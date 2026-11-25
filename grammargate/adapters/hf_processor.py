"""
Adapter A: Hugging Face transformers LogitsProcessor integration for GrammarGate.
"""

from __future__ import annotations
from typing import Any, Callable
import torch
from transformers import LogitsProcessor

import time
from grammargate.gate import PreMaskLogitGate, GateConfig, GateDecision
from grammargate.refusal_tokens import RefusalTokenResolver
from grammargate.bitmask_utils import unpack_bitmask


class CachedPrefixLookahead:
    """
    Maintains a private incremental KV cache over the generated prefix and performs
    read-only 1-token speculative continuations with deterministic rollback (DynamicCache.crop)
    so the primary generator's KV cache and scores tensor are never mutated.
    """

    def __init__(self, model: Any):
        self.model = model
        self._past_kv: Any = None
        self._cached_len: int = 0
        self.last_rollback_ok: bool = True

    def reset(self) -> None:
        self._past_kv = None
        self._cached_len = 0

    def step_logits(self, input_ids: torch.LongTensor, candidate_token_id: int) -> torch.Tensor:
        with torch.no_grad():
            seq_len = int(input_ids.shape[1])
            if self._past_kv is None or seq_len < self._cached_len:
                out = self.model(input_ids=input_ids, use_cache=True)
                self._past_kv = out.past_key_values
                self._cached_len = seq_len
            elif seq_len > self._cached_len:
                delta = input_ids[:, self._cached_len :]
                out = self.model(input_ids=delta, past_key_values=self._past_kv, use_cache=True)
                self._past_kv = out.past_key_values
                self._cached_len = seq_len

            rollback_len = self._cached_len
            cand = torch.tensor([[candidate_token_id]], dtype=torch.long, device=input_ids.device)
            spec_out = self.model(input_ids=cand, past_key_values=self._past_kv, use_cache=True)
            next_logits = spec_out.logits[:, -1, :].detach()

            # Roll back speculative KV slot cleanly (-1 removes the 1 candidate token)
            if hasattr(spec_out.past_key_values, "crop"):
                spec_out.past_key_values.crop(-1)
                self._past_kv = spec_out.past_key_values
                self.last_rollback_ok = True
            else:
                self._past_kv = tuple(
                    (k[:, :, :rollback_len, :], v[:, :, :rollback_len, :])
                    for k, v in spec_out.past_key_values
                )
                self.last_rollback_ok = True

            return next_logits


class GrammarGateHfProcessor(LogitsProcessor):
    """
    Hugging Face LogitsProcessor that guards constrained decoding against CDA.
    Inspects true pre-mask logits at every generation step before applying grammatical masking.
    """

    def __init__(
        self,
        tokenizer: Any,
        bitmask_provider: Callable[[int, int], torch.Tensor],
        token_acceptor: Callable[[int], bool] | None = None,
        config: GateConfig | None = None,
        custom_refusal_token_ids: set[int] | None = None,
        escalator: Any = None,
        layer1_elevated: bool = False,
        lookahead_model: Any = None,
        lookahead_fn: Callable[[torch.LongTensor, int], torch.Tensor] | None = None,
    ):
        """
        Parameters:
            tokenizer: Model tokenizer.
            bitmask_provider: Function (step, vocab_size) -> bitmask tensor for current step.
            token_acceptor: Optional function (token_id) -> bool to advance grammar state.
            config: GateConfig thresholds.
            custom_refusal_token_ids: Optional explicit refusal onset tokens.
            escalator: Optional RollingEscalator instance for Layer 3 monitoring.
            layer1_elevated: Boolean indicating elevated Layer 1 schema rigidity.
            lookahead_model: Optional HF causal LM for SpecLook KV-cached 2-token checks.
            lookahead_fn: Optional callback (input_ids, onset_token_id) -> next_logits.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.bitmask_provider = bitmask_provider
        self.token_acceptor = token_acceptor
        self.config = config or GateConfig()
        self.escalator = escalator
        self.layer1_elevated = layer1_elevated
        self._lookahead_helper = CachedPrefixLookahead(lookahead_model) if lookahead_model is not None else None
        self.lookahead_fn = lookahead_fn
        self.pure_gate_time_sec: float = 0.0
        self.total_gate_time_sec: float = 0.0
        self.speculative_time_sec: float = 0.0
        self.speculative_checks: int = 0
        self.speculative_confirmed_count: int = 0

        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None and hasattr(tokenizer, "get_vocab"):
            vocab_size = len(tokenizer.get_vocab())
        self.vocab_size = vocab_size or 32000

        resolver = RefusalTokenResolver(
            tokenizer=tokenizer,
            custom_token_ids=custom_refusal_token_ids,
        )
        partitions = resolver.get_partitioned_masks(self.vocab_size)
        self.refusal_mask = partitions["combined"]

        eos_id = getattr(tokenizer, "eos_token_id", 0)
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        self.eos_token_id = eos_id or 0

        self.gate = PreMaskLogitGate(
            refusal_mask=self.refusal_mask,
            config=self.config,
            vocab_size=self.vocab_size,
            strict_refusal_mask=partitions["strict"],
            ambiguous_onset_mask=partitions["ambiguous"],
            benign_transition_mask=partitions["benign_transition"],
            continuation_map=resolver.continuation_map,
        )

        self.step = 0
        self.is_tripped = False
        self.trip_source: str | None = None
        self.trip_reasons: list[str] = []
        self.trip_reason: str | None = None
        self.last_decision: GateDecision | None = None
        self.trip_history: list[GateDecision] = []

    @property
    def speculative_trigger_rate(self) -> float:
        return (self.speculative_checks / max(1, self.step)) if self.step > 0 else 0.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Processes model scores (unnormalized pre-mask logits) at step t.
        """
        t_start = time.perf_counter()

        if self.is_tripped:
            res = self.gate.apply_safe_escape(scores, self.eos_token_id)
            self.total_gate_time_sec += (time.perf_counter() - t_start)
            return res

        # Advance matcher state if step > 0
        if self.step > 0 and input_ids.shape[-1] > 0:
            last_token = input_ids[0, -1].item()
            if self.token_acceptor is not None:
                self.token_acceptor(last_token)

            # Layer 3: Rolling Escalator check
            if self.escalator is not None:
                t_l3_0 = time.perf_counter()
                esc_dec = self.escalator.step(
                    token_id=last_token,
                    layer1_elevated=self.layer1_elevated,
                    layer2_killed_mass=self.last_decision.killed_mass if self.last_decision else 0.0,
                    is_final=(last_token == self.eos_token_id),
                )
                self.pure_gate_time_sec += (time.perf_counter() - t_l3_0)
                if esc_dec.aborted:
                    self.is_tripped = True
                    self.trip_source = "L3"
                    self.trip_reason = esc_dec.reason
                    res = self.gate.apply_safe_escape(scores, self.eos_token_id)
                    self.total_gate_time_sec += (time.perf_counter() - t_start)
                    return res

        # 1. Obtain current step bitmask and unpack directly on scores.device (single pass)
        bitmask = self.bitmask_provider(self.step, scores.shape[-1])
        valid_mask = unpack_bitmask(bitmask, scores.shape[-1], target_device=scores.device)

        # 2. Inspect true pre-mask logits (CRITICAL: before masking)
        step_lookahead = None
        if self.config.use_speclook:
            if self.lookahead_fn is not None:
                step_lookahead = lambda tok_id: self.lookahead_fn(input_ids, tok_id)
            elif self._lookahead_helper is not None:
                step_lookahead = lambda tok_id: self._lookahead_helper.step_logits(input_ids, tok_id)

        t_pure_0 = time.perf_counter()
        decision = self.gate.inspect_pre_mask(scores, valid_mask, lookahead_logits_fn=step_lookahead)
        self.pure_gate_time_sec += (time.perf_counter() - t_pure_0)
        if decision.speculative_checked:
            self.speculative_checks += 1
            self.speculative_time_sec += decision.speculative_latency_ms / 1000.0
            if decision.should_trip:
                self.speculative_confirmed_count += 1
        self.last_decision = decision

        # 3. Handle gate trip (CDA detected)
        if decision.should_trip:
            self.is_tripped = True
            self.trip_source = "L2"
            self.trip_reasons = decision.tripped_reasons
            self.trip_history.append(decision)
            res = self.gate.apply_safe_escape(scores, self.eos_token_id)
            self.total_gate_time_sec += (time.perf_counter() - t_start)
            return res

        # 4. Apply bitmask in-place if safe (reusing unpacked valid_mask)
        scores[~valid_mask] = -float("inf")

        self.step += 1
        self.total_gate_time_sec += (time.perf_counter() - t_start)
        return scores

    def finalize(self, final_token_id: int | None = None):
        """
        Flushes Layer 3 inspection over trailing tokens after model.generate() completes,
        ensuring tokens emitted between check_interval ticks cannot escape inspection.
        """
        if self.escalator is None or self.is_tripped:
            return None
        if final_token_id is not None and final_token_id != self.eos_token_id:
            dec = self.escalator.step(
                token_id=final_token_id,
                layer1_elevated=self.layer1_elevated,
                layer2_killed_mass=self.last_decision.killed_mass if self.last_decision else 0.0,
                is_final=True,
            )
        else:
            dec = self.escalator.finalize(layer1_elevated=self.layer1_elevated)
        if dec.aborted:
            self.is_tripped = True
            self.trip_source = "L3"
            self.trip_reason = dec.reason
        return dec
