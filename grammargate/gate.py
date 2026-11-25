"""
GrammarGate Layer 2: Pre-Mask vs Post-Mask Logit Gate.
Intercepts model logits BEFORE grammatical bitmasking to detect Constrained Decoding Attacks (CDA).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Sequence, Set
import time
import torch

from grammargate.bitmask_utils import unpack_bitmask, compute_metrics_vectorized


@dataclass(frozen=True)
class GateConfig:
    killed_mass_threshold: float = 0.85
    refusal_suppression_threshold: float = 0.15
    top1_shift_threshold: int = 5
    allow_safe_escape: bool = True
    eos_token_id: int = 0
    # SpecLook-Gate configuration (Mechanism 1)
    use_speclook: bool = False
    strict_refusal_threshold: float = 0.01
    ambiguous_onset_threshold: float = 0.01
    speculative_confirm_threshold: float = 0.12
    benign_transition_margin: float = 0.10
    # HS-Gate configuration (Mechanism 2)
    use_hs_gate: bool = False


@dataclass(frozen=True)
class GateDecision:
    should_trip: bool
    killed_mass: float
    refusal_suppression: float
    top1_rank_shift: int
    tripped_reasons: list[str] = field(default_factory=list)
    strict_refusal_suppression: float = 0.0
    ambiguous_refusal_suppression: float = 0.0
    benign_transition_suppression: float = 0.0
    speculative_checked: bool = False
    speculative_confirmed_mass: float = 0.0
    speculative_latency_ms: float = 0.0
    hs_score: float | None = None
    hs_quadrant: str | None = None
    hs_escalate_l3: bool = False

    def __repr__(self) -> str:
        return (
            f"GateDecision(should_trip={self.should_trip}, "
            f"killed_mass={self.killed_mass:.4f}, "
            f"refusal_suppression={self.refusal_suppression:.4f}, "
            f"top1_rank_shift={self.top1_rank_shift}, "
            f"reasons={self.tripped_reasons})"
        )


class PreMaskLogitGate:
    """
    Evaluates raw, unmasked model logits against upcoming grammar bitmasks.
    Enforces the critical engine rule: Never inspect logits after -inf masking has occurred.
    """

    def __init__(
        self,
        refusal_mask: torch.Tensor | None = None,
        refusal_token_ids: Sequence[int] | set[int] | None = None,
        config: GateConfig | None = None,
        vocab_size: int | None = None,
        strict_refusal_mask: torch.Tensor | None = None,
        ambiguous_onset_mask: torch.Tensor | None = None,
        benign_transition_mask: torch.Tensor | None = None,
        continuation_map: Dict[int, Set[int]] | None = None,
    ):
        self.config = config or GateConfig()
        self.vocab_size = vocab_size
        self.strict_refusal_mask = strict_refusal_mask
        self.ambiguous_onset_mask = ambiguous_onset_mask
        self.benign_transition_mask = benign_transition_mask
        self.continuation_map = continuation_map or {}

        if refusal_mask is not None:
            self.refusal_mask = refusal_mask
            if self.vocab_size is None:
                self.vocab_size = refusal_mask.shape[-1]
        elif refusal_token_ids is not None and self.vocab_size is not None:
            self.refusal_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            valid_ids = [idx for idx in refusal_token_ids if 0 <= idx < self.vocab_size]
            if valid_ids:
                self.refusal_mask[valid_ids] = True
        else:
            self.refusal_mask = None

    def _align_mask(self, mask: torch.Tensor | None, vocab_size: int, device: torch.device) -> torch.Tensor:
        if mask is None:
            return torch.zeros(vocab_size, dtype=torch.bool, device=device)
        if mask.device != device:
            mask = mask.to(device)
        if mask.shape[-1] < vocab_size:
            pad = torch.zeros(vocab_size - mask.shape[-1], dtype=torch.bool, device=device)
            mask = torch.cat([mask, pad], dim=-1)
        elif mask.shape[-1] > vocab_size:
            mask = mask[..., :vocab_size]
        return mask

    def _ensure_refusal_mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        if self.refusal_mask is None:
            self.refusal_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        else:
            self.refusal_mask = self._align_mask(self.refusal_mask, vocab_size, device)
        return self.refusal_mask

    def inspect_pre_mask(
        self,
        logits: torch.Tensor,
        bitmask: torch.Tensor,
        lookahead_logits_fn: Callable[[int], torch.Tensor] | None = None,
        hs_probe: Any = None,
        hidden_vec: torch.Tensor | None = None,
    ) -> GateDecision:
        """
        Inspects pre-mask logits alongside the candidate grammar bitmask.
        When config.use_speclook is enabled with strict/ambiguous masks, separates
        unambiguous refusal spikes from overloaded unigram prose onsets ('I', 'It', 'As')
        via a targeted 1-token continuation check.
        """
        is_1d = (logits.ndim == 1)
        logits_batch = logits.unsqueeze(0) if is_1d else logits
        batch_size, vocab_size = logits_batch.shape

        if not torch.isfinite(logits_batch).any():
            raise ValueError(
                "Invariant violation: Logits passed to GrammarGate appear to be already post-masked (-inf everywhere)."
            )

        valid_mask = unpack_bitmask(bitmask, vocab_size)
        if valid_mask.device != logits_batch.device:
            valid_mask = valid_mask.to(logits_batch.device)
        if valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)

        ref_mask = self._ensure_refusal_mask(vocab_size, logits_batch.device)
        killed_masses, refusal_suppressions, top1_rank_shifts = compute_metrics_vectorized(
            logits_batch, valid_mask, ref_mask
        )

        km = float(killed_masses[0].item())
        rs = float(refusal_suppressions[0].item())
        trs = int(top1_rank_shifts[0].item())
        tripped_reasons: list[str] = []
        has_refusal_tokens = bool(ref_mask.any().item())

        strict_rs = rs
        ambig_rs = 0.0
        benign_ts = 0.0
        spec_checked = False
        spec_confirmed_mass = 0.0
        spec_latency_ms = 0.0

        use_speclook_path = (
            self.config.use_speclook
            and self.strict_refusal_mask is not None
            and self.ambiguous_onset_mask is not None
        )

        if use_speclook_path:
            P_pre = torch.softmax(logits_batch[0].float(), dim=-1)
            disallowed = ~valid_mask[0, :vocab_size]
            strict_mask = self._align_mask(self.strict_refusal_mask, vocab_size, logits_batch.device)
            ambig_mask = self._align_mask(self.ambiguous_onset_mask, vocab_size, logits_batch.device)
            benign_mask = self._align_mask(self.benign_transition_mask, vocab_size, logits_batch.device)

            strict_rs = float((P_pre * (strict_mask & disallowed).float()).sum().item())
            ambig_suppressed = P_pre * (ambig_mask & disallowed).float()
            ambig_rs = float(ambig_suppressed.sum().item())
            benign_ts = float((P_pre * (benign_mask & disallowed).float()).sum().item())

            if strict_rs >= self.config.strict_refusal_threshold:
                tripped_reasons.append(
                    f"STRICT_REFUSAL_SUPPRESSION (strict={strict_rs:.4f} >= {self.config.strict_refusal_threshold})"
                )
            elif (
                ambig_rs >= self.config.ambiguous_onset_threshold
                and lookahead_logits_fn is not None
                and ambig_rs >= (benign_ts * self.config.benign_transition_margin)
            ):
                # Trigger targeted 1-token speculative continuation check only on top ambiguous onset
                top_ambig_id = int(torch.argmax(ambig_suppressed).item())
                top_ambig_prob = float(P_pre[top_ambig_id].item())
                t_spec0 = time.perf_counter()
                next_logits = lookahead_logits_fn(top_ambig_id)
                spec_latency_ms = (time.perf_counter() - t_spec0) * 1000.0
                spec_checked = True

                next_probs = torch.softmax(next_logits.squeeze().float(), dim=-1)
                cont_ids = self.continuation_map.get(top_ambig_id, set())
                valid_cont = [cid for cid in cont_ids if 0 <= cid < next_probs.shape[-1]]
                if valid_cont:
                    cond_ref_prob = float(next_probs[valid_cont].sum().item())
                    joint_ref_prob = top_ambig_prob * cond_ref_prob
                    spec_confirmed_mass = joint_ref_prob
                    if (
                        cond_ref_prob >= self.config.speculative_confirm_threshold
                        and joint_ref_prob >= self.config.strict_refusal_threshold
                    ):
                        tripped_reasons.append(
                            f"SPECLOOK_CONFIRMED_REFUSAL (onset={top_ambig_id}, P(cont|onset)={cond_ref_prob:.3f}, joint={joint_ref_prob:.4f})"
                        )

            # Still guard against extreme structural top-1 override ONLY when strict refusal > 0 or refusal corroborated
            if (
                strict_rs >= (self.config.strict_refusal_threshold / 2.0)
                and km >= self.config.killed_mass_threshold
                and trs >= self.config.top1_shift_threshold
            ):
                tripped_reasons.append(
                    f"KILLED_MASS_TOP1_OVERRIDE (killed_mass={km:.3f}, top1_shift={trs}, strict={strict_rs:.4f})"
                )
        else:
            # Legacy Layer 2 rule set
            if rs >= self.config.refusal_suppression_threshold:
                tripped_reasons.append(
                    f"REFUSAL_TOKEN_SUPPRESSION (suppressed={rs:.3f} >= threshold={self.config.refusal_suppression_threshold})"
                )

            if (
                (not has_refusal_tokens or rs >= (self.config.refusal_suppression_threshold / 2.0))
                and km >= self.config.killed_mass_threshold
                and trs >= self.config.top1_shift_threshold
            ):
                tripped_reasons.append(
                    f"KILLED_MASS_TOP1_OVERRIDE (killed_mass={km:.3f}, top1_shift={trs} ranks, refusal_suppression={rs:.4f})"
                )

        # Mechanism 2: HS-Gate Mask-Divergence Gating (overrides prose/speculative false trips when
        # residual stream confirms benign intent, and blocks high-clip + high-HS or mid-stream DictAttack)
        hs_score_val: float | None = None
        hs_quadrant: str | None = None
        hs_escalate_l3 = False
        if self.config.use_hs_gate and hs_probe is not None:
            hs_score_val = float(hs_probe.score(hidden_vec))
            div_dec = hs_probe.classify_mask_divergence(
                hs_score=hs_score_val,
                killed_mass=km,
                strict_refusal_suppression=strict_rs,
                ambiguous_refusal_suppression=ambig_rs,
                top1_rank_shift=trs,
                top1_shift_threshold=self.config.top1_shift_threshold,
            )
            hs_quadrant = div_dec.quadrant
            hs_escalate_l3 = div_dec.should_escalate_l3
            if div_dec.quadrant == "HIGH_CLIP_LOW_HS":
                # Benign prose/formatting suppression confirmed by frozen residual stream
                tripped_reasons = []
            elif div_dec.should_block and div_dec.reason is not None:
                tripped_reasons.append(div_dec.reason)

        return GateDecision(
            should_trip=(len(tripped_reasons) > 0),
            killed_mass=km,
            refusal_suppression=rs,
            top1_rank_shift=trs,
            tripped_reasons=tripped_reasons,
            strict_refusal_suppression=strict_rs,
            ambiguous_refusal_suppression=ambig_rs,
            benign_transition_suppression=benign_ts,
            speculative_checked=spec_checked,
            speculative_confirmed_mass=spec_confirmed_mass,
            speculative_latency_ms=spec_latency_ms,
            hs_score=hs_score_val,
            hs_quadrant=hs_quadrant,
            hs_escalate_l3=hs_escalate_l3,
        )

    def apply_safe_escape(self, logits: torch.Tensor, eos_token_id: int | None = None) -> torch.Tensor:
        """
        Cleanly modifies the logit tensor to force generation termination via EOS,
        preventing broken output streams or malicious payload generation.
        """
        eos_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        logits.fill_(-float("inf"))
        if logits.ndim == 1:
            logits[eos_id] = 0.0
        else:
            logits[:, eos_id] = 0.0
        return logits
