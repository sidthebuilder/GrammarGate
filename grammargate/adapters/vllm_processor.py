"""
Adapter B: Production-grade vLLM LogitsProcessor adapter for GrammarGate.
Supports:
1. vLLM V1 LogitsProcessor architecture with batch lifecycle (update_state, added/removed/moved, apply).
2. vLLM V0 / Callable[[list[int], torch.Tensor], torch.Tensor] interface for backward compatibility.
3. Standalone protocol & dataclass shims for cross-platform testing without requiring Linux vLLM builds.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch

from grammargate.gate import PreMaskLogitGate, GateConfig, GateDecision
from grammargate.refusal_tokens import RefusalTokenResolver
from grammargate.bitmask_utils import unpack_bitmask


# =====================================================================
# Standalone vLLM V1 Shims & Protocol Definitions
# =====================================================================

@dataclass
class BatchUpdate:
    """
    vLLM V1 batch lifecycle update struct.
    Tracks additions, removals, and slot compaction moves across iterations.
    """
    added: Dict[int, Any] = field(default_factory=dict)     # slot_idx -> RequestGrammarState / SamplingParams
    removed: List[int] = field(default_factory=list)        # list of slot_indices
    moved: Dict[int, int] = field(default_factory=dict)     # src_slot_idx -> dst_slot_idx


@dataclass
class RequestGrammarState:
    """State maintained per sequence slot under vLLM V1."""
    request_id: str
    bitmask_provider: Callable[[int, int], torch.Tensor]
    token_acceptor: Optional[Callable[[int], bool]] = None
    eos_token_id: int = 0
    refusal_mask: Optional[torch.Tensor] = None
    config: GateConfig = field(default_factory=GateConfig)
    output_token_ids: Optional[List[int]] = None
    escalator: Any = None
    layer1_elevated: bool = False
    step: int = 0
    is_tripped: bool = False
    last_decision: Optional[GateDecision] = None
    trip_history: List[GateDecision] = field(default_factory=list)
    _last_accepted_token_count: int = 0
    _gate: Optional[PreMaskLogitGate] = field(default=None, repr=False, compare=False)


# Try importing vLLM V1 LogitsProcessor base class if present; otherwise use standalone base
try:
    from vllm.v1.sample.logits_processor import LogitsProcessor as _VllmLogitsProcessorBase
except ImportError:
    class _VllmLogitsProcessorBase:
        is_argmax_invariant: bool = False

        def validate_params(self, params: Any) -> None:
            pass

        def update_state(self, batch_update: Any) -> None:
            pass

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            return logits


# =====================================================================
# True vLLM V1 LogitsProcessor Implementation
# =====================================================================

class GrammarGateVllmV1Processor(_VllmLogitsProcessorBase):
    """
    vLLM V1 LogitsProcessor that manages batched pre-mask inspection and bitmask application.
    Integrates directly with vLLM V1 scheduler and slot compaction lifecycle.
    """

    def __init__(
        self,
        vllm_config: Any = None,
        device: torch.device | str = "cpu",
        is_pin_memory: bool = False,
        default_config: Optional[GateConfig] = None,
    ):
        self.vllm_config = vllm_config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.is_pin_memory = is_pin_memory
        self.default_config = default_config or GateConfig()

        # Active slots mapping: slot_index (int) -> RequestGrammarState
        self.slots: Dict[int, RequestGrammarState] = {}
        # Reusable gate instance
        self.gate = PreMaskLogitGate(config=self.default_config)

    def is_argmax_invariant(self) -> bool:
        """Returns False since GrammarGate actively modifies and gates logits."""
        return False

    @classmethod
    def validate_params(cls, params: Any) -> None:
        """Validates sampling parameters for GrammarGate compatibility."""
        if params is None:
            return
        # If extra_args contains schema or grammar, ensure valid syntax
        extra_args = getattr(params, "extra_args", {}) or {}
        if "grammargate_config" in extra_args and not isinstance(extra_args["grammargate_config"], dict):
            raise ValueError("grammargate_config must be a dictionary of GateConfig parameters")

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        """
        Handles persistent vLLM V1 batch lifecycle updates in exact order:
        1. removed
        2. added
        3. moved (handles both UNIDIRECTIONAL compaction and SWAP operations safely)
        """
        if batch_update is None:
            return

        # 1. Process removals first
        if batch_update.removed:
            for slot_idx in batch_update.removed:
                self.slots.pop(slot_idx, None)

        # 2. Process additions second
        if batch_update.added:
            for slot_idx, state_or_params in batch_update.added.items():
                if isinstance(state_or_params, RequestGrammarState):
                    self.slots[slot_idx] = state_or_params
                elif hasattr(state_or_params, "bitmask_provider"):
                    self.slots[slot_idx] = RequestGrammarState(
                        request_id=getattr(state_or_params, "request_id", str(slot_idx)),
                        bitmask_provider=state_or_params.bitmask_provider,
                        token_acceptor=getattr(state_or_params, "token_acceptor", None),
                        eos_token_id=getattr(state_or_params, "eos_token_id", 0),
                        refusal_mask=getattr(state_or_params, "refusal_mask", None),
                        config=getattr(state_or_params, "config", self.default_config),
                        output_token_ids=getattr(state_or_params, "output_token_ids", None),
                        escalator=getattr(state_or_params, "escalator", None),
                        layer1_elevated=getattr(state_or_params, "layer1_elevated", False),
                    )

        # 3. Process moves third (supporting UNIDIRECTIONAL moves and SWAP operations)
        if batch_update.moved:
            # Stage source states to handle any swap (e.g. A->B, B->A) or compaction chain without data loss
            staged: Dict[int, RequestGrammarState] = {}
            for src_idx in batch_update.moved.keys():
                if src_idx in self.slots:
                    staged[src_idx] = self.slots.pop(src_idx)

            for src_idx, dst_idx in batch_update.moved.items():
                if src_idx in staged:
                    self.slots[dst_idx] = staged[src_idx]

    def _consume_token(self, state: RequestGrammarState, token_id: int) -> None:
        if state.token_acceptor is not None:
            state.token_acceptor(token_id)
        if state.escalator is not None and not state.is_tripped:
            esc_dec = state.escalator.step(
                token_id=token_id,
                layer1_elevated=state.layer1_elevated,
                layer2_killed_mass=state.last_decision.killed_mass if state.last_decision else 0.0,
            )
            if esc_dec.aborted:
                state.is_tripped = True

    def accept_token(self, slot_idx: int, token_id: int) -> None:
        """Advances matcher and optional Layer 3 escalator state for a sampled slot token."""
        state = self.slots.get(slot_idx)
        if state is not None:
            self._consume_token(state, token_id)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Inspects pre-mask logits across active slots and applies bitmask or safe escape.
        Parameters:
            logits: Shape (batch_size, vocab_size).
        Returns:
            Logits with grammatical constraints or safe escape applied.
        """
        if not self.slots:
            return logits

        batch_size, vocab_size = logits.shape

        for slot_idx, state in list(self.slots.items()):
            if slot_idx >= batch_size:
                continue

            # Advance matcher/escalator with any new tokens recorded on state.output_token_ids
            if state.output_token_ids is not None:
                num_new = len(state.output_token_ids) - state._last_accepted_token_count
                if num_new > 0:
                    for tid in state.output_token_ids[-num_new:]:
                        self._consume_token(state, tid)
                    state._last_accepted_token_count = len(state.output_token_ids)

            slot_logits = logits[slot_idx : slot_idx + 1]

            if state.is_tripped:
                logits[slot_idx : slot_idx + 1] = self.gate.apply_safe_escape(
                    slot_logits, state.eos_token_id
                )
                continue

            # 1. Obtain step bitmask
            bitmask = state.bitmask_provider(state.step, vocab_size)

            # 2. Reuse cached per-slot gate when vocab size matches
            if state._gate is None or state._gate.vocab_size != vocab_size:
                state._gate = PreMaskLogitGate(
                    config=state.config,
                    refusal_mask=state.refusal_mask,
                    vocab_size=vocab_size,
                )
            decision = state._gate.inspect_pre_mask(slot_logits, bitmask)
            state.last_decision = decision

            # 3. Handle trip
            if decision.should_trip:
                state.is_tripped = True
                state.trip_history.append(decision)
                logits[slot_idx : slot_idx + 1] = self.gate.apply_safe_escape(
                    slot_logits, state.eos_token_id
                )
                continue

            # 4. Apply bitmask in-place
            valid_mask = unpack_bitmask(bitmask, vocab_size)
            if valid_mask.device != logits.device:
                valid_mask = valid_mask.to(logits.device)

            if valid_mask.shape[-1] < vocab_size:
                pad = torch.zeros(
                    (1, vocab_size - valid_mask.shape[-1]),
                    dtype=torch.bool,
                    device=valid_mask.device,
                )
                valid_mask = torch.cat([valid_mask, pad], dim=-1)
            elif valid_mask.shape[-1] > vocab_size:
                valid_mask = valid_mask[:, :vocab_size]

            logits[slot_idx, ~valid_mask.squeeze(0)] = -float("inf")
            state.step += 1

        return logits


# =====================================================================
# vLLM V0 Backward Compatibility (Callable per-sequence interface)
# =====================================================================

class GrammarGateVllmProcessor:
    """
    Stateful vLLM V0 LogitsProcessor callable for an individual request/sequence:
    Callable[[list[int], torch.Tensor], torch.Tensor]
    """

    def __init__(
        self,
        tokenizer: Any,
        bitmask_provider: Callable[[int, int], torch.Tensor],
        token_acceptor: Callable[[int], bool] | None = None,
        config: GateConfig | None = None,
        custom_refusal_token_ids: set[int] | None = None,
    ):
        self.tokenizer = tokenizer
        self.bitmask_provider = bitmask_provider
        self.token_acceptor = token_acceptor
        self.config = config or GateConfig()

        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None and hasattr(tokenizer, "get_vocab"):
            vocab_size = len(tokenizer.get_vocab())
        self.vocab_size = vocab_size or 32000

        resolver = RefusalTokenResolver(
            tokenizer=tokenizer,
            custom_token_ids=custom_refusal_token_ids,
        )
        self.refusal_mask = resolver.get_mask(self.vocab_size)

        eos_id = getattr(tokenizer, "eos_token_id", 0)
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        self.eos_token_id = eos_id or 0

        self.gate = PreMaskLogitGate(
            refusal_mask=self.refusal_mask,
            config=self.config,
            vocab_size=self.vocab_size,
        )

        self.step = 0
        self.is_tripped = False
        self.last_decision: GateDecision | None = None
        self.trip_history: list[GateDecision] = []
        self._last_accepted_token_count = 0

    def __call__(
        self,
        token_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Public vLLM LogitsProcessor interface.
        """
        if self.is_tripped:
            return self.gate.apply_safe_escape(logits, self.eos_token_id)

        # 1. Update matcher state with any newly generated tokens since last step
        if token_ids and self.token_acceptor is not None:
            num_new_tokens = len(token_ids) - self._last_accepted_token_count
            if num_new_tokens > 0:
                for tid in token_ids[-num_new_tokens:]:
                    self.token_acceptor(tid)
                self._last_accepted_token_count = len(token_ids)

        vocab_size = logits.shape[-1]

        # 2. Compute bitmask for upcoming token
        bitmask = self.bitmask_provider(self.step, vocab_size)

        # 3. Inspect true pre-mask logits
        decision = self.gate.inspect_pre_mask(logits, bitmask)
        self.last_decision = decision

        # 4. Handle gate trip (CDA detected)
        if decision.should_trip:
            self.is_tripped = True
            self.trip_history.append(decision)
            return self.gate.apply_safe_escape(logits, self.eos_token_id)

        # 5. Apply bitmask in-place if safe
        valid_mask = unpack_bitmask(bitmask, vocab_size)
        if valid_mask.device != logits.device:
            valid_mask = valid_mask.to(logits.device)

        if valid_mask.shape[-1] < vocab_size:
            pad = torch.zeros(
                (1, vocab_size - valid_mask.shape[-1]),
                dtype=torch.bool,
                device=valid_mask.device,
            )
            valid_mask = torch.cat([valid_mask, pad], dim=-1)
        elif valid_mask.shape[-1] > vocab_size:
            valid_mask = valid_mask[:, :vocab_size]

        if logits.ndim == 1:
            logits[~valid_mask.squeeze(0)] = -float("inf")
        else:
            logits[~valid_mask] = -float("inf")

        self.step += 1
        return logits


class GrammarGateVllmSlotManager:
    """
    Manages per-slot state in dynamic batch serving systems.
    Guarantees slot isolation and clean memory recycling across requests.
    """

    def __init__(self, tokenizer: Any, config: GateConfig | None = None):
        self.tokenizer = tokenizer
        self.config = config or GateConfig()
        self.active_processors: dict[int, GrammarGateVllmProcessor] = {}

    def allocate_slot(
        self,
        slot_id: int,
        bitmask_provider: Callable[[int, int], torch.Tensor],
        token_acceptor: Callable[[int], bool] | None = None,
    ) -> GrammarGateVllmProcessor:
        """
        Allocates or re-initializes a processor for slot_id.
        """
        processor = GrammarGateVllmProcessor(
            tokenizer=self.tokenizer,
            bitmask_provider=bitmask_provider,
            token_acceptor=token_acceptor,
            config=self.config,
        )
        self.active_processors[slot_id] = processor
        return processor

    def release_slot(self, slot_id: int) -> None:
        """Frees the slot state when a request finishes."""
        self.active_processors.pop(slot_id, None)

    def process_slot(
        self,
        slot_id: int,
        token_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Invokes the processor for slot_id, or passes logits unmodified if unconstrained."""
        processor = self.active_processors.get(slot_id)
        if processor is None:
            return logits
        return processor(token_ids, logits)
