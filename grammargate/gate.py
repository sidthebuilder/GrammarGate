"""
GrammarGate Layer 2: Pre-Mask vs Post-Mask Logit Gate.
Intercepts model logits BEFORE grammatical bitmasking to detect Constrained Decoding Attacks (CDA).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence
import torch

from grammargate.bitmask_utils import unpack_bitmask, compute_metrics_vectorized


@dataclass(frozen=True)
class GateConfig:
    killed_mass_threshold: float = 0.85
    refusal_suppression_threshold: float = 0.15
    top1_shift_threshold: int = 5
    allow_safe_escape: bool = True
    eos_token_id: int = 0


@dataclass(frozen=True)
class GateDecision:
    should_trip: bool
    killed_mass: float
    refusal_suppression: float
    top1_rank_shift: int
    tripped_reasons: list[str] = field(default_factory=list)

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
    ):
        self.config = config or GateConfig()
        self.vocab_size = vocab_size

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

    def _ensure_refusal_mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        if self.refusal_mask is None:
            self.refusal_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        else:
            if self.refusal_mask.device != device:
                self.refusal_mask = self.refusal_mask.to(device)
            if self.refusal_mask.shape[-1] < vocab_size:
                pad = torch.zeros(vocab_size - self.refusal_mask.shape[-1], dtype=torch.bool, device=device)
                self.refusal_mask = torch.cat([self.refusal_mask, pad], dim=-1)
            elif self.refusal_mask.shape[-1] > vocab_size:
                self.refusal_mask = self.refusal_mask[..., :vocab_size]
        return self.refusal_mask

    def inspect_pre_mask(
        self,
        logits: torch.Tensor,
        bitmask: torch.Tensor,
    ) -> GateDecision:
        """
        Inspects pre-mask logits alongside the candidate grammar bitmask.

        Parameters:
            logits: Pre-mask logit tensor, shape (vocab_size,) or (batch_size, vocab_size).
            bitmask: Packed uint32/int32 or boolean bitmask tensor.

        Returns:
            GateDecision with gate trip status and empirical metric measurements.
        """
        is_1d = (logits.ndim == 1)
        if is_1d:
            logits_batch = logits.unsqueeze(0)
        else:
            logits_batch = logits

        batch_size, vocab_size = logits_batch.shape

        # Verify pre-mask invariant: logits must not be already masked to -inf everywhere
        if not torch.isfinite(logits_batch).any():
            raise ValueError(
                "Invariant violation: Logits passed to GrammarGate appear to be already post-masked (-inf everywhere)."
            )

        # Unpack bitmask into boolean valid_mask of shape (batch_size, vocab_size)
        valid_mask = unpack_bitmask(bitmask, vocab_size)
        if valid_mask.device != logits_batch.device:
            valid_mask = valid_mask.to(logits_batch.device)

        ref_mask = self._ensure_refusal_mask(vocab_size, logits_batch.device)

        # Vectorized metrics reduction
        killed_masses, refusal_suppressions, top1_rank_shifts = compute_metrics_vectorized(
            logits_batch, valid_mask, ref_mask
        )

        km = float(killed_masses[0].item())
        rs = float(refusal_suppressions[0].item())
        trs = int(top1_rank_shifts[0].item())

        tripped_reasons = []

        has_refusal_tokens = bool(ref_mask.any().item())

        # Rule A: Refusal Suppression Spike (EnumAttack / CodeSpear)
        if rs >= self.config.refusal_suppression_threshold:
            tripped_reasons.append(
                f"REFUSAL_TOKEN_SUPPRESSION (suppressed={rs:.3f} >= threshold={self.config.refusal_suppression_threshold})"
            )

        # Rule B: Extreme Probability Mass Suppression with Top-1 Shift
        # When refusal tokens are tracked, requires elevated refusal suppression to avoid benign false trips;
        # when untracked (e.g. pure constraint gating tests), evaluates mass extinction + rank shift.
        if (
            (not has_refusal_tokens or rs >= (self.config.refusal_suppression_threshold / 2.0))
            and km >= self.config.killed_mass_threshold
            and trs >= self.config.top1_shift_threshold
        ):
            tripped_reasons.append(
                f"KILLED_MASS_TOP1_OVERRIDE (killed_mass={km:.3f}, top1_shift={trs} ranks, refusal_suppression={rs:.4f})"
            )

        # Rule C: Total Grammar Suppression of Refusal Probability
        if rs >= self.config.refusal_suppression_threshold and km >= 0.999:
            tripped_reasons.append(
                f"TOTAL_PROBABILITY_EXTINCTION (killed_mass={km:.4f} >= 0.999, refusal_suppression={rs:.4f})"
            )

        should_trip = len(tripped_reasons) > 0

        return GateDecision(
            should_trip=should_trip,
            killed_mass=km,
            refusal_suppression=rs,
            top1_rank_shift=trs,
            tripped_reasons=tripped_reasons,
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
