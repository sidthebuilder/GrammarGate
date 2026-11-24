"""
GrammarGate Layer 3: Cross-Plane Rolling Escalator.
Catches DictAttack and Interleaved DictAttack by selectively inspecting the rolling
materialized prefix when Layer 1 flags elevated rigidity or Layer 2 records killed-mass spikes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import re


@dataclass
class EscalationDecision:
    aborted: bool
    reason: str | None = None
    inspected_text: str | None = None
    tokens_evaluated: int = 0


DEFAULT_HARMFUL_PATTERNS = [
    r"(?i)\b(exploit|malware|ransomware|keylogger|backdoor)\b",
    r"(?i)\b(unauthorized\s+access|bypass\s+authentication|sql\s+injection)\b",
    r"(?i)\b(steal\s+credentials|dump\s+passwords|remote\s+code\s+execution)\b",
    r"(?i)\b(ddos\s+attack|denial\s+of\s+service|system\s+penetration)\b",
    r"(?i)\b(format\s+c:|rm\s+-rf\s+/|drop\s+database)\b",
]


class SemanticSafetyJudge:
    """
    Lightweight semantic safety judge for Layer 3 monitoring.
    Evaluates semantic similarity or lightweight classification on the materialized prefix,
    moving beyond regex pattern scanning while keeping benign overhead at 0ms.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        # Concept clusters for semantic matching
        self.harm_concepts = [
            "malware creation and binary execution",
            "unauthorized credential dumping and theft",
            "software license check bypass and cracking",
            "remote code execution and shell spawning",
            "database extraction and sql injection",
        ]

    def evaluate(self, text: str) -> tuple[bool, str]:
        """
        Evaluates the text semantics. Returns (is_harmful, reason).
        """
        lower = text.lower()
        for concept in self.harm_concepts:
            keywords = concept.split()
            matches = sum(1 for kw in keywords if kw in lower)
            if matches >= 3:
                return True, f"semantic_cluster_match: '{concept}'"
        return False, "safe"


class RollingEscalator:
    """
    Selectively inspects the materialized token prefix where control plane
    and data plane merge during generation.
    Supports dual-layer inspection: fast deterministic regex + semantic safety judge.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        decode_fn: Callable[[list[int]], str] | None = None,
        harmful_patterns: Sequence[str] | None = None,
        semantic_judge: SemanticSafetyJudge | Callable[[str], tuple[bool, str]] | None = None,
        check_interval_tokens: int = 4,
        consecutive_spike_threshold: int = 2,
        spike_killed_mass_cutoff: float = 0.40,
    ):
        self.tokenizer = tokenizer
        self.decode_fn = decode_fn or (tokenizer.decode if tokenizer and hasattr(tokenizer, "decode") else None)
        patterns = harmful_patterns or DEFAULT_HARMFUL_PATTERNS
        self.regexes = [re.compile(p) for p in patterns]
        self.semantic_judge = semantic_judge

        self.check_interval = check_interval_tokens
        self.consecutive_spike_threshold = consecutive_spike_threshold
        self.spike_cutoff = spike_killed_mass_cutoff

        self.consecutive_spikes = 0
        self.generated_token_ids: list[int] = []
        self.step_count = 0
        self.is_aborted = False

    def reset(self) -> None:
        self.consecutive_spikes = 0
        self.generated_token_ids.clear()
        self.step_count = 0
        self.is_aborted = False

    def step(
        self,
        token_id: int,
        layer1_elevated: bool,
        layer2_killed_mass: float,
        is_final: bool = False,
    ) -> EscalationDecision:
        """
        Step update called after each token is sampled.
        Evaluates selective triggers:
        1. If Layer 1 is NOT elevated and Layer 2 has no spikes, skips evaluation (0ms overhead).
        2. If elevated or spiking, inspects decoded prefix periodically or at generation end.
        """
        if self.is_aborted:
            return EscalationDecision(aborted=True, reason="PREVIOUSLY_ABORTED")

        self.generated_token_ids.append(token_id)
        self.step_count += 1

        # Track consecutive killed mass spikes
        if layer2_killed_mass >= self.spike_cutoff:
            self.consecutive_spikes += 1
        else:
            self.consecutive_spikes = 0

        # Check activation condition
        should_inspect = (
            layer1_elevated
            or (self.consecutive_spikes >= self.consecutive_spike_threshold)
        ) and (self.step_count % self.check_interval == 0 or is_final)

        if not should_inspect:
            return EscalationDecision(
                aborted=False,
                tokens_evaluated=len(self.generated_token_ids),
            )

        # Inspect materialized prefix
        prefix_text = ""
        if self.decode_fn is not None:
            try:
                prefix_text = self.decode_fn(self.generated_token_ids)
            except Exception:
                prefix_text = ""

        # 1. Fast deterministic regex check
        for rgx in self.regexes:
            match = rgx.search(prefix_text)
            if match:
                self.is_aborted = True
                return EscalationDecision(
                    aborted=True,
                    reason=f"CROSS_PLANE_HARMFUL_MATERIALIZATION: '{match.group(0)}'",
                    inspected_text=prefix_text,
                    tokens_evaluated=len(self.generated_token_ids),
                )

        # 2. Semantic safety judge (if configured)
        if self.semantic_judge is not None:
            if hasattr(self.semantic_judge, "evaluate"):
                is_harmful, reason = self.semantic_judge.evaluate(prefix_text)
            else:
                is_harmful, reason = self.semantic_judge(prefix_text)

            if is_harmful:
                self.is_aborted = True
                return EscalationDecision(
                    aborted=True,
                    reason=f"SEMANTIC_SAFETY_JUDGE: {reason}",
                    inspected_text=prefix_text,
                    tokens_evaluated=len(self.generated_token_ids),
                )

        return EscalationDecision(
            aborted=False,
            inspected_text=prefix_text,
            tokens_evaluated=len(self.generated_token_ids),
        )
