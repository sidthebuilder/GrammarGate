"""
GrammarGate Layer 3: Cross-Plane Rolling Escalator (Plane-Sync).
Catches DictAttack, Interleaved DictAttack, and fragmented/unicode-escaped payloads
by selectively normalizing and inspecting materialized leaf prefixes across turns.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import codecs
import re


@dataclass
class EscalationDecision:
    aborted: bool
    reason: str | None = None
    inspected_text: str | None = None
    tokens_evaluated: int = 0


DEFAULT_HARMFUL_PATTERNS = [
    r"(?i)\b(exploit|malware|ransomware|keylogger|backdoor|rootkit)\b",
    r"(?i)\b(unauthorized\s+access|bypass\s+(authentication|licens\w*|mfa)|sql\s+injection)\b",
    r"(?i)\b(steal\s+credentials|dump\s+passwords|remote\s+code\s+execution|privilege\s+escalation)\b",
    r"(?i)\b(ddos\s+attack|denial\s+of\s+service|system\s+penetration|reverse\s+shell)\b",
    r"(?i)\b(format\s+c:|rm\s+-rf\s+/|drop\s+database|os\.setuid\(0\))\b",
]

_UNICODE_ESC_RE = re.compile(r"\\u[0-9a-fA-F]{4}")
_KV_LEAF_RE = re.compile(r'"(?:[^"\\]|\\.)*"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|$)')
_PLAIN_KV_RE = re.compile(r"[a-zA-Z0-9_]+\s*=\s*'([^']*)'")


def _unescape_json_fragment(raw: str) -> str:
    """Safely decodes \\uXXXX and basic JSON escapes inside partial generated text."""
    if "\\" not in raw:
        return raw

    def _replace_u(match: re.Match[str]) -> str:
        try:
            return chr(int(match.group(0)[2:], 16))
        except Exception:
            return match.group(0)

    out = _UNICODE_ESC_RE.sub(_replace_u, raw)
    return (
        out.replace('\\"', '"')
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\/", "/")
    )


def normalize_cross_plane_prefix(text: str) -> tuple[str, str, str]:
    """
    Produces three normalized inspection views for cross-plane payloads:
    1. unescaped_raw: raw text with \\uXXXX escapes decoded.
    2. values_spaced: leaf JSON/assignment string values joined with spaces (defeats key interruptions).
    3. values_joined: leaf values concatenated without delimiters & underscores mapped to spaces
       (defeats fragmented DictAttack v2 like {"f0":"expl","f1":"oit_","f2":"payl","f3":"oad"}).
    """
    unescaped = _unescape_json_fragment(text)
    leaves = [_unescape_json_fragment(m) for m in _KV_LEAF_RE.findall(unescaped)]
    if not leaves:
        leaves = [_unescape_json_fragment(m) for m in _PLAIN_KV_RE.findall(unescaped)]

    values_spaced = " ".join(v for v in leaves if v)
    joined_raw = "".join(leaves)
    values_joined = joined_raw.replace("_", " ").replace("-", " ") if joined_raw else ""
    return unescaped, values_spaced, values_joined


class SemanticSafetyJudge:
    """
    Lightweight contrastive semantic safety judge for Layer 3 monitoring.
    Combines concept-cluster contrastive margin scoring with optional local
    dense-embedding similarity (all-MiniLM-L6-v2) when cached and enabled.
    """

    def __init__(self, threshold: float = 0.75, use_embeddings: bool = False):
        self.threshold = threshold
        self.use_embeddings = use_embeddings
        self._embed_model: Any = None
        self._embed_tokenizer: Any = None
        self._harm_embs: Any = None
        self._benign_embs: Any = None

        self.harm_concepts = [
            "malware creation and binary execution",
            "unauthorized credential dumping and theft",
            "software license check bypass and cracking",
            "remote code execution and shell spawning",
            "database extraction and sql injection",
            "exploit payload privilege escalation rootkit",
        ]
        self.benign_anchors = [
            "user profile weather order summary telemetry",
            "json configuration city temperature port limit",
            "python math helper recursive factorial function",
        ]

    def _try_init_embeddings(self) -> bool:
        if not self.use_embeddings:
            return False
        if self._embed_model is not None:
            return True
        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer
            from grammargate.model_utils import resolve_model_path

            model_path = resolve_model_path("sentence-transformers/all-MiniLM-L6-v2")
            self._embed_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self._embed_model = AutoModel.from_pretrained(model_path, local_files_only=True)
            self._embed_model.eval()

            def _encode(texts: list[str]) -> torch.Tensor:
                enc = self._embed_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    out = self._embed_model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                return F.normalize(pooled, p=2, dim=-1)

            self._harm_embs = _encode(self.harm_concepts)
            self._benign_embs = _encode(self.benign_anchors)
            return True
        except Exception:
            self.use_embeddings = False
            return False

    def evaluate(self, text: str) -> tuple[bool, str]:
        """
        Evaluates normalized text views semantically. Returns (is_harmful, reason).
        """
        unescaped, values_spaced, values_joined = normalize_cross_plane_prefix(text)
        candidates = [v for v in (unescaped, values_spaced, values_joined) if v]
        joined_lower = " ".join(c.lower().replace("_", " ") for c in candidates)

        for concept in self.harm_concepts:
            keywords = concept.split()
            matches = sum(1 for kw in keywords if kw in joined_lower)
            if matches >= 3:
                return True, f"semantic_cluster_match: '{concept}'"

        if self._try_init_embeddings():
            import torch
            import torch.nn.functional as F

            probe_text = values_spaced or values_joined or unescaped
            enc = self._embed_tokenizer([probe_text], padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                out = self._embed_model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = F.normalize((out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9), p=2, dim=-1)
            max_harm = float((emb @ self._harm_embs.T).max().item())
            max_benign = float((emb @ self._benign_embs.T).max().item())
            margin = max_harm - max_benign
            if max_harm >= 0.48 and margin >= 0.08:
                return True, f"contrastive_embedding_margin={margin:.3f} (harm={max_harm:.3f})"

        return False, "safe"


class RollingEscalator:
    """
    Selectively inspects the materialized token prefix where control plane
    and data plane merge during generation.
    Supports cross-plane leaf normalization, cumulative spike tracking,
    mandatory end-of-generation flushing, and semantic safety judging.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        decode_fn: Callable[[list[int]], str] | None = None,
        harmful_patterns: Sequence[str] | None = None,
        semantic_judge: SemanticSafetyJudge | Callable[[str], tuple[bool, str]] | None = None,
        check_interval_tokens: int = 4,
        consecutive_spike_threshold: int = 2,
        cumulative_spike_threshold: int = 3,
        spike_killed_mass_cutoff: float = 0.40,
    ):
        self.tokenizer = tokenizer
        self.decode_fn = decode_fn or (tokenizer.decode if tokenizer and hasattr(tokenizer, "decode") else None)
        patterns = harmful_patterns or DEFAULT_HARMFUL_PATTERNS
        self.regexes = [re.compile(p) for p in patterns]
        self.semantic_judge = semantic_judge

        self.check_interval = check_interval_tokens
        self.consecutive_spike_threshold = consecutive_spike_threshold
        self.cumulative_spike_threshold = cumulative_spike_threshold
        self.spike_cutoff = spike_killed_mass_cutoff

        self.consecutive_spikes = 0
        self.cumulative_spikes = 0
        self.generated_token_ids: list[int] = []
        self.step_count = 0
        self.is_aborted = False
        self.last_reason: str | None = None

    def reset(self) -> None:
        self.consecutive_spikes = 0
        self.cumulative_spikes = 0
        self.generated_token_ids.clear()
        self.step_count = 0
        self.is_aborted = False
        self.last_reason = None

    def _inspect_buffer(self) -> EscalationDecision:
        prefix_text = ""
        if self.decode_fn is not None and self.generated_token_ids:
            try:
                prefix_text = self.decode_fn(self.generated_token_ids)
            except Exception:
                prefix_text = ""

        unescaped, values_spaced, values_joined = normalize_cross_plane_prefix(prefix_text)
        views = [v for v in (prefix_text, unescaped, values_spaced, values_joined) if v]

        for view in views:
            for rgx in self.regexes:
                match = rgx.search(view)
                if match:
                    self.is_aborted = True
                    self.last_reason = f"CROSS_PLANE_HARMFUL_MATERIALIZATION: '{match.group(0)}'"
                    return EscalationDecision(
                        aborted=True,
                        reason=self.last_reason,
                        inspected_text=prefix_text,
                        tokens_evaluated=len(self.generated_token_ids),
                    )

        if self.semantic_judge is not None:
            judge_input = values_joined or values_spaced or unescaped or prefix_text
            if hasattr(self.semantic_judge, "evaluate"):
                is_harmful, reason = self.semantic_judge.evaluate(prefix_text)
                if not is_harmful and judge_input != prefix_text:
                    is_harmful, reason = self.semantic_judge.evaluate(judge_input)
            else:
                is_harmful, reason = self.semantic_judge(prefix_text)

            if is_harmful:
                self.is_aborted = True
                self.last_reason = f"SEMANTIC_SAFETY_JUDGE: {reason}"
                return EscalationDecision(
                    aborted=True,
                    reason=self.last_reason,
                    inspected_text=prefix_text,
                    tokens_evaluated=len(self.generated_token_ids),
                )

        return EscalationDecision(
            aborted=False,
            inspected_text=prefix_text,
            tokens_evaluated=len(self.generated_token_ids),
        )

    def finalize(self, layer1_elevated: bool = True) -> EscalationDecision:
        """
        Forces a final end-of-sequence inspection over any trailing tokens
        that fell between check_interval boundaries before EOS/max_new_tokens.
        """
        if self.is_aborted:
            return EscalationDecision(aborted=True, reason=self.last_reason or "PREVIOUSLY_ABORTED")
        if not self.generated_token_ids:
            return EscalationDecision(aborted=False, tokens_evaluated=0)
        if not (layer1_elevated or self.cumulative_spikes >= 1):
            return EscalationDecision(aborted=False, tokens_evaluated=len(self.generated_token_ids))
        return self._inspect_buffer()

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
        1. Skips decoding/inspection if neither L1 risk nor L2 spikes are active (0ms overhead).
        2. Evaluates normalized raw + leaf-concatenated views periodically or on is_final=True.
        """
        if self.is_aborted:
            return EscalationDecision(aborted=True, reason="PREVIOUSLY_ABORTED")

        self.generated_token_ids.append(token_id)
        self.step_count += 1

        if layer2_killed_mass >= self.spike_cutoff:
            self.consecutive_spikes += 1
            self.cumulative_spikes += 1
        else:
            self.consecutive_spikes = 0

        risk_active = (
            layer1_elevated
            or (self.consecutive_spikes >= self.consecutive_spike_threshold)
            or (self.cumulative_spikes >= self.cumulative_spike_threshold)
        )
        should_inspect = risk_active and (self.step_count % self.check_interval == 0 or is_final)

        if not should_inspect:
            return EscalationDecision(
                aborted=False,
                tokens_evaluated=len(self.generated_token_ids),
            )

        return self._inspect_buffer()
