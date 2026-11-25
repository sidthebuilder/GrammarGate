"""
Resolves refusal onset token IDs and SpecLook-Gate partitioned masks from model tokenizers.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, Set
import torch

DEFAULT_REFUSAL_PHRASES = [
    "I cannot",
    "I can't",
    "I am sorry",
    "I'm sorry",
    "Sorry",
    "sorry",
    "Cannot",
    "cannot",
    "As an AI",
    "As a language model",
    "As an artificial",
    "I am unable",
    "I'm unable",
    "My apologies",
    "Unfortunately",
    "unfortunately",
    "It is illegal",
    "It would be unethical",
    "I must refuse",
    "I am programmed to be",
    "I will not generate",
]

STRICT_REFUSAL_STEMS = {
    "sorry",
    "cannot",
    "can't",
    "unable",
    "apologies",
    "apologize",
    "apologise",
    "unfortunately",
    "refuse",
    "illegal",
    "unethical",
    "prohibited",
}

REFUSAL_CONTINUATION_LEXEMES = {
    "cannot",
    "can't",
    "'t",
    "sorry",
    "unable",
    "refuse",
    "apologies",
    "apologize",
    "apologise",
    "illegal",
    "unethical",
    "prohibited",
    "not",
    "must",
    "ai",
    "language",
    "artificial",
}

WEAK_BRIDGE_LEXEMES = {"am", "can", "will", "is", "would", "an", "a", "'m"}

BENIGN_TRANSITION_PHRASES = [
    "Sure",
    "Certainly",
    "Here",
    "To",
    "This",
    "In",
    "Note",
    "Example",
    "First",
    "Below",
    "#",
    "```",
    '"""',
]


def _decode_token_safe(tokenizer: Any, token_id: int) -> str:
    if hasattr(tokenizer, "decode"):
        try:
            return str(tokenizer.decode([token_id]))
        except Exception:
            pass
    inv_vocab = getattr(tokenizer, "inv_vocab", None)
    if isinstance(inv_vocab, dict) and token_id in inv_vocab:
        return str(inv_vocab[token_id])
    return ""


class RefusalTokenResolver:
    """
    Extracts vocabulary token IDs that represent refusal responses and partitions
    them into unambiguous refusal tokens, ambiguous unigram onsets (e.g. 'I', 'It', 'As'),
    refusal bigram continuations, and benign prose/code transition tokens.
    """

    def __init__(
        self,
        tokenizer: Any,
        phrases: Iterable[str] | None = None,
        custom_token_ids: Iterable[int] | None = None,
        include_eos: bool = False,
    ):
        self.tokenizer = tokenizer
        self.phrases = list(phrases or DEFAULT_REFUSAL_PHRASES)
        self.include_eos = include_eos
        self.refusal_token_ids: Set[int] = set()
        self.strict_refusal_ids: Set[int] = set()
        self.ambiguous_onset_ids: Set[int] = set()
        self.benign_transition_ids: Set[int] = set()
        self.continuation_map: Dict[int, Set[int]] = {}
        self.bridge_continuations: Dict[int, Set[int]] = {}
        self.all_continuation_ids: Set[int] = set()

        if custom_token_ids:
            custom_set = set(custom_token_ids)
            self.refusal_token_ids.update(custom_set)
            self.strict_refusal_ids.update(custom_set)

        self._resolve_tokens()

    def _classify_onset(self, token_id: int, full_ids: list[int]) -> None:
        self.refusal_token_ids.add(token_id)
        decoded = _decode_token_safe(self.tokenizer, token_id).strip().lower()
        is_strict = decoded in STRICT_REFUSAL_STEMS or (len(full_ids) == 1 and len(decoded) >= 4)
        if is_strict:
            self.strict_refusal_ids.add(token_id)
            self.ambiguous_onset_ids.discard(token_id)
        elif token_id not in self.strict_refusal_ids:
            self.ambiguous_onset_ids.add(token_id)

        if len(full_ids) >= 2:
            second = full_ids[1]
            second_dec = _decode_token_safe(self.tokenizer, second).strip().lower()
            # Keep direct high-signal continuations vs multi-hop bridges ('I' -> ' am' -> ' sorry')
            if len(full_ids) == 2 or second_dec in REFUSAL_CONTINUATION_LEXEMES or second_dec not in WEAK_BRIDGE_LEXEMES:
                self.continuation_map.setdefault(token_id, set()).add(second)
                self.all_continuation_ids.add(second)
            if len(full_ids) >= 3:
                self.bridge_continuations.setdefault(second, set()).add(full_ids[2])
                self.continuation_map.setdefault(token_id, set()).add(full_ids[2])
                self.all_continuation_ids.add(full_ids[2])

    def _resolve_tokens(self) -> None:
        if self.tokenizer is None:
            return

        if self.include_eos and hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, int):
                self.refusal_token_ids.add(self.tokenizer.eos_token_id)
                self.strict_refusal_ids.add(self.tokenizer.eos_token_id)
            elif isinstance(self.tokenizer.eos_token_id, list):
                self.refusal_token_ids.update(self.tokenizer.eos_token_id)
                self.strict_refusal_ids.update(self.tokenizer.eos_token_id)

        for phrase in self.phrases:
            for text_variant in [phrase, f" {phrase}", phrase.lower(), f" {phrase.lower()}"]:
                try:
                    ids = (
                        self.tokenizer.encode(text_variant, add_special_tokens=False)
                        if hasattr(self.tokenizer, "encode")
                        else []
                    )
                    if ids:
                        self._classify_onset(ids[0], list(ids))
                except Exception:
                    continue

            try:
                if hasattr(self.tokenizer, "encode"):
                    nl_ids = self.tokenizer.encode(f"\n{phrase}", add_special_tokens=False)
                    if len(nl_ids) > 1:
                        self._classify_onset(nl_ids[1], list(nl_ids[1:]))
            except Exception:
                pass

        for phrase in BENIGN_TRANSITION_PHRASES:
            for variant in [phrase, f" {phrase}", f"\n{phrase}"]:
                try:
                    if hasattr(self.tokenizer, "encode"):
                        ids = self.tokenizer.encode(variant, add_special_tokens=False)
                        if ids and ids[0] not in self.strict_refusal_ids and ids[0] not in self.ambiguous_onset_ids:
                            self.benign_transition_ids.add(ids[0])
                except Exception:
                    continue

    def _build_mask(self, token_ids: Iterable[int], vocab_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        valid_ids = [idx for idx in token_ids if 0 <= idx < vocab_size]
        if valid_ids:
            mask[valid_ids] = True
        return mask

    def get_mask(self, vocab_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """
        Returns a boolean tensor of shape (vocab_size,) where True indicates any refusal token.
        """
        return self._build_mask(self.refusal_token_ids, vocab_size, device=device)

    def get_partitioned_masks(
        self, vocab_size: int, device: torch.device | str = "cpu"
    ) -> dict[str, torch.Tensor]:
        """
        Returns partitioned boolean masks for SpecLook-Gate:
        - strict: unambiguous refusal tokens
        - ambiguous: short/overloaded unigram onsets requiring 2-token continuation check
        - benign_transition: prose/markdown/code comment transitions
        - combined: strict | ambiguous
        """
        return {
            "strict": self._build_mask(self.strict_refusal_ids, vocab_size, device=device),
            "ambiguous": self._build_mask(self.ambiguous_onset_ids, vocab_size, device=device),
            "benign_transition": self._build_mask(self.benign_transition_ids, vocab_size, device=device),
            "combined": self.get_mask(vocab_size, device=device),
        }
