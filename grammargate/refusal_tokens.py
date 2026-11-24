"""
Resolves refusal onset token IDs and refusal prefix masks from model tokenizers.
"""

from __future__ import annotations
from typing import Any, Iterable, Set
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


class RefusalTokenResolver:
    """
    Extracts vocabulary token IDs that represent the beginning of a refusal response.
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

        if custom_token_ids:
            self.refusal_token_ids.update(custom_token_ids)

        self._resolve_tokens()

    def _resolve_tokens(self) -> None:
        if self.tokenizer is None:
            return

        # Add EOS tokens only if explicitly requested (structural EOS != safety refusal)
        if self.include_eos and hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, int):
                self.refusal_token_ids.add(self.tokenizer.eos_token_id)
            elif isinstance(self.tokenizer.eos_token_id, list):
                self.refusal_token_ids.update(self.tokenizer.eos_token_id)

        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        if vocab_size is None and hasattr(self.tokenizer, "get_vocab"):
            vocab_size = len(self.tokenizer.get_vocab())

        for phrase in self.phrases:
            # Tokenize phrase directly and with leading space / variations / newline
            for text_variant in [phrase, f" {phrase}", phrase.lower(), f" {phrase.lower()}"]:
                try:
                    if hasattr(self.tokenizer, "encode"):
                        ids = self.tokenizer.encode(text_variant, add_special_tokens=False)
                    else:
                        ids = []
                    if ids:
                        # Onset token is the first token of the phrase
                        self.refusal_token_ids.add(ids[0])
                except Exception:
                    continue

            # Also check newline variant (e.g. \nI or \nSorry)
            try:
                if hasattr(self.tokenizer, "encode"):
                    nl_ids = self.tokenizer.encode(f"\n{phrase}", add_special_tokens=False)
                    if len(nl_ids) > 1:
                        self.refusal_token_ids.add(nl_ids[1])
            except Exception:
                pass

    def get_mask(self, vocab_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """
        Returns a boolean tensor of shape (vocab_size,) where True indicates a refusal token.
        """
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        valid_ids = [idx for idx in self.refusal_token_ids if 0 <= idx < vocab_size]
        if valid_ids:
            mask[valid_ids] = True
        return mask
