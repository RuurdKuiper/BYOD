"""Text-generation quality metrics shared by training and evaluation."""
from __future__ import annotations

from typing import Any


def distinct_n(text: str, tokenizer: Any, n: int) -> float:
    """Return sliding token-level Distinct-n, excluding special tokens.

    Unlike the former repetition metric, adjacent windows overlap: for
    ``[A, B, C]`` the bigrams are ``(A, B)`` and ``(B, C)``.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    tokens = [
        token
        for token in tokenizer.encode(text, add_special_tokens=False)
        if token not in special_ids
    ]
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return float(len(set(grams)) / len(grams)) if grams else 0.0
