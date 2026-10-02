"""Fill-in-the-middle (docs/api.md#fill-in-the-middle).

Code models such as Qwen2.5 are trained to fill a gap between a prefix and a suffix: given the tokens
``<fim_prefix> prefix <fim_suffix> suffix <fim_middle>`` they write the middle. The FIM tokens are found in the
model's own vocabulary under either spelling (``<|fim_prefix|>`` or ``<fim_prefix>``), the prompt is built from token
ids (the prefix and suffix are encoded separately), and the middle ends at the model's stop tokens or at the first
token that starts another FIM part, pads, or begins a new file or text (:data:`ENDS`). Nothing here depends on
anything but the tokenizer, so a middle is as exact as any other completion.
"""

from __future__ import annotations

from dataclasses import dataclass

from etalii_dllm.tokenization import Tokenizer

PARTS = ("fim_prefix", "fim_suffix", "fim_middle")
ENDS = ("fim_prefix", "fim_suffix", "fim_middle", "fim_pad", "file_sep", "repo_name", "endoftext")
"""Tokens that end a middle (in either spelling), besides the model's own stop tokens."""


def _token(tokenizer: Tokenizer, name: str) -> int | None:
    lookup = getattr(tokenizer, "token_to_id", None)
    if lookup is None:
        return None
    for spelling in (f"<|{name}|>", f"<{name}>"):
        token = lookup(spelling)
        if token is not None:
            return int(token)
    return None


@dataclass(frozen=True)
class FimTokens:
    """A model's fill-in-the-middle tokens."""

    prefix: int
    suffix: int
    middle: int
    ends: frozenset[int]
    """The tokens that end a middle besides the model's stop tokens (:data:`ENDS` found in the vocabulary)."""

    @classmethod
    def of(cls, tokenizer: Tokenizer) -> FimTokens | None:
        """The tokenizer's FIM tokens, or ``None`` when it lacks any of the three parts."""
        prefix, suffix, middle = (_token(tokenizer, name) for name in PARTS)
        if prefix is None or suffix is None or middle is None:
            return None
        ends = (_token(tokenizer, name) for name in ENDS)
        return cls(prefix, suffix, middle, frozenset(t for t in ends if t is not None))

    def prompt(self, tokenizer: Tokenizer, prefix: str, suffix: str) -> list[int]:
        """The prompt tokens: ``<fim_prefix> prefix <fim_suffix> suffix <fim_middle>``."""
        return [self.prefix, *tokenizer.encode(prefix), self.suffix, *tokenizer.encode(suffix), self.middle]


def fim_tokens(tokenizer: Tokenizer) -> FimTokens:
    """:meth:`FimTokens.of`, raising ``ValueError`` for a model without FIM tokens."""
    tokens = FimTokens.of(tokenizer)
    if tokens is None:
        raise ValueError("this model has no fill-in-the-middle tokens (<|fim_prefix|>, <|fim_suffix|>, <|fim_middle|>)")
    return tokens
