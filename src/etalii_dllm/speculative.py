"""Speculative decoding that changes no bit.

Plain decoding runs one forward pass per token. Speculative decoding guesses the next few tokens cheaply (a *draft*),
runs the real model once over all of them, and keeps the guesses the model agrees with. Here the result is not merely
the same distribution but the same tokens, logprobs and text as plain decoding:

- One pass over ``context + draft`` gives, at every position, the logits that decoding the tokens one at a time would
  give, bit for bit (every kernel computes each row on its own, in one order; see ``docs/kernels.md``).
- The sampler is asked for the token at each position in turn, from those logits, with that position's own draw
  from its random stream. A drafted token is kept only when it *is* that token; at the first disagreement the model's
  own token is used and the rest of the draft is dropped. The sampler sees exactly the calls plain decoding makes.

So speculation only changes how much work is done. Drafts come from the text so far (:class:`PromptLookup`: the
tokens that followed the last earlier occurrence of the current ending, which pays off on code, quotes, edits and
structured output) or from a small model with the same tokenizer (:class:`DraftModel`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from etalii_dllm.numerics import argmax

DEFAULT_DRAFT_TOKENS = 8
"""Tokens drafted per step when speculation is on (``--speculate`` without a number)."""


class Drafter(Protocol):
    """Proposes the next tokens of a sequence. A drafter belongs to one generation and may keep state about it."""

    def propose(self, context: Sequence[int], count: int) -> list[int]: ...


class PromptLookup:
    """Drafts by looking the sequence's own ending up in the sequence: the last ``n`` tokens (``n`` from
    ``max_ngram`` down to ``min_ngram``) are matched against their most recent earlier occurrence, and the tokens
    that followed it are the draft. Single tokens repeat too often to predict much, so the default needs two. Pure
    function of the tokens; the n-gram index grows with the sequence."""

    def __init__(self, max_ngram: int = 3, min_ngram: int = 2) -> None:
        if not 1 <= min_ngram <= max_ngram:
            raise ValueError("need 1 <= min_ngram <= max_ngram")
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram
        self._indexed = 0
        self._latest: dict[tuple[int, ...], int] = {}
        """The latest start of each n-gram that has at least one token after it."""

    def propose(self, context: Sequence[int], count: int) -> list[int]:
        if count <= 0:
            return []
        if len(context) < self._indexed:  # a different sequence: start over
            self._indexed, self._latest = 0, {}
        for end in range(max(self._indexed, 1), len(context)):  # n-grams ending just before position `end`
            for n in range(self.min_ngram, self.max_ngram + 1):
                if end - n >= 0:
                    self._latest[tuple(context[end - n : end])] = end - n
        self._indexed = len(context)
        for n in range(min(self.max_ngram, len(context) - 1), self.min_ngram - 1, -1):
            start = self._latest.get(tuple(context[len(context) - n :]))
            if start is not None:
                return list(context[start + n : start + n + count])
        return []


class DraftModel:
    """Drafts greedily with a smaller model that uses the same tokenizer, with its own KV cache. A text-to-text
    draft model (:class:`etalii_dllm.seq2seq.TextToText`, #413) drafts the answer and stops before ``</s>``, which
    ends it."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._cache = model.new_cache()
        self._end = getattr(model, "end_of_source", None)

    def propose(self, context: Sequence[int], count: int) -> list[int]:
        draft: list[int] = []
        tokens = list(context)
        for _ in range(count):
            token = argmax(self.model.forward_cached(tokens, self._cache))
            if not 0 <= token < self.model.vocabulary_size or token == self._end:
                break
            draft.append(token)
            tokens.append(token)
        return draft
