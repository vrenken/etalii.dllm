"""Language models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from etalii_dllm.numerics import FloatArray, fill_gaussian, fingerprint


class LanguageModel(Protocol):
    """An autoregressive language model. Implementations must be pure: the logits depend only on the weights and
    the given tokens, and are bit-identical on every run on the same hardware."""

    @property
    def id(self) -> str: ...

    @property
    def vocabulary_size(self) -> int: ...

    def forward(self, tokens: Sequence[int]) -> FloatArray:
        """Returns the next-token logits for the given context."""
        ...


class BigramModel:
    """The smallest possible language model: a table of next-token logits indexed by the previous token,
    initialised from a seed. It exercises the whole deterministic pipeline (tokenizer, forward pass, sampler, API)
    until imported transformer weights replace it."""

    def __init__(self, vocabulary_size: int, seed: int) -> None:
        if vocabulary_size < 1:
            raise ValueError("vocabulary_size must be positive")
        self.vocabulary_size = vocabulary_size
        self.seed = seed
        self._table = fill_gaussian(seed, vocabulary_size * vocabulary_size).reshape(vocabulary_size, vocabulary_size)
        self._table.flags.writeable = False

    @property
    def id(self) -> str:
        return f"dllm-bigram-{self.vocabulary_size}-{self.seed}"

    @property
    def weights_fingerprint(self) -> str:
        """Fingerprint of the weights, suitable as an OpenAI style ``system_fingerprint``."""
        return fingerprint(self._table)

    def forward(self, tokens: Sequence[int]) -> FloatArray:
        previous = tokens[-1] if tokens else 0
        return np.array(self._table[previous], dtype=np.float32)

    def hidden_states(self, tokens: Sequence[int]) -> FloatArray:
        """One row per position (the logits row of that token), for embeddings."""
        return np.array(self._table[np.asarray(tokens, dtype=np.int64)], dtype=np.float32)
