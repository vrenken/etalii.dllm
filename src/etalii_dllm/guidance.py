"""Exact guided decoding (docs/api.md#guided-decoding): decoding from a combination of several next-token
distributions, each combination an exactly defined float32 computation.

Every guide sees the request's own float32 logits ``l`` after each token, and returns the logits decoding uses
instead; logit bias, penalties, the watermark, temperature and the sampler then apply to those as usual. Reported
log-probabilities stay those of the request's own logits.

- **Classifier-free guidance** (:class:`NegativePrompt`; Sanchez et al. 2023): a second context, the negative prompt,
  runs on the same model and advances with the same chosen tokens; with its logits ``n``, decoding uses
  ``f32(n + f32(f32(scale) * f32(l - n)))``.
- **Contrastive decoding** (:class:`Contrast`; Li et al. 2023, O'Brien and Lewis 2023): a smaller amateur model with
  the same tokenizer runs on the same context; with ``p = softmax(l)`` (the kernel) and the amateur's logits ``a``,
  tokens with ``p < alpha * max(p)`` (a double product) get ``-inf`` and the rest
  ``f32(f32(f32(1 + beta) * l) - f32(f32(beta) * a))``.
- **Ensembles** (:class:`Ensemble`): models with the same tokenizer run on the same context; decoding uses
  ``f32(sum_i (w_i / W) * log_softmax_i)`` with the sum in double in model order (the served model first) and ``W``
  the sum of the weights.

A request uses at most one of them. Each guide keeps its own KV cache, so it costs one more forward pass per token
and model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from etalii_dllm.numerics import log_softmax, softmax


def guided(logits: np.ndarray, negative: np.ndarray, scale: float) -> np.ndarray:
    """Classifier-free guidance of float32 ``logits`` away from the ``negative`` context's."""
    return negative + np.float32(scale) * (logits - negative)


def contrasted(logits: np.ndarray, amateur: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Contrastive decoding of the expert's float32 ``logits`` against the ``amateur``'s."""
    probabilities = softmax(logits)
    limit = float(alpha) * float(probabilities.max())
    combined = np.float32(1.0 + beta) * logits - np.float32(beta) * amateur
    return np.where(probabilities.astype(np.float64) >= limit, combined, np.float32(-np.inf)).astype(np.float32)


def ensembled(members: Sequence[tuple[np.ndarray, float]]) -> np.ndarray:
    """The weighted mean of the members' log-probabilities (logits, weight), in double in member order."""
    total_weight = 0.0
    for _, weight in members:
        total_weight += float(weight)
    total: np.ndarray | None = None
    for logits, weight in members:
        term = (float(weight) / total_weight) * log_softmax(logits).astype(np.float64)
        total = term if total is None else total + term
    assert total is not None
    return total.astype(np.float32)


class Guide(Protocol):
    def combine(self, logits: np.ndarray, generated: Sequence[int]) -> np.ndarray:
        """The logits to decode from, given the request's own ``logits`` after the ``generated`` tokens."""

    def length(self, generated: Sequence[int]) -> int:
        """How many tokens the guide's longest context holds after ``generated``."""


class _Context:
    """A model on a fixed prefix plus the generated tokens, with its own KV cache when the model has one."""

    def __init__(self, model: Any, prefix: Sequence[int]) -> None:
        self.model = model
        self.prefix = list(prefix)
        new_cache = getattr(model, "new_cache", None)
        self.cache = new_cache() if new_cache is not None else None

    def logits(self, generated: Sequence[int]) -> np.ndarray:
        tokens = [*self.prefix, *generated]
        cache = self.cache
        values = self.model.forward(tokens) if cache is None else self.model.forward_cached(tokens, cache)
        return np.asarray(values, dtype=np.float32).reshape(-1)


class NegativePrompt:
    def __init__(self, model: Any, negative: Sequence[int], scale: float) -> None:
        if not negative:
            raise ValueError("the negative prompt needs at least one token")
        self._context = _Context(model, negative)
        self._scale = float(scale)

    def combine(self, logits: np.ndarray, generated: Sequence[int]) -> np.ndarray:
        return guided(logits, self._context.logits(generated), self._scale)

    def length(self, generated: Sequence[int]) -> int:
        return len(self._context.prefix) + len(generated)


class Contrast:
    def __init__(self, amateur: Any, context: Sequence[int], alpha: float, beta: float) -> None:
        self._context = _Context(amateur, context)
        self._alpha, self._beta = float(alpha), float(beta)

    def combine(self, logits: np.ndarray, generated: Sequence[int]) -> np.ndarray:
        return contrasted(logits, self._context.logits(generated), self._alpha, self._beta)

    def length(self, generated: Sequence[int]) -> int:
        return len(self._context.prefix) + len(generated)


class Ensemble:
    def __init__(self, members: Sequence[tuple[Any, float]], weight: float, context: Sequence[int]) -> None:
        self._weight = float(weight)
        self._members = [(_Context(model, context), float(w)) for model, w in members]
        self._length = len(context)

    def combine(self, logits: np.ndarray, generated: Sequence[int]) -> np.ndarray:
        rows = [(logits, self._weight), *((context.logits(generated), w) for context, w in self._members)]
        return ensembled(rows)

    def length(self, generated: Sequence[int]) -> int:
        return self._length + len(generated)
