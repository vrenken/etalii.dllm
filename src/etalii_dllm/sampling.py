"""Deterministic token sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from etalii_dllm.numerics import DeterministicRandom, FloatArray, argmax, softmax


@dataclass(frozen=True)
class SamplingOptions:
    """Controls token selection.

    With the same options, model and prompt the generated tokens are always identical: a temperature above zero
    draws from a seeded ``DeterministicRandom``, never from ambient entropy.
    """

    temperature: float = 0.0
    """0 means greedy decoding."""
    top_k: int = 0
    """Keep only the K most likely tokens; 0 disables."""
    top_p: float = 1.0
    """Nucleus sampling threshold in (0, 1]; 1 disables."""
    seed: int = 0
    """Seed for the sampler's random stream (unsigned 64-bit)."""

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")


GREEDY = SamplingOptions()


class Sampler:
    """Candidates are ordered by (probability descending, token id ascending), a total order, so ties never
    depend on sort stability or hardware."""

    def __init__(self, options: SamplingOptions) -> None:
        self._options = options
        self._random = DeterministicRandom(options.seed & 0xFFFFFFFFFFFFFFFF)

    def sample(self, logits: FloatArray) -> int:
        options = self._options
        if options.temperature == 0:
            return argmax(logits)

        scaled = (np.asarray(logits, dtype=np.float32) / np.float32(options.temperature)).astype(np.float32)
        probabilities = softmax(scaled).tolist()
        order = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))

        keep = len(order)
        if options.top_k > 0:
            keep = min(keep, options.top_k)
        if options.top_p < 1:
            cumulative = 0.0
            for i in range(keep):
                cumulative += probabilities[order[i]]
                if cumulative >= options.top_p:
                    keep = i + 1
                    break

        total = 0.0
        for i in range(keep):
            total += probabilities[order[i]]
        target = self._random.next_double() * total
        running = 0.0
        for i in range(keep):
            running += probabilities[order[i]]
            if target < running:
                return order[i]
        return order[keep - 1]
