"""Expert routing of mixture-of-experts models: which experts every token was sent to, and how often each is used.

The routing comes from a traced forward pass (:func:`etalii_dllm.interpret.trace.trace`), so it is exactly the routing
the model computes when it generates: the same experts and the same weight bits on every run, thread count and
machine, whatever else is in the batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from etalii_dllm.interpret.trace import trace
from etalii_dllm.transformer import Transformer


@dataclass(frozen=True)
class Routing:
    """The routing of ``tokens`` through every mixture-of-experts layer."""

    tokens: tuple[int, ...]
    layers: tuple[int, ...]
    """The 0-based mixture-of-experts layers (dense layers are left out)."""
    experts: np.ndarray
    """``[len(layers), positions, k]`` int64: the chosen experts of each position, in rank order."""
    weights: np.ndarray
    """``[len(layers), positions, k]`` float32: their weights."""
    expert_count: int

    def usage(self) -> np.ndarray:
        """``[len(layers), experts]`` int64: how many positions each expert was chosen for, per layer."""
        counts = np.zeros((len(self.layers), self.expert_count), dtype=np.int64)
        for i in range(len(self.layers)):
            counts[i] = np.bincount(self.experts[i].reshape(-1), minlength=self.expert_count)
        return counts


def routing(model: Transformer, tokens: Sequence[int]) -> Routing:
    """The expert routing of ``tokens`` (CPU); raises ``ValueError`` for a dense model."""
    config = model.config
    if not config.experts:
        raise ValueError("the model has no experts (it is not a mixture-of-experts model)")
    recorded = trace(model, tokens, attention=False, logits=False)
    assert recorded.experts is not None and recorded.expert_weights is not None
    layers = tuple(layer for layer in range(config.layers) if config.is_sparse(layer))
    return Routing(
        tokens=tuple(int(t) for t in tokens),
        layers=layers,
        experts=np.ascontiguousarray(recorded.experts[list(layers)]),
        weights=np.ascontiguousarray(recorded.expert_weights[list(layers)]),
        expert_count=config.experts,
    )
