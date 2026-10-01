"""The logit lens: what the model would predict if it stopped after each layer.

The residual stream after layer ``l`` is put through the final norm and the LM head (with the model's own logits
scaling and soft-cap), exactly as the last layer's output is in a normal forward pass; so the lens of the last layer
is the model's real prediction, bit for bit. Rankings use a total order (higher probability first, ties on the lower
token id), so the output is reproducible byte for byte.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from etalii_dllm.interpret.trace import Trace, trace
from etalii_dllm.numerics import softmax
from etalii_dllm.transformer import Transformer


def top_k(values: np.ndarray, k: int) -> list[int]:
    """Indices of the ``k`` largest ``values``, largest first, ties broken on the lower index (a total order)."""
    order = np.lexsort((np.arange(values.shape[0]), -np.asarray(values, dtype=np.float64)))
    return [int(i) for i in order[:k]]


@dataclass(frozen=True)
class Prediction:
    token: int
    probability: float


@dataclass(frozen=True)
class Lens:
    """``predictions[layer][position]`` lists the top tokens after ``layer`` layers (0: the embeddings alone,
    ``L``: the full model) at each position."""

    tokens: tuple[int, ...]
    predictions: list[list[list[Prediction]]]

    @property
    def layers(self) -> int:
        return len(self.predictions) - 1


def logit_lens(model: Transformer, tokens: Sequence[int], top: int = 5, recorded: Trace | None = None) -> Lens:
    """The top ``top`` next-token predictions after every layer and position of ``tokens``."""
    if top < 1:
        raise ValueError("top must be at least 1")
    recorded = recorded or trace(model, tokens, attention=False, logits=False)
    predictions = []
    for stream in recorded.residual:
        logits = model.logits_from_hidden(model.final_norm(stream))
        rows = []
        for row in logits:
            probabilities = softmax(row)
            rows.append([Prediction(i, float(probabilities[i])) for i in top_k(probabilities, top)])
        predictions.append(rows)
    return Lens(recorded.tokens, predictions)
