"""Activation steering: a direction in the residual stream, found by contrasting prompts, added during generation.

A steering vector for layer ``l`` is the mean residual stream after ``l`` over the positive prompts minus that over
the negative ones (contrastive activation addition). Each prompt's mean over its positions and the mean over prompts
are ``column_mean`` kernels (double sums in a fixed order), so the vector is the same bits on every machine.
Applying it adds ``strength * vector`` (one float32 multiply, then a float32 add per element) to the residual stream
after layer ``l`` at every position; a steered model is a different model, with its own ``system_fingerprint``.

For a T5 text-to-text model the vector steers the decoder (#405): each prompt is read by the decoder as the answer to
an empty source (just ``</s>``), and its mean is over the answer's positions (the start token, the same for every
prompt, is left out), so the vector lives in the residual stream the decoder adds it to when it writes.

Files are JSON with each float32 value written as its exact double, so saving and loading never changes a bit.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm.interpret.text_to_text import trace_text_to_text
from etalii_dllm.interpret.trace import trace
from etalii_dllm.numerics import column_mean
from etalii_dllm.seq2seq import TextToText
from etalii_dllm.tokenization import Tokenizer
from etalii_dllm.transformer import Transformer


def steered_layers(model: Transformer | TextToText) -> int:
    """The layers a steering vector can follow: the decoder's (a T5 model's decoder layers)."""
    return model.config.decoder_layers if isinstance(model, TextToText) else model.config.layers


FORMAT = "dllm-steering"


@dataclass(frozen=True)
class SteeringVector:
    """A direction to add after ``layer`` (1-based: the output of the ``layer``-th decoder layer)."""

    layer: int
    vector: np.ndarray
    strength: float = 4.0
    model_fingerprint: str = ""
    """The weights fingerprint of the model it was built from (empty when unknown)."""
    origin: dict[str, Any] = field(default_factory=dict)
    """How it was made (the prompts), for the record."""

    def scaled(self, strength: float | None = None) -> np.ndarray:
        """``strength * vector`` in float32 (the file's strength unless one is given)."""
        factor = np.float32(self.strength if strength is None else strength)
        return (self.vector * factor).astype(np.float32)

    def to_json(self) -> str:
        payload = {
            "format": FORMAT,
            "layer": self.layer,
            "strength": self.strength,
            "model_fingerprint": self.model_fingerprint,
            "origin": self.origin,
            "vector": [float(v) for v in self.vector],
        }
        return json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> SteeringVector:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}: {error}") from error
        if not isinstance(payload, dict) or payload.get("format") != FORMAT:
            raise ValueError(f"{path}: not a steering vector file")
        return cls(
            layer=int(payload["layer"]),
            vector=np.asarray(payload["vector"], dtype=np.float32),
            strength=float(payload.get("strength", 4.0)),
            model_fingerprint=str(payload.get("model_fingerprint", "")),
            origin=dict(payload.get("origin") or {}),
        )

    def for_model(self, model: Transformer | TextToText, strength: float | None = None) -> dict[int, np.ndarray]:
        """The ``steering`` argument of :class:`Transformer` or :class:`TextToText`: ``{layer index: strength *
        vector}``."""
        layers = steered_layers(model)
        if not 1 <= self.layer <= layers:
            raise ValueError(f"steering layer {self.layer} is not between 1 and {layers}")
        if self.vector.shape != (model.config.hidden_size,):
            raise ValueError(f"steering vector has {self.vector.shape[0]} values, the model {model.config.hidden_size}")
        return {self.layer - 1: self.scaled(strength)}


def mean_activation(
    model: Transformer | TextToText, tokenizer: Tokenizer, prompts: Sequence[str], layer: int
) -> np.ndarray:
    """The residual stream after ``layer`` (1-based), averaged over each prompt's positions and then over prompts
    (a T5 model: the decoder's, over the prompt read as the answer to an empty source)."""
    if not prompts:
        raise ValueError("needs at least one prompt")
    means = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            raise ValueError(f"prompt {prompt!r} has no tokens")
        if isinstance(model, TextToText):
            if model.end_of_source in tokens:
                raise ValueError(f"prompt {prompt!r} contains </s>, which ends a text-to-text answer")
            recorded = trace_text_to_text(model, [model.end_of_source, *tokens], attention=False, logits=False)
            means.append(column_mean(recorded.residual[layer][1:]))
        else:
            means.append(column_mean(trace(model, tokens, attention=False, logits=False).residual[layer]))
    return column_mean(np.stack(means))


def build_steering_vector(
    model: Transformer | TextToText,
    tokenizer: Tokenizer,
    positive: Sequence[str],
    negative: Sequence[str],
    layer: int,
    strength: float = 4.0,
) -> SteeringVector:
    """Mean activation after ``layer`` over ``positive`` minus that over ``negative`` (elementwise float32)."""
    layers = steered_layers(model)
    if not 1 <= layer <= layers:
        raise ValueError(f"layer must be between 1 and {layers}")
    if model.steering:
        raise ValueError("build steering vectors on an unsteered model")
    vector = mean_activation(model, tokenizer, positive, layer) - mean_activation(model, tokenizer, negative, layer)
    return SteeringVector(
        layer,
        vector.astype(np.float32),
        strength,
        model.weights_fingerprint,
        {"positive": list(positive), "negative": list(negative)},
    )
