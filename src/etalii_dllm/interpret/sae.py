"""Sparse autoencoders: interpretable features of one layer's residual stream, trained reproducibly.

A sparse autoencoder (SAE) rewrites a residual stream vector ``x`` (scaled so its mean squared norm is the hidden
size) as a sparse, non-negative mix of learned directions:

    f = relu(W_e (x - b_d) + b_e)        x_hat = W_d f + b_d

trained to minimise ``mean ||x - x_hat||^2 + l1 * mean sum f`` with the decoder's columns kept at unit length, so
each feature is one direction and its activation says how much of it is present. Every product and sum is a
deterministic kernel (``linear``, ``linear_backward``, ``sum_squares``, ``sum``), the update is the ``adamw_step``
kernel, the decoder is initialised from the seeded Gaussian generator and batches follow a seeded shuffle; so equal
runs write byte-identical SAE files on every machine. Files are safetensors with the settings in the metadata.

For a T5 text-to-text model (#408) the SAE reads the decoder's residual stream: each text is read by the decoder as
the answer to an empty source (just ``</s>``), as steering vectors are built, and its rows are the answer's tokens
(the start token, the same for every text, is left out). A feature's direction then steers the decoder.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from etalii_dllm.interpret.lens import top_k
from etalii_dllm.interpret.steering import SteeringVector, steered_layers
from etalii_dllm.interpret.text_to_text import trace_text_to_text
from etalii_dllm.interpret.trace import trace
from etalii_dllm.numerics import DeterministicRandom, fill_gaussian, linear, linear_backward, sum_, sum_squares
from etalii_dllm.seq2seq import TextToText
from etalii_dllm.tokenization import Tokenizer
from etalii_dllm.transformer import Transformer

_MASK64 = (1 << 64) - 1
_NAMES = ("decoder.bias", "decoder.weight", "encoder.bias", "encoder.weight")


@dataclass(frozen=True)
class SaeConfig:
    features: int = 1024
    l1: float = 5.0
    """Weight of the sparsity penalty (mean over the batch of the summed feature activations)."""
    learning_rate: float = 1e-3
    steps: int = 2000
    batch_size: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        if self.features < 1 or self.steps < 1 or self.batch_size < 1:
            raise ValueError("features, steps and batch size must be at least 1")
        if self.l1 < 0 or self.learning_rate <= 0:
            raise ValueError("l1 must not be negative and the learning rate must be positive")


@dataclass(frozen=True)
class Activations:
    """Residual stream rows ``[N, hidden]`` after ``layer`` (1-based) and, per row, its text and token position."""

    layer: int
    values: np.ndarray
    positions: list[tuple[int, int]]
    tokens: list[list[int]]


def collect_activations(
    model: Transformer | TextToText, tokenizer: Tokenizer, texts: Sequence[str], layer: int, *, outlier: float = 10.0
) -> Activations:
    """The residual stream after ``layer`` at every position of every text, in text and position order. Rows whose
    squared norm is more than ``outlier`` times the median are left out (0 keeps all): models park attention on the
    first token or two and their residual stream there is many times larger than elsewhere (an "attention sink"),
    which would otherwise dominate the features. A T5 model's rows are its decoder's, over each text as the answer
    to an empty source."""
    layers = steered_layers(model)
    if not 1 <= layer <= layers:
        raise ValueError(f"layer must be between 1 and {layers}")
    rows, positions, tokens = [], [], []
    for text in texts:
        ids = tokenizer.encode(text)
        if not ids:
            continue
        if isinstance(model, TextToText):
            if model.end_of_source in ids:
                raise ValueError(f"text {text!r} contains </s>, which ends a text-to-text answer")
            answer = [model.end_of_source, *ids]
            rows.append(trace_text_to_text(model, answer, attention=False, logits=False).residual[layer][1:])
        else:
            rows.append(trace(model, ids, attention=False, logits=False).residual[layer])
        positions += [(len(tokens), p) for p in range(len(ids))]
        tokens.append(ids)
    if not rows:
        raise ValueError("the corpus has no tokens")
    values = np.concatenate(rows)
    if outlier > 0:
        squares = _row_squares(values)
        keep = squares <= np.float32(outlier) * np.median(squares)
        values = values[keep]
        positions = [p for p, kept in zip(positions, keep, strict=True) if kept]
    return Activations(layer, np.ascontiguousarray(values), positions, tokens)


def _row_squares(values: np.ndarray) -> np.ndarray:
    """Squared norm of each row, the sum over columns ascending in double (through ``linear``)."""
    squares = np.ascontiguousarray((values * values).astype(np.float32))
    return linear(squares, np.ones((1, values.shape[1]), dtype=np.float32)).numpy()[:, 0]


def _column_sums(values: np.ndarray) -> np.ndarray:
    """Sum of each column of ``values[rows, cols]``, rows ascending in double (through ``linear``)."""
    ones = np.ones((1, values.shape[0]), dtype=np.float32)
    return linear(ones, np.ascontiguousarray(values.T)).numpy()[0]


class SparseAutoencoder:
    def __init__(
        self,
        layer: int,
        params: dict[str, np.ndarray],
        input_scale: float,
        config: SaeConfig,
        model_fingerprint: str = "",
        history: dict[str, Any] | None = None,
    ) -> None:
        self.layer = layer
        self.params = params
        self.input_scale = float(input_scale)
        """Multiplies the residual stream before encoding (makes its mean squared norm the hidden size)."""
        self.config = config
        self.model_fingerprint = model_fingerprint
        self.history = history or {}

    @property
    def features(self) -> int:
        return int(self.params["encoder.weight"].shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.params["encoder.weight"].shape[1])

    @classmethod
    def initial(cls, activations: Activations, config: SaeConfig, model_fingerprint: str = "") -> SparseAutoencoder:
        """Decoder columns from the seeded Gaussian generator, normalised; the encoder its transpose; ``b_d`` the
        mean scaled activation; ``b_e`` zero."""
        rows, hidden = activations.values.shape
        scale = math.sqrt(hidden / (sum_squares(activations.values) / rows))
        x = (activations.values * np.float32(scale)).astype(np.float32)
        decoder = fill_gaussian(config.seed, hidden * config.features).reshape(hidden, config.features)
        decoder = _unit_columns(decoder)
        params = {
            "decoder.weight": decoder,
            "decoder.bias": (_column_sums(x) / np.float32(rows)).astype(np.float32),
            "encoder.weight": np.ascontiguousarray(decoder.T),
            "encoder.bias": np.zeros(config.features, dtype=np.float32),
        }
        return cls(activations.layer, params, scale, config, model_fingerprint)

    def scale(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) * np.float32(self.input_scale)).astype(np.float32)

    def encode(self, x: np.ndarray, *, scaled: bool = False) -> np.ndarray:
        """Feature activations ``[rows, features]`` of residual stream rows ``[rows, hidden]``."""
        xs = np.asarray(x, dtype=np.float32) if scaled else self.scale(x)
        pre = linear(xs - self.params["decoder.bias"], self.params["encoder.weight"], self.params["encoder.bias"])
        return np.maximum(pre.numpy(), np.float32(0.0))

    def decode(self, f: np.ndarray) -> np.ndarray:
        """The reconstruction (in the scaled space) of feature activations ``f``."""
        return linear(f, self.params["decoder.weight"], self.params["decoder.bias"]).numpy()

    def loss_and_gradients(self, x: np.ndarray) -> tuple[float, float, dict[str, np.ndarray]]:
        """For scaled rows ``x[B, hidden]``: the loss, the mean squared reconstruction error, and the gradient of
        the loss for every parameter."""
        p = self.params
        rows = x.shape[0]
        centred = (x - p["decoder.bias"]).astype(np.float32)
        pre = linear(centred, p["encoder.weight"], p["encoder.bias"]).numpy()
        active = pre > 0
        f = np.where(active, pre, np.float32(0.0)).astype(np.float32)
        error = (linear(f, p["decoder.weight"], p["decoder.bias"]).numpy() - x).astype(np.float32)
        mse = sum_squares(error) / rows
        loss = mse + self.config.l1 * sum_(f.reshape(-1)) / rows
        derror = (error * np.float32(2.0 / rows)).astype(np.float32)
        df, ddecoder, ddecoder_bias = linear_backward(f, p["decoder.weight"], derror, with_bias=True)
        dpre = np.where(active, df.numpy() + np.float32(self.config.l1 / rows), np.float32(0.0)).astype(np.float32)
        dcentred, dencoder, dencoder_bias = linear_backward(centred, p["encoder.weight"], dpre, with_bias=True)
        assert ddecoder_bias is not None and dencoder_bias is not None
        gradients = {
            "decoder.weight": ddecoder.numpy(),
            "decoder.bias": (ddecoder_bias.numpy() - _column_sums(dcentred.numpy())).astype(np.float32),
            "encoder.weight": dencoder.numpy(),
            "encoder.bias": dencoder_bias.numpy(),
        }
        return float(loss), float(mse), gradients

    def steering_vector(self, feature: int, strength: float = 4.0) -> SteeringVector:
        """Feature ``feature``'s decoder direction, mapped back to the residual stream's scale, as a steering
        vector for this SAE's layer."""
        if not 0 <= feature < self.features:
            raise ValueError(f"feature must be between 0 and {self.features - 1}")
        direction = (self.params["decoder.weight"][:, feature] / np.float32(self.input_scale)).astype(np.float32)
        origin = {"sae_feature": feature, "sae_config": asdict(self.config)}
        return SteeringVector(self.layer, np.ascontiguousarray(direction), strength, self.model_fingerprint, origin)

    def save(self, path: str | Path) -> None:
        from etalii_dllm.importing.safetensors import write_safetensors

        metadata = {
            "format": "dllm-sae",
            "layer": str(self.layer),
            "input_scale": repr(self.input_scale),
            "config": json.dumps(asdict(self.config), sort_keys=True),
            "model_fingerprint": self.model_fingerprint,
            "history": json.dumps(self.history, sort_keys=True),
        }
        write_safetensors(path, self.params, metadata)

    @classmethod
    def load(cls, path: str | Path) -> SparseAutoencoder:
        from etalii_dllm.importing.safetensors import SafetensorsError, SafetensorsFile

        try:
            file = SafetensorsFile(path)
        except (OSError, SafetensorsError) as error:
            raise ValueError(f"{path}: {error}") from error
        metadata = file.metadata
        if metadata.get("format") != "dllm-sae" or sorted(file.names()) != sorted(_NAMES):
            raise ValueError(f"{path}: not a sparse autoencoder file")
        params = {name: np.array(file[name].to_float32(), dtype=np.float32) for name in _NAMES}
        return cls(
            int(metadata["layer"]),
            params,
            float(metadata["input_scale"]),
            SaeConfig(**json.loads(metadata["config"])),
            metadata.get("model_fingerprint", ""),
            json.loads(metadata.get("history") or "{}"),
        )


def _unit_columns(matrix: np.ndarray) -> np.ndarray:
    """``matrix[rows, cols]`` with every column divided by its length (sums of squares over rows ascending)."""
    norms = np.sqrt(_column_sums((matrix * matrix).astype(np.float32)).astype(np.float64)).astype(np.float32)
    norms = np.where(norms > 0, norms, np.float32(1.0))
    return np.ascontiguousarray((matrix / norms[None, :]).astype(np.float32))


def _order(count: int, seed: int, epoch: int) -> list[int]:
    random = DeterministicRandom((seed * 0x9E3779B97F4A7C15 + epoch + 1) & _MASK64)
    order = list(range(count))
    for i in range(count - 1, 0, -1):
        j = random.next_u64() % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


@dataclass(frozen=True)
class SaeStep:
    step: int
    loss: float
    mse: float


def train_sae(
    activations: Activations,
    config: SaeConfig,
    model_fingerprint: str = "",
    on_step: Callable[[SaeStep], None] | None = None,
) -> SparseAutoencoder:
    """Trains an SAE on ``activations``: ``config.steps`` AdamW steps on batches drawn epoch by epoch from a seeded
    shuffle of the rows, the decoder columns renormalised after every step."""
    from etalii_dllm.training import AdamW, AdamWConfig

    sae = SparseAutoencoder.initial(activations, config, model_fingerprint)
    x = sae.scale(activations.values)
    optimiser = AdamW(
        AdamWConfig(learning_rate=config.learning_rate, weight_decay=0.0, max_grad_norm=0.0, schedule="constant"),
        {name: values.shape for name, values in sae.params.items()},
    )
    count = x.shape[0]
    orders: dict[int, list[int]] = {}
    losses = []
    for step in range(1, config.steps + 1):
        indices = []
        for sample in range((step - 1) * config.batch_size, step * config.batch_size):
            epoch, index = divmod(sample, count)
            if epoch not in orders:
                orders = {epoch: _order(count, config.seed, epoch)}
            indices.append(orders[epoch][index])
        loss, mse, gradients = sae.loss_and_gradients(np.ascontiguousarray(x[indices]))
        optimiser.step(sae.params, gradients, step, config.learning_rate)
        sae.params["decoder.weight"] = _unit_columns(sae.params["decoder.weight"])
        losses.append(loss)
        if on_step is not None:
            on_step(SaeStep(step, loss, mse))
    sae.history = {"rows": count, "final_loss": losses[-1]}
    return sae


@dataclass(frozen=True)
class Example:
    activation: float
    text: int
    position: int


@dataclass(frozen=True)
class Feature:
    index: int
    frequency: float
    """Share of the rows on which the feature is active."""
    max_activation: float
    examples: list[Example]


def feature_report(
    sae: SparseAutoencoder,
    activations: Activations,
    features: Sequence[int] | None = None,
    top: int = 8,
    count: int = 10,
) -> list[Feature]:
    """For ``features`` (default: the ``count`` with the largest maximum activation), how often each is active and
    the ``top`` rows that activate it most; rankings break ties on the lower index."""
    if activations.layer != sae.layer:
        raise ValueError(f"the SAE reads layer {sae.layer}, the activations are from layer {activations.layer}")
    f = sae.encode(activations.values)
    maxima = f.max(axis=0)
    if features is None:
        features = top_k(maxima, min(count, sae.features))
    report = []
    for index in features:
        if not 0 <= index < sae.features:
            raise ValueError(f"feature must be between 0 and {sae.features - 1}")
        column = np.ascontiguousarray(f[:, index])
        active = int(np.count_nonzero(column))
        examples = [Example(float(column[row]), *activations.positions[row]) for row in top_k(column, min(top, active))]
        report.append(Feature(int(index), active / column.shape[0], float(maxima[index]), examples))
    return report
