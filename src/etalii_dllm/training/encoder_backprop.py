"""Reverse-mode gradients of BERT-style encoders (``etalii_dllm.encoder``): embedders and cross-encoders (#352).

The forward pass is the encoder's own, kernel for kernel (``Encoder.hidden_states`` and ``Encoder.classify``), so
its states and logits equal the served model's bit for bit. The backward pass runs the gradient kernels of
``cpp/include/dllm/grad.hpp`` in reverse layer order: ``layer_norm_backward`` (mean and variance recomputed as the
forward kernel computes them; every sum ascending in double), ``attention_backward`` without the causal mask (every
query sees every key), ``gelu_backward`` (the exact erf GELU) or ``gelu_tanh_backward``, ``linear_backward`` with
biases, and ``embedding_backward`` for the word, token-type and position tables (RoBERTa's positions are the
``position_ids`` the forward pass used, and as in transformers, whose embeddings have a ``padding_idx``, the padding
token's row of both tables gets no gradient). Where two gradient paths meet (a residual and a norm, the three
projections reading one state) their float32 contributions are added elementwise in the order written below.

Sentence vectors for training embedders are pooled as the model is (:func:`pool`: ``mean``, ``cls`` or
``last_token``) and L2-normalised, so the dot product of two of them is their cosine similarity. Nothing depends on
threads, batch composition or dictionary order: the gradients of an input are the same bits on every run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.numerics import (
    FloatArray,
    attention,
    attention_backward,
    dot,
    embedding_backward,
    gelu,
    gelu_backward,
    gelu_tanh_backward,
    layer_norm,
    layer_norm_backward,
    linear,
    linear_backward,
    softcap,
    softcap_backward,
    sum_squares,
)

POOLING_MODES = ("mean", "cls", "last_token")


@dataclass
class EncoderPass:
    """One forward pass, with the activations its backward pass needs."""

    tokens: npt.NDArray[np.int64]
    types: npt.NDArray[np.int64]
    positions: npt.NDArray[np.int64]
    states: FloatArray
    """The last layer's states ``[positions, hidden]``."""
    embedded: FloatArray
    layers: list[dict[str, FloatArray]] = field(default_factory=list)
    head: dict[str, FloatArray] | None = None
    """With :meth:`EncoderGradients.classify`: the pooler's input and output and the logits."""

    @property
    def logits(self) -> FloatArray:
        if self.head is None:
            raise ValueError("this pass did not run the classification head")
        return self.head["logits"]


class EncoderGradients:
    """Forward and backward passes of one encoder configuration. ``weights`` maps the names of
    ``TransformerConfig.tensor_shapes()`` to float32 arrays; they are read, never modified."""

    def __init__(self, config: TransformerConfig) -> None:
        if not config.is_encoder:
            raise ValueError(f"{config.family} is a decoder; its gradients are DecoderGradients")
        self.config = config

    # Forward

    def encode(
        self, weights: Mapping[str, npt.ArrayLike], tokens: Sequence[int], types: Sequence[int] | None = None
    ) -> EncoderPass:
        """The forward pass of ``Encoder.hidden_states`` (the same checks, the same kernels in the same order)."""
        config = self.config
        if not tokens:
            raise ValueError("the encoder needs at least one token")
        if len(tokens) > config.context_length:
            raise ValueError(f"{len(tokens)} tokens exceed the encoder's {config.context_length} positions")
        if min(tokens) < 0 or max(tokens) >= config.vocabulary_size:
            raise ValueError("token id out of range")
        ids = np.asarray(tokens, dtype=np.int64)
        kinds = np.zeros(len(ids), dtype=np.int64) if types is None else np.asarray(types, dtype=np.int64)
        if kinds.shape != ids.shape or kinds.min() < 0 or kinds.max() >= config.type_vocabulary_size:
            raise ValueError(f"token types must be one per token, below {config.type_vocabulary_size}")
        positions = np.asarray(config.position_ids(list(tokens)), dtype=np.int64)
        n = len(ids)
        words = _array(weights["token_embedding.weight"])[ids]
        typed = words + _array(weights["token_type_embedding.weight"])[kinds]
        embedded = np.ascontiguousarray(typed + _array(weights["position_embedding.weight"])[positions])
        h = self._norm(weights, embedded, "embedding_norm")
        heads, head_dim = config.heads, config.head_dim
        result = EncoderPass(ids, kinds, positions, h, embedded)
        for i in range(config.layers):
            p = f"layers.{i}."
            q = self._linear(weights, h, p + "attention.q").reshape(n, heads, head_dim)
            k = self._linear(weights, h, p + "attention.k").reshape(n, heads, head_dim)
            v = self._linear(weights, h, p + "attention.v").reshape(n, heads, head_dim)
            mixed = attention(q, k, v, scale=config.attention_scale, causal=False).numpy().reshape(n, -1)
            a = h + self._linear(weights, mixed, p + "attention.o")
            h1 = self._norm(weights, a, p + "attention_norm")
            up = self._linear(weights, h1, p + "mlp.up")
            activated = gelu(up, approximate="tanh" if config.activation == "gelu_tanh" else "none").numpy()
            m = h1 + self._linear(weights, activated, p + "mlp.down")
            result.layers.append(
                {"h": h, "q": q, "k": k, "v": v, "mixed": mixed, "a": a, "h1": h1, "up": up, "act": activated, "m": m}
            )
            h = self._norm(weights, m, p + "mlp_norm")
        result.states = h
        return result

    def hidden_states(
        self, weights: Mapping[str, npt.ArrayLike], tokens: Sequence[int], types: Sequence[int] | None = None
    ) -> FloatArray:
        return self.encode(weights, tokens, types).states

    def classify(
        self, weights: Mapping[str, npt.ArrayLike], tokens: Sequence[int], types: Sequence[int] | None = None
    ) -> EncoderPass:
        """:meth:`encode` with ``Encoder.classify``'s head: ``classifier(tanh(pooler(h[0])))``."""
        if not self.config.classifier_labels:
            raise ValueError("the model has no classification head")
        result = self.encode(weights, tokens, types)
        first = np.ascontiguousarray(result.states[:1])
        before = self._linear(weights, first, "pooler")
        pooled = softcap(before, 1.0).numpy()
        logits = self._linear(weights, pooled, "classifier").reshape(-1)
        result.head = {"first": first, "before": before, "pooled": pooled, "logits": logits}
        return result

    # Backward

    def gradients(
        self,
        weights: Mapping[str, npt.ArrayLike],
        result: EncoderPass,
        dstates: npt.ArrayLike | None = None,
        dlogits: npt.ArrayLike | None = None,
    ) -> dict[str, FloatArray]:
        """The gradient of every weight from ``dstates`` (the loss gradient of the last layer's states) and/or
        ``dlogits`` (of the classifier's logits; the head's gradients join the states' at ``h[0]``). Weights the
        loss does not reach (the head, for an embedding loss) are left out."""
        config = self.config
        n = len(result.tokens)
        grads: dict[str, FloatArray] = {}
        dh = np.zeros_like(result.states) if dstates is None else np.array(dstates, dtype=np.float32)
        if dh.shape != result.states.shape:
            raise ValueError(f"dstates must have the states' shape {result.states.shape}")
        if dlogits is not None:
            head = result.head
            if head is None:
                raise ValueError("dlogits needs a pass that ran the classification head (classify)")
            dlog = np.asarray(dlogits, dtype=np.float32).reshape(1, -1)
            dpooled, grads["classifier.weight"], db = linear_backward(
                head["pooled"], weights["classifier.weight"], dlog, with_bias=True
            )
            grads["classifier.bias"] = _value(db)
            dbefore = softcap_backward(head["before"], dpooled, 1.0)
            dfirst, grads["pooler.weight"], db = linear_backward(
                head["first"], weights["pooler.weight"], dbefore, with_bias=True
            )
            grads["pooler.bias"] = _value(db)
            dh[0] = dh[0] + dfirst.numpy()[0]
        heads, head_dim = config.heads, config.head_dim
        for i in reversed(range(config.layers)):
            p = f"layers.{i}."
            saved = result.layers[i]
            # h_out = LN(m), m = h1 + down(gelu(up(h1)))
            dm = self._norm_backward(weights, saved["m"], dh, grads, p + "mlp_norm")
            dact = self._linear_backward(weights, saved["act"], dm, grads, p + "mlp.down")
            if config.activation == "gelu_tanh":
                dup = gelu_tanh_backward(saved["up"], dact).numpy()
            else:
                dup = gelu_backward(saved["up"], dact).numpy()
            dh1 = dm + self._linear_backward(weights, saved["h1"], dup, grads, p + "mlp.up")
            # h1 = LN(a), a = h + o(attention(q(h), k(h), v(h)))
            da = self._norm_backward(weights, saved["a"], dh1, grads, p + "attention_norm")
            dmixed = self._linear_backward(weights, saved["mixed"], da, grads, p + "attention.o")
            dq, dk, dv = attention_backward(
                saved["q"],
                saved["k"],
                saved["v"],
                dmixed.reshape(n, heads, head_dim),
                scale=config.attention_scale,
                causal=False,
                q_offset=0,
            )
            dproj = None
            for name, d in (("q", dq), ("k", dk), ("v", dv)):
                d_flat = d.numpy().reshape(n, -1)
                dx = self._linear_backward(weights, saved["h"], d_flat, grads, p + "attention." + name)
                dproj = dx if dproj is None else dproj + dx
            dh = da + dproj
        dembedded = self._norm_backward(weights, result.embedded, dh, grads, "embedding_norm")
        shapes = config.tensor_shapes()
        for name, ids in (
            ("token_embedding.weight", result.tokens),
            ("token_type_embedding.weight", result.types),
            ("position_embedding.weight", result.positions),
        ):
            grads[name] = embedding_backward(dembedded, ids, shapes[name][0]).numpy()
        result_grads = {name: np.array(values, dtype=np.float32) for name, values in grads.items()}
        if config.padding_index is not None:  # torch's padding_idx: the padding rows never train
            for name in ("token_embedding.weight", "position_embedding.weight"):
                result_grads[name][config.padding_index] = 0.0
        return result_grads

    # Pieces

    def _linear(self, weights: Mapping[str, npt.ArrayLike], x: np.ndarray, name: str) -> FloatArray:
        return linear(x, weights[name + ".weight"], weights[name + ".bias"]).numpy()

    def _linear_backward(
        self,
        weights: Mapping[str, npt.ArrayLike],
        x: np.ndarray,
        dy: np.ndarray,
        grads: dict[str, FloatArray],
        name: str,
    ) -> FloatArray:
        dx, grads[name + ".weight"], db = linear_backward(x, weights[name + ".weight"], dy, with_bias=True)
        grads[name + ".bias"] = _value(db)
        return dx.numpy()

    def _norm(self, weights: Mapping[str, npt.ArrayLike], x: np.ndarray, name: str) -> FloatArray:
        eps = self.config.rms_norm_eps
        return layer_norm(x, weights[name + ".weight"], weights[name + ".bias"], eps).numpy()

    def _norm_backward(
        self,
        weights: Mapping[str, npt.ArrayLike],
        x: np.ndarray,
        dy: np.ndarray,
        grads: dict[str, FloatArray],
        name: str,
    ) -> FloatArray:
        dx, dw, db = layer_norm_backward(x, weights[name + ".weight"], dy, self.config.rms_norm_eps)
        grads[name + ".weight"], grads[name + ".bias"] = dw.numpy(), db.numpy()
        return dx.numpy()


# Sentence vectors


def pool(states: npt.ArrayLike, mode: str) -> tuple[FloatArray, FloatArray, np.float32]:
    """The sentence vector of ``states`` as ``engine.embed`` pools it (``mean``: each column summed over positions
    ascending in double through the ``linear`` kernel, divided in float32; ``cls``: the first state; ``last_token``:
    the last), then L2-normalised: (the normalised vector, the pooled one, its float32 norm)."""
    values = np.asarray(states, dtype=np.float32)
    if mode == "cls":
        vector = values[0].copy()
    elif mode == "last_token":
        vector = values[-1].copy()
    elif mode == "mean":
        ones = np.ones((1, values.shape[0]), dtype=np.float32)
        total = linear(ones, np.ascontiguousarray(values.T)).numpy().reshape(-1)
        vector = (total / np.float32(values.shape[0])).astype(np.float32)
    else:
        raise ValueError(f"unknown pooling {mode!r}; supported: {', '.join(POOLING_MODES)}")
    norm = np.float32(np.sqrt(sum_squares(vector)))
    if not norm > 0:
        raise ValueError("cannot normalise a zero sentence vector")
    return (vector / norm).astype(np.float32), vector, norm


def pool_backward(
    normalized: npt.ArrayLike, norm: np.float32, dnormalized: npt.ArrayLike, positions: int, mode: str
) -> FloatArray:
    """The gradient ``[positions, hidden]`` of the states from ``dnormalized``, the gradient of :func:`pool`'s
    normalised vector ``u / |u|``: ``du = (d - u_hat (u_hat . d)) / |u|`` (the ``dot`` kernel), then ``du`` to the
    pooled position (``cls``, ``last_token``) or ``du / positions`` to every position (``mean``)."""
    u_hat = np.asarray(normalized, dtype=np.float32)
    d = np.asarray(dnormalized, dtype=np.float32)
    du = ((d - u_hat * np.float32(dot(u_hat, d))) / norm).astype(np.float32)
    dstates = np.zeros((positions, u_hat.shape[0]), dtype=np.float32)
    if mode == "cls":
        dstates[0] = du
    elif mode == "last_token":
        dstates[-1] = du
    else:
        dstates[:] = du / np.float32(positions)
    return dstates


def _array(values: npt.ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _value(tensor: object) -> FloatArray:
    assert tensor is not None
    return tensor.numpy()  # type: ignore[attr-defined,no-any-return]
