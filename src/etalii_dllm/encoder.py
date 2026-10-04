"""BERT-style encoders (BERT, all-MiniLM, bge, RoBERTa, XLM-RoBERTa) running on the deterministic kernels.

An encoder turns a whole input into one hidden state per position and is used for embeddings only. Its forward pass,
as Hugging Face's ``BertModel`` defines it:

1. ``h = LayerNorm((word[t] + type[s]) + position[p])``: the three embedding rows added in float32 in that order,
   ``s`` the token's type (0, or 1 for the second text of a pair) and ``p`` its position id: ``i`` for BERT, and
   for RoBERTa and XLM-RoBERTa the count of non-padding tokens up to ``i`` past the padding id
   (:meth:`~etalii_dllm.architecture.TransformerConfig.position_ids`);
2. per layer, ``h = LayerNorm(h + o(attention(q(h), k(h), v(h))))`` with every projection carrying a bias and
   attention seeing every position (bidirectional, no causal mask), then
   ``h = LayerNorm(h + down(gelu(up(h))))``, the MLP ungated with the exact (erf) GELU.

A sequence-classification model (a cross-encoder) adds a head on the ``[CLS]`` state ``h[0]``:
``logits = classifier(tanh(pooler(h[0])))``, both ``linear`` with a bias and tanh the portable kernel in double
(``softcap`` with cap 1), rounded to float32.

A ModernBERT encoder (``config.family == "modernbert"``), as transformers' ``ModernBertModel`` defines it, has no
position or type embeddings, biases or LayerNorm biases (a LayerNorm without a bias is the kernel with a zero bias,
which adds nothing):

1. ``h = LayerNorm(word[t])``;
2. per layer, ``h = h + o(attention(rope(q(x)), rope(k(x)), v(x)))`` with ``x = LayerNorm(h)`` (the first layer
   reads ``h`` itself), the rotary embedding at positions 0, 1, ... with ``rope_theta`` on global layers and
   ``local_rope_theta`` on local ones, and attention bidirectional: every key on a global layer, the keys closer
   than ``sliding_window`` on a local one (:attr:`~etalii_dllm.architecture.TransformerConfig.sliding_window_layers`);
   then ``h = h + down(gelu(gate(x)) * up(x))`` with ``x = LayerNorm(h)``, the gated MLP with the exact GELU;
3. ``h = LayerNorm(h)`` (the final norm).

Its classification head reads the first state (``classifier_pooling`` ``cls``) or the mean of every state
(``mean``: each column summed over the positions ascending in double by ``linear``, rounded to float32 and divided
in float32, as ``engine.embed`` pools), then
``logits = classifier(LayerNorm(gelu(pooler(pooled))))``, the pooler without a bias and the classifier with one.

A DeBERTa encoder (``config.family == "deberta"``, DeBERTa-v2 and v3 as transformers' ``DebertaV2Model`` defines
them) is BERT without absolute positions and with disentangled attention:

1. ``h = LayerNorm(word[t] (+ type[s]))``, the type row only when the model has token types;
2. per layer, ``h = LayerNorm(h + o(attention))`` and ``h = LayerNorm(h + down(gelu(up(h))))`` as in BERT, where
   attention scores query ``i`` against key ``j`` as
   ``(q_i . k_j + q_i . pk[d(i, j)] + k_j . pq[d(i, j)]) / sqrt(3 head_dim)``: ``pq`` and ``pk`` are the layer's
   own query and key projections (with their biases) of the relative position table ``LayerNorm(relative)``, and
   ``d(i, j)`` is the row of the distance ``i - j`` (:func:`relative_index`). The two position terms are each a
   ``linear`` of a head's queries (keys) against the table rows, gathered per (i, j) and added in float32 into a
   bias; ``biased_attention`` adds that bias to the dot product in double before scaling.

Its classification head (the context pooler) is ``logits = classifier(gelu(pooler(h[0])))``.

A T5 encoder (``config.family == "t5"``: sentence-t5, GTR-T5) has no positions in its embeddings:

1. ``h = word[t]``;
2. per layer, ``h = h + o(attention(RMSNorm(h)))`` and ``h = h + down(mlp(RMSNorm(h)))``, where the RMS norms are
   the ``rms_norm`` kernel (no mean, no bias), the projections have no biases, attention scores query ``i`` against
   key ``j`` as ``q_i . k_j + bias[head, t5_bucket(j - i)]`` without scaling (``biased_attention`` with scale 1), the
   bias table being the first layer's ``relative_attention_bias`` that every layer shares, and the MLP is
   ``relu(up(x))`` (T5 v1.0) or ``act(gate(x)) * up(x)`` (v1.1, :attr:`TransformerConfig.gated_mlp`);
3. ``h = RMSNorm(h)`` (the final norm).

An embedder whose sentence-transformers pipeline has a ``Dense`` module (:attr:`TransformerConfig.projection_size`)
projects the pooled vector with ``linear`` (and the module's activation) before normalising it; ``engine.embed``
applies it (:func:`project`).

LayerNorm is the ``layer_norm`` kernel (mean and variance summed ascending in double); attention, ``linear``, RoPE
and GELU are the decoder's kernels, so each position's state has one fixed evaluation order whatever the thread
count or SIMD path. The residual additions and the gate product are float32 elementwise operations. An encoder runs
on the CPU.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from etalii_dllm.architecture import ENCODER_ONLY, TransformerConfig
from etalii_dllm.numerics import (
    FloatArray,
    attention,
    biased_attention,
    biased_attention_weights,
    gelu,
    layer_norm,
    linear,
    log,
    rms_norm,
    rope,
    rope_inv_freq,
    silu,
    softcap,
)
from etalii_dllm.tensor import Tensor
from etalii_dllm.transformer import _MATRICES, _prepare, quantized_fingerprint


def relative_buckets(distances: npt.ArrayLike, buckets: int, max_position: int) -> np.ndarray:
    """DeBERTa's log-bucketed relative positions of integer ``distances`` (``i - j``), exactly as transformers'
    ``make_log_bucket_position`` computes them in float32: with ``mid = buckets // 2``, a distance ``d`` with
    ``|d| <= mid`` is its own bucket, and a longer one becomes ``sign(d) * (ceil(f32(f32(log(|d| / mid)) /
    f32(log((max_position - 1) / mid))) * (mid - 1)) + mid)``, every operation rounded to float32 and the
    logarithms the portable ``log`` in double rounded to float32. ``buckets <= 0`` (or ``max_position <= 0``) keeps
    every distance."""
    d = np.asarray(distances, dtype=np.int64)
    if buckets <= 0 or max_position <= 0:
        return d.copy()
    mid = buckets // 2
    magnitude = np.abs(d)
    far = magnitude > mid
    ratio = magnitude[far].astype(np.float32) / np.float32(mid)
    logs = np.array([log(float(r)) for r in ratio], dtype=np.float64).astype(np.float32)
    denominator = np.float32(log(float(np.float32((max_position - 1) / mid))))
    scaled = (logs / denominator).astype(np.float32) * np.float32(mid - 1)
    out = d.copy()
    out[far] = np.sign(d[far]) * (np.ceil(scaled).astype(np.int64) + mid)
    return out


def relative_index(config: TransformerConfig, count: int) -> np.ndarray:
    """``[count, count]``: the relative position table row of query ``i`` against key ``j``,
    ``clamp(bucket(i - j) + span, 0, 2 span - 1)`` with ``span = config.relative_span``."""
    span = config.relative_span
    positions = np.arange(count, dtype=np.int64)
    distances = positions[:, None] - positions[None, :]
    buckets = relative_buckets(distances, config.position_buckets, config.max_relative_positions)
    return np.clip(buckets + span, 0, 2 * span - 1)


def t5_relative_buckets(
    distances: npt.ArrayLike, buckets: int, max_distance: int, *, bidirectional: bool = True
) -> np.ndarray:
    """T5's relative position buckets of integer ``distances`` (``j - i``, key minus query), exactly as transformers'
    ``T5Attention._relative_position_bucket`` computes them. Bidirectional (the encoder): half of the ``buckets`` for
    each direction (``n = buckets // 2``, keys after the query from ``n`` on) over ``r = |d|``; one-directional (the
    decoder, whose queries only see earlier keys): ``n = buckets`` over ``r = max(-d, 0)``. With ``exact = n // 2``,
    a distance ``r`` below ``exact`` is its own bucket and a longer one ``min(exact + trunc(f32(f32(log(f32(r) /
    exact)) / f32(log(max_distance / exact))) * (n - exact)), n - 1)``, every operation rounded to float32 and the
    logarithms the portable ``log`` in double rounded to float32."""
    d = np.asarray(distances, dtype=np.int64)
    half = buckets // 2 if bidirectional else buckets
    exact = half // 2
    if exact < 1:
        raise ValueError("t5 needs at least four relative position buckets")
    magnitude = np.abs(d) if bidirectional else np.maximum(-d, 0)
    far = magnitude >= exact
    ratio = (magnitude[far].astype(np.float32) / np.float32(exact)).astype(np.float32)
    logs = np.array([log(float(r)) for r in ratio], dtype=np.float64).astype(np.float32)
    denominator = np.float32(log(max_distance / exact))
    scaled = (logs / denominator).astype(np.float32) * np.float32(half - exact)
    bucket = magnitude.copy()
    bucket[far] = np.minimum(exact + np.trunc(scaled).astype(np.int64), half - 1)
    if not bidirectional:
        return bucket
    return np.where(d > 0, half, 0) + bucket


def t5_bias(config: TransformerConfig, table: npt.ArrayLike, count: int) -> np.ndarray:
    """``[heads, count, count]``: the T5 score bias of query ``i`` and key ``j``, ``table[t5_bucket(j - i), head]``."""
    positions = np.arange(count, dtype=np.int64)
    index = t5_relative_buckets(
        positions[None, :] - positions[:, None], config.position_buckets, config.max_relative_positions
    )
    return np.ascontiguousarray(np.asarray(table, dtype=np.float32)[index].transpose(2, 0, 1))


def activate(x: np.ndarray, activation: str) -> np.ndarray:
    """An MLP activation, elementwise: ``relu`` (``x`` where ``x > 0``, else +0), ``gelu``, ``gelu_tanh`` or
    ``silu``."""
    if activation == "relu":
        return np.where(x > 0, x, np.float32(0)).astype(np.float32)
    if activation == "silu":
        return silu(x).numpy()
    return gelu(x, approximate="tanh" if activation == "gelu_tanh" else "none").numpy()


def project(vector: npt.ArrayLike, weight: npt.ArrayLike, bias: npt.ArrayLike | None, activation: str) -> np.ndarray:
    """A sentence-transformers ``Dense`` module: ``act(linear(vector, weight, bias))`` with ``act`` ``identity`` or
    ``tanh`` (the portable ``tanh``, as ``softcap(x, 1)``)."""
    out = linear(np.asarray(vector, dtype=np.float32).reshape(1, -1), weight, bias)
    if activation == "tanh":
        out = softcap(out, 1.0)
    elif activation != "identity":
        raise ValueError(f"unsupported projection activation {activation!r}")
    return out.numpy().reshape(-1)


class Encoder:
    """A BERT, DeBERTa, ModernBERT or T5 encoder over a model file's tensors (``config.is_encoder``); ``quantize``
    (``q8_0`` or ``q4_0``) runs the linear layers on quantised weights, which changes the output and so the
    fingerprint."""

    def __init__(
        self,
        config: TransformerConfig,
        tensors: Mapping[str, npt.ArrayLike],
        *,
        model_id: str = "dllm-encoder",
        weights_fingerprint: str = "",
        quantize: str | None = None,
        device: str = "cpu",
        release: Callable[[str], None] | None = None,
    ) -> None:
        if not config.is_encoder:
            raise ValueError(f"{config.family} is a decoder; load it with Transformer")
        if device != "cpu":
            raise ValueError("encoder models run on the CPU only")
        expected = config.tensor_shapes()
        missing = sorted(set(expected) - set(tensors))
        if missing:
            raise ValueError(f"missing tensors: {missing}")
        self.config = config
        self.quantization = quantize
        self.device = device
        self.steering: dict[int, np.ndarray] = {}
        self.weights_fingerprint = quantized_fingerprint(weights_fingerprint, quantize)
        self._id = model_id
        weights = {name: Tensor(tensors[name]) for name in expected}
        for name, shape in expected.items():
            if weights[name].shape != shape:
                raise ValueError(f"tensor {name!r} has shape {weights[name].shape}, expected {shape}")
        self.tensors: Mapping[str, Tensor] = weights
        """The float32 source tensors (usually memory-mapped from the model file)."""
        self._w: dict[str, object] = {}
        for name, tensor in weights.items():
            if name.endswith(_MATRICES):
                self._w[name] = _prepare(tensor, quantize, "cpu")
                if release is not None:
                    release(name)
            else:
                self._w[name] = tensor.numpy()
        self._zeros = np.zeros(config.hidden_size, dtype=np.float32)
        self._inv_freq = self._local_inv_freq = np.zeros(0)
        if config.family == "modernbert":
            self._inv_freq = rope_inv_freq(config.head_dim, config.rope_theta)
            local = config.rope_theta if config.local_rope_theta is None else config.local_rope_theta
            self._local_inv_freq = rope_inv_freq(config.head_dim, local)
        self._relative = np.zeros((0, config.hidden_size), dtype=np.float32)
        if config.family == "deberta":
            self._relative = self._norm(np.asarray(self._w["relative_embedding.weight"]), "relative_norm")

    @property
    def id(self) -> str:
        return self._id

    @property
    def vocabulary_size(self) -> int:
        return self.config.vocabulary_size

    @property
    def context_length(self) -> int:
        return self.config.context_length

    def forward(self, tokens: Sequence[int]) -> FloatArray:
        raise ValueError(ENCODER_ONLY)

    def hidden_states(
        self, tokens: Sequence[int], types: Sequence[int] | None = None, recorder: Any = None
    ) -> FloatArray:
        """The last layer's states ``[positions, hidden]`` for ``tokens`` of token ``types`` (all 0 by default;
        ModernBERT has no token types and ignores them). ``recorder`` (T5 only, a :class:`etalii_dllm.seq2seq.Recorder`)
        is shown every layer's intermediates without changing a bit (#402)."""
        config = self.config
        if recorder is not None and config.family != "t5":
            raise ValueError("only T5 encoders can be traced")
        if not tokens:
            raise ValueError("the encoder needs at least one token")
        if len(tokens) > config.context_length:
            raise ValueError(f"{len(tokens)} tokens exceed the encoder's {config.context_length} positions")
        if min(tokens) < 0 or max(tokens) >= config.vocabulary_size:
            raise ValueError("token id out of range")
        if config.family == "modernbert":
            return self._modernbert_states(tokens)
        if config.family == "t5":
            return self._t5_states(tokens, recorder)
        kinds = np.zeros(len(tokens), dtype=np.int64) if types is None else np.asarray(types, dtype=np.int64)
        if config.family == "deberta" and not config.type_vocabulary_size:
            kinds = np.zeros(len(tokens), dtype=np.int64)  # no token types: a pair's second text reads like the first
        if kinds.shape != (len(tokens),) or kinds.min() < 0 or kinds.max() >= max(config.type_vocabulary_size, 1):
            raise ValueError(f"token types must be one per token, below {config.type_vocabulary_size}")
        if config.family == "deberta":
            return self._deberta_states(tokens, kinds)
        w = self._w
        words = np.asarray(w["token_embedding.weight"])[np.asarray(tokens, dtype=np.int64)]
        typed = words + np.asarray(w["token_type_embedding.weight"])[kinds]
        positions = np.asarray(w["position_embedding.weight"])[np.asarray(config.position_ids(tokens), dtype=np.int64)]
        h = self._norm(typed + positions, "embedding_norm")
        heads, head_dim = config.heads, config.head_dim
        for i in range(config.layers):
            p = f"layers.{i}."
            q = self._linear(h, p + "attention.q").reshape(len(tokens), heads, head_dim)
            k = self._linear(h, p + "attention.k").reshape(len(tokens), heads, head_dim)
            v = self._linear(h, p + "attention.v").reshape(len(tokens), heads, head_dim)
            mixed = attention(q, k, v, scale=config.attention_scale, causal=False).numpy()
            h = self._norm(h + self._linear(mixed.reshape(len(tokens), -1), p + "attention.o"), p + "attention_norm")
            up = self._linear(h, p + "mlp.up")
            activated = gelu(up, approximate="tanh" if config.activation == "gelu_tanh" else "none").numpy()
            h = self._norm(h + self._linear(activated, p + "mlp.down"), p + "mlp_norm")
        return h

    def classify(self, tokens: Sequence[int], types: Sequence[int] | None = None) -> FloatArray:
        """The classifier's logits ``[labels]`` for ``tokens`` of token ``types`` (a cross-encoder's pair)."""
        if not self.config.classifier_labels:
            raise ValueError(f"model {self.id} has no classification head")
        states = self.hidden_states(tokens, types)
        if self.config.family == "modernbert":
            pooled = states[:1]
            if self.config.classifier_pooling == "mean":
                ones = np.ones((1, len(states)), dtype=np.float32)
                pooled = linear(ones, np.ascontiguousarray(states.T)).numpy() / np.float32(len(states))
            head = gelu(self._linear(pooled, "pooler")).numpy()
            return self._linear(self._norm(head, "pooler_norm"), "classifier").reshape(-1)
        if self.config.family == "deberta":
            pooled = gelu(self._linear(states[:1], "pooler"), approximate=self._approximate).numpy()
            return self._linear(pooled, "classifier").reshape(-1)
        pooled = softcap(self._linear(states[:1], "pooler"), 1.0).numpy()
        return self._linear(pooled, "classifier").reshape(-1)

    def _deberta_states(self, tokens: Sequence[int], kinds: np.ndarray) -> FloatArray:
        config, w = self.config, self._w
        count, heads, head_dim = len(tokens), config.heads, config.head_dim
        x = np.asarray(w["token_embedding.weight"])[np.asarray(tokens, dtype=np.int64)]
        if config.type_vocabulary_size:
            x = x + np.asarray(w["token_type_embedding.weight"])[kinds]
        h = self._norm(x, "embedding_norm")
        index = relative_index(config, count)
        first = int(index.min())
        rows = np.ascontiguousarray(self._relative[first : int(index.max()) + 1])  # the rows any pair reads
        local = index - first
        scale = 1.0 / math.sqrt(3 * head_dim)
        for i in range(config.layers):
            p = f"layers.{i}."
            q = self._linear(h, p + "attention.q").reshape(count, heads, head_dim)
            k = self._linear(h, p + "attention.k").reshape(count, heads, head_dim)
            v = self._linear(h, p + "attention.v").reshape(count, heads, head_dim)
            pq = self._linear(rows, p + "attention.q").reshape(len(rows), heads, head_dim)
            pk = self._linear(rows, p + "attention.k").reshape(len(rows), heads, head_dim)
            bias = np.empty((heads, count, count), dtype=np.float32)
            for head in range(heads):
                c2p = linear(np.ascontiguousarray(q[:, head]), np.ascontiguousarray(pk[:, head])).numpy()
                p2c = linear(np.ascontiguousarray(k[:, head]), np.ascontiguousarray(pq[:, head])).numpy()
                bias[head] = np.take_along_axis(c2p, local, axis=1) + np.take_along_axis(p2c, local.T, axis=1).T
            mixed = biased_attention(q, k, v, bias, scale).numpy()
            h = self._norm(h + self._linear(mixed.reshape(count, -1), p + "attention.o"), p + "attention_norm")
            activated = gelu(self._linear(h, p + "mlp.up"), approximate=self._approximate).numpy()
            h = self._norm(h + self._linear(activated, p + "mlp.down"), p + "mlp_norm")
        return h

    @property
    def _approximate(self) -> str:
        return "tanh" if self.config.activation == "gelu_tanh" else "none"

    def project(self, vector: npt.ArrayLike, activation: str) -> np.ndarray:
        """The pooled sentence vector through the model's ``Dense`` projection (:func:`project`)."""
        if not self.config.projection_size:
            raise ValueError(f"model {self.id} has no projection")
        return project(vector, self._w["projection.weight"], self._w.get("projection.bias"), activation)

    def _t5_states(self, tokens: Sequence[int], recorder: Any = None) -> FloatArray:
        config, w = self.config, self._w
        count, heads, head_dim = len(tokens), config.heads, config.head_dim
        h = np.ascontiguousarray(np.asarray(w["token_embedding.weight"])[np.asarray(tokens, dtype=np.int64)])
        bias = t5_bias(config, w["relative_bias.weight"], count)
        eps = config.rms_norm_eps
        for i in range(config.layers):
            p = f"layers.{i}."
            if recorder is not None:
                recorder.record("residual", i, h.copy())
            x = rms_norm(h, w[p + "attention_norm.weight"], eps).numpy()  # type: ignore[arg-type]
            q = self._linear(x, p + "attention.q").reshape(count, heads, head_dim)
            k = self._linear(x, p + "attention.k").reshape(count, heads, head_dim)
            v = self._linear(x, p + "attention.v").reshape(count, heads, head_dim)
            mixed = biased_attention(q, k, v, bias, 1.0).numpy()
            out = self._linear(mixed.reshape(count, -1), p + "attention.o")
            h = h + out
            x = rms_norm(h, w[p + "mlp_norm.weight"], eps).numpy()  # type: ignore[arg-type]
            activation = self._t5_mlp(x, p)
            if recorder is not None:
                if recorder.wants_attention:
                    recorder.record("attention", i, biased_attention_weights(q, k, bias, 1.0).numpy())
                recorder.record("attention_output", i, out)
                recorder.record("middle", i, h.copy())
                recorder.record("mlp_activation", i, activation)
            out = self._linear(activation, p + "mlp.down")
            if recorder is not None:
                recorder.record("mlp_output", i, out)
            h = h + out
        if recorder is not None:
            recorder.record("residual", config.layers, h.copy())
        return rms_norm(h, w["final_norm.weight"], eps).numpy()  # type: ignore[arg-type,no-any-return]

    def _t5_mlp(self, x: np.ndarray, p: str) -> np.ndarray:
        """The activated MLP input: ``relu(up(x))`` (or the configured activation), or ``act(gate(x)) * up(x)``."""
        up = self._linear(x, p + "mlp.up")
        if not self.config.gated_mlp:
            return activate(up, self.config.activation)
        return activate(self._linear(x, p + "mlp.gate"), self.config.activation) * up

    def _modernbert_states(self, tokens: Sequence[int]) -> FloatArray:
        config = self.config
        count, heads, head_dim = len(tokens), config.heads, config.head_dim
        positions = np.arange(count, dtype=np.int64)
        h = self._norm(
            np.asarray(self._w["token_embedding.weight"])[np.asarray(tokens, dtype=np.int64)], "embedding_norm"
        )
        for i in range(config.layers):
            p = f"layers.{i}."
            x = self._norm(h, p + "attention_norm") if i else h
            window = config.window(i)
            inv_freq = self._inv_freq if window is None else self._local_inv_freq
            q = rope(self._linear(x, p + "attention.q").reshape(count, heads, head_dim), positions, inv_freq)
            k = rope(self._linear(x, p + "attention.k").reshape(count, heads, head_dim), positions, inv_freq)
            v = self._linear(x, p + "attention.v").reshape(count, heads, head_dim)
            mixed = attention(q, k, v, scale=config.attention_scale, causal=False, window=window).numpy()
            h = h + self._linear(mixed.reshape(count, -1), p + "attention.o")
            x = self._norm(h, p + "mlp_norm")
            gated = gelu(self._linear(x, p + "mlp.gate")).numpy() * self._linear(x, p + "mlp.up")
            h = h + self._linear(gated, p + "mlp.down")
        return self._norm(h, "final_norm")

    def _linear(self, x: np.ndarray, name: str) -> np.ndarray:
        return linear(x, self._w[name + ".weight"], self._w.get(name + ".bias")).numpy()  # type: ignore[arg-type]

    def _norm(self, x: np.ndarray, name: str) -> np.ndarray:
        w = self._w
        bias = w.get(name + ".bias", self._zeros)
        return layer_norm(x, w[name + ".weight"], bias, self.config.rms_norm_eps).numpy()  # type: ignore[arg-type]
