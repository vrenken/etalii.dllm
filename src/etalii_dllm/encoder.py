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

LayerNorm is the ``layer_norm`` kernel (mean and variance summed ascending in double); attention, ``linear``, RoPE
and GELU are the decoder's kernels, so each position's state has one fixed evaluation order whatever the thread
count or SIMD path. The residual additions and the gate product are float32 elementwise operations. An encoder runs
on the CPU.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from etalii_dllm.architecture import ENCODER_ONLY, TransformerConfig
from etalii_dllm.numerics import (
    FloatArray,
    attention,
    gelu,
    layer_norm,
    linear,
    rope,
    rope_inv_freq,
    softcap,
)
from etalii_dllm.tensor import Tensor
from etalii_dllm.transformer import _MATRICES, _prepare, quantized_fingerprint


class Encoder:
    """A BERT or ModernBERT encoder over a model file's tensors (``config.is_encoder``); ``quantize`` (``q8_0`` or
    ``q4_0``) runs the linear layers on quantised weights, which changes the output and so the fingerprint."""

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

    def hidden_states(self, tokens: Sequence[int], types: Sequence[int] | None = None) -> FloatArray:
        """The last layer's states ``[positions, hidden]`` for ``tokens`` of token ``types`` (all 0 by default;
        ModernBERT has no token types and ignores them)."""
        config = self.config
        if not tokens:
            raise ValueError("the encoder needs at least one token")
        if len(tokens) > config.context_length:
            raise ValueError(f"{len(tokens)} tokens exceed the encoder's {config.context_length} positions")
        if min(tokens) < 0 or max(tokens) >= config.vocabulary_size:
            raise ValueError("token id out of range")
        if config.family == "modernbert":
            return self._modernbert_states(tokens)
        kinds = np.zeros(len(tokens), dtype=np.int64) if types is None else np.asarray(types, dtype=np.int64)
        if kinds.shape != (len(tokens),) or kinds.min() < 0 or kinds.max() >= config.type_vocabulary_size:
            raise ValueError(f"token types must be one per token, below {config.type_vocabulary_size}")
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
        pooled = softcap(self._linear(states[:1], "pooler"), 1.0).numpy()
        return self._linear(pooled, "classifier").reshape(-1)

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
