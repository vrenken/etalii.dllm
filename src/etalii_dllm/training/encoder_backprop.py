"""Reverse-mode gradients of BERT, ModernBERT, DeBERTa and T5 encoders (``etalii_dllm.encoder``): embedders and
cross-encoders (#352, #360, #367, #377).

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

ModernBERT's backward pass runs the same kernels through its own forward pass: the rotary embedding's gradient is
the inverse rotation (``rope(..., inverse=True)``, its exact transpose), local layers' ``attention_backward`` sees the
window the forward pass saw, the gated MLP's gradient is ``gelu_backward(gate, dy * up)`` and ``dy * gelu(gate)``,
and bias-free LayerNorms keep only their weight gradients. Every row of its word embedding trains: inputs are
never padded, so transformers' ``padding_idx`` (which freezes the padding token's row) only differs for a text that
spells out the padding token itself.

DeBERTa's backward pass is BERT's with ``biased_attention_backward`` in place of ``attention_backward``: it returns
the gradient of the score bias with the others. The bias of head ``h`` gathers ``c2p[i, d(i, j)] + p2c[j, d(i,
j)]``, so its gradient scatters back with ``embedding_backward`` (each (i, j) pair's value summed into its row of
``c2p`` and of ``p2c``, in (i, j) row-major order, in double), then ``linear_backward`` carries ``c2p = q_h pk_h^T``
and ``p2c = k_h pq_h^T`` to the queries and keys (added after the attention's own ``dq``/``dk``) and to the position
projections ``pq``/``pk``. Those are the layer's query and key projections of the relative table, so their weight
and bias gradients are added to the content path's (content first). The table rows' gradients from every layer
(``pq``'s then ``pk``'s, layers last to first) are added in float32 and go through ``layer_norm_backward`` to
``relative_norm`` and ``relative_embedding``, whose rows no pair reads get zero. The head
``classifier(gelu(pooler(h[0])))`` runs ``gelu_backward``.

T5's backward pass runs ``rms_norm_backward`` for its bias-free RMS norms, ``biased_attention_backward`` with scale 1,
and its MLP's activation gradient (ReLU passes ``dy`` where its input is positive, +0 elsewhere, as torch does; the
gated MLP's gradient is ModernBERT's with that activation). Every layer reads one bucket table, so the score bias
gradient of every layer (last to first, added in float32) scatters back with ``embedding_backward``: each (head, i,
j) value summed in (head, i, j) row-major order into row ``t5_bucket(j - i)`` of that head's column.

An embedder with a sentence-transformers ``Dense`` module (:func:`sentence_vector`) trains it with the rest: the
normalisation's gradient goes through ``softcap_backward`` (``tanh``) and ``linear_backward`` to the projection's
weight and bias and to the pooled vector.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.encoder import activate, relative_index, t5_bias, t5_relative_buckets
from etalii_dllm.numerics import (
    FloatArray,
    attention,
    attention_backward,
    biased_attention,
    biased_attention_backward,
    dot,
    embedding_backward,
    gelu,
    gelu_backward,
    gelu_tanh_backward,
    layer_norm,
    layer_norm_backward,
    linear,
    linear_backward,
    rms_norm,
    rms_norm_backward,
    rope,
    rope_inv_freq,
    silu_backward,
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
    relative: dict[str, np.ndarray] | None = None
    """DeBERTa: the relative table rows the pass read (before and after their LayerNorm) and their indices."""

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
        self.modern = config.family == "modernbert"
        self.deberta = config.family == "deberta"
        self.t5 = config.family == "t5"
        if self.modern:
            local = config.rope_theta if config.local_rope_theta is None else config.local_rope_theta
            self.inv_freq = rope_inv_freq(config.head_dim, config.rope_theta)
            self.local_inv_freq = rope_inv_freq(config.head_dim, local)

    # Forward

    def encode(
        self,
        weights: Mapping[str, npt.ArrayLike],
        tokens: Sequence[int],
        types: Sequence[int] | None = None,
        *,
        add: tuple[int, npt.ArrayLike] | None = None,
    ) -> EncoderPass:
        """The forward pass of ``Encoder.hidden_states`` (the same checks, the same kernels in the same order).
        ``add`` (T5 only) is ``(layer, delta[n, hidden])``: delta added (float32) to the residual stream after that
        0-based layer, as model editing does (#407)."""
        config = self.config
        if not tokens:
            raise ValueError("the encoder needs at least one token")
        if len(tokens) > config.context_length:
            raise ValueError(f"{len(tokens)} tokens exceed the encoder's {config.context_length} positions")
        if min(tokens) < 0 or max(tokens) >= config.vocabulary_size:
            raise ValueError("token id out of range")
        ids = np.asarray(tokens, dtype=np.int64)
        if add is not None and not self.t5:
            raise ValueError("only T5 encoders take a residual delta")
        if self.modern:
            return self._modern_encode(weights, ids)
        if self.deberta:
            return self._deberta_encode(weights, ids, types)
        if self.t5:
            return self._t5_encode(weights, ids, add)
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
        """:meth:`encode` with ``Encoder.classify``'s head: ``classifier(tanh(pooler(h[0])))``, ModernBERT's
        ``classifier(LayerNorm(gelu(pooler(pooled))))`` or DeBERTa's ``classifier(gelu(pooler(h[0])))``."""
        if not self.config.classifier_labels:
            raise ValueError("the model has no classification head")
        result = self.encode(weights, tokens, types)
        if self.modern:
            states = result.states
            first = np.ascontiguousarray(states[:1])
            if self.config.classifier_pooling == "mean":
                ones = np.ones((1, len(states)), dtype=np.float32)
                first = linear(ones, np.ascontiguousarray(states.T)).numpy() / np.float32(len(states))
            before = self._linear(weights, first, "pooler")
            activated = gelu(before).numpy()
            pooled = self._norm(weights, activated, "pooler_norm")
            logits = self._linear(weights, pooled, "classifier").reshape(-1)
            result.head = {"first": first, "before": before, "act": activated, "pooled": pooled, "logits": logits}
            return result
        first = np.ascontiguousarray(result.states[:1])
        before = self._linear(weights, first, "pooler")
        if self.deberta:
            activated = gelu(before, approximate=self.approximate).numpy()
            logits = self._linear(weights, activated, "classifier").reshape(-1)
            result.head = {"first": first, "before": before, "pooled": activated, "logits": logits}
            return result
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
        if dlogits is not None and self.modern:
            head = result.head
            if head is None:
                raise ValueError("dlogits needs a pass that ran the classification head (classify)")
            dlog = np.asarray(dlogits, dtype=np.float32).reshape(1, -1)
            dpooled = self._linear_backward(weights, head["pooled"], dlog, grads, "classifier")
            dact = self._norm_backward(weights, head["act"], dpooled, grads, "pooler_norm")
            dbefore = gelu_backward(head["before"], dact).numpy()
            dfirst = self._linear_backward(weights, head["first"], dbefore, grads, "pooler")[0]
            if self.config.classifier_pooling == "mean":
                dh = dh + dfirst / np.float32(n)
            else:
                dh[0] = dh[0] + dfirst
        elif dlogits is not None:
            head = result.head
            if head is None:
                raise ValueError("dlogits needs a pass that ran the classification head (classify)")
            dlog = np.asarray(dlogits, dtype=np.float32).reshape(1, -1)
            dpooled, grads["classifier.weight"], db = linear_backward(
                head["pooled"], weights["classifier.weight"], dlog, with_bias=True
            )
            grads["classifier.bias"] = _value(db)
            if self.deberta:
                dbefore = self._gelu_backward(head["before"], dpooled.numpy())
            else:
                dbefore = softcap_backward(head["before"], dpooled, 1.0)
            dfirst, grads["pooler.weight"], db = linear_backward(
                head["first"], weights["pooler.weight"], dbefore, with_bias=True
            )
            grads["pooler.bias"] = _value(db)
            dh[0] = dh[0] + dfirst.numpy()[0]
        if self.modern:
            return self._modern_gradients(weights, result, dh, grads)
        if self.deberta:
            return self._deberta_gradients(weights, result, dh, grads)
        if self.t5:
            return self._t5_gradients(weights, result, dh, grads)  # type: ignore[return-value]
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

    # ModernBERT

    def _modern_encode(self, weights: Mapping[str, npt.ArrayLike], ids: npt.NDArray[np.int64]) -> EncoderPass:
        """``Encoder._modernbert_states``, keeping the activations."""
        config = self.config
        n, heads, head_dim = len(ids), config.heads, config.head_dim
        positions = np.arange(n, dtype=np.int64)
        embedded = np.ascontiguousarray(_array(weights["token_embedding.weight"])[ids])
        h = self._norm(weights, embedded, "embedding_norm")
        result = EncoderPass(ids, np.zeros(n, dtype=np.int64), positions, h, embedded)
        for i in range(config.layers):
            p = f"layers.{i}."
            x = self._norm(weights, h, p + "attention_norm") if i else h
            inv_freq = self.inv_freq if config.window(i) is None else self.local_inv_freq
            q = rope(self._linear(weights, x, p + "attention.q").reshape(n, heads, head_dim), positions, inv_freq)
            k = rope(self._linear(weights, x, p + "attention.k").reshape(n, heads, head_dim), positions, inv_freq)
            q, k = q.numpy(), k.numpy()
            v = self._linear(weights, x, p + "attention.v").reshape(n, heads, head_dim)
            window = config.window(i)
            mixed = attention(q, k, v, scale=config.attention_scale, causal=False, window=window).numpy().reshape(n, -1)
            a = h + self._linear(weights, mixed, p + "attention.o")
            x2 = self._norm(weights, a, p + "mlp_norm")
            gate = self._linear(weights, x2, p + "mlp.gate")
            up = self._linear(weights, x2, p + "mlp.up")
            activated = gelu(gate).numpy()
            gated = activated * up
            result.layers.append(
                {
                    "h": h,
                    "x": x,
                    "q": q,
                    "k": k,
                    "v": v,
                    "mixed": mixed,
                    "a": a,
                    "x2": x2,
                    "gate": gate,
                    "up": up,
                    "act": activated,
                    "gated": gated,
                }
            )
            h = a + self._linear(weights, gated, p + "mlp.down")
        result.layers.append({"h": h})
        result.states = self._norm(weights, h, "final_norm")
        return result

    def _modern_gradients(
        self,
        weights: Mapping[str, npt.ArrayLike],
        result: EncoderPass,
        dstates: FloatArray,
        grads: dict[str, FloatArray],
    ) -> dict[str, FloatArray]:
        config = self.config
        n, heads, head_dim = len(result.tokens), config.heads, config.head_dim
        dh = self._norm_backward(weights, result.layers[-1]["h"], dstates, grads, "final_norm")
        for i in reversed(range(config.layers)):
            p = f"layers.{i}."
            saved = result.layers[i]
            # h_out = a + down(gelu(gate(x2)) * up(x2)), x2 = LN(a)
            dgated = self._linear_backward(weights, saved["gated"], dh, grads, p + "mlp.down")
            dgate = gelu_backward(saved["gate"], dgated * saved["up"]).numpy()
            dup = dgated * saved["act"]
            dx2 = self._linear_backward(weights, saved["x2"], dgate, grads, p + "mlp.gate")
            dx2 = dx2 + self._linear_backward(weights, saved["x2"], dup, grads, p + "mlp.up")
            da = dh + self._norm_backward(weights, saved["a"], dx2, grads, p + "mlp_norm")
            # a = h + o(attention(rope(q(x)), rope(k(x)), v(x))), x = LN(h) (h itself on the first layer)
            dmixed = self._linear_backward(weights, saved["mixed"], da, grads, p + "attention.o")
            window = config.window(i)
            dq, dk, dv = attention_backward(
                saved["q"],
                saved["k"],
                saved["v"],
                dmixed.reshape(n, heads, head_dim),
                scale=config.attention_scale,
                causal=False,
                q_offset=0,
                window=window,
            )
            inv_freq = self.inv_freq if window is None else self.local_inv_freq
            dq = rope(dq, result.positions, inv_freq, inverse=True)
            dk = rope(dk, result.positions, inv_freq, inverse=True)
            dx = None
            for name, d in (("q", dq), ("k", dk), ("v", dv)):
                part = self._linear_backward(
                    weights, saved["x"], d.numpy().reshape(n, -1), grads, p + "attention." + name
                )
                dx = part if dx is None else dx + part
            assert dx is not None
            dh = da + (self._norm_backward(weights, saved["h"], dx, grads, p + "attention_norm") if i else dx)
        dembedded = self._norm_backward(weights, result.embedded, dh, grads, "embedding_norm")
        grads["token_embedding.weight"] = embedding_backward(dembedded, result.tokens, config.vocabulary_size).numpy()
        return {name: np.array(values, dtype=np.float32) for name, values in grads.items()}

    # DeBERTa

    def _deberta_encode(
        self, weights: Mapping[str, npt.ArrayLike], ids: npt.NDArray[np.int64], types: Sequence[int] | None
    ) -> EncoderPass:
        """``Encoder._deberta_states``, keeping the activations."""
        config = self.config
        n, heads, head_dim = len(ids), config.heads, config.head_dim
        kinds = np.zeros(n, dtype=np.int64)
        if types is not None and config.type_vocabulary_size:
            kinds = np.asarray(types, dtype=np.int64)
        if kinds.shape != ids.shape or kinds.min() < 0 or kinds.max() >= max(config.type_vocabulary_size, 1):
            raise ValueError(f"token types must be one per token, below {config.type_vocabulary_size}")
        x = _array(weights["token_embedding.weight"])[ids]
        if config.type_vocabulary_size:
            x = x + _array(weights["token_type_embedding.weight"])[kinds]
        embedded = np.ascontiguousarray(x)
        h = self._norm(weights, embedded, "embedding_norm")
        index = relative_index(config, n)
        first = int(index.min())
        table = np.ascontiguousarray(_array(weights["relative_embedding.weight"])[first : int(index.max()) + 1])
        rows = self._norm(weights, table, "relative_norm")
        local = np.ascontiguousarray(index - first)
        result = EncoderPass(ids, kinds, np.arange(n, dtype=np.int64), h, embedded)
        result.relative = {"table": table, "rows": rows, "local": local, "first": np.asarray(first)}
        scale = 1.0 / math.sqrt(3 * head_dim)
        for i in range(config.layers):
            p = f"layers.{i}."
            q = self._linear(weights, h, p + "attention.q").reshape(n, heads, head_dim)
            k = self._linear(weights, h, p + "attention.k").reshape(n, heads, head_dim)
            v = self._linear(weights, h, p + "attention.v").reshape(n, heads, head_dim)
            pq = self._linear(weights, rows, p + "attention.q").reshape(len(rows), heads, head_dim)
            pk = self._linear(weights, rows, p + "attention.k").reshape(len(rows), heads, head_dim)
            bias = np.empty((heads, n, n), dtype=np.float32)
            for head in range(heads):
                c2p = linear(np.ascontiguousarray(q[:, head]), np.ascontiguousarray(pk[:, head])).numpy()
                p2c = linear(np.ascontiguousarray(k[:, head]), np.ascontiguousarray(pq[:, head])).numpy()
                bias[head] = np.take_along_axis(c2p, local, axis=1) + np.take_along_axis(p2c, local.T, axis=1).T
            mixed = biased_attention(q, k, v, bias, scale).numpy().reshape(n, -1)
            a = h + self._linear(weights, mixed, p + "attention.o")
            h1 = self._norm(weights, a, p + "attention_norm")
            up = self._linear(weights, h1, p + "mlp.up")
            activated = gelu(up, approximate=self.approximate).numpy()
            m = h1 + self._linear(weights, activated, p + "mlp.down")
            result.layers.append(
                {
                    "h": h,
                    "q": q,
                    "k": k,
                    "v": v,
                    "pq": pq,
                    "pk": pk,
                    "bias": bias,
                    "mixed": mixed,
                    "a": a,
                    "h1": h1,
                    "up": up,
                    "act": activated,
                    "m": m,
                }
            )
            h = self._norm(weights, m, p + "mlp_norm")
        result.states = h
        return result

    def _deberta_gradients(
        self,
        weights: Mapping[str, npt.ArrayLike],
        result: EncoderPass,
        dstates: FloatArray,
        grads: dict[str, FloatArray],
    ) -> dict[str, FloatArray]:
        config = self.config
        relative = result.relative
        assert relative is not None
        n, heads, head_dim = len(result.tokens), config.heads, config.head_dim
        rows, local = relative["rows"], relative["local"]
        r = len(rows)
        # Flat scatter targets of every (head, i, j): c2p[h, i, d(i, j)] and p2c[h, j, d(i, j)].
        offsets = np.arange(heads, dtype=np.int64)[:, None, None] * (n * r)
        queries = np.arange(n, dtype=np.int64)[:, None] * r
        to_c2p = (offsets + queries + local).reshape(-1)
        to_p2c = (offsets + queries.T + local).reshape(-1)
        scale = 1.0 / math.sqrt(3 * head_dim)
        drows = np.zeros_like(rows)
        dh = dstates
        for i in reversed(range(config.layers)):
            p = f"layers.{i}."
            saved = result.layers[i]
            dm = self._norm_backward(weights, saved["m"], dh, grads, p + "mlp_norm")
            dact = self._linear_backward(weights, saved["act"], dm, grads, p + "mlp.down")
            dup = self._gelu_backward(saved["up"], dact)
            dh1 = dm + self._linear_backward(weights, saved["h1"], dup, grads, p + "mlp.up")
            da = self._norm_backward(weights, saved["a"], dh1, grads, p + "attention_norm")
            dmixed = self._linear_backward(weights, saved["mixed"], da, grads, p + "attention.o")
            dq_t, dk_t, dv_t, dbias_t = biased_attention_backward(
                saved["q"], saved["k"], saved["v"], saved["bias"], dmixed.reshape(n, heads, head_dim), scale
            )
            dq, dk, dbias = dq_t.numpy().copy(), dk_t.numpy().copy(), dbias_t.numpy().reshape(-1, 1)
            dc2p = embedding_backward(dbias, to_c2p, heads * n * r).numpy().reshape(heads, n, r)
            dp2c = embedding_backward(dbias, to_p2c, heads * n * r).numpy().reshape(heads, n, r)
            q, k, pq, pk = saved["q"], saved["k"], saved["pq"], saved["pk"]
            dpq = np.empty_like(pq)
            dpk = np.empty_like(pk)
            for head in range(heads):
                dq_pos, dpk_h, _ = linear_backward(
                    np.ascontiguousarray(q[:, head]), np.ascontiguousarray(pk[:, head]), dc2p[head]
                )
                dk_pos, dpq_h, _ = linear_backward(
                    np.ascontiguousarray(k[:, head]), np.ascontiguousarray(pq[:, head]), dp2c[head]
                )
                dq[:, head] = dq[:, head] + dq_pos.numpy()
                dk[:, head] = dk[:, head] + dk_pos.numpy()
                dpk[:, head] = dpk_h.numpy()
                dpq[:, head] = dpq_h.numpy()
            dproj = None
            for name, d in (("q", dq), ("k", dk), ("v", dv_t.numpy())):
                dx = self._linear_backward(weights, saved["h"], d.reshape(n, -1), grads, p + "attention." + name)
                dproj = dx if dproj is None else dproj + dx
            for name, d in (("q", dpq), ("k", dpk)):
                full = p + "attention." + name
                drow, dw, db = linear_backward(rows, weights[full + ".weight"], d.reshape(r, -1), with_bias=True)
                grads[full + ".weight"] = grads[full + ".weight"] + dw.numpy()
                grads[full + ".bias"] = grads[full + ".bias"] + _value(db)
                drows = drows + drow.numpy()
            dh = da + dproj
        dtable = self._norm_backward(weights, relative["table"], drows, grads, "relative_norm")
        shapes = config.tensor_shapes()
        relative_grad = np.zeros(shapes["relative_embedding.weight"], dtype=np.float32)
        first = int(relative["first"])
        relative_grad[first : first + r] = dtable
        grads["relative_embedding.weight"] = relative_grad
        dembedded = self._norm_backward(weights, result.embedded, dh, grads, "embedding_norm")
        grads["token_embedding.weight"] = embedding_backward(dembedded, result.tokens, config.vocabulary_size).numpy()
        if config.type_vocabulary_size:
            grads["token_type_embedding.weight"] = embedding_backward(
                dembedded, result.types, config.type_vocabulary_size
            ).numpy()
        return {name: np.array(values, dtype=np.float32) for name, values in grads.items()}

    # T5

    def _t5_encode(
        self,
        weights: Mapping[str, npt.ArrayLike],
        ids: npt.NDArray[np.int64],
        add: tuple[int, npt.ArrayLike] | None = None,
    ) -> EncoderPass:
        """``Encoder._t5_states``, keeping the activations; ``add`` puts a delta after one layer."""
        config = self.config
        n, heads, head_dim = len(ids), config.heads, config.head_dim
        embedded = np.ascontiguousarray(_array(weights["token_embedding.weight"])[ids])
        bias = t5_bias(config, weights["relative_bias.weight"], n)
        result = EncoderPass(ids, np.zeros(n, dtype=np.int64), np.arange(n, dtype=np.int64), embedded, embedded)
        h = embedded
        for i in range(config.layers):
            p = f"layers.{i}."
            x = self._rms(weights, h, p + "attention_norm")
            q = self._linear(weights, x, p + "attention.q").reshape(n, heads, head_dim)
            k = self._linear(weights, x, p + "attention.k").reshape(n, heads, head_dim)
            v = self._linear(weights, x, p + "attention.v").reshape(n, heads, head_dim)
            mixed = biased_attention(q, k, v, bias, 1.0).numpy().reshape(n, -1)
            a = h + self._linear(weights, mixed, p + "attention.o")
            x2 = self._rms(weights, a, p + "mlp_norm")
            up = self._linear(weights, x2, p + "mlp.up")
            saved = {"h": h, "x": x, "q": q, "k": k, "v": v, "mixed": mixed, "a": a, "x2": x2, "up": up}
            if config.gated_mlp:
                gate = self._linear(weights, x2, p + "mlp.gate")
                activated = activate(gate, config.activation)
                saved |= {"gate": gate, "act": activated, "inner": activated * up}
            else:
                saved["inner"] = activate(up, config.activation)
            result.layers.append(saved)
            h = a + self._linear(weights, saved["inner"], p + "mlp.down")
            if add is not None and add[0] == i:
                h = (h + np.asarray(add[1], dtype=np.float32)).astype(np.float32)
        result.layers.append({"h": h, "bias": bias})
        result.states = self._rms(weights, h, "final_norm")
        return result

    def residual_gradient(
        self, weights: Mapping[str, npt.ArrayLike], result: EncoderPass, dstates: npt.ArrayLike, layer: int
    ) -> FloatArray:
        """T5: the gradient ``[n, hidden]`` of the residual stream after ``layer`` (0-based) from ``dstates``, the
        loss gradient of the last states: the backward pass of the later layers only (#407)."""
        if not self.t5:
            raise ValueError("the residual gradient is defined for T5 encoders")
        if not 0 <= layer < self.config.layers:
            raise ValueError(f"layer must be between 0 and {self.config.layers - 1}")
        states = np.array(dstates, dtype=np.float32)
        return self._t5_gradients(weights, result, states, {}, stop=layer)  # type: ignore[return-value]

    def _t5_gradients(
        self,
        weights: Mapping[str, npt.ArrayLike],
        result: EncoderPass,
        dstates: FloatArray,
        grads: dict[str, FloatArray],
        stop: int | None = None,
    ) -> dict[str, FloatArray] | FloatArray:
        config = self.config
        n, heads, head_dim, buckets = len(result.tokens), config.heads, config.head_dim, config.position_buckets
        last = result.layers[-1]
        bias = last["bias"]
        # Flat scatter targets of every (head, i, j): row h * buckets + t5_bucket(j - i) of the transposed table.
        positions = np.arange(n, dtype=np.int64)
        index = t5_relative_buckets(positions[None, :] - positions[:, None], buckets, config.max_relative_positions)
        targets = (np.arange(heads, dtype=np.int64)[:, None, None] * buckets + index[None]).reshape(-1)
        dtable = np.zeros((heads, buckets), dtype=np.float32)
        dh = self._rms_backward(weights, last["h"], dstates, grads, "final_norm")
        for i in reversed(range(config.layers)):
            if i == stop:  # the gradient of the residual stream after layer ``stop``
                return np.asarray(dh, dtype=np.float32)
            p = f"layers.{i}."
            saved = result.layers[i]
            # h_out = a + down(mlp(x2)), x2 = RMSNorm(a)
            dinner = self._linear_backward(weights, saved["inner"], dh, grads, p + "mlp.down")
            if config.gated_mlp:  # inner = act(gate) * up
                dgate = self._activation_backward(saved["gate"], dinner * saved["up"])
                dup = dinner * saved["act"]
                dx2 = self._linear_backward(weights, saved["x2"], dgate, grads, p + "mlp.gate")
                dx2 = dx2 + self._linear_backward(weights, saved["x2"], dup, grads, p + "mlp.up")
            else:  # inner = act(up)
                dup = self._activation_backward(saved["up"], dinner)
                dx2 = self._linear_backward(weights, saved["x2"], dup, grads, p + "mlp.up")
            da = dh + self._rms_backward(weights, saved["a"], dx2, grads, p + "mlp_norm")
            # a = h + o(attention(q(x), k(x), v(x), bias)), x = RMSNorm(h)
            dmixed = self._linear_backward(weights, saved["mixed"], da, grads, p + "attention.o")
            dq, dk, dv, dbias = biased_attention_backward(
                saved["q"], saved["k"], saved["v"], bias, dmixed.reshape(n, heads, head_dim), 1.0
            )
            scattered = embedding_backward(dbias.numpy().reshape(-1, 1), targets, heads * buckets)
            dtable = dtable + scattered.numpy().reshape(heads, buckets)
            dx = None
            for name, d in (("q", dq), ("k", dk), ("v", dv)):
                part = self._linear_backward(
                    weights, saved["x"], d.numpy().reshape(n, -1), grads, p + "attention." + name
                )
                dx = part if dx is None else dx + part
            assert dx is not None
            dh = da + self._rms_backward(weights, saved["h"], dx, grads, p + "attention_norm")
        grads["relative_bias.weight"] = np.ascontiguousarray(dtable.T)
        grads["token_embedding.weight"] = embedding_backward(dh, result.tokens, config.vocabulary_size).numpy()
        return {name: np.array(values, dtype=np.float32) for name, values in grads.items()}

    # Pieces

    @property
    def approximate(self) -> str:
        return "tanh" if self.config.activation == "gelu_tanh" else "none"

    def _gelu_backward(self, x: np.ndarray, dy: np.ndarray) -> FloatArray:
        if self.config.activation == "gelu_tanh":
            return gelu_tanh_backward(x, dy).numpy()
        return gelu_backward(x, dy).numpy()

    def _activation_backward(self, x: np.ndarray, dy: np.ndarray) -> FloatArray:
        """``dy * act'(x)`` for T5's activation: ``relu`` passes ``dy`` where ``x > 0`` (+0 elsewhere, as torch)."""
        activation = self.config.activation
        if activation == "relu":
            return np.where(x > 0, dy, np.float32(0)).astype(np.float32)
        if activation == "silu":
            return silu_backward(x, dy).numpy()
        return self._gelu_backward(x, dy)

    def _rms(self, weights: Mapping[str, npt.ArrayLike], x: np.ndarray, name: str) -> FloatArray:
        return rms_norm(x, weights[name + ".weight"], self.config.rms_norm_eps).numpy()

    def _rms_backward(
        self,
        weights: Mapping[str, npt.ArrayLike],
        x: np.ndarray,
        dy: np.ndarray,
        grads: dict[str, FloatArray],
        name: str,
    ) -> FloatArray:
        dx, dw = rms_norm_backward(x, weights[name + ".weight"], dy, self.config.rms_norm_eps)
        grads[name + ".weight"] = dw.numpy()
        return dx.numpy()

    def _linear(self, weights: Mapping[str, npt.ArrayLike], x: np.ndarray, name: str) -> FloatArray:
        return linear(x, weights[name + ".weight"], weights.get(name + ".bias")).numpy()

    def _linear_backward(
        self,
        weights: Mapping[str, npt.ArrayLike],
        x: np.ndarray,
        dy: np.ndarray,
        grads: dict[str, FloatArray],
        name: str,
    ) -> FloatArray:
        biased = name + ".bias" in weights
        dx, grads[name + ".weight"], db = linear_backward(x, weights[name + ".weight"], dy, with_bias=biased)
        if biased:
            grads[name + ".bias"] = _value(db)
        return dx.numpy()

    def _norm(self, weights: Mapping[str, npt.ArrayLike], x: np.ndarray, name: str) -> FloatArray:
        eps, weight = self.config.rms_norm_eps, weights[name + ".weight"]
        bias = weights.get(name + ".bias")
        return layer_norm(
            x, weight, np.zeros(len(_array(weight)), dtype=np.float32) if bias is None else bias, eps
        ).numpy()

    def _norm_backward(
        self,
        weights: Mapping[str, npt.ArrayLike],
        x: np.ndarray,
        dy: np.ndarray,
        grads: dict[str, FloatArray],
        name: str,
    ) -> FloatArray:
        dx, dw, db = layer_norm_backward(x, weights[name + ".weight"], dy, self.config.rms_norm_eps)
        grads[name + ".weight"] = dw.numpy()
        if name + ".bias" in weights:
            grads[name + ".bias"] = db.numpy()
        return dx.numpy()


# Sentence vectors


def pool(states: npt.ArrayLike, mode: str) -> tuple[FloatArray, FloatArray, np.float32]:
    """The sentence vector of ``states`` as ``engine.embed`` pools it (``mean``: each column summed over positions
    ascending in double through the ``linear`` kernel, divided in float32; ``cls``: the first state; ``last_token``:
    the last), then L2-normalised: (the normalised vector, the pooled one, its float32 norm)."""
    vector = _pooled(np.asarray(states, dtype=np.float32), mode)
    normalized, norm = _normalize(vector)
    return normalized, vector, norm


def pool_backward(
    normalized: npt.ArrayLike, norm: np.float32, dnormalized: npt.ArrayLike, positions: int, mode: str
) -> FloatArray:
    """The gradient ``[positions, hidden]`` of the states from ``dnormalized``, the gradient of :func:`pool`'s
    normalised vector ``u / |u|``: ``du = (d - u_hat (u_hat . d)) / |u|`` (the ``dot`` kernel), then ``du`` to the
    pooled position (``cls``, ``last_token``) or ``du / positions`` to every position (``mean``)."""
    return _spread(_normalize_backward(normalized, norm, dnormalized), positions, mode)


@dataclass
class SentenceVector:
    """A training sentence vector: pooled, through the model's ``Dense`` projection when it has one, normalised."""

    normalized: FloatArray
    norm: np.float32
    pooled: FloatArray
    before: FloatArray | None = None
    """The projection's ``linear`` output, before its activation (``None`` without a projection)."""


def sentence_vector(
    weights: Mapping[str, npt.ArrayLike], states: npt.ArrayLike, mode: str, projection: str | None = None
) -> SentenceVector:
    """``engine.embed``'s vector of ``states``: :func:`pool`, then, with ``projection`` (the embedding setting:
    ``identity`` or ``tanh``), the ``Dense`` module ``act(linear(pooled, projection.weight, projection.bias))``
    (``tanh`` as ``softcap(x, 1)``), then L2-normalised."""
    pooled = _pooled(np.asarray(states, dtype=np.float32), mode)
    if not projection:
        normalized, norm = _normalize(pooled)
        return SentenceVector(normalized, norm, pooled)
    before = linear(pooled.reshape(1, -1), weights["projection.weight"], weights.get("projection.bias")).numpy()
    if projection == "tanh":
        projected = softcap(before, 1.0).numpy()
    elif projection == "identity":
        projected = before
    else:
        raise ValueError(f"unsupported projection activation {projection!r}")
    normalized, norm = _normalize(projected.reshape(-1))
    return SentenceVector(normalized, norm, pooled, before.reshape(-1))


def sentence_backward(
    weights: Mapping[str, npt.ArrayLike],
    vector: SentenceVector,
    dnormalized: npt.ArrayLike,
    positions: int,
    mode: str,
    projection: str | None = None,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """The gradient of the states from ``dnormalized`` (:func:`pool_backward`), through the projection when
    :func:`sentence_vector` ran one (``softcap_backward`` for ``tanh``, then ``linear_backward``), and the
    projection's weight (and bias) gradients."""
    d = _normalize_backward(vector.normalized, vector.norm, dnormalized)
    grads: dict[str, FloatArray] = {}
    if projection:
        assert vector.before is not None
        if projection == "tanh":
            d = softcap_backward(vector.before, d, 1.0).numpy()
        biased = "projection.bias" in weights
        dpooled, dw, db = linear_backward(
            vector.pooled.reshape(1, -1), weights["projection.weight"], d.reshape(1, -1), with_bias=biased
        )
        grads["projection.weight"] = dw.numpy()
        if biased:
            grads["projection.bias"] = _value(db).reshape(-1)
        d = dpooled.numpy().reshape(-1)
    return _spread(d, positions, mode), grads


def _pooled(values: np.ndarray, mode: str) -> FloatArray:
    if mode == "cls":
        return values[0].copy()
    if mode == "last_token":
        return values[-1].copy()
    if mode == "mean":
        ones = np.ones((1, values.shape[0]), dtype=np.float32)
        total = linear(ones, np.ascontiguousarray(values.T)).numpy().reshape(-1)
        return (total / np.float32(values.shape[0])).astype(np.float32)
    raise ValueError(f"unknown pooling {mode!r}; supported: {', '.join(POOLING_MODES)}")


def _normalize(vector: np.ndarray) -> tuple[FloatArray, np.float32]:
    norm = np.float32(np.sqrt(sum_squares(vector)))
    if not norm > 0:
        raise ValueError("cannot normalise a zero sentence vector")
    return (vector / norm).astype(np.float32), norm


def _normalize_backward(normalized: npt.ArrayLike, norm: np.float32, dnormalized: npt.ArrayLike) -> FloatArray:
    u_hat = np.asarray(normalized, dtype=np.float32)
    d = np.asarray(dnormalized, dtype=np.float32)
    return ((d - u_hat * np.float32(dot(u_hat, d))) / norm).astype(np.float32)


def _spread(du: np.ndarray, positions: int, mode: str) -> FloatArray:
    dstates = np.zeros((positions, du.shape[0]), dtype=np.float32)
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
