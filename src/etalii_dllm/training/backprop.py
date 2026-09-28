"""Reverse-mode gradients of the decoder (``etalii_dllm.transformer``) with a next-token cross-entropy loss.

The forward pass is the decoder's own, kernel for kernel, so the last row of its logits equals
``Transformer.forward`` bit for bit. The backward pass runs the gradient kernels of ``cpp/include/dllm/grad.hpp``
in reverse layer order; every reduction in them has a fixed order and a double accumulator. Where two gradient
paths meet (the residual stream, the three projections reading one normed input, a tied embedding and head) their
float32 contributions are added elementwise in the fixed order written below. Nothing here depends on threads,
batch composition or dictionary iteration order, so the gradients of a sequence are the same bits on every run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.numerics import (
    FloatArray,
    attention,
    attention_backward,
    cross_entropy,
    embedding_backward,
    linear,
    linear_backward,
    rms_norm,
    rms_norm_backward,
    rope,
    rope_inv_freq,
)
from etalii_dllm.numerics import (
    silu as silu_forward,
)
from etalii_dllm.numerics import (
    silu_backward as silu_grad,
)


class DecoderGradients:
    """Loss and gradients of one decoder configuration. ``weights`` maps the names of
    ``TransformerConfig.tensor_shapes()`` to float32 arrays; they are read, never modified."""

    def __init__(self, config: TransformerConfig) -> None:
        if config.rope_interleaved:
            raise ValueError("training expects the Hugging Face rotary layout (imports convert to it)")
        self.config = config
        self.inv_freq = rope_inv_freq(config.head_dim, config.rope_theta, scaling=config.rope_scaling)

    def logits(self, weights: Mapping[str, npt.ArrayLike], tokens: Sequence[int]) -> FloatArray:
        """Logits ``[len(tokens), vocab]`` for every position (forward pass only)."""
        hidden, _ = self._forward(weights, tokens, keep=False)
        return linear(hidden, self._head(weights)).numpy()

    def loss_and_gradients(
        self,
        weights: Mapping[str, npt.ArrayLike],
        tokens: Sequence[int],
        targets: Sequence[int],
        *,
        scale: float = 1.0,
    ) -> tuple[float, dict[str, FloatArray]]:
        """Summed cross-entropy of predicting ``targets[t]`` after ``tokens[: t + 1]`` (negative targets are
        skipped) and the gradient of ``scale`` times that sum for every weight."""
        config = self.config
        if len(tokens) != len(targets) or not tokens:
            raise ValueError("tokens and targets must be non-empty and equally long")
        n = len(tokens)
        hidden, saved = self._forward(weights, tokens, keep=True)
        head = self._head(weights)
        loss, dlogits = cross_entropy(linear(hidden, head), np.asarray(targets, dtype=np.int64), scale=scale)

        grads: dict[str, FloatArray] = {}
        final_input = saved.pop()
        dhidden, dhead, _ = linear_backward(hidden, head, dlogits)
        dx_t, grads["final_norm.weight"] = rms_norm_backward(
            final_input, weights["final_norm.weight"], dhidden, config.rms_norm_eps
        )
        dx = dx_t.numpy()

        positions = np.arange(n, dtype=np.int64)
        heads, kv_heads, head_dim = config.heads, config.kv_heads, config.head_dim
        for layer in reversed(range(config.layers)):
            p = f"layers.{layer}."
            x_in, h1, q, k, v, attended, x_mid, h2, gate, up, gate_act, product = saved[layer]

            # x_out = x_mid + down(silu(gate(h2)) * up(h2)),  h2 = mlp_norm(x_mid)
            dproduct, grads[p + "mlp.down.weight"], _ = linear_backward(product, weights[p + "mlp.down.weight"], dx)
            dgate = silu_grad(gate, dproduct.numpy() * up)
            dup = dproduct.numpy() * gate_act
            dh2_gate, grads[p + "mlp.gate.weight"], _ = linear_backward(h2, weights[p + "mlp.gate.weight"], dgate)
            dh2_up, grads[p + "mlp.up.weight"], _ = linear_backward(h2, weights[p + "mlp.up.weight"], dup)
            dh2 = dh2_gate.numpy() + dh2_up.numpy()
            dx_mid_norm, grads[p + "mlp_norm.weight"] = rms_norm_backward(
                x_mid, weights[p + "mlp_norm.weight"], dh2, config.rms_norm_eps
            )
            dx_mid = dx + dx_mid_norm.numpy()

            # x_mid = x_in + o(attention(rope(q(h1)), rope(k(h1)), v(h1))),  h1 = attention_norm(x_in)
            dattended, grads[p + "attention.o.weight"], _ = linear_backward(
                attended, weights[p + "attention.o.weight"], dx_mid
            )
            dq, dk, dv = attention_backward(q, k, v, dattended.reshape(n, heads, head_dim), causal=True, q_offset=0)
            dq = rope(dq, positions, self.inv_freq, inverse=True).reshape(n, heads * head_dim)
            dk = rope(dk, positions, self.inv_freq, inverse=True).reshape(n, kv_heads * head_dim)
            dv = dv.reshape(n, kv_heads * head_dim)
            dh1 = None
            for name, d in (("q", dq), ("k", dk), ("v", dv)):
                prefix = f"{p}attention.{name}."
                dh, dw, db = linear_backward(h1, weights[prefix + "weight"], d, with_bias=config.attention_bias)
                grads[prefix + "weight"] = dw
                if db is not None:
                    grads[prefix + "bias"] = db
                dh1 = dh.numpy() if dh1 is None else dh1 + dh.numpy()
            dx_in_norm, grads[p + "attention_norm.weight"] = rms_norm_backward(
                x_in, weights[p + "attention_norm.weight"], dh1, config.rms_norm_eps
            )
            dx = dx_mid + dx_in_norm.numpy()

        dembedding = embedding_backward(dx, np.asarray(tokens, dtype=np.int64), config.vocabulary_size).numpy()
        if config.tie_word_embeddings:
            grads["token_embedding.weight"] = dembedding + dhead.numpy()
        else:
            grads["token_embedding.weight"] = dembedding
            grads["lm_head.weight"] = dhead
        return loss, {name: np.asarray(g, dtype=np.float32) for name, g in grads.items()}

    def _head(self, weights: Mapping[str, npt.ArrayLike]) -> npt.ArrayLike:
        return weights["token_embedding.weight" if self.config.tie_word_embeddings else "lm_head.weight"]

    def _forward(
        self, weights: Mapping[str, npt.ArrayLike], tokens: Sequence[int], *, keep: bool
    ) -> tuple[FloatArray, list]:
        """The decoder forward pass over a whole sequence (``Transformer._layers`` without a cache). With ``keep``,
        returns the activations the backward pass needs: one tuple per layer, then the final norm's input."""
        config = self.config
        w = weights
        n = len(tokens)
        if any(not 0 <= t < config.vocabulary_size for t in tokens):
            raise ValueError("token id out of range")
        positions = np.arange(n, dtype=np.int64)
        embedding = np.asarray(w["token_embedding.weight"], dtype=np.float32)
        x = np.ascontiguousarray(embedding[np.asarray(tokens, dtype=np.int64)])
        saved: list = []
        heads, kv_heads, head_dim = config.heads, config.kv_heads, config.head_dim
        for layer in range(config.layers):
            p = f"layers.{layer}."
            h1 = rms_norm(x, w[p + "attention_norm.weight"], config.rms_norm_eps).numpy()
            q = linear(h1, w[p + "attention.q.weight"], w.get(p + "attention.q.bias"))
            k = linear(h1, w[p + "attention.k.weight"], w.get(p + "attention.k.bias"))
            v = linear(h1, w[p + "attention.v.weight"], w.get(p + "attention.v.bias"))
            q = rope(q.reshape(n, heads, head_dim), positions, self.inv_freq).numpy()
            k = rope(k.reshape(n, kv_heads, head_dim), positions, self.inv_freq).numpy()
            v = v.reshape(n, kv_heads, head_dim).numpy()
            attended = attention(q, k, v, causal=True, q_offset=0).reshape(n, heads * head_dim).numpy()
            x_mid = x + linear(attended, w[p + "attention.o.weight"]).numpy()
            h2 = rms_norm(x_mid, w[p + "mlp_norm.weight"], config.rms_norm_eps).numpy()
            gate = linear(h2, w[p + "mlp.gate.weight"]).numpy()
            up = linear(h2, w[p + "mlp.up.weight"]).numpy()
            gate_act = silu_forward(gate).numpy()
            product = gate_act * up
            x_out = x_mid + linear(product, w[p + "mlp.down.weight"]).numpy()
            if keep:
                saved.append((x, h1, q, k, v, attended, x_mid, h2, gate, up, gate_act, product))
            x = x_out
        if keep:
            saved.append(x)
        return rms_norm(x, w["final_norm.weight"], config.rms_norm_eps).numpy(), saved
