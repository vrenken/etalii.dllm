"""Reverse-mode gradients of the decoder (``etalii_dllm.transformer``) with a next-token cross-entropy loss.

The forward pass is the decoder's own, kernel for kernel, for every architecture it runs (pre-, post- and sandwich
norms, QK-norm per head or over the whole projection, Gemma's unit-offset norms, tanh GELU and soft-caps, Granite's
multipliers, LongRoPE's attention factor, local RoPE bases and sliding windows), so the last row of its logits equals
``Transformer.forward`` bit for bit. Constant scales folded into weights (``transformer.fold_scales``) are folded the
same way here, and their gradients are scaled back elementwise. The backward pass runs the gradient kernels of
``cpp/include/dllm/grad.hpp`` in reverse layer order; every reduction in them has a fixed order and a double
accumulator. Where two gradient paths meet (the residual stream, the three projections reading one normed input, a
tied embedding and head) their float32 contributions are added elementwise in the fixed order written below. Nothing
here depends on threads, batch composition or dictionary iteration order, so the gradients of a sequence are the
same bits on every run.
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
    column_mean,
    cross_entropy,
    dot,
    embedding_backward,
    gelu,
    gelu_tanh_backward,
    linear,
    linear_backward,
    moe_route,
    moe_route_backward,
    rms_norm,
    rms_norm_backward,
    rope,
    rope_inv_freq,
    sigmoid_elementwise,
    silu,
    silu_backward,
    softcap,
    softcap_backward,
    softmax,
)
from etalii_dllm.tensor import Tensor
from etalii_dllm.transformer import fold_scales, rope_scaled_tensors


class DecoderGradients:
    """Loss and gradients of one decoder configuration. ``weights`` maps the names of
    ``TransformerConfig.tensor_shapes()`` to float32 arrays; they are read, never modified."""

    def __init__(self, config: TransformerConfig) -> None:
        if config.rope_interleaved:
            raise ValueError("training expects the Hugging Face rotary layout (imports convert to it)")
        self.config = config
        self.inv_freq = rope_inv_freq(
            config.head_dim, config.rope_theta, rotary_dim=config.rotary_dimension, scaling=config.rope_scaling
        )
        self.local_inv_freq = self.inv_freq
        if config.local_rope_theta is not None:
            self.local_inv_freq = rope_inv_freq(
                config.head_dim, config.local_rope_theta, rotary_dim=config.rotary_dimension
            )

    def logits(self, weights: Mapping[str, npt.ArrayLike], tokens: Sequence[int]) -> FloatArray:
        """Logits ``[len(tokens), vocab]`` for every position (forward pass only)."""
        hidden, _ = self._forward(weights, tokens, keep=False)
        logits = linear(hidden, self._head(weights)).numpy()
        return self._finish(logits)[1]

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
        loss, _, grads = self.losses_and_gradients(weights, tokens, targets, scale=scale)
        return loss, grads

    def losses_and_gradients(
        self,
        weights: Mapping[str, npt.ArrayLike],
        tokens: Sequence[int],
        targets: Sequence[int],
        *,
        scale: float = 1.0,
        router_scale: float = 0.0,
    ) -> tuple[float, float, dict[str, FloatArray]]:
        """As :meth:`loss_and_gradients`, together with the router load-balancing loss of the sequence
        (:meth:`router_loss`; 0 for dense models): (summed cross-entropy, load-balancing loss, the gradient of
        ``scale`` times the cross-entropy plus ``router_scale`` times the load-balancing loss)."""
        config = self.config
        if len(tokens) != len(targets) or not tokens:
            raise ValueError("tokens and targets must be non-empty and equally long")
        hidden, saved = self._forward(weights, tokens, keep=True)
        router_loss, router_gradients = self._router_loss(saved, router_scale)
        head = self._head(weights)
        loss, dx, dhead, grads = self._backward_from_logits(weights, hidden, head, targets, saved, scale)
        for layer in reversed(range(config.layers)):
            dx = self._backward_layer(saved[layer], dx, grads, router_gradients.get(layer))

        if config.embedding_multiplier != 1.0:
            dx = dx * np.float32(config.embedding_multiplier)
        dembedding = embedding_backward(dx, np.asarray(tokens, dtype=np.int64), config.vocabulary_size).numpy()
        if config.tie_word_embeddings:
            grads["token_embedding.weight"] = dembedding + dhead.numpy()
        else:
            grads["token_embedding.weight"] = dembedding
            grads["lm_head.weight"] = dhead
        return loss, router_loss, self._unfold({name: np.asarray(g, dtype=np.float32) for name, g in grads.items()})

    def router_loss(self, weights: Mapping[str, npt.ArrayLike], tokens: Sequence[int]) -> float:
        """The load-balancing loss of transformers' ``load_balancing_loss_func`` for one sequence, over the routers
        of every sparse layer (``T`` rows: the positions of every sparse layer):
        ``experts * sum_e f_e P_e`` with ``f_e`` the number of times expert ``e`` is among a row's chosen experts
        divided by ``T`` and ``P_e`` the mean of its softmax probability over the rows. 0 for dense models."""
        _, saved = self._forward(weights, tokens, keep=True)
        return self._router_loss(saved, 0.0)[0]

    def residual_gradient(
        self,
        weights: Mapping[str, npt.ArrayLike],
        tokens: Sequence[int],
        targets: Sequence[int],
        layer: int,
        delta: npt.ArrayLike,
    ) -> tuple[float, FloatArray]:
        """With ``delta[n, hidden]`` added to the residual stream after ``layer`` (0-based) in the forward pass: the
        summed cross-entropy of ``targets`` (as in :meth:`loss_and_gradients`) and its gradient with respect to that
        residual stream, ``[n, hidden]`` (model editing optimises such a delta)."""
        if len(tokens) != len(targets) or not tokens:
            raise ValueError("tokens and targets must be non-empty and equally long")
        if not 0 <= layer < self.config.layers:
            raise ValueError(f"layer must be between 0 and {self.config.layers - 1}")
        hidden, saved = self._forward(weights, tokens, keep=True, add=(layer, np.asarray(delta, dtype=np.float32)))
        head = self._head(weights)
        loss, dx, _, grads = self._backward_from_logits(weights, hidden, head, targets, saved, 1.0)
        for later in reversed(range(layer + 1, self.config.layers)):
            dx = self._backward_layer(saved[later], dx, grads)
        return loss, np.asarray(dx, dtype=np.float32)

    # The pieces of the forward and backward passes

    def _norm(self, x: npt.ArrayLike, weight: npt.ArrayLike) -> FloatArray:
        config = self.config
        return rms_norm(x, weight, config.rms_norm_eps, add_unit_offset=config.norm_unit_offset).numpy()

    def _norm_backward(
        self, x: npt.ArrayLike, weight: npt.ArrayLike, dy: npt.ArrayLike, grads: dict[str, FloatArray], name: str
    ) -> FloatArray:
        """The gradient of a norm's input; the gradient of its weight goes to ``grads[name]``."""
        config = self.config
        dx, grads[name] = rms_norm_backward(x, weight, dy, config.rms_norm_eps, add_unit_offset=config.norm_unit_offset)
        return dx.numpy()

    def _activation(self, gate: npt.ArrayLike) -> FloatArray:
        if self.config.activation == "gelu_tanh":
            return gelu(gate, approximate="tanh").numpy()
        return silu(gate).numpy()

    def _activation_backward(self, gate: npt.ArrayLike, dy: npt.ArrayLike) -> FloatArray:
        if self.config.activation == "gelu_tanh":
            return gelu_tanh_backward(gate, dy).numpy()
        return silu_backward(gate, dy).numpy()

    def _finish(self, logits: FloatArray) -> tuple[FloatArray, FloatArray]:
        """``Transformer._finish_logits``: (the logits before the soft-cap, the final logits)."""
        config = self.config
        if config.logits_scaling != 1.0:
            logits = logits / np.float32(config.logits_scaling)
        if config.logits_softcap is None:
            return logits, logits
        return logits, softcap(logits, config.logits_softcap).numpy()

    def _backward_from_logits(
        self,
        weights: Mapping[str, npt.ArrayLike],
        hidden: FloatArray,
        head: npt.ArrayLike,
        targets: Sequence[int],
        saved: list,
        scale: float,
    ) -> tuple[float, FloatArray, Tensor, dict[str, FloatArray]]:
        """The loss and the backward pass through the logits, the LM head and the final norm: (loss, the gradient of
        the last layer's output, the gradient of the head, the weight gradients so far)."""
        config = self.config
        uncapped, logits = self._finish(linear(hidden, head).numpy())
        loss, dlogits_t = cross_entropy(logits, np.asarray(targets, dtype=np.int64), scale=scale)
        dlogits = dlogits_t.numpy()
        if config.logits_softcap is not None:
            dlogits = softcap_backward(uncapped, dlogits, config.logits_softcap).numpy()
        if config.logits_scaling != 1.0:
            dlogits = dlogits / np.float32(config.logits_scaling)
        grads: dict[str, FloatArray] = {}
        final_input = saved.pop()
        dhidden, dhead, _ = linear_backward(hidden, head, dlogits)
        dx = self._norm_backward(final_input, weights["final_norm.weight"], dhidden, grads, "final_norm.weight")
        return loss, dx, dhead, grads

    def _backward_layer(
        self,
        saved: dict,
        dx: FloatArray,
        grads: dict[str, FloatArray],
        dprobabilities: FloatArray | None = None,
    ) -> FloatArray:
        """Back through one decoder layer: from the gradient of its output to that of its input, recording the
        gradients of its weights (as the forward pass used them, scales folded in) in ``grads``."""
        config = self.config
        n = dx.shape[0]
        positions = np.arange(n, dtype=np.int64)
        heads, kv_heads, head_dim = config.heads, config.kv_heads, config.head_dim
        w = saved["weights"]
        p = saved["prefix"]

        # x_out = x_mid + post_norm(down(act(gate(h2)) * up(h2))),  h2 = mlp_norm(x_mid)
        dmlp = dx
        if config.has_post_norms:
            name = p + "mlp_post_norm.weight"
            dmlp = self._norm_backward(saved["mlp"], w[name], dx, grads, name)
        if "routing" in saved:
            dh2 = self._experts_backward(saved, dmlp, grads, dprobabilities)
        else:
            dh2 = self._swiglu_backward(saved["h2"], saved, w, p + "mlp.", dmlp, grads)
        if config.has_pre_norms:
            dh2 = self._norm_backward(saved["x_mid"], w[p + "mlp_norm.weight"], dh2, grads, p + "mlp_norm.weight")
        dx_mid = dx + dh2

        # x_mid = x_in + post_norm(o(attention(rope(q(h1)), rope(k(h1)), v(h1)))),  h1 = attention_norm(x_in)
        dout = dx_mid
        if config.has_post_norms:
            name = p + "attention_post_norm.weight"
            dout = self._norm_backward(saved["out"], w[name], dx_mid, grads, name)
        dattended, grads[p + "attention.o.weight"], _ = linear_backward(
            saved["attended"], w[p + "attention.o.weight"], dout
        )
        dq_t, dk_t, dv_t = attention_backward(
            saved["q"],
            saved["k"],
            saved["v"],
            dattended.reshape(n, heads, head_dim),
            scale=config.attention_scale,
            causal=True,
            q_offset=0,
            window=config.window(saved["layer"]),
            softcap=config.attention_softcap,
        )
        inv_freq = self.local_inv_freq if config.uses_local_rope(saved["layer"]) else self.inv_freq
        dq = rope(dq_t, positions, inv_freq, inverse=True).numpy()
        dk = rope(dk_t, positions, inv_freq, inverse=True).numpy()
        if config.qk_norm and config.qk_norm_scope == "head":
            dq = dq.reshape(n * heads, head_dim)
            dk = dk.reshape(n * kv_heads, head_dim)
            q_raw = saved["q_raw"].reshape(n * heads, head_dim)
            k_raw = saved["k_raw"].reshape(n * kv_heads, head_dim)
            dq = self._norm_backward(q_raw, w[p + "attention.q_norm.weight"], dq, grads, p + "attention.q_norm.weight")
            dk = self._norm_backward(k_raw, w[p + "attention.k_norm.weight"], dk, grads, p + "attention.k_norm.weight")
        dq, dk = dq.reshape(n, heads * head_dim), dk.reshape(n, kv_heads * head_dim)
        if config.qk_norm and config.qk_norm_scope == "all":
            dq = self._norm_backward(
                saved["q_raw"], w[p + "attention.q_norm.weight"], dq, grads, p + "attention.q_norm.weight"
            )
            dk = self._norm_backward(
                saved["k_raw"], w[p + "attention.k_norm.weight"], dk, grads, p + "attention.k_norm.weight"
            )
        dv = dv_t.numpy().reshape(n, kv_heads * head_dim)
        h1 = saved["h1"]
        dh1 = None
        for name, d in (("q", dq), ("k", dk), ("v", dv)):
            prefix = f"{p}attention.{name}."
            dh, dw, db = linear_backward(h1, w[prefix + "weight"], d, with_bias=config.attention_bias)
            grads[prefix + "weight"] = dw
            if db is not None:
                grads[prefix + "bias"] = db
            dh1 = dh.numpy() if dh1 is None else dh1 + dh.numpy()
        if config.has_pre_norms:
            name = p + "attention_norm.weight"
            dh1 = self._norm_backward(saved["x_in"], w[name], dh1, grads, name)
        return dx_mid + dh1

    def _swiglu_backward(
        self,
        x: FloatArray,
        saved: Mapping[str, FloatArray],
        w: Mapping[str, npt.ArrayLike],
        prefix: str,
        dy: FloatArray,
        grads: dict[str, FloatArray],
    ) -> FloatArray:
        """Back through ``down(act(gate(x)) * up(x))`` (the weights ``prefix + gate/up/down.weight``): the gradient
        of ``x``, the gate and up paths added in that order; the weight gradients go to ``grads``."""
        dproduct_t, grads[prefix + "down.weight"], _ = linear_backward(saved["product"], w[prefix + "down.weight"], dy)
        dproduct = dproduct_t.numpy()
        dgate = self._activation_backward(saved["gate"], dproduct * saved["up"])
        dup = dproduct * saved["gate_act"]
        dx_gate, grads[prefix + "gate.weight"], _ = linear_backward(x, w[prefix + "gate.weight"], dgate)
        dx_up, grads[prefix + "up.weight"], _ = linear_backward(x, w[prefix + "up.weight"], dup)
        return dx_gate.numpy() + dx_up.numpy()

    def _experts_backward(
        self,
        saved: dict,
        dmlp: FloatArray,
        grads: dict[str, FloatArray],
        dprobabilities: FloatArray | None,
    ) -> FloatArray:
        """Back through a mixture-of-experts block ``out[r] = sum_e weight[r, e] * expert_e(h2[r])`` (experts
        added in increasing order). Each expert's rows (ascending) go back through it as one group; the gradient of
        a routing weight is the ``dot`` kernel of the row's output gradient and the expert's output; the router's
        logits get theirs from ``moe_route_backward``. A shared expert sees every row: its output gradient is scaled
        by the gate's sigmoid ``s``, whose score gets ``dot(dout, y) * s * (1 - s)``. The gradient of ``h2`` adds each
        row's experts in increasing order, then the shared expert's, its gate's and the router's."""
        config = self.config
        w = saved["weights"]
        p = saved["prefix"]
        h2 = saved["h2"]
        chosen, routing, logits = saved["routing"]
        dh2 = np.zeros_like(h2)
        dweights = np.zeros(routing.shape, dtype=np.float32)
        for expert in range(config.experts):
            prefix = f"{p}mlp.experts.{expert}."
            group = saved["experts"].get(expert)
            if group is None:  # routed no row of this sequence
                for name in ("gate", "up", "down"):
                    grads[prefix + name + ".weight"] = np.zeros(np.shape(w[prefix + name + ".weight"]), np.float32)
                continue
            rows, ranks = group["rows"], group["ranks"]
            dout = np.ascontiguousarray(dmlp[rows])
            for i, (row, rank) in enumerate(zip(rows, ranks, strict=True)):
                dweights[row, rank] = dot(dout[i], group["y"][i])
            dy = dout * routing[rows, ranks][:, None]
            dh2[rows] += self._swiglu_backward(group["x"], group, w, prefix, dy, grads)
        shared = saved.get("shared")
        if shared is not None:
            dy = dmlp
            if config.shared_expert_gate:
                s = shared["s"]
                dy = dmlp * s
                ds = np.array([[dot(dmlp[r], shared["y"][r])] for r in range(len(h2))], dtype=np.float32)
                dscore = ds * s * (np.float32(1.0) - s)
            dh2 = dh2 + self._swiglu_backward(h2, shared, w, f"{p}mlp.shared.", np.ascontiguousarray(dy), grads)
            if config.shared_expert_gate:
                name = p + "mlp.shared_gate.weight"
                dh2_gate, grads[name], _ = linear_backward(h2, w[name], dscore)
                dh2 = dh2 + dh2_gate.numpy()
        dlogits = moe_route_backward(logits, chosen, dweights, config.normalize_expert_weights, dprobabilities)
        name = p + "mlp.router.weight"
        dh2_router, grads[name], _ = linear_backward(h2, w[name], dlogits)
        return dh2 + dh2_router.numpy()

    def _router_loss(self, saved: list, router_scale: float) -> tuple[float, dict[int, FloatArray]]:
        """The load-balancing loss (:meth:`router_loss`) and, per sparse layer, the gradient ``[n, experts]`` of
        ``router_scale`` times it with respect to that layer's softmax probabilities: ``router_scale * experts *
        f_e / T`` in every row (``f_e`` counts choices, which have no gradient)."""
        config = self.config
        layers = [entry for entry in saved[:-1] if "routing" in entry]
        if not layers:
            return 0.0, {}
        chosen = np.concatenate([entry["routing"][0] for entry in layers])
        probabilities = np.stack([softmax(row) for entry in layers for row in entry["routing"][2]])
        rows = probabilities.shape[0]
        counts = np.bincount(chosen.reshape(-1), minlength=config.experts)
        fractions = (counts / rows).astype(np.float32)
        loss = config.experts * dot(fractions, column_mean(probabilities))
        if router_scale == 0.0:
            return loss, {}
        row = (router_scale * config.experts * counts / (rows * rows)).astype(np.float32)
        gradient = np.ascontiguousarray(np.broadcast_to(row, (len(saved[0]["x_in"]), config.experts)))
        return loss, {entry["layer"]: gradient for entry in layers}

    def _unfold(self, grads: dict[str, FloatArray]) -> dict[str, FloatArray]:
        """Gradients of the stored weights from those of the folded ones (``fold_scales``): the same elementwise
        float32 scales applied to the gradients."""
        config = self.config
        if config.residual_multiplier != 1.0:
            residual = np.float32(config.residual_multiplier)
            for name in grads:
                if name.endswith(("attention.o.weight", "down.weight")):
                    grads[name] = grads[name] * residual
        if config.rope_attention_factor != 1.0:
            for name, values in grads.items():
                if name.endswith(rope_scaled_tensors(config)):
                    rows = values.reshape(-1, config.head_dim, *values.shape[1:]).copy()
                    rows[:, : config.rotary_dimension] *= np.float32(config.rope_attention_factor)
                    grads[name] = rows.reshape(values.shape)
        return grads

    def _head(self, weights: Mapping[str, npt.ArrayLike]) -> npt.ArrayLike:
        return weights["token_embedding.weight" if self.config.tie_word_embeddings else "lm_head.weight"]

    def _forward(
        self,
        weights: Mapping[str, npt.ArrayLike],
        tokens: Sequence[int],
        *,
        keep: bool,
        add: tuple[int, np.ndarray] | None = None,
    ) -> tuple[FloatArray, list]:
        """The decoder forward pass over a whole sequence (``Transformer._layers`` without a cache). With ``keep``,
        returns the activations the backward pass needs: one dict per layer, then the final norm's input. ``add``,
        ``(layer, rows)``, adds ``rows`` to the residual stream after that layer."""
        config = self.config
        n = len(tokens)
        if any(not 0 <= t < config.vocabulary_size for t in tokens):
            raise ValueError("token id out of range")
        w = self._folded(weights)
        positions = np.arange(n, dtype=np.int64)
        embedding = np.asarray(weights["token_embedding.weight"], dtype=np.float32)
        x = embedding[np.asarray(tokens, dtype=np.int64)]
        if config.embedding_multiplier != 1.0:
            x = x * np.float32(config.embedding_multiplier)
        x = np.ascontiguousarray(x, dtype=np.float32)
        saved: list = []
        heads, kv_heads, head_dim = config.heads, config.kv_heads, config.head_dim
        for layer in range(config.layers):
            p = f"layers.{layer}."
            h1 = self._norm(x, w[p + "attention_norm.weight"]) if config.has_pre_norms else x
            q_t = linear(h1, w[p + "attention.q.weight"], w.get(p + "attention.q.bias"))
            k_t = linear(h1, w[p + "attention.k.weight"], w.get(p + "attention.k.bias"))
            v = linear(h1, w[p + "attention.v.weight"], w.get(p + "attention.v.bias")).numpy()
            q_raw, k_raw = q_t.numpy(), k_t.numpy()
            q, k = q_raw, k_raw
            if config.qk_norm and config.qk_norm_scope == "all":  # OLMo 2: over the whole projection
                q = self._norm(q_raw, w[p + "attention.q_norm.weight"])
                k = self._norm(k_raw, w[p + "attention.k_norm.weight"])
            q, k = q.reshape(n, heads, head_dim), k.reshape(n, kv_heads, head_dim)
            if config.qk_norm and config.qk_norm_scope == "head":  # Qwen3, Gemma 3: over each head, before the rotation
                q = self._norm(q, w[p + "attention.q_norm.weight"])
                k = self._norm(k, w[p + "attention.k_norm.weight"])
            inv_freq = self.local_inv_freq if config.uses_local_rope(layer) else self.inv_freq
            q = rope(q, positions, inv_freq).numpy()
            k = rope(k, positions, inv_freq).numpy()
            v = v.reshape(n, kv_heads, head_dim)
            attended = attention(
                q,
                k,
                v,
                scale=config.attention_scale,
                causal=True,
                q_offset=0,
                window=config.window(layer),
                softcap=config.attention_softcap,
            )
            attended = attended.reshape(n, heads * head_dim).numpy()
            out = linear(attended, w[p + "attention.o.weight"]).numpy()
            out_normed = self._norm(out, w[p + "attention_post_norm.weight"]) if config.has_post_norms else out
            x_mid = x + out_normed
            h2 = self._norm(x_mid, w[p + "mlp_norm.weight"]) if config.has_pre_norms else x_mid
            if config.is_sparse(layer):
                mlp, pieces = self._experts(w, p, layer, h2)
            else:
                pieces = self._swiglu(w, p + "mlp.", h2)
                mlp = pieces["y"]
            mlp_normed = self._norm(mlp, w[p + "mlp_post_norm.weight"]) if config.has_post_norms else mlp
            x_out = x_mid + mlp_normed
            if add is not None and add[0] == layer:
                x_out = x_out + add[1]
            if keep:
                saved.append(
                    {
                        "layer": layer,
                        "prefix": p,
                        "weights": w,
                        "x_in": x,
                        "h1": h1,
                        "q_raw": q_raw,
                        "k_raw": k_raw,
                        "q": q,
                        "k": k,
                        "v": v,
                        "attended": attended,
                        "out": out,
                        "x_mid": x_mid,
                        "h2": h2,
                        "mlp": mlp,
                        **pieces,
                    }
                )
            x = x_out
        if keep:
            saved.append(x)
        return self._norm(x, w["final_norm.weight"]), saved

    def _swiglu(self, w: Mapping[str, npt.ArrayLike], prefix: str, x: FloatArray) -> dict[str, FloatArray]:
        """``down(act(gate(x)) * up(x))`` (as ``"y"``) with the activations its backward pass needs."""
        gate = linear(x, w[prefix + "gate.weight"]).numpy()
        up = linear(x, w[prefix + "up.weight"]).numpy()
        gate_act = self._activation(gate)
        product = gate_act * up
        y = linear(product, w[prefix + "down.weight"]).numpy()
        return {"gate": gate, "up": up, "gate_act": gate_act, "product": product, "y": y}

    def _experts(
        self, w: Mapping[str, npt.ArrayLike], p: str, layer: int, h2: FloatArray
    ) -> tuple[FloatArray, dict[str, object]]:
        """``Transformer._experts``: each expert runs on the rows routed to it (ascending); ``out`` adds each row's
        weighted expert outputs to zero in increasing expert order, then the shared expert's output (times its gate's
        sigmoid when it has one)."""
        config = self.config
        logits = linear(h2, w[p + "mlp.router.weight"]).numpy()
        chosen, routing = moe_route(logits, config.experts_per_token, config.normalize_expert_weights)
        out = np.zeros_like(h2)
        groups: dict[int, dict[str, FloatArray]] = {}
        for expert in np.unique(chosen):
            rows, ranks = np.nonzero(chosen == expert)
            x = np.ascontiguousarray(h2[rows])
            group = self._swiglu(w, f"{p}mlp.experts.{int(expert)}.", x)
            out[rows] += group["y"] * routing[rows, ranks][:, None]
            groups[int(expert)] = {"rows": rows, "ranks": ranks, "x": x, **group}
        pieces: dict[str, object] = {"routing": (chosen, routing, logits), "experts": groups}
        if config.shared_expert_intermediate_size is not None:
            shared = self._swiglu(w, f"{p}mlp.shared.", h2)
            y = shared["y"]
            if config.shared_expert_gate:
                shared["s"] = sigmoid_elementwise(linear(h2, w[p + "mlp.shared_gate.weight"]).numpy())
                y = y * shared["s"]
            out = out + y
            pieces["shared"] = shared
        return out, pieces

    def _folded(self, weights: Mapping[str, npt.ArrayLike]) -> Mapping[str, npt.ArrayLike]:
        """The weights as the decoder uses them (``fold_scales``); the stored ones when nothing is folded."""
        config = self.config
        if config.residual_multiplier == 1.0 and config.rope_attention_factor == 1.0:
            return weights
        if isinstance(weights, dict):
            folded = fold_scales(config, {name: Tensor(values) for name, values in weights.items()})
            return {name: tensor.numpy() for name, tensor in folded.items()}
        return _FoldedWeights(config, weights)  # a lazy mapping (a quantised base) stays lazy


class _FoldedWeights(Mapping[str, npt.ArrayLike]):
    """``fold_scales`` of a lazy weights mapping, one weight at a time when it is read (the same bits)."""

    def __init__(self, config: TransformerConfig, weights: Mapping[str, npt.ArrayLike]) -> None:
        self._config, self._weights = config, weights

    def __getitem__(self, name: str) -> npt.ArrayLike:
        return fold_scales(self._config, {name: Tensor(self._weights[name])})[name].numpy()

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._weights)

    def __len__(self) -> int:
        return len(self._weights)
