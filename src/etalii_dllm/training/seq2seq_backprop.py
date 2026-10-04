"""Reverse-mode gradients of T5 text-to-text models (:mod:`etalii_dllm.seq2seq`, #387).

A training example is a source (ending with ``</s>``) and a target (ending with ``</s>``). The decoder reads
``[0, *target[:-1]]`` (teacher forcing) and is scored on predicting every target token: the summed softmax
cross-entropy, transformers' ``T5ForConditionalGeneration`` loss with ``labels=target`` times the number of targets.

The forward pass runs the whole target at once. Its self-attention is ``biased_attention`` over every decoder
position with the decoder's one-directional bucket bias and ``-inf`` for keys after the query: their exponentials are
exactly 0 and the kernel's maximum starts at key 0, which every query sees, so each row equals the served model's
incremental step (:meth:`TextToText._step`) bit for bit. Cross-attention is ``biased_attention`` over the encoder's
states with a zero bias and scale 1, and the encoder runs the T5 pass of
:class:`~etalii_dllm.training.encoder_backprop.EncoderGradients`.

The backward pass runs the gradient kernels in reverse: ``cross_entropy``, ``linear_backward`` of the LM head (with a
tied head the gradient then takes the forward's float32 factor ``hidden ** -0.5``), ``rms_norm_backward``, and per
decoder layer, last to first, the MLP (as in the encoder), cross-attention (``biased_attention_backward``; the key and
value gradients go through the layer's ``k`` and ``v`` projections to the encoder states, where every layer's
contribution is added in float32, ``k`` then ``v``, last layer first) and self-attention, whose bias gradient
scatters into the decoder's bucket table with ``embedding_backward`` as the encoder's does (the masked entries carry
exact zeros). The encoder states' gradient then runs the encoder's backward pass. The shared word embedding adds its
gradients in float32 in a fixed order: the tied head's, the decoder inputs', then the encoder's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.encoder import activate, t5_relative_buckets
from etalii_dllm.numerics import (
    FloatArray,
    biased_attention,
    biased_attention_backward,
    cross_entropy,
    embedding_backward,
    linear,
    linear_backward,
)
from etalii_dllm.seq2seq import DECODER_START
from etalii_dllm.training.encoder_backprop import EncoderGradients, EncoderPass


@dataclass
class TextToTextPass:
    """One forward pass over a source and the decoder inputs, with what the backward pass needs."""

    source: npt.NDArray[np.int64]
    inputs: npt.NDArray[np.int64]
    encoder: EncoderPass
    logits: FloatArray
    """``[len(inputs), vocabulary]``: the logits after every decoder input."""
    layers: list[dict[str, FloatArray]] = field(default_factory=list)
    final: dict[str, FloatArray] = field(default_factory=dict)


class TextToTextGradients:
    """Forward and backward passes of one T5 text-to-text configuration (``config.is_text_to_text``). ``weights``
    maps the names of ``TransformerConfig.tensor_shapes()`` to float32 arrays; they are read, never modified."""

    def __init__(self, config: TransformerConfig) -> None:
        if not config.is_text_to_text:
            raise ValueError(f"{config.family} is not a text-to-text model")
        self.config = config
        self.encoder = EncoderGradients(replace(config, decoder_layers=0, tie_word_embeddings=True))

    # Forward

    def forward(
        self,
        weights: Mapping[str, npt.ArrayLike],
        source: Sequence[int],
        inputs: Sequence[int],
        *,
        add: tuple[int, npt.ArrayLike] | None = None,
    ) -> TextToTextPass:
        """The logits after every decoder input (``inputs[0]`` is the start token), as the served model computes
        them one step at a time. ``add`` is ``(layer, delta[source, hidden])``, added to the encoder's residual
        stream after that 0-based layer (model editing, #407)."""
        config, e = self.config, self.encoder
        if not inputs:
            raise ValueError("the decoder needs at least one input")
        if min(inputs) < 0 or max(inputs) >= config.vocabulary_size:
            raise ValueError("token id out of range")
        encoded = e.encode(weights, source, add=add)
        states = encoded.states
        n, m, heads, head_dim = len(inputs), len(source), config.heads, config.head_dim
        ids = np.asarray(inputs, dtype=np.int64)
        bias = self._self_bias(weights, n)
        zero = np.zeros((heads, n, m), dtype=np.float32)
        h = np.ascontiguousarray(np.asarray(weights["token_embedding.weight"], dtype=np.float32)[ids])
        result = TextToTextPass(np.asarray(source, dtype=np.int64), ids, encoded, h)
        for i in range(config.decoder_layers):
            p = f"decoder.layers.{i}."
            x = e._rms(weights, h, p + "attention_norm")
            q = e._linear(weights, x, p + "attention.q").reshape(n, heads, head_dim)
            k = e._linear(weights, x, p + "attention.k").reshape(n, heads, head_dim)
            v = e._linear(weights, x, p + "attention.v").reshape(n, heads, head_dim)
            mixed = biased_attention(q, k, v, bias, 1.0).numpy().reshape(n, -1)
            a = h + e._linear(weights, mixed, p + "attention.o")
            x2 = e._rms(weights, a, p + "cross_norm")
            cq = e._linear(weights, x2, p + "cross.q").reshape(n, heads, head_dim)
            ck = e._linear(weights, states, p + "cross.k").reshape(m, heads, head_dim)
            cv = e._linear(weights, states, p + "cross.v").reshape(m, heads, head_dim)
            crossed = biased_attention(cq, ck, cv, zero, 1.0).numpy().reshape(n, -1)
            b = a + e._linear(weights, crossed, p + "cross.o")
            x3 = e._rms(weights, b, p + "mlp_norm")
            up = e._linear(weights, x3, p + "mlp.up")
            saved = {"h": h, "x": x, "q": q, "k": k, "v": v, "mixed": mixed, "a": a, "x2": x2}
            saved |= {"cq": cq, "ck": ck, "cv": cv, "crossed": crossed, "b": b, "x3": x3, "up": up}
            if config.gated_mlp:
                gate = e._linear(weights, x3, p + "mlp.gate")
                activated = activate(gate, config.activation)
                saved |= {"gate": gate, "act": activated, "inner": activated * up}
            else:
                saved["inner"] = activate(up, config.activation)
            result.layers.append(saved)
            h = b + e._linear(weights, saved["inner"], p + "mlp.down")
        out = e._rms(weights, h, "decoder.final_norm")
        if config.tie_word_embeddings:
            out = (out * np.float32(config.hidden_size**-0.5)).astype(np.float32)
        result.final = {"h": h, "out": out, "bias": bias, "zero": zero}
        result.logits = linear(out, self._head(weights)).numpy()
        return result

    def logits(self, weights: Mapping[str, npt.ArrayLike], source: Sequence[int], target: Sequence[int]) -> FloatArray:
        """The logits ``[len(target), vocabulary]`` the model scores ``target`` with (teacher forcing)."""
        return self.forward(weights, source, [DECODER_START, *target[:-1]]).logits

    # Backward

    def loss_and_gradients(
        self,
        weights: Mapping[str, npt.ArrayLike],
        source: Sequence[int],
        target: Sequence[int],
        *,
        scale: float = 1.0,
    ) -> tuple[float, dict[str, FloatArray]]:
        """The summed cross-entropy of ``target`` given ``source`` and the gradient of ``scale`` times it for every
        weight."""
        if not target:
            raise ValueError("the target needs at least one token")
        result = self.forward(weights, source, [DECODER_START, *target[:-1]])
        loss, dlogits = cross_entropy(result.logits, np.asarray(target, dtype=np.int64), scale=scale)
        return loss, self.gradients(weights, result, dlogits.numpy())

    def gradients(
        self, weights: Mapping[str, npt.ArrayLike], result: TextToTextPass, dlogits: npt.ArrayLike
    ) -> dict[str, FloatArray]:
        """The gradient of every weight from ``dlogits``, the loss gradient of ``result.logits``."""
        config, e = self.config, self.encoder
        grads: dict[str, FloatArray] = {}
        dstates, dhead, dinputs = self._decoder_gradients(weights, result, dlogits, grads)
        encoder_grads = e.gradients(weights, result.encoder, dstates)
        dencoder = encoder_grads.pop("token_embedding.weight")
        grads.update(encoder_grads)
        if config.tie_word_embeddings:
            grads["token_embedding.weight"] = dhead + dinputs + dencoder
        else:
            grads["lm_head.weight"] = dhead
            grads["token_embedding.weight"] = dinputs + dencoder
        return {name: np.array(values, dtype=np.float32) for name, values in grads.items()}

    def residual_gradient(
        self,
        weights: Mapping[str, npt.ArrayLike],
        source: Sequence[int],
        target: Sequence[int],
        layer: int,
        delta: npt.ArrayLike,
    ) -> tuple[float, FloatArray]:
        """With ``delta[source, hidden]`` added to the encoder's residual stream after ``layer`` (0-based): the summed
        cross-entropy of ``target`` given ``source`` (teacher forcing, as in :meth:`loss_and_gradients`) and its
        gradient with respect to that residual stream, ``[source, hidden]`` (#407). The decoder's backward pass
        brings the gradient to the encoder states through cross-attention; the encoder's later layers take it on."""
        if not target:
            raise ValueError("the target needs at least one token")
        result = self.forward(weights, source, [DECODER_START, *target[:-1]], add=(layer, delta))
        loss, dlogits = cross_entropy(result.logits, np.asarray(target, dtype=np.int64))
        dstates, _, _ = self._decoder_gradients(weights, result, dlogits.numpy(), {})
        return loss, self.encoder.residual_gradient(weights, result.encoder, dstates, layer)

    def _decoder_gradients(
        self,
        weights: Mapping[str, npt.ArrayLike],
        result: TextToTextPass,
        dlogits: npt.ArrayLike,
        grads: dict[str, FloatArray],
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """The decoder's backward pass: its weights' gradients into ``grads``; returns the gradients of the encoder
        states, of the LM head's matrix and of the decoder inputs' embedding rows."""
        config, e = self.config, self.encoder
        n, m, heads, head_dim = len(result.inputs), len(result.source), config.heads, config.head_dim
        buckets = config.position_buckets
        dout, dhead, _ = linear_backward(result.final["out"], self._head(weights), np.asarray(dlogits, np.float32))
        dh = dout.numpy()
        if config.tie_word_embeddings:
            dh = (dh * np.float32(config.hidden_size**-0.5)).astype(np.float32)
        dh = e._rms_backward(weights, result.final["h"], dh, grads, "decoder.final_norm")
        states = result.encoder.states
        dstates = np.zeros_like(states)
        dtable = np.zeros((heads, buckets), dtype=np.float32)
        targets = self._bias_targets(n)
        bias, zero = result.final["bias"], result.final["zero"]
        for i in reversed(range(config.decoder_layers)):
            p = f"decoder.layers.{i}."
            saved = result.layers[i]
            # h_out = b + down(mlp(x3)), x3 = RMSNorm(b)
            dinner = e._linear_backward(weights, saved["inner"], dh, grads, p + "mlp.down")
            if config.gated_mlp:  # inner = act(gate) * up
                dgate = e._activation_backward(saved["gate"], dinner * saved["up"])
                dup = dinner * saved["act"]
                dx3 = e._linear_backward(weights, saved["x3"], dgate, grads, p + "mlp.gate")
                dx3 = dx3 + e._linear_backward(weights, saved["x3"], dup, grads, p + "mlp.up")
            else:  # inner = act(up)
                dup = e._activation_backward(saved["up"], dinner)
                dx3 = e._linear_backward(weights, saved["x3"], dup, grads, p + "mlp.up")
            db = dh + e._rms_backward(weights, saved["b"], dx3, grads, p + "mlp_norm")
            # b = a + o(attention(q(x2), k(states), v(states))), x2 = RMSNorm(a)
            dcrossed = e._linear_backward(weights, saved["crossed"], db, grads, p + "cross.o")
            dcq, dck, dcv, _ = biased_attention_backward(
                saved["cq"], saved["ck"], saved["cv"], zero, dcrossed.reshape(n, heads, head_dim), 1.0
            )
            for name, d in (("k", dck), ("v", dcv)):
                part = e._linear_backward(weights, states, d.numpy().reshape(m, -1), grads, p + "cross." + name)
                dstates = dstates + part
            dx2 = e._linear_backward(weights, saved["x2"], dcq.numpy().reshape(n, -1), grads, p + "cross.q")
            da = db + e._rms_backward(weights, saved["a"], dx2, grads, p + "cross_norm")
            # a = h + o(attention(q(x), k(x), v(x), bias)), x = RMSNorm(h)
            dmixed = e._linear_backward(weights, saved["mixed"], da, grads, p + "attention.o")
            dq, dk, dv, dbias = biased_attention_backward(
                saved["q"], saved["k"], saved["v"], bias, dmixed.reshape(n, heads, head_dim), 1.0
            )
            scattered = embedding_backward(dbias.numpy().reshape(-1, 1), targets, heads * buckets)
            dtable = dtable + scattered.numpy().reshape(heads, buckets)
            dx = None
            for name, d in (("q", dq), ("k", dk), ("v", dv)):
                part = e._linear_backward(weights, saved["x"], d.numpy().reshape(n, -1), grads, p + "attention." + name)
                dx = part if dx is None else dx + part
            assert dx is not None
            dh = da + e._rms_backward(weights, saved["h"], dx, grads, p + "attention_norm")
        grads["decoder.relative_bias.weight"] = np.ascontiguousarray(dtable.T)
        dinputs = embedding_backward(dh, result.inputs, config.vocabulary_size).numpy()
        return dstates, dhead.numpy(), dinputs

    # Pieces

    def _head(self, weights: Mapping[str, npt.ArrayLike]) -> npt.ArrayLike:
        return weights["token_embedding.weight" if self.config.tie_word_embeddings else "lm_head.weight"]

    def _buckets(self, n: int) -> np.ndarray:
        positions = np.arange(n, dtype=np.int64)
        config = self.config
        return t5_relative_buckets(
            positions[None, :] - positions[:, None],
            config.position_buckets,
            config.max_relative_positions,
            bidirectional=False,
        )

    def _self_bias(self, weights: Mapping[str, npt.ArrayLike], n: int) -> np.ndarray:
        """``[heads, n, n]``: ``table[bucket(j - i), head]`` for keys ``j <= i``, ``-inf`` after the query."""
        table = np.asarray(weights["decoder.relative_bias.weight"], dtype=np.float32)
        bias = table[self._buckets(n)].transpose(2, 0, 1)
        later = np.triu(np.ones((n, n), dtype=bool), 1)
        return np.ascontiguousarray(np.where(later[None], np.float32(-np.inf), bias).astype(np.float32))

    def _bias_targets(self, n: int) -> np.ndarray:
        """Flat scatter targets of every (head, i, j): row ``h * buckets + bucket(j - i)`` of the transposed table."""
        heads, buckets = self.config.heads, self.config.position_buckets
        return (np.arange(heads, dtype=np.int64)[:, None, None] * buckets + self._buckets(n)[None]).reshape(-1)
