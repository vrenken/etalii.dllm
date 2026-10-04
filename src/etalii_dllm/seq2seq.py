"""T5 text-to-text models (T5ForConditionalGeneration, Flan-T5) on the deterministic kernels (#382, #384).

A text-to-text model reads a source text with its encoder (:class:`etalii_dllm.encoder.Encoder`, family ``t5``) and
writes the answer with its decoder, one token at a time. To fit the engine's one-sequence generation loop, the
sequence the generator holds is the source (ending with ``</s>``, which :class:`etalii_dllm.generation.Generator`
appends to every prompt of a text-to-text model) followed by the answer so far: everything up to and including the
last ``</s>`` goes to the encoder, the rest to the decoder after its start token (``<pad>``, id 0). An answer never
contains ``</s>``, since that token ends it.

The decoder, as transformers' ``T5Stack`` defines it (pre-norm, RMS norms without mean or bias, no biases, no score
scaling), for the new token at decoder position ``t``:

1. ``h = embedding[token]`` (the word embedding the encoder shares);
2. per layer, self-attention: ``x = RMSNorm(h)``, its key and value appended to the cache, ``h = h +
   o(biased_attention(q(x), K[:t+1], V[:t+1], bias, 1))`` where ``bias[head, j] = table[t5_bucket(j - t), head]``
   with the decoder's own table and one-directional buckets (``t5_relative_buckets(..., bidirectional=False)``).
   The new token's query only sees keys up to itself, so the cache needs no mask and gives exactly the bits of a
   recompute;
3. cross-attention: ``x = RMSNorm(h)``, ``h = h + o(biased_attention(q(x), K_enc, V_enc, 0, 1))`` where ``K_enc``
   and ``V_enc`` are the layer's key and value projections of the encoder states, computed once per source;
4. the MLP: ``h = h + down(act(up(RMSNorm(h))))`` or ``down(act(gate(x)) * up(x))``;
5. ``RMSNorm(h)`` with the final norm, times ``hidden ** -0.5`` (rounded to float32) when the LM head is the tied
   word embedding (T5 v1.0), then the LM head.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.encoder import Encoder, activate, t5_relative_buckets
from etalii_dllm.numerics import FloatArray, biased_attention, linear, rms_norm
from etalii_dllm.tensor import Tensor
from etalii_dllm.transformer import _MATRICES, _prepare, quantized_fingerprint

DECODER_START = 0
"""The decoder's first input, T5's ``decoder_start_token_id`` (the padding token)."""


@dataclass
class TextToTextCache:
    """A generation's encoder states and decoder keys and values; the rows are exactly what a recompute gives."""

    source: tuple[int, ...] | None = None
    cross: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    """Per decoder layer, the key and value projections ``[source, heads, head_dim]`` of the encoder states."""
    tokens: list[int] = field(default_factory=list)
    """The decoder inputs processed so far (the start token, then the answer)."""
    keys: list[list[np.ndarray]] = field(default_factory=list)
    values: list[list[np.ndarray]] = field(default_factory=list)


class TextToText:
    """A T5 encoder-decoder over a model file's tensors (``config.is_text_to_text``); ``quantize`` (``q8_0`` or
    ``q4_0``) runs the linear layers on quantised weights, which changes the output and so the fingerprint."""

    text_to_text = True

    def __init__(
        self,
        config: TransformerConfig,
        tensors: Mapping[str, npt.ArrayLike],
        *,
        model_id: str = "dllm-t5",
        weights_fingerprint: str = "",
        quantize: str | None = None,
        device: str = "cpu",
        release: Callable[[str], None] | None = None,
    ) -> None:
        if not config.is_text_to_text:
            raise ValueError(f"{config.family} is not a text-to-text model")
        if device != "cpu":
            raise ValueError("text-to-text models run on the CPU only")
        expected = config.tensor_shapes()
        missing = sorted(set(expected) - set(tensors))
        if missing:
            raise ValueError(f"missing tensors: {missing}")
        self.config = config
        self.quantization = quantize
        self.device = device
        self.weights_fingerprint = quantized_fingerprint(weights_fingerprint, quantize)
        self._id = model_id
        encoder_config = replace(config, decoder_layers=0, tie_word_embeddings=True)
        encoder_names = encoder_config.tensor_shapes()
        self.encoder = Encoder(
            encoder_config,
            {name: tensors[name] for name in encoder_names},
            model_id=model_id,
            quantize=quantize,
            release=release,
        )
        weights = {name: Tensor(tensors[name]) for name in expected if name not in encoder_names}
        for name, tensor in weights.items():
            if tensor.shape != expected[name]:
                raise ValueError(f"tensor {name!r} has shape {tensor.shape}, expected {expected[name]}")
        self.tensors: Mapping[str, Tensor] = {**self.encoder.tensors, **weights}
        self._w: dict[str, object] = {}
        for name, tensor in weights.items():
            if name.endswith(_MATRICES) or name == "lm_head.weight":
                self._w[name] = _prepare(tensor, quantize, "cpu")
                if release is not None:
                    release(name)
            else:
                self._w[name] = tensor.numpy()
        embedding = self.encoder.tensors["token_embedding.weight"]
        self._embedding = np.asarray(embedding.numpy(), dtype=np.float32)
        tied = config.tie_word_embeddings
        self._head = _prepare(embedding, quantize, "cpu") if tied else self._w["lm_head.weight"]
        self._head_scale = np.float32(config.hidden_size**-0.5) if config.tie_word_embeddings else None
        self.end_of_source = config.eos_token_ids[0] if config.eos_token_ids else 1
        """The token that ends the source (``</s>``); everything after its last occurrence is the answer."""

    @property
    def id(self) -> str:
        return self._id

    @property
    def vocabulary_size(self) -> int:
        return self.config.vocabulary_size

    @property
    def context_length(self) -> int:
        return self.config.context_length

    def split(self, tokens: Sequence[int]) -> tuple[list[int], list[int]]:
        """``(source, answer)``: the tokens up to and including the last ``</s>``, and the ones after it."""
        tokens = list(tokens)
        ends = [i for i, token in enumerate(tokens) if token == self.end_of_source]
        if not ends:
            raise ValueError("a text-to-text model's sequence needs its source, ending with </s>")
        return tokens[: ends[-1] + 1], tokens[ends[-1] + 1 :]

    def forward(self, tokens: Sequence[int]) -> FloatArray:
        """The next answer token's logits for ``tokens`` (the source, then the answer so far)."""
        return self.forward_cached(tokens, self.new_cache())

    def new_cache(self) -> TextToTextCache:
        return TextToTextCache()

    def forward_cached(self, tokens: Sequence[int], cache: TextToTextCache) -> FloatArray:
        """:meth:`forward`, reusing the encoder states and the decoder's keys and values in ``cache`` when the
        source is the same and the answer extends the one already processed. Gives the same bits as :meth:`forward`."""
        source, answer = self.split(tokens)
        if cache.source != tuple(source):
            self._encode(source, cache)
        inputs = [DECODER_START, *answer]
        if cache.tokens != inputs[: len(cache.tokens)] or len(cache.tokens) == len(inputs):
            cache.tokens = []
            cache.keys = [[] for _ in range(self.config.decoder_layers)]
            cache.values = [[] for _ in range(self.config.decoder_layers)]
        logits = None
        for token in inputs[len(cache.tokens) :]:
            logits = self._step(token, cache)
        assert logits is not None
        return logits

    def encoder_states(self, source: Sequence[int]) -> FloatArray:
        """The encoder's last states ``[source, hidden]`` (after its final norm)."""
        return self.encoder.hidden_states(source)

    def _encode(self, source: Sequence[int], cache: TextToTextCache) -> None:
        config = self.config
        states = self.encoder_states(source)
        count, heads, head_dim = len(source), config.heads, config.head_dim
        cache.source = tuple(source)
        cache.cross = []
        for i in range(config.decoder_layers):
            p = f"decoder.layers.{i}.cross."
            k = self._linear(states, p + "k").reshape(count, heads, head_dim)
            v = self._linear(states, p + "v").reshape(count, heads, head_dim)
            cache.cross.append((k, v))
        cache.tokens, cache.keys, cache.values = [], [], []

    def _step(self, token: int, cache: TextToTextCache) -> FloatArray:
        """Appends ``token`` at the next decoder position and returns the logits after it."""
        config = self.config
        if not 0 <= token < config.vocabulary_size:
            raise ValueError("token id out of range")
        heads, head_dim, eps = config.heads, config.head_dim, config.rms_norm_eps
        if not cache.keys:
            cache.keys = [[] for _ in range(config.decoder_layers)]
            cache.values = [[] for _ in range(config.decoder_layers)]
        position = len(cache.tokens)
        keys_seen = np.arange(position + 1, dtype=np.int64)
        buckets = t5_relative_buckets(
            keys_seen - position, config.position_buckets, config.max_relative_positions, bidirectional=False
        )
        table = np.asarray(self._w["decoder.relative_bias.weight"], dtype=np.float32)
        bias = np.ascontiguousarray(table[buckets].T.reshape(heads, 1, position + 1))
        source_length = len(cache.cross[0][0]) if cache.cross else 0
        zero = np.zeros((heads, 1, source_length), dtype=np.float32)
        h = self._embedding[token : token + 1].copy()
        for i in range(config.decoder_layers):
            p = f"decoder.layers.{i}."
            x = rms_norm(h, self._w[p + "attention_norm.weight"], eps).numpy()  # type: ignore[arg-type]
            q = self._linear(x, p + "attention.q").reshape(1, heads, head_dim)
            cache.keys[i].append(self._linear(x, p + "attention.k").reshape(heads, head_dim))
            cache.values[i].append(self._linear(x, p + "attention.v").reshape(heads, head_dim))
            k, v = np.stack(cache.keys[i]), np.stack(cache.values[i])
            mixed = biased_attention(q, k, v, bias, 1.0).numpy().reshape(1, -1)
            h = h + self._linear(mixed, p + "attention.o")
            x = rms_norm(h, self._w[p + "cross_norm.weight"], eps).numpy()  # type: ignore[arg-type]
            q = self._linear(x, p + "cross.q").reshape(1, heads, head_dim)
            k, v = cache.cross[i]
            mixed = biased_attention(q, k, v, zero, 1.0).numpy().reshape(1, -1)
            h = h + self._linear(mixed, p + "cross.o")
            x = rms_norm(h, self._w[p + "mlp_norm.weight"], eps).numpy()  # type: ignore[arg-type]
            h = h + self._linear(self._mlp(x, p), p + "mlp.down")
        cache.tokens.append(token)
        out = rms_norm(h, self._w["decoder.final_norm.weight"], eps).numpy()  # type: ignore[arg-type]
        if self._head_scale is not None:
            out = (out * self._head_scale).astype(np.float32)
        return linear(out, self._head).numpy().reshape(-1)  # type: ignore[arg-type,no-any-return]

    def _mlp(self, x: np.ndarray, p: str) -> np.ndarray:
        up = self._linear(x, p + "mlp.up")
        if not self.config.gated_mlp:
            return activate(up, self.config.activation)
        return activate(self._linear(x, p + "mlp.gate"), self.config.activation) * up

    def _linear(self, x: np.ndarray, name: str) -> np.ndarray:
        return linear(x, self._w[name + ".weight"]).numpy()  # type: ignore[arg-type]
