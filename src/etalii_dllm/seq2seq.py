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

A steering vector (#405) is added (float32) to ``h`` after its decoder layer, as decoder-only models add theirs. A
:class:`Recorder` passed to the encoder and decoder steps is shown every intermediate (``etalii_dllm.interpret``);
it only copies, so a traced pass has exactly the bits of an untraced one (#402).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.encoder import Encoder, activate, t5_relative_buckets
from etalii_dllm.numerics import FloatArray, biased_attention, biased_attention_weights, linear, rms_norm
from etalii_dllm.tensor import Tensor
from etalii_dllm.transformer import _MATRICES, _prepare, quantized_fingerprint, steered_fingerprint

DECODER_START = 0
"""The decoder's first input, T5's ``decoder_start_token_id`` (the padding token)."""


class Recorder(Protocol):
    """Watches a T5 encoder or decoder pass (:func:`etalii_dllm.interpret.trace`). ``record`` is shown each
    intermediate by name and 0-based layer; the attention probabilities (``attention``, ``cross_attention``:
    ``[rows, heads, keys]``) are computed only when ``wants_attention``."""

    wants_attention: bool

    def record(self, part: str, layer: int, values: np.ndarray) -> None: ...


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

    def reusable(self, prompt: Sequence[int]) -> int:
        """How many tokens of ``prompt`` (a source ending with ``</s>``, then maybe an answer) this cache saves in
        the prompt cache (#414): none unless it holds the very same source (the encoder is bidirectional, so a
        changed token changes every state), else the source and the answer tokens it shares, less the last one,
        which the decoder reads again for its logits."""
        if not self.source:
            return 0
        ends = [i for i, token in enumerate(prompt) if token == self.source[-1]]  # the source ends with </s>
        if not ends or tuple(prompt[: ends[-1] + 1]) != self.source:
            return 0
        answer = list(prompt[ends[-1] + 1 :])
        shared = 0
        for cached, token in zip(self.tokens[1:], answer, strict=False):
            if cached != token:
                break
            shared += 1
        return len(self.source) + min(shared, max(len(answer) - 1, 0))

    def covers(self, other: Any) -> bool:
        """Whether this cache serves every prompt ``other`` would: the same source, and its answer extends
        ``other``'s."""
        return (
            isinstance(other, TextToTextCache)
            and other.source == self.source
            and self.tokens[: len(other.tokens)] == other.tokens
        )

    def export(self) -> tuple[Any, ...]:
        """What :meth:`restore` needs to make an independent copy (beam search's hypotheses each keep one). The
        arrays are shared: they are never changed in place, only appended to the copy's own lists."""
        return self.source, list(self.cross), list(self.tokens), [*map(list, self.keys)], [*map(list, self.values)]

    def restore(
        self,
        source: tuple[int, ...] | None,
        cross: list[tuple[np.ndarray, np.ndarray]],
        tokens: list[int],
        keys: list[list[np.ndarray]],
        values: list[list[np.ndarray]],
    ) -> None:
        """Fills an empty cache with what :meth:`export` returned."""
        if self.source is not None or self.tokens:
            raise ValueError("only an empty cache can be restored")
        self.source, self.cross, self.tokens = source, list(cross), list(tokens)
        self.keys, self.values = [*map(list, keys)], [*map(list, values)]


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
        steering: Mapping[int, npt.ArrayLike] | None = None,
        release: Callable[[str], None] | None = None,
    ) -> None:
        """``steering`` maps 0-based decoder layer indices to vectors ``[hidden]`` added (float32) to the decoder's
        residual stream after that layer at every position (#405); it changes the output and the fingerprint."""
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
        self.steering: dict[int, np.ndarray] = {}
        for layer, vector in (steering or {}).items():
            values = np.ascontiguousarray(vector, dtype=np.float32)
            if not 0 <= int(layer) < config.decoder_layers or values.shape != (config.hidden_size,):
                raise ValueError(
                    f"steering needs a decoder layer in 0..{config.decoder_layers - 1} and a vector of "
                    f"{config.hidden_size}"
                )
            self.steering[int(layer)] = values
        self.weights_fingerprint = quantized_fingerprint(
            steered_fingerprint(weights_fingerprint, self.steering), quantize
        )
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
        return self.forward_batch([tokens], [cache])[0]

    def forward_batch(
        self, sequences: Sequence[Sequence[int]], caches: Sequence[TextToTextCache] | None = None
    ) -> list[FloatArray]:
        """Next-token logits for several independent sequences (#395): each source is encoded on its own (once per
        cache), then the decoder steps of all sequences run together, their rows stacked through every linear
        layer and attention per sequence against its own cache. Every kernel computes each row on its own in a fixed
        order, so each sequence gets exactly the bits of a lone :meth:`forward_cached`. ``caches`` (one per
        sequence, distinct) default to fresh ones."""
        if caches is None:
            caches = [self.new_cache() for _ in sequences]
        if len(caches) != len(sequences):
            raise ValueError("forward_batch needs one cache per sequence")
        if len({id(cache) for cache in caches}) != len(caches):
            raise ValueError("forward_batch needs a distinct cache per sequence")
        pending = [self._pending(tokens, cache) for tokens, cache in zip(sequences, caches, strict=True)]
        results: list[FloatArray | None] = [None] * len(caches)
        step = 0
        while True:  # one decoder position of every sequence that still has tokens to read
            active = [i for i, tokens in enumerate(pending) if step < len(tokens)]
            if not active:
                break
            logits = self._steps([pending[i][step] for i in active], [caches[i] for i in active])
            for row, i in enumerate(active):
                if step == len(pending[i]) - 1:
                    results[i] = logits[row].copy()
            step += 1
        return [logits for logits in results if logits is not None]

    def forward_cached_last(self, tokens: Sequence[int], cache: TextToTextCache, count: int) -> FloatArray:
        """The logits ``[count, vocabulary]`` after each of the last ``count`` tokens of ``tokens`` (the source,
        then the answer so far), reusing ``cache`` like :meth:`forward_cached` (#412). The answer tokens the cache
        does not hold go through the decoder in one pass, their rows stacked through every linear layer, each
        attending to the keys up to its own position, so row ``i`` has exactly the bits of
        ``forward_cached(tokens[: len(tokens) - count + 1 + i])``. The last ``count`` tokens must be answer tokens
        (or the source's closing ``</s>`` for the first row); speculative decoding checks a draft with one call."""
        _, answer = self.split(tokens)
        if not 1 <= count <= len(answer) + 1:
            raise ValueError("count must be between 1 and the number of answer tokens plus one")
        inputs = self._pending(tokens, cache)
        if count > len(inputs):  # the cache holds positions whose logits are asked for: recompute them
            keep = len(cache.tokens) - (count - len(inputs))
            self._truncate(cache, keep)
            inputs = [DECODER_START, *answer][keep:]
        return self._steps(inputs, [cache] * len(inputs))[-count:]

    def _pending(self, tokens: Sequence[int], cache: TextToTextCache) -> list[int]:
        """Prepares ``cache`` for ``tokens`` and returns the decoder inputs it does not hold yet (at least one): the
        source is encoded unless the cache has it, and the decoder keeps the keys of the inputs it shares with the
        new answer (each position's keys depend only on the inputs up to it, so they are what a recompute gives)."""
        source, answer = self.split(tokens)
        if cache.source != tuple(source):
            self._encode(source, cache)
        inputs = [DECODER_START, *answer]
        for token in inputs:
            if not 0 <= token < self.config.vocabulary_size:
                raise ValueError("token id out of range")
        shared = 0
        for cached, token in zip(cache.tokens, inputs, strict=False):
            if cached != token:
                break
            shared += 1
        self._truncate(cache, min(shared, len(inputs) - 1))
        return inputs[len(cache.tokens) :]

    def _truncate(self, cache: TextToTextCache, keep: int) -> None:
        """Drops the decoder positions of ``cache`` from ``keep`` on (the lists are replaced, never changed in
        place, so a copy :meth:`TextToTextCache.export` made keeps its rows)."""
        if keep == len(cache.tokens):
            return
        cache.tokens = cache.tokens[:keep]
        if keep == 0:
            cache.keys = [[] for _ in range(self.config.decoder_layers)]
            cache.values = [[] for _ in range(self.config.decoder_layers)]
        else:
            cache.keys = [layer[:keep] for layer in cache.keys]
            cache.values = [layer[:keep] for layer in cache.values]

    def answer_logits(self, source: Sequence[int], answer: Sequence[int]) -> np.ndarray:
        """The logits ``[len(answer), vocabulary]`` before each answer token: row ``i`` follows the source (ending
        with ``</s>``) and ``answer[:i]``, with exactly the bits :meth:`forward` gives for that sequence (#393). Only
        the answer's last token may be ``</s>``, which ends a complete answer (#400)."""
        source, answer = list(source), [int(token) for token in answer]
        if not source or source[-1] != self.end_of_source or self.end_of_source in answer[:-1]:
            raise ValueError(
                "a text-to-text model scores an answer (</s> only at its end) after a source ending with </s>"
            )
        if not answer:
            return np.zeros((0, self.vocabulary_size), dtype=np.float32)
        if any(not 0 <= token < self.vocabulary_size for token in answer):
            raise ValueError("token id out of range")
        cache = self.new_cache()
        self._encode(source, cache)
        inputs = [DECODER_START, *answer[:-1]]
        return self._steps(inputs, [cache] * len(inputs))  # one pass, the bits of one step per token (#412)

    def encoder_states(self, source: Sequence[int], recorder: Recorder | None = None) -> FloatArray:
        """The encoder's last states ``[source, hidden]`` (after its final norm), shown to ``recorder`` on the way."""
        return self.encoder.hidden_states(source, recorder=recorder)

    def final_norm(self, x: npt.ArrayLike) -> FloatArray:
        """The decoder's final RMSNorm of residual stream rows ``[rows, hidden]``."""
        weight = self._w["decoder.final_norm.weight"]
        return rms_norm(x, weight, self.config.rms_norm_eps).numpy().copy()  # type: ignore[arg-type]

    def logits_from_hidden(self, hidden: npt.ArrayLike) -> FloatArray:
        """Logits ``[rows, vocabulary]`` of final-norm decoder states: the tied head's scale (T5 v1.0), then the LM
        head, exactly as :meth:`forward` applies them."""
        out = np.ascontiguousarray(hidden, dtype=np.float32)
        if self._head_scale is not None:
            out = (out * self._head_scale).astype(np.float32)
        return linear(out, self._head).numpy()  # type: ignore[arg-type,no-any-return]

    def _encode(self, source: Sequence[int], cache: TextToTextCache, recorder: Recorder | None = None) -> None:
        config = self.config
        states = self.encoder_states(source, recorder)
        count, heads, head_dim = len(source), config.heads, config.head_dim
        cache.source = tuple(source)
        cache.cross = []
        for i in range(config.decoder_layers):
            p = f"decoder.layers.{i}.cross."
            k = self._linear(states, p + "k").reshape(count, heads, head_dim)
            v = self._linear(states, p + "v").reshape(count, heads, head_dim)
            cache.cross.append((k, v))
        cache.tokens, cache.keys, cache.values = [], [], []

    def _steps(
        self, tokens: Sequence[int], caches: Sequence[TextToTextCache], recorder: Recorder | None = None
    ) -> np.ndarray:
        """Appends ``tokens[i]`` at the next decoder position of ``caches[i]`` and returns the logits after each
        ``[len(tokens), vocabulary]``: the rows go through the linear layers together, attention runs per row. A
        cache may appear several times (#412): its rows take its next positions in order, each attending to the keys
        up to its own. ``recorder`` (one cache only) sees every intermediate."""
        if recorder is not None and len(caches) != 1:
            raise ValueError("a recorder watches one sequence")
        config = self.config
        heads, head_dim, eps = config.heads, config.head_dim, config.rms_norm_eps
        count = len(tokens)
        table = np.asarray(self._w["decoder.relative_bias.weight"], dtype=np.float32)
        biases, zeros = [], []
        offsets: dict[int, int] = {}
        for cache in caches:
            if not cache.keys:
                cache.keys = [[] for _ in range(config.decoder_layers)]
                cache.values = [[] for _ in range(config.decoder_layers)]
            position = len(cache.tokens) + offsets.get(id(cache), 0)
            offsets[id(cache)] = offsets.get(id(cache), 0) + 1
            keys_seen = np.arange(position + 1, dtype=np.int64)
            buckets = t5_relative_buckets(
                keys_seen - position, config.position_buckets, config.max_relative_positions, bidirectional=False
            )
            biases.append(np.ascontiguousarray(table[buckets].T.reshape(heads, 1, position + 1)))
            source_length = len(cache.cross[0][0]) if cache.cross else 0
            zeros.append(np.zeros((heads, 1, source_length), dtype=np.float32))
        h = self._embedding[list(tokens)].copy()
        watcher = recorder if recorder is not None and recorder.wants_attention else None
        for i in range(config.decoder_layers):
            p = f"decoder.layers.{i}."
            if recorder is not None:
                recorder.record("residual", i, h.copy())
            x = rms_norm(h, self._w[p + "attention_norm.weight"], eps).numpy()  # type: ignore[arg-type]
            q = self._linear(x, p + "attention.q").reshape(count, 1, heads, head_dim)
            k_new = self._linear(x, p + "attention.k").reshape(count, heads, head_dim)
            v_new = self._linear(x, p + "attention.v").reshape(count, heads, head_dim)
            mixed = np.empty((count, heads * head_dim), dtype=np.float32)
            for row, cache in enumerate(caches):
                cache.keys[i].append(k_new[row])
                cache.values[i].append(v_new[row])
                k, v = np.stack(cache.keys[i]), np.stack(cache.values[i])
                mixed[row] = biased_attention(q[row], k, v, biases[row], 1.0).numpy().reshape(-1)
                if watcher is not None:
                    watcher.record("attention", i, biased_attention_weights(q[row], k, biases[row], 1.0).numpy())
            out = self._linear(mixed, p + "attention.o")
            h = h + out
            if recorder is not None:
                recorder.record("attention_output", i, out)
                recorder.record("middle", i, h.copy())
            x = rms_norm(h, self._w[p + "cross_norm.weight"], eps).numpy()  # type: ignore[arg-type]
            q = self._linear(x, p + "cross.q").reshape(count, 1, heads, head_dim)
            for row, cache in enumerate(caches):
                k, v = cache.cross[i]
                mixed[row] = biased_attention(q[row], k, v, zeros[row], 1.0).numpy().reshape(-1)
                if watcher is not None:
                    watcher.record("cross_attention", i, biased_attention_weights(q[row], k, zeros[row], 1.0).numpy())
            out = self._linear(mixed, p + "cross.o")
            h = h + out
            if recorder is not None:
                recorder.record("cross_attention_output", i, out)
                recorder.record("cross_middle", i, h.copy())
            x = rms_norm(h, self._w[p + "mlp_norm.weight"], eps).numpy()  # type: ignore[arg-type]
            activation = self._mlp(x, p)
            out = self._linear(activation, p + "mlp.down")
            if recorder is not None:
                recorder.record("mlp_activation", i, activation)
                recorder.record("mlp_output", i, out)
            h = h + out
            if i in self.steering:
                h = h + self.steering[i]
        if recorder is not None:
            recorder.record("residual", config.decoder_layers, h.copy())
        for token, cache in zip(tokens, caches, strict=True):
            cache.tokens.append(int(token))
        return self.logits_from_hidden(self.final_norm(h))

    def _mlp(self, x: np.ndarray, p: str) -> np.ndarray:
        up = self._linear(x, p + "mlp.up")
        if not self.config.gated_mlp:
            return activate(up, self.config.activation)
        return activate(self._linear(x, p + "mlp.gate"), self.config.activation) * up

    def _linear(self, x: np.ndarray, name: str) -> np.ndarray:
        return linear(x, self._w[name + ".weight"]).numpy()  # type: ignore[arg-type]
