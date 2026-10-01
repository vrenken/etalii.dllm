"""Llama-style decoder (Llama, SmolLM2, TinyLlama, Qwen2, Qwen3) running on the deterministic kernels.

Every reduction goes through :mod:`etalii_dllm.numerics`, whose kernels give each output element one fixed
accumulation order, independent of how many tokens are processed together. Consequently the KV cache is an
optimisation only: prefilling a prompt at once, feeding it token by token, or recomputing from scratch all give
bit-identical logits (``tests/test_transformer.py``). Residual additions and the SwiGLU product are elementwise
float32 operations, as in the reference implementations.

With ``device="cuda"`` the weights, activations and KV cache live on the GPU and the same steps run there
(:mod:`etalii_dllm.cuda`); only the embedding rows go up and the logits come back. The GPU kernels produce the CPU
bits, so the logits, and the ``system_fingerprint``, do not depend on the device (``tests/test_cuda.py``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from etalii_dllm import cuda
from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.cuda import CudaTensor
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.numerics import (
    CudaQuantizedWeight,
    CudaWeight,
    FloatArray,
    PackedWeight,
    QuantizedWeight,
    attention,
    attention_weights,
    linear,
    rms_norm,
    rope,
    rope_inv_freq,
    softcap,
    swiglu,
)
from etalii_dllm.tensor import Tensor

_MATRICES = tuple(f".{m}.weight" for m in ("q", "k", "v", "o", "gate", "up", "down"))


Weight = PackedWeight | QuantizedWeight | CudaWeight | CudaQuantizedWeight


class LayerHook:
    """Observes (and may change) the activations of a CPU forward pass, layer by layer; see
    :meth:`Transformer.run_hooked`. Every method gets a fresh float32 array it may keep. ``residual`` may return a
    replacement for the residual stream (same shape and dtype), which the rest of the pass then uses; returning
    ``None`` leaves it unchanged, so a hook that only observes cannot change a single bit of the output."""

    wants_attention = False
    """Whether :meth:`attention` should be called (computing the probabilities costs a second pass over the keys)."""

    def residual(self, layer: int, point: str, x: np.ndarray) -> np.ndarray | None:
        """The residual stream ``[positions, hidden]`` of ``layer`` at ``point``: ``"input"`` (before the layer;
        layer 0 sees the embeddings), ``"middle"`` (after the attention block's residual add) or ``"output"``."""
        return None

    def attention(self, layer: int, probabilities: np.ndarray) -> None:
        """The attention probabilities ``[positions, heads, keys]`` of ``layer``."""

    def attention_output(self, layer: int, out: np.ndarray) -> None:
        """What the attention block adds to the residual stream, ``[positions, hidden]``."""

    def mlp_activation(self, layer: int, activation: np.ndarray) -> None:
        """The MLP's hidden activation ``act(gate) * up``, ``[positions, intermediate]``: the input of the down
        projection (the "keys" of model editing)."""

    def mlp_output(self, layer: int, out: np.ndarray) -> None:
        """What the MLP block adds to the residual stream, ``[positions, hidden]``."""


def _prepare(weight: Tensor, quantize: str | None, device: str) -> Weight:
    if quantize and QuantizedWeight.supports(weight):
        quantized = QuantizedWeight(weight, quantize)
        return CudaQuantizedWeight(quantized) if device == "cuda" else quantized
    return CudaWeight(weight) if device == "cuda" else PackedWeight(weight)


def steered_fingerprint(fingerprint: str, steering: Mapping[int, np.ndarray] | None) -> str:
    """The weights fingerprint of a model run with ``steering`` (layer index -> vector added to that layer's
    output): unchanged without it, else the SHA-256 of the fingerprint and every layer's vector bits, so a steered
    model never shares a ``system_fingerprint`` with the plain one."""
    if not steering:
        return fingerprint
    digest = hashlib.sha256(f"{fingerprint}:steering".encode())
    for layer in sorted(steering):
        digest.update(f":{layer}:".encode())
        digest.update(np.ascontiguousarray(steering[layer], dtype="<f4").tobytes())
    return digest.hexdigest()


def quantized_fingerprint(fingerprint: str, quantize: str | None) -> str:
    """The weights fingerprint of a model run with ``quantize``: unchanged without quantisation, else the SHA-256
    of ``"<fingerprint>:<quantisation>"``, so quantised and float runs never share a ``system_fingerprint``."""
    if not quantize:
        return fingerprint
    return hashlib.sha256(f"{fingerprint}:{quantize}".encode()).hexdigest()


class KVCache:
    """Keys and values of the tokens processed so far, per layer, in ``[positions, kv_heads, head_dim]`` buffers
    that grow by doubling (in GPU memory for ``device="cuda"``). The stored rows are exactly what a full recompute
    would produce, so reading them back cannot change a result."""

    def __init__(self, config: TransformerConfig, device: str = "cpu") -> None:
        self._config = config
        self.device = device
        self.tokens: list[int] = []
        self._keys: list = []
        self._values: list = []

    def __len__(self) -> int:
        return len(self.tokens)

    def _reserve(self, length: int) -> None:
        capacity = self._keys[0].shape[0] if self._keys else 0
        if length <= capacity:
            return
        capacity = max(16, capacity)
        while capacity < length:
            capacity *= 2
        shape = (capacity, self._config.kv_heads, self._config.head_dim)
        if self.device == "cuda":
            for layer in range(self._config.layers):
                keys, values = CudaTensor.empty(shape), CudaTensor.empty(shape)
                if layer < len(self._keys):
                    self._keys[layer][: len(self)].copy_to(keys[: len(self)])
                    self._values[layer][: len(self)].copy_to(values[: len(self)])
                    self._keys[layer], self._values[layer] = keys, values
                else:
                    self._keys.append(keys)
                    self._values.append(values)
            return
        for layer in range(self._config.layers):
            keys = np.zeros(shape, dtype=np.float32)
            values = np.zeros(shape, dtype=np.float32)
            if layer < len(self._keys):
                keys[: len(self)] = self._keys[layer][: len(self)]
                values[: len(self)] = self._values[layer][: len(self)]
                self._keys[layer], self._values[layer] = keys, values
            else:
                self._keys.append(keys)
                self._values.append(values)

    def append(
        self, layer: int, start: int, keys: Tensor | np.ndarray, values: Tensor | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stores rows ``start ..`` of ``layer`` and returns all keys and values up to the new end (views)."""
        end = start + keys.shape[0]
        self._reserve(end)
        if isinstance(keys, CudaTensor):
            keys.copy_to(self._keys[layer][start:end])
            values.copy_to(self._values[layer][start:end])
            return self._keys[layer][:end], self._values[layer][:end]
        self._keys[layer][start:end] = keys.numpy() if isinstance(keys, Tensor) else keys
        self._values[layer][start:end] = values.numpy() if isinstance(values, Tensor) else values
        return self._keys[layer][:end], self._values[layer][:end]

    def truncate(self, length: int) -> None:
        del self.tokens[length:]


def _rope_scaled(name: str, tensor: Tensor, config: TransformerConfig) -> Tensor:
    """``tensor`` with the rows feeding the rotated dimensions of each query or key head multiplied (elementwise,
    float32) by the LongRoPE attention factor; other tensors unchanged."""
    if not name.endswith(("attention.q.weight", "attention.q.bias", "attention.k.weight", "attention.k.bias")):
        return tensor
    values = tensor.numpy()
    heads = values.shape[0] // config.head_dim
    rows = values.reshape(heads, config.head_dim, *values.shape[1:]).copy()
    rows[:, : config.rotary_dimension] *= np.float32(config.rope_attention_factor)
    return Tensor(rows.reshape(values.shape))


class Transformer:
    """A decoder-only transformer with the architecture and tensor names of ``docs/model-format.md``."""

    def __init__(
        self,
        config: TransformerConfig,
        tensors: Mapping[str, npt.ArrayLike],
        *,
        model_id: str = "dllm-transformer",
        weights_fingerprint: str = "",
        quantize: str | None = None,
        device: str = "cpu",
        steering: Mapping[int, npt.ArrayLike] | None = None,
    ) -> None:
        """``steering`` maps 0-based layer indices to vectors ``[hidden]`` added (elementwise, float32) to the
        residual stream after that layer at every position (activation steering); it changes the output and so the
        fingerprint."""
        expected = config.tensor_shapes()
        missing = sorted(set(expected) - set(tensors))
        if missing:
            raise ValueError(f"missing tensors: {missing}")
        self.config = config
        self.quantization = quantize
        self.device = cuda.check_device(device)
        """``"cpu"`` or ``"cuda"``; never changes the output, so it is not part of the fingerprint."""
        self.steering: dict[int, np.ndarray] = {}
        for layer, vector in (steering or {}).items():
            values = np.ascontiguousarray(vector, dtype=np.float32)
            if not 0 <= layer < config.layers or values.shape != (config.hidden_size,):
                raise ValueError(
                    f"steering needs a layer in 0..{config.layers - 1} and a vector of {config.hidden_size}"
                )
            self.steering[int(layer)] = values
        self.weights_fingerprint = quantized_fingerprint(
            steered_fingerprint(weights_fingerprint, self.steering), quantize
        )
        self._id = model_id
        weights = {name: Tensor(tensors[name]) for name in expected}
        for name, shape in expected.items():
            if weights[name].shape != shape:
                raise ValueError(f"tensor {name!r} has shape {weights[name].shape}, expected {shape}")
        self.tensors: Mapping[str, Tensor] = weights
        """The float32 source tensors (usually memory-mapped from the model file)."""
        self._embedding = weights["token_embedding.weight"].numpy()
        if config.residual_multiplier != 1.0:
            # Granite scales the outputs of attention and the MLP before the residual add; scaling the two output
            # projections once (elementwise, float32) is the same map and keeps CPU and GPU on one code path.
            residual = np.float32(config.residual_multiplier)
            scaled = ("attention.o.weight", "mlp.down.weight")
            weights = {
                name: Tensor(tensor.numpy() * residual) if name.endswith(scaled) else tensor
                for name, tensor in weights.items()
            }
        if config.rope_attention_factor != 1.0:
            if config.qk_norm:
                raise ValueError("a LongRoPE attention factor together with QK-norm is not supported")
            # LongRoPE scales the rotated query and key dimensions by its attention factor. Rotation is linear, so
            # scaling the rows of the q/k projections that produce those dimensions is the same map.
            weights = {name: _rope_scaled(name, tensor, config) for name, tensor in weights.items()}
        head = weights["token_embedding.weight" if config.tie_word_embeddings else "lm_head.weight"]
        # Matrices are packed (or quantised, or uploaded to the GPU) once here; norms and biases stay plain (on the
        # GPU they are uploaded too); the embedding table stays on the host, which looks up the rows.
        self._w: dict[str, Tensor | Weight | CudaTensor] = {}
        for name, tensor in weights.items():
            if name.endswith(_MATRICES):
                self._w[name] = _prepare(tensor, quantize, self.device)
            elif self.device == "cuda" and name not in ("token_embedding.weight", "lm_head.weight"):
                self._w[name] = CudaTensor.upload(tensor.numpy())
            else:
                self._w[name] = tensor
        self._lm_head = _prepare(head, quantize, self.device)
        self._inv_freq = rope_inv_freq(
            config.head_dim, config.rope_theta, rotary_dim=config.rotary_dimension, scaling=config.rope_scaling
        )
        self._local_inv_freq = self._inv_freq
        if config.local_rope_theta is not None:
            self._local_inv_freq = rope_inv_freq(
                config.head_dim, config.local_rope_theta, rotary_dim=config.rotary_dimension
            )
        if self.device == "cuda":
            self._gpu_inv_freq = cuda.upload_raw(self._inv_freq)
            self._gpu_local_inv_freq = cuda.upload_raw(self._local_inv_freq)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        verify: bool = True,
        quantize: str | None = None,
        device: str = "cpu",
        steering: Mapping[int, npt.ArrayLike] | None = None,
    ) -> Transformer:
        model = ModelFile(path, verify=verify)
        source = model.source.get("repository") or Path(path).stem
        return cls(
            model.config,
            model.tensors,
            model_id=str(source),
            weights_fingerprint=model.fingerprint,
            quantize=quantize,
            device=device,
            steering=steering,
        )

    @property
    def id(self) -> str:
        return self._id

    @property
    def vocabulary_size(self) -> int:
        return self.config.vocabulary_size

    def new_cache(self) -> KVCache:
        return KVCache(self.config, self.device)

    def forward(self, tokens: Sequence[int]) -> FloatArray:
        """Next-token logits after ``tokens``, recomputed from scratch (no shared state, safe to call
        concurrently)."""
        return self.forward_cached(tokens, self.new_cache())

    def hidden_states(self, tokens: Sequence[int]) -> FloatArray:
        """The final-norm hidden states ``[positions, hidden]`` (what the LM head sees), for embeddings."""
        if not tokens:
            raise ValueError("the decoder needs at least one token of context")
        segments = [(list(tokens), 0, self.new_cache())]
        if self.device == "cuda":
            return self._layers_gpu(segments).numpy()
        return self._layers(segments).numpy().copy()

    def forward_cached(self, tokens: Sequence[int], cache: KVCache) -> FloatArray:
        """Next-token logits after ``tokens``, reusing the longest prefix already in ``cache`` and appending the
        rest. Gives the same bits as :meth:`forward`."""
        return self.forward_batch([tokens], [cache])[0]

    def forward_batch(
        self, sequences: Sequence[Sequence[int]], caches: Sequence[KVCache] | None = None
    ) -> list[FloatArray]:
        """Next-token logits for several independent sequences in one pass: their new tokens go through every
        linear layer as one stacked batch, attention runs per sequence against its own cache. Every kernel computes
        each row on its own in a fixed order, so each sequence gets exactly the bits of a lone :meth:`forward_cached`
        (``tests/test_batch_invariance.py``). ``caches`` (one per sequence, distinct) default to fresh ones."""
        if caches is None:
            caches = [self.new_cache() for _ in sequences]
        if len(caches) != len(sequences):
            raise ValueError("forward_batch needs one cache per sequence")
        if len({id(cache) for cache in caches}) != len(caches):
            raise ValueError("forward_batch needs a distinct cache per sequence")
        segments = []
        for tokens, cache in zip(sequences, caches, strict=True):
            if not tokens:
                raise ValueError("the decoder needs at least one token of context")
            common = 0
            for cached, token in zip(cache.tokens, tokens, strict=False):
                if cached != token:
                    break
                common += 1
            if common == len(tokens):  # nothing new: recompute the last position so there is a hidden state
                common -= 1
            cache.truncate(common)
            segments.append((list(tokens[common:]), common, cache))
        ends = np.cumsum([len(tokens) for tokens, _, _ in segments]) - 1
        if self.device == "cuda":
            states = self._layers_gpu(segments)
            last = CudaTensor.empty((len(ends), self.config.hidden_size))
            for i, end in enumerate(ends):
                states[end : end + 1].copy_to(last[i : i + 1])
            logits = cuda.linear(last, self._lm_head).numpy()  # type: ignore[arg-type]
            logits = self._finish_logits(logits)
        else:
            hidden = self._layers(segments).numpy()
            logits = self.logits_from_hidden(hidden[ends])
        for tokens, _, cache in segments:
            cache.tokens.extend(tokens)
        return [row.copy() for row in logits]

    def logits_from_hidden(self, hidden: npt.ArrayLike) -> FloatArray:
        """Logits ``[rows, vocabulary]`` of final-norm hidden states ``[rows, hidden]``: the LM head, then the
        logits scaling and soft-cap, exactly as :meth:`forward` applies them (CPU)."""
        values = np.ascontiguousarray(hidden, dtype=np.float32)
        head = self._lm_head
        if self.device == "cuda":
            head = self._host_lm_head()
        return self._finish_logits(linear(values, head).numpy())

    def final_norm(self, x: npt.ArrayLike) -> FloatArray:
        """The final RMSNorm of residual stream rows ``[rows, hidden]`` (what :meth:`hidden_states` returns for the
        last layer's output)."""
        weight = self.tensors["final_norm.weight"]
        normed = rms_norm(x, weight, self.config.rms_norm_eps, add_unit_offset=self.config.norm_unit_offset)
        return normed.numpy().copy()

    def run_hooked(self, tokens: Sequence[int], hook: LayerHook) -> FloatArray:
        """Runs ``tokens`` from scratch on the CPU with ``hook`` watching every layer and returns the final-norm
        hidden states ``[positions, hidden]``. With a hook that changes nothing these are the bits of
        :meth:`hidden_states`, and ``logits_from_hidden`` of the last row is the bits of :meth:`forward`."""
        if not tokens:
            raise ValueError("the decoder needs at least one token of context")
        if self.device != "cpu":
            raise ValueError("hooked forward passes run on the CPU; load the model with device='cpu'")
        return self._layers([(list(tokens), 0, self.new_cache())], hook).numpy().copy()

    def _finish_logits(self, logits: np.ndarray) -> FloatArray:
        if self.config.logits_scaling != 1.0:
            logits = logits / np.float32(self.config.logits_scaling)
        if self.config.logits_softcap is not None:
            logits = softcap(logits, self.config.logits_softcap).numpy()
        return logits

    def _host_lm_head(self) -> Weight:
        if not hasattr(self, "_cpu_lm_head"):
            name = "token_embedding.weight" if self.config.tie_word_embeddings else "lm_head.weight"
            self._cpu_lm_head = _prepare(self.tensors[name], self.quantization, "cpu")
        return self._cpu_lm_head

    def _layers(self, segments: list[tuple[list[int], int, KVCache]], hook: LayerHook | None = None) -> Tensor:
        """Runs the new ``tokens`` of each ``(tokens, start, cache)`` segment, stacked, through the decoder."""
        config = self.config
        w = self._w
        tokens, positions, bounds = self._embed(segments)
        count = len(tokens)
        x = self._embeddings(tokens)
        unit = config.norm_unit_offset

        def norm(values: npt.ArrayLike | Tensor, name: str) -> Tensor:
            return rms_norm(values, w[name], config.rms_norm_eps, add_unit_offset=unit)

        def observe(layer: int, point: str, values: np.ndarray) -> np.ndarray:
            if hook is None:
                return values
            replaced = hook.residual(layer, point, values.copy())
            return values if replaced is None else np.ascontiguousarray(replaced, dtype=np.float32)

        for layer in range(config.layers):
            p = f"layers.{layer}."
            x = observe(layer, "input", x)
            h = norm(x, p + "attention_norm.weight") if config.has_pre_norms else x
            q = linear(h, w[p + "attention.q.weight"], w.get(p + "attention.q.bias"))
            k = linear(h, w[p + "attention.k.weight"], w.get(p + "attention.k.bias"))
            v = linear(h, w[p + "attention.v.weight"], w.get(p + "attention.v.bias"))
            if config.qk_norm and config.qk_norm_scope == "all":  # OLMo 2: over the whole projection
                q, k = norm(q, p + "attention.q_norm.weight"), norm(k, p + "attention.k_norm.weight")
            q, k = q.reshape(count, config.heads, config.head_dim), k.reshape(count, config.kv_heads, config.head_dim)
            if config.qk_norm and config.qk_norm_scope == "head":  # Qwen3: over each head, before the rotation
                q, k = norm(q, p + "attention.q_norm.weight"), norm(k, p + "attention.k_norm.weight")
            inv_freq = self._local_inv_freq if config.uses_local_rope(layer) else self._inv_freq
            q = rope(q, positions, inv_freq).numpy()
            k = rope(k, positions, inv_freq).numpy()
            v = v.numpy().reshape(count, config.kv_heads, config.head_dim)
            attended = []
            for (_, start, cache), lo, hi in zip(segments, bounds[:-1], bounds[1:], strict=True):
                keys, values = cache.append(layer, start, k[lo:hi], v[lo:hi])
                attended.append(
                    attention(
                        q[lo:hi],
                        keys,
                        values,
                        scale=config.attention_scale,
                        causal=True,
                        q_offset=start,
                        window=config.window(layer),
                        softcap=config.attention_softcap,
                    ).numpy()
                )
                if hook is not None and hook.wants_attention:
                    hook.attention(
                        layer,
                        attention_weights(
                            q[lo:hi],
                            keys,
                            scale=config.attention_scale,
                            causal=True,
                            q_offset=start,
                            window=config.window(layer),
                            softcap=config.attention_softcap,
                        ).numpy(),
                    )
            a = attended[0] if len(attended) == 1 else np.concatenate(attended)
            out = linear(a.reshape(count, config.heads * config.head_dim), w[p + "attention.o.weight"])
            if config.has_post_norms:
                out = norm(out, p + "attention_post_norm.weight")
            if hook is not None:
                hook.attention_output(layer, out.numpy().copy())
            x = observe(layer, "middle", x + out.numpy())
            h = norm(x, p + "mlp_norm.weight") if config.has_pre_norms else x
            gate = linear(h, w[p + "mlp.gate.weight"])
            up = linear(h, w[p + "mlp.up.weight"])
            activation = swiglu(gate, up, config.activation).numpy()
            if hook is not None:
                hook.mlp_activation(layer, activation.copy())
            out = linear(activation, w[p + "mlp.down.weight"])
            if config.has_post_norms:
                out = norm(out, p + "mlp_post_norm.weight")
            if hook is not None:
                hook.mlp_output(layer, out.numpy().copy())
            x = x + out.numpy()
            if layer in self.steering:
                x = x + self.steering[layer]
            x = observe(layer, "output", x)
        return norm(x, "final_norm.weight")

    def _embeddings(self, tokens: list[int]) -> np.ndarray:
        rows = self._embedding[np.asarray(tokens, dtype=np.int64)]
        if self.config.embedding_multiplier != 1.0:
            rows = rows * np.float32(self.config.embedding_multiplier)
        return np.ascontiguousarray(rows, dtype=np.float32)

    def _embed(self, segments: list[tuple[list[int], int, KVCache]]) -> tuple[list[int], np.ndarray, np.ndarray]:
        """(token ids, absolute positions, segment bounds) of the stacked new tokens."""
        tokens = [t for segment, _, _ in segments for t in segment]
        if any(not 0 <= t < self.config.vocabulary_size for t in tokens):
            raise ValueError("token id out of range")
        positions = np.concatenate(
            [np.arange(start, start + len(segment), dtype=np.int64) for segment, start, _ in segments]
        )
        bounds = np.concatenate([[0], np.cumsum([len(segment) for segment, _, _ in segments])])
        return tokens, positions, bounds

    def _layers_gpu(self, segments: list[tuple[list[int], int, KVCache]]) -> CudaTensor:
        """:meth:`_layers` on the GPU: the same steps in the same order, on device tensors."""
        config = self.config
        w: dict = self._w
        tokens, positions, bounds = self._embed(segments)
        count = len(tokens)
        scale = config.attention_scale
        x = CudaTensor.upload(self._embeddings(tokens))
        on_gpu_positions = cuda.upload_raw(positions)
        unit = config.norm_unit_offset

        def norm(values: CudaTensor, name: str) -> CudaTensor:
            return cuda.rms_norm(values, w[name], config.rms_norm_eps, add_unit_offset=unit)

        for layer in range(config.layers):
            p = f"layers.{layer}."
            h = norm(x, p + "attention_norm.weight") if config.has_pre_norms else x
            q = cuda.linear(h, w[p + "attention.q.weight"], w.get(p + "attention.q.bias"))
            k = cuda.linear(h, w[p + "attention.k.weight"], w.get(p + "attention.k.bias"))
            v = cuda.linear(h, w[p + "attention.v.weight"], w.get(p + "attention.v.bias"))
            if config.qk_norm and config.qk_norm_scope == "all":
                q, k = norm(q, p + "attention.q_norm.weight"), norm(k, p + "attention.k_norm.weight")
            q, k = q.reshape(count, config.heads, config.head_dim), k.reshape(count, config.kv_heads, config.head_dim)
            if config.qk_norm and config.qk_norm_scope == "head":
                q, k = norm(q, p + "attention.q_norm.weight"), norm(k, p + "attention.k_norm.weight")
            inv_freq = self._gpu_local_inv_freq if config.uses_local_rope(layer) else self._gpu_inv_freq
            q = cuda.rope(q, on_gpu_positions, inv_freq)
            k = cuda.rope(k, on_gpu_positions, inv_freq)
            v = v.reshape(count, config.kv_heads, config.head_dim)
            parts = []
            for (_, start, cache), lo, hi in zip(segments, bounds[:-1], bounds[1:], strict=True):
                keys, values = cache.append(layer, start, k[lo:hi], v[lo:hi])
                end = start + int(hi - lo)
                parts.append(
                    cuda.attention(
                        q[lo:hi],
                        keys,
                        values,
                        kv_len=end,
                        scale=scale,
                        causal=True,
                        q_offset=start,
                        window=config.window(layer),
                        softcap=config.attention_softcap,
                    )
                )
            if len(parts) == 1:
                attended = parts[0]
            else:
                attended = CudaTensor.empty((count, config.heads, config.head_dim))
                for part, lo, hi in zip(parts, bounds[:-1], bounds[1:], strict=True):
                    part.copy_to(attended[lo:hi])
            flat = attended.reshape(count, config.heads * config.head_dim)
            out = cuda.linear(flat, w[p + "attention.o.weight"])
            if config.has_post_norms:
                out = norm(out, p + "attention_post_norm.weight")
            x = cuda.add(x, out)
            h = norm(x, p + "mlp_norm.weight") if config.has_pre_norms else x
            gate = cuda.linear(h, w[p + "mlp.gate.weight"])
            up = cuda.linear(h, w[p + "mlp.up.weight"])
            out = cuda.linear(cuda.swiglu(gate, up, config.activation), w[p + "mlp.down.weight"])
            if config.has_post_norms:
                out = norm(out, p + "mlp_post_norm.weight")
            x = cuda.add(x, out)
            if layer in self.steering:
                rows = np.ascontiguousarray(np.broadcast_to(self.steering[layer], (count, config.hidden_size)))
                x = cuda.add(x, CudaTensor.upload(rows))
        return norm(x, "final_norm.weight")
