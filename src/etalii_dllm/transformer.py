"""Llama-style decoder (Llama, SmolLM2, TinyLlama, Qwen2) running on the deterministic kernels.

Every reduction goes through :mod:`etalii_dllm.numerics`, whose kernels give each output element one fixed
accumulation order, independent of how many tokens are processed together. Consequently the KV cache is an
optimisation only: prefilling a prompt at once, feeding it token by token, or recomputing from scratch all give
bit-identical logits (``tests/test_transformer.py``). Residual additions and the SwiGLU product are elementwise
float32 operations, as in the reference implementations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.numerics import FloatArray, attention, linear, rms_norm, rope, rope_inv_freq, silu
from etalii_dllm.tensor import Tensor


class KVCache:
    """Keys and values of the tokens processed so far, per layer, in ``[positions, kv_heads, head_dim]`` buffers
    that grow by doubling. The stored rows are exactly what a full recompute would produce, so reading them back
    cannot change a result."""

    def __init__(self, config: TransformerConfig) -> None:
        self._config = config
        self.tokens: list[int] = []
        self._keys: list[np.ndarray] = []
        self._values: list[np.ndarray] = []

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

    def append(self, layer: int, start: int, keys: Tensor, values: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Stores rows ``start ..`` of ``layer`` and returns all keys and values up to the new end (views)."""
        end = start + keys.shape[0]
        self._reserve(end)
        self._keys[layer][start:end] = keys.numpy()
        self._values[layer][start:end] = values.numpy()
        return self._keys[layer][:end], self._values[layer][:end]

    def truncate(self, length: int) -> None:
        del self.tokens[length:]


class Transformer:
    """A decoder-only transformer with the architecture and tensor names of ``docs/model-format.md``."""

    def __init__(
        self,
        config: TransformerConfig,
        tensors: Mapping[str, npt.ArrayLike],
        *,
        model_id: str = "dllm-transformer",
        weights_fingerprint: str = "",
    ) -> None:
        expected = config.tensor_shapes()
        missing = sorted(set(expected) - set(tensors))
        if missing:
            raise ValueError(f"missing tensors: {missing}")
        self.config = config
        self.weights_fingerprint = weights_fingerprint
        self._id = model_id
        self._w = {name: Tensor(tensors[name]) for name in expected}
        for name, shape in expected.items():
            if self._w[name].shape != shape:
                raise ValueError(f"tensor {name!r} has shape {self._w[name].shape}, expected {shape}")
        self._embedding = self._w["token_embedding.weight"].numpy()
        self._lm_head = self._w["token_embedding.weight" if config.tie_word_embeddings else "lm_head.weight"]
        self._inv_freq = rope_inv_freq(config.head_dim, config.rope_theta, scaling=config.rope_scaling)

    @classmethod
    def from_file(cls, path: str | Path, verify: bool = True) -> Transformer:
        model = ModelFile(path, verify=verify)
        source = model.source.get("repository") or Path(path).stem
        return cls(model.config, model.tensors, model_id=str(source), weights_fingerprint=model.fingerprint)

    @property
    def id(self) -> str:
        return self._id

    @property
    def vocabulary_size(self) -> int:
        return self.config.vocabulary_size

    def new_cache(self) -> KVCache:
        return KVCache(self.config)

    def forward(self, tokens: Sequence[int]) -> FloatArray:
        """Next-token logits after ``tokens``, recomputed from scratch (no shared state, safe to call
        concurrently)."""
        return self.forward_cached(tokens, self.new_cache())

    def forward_cached(self, tokens: Sequence[int], cache: KVCache) -> FloatArray:
        """Next-token logits after ``tokens``, reusing the longest prefix already in ``cache`` and appending the
        rest. Gives the same bits as :meth:`forward`."""
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
        hidden = self._layers(list(tokens[common:]), common, cache)
        cache.tokens.extend(tokens[common:])
        return linear(hidden[-1:], self._lm_head).numpy().reshape(self.vocabulary_size).copy()

    def _layers(self, tokens: list[int], start: int, cache: KVCache) -> Tensor:
        config = self.config
        w = self._w
        count = len(tokens)
        if any(not 0 <= t < config.vocabulary_size for t in tokens):
            raise ValueError("token id out of range")
        positions = np.arange(start, start + count, dtype=np.int64)
        x = np.ascontiguousarray(self._embedding[np.asarray(tokens, dtype=np.int64)])
        for layer in range(config.layers):
            p = f"layers.{layer}."
            h = rms_norm(x, w[p + "attention_norm.weight"], config.rms_norm_eps)
            q = linear(h, w[p + "attention.q.weight"], w.get(p + "attention.q.bias"))
            k = linear(h, w[p + "attention.k.weight"], w.get(p + "attention.k.bias"))
            v = linear(h, w[p + "attention.v.weight"], w.get(p + "attention.v.bias"))
            q = rope(q.reshape(count, config.heads, config.head_dim), positions, self._inv_freq)
            k = rope(k.reshape(count, config.kv_heads, config.head_dim), positions, self._inv_freq)
            keys, values = cache.append(layer, start, k, v.reshape(count, config.kv_heads, config.head_dim))
            a = attention(q, keys, values, causal=True, q_offset=start)
            x = x + linear(a.reshape(count, config.heads * config.head_dim), w[p + "attention.o.weight"]).numpy()
            h = rms_norm(x, w[p + "mlp_norm.weight"], config.rms_norm_eps)
            gate = silu(linear(h, w[p + "mlp.gate.weight"])).numpy()
            up = linear(h, w[p + "mlp.up.weight"]).numpy()
            x = x + linear(gate * up, w[p + "mlp.down.weight"]).numpy()
        return rms_norm(x, w["final_norm.weight"], config.rms_norm_eps)
