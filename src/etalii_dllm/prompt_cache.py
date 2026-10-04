"""Prompt caching: KV caches of earlier requests, reused for the longest shared token prefix of the next one.

A multi-turn chat, a shared system prompt or a tool loop sends a prompt that starts with the tokens of an earlier
one. :class:`PromptCache` keeps the KV caches of recent generations and hands the next generation the one that shares
the longest prefix with its prompt; :meth:`Transformer.forward_cached` then only computes the new tokens. The KV
cache is an optimisation only (its rows are exactly what a recompute gives), so a cache hit produces the same bits
as a cold run (``tests/test_prompt_cache.py``); only the work, and the reported ``cached_tokens``, differ.

Each cache is lent to one generation at a time, so concurrent requests never share one. Choice and eviction are
deterministic (longest prefix, then most recently returned; the least recently returned entry goes first) and
depend only on the order of requests, though neither can affect the output anyway.

With a :class:`CacheStore` (``--persistent-cache``) the idle caches are also kept on disk and loaded again when the
engine starts, so a restarted server still skips the shared prefixes it has seen. A stored cache is only used by the
same weights and engine version, and only after its checksum matches; it holds the same bits a recompute gives, so
it cannot change the output either.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Generic, Protocol, TypeVar

import numpy as np


class TokenCache(Protocol):
    tokens: list[int]
    """The tokens the cache holds. A cache can define ``reusable(prompt)`` and ``covers(other)`` instead of the
    shared-prefix rules (a T5 cache, whose encoder states only serve the very same source)."""


C = TypeVar("C", bound=TokenCache)

DEFAULT_PROMPT_CACHE_SIZE = 4
"""How many KV caches the served engine keeps by default (``--prompt-cache``, ``DLLM_PROMPT_CACHE``)."""


def shared_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the common prefix of two token sequences."""
    length = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        length += 1
    return length


def reusable(cache_tokens: Sequence[int], prompt: Sequence[int]) -> int:
    """How many prompt tokens a cache holding ``cache_tokens`` saves: the shared prefix, less one when it covers the
    whole prompt (the last position is recomputed to get its hidden state)."""
    return min(shared_prefix(cache_tokens, prompt), max(len(prompt) - 1, 0))


def _reusable(cache: TokenCache, prompt: Sequence[int]) -> int:
    own = getattr(cache, "reusable", None)
    return int(own(prompt)) if own is not None else reusable(cache.tokens, prompt)


def _covers(cache: TokenCache, other: TokenCache) -> bool:
    """Whether ``cache`` serves every prompt ``other`` would (``other`` holds a prefix of it)."""
    own = getattr(cache, "covers", None)
    if own is not None:
        return bool(own(other))
    return shared_prefix(other.tokens, cache.tokens) >= len(other.tokens)


_MAGIC = b"DLLMKV1\n"
_SUFFIX = ".kv"


class CacheStore:
    """KV caches in a directory, one file each, for one model (``key``: the weights and the engine version).

    A file holds a JSON header (the key, the tokens, the shape and a SHA-256 of the tokens and the data) and the keys
    and values as little-endian float32. Files are written to a temporary name and renamed, so a crash never leaves
    a half-written cache under a real name; a file that does not match its checksum or the model is ignored. Files of
    other models in the same directory are left alone."""

    def __init__(self, directory: str | Path, key: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.key = key
        self._prefix = hashlib.sha256(key.encode()).hexdigest()[:16]
        self._lock = threading.Lock()

    def path(self, tokens: Sequence[int]) -> Path:
        digest = hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()[:32]
        return self.directory / f"{self._prefix}-{digest}{_SUFFIX}"

    def files(self) -> list[Path]:
        """This model's cache files, in name order."""
        return sorted(self.directory.glob(f"{self._prefix}-*{_SUFFIX}"))

    @staticmethod
    def _checksum(tokens: Sequence[int], keys: bytes, values: bytes) -> str:
        digest = hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode())
        digest.update(keys)
        digest.update(values)
        return digest.hexdigest()

    def save(self, tokens: Sequence[int], keys: np.ndarray, values: np.ndarray) -> None:
        """Writes a cache (what :meth:`etalii_dllm.transformer.KVCache.export` returns) unless its file exists."""
        path = self.path(tokens)
        if path.exists() or not tokens:
            return
        key_bytes, value_bytes = keys.astype("<f4").tobytes(), values.astype("<f4").tobytes()
        header = json.dumps(
            {
                "key": self.key,
                "tokens": list(tokens),
                "shape": list(keys.shape),
                "sha256": self._checksum(tokens, key_bytes, value_bytes),
            },
            separators=(",", ":"),
        ).encode()
        with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as file:
            file.write(_MAGIC + struct.pack("<Q", len(header)) + header)
            file.write(key_bytes)
            file.write(value_bytes)
        os.replace(file.name, path)

    def read(self, path: Path) -> tuple[list[int], np.ndarray, np.ndarray] | None:
        """The tokens, keys and values of a cache file, or ``None`` when it is damaged or another model's."""
        try:
            data = path.read_bytes()
            if not data.startswith(_MAGIC):
                return None
            (length,) = struct.unpack_from("<Q", data, len(_MAGIC))
            start = len(_MAGIC) + 8
            header = json.loads(data[start : start + length])
            shape = tuple(int(n) for n in header["shape"])
            tokens = [int(t) for t in header["tokens"]]
            size = 4 * int(np.prod(shape))
            body = data[start + length :]
            key_bytes, value_bytes = body[:size], body[size:]
            if header["key"] != self.key or len(body) != 2 * size or len(tokens) != shape[1]:
                return None
            if header["sha256"] != self._checksum(tokens, key_bytes, value_bytes):
                return None
        except (OSError, ValueError, KeyError, TypeError, IndexError, struct.error):
            return None
        keys = np.frombuffer(key_bytes, dtype="<f4").reshape(shape).astype(np.float32)
        values = np.frombuffer(value_bytes, dtype="<f4").reshape(shape).astype(np.float32)
        return tokens, keys, values

    def load(self, new_cache: Callable[[], C], capacity: int) -> list[C]:
        """Up to ``capacity`` stored caches (the last ones in name order), restored into new caches."""
        caches: list[C] = []
        for path in reversed(self.files()):
            if len(caches) >= capacity:
                break
            stored = self.read(path)
            if stored is None:
                continue
            cache = new_cache()
            try:
                cache.restore(*stored)  # type: ignore[attr-defined]
            except ValueError:
                continue
            caches.append(cache)
        return caches[::-1]

    def sync(self, exported: tuple[list[int], np.ndarray, np.ndarray] | None, keep: Sequence[Sequence[int]]) -> None:
        """Writes ``exported`` (when given) and deletes this model's files other than those of the token lists in
        ``keep`` (the idle caches)."""
        with self._lock:
            if exported is not None:
                self.save(*exported)
            kept = {self.path(tokens) for tokens in keep}
            for path in self.files():
                if path not in kept:
                    path.unlink(missing_ok=True)


class PromptCache(Generic[C]):
    """Up to ``capacity`` idle KV caches, lent out by :meth:`acquire` and given back by :meth:`release`; with a
    ``store`` they are also kept on disk across restarts."""

    def __init__(
        self, new_cache: Callable[[], C], capacity: int = DEFAULT_PROMPT_CACHE_SIZE, store: CacheStore | None = None
    ) -> None:
        if capacity < 0:
            raise ValueError("the prompt cache capacity must be non-negative")
        self._new_cache = new_cache
        self.capacity = capacity
        self.store = store
        self._idle: list[C] = store.load(new_cache, capacity) if store is not None else []
        """Oldest returned first."""
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._idle)

    def acquire(self, prompt: Sequence[int]) -> tuple[C, int]:
        """A cache for a generation starting with ``prompt`` and how many of its tokens it already holds: the idle
        cache sharing the longest prefix (the most recently returned on a tie), else a new one. The cache is taken
        out of the pool until :meth:`release`."""
        with self._lock:
            best, saved = -1, 0
            for index, cache in enumerate(self._idle):
                length = _reusable(cache, prompt)
                if length > 0 and length >= saved:
                    best, saved = index, length
            if best >= 0:
                return self._idle.pop(best), saved
        return self._new_cache(), 0

    def release(self, cache: C) -> None:
        """Returns a cache to the pool. Idle caches whose tokens are a prefix of it are dropped (it serves every
        prompt they would), then the oldest ones beyond ``capacity``."""
        if self.capacity == 0:
            return
        with self._lock:
            tokens = cache.tokens
            self._idle = [c for c in self._idle if c is not cache and not _covers(cache, c)]
            self._idle.append(cache)
            del self._idle[: max(0, len(self._idle) - self.capacity)]
            if self.store is None:
                return
            # Copied while no generation can hold the cache; writing happens outside the lock.
            kept = [list(c.tokens) for c in self._idle]
            stored = cache in self._idle and not self.store.path(tokens).exists()
            exported = cache.export() if stored else None  # type: ignore[attr-defined]
        self.store.sync(exported, kept)

    def clear(self) -> None:
        with self._lock:
            self._idle.clear()
