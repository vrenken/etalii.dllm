"""Prompt caching: KV caches of earlier requests, reused for the longest shared token prefix of the next one.

A multi-turn chat, a shared system prompt or a tool loop sends a prompt that starts with the tokens of an earlier
one. :class:`PromptCache` keeps the KV caches of recent generations and hands the next generation the one that shares
the longest prefix with its prompt; :meth:`Transformer.forward_cached` then only computes the new tokens. The KV
cache is an optimisation only (its rows are exactly what a recompute gives), so a cache hit produces the same bits
as a cold run (``tests/test_prompt_cache.py``); only the work, and the reported ``cached_tokens``, differ.

Each cache is lent to one generation at a time, so concurrent requests never share one. Choice and eviction are
deterministic (longest prefix, then most recently returned; the least recently returned entry goes first) and
depend only on the order of requests, though neither can affect the output anyway.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Generic, Protocol, TypeVar


class TokenCache(Protocol):
    tokens: list[int]


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


class PromptCache(Generic[C]):
    """Up to ``capacity`` idle KV caches, lent out by :meth:`acquire` and given back by :meth:`release`."""

    def __init__(self, new_cache: Callable[[], C], capacity: int = DEFAULT_PROMPT_CACHE_SIZE) -> None:
        if capacity < 0:
            raise ValueError("the prompt cache capacity must be non-negative")
        self._new_cache = new_cache
        self.capacity = capacity
        self._idle: list[C] = []
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
                length = reusable(cache.tokens, prompt)
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
            self._idle = [c for c in self._idle if c is not cache and shared_prefix(c.tokens, tokens) < len(c.tokens)]
            self._idle.append(cache)
            del self._idle[: max(0, len(self._idle) - self.capacity)]

    def clear(self) -> None:
        with self._lock:
            self._idle.clear()
