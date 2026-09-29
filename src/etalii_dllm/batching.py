"""Continuous batching: concurrent generations share one forward pass per step.

Every generation calls :meth:`Batcher.forward_cached` for its next logits. When several do so at the same time
(concurrent server requests), their calls are gathered and run as one :meth:`Transformer.forward_batch`: the new
tokens of all of them (a prompt being read, a single token being decoded) go through every linear layer as one
stacked matrix, so the weights are read once per step instead of once per request. A request joins at the next step
after it arrives and leaves when it finishes; nobody waits for a whole batch to end.

Which requests share a step depends on timing, but the output cannot: every kernel computes each row on its own in
a fixed order and attention runs per sequence, so each sequence gets exactly the bits of a lone forward pass
(``tests/test_batch_invariance.py``). That is the guarantee mainstream batching servers do not give.

There is no scheduler thread. A caller that finds no batch running becomes the leader: it takes every waiting call
(its own included), runs them, hands out the results and wakes the others; one of the callers that arrived meanwhile
leads the next batch.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from etalii_dllm.numerics import FloatArray


class BatchModel(Protocol):
    def forward_batch(self, sequences: Sequence[Sequence[int]], caches: Sequence[Any]) -> list[FloatArray]: ...


@dataclass(eq=False)
class _Call:
    tokens: list[int]
    cache: Any
    done: bool = False
    logits: FloatArray | None = None
    error: BaseException | None = field(default=None, repr=False)

    def result(self) -> FloatArray:
        if self.error is not None:
            raise self.error
        assert self.logits is not None
        return self.logits


class Batcher:
    """Runs concurrent ``forward_cached`` calls on ``model`` as batches. ``max_batch`` caps how many sequences share
    a step (0: no limit); the rest wait for the next one."""

    def __init__(self, model: BatchModel, max_batch: int = 0) -> None:
        if max_batch < 0:
            raise ValueError("max_batch must be non-negative")
        self.model = model
        self.max_batch = max_batch
        self._waiting: list[_Call] = []
        self._running = False
        self._condition = threading.Condition()
        self.batches = 0
        """Forward passes run so far (for tests and statistics)."""
        self.largest_batch = 0

    def forward_cached(self, tokens: Sequence[int], cache: Any) -> FloatArray:
        """Next-token logits after ``tokens``, reusing and extending ``cache``, exactly as the model's own
        ``forward_cached``. Blocks until the batch holding this call has run."""
        call = _Call(list(tokens), cache)
        with self._condition:
            self._waiting.append(call)
        while True:
            with self._condition:
                while self._running and not call.done:
                    self._condition.wait()
                if call.done:
                    return call.result()
                self._running = True
                batch = self._take()
            try:
                self._run(batch)
            finally:
                with self._condition:
                    self._running = False
                    self._condition.notify_all()

    def _take(self) -> list[_Call]:
        """The oldest waiting calls, at most ``max_batch`` of them; the caller holds the lock."""
        count = len(self._waiting) if self.max_batch == 0 else min(self.max_batch, len(self._waiting))
        batch, self._waiting = self._waiting[:count], self._waiting[count:]
        return batch

    def _run(self, batch: list[_Call]) -> None:
        self.batches += 1
        self.largest_batch = max(self.largest_batch, len(batch))
        try:
            results = self.model.forward_batch([c.tokens for c in batch], [c.cache for c in batch])
        except Exception:
            # One bad call must not fail the others: run each alone, so each gets its own result or error.
            for call in batch:
                _run_alone(call, self.model)
            return
        for call, logits in zip(batch, results, strict=True):
            call.logits = logits
            call.done = True


def _run_alone(call: _Call, model: BatchModel) -> None:
    """Runs one call alone and stores its logits or its error."""
    try:
        call.logits = model.forward_batch([call.tokens], [call.cache])[0]
    except Exception as error:
        call.error = error
    call.done = True
