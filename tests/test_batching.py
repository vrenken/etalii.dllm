"""Continuous batching: concurrent generations share forward passes and still get their solo bits."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from test_batch_invariance import chat_model_path  # noqa: F401 - fixture

from etalii_dllm.batching import Batcher
from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import ChatRequest, DllmEngine
from etalii_dllm.sampling import SamplingOptions

pytest.importorskip("tokenizers")


class FakeCache:
    def __init__(self) -> None:
        self.tokens: list[int] = []


class SlowModel:
    """Logits are the sum of the tokens; the first pass waits so that the other callers queue up behind it."""

    def __init__(self) -> None:
        self.batches: list[list[list[int]]] = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def forward_batch(self, sequences, caches):
        if not self.batches:
            self.entered.set()
            self.release.wait(5)
        self.batches.append([list(s) for s in sequences])
        if any(-1 in s for s in sequences):
            raise ValueError("token id out of range")
        for tokens, cache in zip(sequences, caches, strict=True):
            cache.tokens = list(tokens)
        return [np.array([float(sum(s))], dtype=np.float32) for s in sequences]


def _submit(batcher: Batcher, model: SlowModel, sequences: list[list[int]]):
    """Starts the first call, waits until it is inside the model, then starts the rest and waits until they queue."""
    pool = ThreadPoolExecutor(len(sequences))
    futures = [pool.submit(batcher.forward_cached, sequences[0], FakeCache())]
    assert model.entered.wait(5)
    futures += [pool.submit(batcher.forward_cached, s, FakeCache()) for s in sequences[1:]]
    while len(batcher._waiting) < len(sequences) - 1:
        time.sleep(0.001)
    model.release.set()
    return pool, futures


def test_waiting_calls_run_as_one_batch():
    model = SlowModel()
    batcher = Batcher(model)
    pool, futures = _submit(batcher, model, [[1], [2, 3], [4, 5, 6], [7]])  # one leads, three wait
    assert [f.result()[0] for f in futures] == [1.0, 5.0, 15.0, 7.0]
    pool.shutdown()
    assert len(model.batches) == 2 and len(model.batches[1]) == 3
    assert batcher.largest_batch == 3


def test_max_batch_caps_a_step():
    model = SlowModel()
    batcher = Batcher(model, max_batch=2)
    pool, futures = _submit(batcher, model, [[1], [2], [3], [4], [5]])
    assert [f.result()[0] for f in futures] == [1.0, 2.0, 3.0, 4.0, 5.0]
    pool.shutdown()
    assert [len(b) for b in model.batches] == [1, 2, 2]


def test_a_failing_call_does_not_fail_the_others():
    model = SlowModel()
    batcher = Batcher(model)
    pool, futures = _submit(batcher, model, [[1], [2], [-1], [3]])
    assert futures[0].result()[0] == 1.0
    assert futures[1].result()[0] == 2.0 and futures[3].result()[0] == 3.0
    with pytest.raises(ValueError, match="out of range"):
        futures[2].result()
    pool.shutdown()


def test_concurrent_chats_are_batched_and_keep_their_bits(chat_model_path):  # noqa: F811
    requests = [
        ChatRequest(
            [ChatMessage("user", f"Request {i}: count to {i % 4 + 2}.")],
            max_tokens=6 + i % 5,
            options=SamplingOptions(temperature=0.0 if i % 3 == 0 else 0.9, seed=i),
            top_logprobs=2 if i % 2 else None,
        )
        for i in range(10)
    ]
    alone = DllmEngine.from_model_file(chat_model_path, prompt_cache=0)
    expected = [alone.chat_completion(r) for r in requests]
    engine = DllmEngine.from_model_file(chat_model_path, prompt_cache=0)
    batcher = engine._generator.batcher
    assert batcher is not None
    # Hold the first forward pass until every other request is waiting, so the rest must share steps.
    forward_batch = engine.model.forward_batch
    gate = threading.Event()

    def gated(sequences, caches):
        gate.wait(10)
        return forward_batch(sequences, caches)

    batcher.model = type("Gated", (), {"forward_batch": staticmethod(gated)})()
    with ThreadPoolExecutor(len(requests)) as pool:
        futures = [pool.submit(engine.chat_completion, r) for r in requests]
        while len(batcher._waiting) < len(requests) - 1:
            time.sleep(0.001)
        gate.set()
        results = [f.result() for f in futures]
    assert results == expected
    assert batcher.largest_batch >= len(requests) - 1
    assert batcher.batches < sum(r.completion_tokens + 1 for r in expected)
    cached = DllmEngine.from_model_file(chat_model_path, prompt_cache=4)
    with ThreadPoolExecutor(4) as pool:
        for _ in range(2):
            assert [replace(r, cached_tokens=0) for r in pool.map(cached.chat_completion, requests)] == expected
