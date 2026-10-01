"""Prompt caching reuses KV caches across requests and never changes a bit of the output."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_engine_import import model_path, served  # noqa: F401 - fixtures

from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import ChatRequest, DllmEngine, configured_prompt_cache
from etalii_dllm.prompt_cache import DEFAULT_PROMPT_CACHE_SIZE, PromptCache, reusable
from etalii_dllm.sampling import SamplingOptions

pytest.importorskip("tokenizers")


@dataclass
class FakeCache:
    tokens: list[int] = field(default_factory=list)


def test_reusable_keeps_the_last_prompt_position_to_recompute():
    assert reusable([1, 2, 3], [1, 2, 4, 5]) == 2
    assert reusable([1, 2, 3], [1, 2, 3]) == 2
    assert reusable([1, 2, 3, 4], [1, 2]) == 1
    assert reusable([9], [1, 2]) == 0


def test_acquire_takes_the_longest_prefix_and_lends_it_out():
    pool: PromptCache[FakeCache] = PromptCache(FakeCache, capacity=3)
    short, long = FakeCache([1, 2]), FakeCache([1, 2, 3, 4])
    other = FakeCache([7, 8, 9])
    for cache in (long, short, other):
        pool.release(cache)
    assert len(pool) == 3
    cache, saved = pool.acquire([1, 2, 3, 4, 5])
    assert cache is long and saved == 4
    assert len(pool) == 2  # lent out until released
    fresh, saved = pool.acquire([5, 6])
    assert fresh not in (short, other) and saved == 0


def test_release_drops_prefixes_and_evicts_the_oldest():
    pool: PromptCache[FakeCache] = PromptCache(FakeCache, capacity=2)
    first, second = FakeCache([1]), FakeCache([2])
    pool.release(first)
    pool.release(second)
    pool.release(FakeCache([1, 5]))  # supersedes [1]
    assert [c.tokens for c in pool._idle] == [[2], [1, 5]]
    pool.release(FakeCache([3]))
    assert [c.tokens for c in pool._idle] == [[1, 5], [3]]
    disabled: PromptCache[FakeCache] = PromptCache(FakeCache, capacity=0)
    disabled.release(FakeCache([1]))
    assert len(disabled) == 0


def _conversation(engine: DllmEngine, turns: int, options: SamplingOptions) -> list:
    messages = [ChatMessage("system", "You are terse."), ChatMessage("user", "Hello there")]
    results = []
    for turn in range(turns):
        request = ChatRequest(messages, max_tokens=10, options=options, top_logprobs=2)
        results.append(engine.chat_completion(request))
        messages = [*messages, ChatMessage("assistant", results[-1].content), ChatMessage("user", f"And {turn}?")]
    return results


@pytest.mark.parametrize("options", [SamplingOptions(), SamplingOptions(temperature=0.9, seed=5)])
def test_cache_hits_give_the_bits_of_a_cold_run(model_path, options):  # noqa: F811
    cached = DllmEngine.from_model_file(model_path, prompt_cache=2)
    cold = DllmEngine.from_model_file(model_path, prompt_cache=0)
    warm_results = _conversation(cached, 4, options)
    cold_results = _conversation(cold, 4, options)
    assert [r.cached_tokens for r in cold_results] == [0, 0, 0, 0]
    assert warm_results[0].cached_tokens == 0
    assert all(r.cached_tokens > 0 for r in warm_results[1:])
    assert [replace(r, cached_tokens=0) for r in warm_results] == cold_results
    # The same request again reuses all but the last prompt position.
    again = _conversation(cached, 1, options)[0]
    assert again.cached_tokens == again.prompt_tokens - 1
    assert replace(again, cached_tokens=0) == cold_results[0]


def test_concurrent_requests_with_a_shared_prefix(model_path):  # noqa: F811
    cold = DllmEngine.from_model_file(model_path, prompt_cache=0)
    cached = DllmEngine.from_model_file(model_path, prompt_cache=2)
    requests = [
        ChatRequest([ChatMessage("system", "Answer briefly."), ChatMessage("user", f"Question {i}")], max_tokens=8)
        for i in range(6)
    ]
    expected = [cold.chat_completion(r) for r in requests]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in range(2):
            results = list(pool.map(cached.chat_completion, requests))
            assert [replace(r, cached_tokens=0) for r in results] == expected
    assert len(cached._generator.prompt_cache) <= 2


def test_prompt_cache_setting(monkeypatch):
    monkeypatch.delenv("DLLM_PROMPT_CACHE", raising=False)
    assert configured_prompt_cache() == DEFAULT_PROMPT_CACHE_SIZE
    monkeypatch.setenv("DLLM_PROMPT_CACHE", "0")
    assert configured_prompt_cache() == 0
    monkeypatch.setenv("DLLM_PROMPT_CACHE", "-1")
    with pytest.raises(ValueError):
        configured_prompt_cache()


def test_the_apis_report_cached_tokens(served):  # noqa: F811
    from etalii_dllm.server.app import app

    client = TestClient(app)
    body = {"model": "m", "messages": [{"role": "user", "content": "Cache me"}], "max_tokens": 6}
    first = client.post("/v1/chat/completions", json=body).json()
    second = client.post("/v1/chat/completions", json=body).json()
    assert first["choices"] == second["choices"] and first["id"] == second["id"]
    prompt = second["usage"]["prompt_tokens"]
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] == prompt - 1

    anthropic = {"model": "m", "messages": [{"role": "user", "content": "Cache me"}], "max_tokens": 6}
    reply = client.post("/v1/messages", json=anthropic).json()
    counted = client.post("/v1/messages/count_tokens", json=anthropic).json()["input_tokens"]
    usage = reply["usage"]
    assert usage["input_tokens"] + usage["cache_read_input_tokens"] == counted
    assert usage["cache_read_input_tokens"] == counted - 1


# -- Phase 15: a prompt cache that survives restarts ----------------------------------------------------------------


@pytest.mark.parametrize("quantize", [None, "q8_0"])
def test_stored_caches_survive_a_restart_with_the_same_bits(model_path, tmp_path, quantize):  # noqa: F811
    options = SamplingOptions(temperature=0.9, seed=5)
    cold = _conversation(DllmEngine.from_model_file(model_path, quantize=quantize, prompt_cache=0), 3, options)
    first = DllmEngine.from_model_file(model_path, quantize=quantize, prompt_cache=2, prompt_cache_dir=tmp_path)
    _conversation(first, 2, options)
    files = sorted(tmp_path.glob("*.kv"))
    assert 1 <= len(files) <= 2 and not list(tmp_path.glob("*.tmp"))

    # A new engine (a restarted server) starts with the stored caches and gives the cold bits.
    restarted = DllmEngine.from_model_file(model_path, quantize=quantize, prompt_cache=2, prompt_cache_dir=tmp_path)
    assert len(restarted._generator.prompt_cache) == len(files)
    results = _conversation(restarted, 3, options)
    assert results[0].cached_tokens == results[0].prompt_tokens - 1
    assert all(r.cached_tokens > 0 for r in results)
    assert [replace(r, cached_tokens=0) for r in results] == cold
    assert len(list(tmp_path.glob("*.kv"))) <= 2


def test_stores_of_other_weights_and_damaged_files_are_ignored(model_path, tmp_path):  # noqa: F811
    request = ChatRequest([ChatMessage("user", "Hello there")], max_tokens=6)
    float_engine = DllmEngine.from_model_file(model_path, prompt_cache=2, prompt_cache_dir=tmp_path)
    expected = float_engine.chat_completion(request)
    (stored,) = tmp_path.glob("*.kv")
    other = tmp_path / "unrelated.txt"
    other.write_text("not a cache", encoding="utf-8")

    # Quantised weights have other KV rows: they neither load nor delete the float32 file.
    quantized = DllmEngine.from_model_file(model_path, quantize="q8_0", prompt_cache=2, prompt_cache_dir=tmp_path)
    assert len(quantized._generator.prompt_cache) == 0
    quantized.chat_completion(request)
    assert stored.exists() and len(list(tmp_path.glob("*.kv"))) == 2 and other.exists()

    # One flipped bit in the data: the checksum fails and the cache is not used.
    data = bytearray(stored.read_bytes())
    data[-1] ^= 1
    stored.write_bytes(bytes(data))
    damaged = DllmEngine.from_model_file(model_path, prompt_cache=2, prompt_cache_dir=tmp_path)
    assert len(damaged._generator.prompt_cache) == 0
    result = damaged.chat_completion(request)
    assert result.cached_tokens == 0 and replace(result, cached_tokens=0) == replace(expected, cached_tokens=0)


def test_cache_store_rejects_files_that_do_not_fit(tmp_path):
    from etalii_dllm.prompt_cache import CacheStore

    store = CacheStore(tmp_path / "kv", "model-a")
    keys = np.arange(2 * 3 * 1 * 4, dtype=np.float32).reshape(2, 3, 1, 4)
    store.save([1, 2, 3], keys, keys + 1)
    store.save([1, 2, 3], keys, keys + 1)  # already stored: nothing written
    store.save([], keys[:, :0], keys[:, :0])  # nothing to store
    (path,) = store.files()
    tokens, read_keys, read_values = store.read(path)
    assert tokens == [1, 2, 3] and np.array_equal(read_keys, keys) and np.array_equal(read_values, keys + 1)
    assert CacheStore(tmp_path / "kv", "model-b").read(path) is None
    data = path.read_bytes()
    for broken in (b"", b"other" + data, data[:20], data[:-4]):
        path.write_bytes(broken)
        assert store.read(path) is None

    class Unfitting:
        tokens = ()

        def restore(self, *stored):
            raise ValueError("does not fit")

    store.save([4, 5, 6], keys, keys)
    assert store.load(Unfitting, 4) == []
    store.sync(None, [])
    assert store.files() == []


def test_kv_cache_export_and_restore(model_path):  # noqa: F811
    engine = DllmEngine.from_model_file(model_path, prompt_cache=0)
    model = engine.model
    cache = model.new_cache()
    model.forward_cached([1, 2, 3, 4], cache)
    tokens, keys, values = cache.export()
    copy = model.new_cache()
    copy.restore(tokens, keys, values)
    assert copy.tokens == tokens
    assert all(np.array_equal(a, b) for a, b in zip(copy.export()[1:], (keys, values), strict=True))
    assert np.array_equal(model.forward_cached([1, 2, 3, 4, 5], copy), model.forward([1, 2, 3, 4, 5]))
    with pytest.raises(ValueError, match="does not fit"):
        copy.restore(tokens, keys, values)  # not empty
    with pytest.raises(ValueError, match="does not fit"):
        model.new_cache().restore(tokens, keys[:1], values[:1])
    empty = model.new_cache().export()
    assert empty[0] == [] and empty[1].shape[1] == 0


def test_prompt_cache_dir_setting(model_path, tmp_path, monkeypatch):  # noqa: F811
    from etalii_dllm import engine as engine_module

    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(model_path))
    monkeypatch.setenv(engine_module.PROMPT_CACHE_DIR_ENVIRONMENT_VARIABLE, "")
    engine_module.use_model_file(None, prompt_cache_dir=tmp_path / "kv")
    try:
        engine = engine_module.default_engine()
        engine.chat_completion(ChatRequest([ChatMessage("user", "Hi")], max_tokens=3))
        assert len(list((tmp_path / "kv").glob("*.kv"))) == 1
    finally:
        engine_module.default_engine.cache_clear()


def test_cache_store_loads_at_most_the_capacity(model_path, tmp_path):  # noqa: F811
    from etalii_dllm.prompt_cache import CacheStore

    model = DllmEngine.from_model_file(model_path, prompt_cache=0).model
    store = CacheStore(tmp_path, "key")
    for tokens in ([1, 2, 3], [4, 5], [6]):
        cache = model.new_cache()
        model.forward_cached(tokens, cache)
        store.save(*cache.export())
    loaded = store.load(model.new_cache, 2)
    assert [c.tokens for c in loaded] == [store.read(p)[0] for p in store.files()[1:]]
