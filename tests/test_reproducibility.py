"""Exact-hash tests: any run-to-run drift fails the build. The current kernels are portable, so the same values hold
on every CI platform; that is a bonus, not a requirement (see docs/research/deterministic-inference.md)."""

from concurrent.futures import ThreadPoolExecutor

from golden_values import GREEDY_FINGERPRINT, SAMPLED_FINGERPRINT, SYSTEM_FINGERPRINT

from etalii_dllm.engine import DllmEngine
from etalii_dllm.sampling import GREEDY, SamplingOptions


def test_weights_are_bit_exact():
    assert DllmEngine.create_default().system_fingerprint == SYSTEM_FINGERPRINT


def test_greedy_generation_is_bit_exact():
    result = DllmEngine.create_default().complete("Hello, world", 32, GREEDY)
    assert result.fingerprint == GREEDY_FINGERPRINT


def test_sampled_generation_is_bit_exact():
    options = SamplingOptions(temperature=0.8, top_k=40, top_p=0.95, seed=1234)
    result = DllmEngine.create_default().complete("Hello, world", 32, options)
    assert result.fingerprint == SAMPLED_FINGERPRINT


def test_same_request_gives_same_output_across_engine_instances():
    options = SamplingOptions(temperature=1.0, seed=99)
    a = DllmEngine.create_default().complete("determinism", 64, options)
    b = DllmEngine.create_default().complete("determinism", 64, options)
    assert a.tokens == b.tokens


def test_different_seeds_give_different_output():
    engine = DllmEngine.create_default()
    a = engine.complete("determinism", 64, SamplingOptions(temperature=1.0, seed=1))
    b = engine.complete("determinism", 64, SamplingOptions(temperature=1.0, seed=2))
    assert a.tokens != b.tokens


def test_concurrent_requests_do_not_affect_each_other():
    engine = DllmEngine.create_default()
    options = SamplingOptions(temperature=0.9, seed=7)
    alone = engine.complete("context window", 48, options)

    def request(i: int):
        if i % 2 == 0:
            return engine.complete("context window", 48, options)
        return engine.complete(f"other traffic {i}", 48, SamplingOptions(temperature=1.0, seed=i))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(request, range(16)))
    for result in results[::2]:
        assert result.tokens == alone.tokens
