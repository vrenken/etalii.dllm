"""Phase 50: exact adaptive samplers: Mirostat 2.0 (#312), Mirostat 1.0 (#313), dynamic temperature (#314), in every
API, receipts, the specification and the reference implementation (#315)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from golden_values import ADAPTIVE_SAMPLERS_FINGERPRINT
from test_model_building import BYTE_LEVEL_TOKENIZER, _import

from etalii_dllm import reference, verify
from etalii_dllm.cli import main
from etalii_dllm.engine import DllmEngine
from etalii_dllm.numerics import log
from etalii_dllm.sampling import (
    GREEDY,
    LN2,
    Sampler,
    SamplingOptions,
    dynamic_temperature,
    mirostat_k,
    power,
)
from etalii_dllm.server.app import app
from etalii_dllm.server.contracts import id_payload

PROMPT = "Adaptive samplers keep the bits"
ADAPTIVE = SamplingOptions(temperature=0.9, seed=4, mirostat=2, mirostat_tau=3.0, dynatemp_range=0.5)


@pytest.fixture
def client():
    return TestClient(app)


def _zipf(n: int, s: float) -> np.ndarray:
    """Logits of a Zipf distribution with exponent ``s`` over ``n`` tokens (token 0 the most likely)."""
    return np.array([-s * math.log(i + 1) for i in range(n)], dtype=np.float32)


# -- options ------------------------------------------------------------------------------------------------------


def test_defaults_keep_the_original_record():
    assert GREEDY.record() == {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "seed": 0}
    record = ADAPTIVE.record()
    assert record["mirostat"] == 2 and record["dynatemp_range"] == 0.5 and "mirostat_eta" not in record
    assert SamplingOptions.from_record(record) == ADAPTIVE


BAD_OPTIONS = [
    {"mirostat": 3},
    {"mirostat_tau": -1.0},
    {"mirostat_eta": math.nan},
    {"dynatemp_range": -0.1},
    {"dynatemp_exponent": math.inf},
]


@pytest.mark.parametrize("bad", BAD_OPTIONS)
def test_options_are_validated(bad):
    with pytest.raises(ValueError):
        SamplingOptions(**bad)


# -- the pieces ---------------------------------------------------------------------------------------------------


def test_power():
    assert power(2.0, 0.0) == 1.0 and power(0.0, 3.0) == 0.0
    assert power(2.0, 10.0) == pytest.approx(1024.0, rel=1e-15)
    assert power(9.0, 0.5) == pytest.approx(3.0, rel=1e-15)


def test_dynamic_temperature_follows_the_entropy():
    flat = np.zeros(8, dtype=np.float32)
    peaked = np.array([20.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    assert dynamic_temperature(flat, 1.0, 0.5, 1.0) == pytest.approx(1.5, rel=1e-12)  # maximal entropy: the top
    assert dynamic_temperature(peaked, 1.0, 0.5, 1.0) == pytest.approx(0.5, abs=1e-6)  # almost none: the bottom
    assert dynamic_temperature(peaked, 0.3, 0.5, 1.0) >= 0.0  # the bottom never goes below 0
    assert dynamic_temperature(np.zeros(1, dtype=np.float32), 0.7, 0.5, 1.0) == 0.7
    half = np.array([0.0, 0.0, -1e9, -1e9], dtype=np.float32)  # entropy log 2 of a possible log 4
    assert dynamic_temperature(half, 1.0, 1.0, 2.0) == pytest.approx(0.0 + 2.0 * 0.25, rel=1e-12)


def test_a_zero_dynamic_temperature_is_greedy():
    peaked = np.array([30.0, 1.0, 0.0], dtype=np.float32)
    sampler = Sampler(SamplingOptions(temperature=0.5, dynatemp_range=0.5, seed=1))
    assert all(sampler.sample(peaked) == 0 for _ in range(5))


def test_mirostat_k_fits_zipf():
    probabilities = np.exp(_zipf(1000, 1.2).astype(np.float64)) / np.exp(_zipf(1000, 1.2).astype(np.float64)).sum()
    order = list(range(1000))
    p = probabilities.tolist()
    small, large = mirostat_k(p, order, 2.0), mirostat_k(p, order, 8.0)
    assert 1 <= small < large <= 1000  # a higher surprise limit keeps more candidates
    assert mirostat_k([1.0], [0], 10.0) == 1  # nothing to fit
    assert mirostat_k([0.5, 0.5], [0, 1], 10.0) == 2  # flat: s = 0, keep all
    assert mirostat_k([0.9, 0.1, 0.0], [0, 1, 2], -50.0) == 1  # a tiny k still keeps one
    assert mirostat_k([0.9, 0.1], [0, 1], 2000.0) == 2  # 2^mu overflows: keep all
    steep = [0.999999, 1e-6, 1e-12]
    assert mirostat_k(steep, [0, 1, 2], 1.0) == 1


def test_mirostat_2_truncates_by_surprise_and_learns():
    logits = _zipf(64, 1.0)
    options = SamplingOptions(temperature=1.0, mirostat=2, mirostat_tau=3.0, mirostat_eta=0.1, seed=3)
    sampler = Sampler(options)
    assert sampler.mu == 6.0
    probabilities = np.exp(logits.astype(np.float64)) / np.exp(logits.astype(np.float64)).sum()
    token = sampler.sample(logits)
    # Every candidate with surprise above mu = 6 bits is cut: those below 2^-6 likely.
    assert probabilities[token] >= 2.0**-6 * 0.99 or token == 0
    assert sampler.mu != 6.0
    mus = [sampler.mu]
    for _ in range(200):
        sampler.sample(logits)
        mus.append(sampler.mu)
    # mu settles so that the average surprise is near tau.
    assert abs(np.mean(mus[100:]) - 6.0) < 3.0


def test_mirostat_surprise_is_in_bits():
    sampler = Sampler(SamplingOptions(temperature=1.0, mirostat=2, mirostat_tau=1.0, mirostat_eta=1.0))
    token = sampler.sample(np.zeros(2, dtype=np.float32))  # mu = 2 keeps both; the drawn one has surprise 1 bit
    assert token in (0, 1)
    assert sampler.mu == pytest.approx(2.0 - (-log(0.5) / LN2 - 1.0), abs=1e-12)


def test_samplers_match_the_reference_on_random_settings():
    rng = np.random.default_rng(50)
    for trial in range(48):
        options = SamplingOptions(
            temperature=[0.5, 1.0, 1.4][trial % 3],
            seed=trial,
            mirostat=[0, 1, 2][(trial // 3) % 3],
            mirostat_tau=[5.0, 2.0][trial % 2],
            mirostat_eta=[0.1, 0.5][(trial // 2) % 2],
            dynatemp_range=[0.0, 0.4, 1.2][(trial // 9) % 3],
            dynatemp_exponent=[1.0, 0.5, 2.0][trial % 3],
            top_k=[0, 10][trial % 2],
            repetition_penalty=[1.0, 1.2][trial % 2],
        )
        engine, twin = Sampler(options, [1, 2]), reference.sampler(options)
        twin.begin([1, 2])
        for step in range(12):
            logits = (rng.standard_normal(40) * 3).astype(np.float32)
            allowed = sorted(int(t) for t in rng.choice(40, 7, replace=False)) if step % 5 == 4 else None
            token = engine.sample(logits, allowed)
            assert token == twin.sample(logits, allowed), (trial, step)
            assert engine.mu == twin.mu
            engine.accept(token)
            twin.accept(token)


# -- generation, APIs and the CLI (#315) ----------------------------------------------------------------------------


def test_golden_adaptive_samplers():
    engine = DllmEngine.create_default()
    result = engine.complete(PROMPT, 48, ADAPTIVE)
    assert result.fingerprint == ADAPTIVE_SAMPLERS_FINGERPRINT
    assert engine.complete(PROMPT, 48, ADAPTIVE).tokens == result.tokens
    assert engine.complete(PROMPT, 48, SamplingOptions(temperature=0.9, seed=4)).tokens != result.tokens
    v1 = engine.complete(PROMPT, 48, SamplingOptions(temperature=0.9, seed=4, mirostat=1))
    assert v1.tokens == engine.complete(PROMPT, 48, SamplingOptions(temperature=0.9, seed=4, mirostat=1)).tokens


def test_the_apis_take_the_samplers(client):
    body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 32, "temperature": 0.9, "seed": 4}
    plain = client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]["content"]
    adaptive = {**body, "mirostat": 2, "mirostat_tau": 3.0, "dynatemp_range": 0.5}
    first = client.post("/v1/chat/completions", json=adaptive).json()["choices"][0]["message"]["content"]
    assert first != plain
    assert client.post("/v1/chat/completions", json={**adaptive, "mirostat": 5}).status_code == 400
    completion = {"prompt": PROMPT, "max_tokens": 32, "temperature": 0.9, "seed": 4}
    a = client.post("/v1/completions", json=completion).json()["choices"][0]["text"]
    b = client.post("/v1/completions", json={**completion, "mirostat": 1}).json()["choices"][0]["text"]
    assert a != b
    ollama = {"model": "dllm", "prompt": PROMPT, "stream": False, "options": {"temperature": 0.9, "seed": 4}}
    o1 = client.post("/api/generate", json=ollama).json()["response"]
    ollama["options"].update(mirostat=2, mirostat_tau=3.0, mirostat_eta=0.2)
    o2 = client.post("/api/generate", json=ollama).json()["response"]
    assert o1 != o2 and o2 == client.post("/api/generate", json=ollama).json()["response"]


def test_ids_of_requests_without_the_samplers_are_unchanged():
    assert id_payload({"messages": [], "mirostat": None, "dynatemp_range": None}) == {"messages": []}
    assert id_payload({"options": {"seed": 1, "mirostat_tau": None}}) == {"options": {"seed": 1}}


def test_cli_flags(capsys):
    args = ["generate", "--prompt", PROMPT, "--max-tokens", "48", "--temperature", "0.9", "--seed", "4"]
    args += ["--mirostat", "2", "--mirostat-tau", "3", "--dynatemp-range", "0.5"]
    assert main(args) == 0
    assert ADAPTIVE_SAMPLERS_FINGERPRINT in capsys.readouterr().err


def test_verify_reference_checks_the_samplers(tmp_path):
    engine = DllmEngine.from_model_file(_import(tmp_path, tokenizer=BYTE_LEVEL_TOKENIZER, vocabulary=264))
    assert verify.check_reference(engine, max_tokens=8).results["adaptive"] == "equal"
