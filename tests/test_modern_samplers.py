"""Phase 49: exact modern samplers: DRY (#307), XTC (#308), locally typical and top-n-sigma sampling (#309), in every
API, receipts, the specification and the reference implementation (#310)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from golden_values import MODERN_SAMPLERS_FINGERPRINT
from test_model_building import BYTE_LEVEL_TOKENIZER, _import

from etalii_dllm import reference, verify
from etalii_dllm.cli import main
from etalii_dllm.engine import DllmEngine
from etalii_dllm.sampling import (
    DRY_BREAKERS,
    DRY_MAX_MATCH,
    GREEDY,
    Sampler,
    SamplingOptions,
    dry_breakers,
    dry_penalties,
    sampler_fields,
    sigma_count,
    typical,
)
from etalii_dllm.server.app import app
from etalii_dllm.server.contracts import id_payload

PROMPT = "Modern samplers keep the bits"
MODERN = SamplingOptions(
    temperature=1.0,
    seed=5,
    top_n_sigma=2.0,
    typical_p=0.9,
    xtc_probability=0.5,
    xtc_threshold=0.05,
    dry_multiplier=0.8,
    dry_allowed_length=1,
)


@pytest.fixture
def client():
    return TestClient(app)


# -- options ------------------------------------------------------------------------------------------------------


def test_defaults_keep_the_original_record():
    assert GREEDY.record() == {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "seed": 0}
    assert not GREEDY.dry and not GREEDY.adjusts_logits
    record = MODERN.record()
    assert record["typical_p"] == 0.9 and "dry_sequence_breakers" not in record
    assert SamplingOptions.from_record(record) == MODERN
    custom = SamplingOptions(dry_multiplier=1.0, dry_sequence_breakers=("\n", "."))
    assert custom.record()["dry_sequence_breakers"] == ["\n", "."]
    assert SamplingOptions.from_record(custom.record()) == custom
    assert custom.adjusts_logits and not SamplingOptions(dry_multiplier=1.0, dry_penalty_last_n=0).dry


@pytest.mark.parametrize(
    "bad",
    [
        {"typical_p": 0.0},
        {"typical_p": 1.5},
        {"top_n_sigma": -1.0},
        {"top_n_sigma": math.inf},
        {"xtc_probability": 1.5},
        {"xtc_threshold": -0.1},
        {"dry_multiplier": -1.0},
        {"dry_base": 0.5},
        {"dry_allowed_length": 0},
        {"dry_penalty_last_n": -2},
        {"dry_sequence_breakers": ("",)},
    ],
)
def test_options_are_validated(bad):
    with pytest.raises(ValueError):
        SamplingOptions(**bad)


def test_sampler_fields_pick_what_a_request_sets():
    request = SimpleNamespace(typical_p=0.5, dry_sequence_breakers=["\n"], xtc_probability=None)
    assert sampler_fields(request) == {"typical_p": 0.5, "dry_sequence_breakers": ("\n",)}


# -- DRY (#307) ---------------------------------------------------------------------------------------------------


def test_dry_penalises_the_token_that_extends_a_repeat():
    # ... 1 2 3 4 ... 1 2 3: the 4 that followed "1 2 3" would repeat a run of three.
    window = [1, 2, 3, 4, 9, 1, 2, 3]
    assert dry_penalties(window, (), 0.8, 1.75, 2) == {4: 0.8 * 1.75}
    assert dry_penalties(window, (), 0.8, 1.75, 3) == {4: 0.8}
    assert dry_penalties(window, (), 0.8, 1.75, 4) == {}
    # A breaker inside the run cuts it short; a breaker at the end stops DRY.
    assert dry_penalties(window, {2}, 0.8, 1.75, 1) == {4: 0.8}
    assert dry_penalties([*window, 7], {7}, 0.8, 1.75, 1) == {}
    assert dry_penalties([5], (), 1.0, 2.0, 1) == {}
    # The longest run of each token counts: "a b" -> c (2) and "b" -> d (1).
    assert dry_penalties([1, 2, 3, 2, 4, 1, 2], (), 1.0, 2.0, 1) == {3: 2.0, 4: 1.0}


def test_dry_runs_are_capped():
    window = [7] * (DRY_MAX_MATCH + 50)
    penalties = dry_penalties(window, (), 1.0, 1.0, 1)
    assert penalties == {7: 1.0}
    assert dry_penalties(window, (), 1.0, 2.0, 1)[7] == 2.0 ** (DRY_MAX_MATCH - 1)


def test_dry_breakers_from_token_bytes():
    assert dry_breakers([b"a", b"\n", b"x:y", b"", b"**"], DRY_BREAKERS) == {1, 2, 4}


def test_dry_changes_greedy_answers_too():
    sampler = Sampler(SamplingOptions(dry_multiplier=5.0, dry_allowed_length=1), [1, 2, 1])
    logits = np.zeros(4, dtype=np.float32)
    logits[2] = 1.0
    assert Sampler(GREEDY, [1, 2, 1]).sample(logits) == 2
    assert sampler.sample(logits) == 0
    sampler.accept(0)
    adjusted = sampler.adjust(np.zeros(4, dtype=np.float32))
    assert adjusted.tolist() == [0.0, 0.0, 0.0, 0.0]  # "... 1 0": 0 never followed 0 before


# -- XTC (#308) ---------------------------------------------------------------------------------------------------


def test_xtc_drops_all_but_the_least_likely_top_choice():
    logits = np.log(np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32))
    always = SamplingOptions(temperature=1.0, xtc_probability=1.0, xtc_threshold=0.1, seed=1)
    draws = {Sampler(always.for_choice(i)).sample(logits) for i in range(40)}
    assert draws == {2, 3}  # 0 and 1 are excluded, 2 is the least likely choice above 0.1
    one = SamplingOptions(temperature=1.0, xtc_probability=1.0, xtc_threshold=0.4, seed=1)
    assert {Sampler(one.for_choice(i)).sample(logits) for i in range(40)} == {0, 1, 2, 3}


def test_xtc_off_keeps_the_random_stream():
    logits = np.random.default_rng(3).standard_normal(50).astype(np.float32)
    plain = SamplingOptions(temperature=1.0, seed=9)
    off = SamplingOptions(temperature=1.0, seed=9, xtc_threshold=0.3)
    a, b = Sampler(plain), Sampler(off)
    assert [a.sample(logits) for _ in range(20)] == [b.sample(logits) for _ in range(20)]


# -- typical-p and top-n-sigma (#309) -----------------------------------------------------------------------------


def test_sigma_count():
    assert sigma_count([0.0, 0.0, 0.0], 1.0) == 3
    assert sigma_count([10.0, 0.0, 0.0, 0.0], 1.0) == 1
    # Mean 0.5 and sigma sqrt(1/6) of the finite logits (-inf is left out): one is within 1 sigma, two within 2.
    assert sigma_count([1.0, 0.5, -math.inf, 0.0], 1.0) == 1
    assert sigma_count([1.0, 0.5, -math.inf, 0.0], 2.0) == 2
    assert sigma_count([-math.inf, -math.inf], 1.0) == 2


def test_typical_keeps_the_candidates_closest_to_the_entropy():
    probabilities = [0.6, 0.3, 0.07, 0.03]
    assert typical([0, 1, 2, 3], probabilities, 1.0) == [0, 1, 2, 3]
    # H = 1.0045; surprises 0.51, 1.20, 2.66, 3.51: 1 then 0 are closest.
    assert typical([0, 1, 2, 3], probabilities, 0.25) == [1]
    assert typical([0, 1, 2, 3], probabilities, 0.8) == [0, 1]
    assert typical([0, 1], [0.5, 0.5, 0.0], 0.4) == [0]  # ties go to the earlier candidate
    assert typical([0, 1], [1.0, 0.0], 0.5) == [0]


def test_sampler_matches_the_reference_on_random_settings():
    rng = np.random.default_rng(49)
    for trial in range(60):
        options = SamplingOptions(
            temperature=[0.0, 0.7, 1.3][trial % 3],
            seed=trial,
            top_k=[0, 12][trial % 2],
            top_p=[1.0, 0.95][trial % 2],
            min_p=[0.0, 0.03][trial % 3 == 1],
            top_n_sigma=[0.0, 1.0, 2.5][trial % 3],
            typical_p=[1.0, 0.7, 0.95][(trial // 3) % 3],
            xtc_probability=[0.0, 0.5, 1.0][(trial // 2) % 3],
            xtc_threshold=[0.1, 0.02][trial % 2],
            dry_multiplier=[0.0, 0.8, 3.0][(trial // 4) % 3],
            dry_allowed_length=[1, 2][trial % 2],
            dry_penalty_last_n=[-1, 6][(trial // 5) % 2],
            dry_base=[1.75, 3.0][trial % 2],
        )
        breakers = frozenset(int(t) for t in rng.integers(0, 32, 3))
        prompt = [int(t) for t in rng.integers(0, 8, 14)]
        engine = Sampler(options, prompt, breakers)
        twin = reference.sampler(options, sorted(breakers))
        twin.begin(prompt)
        for step in range(12):
            logits = (rng.standard_normal(32) * 2).astype(np.float32)
            allowed = sorted(int(t) for t in rng.choice(32, 9, replace=False)) if step % 4 == 3 else None
            token = engine.sample(logits, allowed)
            assert token == twin.sample(logits, allowed), (trial, step)
            engine.accept(token)
            twin.accept(token)


# -- generation, APIs and the CLI (#310) ----------------------------------------------------------------------------


def test_golden_modern_samplers():
    engine = DllmEngine.create_default()
    result = engine.complete(PROMPT, 48, MODERN)
    assert result.fingerprint == MODERN_SAMPLERS_FINGERPRINT
    assert engine.complete(PROMPT, 48, MODERN).tokens == result.tokens
    plain = engine.complete(PROMPT, 48, SamplingOptions(temperature=1.0, seed=5))
    assert plain.tokens != result.tokens


def test_the_openai_apis_take_the_samplers(client):
    body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 24, "temperature": 1.0, "seed": 5}
    plain = client.post("/v1/chat/completions", json=body).json()
    modern = {**body, "typical_p": 0.9, "top_n_sigma": 2.0, "xtc_probability": 0.5, "xtc_threshold": 0.05,
              "dry_multiplier": 0.8, "dry_allowed_length": 1, "dry_sequence_breakers": ["\n"]}  # fmt: skip
    first = client.post("/v1/chat/completions", json=modern).json()
    assert first == client.post("/v1/chat/completions", json=modern).json()
    assert first["choices"][0]["message"]["content"] != plain["choices"][0]["message"]["content"]
    assert client.post("/v1/chat/completions", json={**modern, "typical_p": 0}).status_code == 400
    completion = {"prompt": PROMPT, "max_tokens": 48, "temperature": 1.0, "seed": 5}
    dry = {**completion, "dry_multiplier": 5.0, "dry_allowed_length": 1}
    a = client.post("/v1/completions", json=completion).json()["choices"][0]["text"]
    b = client.post("/v1/completions", json=dry).json()["choices"][0]["text"]
    assert a != b


def test_the_ollama_api_takes_the_samplers(client):
    body = {"model": "dllm", "prompt": PROMPT, "stream": False, "options": {"temperature": 1.0, "seed": 5}}
    plain = client.post("/api/generate", json=body).json()["response"]
    body["options"].update(typical_p=0.7, xtc_probability=1.0, dry_multiplier=1.0)
    modern = client.post("/api/generate", json=body).json()["response"]
    assert modern != plain and modern == client.post("/api/generate", json=body).json()["response"]


def test_ids_of_requests_without_the_samplers_are_unchanged():
    assert id_payload({"messages": [], "typical_p": None, "dry_sequence_breakers": None}) == {"messages": []}
    assert id_payload({"options": {"seed": 1, "xtc_probability": None}}) == {"options": {"seed": 1}}


def test_cli_flags(capsys):
    args = ["generate", "--prompt", PROMPT, "--max-tokens", "48", "--temperature", "1.0", "--seed", "5"]
    args += ["--top-n-sigma", "2", "--typical-p", "0.9", "--xtc-probability", "0.5", "--xtc-threshold", "0.05"]
    args += ["--dry-multiplier", "0.8", "--dry-allowed-length", "1"]
    assert main(args) == 0
    assert MODERN_SAMPLERS_FINGERPRINT in capsys.readouterr().err
    breakers = ["generate", "--prompt", PROMPT, "--dry-multiplier", "1", "--dry-sequence-breaker", "."]
    assert main(breakers) == 0
    assert main(["generate", "--typical-p", "2"]) == 2


def test_verify_reference_checks_the_samplers(tmp_path):
    engine = DllmEngine.from_model_file(_import(tmp_path, tokenizer=BYTE_LEVEL_TOKENIZER, vocabulary=264))
    assert verify.check_reference(engine, max_tokens=8).results["modern"] == "equal"
