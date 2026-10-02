"""Phase 21 decoding controls: penalties, min-p and logit bias (#166), several choices (#167), regex-constrained
output (#168), and their agreement with the reference implementation (#169)."""

from __future__ import annotations

import json
import random
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient
from golden_values import DECODING_CONTROLS_FINGERPRINT
from test_model_building import BYTE_LEVEL_TOKENIZER, _import

from etalii_dllm import reference, verify
from etalii_dllm.chat import ChatMessage
from etalii_dllm.cli import main
from etalii_dllm.engine import ChatRequest, DllmEngine, ResponseFormat, default_engine
from etalii_dllm.grammar import Grammar, GrammarError
from etalii_dllm.receipts import request_from_record, request_record
from etalii_dllm.regexp import compile_regex, utf8_sequences
from etalii_dllm.sampling import GREEDY, Sampler, SamplingOptions
from etalii_dllm.server.app import app
from etalii_dllm.server.contracts import id_payload

PROMPT = "Deterministic decoding controls are"
CONTROLLED = SamplingOptions(
    temperature=0.9,
    seed=3,
    min_p=0.05,
    repetition_penalty=1.4,
    repeat_last_n=-1,
    frequency_penalty=0.5,
    presence_penalty=0.3,
    logit_bias=((32, -1.5), (101, 2.0)),
)


@pytest.fixture
def client():
    return TestClient(app)


def _logits(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


# -- the sampler (#166) -------------------------------------------------------------------------------------------


def test_defaults_keep_the_original_record():
    assert GREEDY.record() == {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "seed": 0}
    assert not GREEDY.adjusts_logits
    assert SamplingOptions.from_record(CONTROLLED.record()) == CONTROLLED
    assert CONTROLLED.record()["logit_bias"] == [[32, -1.5], [101, 2.0]]


def test_options_are_validated():
    for bad in (
        {"min_p": 1.5},
        {"repetition_penalty": 0.0},
        {"repetition_penalty": float("nan")},
        {"repeat_last_n": -2},
        {"frequency_penalty": float("inf")},
        {"logit_bias": ((3, 1.0), (1, 1.0))},
        {"logit_bias": ((1, 1.0), (1, 2.0))},
        {"logit_bias": ((-1, 1.0),)},
        {"logit_bias": ((1, float("nan")),)},
    ):
        with pytest.raises(ValueError):
            SamplingOptions(**bad)
    assert SamplingOptions.bias({"7": 1, 2: -3.5}) == ((2, -3.5), (7, 1.0))
    assert SamplingOptions.bias(None) == ()


def test_logit_bias_steers_greedy_decoding():
    logits = _logits([1.0, 3.0, 2.0])
    assert Sampler(GREEDY).sample(logits) == 1
    assert Sampler(SamplingOptions(logit_bias=((2, 1.5),))).sample(logits) == 2
    assert Sampler(SamplingOptions(logit_bias=((9, 100.0),))).sample(logits) == 1  # past the vocabulary: ignored


def test_penalties_follow_the_documented_formulas():
    options = SamplingOptions(
        repetition_penalty=2.0, repeat_last_n=2, frequency_penalty=0.25, presence_penalty=0.5, logit_bias=((0, 1.0),)
    )
    sampler = Sampler(options, prompt=[3, 0])
    sampler.accept(1)
    sampler.accept(1)
    adjusted = sampler.adjust(_logits([1.0, -2.0, 4.0, 8.0]))
    # the window holds the last two tokens (1, 1): only token 1 gets the repetition penalty; token 0 is biased
    expected = [np.float32(2.0), np.float32(np.float32(-2.0 * 2.0) - np.float32(2 * 0.25 + 0.5)), 4.0, 8.0]
    assert adjusted.tolist() == [float(v) for v in expected]
    every = Sampler(SamplingOptions(repetition_penalty=2.0, repeat_last_n=-1), prompt=[0, 3])
    assert every.adjust(_logits([1.0, 1.0, 1.0, -1.0])).tolist() == [0.5, 1.0, 1.0, -2.0]
    off = Sampler(SamplingOptions(repetition_penalty=2.0, repeat_last_n=0), prompt=[0])
    assert off.adjust(_logits([1.0])).tolist() == [1.0]


def test_min_p_drops_unlikely_candidates():
    logits = _logits([5.0, 4.9, 0.0, -1.0])
    draws = {Sampler(SamplingOptions(temperature=1.0, seed=s, min_p=0.5)).sample(logits) for s in range(64)}
    assert draws == {0, 1}
    loose = {Sampler(SamplingOptions(temperature=1.0, seed=s)).sample(_logits([1.0, 1.0, 0.9, 0.8])) for s in range(64)}
    assert loose == {0, 1, 2, 3}


def test_penalties_reduce_repetition_and_stay_reproducible():
    engine = default_engine()
    plain = engine.complete(PROMPT, 48, SamplingOptions(temperature=0.9, seed=3))
    controlled = engine.complete(PROMPT, 48, CONTROLLED)
    assert controlled.tokens != plain.tokens
    assert engine.complete(PROMPT, 48, CONTROLLED).tokens == controlled.tokens
    assert controlled.fingerprint == DECODING_CONTROLS_FINGERPRINT
    assert len(set(controlled.tokens)) >= len(set(plain.tokens))


def test_logit_bias_beyond_the_vocabulary_is_rejected():
    engine = default_engine()
    with pytest.raises(ValueError, match="vocabulary size"):
        engine.complete(PROMPT, 4, SamplingOptions(logit_bias=((100000, 1.0),)))


def test_receipts_record_the_controls_and_replay_them():
    request = ChatRequest(
        [ChatMessage("user", "hi")], 8, CONTROLLED, response_format=ResponseFormat("regex", pattern="a+")
    )
    record = request_record(request)
    assert record["options"]["min_p"] == 0.05 and record["response_format"]["pattern"] == "a+"
    assert request_from_record(record) == request
    plain = request_record(ChatRequest([ChatMessage("user", "hi")], 8))
    assert set(plain["options"]) == {"temperature", "top_k", "top_p", "seed"}
    assert plain["response_format"] == {"type": "text", "schema": None}


def test_the_openai_api_takes_the_controls(client):
    body = {
        "messages": [{"role": "user", "content": "Repeat after me."}],
        "max_tokens": 24,
        "temperature": 0.9,
        "seed": 3,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "logit_bias": {"101": 2.0, "32": -1.5},
        "min_p": 0.05,
        "repetition_penalty": 1.4,
        "repeat_last_n": -1,
    }
    first = client.post("/v1/chat/completions", json=body).json()
    assert first == client.post("/v1/chat/completions", json=body).json()
    plain = client.post(
        "/v1/chat/completions", json={k: body[k] for k in ("messages", "max_tokens", "temperature", "seed")}
    )
    assert first["choices"][0]["message"]["content"] != plain.json()["choices"][0]["message"]["content"]
    bad = client.post("/v1/chat/completions", json={**body, "logit_bias": {"999999": 1.0}})
    assert bad.status_code == 400


def test_ids_of_requests_without_the_controls_are_unchanged():
    dumped = {"messages": [], "min_p": None, "logit_bias": None, "options": {"seed": 1, "repeat_penalty": None}}
    dumped["response_format"] = {"type": "json_object", "json_schema": None, "regex": None}
    assert id_payload(dumped) == {
        "messages": [],
        "options": {"seed": 1},
        "response_format": {"type": "json_object", "json_schema": None},
    }
    assert id_payload({"min_p": 0.1}) == {"min_p": 0.1}


def test_the_ollama_api_takes_the_controls(client):
    body = {
        "model": "dllm",
        "prompt": "Repeat after me.",
        "stream": False,
        "options": {"temperature": 0.9, "seed": 3, "num_predict": 24},
    }
    plain = client.post("/api/generate", json=body).json()["response"]
    body["options"].update(
        repeat_penalty=1.4, repeat_last_n=-1, frequency_penalty=0.5, presence_penalty=0.3, min_p=0.05
    )
    controlled = client.post("/api/generate", json=body).json()["response"]
    assert controlled != plain
    assert controlled == client.post("/api/generate", json=body).json()["response"]


def test_cli_flags(capsys):
    args = ["generate", "--prompt", PROMPT, "--max-tokens", "48", "--temperature", "0.9", "--seed", "3"]
    args += ["--min-p", "0.05", "--repetition-penalty", "1.4", "--repeat-last-n", "-1", "--frequency-penalty", "0.5"]
    args += ["--presence-penalty", "0.3", "--logit-bias", "101=2", "--logit-bias", "32=-1.5"]
    assert main(args) == 0
    assert DECODING_CONTROLS_FINGERPRINT in capsys.readouterr().err
    assert main(["generate", "--logit-bias", "nope"]) == 2
    assert "TOKEN=BIAS" in capsys.readouterr().err


# -- several choices (#167) ---------------------------------------------------------------------------------------


def test_choice_seeds_wrap_around():
    options = SamplingOptions(temperature=1.0, seed=2**64 - 1)
    assert options.for_choice(0) is options
    assert options.for_choice(1).seed == 0 and options.for_choice(2).seed == 1


def test_each_choice_is_its_solo_answer(client):
    body = {"messages": [{"role": "user", "content": "Pick a word."}], "max_tokens": 12, "temperature": 1.0}
    body["seed"] = 40
    response = client.post("/v1/chat/completions", json={**body, "n": 3}).json()
    assert [c["index"] for c in response["choices"]] == [0, 1, 2]
    for index, choice in enumerate(response["choices"]):
        solo = client.post("/v1/chat/completions", json={**body, "seed": 40 + index}).json()
        assert choice["message"] == solo["choices"][0]["message"]
        assert choice["finish_reason"] == solo["choices"][0]["finish_reason"]
    solo_tokens = [client.post("/v1/chat/completions", json={**body, "seed": 40 + i}).json() for i in range(3)]
    assert response["usage"]["completion_tokens"] == sum(s["usage"]["completion_tokens"] for s in solo_tokens)
    assert len({c["message"]["content"] for c in response["choices"]}) > 1


def test_choices_on_a_batching_model_keep_their_solo_bits(tmp_path):
    engine = DllmEngine.from_model_file(_import(tmp_path, tokenizer=BYTE_LEVEL_TOKENIZER, vocabulary=264))
    request = ChatRequest([ChatMessage("user", "hello")], 10, SamplingOptions(temperature=1.0, seed=9))
    choices = engine.chat_choices(request, 4)
    solo = [engine.chat_completion(engine.choice_request(request, i)) for i in range(4)]
    assert [c.fingerprint for c in choices] == [s.fingerprint for s in solo]
    with pytest.raises(ValueError, match="between 1 and"):
        engine.chat_choices(request, 0)


def test_streamed_choices_come_one_after_another(client):
    body = {"messages": [{"role": "user", "content": "Pick a word."}], "max_tokens": 8, "temperature": 1.0, "seed": 1}
    body.update(n=2, stream=True, stream_options={"include_usage": True}, receipt=True)
    lines = [line[6:] for line in client.post("/v1/chat/completions", json=body).text.splitlines() if line]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in lines[:-1]]
    indices = [c["choices"][0]["index"] for c in chunks if c["choices"]]
    assert indices == sorted(indices) and set(indices) == {0, 1}
    texts = {
        i: "".join(
            c["choices"][0]["delta"].get("content") or ""
            for c in chunks
            if c["choices"] and c["choices"][0]["index"] == i
        )
        for i in (0, 1)
    }
    unstreamed = client.post("/v1/chat/completions", json={**body, "stream": False}).json()
    assert [texts[0], texts[1]] == [c["message"]["content"] for c in unstreamed["choices"]]
    assert chunks[-1]["usage"]["completion_tokens"] == unstreamed["usage"]["completion_tokens"]
    receipts = [c["receipt"] for c in chunks if "receipt" in c]
    assert len(receipts) == 2 and receipts[0]["id"] == unstreamed["receipt"]["id"]
    assert [c["receipt"]["id"] for c in unstreamed["choices"]] == [r["id"] for r in receipts]
    assert client.post("/v1/chat/completions", json={**body, "n": 0}).status_code == 400


def test_cli_generates_several_choices(capsys):
    assert (
        main(["generate", "--prompt", "Hi", "--max-tokens", "6", "--temperature", "1", "--seed", "4", "--n", "2"]) == 0
    )
    err = capsys.readouterr().err
    assert "choice 0 (seed 4)" in err and "choice 1 (seed 5)" in err
    assert main(["generate", "--n", "0"]) == 2


# -- regex-constrained output (#168) ------------------------------------------------------------------------------

PATTERNS = [
    r"\d{3}-\d{4}",
    r"(yes|no|maybe)",
    r"[a-z]+@[a-z]+\.(com|org)",
    r"\w+( \w+)*\.",
    r"[^aeiou\n]{2,5}",
    r"(?:ab|a)*b?",
    "\u00e9+[\u03b1-\u03c9]*",
    r".{0,3}x",
    r"^\$\d+(\.\d\d)?$",
    r"(?P<pair>[A-F0-9]{2}){1,3}",
    r"(?:a){2,}",
    r"[\]\-a]+",
    r"x{",
    r"x{2,1a}",
    r"\x41é+\t?",
    r"a??b",
    r"\D\W\S[\d\s]",
    r"[^\x00-\x7f]",
]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_regex_automata_match_like_python(pattern):
    automaton = compile_regex(pattern)
    rng = random.Random(pattern)
    alphabet = "abcdeouxyz0123456789-@.$ ABCF\t\n\u00e9\u03b1\u20ac\U0001f600]{}"
    for _ in range(1500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 8)))
        assert automaton.matches(text.encode()) == bool(re.fullmatch(pattern, text, re.ASCII)), text
    assert not automaton.matches(b"\xff")


@pytest.mark.parametrize(
    "pattern",
    [r"(a", r"a)", r"*a", r"a**", r"a{2}{3}", r"\1", r"(?=a)", r"\b", r"a{3,1}", r"[z-a]", r"(?i)a", r"(?<=a)b",
     r"(?P=n)", r"[[:alpha:]]", r"a|*", r"a{1001}", r"\xZZ", r"\ud800", r"[a-\d]", r"a^", r"$a", r"[a", r"(?<n"],
)  # fmt: skip
def test_unsupported_regex_syntax_is_rejected(pattern):
    with pytest.raises(GrammarError):
        compile_regex(pattern)


def test_named_groups():
    assert compile_regex(r"(?<n>a){2,}").matches(b"aaa")
    assert compile_regex(r"(?P<n>ab)+").matches(b"abab")


def test_huge_regexes_are_rejected():
    with pytest.raises(GrammarError, match="too large"):
        compile_regex(r"(\w{1000}){1000}")


def test_utf8_sequences_cover_exactly_the_code_points():
    for low, high in [(0, 0x7F), (0x80, 0x7FF), (0x41, 0x10FFFF), (0x7A0, 0x3001), (0xD000, 0xE100)]:
        sequences = utf8_sequences(low, high)
        count = sum(int(np.prod([b - a + 1 for a, b in s])) for s in sequences)
        assert count == sum(1 for c in range(low, high + 1) if not 0xD800 <= c <= 0xDFFF)
        for code in random.Random(low).sample(range(low, high + 1), min(500, high - low + 1)):
            if 0xD800 <= code <= 0xDFFF:
                continue
            data = chr(code).encode()
            same = [s for s in sequences if len(s) == len(data)]
            fits = [all(a <= b <= c for (a, c), b in zip(s, data, strict=True)) for s in same]
            assert any(fits)


def test_regex_constrained_generation_matches_the_pattern(capsys):
    engine = default_engine()
    pattern = r"\d{3}-\d{3}-\d{4}"
    for seed in range(3):
        result = engine.complete_stream(
            "Call ", 40, SamplingOptions(temperature=1.0, seed=seed), regex=pattern
        ).result()
        assert re.fullmatch(pattern, result.text), result.text
        assert result.finish_reason == "stop"
    assert main(["generate", "--prompt", "x", "--regex", "(a", "--max-tokens", "4"]) == 2
    assert main(["generate", "--prompt", "Call ", "--regex", pattern, "--max-tokens", "40"]) == 0
    assert re.search(pattern, capsys.readouterr().out)


def test_regex_in_a_sequence_grammar():
    grammar = Grammar.sequence([Grammar.literal("<"), Grammar.regex(r"[0-9]+"), Grammar.literal(">")])
    matcher = grammar.matcher()
    assert matcher.matches(b"<42>") and not matcher.matches(b"<>") and not matcher.matches(b"<4a>")
    assert Grammar.regex("").matcher().matches(b"")


def test_regex_through_the_openai_api(client):
    body = {"messages": [{"role": "user", "content": "A date?"}], "max_tokens": 30, "temperature": 0.8, "seed": 2}
    pattern = r"\d{4}-\d{2}-\d{2}"
    a = client.post("/v1/chat/completions", json={**body, "guided_regex": pattern}).json()
    b = client.post("/v1/chat/completions", json={**body, "response_format": {"type": "regex", "regex": pattern}})
    assert re.fullmatch(pattern, a["choices"][0]["message"]["content"])
    assert b.json()["choices"][0]["message"] == a["choices"][0]["message"]
    both = {**body, "guided_regex": pattern, "response_format": {"type": "json_object"}}
    assert client.post("/v1/chat/completions", json=both).status_code == 400
    missing = {**body, "response_format": {"type": "regex"}}
    assert client.post("/v1/chat/completions", json=missing).status_code == 400
    bad = client.post("/v1/chat/completions", json={**body, "guided_regex": "(a"})
    assert bad.status_code == 400


def test_regex_in_chat_on_the_command_line(capsys):
    assert main(["chat", "Answer", "--regex", "(yes|no)", "--max-tokens", "8"]) == 0
    assert capsys.readouterr().out.strip() in ("yes", "no")
    assert main(["chat", "Answer", "--regex", "a", "--json"]) == 1


def test_response_format_validation():
    with pytest.raises(ValueError, match="needs a pattern"):
        ResponseFormat("regex")
    with pytest.raises(ValueError, match="only regex and grammar"):
        ResponseFormat("text", pattern="a")


# -- the reference implementation (#169) --------------------------------------------------------------------------


def test_the_reference_sampler_gives_the_same_tokens():
    rng = np.random.default_rng(5)
    for trial in range(20):
        options = SamplingOptions(
            temperature=[0.0, 0.7, 1.3][trial % 3],
            top_k=[0, 5][trial % 2],
            top_p=[1.0, 0.9][trial % 2],
            seed=trial,
            min_p=[0.0, 0.1][trial % 2],
            repetition_penalty=[1.0, 1.3, 0.8][trial % 3],
            repeat_last_n=[64, -1, 3][trial % 3],
            frequency_penalty=[0.0, 0.7, -0.4][trial % 3],
            presence_penalty=[0.0, 0.2][trial % 2],
            logit_bias=((1, 3.0), (17, -2.5)) if trial % 2 else (),
        )
        prompt = [int(t) for t in rng.integers(0, 32, 10)]
        engine, twin = Sampler(options, prompt), reference.sampler(options)
        twin.begin(prompt)
        for _ in range(12):
            logits = rng.standard_normal(32).astype(np.float32)
            token = engine.sample(logits)
            assert token == twin.sample(logits)
            engine.accept(token)
            twin.accept(token)


def test_verify_reference_checks_the_controls(tmp_path):
    engine = DllmEngine.from_model_file(_import(tmp_path, tokenizer=BYTE_LEVEL_TOKENIZER, vocabulary=264))
    assert verify.check_reference(engine, max_tokens=8).results["controlled"] == "equal"
