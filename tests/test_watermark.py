"""Phase 27: verifiable text watermarks. The green list is integer arithmetic only, the sampler adds delta to exactly
the green logits after bias and penalties, watermarked answers replay from receipts, detection needs only the
tokenizer, and the reference implementation agrees (the conformance vectors carry watermark cases)."""

from __future__ import annotations

import hashlib
import itertools
import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from golden_values import WATERMARK_FINGERPRINTS
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm import engine as engine_module
from etalii_dllm import receipts, reference, watermark
from etalii_dllm.chat import ChatMessage
from etalii_dllm.cli import main
from etalii_dllm.engine import ChatRequest, DllmEngine, default_engine
from etalii_dllm.numerics import fill_gaussian
from etalii_dllm.sampling import Sampler, SamplingOptions

MASK = (1 << 64) - 1


def splitmix_output(z: int) -> int:
    z &= MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


@pytest.fixture(scope="module")
def tiny(model_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(model_path)


@pytest.fixture
def served(model_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(model_path))
    default_engine.cache_clear()
    yield default_engine()
    default_engine.cache_clear()


# The green list


def test_green_list_is_the_documented_integer_rule():
    key = watermark.key_hash("secret")
    assert key == int.from_bytes(hashlib.sha256(b"dllm-watermark/1\0secret").digest()[:8], "little")
    for z in (0, 1, 2**63, MASK, 12345678901234567890):
        assert watermark.mix(z) == splitmix_output(z)
    seed = splitmix_output(key + 8 * 0x9E3779B97F4A7C15)
    limit = math.floor(0.25 * 2**64)
    mask = watermark.green_mask(key, 7, 5000, 0.25)
    expected = [splitmix_output(seed + (t + 1) * 0x9E3779B97F4A7C15) < limit for t in range(5000)]
    assert mask.tolist() == expected
    assert all(watermark.is_green(key, 7, t, 0.25) == expected[t] for t in range(0, 5000, 97))
    assert 0.23 < mask.mean() < 0.27
    assert not np.array_equal(mask, watermark.green_mask(key, 8, 5000, 0.25))  # depends on the previous token
    assert not np.array_equal(mask, watermark.green_mask(watermark.key_hash("other"), 7, 5000, 0.25))
    assert watermark.green_mask(key, -1, 10, 0.5).dtype == bool
    assert watermark.threshold(0.5) == 2**63
    for gamma in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="gamma"):
            watermark.threshold(gamma)


def test_detection_counts_and_scores():
    key = "k"
    hashed = watermark.key_hash(key)
    tokens = [5, 9, 12, 3, 3, 40, 2, 17]
    green = sum(watermark.is_green(hashed, a, b, 0.25) for a, b in itertools.pairwise(tokens))
    result = watermark.detect(tokens, key)
    assert (result.tokens, result.green) == (7, green)
    assert result.z == (green - 0.25 * 7) / math.sqrt(7 * 0.25 * 0.75)
    assert result.watermarked == (result.z >= watermark.THRESHOLD)
    with_prompt = watermark.detect(tokens, key, previous=1)
    assert with_prompt.tokens == 8
    assert with_prompt.green == green + watermark.is_green(hashed, 1, 5, 0.25)
    assert watermark.detect([4], key).to_json() == {
        "tokens": 0, "green": 0, "z": 0.0, "watermarked": False, "gamma": 0.25, "threshold": 4.0
    }  # fmt: skip


# The sampler


def test_sampler_adds_delta_to_the_green_logits_last():
    logits = fill_gaussian(3, 300) * np.float32(3.0)
    options = SamplingOptions(
        watermark_key="key", watermark_gamma=0.3, watermark_delta=1.5, logit_bias=((4, 2.0),), presence_penalty=0.5
    )
    sampler = Sampler(options, [11, 22])
    plain = Sampler(SamplingOptions(logit_bias=((4, 2.0),), presence_penalty=0.5), [11, 22])
    green = watermark.green_mask(watermark.key_hash("key"), 22, 300, 0.3)
    expected = plain.adjust(logits)
    expected[green] = expected[green] + np.float32(1.5)
    assert np.array_equal(sampler.adjust(logits), expected)
    sampler.accept(4)
    plain.accept(4)
    green = watermark.green_mask(watermark.key_hash("key"), 4, 300, 0.3)
    expected = plain.adjust(logits)
    expected[green] = expected[green] + np.float32(1.5)
    assert np.array_equal(sampler.adjust(logits), expected)
    # Without a prompt, the first token is judged after -1; with the repetition penalty, the history keeps the prompt.
    first = Sampler(SamplingOptions(watermark_key="key"))
    green = watermark.green_mask(watermark.key_hash("key"), -1, 300, 0.25)
    assert np.array_equal(first.adjust(logits), np.where(green, logits + np.float32(2.0), logits))
    repeating = Sampler(SamplingOptions(watermark_key="key", repetition_penalty=1.2), [9])
    assert repeating._previous == 9
    # The reference implementation agrees, greedy and sampled.
    for temperature in (0.0, 0.8):
        choices = SamplingOptions(temperature=temperature, seed=3, watermark_key="r", watermark_delta=3.0)
        ours, theirs = Sampler(choices, [7]), reference.sampler(choices)
        theirs.begin([7])
        for row in (fill_gaussian(s, 300) for s in range(10)):
            token = ours.sample(row)
            assert token == theirs.sample(row)
            ours.accept(token)
            theirs.accept(token)


def test_options_record_only_when_set():
    assert "watermark_key" not in SamplingOptions(temperature=0.5).record()  # older records keep their bytes
    options = SamplingOptions(watermark_key="k", watermark_gamma=0.5, watermark_delta=1.0)
    assert SamplingOptions.from_record(json.loads(json.dumps(options.record()))) == options
    assert options.adjusts_logits and not SamplingOptions().adjusts_logits
    for bad, message in [
        ({"watermark_key": ""}, "empty"),
        ({"watermark_gamma": 1.0}, "gamma"),
        ({"watermark_delta": math.inf}, "delta"),
    ]:
        with pytest.raises(ValueError, match=message):
            SamplingOptions(**bad)


# Generation, receipts and detection


def test_watermarked_generation_is_reproducible_and_detectable(tiny):
    prompt = "Once upon a time"
    options = SamplingOptions(temperature=0.7, seed=4, watermark_key="story", watermark_delta=6.0)
    first = tiny.complete(prompt, 24, options)
    assert first.tokens == tiny.complete(prompt, 24, options).tokens
    plain = tiny.complete(prompt, 24, SamplingOptions(temperature=0.7, seed=4))
    assert first.tokens != plain.tokens
    previous = tiny.tokenizer.encode(prompt)[-1]
    marked = watermark.detect(first.tokens, "story", previous=previous)
    unmarked = watermark.detect(plain.tokens, "story", previous=previous)
    assert marked.green > unmarked.green and marked.watermarked
    assert not watermark.detect(first.tokens, "another key", previous=previous).watermarked
    assert first.fingerprint == WATERMARK_FINGERPRINTS["generation"]
    digest = hashlib.sha256(json.dumps(marked.to_json(), sort_keys=True).encode()).hexdigest()
    assert digest == WATERMARK_FINGERPRINTS["detection"]


def test_receipts_replay_watermarked_answers(tiny):
    request = ChatRequest(
        [ChatMessage("user", "Tell me a story.")], 12, SamplingOptions(temperature=0.8, seed=2, watermark_key="r")
    )
    result = tiny.chat_completion(request)
    assert result.receipt is not None and result.receipt["request"]["options"]["watermark_key"] == "r"
    assert receipts.verify(tiny, result.receipt).ok
    plain = tiny.chat_completion(ChatRequest(request.messages, 12, SamplingOptions(temperature=0.8, seed=2)))
    assert "watermark_key" not in plain.receipt["request"]["options"]


def test_front_ends(served, capsys, tmp_path):
    client = TestClient(__import__("etalii_dllm.server.app", fromlist=["app"]).app)
    messages = [{"role": "user", "content": "Tell me a story."}]
    body = {"messages": messages, "max_tokens": 16, "temperature": 0.7, "seed": 3}
    plain = client.post("/v1/chat/completions", json=body).json()
    marked_body = {**body, "watermark": {"key": "api", "delta": 8.0}}
    marked = client.post("/v1/chat/completions", json=marked_body).json()
    assert marked["id"] != plain["id"]
    again = client.post("/v1/chat/completions", json=marked_body).json()
    assert (again["id"], again["choices"]) == (marked["id"], marked["choices"])  # usage counts prompt-cache hits
    expected = served.chat_completion(
        ChatRequest(
            [ChatMessage("user", "Tell me a story.")],
            16,
            SamplingOptions(temperature=0.7, seed=3, watermark_key="api", watermark_delta=8.0),
        )
    )
    assert marked["choices"][0]["message"]["content"] == expected.content
    anthropic = client.post(
        "/v1/messages",
        json={
            "max_tokens": 16,
            "temperature": 0.7,
            "seed": 3,
            "messages": messages,
            "watermark": {"key": "api", "delta": 8.0},
        },
    ).json()
    assert anthropic["content"][0]["text"] == expected.content
    ollama = client.post(
        "/api/chat",
        json={
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": 16,
                "temperature": 0.7,
                "seed": 3,
                "watermark_key": "api",
                "watermark_delta": 8.0,
            },
        },
    ).json()
    assert ollama["message"]["content"] == expected.content

    text = expected.content
    detected = client.post("/v1/watermark/detect", json={"text": text, "key": "api"}).json()
    assert detected == watermark.detect(served.tokenizer.encode(text), "api").to_json()
    assert client.post("/v1/watermark/detect", json={"text": text, "key": "api", "gamma": 2}).status_code == 400

    (tmp_path / "answer.txt").write_text(text, encoding="utf-8")
    code = main(["watermark", "detect", str(tmp_path / "answer.txt"), "--key", "api", "--json"])
    assert json.loads(capsys.readouterr().out) == detected and code == (0 if detected["watermarked"] else 1)
    main(["watermark", "detect", str(tmp_path / "answer.txt"), "--key", "api"])
    assert "green:" in capsys.readouterr().out
    assert main(["watermark", "detect", str(tmp_path / "missing.txt"), "--key", "api"]) == 2
    assert capsys.readouterr().err.startswith("dllm watermark: ")
    args = ["generate", "--prompt", "Hi", "--max-tokens", "6", "--temperature", "0.5", "--watermark-key", "cli"]
    assert main(args) == 0
    cli_text = capsys.readouterr().out
    assert main(args) == 0 and capsys.readouterr().out == cli_text
