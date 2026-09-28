"""Streaming, stop sequences, logprobs and constrained decoding in the generator and the engine."""

import json
import math

import jsonschema
import pytest
from golden_values import CONSTRAINED_FINGERPRINT, EMBEDDING_FINGERPRINT

from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import ChatRequest, DllmEngine, ResponseFormat, TextDelta
from etalii_dllm.generation import _complete_prefix, _held_back
from etalii_dllm.numerics import fingerprint, softmax
from etalii_dllm.sampling import GREEDY, SamplingOptions

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "pets": {"type": "array", "items": {"enum": ["cat", "dog"]}},
        "happy": {"type": "boolean"},
    },
    "required": ["name", "age", "happy"],
}


@pytest.fixture(scope="module")
def engine():
    return DllmEngine.create_default()


def test_streamed_text_equals_the_generated_text(engine):
    for seed in range(5):
        options = SamplingOptions(temperature=1.0, seed=seed)
        generation = engine.complete_stream("stream me", 40, options)
        steps = list(generation)
        result = engine.complete("stream me", 40, options)
        assert "".join(s.text for s in steps) == result.text
        assert tuple(s.token for s in steps if s.token is not None) == result.tokens
        assert steps[-1].finish_reason == result.finish_reason
        assert all(s.finish_reason is None for s in steps[:-1])


def test_incomplete_utf8_is_held_back():
    assert _complete_prefix("a€".encode()[:-1]) == 1
    assert _complete_prefix("a€".encode()) == 4
    assert _complete_prefix(b"a\x80\x80\x80\x80") == 5  # invalid bytes are released (as replacement characters)
    assert _held_back("hello wo", ["world", "xyz"]) == 2
    assert _held_back("hello", ["world"]) == 0


def test_stop_sequences_end_the_text(engine):
    options = SamplingOptions(temperature=1.0, seed=3)
    full = engine.complete("stop test", 60, options).text
    start = next(i for i in range(5, len(full) - 1) if full[i : i + 2].isascii() and full[i : i + 2].isprintable())
    stop = full[start : start + 2]
    index = full.find(stop)
    generation = engine._generator.stream("stop test", 60, options, stop=[stop, "never-there"])
    steps = list(generation)
    result = generation.result()
    assert result.text == full[:index]
    assert result.finish_reason == "stop" and generation.stop_sequence == stop
    assert "".join(s.text for s in steps) == result.text


def test_logprobs_are_the_model_distribution(engine):
    result = engine._generator.generate("logprobs", 6, GREEDY, top_logprobs=3)
    assert len(result.logprobs) == len(result.tokens) == 6
    context = engine.tokenizer.encode("logprobs")
    probabilities = softmax(engine.model.forward(context))
    first = result.logprobs[0]
    assert first.token == result.tokens[0]
    assert first.logprob == pytest.approx(math.log(float(probabilities[first.token])), abs=1e-5)
    assert first.top[0].token == first.token  # greedy picks the most likely token
    values = [t.logprob for t in first.top]
    assert values == sorted(values, reverse=True)
    with pytest.raises(ValueError):
        engine._generator.generate("x", 1, GREEDY, top_logprobs=21)


def chat(engine, response_format, options, max_tokens=300):
    request = ChatRequest([ChatMessage("user", "Describe a person.")], max_tokens, options,
                          response_format=response_format)  # fmt: skip
    return engine.chat_completion(request)


def test_structured_output_is_valid_json(engine):
    # The placeholder model babbles, so some strings never close within the budget; every finished answer must be
    # valid, and most must finish.
    finished = 0
    for seed in range(8):
        result = chat(engine, ResponseFormat("json_schema", SCHEMA), SamplingOptions(temperature=1.0, seed=seed))
        if result.finish_reason == "stop":
            finished += 1
            jsonschema.validate(json.loads(result.content), SCHEMA)
    assert finished >= 4


def test_json_object_mode(engine):
    for seed in range(4):
        result = chat(engine, ResponseFormat("json_object"), SamplingOptions(temperature=0.7, seed=seed), 400)
        if result.finish_reason == "stop":
            assert isinstance(json.loads(result.content), dict)


def test_greedy_shortcut_equals_the_full_mask(engine, monkeypatch):
    fast = chat(engine, ResponseFormat("json_schema", SCHEMA), GREEDY)
    from etalii_dllm.sampling import Sampler

    monkeypatch.setattr(Sampler, "greedy", property(lambda self: False))
    slow = chat(engine, ResponseFormat("json_schema", SCHEMA), GREEDY)
    assert fast == slow


def test_constrained_generation_is_bit_exact(engine):
    result = chat(engine, ResponseFormat("json_schema", SCHEMA), SamplingOptions(temperature=0.8, seed=11))
    assert result.fingerprint == CONSTRAINED_FINGERPRINT


def test_chat_stream_matches_chat_completion(engine):
    request = ChatRequest([ChatMessage("user", "Hi")], 30, SamplingOptions(temperature=0.9, seed=4), top_logprobs=1)
    events = list(engine.chat_stream(request))
    result = engine.chat_completion(request)
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == result.content
    assert len(result.logprobs) == result.completion_tokens == 30


def test_embeddings_are_normalised_and_bit_exact(engine):
    first = engine.embed("embed this text")
    assert first.tokens == len("embed this text")
    assert sum(float(v) * float(v) for v in first.vector) == pytest.approx(1.0, abs=1e-5)
    assert fingerprint(first.vector) == EMBEDDING_FINGERPRINT
    assert fingerprint(DllmEngine.create_default().embed("embed this text").vector) == EMBEDDING_FINGERPRINT
    short = engine.embed("embed this text", dimensions=16)
    assert short.vector.shape == (16,)
    with pytest.raises(ValueError):
        engine.embed("")
