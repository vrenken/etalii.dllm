"""Speculative decoding: drafted tokens are checked in one pass and kept only when they are the tokens plain decoding
chooses, so every output (tokens, text, logprobs, finish reason) is identical with and without it."""

from __future__ import annotations

import json
import sys

import pytest
from test_engine_import import model_path  # noqa: F401 - fixture
from test_tools import WEATHER

from etalii_dllm import engine as engine_module
from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import ChatRequest, DllmEngine, ResponseFormat, configured_speculate, default_engine
from etalii_dllm.generation import Generator
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.speculative import DEFAULT_DRAFT_TOKENS, DraftModel, PromptLookup
from etalii_dllm.tools import ToolChoice

pytest.importorskip("tokenizers")

SAMPLED = SamplingOptions(temperature=0.9, seed=11)
REPETITIVE = [5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6]


def test_prompt_lookup_drafts_what_followed_the_last_occurrence():
    lookup = PromptLookup()
    assert lookup.propose(REPETITIVE, 4) == [7, 8, 9, 5]  # "5 6" last seen at index 5, then "7 8 9 5"
    assert lookup.propose([*REPETITIVE, 7], 2) == [8, 9]  # the index grows with the sequence
    assert lookup.propose([1, 2, 3], 4) == []  # a shorter, different sequence starts over: no repeat
    assert lookup.propose([1, 2, 3, 1, 2], 0) == []
    assert lookup.propose([1, 2, 3, 1, 2], 9) == [3, 1, 2]  # only what exists
    assert PromptLookup(min_ngram=1).propose([4, 1, 9, 1], 2) == [9, 1]  # single-token matches when allowed
    assert PromptLookup().propose([4, 1, 9, 1], 2) == []
    with pytest.raises(ValueError):
        PromptLookup(max_ngram=1, min_ngram=2)


def test_forward_cached_last_gives_the_one_at_a_time_bits(model_path):  # noqa: F811
    model = DllmEngine.from_model_file(model_path).model
    tokens = [3, 17, 42, 9, 9, 31, 8]
    rows = model.forward_cached_last(tokens, model.new_cache(), 4)
    for i in range(4):
        alone = model.forward(tokens[: len(tokens) - 3 + i])
        assert rows[i].tobytes() == alone.tobytes()
    cache = model.new_cache()
    model.forward_cached(tokens, cache)  # every position already cached: the last ones are recomputed
    assert model.forward_cached_last(tokens, cache, 3).tobytes() == rows[1:].tobytes()
    with pytest.raises(ValueError):
        model.forward_cached_last(tokens, cache, 8)


def _generations(engine: DllmEngine, speculate: int, draft_model=None, **kwargs):
    generator = Generator(engine.model, engine.tokenizer, engine._generator.stop_tokens, 0, speculate, draft_model)
    prompt = [*REPETITIVE, *REPETITIVE]
    results, accepted = [], 0
    for options in (SamplingOptions(), SAMPLED):
        generation = generator.stream(prompt, 40, options, top_logprobs=2, **kwargs)
        steps = list(generation)
        results.append((steps, generation.result()))
        accepted += generation.accepted_tokens
    return results, accepted


@pytest.mark.parametrize("speculate", [1, 3, DEFAULT_DRAFT_TOKENS])
def test_speculation_never_changes_the_output(model_path, speculate):  # noqa: F811
    engine = DllmEngine.from_model_file(model_path, prompt_cache=0)
    plain, _ = _generations(engine, 0)
    fast, accepted = _generations(engine, speculate)
    assert fast == plain
    assert accepted > 0  # the repetitive prompt makes some drafts right
    stopped, _ = _generations(engine, speculate, stop=["e"])
    assert stopped == _generations(engine, 0, stop=["e"])[0]


def test_a_draft_model_that_agrees_is_kept(model_path):  # noqa: F811
    engine = DllmEngine.from_model_file(model_path, prompt_cache=0)
    plain, _ = _generations(engine, 0)
    drafted, accepted = _generations(engine, 4, draft_model=engine.model)  # the model drafts for itself
    assert drafted == plain
    greedy_tokens = len(plain[0][1].tokens)
    assert accepted >= greedy_tokens - greedy_tokens // 5 - 1  # greedy drafts are all right, bar the bonus tokens
    assert DraftModel(engine.model).propose([1, 2, 3], 0) == []


def _requests() -> list[ChatRequest]:
    messages = [ChatMessage("user", "Say hello hello hello hello hello")]
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    return [
        ChatRequest(messages, 24, SAMPLED, top_logprobs=1),
        ChatRequest(messages, 24, response_format=ResponseFormat("json_schema", schema)),
        ChatRequest(messages, 24, SAMPLED, tools=[WEATHER], tool_choice=ToolChoice("required")),
        ChatRequest(messages, 24, stop=["l"]),
    ]


def test_engine_speculation_with_grammars_tools_and_stops(model_path, tmp_path):  # noqa: F811
    plain = DllmEngine.from_model_file(model_path)
    fast = DllmEngine.from_model_file(model_path, speculate=5)
    drafted = DllmEngine.from_model_file(model_path, draft_model=model_path)
    assert fast.system_fingerprint == drafted.system_fingerprint == plain.system_fingerprint
    assert drafted._generator.speculate == DEFAULT_DRAFT_TOKENS
    for request in _requests():
        expected = plain.chat_completion(request)
        assert fast.chat_completion(request) == expected
        assert drafted.chat_completion(request) == expected
        assert list(fast.chat_stream(request)) == list(plain.chat_stream(request))


def test_a_draft_model_needs_the_same_tokenizer(model_path, tmp_path):  # noqa: F811
    from etalii_dllm.modelfile import ModelFile

    file = ModelFile(model_path)
    header = json.loads(json.dumps(file.tokenizer))
    header["tokenizer_json"]["model"]["vocab"] = {"a": 0}
    assert engine_module._vocabulary(header) != engine_module._vocabulary(file.tokenizer)
    assert engine_module._vocabulary(None) is None
    gguf = {"format": "gguf", "tokenizer.ggml.tokens": ["a"], "general.name": "x"}
    assert engine_module._vocabulary(gguf) == {"tokenizer.ggml.tokens": ["a"]}
    other = tmp_path / "other.dllm"
    other.write_bytes(model_path.read_bytes())
    original = engine_module._vocabulary
    try:
        engine_module._vocabulary = lambda tokenizer: id(tokenizer)  # every header differs
        with pytest.raises(ValueError, match="tokenizer differs"):
            DllmEngine.from_model_file(model_path, draft_model=other)
    finally:
        engine_module._vocabulary = original


def test_models_without_multi_position_logits_never_speculate():
    engine = DllmEngine.create_default()
    generator = Generator(engine.model, engine.tokenizer, speculate=4)
    assert generator.speculate == 0 and generator.new_drafter() is None
    with pytest.raises(ValueError):
        Generator(engine.model, engine.tokenizer, speculate=-1)


def test_speculate_configuration(model_path, monkeypatch):  # noqa: F811
    monkeypatch.delenv(engine_module.SPECULATE_ENVIRONMENT_VARIABLE, raising=False)
    assert configured_speculate() is None
    monkeypatch.setenv(engine_module.SPECULATE_ENVIRONMENT_VARIABLE, "x")
    with pytest.raises(ValueError, match="DLLM_SPECULATE"):
        configured_speculate()
    from etalii_dllm import cli

    try:
        monkeypatch.setattr(sys, "argv", ["dllm"])
        code = cli.main(["--model", str(model_path), "--speculate", "3", "--draft-model", str(model_path), "info"])
        assert code in (0, None)
        assert configured_speculate() == 3
        engine = default_engine()
        assert engine._generator.speculate == 3 and engine._generator.draft_model is not None
    finally:
        for name in (engine_module.SPECULATE_ENVIRONMENT_VARIABLE, engine_module.DRAFT_MODEL_ENVIRONMENT_VARIABLE):
            monkeypatch.delenv(name, raising=False)
            import os

            os.environ.pop(name, None)
        default_engine.cache_clear()
