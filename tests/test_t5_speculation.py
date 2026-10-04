"""Fast exact decoding of T5 text-to-text models (#412-#415): several answer tokens per decoder pass with the bits of
one-at-a-time steps, speculative decoding by prompt lookup or a T5 draft model, and a prompt cache that reuses the
encoder states of the same source. None of them changes a bit of the output."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_t5 import TEXTS
from test_t5_decoding import END, build, source_of
from test_t5_generation import FLAN

from etalii_dllm.engine import ChatRequest, DllmEngine
from etalii_dllm.generation import Generator
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.seq2seq import DECODER_START, TextToTextCache
from etalii_dllm.speculative import DraftModel

pytest.importorskip("tokenizers")
pytest.importorskip("sentencepiece")

SAMPLED = SamplingOptions(temperature=0.9, seed=11)
COPY = "copy the words: alpha beta gamma delta alpha beta gamma delta alpha beta"
"""A repetitive source, so that prompt lookup finds drafts."""


@pytest.fixture(scope="module")
def flan(tmp_path_factory):
    return build(tmp_path_factory.mktemp("flan-speculation"), **FLAN)


@pytest.fixture(scope="module")
def t5(tmp_path_factory):
    return build(tmp_path_factory.mktemp("t5-speculation"))


@pytest.fixture(scope="module")
def ending(tmp_path_factory):
    """A tiny T5 whose ``</s>`` is likely, so that answers (and drafts) end."""
    return build(tmp_path_factory.mktemp("t5-ending"), 1.5, **FLAN)


def path_of(model) -> str:
    return str(model[0].parent / "model.dllm")


# Several answer tokens per decoder pass (#412)


def test_forward_cached_last_gives_the_one_at_a_time_bits(t5, flan):
    for _, engine in (t5, flan):
        model = engine.model
        source = source_of(engine, TEXTS[0])
        tokens = [*source, 5, 7, 9, 11, 3]
        cache = model.new_cache()
        rows = model.forward_cached_last(tokens, cache, 6)
        assert rows.shape == (6, model.vocabulary_size)
        for i in range(6):
            assert rows[i].tobytes() == model.forward(tokens[: len(source) + i]).tobytes()
        assert cache.tokens == [DECODER_START, 5, 7, 9, 11, 3]
        # Every position cached: the asked-for ones are read again.
        assert model.forward_cached_last(tokens, cache, 3).tobytes() == rows[3:].tobytes()
        # A diverging answer keeps the shared prefix of the cache.
        other = [*source, 5, 7, 2, 2]
        again = model.forward_cached_last(other, cache, 3)
        for i in range(3):
            assert again[i].tobytes() == model.forward(other[: len(source) + 2 + i]).tobytes()
        assert cache.tokens == [DECODER_START, 5, 7, 2, 2]
        assert model.forward_cached(tokens, cache).tobytes() == rows[-1].tobytes()
        with pytest.raises(ValueError, match="count"):
            model.forward_cached_last(tokens, cache, 7)
        with pytest.raises(ValueError, match="count"):
            model.forward_cached_last(tokens, cache, 0)


def test_truncation_keeps_exported_copies(flan):
    _, engine = flan
    model = engine.model
    source = source_of(engine, TEXTS[1])
    cache = model.new_cache()
    model.forward_cached([*source, 4, 8, 15], cache)
    copy = model.new_cache()
    copy.restore(*cache.export())
    expected = model.forward([*source, 4, 8, 15, 16])
    model.forward_cached([*source, 23], cache)  # truncates the original to the start token
    assert model.forward_cached([*source, 4, 8, 15, 16], copy).tobytes() == expected.tobytes()


# Speculative decoding (#413)


def generations(engine: DllmEngine, speculate: int, draft_model=None, prompt: str = COPY, max_tokens: int = 24):
    generator = Generator(engine.model, engine.tokenizer, engine._generator.stop_tokens, 0, speculate, draft_model)
    results, drafted, accepted = [], 0, 0
    for options in (SamplingOptions(), SAMPLED):
        generation = generator.stream(prompt, max_tokens, options, top_logprobs=2, new_text=True)
        steps = list(generation)
        results.append((steps, generation.result()))
        drafted += generation.drafted_tokens
        accepted += generation.accepted_tokens
    return results, drafted, accepted


@pytest.mark.parametrize("speculate", [1, 3, 8])
def test_prompt_lookup_never_changes_the_output(flan, speculate):
    _, engine = flan
    accepted = 0
    for prompt in (COPY, TEXTS[2]):
        plain, _, _ = generations(engine, 0, prompt=prompt, max_tokens=40)
        fast, _, kept = generations(engine, speculate, prompt=prompt, max_tokens=40)
        assert fast == plain
        accepted += kept
    assert accepted > 0  # the answer repeats itself, so some drafts are right


def test_a_text_to_text_draft_model_that_agrees_is_kept(flan, t5):
    _, engine = flan
    plain, _, _ = generations(engine, 0)
    drafted, count, accepted = generations(engine, 4, draft_model=engine.model)  # the model drafts for itself
    assert drafted == plain
    greedy = len(plain[0][1].tokens)
    assert count > 0 and accepted >= greedy - greedy // 5 - 1  # greedy drafts are all right, bar the bonus tokens
    other, _, _ = generations(engine, 4, draft_model=t5[1].model)  # another model's drafts: same output
    assert other == plain


def test_drafts_stop_before_the_end_of_source(ending):
    _, engine = ending
    model = engine.model
    source = source_of(engine, TEXTS[0])
    greedy = engine.complete(TEXTS[0], 30, SamplingOptions())
    assert greedy.finish_reason == "stop"  # the answer ends with </s>
    draft = DraftModel(model).propose(source, 40)
    assert END not in draft and len(draft) == len(greedy.tokens)

    class Ender:  # a drafter that proposes </s> mid-draft
        def propose(self, context, count):
            return [5, END, 7][:count]

    generator = Generator(model, engine.tokenizer, engine._generator.stop_tokens, 0, 3)
    generator.new_drafter = Ender  # type: ignore[method-assign]
    generation = generator.stream(TEXTS[0], 30, SamplingOptions(), new_text=True)
    assert generation.result().tokens == greedy.tokens
    assert generation.drafted_tokens > 0


def test_engine_speculation_and_draft_models(flan, t5):
    plain = DllmEngine.from_model_file(path_of(flan), prompt_cache=0)
    fast = DllmEngine.from_model_file(path_of(flan), speculate=5, prompt_cache=0)
    drafted = DllmEngine.from_model_file(path_of(flan), draft_model=path_of(t5), prompt_cache=0)
    assert fast._generator.speculate == 5 and drafted._generator.speculate == 8
    for text in (COPY, *TEXTS[:3]):
        for options in (SamplingOptions(), SAMPLED):
            request = ChatRequest((), 20, options, prompt=text, top_logprobs=1)
            expected = plain.chat_completion(request)
            assert fast.chat_completion(request) == expected
            assert drafted.chat_completion(request) == expected


# The prompt cache (#414)


def test_cache_reuse_rules():
    source = (3, 4, 1)
    cache = TextToTextCache(source=source, tokens=[DECODER_START, 9, 8])
    assert cache.reusable([3, 4, 1]) == 3  # the whole source: the encoder is not run again
    assert cache.reusable([3, 4, 1, 9, 8, 7]) == 5
    assert cache.reusable([3, 4, 1, 9, 8]) == 4  # the last answer token is read again
    assert cache.reusable([3, 5, 1]) == 0  # another source never reuses an encoder pass
    assert cache.reusable([3, 4]) == 0
    assert TextToTextCache().reusable([3, 4, 1]) == 0
    shorter = TextToTextCache(source=source, tokens=[DECODER_START, 9])
    assert cache.covers(shorter) and not shorter.covers(cache)
    assert not cache.covers(TextToTextCache(source=(3, 1), tokens=[DECODER_START]))
    assert not cache.covers(object())


@pytest.mark.parametrize("options", [SamplingOptions(), SAMPLED])
def test_cache_hits_give_the_bits_of_a_cold_run(flan, options):
    cached = DllmEngine.from_model_file(path_of(flan), prompt_cache=2)
    cold = DllmEngine.from_model_file(path_of(flan), prompt_cache=0)
    requests = [ChatRequest((), 12, options, prompt=text, top_logprobs=2) for text in (*TEXTS[:3], TEXTS[2])]
    warm = [cached.chat_completion(r) for r in requests]
    expected = [cold.chat_completion(r) for r in requests]
    assert [r.cached_tokens for r in expected] == [0, 0, 0, 0]
    assert [r.cached_tokens for r in warm[:3]] == [0, 0, 0]
    assert warm[3].cached_tokens == warm[3].prompt_tokens  # the same source: its encoder pass is reused
    assert [replace(r, cached_tokens=0) for r in warm] == expected
    assert len(cached._generator.prompt_cache) == 2
    fast = DllmEngine.from_model_file(path_of(flan), prompt_cache=2, speculate=4)
    both = [fast.chat_completion(r) for r in (*requests, requests[1])]
    assert [replace(r, cached_tokens=0) for r in both] == [*expected, expected[1]]
    assert both[-1].cached_tokens == both[-1].prompt_tokens


def test_persistent_cache_is_refused(flan, tmp_path):
    with pytest.raises(ValueError, match="prompt_cache_dir"):
        DllmEngine.from_model_file(path_of(flan), prompt_cache_dir=tmp_path / "kv")
    _, engine = flan
    with pytest.raises(ValueError, match="in memory"):
        Generator(engine.model, engine.tokenizer, (), 2, prompt_cache_dir=str(tmp_path / "kv"))


def test_answer_scores_in_one_pass(flan):
    _, engine = flan
    model = engine.model
    source, answer = source_of(engine, TEXTS[1]), [6, 2, 9, END]
    rows = model.answer_logits(source, answer)
    for i in range(len(answer)):
        assert np.array_equal(rows[i], model.forward([*source, *answer[:i]]))
