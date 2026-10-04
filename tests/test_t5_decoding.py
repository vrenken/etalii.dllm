"""Beam search, scoring, guidance and batching for T5 text-to-text models (#392-#395): beam search token for token
against transformers' ``generate(num_beams=...)``, scores of a target given its source against transformers' logits,
negative sources, contrast and ensembles of text-to-text models, batched decoder steps with the bits of lone ones,
receipts, the front ends and goldens."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest
from test_t5 import TEXTS, write_t5_checkpoint
from test_t5_generation import FLAN, write_checkpoint

from etalii_dllm import beam, evaluation, guidance, scoring
from etalii_dllm.engine import ChatMessage, ChatRequest, DllmEngine
from etalii_dllm.importing import import_model
from etalii_dllm.numerics import log_softmax
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.seq2seq import TextToTextCache

pytest.importorskip("tokenizers")
pytest.importorskip("sentencepiece")

END = 1
"""``</s>``, which ends every source."""


def build(directory: Path, end_bump: float = 0.0, **changes) -> tuple[Path, DllmEngine]:
    """A tiny T5 (``changes`` as in test_t5_generation); ``end_bump`` makes ``</s>`` likelier, so that beams finish."""
    from safetensors.numpy import load_file, save_file

    write_checkpoint(directory / "checkpoint", **changes)
    if end_bump:
        path = directory / "checkpoint" / "model.safetensors"
        weights = load_file(path)
        head = "shared.weight" if changes.get("tie_word_embeddings", True) else "lm_head.weight"
        row = weights[head].copy()
        row[END] += np.float32(end_bump) * np.sign(row[END])
        weights[head] = row
        save_file(weights, path, {"format": "pt"})
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-t5-decoding")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


@pytest.fixture(scope="module")
def t5(tmp_path_factory) -> tuple[Path, DllmEngine]:
    return build(tmp_path_factory.mktemp("t5-decoding"))


@pytest.fixture(scope="module")
def flan(tmp_path_factory) -> tuple[Path, DllmEngine]:
    return build(tmp_path_factory.mktemp("flan-decoding"), **FLAN)


def source_of(engine: DllmEngine, text: str) -> list[int]:
    return [*engine.tokenizer.encode(text), END]


# Beam search (#392)


@pytest.mark.parametrize("variant", ["t5", "flan"])
def test_beam_search_matches_transformers(variant, tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    changes = FLAN if variant == "flan" else {}
    finished = set()
    for bump in (0.0, 1.5):
        checkpoint, engine = build(tmp_path / str(bump), bump, **changes)
        model = transformers.T5ForConditionalGeneration.from_pretrained(checkpoint).eval()
        for text in TEXTS[:3]:
            source = source_of(engine, text)
            for width, length_penalty in ((2, 1.0), (3, 0.0), (4, 2.0)):
                result = beam.search(engine, ChatRequest((), 10, prompt=text), width, 1, length_penalty)
                best = result.hypotheses[0]
                finished.add(best.finish_reason)
                with torch.no_grad():
                    output = model.generate(
                        torch.tensor([source]),
                        max_new_tokens=10,
                        num_beams=width,
                        early_stopping=True,
                        length_penalty=length_penalty,
                        do_sample=False,
                        use_cache=variant == "flan",  # a decoder deeper than the encoder breaks transformers' cache
                    )[0].tolist()
                expected = [t for t in output[1:] if t != 0]
                if expected and expected[-1] == END:
                    expected = expected[:-1]
                assert list(best.tokens) == expected, (bump, text, width, length_penalty)
                assert result.prompt_tokens == len(source)
    assert finished == {"stop", "length"}


def test_beam_search_receipts_and_front_ends(flan, capsys, cli_environment, tmp_path):
    from golden_values import T5_DECODING_FINGERPRINT

    from etalii_dllm.cli import main

    checkpoint, engine = flan
    request = ChatRequest([ChatMessage("user", TEXTS[1])], 8)
    result = beam.search(engine, request, 3, 2)
    assert len(result.hypotheses) == 2 and result.hypotheses[0].score >= result.hypotheses[1].score
    assert result.hypotheses[0].fingerprint == T5_DECODING_FINGERPRINT["beam"]
    receipt = beam.record(engine, request, result)
    assert beam.verify(engine, receipt).ok
    path = checkpoint.parent / "model.dllm"
    receipt_path = tmp_path / "beam.json"
    arguments = ["--model", str(path), "generate", "--prompt", TEXTS[1], "--max-tokens", "8", "--beams", "3"]
    assert main([*arguments, "--receipt", str(receipt_path)]) == 0
    assert beam.search(engine, ChatRequest((), 8, prompt=TEXTS[1]), 3).hypotheses[0].text in capsys.readouterr().out
    assert main(["--model", str(path), "replay", str(receipt_path)]) == 0


def test_cache_export_and_restore(flan):
    _, engine = flan
    model = engine.model
    source = source_of(engine, TEXTS[0])
    cache = model.new_cache()
    model.forward_cached([*source, 5, 9], cache)
    copy = TextToTextCache()
    copy.restore(*cache.export())
    after_copy = model.forward_cached([*source, 5, 9, 12], copy)
    assert after_copy.tobytes() == model.forward([*source, 5, 9, 12]).tobytes()
    assert len(cache.tokens) == 3 and len(copy.tokens) == 4  # the copy's appends leave the original alone
    assert model.forward_cached([*source, 5, 9, 30], cache).tobytes() == model.forward([*source, 5, 9, 30]).tobytes()
    with pytest.raises(ValueError, match="empty cache"):
        copy.restore(*cache.export())


# Scores of a target given its source (#393)


def test_scores_match_transformers(t5, flan):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    for checkpoint, engine in (t5, flan):
        model = transformers.T5ForConditionalGeneration.from_pretrained(checkpoint).eval()
        source, target = source_of(engine, TEXTS[0]), engine.tokenizer.encode(TEXTS[2])
        score = scoring.score_text(engine, TEXTS[2], 3, source=TEXTS[0])
        assert [t.token for t in score.tokens] == target and score.scored == len(target)
        for i, entry in enumerate(score.tokens):
            row = engine.model.forward([*source, *target[:i]])
            assert entry.logprob == float(log_softmax(row)[target[i]])
            assert [a.token for a in entry.top] == sorted(range(len(row)), key=lambda j: (-row[j], j))[:3]
        with torch.no_grad():
            logits = model(
                input_ids=torch.tensor([source]), decoder_input_ids=torch.tensor([[0, *target[:-1]]])
            ).logits[0]
        expected = torch.log_softmax(logits.double(), -1)[range(len(target)), target].numpy()
        np.testing.assert_allclose([t.logprob for t in score.tokens], expected, rtol=1e-4, atol=1e-4)
        assert score.perplexity is not None and np.isfinite(score.log_likelihood)
    assert scoring.score_tokens(flan[1], [], source=[]).tokens == ()


def test_score_refusals_receipts_and_front_ends(flan, capsys, cli_environment, tmp_path):
    from golden_values import T5_DECODING_FINGERPRINT

    from etalii_dllm.cli import main

    checkpoint, engine = flan
    with pytest.raises(ValueError, match="answer to a source"):
        scoring.score_text(engine, TEXTS[2])
    with pytest.raises(ValueError, match="only text-to-text"):
        scoring.score_text(DllmEngine.create_default(), "hello", source="hi")
    with pytest.raises(ValueError, match="</s> only at its end"):
        engine.model.answer_logits(source_of(engine, TEXTS[0]), [END, 5])
    with pytest.raises(ValueError, match="out of range"):
        engine.model.answer_logits(source_of(engine, TEXTS[0]), [10**6])
    assert engine.model.answer_logits(source_of(engine, TEXTS[0]), []).shape == (0, engine.model.vocabulary_size)
    score = scoring.score_text(engine, TEXTS[2], source=TEXTS[0])
    assert score.fingerprint == T5_DECODING_FINGERPRINT["score"]
    receipt = scoring.record(engine, TEXTS[2], 0, score, TEXTS[0])
    assert receipt["source"] == TEXTS[0] and scoring.verify(engine, receipt).ok
    edited = {**receipt, "source": TEXTS[1]}
    assert not scoring.verify(engine, edited).ok
    text = tmp_path / "target.txt"
    text.write_text(TEXTS[2], encoding="utf-8")
    path = str(checkpoint.parent / "model.dllm")
    receipt_path = tmp_path / "score.json"
    arguments = ["--model", path, "score", str(text), "--source", TEXTS[0], "--json", "--receipt", str(receipt_path)]
    assert main(arguments) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["fingerprint"] == score.fingerprint and printed["scored"] == len(score.tokens)
    assert main(["--model", path, "replay", str(receipt_path)]) == 0
    assert main(["--model", path, "score", str(text)]) == 2
    assert "answer to a source" in capsys.readouterr().err


def test_evaluation_scores_continuations_as_answers(flan):
    _, engine = flan
    context, continuation = engine.tokenizer.encode(TEXTS[0]), engine.tokenizer.encode(TEXTS[1])
    scored = evaluation.score(engine, context, continuation)
    expected = scoring.score_tokens(engine, continuation, source=context)
    assert scored.logprobs.tolist() == [t.logprob for t in expected.tokens]


# Negative sources, contrast and ensembles (#394)


def greedy_guided(engine: DllmEngine, source: list[int], combine, max_tokens: int) -> list[int]:
    answer: list[int] = []
    for _ in range(max_tokens):
        logits = combine(engine.model.forward([*source, *answer]), answer)
        token = int(np.argmax(logits))
        if token in engine.stop_tokens:
            break
        answer.append(token)
    return answer


def test_negative_source(flan):
    from golden_values import T5_DECODING_FINGERPRINT

    _, engine = flan
    options = SamplingOptions(temperature=0.0, negative_prompt=TEXTS[3], guidance_scale=2.0)
    result = engine.complete(TEXTS[0], 8, options)
    negative = source_of(engine, TEXTS[3])

    def combine(logits, answer):
        return guidance.guided(logits, engine.model.forward([*negative, *answer]), 2.0)

    assert list(result.tokens) == greedy_guided(engine, source_of(engine, TEXTS[0]), combine, 8)
    assert result.tokens != engine.complete(TEXTS[0], 8, SamplingOptions(temperature=0.0)).tokens
    chat = ChatRequest([ChatMessage("user", TEXTS[0])], 8, options)
    content = "".join(getattr(event, "text", "") or "" for event in engine.chat_stream(chat))
    assert content == result.text
    assert result.fingerprint == T5_DECODING_FINGERPRINT["negative"]
    empty = engine.complete(TEXTS[0], 4, SamplingOptions(temperature=0.0, negative_prompt=""))
    assert len(empty.tokens) <= 4


def test_contrast_and_ensembles_of_text_to_text_models(t5, flan, tmp_path):
    _, small = t5
    checkpoint, engine = flan
    path, other = checkpoint.parent / "model.dllm", t5[0].parent / "model.dllm"
    contrasted = DllmEngine.from_model_file(path, contrast_model=other)
    options = SamplingOptions(temperature=0.0, contrast_beta=0.5, contrast_alpha=0.1)
    result = contrasted.complete(TEXTS[0], 8, options)
    source = source_of(engine, TEXTS[0])

    def contrast(logits, answer):
        return guidance.contrasted(logits, small.model.forward([*source, *answer]), 0.1, 0.5)

    assert list(result.tokens) == greedy_guided(engine, source, contrast, 8)
    assert contrasted.system_fingerprint != engine.system_fingerprint
    ensembled = DllmEngine.from_model_file(path, ensemble=[(other, 2.0)], ensemble_weight=1.0)
    together = ensembled.complete(TEXTS[0], 8, SamplingOptions(temperature=0.0))

    def ensemble(logits, answer):
        return guidance.ensembled([(logits, 1.0), (small.model.forward([*source, *answer]), 2.0)])

    assert list(together.tokens) == greedy_guided(engine, source, ensemble, 8)
    encoder = tmp_path / "encoder"
    write_t5_checkpoint(encoder / "checkpoint")
    import_model(encoder / "checkpoint", encoder / "model.dllm", repository="example/tiny-t5-encoder")
    with pytest.raises(ValueError, match="must be a text-to-text model"):
        DllmEngine.from_model_file(path, contrast_model=encoder / "model.dllm")
    with pytest.raises(ValueError, match="text-to-text"):
        DllmEngine.from_model_file(path, speculate=4)


# Batched decoder steps (#395)


def test_forward_batch_gives_the_bits_of_lone_steps(t5, flan):
    for _, engine in (t5, flan):
        model = engine.model
        sequences = [
            source_of(engine, TEXTS[0]),
            [*source_of(engine, TEXTS[1]), 7, 12, 30],
            [*source_of(engine, TEXTS[2]), 4],
            [*source_of(engine, TEXTS[0]), 9, 9],
        ]
        expected = [model.forward(tokens).tobytes() for tokens in sequences]
        caches = [model.new_cache() for _ in sequences]
        assert [row.tobytes() for row in model.forward_batch(sequences, caches)] == expected
        longer = [[*tokens, 3] for tokens in sequences]
        again = [model.forward(tokens).tobytes() for tokens in longer]
        assert [row.tobytes() for row in model.forward_batch(longer, caches)] == again
        assert [row.tobytes() for row in model.forward_batch(sequences)] == expected
        with pytest.raises(ValueError, match="one cache per sequence"):
            model.forward_batch(sequences, caches[:2])
        with pytest.raises(ValueError, match="distinct cache"):
            model.forward_batch(sequences[:2], [caches[0], caches[0]])
        with pytest.raises(ValueError, match="out of range"):
            model.forward_batch([[*sequences[0], 10**6]])


def test_concurrent_generations_share_batched_steps(flan):
    _, engine = flan
    options = [SamplingOptions(temperature=0.8, seed=seed) for seed in range(4)]
    alone = [engine.complete(TEXTS[i % len(TEXTS)], 10, o) for i, o in enumerate(options)]
    batcher = engine._generator.batcher
    assert batcher is not None  # a text-to-text model batches (forward_batch)
    batches = batcher.batches
    results: list = [None] * len(options)
    barrier = threading.Barrier(len(options))

    def run(i: int) -> None:
        barrier.wait()
        results[i] = engine.complete(TEXTS[i % len(TEXTS)], 10, options[i])

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(options))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert [r.tokens for r in results] == [r.tokens for r in alone]
    assert [r.fingerprint for r in results] == [r.fingerprint for r in alone]
    assert batcher.batches > batches
