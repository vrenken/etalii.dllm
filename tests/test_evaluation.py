"""Reproducible evaluation (``dllm eval``): log-likelihood scores with exact golden fingerprints, the same bits for
a prefill and token-by-token decoding, on every thread count, and clear errors for bad task files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from golden_values import EVAL_FINGERPRINTS
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm import _kernels, evaluation, numerics
from etalii_dllm.cli import main
from etalii_dllm.engine import DllmEngine, default_engine
from etalii_dllm.numerics import log_softmax, sum_

DATA = Path(__file__).parent / "data"
TASKS = {"multiple_choice": DATA / "eval-tiny.jsonl", "perplexity": DATA / "eval-text.jsonl"}


@pytest.fixture
def placeholder():
    return DllmEngine.create_default()


@pytest.mark.parametrize("kind", sorted(TASKS))
def test_placeholder_scores_are_golden(placeholder, kind):
    report = evaluation.evaluate(placeholder, evaluation.read_task(TASKS[kind]))
    assert report["kind"] == kind
    assert report["fingerprint"] == EVAL_FINGERPRINTS[f"bigram_{kind}"]


@pytest.mark.parametrize("kind", sorted(TASKS))
def test_transformer_scores_are_golden_on_every_thread_count(model_path, kind):  # noqa: F811
    engine = DllmEngine.from_model_file(model_path, prompt_cache=0)
    items = evaluation.read_task(TASKS[kind])
    original = numerics.threads()
    try:
        reports = []
        for count in (1, 3):
            numerics.set_threads(count)
            reports.append(evaluation.evaluate(engine, items))
    finally:
        numerics.set_threads(original)
    assert reports[0] == reports[1]
    assert reports[0]["fingerprint"] == EVAL_FINGERPRINTS[f"tiny_{kind}"]


def test_scores_equal_token_by_token_decoding(model_path):  # noqa: F811
    engine = DllmEngine.from_model_file(model_path, prompt_cache=0)
    context, continuation = [2, 5, 9, 11], [7, 3, 8]
    scored = evaluation.score(engine, context, continuation)
    tokens = [*context, *continuation]
    expected = [log_softmax(engine.model.forward(tokens[:i]))[tokens[i]] for i in range(len(context), len(tokens))]
    assert scored.logprobs.tobytes() == np.asarray(expected, dtype=np.float32).tobytes()
    assert scored.log_likelihood == sum_(expected)


def test_multiple_choice_report(placeholder):
    items = [
        {"context": "A", "choices": [" b", " c"], "answer": 1},
        {"context": "A ", "choices": ["b", "c"], "answer": 0},
    ]
    report = evaluation.evaluate(placeholder, items, task="mine")
    first, second = report["results"]
    # Trailing context spaces move to the choices, so both items score the same continuations.
    assert first["log_likelihoods"] == second["log_likelihoods"]
    assert report["accuracy"] == 0.5 and report["task"] == "mine" and report["items"] == 2
    best = max(range(2), key=lambda i: (first["log_likelihoods"][i], -i))
    assert first["prediction"] == best


def test_perplexity_windows(placeholder):
    text = "abcdefghij"
    whole = evaluation.evaluate(placeholder, [{"text": text}])
    windowed = evaluation.evaluate(placeholder, [{"text": text}], max_length=4)
    # The bigram model only looks at the previous token, so windows starting one token early change nothing.
    assert whole["log_likelihood"] == windowed["log_likelihood"] and whole["tokens"] == len(text)
    assert whole["perplexity"] == _kernels.exp(-whole["log_likelihood"] / whole["tokens"])
    assert whole["bits_per_byte"] == -whole["log_likelihood"] / len(text) / _kernels.log(2.0)
    empty = evaluation.evaluate(placeholder, [{"text": ""}])
    assert empty["tokens"] == 0 and empty["perplexity"] is None and empty["bits_per_byte"] is None
    with pytest.raises(evaluation.EvaluationError, match="at least 2"):
        evaluation.evaluate(placeholder, [{"text": text}], max_length=1)


def test_score_edge_cases(placeholder):
    assert evaluation.score(placeholder, [1], []).log_likelihood == 0.0
    with pytest.raises(evaluation.EvaluationError, match="context"):
        evaluation.score(placeholder, [], [1])


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (["[1]"], "expected a JSON object"),
        (["{"], "eval.jsonl:1"),
        ([""], "no items"),
        (['{"context": "a", "choices": ["b"], "answer": 1}'], "valid answer index"),
        (['{"context": "a", "choices": ["b"], "answer": true}'], "valid answer index"),
        (['{"text": "a"}', '{"context": "a", "choices": ["b"], "answer": 0}'], "mixes"),
    ],
)
def test_bad_task_files(placeholder, tmp_path, lines, message):
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(evaluation.EvaluationError, match=message):
        evaluation.evaluate(placeholder, evaluation.read_task(path))


def test_text_files_and_missing_files(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text("Once upon a time\n\nthe end", encoding="utf-8")
    assert evaluation.read_task(path) == [{"text": "Once upon a time\n\nthe end"}]
    with pytest.raises(evaluation.EvaluationError, match=r"missing\.txt"):
        evaluation.read_task(tmp_path / "missing.txt")


def test_eval_command(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DLLM_MODEL", raising=False)
    default_engine.cache_clear()
    output = tmp_path / "report.json"
    assert main(["eval", str(TASKS["multiple_choice"]), "-o", str(output)]) == 0
    captured = capsys.readouterr()
    assert f"fingerprint:        {EVAL_FINGERPRINTS['bigram_multiple_choice']}" in captured.out
    assert "item 5/5" in captured.err
    assert json.loads(output.read_text(encoding="utf-8"))["task"] == "eval-tiny.jsonl"
    assert main(["eval", str(TASKS["perplexity"]), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["fingerprint"] == EVAL_FINGERPRINTS["bigram_perplexity"]
    assert main(["eval", str(tmp_path / "missing.jsonl")]) == 1
    assert capsys.readouterr().err.startswith("dllm eval: ")
    default_engine.cache_clear()
