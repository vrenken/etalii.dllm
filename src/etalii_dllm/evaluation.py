"""Reproducible evaluation (``dllm eval``): perplexity and multiple-choice accuracy with the same bits everywhere.

Scores come from the model's log-likelihoods, the way lm-evaluation-harness computes them, but every number is a
pure function of the weights and the task file: logits from the decoder's fixed-order kernels, log-probabilities from
the ``log_softmax`` kernel, sums with ``numerics.sum_`` (index order, double accumulator) and ``exp`` from the
portable kernels. Each result carries a fingerprint over every per-token log-probability, so two machines, devices,
thread counts or engine versions can be compared bit for bit with one line (``docs/evaluation.md``).

A task file is JSON lines, one item per line:

- ``{"context": "...", "choices": ["...", ...], "answer": 0}``: multiple choice. Each choice is scored as a
  continuation of the context; the prediction is the most likely choice (``accuracy``) and the most likely per byte
  of the choice (``accuracy_norm``), ties going to the first.
- ``{"text": "..."}``: perplexity over the text. A plain ``.txt`` file is one such item.

Every sequence starts with the tokenizer's begin-of-sequence token (its end-of-sequence token when it has none), so
the first token of a text is scored too. A text longer than ``max_length`` is scored in consecutive windows, each
starting from the token before its first target.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from etalii_dllm import _kernels
from etalii_dllm.numerics import argmax, fingerprint, log_softmax, sum_

if TYPE_CHECKING:
    from etalii_dllm.engine import DllmEngine

DEFAULT_MAX_LENGTH = 1024
"""The longest window a perplexity text is scored in, unless the model's context is shorter."""


class EvaluationError(ValueError):
    """A task file that cannot be evaluated."""


@dataclass(frozen=True)
class Scored:
    """The log-probabilities of a continuation's tokens, and whether each was the greedy choice."""

    logprobs: np.ndarray
    """float32, one per scored token."""
    greedy: bool

    @property
    def log_likelihood(self) -> float:
        return sum_(self.logprobs) if len(self.logprobs) else 0.0


def _start_token(engine: DllmEngine) -> int:
    begin = getattr(engine.tokenizer, "begin_of_sequence", -1)
    return begin if begin is not None and begin >= 0 else engine.tokenizer.end_of_sequence


def _rows(model: Any, tokens: Sequence[int], count: int) -> np.ndarray:
    """Next-token logits after each of the last ``count`` tokens."""
    if hasattr(model, "forward_cached_last"):
        return model.forward_cached_last(tokens, model.new_cache(), count)
    return np.stack([model.forward(tokens[: len(tokens) - count + 1 + i]) for i in range(count)])


def score(engine: DllmEngine, context: Sequence[int], continuation: Sequence[int]) -> Scored:
    """Log-probabilities of ``continuation`` after ``context`` (which must not be empty)."""
    if not context:
        raise EvaluationError("a continuation needs at least one token of context")
    if not continuation:
        return Scored(np.zeros(0, dtype=np.float32), True)
    tokens = [*context, *continuation[:-1]]
    rows = _rows(engine.model, tokens, len(continuation))
    logprobs = np.empty(len(continuation), dtype=np.float32)
    greedy = True
    for i, target in enumerate(continuation):
        logprobs[i] = log_softmax(rows[i])[target]
        greedy = greedy and argmax(rows[i]) == target
    return Scored(logprobs, greedy)


def _split_whitespace(context: str, continuation: str) -> tuple[str, str]:
    """Moves trailing spaces of the context to the continuation, as lm-evaluation-harness does, so ``"Q: A"`` +
    ``" B"`` and ``"Q: A "`` + ``"B"`` tokenize alike."""
    stripped = context.rstrip(" ")
    return stripped, context[len(stripped) :] + continuation


def _choice(engine: DllmEngine, context: str, choice: str) -> Scored:
    context, choice = _split_whitespace(context, choice)
    start = [_start_token(engine), *engine.tokenizer.encode(context)]
    return score(engine, start, engine.tokenizer.encode(choice))


def _text(engine: DllmEngine, text: str, max_length: int) -> Scored:
    tokens = [_start_token(engine), *engine.tokenizer.encode(text)]
    parts: list[np.ndarray] = []
    greedy = True
    step = max_length - 1
    for first in range(1, len(tokens), step):
        window = tokens[first - 1 : first + step]
        scored = score(engine, window[:1], window[1:])
        parts.append(scored.logprobs)
        greedy = greedy and scored.greedy
    return Scored(np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32), greedy)


def _float(value: float) -> str:
    """An exact, portable spelling of a double for fingerprints."""
    return float(value).hex()


def read_task(path: str | Path) -> list[dict[str, Any]]:
    """The items of a task file: JSON lines, or a ``.txt`` file as one perplexity text."""
    path = Path(path)
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise EvaluationError(f"{path}: {error.strerror or error}") from None
    if path.suffix.lower() != ".jsonl":
        return [{"text": content}]
    items = []
    for number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationError(f"{path}:{number}: {error.msg}") from None
        if not isinstance(item, dict):
            raise EvaluationError(f"{path}:{number}: expected a JSON object")
        items.append(item)
    if not items:
        raise EvaluationError(f"{path}: no items")
    return items


def _kind(items: Sequence[Mapping[str, Any]]) -> str:
    kinds = set()
    for number, item in enumerate(items, 1):
        if "text" in item and isinstance(item["text"], str):
            kinds.add("perplexity")
            continue
        choices, answer = item.get("choices"), item.get("answer")
        if (
            isinstance(item.get("context"), str)
            and isinstance(choices, list)
            and choices
            and all(isinstance(c, str) for c in choices)
            and isinstance(answer, int)
            and not isinstance(answer, bool)
            and 0 <= answer < len(choices)
        ):
            kinds.add("multiple_choice")
            continue
        raise EvaluationError(
            f"item {number}: expected {{'text'}} or {{'context', 'choices', 'answer'}} with a valid answer index"
        )
    if len(kinds) > 1:
        raise EvaluationError("a task mixes perplexity texts and multiple-choice items; split it into two files")
    return kinds.pop()


def evaluate(
    engine: DllmEngine,
    items: Sequence[Mapping[str, Any]],
    *,
    task: str = "task",
    max_length: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Scores ``items`` (see :func:`read_task`) and returns the report as JSON-ready data."""
    from etalii_dllm import __version__

    kind = _kind(items)
    context_length = getattr(getattr(engine.model, "config", None), "context_length", 0) or DEFAULT_MAX_LENGTH
    max_length = max_length or min(DEFAULT_MAX_LENGTH, context_length)
    if max_length < 2:
        raise EvaluationError("max_length must be at least 2")
    report: dict[str, Any] = {
        "task": task,
        "kind": kind,
        "engine": __version__,
        "model": engine.model.id,
        "system_fingerprint": engine.system_fingerprint,
        "items": len(items),
    }
    rows: list[dict[str, Any]] = []
    if kind == "perplexity":
        totals, counts, sizes = [], 0, 0
        for index, item in enumerate(items):
            scored = _text(engine, item["text"], max_length)
            totals.append(scored.log_likelihood)
            counts += len(scored.logprobs)
            sizes += len(item["text"].encode("utf-8"))
            rows.append(
                {
                    "index": index,
                    "tokens": len(scored.logprobs),
                    "log_likelihood": scored.log_likelihood,
                    "logprobs": fingerprint(scored.logprobs),
                }
            )
            if progress:
                progress(index + 1, len(items))
        total = _sum_doubles(totals)
        report.update(
            tokens=counts,
            bytes=sizes,
            log_likelihood=total,
            perplexity=_kernels.exp(-total / counts) if counts else None,
            bits_per_byte=(-total / sizes) / _kernels.log(2.0) if sizes else None,
        )
    else:
        correct, correct_norm = 0, 0
        for index, item in enumerate(items):
            scores = [_choice(engine, item["context"], choice) for choice in item["choices"]]
            likelihoods = [s.log_likelihood for s in scores]
            normalised = [
                ll / max(len(c.encode("utf-8")), 1) for ll, c in zip(likelihoods, item["choices"], strict=True)
            ]
            prediction = _best(likelihoods)
            prediction_norm = _best(normalised)
            correct += prediction == item["answer"]
            correct_norm += prediction_norm == item["answer"]
            rows.append(
                {
                    "index": index,
                    "answer": item["answer"],
                    "prediction": prediction,
                    "prediction_norm": prediction_norm,
                    "log_likelihoods": likelihoods,
                    "greedy": [s.greedy for s in scores],
                    "logprobs": [fingerprint(s.logprobs) for s in scores],
                }
            )
            if progress:
                progress(index + 1, len(items))
        report.update(accuracy=correct / len(items), accuracy_norm=correct_norm / len(items))
    report["fingerprint"] = hashlib.sha256(_canonical(rows).encode()).hexdigest()
    report["results"] = rows
    return report


def _best(values: Sequence[float]) -> int:
    """The index of the largest value, the first on a tie (a total order)."""
    best = 0
    for index, value in enumerate(values):
        if value > values[best]:
            best = index
    return best


def _sum_doubles(values: Sequence[float]) -> float:
    """Sum in index order (Python floats are IEEE doubles, so this is the same everywhere)."""
    total = 0.0
    for value in values:
        total += value
    return total


def _canonical(rows: Sequence[Mapping[str, Any]]) -> str:
    def exact(value: Any) -> Any:
        if isinstance(value, float):
            return _float(value)
        if isinstance(value, list):
            return [exact(v) for v in value]
        if isinstance(value, dict):
            return {k: exact(v) for k, v in value.items()}
        return value

    return json.dumps(exact(list(rows)), sort_keys=True, separators=(",", ":"))
