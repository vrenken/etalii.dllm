"""Exact beam search (``beam`` on chat completions and completions, ``dllm generate --beams N``;
docs/api.md#beam-search).

Beam search keeps the ``width`` most likely partial answers and extends them together, with every step fixed:

- A hypothesis's log-likelihood is the sum, in double and in token order, of the float32 log-probabilities
  (``log_softmax``, the kernel) of its tokens, the stop token included when one ended it.
- Each step, every live hypothesis proposes its ``2 * width`` most likely next tokens (log-probability descending,
  then id ascending). The candidates are ranked by log-likelihood descending, then by token sequence ascending (a
  total order: no two candidates share a sequence), and taken in that order: a candidate whose token is a stop
  token, or whose text now contains a stop sequence, finishes when its rank is below ``width`` (else it is dropped);
  any other becomes one of the next step's live hypotheses, until there are ``width`` of them.
- The search ends when ``width`` hypotheses have finished, when no hypothesis is live, or when ``max_tokens`` (or the
  context window) is reached; then every live hypothesis finishes with ``length``.
- Finished hypotheses are ranked by their score ``log_likelihood / length ** length_penalty`` (``length`` counts the
  scored tokens; the power is ``exp(length_penalty * log(length))`` with the portable ``exp`` and ``log``) descending,
  then by token sequence ascending; the best ``n_best`` are the answers.

Temperature, top-k, top-p and seeds play no part. Every hypothesis keeps its own KV cache (a copy of its parent's,
whose rows are the bits a recompute gives), and the logits of all live hypotheses come from one batched pass, which
gives each the bits of a lone pass. So the same request gives the same answers on every machine. :func:`record`
makes a receipt that :func:`verify` (``dllm replay``) checks by searching again.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from etalii_dllm import _kernels, receipts
from etalii_dllm.generation import ContextLengthError, _complete_prefix
from etalii_dllm.numerics import fingerprint, log_softmax
from etalii_dllm.receipts import Verification, canonical_json
from etalii_dllm.sampling import _DEFAULTS

if TYPE_CHECKING:
    from etalii_dllm.engine import ChatRequest, DllmEngine

FORMAT = "dllm-beam/1"
"""The beam search receipt format; a reader refuses others."""

MAX_WIDTH = 16


def validate(width: int, n_best: int, length_penalty: float) -> None:
    if not 1 <= width <= MAX_WIDTH:
        raise ValueError(f"a beam search needs a width between 1 and {MAX_WIDTH}")
    if not 1 <= n_best <= width:
        raise ValueError("n_best must be between 1 and the beam width")
    if not np.isfinite(length_penalty):
        raise ValueError("the length penalty must be a finite number")


def score(log_likelihood: float, length: int, length_penalty: float) -> float:
    """``log_likelihood / length ** length_penalty``, the power from the portable ``exp`` and ``log``."""
    if length <= 0:
        return log_likelihood
    return log_likelihood / float(_kernels.exp(float(length_penalty) * float(_kernels.log(float(length)))))


@dataclass(frozen=True)
class Hypothesis:
    tokens: tuple[int, ...]
    """The answer's tokens (a stop token that ended it is not part of them)."""
    logprobs: tuple[float, ...]
    """The float32 log-probability of every scored token, the stop token included."""
    finish_reason: str
    log_likelihood: float
    score: float
    text: str = ""

    @property
    def fingerprint(self) -> str:
        """Hash of the answer's token ids, as :attr:`~etalii_dllm.generation.GenerationResult.fingerprint`."""
        return fingerprint(self.tokens, dtype="<i4")

    def to_json(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tokens": len(self.tokens),
            "finish_reason": self.finish_reason,
            "log_likelihood": self.log_likelihood,
            "score": self.score,
            "fingerprint": self.fingerprint,
        }

    def record(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "finish_reason": self.finish_reason,
            "log_likelihood": self.log_likelihood.hex(),
            "score": self.score.hex(),
        }


@dataclass
class _Beam:
    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    total: float
    cache: Any


def _clone(model: Any, cache: Any) -> Any:
    if cache is None:
        return None
    copy = model.new_cache()
    copy.restore(*cache.export())
    return copy


def search_tokens(
    model: Any,
    context: Sequence[int],
    width: int,
    max_tokens: int,
    stop_tokens: frozenset[int] | set[int],
    *,
    n_best: int = 1,
    length_penalty: float = 1.0,
    window: int | None = None,
    stopped: Callable[[tuple[int, ...]], bool] | None = None,
) -> list[Hypothesis]:
    """The best ``n_best`` hypotheses after ``context`` (see the module docstring). ``stopped`` says whether an
    answer's tokens end in a stop sequence; ``window`` is the model's context window."""
    validate(width, n_best, length_penalty)
    context = [int(t) for t in context]
    new_cache = getattr(model, "new_cache", None)
    batched = new_cache is not None and hasattr(model, "forward_batch")
    live = [_Beam((), (), 0.0, new_cache() if new_cache is not None else None)]
    finished: list[tuple[tuple[int, ...], tuple[float, ...], str, float]] = []
    exhausted = True
    for step in range(max_tokens):
        if window is not None and len(context) + step >= window:
            break
        sequences = [[*context, *beam.tokens] for beam in live]
        if batched:
            rows = model.forward_batch(sequences, [beam.cache for beam in live])
        elif new_cache is not None:
            rows = [model.forward_cached(tokens, beam.cache) for tokens, beam in zip(sequences, live, strict=True)]
        else:
            rows = [model.forward(tokens) for tokens in sequences]
        candidates: list[tuple[float, tuple[int, ...], int, int, float]] = []
        for index, (beam, row) in enumerate(zip(live, rows, strict=True)):
            values = log_softmax(np.asarray(row, dtype=np.float32).reshape(-1))
            order = np.lexsort((np.arange(len(values)), -values.astype(np.float64)))[: 2 * width]
            for token in order.tolist():
                logprob = float(values[token])
                candidates.append((beam.total + logprob, (*beam.tokens, token), index, token, logprob))
        candidates.sort(key=lambda c: (-c[0], c[1]))
        following: list[_Beam] = []
        taken: set[int] = set()
        for rank, (total, sequence, index, token, logprob) in enumerate(candidates):
            if len(following) == width:
                break
            parent = live[index]
            logprobs = (*parent.logprobs, logprob)
            if token in stop_tokens:
                if rank < width:
                    finished.append((parent.tokens, logprobs, "stop", total))
                continue
            if stopped is not None and stopped(sequence):
                if rank < width:
                    finished.append((sequence, logprobs, "stop", total))
                continue
            cache = parent.cache if index not in taken else _clone(model, parent.cache)
            taken.add(index)
            following.append(_Beam(sequence, logprobs, total, cache))
        live = following
        if len(finished) >= width or not live:
            exhausted = False
            break
    if exhausted:
        finished.extend((beam.tokens, beam.logprobs, "length", beam.total) for beam in live)
    hypotheses = [
        Hypothesis(tokens, logprobs, reason, total, score(total, len(logprobs), length_penalty))
        for tokens, logprobs, reason, total in finished
    ]
    hypotheses.sort(key=lambda h: (-h.score, h.tokens))
    return hypotheses[:n_best]


@dataclass(frozen=True)
class BeamResult:
    hypotheses: tuple[Hypothesis, ...]
    """The best ``n_best``, best first."""
    prompt_tokens: int
    width: int
    n_best: int
    length_penalty: float

    @property
    def completion_tokens(self) -> int:
        return sum(len(h.tokens) for h in self.hypotheses)

    def to_json(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "n_best": self.n_best,
            "length_penalty": self.length_penalty,
            "hypotheses": [h.to_json() for h in self.hypotheses],
        }


def _check(request: ChatRequest) -> None:
    """Beam search ranks by the model's own log-probabilities: what would change them, or the answer's shape, is
    refused."""
    if request.tools and request.tool_choice.mode != "none":
        raise ValueError("beam search cannot use tools")
    if request.response_format.grammar() is not None:
        raise ValueError("beam search cannot be combined with structured output or a regex")
    if request.context_overflow != "stop":
        raise ValueError("beam search cannot roll the context")
    if request.token_healing:
        raise ValueError("beam search cannot heal tokens")
    if request.suffix is not None:
        raise ValueError("beam search cannot fill in the middle (suffix)")
    if request.top_logprobs:
        raise ValueError("beam search reports no alternative tokens (top_logprobs)")
    changed = sorted(name for name, default in _DEFAULTS.items() if getattr(request.options, name) != default)
    if changed:
        raise ValueError(
            "beam search ranks by the model's own log-probabilities; it cannot be combined with " + ", ".join(changed)
        )


def search(
    engine: DllmEngine, request: ChatRequest, width: int, n_best: int = 1, length_penalty: float = 1.0
) -> BeamResult:
    """A beam search for ``request``'s answer (the conversation's, or the raw prompt's continuation), honouring its
    ``max_tokens`` and ``stop``."""
    validate(width, n_best, length_penalty)
    _check(request)
    _, prompt = engine.request_prompt(request)
    tokenizer = engine.tokenizer
    context = tokenizer.encode(prompt)
    window = getattr(getattr(engine.model, "config", None), "context_length", None) or None
    if window is not None and len(context) >= window:
        raise ContextLengthError(
            f"the prompt has {len(context)} tokens; the model's context window holds {window}, answer included"
        )
    if request.max_tokens < 1:
        raise ValueError("beam search needs max_tokens of at least 1")
    stop = [s for s in request.stop if s]
    strip = request.prompt is None and bool(getattr(tokenizer, "strips_leading_space", False))

    def decoded(tokens: Sequence[int]) -> bytearray:
        data = bytearray(b"".join(tokenizer.decode_bytes([t]) for t in tokens))
        if strip and data[:1] == b" ":
            del data[0]
        return data

    def text_of(tokens: Sequence[int]) -> str:
        data = decoded(tokens)
        return bytes(data[: _complete_prefix(data)]).decode("utf-8", errors="replace")

    def stopped(tokens: tuple[int, ...]) -> bool:
        return bool(stop) and any(s in text_of(tokens) for s in stop)

    found = search_tokens(
        engine.model,
        context,
        width,
        request.max_tokens,
        engine.stop_tokens,
        n_best=n_best,
        length_penalty=length_penalty,
        window=window,
        stopped=stopped if stop else None,
    )
    hypotheses = []
    for hypothesis in found:
        text = bytes(decoded(hypothesis.tokens)).decode("utf-8", errors="replace")
        cuts = [i for s in stop if (i := text.find(s)) >= 0]
        values = {f.name: getattr(hypothesis, f.name) for f in fields(hypothesis)}
        hypotheses.append(Hypothesis(**{**values, "text": text[: min(cuts)] if cuts else text}))
    return BeamResult(tuple(hypotheses), len(context), width, n_best, float(length_penalty))


def _id(body: Mapping[str, Any]) -> str:
    content = {k: v for k, v in body.items() if k not in ("id", "signature")}
    return "beam_" + hashlib.sha256(canonical_json(content).encode()).hexdigest()[:32]


def record(engine: DllmEngine, request: ChatRequest, result: BeamResult) -> dict[str, Any]:
    """A receipt for a beam search: the request, the search settings and every answer's fingerprint and scores."""
    from etalii_dllm import __version__

    body: dict[str, Any] = {
        "beam": FORMAT,
        "engine": __version__,
        "model": engine.model.id,
        "system_fingerprint": engine.system_fingerprint,
        "request": receipts.request_record(request),
        "width": result.width,
        "n_best": result.n_best,
        "length_penalty": result.length_penalty,
        "output": {"hypotheses": [h.record() for h in result.hypotheses]},
    }
    return {**body, "id": _id(body)}


def verify(engine: DllmEngine, receipt: Mapping[str, Any]) -> Verification:
    """Searches again with the recorded request and settings and compares every answer."""
    from etalii_dllm import __version__

    if receipt.get("beam") != FORMAT:
        raise ValueError(f"not a {FORMAT} receipt")
    reasons: list[str] = []
    if receipt.get("id") != _id(receipt):
        reasons.append("the receipt was edited: its id does not match its content")
    if receipt["system_fingerprint"] != engine.system_fingerprint:
        reasons.append(
            f"different weights or settings: the receipt was made with {receipt['system_fingerprint']}, "
            f"this engine is {engine.system_fingerprint}"
        )
    notes = []
    if receipt["engine"] != __version__:
        notes.append(f"made by engine version {receipt['engine']}, replayed with {__version__}")
    request = receipts.request_from_record(receipt["request"])
    result = search(engine, request, int(receipt["width"]), int(receipt["n_best"]), float(receipt["length_penalty"]))
    replayed = record(engine, request, result)
    expected, actual = receipt["output"]["hypotheses"], replayed["output"]["hypotheses"]
    if len(expected) != len(actual):
        reasons.append(f"the number of answers differs: recorded {len(expected)}, replayed {len(actual)}")
    for index, (before, after) in enumerate(zip(expected, actual, strict=False)):
        if before != after:
            reasons.append(f"answer {index} differs: recorded {before!r}, replayed {after!r}")
    return Verification(not reasons, tuple(reasons), tuple(notes), replayed)
