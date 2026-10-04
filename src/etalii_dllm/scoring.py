"""Exact prompt scoring (``dllm score``, ``echo`` with ``logprobs`` on ``/v1/completions``; docs/api.md#scoring).

The score of a text is the log-probability of each of its tokens given the tokens before it:

- The text is tokenized exactly as a prompt is (``tokenizer.encode``), so a completion's echoed prompt and a
  ``dllm score`` of the same text carry the same numbers.
- Token ``i`` (for ``i >= 1``) gets ``log_softmax(logits after tokens 0 .. i-1)[token i]``, a float32 from the
  decoder's fixed-order kernels and the ``log_softmax`` kernel. The first token has nothing before it and is not
  scored (``None``, as in OpenAI's completions API).
- With ``top`` alternatives, each scored position also lists the ``top`` most likely tokens, ordered by log-probability
  descending, then id ascending.
- A text-to-text model (:mod:`etalii_dllm.seq2seq`, #393) scores a text as the answer to a ``source``: every
  token is scored, token ``i`` given the source and the tokens before it, from the decoder's logits with the bits of
  incremental decoding (``TextToText.answer_logits``).
- ``log_likelihood`` sums the float32 log-probabilities in token order in double (``numerics.sum_``), and
  ``perplexity`` is ``exp(-log_likelihood / scored)`` with the portable ``exp``.

Every number is a pure function of the weights and the text, so the same text scores the same on every machine.
:func:`record` turns a score into a receipt that :func:`verify` (``dllm replay``) checks by scoring again.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from etalii_dllm import _kernels
from etalii_dllm.evaluation import _rows
from etalii_dllm.generation import ContextLengthError, TokenLogprob, with_end_of_source
from etalii_dllm.numerics import log_softmax, sum_
from etalii_dllm.receipts import Verification, canonical_json

if TYPE_CHECKING:
    from etalii_dllm.engine import DllmEngine

FORMAT = "dllm-score/1"
"""The score receipt format; a reader refuses others."""


@dataclass(frozen=True)
class TokenScore:
    token: int
    logprob: float | None
    """``None`` for the first token, which has no context."""
    top: tuple[TokenLogprob, ...] = ()


@dataclass(frozen=True)
class TextScore:
    tokens: tuple[TokenScore, ...]

    @property
    def scored(self) -> int:
        return sum(t.logprob is not None for t in self.tokens)

    @property
    def log_likelihood(self) -> float:
        values = np.asarray([t.logprob for t in self.tokens if t.logprob is not None], dtype=np.float32)
        return sum_(values) if len(values) else 0.0

    @property
    def perplexity(self) -> float | None:
        return float(_kernels.exp(-self.log_likelihood / self.scored)) if self.scored else None

    @property
    def fingerprint(self) -> str:
        """A hash of every token id and the exact bits of its log-probability (and alternatives)."""
        entries = [
            [t.token, None if t.logprob is None else t.logprob.hex(), [[a.token, a.logprob.hex()] for a in t.top]]
            for t in self.tokens
        ]
        return hashlib.sha256(canonical_json(entries).encode()).hexdigest()

    def to_json(self, engine: DllmEngine) -> dict[str, Any]:
        def text(token: int) -> str:
            return engine.tokenizer.decode_bytes([token]).decode("utf-8", errors="replace")

        return {
            "tokens": [
                {
                    "token": t.token,
                    "text": text(t.token),
                    "logprob": t.logprob,
                    **({"top_logprobs": [{"token": a.token, "text": text(a.token), "logprob": a.logprob}
                                         for a in t.top]} if t.top else {}),
                }
                for t in self.tokens
            ],
            "scored": self.scored,
            "log_likelihood": self.log_likelihood,
            "perplexity": self.perplexity,
            "fingerprint": self.fingerprint,
        }  # fmt: skip


def score_tokens(
    engine: DllmEngine, tokens: Sequence[int], top: int = 0, source: Sequence[int] | None = None
) -> TextScore:
    """The score of ``tokens`` (see the module docstring); ``source`` is a text-to-text model's source, which it
    needs. Raises :class:`ContextLengthError` when they do not fit the model's context window."""
    if not 0 <= top <= 20:
        raise ValueError("top logprobs must be between 0 and 20")
    tokens = [int(t) for t in tokens]
    model = engine.model
    text_to_text = getattr(model, "end_of_source", None) is not None
    if text_to_text and source is None:
        raise ValueError("a text-to-text model scores a text as the answer to a source")
    if not text_to_text and source is not None:
        raise ValueError("only text-to-text models score a text given a source")
    window = getattr(getattr(model, "config", None), "context_length", None)
    if window and len(tokens) > window:
        raise ContextLengthError(f"the text has {len(tokens)} tokens; the model's context window holds {window}")
    if not tokens:
        return TextScore(())
    if text_to_text:
        assert source is not None
        rows = model.answer_logits(with_end_of_source(model, source), tokens)  # type: ignore[attr-defined]
        targets = tokens
        scores = []
    else:
        rows = _rows(model, tokens[:-1], len(tokens) - 1) if len(tokens) > 1 else np.zeros((0, 0), np.float32)
        targets = tokens[1:]
        scores = [TokenScore(tokens[0], None)]
    for i, target in enumerate(targets):
        values = log_softmax(rows[i])
        alternatives: tuple[TokenLogprob, ...] = ()
        if top:
            order = np.lexsort((np.arange(len(values)), -values.astype(np.float64)))[:top]
            alternatives = tuple(TokenLogprob(int(j), float(values[j])) for j in order)
        scores.append(TokenScore(target, float(values[target]), alternatives))
    return TextScore(tuple(scores))


def score_text(engine: DllmEngine, text: str, top: int = 0, source: str | None = None) -> TextScore:
    """The score of ``text``; for a text-to-text model, as the answer to ``source``."""
    encoded = None if source is None else engine.tokenizer.encode(source)
    return score_tokens(engine, engine.tokenizer.encode(text), top, encoded)


def _id(body: Mapping[str, Any]) -> str:
    content = {k: v for k, v in body.items() if k not in ("id", "signature")}
    return "score_" + hashlib.sha256(canonical_json(content).encode()).hexdigest()[:32]


def record(engine: DllmEngine, text: str, top: int, score: TextScore, source: str | None = None) -> dict[str, Any]:
    """A receipt for a score: the text (and a text-to-text model's source), the weights and the score's
    fingerprint."""
    from etalii_dllm import __version__

    body: dict[str, Any] = {
        "score": FORMAT,
        "engine": __version__,
        "model": engine.model.id,
        "system_fingerprint": engine.system_fingerprint,
        "text": text,
        **({"source": source} if source is not None else {}),
        "top_logprobs": top,
        "output": {
            "tokens": len(score.tokens),
            "log_likelihood": score.log_likelihood.hex(),
            "fingerprint": score.fingerprint,
        },
    }
    return {**body, "id": _id(body)}


def verify(engine: DllmEngine, receipt: Mapping[str, Any]) -> Verification:
    """Scores the recorded text again and compares the result with the receipt."""
    from etalii_dllm import __version__

    if receipt.get("score") != FORMAT:
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
    top = int(receipt["top_logprobs"])
    source = receipt.get("source")
    replayed = record(engine, receipt["text"], top, score_text(engine, receipt["text"], top, source), source)
    for key, label in (("tokens", "the number of tokens"), ("fingerprint", "the log-probabilities"),
                       ("log_likelihood", "the log-likelihood")):  # fmt: skip
        if receipt["output"].get(key) != replayed["output"][key]:
            reasons.append(
                f"{label} differ: recorded {receipt['output'].get(key)!r}, replayed {replayed['output'][key]!r}"
            )
    return Verification(not reasons, tuple(reasons), tuple(notes), replayed)
