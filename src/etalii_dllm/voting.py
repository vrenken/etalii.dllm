"""Self-consistency voting (``vote`` on chat completions, ``dllm chat --vote N``; docs/api.md#voting).

A vote samples ``n`` answers to one request and returns the most common one, with every step fixed:

- The answers are the request's choices ``0 .. n-1`` (:meth:`~etalii_dllm.engine.DllmEngine.choice_request`: the
  seeds ``seed + i``), so each is exactly the answer a single request with that seed gets.
- Each answer is normalised (:func:`normalize`): with an ``extract`` regex, its last match (the first group when
  the pattern has groups) is taken from the answer, and an answer without a match casts no vote; then NFKC and lower
  case from the pinned Unicode tables, runs of ASCII whitespace become one space, and the ends are stripped. An empty
  result casts no vote.
- The winner is the answer with the most votes; a tie goes to the answer whose first vote came from the lowest choice
  index. The response is that first choice's full answer. With no votes at all, choice 0 is returned.

The same request gives the same answers, votes and winner on every machine. :func:`record` makes a receipt of a vote
that :func:`verify` (``dllm replay``) checks by voting again.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from etalii_dllm import receipts, unicode
from etalii_dllm.receipts import Verification, canonical_json

if TYPE_CHECKING:
    from etalii_dllm.engine import ChatRequest, ChatResult, DllmEngine

FORMAT = "dllm-vote/1"
"""The vote receipt format; a reader refuses others."""

MAX_VOTERS = 16
WHITESPACE = " \t\n\r\f\v"


def normalize(text: str, extract: str | None = None) -> str | None:
    """The answer ``text`` votes for, or ``None`` when it casts no vote (see the module docstring)."""
    if extract is not None:
        matches = list(unicode.compile(extract).finditer(text))
        if not matches:
            return None
        last = matches[-1]
        text = (last.group(1) if last.re.groups else last.group(0)) or ""
    text = unicode.lower(unicode.normalize("NFKC", text))
    words: list[str] = []
    word = ""
    for character in text:
        if character in WHITESPACE:
            if word:
                words.append(word)
            word = ""
        else:
            word += character
    if word:
        words.append(word)
    return " ".join(words) or None


def validate(n: int, extract: str | None) -> None:
    if not 1 <= n <= MAX_VOTERS:
        raise ValueError(f"a vote needs between 1 and {MAX_VOTERS} answers")
    if extract is not None:
        try:
            unicode.compile(extract)
        except Exception as error:  # regex.error, or a ValueError from the pinned translation
            raise ValueError(f"vote extract is not a valid regex: {error}") from error


@dataclass(frozen=True)
class Ballot:
    answer: str
    choices: tuple[int, ...]
    """The choice indexes that voted for it, ascending."""

    @property
    def votes(self) -> int:
        return len(self.choices)

    def to_json(self) -> dict[str, Any]:
        return {"answer": self.answer, "votes": self.votes, "choices": list(self.choices)}


def tally(answers: Sequence[str | None]) -> tuple[tuple[Ballot, ...], int]:
    """The ballots, most votes first (ties: the earliest first vote first), and the winning choice index."""
    voters: dict[str, list[int]] = {}
    for index, answer in enumerate(answers):
        if answer is not None:
            voters.setdefault(answer, []).append(index)
    ballots = sorted((Ballot(a, tuple(c)) for a, c in voters.items()), key=lambda b: (-b.votes, b.choices[0]))
    return tuple(ballots), ballots[0].choices[0] if ballots else 0


@dataclass(frozen=True)
class VoteResult:
    results: tuple[ChatResult, ...]
    """Every sampled answer, in choice order."""
    answers: tuple[str | None, ...]
    ballots: tuple[Ballot, ...]
    winner: int
    extract: str | None

    @property
    def result(self) -> ChatResult:
        return self.results[self.winner]

    @property
    def answer(self) -> str | None:
        return self.answers[self.winner]

    def to_json(self) -> dict[str, Any]:
        return {
            "n": len(self.results),
            "extract": self.extract,
            "winner": self.winner,
            "answer": self.answer,
            "ballots": [b.to_json() for b in self.ballots],
            "answers": list(self.answers),
        }


def vote(engine: DllmEngine, request: ChatRequest, n: int, extract: str | None = None) -> VoteResult:
    validate(n, extract)
    results = tuple(engine.chat_choices(request, n))
    answers = tuple(normalize(r.content, extract) for r in results)
    ballots, winner = tally(answers)
    return VoteResult(results, answers, ballots, winner, extract)


def _id(body: Mapping[str, Any]) -> str:
    content = {k: v for k, v in body.items() if k not in ("id", "signature")}
    return "vote_" + hashlib.sha256(canonical_json(content).encode()).hexdigest()[:32]


def record(engine: DllmEngine, request: ChatRequest, result: VoteResult) -> dict[str, Any]:
    """A receipt for a vote: the request, and every answer's receipt id, the ballots and the winner."""
    from etalii_dllm import __version__

    body: dict[str, Any] = {
        "vote": FORMAT,
        "engine": __version__,
        "model": engine.model.id,
        "system_fingerprint": engine.system_fingerprint,
        "request": receipts.request_record(request),
        "n": len(result.results),
        "extract": result.extract,
        "output": {
            "receipts": [r.receipt["id"] if r.receipt else None for r in result.results],
            "ballots": [b.to_json() for b in result.ballots],
            "winner": result.winner,
        },
    }
    return {**body, "id": _id(body)}


def verify(engine: DllmEngine, receipt: Mapping[str, Any]) -> Verification:
    """Votes again with the recorded request and compares every answer, the ballots and the winner."""
    from etalii_dllm import __version__

    if receipt.get("vote") != FORMAT:
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
    validate(int(receipt["n"]), receipt["extract"])
    results = tuple(
        engine.chat_completion(engine.choice_request(request, i), fresh=True) for i in range(int(receipt["n"]))
    )
    answers = tuple(normalize(r.content, receipt["extract"]) for r in results)
    ballots, winner = tally(answers)
    replayed = record(engine, request, VoteResult(results, answers, ballots, winner, receipt["extract"]))
    expected, actual = receipt["output"], replayed["output"]
    for index, (before, after) in enumerate(zip(expected["receipts"], actual["receipts"], strict=False)):
        if before != after:
            reasons.append(f"answer {index} differs: recorded {before!r}, replayed {after!r}")
    for key, label in (("ballots", "the ballots"), ("winner", "the winner")):
        if expected.get(key) != actual[key]:
            reasons.append(f"{label} differ: recorded {expected.get(key)!r}, replayed {actual[key]!r}")
    return Verification(not reasons, tuple(reasons), tuple(notes), replayed)
