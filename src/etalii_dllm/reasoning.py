"""Reasoning ("thinking") models: telling the thinking from the answer, and the exact thinking budget.

A thinking model writes ``<think>`` ... ``</think>`` before its answer; some chat templates (DeepSeek-R1 style) open
the block in the prompt, so the output starts inside it. The rule (``docs/specification.md#reasoning``) is fixed text
matching, applied the same way to streamed and finished output:

- The output *starts in thinking* when the prompt, stripped of trailing whitespace, ends with ``<think>``. Otherwise
  it thinks when its text, stripped of leading whitespace, starts with ``<think>``.
- The thinking runs to the first ``</think>``; the answer is the rest without its leading whitespace. The reasoning
  is the text in between with surrounding whitespace removed. An output that never closes the block has no answer.
- An output that does not think is all answer, unchanged.

A budget of N tokens counts the generated tokens from the first one of the block (the first output token when the
output starts in thinking, else the token that completes ``<think>``). When N tokens are spent and the block is still
open, the generator appends the tokens of :func:`closing_text` itself; they are part of the output.
"""

from __future__ import annotations

from dataclasses import dataclass

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def is_thinking_template(source: str | None) -> bool:
    """Whether a chat template belongs to a thinking model (it writes or strips ``<think>`` blocks)."""
    return source is not None and THINK_OPEN in source


def starts_in_thinking(prompt: str) -> bool:
    return prompt.rstrip().endswith(THINK_OPEN)


def closing_text(text: str) -> str:
    """What the generator appends to close a block whose text so far is ``text``."""
    return ("" if text.endswith("\n") else "\n") + THINK_CLOSE + "\n\n"


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest proper prefix of ``tag`` that ``text`` ends with."""
    for length in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


@dataclass(frozen=True)
class Split:
    reasoning: str | None
    """``None`` when the output does not think."""
    answer: str
    state: str
    """``undecided`` (it may still open a block), ``thinking`` (inside an open block) or ``answer``."""


def split(text: str, started: bool) -> Split:
    """The thinking and the answer of a finished ``text`` (see the module docstring)."""
    if started:
        body = text
    else:
        stripped = text.lstrip()
        if stripped.startswith(THINK_OPEN):
            body = stripped[len(THINK_OPEN) :]
        elif THINK_OPEN.startswith(stripped):
            return Split(None, text, "undecided")
        else:
            return Split(None, text, "answer")
    end = body.find(THINK_CLOSE)
    if end < 0:
        return Split(body.strip(), "", "thinking")
    return Split(body[:end].strip(), body[end + len(THINK_CLOSE) :].lstrip(), "answer")


def streamable(text: str, started: bool) -> tuple[str, str]:
    """The reasoning and answer text of an output still being generated that can be sent already: both are prefixes
    of what :func:`split` gives for any continuation of ``text``."""
    parts = split(text, started)
    if parts.state == "undecided":
        return "", ""
    if parts.reasoning is None:
        return "", parts.answer
    if parts.state == "answer":
        return parts.reasoning, parts.answer
    body = text if started else text.lstrip()[len(THINK_OPEN) :]
    body = body[: len(body) - _partial_suffix(body, THINK_CLOSE)]
    return body.strip(), ""


class Tracker:
    """Follows an output token by token for the budget: whether a block is open and how many tokens it holds."""

    def __init__(self, started: bool, budget: int | None) -> None:
        self.started = started
        self.budget = budget
        self.tokens = 0
        """Tokens of the block so far (the reasoning tokens of the usage)."""
        self._opened = started
        self._closed = False

    def step(self, text: str) -> None:
        """Called after each generated token with the output text so far."""
        if self._closed:
            return
        if not self._opened:
            stripped = text.lstrip()
            if stripped.startswith(THINK_OPEN):
                self._opened = True
            elif not THINK_OPEN.startswith(stripped):
                self._closed = True  # no thinking at all
                return
            else:
                return
        self.tokens += 1
        body = text if self.started else text.lstrip()[len(THINK_OPEN) :]
        if THINK_CLOSE in body:
            self._closed = True

    def close(self) -> None:
        """The generator closed the block (or could not, out of room); nothing more to count."""
        self._closed = True

    @property
    def over_budget(self) -> bool:
        """The block is open and has spent its budget: close it now."""
        return self.budget is not None and self._opened and not self._closed and self.tokens >= self.budget
