"""Flattens a chat conversation into a single prompt, in a fixed, documented format."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


def render(messages: Iterable[ChatMessage]) -> str:
    parts = [f"<|{m.role}|>\n{m.content}\n" for m in messages]
    return "".join(parts) + "<|assistant|>\n"
