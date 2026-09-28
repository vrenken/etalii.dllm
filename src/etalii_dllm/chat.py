"""Chat messages, and the fixed, documented prompt format for models without a chat template."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


@dataclass(frozen=True)
class ToolCall:
    """A function call made by the assistant. ``arguments`` is JSON text, as in the OpenAI API."""

    id: str
    name: str
    arguments: str

    def arguments_object(self) -> object:
        """The parsed arguments (the raw text when it is not valid JSON)."""
        try:
            return json.loads(self.arguments)
        except json.JSONDecodeError:
            return self.arguments

    def render(self) -> str:
        """The call in the Hermes/Qwen format the engine asks models to use."""
        call = {"name": self.name, "arguments": self.arguments_object()}
        return f"{TOOL_CALL_OPEN}\n{json.dumps(call, ensure_ascii=False)}\n{TOOL_CALL_CLOSE}"


@dataclass(frozen=True)
class ChatMessage:
    role: str
    """``system``, ``user``, ``assistant`` or ``tool``."""
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    """Calls made by an assistant message."""
    tool_call_id: str | None = None
    """For ``tool`` messages: the call this is the result of."""
    name: str | None = None
    """For ``tool`` messages: the function's name."""


def render(messages: Iterable[ChatMessage]) -> str:
    parts = []
    for m in messages:
        content = "\n".join([*([m.content] if m.content else []), *(call.render() for call in m.tool_calls)])
        parts.append(f"<|{m.role}|>\n{content}\n")
    return "".join(parts) + "<|assistant|>\n"
