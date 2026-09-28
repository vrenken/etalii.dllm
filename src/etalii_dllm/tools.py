"""Tool (function) calling: presenting tools to the model, constraining its calls and reading them back.

Models call tools in the Hermes format that Qwen2.5 and most tool-trained small models use::

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Paris"}}
    </tool_call>

When the model's chat template knows about tools (it references ``tools``), the template presents them, exactly as
``transformers`` would; otherwise the engine adds the Hermes instructions to the system message. Calls are
constrained by a grammar: with ``tool_choice`` ``auto`` the constraint starts once the model writes
``<tool_call>`` (before that it may answer in text), so a call always has a known function name and arguments that
fit its parameter schema; with ``required`` or a named tool the output must be exactly one call.

Tool call ids are derived from the request and the call's position, never from a clock or random source.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from etalii_dllm.chat import TOOL_CALL_CLOSE, TOOL_CALL_OPEN, ChatMessage
from etalii_dllm.grammar import Grammar


@dataclass(frozen=True)
class Tool:
    name: str
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    """JSON schema of the arguments object."""

    def to_openai(self) -> dict[str, Any]:
        """The OpenAI/Hugging Face form chat templates expect."""
        function: dict[str, Any] = {"name": self.name}
        if self.description:
            function["description"] = self.description
        function["parameters"] = dict(self.parameters)
        return {"type": "function", "function": function}


@dataclass(frozen=True)
class ToolChoice:
    mode: str = "auto"
    """``auto`` (the model decides), ``none`` (no tools), ``required`` (at least one call) or ``named``."""
    name: str | None = None
    """The function to call when ``mode`` is ``named``."""

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "none", "required", "named"):
            raise ValueError(f"unknown tool_choice {self.mode!r}")
        if (self.mode == "named") != (self.name is not None):
            raise ValueError("a named tool_choice needs exactly one function name")


AUTO = ToolChoice()

HERMES_INSTRUCTIONS = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> "
    'XML tags:\n<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>'
)


def validate_tools(tools: Sequence[Tool], choice: ToolChoice) -> None:
    names = [t.name for t in tools]
    if len(set(names)) != len(names):
        raise ValueError("tool names must be unique")
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", name):
            raise ValueError(f"invalid tool name {name!r}")
    if choice.mode in ("required", "named") and not tools:
        raise ValueError(f"tool_choice {choice.mode!r} needs tools")
    if choice.mode == "named" and choice.name not in names:
        raise ValueError(f"tool_choice names an unknown tool {choice.name!r}")
    for tool in tools:
        call_schema(tool)  # raises for schemas that are not objects


def template_supports_tools(template_source: str | None) -> bool:
    return template_source is not None and "tools" in template_source


def instructions(tools: Sequence[Tool]) -> str:
    return HERMES_INSTRUCTIONS.format(tools="\n".join(json.dumps(t.to_openai(), ensure_ascii=False) for t in tools))


def with_instructions(messages: Sequence[ChatMessage], tools: Sequence[Tool]) -> list[ChatMessage]:
    """For models whose template does not present tools: the Hermes instructions in the system message, earlier
    calls as ``<tool_call>`` text and tool results as ``<tool_response>`` user turns, the way Hermes-format models
    see them."""
    converted: list[ChatMessage] = []
    for message in messages:
        if message.role == "tool":
            response = f"<tool_response>\n{message.content}\n</tool_response>"
            previous = converted[-1] if converted else None
            if previous is not None and previous.role == "user" and previous.content.startswith("<tool_response>"):
                converted[-1] = ChatMessage("user", previous.content + "\n" + response)
            else:
                converted.append(ChatMessage("user", response))
        else:
            converted.append(message)
    if tools:
        text = instructions(tools)
        if converted and converted[0].role == "system":
            converted[0] = ChatMessage("system", converted[0].content + "\n\n" + text)
        else:
            converted.insert(0, ChatMessage("system", text))
    return converted


def template_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Messages in the Hugging Face form tool-aware chat templates expect (arguments as objects)."""
    result = []
    for message in messages:
        entry: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_object()},
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id is not None:
            entry["tool_call_id"] = message.tool_call_id
        if message.name is not None:
            entry["name"] = message.name
        result.append(entry)
    return result


def call_schema(tool: Tool) -> dict[str, Any]:
    parameters = dict(tool.parameters) if tool.parameters else {"type": "object", "properties": {}}
    if parameters.get("type", "object") != "object":
        raise ValueError(f"the parameters of tool {tool.name!r} must be a JSON schema of type object")
    return {
        "type": "object",
        "properties": {"name": {"const": tool.name}, "arguments": parameters},
        "required": ["name", "arguments"],
    }


def call_grammar(tools: Sequence[Tool]) -> Grammar:
    """The JSON object and closing tag after ``<tool_call>``."""
    calls = Grammar.choice([Grammar.json_schema(call_schema(t), lenient=True) for t in tools])
    return Grammar.sequence([Grammar.whitespace(), calls, Grammar.whitespace(), Grammar.literal(TOOL_CALL_CLOSE)])


def forced_grammar(tools: Sequence[Tool], choice: ToolChoice) -> Grammar:
    """Exactly one call, for ``required`` and named tool choices."""
    chosen = [t for t in tools if t.name == choice.name] if choice.mode == "named" else list(tools)
    return Grammar.sequence([Grammar.literal(TOOL_CALL_OPEN), call_grammar(chosen)])


_BLOCK = re.compile(re.escape(TOOL_CALL_OPEN) + r"(.*?)(?:" + re.escape(TOOL_CALL_CLOSE) + r"|\Z)", re.DOTALL)


def _as_call(text: str, tools: Mapping[str, Tool]) -> tuple[str, str] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or value.get("name") not in tools:
        return None
    arguments = value.get("arguments", value.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return str(value["name"]), json.dumps(arguments, ensure_ascii=False)


def parse_calls(text: str, tools: Sequence[Tool]) -> tuple[str, list[tuple[str, str]]]:
    """Splits generated text into answer text (stripped) and ``(name, arguments JSON)`` calls. ``<tool_call>``
    blocks that do not hold a call to a known tool stay in the text. Without blocks, a reply that is nothing but a
    JSON call object (the Llama 3 style) counts as a call too."""
    by_name = {t.name: t for t in tools}
    calls: list[tuple[str, str]] = []
    kept: list[str] = []
    position = 0
    for match in _BLOCK.finditer(text):
        call = _as_call(match.group(1).strip(), by_name)
        if call is None:
            continue
        kept.append(text[position : match.start()])
        calls.append(call)
        position = match.end()
    kept.append(text[position:])
    if not calls:
        call = _as_call(text.strip(), by_name) if text.strip().startswith("{") else None
        if call is not None:
            return "", [call]
        return text.strip(), []
    return "".join(kept).strip(), calls
