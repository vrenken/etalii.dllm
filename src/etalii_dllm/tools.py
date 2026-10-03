"""Tool (function) calling: presenting tools to the model, constraining its calls and reading them back.

Models call tools in the format they were trained on, which :func:`detect_format` reads from the model's own chat
template (:data:`FORMATS`):

- ``hermes`` (Qwen2.5, Qwen3, SmolLM, Hermes and most tool-trained small models)::

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Paris"}}
    </tool_call>

- ``llama3`` (Llama 3.1/3.2/3.3): the whole reply is one bare call,
  ``{"name": "get_weather", "parameters": {"city": "Paris"}}``.
- ``mistral`` (Mistral, Mixtral, Ministral): ``[TOOL_CALLS][{"name": "get_weather", "arguments": {...}}]``.
- ``granite`` (IBM Granite 3): ``<|tool_call|>[{"name": "get_weather", "arguments": {...}}]``.

When the model's chat template knows about tools (it references ``tools``), the template presents them, exactly as
``transformers`` would; otherwise the engine adds the Hermes instructions to the system message and the model calls
tools in the Hermes format. Calls are constrained by a grammar: with ``tool_choice`` ``auto`` the constraint starts
once the model writes the format's marker (``<tool_call>``, ``[TOOL_CALLS]``, ``<|tool_call|>``; for Llama 3 once
the reply starts with ``{``), before which it may answer in text, so a call always has a known function name and
arguments that fit its parameter schema; with ``required`` or a named tool the output must be exactly one call.

Tool call ids are derived from the request and the call's position, never from a clock or random source. A template
that insists on its own id shape (Mistral's nine letters and digits) sees ids derived from the caller's ids.
"""

from __future__ import annotations

import hashlib
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


@dataclass(frozen=True)
class ToolFormat:
    """How a model family writes tool calls."""

    name: str
    open: str
    """The marker before the calls (``""`` when a call is a bare JSON object)."""
    close: str = ""
    """The marker after each call (Hermes ``</tool_call>``)."""
    arguments: str = "arguments"
    """The key of the arguments object."""
    listed: bool = False
    """The calls are one JSON array after the marker (else one JSON object per marker)."""
    id_length: int | None = None
    """The template accepts only call ids of exactly this many letters and digits."""


HERMES = ToolFormat("hermes", TOOL_CALL_OPEN, TOOL_CALL_CLOSE)
LLAMA3 = ToolFormat("llama3", "", arguments="parameters")
MISTRAL = ToolFormat("mistral", "[TOOL_CALLS]", listed=True, id_length=9)
GRANITE = ToolFormat("granite", "<|tool_call|>", listed=True)
FORMATS = {f.name: f for f in (HERMES, LLAMA3, MISTRAL, GRANITE)}
"""The tool call formats, by name."""


def detect_format(template_source: str | None) -> ToolFormat:
    """The tool call format of a model, from its chat template: the template's own call marker decides, in a fixed
    order; Llama 3's bare calls are recognised by its ``ipython`` role and ``"parameters"`` key. Templates that do
    not present tools get the Hermes instructions, and so the Hermes format."""
    if not template_supports_tools(template_source):
        return HERMES
    assert template_source is not None
    for marker, found in (("[TOOL_CALLS]", MISTRAL), ("<|tool_call|>", GRANITE), (TOOL_CALL_OPEN, HERMES)):
        if marker in template_source:
            return found
    if "ipython" in template_source and "parameters" in template_source:
        return LLAMA3
    return HERMES


def tool_format(name: str) -> ToolFormat:
    """The format called ``name`` (``ValueError`` for unknown names)."""
    if name not in FORMATS:
        raise ValueError(f"unknown tool call format {name!r}; known: {', '.join(sorted(FORMATS))}")
    return FORMATS[name]


_ALPHANUMERIC = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def template_id(call_id: str, length: int) -> str:
    """A call id of ``length`` letters and digits derived from ``call_id`` (ids that already fit stay as they are),
    so a call and its result keep matching ids."""
    if len(call_id) == length and all(c in _ALPHANUMERIC for c in call_id):
        return call_id
    number = int.from_bytes(hashlib.sha256(call_id.encode("utf-8")).digest(), "big")
    digits = []
    for _ in range(length):
        number, digit = divmod(number, len(_ALPHANUMERIC))
        digits.append(_ALPHANUMERIC[digit])
    return "".join(digits)


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


def template_messages(messages: Sequence[ChatMessage], fmt: ToolFormat = HERMES) -> list[dict[str, Any]]:
    """Messages in the Hugging Face form tool-aware chat templates expect (arguments as objects; call ids in the
    shape the format's template accepts)."""

    def ident(call_id: str) -> str:
        return template_id(call_id, fmt.id_length) if fmt.id_length is not None else call_id

    result = []
    for message in messages:
        entry: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": ident(call.id),
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_object()},
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id is not None:
            entry["tool_call_id"] = ident(message.tool_call_id)
        if message.name is not None:
            entry["name"] = message.name
        result.append(entry)
    return result


def call_schema(tool: Tool, fmt: ToolFormat = HERMES) -> dict[str, Any]:
    parameters = dict(tool.parameters) if tool.parameters else {"type": "object", "properties": {}}
    if parameters.get("type", "object") != "object":
        raise ValueError(f"the parameters of tool {tool.name!r} must be a JSON schema of type object")
    return {
        "type": "object",
        "properties": {"name": {"const": tool.name}, fmt.arguments: parameters},
        "required": ["name", fmt.arguments],
    }


def _calls(tools: Sequence[Tool], fmt: ToolFormat) -> Grammar:
    """One call to any of ``tools``, or for listed formats a non-empty JSON array of them."""
    if fmt.listed:
        items = {"anyOf": [call_schema(t, fmt) for t in tools]}
        return Grammar.json_schema({"type": "array", "items": items, "minItems": 1}, lenient=True)
    return Grammar.choice([Grammar.json_schema(call_schema(t, fmt), lenient=True) for t in tools])


def call_grammar(tools: Sequence[Tool], fmt: ToolFormat = HERMES) -> Grammar:
    """What follows the format's marker: the call (or call list) and the closing marker."""
    parts = [Grammar.whitespace(), _calls(tools, fmt)]
    if fmt.close:
        parts += [Grammar.whitespace(), Grammar.literal(fmt.close)]
    return Grammar.sequence(parts)


def forced_grammar(tools: Sequence[Tool], choice: ToolChoice, fmt: ToolFormat = HERMES) -> Grammar:
    """Exactly one call (or call list), for ``required`` and named tool choices."""
    chosen = [t for t in tools if t.name == choice.name] if choice.mode == "named" else list(tools)
    return Grammar.sequence([Grammar.literal(fmt.open), call_grammar(chosen, fmt)])


FREE_TEXT = r"[ \t\r\n]*([^{ \t\r\n][\s\S]*)?"
"""Text that does not start like a JSON object (an answer, for formats whose calls are bare objects)."""


def auto_grammar(tools: Sequence[Tool], fmt: ToolFormat) -> tuple[Grammar, str | None]:
    """The constraint for ``tool_choice`` ``auto`` and the marker that starts it (``None``: from the start)."""
    if fmt.open:
        return call_grammar(tools, fmt), fmt.open
    return Grammar.either([Grammar.regex(FREE_TEXT), forced_grammar(tools, AUTO, fmt)]), None


def _block(fmt: ToolFormat) -> re.Pattern[str]:
    return re.compile(re.escape(fmt.open) + r"(.*?)(?:" + re.escape(fmt.close) + r"|\Z)", re.DOTALL)


def _as_call(value: Any, tools: Mapping[str, Tool]) -> tuple[str, str] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
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


def _listed_calls(text: str, fmt: ToolFormat, tools: Mapping[str, Tool]) -> tuple[str, list[tuple[str, str]]] | None:
    """The text before the marker and the calls of the JSON array after it, when it is one of known calls."""
    cut = text.find(fmt.open)
    if cut < 0:
        return None
    rest = text[cut + len(fmt.open) :].lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(rest)
    except json.JSONDecodeError:
        return None
    items = value if isinstance(value, list) else [value]
    calls = [_as_call(item, tools) for item in items]
    if not calls or None in calls or rest[end:].strip():
        return None
    return text[:cut].strip(), [call for call in calls if call is not None]


def parse_calls(text: str, tools: Sequence[Tool], fmt: ToolFormat = HERMES) -> tuple[str, list[tuple[str, str]]]:
    """Splits generated text into answer text (stripped) and ``(name, arguments JSON)`` calls in the format
    ``fmt``. Blocks that do not hold calls to known tools stay in the text. Without blocks, a reply that is nothing
    but a JSON call object (the Llama 3 style) counts as a call in every format."""
    by_name = {t.name: t for t in tools}
    calls: list[tuple[str, str]] = []
    kept: list[str] = []
    if fmt.listed:
        found = _listed_calls(text, fmt, by_name)
        if found is not None:
            return found
    elif fmt.open:
        position = 0
        for match in _block(fmt).finditer(text):
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
