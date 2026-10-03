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
- ``xml`` (Qwen3-Coder): parameters as XML, string values as raw text and other values as JSON::

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Paris
    </parameter>
    </function>
    </tool_call>

- ``deepseek`` (DeepSeek V3 and R1, and the R1 distills of Qwen and Llama): the calls between the special tokens
  :data:`DEEPSEEK_CALLS_BEGIN` and :data:`DEEPSEEK_CALLS_END`, each :data:`DEEPSEEK_CALL_BEGIN`, ``function``,
  :data:`DEEPSEEK_SEP`, the name, the arguments in a fenced ``json`` block and :data:`DEEPSEEK_CALL_END`.
- ``pythonic`` (Llama 3.2/4-style templates that ask for Python calls): the whole reply is a Python list of calls,
  ``[get_weather(city="Paris"), is_noon()]``, values as Python literals (``True``, ``False``, ``None``).
- ``deepseek-v3.1`` (DeepSeek V3.1): V3's markers without the function type and the fenced block,
  :data:`DEEPSEEK_CALL_BEGIN`, the name, :data:`DEEPSEEK_SEP`, the arguments as JSON and :data:`DEEPSEEK_CALL_END`.
- ``phi4-mini`` (Phi-4-mini): ``functools[{"name": "get_weather", "arguments": {...}}]``; the template takes the
  tools as a JSON string on the system message and shows earlier calls as the assistant's text.
- ``command-r7b`` (Command R7B): ``<|START_ACTION|>[{"tool_call_id": "0", "tool_name": "get_weather", "parameters":
  {...}}]<|END_ACTION|>``.

When the model's chat template knows about tools (it references ``tools``), the template presents them, exactly as
``transformers`` would; otherwise the engine adds the Hermes instructions to the system message and the model calls
tools in the Hermes format. Calls are constrained by a grammar: with ``tool_choice`` ``auto`` the constraint starts
once the model writes the format's marker (``<tool_call>``, ``[TOOL_CALLS]``, ``<|tool_call|>``, DeepSeek's; for
Llama 3 and Python calls once the reply starts with ``{`` or ``[``), before which it may answer in text, so a call
always has a known function name and arguments that fit its parameter schema; with ``required`` or a named tool the
output must be exactly one call. A :class:`ToolChoice` may also narrow the tools the model may call (``allowed``,
while every tool stays in the prompt) and limit an answer to one call (``parallel``); a ``strict`` tool's arguments
are constrained by the whole schema rather than leniently.

Tool call ids are derived from the request and the call's position, never from a clock or random source. A template
that insists on its own id shape (Mistral's nine letters and digits) sees ids derived from the caller's ids.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from etalii_dllm.chat import TOOL_CALL_CLOSE, TOOL_CALL_OPEN, ChatMessage
from etalii_dllm.grammar import ANNOTATIONS, Grammar, GrammarError


@dataclass(frozen=True)
class Tool:
    name: str
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    """JSON schema of the arguments object."""
    strict: bool = False
    """The arguments must satisfy the whole schema (else the grammar is lenient: keywords it cannot check are
    ignored). Not shown to the model."""

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
    allowed: tuple[str, ...] | None = None
    """The tools the model may call (``auto`` and ``required``); every tool is still shown to it."""
    parallel: bool = True
    """Several calls in one answer (``False``: at most one)."""

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "none", "required", "named"):
            raise ValueError(f"unknown tool_choice {self.mode!r}")
        if (self.mode == "named") != (self.name is not None):
            raise ValueError("a named tool_choice needs exactly one function name")
        if self.allowed is not None:
            if self.mode not in ("auto", "required"):
                raise ValueError("allowed tools need the mode 'auto' or 'required'")
            if not self.allowed:
                raise ValueError("allowed tools need at least one tool")
            if len(set(self.allowed)) != len(self.allowed):
                raise ValueError("allowed tools must be unique")

    def callable(self, tools: Sequence[Tool]) -> list[Tool]:
        """The tools the model may call, in their order in ``tools``."""
        if self.mode == "named":
            return [t for t in tools if t.name == self.name]
        if self.allowed is not None:
            return [t for t in tools if t.name in self.allowed]
        return list(tools)

    def record(self) -> dict[str, Any]:
        """The choice as JSON for receipts (the newer fields only when set, so older receipts stay the same)."""
        record: dict[str, Any] = {"mode": self.mode, "name": self.name}
        if self.allowed is not None:
            record["allowed"] = list(self.allowed)
        if not self.parallel:
            record["parallel"] = False
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ToolChoice:
        allowed = record.get("allowed")
        return cls(
            record["mode"],
            record["name"],
            None if allowed is None else tuple(allowed),
            bool(record.get("parallel", True)),
        )


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
    style: str = "json"
    """How a call is written: ``json`` objects, ``xml`` parameters, ``deepseek`` markers or ``pythonic`` calls."""
    markers: tuple[str, ...] = ()
    """Special tokens the format writes, kept visible in decoded text (with :attr:`open`)."""
    string_arguments: bool = False
    """The template expects earlier calls' arguments as JSON text rather than objects."""
    name_key: str = "name"
    """The key of the function name in a JSON call (Command R7B: ``tool_name``)."""
    id_key: str | None = None
    """A key the model writes with each JSON call for its own call number (Command R7B: ``tool_call_id``)."""
    system_tools: bool = False
    """The template takes the tools as a JSON string on the system message (Phi-4-mini), not as ``tools``."""
    content_calls: bool = False
    """The template does not render tool calls: earlier calls are written into the assistant's text."""

    @property
    def bare(self) -> str:
        """The first character of a reply that is a call without a marker (``""`` for formats with a marker)."""
        if self.open:
            return ""
        return "[" if self.style == "pythonic" else "{"


DEEPSEEK_CALLS_BEGIN = "<\uff5ctool\u2581calls\u2581begin\uff5c>"
DEEPSEEK_CALLS_END = "<\uff5ctool\u2581calls\u2581end\uff5c>"
DEEPSEEK_CALL_BEGIN = "<\uff5ctool\u2581call\u2581begin\uff5c>"
DEEPSEEK_CALL_END = "<\uff5ctool\u2581call\u2581end\uff5c>"
DEEPSEEK_SEP = "<\uff5ctool\u2581sep\uff5c>"

HERMES = ToolFormat("hermes", TOOL_CALL_OPEN, TOOL_CALL_CLOSE)
LLAMA3 = ToolFormat("llama3", "", arguments="parameters")
MISTRAL = ToolFormat("mistral", "[TOOL_CALLS]", listed=True, id_length=9)
GRANITE = ToolFormat("granite", "<|tool_call|>", listed=True)
XML = ToolFormat("xml", TOOL_CALL_OPEN, TOOL_CALL_CLOSE, style="xml")
DEEPSEEK = ToolFormat(
    "deepseek",
    DEEPSEEK_CALLS_BEGIN,
    DEEPSEEK_CALLS_END,
    listed=True,
    style="deepseek",
    markers=(DEEPSEEK_CALLS_END, DEEPSEEK_CALL_BEGIN, DEEPSEEK_CALL_END, DEEPSEEK_SEP),
    string_arguments=True,
)
PYTHONIC = ToolFormat("pythonic", "", style="pythonic")
DEEPSEEK_V31 = ToolFormat(
    "deepseek-v3.1",
    DEEPSEEK_CALLS_BEGIN,
    DEEPSEEK_CALLS_END,
    listed=True,
    style="deepseek-v3.1",
    markers=DEEPSEEK.markers,
    string_arguments=True,
)
PHI4_MINI = ToolFormat("phi4-mini", "functools", listed=True, system_tools=True, content_calls=True)
COMMAND_R7B_OPEN = "<|START_ACTION|>"
COMMAND_R7B_CLOSE = "<|END_ACTION|>"
COMMAND_R7B = ToolFormat(
    "command-r7b",
    COMMAND_R7B_OPEN,
    COMMAND_R7B_CLOSE,
    arguments="parameters",
    listed=True,
    markers=(COMMAND_R7B_CLOSE,),
    name_key="tool_name",
    id_key="tool_call_id",
)
FORMATS = {
    f.name: f for f in (HERMES, LLAMA3, MISTRAL, GRANITE, XML, DEEPSEEK, PYTHONIC, DEEPSEEK_V31, PHI4_MINI, COMMAND_R7B)
}
"""The tool call formats, by name."""


def detect_format(template_source: str | None) -> ToolFormat:
    """The tool call format of a model, from its chat template: the template's own call marker decides, in a fixed
    order (``[TOOL_CALLS]``, DeepSeek's calls marker (V3 and R1 with a fenced ``json`` block, else V3.1), Command
    R7B's ``<|START_ACTION|>``, Phi-4-mini's ``<|tool|>``, ``<|tool_call|>``, Qwen3-Coder's ``<function=``,
    ``<tool_call>``, the Python call example ``func_name1(``); Llama 3's bare calls are recognised by its ``ipython``
    role and ``"parameters"`` key. Templates that do not present tools get the Hermes instructions, and so the Hermes
    format."""
    if not template_supports_tools(template_source):
        return HERMES
    assert template_source is not None
    markers = (
        ("[TOOL_CALLS]", MISTRAL),
        (DEEPSEEK_CALLS_BEGIN, DEEPSEEK if "```json" in template_source else DEEPSEEK_V31),
        (COMMAND_R7B_OPEN, COMMAND_R7B),
        ("<|tool|>", PHI4_MINI),
        ("<|tool_call|>", GRANITE),
        ("<function=", XML),
        (TOOL_CALL_OPEN, HERMES),
        ("func_name1(", PYTHONIC),
    )
    for marker, found in markers:
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
    for name in choice.allowed or ():
        if name not in names:
            raise ValueError(f"allowed tools name an unknown tool {name!r}")
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


def template_messages(
    messages: Sequence[ChatMessage], fmt: ToolFormat = HERMES, tools: Sequence[Tool] = ()
) -> list[dict[str, Any]]:
    """Messages in the Hugging Face form tool-aware chat templates expect (arguments as objects; call ids in the
    shape the format's template accepts). For a format whose template takes the tools on the system message
    (:attr:`ToolFormat.system_tools`), ``tools`` go there; earlier calls a template cannot render
    (:attr:`ToolFormat.content_calls`) are written into the assistant's text, in the format."""

    def ident(call_id: str) -> str:
        return template_id(call_id, fmt.id_length) if fmt.id_length is not None else call_id

    result: list[dict[str, Any]] = []
    for message in messages:
        entry: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls and fmt.content_calls:
            listed = [{fmt.name_key: c.name, fmt.arguments: c.arguments_object()} for c in message.tool_calls]
            entry["content"] = message.content + fmt.open + json.dumps(listed, ensure_ascii=False)
        elif message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": ident(call.id),
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments if fmt.string_arguments else call.arguments_object(),
                    },
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id is not None:
            entry["tool_call_id"] = ident(message.tool_call_id)
        if message.name is not None:
            entry["name"] = message.name
        result.append(entry)
    if tools and fmt.system_tools:
        if not result or result[0]["role"] != "system":
            result.insert(0, {"role": "system", "content": ""})
        result[0]["tools"] = json.dumps([t.to_openai()["function"] for t in tools], ensure_ascii=False)
    return result


CALL_NUMBERS = [str(n) for n in range(10)]
"""The call numbers a format with :attr:`ToolFormat.id_key` may write ("0" to "9"; their value is not used)."""


def call_schema(tool: Tool, fmt: ToolFormat = HERMES) -> dict[str, Any]:
    parameters = dict(tool.parameters) if tool.parameters else {"type": "object", "properties": {}}
    if parameters.get("type", "object") != "object":
        raise ValueError(f"the parameters of tool {tool.name!r} must be a JSON schema of type object")
    numbered = {fmt.id_key: {"enum": CALL_NUMBERS}} if fmt.id_key else {}
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {**numbered, fmt.name_key: {"const": tool.name}, fmt.arguments: parameters},
        "required": [*numbered, fmt.name_key, fmt.arguments],
    }
    # The arguments' definitions, where their "#/$defs/..." references look from the call object.
    schema.update({key: parameters[key] for key in _DEFINITIONS if key in parameters})
    return schema


_DEFINITIONS = ("$defs", "definitions")
_ARGUMENTS_OBJECT = frozenset({"type", "properties", "required", "additionalProperties", *_DEFINITIONS})
"""The keywords of a strict tool's arguments object that formats writing arguments one by one can enforce
(``additionalProperties`` holds trivially: only the declared arguments are written)."""
_RAW_STRING = frozenset({"type", "pattern", "minLength", "maxLength", *_DEFINITIONS})
"""The keywords of a strict string parameter written as raw XML text."""


def _parameters(tool: Tool) -> tuple[list[tuple[str, dict[str, Any], bool]], Mapping[str, Any]]:
    """A tool's parameters as ``(name, schema, required)`` in schema order, and its parameter schema."""
    schema = call_schema(tool)["properties"]["arguments"]
    required = set(schema.get("required", ()))
    properties = schema.get("properties") or {}
    return [
        (name, dict(sub) if isinstance(sub, Mapping) else {}, name in required) for name, sub in properties.items()
    ], schema


def _written_parameters(tool: Tool, fmt: ToolFormat) -> list[tuple[str, dict[str, Any], bool]]:
    """The parameters of a format that writes arguments one by one (XML, Python calls). For a strict tool the
    arguments object may only use what that enforces, and each parameter carries the definitions it may refer to."""
    parameters, schema = _parameters(tool)
    if not tool.strict:
        return parameters
    extra = sorted(set(schema) - ANNOTATIONS - _ARGUMENTS_OBJECT)
    if extra:
        raise GrammarError(
            f"strict tool {tool.name!r}: the {fmt.name} format writes arguments one by one and cannot enforce "
            f"{', '.join(extra)} on the arguments object"
        )
    missing = sorted(set(schema.get("required", ())) - {name for name, _, _ in parameters})
    if missing:
        raise GrammarError(f"strict tool {tool.name!r}: required arguments without properties: {', '.join(missing)}")
    definitions = {key: schema[key] for key in _DEFINITIONS if key in schema}
    return [(name, {**definitions, **sub}, needed) for name, sub, needed in parameters]


def _keyword_arguments(parameters: list[tuple[str, dict[str, Any], bool]], written: Any, separator: str) -> Grammar:
    """Parameters in schema order, each written by ``written(name, schema)``, required ones always and optional
    ones when chosen, ``separator`` between them."""
    parts: list[tuple[Grammar, bool]] = [(written(name, sub), needed) for name, sub, needed in parameters]
    if not separator:
        return Grammar.sequence(g if needed else Grammar.optional(g) for g, needed in parts)
    # With a separator, the arguments after the first written one each start with it: a small automaton over the
    # positions (``rest[i]``: the arguments from ``i`` on, after at least one was written).
    comma = Grammar.literal(separator)
    rest: list[Grammar] = [Grammar([])] * (len(parts) + 1)
    first: list[Grammar] = [Grammar([])] * (len(parts) + 1)
    for i in range(len(parts) - 1, -1, -1):
        grammar, needed = parts[i]
        with_it = Grammar.sequence([comma, grammar, rest[i + 1]])
        rest[i] = with_it if needed else Grammar.one_of([with_it, rest[i + 1]])
        starts = Grammar.sequence([grammar, rest[i + 1]])
        first[i] = starts if needed else Grammar.one_of([starts, first[i + 1]])
    return first[0]


_XML_TEXT = r"[^<]*"
"""A string parameter's raw text in the XML format (no ``<``, so it cannot run into the closing tag)."""


def _xml_value(sub: Mapping[str, Any], strict: bool) -> Grammar:
    if sub.get("type") == "string" and strict:
        extra = sorted(set(sub) - ANNOTATIONS - _RAW_STRING - {"enum", "const"})
        if extra:
            raise GrammarError(f"a strict string parameter in the xml format cannot use {', '.join(extra)}")
    if sub.get("type") == "string" and (isinstance(sub.get("enum"), list) or (strict and "const" in sub)):
        values = [v for v in (sub["enum"] if "enum" in sub else [sub["const"]]) if isinstance(v, str)]
        if strict:
            rest = Grammar.raw_string(sub, _XML_TEXT).matcher()
            values = [v for v in values if rest.matches(v.encode("utf-8"))]
            if "enum" in sub and "const" in sub:
                values = [v for v in values if v == sub["const"]]
        return Grammar.one_of([Grammar.literal(v) for v in values] or [Grammar([])])
    if sub.get("type") == "string":
        return Grammar.raw_string(sub, _XML_TEXT) if strict else Grammar.regex(_XML_TEXT)
    return Grammar.json_schema(sub, lenient=not strict)


def _xml_call(tool: Tool) -> Grammar:
    def parameter(name: str, sub: dict[str, Any]) -> Grammar:
        value = _xml_value(sub, tool.strict)
        return Grammar.sequence([Grammar.literal(f"<parameter={name}>\n"), value, Grammar.literal("\n</parameter>\n")])

    return Grammar.sequence(
        [
            Grammar.literal(f"\n<function={tool.name}>\n"),
            _keyword_arguments(_written_parameters(tool, XML), parameter, ""),
            Grammar.literal("</function>\n"),
        ]
    )


def _deepseek_call(tool: Tool) -> Grammar:
    arguments = Grammar.json_schema(call_schema(tool)["properties"]["arguments"], lenient=not tool.strict)
    return Grammar.sequence(
        [
            Grammar.literal(f"{DEEPSEEK_CALL_BEGIN}function{DEEPSEEK_SEP}{tool.name}\n```json\n"),
            arguments,
            Grammar.literal(f"\n```{DEEPSEEK_CALL_END}"),
        ]
    )


def _deepseek_v31_call(tool: Tool) -> Grammar:
    arguments = Grammar.json_schema(call_schema(tool)["properties"]["arguments"], lenient=not tool.strict)
    return Grammar.sequence(
        [
            Grammar.literal(f"{DEEPSEEK_CALL_BEGIN}{tool.name}{DEEPSEEK_SEP}"),
            arguments,
            Grammar.literal(DEEPSEEK_CALL_END),
        ]
    )


def _python_value(sub: Mapping[str, Any], strict: bool) -> Grammar:
    if sub.get("type") == "boolean":
        return Grammar.one_of([Grammar.literal("True"), Grammar.literal("False")])
    if sub.get("type") == "null":
        return Grammar.literal("None")
    return Grammar.json_schema(sub, lenient=not strict)


def _python_call(tool: Tool) -> Grammar:
    def argument(name: str, sub: dict[str, Any]) -> Grammar:
        return Grammar.sequence([Grammar.literal(f"{name}="), _python_value(sub, tool.strict)])

    return Grammar.sequence(
        [
            Grammar.literal(f"{tool.name}("),
            _keyword_arguments(_written_parameters(tool, PYTHONIC), argument, ", "),
            Grammar.literal(")"),
        ]
    )


def _json_call(tool: Tool, fmt: ToolFormat) -> Grammar:
    return Grammar.json_schema(call_schema(tool, fmt), lenient=not tool.strict)


def _calls(tools: Sequence[Tool], fmt: ToolFormat, single: bool) -> Grammar:
    """One call to any of ``tools``, or for listed formats a non-empty list of them (of one, when ``single``)."""
    if fmt.style == "xml":
        return Grammar.one_of([_xml_call(t) for t in tools])
    if fmt.style == "deepseek":
        call = Grammar.one_of([_deepseek_call(t) for t in tools])
        return call if single else Grammar.repeat(call, Grammar.literal("\n"))
    if fmt.style == "deepseek-v3.1":
        call = Grammar.one_of([_deepseek_v31_call(t) for t in tools])
        return call if single else Grammar.repeat(call)
    if fmt.style == "pythonic":
        call = Grammar.one_of([_python_call(t) for t in tools])
        calls = call if single else Grammar.repeat(call, Grammar.literal(", "))
        return Grammar.sequence([Grammar.literal("["), calls, Grammar.literal("]")])
    if fmt.listed and not any(t.strict for t in tools):
        items = {"anyOf": [call_schema(t, fmt) for t in tools]}
        array: dict[str, Any] = {"type": "array", "items": items, "minItems": 1, **({"maxItems": 1} if single else {})}
        return Grammar.json_schema(array, lenient=True)
    call = Grammar.choice([_json_call(t, fmt) for t in tools])
    if not fmt.listed:
        return call
    # Strict tools are compiled one by one (a lenient neighbour must not loosen them), so the list is spelled out.
    space = Grammar.whitespace()
    comma = Grammar.sequence([space, Grammar.literal(","), space])
    calls = call if single else Grammar.repeat(call, comma)
    return Grammar.sequence([Grammar.literal("["), space, calls, space, Grammar.literal("]")])


def call_grammar(tools: Sequence[Tool], fmt: ToolFormat = HERMES, *, single: bool = False) -> Grammar:
    """What follows the format's marker: the call (or call list, of one call when ``single``) and the closing
    marker."""
    if fmt.style in ("xml", "deepseek", "deepseek-v3.1"):
        return Grammar.sequence([_calls(tools, fmt, single), Grammar.literal(fmt.close)])
    parts = [Grammar.whitespace(), _calls(tools, fmt, single)]
    if fmt.close:
        parts += [Grammar.whitespace(), Grammar.literal(fmt.close)]
    return Grammar.sequence(parts)


def forced_grammar(tools: Sequence[Tool], choice: ToolChoice, fmt: ToolFormat = HERMES) -> Grammar:
    """Exactly one call (or call list) to the tools ``choice`` lets the model call: for ``required`` and named tool
    choices, and as the call branch of the others."""
    return Grammar.sequence(
        [Grammar.literal(fmt.open), call_grammar(choice.callable(tools), fmt, single=not choice.parallel)]
    )


FREE_TEXT = r"[ \t\r\n]*([^{ \t\r\n][\s\S]*)?"
"""Text that does not start like a JSON object (an answer, for formats whose calls are bare objects)."""
FREE_TEXT_PYTHONIC = r"[ \t\r\n]*([^\[ \t\r\n][\s\S]*)?"
"""Text that does not start like a list (an answer, for Python-style calls)."""


def auto_grammar(tools: Sequence[Tool], fmt: ToolFormat, choice: ToolChoice = AUTO) -> tuple[Grammar, str | None]:
    """The constraint for ``tool_choice`` ``auto`` and the marker that starts it (``None``: from the start). With
    ``choice.parallel`` off the constraint must also end the answer after the first call (``once``)."""
    if fmt.open:
        return call_grammar(choice.callable(tools), fmt, single=not choice.parallel), fmt.open
    free = FREE_TEXT_PYTHONIC if fmt.style == "pythonic" else FREE_TEXT
    return Grammar.either([Grammar.regex(free), forced_grammar(tools, choice, fmt)]), None


def _block(fmt: ToolFormat) -> re.Pattern[str]:
    return re.compile(re.escape(fmt.open) + r"(.*?)(?:" + re.escape(fmt.close) + r"|\Z)", re.DOTALL)


def _as_call(value: Any, tools: Mapping[str, Tool], name_key: str = "name") -> tuple[str, str] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict) or value.get(name_key) not in tools:
        return None
    arguments = value.get("arguments", value.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return str(value[name_key]), json.dumps(arguments, ensure_ascii=False)


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
    calls = [_as_call(item, tools, fmt.name_key) for item in items]
    after = rest[end:].strip()
    if fmt.close and after.startswith(fmt.close):  # text may follow a closed block
        after = ""
    if not calls or None in calls or after:
        return None
    return text[:cut].strip(), [call for call in calls if call is not None]


_XML_FUNCTION = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
_XML_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _xml_call_of(body: str, tools: Mapping[str, Tool]) -> tuple[str, str] | None:
    """The call in one XML block: string parameters as their text, others as JSON (as text when they are not)."""
    match = _XML_FUNCTION.fullmatch(body.strip())
    if match is None or match.group(1) not in tools:
        return None
    properties = _parameters(tools[match.group(1)])[1].get("properties") or {}
    arguments: dict[str, Any] = {}
    for name, value in _XML_PARAMETER.findall(match.group(2)):
        sub = properties.get(name)
        if isinstance(sub, Mapping) and sub.get("type") == "string":
            arguments[name] = value
            continue
        try:
            arguments[name] = json.loads(value)
        except json.JSONDecodeError:
            arguments[name] = value
    return match.group(1), json.dumps(arguments, ensure_ascii=False)


_DEEPSEEK_CALL = re.compile(
    re.escape(DEEPSEEK_CALL_BEGIN)
    + r"(?:function)?"
    + re.escape(DEEPSEEK_SEP)
    + r"([^\n]+)\n```(?:json)?\n(.*?)\n```"
    + re.escape(DEEPSEEK_CALL_END),
    re.DOTALL,
)


_DEEPSEEK_V31_CALL = re.compile(
    re.escape(DEEPSEEK_CALL_BEGIN) + r"([^\n<]+?)" + re.escape(DEEPSEEK_SEP) + r"(.*?)" + re.escape(DEEPSEEK_CALL_END),
    re.DOTALL,
)


def _deepseek_calls(
    text: str, tools: Mapping[str, Tool], pattern: re.Pattern[str] = _DEEPSEEK_CALL
) -> tuple[str, list[tuple[str, str]]] | None:
    """The text before the calls marker and the calls after it, when they are all calls of known tools."""
    cut = text.find(DEEPSEEK_CALLS_BEGIN)
    if cut < 0:
        return None
    rest = text[cut + len(DEEPSEEK_CALLS_BEGIN) :]
    end = rest.find(DEEPSEEK_CALLS_END)
    block = rest if end < 0 else rest[:end]
    calls = [_as_call({"name": name, "arguments": body}, tools) for name, body in pattern.findall(block)]
    if not calls or None in calls or pattern.sub("", block).strip():
        return None
    return text[:cut].strip(), [call for call in calls if call is not None]


def _python_literal(node: ast.expr, source: str) -> Any:
    """The value of a Python literal, with JSON's ``true``/``false``/``null`` accepted too (nested JSON values);
    double-quoted strings are read as JSON strings, as the grammar writes them."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        written = ast.get_source_segment(source, node) or ""
        if written.startswith('"') and not written.startswith('"""'):
            try:
                return json.loads(written)
            except json.JSONDecodeError:
                pass
        return node.value
    if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, (bool, int, float))):
        return node.value
    if isinstance(node, ast.Name) and node.id in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _python_literal(node.operand, source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return -value
    if isinstance(node, ast.List):
        return [_python_literal(item, source) for item in node.elts]
    if isinstance(node, ast.Dict) and None not in node.keys:
        keys = [_python_literal(key, source) for key in node.keys if key is not None]
        if all(isinstance(key, str) for key in keys):
            return dict(zip(keys, (_python_literal(value, source) for value in node.values), strict=True))
    raise ValueError("not a literal")


def _python_calls(text: str, tools: Mapping[str, Tool]) -> list[tuple[str, str]] | None:
    """The calls of a reply that is a Python list of calls to known tools with literal keyword arguments."""
    source = text.strip()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)  # JSON escapes such as \/ (read as JSON below)
            tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError):
        return None
    if not isinstance(tree.body, ast.List) or not tree.body.elts:
        return None
    calls = []
    for item in tree.body.elts:
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Name) or item.args:
            return None
        if item.func.id not in tools or any(keyword.arg is None for keyword in item.keywords):
            return None
        try:
            arguments = {keyword.arg: _python_literal(keyword.value, source) for keyword in item.keywords}
        except ValueError:
            return None
        calls.append((item.func.id, json.dumps(arguments, ensure_ascii=False)))
    return calls


def parse_calls(text: str, tools: Sequence[Tool], fmt: ToolFormat = HERMES) -> tuple[str, list[tuple[str, str]]]:
    """Splits generated text into answer text (stripped) and ``(name, arguments JSON)`` calls in the format
    ``fmt``. Blocks that do not hold calls to known tools stay in the text. Without blocks, a reply that is nothing
    but a JSON call object (the Llama 3 style) counts as a call in every format."""
    by_name = {t.name: t for t in tools}
    calls: list[tuple[str, str]] = []
    kept: list[str] = []
    if fmt.style in ("deepseek", "deepseek-v3.1"):
        found = _deepseek_calls(text, by_name, _DEEPSEEK_CALL if fmt.style == "deepseek" else _DEEPSEEK_V31_CALL)
        if found is not None:
            return found
    elif fmt.style == "pythonic":
        listed = _python_calls(text, by_name) if text.strip().startswith("[") else None
        if listed is not None:
            return "", listed
    elif fmt.listed:
        found = _listed_calls(text, fmt, by_name)
        if found is not None:
            return found
    elif fmt.open:
        position = 0
        for match in _block(fmt).finditer(text):
            body = match.group(1).strip()
            call = _xml_call_of(body, by_name) if fmt.style == "xml" else _as_call(body, by_name)
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
