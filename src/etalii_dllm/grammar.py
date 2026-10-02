"""Constrained decoding: byte-level JSON grammars and the token masks they induce.

A :class:`Grammar` describes the bytes the model may produce: literal text, JSON values of a JSON schema, text
matching a regular expression (:mod:`etalii_dllm.regexp`), or a sequence of those (a tool call is ``<tool_call>``
+ a JSON object + ``</tool_call>``). It is recognised by a small nondeterministic pushdown automaton over bytes, so
tokens that end mid-character (byte-level BPE) are handled exactly.

:class:`TokenConstraint` turns a grammar into the set of tokens allowed at each step. Every token's bytes are put in
a trie once per tokenizer; a depth-first walk over the trie, pruned as soon as the automaton rejects a prefix, finds
the allowed tokens. Automaton states are interned to integers and their byte transitions memoised, so the walk is
mostly dictionary lookups.

Determinism: the automaton is a pure function of the grammar and the bytes; stacks are kept in insertion order
(never in set iteration order), and the allowed tokens are returned in ascending id order.

Supported JSON schema keywords: ``type`` (a name or a list of names), ``properties``, ``required``,
``additionalProperties`` (``false``, or a free-form object when there are no ``properties``), ``items``,
``minItems``, ``maxItems``, ``enum``, ``const``, ``anyOf``, ``oneOf`` (treated as ``anyOf``), ``allOf`` with a
single schema, ``$ref`` to ``#/$defs/...`` or ``#/definitions/...``, and ``nullable``. Value constraints are
compiled to byte automata (:mod:`etalii_dllm.regexp`): on strings ``pattern`` (found anywhere unless anchored by a
leading ``^`` or trailing ``$``), the :data:`FORMATS` and ``minLength``/``maxLength`` in code points (such strings are
written without escape sequences, so they hold no quote, backslash or control character); on integers ``minimum``,
``maximum``, ``exclusiveMinimum`` and ``exclusiveMaximum``. Annotations (``title``, ``description``, ``default``,
``examples``, other ``format`` values, ``$schema``, ``$id``, ``strict``) are ignored. Keywords the automaton does not
check (``multipleOf``, ``uniqueItems``, bounds on non-integer numbers, ...) are rejected with a :class:`GrammarError`
rather than silently ignored; ``lenient`` grammars (tool parameters) ignore them and the value constraints.

Object properties are generated in the order the schema lists them; optional properties may be left out. Between
tokens the model may write up to :data:`MAX_WHITESPACE` whitespace bytes, enough for pretty-printed JSON but not
for endless padding.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

MAX_WHITESPACE = 16
"""Longest run of whitespace allowed between JSON tokens."""

_WHITESPACE = frozenset(b" \t\n\r")
_DIGITS = frozenset(b"0123456789")
_HEX = frozenset(b"0123456789abcdefABCDEF")
_ESCAPES = frozenset(b'"\\/bfnrt')

_ANNOTATIONS = frozenset(
    {"title", "description", "default", "examples", "format", "$schema", "$id", "$comment", "strict", "deprecated",
     "readOnly", "writeOnly"}
)  # fmt: skip
_STRUCTURE = frozenset(
    {"type", "properties", "required", "additionalProperties", "items", "minItems", "maxItems", "enum", "const",
     "anyOf", "oneOf", "allOf", "$ref", "$defs", "definitions", "nullable"}
)  # fmt: skip
_STRING_KEYWORDS = ("pattern", "minLength", "maxLength")
_BOUNDS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
_CONSTRAINTS = frozenset({*_STRING_KEYWORDS, *_BOUNDS})
"""Value constraints compiled to byte automata (:func:`string_automaton`, :func:`integer_automaton`)."""

_CONTENT = r'[^"\\\x00-\x1f]*'
"""JSON string content written without escapes: no quote, backslash or control character."""
_DATE = r"\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
_TIME = r"(?:[01]\d|2[0-3]):[0-5]\d:(?:[0-5]\d|60)(?:\.\d+)?(?:[Zz]|[+-](?:[01]\d|2[0-3]):[0-5]\d)"
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
FORMATS = {
    "date": _DATE,
    "time": _TIME,
    "date-time": _DATE + "[Tt]" + _TIME,
    "uuid": r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
    "ipv4": _OCTET + r"(?:\." + _OCTET + "){3}",
}
"""The ``format`` values that constrain strings (RFC 3339 dates and times, with days 01 to 31 in any month; other
formats stay annotations)."""


class GrammarError(ValueError):
    """The schema uses a feature constrained decoding does not support."""


# -- schema nodes -------------------------------------------------------------------------------------------------


class _Node:
    __slots__ = ()


class _String(_Node):
    __slots__ = ()


class _Number(_Node):
    __slots__ = ("integer",)

    def __init__(self, integer: bool) -> None:
        self.integer = integer


class _Literals(_Node):
    """One of a fixed set of JSON texts (``enum``, ``const``, booleans, ``null``)."""

    __slots__ = ("options",)

    def __init__(self, options: Iterable[bytes]) -> None:
        self.options = tuple(dict.fromkeys(options))


class _Union(_Node):
    __slots__ = ("options",)

    def __init__(self, options: Sequence[_Node]) -> None:
        self.options = tuple(options)


class _Object(_Node):
    """``properties`` as ``(json-encoded name, schema, required)``; ``free`` objects take any members."""

    __slots__ = ("free", "properties")

    def __init__(self, properties: Sequence[tuple[bytes, _Node, bool]], free: bool) -> None:
        self.properties = tuple(properties)
        self.free = free


class _Array(_Node):
    __slots__ = ("items", "maximum", "minimum")

    def __init__(self, items: _Node, minimum: int, maximum: int | None) -> None:
        self.items = items
        self.minimum = minimum
        self.maximum = maximum


class _Text(_Node):
    """A JSON string whose content (written without escapes) the automaton ``dfa`` matches."""

    __slots__ = ("dfa",)

    def __init__(self, dfa: Any) -> None:
        self.dfa = dfa


class _Digits(_Node):
    """An integer whose JSON text the automaton ``dfa`` matches."""

    __slots__ = ("dfa",)

    def __init__(self, dfa: Any) -> None:
        self.dfa = dfa


class _Any(_Node):
    """Any JSON value."""

    __slots__ = ()


_ANY = _Any()
_STRING = _String()
_FREE_OBJECT = _Object((), free=True)
_ANY_ARRAY = _Array(_ANY, 0, None)
_ANY_SCALARS = (_STRING, _Number(integer=False), _Literals([b"true", b"false", b"null"]))


def _encode(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _top_level_alternation(pattern: str) -> bool:
    """Whether ``pattern`` has a ``|`` outside every group and class."""
    depth, index, in_class = 0, 0, False
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            index += 2
            continue
        if in_class:
            in_class = char != "]"
        elif char == "[":
            in_class = True
            if pattern[index + 1 : index + 2] == "]":  # a leading "]" is a literal
                index += 1
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "|" and depth == 0:
            return True
        index += 1
    return False


def _searched(pattern: str) -> str:
    """A JSON Schema ``pattern`` (found anywhere in the string unless anchored by a leading ``^`` or a trailing
    ``$``) as a full-match regex."""
    body = pattern[1:] if pattern.startswith("^") else pattern
    anchored_end = False
    if body.endswith("$"):
        backslashes = len(body[:-1]) - len(body[:-1].rstrip("\\"))
        anchored_end = backslashes % 2 == 0  # an escaped "\$" is a literal dollar sign
    if anchored_end:
        body = body[:-1]
    anchored = pattern.startswith("^") or anchored_end
    if anchored and _top_level_alternation(body):
        raise GrammarError(f"pattern {pattern!r}: anchors around a top-level alternation are ambiguous; group it")
    anything = r"[\s\S]*"
    return f"{'' if pattern.startswith('^') else anything}(?:{body}){'' if anchored_end else anything}"


@functools.lru_cache(maxsize=256)
def string_automaton(pattern: str | None, text_format: str | None, min_length: int, max_length: int | None) -> Any:
    """The byte automaton of string content written without escapes that matches ``pattern`` (JSON Schema
    semantics), the format and the length bounds (in code points): the intersection of their automata, trimmed so
    that every state can still reach a match."""
    from etalii_dllm.regexp import Counted, compile_regex, intersect

    dfa = compile_regex(_CONTENT)
    if pattern is not None:
        dfa = intersect(dfa, compile_regex(_searched(pattern)))
    if text_format is not None:
        dfa = intersect(dfa, compile_regex(FORMATS[text_format]))
    if min_length or max_length is not None:
        return Counted(dfa, min_length, max_length)
    return dfa


def _same_length(low: str, high: str) -> list[str]:
    """Regex branches for the digit strings from ``low`` to ``high`` (of equal length)."""
    if low == high:
        return [low]
    if len(low) == 1:
        return [f"[{low}-{high}]"]
    rest = len(low) - 1
    if low[1:] == "0" * rest and high[1:] == "9" * rest:
        return [f"[{low[0]}-{high[0]}]\\d{{{rest}}}"]
    if low[0] == high[0]:
        return [low[0] + "(?:" + "|".join(_same_length(low[1:], high[1:])) + ")"]
    branches = [low[0] + "(?:" + "|".join(_same_length(low[1:], "9" * rest)) + ")"]
    if int(high[0]) - int(low[0]) > 1:
        branches.append(f"[{int(low[0]) + 1}-{int(high[0]) - 1}]\\d{{{rest}}}")
    branches.append(high[0] + "(?:" + "|".join(_same_length("0" * rest, high[1:])) + ")")
    return branches


def _magnitudes(low: int, high: int | None) -> list[str]:
    """Regex branches for the decimal texts (no leading zeros) of the integers ``low .. high`` (``None``: no end)."""
    end = high if high is not None else 10 ** len(str(low)) - 1
    branches: list[str] = []
    for length in range(len(str(low)), len(str(end)) + 1):
        first = max(low, 10 ** (length - 1) if length > 1 else 0)
        last = min(end, 10**length - 1)
        if first <= last:
            branches += _same_length(str(first), str(last))
    if high is None:
        branches.append(f"[1-9]\\d{{{len(str(low))},}}")
    return branches


@functools.lru_cache(maxsize=256)
def integer_automaton(low: int | None, high: int | None) -> Any:
    """The byte automaton of the JSON texts of the integers ``low .. high`` (``None``: unbounded); ``-0`` is not
    one of them."""
    from etalii_dllm.regexp import compile_regex

    if low is not None and high is not None and low > high:
        raise GrammarError(f"no integer lies between {low} and {high}")
    branches: list[str] = []
    if high is None or high >= 0:
        branches += _magnitudes(max(low, 0) if low is not None else 0, high)
    if low is None or low < 0:
        closest = min(high, -1) if high is not None else -1
        branches.append("-(?:" + "|".join(_magnitudes(-closest, None if low is None else -low)) + ")")
    return compile_regex("|".join(branches))


def _bounds(schema: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """The integer range of ``minimum``/``maximum``/``exclusiveMinimum``/``exclusiveMaximum`` (numbers, or the
    draft 4 booleans)."""
    low = high = None

    def number(name: str) -> float:
        value = schema[name]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise GrammarError(f"'{name}' must be a finite number")
        return value

    if "minimum" in schema:
        low = math.ceil(number("minimum"))
        if schema.get("exclusiveMinimum") is True and low == number("minimum"):
            low += 1
    if "maximum" in schema:
        high = math.floor(number("maximum"))
        if schema.get("exclusiveMaximum") is True and high == number("maximum"):
            high -= 1
    if "exclusiveMinimum" in schema and not isinstance(schema["exclusiveMinimum"], bool):
        bound = math.floor(number("exclusiveMinimum")) + 1
        low = bound if low is None else max(low, bound)
    if "exclusiveMaximum" in schema and not isinstance(schema["exclusiveMaximum"], bool):
        bound = math.ceil(number("exclusiveMaximum")) - 1
        high = bound if high is None else min(high, bound)
    return low, high


class _SchemaCompiler:
    def __init__(self, root: Mapping[str, Any], lenient: bool) -> None:
        self._root = root
        self._lenient = lenient
        self._refs: dict[str, _Node] = {}

    def compile(self, schema: Any) -> _Node:
        if schema is True or schema == {}:
            return _ANY
        if not isinstance(schema, Mapping):
            raise GrammarError(f"a schema must be an object, got {schema!r}")
        unsupported = sorted(set(schema) - _ANNOTATIONS - _STRUCTURE - _CONSTRAINTS)
        if unsupported and not self._lenient:
            names = ", ".join(unsupported)
            raise GrammarError(f"JSON schema keyword(s) not supported by constrained decoding: {names}")
        node = self._compile(schema)
        if schema.get("nullable") is True:
            node = _Union([node, _Literals([b"null"])])
        return node

    def _compile(self, schema: Mapping[str, Any]) -> _Node:
        if "$ref" in schema:
            return self._reference(str(schema["$ref"]))
        if "const" in schema:
            return _Literals([_encode(schema["const"])])
        if "enum" in schema:
            values = list(schema["enum"])
            if not values:
                raise GrammarError("'enum' must not be empty")
            return _Literals(_encode(v) for v in values)
        for keyword in ("anyOf", "oneOf"):
            if keyword in schema:
                return _Union([self.compile(s) for s in schema[keyword]])
        if "allOf" in schema:
            if len(schema["allOf"]) != 1:
                raise GrammarError("'allOf' is only supported with a single schema")
            return self.compile(schema["allOf"][0])
        kind = schema.get("type")
        if isinstance(kind, list):
            return _Union([self._typed(str(k), schema) for k in kind])
        if kind is None:
            if "properties" in schema:
                kind = "object"
            elif "items" in schema:
                kind = "array"
            else:
                return _ANY
        return self._typed(str(kind), schema)

    def _typed(self, kind: str, schema: Mapping[str, Any]) -> _Node:
        if kind == "string":
            text_format = schema.get("format") if schema.get("format") in FORMATS else None
            if self._lenient or (text_format is None and not any(k in schema for k in _STRING_KEYWORDS)):
                return _STRING
            pattern = schema.get("pattern")
            if pattern is not None and not isinstance(pattern, str):
                raise GrammarError("'pattern' must be a string")
            min_length, max_length = int(schema.get("minLength", 0)), schema.get("maxLength")
            if max_length is not None and int(max_length) < min_length:
                raise GrammarError("'maxLength' is smaller than 'minLength'")
            maximum = None if max_length is None else int(max_length)
            return _Text(string_automaton(pattern, text_format, min_length, maximum))
        if kind in ("number", "integer"):
            if self._lenient or not any(k in schema for k in _BOUNDS):
                return _Number(integer=kind == "integer")
            if kind == "number":
                raise GrammarError("minimum, maximum and their exclusive forms are supported on integers only")
            return _Digits(integer_automaton(*_bounds(schema)))
        if kind == "boolean":
            return _Literals([b"true", b"false"])
        if kind == "null":
            return _Literals([b"null"])
        if kind == "array":
            items = self.compile(schema.get("items", True))
            minimum = int(schema.get("minItems", 0))
            maximum = schema.get("maxItems")
            if maximum is not None and int(maximum) < minimum:
                raise GrammarError("'maxItems' is smaller than 'minItems'")
            return _Array(items, minimum, None if maximum is None else int(maximum))
        if kind == "object":
            properties = schema.get("properties") or {}
            required = set(schema.get("required") or ())
            unknown = sorted(required - set(properties))
            if unknown and properties:
                raise GrammarError(f"required properties without a schema: {', '.join(unknown)}")
            if not properties:
                if schema.get("additionalProperties", True) is False:
                    return _Object((), free=False)
                return _FREE_OBJECT
            # With declared properties the model writes exactly those (``additionalProperties`` is not used).
            return _Object(
                [(_encode(str(name)), self.compile(sub), name in required) for name, sub in properties.items()],
                free=False,
            )
        raise GrammarError(f"unknown JSON schema type {kind!r}")

    def _reference(self, reference: str) -> _Node:
        node = self._refs.get(reference)
        if node is not None:
            return node
        prefixes = ("#/$defs/", "#/definitions/")
        if reference == "#":
            target: Any = self._root
        elif reference.startswith(prefixes):
            section, _, name = reference[2:].partition("/")
            target = (self._root.get(section) or {}).get(name)
        else:
            raise GrammarError(f"only local $ref values are supported, got {reference!r}")
        if target is None:
            raise GrammarError(f"unresolved $ref {reference!r}")
        # Recursive schemas: a placeholder union is filled in after compiling, so references to it resolve lazily.
        placeholder = _Union(())
        self._refs[reference] = placeholder
        placeholder.options = (self.compile(target),)
        return placeholder


# -- automaton ----------------------------------------------------------------------------------------------------
#
# A stack is a persistent linked list ``(item, rest)`` with ``None`` for the empty stack; items are tuples whose first
# element is a tag. Consuming items (literal, alternatives, whitespace, string, number) read bytes; the others
# expand into consuming items without reading.

_LIT, _ALT, _WS, _STR, _NUM, _VALUE, _OBJ, _ARR, _RE = range(9)

Stack = tuple[Any, Any] | None

# String sub-states: 0 plain, 1 after a backslash, 11..14 hex digits of \uXXXX left (10 + k), 21..23 UTF-8
# continuation bytes left (20 + k), 31..34 the restricted second byte of some three- and four-byte sequences.
# Number phases: 0 start, 1 after '-', 2 after a leading 0, 3 integer digits, 4 after '.', 5 fraction digits,
# 6 after 'e', 7 after the exponent sign, 8 exponent digits.
_NUMBER_ACCEPTING = frozenset({2, 3, 5, 8})


def _push(rest: Stack, items: Sequence[tuple[Any, ...]]) -> Stack:
    stack = rest
    for item in reversed(items):
        stack = (item, stack)
    return stack


def _value(node: _Node) -> tuple[Any, ...]:
    return (_VALUE, node)


def _literal(data: bytes) -> tuple[Any, ...]:
    return (_LIT, data, 0)


_WS_ITEM = (_WS, MAX_WHITESPACE)
_COMMA = _literal(b",")
_COLON = _literal(b":")


def _expand(node: _Node, rest: Stack) -> list[Stack]:
    if isinstance(node, _String):
        return [((_LIT, b'"', 0), ((_STR, 0), rest))]
    if isinstance(node, _Number):
        return [((_NUM, node.integer, 0), rest)]
    if isinstance(node, _Text):
        return [_push(rest, [_literal(b'"'), (_RE, node.dfa, 0), _literal(b'"')])]
    if isinstance(node, _Digits):
        return [((_RE, node.dfa, 0), rest)]
    if isinstance(node, _Literals):
        return [((_ALT, node.options, 0), rest)] if node.options else []
    if isinstance(node, _Union):
        return [s for option in node.options for s in _expand(option, rest)]
    if isinstance(node, _Object):
        return [_push(rest, [_literal(b"{"), _WS_ITEM, (_OBJ, node, 0, False)])]
    if isinstance(node, _Array):
        return [_push(rest, [_literal(b"["), _WS_ITEM, (_ARR, node, 0)])]
    if isinstance(node, _Any):
        # Shared node objects: stacks compare nodes by identity, so fresh ones would defeat state interning.
        return [s for option in (_FREE_OBJECT, _ANY_ARRAY, *_ANY_SCALARS) for s in _expand(option, rest)]
    raise TypeError(f"unknown node {node!r}")


def _object_steps(item: tuple[Any, ...], rest: Stack) -> list[Stack]:
    _, node, index, written = item
    lead = [_COMMA, _WS_ITEM] if written else []
    close = _push(rest, [_literal(b"}")])
    if node.free:
        member = [*lead, _value(_STRING), _WS_ITEM, _COLON, _WS_ITEM, _value(_ANY), _WS_ITEM, (_OBJ, node, 0, True)]
        return [close, _push(rest, member)]
    stacks: list[Stack] = []
    properties = node.properties
    for position in range(index, len(properties)):
        name, schema, required = properties[position]
        member = [*lead, _literal(name), _WS_ITEM, _COLON, _WS_ITEM, _value(schema), _WS_ITEM]
        stacks.append(_push(rest, [*member, (_OBJ, node, position + 1, True)]))
        if required:
            return stacks
    stacks.append(close)
    return stacks


def _array_steps(item: tuple[Any, ...], rest: Stack) -> list[Stack]:
    _, node, count = item
    stacks: list[Stack] = []
    if count >= node.minimum:
        stacks.append(_push(rest, [_literal(b"]")]))
    if node.maximum is None or count < node.maximum:
        lead = [_COMMA, _WS_ITEM] if count else []
        stacks.append(_push(rest, [*lead, _value(node.items), _WS_ITEM, (_ARR, node, count + 1)]))
    return stacks


def _closure(stack: Stack, out: dict[Stack, None]) -> None:
    """Adds to ``out`` the stacks reachable from ``stack`` without reading, whose top reads a byte (or which are
    empty)."""
    if stack is None:
        out[None] = None
        return
    item, rest = stack
    tag = item[0]
    if tag in (_LIT, _ALT, _STR):
        out[stack] = None
    elif tag == _WS:
        out[stack] = None
        _closure(rest, out)
    elif tag == _NUM:
        out[stack] = None
        if item[2] in _NUMBER_ACCEPTING:
            _closure(rest, out)
    elif tag == _VALUE:
        for expanded in _expand(item[1], rest):
            _closure(expanded, out)
    elif tag == _OBJ:
        for expanded in _object_steps(item, rest):
            _closure(expanded, out)
    elif tag == _ARR:
        for expanded in _array_steps(item, rest):
            _closure(expanded, out)
    elif tag == _RE:
        _, dfa, state = item
        if dfa.reads(state):
            out[stack] = None
        if dfa.accepting(state):
            _closure(rest, out)
    else:  # pragma: no cover
        raise TypeError(f"unknown grammar item {item!r}")


# Well-formed UTF-8 only (no overlong forms, surrogates or code points above U+10FFFF), so generated strings always
# decode without replacement characters.
_UTF8_LEADS = {0xE0: 31, 0xED: 32, 0xF0: 33, 0xF4: 34}
_UTF8_LEADS.update({lead: 22 for lead in (*range(0xE1, 0xED), 0xEE, 0xEF)})
_UTF8_LEADS.update({lead: 23 for lead in range(0xF1, 0xF4)})
_UTF8_SECOND = {31: (0xA0, 0xBF, 21), 32: (0x80, 0x9F, 21), 33: (0x90, 0xBF, 22), 34: (0x80, 0x8F, 22)}


def _string_step(sub: int, byte: int) -> int | None:
    """The next string sub-state, -1 for the closing quote, ``None`` when ``byte`` is not allowed."""
    if sub == 0:
        if byte == 0x22:
            return -1
        if byte == 0x5C:
            return 1
        if byte < 0x20:
            return None
        if byte < 0x80:
            return 0
        if 0xC2 <= byte <= 0xDF:
            return 21
        return _UTF8_LEADS.get(byte)
    if sub == 1:
        if byte in _ESCAPES:
            return 0
        return 14 if byte == 0x75 else None
    if sub > 30:  # the restricted second byte after E0, ED, F0 or F4
        low, high, following = _UTF8_SECOND[sub]
        return following if low <= byte <= high else None
    if sub > 20:
        return (sub - 1 if sub > 21 else 0) if 0x80 <= byte <= 0xBF else None
    return (sub - 1 if sub > 11 else 0) if byte in _HEX else None


def _number_step(integer: bool, phase: int, byte: int) -> int | None:
    digit = byte in _DIGITS
    if phase == 0:
        return 1 if byte == 0x2D else (2 if byte == 0x30 else (3 if digit else None))
    if phase == 1:
        return 2 if byte == 0x30 else (3 if digit else None)
    if phase in (2, 3):
        if digit and phase == 3:
            return 3
        if integer:
            return None
        if byte == 0x2E:
            return 4
        return 6 if byte in (0x65, 0x45) else None
    if phase in (4, 5):
        if digit:
            return 5
        return 6 if phase == 5 and byte in (0x65, 0x45) else None
    if phase == 6:
        return 7 if byte in (0x2B, 0x2D) else (8 if digit else None)
    return 8 if digit else None  # phases 7 and 8


def _consume(stack: Stack, byte: int, out: dict[Stack, None]) -> None:
    """Adds to ``out`` the stacks after ``stack`` (whose top reads bytes) reads ``byte``."""
    assert stack is not None
    item, rest = stack
    tag = item[0]
    if tag == _LIT:
        _, data, position = item
        if data[position] == byte:
            out[rest if position + 1 == len(data) else ((_LIT, data, position + 1), rest)] = None
    elif tag == _ALT:
        _, options, position = item
        longer = []
        for option in options:
            if option[position] == byte:
                if position + 1 == len(option):
                    out[rest] = None
                else:
                    longer.append(option)
        if longer:
            out[((_ALT, tuple(longer), position + 1), rest)] = None
    elif tag == _WS:
        if byte in _WHITESPACE and item[1] > 0:
            out[((_WS, item[1] - 1), rest)] = None
    elif tag == _STR:
        sub = _string_step(item[1], byte)
        if sub is not None:
            out[rest if sub < 0 else ((_STR, sub), rest)] = None
    elif tag == _NUM:
        phase = _number_step(item[1], item[2], byte)
        if phase is not None:
            out[((_NUM, item[1], phase), rest)] = None
    elif tag == _RE:
        state = item[1].step(item[2], byte)
        if state >= 0:
            out[((_RE, item[1], state), rest)] = None


class Grammar:
    """The bytes a constrained generation may produce. Build one with :meth:`json_schema`, :meth:`json_object`,
    :meth:`literal` or :meth:`sequence`."""

    def __init__(self, items: Sequence[tuple[Any, ...]], *, alternatives: Sequence[Grammar] = ()) -> None:
        self._items = tuple(items)
        self._alternatives = tuple(alternatives)

    @classmethod
    def json_schema(cls, schema: Mapping[str, Any] | bool, *, lenient: bool = False) -> Grammar:
        """A JSON value valid under ``schema`` (see the module docstring for the supported keywords). ``lenient``
        ignores unsupported keywords instead of raising (for tool parameters, which only guide the model)."""
        root = schema if isinstance(schema, Mapping) else {}
        return cls([_value(_SchemaCompiler(root, lenient).compile(schema))])

    @classmethod
    def json_object(cls) -> Grammar:
        """Any JSON object (OpenAI ``response_format: {"type": "json_object"}``)."""
        return cls([_value(_FREE_OBJECT)])

    @classmethod
    def regex(cls, pattern: str) -> Grammar:
        """Text matching ``pattern`` in full (:mod:`etalii_dllm.regexp` lists the supported syntax)."""
        from etalii_dllm.regexp import compile_regex

        return cls([(_RE, compile_regex(pattern), 0)])

    @classmethod
    def literal(cls, text: str) -> Grammar:
        return cls([_literal(text.encode("utf-8"))] if text else [])

    @classmethod
    def whitespace(cls) -> Grammar:
        """Up to :data:`MAX_WHITESPACE` optional whitespace bytes."""
        return cls([_WS_ITEM])

    @classmethod
    def sequence(cls, parts: Iterable[Grammar]) -> Grammar:
        parts = list(parts)
        if any(part._alternatives for part in parts):
            raise ValueError("either() grammars cannot be part of a sequence")
        return cls([item for part in parts for item in part._items])

    @classmethod
    def either(cls, grammars: Sequence[Grammar]) -> Grammar:
        """Any one of ``grammars`` (for example a tool call or a JSON answer)."""
        flat = [alternative for g in grammars for alternative in (g._alternatives or (g,))]
        return cls((), alternatives=flat)

    @classmethod
    def choice(cls, grammars: Sequence[Grammar]) -> Grammar:
        """One of several JSON-schema grammars (each a single JSON value)."""
        nodes = []
        for grammar in grammars:
            if len(grammar._items) != 1 or grammar._items[0][0] != _VALUE:
                raise ValueError("choice() takes single JSON-value grammars")
            nodes.append(grammar._items[0][1])
        return cls([_value(_Union(nodes))])

    def matcher(self) -> Matcher:
        return Matcher([_push(None, g._items) for g in (self._alternatives or (self,))])


class Matcher:
    """Runs a grammar's automaton. States are interned integers; ``-1`` is the dead state."""

    DEAD = -1

    def __init__(self, starts: Sequence[Stack]) -> None:
        self._states: list[tuple[Stack, ...]] = []
        self._ids: dict[tuple[Stack, ...], int] = {}
        self._transitions: dict[tuple[int, int], int] = {}
        self._accepting: dict[int, tuple[bool, bool]] = {}
        closed: dict[Stack, None] = {}
        for start in starts:
            _closure(start, closed)
        self.start = self._intern(tuple(closed))

    def _intern(self, stacks: tuple[Stack, ...]) -> int:
        state = self._ids.get(stacks)
        if state is None:
            state = len(self._states)
            self._states.append(stacks)
            self._ids[stacks] = state
        return state

    def step(self, state: int, byte: int) -> int:
        """The state after reading ``byte``."""
        key = (state, byte)
        result = self._transitions.get(key)
        if result is not None:
            return result
        consumed: dict[Stack, None] = {}
        for stack in self._states[state]:
            if stack is not None:
                _consume(stack, byte, consumed)
        closed: dict[Stack, None] = {}
        for stack in consumed:
            _closure(stack, closed)
        result = self._intern(tuple(closed)) if closed else self.DEAD
        self._transitions[key] = result
        return result

    def advance(self, state: int, data: bytes) -> int:
        for byte in data:
            if state == self.DEAD:
                break
            state = self.step(state, byte)
        return state

    def _flags(self, state: int) -> tuple[bool, bool]:
        flags = self._accepting.get(state)
        if flags is None:
            stacks = self._states[state]
            flags = (None in stacks, any(s is not None for s in stacks))
            self._accepting[state] = flags
        return flags

    def accepting(self, state: int) -> bool:
        """The bytes read so far form a complete match."""
        return state != self.DEAD and self._flags(state)[0]

    def finished(self, state: int) -> bool:
        """Complete, and no further byte can extend the match."""
        return state != self.DEAD and self._flags(state) == (True, False)

    def matches(self, data: bytes) -> bool:
        return self.accepting(self.advance(self.start, data))


# -- token masks --------------------------------------------------------------------------------------------------


class TokenTrie:
    """The byte strings of a vocabulary, as a trie. Build it once per tokenizer (it is immutable)."""

    def __init__(self, token_bytes: Sequence[bytes]) -> None:
        self.vocabulary_size = len(token_bytes)
        self.token_bytes = tuple(token_bytes)
        self._children: list[dict[int, int]] = [{}]
        self._tokens: list[list[int]] = [[]]
        for token, data in enumerate(token_bytes):
            if not data:
                continue  # special tokens: never produced under a constraint (stop tokens are handled apart)
            node = 0
            for byte in data:
                child = self._children[node].get(byte)
                if child is None:
                    child = len(self._children)
                    self._children[node][byte] = child
                    self._children.append({})
                    self._tokens.append([])
                node = child
            self._tokens[node].append(token)

    def allowed(self, matcher: Matcher, state: int) -> list[int]:
        """Ids of the non-empty tokens whose bytes ``matcher`` accepts from ``state``, ascending."""
        allowed: list[int] = []
        pending = [(0, state)]
        children, tokens = self._children, self._tokens
        while pending:
            node, current = pending.pop()
            for byte, child in children[node].items():
                following = matcher.step(current, byte)
                if following == Matcher.DEAD:
                    continue
                allowed.extend(tokens[child])
                if children[child]:
                    pending.append((child, following))
        allowed.sort()
        return allowed


class TokenConstraint:
    """The per-step view of a grammar for :class:`etalii_dllm.generation.Generator`.

    ``trigger`` makes the constraint lazy: generation is free until the generated text contains ``trigger``, then
    the rest must match the grammar; once it has matched completely generation is free again until the next
    trigger (this is how tool calls are constrained while the model may still answer in plain text). Without a
    trigger the whole output must match, and generation stops as soon as the match cannot be extended.
    """

    def __init__(self, grammar: Grammar, trie: TokenTrie, *, trigger: str | None = None) -> None:
        self._grammar = grammar
        self._matcher = grammar.matcher()
        self._trie = trie
        self._trigger = trigger.encode("utf-8") if trigger else None
        self._state = self._matcher.start if self._trigger is None else None
        self._pending = b""
        """Generated bytes since the last completed match, while waiting for the trigger."""
        self._masks: dict[int, list[int]] = {}

    @property
    def active(self) -> bool:
        """Whether the next token is constrained."""
        return self._state is not None

    def allows(self, token: int) -> bool:
        assert self._state is not None
        data = self._trie.token_bytes[token]
        return bool(data) and self._matcher.advance(self._state, data) != Matcher.DEAD

    def allowed(self) -> list[int]:
        """The allowed (non-empty) tokens, ascending."""
        assert self._state is not None
        mask = self._masks.get(self._state)
        if mask is None:
            mask = self._trie.allowed(self._matcher, self._state)
            self._masks[self._state] = mask
        return mask

    @property
    def may_stop(self) -> bool:
        """Whether a stop token may end the generation now."""
        if self._state is None:
            return True
        return self._matcher.accepting(self._state) and self._trigger is None

    @property
    def finished(self) -> bool:
        """The whole output has matched and cannot be extended: stop without sampling."""
        return self._trigger is None and self._state is not None and self._matcher.finished(self._state)

    def accept(self, token: int) -> None:
        """Records a generated token."""
        data = self._trie.token_bytes[token]
        if self._state is not None:
            self._state = self._matcher.advance(self._state, data)
            if self._state == Matcher.DEAD:  # only reachable in lazy mode, from bytes after the trigger
                self._state = None
                self._trigger = None
                return
            if self._trigger is not None and self._matcher.finished(self._state):
                self._state = None
                self._pending = b""
            return
        if self._trigger is None:
            return
        self._pending += data
        found = self._pending.find(self._trigger)
        if found < 0:
            self._pending = self._pending[-(len(self._trigger) - 1) :] if len(self._trigger) > 1 else b""
            return
        state = self._matcher.advance(self._matcher.start, self._pending[found + len(self._trigger) :])
        self._pending = b""
        if state == Matcher.DEAD:  # the model went its own way within the token: leave it unconstrained
            self._trigger = None
            return
        self._state = state
        if self._matcher.finished(state):
            self._state = None
