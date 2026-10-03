"""Constrained decoding: byte-level JSON grammars and the token masks they induce.

A :class:`Grammar` describes the bytes the model may produce: literal text, JSON values of a JSON schema, text
matching a regular expression (:mod:`etalii_dllm.regexp`), text a GBNF grammar derives (:mod:`etalii_dllm.gbnf`),
or a sequence of those (a tool call is ``<tool_call>`` + a JSON object + ``</tool_call>``). It is recognised by a
small nondeterministic pushdown automaton over bytes, so tokens that end mid-character (byte-level BPE) are handled
exactly.

:class:`TokenConstraint` turns a grammar into the set of tokens allowed at each step. Every token's bytes are put in
a trie once per tokenizer; a depth-first walk over the trie, pruned as soon as the automaton rejects a prefix, finds
the allowed tokens. Automaton states are interned to integers and their byte transitions memoised, so the walk is
mostly dictionary lookups.

Determinism: the automaton is a pure function of the grammar and the bytes; stacks are kept in insertion order
(never in set iteration order), and the allowed tokens are returned in ascending id order.

Supported JSON schema keywords: ``type`` (a name or a list of names), ``properties``, ``required``,
``additionalProperties`` (``false``, or, when there are no ``properties``, a free-form object whose values follow
the schema it gives), ``patternProperties``, ``propertyNames``, ``minProperties``/``maxProperties``, ``items``,
``prefixItems`` (and ``items`` arrays with ``additionalItems``), ``minItems``, ``maxItems``, ``uniqueItems`` (over
items from a finite set of literals), ``contains`` with ``minContains`` (and ``maxContains`` over items from a finite
set), ``enum``, ``const``, ``anyOf``, ``oneOf`` (treated as ``anyOf``), ``allOf``, ``not`` and ``if``/``then``/
``else`` (rewritten exactly by :mod:`etalii_dllm.schema_algebra`), ``$ref`` to ``#/$defs/...`` or
``#/definitions/...``, and ``nullable``. Value constraints are compiled to byte automata
(:mod:`etalii_dllm.regexp`): on strings ``pattern`` (found anywhere unless anchored by a leading ``^`` or trailing
``$``), the :data:`FORMATS` and ``minLength``/``maxLength`` in code points (such strings are written without escape
sequences, so they hold no quote, backslash or control character); on numbers ``minimum``, ``maximum``,
``exclusiveMinimum``, ``exclusiveMaximum`` and ``multipleOf``, compared in exact decimal arithmetic
(:mod:`etalii_dllm.numeric_automata`). Annotations (``title``, ``description``, ``default``, ``examples``, other
``format`` values, ``$schema``, ``$id``, ``strict``) are ignored. Keywords the automaton does not check
(``dependentSchemas``, ``unevaluatedProperties``, ...) are rejected with a :class:`GrammarError` rather than silently
ignored; ``lenient`` grammars (tool parameters) ignore them and the value constraints.

Object properties are generated in the order the schema lists them; optional properties may be left out. Between
tokens the model may write up to :data:`MAX_WHITESPACE` whitespace bytes, enough for pretty-printed JSON but not
for endless padding.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal
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
ANNOTATIONS = _ANNOTATIONS
"""Schema keywords that never restrict a value."""
_STRUCTURE = frozenset(
    {"type", "properties", "required", "additionalProperties", "items", "minItems", "maxItems", "enum", "const",
     "anyOf", "oneOf", "allOf", "$ref", "$defs", "definitions", "nullable", "prefixItems", "additionalItems",
     "uniqueItems", "propertyNames", "not", "if", "then", "else", "patternProperties", "contains", "minContains",
     "maxContains"}
)  # fmt: skip
_STRING_KEYWORDS = ("pattern", "minLength", "maxLength")
_TYPES = ("object", "array", "string", "number", "boolean", "null")
_SHAPES = ("type", "enum", "const", "$ref", "anyOf", "oneOf", "allOf")
"""Keywords that say which values a schema has, so a schema without them takes any value."""
_BOUNDS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
_CONSTRAINTS = frozenset({*_STRING_KEYWORDS, *_BOUNDS, "multipleOf", "minProperties", "maxProperties"})
"""Value constraints compiled to byte automata (:func:`string_automaton`, :func:`integer_automaton`,
:func:`etalii_dllm.numeric_automata.number_automaton`) or checked by the object automaton (property counts)."""

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


class _Unsatisfiable(GrammarError):
    """No value satisfies the schema: an optional property or an ``anyOf`` branch of it is left out."""


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
    """``properties`` as ``(json-encoded name, schema, required)``; ``free`` objects take members of the ``kinds``
    ``(names, values)``: a name valid under ``names`` with a value valid under ``values``. ``minimum``/``maximum``
    bound the number of members."""

    __slots__ = ("free", "kinds", "maximum", "minimum", "properties")

    def __init__(
        self,
        properties: Sequence[tuple[bytes, _Node, bool]],
        free: bool,
        *,
        kinds: Sequence[tuple[_Node, _Node]] = (),
        minimum: int = 0,
        maximum: int | None = None,
    ) -> None:
        self.properties = tuple(properties)
        self.free = free
        self.kinds = tuple(kinds)
        self.minimum = minimum
        self.maximum = maximum


class _Array(_Node):
    """Elements valid under ``prefix`` (one schema per leading position) and then ``items`` (``None``: no more), or,
    with ``choices``, among those JSON texts (``(text, group, hit)``: equal values share a group; ``hit``: the value
    matches ``contains``), distinct ones when ``unique``. Between ``least`` and ``most`` elements match
    ``contains``: for ``choices`` the hits are counted; otherwise ``witnesses`` (per prefix position) and
    ``witness`` (after it) are the element nodes that match ``contains``, and only ``least`` is enforced."""

    __slots__ = ("choices", "items", "least", "maximum", "minimum", "most", "prefix", "unique", "witness",
                 "witnesses")  # fmt: skip

    def __init__(
        self,
        items: _Node | None,
        minimum: int,
        maximum: int | None,
        *,
        prefix: Sequence[_Node] = (),
        choices: Sequence[tuple[bytes, int, bool]] | None = None,
        unique: bool = False,
        least: int = 0,
        most: int | None = None,
        witnesses: Sequence[_Node | None] = (),
        witness: _Node | None = None,
    ) -> None:
        self.items = items
        self.minimum = minimum
        self.maximum = maximum
        self.prefix = tuple(prefix)
        self.choices = None if choices is None else tuple(choices)
        self.unique = unique
        self.least = least
        self.most = most
        self.witnesses = tuple(witnesses)
        self.witness = witness


class _Text(_Node):
    """A JSON string whose content (written without escapes) the automaton ``dfa`` matches."""

    __slots__ = ("dfa",)

    def __init__(self, dfa: Any) -> None:
        self.dfa = dfa


class _Digits(_Node):
    """A number whose JSON text the automaton ``dfa`` matches."""

    __slots__ = ("dfa",)

    def __init__(self, dfa: Any) -> None:
        self.dfa = dfa


class _Any(_Node):
    """Any JSON value."""

    __slots__ = ()


class _Rule(_Node):
    """A grammar rule (:mod:`etalii_dllm.gbnf`): alternatives that are sequences of automaton items. Filled in after
    creation, so rules can refer to each other and to themselves."""

    __slots__ = ("alternatives", "name")

    def __init__(self, name: str) -> None:
        self.name = name
        self.alternatives: tuple[tuple[tuple[Any, ...], ...], ...] = ()


_ANY = _Any()
_STRING = _String()
_FREE_OBJECT = _Object((), free=True, kinds=[(_STRING, _ANY)])
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
def string_automaton(
    pattern: str | tuple[str, ...] | None,
    text_format: str | None,
    min_length: int,
    max_length: int | None,
    content: str = _CONTENT,
) -> Any:
    """The byte automaton of string content written without escapes (``content``: JSON's by default) that matches
    ``pattern`` (JSON Schema semantics; a tuple: every one of them), the format and the length bounds (in code
    points): the intersection of their automata, trimmed so that every state can still reach a match."""
    from etalii_dllm.regexp import Counted, compile_regex, intersect

    dfa = compile_regex(content)
    try:
        for each in () if pattern is None else pattern if isinstance(pattern, tuple) else (pattern,):
            dfa = intersect(dfa, compile_regex(_searched(each)))
        if text_format is not None:
            dfa = intersect(dfa, compile_regex(FORMATS[text_format]))
        if min_length or max_length is not None:
            return Counted(dfa, min_length, max_length)
    except GrammarError as error:
        if "no text satisfies" in str(error):
            raise _Unsatisfiable(f"no string satisfies the schema ({error})") from None
        raise
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
        raise _Unsatisfiable(f"no integer lies between {low} and {high}")
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


def _identity(value: Any) -> Any:
    """A key equal for equal JSON values: numbers by exact value (``1`` equals ``1.0``), never equal to booleans."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return (type(value).__name__, value)
    if isinstance(value, int | float):
        from decimal import Decimal

        return ("number", Decimal(repr(value) if isinstance(value, float) else value).normalize())
    if isinstance(value, list):
        return ("array", tuple(_identity(v) for v in value))
    return ("object", tuple(sorted((k, _identity(v)) for k, v in value.items())))


def _choices(node: _Node) -> list[tuple[bytes, int]] | None:
    """The JSON texts of a node that is a finite set of literals (unions included), each with a group number shared
    by equal values; ``None`` for other nodes."""
    texts: list[bytes] = []

    def collect(current: _Node) -> bool:
        if isinstance(current, _Literals):
            texts.extend(current.options)
            return True
        if isinstance(current, _Union):
            return all(collect(option) for option in current.options)
        return False

    if not collect(node):
        return None
    groups: dict[Any, int] = {}
    choices = []
    for text in dict.fromkeys(texts):
        key = _identity(json.loads(text))
        choices.append((text, groups.setdefault(key, len(groups))))
    return choices


class _SchemaCompiler:
    def __init__(self, root: Mapping[str, Any], lenient: bool) -> None:
        self._root = root
        self._lenient = lenient
        self._refs: dict[str, _Node] = {}
        self._combined: dict[str, _Node] = {}
        self._algebra: Any = None

    def compile(self, schema: Any) -> _Node:
        if schema is True or schema == {}:
            return _ANY
        if schema is False:
            raise _Unsatisfiable("no value satisfies the schema false")
        if not isinstance(schema, Mapping):
            raise GrammarError(f"a schema must be an object, got {schema!r}")
        unsupported = sorted(set(schema) - _ANNOTATIONS - _STRUCTURE - _CONSTRAINTS)
        if unsupported and not self._lenient:
            names = ", ".join(unsupported)
            raise GrammarError(f"JSON schema keyword(s) not supported by constrained decoding: {names}")
        if "$ref" not in schema and not self._lenient and self._combinators(schema):
            return self._combine(schema)
        node = self._compile(schema)
        if isinstance(schema.get("not"), tuple) and not self._lenient:
            node = self._without(node, schema["not"])
        if schema.get("nullable") is True:
            node = _Union([node, _Literals([b"null"])])
        return node

    @staticmethod
    def _combinators(schema: Mapping[str, Any]) -> bool:
        return "allOf" in schema or "if" in schema or ("not" in schema and not isinstance(schema["not"], tuple))

    def _combine(self, schema: Mapping[str, Any]) -> _Node:
        """``allOf``/``not``/``if`` rewritten by :mod:`etalii_dllm.schema_algebra`; equal rewrites share one node, so
        recursive schemas end."""
        from etalii_dllm.schema_algebra import Algebra, canonical

        key = canonical(schema)
        node = self._combined.get(key)
        if node is not None:
            return node
        if self._algebra is None:
            self._algebra = Algebra(self._target)
        simple = self._algebra.simplify(schema)
        if simple is False:
            raise _Unsatisfiable("no value satisfies all of the schema's conditions")
        placeholder = _Union(())
        self._combined[key] = placeholder
        try:
            placeholder.options = (self.compile(simple),)
        except GrammarError:
            del self._combined[key]
            raise
        return placeholder

    def _compile(self, schema: Mapping[str, Any]) -> _Node:
        if "$ref" in schema:
            return self._reference(str(schema["$ref"]))
        if "const" in schema or "enum" in schema:
            values = [schema["const"]] if "const" in schema else list(schema["enum"])
            if not values:
                raise GrammarError("'enum' must not be empty")
            return self._values(values, schema)
        for keyword in ("anyOf", "oneOf"):
            if keyword in schema:
                return self._union(schema[keyword])
        if "allOf" in schema:  # lenient grammars only
            return self.compile(schema["allOf"][0]) if schema["allOf"] else _ANY
        kind = schema.get("type")
        if isinstance(kind, list):
            options = []
            for name in kind:
                try:
                    options.append(self._typed(str(name), schema))
                except _Unsatisfiable:
                    continue
            if not options and kind:
                raise _Unsatisfiable("no type of the schema has a value")
            return _Union(options)
        if kind is None:
            if "properties" in schema or "patternProperties" in schema:
                kind = "object"
            elif "items" in schema or "prefixItems" in schema:
                kind = "array"
            elif any(k in _CONSTRAINTS or k in ("contains", "propertyNames", "uniqueItems") for k in schema):
                # Keywords constrain only values of their type: every other type stays free.
                return _Union([self._typed(name, schema) for name in _TYPES])
            else:
                return _ANY
        return self._typed(str(kind), schema)

    def _union(self, schemas: Sequence[Any]) -> _Node:
        """The values of any of ``schemas``; branches no value satisfies are left out."""
        options = []
        for schema in schemas:
            try:
                options.append(self.compile(schema))
            except _Unsatisfiable:
                continue
        if not options and schemas:
            raise _Unsatisfiable("no branch of 'anyOf' has a value")
        return _Union(options)

    def _values(self, values: Sequence[Any], schema: Mapping[str, Any]) -> _Node:
        """``enum``/``const`` values, without those the schema's other keywords rule out."""
        texts = [_encode(v) for v in values]
        rest = {k: v for k, v in schema.items() if k not in ("const", "enum", "nullable") and k not in _ANNOTATIONS}
        if rest and not self._lenient:
            try:
                matcher = Grammar([_value(self.compile(rest))]).matcher()
                texts = [text for text in texts if matcher.matches(text)]
            except _Unsatisfiable:
                texts = []
            if not texts:
                raise _Unsatisfiable("no value of 'enum'/'const' satisfies the rest of the schema")
        return _Literals(texts)

    def _typed(self, kind: str, schema: Mapping[str, Any]) -> _Node:
        if kind == "string":
            text_format = schema.get("format") if schema.get("format") in FORMATS else None
            if self._lenient or (text_format is None and not any(k in schema for k in _STRING_KEYWORDS)):
                return _STRING
            pattern = schema.get("pattern")
            if pattern is not None and not isinstance(pattern, str | tuple):
                raise GrammarError("'pattern' must be a string")
            min_length, max_length = int(schema.get("minLength", 0)), schema.get("maxLength")
            if max_length is not None and int(max_length) < min_length:
                raise _Unsatisfiable("'maxLength' is smaller than 'minLength'")
            maximum = None if max_length is None else int(max_length)
            return _Text(string_automaton(pattern, text_format, min_length, maximum))
        if kind in ("number", "integer"):
            if self._lenient or not any(k in schema for k in (*_BOUNDS, "multipleOf")):
                return _Number(integer=kind == "integer")
            from etalii_dllm.numeric_automata import decimal_bounds, exact, number_automaton

            if kind == "integer" and "multipleOf" not in schema:
                return _Digits(integer_automaton(*_bounds(schema)))
            multiple = exact(schema["multipleOf"], "multipleOf") if "multipleOf" in schema else None
            return _Digits(number_automaton(*decimal_bounds(schema), multiple, kind == "integer"))
        if kind == "boolean":
            return _Literals([b"true", b"false"])
        if kind == "null":
            return _Literals([b"null"])
        if kind == "array":
            return self._array(schema)
        if kind == "object":
            return self._object(schema)
        raise GrammarError(f"unknown JSON schema type {kind!r}")

    def _object(self, schema: Mapping[str, Any]) -> _Object:
        from etalii_dllm.schema_algebra import all_of, searches

        properties = schema.get("properties") or {}
        required = set(schema.get("required") or ())
        unknown = sorted(required - set(properties))
        if unknown and properties:
            raise GrammarError(f"required properties without a schema: {', '.join(unknown)}")
        minimum, maximum = self._counts(schema)
        additional = schema.get("additionalProperties", True)
        patterns = {} if self._lenient else dict(schema.get("patternProperties") or {})
        names = self._names(schema)
        declared = bool(properties)
        if names is not None and properties:
            properties, required = self._named(properties, required, names)
        # With declared properties the model writes exactly those (``additionalProperties`` is not used); each one
        # also obeys the pattern properties its name matches.
        members = []
        for name, sub in properties.items():
            matched = [p_schema for pattern, p_schema in patterns.items() if searches(pattern, str(name))]
            try:
                node = self.compile(all_of(sub, *matched) if matched else sub)
            except _Unsatisfiable:
                if name in required:
                    raise GrammarError(f"required property {name!r} has no value under the schema") from None
                continue
            members.append((_encode(str(name)), node, name in required))
        if members:
            if minimum > len(members) or (maximum is not None and len(required) > maximum):
                raise GrammarError(
                    "no object of the declared properties has between 'minProperties' and 'maxProperties' members"
                )
            return _Object(members, free=False, minimum=minimum, maximum=maximum)
        if (additional is False and not patterns) or declared:
            if minimum:
                raise GrammarError("no object without properties has 'minProperties' members")
            return _Object((), free=False)
        if minimum > 1:
            # Free names may repeat, and a repeated name counts once: only one member is certain.
            raise GrammarError("'minProperties' above 1 needs declared 'properties'")
        if not patterns:
            values = self._free_values(additional)
            if values is _ANY and names is None and (minimum, maximum) == (0, None):
                return _FREE_OBJECT
            kinds = [] if values is None else [(names or _STRING, values)]
        else:
            kinds = self._pattern_kinds(names, patterns, additional)
        if minimum and not kinds:
            raise GrammarError("no object without properties has 'minProperties' members")
        return _Object((), free=True, kinds=kinds, minimum=minimum, maximum=maximum)

    def _free_values(self, schema: Any) -> _Node | None:
        """The node of the values of free members, ``None`` when there are none."""
        if schema is True or self._lenient:
            return _ANY
        try:
            return self.compile(schema)
        except _Unsatisfiable:
            return None

    def _pattern_kinds(
        self, names: _Node | None, patterns: Mapping[str, Any], additional: Any
    ) -> list[tuple[_Node, _Node]]:
        """Free members under ``patternProperties``: the names, split by which patterns they match (byte automata
        intersected and subtracted), each with the merged schemas of those patterns (``additional`` for none)."""
        from etalii_dllm.regexp import Counted, compile_regex, difference, intersect
        from etalii_dllm.schema_algebra import MAX_BRANCHES, all_of

        automata = [compile_regex(_searched(str(p))) for p in patterns]
        schemas = list(patterns.values())
        regions: list[tuple[tuple[int, ...], _Node]] = []
        literal_regions: dict[tuple[int, ...], list[bytes]] = {}
        for option in _flatten(names or _STRING):
            if isinstance(option, _Literals):
                for text in option.options:
                    name = json.loads(text).encode("utf-8")
                    matched = tuple(i for i, automaton in enumerate(automata) if automaton.matches(name))
                    literal_regions.setdefault(matched, []).append(text)
                continue
            dfa = option.dfa if isinstance(option, _Text) else string_automaton(None, None, 0, None)
            counted = (dfa.minimum, dfa.maximum) if isinstance(dfa, Counted) else None
            split = [((), dfa.dfa if counted else dfa)]
            for index, automaton in enumerate(automata):
                following = []
                for matched, region in split:
                    for marks, combine in (((*matched, index), intersect), (matched, difference)):
                        try:
                            following.append((marks, combine(region, automaton)))
                        except GrammarError as error:
                            if "no text satisfies" not in str(error):
                                raise  # pragma: no cover - the size limit
                split = following
                if len(split) > MAX_BRANCHES:
                    raise GrammarError(f"'patternProperties' split the names into more than {MAX_BRANCHES} parts")
            for matched, region in split:
                try:
                    regions.append((matched, _Text(Counted(region, *counted) if counted else region)))
                except GrammarError:
                    continue
        regions += [(matched, _Literals(texts)) for matched, texts in literal_regions.items()]
        kinds = []
        for matched, name_node in regions:
            values = self._free_values(all_of(*(schemas[i] for i in matched)) if matched else additional)
            if values is not None:
                kinds.append((name_node, values))
        return kinds

    def _array(self, schema: Mapping[str, Any]) -> _Array:
        from etalii_dllm.schema_algebra import all_of

        prefix_schemas = schema.get("prefixItems")
        rest = schema.get("items", True)
        if prefix_schemas is None and isinstance(rest, list):  # draft 2019 and older: an items array is a tuple
            prefix_schemas, rest = rest, schema.get("additionalItems", True)
        if prefix_schemas is not None and not isinstance(prefix_schemas, list):
            raise GrammarError("'prefixItems' must be a list of schemas")
        prefix = [self.compile(sub) for sub in prefix_schemas or ()]
        items = None if rest is False else self.compile(rest)
        minimum = int(schema.get("minItems", 0))
        maximum = None if schema.get("maxItems") is None else int(schema["maxItems"])
        if maximum is not None and maximum < minimum:
            raise GrammarError("'maxItems' is smaller than 'minItems'")
        if items is None:
            maximum = len(prefix) if maximum is None else min(maximum, len(prefix))
        least, most = self._contains_counts(schema)
        unique = schema.get("uniqueItems") is True and not self._lenient
        choices = None
        if unique or most is not None or least:
            found = _choices(items) if items is not None and not prefix else None
            if found is not None and (unique or least or most is not None):
                hits = self._hits(schema.get("contains"), found) if least or most is not None else {}
                choices = [(text, group, hits.get(group, False)) for text, group in found]
        if unique and choices is None:
            raise GrammarError(
                "'uniqueItems' needs items from a finite set (enum, const, boolean, null) and no 'prefixItems'"
            )
        if most is not None and choices is None:
            raise GrammarError("'maxContains' needs items from a finite set (enum, const, boolean, null)")
        if unique and choices is not None:
            groups = len({group for _, group, _ in choices})
            maximum = groups if maximum is None else min(maximum, groups)
        if maximum is not None and maximum < minimum:
            raise GrammarError("no array has 'minItems' elements under the schema")
        witnesses: list[_Node | None] = []
        witness = None
        if least and choices is None:
            contains = schema["contains"]
            witnesses = [self._witness(all_of(sub, contains)) for sub in prefix_schemas or ()]
            witness = None if items is None else self._witness(all_of(rest, contains))
        node = _Array(items, minimum, maximum, prefix=prefix, choices=choices, unique=unique, least=least, most=most,
                      witnesses=witnesses, witness=witness)  # fmt: skip
        if not _array_can_finish(node, 0, (), 0):
            raise GrammarError("no array has 'minContains' elements matching 'contains' under the schema")
        return node

    def _contains_counts(self, schema: Mapping[str, Any]) -> tuple[int, int | None]:
        """How many elements must (at least) and may (at most, ``None``: any) match ``contains``."""
        if "contains" not in schema or self._lenient:
            return 0, None
        least, most = schema.get("minContains", 1), schema.get("maxContains")
        for name, value in (("minContains", least), ("maxContains", most)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise GrammarError(f"'{name}' must be a non-negative integer")
        if most is not None and most < least:
            raise GrammarError("'maxContains' is smaller than 'minContains'")
        return least, most

    def _hits(self, contains: Any, choices: Sequence[tuple[bytes, int]]) -> dict[int, bool]:
        """Per group of ``choices``: whether its value matches ``contains``."""
        try:
            matcher = Grammar([_value(self.compile(contains))]).matcher()
        except _Unsatisfiable:
            return {}
        hits: dict[int, bool] = {}
        for text, group in choices:
            hits.setdefault(group, matcher.matches(text))
        return hits

    def _witness(self, schema: Any) -> _Node | None:
        try:
            return self.compile(schema)
        except _Unsatisfiable:
            return None

    def _without(self, node: _Node, exclusions: tuple[tuple[str, Any], ...]) -> _Node:
        """``node`` without the strings and numbers that internal exclusions (:mod:`etalii_dllm.schema_algebra`)
        rule out: values, patterns, formats and divisors."""
        result = _excluded(node, exclusions)
        if result is None:
            raise _Unsatisfiable("no value satisfies the schema's 'not'")
        return result

    def _names(self, schema: Mapping[str, Any]) -> _Node | None:
        """The node of ``propertyNames`` (a string schema), or ``None`` when names are free."""
        names = schema.get("propertyNames", True)
        if self._lenient or names is True or names == {}:
            return None
        if isinstance(names, Mapping) and not any(k in names for k in _SHAPES):
            names = {**names, "type": "string"}  # names are strings: string keywords describe them
        elif isinstance(names, Mapping) and self._combinators(names):
            names = {"allOf": [names, {"type": "string"}]}
        try:
            node = self.compile(names)
        except _Unsatisfiable:
            return _Union(())

        def strings(current: _Node) -> bool:
            if isinstance(current, _String | _Text):
                return True
            if isinstance(current, _Literals):
                return all(option.startswith(b'"') for option in current.options)
            return isinstance(current, _Union) and all(strings(option) for option in current.options)

        if not strings(node):
            raise GrammarError("'propertyNames' must describe strings")
        return node

    def _named(
        self, properties: Mapping[str, Any], required: set[str], names: _Node
    ) -> tuple[dict[str, Any], set[str]]:
        """The declared properties whose names are valid under ``names``; a required one that is not is refused."""
        matcher = Grammar([_value(names)]).matcher()
        kept = {name: sub for name, sub in properties.items() if matcher.matches(_encode(str(name)))}
        broken = sorted(required - set(kept))
        if broken:
            raise GrammarError(f"required properties break 'propertyNames': {', '.join(broken)}")
        return kept, required

    def _counts(self, schema: Mapping[str, Any]) -> tuple[int, int | None]:
        if self._lenient:
            return 0, None
        minimum, maximum = schema.get("minProperties", 0), schema.get("maxProperties")
        for name, value in (("minProperties", minimum), ("maxProperties", maximum)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise GrammarError(f"'{name}' must be a non-negative integer")
        if maximum is not None and maximum < minimum:
            raise GrammarError("'maxProperties' is smaller than 'minProperties'")
        return minimum, maximum

    def _target(self, reference: str) -> Any:
        """The schema a local ``$ref`` points to."""
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
        return target

    def _reference(self, reference: str) -> _Node:
        node = self._refs.get(reference)
        if node is not None:
            return node
        target = self._target(reference)
        # Recursive schemas: a placeholder union is filled in after compiling, so references to it resolve lazily.
        placeholder = _Union(())
        self._refs[reference] = placeholder
        try:
            placeholder.options = (self.compile(target),)
        except GrammarError:
            del self._refs[reference]
            raise
        return placeholder


def _flatten(node: _Node) -> list[_Node]:
    """The options of nested unions."""
    if isinstance(node, _Union):
        return [leaf for option in node.options for leaf in _flatten(option)]
    return [node]


def _number_text(value: Decimal) -> str:
    """A regex of the decimal texts (as constrained decoding writes them) whose value is ``value``."""
    whole, _, fraction = format(abs(value), "f").partition(".")
    fraction = fraction.rstrip("0")
    sign = "-" if value < 0 else ""
    return f"{sign}{whole}\\.{fraction}0*" if fraction else f"{sign}{whole}(?:\\.0+)?"


def _excluded_literal(text: bytes, exclusions: tuple[tuple[str, Any], ...]) -> bool:
    """Whether excluded values hold ``text``. (``enum`` values are already checked against the other exclusions, and
    ``true``, ``false`` and ``null`` have no patterns or divisors.)"""
    value = _identity(json.loads(text))
    return any(kind == "values" and value in {_identity(v) for v in payload} for kind, payload in exclusions)


def _excluded(node: _Node, exclusions: tuple[tuple[str, Any], ...]) -> _Node | None:
    """``node`` without the excluded strings and numbers; ``None`` when nothing is left."""
    from etalii_dllm.regexp import Counted, compile_regex, difference, literal_automaton

    if isinstance(node, _Union):
        options = [o for o in (_excluded(option, exclusions) for option in node.options) if o is not None]
        return _Union(options) if options else None
    if isinstance(node, _Any):
        return _excluded(_Union([_FREE_OBJECT, _ANY_ARRAY, *_ANY_SCALARS]), exclusions)
    if isinstance(node, _Literals):
        kept = [text for text in node.options if not _excluded_literal(text, exclusions)]
        return _Literals(kept) if kept else None
    if isinstance(node, _Object | _Array):
        kind = dict if isinstance(node, _Object) else list
        if any(k == "values" and any(isinstance(v, kind) for v in payload) for k, payload in exclusions):
            raise GrammarError("'not' can only exclude strings, numbers, booleans and null")
        return node
    try:
        if isinstance(node, _String | _Text):
            dfa = node.dfa if isinstance(node, _Text) else string_automaton(None, None, 0, None)
            counted = (dfa.minimum, dfa.maximum) if isinstance(dfa, Counted) else None
            inner = dfa.dfa if counted else dfa
            for kind, payload in exclusions:
                if kind == "values":
                    texts = [v.encode("utf-8") for v in payload if isinstance(v, str)]
                    excluded = literal_automaton(texts) if texts else None
                elif kind == "pattern":
                    excluded = compile_regex(_searched(payload))
                elif kind == "format":
                    excluded = compile_regex(FORMATS[payload])
                else:
                    excluded = None
                if excluded is not None:
                    inner = difference(inner, excluded)
            return _Text(Counted(inner, *counted) if counted else inner)
        if isinstance(node, _Number | _Digits):
            from etalii_dllm.numeric_automata import _divisible, exact, number_automaton

            dfa = node.dfa if isinstance(node, _Digits) else number_automaton(None, None, None, node.integer)
            for kind, payload in exclusions:
                if kind == "values":
                    numbers = [v for v in payload if isinstance(v, int | float) and not isinstance(v, bool)]
                    if numbers:
                        regex = "|".join(_number_text(exact(v, "value").normalize()) for v in numbers)
                        dfa = difference(dfa, compile_regex(regex))
                elif kind == "multipleOf":
                    dfa = difference(dfa, _divisible(exact(payload, "multipleOf")))
            return _Digits(dfa)
    except GrammarError as error:
        if "no text satisfies" in str(error):
            return None
        raise  # pragma: no cover - the size limit
    raise TypeError(f"unknown node {node!r}")  # pragma: no cover


def _array_can_finish(node: _Array, count: int, used: tuple[int, ...], found: int) -> bool:
    """Whether an array with ``count`` elements, the unique groups ``used`` and ``found`` elements matching
    ``contains`` can still be completed within the bounds."""
    infinite = math.inf
    maximum = infinite if node.maximum is None else node.maximum
    needed = max(0, node.least - found)
    if node.choices is not None:
        hit_groups = {group for _, group, hit in node.choices if hit and group not in used}
        other_groups = {group for _, group, hit in node.choices if not hit and group not in used}
        hits = len(hit_groups) if node.unique else (infinite if hit_groups else 0)
        others = len(other_groups) if node.unique else (infinite if other_groups else 0)
        most_hits = min(hits, infinite if node.most is None else node.most - found)
        return needed <= most_hits and count + needed <= maximum and count + most_hits + others >= node.minimum
    if not needed:
        return True
    available = sum(1 for p in range(count, min(len(node.prefix), int(min(maximum, 10**9)))) if node.witnesses[p])
    if node.witness is not None:
        available += maximum - max(count, len(node.prefix)) if maximum != infinite else infinite
    return available >= needed


# -- automaton ----------------------------------------------------------------------------------------------------
#
# A stack is a persistent linked list ``(item, rest)`` with ``None`` for the empty stack; items are tuples whose first
# element is a tag. Consuming items (literal, alternatives, whitespace, string, number) read bytes; a token item
# ``(_TOK, ids, negated)`` reads one whole token (one in ``ids``, or with ``negated`` one not in them, GBNF token
# references); the others expand into consuming items without reading.

_LIT, _ALT, _WS, _STR, _NUM, _VALUE, _OBJ, _ARR, _RE, _TOK = range(10)

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
        return [_push(rest, [_literal(b"{"), _WS_ITEM, (_OBJ, node, 0, 0)])]
    if isinstance(node, _Array):
        return [_push(rest, [_literal(b"["), _WS_ITEM, (_ARR, node, 0, (), 0)])]
    if isinstance(node, _Rule):
        return [_push(rest, alternative) for alternative in node.alternatives]
    if isinstance(node, _Any):
        # Shared node objects: stacks compare nodes by identity, so fresh ones would defeat state interning.
        return [s for option in (_FREE_OBJECT, _ANY_ARRAY, *_ANY_SCALARS) for s in _expand(option, rest)]
    raise TypeError(f"unknown node {node!r}")


def _object_steps(item: tuple[Any, ...], rest: Stack) -> list[Stack]:
    """The ways on from an object after ``count`` members, the last of them declared property ``index - 1``."""
    _, node, index, count = item
    lead = [_COMMA, _WS_ITEM] if count else []
    close = _push(rest, [_literal(b"}")])
    if node.free:
        stacks = [close] if count >= node.minimum else []
        if node.maximum is None or count < node.maximum:
            # Counts past what the bounds tell apart are alike, so states stay few.
            following = min(count + 1, node.maximum if node.maximum is not None else max(node.minimum, 1))
            for names, values in node.kinds:
                member = [*lead, _value(names), _WS_ITEM, _COLON, _WS_ITEM, _value(values), _WS_ITEM]
                stacks.append(_push(rest, [*member, (_OBJ, node, 0, following)]))
        return stacks
    stacks: list[Stack] = []
    properties = node.properties
    for position in range(index, len(properties)):
        name, schema, required = properties[position]
        later_required = sum(1 for p in properties[position + 1 :] if p[2])
        fits = node.maximum is None or count + 1 + later_required <= node.maximum
        reaches = count + len(properties) - position >= node.minimum  # this one and every later one
        if fits and reaches:
            member = [*lead, _literal(name), _WS_ITEM, _COLON, _WS_ITEM, _value(schema), _WS_ITEM]
            stacks.append(_push(rest, [*member, (_OBJ, node, position + 1, count + 1)]))
        if required:
            return stacks
    if count >= node.minimum:
        stacks.append(close)
    return stacks


def _array_steps(item: tuple[Any, ...], rest: Stack) -> list[Stack]:
    """The ways on from an array after ``count`` elements (``used``: the groups of the unique choices taken;
    ``found``: the elements that match ``contains``, counted up to what the bounds tell apart)."""
    _, node, count, used, found = item
    stacks: list[Stack] = []
    if count >= node.minimum and found >= node.least:
        stacks.append(_push(rest, [_literal(b"]")]))
    if node.maximum is not None and count >= node.maximum:
        return stacks
    lead = [_COMMA, _WS_ITEM] if count else []
    # Counts past what the bounds and the prefix tell apart are alike, so states stay few.
    cap = node.maximum if node.maximum is not None else max(node.minimum, len(node.prefix), 1)
    following = count + 1 if node.unique else min(count + 1, cap)
    if node.choices is not None:
        found_cap = node.most if node.most is not None else node.least
        for text, group, hit in node.choices:
            if node.unique and group in used:
                continue
            taken = tuple(sorted((*used, group))) if node.unique else used
            now = min(found + hit, found_cap)
            if (node.most is None or found + hit <= node.most) and _array_can_finish(node, following, taken, now):
                stacks.append(_push(rest, [*lead, _literal(text), _WS_ITEM, (_ARR, node, following, taken, now)]))
        return stacks
    element = node.prefix[count] if count < len(node.prefix) else node.items
    if element is not None and _array_can_finish(node, following, used, found):
        stacks.append(_push(rest, [*lead, _value(element), _WS_ITEM, (_ARR, node, following, used, found)]))
    if found < node.least:
        witness = node.witnesses[count] if count < len(node.prefix) else node.witness
        if witness is not None and _array_can_finish(node, following, used, found + 1):
            stacks.append(_push(rest, [*lead, _value(witness), _WS_ITEM, (_ARR, node, following, used, found + 1)]))
    return stacks


def _closure(stack: Stack, out: dict[Stack, None]) -> None:
    """Adds to ``out`` the stacks reachable from ``stack`` without reading, whose top reads a byte (or which are
    empty)."""
    if stack is None:
        out[None] = None
        return
    item, rest = stack
    tag = item[0]
    if tag in (_LIT, _ALT, _STR, _TOK):
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
    def gbnf(cls, text: str, tokens: Any = None) -> Grammar:
        """Text the GBNF grammar ``text`` derives from its ``root`` rule (:mod:`etalii_dllm.gbnf` lists the syntax);
        ``tokens`` (a :class:`etalii_dllm.gbnf.TokenTable`) resolves its token references."""
        from etalii_dllm.gbnf import compile_gbnf

        return compile_gbnf(text, tokens)

    @classmethod
    def raw_string(cls, schema: Mapping[str, Any], content: str) -> Grammar:
        """A string written as raw text, each byte allowed by the regex ``content`` (XML tool parameters: no
        ``<``), that satisfies ``schema``'s ``pattern``, ``format`` and length bounds exactly."""
        text_format = schema.get("format") if schema.get("format") in FORMATS else None
        pattern = schema.get("pattern")
        if pattern is not None and not isinstance(pattern, str):
            raise GrammarError("'pattern' must be a string")
        min_length, max_length = int(schema.get("minLength", 0)), schema.get("maxLength")
        if max_length is not None and int(max_length) < min_length:
            raise GrammarError("'maxLength' is smaller than 'minLength'")
        maximum = None if max_length is None else int(max_length)
        return cls([(_RE, string_automaton(pattern, text_format, min_length, maximum, content), 0)])

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
    def one_of(cls, grammars: Sequence[Grammar]) -> Grammar:
        """Any one of ``grammars``, as one part that can go in a :meth:`sequence` (unlike :meth:`either`)."""
        rule = _Rule("one of several parts")
        rule.alternatives = tuple(alt._items for g in grammars for alt in (g._alternatives or (g,)))
        return cls([_value(rule)])

    @classmethod
    def optional(cls, grammar: Grammar) -> Grammar:
        """``grammar`` or nothing."""
        return cls.one_of([grammar, cls([])])

    @classmethod
    def repeat(cls, grammar: Grammar, separator: Grammar | None = None) -> Grammar:
        """One or more of ``grammar``, ``separator`` between them."""
        if grammar._alternatives:
            grammar = cls.one_of([grammar])
        loop = _Rule("a repetition")
        lead = separator._items if separator is not None else ()
        loop.alternatives = ((*lead, *grammar._items, _value(loop)), ())
        return cls([*grammar._items, _value(loop)])

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
        self._token_items: dict[int, tuple[tuple[frozenset[int], bool], ...]] = {}
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

    def token_items(self, state: int) -> tuple[tuple[frozenset[int], bool], ...]:
        """The token references that may read the next token: ``(ids, negated)`` pairs, in stack order."""
        found = self._token_items.get(state)
        if found is None:
            found = ()
            if state != self.DEAD:
                tops = (s[0] for s in self._states[state] if s is not None and s[0][0] == _TOK)
                found = tuple(dict.fromkeys((top[1], top[2]) for top in tops))
            self._token_items[state] = found
        return found

    def step_token(self, state: int, token: int) -> int:
        """The state after a token reference reads ``token`` as a whole (``DEAD`` when none may)."""
        if not self.token_items(state):
            return self.DEAD
        key = (state, -1 - token)
        result = self._transitions.get(key)
        if result is not None:
            return result
        closed: dict[Stack, None] = {}
        for stack in self._states[state]:
            if stack is not None and stack[0][0] == _TOK and (token in stack[0][1]) != stack[0][2]:
                _closure(stack[1], closed)
        result = self._intern(tuple(closed)) if closed else self.DEAD
        self._transitions[key] = result
        return result

    def advance_token(self, state: int, token: int, data: bytes) -> int:
        """The state after the token ``token`` with bytes ``data``: read byte by byte, or whole by a token
        reference, whichever applies (both, merged). A token without bytes that no reference reads leaves the
        state as it is (stop tokens)."""
        whole = self.step_token(state, token)
        if not data:
            return state if whole == self.DEAD else whole
        by_bytes = self.advance(state, data)
        if whole == self.DEAD or by_bytes == self.DEAD:
            return by_bytes if whole == self.DEAD else whole
        return self._intern(tuple(dict.fromkeys((*self._states[by_bytes], *self._states[whole]))))

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
        """Ids of the non-empty tokens whose bytes ``matcher`` accepts from ``state``, and of the tokens a token
        reference reads there, ascending."""
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
        items = matcher.token_items(state)
        if items:
            whole: set[int] = set(allowed)
            for ids, negated in items:
                if negated:
                    whole.update(t for t in range(self.vocabulary_size) if t not in ids)
                else:
                    whole.update(t for t in ids if t < self.vocabulary_size)
            return sorted(whole)
        allowed.sort()
        return allowed

    def healing(self, prefix: bytes, fits: Callable[[bytes], bool]) -> list[int]:
        """Ids of the non-empty tokens whose bytes are a prefix of ``prefix``, or extend it and ``fits`` accepts,
        ascending (token healing, :class:`HealingConstraint`)."""
        allowed: list[int] = []
        children, tokens = self._children, self._tokens
        node = 0
        for byte in prefix:
            child = children[node].get(byte)
            if child is None:
                return sorted(allowed)
            allowed.extend(tokens[child])
            node = child
        pending = list(children[node].values())
        while pending:
            child = pending.pop()
            allowed.extend(token for token in tokens[child] if fits(self.token_bytes[token]))
            pending.extend(children[child].values())
        allowed.sort()
        return allowed


class TokenConstraint:
    """The per-step view of a grammar for :class:`etalii_dllm.generation.Generator`.

    ``trigger`` makes the constraint lazy: generation is free until the generated text contains ``trigger``, then
    the rest must match the grammar; once it has matched completely generation is free again until the next
    trigger (this is how tool calls are constrained while the model may still answer in plain text), or with
    ``once`` the output ends there (an answer with at most one tool call). ``lazy``
    (trigger words, as llama.cpp's lazy grammars) also leaves generation free until one of the words appears (the
    earliest, ties to the first listed), but then the grammar matches from the start of that word and constrains
    the rest of the output, as without a trigger. Without either the whole output must match, and generation stops
    as soon as the match cannot be extended. If the token that completes a trigger already goes against the grammar
    after it, the rest of the output stays free.
    """

    def __init__(
        self,
        grammar: Grammar,
        trie: TokenTrie,
        *,
        trigger: str | None = None,
        lazy: Sequence[str] = (),
        once: bool = False,
    ) -> None:
        if trigger is not None and lazy:
            raise ValueError("a constraint takes a trigger or lazy trigger words, not both")
        if once and trigger is None:
            raise ValueError("'once' needs a trigger")
        if any(not word for word in lazy):
            raise ValueError("a lazy grammar's trigger words cannot be empty")
        self._grammar = grammar
        self._matcher = grammar.matcher()
        self._trie = trie
        self._triggers = (trigger.encode("utf-8"),) if trigger else tuple(word.encode("utf-8") for word in lazy)
        self._rearm = trigger is not None
        """Free again after each complete match (tool calls); else the trigger is fed to the grammar (``lazy``)."""
        self._once = once
        for word in self._triggers if not self._rearm else ():
            if self._matcher.advance(self._matcher.start, word) == Matcher.DEAD:
                raise GrammarError(f"the grammar cannot start with its trigger word {word.decode('utf-8')!r}")
        self._state = self._matcher.start if not self._triggers else None
        self._pending = b""
        """Generated bytes since the last completed match, while waiting for the trigger."""
        self._masks: dict[int, list[int]] = {}

    @property
    def active(self) -> bool:
        """Whether the next token is constrained."""
        return self._state is not None

    def allows(self, token: int) -> bool:
        assert self._state is not None
        if self._matcher.step_token(self._state, token) != Matcher.DEAD:
            return True
        data = self._trie.token_bytes[token]
        return bool(data) and self._matcher.advance(self._state, data) != Matcher.DEAD

    def allows_bytes(self, data: bytes) -> bool:
        """Whether ``data`` may come next (always, while a lazy constraint waits for its trigger)."""
        return self._state is None or self._matcher.advance(self._state, data) != Matcher.DEAD

    def allowed(self) -> list[int]:
        """The allowed tokens (non-empty ones, and those a token reference reads), ascending."""
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
        return self._matcher.accepting(self._state) and not self._rearm

    @property
    def finished(self) -> bool:
        """The whole output has matched and cannot be extended: stop without sampling."""
        return not self._rearm and self._state is not None and self._matcher.finished(self._state)

    def accept(self, token: int) -> None:
        """Records a generated token."""
        data = self._trie.token_bytes[token]
        if self._state is not None:
            self._moved(self._matcher.advance_token(self._state, token, data))
        else:
            self.accept_bytes(data)

    def _moved(self, state: int) -> None:
        if state == Matcher.DEAD:  # only reachable from bytes after a trigger
            self._state = None
            self._triggers = ()
            return
        self._state = state
        if self._rearm and self._matcher.finished(state):
            if self._once:  # the match ends the output
                self._rearm = False
                return
            self._state = None
            self._pending = b""

    def accept_bytes(self, data: bytes) -> None:
        """Records generated bytes."""
        if self._state is not None:
            self._moved(self._matcher.advance(self._state, data))
            return
        if not self._triggers:
            return
        self._pending += data
        found = [(at, n) for n, word in enumerate(self._triggers) if (at := self._pending.find(word)) >= 0]
        if not found:
            longest = max(len(word) for word in self._triggers)
            self._pending = self._pending[-(longest - 1) :] if longest > 1 else b""
            return
        at, which = min(found)
        start = at if not self._rearm else at + len(self._triggers[which])
        state = self._matcher.advance(self._matcher.start, self._pending[start:])
        self._pending = b""
        if not self._rearm:
            self._triggers = ()
        self._moved(state)


class HealingConstraint:
    """Token healing: the prompt's last token was taken back, and the output must start with its bytes ``prefix``;
    then ``inner`` (or nothing) constrains the rest. While part of the prefix is left, a token is allowed when its
    bytes are a prefix of what is left, or start with all of it and ``inner`` allows the bytes after it. A stop
    token is not allowed until the prefix is written. Same interface as :class:`TokenConstraint`."""

    def __init__(self, prefix: bytes, trie: TokenTrie, inner: TokenConstraint | None = None) -> None:
        if not prefix:
            raise ValueError("token healing needs the bytes of a token")
        self._left = prefix
        self._trie = trie
        self._inner = inner
        self._masks: dict[bytes, list[int]] = {}

    @property
    def active(self) -> bool:
        return bool(self._left) or (self._inner is not None and self._inner.active)

    def _fits(self, data: bytes) -> bool:
        left = self._left
        if len(data) <= len(left):
            return bool(data) and left.startswith(data)
        return data.startswith(left) and (self._inner is None or self._inner.allows_bytes(data[len(left) :]))

    def allows(self, token: int) -> bool:
        if self._left:
            return self._fits(self._trie.token_bytes[token])
        assert self._inner is not None
        return self._inner.allows(token)

    def allowed(self) -> list[int]:
        """The allowed (non-empty) tokens, ascending."""
        if not self._left:
            assert self._inner is not None
            return self._inner.allowed()
        mask = self._masks.get(self._left)  # the inner constraint does not move while the prefix is written
        if mask is None:
            mask = self._trie.healing(self._left, self._fits)
            self._masks[self._left] = mask
        return mask

    @property
    def may_stop(self) -> bool:
        if self._left:
            return False
        return self._inner is None or self._inner.may_stop

    @property
    def finished(self) -> bool:
        return not self._left and self._inner is not None and self._inner.finished

    def accept(self, token: int) -> None:
        data = self._trie.token_bytes[token]
        if not self._left:
            if self._inner is not None:
                self._inner.accept(token)
            return
        if len(data) <= len(self._left):
            self._left = self._left[len(data) :]
            return
        tail, self._left = data[len(self._left) :], b""
        if self._inner is not None:
            self._inner.accept_bytes(tail)
