"""Constrained decoding: byte-level JSON grammars and the token masks they induce.

A :class:`Grammar` describes the bytes the model may produce: literal text, JSON values of a JSON schema, or a
sequence of those (a tool call is ``<tool_call>`` + a JSON object + ``</tool_call>``). It is recognised by a small
nondeterministic pushdown automaton over bytes, so tokens that end mid-character (byte-level BPE) are handled
exactly.

:class:`TokenConstraint` turns a grammar into the set of tokens allowed at each step. Every token's bytes are put in
a trie once per tokenizer; a depth-first walk over the trie, pruned as soon as the automaton rejects a prefix, finds
the allowed tokens. Automaton states are interned to integers and their byte transitions memoised, so the walk is
mostly dictionary lookups.

Determinism: the automaton is a pure function of the grammar and the bytes; stacks are kept in insertion order
(never in set iteration order), and the allowed tokens are returned in ascending id order.

Supported JSON schema keywords: ``type`` (a name or a list of names), ``properties``, ``required``,
``additionalProperties`` (``false``, or a free-form object when there are no ``properties``), ``items``,
``minItems``, ``maxItems``, ``enum``, ``const``, ``anyOf``, ``oneOf`` (treated as ``anyOf``), ``allOf`` with a
single schema, ``$ref`` to ``#/$defs/...`` or ``#/definitions/...``, and ``nullable``. Annotations (``title``,
``description``, ``default``, ``examples``, ``format``, ``$schema``, ``$id``, ``strict``) are ignored. Keywords
that constrain values in ways the automaton does not check (``pattern``, ``minLength``, ``minimum``, ...) are
rejected with a :class:`GrammarError` rather than silently ignored.

Object properties are generated in the order the schema lists them; optional properties may be left out. Between
tokens the model may write up to :data:`MAX_WHITESPACE` whitespace bytes, enough for pretty-printed JSON but not
for endless padding.
"""

from __future__ import annotations

import json
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
        unsupported = sorted(set(schema) - _ANNOTATIONS - _STRUCTURE)
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
            return _STRING
        if kind in ("number", "integer"):
            return _Number(integer=kind == "integer")
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

_LIT, _ALT, _WS, _STR, _NUM, _VALUE, _OBJ, _ARR = range(8)

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
