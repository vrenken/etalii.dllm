"""GBNF grammars (the format of llama.cpp) compiled to the byte pushdown automaton of :mod:`etalii_dllm.grammar`.

A grammar is a list of rules ``name ::= alternatives``; the output is the text the ``root`` rule derives, matched in
full. Supported syntax:

- rule names of letters, digits, ``-`` and ``_``; a name followed by ``::=`` starts a new rule, so a rule may span
  lines; ``#`` starts a comment that runs to the end of the line;
- alternatives separated by ``|`` (an empty alternative matches nothing), sequences, groups ``( ... )``;
- string literals ``"..."`` and character classes ``[...]``/``[^...]`` with ranges ``a-z``, both with the escapes
  ``\\n \\r \\t \\\\ \\" \\[ \\] \\-``, ``\\xHH``, ``\\uHHHH`` and ``\\UHHHHHHHH``; ``.`` is any character;
- the repetitions ``*``, ``+``, ``?``, ``{m}``, ``{m,}`` and ``{m,n}`` (counts up to
  :data:`etalii_dllm.regexp.MAX_REPEAT`).

Characters are Unicode code points (never surrogates) and the output is their UTF-8 encoding, so it is always
well-formed. Rules may refer to each other and to themselves, but not on the left: left recursion (``a ::= a "x"``,
also through rules that can match nothing) and unbounded repetitions of something that can match nothing would make
the automaton loop, so they raise :class:`etalii_dllm.grammar.GrammarError`, as do undefined or doubly defined rules,
a missing ``root`` and token references (``<...>``). Alternatives that can never finish (rules without a way out of
their recursion) are dropped, so constrained decoding never runs into an answer it cannot end; a grammar that
derives no text at all is refused.

Determinism: the automaton is a pure function of the grammar text. Rules and alternatives keep their written order
and every analysis walks them in that order.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

from etalii_dllm.grammar import _RE, _VALUE, Grammar, GrammarError, _literal, _Rule, _value
from etalii_dllm.regexp import _ALL, MAX_REPEAT, Ranges, _complement, _normalise, compile_regex

_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
_ESCAPES = {"n": 0x0A, "r": 0x0D, "t": 0x09, "\\": 0x5C, '"': 0x22, "[": 0x5B, "]": 0x5D, "-": 0x2D}
_HEX_DIGITS = {"x": 2, "u": 4, "U": 8}


# -- syntax tree ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Literal:
    text: str


@dataclass(frozen=True)
class _Class:
    ranges: Ranges


@dataclass(frozen=True)
class _Reference:
    name: str


@dataclass(frozen=True)
class _Group:
    alternatives: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class _Repeat:
    element: Any
    minimum: int
    maximum: int | None


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0

    def error(self, message: str) -> GrammarError:
        line = self.text.count("\n", 0, self.position) + 1
        column = self.position - (self.text.rfind("\n", 0, self.position) + 1) + 1
        return GrammarError(f"grammar line {line}, column {column}: {message}")

    def _skip(self) -> None:
        """Skips whitespace (newlines included) and comments."""
        text = self.text
        while self.position < len(text):
            char = text[self.position]
            if char in " \t\r\n":
                self.position += 1
            elif char == "#":
                end = text.find("\n", self.position)
                self.position = len(text) if end < 0 else end + 1
            else:
                break

    def _peek(self) -> str:
        return self.text[self.position : self.position + 1]

    def _name(self) -> str:
        start = self.position
        while self.position < len(self.text) and self.text[self.position] in _NAME_CHARS:
            self.position += 1
        return self.text[start : self.position]

    def _rule_starts(self) -> bool:
        """Whether a rule name followed by ``::=`` comes next (it ends the previous rule)."""
        saved = self.position
        found = bool(self._name())
        self._skip()
        found = found and self.text.startswith("::=", self.position)
        self.position = saved
        return found

    def parse(self) -> dict[str, tuple[tuple[Any, ...], ...]]:
        rules: dict[str, tuple[tuple[Any, ...], ...]] = {}
        self._skip()
        while self.position < len(self.text):
            start = self.position
            name = self._name()
            if not name:
                raise self.error(f"expected a rule name, found {self._peek()!r}")
            self._skip()
            if not self.text.startswith("::=", self.position):
                raise self.error(f"expected '::=' after the rule name {name!r}")
            if name in rules:
                self.position = start
                raise self.error(f"rule {name!r} is defined twice")
            self.position += 3
            rules[name] = self._alternatives(nested=False)
        if not rules:
            raise GrammarError("the grammar has no rules")
        return rules

    def _alternatives(self, nested: bool) -> tuple[tuple[Any, ...], ...]:
        alternatives = [self._sequence(nested)]
        while self._peek() == "|":
            self.position += 1
            alternatives.append(self._sequence(nested))
        return tuple(alternatives)

    def _sequence(self, nested: bool) -> tuple[Any, ...]:
        elements: list[Any] = []
        while True:
            self._skip()
            char = self._peek()
            if char in ("", "|") or (char == ")" and nested):
                return tuple(elements)
            if not nested and self._rule_starts():
                return tuple(elements)
            if char in "*+?{":
                if not elements:
                    raise self.error(f"nothing to repeat before {char!r}")
                elements[-1] = self._repeat(elements[-1])
                continue
            elements.append(self._element())

    def _element(self) -> Any:
        char = self._peek()
        if char == '"':
            self.position += 1
            codes: list[int] = []
            while self._peek() != '"':
                if not self._peek() or self._peek() == "\n":
                    raise self.error("unterminated string literal")
                codes.append(self._char())
            self.position += 1
            return _Literal("".join(map(chr, codes)))
        if char == "[":
            return self._class()
        if char == ".":
            self.position += 1
            return _Class(_ALL)
        if char == "(":
            self.position += 1
            alternatives = self._alternatives(nested=True)
            if self._peek() != ")":
                raise self.error("missing ')'")
            self.position += 1
            return _Group(alternatives)
        if char == "<":
            raise self.error("token references (<...>) are not supported")
        if char in _NAME_CHARS:
            return _Reference(self._name())
        raise self.error(f"unexpected {char!r}")

    def _char(self) -> int:
        """One code point of a literal or class, escapes resolved."""
        char = self.text[self.position]
        self.position += 1
        if char != "\\":
            if 0xD800 <= ord(char) <= 0xDFFF:
                raise self.error("a surrogate code point cannot be matched")
            return ord(char)
        escape = self._peek()
        self.position += 1
        if escape in _ESCAPES:
            return _ESCAPES[escape]
        if escape in _HEX_DIGITS:
            digits = self.text[self.position : self.position + _HEX_DIGITS[escape]]
            if len(digits) != _HEX_DIGITS[escape] or any(d not in "0123456789abcdefABCDEF" for d in digits):
                raise self.error(f"bad \\{escape} escape")
            self.position += len(digits)
            code = int(digits, 16)
            if code > 0x10FFFF or 0xD800 <= code <= 0xDFFF:
                raise self.error(f"\\{escape}{digits} is not a Unicode scalar value")
            return code
        raise self.error(f"unsupported escape \\{escape}")

    def _class(self) -> _Class:
        self.position += 1
        negated = self._peek() == "^"
        if negated:
            self.position += 1
        items: list[tuple[int, int]] = []
        while self._peek() != "]":
            if not self._peek():
                raise self.error("unterminated character class")
            low = self._char()
            high = low
            if self._peek() == "-" and self.text[self.position + 1 : self.position + 2] not in ("]", ""):
                self.position += 1
                high = self._char()
                if high < low:
                    raise self.error("a class range runs backwards")
            items.append((low, high))
        self.position += 1
        ranges = _normalise(items)
        ranges = _complement(ranges) if negated else ranges
        if not ranges:
            raise self.error("the character class matches no character")
        return _Class(ranges)

    def _count(self) -> int:
        start = self.position
        while self._peek().isdigit() and self._peek().isascii():
            self.position += 1
        if start == self.position:
            raise self.error("expected a repetition count")
        count = int(self.text[start : self.position])
        if count > MAX_REPEAT:
            raise self.error(f"repetition counts are limited to {MAX_REPEAT}")
        return count

    def _repeat(self, element: Any) -> _Repeat:
        char = self._peek()
        self.position += 1
        if char == "*":
            return _Repeat(element, 0, None)
        if char == "+":
            return _Repeat(element, 1, None)
        if char == "?":
            return _Repeat(element, 0, 1)
        self._skip()
        minimum = self._count()
        maximum: int | None = minimum
        self._skip()
        if self._peek() == ",":
            self.position += 1
            self._skip()
            maximum = None if self._peek() == "}" else self._count()
            self._skip()
        if self._peek() != "}":
            raise self.error("missing '}'")
        self.position += 1
        if maximum is not None and maximum < minimum:
            raise self.error("the repetition's maximum is smaller than its minimum")
        return _Repeat(element, minimum, maximum)


# -- compilation ---------------------------------------------------------------------------------------------------


def _class_pattern(ranges: Ranges) -> str:
    """A regex class matching exactly ``ranges`` (for :func:`etalii_dllm.regexp.compile_regex`)."""

    def char(code: int) -> str:
        if code <= 0xFFFF:
            return f"\\u{code:04x}"
        return chr(code)

    return "[" + "".join(char(low) if low == high else f"{char(low)}-{char(high)}" for low, high in ranges) + "]"


@functools.lru_cache(maxsize=1024)
def _class_item(ranges: Ranges) -> tuple[Any, ...]:
    return (_RE, compile_regex(_class_pattern(ranges)), 0)


class _Compiler:
    def __init__(self, syntax: dict[str, tuple[tuple[Any, ...], ...]]) -> None:
        self.syntax = syntax
        self.rules = {name: _Rule(name) for name in syntax}
        self.created: list[_Rule] = list(self.rules.values())
        self.repetitions: set[int] = set()
        """Ids of the rules that repeat an element without bound (for the error message)."""
        self._owner = ""

    def compile(self) -> _Rule:
        if "root" not in self.rules:
            raise GrammarError("the grammar has no 'root' rule")
        for name, alternatives in self.syntax.items():
            self._owner = name
            self.rules[name].alternatives = self._alternatives(alternatives)
        return self.rules["root"]

    def _rule(self, kind: str) -> _Rule:
        rule = _Rule(f"{kind} in rule {self._owner!r}")
        self.created.append(rule)
        return rule

    def _alternatives(self, alternatives: tuple[tuple[Any, ...], ...]) -> tuple[tuple[tuple[Any, ...], ...], ...]:
        return tuple(dict.fromkeys(self._sequence(sequence) for sequence in alternatives))

    def _sequence(self, sequence: tuple[Any, ...]) -> tuple[tuple[Any, ...], ...]:
        return tuple(item for element in sequence for item in self._items(element))

    def _items(self, element: Any) -> list[tuple[Any, ...]]:
        if isinstance(element, _Literal):
            return [_literal(element.text.encode("utf-8"))] if element.text else []
        if isinstance(element, _Class):
            return [_class_item(element.ranges)]
        if isinstance(element, _Reference):
            rule = self.rules.get(element.name)
            if rule is None:
                raise GrammarError(f"rule {self._owner!r} refers to the undefined rule {element.name!r}")
            return [_value(rule)]
        if isinstance(element, _Group):
            alternatives = self._alternatives(element.alternatives)
            if len(alternatives) == 1:
                return list(alternatives[0])
            group = self._rule("a group")
            group.alternatives = alternatives
            return [_value(group)]
        assert isinstance(element, _Repeat)
        items = self._items(element.element)
        if not items:
            return []
        if len(items) == 1:
            item = items[0]
        else:
            body = self._rule("a repeated sequence")
            body.alternatives = (tuple(items),)
            item = _value(body)
        result = [item] * element.minimum
        if element.maximum is None:
            loop = self._rule("a repetition")
            loop.alternatives = ((item, _value(loop)), ())
            self.repetitions.add(id(loop))
            result.append(_value(loop))
        elif element.maximum > element.minimum:
            optional = self._rule("an optional part")
            optional.alternatives = ((item,), ())
            for _ in range(element.maximum - element.minimum - 1):
                outer = self._rule("an optional part")
                outer.alternatives = ((item, _value(optional)), ())
                optional = outer
            result.append(_value(optional))
        return result


def _reachable(root: _Rule) -> list[_Rule]:
    """The rules ``root`` reaches, in the order a depth-first walk over their written items first meets them."""
    seen: dict[int, _Rule] = {}
    pending = [root]
    while pending:
        rule = pending.pop()
        if id(rule) in seen:
            continue
        seen[id(rule)] = rule
        references = [item[1] for alternative in rule.alternatives for item in alternative if item[0] == _VALUE]
        pending.extend(reversed(references))
    return list(seen.values())


def _fixpoint(rules: list[_Rule], holds: Any) -> set[int]:
    """Ids of the rules with an alternative all of whose items satisfy ``holds(item, found)``, iterated until no
    rule is added."""
    found: set[int] = set()
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if id(rule) not in found and any(all(holds(i, found) for i in alt) for alt in rule.alternatives):
                found.add(id(rule))
                changed = True
    return found


def _productive(item: tuple[Any, ...], found: set[int]) -> bool:
    return item[0] != _VALUE or id(item[1]) in found


def _nullable(item: tuple[Any, ...], found: set[int]) -> bool:
    return item[0] == _VALUE and id(item[1]) in found


def _check_left_recursion(rules: list[_Rule], repetitions: set[int]) -> None:
    nullable = _fixpoint(rules, _nullable)
    edges: dict[int, list[_Rule]] = {}
    for rule in rules:
        targets: list[_Rule] = []
        for alternative in rule.alternatives:
            for item in alternative:
                if item[0] != _VALUE:
                    break
                targets.append(item[1])
                if id(item[1]) not in nullable:
                    break
        edges[id(rule)] = targets
    state: dict[int, int] = {}  # 1 on the path, 2 done
    path: list[_Rule] = []

    def visit(rule: _Rule) -> None:
        state[id(rule)] = 1
        path.append(rule)
        for target in edges[id(rule)]:
            if state.get(id(target)) == 1:
                cycle = path[path.index(target) :]
                loops = [r for r in cycle if id(r) in repetitions]
                if loops:
                    raise GrammarError(f"{loops[0].name}: it repeats something that can match nothing")
                names = " -> ".join(r.name for r in [*cycle, target])
                raise GrammarError(f"left recursion is not supported: {names}")
            if id(target) not in state:
                visit(target)
        path.pop()
        state[id(rule)] = 2

    for rule in rules:
        if id(rule) not in state:
            visit(rule)


@functools.lru_cache(maxsize=64)
def compile_gbnf(text: str) -> Grammar:
    """The grammar of the GBNF ``text`` (see the module docstring); raises :class:`GrammarError` when it is not
    supported."""
    compiler = _Compiler(_Parser(text).parse())
    root = compiler.compile()
    rules = _reachable(root)
    productive = _fixpoint(rules, _productive)
    if id(root) not in productive:
        raise GrammarError("the grammar derives no text: every way through 'root' recurses forever")
    for rule in rules:
        rule.alternatives = tuple(a for a in rule.alternatives if all(_productive(i, productive) for i in a))
    rules = _reachable(root)
    _check_left_recursion(rules, compiler.repetitions)
    return Grammar([_value(root)])


__all__ = ["compile_gbnf"]
