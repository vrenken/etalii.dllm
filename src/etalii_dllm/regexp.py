"""Regular expressions compiled to byte-level automata, for regex-constrained decoding.

A pattern describes the whole output (it is matched in full, as if wrapped in ``^...$``). It is parsed into code
point sets, the sets are encoded as UTF-8 byte ranges, the result is built into a Thompson NFA over bytes and that
is determinised into a :class:`Dfa`. Only well-formed UTF-8 is accepted, and every state of the automaton can still
reach a match, so constrained decoding never runs into a dead end.

Supported syntax: literal characters, ``.`` (any character but a newline), classes ``[...]`` and ``[^...]`` with
ranges (any code points), the escapes ``\\d \\w \\s`` and their negations (ASCII meanings: ``[0-9]``,
``[A-Za-z0-9_]``, ``[ \\t\\n\\r\\f\\v]``), ``\\n \\t \\r \\f \\v \\0``, ``\\xHH``, ``\\uHHHH``, an escaped
punctuation character, groups ``(...)``, ``(?:...)``, ``(?P<name>...)`` and ``(?<name>...)``, alternation ``|``,
the quantifiers ``* + ? {m} {m,} {m,n}`` (a trailing lazy ``?`` is accepted; it makes no difference to a full
match) and ``^``/``$`` at the very start/end. Everything else (backreferences, lookaround, word boundaries, flags)
raises :class:`etalii_dllm.grammar.GrammarError`.

Determinism: the automaton is a pure function of the pattern. NFA states are numbered as they are built, DFA states
in the order a breadth-first walk over the bytes 0..255 first reaches them, so the allowed tokens never depend on
set iteration order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

MAX_REPEAT = 1000
"""Largest count a ``{m,n}`` quantifier may use."""
MAX_NFA_STATES = 50_000
MAX_DFA_STATES = 10_000

Ranges = tuple[tuple[int, int], ...]
"""Sorted, disjoint, non-adjacent code point ranges without surrogates."""

_SURROGATES = (0xD800, 0xDFFF)
_ALL: Ranges = ((0, 0xD7FF), (0xE000, 0x10FFFF))
_DIGIT: Ranges = ((0x30, 0x39),)
_WORD: Ranges = ((0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A))
_SPACE: Ranges = ((0x09, 0x0D), (0x20, 0x20))
_SIMPLE_ESCAPES = {"n": 0x0A, "t": 0x09, "r": 0x0D, "f": 0x0C, "v": 0x0B, "0": 0x00}


def _error(message: str) -> Exception:
    from etalii_dllm.grammar import GrammarError

    return GrammarError(message)


def _normalise(ranges: Sequence[tuple[int, int]]) -> Ranges:
    """Sorted and merged, with the surrogates taken out."""
    pieces: list[tuple[int, int]] = []
    for low, high in ranges:
        if low > high:
            continue
        if low < _SURROGATES[0] and high > _SURROGATES[1]:
            pieces += [(low, _SURROGATES[0] - 1), (_SURROGATES[1] + 1, high)]
        elif _SURROGATES[0] <= low <= _SURROGATES[1] or _SURROGATES[0] <= high <= _SURROGATES[1]:
            if low < _SURROGATES[0]:
                pieces.append((low, _SURROGATES[0] - 1))
            if high > _SURROGATES[1]:
                pieces.append((_SURROGATES[1] + 1, high))
        else:
            pieces.append((low, high))
    merged: list[tuple[int, int]] = []
    for low, high in sorted(pieces):
        if merged and low <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return tuple(merged)


def _complement(ranges: Ranges) -> Ranges:
    result: list[tuple[int, int]] = []
    start = 0
    for low, high in ranges:
        if low > start:
            result.append((start, low - 1))
        start = high + 1
    if start <= 0x10FFFF:
        result.append((start, 0x10FFFF))
    return _normalise(result)


# -- parsing ------------------------------------------------------------------------------------------------------

# Nodes: ("chars", ranges) | ("concat", [nodes]) | ("alt", [nodes]) | ("repeat", node, minimum, maximum or None)
Node = tuple[Any, ...]
_EMPTY: Node = ("concat", [])


class _Parser:
    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.position = 0

    def parse(self) -> Node:
        if self.pattern.startswith("^"):
            self.position = 1
        node = self._alternation()
        if self.position < len(self.pattern):
            raise _error(f"unbalanced ')' at position {self.position} of the regex")
        return node

    def _peek(self) -> str | None:
        return self.pattern[self.position] if self.position < len(self.pattern) else None

    def _take(self) -> str:
        if self.position >= len(self.pattern):
            raise _error("the regex ends unexpectedly")
        char = self.pattern[self.position]
        self.position += 1
        return char

    def _alternation(self) -> Node:
        branches = [self._concatenation()]
        while self._peek() == "|":
            self.position += 1
            branches.append(self._concatenation())
        return branches[0] if len(branches) == 1 else ("alt", branches)

    def _concatenation(self) -> Node:
        items: list[Node] = []
        while (char := self._peek()) is not None and char not in "|)":
            if char == "$" and self.position == len(self.pattern) - 1:
                self.position += 1
                break
            items.append(self._quantified(self._atom()))
        return items[0] if len(items) == 1 else ("concat", items)

    def _quantified(self, atom: Node) -> Node:
        char = self._peek()
        if char == "*":
            bounds: tuple[int, int | None] | None = (0, None)
        elif char == "+":
            bounds = (1, None)
        elif char == "?":
            bounds = (0, 1)
        elif char == "{":
            bounds = self._braces()
            if bounds is None:
                return atom
        else:
            return atom
        if char != "{":
            self.position += 1
        if self._peek() == "?":
            self.position += 1  # lazy: the same strings match in full
        if self._peek() in ("*", "+", "?") or (self._peek() == "{" and self._braces(peek=True) is not None):
            raise _error(f"nested quantifier at position {self.position} of the regex")
        return ("repeat", atom, bounds[0], bounds[1])

    def _braces(self, peek: bool = False) -> tuple[int, int | None] | None:
        """``{m}``, ``{m,}`` or ``{m,n}`` at the position (consumed unless ``peek``); ``None`` when the brace is a
        literal."""
        end = self.pattern.find("}", self.position)
        if end < 0:
            return None
        body = self.pattern[self.position + 1 : end]
        low, comma, high = body.partition(",")
        if not low.isdigit() or not (high.isdigit() or high == "") or (not comma and high):
            return None
        minimum = int(low)
        maximum = None if comma and high == "" else int(high if comma else low)
        if maximum is not None and maximum < minimum:
            raise _error(f"bad quantifier {{{body}}} in the regex")
        if max(minimum, maximum or 0) > MAX_REPEAT:
            raise _error(f"quantifier counts above {MAX_REPEAT} are not supported")
        if not peek:
            self.position = end + 1
        return minimum, maximum

    def _atom(self) -> Node:
        char = self._take()
        if char == "(":
            if self._peek() == "?":
                self.position += 1
                if self._peek() == ":":
                    self.position += 1
                elif self.pattern.startswith(("P<", "<"), self.position) and self.pattern[
                    self.position + (2 if self._peek() == "P" else 1) :
                ][:1] not in ("=", "!", ""):
                    end = self.pattern.find(">", self.position)
                    if end < 0:
                        raise _error("unterminated group name in the regex")
                    self.position = end + 1
                else:
                    raise _error(f"unsupported group syntax at position {self.position - 2} of the regex")
            node = self._alternation()
            if self._peek() != ")":
                raise _error("missing ')' in the regex")
            self.position += 1
            return node
        if char == "[":
            return ("chars", self._class())
        if char == ".":
            return ("chars", _complement(((0x0A, 0x0A),)))
        if char == "\\":
            return ("chars", self._escape(in_class=False))
        if char in "*+?":
            raise _error(f"nothing to repeat at position {self.position - 1} of the regex")
        if char in "^$":
            raise _error(f"'{char}' is only supported at the start/end of the regex")
        return ("chars", ((ord(char), ord(char)),))

    def _escape(self, in_class: bool) -> Ranges:
        char = self._take()
        sets = {"d": _DIGIT, "w": _WORD, "s": _SPACE}
        if char in sets:
            return sets[char]
        if char.lower() in sets:
            return _complement(sets[char.lower()])
        if char in _SIMPLE_ESCAPES:
            code = _SIMPLE_ESCAPES[char]
            return ((code, code),)
        if char in ("x", "u"):
            digits = self.pattern[self.position : self.position + (2 if char == "x" else 4)]
            if len(digits) != (2 if char == "x" else 4) or any(d not in "0123456789abcdefABCDEF" for d in digits):
                raise _error(f"bad \\{char} escape in the regex")
            self.position += len(digits)
            code = int(digits, 16)
            return _normalise([(code, code)]) or _error_ranges("a surrogate code point cannot be matched")
        if char.isalnum() or char == "_":
            raise _error(f"unsupported escape \\{char} in the regex")
        return ((ord(char), ord(char)),)

    def _class(self) -> Ranges:
        negate = self._peek() == "^"
        if negate:
            self.position += 1
        items: list[tuple[int, int]] = []
        first = True
        while True:
            char = self._take()
            if char == "]" and not first:
                break
            first = False
            if char == "\\":
                ranges = self._escape(in_class=True)
                if len(ranges) != 1 or ranges[0][0] != ranges[0][1]:
                    items += ranges
                    continue
                low = ranges[0][0]
            elif char == "[" and self._peek() == ":":
                raise _error("POSIX character classes are not supported in the regex")
            else:
                low = ord(char)
            if self._peek() == "-" and self.pattern[self.position + 1 : self.position + 2] not in ("]", ""):
                self.position += 1
                end = self._take()
                if end == "\\":
                    ranges = self._escape(in_class=True)
                    if len(ranges) != 1 or ranges[0][0] != ranges[0][1]:
                        raise _error("a class range cannot end in a character set")
                    high = ranges[0][0]
                else:
                    high = ord(end)
                if high < low:
                    raise _error("a class range runs backwards in the regex")
                items.append((low, high))
            else:
                items.append((low, low))
        ranges = _normalise(items)
        return _complement(ranges) if negate else ranges


def _error_ranges(message: str) -> Ranges:
    raise _error(message)


# -- UTF-8 -------------------------------------------------------------------------------------------------------


def _encode(code: int) -> bytes:
    return chr(code).encode("utf-8")


def utf8_sequences(low: int, high: int) -> list[list[tuple[int, int]]]:
    """Byte-range sequences whose concatenations are exactly the UTF-8 encodings of ``low..high`` (surrogates
    left out)."""
    result: list[list[tuple[int, int]]] = []
    pending = list(reversed(_normalise([(low, high)])))
    while pending:
        start, end = pending.pop()
        split = False
        for limit in (0x7F, 0x7FF, 0xFFFF):
            if start <= limit < end:
                pending += [(limit + 1, end), (start, limit)]
                split = True
                break
        if split:
            continue
        if end <= 0x7F:
            result.append([(start, end)])
            continue
        for i in range(1, 4):
            mask = (1 << (6 * i)) - 1
            if start & ~mask != end & ~mask:
                if start & mask:
                    pending += [((start | mask) + 1, end), (start, start | mask)]
                    split = True
                    break
                if end & mask != mask:
                    pending += [(end & ~mask, end), (start, (end & ~mask) - 1)]
                    split = True
                    break
        if split:
            continue
        result.append(list(zip(_encode(start), _encode(end), strict=True)))
    return result


# -- automata ----------------------------------------------------------------------------------------------------


class _Nfa:
    def __init__(self) -> None:
        self.epsilon: list[list[int]] = []
        self.moves: list[list[tuple[int, int, int]]] = []

    def state(self) -> int:
        if len(self.epsilon) >= MAX_NFA_STATES:
            raise _error("the regex is too large")
        self.epsilon.append([])
        self.moves.append([])
        return len(self.epsilon) - 1

    def build(self, node: Node) -> tuple[int, int]:
        kind = node[0]
        if kind == "chars":
            start, end = self.state(), self.state()
            for low, high in node[1]:
                for sequence in utf8_sequences(low, high):
                    current = start
                    for i, (byte_low, byte_high) in enumerate(sequence):
                        target = end if i == len(sequence) - 1 else self.state()
                        self.moves[current].append((byte_low, byte_high, target))
                        current = target
            return start, end
        if kind == "concat":
            start = end = self.state()
            for item in node[1]:
                first, last = self.build(item)
                self.epsilon[end].append(first)
                end = last
            return start, end
        if kind == "alt":
            start, end = self.state(), self.state()
            for branch in node[1]:
                first, last = self.build(branch)
                self.epsilon[start].append(first)
                self.epsilon[last].append(end)
            return start, end
        _, item, minimum, maximum = node
        start = end = self.state()
        for _ in range(minimum):
            first, last = self.build(item)
            self.epsilon[end].append(first)
            end = last
        if maximum is None:
            first, last = self.build(item)
            loop = self.state()
            self.epsilon[end].append(loop)
            self.epsilon[loop].append(first)
            self.epsilon[last].append(loop)
            return start, loop
        exit_state = self.state()
        for _ in range(maximum - minimum):
            self.epsilon[end].append(exit_state)
            first, last = self.build(item)
            self.epsilon[end].append(first)
            end = last
        self.epsilon[end].append(exit_state)
        return start, exit_state

    def closure(self, states: set[int]) -> frozenset[int]:
        seen = set(states)
        stack = list(states)
        while stack:
            for target in self.epsilon[stack.pop()]:
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        return frozenset(seen)


class Dfa:
    """A deterministic automaton over bytes. State 0 is the start; :meth:`step` returns -1 for a byte that cannot
    follow."""

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        nfa = _Nfa()
        start, accept = nfa.build(_Parser(pattern).parse())
        first = nfa.closure({start})
        ids = {first: 0}
        sets = [first]
        self._table: list[dict[int, int]] = []
        self._accepting: list[bool] = []
        index = 0
        while index < len(sets):
            current = sets[index]
            index += 1
            moves: dict[int, set[int]] = {}
            for state in sorted(current):
                for low, high, target in nfa.moves[state]:
                    for byte in range(low, high + 1):
                        moves.setdefault(byte, set()).add(target)
            row: dict[int, int] = {}
            for byte in sorted(moves):
                following = nfa.closure(moves[byte])
                if following not in ids:
                    if len(sets) >= MAX_DFA_STATES:
                        raise _error("the regex is too large")
                    ids[following] = len(sets)
                    sets.append(following)
                row[byte] = ids[following]
            self._table.append(row)
            self._accepting.append(accept in current)

    @property
    def size(self) -> int:
        return len(self._table)

    def step(self, state: int, byte: int) -> int:
        return self._table[state].get(byte, -1)

    def accepting(self, state: int) -> bool:
        return self._accepting[state]

    def reads(self, state: int) -> bool:
        """Whether any byte can follow (else the match is complete)."""
        return bool(self._table[state])

    def matches(self, data: bytes) -> bool:
        state = 0
        for byte in data:
            state = self.step(state, byte)
            if state < 0:
                return False
        return self.accepting(state)


def compile_regex(pattern: str) -> Dfa:
    """The automaton of ``pattern``; raises :class:`etalii_dllm.grammar.GrammarError` for unsupported syntax."""
    return Dfa(pattern)
