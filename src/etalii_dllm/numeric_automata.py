"""Byte automata of JSON numbers under ``minimum``/``maximum`` (and their exclusive forms) and ``multipleOf``.

Constrained decoding (:mod:`etalii_dllm.grammar`) writes a constrained number as plain decimal text: an optional
``-``, the integer part without leading zeros and, for ``number``, an optional fraction, never an exponent and never
a negative zero. Bounds and divisors are exact decimals (a JSON number is taken as written: an integer, or the
shortest decimal that reads back as the same double), and the automaton accepts exactly the texts whose decimal
value satisfies them, so the checks involve no floating point arithmetic at all.

- A one-sided bound is a regular expression over the texts: for ``x >= 2.5``, an integer part of at least 3 with any
  fraction, or ``2`` with a fraction of at least ``.5`` (digit strings compared from the left). Negative values are
  the mirror image.
- ``multipleOf`` ``p / 10^k`` (``p`` a positive integer) holds when ``|x| * 10^k`` is an integer divisible by
  ``p``: a table automaton tracks that remainder digit by digit.
- The automata are intersected (:func:`etalii_dllm.regexp.intersect`) and trimmed, so decoding never runs into a
  dead end; constraints no number satisfies are refused.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from etalii_dllm.regexp import MAX_DFA_STATES, Dfa, _trimmed, compile_regex, intersect

Bound = tuple[Decimal, bool] | None
"""A bound and whether it is exclusive; ``None``: no bound."""

_INTEGER = r"0|-?[1-9]\d*"
_NUMBER = r"(?:0|[1-9]\d*)(?:\.\d+)?|-(?:0\.\d*[1-9]\d*|[1-9]\d*(?:\.\d+)?)"
"""Decimal texts without an exponent and without a negative zero."""
_ANY_FRACTION = r"(?:\.\d+)?"


def _error(message: str) -> Exception:
    from etalii_dllm.grammar import GrammarError

    return GrammarError(message)


def exact(value: Any, name: str) -> Decimal:
    """A JSON number as an exact decimal: an integer as is, a float as its shortest round-trip decimal (a decimal
    from a merged schema as is)."""
    if isinstance(value, Decimal) and value.is_finite():
        return value
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise _error(f"'{name}' must be a finite number")
    return Decimal(value) if isinstance(value, int) else Decimal(repr(value))


def decimal_bounds(schema: Mapping[str, Any]) -> tuple[Bound, Bound]:
    """The lower and upper bound of ``minimum``/``maximum``/``exclusiveMinimum``/``exclusiveMaximum`` (numbers, or
    the draft 4 booleans); of two lower (upper) bounds the tighter one counts."""

    def tighter(first: Bound, second: Bound, lower: bool) -> Bound:
        if first is None or second is None:
            return first or second
        if first[0] != second[0]:
            return first if (first[0] > second[0]) == lower else second
        return (first[0], first[1] or second[1])

    low: Bound = None
    high: Bound = None
    if "minimum" in schema:
        low = (exact(schema["minimum"], "minimum"), schema.get("exclusiveMinimum") is True)
    if "maximum" in schema:
        high = (exact(schema["maximum"], "maximum"), schema.get("exclusiveMaximum") is True)
    if "exclusiveMinimum" in schema and not isinstance(schema["exclusiveMinimum"], bool):
        low = tighter(low, (exact(schema["exclusiveMinimum"], "exclusiveMinimum"), True), lower=True)
    if "exclusiveMaximum" in schema and not isinstance(schema["exclusiveMaximum"], bool):
        high = tighter(high, (exact(schema["exclusiveMaximum"], "exclusiveMaximum"), True), lower=False)
    return low, high


def _parts(value: Decimal) -> tuple[int, str]:
    """The integer part and the fraction digits (no trailing zeros) of ``|value|``."""
    text = format(abs(value), "f")
    whole, _, fraction = text.partition(".")
    return int(whole), fraction.rstrip("0")


def _fraction(digits: str, relation: str) -> tuple[bool, str | None]:
    """The digit strings ``d`` with ``0.d`` related to ``0.digits`` (``ge``, ``gt``, ``le``, ``lt``), as (whether the
    empty string is one, a regex of the non-empty ones or ``None``)."""
    if not digits:
        return {"ge": (True, r"\d+"), "gt": (False, r"\d*[1-9]\d*"), "le": (True, "0+"), "lt": (False, None)}[relation]
    first = int(digits[0])
    rest_empty, rest = _fraction(digits[1:], relation)
    branches = []
    if relation in ("ge", "gt") and first < 9:
        branches.append(f"[{first + 1}-9]\\d*")
    if relation in ("le", "lt") and first > 0:
        branches.append(f"[0-{first - 1}]\\d*")
    if rest is not None:
        branches.append(f"{first}(?:{rest})?" if rest_empty else f"{first}(?:{rest})")
    elif rest_empty:
        branches.append(str(first))
    return relation in ("le", "lt"), "|".join(branches) or None


def _fraction_text(digits: str, relation: str) -> str | None:
    """The optional ``.digits`` part of a text whose fraction is related to ``0.digits``; ``None``: none is."""
    empty, regex = _fraction(digits, relation)
    if regex is None:
        return "" if empty else None
    return f"(?:\\.(?:{regex}))?" if empty else f"\\.(?:{regex})"


def _magnitude(value: Decimal, strict: bool, above: bool) -> str | None:
    """A regex of the non-negative texts at least (``above``) or at most ``|value|``, ``strict``ly or not;
    ``None``: there are none."""
    from etalii_dllm.grammar import _magnitudes

    whole, digits = _parts(value)
    relation = ("gt" if strict else "ge") if above else ("lt" if strict else "le")
    branches = []
    if above:
        branches += [f"(?:{b}){_ANY_FRACTION}" for b in _magnitudes(whole + 1, None)]
    elif whole > 0:
        branches += [f"(?:{b}){_ANY_FRACTION}" for b in _magnitudes(0, whole - 1)]
    tail = _fraction_text(digits, relation)
    if tail is not None:
        branches.append(f"{whole}{tail}")
    return "|".join(branches) or None


def _one_sided(bound: tuple[Decimal, bool], lower: bool) -> str:
    """A regex of the signed texts on the right side of ``bound`` (a lower bound when ``lower``)."""
    value, strict = bound
    everything = rf"(?:0|[1-9]\d*){_ANY_FRACTION}"
    # Above a non-negative bound or below a negative one, every text has the same sign and a magnitude beyond
    # ``|value|``; otherwise every text of the other sign qualifies, and those of this sign up to ``|value|``.
    beyond = (value >= 0) == lower
    magnitudes = _magnitude(value, strict, above=beyond)
    if beyond:
        assert magnitudes is not None
        return magnitudes if lower else f"-(?:{magnitudes})"
    if lower:
        return f"{everything}|-(?:{magnitudes})"
    return f"-{everything}" if magnitudes is None else f"-{everything}|{magnitudes}"


def _divisible(divisor: Decimal) -> Dfa:
    """The texts ``x`` with ``|x| * 10^k`` an integer divisible by ``p``, where ``divisor = p / 10^k``."""
    if divisor <= 0:
        raise _error("'multipleOf' must be greater than 0")
    _, digit_tuple, exponent = divisor.normalize().as_tuple()
    assert isinstance(exponent, int)
    p = int("".join(map(str, digit_tuple))) * 10 ** max(exponent, 0)
    k = max(-exponent, 0)
    # States: ("start"), ("sign"), ("whole", r), ("point", r), ("fraction", r, j) with r the remainder of the digits
    # so far (fraction digits j <= k included).
    start: tuple[Any, ...] = ("start",)
    ids = {start: 0}
    states = [start]
    table: list[dict[int, int]] = []
    accepting: list[bool] = []
    index = 0
    while index < len(states):
        state = states[index]
        index += 1
        moves: dict[int, tuple[Any, ...]] = {}
        kind = state[0]
        if kind == "start":
            moves[0x2D] = ("sign",)
        if kind in ("start", "sign"):
            for digit in range(10):
                moves[0x30 + digit] = ("whole", digit % p)
        elif kind == "whole":
            for digit in range(10):
                moves[0x30 + digit] = ("whole", (state[1] * 10 + digit) % p)
            moves[0x2E] = ("point", state[1])
        elif kind in ("point", "fraction"):
            read = 0 if kind == "point" else state[2]
            for digit in range(10):
                if read < k:
                    moves[0x30 + digit] = ("fraction", (state[1] * 10 + digit) % p, read + 1)
                elif digit == 0:
                    moves[0x30] = ("fraction", state[1], read)
        row: dict[int, int] = {}
        for byte, target in sorted(moves.items()):
            if target not in ids:
                if len(states) >= MAX_DFA_STATES:
                    raise _error(f"'multipleOf' {divisor} is too fine for constrained decoding")
                ids[target] = len(states)
                states.append(target)
            row[byte] = ids[target]
        table.append(row)
        if kind == "whole":
            accepting.append(state[1] * pow(10, k, p) % p == 0)
        elif kind == "fraction":
            accepting.append(state[1] * pow(10, k - state[2], p) % p == 0)
        else:
            accepting.append(False)
    return _trimmed(f"a multiple of {divisor}", table, accepting)


@functools.lru_cache(maxsize=256)
def number_automaton(low: Bound, high: Bound, multiple: Decimal | None, integer: bool) -> Dfa:
    """The texts of the numbers (``integer``: the integers) within ``low`` and ``high`` that are multiples of
    ``multiple``, written as described in the module docstring."""
    from etalii_dllm.grammar import GrammarError, _Unsatisfiable

    dfa = compile_regex(_INTEGER if integer else _NUMBER)
    try:
        for bound, lower in ((low, True), (high, False)):
            if bound is not None:
                dfa = intersect(dfa, compile_regex(_one_sided(bound, lower)))
        if multiple is not None:
            dfa = intersect(dfa, _divisible(multiple))
    except GrammarError as error:
        if "no text satisfies" not in str(error):
            raise
        kind = "integer" if integer else "number"
        raise _Unsatisfiable(f"no {kind} satisfies the bounds and 'multipleOf' of the schema") from None
    return dfa
