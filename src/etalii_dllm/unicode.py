"""Unicode behaviour pinned to one Unicode version, whatever Python or ``regex`` is installed (issue #98).

``unicodedata``, ``str.lower`` and the ``regex`` package's ``\\p{..}`` classes use the Unicode tables their own
release shipped with (Python 3.11: Unicode 14.0, 3.12: 15.0, 3.13: 15.1), so the same text could tokenize differently
on two machines. The tokenizer uses this module instead: normalisation (NFC, NFD, NFKC, NFKD), lower-casing and the
general categories in regular expressions all come from ``unicode_data.json.gz``, generated once from Python 3.13's
``unicodedata`` by ``scripts/generate_unicode_data.py``. Characters outside those tables (unassigned in
:func:`unicode_version`) are left unchanged and have category ``Cn``, on every installation.

The rest of the ``regex`` engine (matching semantics, ``\\s``, case folding) and ``str.isspace`` have not changed in
the Unicode versions we support; ``tests/test_unicode.py`` pins them so that a change would fail loudly.
"""

from __future__ import annotations

import bisect
import gzip
import json
from functools import cache, lru_cache
from importlib import resources
from typing import Any

import regex


@cache
def _data() -> dict[str, Any]:
    raw = resources.files("etalii_dllm").joinpath("unicode_data.json.gz").read_bytes()
    return json.loads(gzip.decompress(raw))


def unicode_version() -> str:
    """The Unicode version the tables were generated from."""
    return str(_data()["unicode_version"])


class _RangeTable:
    """Sorted, disjoint ``[start, end, value]`` ranges (end inclusive) with bisect lookup."""

    def __init__(self, ranges: list[list[Any]]) -> None:
        self.starts = [r[0] for r in ranges]
        self.ends = [r[1] for r in ranges]
        self.values = [r[2] if len(r) > 2 else True for r in ranges]

    def get(self, cp: int, default: Any = None) -> Any:
        i = bisect.bisect_right(self.starts, cp) - 1
        if i >= 0 and cp <= self.ends[i]:
            return self.values[i]
        return default


@cache
def _tables() -> dict[str, Any]:
    data = _data()
    decompositions = {int(cp): (tag, tuple(parts)) for cp, (tag, parts) in data["decompositions"].items()}
    excluded = set(data["composition_exclusions"])
    compositions = {
        parts: cp for cp, (tag, parts) in decompositions.items() if not tag and len(parts) == 2 and cp not in excluded
    }
    return {
        "categories": _RangeTable(data["categories"]),
        "combining": _RangeTable(data["combining"]),
        "decompositions": decompositions,
        "compositions": compositions,
        "lower": {int(cp): "".join(map(chr, cps)) for cp, cps in data["lower"].items()},
        "case_ignorable": _RangeTable(data["case_ignorable"]),
        "cased": _RangeTable(data["cased"]),
    }


def category(character: str) -> str:
    """General category (``Lu``, ``Nd``, ...); ``Cn`` for characters unassigned in :func:`unicode_version`."""
    return str(_tables()["categories"].get(ord(character), "Cn"))


def combining(character: str) -> int:
    """Canonical combining class."""
    return int(_tables()["combining"].get(ord(character), 0))


# --- normalisation (Unicode Standard Annex #15) --------------------------------------------------------------------

_S_BASE, _L_BASE, _V_BASE, _T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
_L_COUNT, _V_COUNT, _T_COUNT = 19, 21, 28
_N_COUNT = _V_COUNT * _T_COUNT
_S_COUNT = _L_COUNT * _N_COUNT


def _decompose(cp: int, compatibility: bool, out: list[int]) -> None:
    if _S_BASE <= cp < _S_BASE + _S_COUNT:
        index = cp - _S_BASE
        out.append(_L_BASE + index // _N_COUNT)
        out.append(_V_BASE + (index % _N_COUNT) // _T_COUNT)
        if index % _T_COUNT:
            out.append(_T_BASE + index % _T_COUNT)
        return
    entry = _tables()["decompositions"].get(cp)
    if entry is None or (entry[0] and not compatibility):
        out.append(cp)
        return
    for part in entry[1]:
        _decompose(part, compatibility, out)


def _canonical_order(cps: list[int]) -> None:
    """Stable sort of every run of non-starters by combining class."""
    ccc = _tables()["combining"]
    i = 0
    while i < len(cps):
        if ccc.get(cps[i], 0) == 0:
            i += 1
            continue
        j = i
        while j < len(cps) and ccc.get(cps[j], 0) != 0:
            j += 1
        cps[i:j] = sorted(cps[i:j], key=lambda cp: ccc.get(cp, 0))
        i = j


def _compose(cps: list[int]) -> list[int]:
    """Canonical composition of a canonically ordered sequence."""
    ccc = _tables()["combining"]
    compositions = _tables()["compositions"]
    out: list[int] = []
    starter = -1  # index in out of the last starter
    for cp in cps:
        cls = ccc.get(cp, 0)
        # Everything after the starter is a non-starter in canonical order, so only the last one can block.
        if starter >= 0 and (len(out) - 1 == starter or 0 < ccc.get(out[-1], 0) < cls):
            first = out[starter]
            if _L_BASE <= first < _L_BASE + _L_COUNT and _V_BASE <= cp < _V_BASE + _V_COUNT:
                composed: int | None = _S_BASE + ((first - _L_BASE) * _V_COUNT + (cp - _V_BASE)) * _T_COUNT
            elif (
                _S_BASE <= first < _S_BASE + _S_COUNT
                and (first - _S_BASE) % _T_COUNT == 0
                and _T_BASE < cp < _T_BASE + _T_COUNT
            ):
                composed = first + (cp - _T_BASE)
            else:
                composed = compositions.get((first, cp))
            if composed is not None:
                out[starter] = composed
                continue
        if cls == 0:
            starter = len(out)
        out.append(cp)
    return out


def normalize(form: str, text: str) -> str:
    """``unicodedata.normalize`` with this module's Unicode version."""
    if form not in ("NFC", "NFD", "NFKC", "NFKD"):
        raise ValueError(f"invalid normalization form {form!r}")
    if text.isascii():
        return text
    cps: list[int] = []
    compatibility = form in ("NFKC", "NFKD")
    for character in text:
        _decompose(ord(character), compatibility, cps)
    _canonical_order(cps)
    if form in ("NFC", "NFKC"):
        cps = _compose(cps)
    return "".join(map(chr, cps))


# --- case -----------------------------------------------------------------------------------------------------------


def _final_sigma(text: str, index: int) -> bool:
    """Python's (and Unicode's) Final_Sigma condition for the capital sigma at ``index``."""
    ignorable, cased = _tables()["case_ignorable"], _tables()["cased"]
    j = index - 1
    while j >= 0 and ignorable.get(ord(text[j]), False):
        j -= 1
    if j < 0 or not cased.get(ord(text[j]), False):
        return False
    j = index + 1
    while j < len(text) and ignorable.get(ord(text[j]), False):
        j += 1
    return j == len(text) or not cased.get(ord(text[j]), False)


def lower(text: str) -> str:
    """``str.lower`` (full case mapping, final sigma) with this module's Unicode version."""
    if text.isascii():
        return text.lower()
    mapping = _tables()["lower"]
    out = []
    for index, character in enumerate(text):
        if character == "\u03a3":
            out.append("\u03c2" if _final_sigma(text, index) else "\u03c3")
        else:
            out.append(mapping.get(ord(character), character))
    return "".join(out)


# --- regular expressions --------------------------------------------------------------------------------------------

_CATEGORY_GROUPS = {"L", "M", "N", "P", "S", "Z", "C"}
_CATEGORY_ALIASES = {
    "Letter": "L",
    "Mark": "M",
    "Number": "N",
    "Punctuation": "P",
    "Symbol": "S",
    "Separator": "Z",
    "Other": "C",
}


@cache
def _category_ranges(name: str) -> tuple[tuple[int, int], ...]:
    name = _CATEGORY_ALIASES.get(name, name)
    if name.startswith("L&"):
        wanted: set[str] = {"Lu", "Ll", "Lt"}
    elif name in _CATEGORY_GROUPS:
        wanted = {c for c in _all_categories() if c.startswith(name)}
    else:
        wanted = {name}
    if not wanted <= _all_categories() | {"Cn"}:
        raise ValueError(f"unsupported Unicode property {name!r}")
    out: list[tuple[int, int]] = []
    table = _tables()["categories"]
    spans = list(zip(table.starts, table.ends, table.values, strict=True))
    if "Cn" in wanted:  # unassigned: the gaps between the table's ranges (surrogates are Cs, so assigned)
        previous = 0
        gaps = []
        for start, end, _ in spans:
            if start > previous:
                gaps.append((previous, start - 1, "Cn"))
            previous = end + 1
        if previous < 0x110000:
            gaps.append((previous, 0x10FFFF, "Cn"))
        spans = sorted(spans + gaps)
    for start, end, value in spans:
        if value in wanted:
            if out and out[-1][1] == start - 1:
                out[-1] = (out[-1][0], end)
            else:
                out.append((start, end))
    return tuple(out)


@cache
def _all_categories() -> frozenset[str]:
    return frozenset(_tables()["categories"].values)


def _complement(ranges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    out = []
    previous = 0
    for start, end in ranges:
        if start > previous:
            out.append((previous, start - 1))
        previous = end + 1
    if previous <= 0x10FFFF:
        out.append((previous, 0x10FFFF))
    return tuple(out)


def _class_body(ranges: tuple[tuple[int, int], ...]) -> str:
    def escape(cp: int) -> str:
        return f"\\U{cp:08x}"

    return "".join(escape(a) if a == b else f"{escape(a)}-{escape(b)}" for a, b in ranges)


def _property(pattern: str, i: int) -> tuple[str, bool, int]:
    """Parses ``\\p{..}``/``\\pL`` at ``pattern[i]`` (the backslash): (name, negated, next index)."""
    negated = pattern[i + 1] == "P"
    if i + 2 >= len(pattern):
        raise ValueError("incomplete \\p escape")
    if pattern[i + 2] == "{":
        end = pattern.index("}", i + 3)
        name = pattern[i + 3 : end]
        next_index = end + 1
    else:
        name = pattern[i + 2]
        next_index = i + 3
    if name.startswith("^"):
        name, negated = name[1:], not negated
    for prefix in ("gc=", "General_Category=", "Script="):
        if name.startswith(prefix):
            if prefix == "Script=":
                raise ValueError("Unicode script properties are not supported")
            name = name[len(prefix) :]
    return name, negated, next_index


def translate(pattern: str) -> str:
    """Replaces every ``\\p{..}``/``\\P{..}`` general-category class with explicit code point ranges from the pinned
    tables, so that the ``regex`` package's own Unicode version no longer matters."""
    if "\\p" not in pattern and "\\P" not in pattern:
        return pattern
    out: list[str] = []
    i = 0
    in_class = False
    while i < len(pattern):
        c = pattern[i]
        if c == "\\" and i + 1 < len(pattern):
            if pattern[i + 1] in "pP":
                name, negated, i = _property(pattern, i)
                ranges = _category_ranges(name)
                if negated:
                    ranges = _complement(ranges)
                body = _class_body(ranges)
                out.append(body if in_class else f"[{body}]")
                continue
            out.append(pattern[i : i + 2])
            i += 2
            continue
        if c == "[" and not in_class:
            in_class = True
            out.append(c)
            i += 1
            if i < len(pattern) and pattern[i] == "^":
                out.append("^")
                i += 1
            if i < len(pattern) and pattern[i] == "]":  # a literal ] first in the class
                out.append("\\]")
                i += 1
            continue
        if c == "[" and in_class:
            raise ValueError("nested character classes are not supported")
        if c == "]" and in_class:
            in_class = False
        out.append(c)
        i += 1
    return "".join(out)


@lru_cache(maxsize=64)
def compile(pattern: str) -> regex.Pattern[str]:
    """``regex.compile`` of :func:`translate` (the pattern's Unicode classes from the pinned tables)."""
    return regex.compile(translate(pattern))
