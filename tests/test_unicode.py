"""Tokenizer text handling that does not depend on the installed Python or ``regex`` version (issue #98)."""

from __future__ import annotations

import unicodedata

import pytest
import regex

from etalii_dllm import bpe, unicode

FORMS = ("NFC", "NFD", "NFKC", "NFKD")
SAME_VERSION = unicodedata.unidata_version == unicode.unicode_version()
# Assigned in Unicode 15.0 (CJK Extension H, a Lo letter) and 15.1 (CJK Extension I): Cn on Python 3.11/3.12.
NEW_LETTERS = "\U00031350\U0002ebf0"


def _code_points():
    return (cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF)


def _normalization_relevant(c: str) -> bool:
    """Characters normalisation can change or move (for all others every form is the identity, in both)."""
    return (
        bool(unicodedata.decomposition(c) or unicodedata.combining(c))
        or "\u1100" <= c <= "\u11ff"
        or ("\uac00" <= c <= "\ud7a3")
    )


def test_tables_are_unicode_15_1():
    assert unicode.unicode_version() == "15.1.0"


@pytest.mark.skipif(not SAME_VERSION, reason="exhaustive comparison needs Python's own Unicode 15.1 (3.13)")
def test_every_code_point_matches_python_with_the_same_unicode_version():
    for cp in _code_points():
        c = chr(cp)
        assert unicode.category(c) == unicodedata.category(c), hex(cp)
        assert unicode.combining(c) == unicodedata.combining(c), hex(cp)
        assert unicode.lower(c) == c.lower(), hex(cp)
        if not _normalization_relevant(c) and unicode.normalize("NFKD", c) == c:
            continue
        for form in FORMS:
            assert unicode.normalize(form, c) == unicodedata.normalize(form, c), (form, hex(cp))


def test_characters_python_knows_normalize_the_same():
    """Normalisation of assigned characters is stable across Unicode versions, so every Python must agree on the
    characters its own tables know."""
    for cp in _code_points():
        c = chr(cp)
        if unicodedata.category(c) == "Cn" or not _normalization_relevant(c):
            continue
        for form in FORMS:
            assert unicode.normalize(form, c) == unicodedata.normalize(form, c), (form, hex(cp))


@pytest.mark.parametrize(
    "text",
    [
        "e\u0301 vs \xe9",
        "\u1100\u1161\u11a8 \uac00\u11a8",  # Hangul jamo composing into syllables
        "a\u0328\u0301\u0323 q\u0307\u0323",  # reordering of combining marks
        "\u212b \u2126 \ufb01 \u2460 \uff21",  # singletons and compatibility characters
        "\u0915\u093c \u0958",  # a composition exclusion
        "\u1f48\u03b4\u03c5\u03c3\u03c3\u03b5\u03cd\u03c2",
    ],
)
def test_sequences_match_python(text):
    for form in FORMS:
        assert unicode.normalize(form, text) == unicodedata.normalize(form, text)


@pytest.mark.parametrize(
    "text",
    [
        "\u039f\u0394\u03a5\u03a3\u03a3\u0395\u03a5\u03a3",
        "\u03a3\u0391 \u03a3 \u0391\u03a3. \u0391\u03a3'\u03a3",
        "\u0130STANBUL",
        "\u01c5 \u1e9e \u03a9",
        "Ab\u0345C",
    ],
)
def test_lower_matches_python_including_final_sigma(text):
    assert unicode.lower(text) == text.lower()


def test_new_letters_are_letters_on_every_python():
    words = unicode.compile(bpe.GPT2_PATTERN)
    assert [m.group() for m in words.finditer(f"a{NEW_LETTERS}b, 1")] == [f"a{NEW_LETTERS}b", ",", " 1"]
    assert {unicode.category(c) for c in NEW_LETTERS} == {"Lo"}


def test_translate():
    assert unicode.translate(r"\d\\p{L}") == r"\d\\p{L}"  # an escaped backslash is not a property
    letters = unicode.compile(r"\p{L}+")
    not_letters = unicode.compile(r"\P{L}+")
    in_class = unicode.compile(r"[^\s\p{L}\p{N}]+")
    assert letters.fullmatch("Stra\u00dfe")
    assert not_letters.fullmatch("12 ,.")
    assert in_class.fullmatch("!?\u2014") and not in_class.search("a1 ")
    assert unicode.compile(r"\p{Nd}+").fullmatch("0123\u0664\u0665")
    assert unicode.compile(r"\pN").fullmatch("5")
    with pytest.raises(ValueError):
        unicode.translate(r"\p{Script=Latin}")
    with pytest.raises(ValueError):
        unicode.translate(r"\p{NoSuchCategory}")


# What the tokenizer still takes from Python and ``regex``: pinned here, so a version that changes it fails loudly.

WHITE_SPACE = [
    *range(0x09, 0x0E),
    0x20,
    0x85,
    0xA0,
    0x1680,
    *range(0x2000, 0x200B),
    0x2028,
    0x2029,
    0x202F,
    0x205F,
    0x3000,
]


ALL_CHARACTERS = "".join(map(chr, _code_points()))


def test_regex_whitespace_is_pinned():
    assert [ord(c) for c in regex.findall(r"\s", ALL_CHARACTERS)] == WHITE_SPACE


def test_str_isspace_is_pinned():
    expected = sorted([*WHITE_SPACE, 0x1C, 0x1D, 0x1E, 0x1F])
    assert [cp for cp in _code_points() if chr(cp).isspace()] == expected


def test_regex_case_folding_of_contractions_is_pinned():
    matched = {ch: [ord(c) for c in regex.findall(f"(?i:{ch})", ALL_CHARACTERS)] for ch in "strevmld"}
    expected = {ch: [ord(ch.upper()), ord(ch)] for ch in "strevmld"}
    expected["s"].append(0x17F)  # LATIN SMALL LETTER LONG S
    assert matched == expected
