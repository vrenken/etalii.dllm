"""WordPiece tokenizers (BERT and the encoder models built on it), as the ``tokenizers`` library runs them.

Three pieces plug into :class:`etalii_dllm.bpe.BpeTokenizer`: the ``BertNormalizer`` (control characters removed,
whitespace made spaces, CJK ideographs set apart, accents stripped, characters lower-cased one by one), the
``BertPreTokenizer`` (split on whitespace, every punctuation character on its own) and the WordPiece model itself
(greedy longest match from the start of each word, continuation pieces prefixed with ``##``, the whole word unknown
when a piece cannot be found). Decoding follows the ``WordPiece`` decoder.

``tokenizers`` classifies characters with the Rust ``unicode_categories`` crate, whose tables are older than the
pinned Unicode version of :mod:`etalii_dllm.unicode`. The ranges below are exactly the characters on which the two
disagree (checked over every code point in ``tests/test_wordpiece.py``), so the classes are fixed here and do not
depend on any installed library. Lower-casing uses the pinned tables: characters assigned after that version (and
lower-cased by newer Rust releases) are left as they are.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Mapping
from functools import cache

from etalii_dllm import unicode

# Rust's char::is_whitespace (the Unicode White_Space property), which the pre-tokenizer splits on.
WHITESPACE = frozenset(
    [0x9, 0xA, 0xB, 0xC, 0xD, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
)

# The CJK ideograph blocks the normaliser surrounds with spaces (tokenizers' is_chinese_char).
_CHINESE = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B920, 0x2CEAF),
    (0x2F800, 0x2FA1F),
)

# Characters "other" (C*) in one table but not the other; unassigned characters (Cn) are never removed.
_CONTROL_DIFFERENCES: tuple[tuple[int, int], ...] = (
    (0x890, 0x891),
    (0x8E2, 0x8E2),
    (0x110CD, 0x110CD),
    (0x13430, 0x1343F),
)
# Characters a non-spacing mark (Mn) in one table but not the other (only these are stripped as accents).
_MARK_DIFFERENCES: tuple[tuple[int, int], ...] = (
    (0x7FD, 0x7FD),
    (0x898, 0x89F),
    (0x8CA, 0x8E1),
    (0x9FE, 0x9FE),
    (0xAFA, 0xAFF),
    (0xB55, 0xB55),
    (0xC04, 0xC04),
    (0xC3C, 0xC3C),
    (0xD00, 0xD00),
    (0xD3B, 0xD3C),
    (0xD81, 0xD81),
    (0xEBA, 0xEBA),
    (0xECE, 0xECE),
    (0x1734, 0x1734),
    (0x180F, 0x180F),
    (0x1885, 0x1886),
    (0x1ABF, 0x1ACE),
    (0x1DF6, 0x1DFB),
    (0xA82C, 0xA82C),
    (0xA8C5, 0xA8C5),
    (0xA8FF, 0xA8FF),
    (0xA9BD, 0xA9BD),
    (0x10D24, 0x10D27),
    (0x10EAB, 0x10EAC),
    (0x10EFD, 0x10EFF),
    (0x10F46, 0x10F50),
    (0x10F82, 0x10F85),
    (0x11070, 0x11070),
    (0x11073, 0x11074),
    (0x110C2, 0x110C2),
    (0x111C9, 0x111C9),
    (0x111CF, 0x111CF),
    (0x1123E, 0x1123E),
    (0x11241, 0x11241),
    (0x1133B, 0x1133B),
    (0x11438, 0x1143F),
    (0x11442, 0x11444),
    (0x11446, 0x11446),
    (0x1145E, 0x1145E),
    (0x1182F, 0x11837),
    (0x11839, 0x1183A),
    (0x1193B, 0x1193C),
    (0x1193E, 0x1193E),
    (0x11943, 0x11943),
    (0x119D4, 0x119D7),
    (0x119DA, 0x119DB),
    (0x119E0, 0x119E0),
    (0x11A01, 0x11A0A),
    (0x11A33, 0x11A38),
    (0x11A3B, 0x11A3E),
    (0x11A47, 0x11A47),
    (0x11A51, 0x11A56),
    (0x11A59, 0x11A5B),
    (0x11A8A, 0x11A96),
    (0x11A98, 0x11A99),
    (0x11C30, 0x11C36),
    (0x11C38, 0x11C3D),
    (0x11C3F, 0x11C3F),
    (0x11C92, 0x11CA7),
    (0x11CAA, 0x11CB0),
    (0x11CB2, 0x11CB3),
    (0x11CB5, 0x11CB6),
    (0x11D31, 0x11D36),
    (0x11D3A, 0x11D3A),
    (0x11D3C, 0x11D3D),
    (0x11D3F, 0x11D45),
    (0x11D47, 0x11D47),
    (0x11D90, 0x11D91),
    (0x11D95, 0x11D95),
    (0x11D97, 0x11D97),
    (0x11EF3, 0x11EF4),
    (0x11F00, 0x11F01),
    (0x11F36, 0x11F3A),
    (0x11F40, 0x11F40),
    (0x11F42, 0x11F42),
    (0x13440, 0x13440),
    (0x13447, 0x13455),
    (0x16F4F, 0x16F4F),
    (0x16FE4, 0x16FE4),
    (0x1CF00, 0x1CF2D),
    (0x1CF30, 0x1CF46),
    (0x1E000, 0x1E006),
    (0x1E008, 0x1E018),
    (0x1E01B, 0x1E021),
    (0x1E023, 0x1E024),
    (0x1E026, 0x1E02A),
    (0x1E08F, 0x1E08F),
    (0x1E130, 0x1E136),
    (0x1E2AE, 0x1E2AE),
    (0x1E2EC, 0x1E2EF),
    (0x1E4EC, 0x1E4EF),
    (0x1E944, 0x1E94A),
)
# Characters punctuation (P*) in one table but not the other.
_PUNCTUATION_DIFFERENCES: tuple[tuple[int, int], ...] = (
    (0x61D, 0x61D),
    (0x9FD, 0x9FD),
    (0xA76, 0xA76),
    (0xC77, 0xC77),
    (0xC84, 0xC84),
    (0x166D, 0x166D),
    (0x1B7D, 0x1B7E),
    (0x2E43, 0x2E4F),
    (0x2E52, 0x2E5D),
    (0x10EAD, 0x10EAD),
    (0x10F55, 0x10F59),
    (0x10F86, 0x10F89),
    (0x111C9, 0x111C9),
    (0x1144B, 0x1144F),
    (0x1145A, 0x1145B),
    (0x1145D, 0x1145D),
    (0x11660, 0x1166C),
    (0x116B9, 0x116B9),
    (0x1183B, 0x1183B),
    (0x11944, 0x11946),
    (0x119E2, 0x119E2),
    (0x11A3F, 0x11A46),
    (0x11A9A, 0x11A9C),
    (0x11A9E, 0x11AA2),
    (0x11B00, 0x11B09),
    (0x11C41, 0x11C45),
    (0x11C70, 0x11C71),
    (0x11EF7, 0x11EF8),
    (0x11F43, 0x11F4F),
    (0x11FFF, 0x11FFF),
    (0x12FF1, 0x12FF2),
    (0x16E97, 0x16E9A),
    (0x16FE2, 0x16FE2),
    (0x1E95E, 0x1E95F),
)
# Characters whose canonical decomposition tokenizers' Unicode normalisation does not know yet (Dives Akuru
# vowel sign O, Unicode 13): accent stripping leaves them whole.
_UNDECOMPOSED = "\U00011938"


def _nfd(text: str) -> str:
    parts = text.split(_UNDECOMPOSED)
    return _UNDECOMPOSED.join(unicode.normalize("NFD", part) for part in parts)


def _in(ranges: tuple[tuple[int, int], ...], cp: int) -> bool:
    i = bisect.bisect_right(ranges, (cp, 0x110000)) - 1
    return i >= 0 and ranges[i][0] <= cp <= ranges[i][1]


def _is_control(character: str) -> bool:
    if character in "\t\n\r":
        return False
    category = unicode.category(character)
    other = category.startswith("C") and category != "Cn"
    return other != _in(_CONTROL_DIFFERENCES, ord(character))


def _is_mark(character: str) -> bool:
    return (unicode.category(character) == "Mn") != _in(_MARK_DIFFERENCES, ord(character))


def is_punctuation(character: str) -> bool:
    """tokenizers' ``is_bert_punc``: ASCII punctuation or a Unicode punctuation character."""
    cp = ord(character)
    if cp < 0x80:
        return 0x21 <= cp <= 0x7E and not character.isalnum()
    return unicode.category(character).startswith("P") != _in(_PUNCTUATION_DIFFERENCES, cp)


@cache
def bert_normalizer(clean_text: bool, chinese: bool, strip_accents: bool, lowercase: bool) -> Callable[[str], str]:
    """The ``BertNormalizer`` as a function of text (cached per option set)."""

    def normalize(text: str) -> str:
        if clean_text:
            text = "".join(
                " " if ord(c) in WHITESPACE else c for c in text if not (c == "\0" or c == "\ufffd" or _is_control(c))
            )
        if chinese:
            text = "".join(f" {c} " if _in(_CHINESE, ord(c)) else c for c in text)
        if strip_accents:
            text = "".join(c for c in _nfd(text) if not _is_mark(c))
        if lowercase:
            text = unicode.lower_characters(text)
        return text

    return normalize


def normalizer(spec: Mapping[str, object]) -> Callable[[str], str]:
    """The ``BertNormalizer`` of a ``tokenizer.json`` normalizer entry (``strip_accents`` null follows
    ``lowercase``)."""
    lowercase = bool(spec.get("lowercase", True))
    strip = spec.get("strip_accents")
    return bert_normalizer(
        bool(spec.get("clean_text", True)),
        bool(spec.get("handle_chinese_chars", True)),
        lowercase if strip is None else bool(strip),
        lowercase,
    )


def pre_tokenize(text: str) -> list[str]:
    """The ``BertPreTokenizer``: whitespace removed, each punctuation character a piece of its own."""
    pieces: list[str] = []
    word: list[str] = []
    for c in text:
        if ord(c) in WHITESPACE or is_punctuation(c):
            if word:
                pieces.append("".join(word))
                word = []
            if ord(c) not in WHITESPACE:
                pieces.append(c)
        else:
            word.append(c)
    if word:
        pieces.append("".join(word))
    return pieces


class WordPiece:
    """The WordPiece model: a vocabulary, the unknown token, the continuation prefix and the longest word."""

    def __init__(self, vocab: Mapping[str, int], unknown: str, prefix: str = "##", max_characters: int = 100) -> None:
        if unknown not in vocab:
            raise ValueError(f"the unknown token {unknown!r} is not in the vocabulary")
        self.vocab = dict(vocab)
        self.unknown = vocab[unknown]
        self.prefix = prefix
        self.max_characters = max_characters

    def tokenize(self, word: str) -> list[int]:
        """Greedy longest match from the start; a word with a piece not in the vocabulary (or longer than
        ``max_characters``) is the unknown token."""
        if len(word) > self.max_characters:
            return [self.unknown]
        ids: list[int] = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                piece = word[start:end] if start == 0 else self.prefix + word[start:end]
                found = self.vocab.get(piece)
                if found is not None:
                    break
                end -= 1
            if found is None:
                return [self.unknown]
            ids.append(found)
            start = end
        return ids


_CLEANUP = (
    (" .", "."),
    (" ?", "?"),
    (" !", "!"),
    (" ,", ","),
    (" ' ", "'"),
    (" n't", "n't"),
    (" 'm", "'m"),
    (" do not", " don't"),
    (" 's", "'s"),
    (" 've", "'ve"),
    (" 're", "'re"),
)


def decode_piece(token: str, first: bool, prefix: str, cleanup: bool) -> str:
    """One token as the ``WordPiece`` decoder writes it: continuation pieces lose the prefix, others (but the first)
    get a space, then the clean-up replacements of that token."""
    if not first:
        token = token[len(prefix) :] if token.startswith(prefix) else " " + token
    if cleanup:
        for dirty, clean in _CLEANUP:
            token = token.replace(dirty, clean)
    return token
