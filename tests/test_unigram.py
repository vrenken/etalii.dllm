"""Unigram tokenizers (issue #347): SentencePiece's precompiled normaliser on every code point and on grapheme
clusters, the Viterbi segmentation with its ties, unknown pieces and byte fallback, XLM-RoBERTa's whole pipeline and
pair encoding, all against the ``tokenizers`` library."""

from __future__ import annotations

import base64
import json
import random
import struct
from functools import cache
from pathlib import Path

import pytest

from etalii_dllm import unicode, unigram
from etalii_dllm.bpe import BpeTokenizer, TokenizerError

tokenizers = pytest.importorskip("tokenizers")
spm = pytest.importorskip("sentencepiece")

ROOT = Path(__file__).resolve().parent.parent
TEXTS = [
    "Hello, world! How are you?",
    "The quick brown fox jumps over the lazy dog.",
    "naïve café façade ﬁ ½ Ⅻ \uff46\uff55\uff4c\uff4c",
    "  two  spaces   and\ttabs\nand lines  ",
    "日本語のテキストです。",
    "Привет, мир! Ελληνικά.",
    "मनुष्य क्षत्रिय हिन्दी",
    "é \r\n x‍y 🇳🇱🇧🇪🇩 👩‍👩‍👧 ✋🏽",
    "``quoted'' and \"plain\"",
    "<mask> hello<s>there</s>",
    "",
    "     ",
    "\u0000\u0007 control ­ soft hyphen",
]


def _varint(data: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[i]
        i += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, i


def _fields(data: bytes) -> list[tuple[int, bytes | int]]:
    """The fields of a protobuf message (enough to read a SentencePiece model without protobuf installed)."""
    out: list[tuple[int, bytes | int]] = []
    i = 0
    while i < len(data):
        key, i = _varint(data, i)
        wire = key & 7
        if wire == 0:
            value, i = _varint(data, i)
            out.append((key >> 3, value))
        elif wire == 2:
            size, i = _varint(data, i)
            out.append((key >> 3, data[i : i + size]))
            i += size
        else:
            width = 4 if wire == 5 else 8
            out.append((key >> 3, data[i : i + width]))
            i += width
    return out


@cache
def sentencepiece_model(rules: tuple[int, ...] = ()) -> tuple[list[tuple[str, float]], bytes]:
    """The pieces with scores and the precompiled charsmap of a small Unigram model trained on the README, with the
    default ``nmt_nfkc`` normaliser or, with ``rules``, one that maps each of those code points to itself and ``|``."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        options = {
            "input": str(ROOT / "README.md"),
            "model_prefix": f"{directory}/m",
            "vocab_size": 400,
            "model_type": "unigram",
            "character_coverage": 0.9995,
            "minloglevel": 2,
        }
        if rules:
            table = Path(directory) / "rules.tsv"
            table.write_text("".join(f"{cp:X}\t{cp:X} 7C\n" for cp in rules), encoding="utf-8")
            options["normalization_rule_tsv"] = str(table)
        spm.SentencePieceTrainer.train(**options)
        raw = Path(f"{directory}/m.model").read_bytes()
    pieces: list[tuple[str, float]] = []
    charsmap = b""
    for number, value in _fields(raw):
        if number == 1:
            piece = dict(_fields(value))  # type: ignore[arg-type]
            score = struct.unpack("<f", piece[2])[0] if 2 in piece else 0.0  # type: ignore[arg-type]
            pieces.append((piece[1].decode("utf-8"), score))  # type: ignore[union-attr]
        elif number == 3:
            charsmap = dict(_fields(value)).get(2, b"")  # type: ignore[assignment,arg-type]
    return pieces, charsmap


def xlmr_tokenizer(byte_fallback: bool = False, legacy: bool = False):
    """A tokenizer laid out as transformers converts XLM-RoBERTa's (``legacy``: the older Metaspace and normaliser
    of files like paraphrase-multilingual-MiniLM-L12-v2's)."""
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

    pieces, charsmap = sentencepiece_model()
    vocab = [("<s>", 0.0), ("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), *pieces[3:], ("<mask>", 0.0)]
    if byte_fallback:
        vocab += [(f"<0x{b:02X}>", -20.0) for b in range(256)]
    tokenizer = Tokenizer(models.Unigram(vocab, 3, byte_fallback))
    if legacy:
        tokenizer.normalizer = normalizers.Precompiled(charsmap)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")]
        )
    else:
        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Replace("``", '"'),
                normalizers.Replace("''", '"'),
                normalizers.Precompiled(charsmap),
                normalizers.Replace(Regex(" {2,}"), " "),
            ]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    tokenizer.add_special_tokens(
        [
            AddedToken("<s>", special=True),
            AddedToken("<pad>", special=True),
            AddedToken("</s>", special=True),
            AddedToken("<unk>", special=True),
            AddedToken("<mask>", special=True, lstrip=True),
        ]
    )
    return tokenizer, BpeTokenizer(json.loads(tokenizer.to_str()))


def _grapheme_pool() -> list[int]:
    """Code points from both ends of every grapheme category's ranges, linkers and InCB extenders."""
    categories, extend, linkers = unicode._grapheme_tables()
    by_category: dict[str, list[int]] = {}
    for start, end, value in zip(categories.starts, categories.ends, categories.values, strict=True):
        by_category.setdefault(value, []).extend([start, end])
    by_category["Any"] = [0x20, 0x30, 0x41, 0x5D0, 0x915, 0x928, 0xAC00]
    by_category["Linker"] = sorted(linkers)
    by_category["InCB extend"] = [extend.starts[0], extend.starts[5], extend.starts[100], extend.ends[-1]]
    return sorted({cp for values in by_category.values() for cp in values[:8] + values[-6:]} - {0})


# Grapheme clusters


def test_grapheme_clusters():
    assert unicode.graphemes("") == []
    assert unicode.graphemes("ab") == ["a", "b"]
    assert unicode.graphemes("éx") == ["é", "x"]
    assert unicode.graphemes("\r\n\n") == ["\r\n", "\n"]
    assert unicode.graphemes("🇳🇱🇧🇪🇩") == ["🇳🇱", "🇧🇪", "🇩"]
    assert unicode.graphemes("👩‍👩‍👧!") == ["👩‍👩‍👧", "!"]
    assert unicode.graphemes("a‍👧") == ["a‍", "👧"]
    assert unicode.graphemes("क्ष") == ["क्ष"]  # GB9c: consonant, virama (linker), consonant
    assert unicode.graphemes("क्‍ष") == ["क्‍ष"]
    assert unicode.graphemes("कष") == ["क", "ष"]
    assert unicode.graphemes("؀x") == ["؀x"]  # a prepended concatenation mark
    assert unicode.graphemes("각ᄀ") == ["각", "ᄀ"] and unicode.graphemes("각") == ["각"]
    assert unicode.grapheme_category("́") == "Extend" and unicode.grapheme_category("a") == "Any"


def test_grapheme_clusters_split_like_tokenizers():
    """A charsmap that maps each pool character to itself and ``|`` shows every cluster shorter than six bytes: the
    reference replaces such a cluster by its first character's mapping, dropping the rest."""
    from tokenizers import normalizers

    pool = _grapheme_pool()
    _, charsmap = sentencepiece_model(tuple(pool))
    reference, ours = normalizers.Precompiled(charsmap), unigram.Precompiled(charsmap)
    rng = random.Random(1)
    joined = 0
    for _ in range(20000):
        text = "".join(chr(rng.choice(pool)) for _ in range(rng.randint(2, 7)))
        expected = reference.normalize_str(text)
        joined += expected.count("|") < len(text)
        assert ours(text) == expected, [hex(ord(c)) for c in text]
    assert joined > 1000  # the strings do form clusters


# The precompiled normaliser


def test_precompiled_matches_tokenizers_on_every_code_point():
    from tokenizers import normalizers

    _, charsmap = sentencepiece_model()
    reference, ours = normalizers.Precompiled(charsmap), unigram.Precompiled(charsmap)
    code_points = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    for i in range(0, len(code_points), 8192):
        text = "\n".join(chr(cp) for cp in code_points[i : i + 8192])
        assert ours(text) == reference.normalize_str(text), hex(code_points[i])
    for text in TEXTS:
        assert ours(text) == reference.normalize_str(text)


def test_precompiled_refuses_broken_charsmaps():
    with pytest.raises(ValueError, match="too short"):
        unigram.Precompiled(b"\x01")
    with pytest.raises(ValueError, match="truncated"):
        unigram.Precompiled(struct.pack("<I", 64) + b"\0" * 8)
    with pytest.raises(ValueError, match="UTF-8"):
        unigram.Precompiled(struct.pack("<II", 4, 0) + b"\xff\0")
    spec = {"type": "Precompiled", "precompiled_charsmap": base64.b64encode(b"\x01").decode()}
    with pytest.raises(TokenizerError, match="Precompiled normalizer"):
        BpeTokenizer({"model": {"type": "Unigram", "vocab": [["a", 0.0]]}, "normalizer": spec})
    # A trie without the looked-up bytes, and a key with a NUL byte, map nothing.
    empty = unigram.Precompiled(struct.pack("<II", 4, 0))
    assert empty("abc\0") == "abc\0" and empty.transform("\0") is None


# The Unigram model


@pytest.mark.parametrize("variant", ["current", "legacy", "byte_fallback"])
def test_xlm_roberta_tokenizer_matches_tokenizers(variant):
    reference, ours = xlmr_tokenizer(byte_fallback=variant == "byte_fallback", legacy=variant == "legacy")
    lines = [line for line in (ROOT / "docs" / "getting-started.md").read_text(encoding="utf-8").splitlines() if line]
    for text in [*TEXTS, *lines[:200]]:
        expected = reference.encode(text)
        assert ours.encode(text, add_special_tokens=True) == expected.ids, text
        assert ours.encode(text) == reference.encode(text, add_special_tokens=False).ids
        assert ours.decode(expected.ids) == reference.decode(expected.ids)
        assert ours.decode(expected.ids, skip_special_tokens=False) == reference.decode(
            expected.ids, skip_special_tokens=False
        )
    assert ours.vocabulary_size == reference.get_vocab_size()
    assert ours.id_to_token(5) == reference.id_to_token(5)


def test_xlm_roberta_pairs_match_tokenizers():
    reference, ours = xlmr_tokenizer()
    texts = [TEXTS[0], TEXTS[2], TEXTS[5] * 3, "<mask> masked", ""]
    for first in texts:
        for second in texts:
            reference.no_truncation()
            expected = reference.encode(first, second)
            assert ours.encode_pair(first, second) == (expected.ids, expected.type_ids)
            for limit in (6, 7, 9, 12, 20):
                reference.enable_truncation(limit, strategy="longest_first")
                expected = reference.encode(first, second)
                assert ours.encode_pair(first, second, max_tokens=limit) == (expected.ids, expected.type_ids)


def test_viterbi_ties_unknown_pieces_and_byte_fallback():
    """Random vocabularies with equal scores, repeated pieces, unknown characters and byte fallback."""
    from tokenizers import Tokenizer, decoders, models

    rng = random.Random(3)
    letters = "abcdé"
    for trial in range(150):
        vocab = [("<unk>", 0.0)] if trial % 3 else []
        for _ in range(rng.randint(3, 25)):
            piece = "".join(rng.choice(letters) for _ in range(rng.randint(1, 4)))
            vocab.append((piece, rng.choice([-1.0, -2.0, -0.5, -1.5, -3.0, rng.uniform(-5, 0)])))
        byte_fallback = trial % 2 == 0
        if byte_fallback:
            vocab += [(f"<0x{b:02X}>", -4.0) for b in b"\xc3\xa9xz"]
        reference = Tokenizer(models.Unigram(vocab, 0 if trial % 3 else None, byte_fallback))
        reference.decoder = decoders.Metaspace()
        ours = BpeTokenizer(json.loads(reference.to_str()))
        for _ in range(40):
            text = "".join(rng.choice(letters + "xz") for _ in range(rng.randint(1, 12)))
            try:
                expected: list[int] | str = reference.encode(text).ids
            except Exception:  # the reference raises a plain Exception without an unk_id
                expected = "error"
            try:
                actual: list[int] | str = ours.encode(text)
            except TokenizerError:
                actual = "error"
            assert actual == expected, (vocab, text)


def test_unigram_model_directly():
    model = unigram.Unigram([("<unk>", 0.0), ("ab", -1.0), ("a", -1.0), ("b", -1.0), ("ab", -0.5)], 0, False)
    assert model.ids["ab"] == 4  # a repeated piece answers to its last id
    assert model.segment("") == [] and model.segment("abab") == ["ab", "ab"]
    assert model.segment("xyab") == ["xy", "ab"] and model.tokenize("xyab") == [0, 4]
    with pytest.raises(ValueError, match="outside the vocabulary"):
        unigram.Unigram([("a", 0.0)], 1, False)
    lonely = unigram.Unigram([("a", -1.0)], None, True)
    with pytest.raises(ValueError, match="no unk_id"):
        lonely.segment("ax")
    assert lonely.tokenize("aaa") == [0, 0, 0]
    tokenizer = BpeTokenizer(
        {
            "model": {"type": "Unigram", "vocab": [["a", -1.0], ["a", -2.0]], "unk_id": None},
            "decoder": {"type": "Metaspace"},
        }
    )
    assert tokenizer.id_to_token(0) == "a" and tokenizer.token_to_id("a") == 1
    with pytest.raises(TokenizerError, match="no unk_id"):
        tokenizer.encode("ab")
    with pytest.raises(TokenizerError, match="outside the vocabulary"):
        BpeTokenizer({"model": {"type": "Unigram", "vocab": [["a", 0.0]], "unk_id": 3}})
    with pytest.raises(TokenizerError, match="BPE, WordPiece and Unigram"):
        BpeTokenizer({"model": {"type": "WordLevel", "vocab": {}}})


def test_whitespace_split_and_roberta_processing():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

    reference = Tokenizer(models.Unigram([("<s>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("ab", -1.0)], 2, False))
    reference.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    reference.post_processor = processors.RobertaProcessing(("</s>", 1), ("<s>", 0))
    reference.decoder = decoders.Metaspace()
    ours = BpeTokenizer(json.loads(reference.to_str()))
    for first, second in [("ab ab\u2003ab\u0085", "x ab"), (" \t ", "ab")]:
        expected = reference.encode(first)
        assert ours.encode(first, add_special_tokens=True) == expected.ids
        pair = reference.encode(first, second)
        assert ours.encode_pair(first, second) == (pair.ids, pair.type_ids)
