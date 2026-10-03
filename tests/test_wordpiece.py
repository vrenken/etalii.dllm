"""WordPiece tokenizers (BERT and the encoders built on it) against the ``tokenizers`` library (issue #337)."""

from __future__ import annotations

import json

import pytest

from etalii_dllm import unicode, wordpiece
from etalii_dllm.bpe import BpeTokenizer, TokenizerError

tokenizers = pytest.importorskip("tokenizers")

TEXTS = [
    "Hello, world! How are you?",
    "The quick brown fox jumps over the lazy dog.",
    "Ünïcödé Àccents and naïve café façade",
    "don't stop: it's 3.14 or 2,718 (maybe)",
    "中文字符和日本語のテキスト",
    "tab\tnew\nline\r\xa0nbsp\u2003em\u3000ideographic",
    "zero\u200bwidth\ufeffbom\x00nul\x07bell",
    "emoji 🥰 and symbols ∑ ≈ € §",
    "ΣΊΣΥΦΟΣ ὈΔΥΣΣΕΎΣ",
    "unknownwordthatislongerthanmostwordsinthisvocabulary " * 2,
    "a" * 120 + " short",
    "[CLS] added [SEP] tokens [MASK]",
    "",
]


def vocabulary() -> list[str]:
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    words = ["hello", "world", "how", "are", "you", "the", "quick", "brown", "fox", "over", "lazy", "dog", "un"]
    characters: list[str] = []
    for text in TEXTS:
        normalized = wordpiece.bert_normalizer(True, True, True, True)(text)
        for c in normalized:
            if c not in characters and not c.isspace():
                characters.append(c)
    pieces = ["##" + c for c in characters if c.isascii()] + ["##ing", "##s", "##ed", "##own"]
    seen: list[str] = []
    for token in [*special, *words, *characters, *pieces]:
        if token not in seen:
            seen.append(token)
    return seen


def bert_tokenizer(lowercase: bool = True, strip_accents: bool | None = None, chinese: bool = True):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

    vocab = {token: i for i, token in enumerate(vocabulary())}
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=100))
    tokenizer.normalizer = normalizers.BertNormalizer(
        clean_text=True, handle_chinese_chars=chinese, strip_accents=strip_accents, lowercase=lowercase
    )
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##", cleanup=True)
    tokenizer.add_special_tokens(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    return tokenizer, BpeTokenizer(json.loads(tokenizer.to_str()))


@pytest.fixture(scope="module", params=[(True, None, True), (False, None, True), (True, False, False)])
def pair(request):
    return bert_tokenizer(*request.param)


@pytest.mark.parametrize("text", TEXTS)
def test_encode_matches_reference(pair, text):
    reference, ours = pair
    assert ours.encode(text) == reference.encode(text, add_special_tokens=False).ids
    assert ours.encode(text, add_special_tokens=True) == reference.encode(text).ids


@pytest.mark.parametrize("text", TEXTS)
def test_decode_matches_reference(pair, text):
    reference, ours = pair
    ids = reference.encode(text).ids
    assert ours.decode(ids) == reference.decode(ids)
    assert ours.decode(ids, skip_special_tokens=False) == reference.decode(ids, skip_special_tokens=False)


def test_streamed_bytes_join_to_the_decoded_text():
    _, ours = bert_tokenizer()
    ids = ours.encode("Hello, world! How are you? unknown")
    streamed = b"".join(ours.decode_bytes([token]) for token in ids).decode()
    assert streamed.removeprefix(" ") == ours.decode(ids)


def test_every_code_point_is_normalised_and_split_like_tokenizers():
    """The character classes (control, non-spacing mark, punctuation, whitespace, CJK) on every code point; only
    characters unassigned in the pinned Unicode version may lower-case differently."""
    from tokenizers import normalizers, pre_tokenizers

    full = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=False)
    lower = normalizers.BertNormalizer(
        clean_text=False, handle_chinese_chars=False, strip_accents=False, lowercase=True
    )
    pre = pre_tokenizers.BertPreTokenizer()
    ours_full = wordpiece.bert_normalizer(True, True, True, False)
    ours_lower = wordpiece.bert_normalizer(False, False, False, True)
    mismatches = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        if full.normalize_str(c) != ours_full(c):
            mismatches.append(("normalize", hex(cp)))
        if unicode.category(c) != "Cn" and lower.normalize_str(c) != ours_lower(c):
            mismatches.append(("lowercase", hex(cp)))
        text = "a" + c + "b"
        if [piece for piece, _ in pre.pre_tokenize_str(text)] != wordpiece.pre_tokenize(text):
            mismatches.append(("split", hex(cp)))
    assert mismatches == []


def test_longest_match_and_unknown_words():
    piece = wordpiece.WordPiece({"[UNK]": 0, "un": 1, "##want": 2, "##ed": 3, "runn": 4, "##ing": 5}, "[UNK]")
    assert piece.tokenize("unwanted") == [1, 2, 3]
    assert piece.tokenize("running") == [4, 5]
    assert piece.tokenize("unwantedx") == [0]  # one piece missing: the whole word is unknown
    assert wordpiece.WordPiece({"[UNK]": 0, "a": 1}, "[UNK]", max_characters=3).tokenize("aaaa") == [0]
    with pytest.raises(ValueError, match="unknown token"):
        wordpiece.WordPiece({"a": 1}, "[UNK]")


def test_bert_processing_post_processor():
    reference, _ = bert_tokenizer()
    spec = json.loads(reference.to_str())
    spec["post_processor"] = {"type": "BertProcessing", "sep": ["[SEP]", 3], "cls": ["[CLS]", 2]}
    ours = BpeTokenizer(spec)
    assert ours.encode("hello", add_special_tokens=True) == [2, ours.token_to_id("hello"), 3]


def test_a_vocabulary_without_the_unknown_token_is_refused():
    reference, _ = bert_tokenizer()
    spec = json.loads(reference.to_str())
    spec["model"]["unk_token"] = "[NOPE]"
    with pytest.raises(TokenizerError, match="unknown token"):
        BpeTokenizer(spec)
