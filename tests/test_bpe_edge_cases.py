"""BPE tokenizer edge cases: less common normalisers, split behaviours, model options, post-processors, added-token
stripping, fallbacks for unknown characters and the ``model.dllm``/GGUF entry points. Where the ``tokenizers``
reference supports a behaviour, the output must match it exactly."""

from __future__ import annotations

import json

import pytest
from test_bpe import SAMPLES, SPECIAL, train

from etalii_dllm.bpe import BpeTokenizer, TokenizerError, from_model_header, spec_from_gguf, special_token_text

tokenizers = pytest.importorskip("tokenizers")

BYTE_LEVEL_NO_REGEX = {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False}
SPLIT_SAMPLES = [
    *SAMPLES,
    "a-b--c---d",
    "--leading and trailing--",
    "x - y -- z",
    "no delimiters here",
]


def ours_from(reference, **options) -> BpeTokenizer:
    return BpeTokenizer(json.loads(reference.to_str()), **options)


def assert_same_encoding(reference, ours: BpeTokenizer, texts) -> None:
    for text in texts:
        assert ours.encode(text) == reference.encode(text, add_special_tokens=False).ids, text
        assert ours.encode(text, add_special_tokens=True) == reference.encode(text).ids, text


def minimal_spec(**overrides) -> dict:
    """A tiny valid byte-level BPE description: the letters a and b and their merge."""
    spec = {
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]},
        "pre_tokenizer": BYTE_LEVEL_NO_REGEX,
        "decoder": {"type": "ByteLevel"},
    }
    for key, value in overrides.items():
        if key == "model":
            spec["model"] = {**spec["model"], **value}
        else:
            spec[key] = value
    return spec


def reference_from(spec: dict):
    """The reference tokenizer for a hand-written ``spec``; ``tokenizers`` wants every field spelled out."""
    model = {
        "dropout": None, "unk_token": None, "continuing_subword_prefix": None, "end_of_word_suffix": None,
        "fuse_unk": False, "byte_fallback": False, "ignore_merges": False, **spec["model"],
    }  # fmt: skip
    full = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": None,
        "post_processor": None, **spec,
        "pre_tokenizer": {"trim_offsets": True, **spec["pre_tokenizer"]},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": model,
    }  # fmt: skip
    return tokenizers.Tokenizer.from_str(json.dumps(full))


# --- normalisers -------------------------------------------------------------------------------------------------


def test_lowercase_and_sequence_normalizers_match_reference():
    from tokenizers import normalizers, pre_tokenizers

    reference = train(
        pre_tokenizers.ByteLevel(add_prefix_space=False),
        normalizer=normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()]),
    )
    ours = ours_from(reference)
    assert_same_encoding(
        reference,
        ours,
        [
            *SAMPLES,
            "MiXeD CaSe ÉLAN \N{FULLWIDTH LATIN CAPITAL LETTER A}\N{FULLWIDTH LATIN CAPITAL LETTER B} ﬁ Ⅻ İstanbul",
        ],
    )
    # The sequence runs in order: NFKC folds the full-width letters before lowercasing.
    assert ours.encode("\N{FULLWIDTH LATIN CAPITAL LETTER A}\N{FULLWIDTH LATIN CAPITAL LETTER B}") == ours.encode("ab")


def test_lowercase_normalizer_alone():
    from tokenizers import normalizers, pre_tokenizers

    reference = train(pre_tokenizers.ByteLevel(add_prefix_space=False), normalizer=normalizers.Lowercase())
    ours = ours_from(reference)
    assert_same_encoding(reference, ours, SAMPLES)
    assert ours.encode("HELLO World") == ours.encode("hello world")


def test_unsupported_normalizer_fails_at_load():
    with pytest.raises(TokenizerError, match="normalizer 'ByteLevel' is not supported"):
        BpeTokenizer(minimal_spec(normalizer={"type": "ByteLevel"}))
    # Also when nested inside a sequence.
    with pytest.raises(TokenizerError, match="normalizer 'Precompiled' is not supported"):
        BpeTokenizer(
            minimal_spec(normalizer={"type": "Sequence", "normalizers": [{"type": "NFC"}, {"type": "Precompiled"}]})
        )


# --- Split pre-tokenizer -----------------------------------------------------------------------------------------


SPLIT_BEHAVIOURS = ["isolated", "removed", "merged_with_previous", "merged_with_next", "contiguous"]


@pytest.mark.parametrize(
    ("behavior", "invert"),
    [(behavior, invert) for behavior in SPLIT_BEHAVIOURS for invert in (False, True)],
)
def test_split_behaviours_match_reference(behavior, invert):
    from tokenizers import Regex, pre_tokenizers

    split = pre_tokenizers.Split(Regex(r"[\s\-]"), behavior=behavior, invert=invert)
    reference = train(
        pre_tokenizers.Sequence([split, pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    )
    assert_same_encoding(reference, ours_from(reference), SPLIT_SAMPLES)


@pytest.mark.parametrize("behavior", ["isolated", "removed", "merged_with_previous", "contiguous"])
def test_split_on_a_literal_string_matches_reference(behavior):
    """A ``String`` pattern is matched literally: regex metacharacters in it have no special meaning."""
    from tokenizers import pre_tokenizers

    split = pre_tokenizers.Split("-.", behavior=behavior)
    reference = train(
        pre_tokenizers.Sequence([split, pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    )
    ours = ours_from(reference)
    assert json.loads(reference.to_str())["pre_tokenizer"]["pretokenizers"][0]["pattern"] == {"String": "-."}
    assert_same_encoding(reference, ours, [*SPLIT_SAMPLES, "a-.b-xc-.-.d", "1.5-.2"])


def test_split_ignores_empty_matches():
    """A pattern that can match the empty string must not produce empty pieces or loop forever."""
    from tokenizers import Regex, pre_tokenizers

    split = pre_tokenizers.Split(Regex(r"\d*"), behavior="isolated")
    reference = train(
        pre_tokenizers.Sequence([split, pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    )
    assert_same_encoding(reference, ours_from(reference), [*SAMPLES, "abc", "a1b22c333"])


def test_unsupported_split_behaviour_fails():
    split = {"type": "Split", "pattern": {"Regex": r"\s"}, "behavior": "Sideways", "invert": False}
    tokenizer = BpeTokenizer(
        minimal_spec(pre_tokenizer={"type": "Sequence", "pretokenizers": [split, BYTE_LEVEL_NO_REGEX]})
    )
    # Split behaviour is only inspected when there is text to split.
    with pytest.raises(TokenizerError, match="split behaviour 'Sideways' is not supported"):
        tokenizer.encode("a b")


# --- model options and load-time validation -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ({"continuing_subword_prefix": "##"}, "continuing_subword_prefix is not supported"),
        ({"end_of_word_suffix": "</w>"}, "end_of_word_suffix is not supported"),
        ({"dropout": 0.1}, "dropout is not deterministic"),
        ({"merges": ["a c"]}, "merge 'a' 'c' refers to an unknown token"),
        ({"vocab": {"a": 0, "b": 1}}, "merge 'a' 'b' refers to an unknown token"),  # the merged token is missing
    ],
)
def test_invalid_model_options_fail_at_load(model, message):
    with pytest.raises(TokenizerError, match=message):
        BpeTokenizer(minimal_spec(model=model))


def test_falsy_model_options_are_accepted():
    """``tokenizers`` writes ``null``/empty values for unused options; those must not be rejected."""
    tokenizer = BpeTokenizer(
        minimal_spec(model={"continuing_subword_prefix": None, "end_of_word_suffix": "", "dropout": None})
    )
    assert tokenizer.encode("ab") == [2]


def test_empty_text_without_added_tokens_encodes_to_nothing():
    """Without added tokens the whole (empty) text reaches the pre-tokenizer, which yields one empty piece."""
    tokenizer = BpeTokenizer(minimal_spec())
    assert tokenizer.encode("") == []
    assert tokenizer.encode("", add_special_tokens=True) == []


def test_merges_given_as_pairs_equal_merges_given_as_strings():
    as_pairs = BpeTokenizer(minimal_spec(model={"merges": [["a", "b"]]}))
    as_strings = BpeTokenizer(minimal_spec())
    assert as_pairs.encode("aabb") == as_strings.encode("aabb") == [0, 2, 1]


def test_unsupported_decoder_fails_at_load():
    with pytest.raises(TokenizerError, match="decoder 'WordPiece' is not supported"):
        BpeTokenizer(minimal_spec(decoder={"type": "WordPiece"}))
    with pytest.raises(TokenizerError, match="decoder None is not supported"):
        BpeTokenizer(minimal_spec(decoder=None))


def test_unsupported_post_processor_fails_at_load():
    with pytest.raises(TokenizerError, match="post-processor 'RobertaProcessing' is not supported"):
        BpeTokenizer(minimal_spec(post_processor={"type": "RobertaProcessing"}))
    nested = {"type": "Sequence", "processors": [{"type": "ByteLevel"}, {"type": "RobertaProcessing"}]}
    with pytest.raises(TokenizerError, match="post-processor 'RobertaProcessing' is not supported"):
        BpeTokenizer(minimal_spec(post_processor=nested))


def test_sequence_post_processor_matches_reference():
    from tokenizers import pre_tokenizers, processors

    reference = train(pre_tokenizers.ByteLevel(add_prefix_space=False))
    reference.post_processor = processors.Sequence(
        [
            processors.ByteLevel(trim_offsets=False),
            processors.TemplateProcessing(
                single="<|im_start|> $A <|im_end|>", special_tokens=[("<|im_start|>", 1), ("<|im_end|>", 2)]
            ),
        ]
    )
    ours = ours_from(reference)
    assert_same_encoding(reference, ours, SAMPLES)
    assert ours.encode("", add_special_tokens=True) == [1, 2]


# --- special tokens and vocabulary lookups ------------------------------------------------------------------------


def test_end_and_begin_of_sequence_resolve_from_text_or_id():
    reference = train(tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False))
    by_text = ours_from(reference, end_of_sequence="<|im_end|>", begin_of_sequence="<|im_start|>")
    assert (by_text.end_of_sequence, by_text.begin_of_sequence) == (2, 1)
    by_id = ours_from(reference, end_of_sequence=7, begin_of_sequence=0)
    assert (by_id.end_of_sequence, by_id.begin_of_sequence) == (7, 0)
    unset = ours_from(reference)
    assert (unset.end_of_sequence, unset.begin_of_sequence) == (-1, -1)
    with pytest.raises(TokenizerError, match="unknown token '<eos>'"):
        ours_from(reference, end_of_sequence="<eos>")
    with pytest.raises(TokenizerError, match="unknown token '<bos>'"):
        ours_from(reference, begin_of_sequence="<bos>")


def test_token_and_id_lookups_match_reference():
    reference = train(tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False))
    ours = ours_from(reference)
    assert ours.vocabulary_size == reference.get_vocab_size()
    for index in range(reference.get_vocab_size()):
        token = reference.id_to_token(index)
        assert ours.id_to_token(index) == token
        assert ours.token_to_id(token) == index == reference.token_to_id(token)
    assert ours.token_to_id("definitely not a token") is None is reference.token_to_id("definitely not a token")
    assert ours.id_to_token(10**6) is None is reference.id_to_token(10**6)


def test_added_tokens_outside_the_vocabulary_match_reference():
    """Added tokens may be absent from the model vocabulary (ids right after it, as ``tokenizers`` writes them);
    they must still encode, decode and resolve."""
    spec = minimal_spec(
        added_tokens=[
            {"id": 3, "content": "<eos>", "special": True, "lstrip": False, "rstrip": False, "normalized": False,
             "single_word": False},
            {"id": 4, "content": "[tool]", "special": False, "lstrip": False, "rstrip": False, "normalized": False,
             "single_word": False},
        ]
    )  # fmt: skip
    reference = reference_from(spec)
    ours = BpeTokenizer(spec, end_of_sequence="<eos>")
    assert ours.end_of_sequence == 3 == reference.token_to_id("<eos>")
    assert ours.token_to_id("[tool]") == 4 == reference.token_to_id("[tool]")
    assert ours.id_to_token(3) == "<eos>" == reference.id_to_token(3)
    assert ours.vocabulary_size == 5 == reference.get_vocab_size()
    for text in ["ab<eos>", "a[tool]b<eos>ab", "<eos><eos>", "[tool]"]:
        ids = ours.encode(text)
        assert ids == reference.encode(text, add_special_tokens=False).ids
        assert ours.decode(ids) == reference.decode(ids)
        assert ours.decode(ids, skip_special_tokens=False) == reference.decode(ids, skip_special_tokens=False)


@pytest.mark.parametrize(("lstrip", "rstrip"), [(True, False), (False, True), (True, True)])
def test_added_token_whitespace_stripping_matches_reference(lstrip, rstrip):
    from tokenizers import AddedToken, pre_tokenizers

    reference = train(pre_tokenizers.ByteLevel(add_prefix_space=False))
    reference.add_special_tokens([AddedToken("<mask>", lstrip=lstrip, rstrip=rstrip, normalized=False)])
    ours = ours_from(reference)
    texts = ["a <mask> b", "a   <mask>\n\t b", "<mask>   ", "   <mask>", "x<mask>y", "a \n<mask> <mask>  end", "<mask>"]
    for text in texts:
        assert ours.encode(text) == reference.encode(text, add_special_tokens=False).ids, text


# --- characters outside the vocabulary ----------------------------------------------------------------------------


def partial_alphabet_spec(**model) -> dict:
    """Only the byte-level characters of 'a', 'b' and 'c': anything else is unknown to the model."""
    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3}
    vocab.update({f"<0x{b:02X}>": 4 + b for b in range(256)})
    vocab["<unk>"] = 260
    return minimal_spec(model={"vocab": vocab, "merges": ["a b"], **model})


def test_byte_fallback_matches_reference():
    spec = partial_alphabet_spec(byte_fallback=True)
    reference = reference_from(spec)
    ours = BpeTokenizer(spec)
    for text in ["abc", "ab d", "é", "a🙂b", "ÿ"]:
        assert ours.encode(text) == reference.encode(text, add_special_tokens=False).ids, text
    # ' ' is byte-level 'Ġ' (U+0120), which falls back to its UTF-8 bytes C4 A0.
    assert ours.encode(" ") == [4 + 0xC4, 4 + 0xA0]


def test_unknown_token_matches_reference():
    spec = partial_alphabet_spec(unk_token="<unk>")
    reference = reference_from(spec)
    ours = BpeTokenizer(spec)
    for text in ["abc", "a d", "zz", "ab🙂"]:
        assert ours.encode(text) == reference.encode(text, add_special_tokens=False).ids, text
    assert ours.encode("a z") == [0, 260, 260]  # one unknown per character, not fused


def test_unknown_token_missing_from_the_vocabulary_is_ignored():
    """An ``unk_token`` the vocabulary lacks leaves unknown characters an error, as with no ``unk_token``."""
    ours = BpeTokenizer(partial_alphabet_spec(unk_token="<missing>"))
    with pytest.raises(TokenizerError, match="character 'Ġ' is not in the vocabulary"):
        ours.encode("a b")


def test_character_outside_the_vocabulary_without_fallback_fails():
    ours = BpeTokenizer(partial_alphabet_spec())
    assert ours.encode("cab") == [2, 3]
    with pytest.raises(TokenizerError, match="character 'z' is not in the vocabulary"):
        ours.encode("abz")


# --- decoding ------------------------------------------------------------------------------------------------------


def test_decode_skips_ids_outside_the_vocabulary():
    reference = train(tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False))
    ours = ours_from(reference)
    ids = ours.encode("hello world")
    with_unknown = [10**6, *ids, reference.get_vocab_size() + 5]
    assert ours.decode(with_unknown) == "hello world" == reference.decode(with_unknown)
    assert ours.decode_bytes([10**6]) == b""


# --- model.dllm headers and GGUF metadata -------------------------------------------------------------------------


def gguf_metadata(reference) -> dict:
    spec = json.loads(reference.to_str())
    tokens = [token for token, _ in sorted(spec["model"]["vocab"].items(), key=lambda item: item[1])]
    return {
        "format": "gguf",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "gpt-2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": [3 if token in SPECIAL else 1 for token in tokens],
        "tokenizer.ggml.merges": [m if isinstance(m, str) else " ".join(m) for m in spec["model"]["merges"]],
        "tokenizer.ggml.eos_token_id": tokens.index("<|im_end|>"),
        "tokenizer.ggml.bos_token_id": tokens.index("<|im_start|>"),
    }


def test_gguf_model_header_matches_reference():
    reference = train(tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False))
    ours = from_model_header(gguf_metadata(reference))
    assert (ours.end_of_sequence, ours.begin_of_sequence) == (2, 1)
    for text in SAMPLES:
        assert ours.encode(text) == reference.encode(text, add_special_tokens=False).ids, text


def test_gguf_without_token_types_or_merges():
    """Token types and merges are optional in GGUF: then every token is normal and nothing merges."""
    metadata = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.tokens": ["a", "b", "ab"]}
    spec = spec_from_gguf(metadata)
    assert spec["added_tokens"] == [] and spec["model"]["merges"] == []
    assert BpeTokenizer(spec).encode("ab") == [0, 1]


@pytest.mark.parametrize("model", ["bert", "t5", None])
def test_gguf_non_bpe_model_is_rejected(model):
    with pytest.raises(TokenizerError, match=f"GGUF tokenizer model {model!r} is not supported"):
        spec_from_gguf({"tokenizer.ggml.model": model, "tokenizer.ggml.tokens": ["a"]})


def test_huggingface_model_header_reads_special_tokens_from_the_config():
    reference = train(tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False))
    header = {
        "format": "huggingface",
        "tokenizer_json": json.loads(reference.to_str()),
        "tokenizer_config": {
            "eos_token": {"content": "<|im_end|>", "lstrip": False, "rstrip": False, "special": True},
            "bos_token": "<|im_start|>",
        },
    }
    ours = from_model_header(header)
    assert (ours.end_of_sequence, ours.begin_of_sequence) == (2, 1)
    without_config = from_model_header({"format": "huggingface", "tokenizer_json": header["tokenizer_json"]})
    assert (without_config.end_of_sequence, without_config.begin_of_sequence) == (-1, -1)


@pytest.mark.parametrize("header", [{"format": "sentencepiece"}, {}])
def test_unknown_model_header_format_is_rejected(header):
    with pytest.raises(TokenizerError, match=f"unknown tokenizer format {header.get('format')!r}"):
        from_model_header(header)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("<|im_end|>", "<|im_end|>"),
        ({"__type": "AddedToken", "content": "</s>", "lstrip": False}, "</s>"),
        ({"lstrip": False}, None),
        (None, None),
    ],
)
def test_special_token_text(value, expected):
    assert special_token_text(value) == expected
