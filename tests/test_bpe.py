"""BPE tokenizer tests: identical ids and text to the Hugging Face ``tokenizers`` reference."""

from __future__ import annotations

import json

import pytest

from etalii_dllm.bpe import BpeTokenizer, TokenizerError

tokenizers = pytest.importorskip("tokenizers")

CORPUS = [
    "The quick brown fox jumps over the lazy dog. It's 2026-09-28 and we've got 12345 tokens!",
    "Deterministic inference: same weights, same prompt, same output. Élan, naïve café, straße.",
    "def main():\n    return sum(x * 2 for x in range(10))  # comment\n\n\tindented\r\n",
    "Emoji 🙂🚀 and CJK 你好世界, Arabic مرحبا, Hindi नमस्ते, math ∑∫√ and combining é.",
    "<|im_start|>user\nHello there<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>",
]

SAMPLES = [
    *CORPUS,
    "",
    " ",
    "   leading and trailing   ",
    "a\n\n\nb   \n  c",
    "1234567890 3.14159 -42 1e10",
    "don't DON'T we'LL you'Re",
    "unseen words: zyxwvut qqq ǅ ﬁ Ⅻ ①",
    "<|im_start|><|im_start|>x<|im_end|>tail<|endoftext|>",
    "special inside<|im_end|>word and <|im_start partial",
    "\N{NO-BREAK SPACE}non-breaking\N{EM SPACE}em space\N{ZERO WIDTH SPACE}zero width",
    "e\N{COMBINING ACUTE ACCENT} vs \N{LATIN SMALL LETTER E WITH ACUTE}",
    "🙂" * 5,
]

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
QWEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def train(pre_tokenizer, normalizer=None, vocab_size=400, **model_options):
    from tokenizers import Tokenizer, decoders, models, trainers

    tokenizer = Tokenizer(models.BPE(**model_options))
    tokenizer.pre_tokenizer = pre_tokenizer
    if normalizer is not None:
        tokenizer.normalizer = normalizer
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL,
        initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(CORPUS * 3, trainer)
    return tokenizer


def smollm2_style():
    from tokenizers import pre_tokenizers

    return train(
        pre_tokenizers.Sequence(
            [pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.ByteLevel(add_prefix_space=False)]
        )
    )


def qwen2_style():
    from tokenizers import Regex, normalizers, pre_tokenizers

    return train(
        pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(QWEN_PATTERN), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        ),
        normalizer=normalizers.NFC(),
    )


def gpt2_style():
    from tokenizers import pre_tokenizers, processors

    tokenizer = train(pre_tokenizers.ByteLevel(add_prefix_space=True), vocab_size=300)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|endoftext|> $A", special_tokens=[("<|endoftext|>", 0)]
    )
    return tokenizer


def llama3_style():
    from tokenizers import Regex, pre_tokenizers

    return train(
        pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(QWEN_PATTERN), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        ),
        ignore_merges=True,
    )


@pytest.fixture(scope="module", params=[smollm2_style, qwen2_style, gpt2_style, llama3_style])
def pair(request):
    reference = request.param()
    ours = BpeTokenizer(json.loads(reference.to_str()), end_of_sequence="<|im_end|>")
    return reference, ours


@pytest.mark.parametrize("text", SAMPLES)
def test_encode_matches_reference(pair, text):
    reference, ours = pair
    assert ours.encode(text) == reference.encode(text, add_special_tokens=False).ids
    assert ours.encode(text, add_special_tokens=True) == reference.encode(text).ids


@pytest.mark.parametrize("text", SAMPLES)
def test_decode_round_trips(pair, text):
    reference, ours = pair
    ids = ours.encode(text)
    assert ours.decode(ids, skip_special_tokens=False) == reference.decode(ids, skip_special_tokens=False)
    assert ours.decode(ids) == reference.decode(ids)


def test_merges_resolve_by_rank_then_position():
    spec = {
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "aa": 2, "ab": 3, "aab": 4},
            "merges": ["a a", "a b", "aa b"],
        },
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
        "decoder": {"type": "ByteLevel"},
    }
    tokenizer = BpeTokenizer(spec)
    assert tokenizer.encode("aaab") == [2, 3]  # (a a) first at position 0, then (a b)
    assert tokenizer.encode("aab") == [4]


def test_unsupported_components_fail_loudly():
    base = {"model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}, "decoder": {"type": "ByteLevel"}}
    with pytest.raises(TokenizerError, match="WordPiece"):
        BpeTokenizer({**base, "model": {"type": "WordPiece", "vocab": {}}})
    with pytest.raises(TokenizerError, match="byte-level"):
        BpeTokenizer(base)
    with pytest.raises(TokenizerError, match="Metaspace"):
        BpeTokenizer({**base, "pre_tokenizer": {"type": "Metaspace"}})


@pytest.mark.parametrize(("style", "pre"), [(smollm2_style, "smollm"), (qwen2_style, "qwen2")])
def test_gguf_tokenizer_metadata_matches_the_original(style, pre):
    """llama.cpp stores the vocabulary, merges and a pre-tokenizer name; rebuilt from those, the tokenizer must
    agree with the Hugging Face tokenizer it came from."""
    from etalii_dllm.bpe import spec_from_gguf

    reference = style()
    spec = json.loads(reference.to_str())
    vocab = spec["model"]["vocab"]
    tokens = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
    special = {t["content"] for t in spec["added_tokens"] if t["special"]}
    metadata = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": pre,
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": [3 if token in special else 1 for token in tokens],
        "tokenizer.ggml.merges": [m if isinstance(m, str) else " ".join(m) for m in spec["model"]["merges"]],
    }
    ours = BpeTokenizer(spec_from_gguf(metadata))
    for text in SAMPLES:
        assert ours.encode(text) == reference.encode(text, add_special_tokens=False).ids
    with pytest.raises(TokenizerError, match="pre-tokenizer"):
        spec_from_gguf({**metadata, "tokenizer.ggml.pre": "deepseek-coder"})
