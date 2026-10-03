"""Phase 47: GGUF files with a SentencePiece vocabulary (#297), tokenized exactly like the sentencepiece library
(#298), SentencePiece-style models exported to GGUF and back (#299), with a golden value (#300)."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
import sentencepiece
from gguf import GGUFReader
from golden_values import SENTENCEPIECE_GGUF_TOKENS
from model_fixtures import tiny_config, write_hf_checkpoint
from test_bpe import CORPUS, SAMPLES, llama2_style, mistral_style

import etalii_dllm
from etalii_dllm.bpe import BpeTokenizer, TokenizerError, from_model_header, sentencepiece_merges, spec_from_gguf
from etalii_dllm.exporting import ExportError, export_gguf, export_safetensors
from etalii_dllm.importing import import_model
from etalii_dllm.modelfile import ModelFile

EXTRA = ["Hello world  x\né🙂", "  two leading", "trailing  ", "aaaaaa bbbb", "ababab abab", "東京", "\t\r\n", "x"]
# A larger corpus that every test environment has (wheel tests run without the docs): the package's own sources.
SOURCES = [
    line
    for path in sorted(Path(etalii_dllm.__file__).parent.glob("*.py"))
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]


def _kind(processor: sentencepiece.SentencePieceProcessor, index: int) -> int:
    """The GGUF token type of a piece: 2 unknown, 3 control, 6 byte, 5 unused, 1 normal."""
    if processor.is_unknown(index):
        return 2
    if processor.is_control(index):
        return 3
    if processor.is_byte(index):
        return 6
    return 5 if processor.is_unused(index) else 1


def trained(lines: list[str], vocabulary: int, prefix: bool) -> tuple[sentencepiece.SentencePieceProcessor, dict]:
    """A SentencePiece BPE model trained like Llama's (identity normalisation, byte fallback) and its GGUF metadata."""
    model = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(lines), model_writer=model, model_type="bpe", vocab_size=vocabulary,
        byte_fallback=True, normalization_rule_name="identity", add_dummy_prefix=prefix,
        remove_extra_whitespaces=False, character_coverage=0.98, hard_vocab_limit=False, split_digits=True,
        num_threads=1, minloglevel=2,
    )  # fmt: skip
    processor = sentencepiece.SentencePieceProcessor(model_proto=model.getvalue())
    size = processor.get_piece_size()
    metadata = {
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": [processor.id_to_piece(i) for i in range(size)],
        "tokenizer.ggml.scores": [processor.get_score(i) for i in range(size)],
        "tokenizer.ggml.token_type": [_kind(processor, i) for i in range(size)],
        "tokenizer.ggml.add_space_prefix": prefix,
    }
    return processor, metadata


# -- the sentencepiece library is the reference ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lines", "vocabulary", "prefix"),
    [(CORPUS * 3, 400, True), (SOURCES, 3000, True), (SOURCES, 1500, False)],
    ids=["small", "sources", "sources-no-prefix"],
)
def test_tokens_match_the_sentencepiece_library(lines, vocabulary, prefix):
    processor, metadata = trained(lines, vocabulary, prefix)
    tokenizer = BpeTokenizer(spec_from_gguf(metadata))
    texts = [*SAMPLES, *EXTRA, *SOURCES[::7], *(line[::-1] for line in SOURCES[::11])]
    for text in texts:
        ids = tokenizer.encode(text)
        assert ids == processor.encode(text), text
        assert tokenizer.decode(ids) == processor.decode(ids), text


def test_merges_follow_scores_and_ties_go_left():
    tokens = ["<unk>", "<s>", "a", "b", "ab", "ba", "aba", "<0x41>"]
    scores = [0.0, 0.0, -10.0, -11.0, -1.0, -1.0, 0.5, 0.0]
    types = [2, 3, 1, 1, 1, 1, 1, 6]
    merges, ranks = sentencepiece_merges(tokens, scores, types)
    assert merges == [["a", "ba"], ["ab", "a"], ["a", "b"], ["b", "a"]] and ranks == [0, 0, 1, 1]
    metadata = {"tokenizer.ggml.model": "llama", "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.scores": scores,
                "tokenizer.ggml.token_type": types, "tokenizer.ggml.add_space_prefix": False}  # fmt: skip
    tokenizer = BpeTokenizer(spec_from_gguf(metadata))
    # a b a b a: "ab" and "ba" score the same, so the leftmost pair merges first; then "aba" outscores both.
    assert tokenizer.encode("ababa") == [6, 5]
    assert tokenizer.encode("A") == [7] and tokenizer.encode("c") == [0]  # byte fallback, else unknown
    assert tokenizer.encode("a<s>b") == [2, 1, 3] and tokenizer.decode([2, 1, 3]) == "ab"

    with pytest.raises(TokenizerError, match="not a number"):
        sentencepiece_merges(["ab", "a", "b"], [float("nan"), 0.0, 0.0], [1, 1, 1])
    with pytest.raises(TokenizerError, match="one score and one type"):
        spec_from_gguf({**metadata, "tokenizer.ggml.scores": [0.0]})
    with pytest.raises(TokenizerError, match="llama SentencePiece"):
        spec_from_gguf({**metadata, "tokenizer.ggml.model": "bert"})
    plain = spec_from_gguf({"tokenizer.ggml.model": "llama", "tokenizer.ggml.tokens": ["a", "b", "ab"]})
    assert plain["model"]["unk_token"] is None and plain["normalizer"]["type"] == "Sequence"


# -- models through GGUF -----------------------------------------------------------------------------------------


def synthetic() -> dict:
    """GGUF metadata of a SentencePiece vocabulary built without any trainer, so it is the same everywhere: control
    and byte pieces, the corpus's characters, and its most frequent two- to four-character pieces (spaces as ``▁``),
    each scored by its rank."""
    counts: dict[str, int] = {}
    for line in CORPUS:
        text = "\u2581" + line.replace(" ", "\u2581")
        for size in (2, 3, 4):
            for start in range(len(text) - size + 1):
                piece = text[start : start + size]
                counts[piece] = counts.get(piece, 0) + 1
    characters = sorted({c for line in CORPUS for c in "\u2581" + line.replace(" ", "\u2581")})
    pieces = sorted((p for p, n in counts.items() if n >= 3), key=lambda p: (-counts[p], p))
    tokens = ["<unk>", "<s>", "</s>", *(f"<0x{b:02X}>" for b in range(256)), *pieces, *characters]
    types = [2, 3, 3, *[6] * 256, *[1] * (len(pieces) + len(characters))]
    scores = [0.0] * 259 + [-float(i) for i in range(len(pieces) + len(characters))]
    return {"tokenizer.ggml.model": "llama", "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.scores": scores,
            "tokenizer.ggml.token_type": types, "tokenizer.ggml.add_space_prefix": True}  # fmt: skip


def test_golden_tokens():
    tokenizer = BpeTokenizer(spec_from_gguf(synthetic()))
    digest = hashlib.sha256(json.dumps([tokenizer.encode(text) for text in SAMPLES]).encode()).hexdigest()
    assert digest == SENTENCEPIECE_GGUF_TOKENS


def _hf_model(directory, spec: dict) -> ModelFile:
    vocabulary = max(spec["model"]["vocab"].values()) + 1
    config = {**tiny_config("llama"), "vocab_size": vocabulary}
    write_hf_checkpoint(directory / "checkpoint", config, tokenizer_json=spec)
    import_model(directory / "checkpoint", directory / "model.dllm")
    return ModelFile(directory / "model.dllm")


def test_sentencepiece_models_round_trip_through_gguf(tmp_path):
    from tokenizers import Tokenizer

    # A Hugging Face tokenizer.json of a SentencePiece model (as transformers converts one: every split of a piece,
    # in score order); with distinct scores, the tokenizers library and SentencePiece's order agree.
    spec = spec_from_gguf(synthetic())
    del spec["model"]["merge_ranks"]
    reference = Tokenizer.from_str(json.dumps(spec))
    (tmp_path / "hf").mkdir()
    (tmp_path / "gguf").mkdir()
    original = _hf_model(tmp_path / "hf", spec)
    ours = from_model_header(original.tokenizer)
    assert all(reference.encode(text, add_special_tokens=False).ids == ours.encode(text) for text in SAMPLES)

    export_gguf(original, tmp_path / "hf" / "model.gguf")
    reader = GGUFReader(tmp_path / "hf" / "model.gguf")
    model_field = reader.fields["tokenizer.ggml.model"]
    assert bytes(model_field.parts[model_field.data[0]]) == b"llama" and "tokenizer.ggml.scores" in reader.fields

    imported = import_model(tmp_path / "hf" / "model.gguf", tmp_path / "gguf" / "model.dllm")
    assert imported.fingerprint == original.fingerprint
    back = ModelFile(tmp_path / "gguf" / "model.dllm")
    assert back.tokenizer["tokenizer.ggml.model"] == "llama"
    theirs = from_model_header(back.tokenizer)
    for text in [*SAMPLES, *EXTRA, "a</s>b"]:
        ids = ours.encode(text)
        assert theirs.encode(text) == ids and theirs.decode(ids) == ours.decode(ids)

    # The GGUF import writes the same GGUF file again, byte for byte.
    export_gguf(back, tmp_path / "gguf" / "model.gguf")
    assert (tmp_path / "gguf" / "model.gguf").read_bytes() == (tmp_path / "hf" / "model.gguf").read_bytes()

    # And a safetensors export of it is a tokenizer.json the tokenizers library reads (merges in SentencePiece order).
    export_safetensors(back, tmp_path / "exported")
    exported = json.loads((tmp_path / "exported" / "tokenizer.json").read_text(encoding="utf-8"))
    assert "merge_ranks" not in exported["model"]
    reference = Tokenizer.from_str(json.dumps(exported))
    assert all(reference.encode(text, add_special_tokens=False).ids == theirs.encode(text) for text in SAMPLES)


def test_hugging_face_trained_tokenizers_export_when_they_encode_alike(tmp_path):
    model = _hf_model(tmp_path, json.loads(llama2_style().to_str()))
    try:
        export_gguf(model, tmp_path / "model.gguf")
    except ExportError as error:  # a trained merge list SentencePiece's order cannot reproduce on the probes
        assert "does not encode like a GGUF llama tokenizer" in str(error)
    else:
        imported = ModelFile(import_model(tmp_path / "model.gguf", tmp_path / "back.dllm").path)
        ours, theirs = from_model_header(model.tokenizer), from_model_header(imported.tokenizer)
        assert all(theirs.encode(text) == ours.encode(text) for text in SAMPLES[:6])


def test_tokenizers_gguf_cannot_express_are_refused(tmp_path):
    # Metaspace with prepend_scheme "first" adds no space after a special token; a GGUF llama tokenizer always does.
    model = _hf_model(tmp_path, json.loads(mistral_style().to_str()))
    with pytest.raises(ExportError, match="does not encode like a GGUF llama tokenizer"):
        export_gguf(model, tmp_path / "model.gguf")
