"""T5 models through GGUF (#417-#420): llama.cpp's ``t5`` and ``t5encoder`` layouts checked against the ``gguf``
package, Unigram vocabularies rebuilt with the token ids of ``tokenizers``, and exports that import back to the same
model, bit for bit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from test_t5 import TEXTS, t5_tokenizer, write_t5_checkpoint
from test_t5_decoding import build, source_of
from test_t5_generation import FLAN

from etalii_dllm import t5_gguf
from etalii_dllm.engine import DllmEngine
from etalii_dllm.exporting import ExportError, export_gguf
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.importing import importer as importer_module
from etalii_dllm.importing.gguf import GgufFile
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.sampling import SamplingOptions

gguf = pytest.importorskip("gguf")
pytest.importorskip("tokenizers")
pytest.importorskip("sentencepiece")

LLAMA_CPP = {"relative_attention_max_distance": 128}
"""llama.cpp's fixed maximum distance of the relative buckets."""
WORDS = [
    *TEXTS,
    "  extra   spaces\tand\nnew lines  ",
    "\uff26\uff55\uff4c\uff4c ① ﬁ café naïve ÅNGSTRÖM",
    "translate English to German: The house is wonderful.",
    "東京 🙂 Привет",
    "",
]


@pytest.fixture(scope="module")
def t5(tmp_path_factory):
    return build(tmp_path_factory.mktemp("t5-gguf"), **LLAMA_CPP)


@pytest.fixture(scope="module")
def flan(tmp_path_factory):
    return build(tmp_path_factory.mktemp("flan-gguf"), **LLAMA_CPP, **FLAN)


@pytest.fixture(scope="module")
def embedder(tmp_path_factory):
    directory = tmp_path_factory.mktemp("t5-encoder-gguf")
    write_t5_checkpoint(directory / "checkpoint", **LLAMA_CPP)
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-t5")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


def model_file(model) -> ModelFile:
    return ModelFile(model[0].parent / "model.dllm")


def round_trip(model, tmp_path: Path, name: str) -> tuple[ModelFile, ModelFile, Path]:
    original = model_file(model)
    path = export_gguf(original, tmp_path / f"{name}.gguf")
    import_model(path, tmp_path / f"{name}.dllm")
    return original, ModelFile(tmp_path / f"{name}.dllm"), path


# The layout (#418)


@pytest.mark.parametrize("which", ["t5", "flan", "embedder"])
def test_tensor_names_follow_llama_cpp(which, request):
    from etalii_dllm.encoder_export import t5_tensor_name, text_to_text_tensor_name

    model = model_file(request.getfixturevalue(which))
    config = model.config
    arch = gguf.MODEL_ARCH.T5 if config.is_text_to_text else gguf.MODEL_ARCH.T5ENCODER
    names = gguf.TensorNameMap(arch, max(config.layers, config.decoder_layers))
    for ours in model.tensors:
        hf = text_to_text_tensor_name(ours, config) if config.is_text_to_text else t5_tensor_name(ours, config)
        expected = names.get_name(hf, try_suffixes=(".weight",))
        assert t5_gguf.gguf_t5_tensor_name(ours) == expected, (ours, hf)
        assert t5_gguf.t5_tensor_name(expected) == ours
    assert t5_gguf.t5_tensor_name("enc.blk.0.cross_attn_q.weight") is None
    assert t5_gguf.t5_tensor_name("blk.0.attn_q.weight") is None
    assert t5_gguf.t5_tensor_name("enc.blk.0.attn_qkv.weight") is None
    with pytest.raises(ExportError, match="no place"):
        t5_gguf.gguf_t5_tensor_name("layers.0.mlp.experts.0.up.weight")


def test_the_gguf_package_reads_the_files(flan, embedder, tmp_path):
    for name, model in (("flan", flan), ("encoder", embedder)):
        original = model_file(model)
        path = export_gguf(original, tmp_path / f"{name}.gguf")
        reader = gguf.GGUFReader(path)
        fields = {key: field.contents() for key, field in reader.fields.items()}
        config = original.config
        arch = "t5" if config.is_text_to_text else "t5encoder"
        keys = gguf.Keys
        assert fields["general.architecture"] == arch
        assert fields[keys.LLM.CONTEXT_LENGTH.format(arch=arch)] == config.context_length
        assert fields[keys.LLM.EMBEDDING_LENGTH.format(arch=arch)] == config.hidden_size
        assert fields[keys.LLM.FEED_FORWARD_LENGTH.format(arch=arch)] == config.intermediate_size
        assert fields[keys.LLM.BLOCK_COUNT.format(arch=arch)] == config.layers
        assert fields[keys.Attention.HEAD_COUNT.format(arch=arch)] == config.heads
        assert fields[keys.Attention.KEY_LENGTH.format(arch=arch)] == config.head_dim
        assert fields[keys.Attention.REL_BUCKETS_COUNT.format(arch=arch)] == config.position_buckets
        if config.is_text_to_text:
            assert fields[keys.LLM.DECODER_START_TOKEN_ID.format(arch=arch)] == 0
            assert fields[keys.LLM.DECODER_BLOCK_COUNT.format(arch=arch)] == config.decoder_layers
        assert fields[keys.Tokenizer.MODEL] == "t5"
        tokens = fields[keys.Tokenizer.LIST]
        assert len(tokens) == config.vocabulary_size and tokens[-1] == f"[PAD{config.vocabulary_size - 1}]"
        charsmap = reader.fields[keys.Tokenizer.PRECOMPILED_CHARSMAP]
        assert charsmap.types == [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT8]
        types = fields[keys.Tokenizer.TOKEN_TYPE]
        assert types[:3] == [3, 3, 2] and types[-1] == 5  # <pad> and </s> control, <unk> unknown, padding unused
        assert fields[keys.Tokenizer.EOS_ID] == 1 and fields[keys.Tokenizer.ADD_EOS] is True
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        assert set(tensors) == {t5_gguf.gguf_t5_tensor_name(name) for name in original.tensors}
        for ours, values in original.tensors.items():
            data = np.asarray(tensors[t5_gguf.gguf_t5_tensor_name(ours)].data)
            assert data.tobytes() == np.ascontiguousarray(values, dtype="<f4").tobytes()
        again = export_gguf(original, tmp_path / f"{name}-again.gguf")
        assert again.read_bytes() == path.read_bytes()  # byte-identical files


# Import (#419)


def test_text_to_text_round_trips_give_the_same_bits(t5, flan, tmp_path):
    for name, model in (("t5", t5), ("flan", flan)):
        original, again, _ = round_trip(model, tmp_path, name)
        assert again.fingerprint == original.fingerprint and again.config == original.config
        assert again.tokenizer["tokenizer_json"] == original.tokenizer["tokenizer_json"]
        engine = model[1]
        back = DllmEngine.from_model_file(tmp_path / f"{name}.dllm")
        assert back.system_fingerprint == engine.system_fingerprint
        for text in TEXTS[:3]:
            for options in (SamplingOptions(), SamplingOptions(temperature=0.8, seed=5)):
                assert back.complete(text, 12, options) == engine.complete(text, 12, options)
        source, answer = source_of(engine, TEXTS[1]), [5, 9, 1]
        assert np.array_equal(back.model.answer_logits(source, answer), engine.model.answer_logits(source, answer))


def test_encoder_round_trips_give_the_same_embeddings(embedder, tmp_path):
    original, again, _ = round_trip(embedder, tmp_path, "encoder")
    assert again.fingerprint == original.fingerprint and again.config == original.config
    assert again.embedding == original.embedding
    back = DllmEngine.from_model_file(tmp_path / "encoder.dllm")
    for text in TEXTS:
        assert back.embed(text).vector.tobytes() == embedder[1].embed(text).vector.tobytes()


class _Edited(GgufFile):
    """A GGUF file whose metadata ``EDIT`` changes, as another converter would have written it."""

    EDIT: ClassVar[dict] = {}

    def __init__(self, path) -> None:
        super().__init__(path)
        for key, value in self.EDIT.items():
            if value is None:
                self.metadata.pop(key, None)
            else:
                self.metadata[key] = value


def import_edited(monkeypatch, path: Path, target: Path, **edits) -> ModelFile:
    edited = type("Edited", (_Edited,), {"EDIT": {key.replace("__", "."): value for key, value in edits.items()}})
    monkeypatch.setattr(importer_module, "GgufFile", edited)
    import_model(path, target)
    return ModelFile(target)


def test_files_without_the_huggingface_tokenizer(flan, monkeypatch, tmp_path):
    from tokenizers import Tokenizer

    original, _, path = round_trip(flan, tmp_path, "flan")
    llama_cpp = import_edited(monkeypatch, path, tmp_path / "plain.dllm", tokenizer__huggingface__json=None)
    assert llama_cpp.fingerprint == original.fingerprint
    spec = llama_cpp.tokenizer["tokenizer_json"]
    reference = t5_tokenizer()
    rebuilt = Tokenizer.from_str(json.dumps(spec))
    ours = DllmEngine.from_model_file(tmp_path / "plain.dllm").tokenizer
    for text in WORDS:
        expected = reference.encode(text).ids
        assert rebuilt.encode(text).ids == expected
        assert [*ours.encode(text), 1] == expected
    assert llama_cpp.tokenizer["tokenizer_config"] == {
        "model_max_length": 512, "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"
    }  # fmt: skip


def test_import_refusals(flan, embedder, monkeypatch, tmp_path):
    _, _, path = round_trip(flan, tmp_path, "flan")
    _, _, encoder = round_trip(embedder, tmp_path, "encoder")
    with pytest.raises(ModelImportError, match="context-length"):
        import_model(path, tmp_path / "x.dllm", context_length=64)
    refusals = [
        (path, {"t5__decoder_start_token_id": 3}, "starts with token 0"),
        (path, {"t5__attention__value_length": 4}, "different lengths"),
        (path, {"t5__embedding_length": None}, "t5.embedding_length is missing"),
        (path, {"t5__attention__head_count": 0}, "does not describe a T5 model"),
        (path, {"tokenizer__huggingface__json": None, "tokenizer__ggml__model": "llama"}, "not supported"),
        (path, {"tokenizer__huggingface__json": None, "tokenizer__ggml__scores": [0.0]}, "cannot be read"),
        (path, {"general__architecture": "t5encoder", "t5encoder__block_count": 2}, "unexpected tensor"),
        (encoder, {"t5encoder__pooling_type": 7}, "pooling type 7"),
    ]
    for source, edits, message in refusals:
        with pytest.raises(ModelImportError, match=message):
            import_edited(monkeypatch, source, tmp_path / "x.dllm", **edits)


# Export refusals (#418) and the vocabulary (#417)


def test_export_refusals(tmp_path):
    for name, changes, message in (
        ("distance", {}, "maximum distance at 128, not 16"),
        ("silu", {**LLAMA_CPP, "feed_forward_proj": "gated-silu"}, "tanh GELU, not silu"),
        ("gelu", {**LLAMA_CPP, "feed_forward_proj": "gelu"}, "ReLU, not gelu"),
    ):
        write_t5_checkpoint(tmp_path / name, **changes)
        import_model(tmp_path / name, tmp_path / f"{name}.dllm")
        with pytest.raises(ExportError, match=message):
            export_gguf(ModelFile(tmp_path / f"{name}.dllm"), tmp_path / f"{name}.gguf")
    write_t5_checkpoint(tmp_path / "dense", dense="identity", **LLAMA_CPP)
    import_model(tmp_path / "dense", tmp_path / "dense.dllm")
    with pytest.raises(ExportError, match="Dense projection"):
        export_gguf(ModelFile(tmp_path / "dense.dllm"), tmp_path / "dense.gguf")


def spec() -> dict:
    return json.loads(t5_tokenizer().to_str())


def test_vocabulary_refusals_and_variants():
    base = spec()
    metadata = {key: value for key, _, value in t5_gguf.unigram_vocabulary(base, 310)}
    assert metadata["tokenizer.ggml.remove_extra_whitespaces"] is True
    assert metadata["tokenizer.ggml.add_space_prefix"] is True
    bpe = {**base, "model": {"type": "BPE", "vocab": {}, "merges": []}}
    first = {**base, "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first"}}
    star = {**base, "pre_tokenizer": {"type": "Metaspace", "replacement": "*"}}
    lowercase = {**base, "normalizer": {"type": "Lowercase"}}
    split = {**base, "pre_tokenizer": {"type": "WhitespaceSplit"}}
    stray = {**base, "added_tokens": [{"id": 5, "content": "<nope>", "special": True}]}
    for broken, message in (
        (bpe, "only Unigram"),
        (first, "prepend_scheme 'first'"),
        (star, "with ▁"),
        (lowercase, "normaliser"),
        (split, "pre-tokenizer"),
        (stray, "added token"),
    ):
        with pytest.raises(ExportError, match=message):
            t5_gguf.unigram_vocabulary(broken, 310)
    with pytest.raises(ExportError, match="more pieces"):
        t5_gguf.unigram_vocabulary(base, 10)
    # Without the space-collapsing step, or the prefix: plain Metaspace, as the GGUF flags say.
    plain = {**base, "normalizer": base["normalizer"]["normalizers"][0], "pre_tokenizer": {
        "type": "Metaspace", "replacement": "▁", "add_prefix_space": False}}  # fmt: skip
    values = {key: value for key, _, value in t5_gguf.unigram_vocabulary(plain, 303)}
    assert values["tokenizer.ggml.remove_extra_whitespaces"] is False
    assert values["tokenizer.ggml.add_space_prefix"] is False
    flat = {key: v[1] if isinstance(v, tuple) else v for key, v in values.items()}
    rebuilt = t5_gguf.unigram_spec(flat)
    assert rebuilt["pre_tokenizer"]["prepend_scheme"] == "never" and rebuilt["normalizer"]["type"] == "Precompiled"
    assert rebuilt["post_processor"] is None  # no end of sequence named
    # Byte pieces keep their type and turn byte fallback on.
    pieces = [["<unk>", 0.0], ["<0x41>", -1.0], ["a", -2.0]]
    raw = {"model": {"type": "Unigram", "unk_id": 0, "vocab": pieces, "byte_fallback": True},
           "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always"}}  # fmt: skip
    values = {key: v[1] if isinstance(v, tuple) else v for key, _, v in t5_gguf.unigram_vocabulary(raw, 3)}
    assert values["tokenizer.ggml.token_type"] == [2, 6, 1]
    again = t5_gguf.unigram_spec(values)
    assert again["model"]["byte_fallback"] is True and again["normalizer"] is None
    assert again["model"]["unk_id"] == 0
    with pytest.raises(ValueError, match="one score"):
        t5_gguf.unigram_spec({"tokenizer.ggml.tokens": ["a"], "tokenizer.ggml.scores": []})


def test_cli_export_and_import(flan, tmp_path, cli_environment, capsys):
    from etalii_dllm.cli import main as cli

    path = flan[0].parent / "model.dllm"
    out = tmp_path / "flan.gguf"
    assert cli(["export", str(path), "--format", "gguf", "-o", str(out)]) == 0
    assert cli(["import", str(out), "-o", str(tmp_path / "back.dllm")]) == 0
    assert ModelFile(tmp_path / "back.dllm").fingerprint == ModelFile(path).fingerprint
