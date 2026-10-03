"""Exporting encoders (issue #355): BERT, RoBERTa and XLM-RoBERTa embedders and cross-encoders back to Hugging Face
safetensors (with the sentence-transformers modules) and BERT to GGUF in llama.cpp's layout; every export imports
again to the same model, transformers loads the safetensors, and the gguf package reads the GGUF."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_cross_encoders import write_cross_encoder
from test_encoders import write_bert_checkpoint
from test_roberta_encoders import write_roberta_checkpoint

from etalii_dllm import exporting
from etalii_dllm.cli import main as cli
from etalii_dllm.encoder_export import phantom_vocabulary, wordpiece_vocabulary
from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.modelfile import ModelFile

tokenizers = pytest.importorskip("tokenizers")

TEXTS = ["The quick brown fox.", "Hello, world! How are you?", "naïve café façade"]


def imported(tmp_path: Path, name: str) -> Path:
    """A tiny encoder of each kind, imported."""
    directory = tmp_path / name
    if name == "bert":
        write_bert_checkpoint(directory)
    elif name == "bert-cls":
        write_bert_checkpoint(directory, pooling="pooling_mode_cls_token")
    elif name == "cross":
        write_cross_encoder(directory)
    elif name == "cross-sigmoid":
        write_cross_encoder(directory, 2, activation="torch.nn.modules.activation.Sigmoid")
    else:
        pytest.importorskip("sentencepiece")
        model_type, _, cross = name.partition("-cross")
        write_roberta_checkpoint(directory, model_type, labels=1 if cross == "" and name.endswith("-cross") else 0)
    import_model(directory, tmp_path / f"{name}.dllm", repository=f"example/{name}")
    return tmp_path / f"{name}.dllm"


def same_model(a: Path, b: Path) -> None:
    first, second = ModelFile(a), ModelFile(b)
    assert first.fingerprint == second.fingerprint
    assert first.config == second.config
    assert (first.embedding, first.classifier) == (second.embedding, second.classifier)


@pytest.mark.parametrize("name", ["bert", "bert-cls", "cross", "cross-sigmoid", "xlm-roberta", "roberta"])
def test_safetensors_exports_import_back(tmp_path, name):
    path = imported(tmp_path, name)
    files = exporting.export_model(path, tmp_path / "export", "safetensors")
    names = {p.relative_to(tmp_path / "export").as_posix() for p in files}
    assert {"config.json", "model.safetensors", "tokenizer.json", "README.md"} <= names
    assert ("modules.json" in names) == ("cross" not in name)
    import_model(tmp_path / "export", tmp_path / "again.dllm", repository=f"example/{name}")
    same_model(path, tmp_path / "again.dllm")
    # byte-identical exports
    again = exporting.export_model(path, tmp_path / "export-2", "safetensors")
    assert [p.read_bytes() for p in files] == [p.read_bytes() for p in again]


def test_roberta_cross_encoder_exports(tmp_path):
    pytest.importorskip("sentencepiece")
    write_roberta_checkpoint(tmp_path / "checkpoint", labels=2)
    import_model(tmp_path / "checkpoint", tmp_path / "cross.dllm")
    exporting.export_model(tmp_path / "cross.dllm", tmp_path / "export", "safetensors")
    config = json.loads((tmp_path / "export" / "config.json").read_text())
    assert config["architectures"] == ["XLMRobertaForSequenceClassification"] and config["pad_token_id"] == 1
    import_model(tmp_path / "export", tmp_path / "again.dllm")
    same_model(tmp_path / "cross.dllm", tmp_path / "again.dllm")


def test_transformers_loads_the_exports(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    for name in ("bert", "cross"):
        path = imported(tmp_path, name)
        exporting.export_model(path, tmp_path / f"{name}-export", "safetensors")
        engine = DllmEngine.from_model_file(path)
        tokens = [2, 7, 30, 11, 3]
        if name == "bert":
            model = transformers.AutoModel.from_pretrained(tmp_path / f"{name}-export").eval()
            with torch.no_grad():
                theirs = model(torch.tensor([tokens])).last_hidden_state[0].numpy()
            assert np.allclose(engine.model.hidden_states(tokens), theirs, atol=1e-5)
        else:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(tmp_path / f"{name}-export")
            with torch.no_grad():
                theirs = model.eval()(torch.tensor([tokens])).logits[0].numpy()
            assert np.allclose(engine.model.classify(tokens), theirs, atol=1e-5)
    settings = json.loads((tmp_path / "bert-export" / "1_Pooling" / "config.json").read_text())
    assert settings["pooling_mode_mean_tokens"] and not settings["pooling_mode_cls_token"]
    assert json.loads((tmp_path / "bert-export" / "sentence_bert_config.json").read_text())["max_seq_length"] == 12


@pytest.mark.parametrize("name", ["bert", "bert-cls", "cross", "cross-sigmoid"])
def test_gguf_exports_import_back(tmp_path, name):
    gguf = pytest.importorskip("gguf")
    path = imported(tmp_path, name)
    out = exporting.export_model(path, tmp_path / "model.gguf", "gguf")[0]
    import_model(out, tmp_path / "again.dllm", repository=f"example/{name}")
    same_model(path, tmp_path / "again.dllm")
    reader = gguf.GGUFReader(str(out))
    fields = reader.fields
    assert bytes(fields["general.architecture"].parts[-1]).decode() == "bert"
    pooling = int(fields["bert.pooling_type"].parts[-1][0])
    assert pooling == {"bert": 1, "bert-cls": 2}.get(name, 4)
    assert not bool(fields["bert.attention.causal"].parts[-1][0])
    names = {tensor.name for tensor in reader.tensors}
    assert {"token_embd.weight", "token_types.weight", "position_embd.weight", "blk.1.layer_output_norm.bias"} <= names
    assert ("cls.output.weight" in names) == name.startswith("cross")
    tokens = [bytes(fields["tokenizer.ggml.tokens"].parts[i]).decode() for i in fields["tokenizer.ggml.tokens"].data]
    assert "[CLS]" in tokens and "▁the" in tokens
    # the same bytes again
    assert exporting.export_model(path, tmp_path / "again.gguf", "gguf")[0].read_bytes() == out.read_bytes()


def test_llama_cpp_style_bert_gguf_imports(tmp_path):
    """A GGUF written the way llama.cpp's converter writes BERT (no tokenizer.huggingface.json): the WordPiece
    vocabulary comes back from the phantom-space tokens with BERT's lower-casing normaliser."""
    gguf = pytest.importorskip("gguf")
    path = imported(tmp_path, "bert")
    model = ModelFile(path)
    from etalii_dllm.encoder_export import gguf_encoder_tensor_name

    spec = model.tokenizer["tokenizer_json"]
    tokens, kinds = phantom_vocabulary(spec)
    writer = gguf.GGUFWriter(str(tmp_path / "llama.gguf"), "bert")
    config = model.config
    writer.add_context_length(config.context_length)
    writer.add_embedding_length(config.hidden_size)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_block_count(config.layers)
    writer.add_head_count(config.heads)
    writer.add_layer_norm_eps(config.rms_norm_eps)
    writer.add_causal_attention(False)
    writer.add_pooling_type(gguf.PoolingType.MEAN)
    writer.add_tokenizer_model("bert")
    writer.add_token_list(tokens)
    writer.add_token_types(kinds)
    writer.add_token_type_count(config.type_vocabulary_size)
    pieces = wordpiece_vocabulary(tokens, kinds)
    writer.add_bos_token_id(pieces.index("[CLS]"))  # llama.cpp keeps BERT's [CLS] as the bos token
    writer.add_sep_token_id(pieces.index("[SEP]"))
    writer.add_unk_token_id(pieces.index("[UNK]"))
    for name, values in model.tensors.items():
        writer.add_tensor(gguf_encoder_tensor_name(name), np.asarray(values, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    import_model(tmp_path / "llama.gguf", tmp_path / "llama.dllm", licence="apache-2.0")
    again = ModelFile(tmp_path / "llama.dllm")
    assert again.fingerprint == model.fingerprint
    assert again.embedding["pooling"] == "mean"
    ours, original = DllmEngine.from_model_file(tmp_path / "llama.dllm"), DllmEngine.from_model_file(path)
    for text in [*TEXTS, "UPPER Case Ünïcödé"]:
        assert ours.tokenizer.encode(text, add_special_tokens=True) == original.tokenizer.encode(
            text, add_special_tokens=True
        )


def test_export_refusals(tmp_path):
    pytest.importorskip("sentencepiece")
    roberta = imported(tmp_path, "xlm-roberta")
    with pytest.raises(exporting.ExportError, match="RoBERTa and XLM-RoBERTa encoders cannot be written to GGUF"):
        exporting.export_model(roberta, tmp_path / "x.gguf", "gguf")
    write_bert_checkpoint(tmp_path / "tanh", config={"hidden_act": "gelu_new"})
    import_model(tmp_path / "tanh", tmp_path / "tanh.dllm")
    with pytest.raises(exporting.ExportError, match="no tanh-GELU"):
        exporting.export_model(tmp_path / "tanh.dllm", tmp_path / "x.gguf", "gguf")
    exporting.export_model(tmp_path / "tanh.dllm", tmp_path / "tanh-export", "safetensors")
    assert json.loads((tmp_path / "tanh-export" / "config.json").read_text())["hidden_act"] == "gelu_new"
    with pytest.raises(exporting.ExportError, match="## continuations"):
        phantom_vocabulary({"model": {"vocab": {"a": 0}, "continuing_subword_prefix": "@@"}})
    with pytest.raises(exporting.ExportError, match="gaps"):
        phantom_vocabulary({"model": {"vocab": {"a": 0, "b": 2}}})
    with pytest.raises(exporting.ExportError, match="cannot keep apart"):
        phantom_vocabulary({"model": {"vocab": {"##▁a": 0, "▁a": 1}}})


def test_gguf_bert_import_refusals(tmp_path):
    gguf = pytest.importorskip("gguf")

    def write(name: str, *, causal: bool = False, tensors: dict | None = None, model: str = "bert") -> Path:
        writer = gguf.GGUFWriter(str(tmp_path / name), "bert")
        writer.add_block_count(1)
        writer.add_causal_attention(causal)
        writer.add_tokenizer_model(model)
        writer.add_token_list(["[CLS]", "[SEP]", "[UNK]", "▁a"])
        writer.add_token_types([3, 3, 3, 1])
        for key, values in (tensors or {"token_embd.weight": np.zeros((4, 4), np.float32)}).items():
            writer.add_tensor(key, values)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return tmp_path / name

    with pytest.raises(ModelImportError, match="causal"):
        import_model(write("causal.gguf", causal=True), tmp_path / "x.dllm", licence="mit")
    with pytest.raises(ModelImportError, match="unexpected tensor"):
        odd = write("odd.gguf", tensors={"blk.0.odd.weight": np.zeros(4, np.float32)})
        import_model(odd, tmp_path / "x.dllm", licence="mit")
    pieces = {"cls.output.weight": np.zeros((1, 4), np.float32)}
    with pytest.raises(ModelImportError, match="without its pooler"):
        import_model(write("head.gguf", tensors=pieces), tmp_path / "x.dllm", licence="mit")
    with pytest.raises(ModelImportError, match="embedding_length is missing"):
        import_model(write("short.gguf"), tmp_path / "x.dllm", licence="mit")


def test_cli_exports_an_encoder(tmp_path, capsys):
    path = imported(tmp_path, "cross")
    assert cli(["export", str(path), "--format", "gguf", "-o", str(tmp_path / "cross.gguf")]) == 0
    assert "cross.gguf" in capsys.readouterr().out
    assert cli(["export", str(path), "--format", "safetensors", "-o", str(tmp_path / "cross")]) == 0
    assert (tmp_path / "cross" / "model.safetensors").exists()
