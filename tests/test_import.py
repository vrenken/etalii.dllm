"""Importer tests on tiny synthetic models: readers, dequantisation, the model.dllm container and ``dllm import``."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from golden_values import TINY_IMPORT_FINGERPRINT
from model_fixtures import (
    TINY_LLAMA_CONFIG,
    bf16_to_float32,
    to_bf16_bits,
    write_gguf,
    write_hf_checkpoint,
    write_safetensors,
)

from etalii_dllm.cli import main as cli
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.importing.gguf import GgufFile
from etalii_dllm.importing.hub import download, parse_reference
from etalii_dllm.importing.quants import BLOCK_FORMATS, dequantize
from etalii_dllm.importing.safetensors import SafetensorsError, SafetensorsFile
from etalii_dllm.modelfile import ModelFile, ModelFileError
from etalii_dllm.numerics import fill_gaussian


def bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype="<f4").view("<u4")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- safetensors -------------------------------------------------------------------------------------------------


def test_safetensors_widening_is_exact(tmp_path):
    values = fill_gaussian(3, 24).reshape(4, 6)
    half = values.astype("<f2")
    brain = to_bf16_bits(values)
    write_safetensors(
        tmp_path / "t.safetensors",
        {"f32": ("F32", values), "f16": ("F16", half), "bf16": ("BF16", brain), "ids": ("I64", np.arange(3))},
    )
    file = SafetensorsFile(tmp_path / "t.safetensors")
    assert file.names() == ["f32", "f16", "bf16", "ids"]
    assert np.array_equal(bits(file["f32"].to_float32()), bits(values))
    assert np.array_equal(bits(file["f16"].to_float32()), bits(half.astype(np.float32)))
    assert np.array_equal(bits(file["bf16"].to_float32()), bits(bf16_to_float32(brain)))
    with pytest.raises(SafetensorsError):
        file["ids"].to_float32()


def test_safetensors_reads_files_written_by_the_reference_library(tmp_path):
    numpy_api = pytest.importorskip("safetensors.numpy")
    values = {"a": fill_gaussian(5, 12).reshape(3, 4), "b": fill_gaussian(6, 5).astype(np.float16)}
    numpy_api.save_file(values, str(tmp_path / "ref.safetensors"), metadata={"format": "np"})
    file = SafetensorsFile(tmp_path / "ref.safetensors")
    assert file.metadata == {"format": "np"}
    for name, expected in values.items():
        assert file[name].shape == expected.shape
        assert np.array_equal(file[name].raw, expected)


def test_safetensors_rejects_overlapping_tensors(tmp_path):
    header = {
        "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
    }
    encoded = json.dumps(header).encode()
    (tmp_path / "bad.safetensors").write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(12))
    with pytest.raises(SafetensorsError, match="overlap"):
        SafetensorsFile(tmp_path / "bad.safetensors")


# --- GGUF --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("type_id", sorted(BLOCK_FORMATS))
def test_dequantisation_is_bit_identical_to_llama_cpp(type_id):
    gguf = pytest.importorskip("gguf")
    block = BLOCK_FORMATS[type_id]
    blocks = 40
    # Random block bytes, with every float16 scale replaced by a finite value.
    raw = np.frombuffer(hashlib.shake_256(block.name.encode()).digest(blocks * block.block_bytes), dtype=np.uint8)
    raw = raw.reshape(blocks, block.block_bytes).copy()
    scales = to_float16_bytes(fill_gaussian(type_id, blocks * 2) * np.float32(0.05)).reshape(blocks, 2, 2)
    scale_columns = {"Q6_K": [208]}.get(block.name, [0, 2] if block.name in ("Q4_1", "Q5_1", "Q4_K", "Q5_K") else [0])
    for i, column in enumerate(scale_columns):
        raw[:, column : column + 2] = scales[:, i]
    ours = dequantize(raw.reshape(-1), block, blocks * block.block_size)
    reference = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(type_id)).reshape(-1)
    assert np.array_equal(bits(ours), bits(reference))


def to_float16_bytes(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype="<f2").view(np.uint8)


def test_gguf_reader_parses_metadata_and_tensors(tmp_path):
    pytest.importorskip("gguf")
    weights = write_gguf(tmp_path / "tiny.gguf")
    file = GgufFile(tmp_path / "tiny.gguf")
    assert file.version == 3
    assert file.metadata["general.architecture"] == "llama"
    assert file.metadata["llama.block_count"] == 2
    assert file.metadata["tokenizer.ggml.tokens"][:2] == ["t0", "t1"]
    embedding = file["token_embd.weight"]
    assert embedding.shape == (64, 16)
    assert np.array_equal(bits(embedding.to_float32()), bits(weights["model.embed_tokens.weight"]))


# --- model.dllm and import ---------------------------------------------------------------------------------------


def test_import_huggingface_checkpoint(tmp_path):
    weights = write_hf_checkpoint(tmp_path / "tiny")
    result = import_model(tmp_path / "tiny", tmp_path / "tiny.dllm", repository="example/tiny", revision="abc123")
    model = ModelFile(tmp_path / "tiny.dllm")
    assert model.fingerprint == result.fingerprint == TINY_IMPORT_FINGERPRINT
    config = model.config
    assert (config.family, config.layers, config.heads, config.kv_heads, config.head_dim) == ("llama", 2, 4, 2, 4)
    assert config.tie_word_embeddings and "lm_head.weight" not in model.tensors
    assert config.eos_token_ids == (2, 3)
    assert np.array_equal(
        bits(model.tensors["layers.1.attention.q.weight"]), bits(weights["model.layers.1.self_attn.q_proj.weight"])
    )
    assert model.tensors["token_embedding.weight"].ctypes.data % 64 == 0
    assert model.source["repository"] == "example/tiny" and model.source["revision"] == "abc123"
    assert {f["path"] for f in model.source["files"]} >= {"config.json", "model.safetensors", "README.md"}
    assert model.licence["spdx"] == "Apache-2.0" and model.licence["redistributable"]
    assert "Apache License" in model.licence["text"] and "example/tiny" in model.licence["attribution"]
    assert model.chat_template == json.loads((tmp_path / "tiny" / "tokenizer_config.json").read_text())["chat_template"]
    assert model.tokenizer["tokenizer_json"]["model"]["type"] == "BPE"


def test_import_is_byte_for_byte_reproducible(tmp_path):
    write_hf_checkpoint(tmp_path / "tiny")
    import_model(tmp_path / "tiny", tmp_path / "a.dllm")
    import_model(tmp_path / "tiny", tmp_path / "b.dllm")
    assert file_sha256(tmp_path / "a.dllm") == file_sha256(tmp_path / "b.dllm")


def test_gguf_and_huggingface_imports_give_the_same_weights(tmp_path):
    """llama.cpp permutes Q/K rows for Llama; the importer undoes it, so an F32 GGUF and the original checkpoint
    convert to the same tensors and therefore the same fingerprint."""
    pytest.importorskip("gguf")
    write_hf_checkpoint(tmp_path / "tiny")
    write_gguf(tmp_path / "tiny.gguf")
    from_hf = import_model(tmp_path / "tiny", tmp_path / "hf.dllm")
    from_gguf = import_model(tmp_path / "tiny.gguf", tmp_path / "gguf.dllm")
    assert from_gguf.fingerprint == from_hf.fingerprint
    assert from_gguf.config.eos_token_ids == (2,)
    model = ModelFile(tmp_path / "gguf.dllm")
    assert model.tokenizer["tokenizer.ggml.model"] == "gpt2"
    assert model.chat_template is not None


def test_qwen2_gguf_is_not_permuted(tmp_path):
    pytest.importorskip("gguf")
    config = {**TINY_LLAMA_CONFIG, "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}
    write_hf_checkpoint(tmp_path / "qwen", config)
    write_gguf(tmp_path / "qwen.gguf", config)
    from_hf = import_model(tmp_path / "qwen", tmp_path / "hf.dllm")
    from_gguf = import_model(tmp_path / "qwen.gguf", tmp_path / "gguf.dllm")
    assert from_hf.config.attention_bias and from_gguf.config.attention_bias
    assert from_gguf.fingerprint == from_hf.fingerprint


def test_quantised_gguf_imports_deterministically(tmp_path):
    gguf = pytest.importorskip("gguf")
    write_gguf(tmp_path / "q8.gguf", quantization="Q8_0")
    first = import_model(tmp_path / "q8.gguf", tmp_path / "a.dllm")
    second = import_model(tmp_path / "q8.gguf", tmp_path / "b.dllm")
    assert first.fingerprint == second.fingerprint
    model = ModelFile(tmp_path / "a.dllm")
    source = GgufFile(tmp_path / "q8.gguf")["blk.0.ffn_down.weight"]
    assert source.type_name == "Q8_0"
    reference = gguf.quants.dequantize(np.asarray(source.raw), gguf.GGMLQuantizationType.Q8_0).reshape(16, 32)
    assert np.array_equal(bits(model.tensors["layers.0.mlp.down.weight"]), bits(reference))
    entry = next(t for t in model.header["tensors"] if t["name"] == "layers.0.mlp.down.weight")
    assert entry["source_dtype"] == "Q8_0"


def test_corrupt_model_file_is_detected(tmp_path):
    write_hf_checkpoint(tmp_path / "tiny")
    import_model(tmp_path / "tiny", tmp_path / "tiny.dllm")
    data = bytearray((tmp_path / "tiny.dllm").read_bytes())
    data[-100] ^= 1
    (tmp_path / "tiny.dllm").write_bytes(bytes(data))
    with pytest.raises(ModelFileError, match="fingerprint"):
        ModelFile(tmp_path / "tiny.dllm")
    ModelFile(tmp_path / "tiny.dllm", verify=False)


# --- refusals ----------------------------------------------------------------------------------------------------


def test_licence_must_be_permissive_or_accepted(tmp_path):
    write_hf_checkpoint(tmp_path / "tiny", card="---\nlicense: llama3.2\n---\n")
    with pytest.raises(ModelImportError, match="accept-licence"):
        import_model(tmp_path / "tiny", tmp_path / "out.dllm")
    (tmp_path / "tiny" / "LICENSE").write_text("Custom terms", encoding="utf-8")
    result = import_model(tmp_path / "tiny", tmp_path / "out.dllm", accept_licence=True)
    assert result.licence == {
        "spdx": "llama3.2",
        "text": "Custom terms",
        "attribution": result.licence["attribution"],
        "redistributable": False,
    }


def test_missing_licence_is_refused(tmp_path):
    write_hf_checkpoint(tmp_path / "tiny", card="")
    with pytest.raises(ModelImportError, match="--licence"):
        import_model(tmp_path / "tiny", tmp_path / "out.dllm")
    assert import_model(tmp_path / "tiny", tmp_path / "out.dllm", licence="mit").licence["spdx"] == "MIT"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model_type": "gpt2"}, "not supported"),
        ({"hidden_act": "gelu"}, "activation"),
        ({"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}, "yarn"),
        ({"mlp_bias": True}, "MLP biases"),
    ],
)
def test_unsupported_architectures_fail_loudly(tmp_path, change, message):
    write_hf_checkpoint(tmp_path / "tiny", {**TINY_LLAMA_CONFIG, **change})
    with pytest.raises(ModelImportError, match=message):
        import_model(tmp_path / "tiny", tmp_path / "out.dllm")


def test_unexpected_tensors_fail_loudly(tmp_path):
    weights = write_hf_checkpoint(tmp_path / "tiny")
    tensors = {name: ("F32", values) for name, values in weights.items()}
    write_safetensors(
        tmp_path / "tiny" / "model.safetensors", {**tensors, "model.layers.0.mystery": ("F32", np.zeros(2, np.float32))}
    )
    with pytest.raises(ModelImportError, match="mystery"):
        import_model(tmp_path / "tiny", tmp_path / "out.dllm")


# --- Hugging Face download and CLI -------------------------------------------------------------------------------


def test_hub_download_pins_the_commit(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    write_hf_checkpoint(checkpoint)
    commit = "0123456789abcdef0123456789abcdef01234567"
    files = [*sorted(p.name for p in checkpoint.iterdir()), "pytorch_model.bin", "onnx/model.onnx"]
    requested = []

    def opener(url: str):
        requested.append(url)
        if "/api/models/" in url:
            return io.BytesIO(json.dumps({"sha": commit, "siblings": [{"rfilename": f} for f in files]}).encode())
        return io.BytesIO((checkpoint / url.rsplit("/", 1)[1]).read_bytes())

    assert parse_reference("hf:org/tiny") == ("org/tiny", "main")
    result = import_model("hf:org/tiny@v1", tmp_path / "out.dllm", cache=tmp_path / "cache", opener=opener)
    assert requested[0] == "https://huggingface.co/api/models/org/tiny/revision/v1"
    assert all(f"/resolve/{commit}/" in url for url in requested[1:])
    assert not any(url.endswith((".bin", ".onnx")) for url in requested)
    assert result.source["repository"] == "org/tiny" and result.source["revision"] == commit
    assert result.fingerprint == TINY_IMPORT_FINGERPRINT
    requested.clear()
    download("org/tiny", "v1", tmp_path / "cache", opener)
    assert len(requested) == 1  # cached files are not downloaded again


def test_cli_import_and_inspect(tmp_path, capsys):
    write_hf_checkpoint(tmp_path / "tiny")
    assert cli(["import", str(tmp_path / "tiny"), "-o", str(tmp_path / "tiny.dllm"), "--repo", "example/tiny"]) == 0
    assert TINY_IMPORT_FINGERPRINT in capsys.readouterr().out
    assert cli(["inspect", str(tmp_path / "tiny.dllm")]) == 0
    output = capsys.readouterr().out
    assert "Apache-2.0" in output and "example/tiny" in output and TINY_IMPORT_FINGERPRINT in output
    assert cli(["import", str(tmp_path / "missing"), "-o", str(tmp_path / "x.dllm")]) == 1
