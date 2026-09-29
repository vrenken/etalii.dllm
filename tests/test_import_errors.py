"""Error paths of the importers: malformed or truncated GGUF and safetensors files, configs the decoder does not
implement, licence checks, the Hub client and adapter imports. Every refusal is checked for its type and message.

Valid files come from the reference ``gguf``/``safetensors`` packages, the fixtures in ``model_fixtures`` or the
small GGUF builder below (which the reference reader is used to validate), and are then corrupted."""

from __future__ import annotations

import io
import json
import struct
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from golden_values import TINY_IMPORT_FINGERPRINT
from model_fixtures import (
    TINY_LLAMA_CONFIG,
    gguf_name,
    hf_weights,
    llama_cpp_permute,
    tiny_config,
    to_bf16_bits,
    write_gguf,
    write_hf_checkpoint,
    write_safetensors,
)

from etalii_dllm.cli import main as cli
from etalii_dllm.importing import ModelImportError, hub, import_model
from etalii_dllm.importing.gguf import GgufError, GgufFile
from etalii_dllm.importing.quants import BLOCK_FORMATS, dequantize
from etalii_dllm.importing.safetensors import SafetensorsError, SafetensorsFile, open_checkpoint
from etalii_dllm.lora import LoraConfig, init_adapters, write_peft
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.numerics import fill_gaussian


def bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype="<f4").view("<u4")


# --- a minimal GGUF writer with full control over every field ----------------------------------------------------

_U32, _F32, _BOOL, _STRING, _ARRAY = 4, 6, 7, 8, 9
F32, F16, Q8_0, I32, BF16 = 0, 1, 8, 26, 30


class Raw:
    """A metadata value written as given: its type id and payload bytes."""

    def __init__(self, type_id: int, payload: bytes) -> None:
        self.type_id, self.payload = type_id, payload


def _string(value: str | bytes) -> bytes:
    data = value.encode() if isinstance(value, str) else value
    return struct.pack("<Q", len(data)) + data


def _value(value: Any) -> tuple[int, bytes]:
    if isinstance(value, Raw):
        return value.type_id, value.payload
    if isinstance(value, bool):
        return _BOOL, struct.pack("<?", value)
    if isinstance(value, int):
        return _U32, struct.pack("<I", value)
    if isinstance(value, float):
        return _F32, struct.pack("<f", value)
    if isinstance(value, str):
        return _STRING, _string(value)
    if isinstance(value, list):
        item_type = _value(value[0])[0]
        return _ARRAY, struct.pack("<IQ", item_type, len(value)) + b"".join(_value(v)[1] for v in value)
    raise TypeError(value)


def build_gguf(
    metadata: list[tuple[str | bytes, Any]],
    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = (),
    *,
    version: bytes = struct.pack("<I", 3),
    magic: bytes = b"GGUF",
    offsets: dict[str, int] | None = None,
) -> bytes:
    """GGUF bytes. ``tensors`` are (name, NumPy shape, GGML type, data); data is laid out at aligned offsets unless
    ``offsets`` overrides one."""
    alignment = next((v for k, v in metadata if k == "general.alignment" and isinstance(v, int)), 32)
    out = magic + version + struct.pack("<QQ", len(tensors), len(metadata))
    for key, value in metadata:
        type_id, payload = _value(value)
        out += _string(key) + struct.pack("<I", type_id) + payload
    data = b""
    for name, shape, type_id, raw in tensors:
        data += b"\0" * (-len(data) % alignment)
        offset = (offsets or {}).get(name, len(data))
        out += _string(name) + struct.pack("<I", len(shape))
        out += b"".join(struct.pack("<Q", d) for d in reversed(shape)) + struct.pack("<IQ", type_id, offset)
        data += raw
    return out + b"\0" * (-len(out) % max(alignment, 1)) + data


def tiny_gguf_metadata(config: dict | None = None) -> list[tuple[str, Any]]:
    config = config or TINY_LLAMA_CONFIG
    family = config["model_type"]
    head = config["hidden_size"] // config["num_attention_heads"]
    return [
        ("general.architecture", family),
        ("general.name", "Tiny test model"),
        ("general.license", "apache-2.0"),
        (f"{family}.context_length", config["max_position_embeddings"]),
        (f"{family}.embedding_length", config["hidden_size"]),
        (f"{family}.block_count", config["num_hidden_layers"]),
        (f"{family}.feed_forward_length", config["intermediate_size"]),
        (f"{family}.attention.head_count", config["num_attention_heads"]),
        (f"{family}.attention.head_count_kv", config["num_key_value_heads"]),
        (f"{family}.rope.dimension_count", head),
        (f"{family}.attention.key_length", head),
        (f"{family}.rope.freq_base", float(config["rope_theta"])),
        (f"{family}.attention.layer_norm_rms_epsilon", float(config["rms_norm_eps"])),
        (f"{family}.vocab_size", config["vocab_size"]),
        ("tokenizer.ggml.model", "gpt2"),
        ("tokenizer.ggml.tokens", [f"t{i}" for i in range(config["vocab_size"])]),
        ("tokenizer.ggml.bos_token_id", config["bos_token_id"]),
        ("tokenizer.ggml.eos_token_id", config["eos_token_id"]),
    ]


def tiny_gguf_tensors(config: dict | None = None) -> list[tuple[str, tuple[int, ...], int, bytes]]:
    """The fixture weights as F32 GGUF tensors, with Q/K permuted the way llama.cpp does for Llama."""
    config = config or TINY_LLAMA_CONFIG
    tensors = []
    for name, values in hf_weights(config).items():
        if ".q_proj." in name:
            values = llama_cpp_permute(values, config["num_attention_heads"])
        elif ".k_proj." in name:
            values = llama_cpp_permute(values, config["num_key_value_heads"])
        tensors.append((gguf_name(name), values.shape, F32, np.ascontiguousarray(values, "<f4").tobytes()))
    return tensors


def write_tiny_gguf(
    path: Path,
    *,
    drop: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
    extra_tensors: list[tuple[str, tuple[int, ...], int, bytes]] = (),
) -> Path:
    entries = [(k, v) for k, v in tiny_gguf_metadata() if k not in drop and k not in (metadata or {})]
    path.write_bytes(build_gguf(entries + list((metadata or {}).items()), tiny_gguf_tensors() + list(extra_tensors)))
    return path


def f32(values: list[float]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


# --- GGUF reader ---------------------------------------------------------------------------------------------------


def test_gguf_builder_agrees_with_the_reference_reader(tmp_path):
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "t.gguf"
    path.write_bytes(build_gguf([("a.list", [1, 2, 3]), ("a.name", "x")], [("w", (2, 3), F32, f32(list(range(6))))]))
    reader = gguf.GGUFReader(str(path))
    assert reader.fields["a.list"].contents() == [1, 2, 3]
    assert reader.fields["a.name"].contents() == "x"
    assert np.array_equal(reader.tensors[0].data, np.arange(6, dtype=np.float32).reshape(2, 3))


def test_gguf_reads_scalar_arrays_and_every_plain_float_type(tmp_path):
    values = fill_gaussian(4, 6)
    brain = to_bf16_bits(values)
    tensors = [
        ("f32", (2, 3), F32, values.tobytes()),
        ("f16", (2, 3), F16, values.astype("<f2").tobytes()),
        ("bf16", (6,), BF16, brain.tobytes()),
        ("ids", (3,), I32, np.arange(3, dtype="<i4").tobytes()),
    ]
    metadata = [("ints", [7, 8, 9]), ("floats", [0.5, 1.5]), ("flags", [True, False]), ("general.alignment", 64)]
    (tmp_path / "t.gguf").write_bytes(build_gguf(metadata, tensors))
    file = GgufFile(tmp_path / "t.gguf")
    assert file.metadata["ints"] == [7, 8, 9] and file.metadata["floats"] == [0.5, 1.5]
    assert file.metadata["flags"] == [True, False] and file.alignment == 64
    assert len(file) == 4 and "f16" in file and "missing" not in file
    assert [t.name for t in file] == ["f32", "f16", "bf16", "ids"]
    assert np.array_equal(bits(file["f32"].to_float32()), bits(values.reshape(2, 3)))
    assert np.array_equal(bits(file["f16"].to_float32()), bits(values.astype("<f2").astype("<f4").reshape(2, 3)))
    assert np.array_equal(file["bf16"].to_float32().view("<u4") >> 16, brain.astype("<u4"))
    assert file["bf16"].lossless and not file["ids"].lossless
    with pytest.raises(GgufError, match="cannot convert I32 to float32"):
        file["ids"].to_float32()


def _valid_header() -> list[tuple[str, Any]]:
    return [("general.architecture", "llama")]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (build_gguf(_valid_header(), magic=b"GGML"), "not a GGUF file"),
        (build_gguf(_valid_header(), version=struct.pack("<I", 1)), "unsupported GGUF version 1"),
        (build_gguf(_valid_header(), version=struct.pack(">I", 3)), "big-endian"),
        (build_gguf([(b"bad\xff\xfekey", 1)]), "not valid UTF-8"),
        (build_gguf([("key", Raw(99, b""))]), "unknown metadata value type 99"),
        (build_gguf([("key", 1), ("key", 2)]), "duplicate metadata key 'key'"),
        (build_gguf([("general.alignment", 48)]), "48 is not a power of two"),
        (build_gguf([("general.alignment", 0)]), "0 is not a power of two"),
        (build_gguf([], [("q", (16,), Q8_0, bytes(34))]), "not a whole number of Q8_0 blocks"),
        (build_gguf([], [("w", (2,), 99, bytes(8))]), "unsupported GGML type 99"),
        (build_gguf([], [("w", (2,), F32, f32([1, 2]))], offsets={"w": 4}), "'w' has an invalid offset"),
        (build_gguf([], [("w", (2,), F32, f32([1, 2]))], offsets={"w": 64}), "'w' has an invalid offset"),
        (build_gguf([], [("w", (1,), F32, f32([1])), ("w", (1,), F32, f32([2]))]), "duplicate tensor 'w'"),
    ],
    ids=[
        "magic",
        "version",
        "big-endian",
        "utf-8",
        "value-type",
        "duplicate-key",
        "alignment-48",
        "alignment-0",
        "partial-block",
        "ggml-type",
        "misaligned-offset",
        "offset-past-end",
        "duplicate-tensor",
    ],
)
def test_malformed_gguf_is_refused(tmp_path, data, message):
    (tmp_path / "bad.gguf").write_bytes(data)
    with pytest.raises(GgufError, match=message):
        GgufFile(tmp_path / "bad.gguf")


@pytest.mark.parametrize("keep", [3, 12, 40, 200, 2000])
def test_truncated_reference_gguf_header_is_refused(tmp_path, keep):
    """A file written by gguf-py, cut off inside its header (magic, counts, metadata or tensor infos)."""
    pytest.importorskip("gguf")
    write_gguf(tmp_path / "tiny.gguf")
    data = (tmp_path / "tiny.gguf").read_bytes()
    (tmp_path / "cut.gguf").write_bytes(data[:keep])
    with pytest.raises(GgufError, match=r"unexpected end of file|not a GGUF file"):
        GgufFile(tmp_path / "cut.gguf")


def test_truncated_reference_gguf_data_is_refused(tmp_path):
    pytest.importorskip("gguf")
    write_gguf(tmp_path / "tiny.gguf")
    data = (tmp_path / "tiny.gguf").read_bytes()
    (tmp_path / "cut.gguf").write_bytes(data[:-64])
    with pytest.raises(GgufError, match="invalid offset"):
        GgufFile(tmp_path / "cut.gguf")


# --- safetensors reader --------------------------------------------------------------------------------------------


def _reference_file(tmp_path: Path) -> Path:
    numpy_api = pytest.importorskip("safetensors.numpy")
    path = tmp_path / "ref.safetensors"
    numpy_api.save_file({"a": fill_gaussian(5, 12).reshape(3, 4)}, str(path), metadata={"format": "np"})
    return path


def _with_header(path: Path, header: bytes | dict | list, data: bytes = bytes(48)) -> Path:
    encoded = header if isinstance(header, bytes) else json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    return path


def test_safetensors_container_protocol(tmp_path):
    file = SafetensorsFile(_reference_file(tmp_path))
    assert len(file) == 1 and "a" in file and "b" not in file
    assert [t.name for t in file] == ["a"]


def test_truncated_safetensors_is_refused(tmp_path):
    data = _reference_file(tmp_path).read_bytes()
    (tmp_path / "short.safetensors").write_bytes(data[:5])
    with pytest.raises(SafetensorsError, match="too short"):
        SafetensorsFile(tmp_path / "short.safetensors")
    (tmp_path / "cut.safetensors").write_bytes(data[:40])
    with pytest.raises(SafetensorsError, match="invalid header length"):
        SafetensorsFile(tmp_path / "cut.safetensors")
    (tmp_path / "data.safetensors").write_bytes(data[:-4])
    with pytest.raises(SafetensorsError, match="'a' has invalid data offsets"):
        SafetensorsFile(tmp_path / "data.safetensors")


def test_absurd_safetensors_header_length_is_refused(tmp_path):
    (tmp_path / "huge.safetensors").write_bytes(struct.pack("<Q", 2**40) + b"{}")
    with pytest.raises(SafetensorsError, match="invalid header length 1099511627776"):
        SafetensorsFile(tmp_path / "huge.safetensors")


def _entry(**changes: Any) -> dict:
    return {"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24], **changes}}


@pytest.mark.parametrize(
    ("header", "message"),
    [
        (b"\xff\xfe{}", "header is not valid JSON"),
        (b'{"w": ', "header is not valid JSON"),
        ([1, 2], "header is not a JSON object"),
        ({"__metadata__": {"format": 1}}, "__metadata__ must map strings to strings"),
        ({"__metadata__": ["format"]}, "__metadata__ must map strings to strings"),
        (_entry(dtype="C64"), "'w' has unsupported dtype 'C64'"),
        (_entry(dtype=None), "'w' has unsupported dtype None"),
        (_entry(shape=[2, -3]), r"'w' has invalid shape \(2, -3\)"),
        (_entry(shape=[2.0, 3]), r"'w' has invalid shape"),
        (_entry(data_offsets=[0, 20]), "'w' has invalid data offsets"),
        (_entry(data_offsets=[0, "24"]), "'w' has invalid data offsets"),
        (_entry(data_offsets=[40, 64]), "'w' has invalid data offsets"),
        (_entry(data_offsets=[-8, 16]), "'w' has invalid data offsets"),
        ({"w": {"dtype": "F32", "shape": [2, 3]}}, "'w' has invalid data offsets"),
    ],
)
def test_malformed_safetensors_header_is_refused(tmp_path, header, message):
    with pytest.raises(SafetensorsError, match=message):
        SafetensorsFile(_with_header(tmp_path / "bad.safetensors", header))


def test_safetensors_tensor_entry_must_be_an_object(tmp_path):
    with pytest.raises(SafetensorsError):
        SafetensorsFile(_with_header(tmp_path / "bad.safetensors", {"w": 5}))


def test_sharded_checkpoint_imports_like_a_single_file(tmp_path):
    weights = write_hf_checkpoint(tmp_path / "tiny")
    (tmp_path / "tiny" / "model.safetensors").unlink()
    names = sorted(weights)
    shards = {"model-00002-of-00002.safetensors": names[::2], "model-00001-of-00002.safetensors": names[1::2]}
    for shard, members in shards.items():
        write_safetensors(tmp_path / "tiny" / shard, {n: ("BF16", to_bf16_bits(weights[n])) for n in members})
    index = {"metadata": {}, "weight_map": {n: s for s, members in shards.items() for n in members}}
    (tmp_path / "tiny" / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    assert sorted(open_checkpoint(tmp_path / "tiny")) == names
    assert import_model(tmp_path / "tiny", tmp_path / "out.dllm").fingerprint == TINY_IMPORT_FINGERPRINT


def test_checkpoint_without_index_reads_every_safetensors_file(tmp_path):
    write_safetensors(tmp_path / "b.safetensors", {"y": ("F32", np.ones(2, np.float32))})
    write_safetensors(tmp_path / "a.safetensors", {"x": ("F32", np.zeros(2, np.float32))})
    assert sorted(open_checkpoint(tmp_path)) == ["x", "y"]


def test_checkpoint_without_safetensors_is_refused(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(SafetensorsError, match="no safetensors files found"):
        open_checkpoint(tmp_path)


def test_tensor_in_two_shards_is_refused(tmp_path):
    write_safetensors(tmp_path / "a.safetensors", {"x": ("F32", np.zeros(2, np.float32))})
    write_safetensors(tmp_path / "b.safetensors", {"x": ("F32", np.ones(2, np.float32))})
    with pytest.raises(SafetensorsError, match="'x' appears in more than one shard"):
        open_checkpoint(tmp_path)


# --- Hugging Face Hub client (no network) --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("org/tiny", "not a Hugging Face reference"),
        ("https://huggingface.co/org/tiny", "not a Hugging Face reference"),
        ("hf:tiny", "invalid repository name 'tiny'"),
        ("hf:org/tiny/extra", "invalid repository name"),
        ("hf:../tiny", "invalid repository name"),
        ("hf:org/-tiny", "invalid repository name"),
    ],
)
def test_bad_hub_references_are_refused(reference, message):
    with pytest.raises(ValueError, match=message):
        hub.parse_reference(reference)


def test_hub_reference_with_revision():
    assert hub.parse_reference("hf:Org.x/Tiny_1-b@v1.0") == ("Org.x/Tiny_1-b", "v1.0")


@pytest.mark.parametrize("sha", [None, "main", "0123456789ABCDEF0123456789ABCDEF01234567", "0" * 39, 12])
def test_hub_must_return_a_commit_hash(tmp_path, sha):
    info = {"siblings": [{"rfilename": "config.json"}], **({} if sha is None else {"sha": sha})}

    def opener(url: str):
        return io.BytesIO(json.dumps(info).encode())

    with pytest.raises(ValueError, match="org/tiny@main: the Hub did not return a commit hash"):
        hub.download("org/tiny", "main", tmp_path, opener)
    assert not any(tmp_path.iterdir())


class _Response(io.BytesIO):
    pass


@pytest.mark.parametrize("token", [None, "secret-token"])
def test_default_opener_sends_the_token_only_when_set(tmp_path, monkeypatch, token):
    commit = "0123456789abcdef0123456789abcdef01234567"
    seen: list[tuple[str, str | None, float]] = []

    def urlopen(request: urllib.request.Request, timeout: float):
        seen.append((request.full_url, request.get_header("Authorization"), timeout))
        if "/api/models/" in request.full_url:
            return _Response(json.dumps({"sha": commit, "siblings": [{"rfilename": "config.json"}]}).encode())
        return _Response(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    if token is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", token)
    snapshot = hub.download("org/tiny", "refs/pr/1", tmp_path)
    assert seen[0][0] == "https://huggingface.co/api/models/org/tiny/revision/refs%2Fpr%2F1"
    assert seen[1][0] == f"https://huggingface.co/org/tiny/resolve/{commit}/config.json"
    expected = None if token is None else f"Bearer {token}"
    assert all(header == expected and timeout == 60 for _, header, timeout in seen)
    assert snapshot.files == ("config.json",) and (snapshot.directory / "config.json").read_bytes() == b"{}"


# --- Hugging Face checkpoints: configs, tensors, templates, licences -----------------------------------------------


@pytest.mark.parametrize(
    ("change", "theta", "scaling"),
    [
        ({"rope_parameters": {"rope_theta": 5e5, "rope_type": "default"}}, 5e5, None),
        ({"rope_parameters": {"rope_type": "linear", "factor": 2.0}}, 1e5, {"rope_type": "linear", "factor": 2.0}),
        ({"rope_scaling": {"type": "linear", "factor": 4.0}}, 1e5, {"rope_type": "linear", "factor": 4.0}),
        (
            {"rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0}},
            1e5,
            {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0},
        ),
        ({"rope_scaling": {"rope_type": "default"}}, 1e5, None),
    ],
    ids=["v5-default", "v5-linear", "legacy-type-key", "llama3", "default"],
)
def test_rope_settings_are_mapped(tmp_path, change, theta, scaling):
    write_hf_checkpoint(tmp_path / "tiny", {**TINY_LLAMA_CONFIG, **change})
    config = import_model(tmp_path / "tiny", tmp_path / "out.dllm").config
    assert config.rope_theta == theta and config.rope_scaling == scaling


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_sliding_window_attention_is_refused(tmp_path, family):
    write_hf_checkpoint(tmp_path / "qwen", {**tiny_config(family), "use_sliding_window": True})
    with pytest.raises(ModelImportError, match="sliding-window attention is not supported"):
        import_model(tmp_path / "qwen", tmp_path / "out.dllm")


def test_checkpoint_without_config_is_refused(tmp_path):
    write_hf_checkpoint(tmp_path / "tiny")
    (tmp_path / "tiny" / "config.json").unlink()
    with pytest.raises(ModelImportError, match=r"no config\.json"):
        import_model(tmp_path / "tiny", tmp_path / "out.dllm")


def test_source_that_is_neither_a_directory_nor_gguf_is_refused(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"x")
    with pytest.raises(ModelImportError, match=r"expected a checkpoint directory, a \.gguf file"):
        import_model(tmp_path / "model.bin", tmp_path / "out.dllm")


def _rewrite_weights(directory: Path, weights: dict[str, np.ndarray], extra: dict[str, tuple[str, np.ndarray]]):
    tensors = {name: ("BF16", to_bf16_bits(values)) for name, values in weights.items()}
    write_safetensors(directory / "model.safetensors", {**tensors, **extra})


def test_rotary_buffers_and_an_identical_tied_head_are_skipped(tmp_path):
    weights = write_hf_checkpoint(tmp_path / "tiny")
    embedding = to_bf16_bits(weights["model.embed_tokens.weight"])
    extra = {
        "model.layers.0.self_attn.rotary_emb.inv_freq": ("F32", np.ones(2, np.float32)),
        "lm_head.weight": ("BF16", embedding),
    }
    _rewrite_weights(tmp_path / "tiny", weights, extra)
    result = import_model(tmp_path / "tiny", tmp_path / "out.dllm")
    assert result.fingerprint == TINY_IMPORT_FINGERPRINT


def test_tied_head_that_differs_from_the_embedding_is_refused(tmp_path):
    weights = write_hf_checkpoint(tmp_path / "tiny")
    head = to_bf16_bits(weights["model.embed_tokens.weight"]).copy()
    head[0, 0] ^= 1
    _rewrite_weights(tmp_path / "tiny", weights, {"lm_head.weight": ("BF16", head)})
    with pytest.raises(ModelImportError, match="lm_head differs from the embedding"):
        import_model(tmp_path / "tiny", tmp_path / "out.dllm")


@pytest.mark.parametrize(("dtype", "cast"), [("F64", "<f8"), ("I32", "<i4")])
def test_weights_that_cannot_widen_exactly_are_refused(tmp_path, dtype, cast):
    weights = write_hf_checkpoint(tmp_path / "tiny")
    norm = weights.pop("model.norm.weight")
    _rewrite_weights(tmp_path / "tiny", weights, {"model.norm.weight": (dtype, norm.astype(cast))})
    with pytest.raises(ModelImportError, match=f"'model.norm.weight' has dtype {dtype}; only F32, F16 and BF16"):
        import_model(tmp_path / "tiny", tmp_path / "out.dllm")


def test_chat_template_file_wins_over_tokenizer_config(tmp_path):
    write_hf_checkpoint(tmp_path / "tiny")
    (tmp_path / "tiny" / "chat_template.jinja").write_text("{{ messages[0]['content'] }}", encoding="utf-8")
    import_model(tmp_path / "tiny", tmp_path / "out.dllm")
    model = ModelFile(tmp_path / "out.dllm")
    assert model.chat_template == "{{ messages[0]['content'] }}"
    assert "chat_template" not in model.tokenizer["tokenizer_config"]


@pytest.mark.parametrize(
    ("templates", "expected"),
    [
        ([{"name": "tool_use", "template": "tools"}, {"name": "default", "template": "plain"}], "plain"),
        ([{"name": "tool_use", "template": "tools"}], None),
    ],
)
def test_named_chat_templates_keep_the_default(tmp_path, templates, expected):
    write_hf_checkpoint(tmp_path / "tiny")
    (tmp_path / "tiny" / "tokenizer_config.json").write_text(json.dumps({"chat_template": templates}), "utf-8")
    import_model(tmp_path / "tiny", tmp_path / "out.dllm")
    assert ModelFile(tmp_path / "out.dllm").chat_template == expected


@pytest.mark.parametrize("card", ["# No front matter\n\nlicense: apache-2.0\n", "---\nlicense:\n---\n"])
def test_model_card_without_a_licence_is_refused(tmp_path, card):
    write_hf_checkpoint(tmp_path / "tiny", card=card)
    with pytest.raises(ModelImportError, match="does not state a licence"):
        import_model(tmp_path / "tiny", tmp_path / "out.dllm")


def test_other_licence_needs_its_text_and_keeps_its_link(tmp_path):
    card = "---\nlicense: other\nlicense_name: tiny-community\nlicense_link: 'https://example.com/terms'\n---\n"
    write_hf_checkpoint(tmp_path / "tiny", card=card)
    with pytest.raises(ModelImportError, match=r"'tiny-community' is not Apache-2\.0 or MIT"):
        import_model(tmp_path / "tiny", tmp_path / "out.dllm")
    with pytest.raises(ModelImportError, match="no text found for licence 'tiny-community'; pass --licence-file"):
        import_model(tmp_path / "tiny", tmp_path / "out.dllm", accept_licence=True)
    (tmp_path / "terms.txt").write_text("Community terms", encoding="utf-8")
    result = import_model(
        tmp_path / "tiny", tmp_path / "out.dllm", accept_licence=True, licence_file=tmp_path / "terms.txt"
    )
    assert result.licence["spdx"] == "tiny-community" and result.licence["text"] == "Community terms"
    assert result.licence["link"] == "https://example.com/terms" and result.licence["redistributable"] is False


def test_licence_override_is_canonicalised(tmp_path):
    write_hf_checkpoint(tmp_path / "tiny", card="---\nlicense: llama3.2\n---\n")
    result = import_model(tmp_path / "tiny", tmp_path / "out.dllm", licence="MIT")
    assert result.licence["spdx"] == "MIT" and result.licence["redistributable"] is True
    assert "MIT License" in result.licence["text"]


def test_unclosed_front_matter_is_still_read(tmp_path):
    write_hf_checkpoint(tmp_path / "tiny", card="---\nlicense: mit\nlibrary_name: transformers\n")
    assert import_model(tmp_path / "tiny", tmp_path / "out.dllm").licence["spdx"] == "MIT"


def test_checkpoint_without_tokenizer_imports_without_one(tmp_path):
    write_hf_checkpoint(tmp_path / "tiny")
    (tmp_path / "tiny" / "tokenizer.json").unlink()
    import_model(tmp_path / "tiny", tmp_path / "out.dllm")
    assert ModelFile(tmp_path / "out.dllm").tokenizer is None


def test_tensor_outside_the_layers_is_refused(tmp_path):
    weights = write_hf_checkpoint(tmp_path / "tiny")
    _rewrite_weights(tmp_path / "tiny", weights, {"model.mystery.weight": ("F32", np.zeros(2, np.float32))})
    with pytest.raises(ModelImportError, match=r"unexpected tensor 'model\.mystery\.weight'"):
        import_model(tmp_path / "tiny", tmp_path / "out.dllm")


def test_dequantising_a_partial_block_is_refused():
    with pytest.raises(ValueError, match="Q8_0: 16 values is not a whole number of blocks"):
        dequantize(np.zeros(34, np.uint8), BLOCK_FORMATS[Q8_0], 16)


# --- GGUF import ---------------------------------------------------------------------------------------------------


def test_built_gguf_imports_like_the_checkpoint(tmp_path):
    """Sanity check of the builder the tests below corrupt: it gives the same model as the reference fixture."""
    result = import_model(write_tiny_gguf(tmp_path / "tiny.gguf"), tmp_path / "out.dllm")
    assert result.fingerprint == TINY_IMPORT_FINGERPRINT
    assert result.config.vocabulary_size == 64 and result.config.eos_token_ids == (2,)
    assert "repository" not in result.source and "url" not in result.source


@pytest.mark.parametrize(
    ("drop", "metadata", "message"),
    [
        ((), {"general.architecture": "gpt2"}, "GGUF architecture 'gpt2' is not supported"),
        (("general.architecture",), {}, "GGUF architecture None is not supported"),
        (("llama.embedding_length",), {}, "GGUF metadata llama.embedding_length is missing"),
        (("llama.feed_forward_length",), {}, "GGUF metadata llama.feed_forward_length is missing"),
        (("llama.attention.layer_norm_rms_epsilon",), {}, "llama.attention.layer_norm_rms_epsilon is missing"),
        ((), {"llama.rope.dimension_count": 2}, "partial rotary embeddings are not supported"),
        ((), {"llama.rope.scaling.type": "yarn"}, "GGUF RoPE scaling 'yarn' is not supported"),
        ((), {"llama.rope.scaling.type": "linear"}, "GGUF metadata llama.rope.scaling.factor is missing"),
    ],
)
def test_unsupported_gguf_metadata_fails_loudly(tmp_path, drop, metadata, message):
    write_tiny_gguf(tmp_path / "tiny.gguf", drop=drop, metadata=metadata)
    with pytest.raises(ModelImportError, match=message):
        import_model(tmp_path / "tiny.gguf", tmp_path / "out.dllm")


def test_gguf_metadata_defaults_and_optional_keys(tmp_path):
    metadata = {
        "llama.rope.scaling.type": "linear",
        "llama.rope.scaling.factor": 2.0,
        "tokenizer.ggml.eot_token_id": 7,
        "general.source.huggingface.repository": "example/tiny",
        "general.url": "https://example.com/tiny",
        "general.license": "other",
        "general.license.name": "mit",
        "general.license.link": "https://example.com/licence",
    }
    path = write_tiny_gguf(tmp_path / "tiny.gguf", drop=("llama.vocab_size",), metadata=metadata)
    result = import_model(path, tmp_path / "out.dllm")
    assert result.config.vocabulary_size == 64  # from the embedding's rows
    assert result.config.rope_scaling == {"rope_type": "linear", "factor": 2.0}
    assert result.config.eos_token_ids == (2, 7)
    assert result.source["repository"] == "example/tiny" and result.source["url"] == "https://example.com/tiny"
    assert result.licence["spdx"] == "MIT" and result.licence["link"] == "https://example.com/licence"
    assert "(https://huggingface.co/example/tiny)" in result.licence["attribution"]


def test_gguf_end_of_turn_token_equal_to_eos_is_not_repeated(tmp_path):
    path = write_tiny_gguf(tmp_path / "tiny.gguf", metadata={"tokenizer.ggml.eot_token_id": 2})
    assert import_model(path, tmp_path / "out.dllm").config.eos_token_ids == (2,)


def test_gguf_without_vocabulary_size_or_embedding_is_refused(tmp_path):
    entries = [(k, v) for k, v in tiny_gguf_metadata() if k != "llama.vocab_size"]
    tensors = [t for t in tiny_gguf_tensors() if t[0] != "token_embd.weight"]
    (tmp_path / "tiny.gguf").write_bytes(build_gguf(entries, tensors))
    with pytest.raises(ModelImportError, match=r"GGUF metadata llama\.vocab_size is missing"):
        import_model(tmp_path / "tiny.gguf", tmp_path / "out.dllm")


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("rope_freqs.weight", "rope_freqs \\(Llama 3 RoPE scaling\\) is not supported"),
        ("blk.0.attn_mystery.weight", "unexpected tensor 'blk.0.attn_mystery.weight'"),
        ("mystery", "unexpected tensor 'mystery'"),
    ],
)
def test_unexpected_gguf_tensors_fail_loudly(tmp_path, name, message):
    path = write_tiny_gguf(tmp_path / "tiny.gguf", extra_tensors=[(name, (2,), F32, f32([1, 2]))])
    with pytest.raises(ModelImportError, match=message):
        import_model(path, tmp_path / "out.dllm")


def test_gguf_with_a_non_permissive_licence_needs_a_licence_file(tmp_path):
    path = write_tiny_gguf(tmp_path / "tiny.gguf", metadata={"general.license": "llama3.2"})
    with pytest.raises(ModelImportError, match=r"no text found for licence 'llama3\.2'"):
        import_model(path, tmp_path / "out.dllm", accept_licence=True)


def test_cli_import_of_a_corrupt_gguf_reports_the_error(tmp_path):
    (tmp_path / "bad.gguf").write_bytes(b"GGUF" + struct.pack("<I", 7))
    assert cli(["import", str(tmp_path / "bad.gguf"), "-o", str(tmp_path / "out.dllm")]) == 1


def test_cli_import_of_a_corrupt_checkpoint_reports_the_error(tmp_path):
    write_hf_checkpoint(tmp_path / "tiny")
    (tmp_path / "tiny" / "model.safetensors").write_bytes(b"\x01")
    assert cli(["import", str(tmp_path / "tiny"), "-o", str(tmp_path / "out.dllm")]) == 1


# --- LoRA adapter import -------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_model(tmp_path_factory) -> ModelFile:
    directory = tmp_path_factory.mktemp("import-errors-base")
    write_hf_checkpoint(directory / "checkpoint")
    import_model(directory / "checkpoint", directory / "base.dllm", repository="example/base")
    return ModelFile(directory / "base.dllm")


def test_adapter_without_weights_is_refused(base_model, tmp_path):
    lora = LoraConfig(rank=2, alpha=4.0)
    write_peft(tmp_path / "adapter", init_adapters(base_model.config, lora, 1), lora)
    (tmp_path / "adapter" / "adapter_model.safetensors").unlink()
    with pytest.raises(ModelImportError, match=r"no adapter_model\.safetensors"):
        import_model(tmp_path / "adapter", tmp_path / "out.dllm", base=base_model.path, licence="mit")


def test_adapter_import_records_repository_and_revision(base_model, tmp_path):
    lora = LoraConfig(rank=2, alpha=4.0)
    write_peft(tmp_path / "adapter", init_adapters(base_model.config, lora, 1), lora)
    result = import_model(
        tmp_path / "adapter",
        tmp_path / "out.dllm",
        base=base_model.path,
        licence="mit",
        repository="example/adapter",
        revision="abc123",
    )
    adapter = ModelFile(tmp_path / "out.dllm").adapter
    assert adapter["source"]["repository"] == "example/adapter" and adapter["source"]["revision"] == "abc123"
    assert adapter["base_fingerprint"] == base_model.fingerprint and adapter["licence"] == "MIT"
    assert "LoRA adapter example/adapter, licensed under MIT" in result.licence["attribution"]
    assert result.licence["redistributable"] is True
