"""The ``model.dllm`` container (``docs/model-format.md``) and the architecture description it stores: a round trip,
byte-for-byte reproducible writes, and every refusal when writing or reading a malformed file."""

from __future__ import annotations

import dataclasses
import json
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.modelfile import (
    ALIGNMENT,
    FORMAT_VERSION,
    MAGIC,
    ModelFile,
    ModelFileError,
    TensorSource,
    canonical_json,
    data_fingerprint,
    tensor_order,
    write_model_file,
)
from etalii_dllm.numerics import fill_gaussian

CONFIG = TransformerConfig(
    family="llama",
    vocabulary_size=8,
    hidden_size=4,
    intermediate_size=6,
    layers=2,
    heads=2,
    kv_heads=1,
    head_dim=2,
    context_length=16,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    eos_token_ids=(2,),
)
METADATA = {"source": {"format": "test"}, "licence": {"spdx": "MIT"}, "tokenizer": None, "chat_template": "{{ x }}"}
_PREFIX = struct.Struct("<4sIQ")


def weights(config: TransformerConfig = CONFIG) -> dict[str, np.ndarray]:
    return {
        name: fill_gaussian(index, int(np.prod(shape))).reshape(shape)
        for index, (name, shape) in enumerate(sorted(config.tensor_shapes().items()))
    }


def sources(values: dict[str, np.ndarray]) -> dict[str, TensorSource]:
    return {name: TensorSource(v.shape, lambda v=v: v, "BF16") for name, v in values.items()}


@pytest.fixture
def model_path(tmp_path) -> Path:
    write_model_file(tmp_path / "model.dllm", CONFIG, sources(weights()), METADATA)
    return tmp_path / "model.dllm"


def rewrite(path: Path, change: Callable[[dict], None] | None = None, *, header: bytes | None = None, **prefix):
    """Rewrites the header of a model file (the data section is kept as it is)."""
    data = path.read_bytes()
    magic, version, length = _PREFIX.unpack_from(data)
    parsed = json.loads(data[_PREFIX.size : _PREFIX.size + length])
    if change:
        change(parsed)
    encoded = canonical_json(parsed) if header is None else header
    magic, version = prefix.get("magic", magic), prefix.get("version", version)
    path.write_bytes(_PREFIX.pack(magic, version, len(encoded)) + encoded + data[_PREFIX.size + length :])


def entry(header: dict, name: str) -> dict:
    return next(t for t in header["tensors"] if t["name"] == name)


# --- writing and reading -------------------------------------------------------------------------------------------


def test_round_trip(model_path):
    values = weights()
    model = ModelFile(model_path)
    assert model.config == CONFIG
    assert model.fingerprint == data_fingerprint(values)
    assert model.source == {"format": "test"} and model.licence == {"spdx": "MIT"}
    assert model.tokenizer is None and model.chat_template == "{{ x }}"
    assert model.fine_tuning is None and model.adapter is None
    assert set(model.tensors) == set(values)
    for name, tensor in model.tensors.items():
        assert tensor.dtype == np.dtype("<f4") and tensor.ctypes.data % ALIGNMENT == 0
        assert not tensor.flags.writeable
        assert np.array_equal(tensor.view("<u4"), values[name].view("<u4"))
    assert [t["name"] for t in model.header["tensors"]] == tensor_order(values)
    assert {t["source_dtype"] for t in model.header["tensors"]} == {"BF16"}


def test_writes_are_byte_for_byte_reproducible(tmp_path):
    values = weights()
    reversed_sources = dict(reversed(list(sources(values).items())))
    first = write_model_file(tmp_path / "a.dllm", CONFIG, sources(values), METADATA)
    second = write_model_file(tmp_path / "b.dllm", CONFIG, reversed_sources, dict(reversed(list(METADATA.items()))))
    assert first == second
    assert (tmp_path / "a.dllm").read_bytes() == (tmp_path / "b.dllm").read_bytes()


def test_file_layout(model_path):
    data = model_path.read_bytes()
    magic, version, length = _PREFIX.unpack_from(data)
    assert (magic, version) == (MAGIC, FORMAT_VERSION)
    assert (_PREFIX.size + length) % ALIGNMENT == 0
    header = json.loads(data[_PREFIX.size : _PREFIX.size + length])
    assert data[_PREFIX.size : _PREFIX.size + length].rstrip(b" ") == canonical_json(header)
    assert all(t["offset"] % ALIGNMENT == 0 for t in header["tensors"])


def test_optional_sections_are_written_only_when_present(tmp_path):
    metadata = {**METADATA, "fine_tuning": {"steps": 3}, "adapter": None}
    write_model_file(tmp_path / "tuned.dllm", CONFIG, sources(weights()), metadata)
    model = ModelFile(tmp_path / "tuned.dllm")
    assert model.fine_tuning == {"steps": 3} and "adapter" not in model.header


def test_tensor_order_is_natural():
    names = ["layers.10.a", "layers.2.b", "layers.2.a", "final", "layers.1.a"]
    assert tensor_order(names) == ["final", "layers.1.a", "layers.2.a", "layers.2.b", "layers.10.a"]


def test_data_fingerprint_does_not_depend_on_insertion_order():
    values = weights()
    assert data_fingerprint(values) == data_fingerprint(dict(reversed(list(values.items()))))


def test_canonical_json_refuses_nan():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


# --- refusals when writing -----------------------------------------------------------------------------------------


def test_missing_and_unexpected_tensors_are_refused(tmp_path):
    values = weights()
    del values["final_norm.weight"]
    values["layers.0.mystery"] = np.zeros(2, np.float32)
    message = r"missing \['final_norm.weight'\], unexpected \['layers.0.mystery'\]"
    with pytest.raises(ModelFileError, match=message):
        write_model_file(tmp_path / "out.dllm", CONFIG, sources(values), METADATA)
    assert not (tmp_path / "out.dllm").exists()


def test_declared_shape_must_match_the_architecture(tmp_path):
    values = weights()
    values["layers.1.mlp.up.weight"] = values["layers.1.mlp.up.weight"].T
    with pytest.raises(ModelFileError, match=r"'layers.1.mlp.up.weight' has shape \(4, 6\), expected \(6, 4\)"):
        write_model_file(tmp_path / "out.dllm", CONFIG, sources(values), METADATA)


def test_loaded_shape_must_match_the_declared_shape(tmp_path):
    tensors = sources(weights())
    tensors["final_norm.weight"] = TensorSource((4,), lambda: np.zeros(5, np.float32))
    with pytest.raises(ModelFileError, match=r"'final_norm.weight' loaded with shape \(5,\)"):
        write_model_file(tmp_path / "out.dllm", CONFIG, tensors, METADATA)


# --- refusals when reading -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("keep", [0, 5, _PREFIX.size - 1])
def test_truncated_prefix_is_refused(model_path, keep):
    model_path.write_bytes(model_path.read_bytes()[:keep])
    with pytest.raises(ModelFileError, match=r"not a model\.dllm file"):
        ModelFile(model_path)


def test_wrong_magic_is_refused(model_path):
    rewrite(model_path, magic=b"GGUF")
    with pytest.raises(ModelFileError, match=r"not a model\.dllm file"):
        ModelFile(model_path)


def test_unknown_format_version_is_refused(model_path):
    rewrite(model_path, version=FORMAT_VERSION + 1)
    with pytest.raises(ModelFileError, match=f"unsupported format version {FORMAT_VERSION + 1}"):
        ModelFile(model_path)


@pytest.mark.parametrize("header", [b"{not json", b"\xff\xfe{}"])
def test_header_that_is_not_json_is_refused(model_path, header):
    rewrite(model_path, header=header)
    with pytest.raises(ModelFileError, match="header is not valid JSON"):
        ModelFile(model_path)


def test_truncated_header_is_refused(model_path):
    data = model_path.read_bytes()
    model_path.write_bytes(data[: _PREFIX.size + 40])
    with pytest.raises(ModelFileError, match="header is not valid JSON"):
        ModelFile(model_path)


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": "F16"},
        {"offset": 4},
        {"offset": 1 << 20},
        {"nbytes": 1 << 20},
    ],
    ids=["dtype", "misaligned", "offset-past-end", "nbytes-past-end"],
)
def test_invalid_tensor_layout_is_refused(model_path, change):
    rewrite(model_path, lambda header: entry(header, "final_norm.weight").update(change))
    with pytest.raises(ModelFileError, match=r"'final_norm\.weight' has an invalid layout"):
        ModelFile(model_path, verify=False)


def test_truncated_data_is_refused(model_path):
    model_path.write_bytes(model_path.read_bytes()[:-ALIGNMENT])
    with pytest.raises(ModelFileError, match="invalid layout"):
        ModelFile(model_path, verify=False)


def test_tensor_list_must_match_the_architecture(model_path):
    def drop(header: dict) -> None:
        header["tensors"] = [t for t in header["tensors"] if t["name"] != "lm_head.weight"]

    rewrite(model_path, drop)
    with pytest.raises(ModelFileError, match="tensors do not match the architecture"):
        ModelFile(model_path, verify=False)


def test_changed_architecture_is_detected(model_path):
    rewrite(model_path, lambda header: header["architecture"].update(tie_word_embeddings=True))
    with pytest.raises(ModelFileError, match="tensors do not match the architecture"):
        ModelFile(model_path, verify=False)


def test_tampered_fingerprint_is_detected(model_path):
    rewrite(model_path, lambda header: header.update(fingerprint="f" * 64))
    with pytest.raises(ModelFileError, match="does not match the fingerprint"):
        ModelFile(model_path)
    assert ModelFile(model_path, verify=False).fingerprint == "f" * 64


def test_tensor_shapes_must_match_the_architecture(model_path):
    rewrite(model_path, lambda header: entry(header, "layers.0.mlp.up.weight").update(shape=[4, 6]))
    with pytest.raises(ModelFileError):
        ModelFile(model_path)


def test_header_without_architecture_is_refused(model_path):
    rewrite(model_path, lambda header: header.pop("architecture"))
    with pytest.raises(ModelFileError):
        ModelFile(model_path)


# --- architecture --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"family": "falcon"}, "unsupported model family 'falcon'"),
        ({"vocabulary_size": 0}, "vocabulary_size must be positive"),
        ({"layers": -1}, "layers must be positive"),
        ({"kv_heads": 0}, "kv_heads must be positive"),
        ({"heads": 3, "kv_heads": 2}, "heads must be a multiple of kv_heads"),
        ({"head_dim": 3}, "head_dim must be even"),
        ({"activation": "swish"}, "unsupported activation 'swish'"),
        ({"activation": "gelu"}, "are for encoders only"),
    ],
)
def test_invalid_architectures_are_refused(change, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(CONFIG, **change)


def test_architecture_round_trips_through_its_dictionary():
    qwen3 = dataclasses.replace(CONFIG, family="qwen3", qk_norm=True, rope_scaling={"rope_type": "linear"})
    for config in (CONFIG, qwen3):
        values = json.loads(json.dumps(config.to_dict()))
        assert ("qk_norm" in values) == config.qk_norm
        assert TransformerConfig.from_dict(values) == config


# --- memory --------------------------------------------------------------------------------------------------------


def _rss_file_kb() -> int | None:
    try:
        lines = Path("/proc/self/status").read_text().splitlines()
    except OSError:
        return None
    return next((int(line.split()[1]) for line in lines if line.startswith("RssFile:")), None)


def test_released_tensors_read_back_the_same_bits(tmp_path):
    big = dataclasses.replace(CONFIG, vocabulary_size=1 << 20)  # a 16 MB embedding: whole pages to drop
    values = weights(big)
    write_model_file(tmp_path / "big.dllm", big, sources(values), METADATA)
    model = ModelFile(tmp_path / "big.dllm")  # verify() already released the pages it read
    embedding = model.tensors["token_embedding.weight"]
    before = _rss_file_kb()
    checksum = int(embedding.view("<u4").astype(np.uint64).sum())  # touch every page
    touched = _rss_file_kb()
    model.release("token_embedding.weight")
    released = _rss_file_kb()
    if before is not None and touched is not None and released is not None:
        assert touched - released > 8 * 1024  # most of the 16 MB left the process
    assert np.array_equal(embedding.view("<u4"), values["token_embedding.weight"].view("<u4"))  # read back
    assert int(embedding.view("<u4").astype(np.uint64).sum()) == checksum
    model.release()
    for name, tensor in model.tensors.items():
        assert np.array_equal(tensor.view("<u4"), values[name].view("<u4"))


def test_release_is_a_no_op_without_madvise(model_path, monkeypatch):
    import mmap

    monkeypatch.delattr(mmap, "MADV_DONTNEED", raising=False)
    model = ModelFile(model_path)
    model.release()
    assert np.array_equal(model.tensors["token_embedding.weight"], weights()["token_embedding.weight"])


def test_release_with_large_pages(model_path, monkeypatch):
    import mmap

    monkeypatch.setattr(mmap, "PAGESIZE", 1 << 16)  # 16 KB pages (macOS arm64) and larger: the file is one partial page
    model = ModelFile(model_path)
    model.release()
    assert np.array_equal(model.tensors["token_embedding.weight"], weights()["token_embedding.weight"])


def test_lineage_of_older_files_and_its_problems():
    from etalii_dllm import modelfile

    source = {"format": "gguf", "files": []}
    header = {
        "source": source,
        "adapter": {"base_fingerprint": "a"},
        "fine_tuning": {"base_fingerprint": "b", "data_fingerprint": "d", "steps_completed": 2, "run": {}},
        "edits": [{"base_fingerprint": "c", "method": "rome"}],
    }
    steps = modelfile.lineage(header)
    assert [s["step"] for s in steps] == ["import", "adapter", "fine_tune", "edit"]
    assert [s.get("output") for s in steps] == ["a", "b", "c", None]  # inferred from the next step's input
    assert modelfile.lineage_problems(steps) == []
    steps[1]["output"] = "z"
    assert modelfile.lineage_problems(steps) == ["step 2 (fine_tune) starts from b, but step 1 gave z"]
    assert modelfile.lineage({"lineage": steps}) == steps
    assert modelfile.lineage_problems([]) == ["the lineage does not start with an import"]
    assert modelfile.extend_lineage([], "x", {"step": "import"}) == [{"step": "import"}]
