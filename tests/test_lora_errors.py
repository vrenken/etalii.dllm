"""Malformed and unsupported PEFT adapters are refused with a precise reason, and the formats PEFT writes besides
float32 (float16, bfloat16, rsLoRA) are read exactly."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from model_fixtures import to_bf16_bits
from test_lora import LORA, base, random_adapters  # noqa: F401 - fixture

from etalii_dllm import lora as lora_module
from etalii_dllm.lora import AdapterError, LoraConfig, init_adapters, peft_key


def write_raw_safetensors(path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """A safetensors file whose tensors keep their own dtype (``write_safetensors`` always writes F32)."""
    header, offset, blobs = {}, 0, []
    for name in sorted(tensors):
        dtype, values = tensors[name]
        data = np.ascontiguousarray(values).tobytes()
        header[name] = {"dtype": dtype, "shape": list(values.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def valid_adapter(directory, config, lora: LoraConfig = LORA) -> dict[str, np.ndarray]:
    adapters = random_adapters(config, lora)
    lora_module.write_peft(directory, adapters, lora)
    return adapters


def edit_config(directory, **changes) -> None:
    path = directory / "adapter_config.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**settings, **changes}), encoding="utf-8")


def peft_tensors(adapters: dict[str, np.ndarray]) -> dict[str, tuple[str, np.ndarray]]:
    return {peft_key(name): ("F32", values) for name, values in adapters.items()}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"fan_in_fan_out": True}, "uses fan_in_fan_out, which is not supported"),
        ({"layers_to_transform": [0]}, "layers_to_transform/layer_replication"),
        ({"layers_to_transform": 0}, "layers_to_transform/layer_replication"),
        ({"layer_replication": [[0, 1]]}, "layers_to_transform/layer_replication"),
        ({"alpha_pattern": {"v_proj": 2}}, "per-layer ranks or alphas"),
        ({"bias": "lora_only"}, "bias='lora_only'"),
        ({"peft_type": "LOHA"}, "peft_type 'LOHA' \\(only LORA\\)"),
    ],
)
def test_more_unsupported_peft_features_are_refused(base, tmp_path, change, message):  # noqa: F811
    valid_adapter(tmp_path, base.config)
    edit_config(tmp_path, **change)
    with pytest.raises(AdapterError, match=message):
        lora_module.read_peft(tmp_path, base.config)


@pytest.mark.parametrize(
    "change",
    [
        {"fan_in_fan_out": False, "use_dora": False, "bias": "none", "modules_to_save": None},
        {"layers_to_transform": None, "layer_replication": None, "rank_pattern": {}, "alpha_pattern": {}},
    ],
)
def test_default_valued_options_are_accepted(base, tmp_path, change):  # noqa: F811
    adapters = valid_adapter(tmp_path, base.config)
    edit_config(tmp_path, **change)
    _, read = lora_module.read_peft(tmp_path, base.config)
    assert all(read[k].tobytes() == v.tobytes() for k, v in adapters.items())


def test_missing_files_are_reported(base, tmp_path):  # noqa: F811
    with pytest.raises(AdapterError, match=r"no adapter_config\.json"):
        lora_module.read_peft(tmp_path, base.config)
    valid_adapter(tmp_path, base.config)
    (tmp_path / "adapter_model.safetensors").unlink()
    (tmp_path / "adapter_model.bin").write_bytes(b"pickle")
    with pytest.raises(AdapterError, match=r"no adapter_model\.safetensors \(adapter_model\.bin is not read"):
        lora_module.read_peft(tmp_path, base.config)


@pytest.mark.parametrize("content", [b"", b"\x05\x00", struct.pack("<Q", 4) + b"nope", struct.pack("<Q", 1 << 40)])
def test_corrupt_weights_are_an_adapter_error(base, tmp_path, content):  # noqa: F811
    valid_adapter(tmp_path, base.config)
    (tmp_path / "adapter_model.safetensors").write_bytes(content)
    with pytest.raises(AdapterError, match=r"adapter_model\.safetensors"):
        lora_module.read_peft(tmp_path, base.config)


@pytest.mark.parametrize(
    "name",
    [
        "base_model.model.lm_head.lora_A.weight",
        "base_model.model.model.embed_tokens.lora_embedding_A",
        "base_model.model.model.layers.0.self_attn.rotary_emb.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_C.weight",
    ],
)
def test_unknown_tensors_are_refused(base, tmp_path, name):  # noqa: F811
    adapters = valid_adapter(tmp_path, base.config)
    tensors = peft_tensors(adapters)
    tensors[name] = ("F32", np.zeros((2, 16), dtype=np.float32))
    write_raw_safetensors(tmp_path / "adapter_model.safetensors", tensors)
    with pytest.raises(AdapterError, match=f"unsupported adapter tensor '{name}'"):
        lora_module.read_peft(tmp_path, base.config)


@pytest.mark.parametrize(
    "name",
    [
        "base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight",  # the tiny model has two layers
        "base_model.model.model.layers.0.mlp.q_proj.lora_A.weight",  # q_proj lives under self_attn
        "base_model.model.model.layers.1.self_attn.down_proj.lora_B.weight",  # down_proj lives under mlp
    ],
)
def test_tensors_that_do_not_fit_the_model_are_refused(base, tmp_path, name):  # noqa: F811
    adapters = valid_adapter(tmp_path, base.config)
    tensors = peft_tensors(adapters)
    tensors[name] = ("F32", np.zeros((2, 16), dtype=np.float32))
    write_raw_safetensors(tmp_path / "adapter_model.safetensors", tensors)
    with pytest.raises(AdapterError, match=f"tensor '{name}' does not fit the model"):
        lora_module.read_peft(tmp_path, base.config)


@pytest.mark.parametrize(("dtype", "numpy_dtype"), [("F64", "<f8"), ("I32", "<i4"), ("U8", "u1")])
def test_non_float_or_lossy_dtypes_are_refused(base, tmp_path, dtype, numpy_dtype):  # noqa: F811
    adapters = valid_adapter(tmp_path, base.config)
    tensors = peft_tensors(adapters)
    first = sorted(tensors)[0]
    tensors[first] = (dtype, tensors[first][1].astype(numpy_dtype))
    write_raw_safetensors(tmp_path / "adapter_model.safetensors", tensors)
    with pytest.raises(AdapterError, match=f"has dtype {dtype}"):
        lora_module.read_peft(tmp_path, base.config)


def test_half_precision_adapters_are_read_exactly(base, tmp_path):  # noqa: F811
    adapters = valid_adapter(tmp_path, base.config)
    tensors = {}
    expected = {}
    for index, (name, values) in enumerate(sorted(adapters.items())):
        if index % 2:
            half = values.astype(np.float16)
            tensors[peft_key(name)] = ("F16", half)
            expected[name] = half.astype(np.float32)
        else:
            bits = to_bf16_bits(values)
            tensors[peft_key(name)] = ("BF16", bits)
            expected[name] = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    write_raw_safetensors(tmp_path / "adapter_model.safetensors", tensors)
    lora, read = lora_module.read_peft(tmp_path, base.config)
    assert lora == LORA
    assert set(read) == set(expected)
    for name, values in read.items():
        assert values.dtype == np.float32 and values.flags.c_contiguous
        assert values.tobytes() == expected[name].tobytes()


def test_empty_adapter_is_refused(base, tmp_path):  # noqa: F811
    valid_adapter(tmp_path, base.config)
    write_raw_safetensors(tmp_path / "adapter_model.safetensors", {})
    with pytest.raises(AdapterError, match="the adapter has no LoRA tensors"):
        lora_module.read_peft(tmp_path, base.config)


def test_adapter_must_cover_every_layer(base, tmp_path):  # noqa: F811
    adapters = valid_adapter(tmp_path, base.config)
    first_layer = {name: values for name, values in adapters.items() if name.startswith("layers.0.")}
    write_raw_safetensors(tmp_path / "adapter_model.safetensors", peft_tensors(first_layer))
    with pytest.raises(AdapterError, match=r"does not cover every layer \(missing \['layers\.1\."):
        lora_module.read_peft(tmp_path, base.config)


def test_adapter_missing_one_factor_is_refused(base, tmp_path):  # noqa: F811
    adapters = valid_adapter(tmp_path, base.config)
    without = {name: values for name, values in adapters.items() if name != sorted(adapters)[0]}
    write_raw_safetensors(tmp_path / "adapter_model.safetensors", peft_tensors(without))
    with pytest.raises(AdapterError, match=f"missing \\['{sorted(adapters)[0]}'\\]"):
        lora_module.read_peft(tmp_path, base.config)


def test_rank_in_the_config_must_match_the_tensors(base, tmp_path):  # noqa: F811
    valid_adapter(tmp_path, base.config)
    edit_config(tmp_path, r=4)
    with pytest.raises(AdapterError, match=r"lora_a has shape \(2, 16\), expected \(4, 16\)"):
        lora_module.read_peft(tmp_path, base.config)


def test_transposed_factor_is_refused(base, tmp_path):  # noqa: F811
    adapters = valid_adapter(tmp_path, base.config)
    name = next(n for n in sorted(adapters) if n.endswith(".lora_b"))
    tensors = peft_tensors(adapters)
    tensors[peft_key(name)] = ("F32", np.ascontiguousarray(adapters[name].T))
    write_raw_safetensors(tmp_path / "adapter_model.safetensors", tensors)
    expected = adapters[name].shape
    with pytest.raises(AdapterError, match=f"{name} has shape \\({expected[1]}, {expected[0]}\\)"):
        lora_module.read_peft(tmp_path, base.config)


def test_rslora_round_trips(base, tmp_path):  # noqa: F811
    lora = LoraConfig(4, 8.0, ("v", "q"), rslora=True)
    assert lora.to_dict() == {"rank": 4, "alpha": 8.0, "targets": ["q", "v"], "rslora": True}
    assert LoraConfig.from_dict(lora.to_dict()) == lora
    assert "rslora" not in LoraConfig(4, 8.0).to_dict()
    assert LoraConfig.from_dict(LoraConfig(4, 8.0).to_dict()) == LoraConfig(4, 8.0)
    lora_module.write_peft(tmp_path, init_adapters(base.config, lora, 3), lora)
    assert json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))["use_rslora"] is True
    read, _ = lora_module.read_peft(tmp_path, base.config)
    assert read == lora and read.scale == 4.0


def test_default_alpha_when_the_config_has_none(base, tmp_path):  # noqa: F811
    valid_adapter(tmp_path, base.config)
    path = tmp_path / "adapter_config.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    del settings["lora_alpha"], settings["use_rslora"]
    path.write_text(json.dumps(settings), encoding="utf-8")
    lora, _ = lora_module.read_peft(tmp_path, base.config)
    assert lora.alpha == 8.0 and not lora.rslora  # PEFT's defaults


def test_adapter_for_another_architecture_is_refused(base, tmp_path):  # noqa: F811
    # An adapter for a wider model: same names, different shapes.
    config = base.config
    wider = type(config)(**{**config.to_dict(), "eos_token_ids": config.eos_token_ids, "hidden_size": 32})
    lora_module.write_peft(tmp_path, random_adapters(wider, LORA), LORA)
    with pytest.raises(AdapterError, match=r"has shape \(\d+, \d+\), expected"):
        lora_module.read_peft(tmp_path, config)
