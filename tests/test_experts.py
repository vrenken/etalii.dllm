"""Mixture-of-experts models (Phase 42; shared experts Phase 44): exact routing, batch invariance, imports and
exports, the reference semantics, routing traces and ``dllm experts``.

The decoders themselves are checked against the float64 transcription of transformers in ``test_transformer.py``
(families ``mixtral``, ``olmoe``, ``qwen3_moe``, ``qwen2_moe`` and ``granitemoe``) and against the reference
implementation in ``test_reference.py``; the routing kernel against the reference in the conformance vectors.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
from model_fixtures import MIXTRAL_EXPERT_NAMES, gguf_name, head_dim, hf_weights, tiny_config, write_hf_checkpoint
from test_model_building import BYTE_LEVEL_TOKENIZER

from etalii_dllm import _kernels, exporting, numerics, reference
from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.cli import main
from etalii_dllm.engine import default_engine
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.importing.importer import hf_config
from etalii_dllm.interpret import routing, trace
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.numerics import PackedWeight, QuantizedWeight, fingerprint, moe_route
from etalii_dllm.transformer import LayerHook, Transformer

PROMPT = [1, 17, 42, 5, 63, 0, 9, 9, 30]
MOE_FAMILIES = ["granitemoe", "granitemoeshared", "mixtral", "olmoe", "qwen2_moe", "qwen3_moe"]
GGUF_ARCHITECTURES = {"mixtral": "llama", "olmoe": "olmoe", "qwen3_moe": "qwen3moe", "qwen2_moe": "qwen2moe"}


def gaussian(seed: int, *shape: int) -> np.ndarray:
    return numerics.fill_gaussian(seed, math.prod(shape)).reshape(shape)


@pytest.fixture(autouse=True)
def restore_kernel_settings():
    yield
    numerics.set_threads(0)
    _kernels.set_isa("best")


@pytest.fixture(scope="module", params=MOE_FAMILIES)
def model_path(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(request.param)
    config = {**tiny_config(request.param), "vocab_size": 264}
    write_hf_checkpoint(directory / "checkpoint", config, tokenizer_json=BYTE_LEVEL_TOKENIZER)
    import_model(directory / "checkpoint", directory / "model.dllm")
    return directory / "model.dllm"


# -- routing ------------------------------------------------------------------------------------------------------


def test_routing_takes_the_top_k_in_a_total_order():
    logits = np.array([[1.0, 3.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0], [5.0, -1.0, 4.0, 4.0]], np.float32)
    experts, weights = moe_route(logits, 2, False)
    assert experts.tolist() == [[1, 2], [0, 1], [0, 2]]  # ties go to the lower expert
    probabilities = np.stack([numerics.softmax(row) for row in logits])
    assert weights.tobytes() == np.take_along_axis(probabilities, experts, 1).tobytes()
    experts, normalized = moe_route(logits, 2, True)
    assert normalized[0].tolist() == [0.5, 0.5] and normalized[1].tolist() == [0.5, 0.5]
    total = float(weights[2, 0]) + float(weights[2, 1])
    assert normalized[2].tolist() == [float(np.float32(float(w) / total)) for w in weights[2]]
    assert moe_route(logits[0], 4, False)[0].tolist() == [[1, 2, 0, 3]]  # one row; k == experts
    for k in (0, 5):
        with pytest.raises(ValueError, match="experts per token"):
            moe_route(logits, k, False)


def test_routing_matches_the_reference_and_ignores_the_batch():
    logits = gaussian(1, 50, 16) * np.float32(4.0)
    logits[7] = logits[3]  # equal rows route equally wherever they are
    for k, normalize in ((1, False), (3, True), (16, True)):
        experts, weights = moe_route(logits, k, normalize)
        chosen, expected = reference.moe_route(logits, k, normalize)
        assert experts.tolist() == chosen and weights.tobytes() == expected.tobytes()
        assert experts[7].tolist() == experts[3].tolist()
        for row in (0, 13, 49):  # a row alone gives the bits it gets among the others
            alone = moe_route(logits[row : row + 1], k, normalize)
            assert alone[0].tolist() == experts[row : row + 1].tolist()
            assert alone[1].tobytes() == weights[row : row + 1].tobytes()


def test_expert_settings_are_validated():
    base = TransformerConfig.from_dict(
        {
            "family": "qwen3_moe",
            "vocabulary_size": 8,
            "hidden_size": 8,
            "intermediate_size": 8,
            "layers": 2,
            "heads": 2,
            "kv_heads": 1,
            "head_dim": 4,
            "context_length": 16,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1e4,
            "experts": 4,
            "experts_per_token": 2,
        }
    )
    assert base.is_sparse(0) and base.expert_size == 8
    for change, message in [
        ({"experts_per_token": 5}, "experts_per_token"),
        ({"experts_per_token": 0}, "experts_per_token"),
        ({"experts": 0}, "need experts"),
        ({"expert_intermediate_size": 0}, "expert_intermediate_size"),
        ({"dense_layers": (2,)}, "dense_layers"),
    ]:
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(base, **change)
    sparse = dataclasses.replace(base, dense_layers=(0,), expert_intermediate_size=4)
    assert not sparse.is_sparse(0) and sparse.is_sparse(1) and sparse.expert_size == 4
    shapes = sparse.tensor_shapes()
    assert "layers.0.mlp.gate.weight" in shapes and "layers.0.mlp.router.weight" not in shapes
    assert shapes["layers.1.mlp.router.weight"] == (4, 8) and shapes["layers.1.mlp.experts.3.down.weight"] == (8, 4)
    assert TransformerConfig.from_dict(sparse.to_dict()) == sparse
    dense = dataclasses.replace(base, experts=0, experts_per_token=0)
    assert not any(key.startswith(("experts", "dense", "normalize")) for key in dense.to_dict())


# -- the decoder --------------------------------------------------------------------------------------------------


def test_mixtures_of_experts_are_batch_invariant(model_path):
    model = Transformer.from_file(model_path)
    sequences = [PROMPT, PROMPT[:3], [4, 4, 4, 4, 4], PROMPT[::-1]]
    alone = [model.forward(tokens).tobytes() for tokens in sequences]
    for count in (1, 3, 8):
        numerics.set_threads(count)
        assert [x.tobytes() for x in model.forward_batch(sequences)] == alone
    cache = model.new_cache()
    stepwise = [model.forward_cached(PROMPT[: i + 1], cache) for i in range(len(PROMPT))]
    assert stepwise[-1].tobytes() == alone[0]
    last = model.forward_cached_last(PROMPT, model.new_cache(), 4)
    assert [row.tobytes() for row in last] == [s.tobytes() for s in stepwise[-4:]]


def test_quantised_experts_keep_a_float_router():
    from test_reference import _model

    model = _model("qwen3_moe", "q8_0")
    config = model.config
    layer = next(i for i in range(config.layers) if config.is_sparse(i))
    assert isinstance(model._w[f"layers.{layer}.mlp.router.weight"], PackedWeight)
    assert isinstance(model._w[f"layers.{layer}.mlp.experts.0.gate.weight"], QuantizedWeight)
    twin = reference.ReferenceTransformer.from_engine_model(model)
    assert model.forward(PROMPT).tobytes() == twin.forward(PROMPT).tobytes()


def test_routing_hooks_and_traces(model_path):
    model = Transformer.from_file(model_path)
    config = model.config
    seen = {}

    class Watcher(LayerHook):
        def routing(self, layer, experts, weights):
            seen[layer] = (experts, weights)

    hidden = model.run_hooked(PROMPT, Watcher())
    assert model.logits_from_hidden(hidden[-1:])[0].tobytes() == model.forward(PROMPT).tobytes()
    assert sorted(seen) == [i for i in range(config.layers) if config.is_sparse(i)]
    for experts, weights in seen.values():
        assert experts.shape == weights.shape == (len(PROMPT), config.experts_per_token)
    recorded = trace(model, PROMPT)
    assert recorded.mlp_activation is None and recorded.experts is not None
    for layer in range(config.layers):
        if config.is_sparse(layer):
            assert recorded.experts[layer].tolist() == seen[layer][0].tolist()
        else:
            assert (recorded.experts[layer] == -1).all() and (recorded.expert_weights[layer] == 0).all()
    assert recorded.fingerprint() == trace(model, PROMPT).fingerprint()
    result = routing(model, PROMPT)
    assert result.layers == tuple(sorted(seen))
    usage = result.usage()
    assert usage.shape == (len(result.layers), config.experts)
    assert (usage.sum(1) == len(PROMPT) * config.experts_per_token).all()


def test_routing_needs_experts(tmp_path):
    write_hf_checkpoint(tmp_path / "dense", tiny_config("llama"))
    import_model(tmp_path / "dense", tmp_path / "dense.dllm")
    with pytest.raises(ValueError, match="no experts"):
        routing(Transformer.from_file(tmp_path / "dense.dllm"), PROMPT)


# -- imports and exports ------------------------------------------------------------------------------------------


def test_transformers_v5_stacked_experts_import_the_same(tmp_path):
    from model_fixtures import to_bf16_bits, write_safetensors

    config = tiny_config("qwen3_moe")
    weights = write_hf_checkpoint(tmp_path / "split", config)
    original = import_model(tmp_path / "split", tmp_path / "split.dllm")
    stacked = {}
    for name, values in weights.items():
        if ".mlp.experts." not in name:
            stacked[name] = values
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}.mlp.experts."
        if p + "0.gate_proj.weight" not in weights:
            continue
        experts = range(config["num_experts"])
        stacked[p + "gate_up_proj"] = np.stack(
            [np.concatenate([weights[f"{p}{e}.gate_proj.weight"], weights[f"{p}{e}.up_proj.weight"]]) for e in experts]
        )
        stacked[p + "down_proj"] = np.stack([weights[f"{p}{e}.down_proj.weight"] for e in experts])
    write_hf_checkpoint(tmp_path / "stacked", config)
    write_safetensors(
        tmp_path / "stacked" / "model.safetensors",
        {name: ("BF16", to_bf16_bits(values)) for name, values in stacked.items()},
    )
    assert import_model(tmp_path / "stacked", tmp_path / "stacked.dllm").fingerprint == original.fingerprint
    bad = {**stacked, "model.layers.1.mlp.experts.down_proj": stacked["model.layers.1.mlp.experts.down_proj"][:2]}
    write_safetensors(
        tmp_path / "stacked" / "model.safetensors", {k: ("BF16", to_bf16_bits(v)) for k, v in bad.items()}
    )
    with pytest.raises(ModelImportError, match="expected"):
        import_model(tmp_path / "stacked", tmp_path / "bad.dllm")


def test_hugging_face_settings():
    config = tiny_config("qwen3_moe")
    ours = hf_config(config)
    assert (ours.experts, ours.experts_per_token, ours.expert_intermediate_size) == (4, 2, 8)
    assert ours.normalize_expert_weights and ours.dense_layers == (0,)
    stepped = hf_config({**config, "mlp_only_layers": [], "decoder_sparse_step": 2})
    assert stepped.dense_layers == (0,) and stepped.is_sparse(1)
    assert hf_config({**config, "mlp_only_layers": []}).dense_layers is None
    mixtral = hf_config(tiny_config("mixtral"))
    assert mixtral.family == "mixtral" and mixtral.normalize_expert_weights and mixtral.expert_size == 32
    olmoe = hf_config(tiny_config("olmoe"))
    assert not olmoe.normalize_expert_weights and olmoe.qk_norm_scope == "all"
    qwen2 = hf_config(tiny_config("qwen2_moe"))
    assert not qwen2.normalize_expert_weights and qwen2.attention_bias and qwen2.dense_layers is None
    assert (qwen2.expert_intermediate_size, qwen2.shared_expert_intermediate_size, qwen2.shared_expert_gate) == (
        8,
        12,
        True,
    )
    granite = hf_config(tiny_config("granitemoe"))
    assert granite.family == "granitemoe" and granite.normalize_expert_weights and granite.residual_multiplier == 0.22
    assert granite.shared_expert_intermediate_size is None
    shared = hf_config(tiny_config("granitemoeshared"))
    assert shared.shared_expert_intermediate_size == 12 and not shared.shared_expert_gate
    for change, message in [
        ({"clip_qkv": 8.0}, "clip_qkv"),
        ({"shared_expert_intermediate_size": 16}, "shared expert"),
        ({"n_shared_experts": 2}, "shared expert"),
        ({"num_experts_per_tok": 9}, "experts_per_token"),
    ]:
        with pytest.raises(ModelImportError, match=message):
            hf_config({**tiny_config("olmoe"), **change})


def write_moe_gguf(path, family: str, stacked: bool = True):
    """The tiny Hugging Face weights of ``family`` written as llama.cpp's converter does: Mixtral as ``llama`` with
    permuted Q/K rows, experts stacked per projection (or, ``stacked=False``, one tensor per expert as old Mixtral
    files have them)."""
    import gguf

    config = tiny_config(family)
    weights = hf_weights(config)
    architecture = GGUF_ARCHITECTURES[family]
    writer = gguf.GGUFWriter(str(path), architecture)
    writer.add_name("Tiny MoE")
    writer.add_string("general.license", "apache-2.0")
    writer.add_context_length(config["max_position_embeddings"])
    writer.add_embedding_length(config["hidden_size"])
    writer.add_block_count(config["num_hidden_layers"])
    writer.add_feed_forward_length(config["intermediate_size"])
    writer.add_head_count(config["num_attention_heads"])
    writer.add_head_count_kv(config["num_key_value_heads"])
    writer.add_rope_dimension_count(head_dim(config))
    writer.add_key_length(head_dim(config))
    writer.add_rope_freq_base(config["rope_theta"])
    writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
    writer.add_vocab_size(config["vocab_size"])
    writer.add_expert_count(config.get("num_local_experts") or config["num_experts"])
    writer.add_expert_used_count(config["num_experts_per_tok"])
    if family in ("qwen2_moe", "qwen3_moe"):
        writer.add_expert_feed_forward_length(config["moe_intermediate_size"])
    if family == "qwen2_moe":
        writer.add_expert_shared_feed_forward_length(config["shared_expert_intermediate_size"])
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list([f"t{i}" for i in range(config["vocab_size"])])
    experts: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    for name, values in weights.items():
        parts = name.split(".")
        if "experts" in parts:
            layer, expert, projection = parts[2], int(parts[parts.index("experts") + 1]), parts[-2]
            projection = {v: k for k, v in MIXTRAL_EXPERT_NAMES.items()}.get(
                projection, projection.removesuffix("_proj")
            )
            experts.setdefault((layer, projection), {})[expert] = values
            continue
        if name.endswith(("mlp.gate.weight", "block_sparse_moe.gate.weight")):
            writer.add_tensor(f"blk.{parts[2]}.ffn_gate_inp.weight", values)
            continue
        if name.endswith("mlp.shared_expert_gate.weight"):  # one row, stored 1-D by llama.cpp
            writer.add_tensor(f"blk.{parts[2]}.ffn_gate_inp_shexp.weight", values.reshape(-1))
            continue
        if architecture == "llama" and ".q_proj." in name:
            values = exporting.permute_rotary(values, config["num_attention_heads"])
        elif architecture == "llama" and ".k_proj." in name:
            values = exporting.permute_rotary(values, config["num_key_value_heads"])
        writer.add_tensor(gguf_name(name), np.ascontiguousarray(values, dtype=np.float32))
    for (layer, projection), parts in experts.items():
        if stacked:
            values = np.stack([parts[e] for e in sorted(parts)])
            writer.add_tensor(f"blk.{layer}.ffn_{projection}_exps.weight", values)
        else:
            for expert in sorted(parts):
                writer.add_tensor(f"blk.{layer}.ffn_{projection}.{expert}.weight", parts[expert])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.parametrize(("family", "stacked"), [("mixtral", True), ("mixtral", False), ("olmoe", True),
                                                 ("qwen3_moe", True), ("qwen2_moe", True)])  # fmt: skip
def test_gguf_imports_give_the_hugging_face_weights(tmp_path, family, stacked):
    pytest.importorskip("gguf")
    write_hf_checkpoint(tmp_path / "hf", tiny_config(family))
    original = import_model(tmp_path / "hf", tmp_path / "hf.dllm")
    write_moe_gguf(tmp_path / "model.gguf", family, stacked)
    imported = import_model(tmp_path / "model.gguf", tmp_path / "gguf.dllm")
    assert imported.fingerprint == original.fingerprint
    expected = dataclasses.replace(
        original.config,
        rms_norm_eps=float(np.float32(original.config.rms_norm_eps)),
        eos_token_ids=(),
        bos_token_id=None,
    )
    assert imported.config == dataclasses.replace(expected, tie_word_embeddings=imported.config.tie_word_embeddings)
    assert fingerprint(Transformer.from_file(tmp_path / "gguf.dllm").forward(PROMPT)) == fingerprint(
        Transformer.from_file(tmp_path / "hf.dllm").forward(PROMPT)
    )


def test_gguf_expert_errors(tmp_path):
    pytest.importorskip("gguf")
    import gguf

    from etalii_dllm.importing.gguf import GgufFile
    from etalii_dllm.importing.importer import gguf_config

    write_moe_gguf(tmp_path / "model.gguf", "olmoe")
    assert GgufFile(tmp_path / "model.gguf")["blk.0.ffn_up_exps.weight"].item(1).shape == (8, 16)
    for key, value, message in [
        ("olmoe.expert_shared_count", 1, "shared experts"),
        ("olmoe.expert_gating_func", 2, "softmax expert gating"),
    ]:
        writer = gguf.GGUFWriter(str(tmp_path / "odd.gguf"), "olmoe")
        for name, field in GgufFile(tmp_path / "model.gguf").metadata.items():
            if name.startswith("olmoe.") and isinstance(field, int):
                writer.add_uint32(name, field)
        writer.add_uint32(key, value)
        writer.add_tensor("blk.0.ffn_gate_inp.weight", np.zeros((4, 16), np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        with pytest.raises(ModelImportError, match=message):
            gguf_config(GgufFile(tmp_path / "odd.gguf"))


@pytest.mark.parametrize("family", MOE_FAMILIES)
def test_exports_import_back(tmp_path, family):
    pytest.importorskip("gguf")
    config = {**tiny_config(family), "vocab_size": 264}
    write_hf_checkpoint(tmp_path / "hf", config, tokenizer_json=BYTE_LEVEL_TOKENIZER)
    original = import_model(tmp_path / "hf", tmp_path / "model.dllm")
    exporting.export_model(tmp_path / "model.dllm", tmp_path / "out", "safetensors")
    exported = json.loads((tmp_path / "out" / "config.json").read_text())
    assert exported["model_type"] == family and exported["num_experts_per_tok"] == 2
    assert import_model(tmp_path / "out", tmp_path / "again.dllm").fingerprint == original.fingerprint
    if family.startswith("granite"):  # llama.cpp has no Granite MoE architecture
        with pytest.raises(exporting.ExportError):
            exporting.export_model(tmp_path / "model.dllm", tmp_path / "model.gguf", "gguf")
        return
    exporting.export_model(tmp_path / "model.dllm", tmp_path / "model.gguf", "gguf")
    again = import_model(tmp_path / "model.gguf", tmp_path / "gguf.dllm")
    assert again.fingerprint == original.fingerprint
    assert again.config == dataclasses.replace(
        original.config, rms_norm_eps=float(np.float32(original.config.rms_norm_eps))
    )
    import gguf

    reader = gguf.GGUFReader(str(tmp_path / "model.gguf"))
    architecture = GGUF_ARCHITECTURES[family]
    assert reader.fields["general.architecture"].contents() == architecture
    assert reader.fields[f"{architecture}.expert_count"].contents() == 4
    names = {t.name for t in reader.tensors}
    assert "blk.1.ffn_gate_exps.weight" in names and "blk.1.ffn_gate_inp.weight" in names
    if family == "qwen2_moe":
        shared = {t.name: t for t in reader.tensors}["blk.1.ffn_gate_inp_shexp.weight"]
        assert len(shared.shape) == 1 and "blk.1.ffn_down_shexp.weight" in names
        assert reader.fields[f"{architecture}.expert_shared_feed_forward_length"].contents() == 12


# -- the command line ---------------------------------------------------------------------------------------------


@pytest.fixture
def isolated(monkeypatch):
    from etalii_dllm import engine

    for name in (engine.MODEL_ENVIRONMENT_VARIABLE, engine.QUANTIZE_ENVIRONMENT_VARIABLE):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    default_engine.cache_clear()
    yield
    default_engine.cache_clear()


def test_experts_command(model_path, capsys, isolated):
    config = ModelFile(model_path).config
    sparse = [i + 1 for i in range(config.layers) if config.is_sparse(i)]
    model = ["--model", str(model_path)]
    assert main([*model, "experts", "--prompt", "hello there"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"layer {sparse[0]}:") and "usage:" in out
    assert main([*model, "experts", "--prompt", "hello there", "--json", "--layer", str(sparse[-1])]) == 0
    result = json.loads(capsys.readouterr().out)
    layer = result["layers"][str(sparse[-1])]
    assert list(result["layers"]) == [str(sparse[-1])]
    assert len(layer["experts"]) == len(result["tokens"]) and sum(layer["usage"]) == 2 * len(result["tokens"])
    assert main([*model, "experts", "--prompt", "hello", "--layer", "9"]) == 1
    assert "mixture-of-experts layer" in capsys.readouterr().err
    assert main(["inspect", str(model_path)]) == 0
    assert "experts:            4, 2 per token (" in capsys.readouterr().out


def test_import_prints_the_experts(tmp_path, capsys):
    write_hf_checkpoint(tmp_path / "hf", tiny_config("olmoe"))
    assert main(["import", str(tmp_path / "hf"), "-o", str(tmp_path / "model.dllm")]) == 0
    assert "experts:            4, 2 per token" in capsys.readouterr().out
