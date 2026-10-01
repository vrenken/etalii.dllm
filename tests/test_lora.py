"""LoRA adapters: merging, adapter gradients against finite differences, reproducible LoRA runs, the PEFT format,
and the same bits whether an adapter is merged into a file or applied when the model loads."""

from __future__ import annotations

import json

import numpy as np
import pytest
from golden_values import LORA_FINETUNE_FINGERPRINT
from model_fixtures import tiny_config, write_hf_checkpoint
from test_engine_import import model_path  # noqa: F401 - fixture
from test_training import TARGETS, TOKENS, ascii_data, reference_loss

from etalii_dllm import lora as lora_module
from etalii_dllm import numerics
from etalii_dllm.cli import main as cli
from etalii_dllm.engine import ADAPTER_ENVIRONMENT_VARIABLE, MODEL_ENVIRONMENT_VARIABLE, DllmEngine, default_engine
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.importing.safetensors import SafetensorsFile
from etalii_dllm.lora import AdapterError, LoraConfig, adapter_gradients, init_adapters, merge, merged_weights
from etalii_dllm.modelfile import ModelFile, data_fingerprint
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.training import AdamWConfig, CheckpointError, DecoderGradients, FineTuner, RunConfig
from etalii_dllm.transformer import Transformer

LORA = LoraConfig(rank=2, alpha=4.0)
RUN = RunConfig(6, 3, 8, 3, AdamWConfig(learning_rate=3e-2), LORA)


@pytest.fixture(scope="module", params=["llama", "qwen3"])
def base(request, tmp_path_factory) -> ModelFile:
    directory = tmp_path_factory.mktemp(f"lora-{request.param}")
    write_hf_checkpoint(directory / "checkpoint", tiny_config(request.param))
    import_model(directory / "checkpoint", directory / "base.dllm")
    return ModelFile(directory / "base.dllm")


def random_adapters(config, lora: LoraConfig, seed: int = 9) -> dict[str, np.ndarray]:
    """Adapters with both factors non-zero (fresh ones have B = 0)."""
    adapters = init_adapters(config, lora, seed)
    for index, name in enumerate(sorted(adapters)):
        if name.endswith(".lora_b"):
            shape = adapters[name].shape
            adapters[name] = numerics.fill_gaussian(seed + index, shape[0] * shape[1]).reshape(shape) * np.float32(0.1)
    return adapters


def test_merge_matches_float64():
    a = numerics.fill_gaussian(1, 3 * 16).reshape(3, 16)
    b = numerics.fill_gaussian(2, 8 * 3).reshape(8, 3)
    w = numerics.fill_gaussian(3, 8 * 16).reshape(8, 16)
    merged = merge(w, a, b, 0.5)
    expected = w.astype(np.float64) + 0.5 * (b.astype(np.float64) @ a.astype(np.float64))
    assert merged.dtype == np.float32
    np.testing.assert_allclose(merged, expected, rtol=0, atol=1e-6)
    assert merge(w, a, np.zeros_like(b), 0.5).tobytes() == w.tobytes()


def test_adapter_gradients_are_the_chain_rule():
    a = numerics.fill_gaussian(4, 3 * 16).reshape(3, 16)
    b = numerics.fill_gaussian(5, 8 * 3).reshape(8, 3)
    dw = numerics.fill_gaussian(6, 8 * 16).reshape(8, 16)
    da, db = adapter_gradients(dw, a, b, 2.0)
    a64, b64, dw64 = (x.astype(np.float64) for x in (a, b, dw))
    np.testing.assert_allclose(da, 2.0 * b64.T @ dw64, rtol=0, atol=1e-5)
    np.testing.assert_allclose(db, 2.0 * dw64 @ a64.T, rtol=0, atol=1e-5)


def test_lora_gradients_match_finite_differences(base):
    config = base.config
    adapters = random_adapters(config, LORA)
    weights = merged_weights(config, base.tensors, adapters, LORA)
    loss, grads = DecoderGradients(config).loss_and_gradients(weights, TOKENS, TARGETS)
    w64 = {name: np.asarray(values, dtype=np.float64) for name, values in base.tensors.items()}

    def loss_at(changed: dict[str, np.ndarray]) -> float:
        merged = dict(w64)
        for name in lora_module.target_weights(config, LORA):
            a, b = changed[name + ".lora_a"].astype(np.float64), changed[name + ".lora_b"].astype(np.float64)
            merged[name] = w64[name] + LORA.scale * (b @ a)
        return reference_loss(config, merged, TOKENS, TARGETS)

    assert abs(loss - loss_at(adapters)) < 1e-4
    random = numerics.DeterministicRandom(5)
    for name in ("layers.0.attention.q.weight", "layers.1.mlp.down.weight"):
        a, b = adapters[name + ".lora_a"], adapters[name + ".lora_b"]
        analytic = dict(zip(("lora_a", "lora_b"), adapter_gradients(grads[name], a, b, LORA.scale), strict=True))
        for kind in ("lora_a", "lora_b"):
            grad = analytic[kind]
            index = np.unravel_index(random.next_u64() % grad.size, grad.shape)
            eps = 1e-5

            def at(delta, kind=kind, index=index, name=name):
                changed = {k: v.astype(np.float64) for k, v in adapters.items()}
                changed[f"{name}.{kind}"][index] += delta
                return loss_at(changed)

            numeric = (at(eps) - at(-eps)) / (2 * eps)
            assert abs(float(grad[index]) - numeric) < 2e-4 * max(1.0, abs(numeric)), (name, kind, index)


def test_lora_loss_decreases(base):
    run = RunConfig(20, 4, 8, 1, AdamWConfig(learning_rate=5e-2, schedule="constant"), LORA)
    results = FineTuner.from_model_file(base, ascii_data(), run).train()
    assert np.mean([r.loss for r in results[-4:]]) < np.mean([r.loss for r in results[:4]]) - 0.03


def test_fresh_adapters_leave_the_model_unchanged(base):
    adapters = init_adapters(base.config, LORA, 0)
    assert init_adapters(base.config, LORA, 0).keys() == adapters.keys()
    assert all(np.array_equal(adapters[k], v) for k, v in init_adapters(base.config, LORA, 0).items())
    merged = merged_weights(base.config, base.tensors, adapters, LORA)
    assert data_fingerprint(merged) == base.fingerprint


def test_lora_runs_are_byte_identical_and_resume_bit_for_bit(base, tmp_path):
    data = ascii_data()
    first = FineTuner.from_model_file(base, data, RUN)
    first.train()
    assert set(first.params) == set(lora_module.adapter_shapes(base.config, LORA))
    first.save_checkpoint(tmp_path / "first.dllmckpt")
    first.export_adapter(tmp_path / "first-adapter")

    second = FineTuner.from_model_file(base, data, RUN)
    second.train(until=2)
    second.save_checkpoint(tmp_path / "middle.dllmckpt")
    with pytest.raises(CheckpointError, match="base model"):
        FineTuner.load_checkpoint(tmp_path / "middle.dllmckpt", data)
    resumed = FineTuner.load_checkpoint(tmp_path / "middle.dllmckpt", data, base)
    resumed.train()
    resumed.save_checkpoint(tmp_path / "resumed.dllmckpt")
    resumed.export_adapter(tmp_path / "resumed-adapter")
    assert (tmp_path / "resumed.dllmckpt").read_bytes() == (tmp_path / "first.dllmckpt").read_bytes()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        assert (tmp_path / "first-adapter" / name).read_bytes() == (tmp_path / "resumed-adapter" / name).read_bytes()

    # The base weights are never written; the merged export is the base plus the adapters.
    fingerprint = first.export(tmp_path / "merged.dllm")
    assert fingerprint == LORA_FINETUNE_FINGERPRINT[base.config.family]
    assert ModelFile(tmp_path / "merged.dllm").fine_tuning["run"]["lora"] == LORA.to_dict()
    # Importing the exported adapter onto the base writes the same weights.
    (tmp_path / "first-adapter" / "README.md").write_text("---\nlicense: mit\n---\n", encoding="utf-8")
    imported = import_model(tmp_path / "first-adapter", tmp_path / "imported.dllm", base=base.path)
    assert imported.fingerprint == fingerprint
    model = ModelFile(tmp_path / "imported.dllm")
    assert model.adapter["lora"] == LORA.to_dict() and model.adapter["base_fingerprint"] == base.fingerprint
    assert "LoRA adapter first-adapter, licensed under MIT" in model.licence["attribution"]
    assert "otherwise unmodified" not in model.licence["attribution"]
    assert [step["step"] for step in model.lineage] == ["import", "adapter"]
    assert model.lineage[1]["input"] == base.fingerprint and model.lineage[1]["output"] == fingerprint
    # ... and so does applying it at load time.
    assert data_fingerprint(lora_module.apply_adapter(base.config, base.tensors, tmp_path / "first-adapter")[0]) == (
        fingerprint
    )


def test_peft_files(base, tmp_path):
    adapters = random_adapters(base.config, LoraConfig(2, 8.0, ("q", "v")))
    lora_module.write_peft(tmp_path, adapters, LoraConfig(2, 8.0, ("v", "q")), "example/base")
    settings = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert settings["target_modules"] == ["q_proj", "v_proj"] and settings["r"] == 2 and settings["lora_alpha"] == 8.0
    names = SafetensorsFile(tmp_path / "adapter_model.safetensors").names()
    assert "base_model.model.model.layers.1.self_attn.v_proj.lora_B.weight" in names and len(names) == 8
    read_lora, read_adapters = lora_module.read_peft(tmp_path, base.config)
    assert read_lora == LoraConfig(2, 8.0, ("q", "v"))
    assert all(read_adapters[k].tobytes() == v.tobytes() for k, v in adapters.items())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"use_dora": True}, "DoRA"),
        ({"bias": "all"}, "bias"),
        ({"rank_pattern": {"q_proj": 4}}, "rank_pattern"),
        ({"peft_type": "IA3"}, "peft_type"),
        ({"modules_to_save": ["lm_head"]}, "modules_to_save"),
    ],
)
def test_unsupported_peft_features_are_refused(base, tmp_path, change, message):
    lora_module.write_peft(tmp_path, init_adapters(base.config, LORA, 1), LORA)
    settings = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({**settings, **change}), encoding="utf-8")
    with pytest.raises(AdapterError, match=message):
        lora_module.read_peft(tmp_path, base.config)


def test_adapter_import_needs_a_base_and_a_licence(base, tmp_path):
    lora_module.write_peft(tmp_path / "adapter", init_adapters(base.config, LORA, 1), LORA)
    with pytest.raises(ModelImportError, match="--base"):
        import_model(tmp_path / "adapter", tmp_path / "out.dllm")
    with pytest.raises(ModelImportError, match="licence"):
        import_model(tmp_path / "adapter", tmp_path / "out.dllm", base=base.path)
    (tmp_path / "adapter" / "README.md").write_text("---\nlicense: cc-by-nc-4.0\n---\n", encoding="utf-8")
    with pytest.raises(ModelImportError, match="accept-licence"):
        import_model(tmp_path / "adapter", tmp_path / "out.dllm", base=base.path)
    result = import_model(tmp_path / "adapter", tmp_path / "out.dllm", base=base.path, accept_licence=True)
    assert result.licence["redistributable"] is False
    with pytest.raises(ModelImportError, match="not a LoRA adapter"):
        import_model(tmp_path / "out.dllm", tmp_path / "again.dllm", base=base.path)


def test_runtime_adapter_gives_the_merged_model_bits(model_path, tmp_path):  # noqa: F811
    file = ModelFile(model_path)
    lora = LoraConfig(4, 8.0, ("q", "k", "v", "o"))
    lora_module.write_peft(tmp_path / "adapter", random_adapters(file.config, lora), lora)
    import_model(tmp_path / "adapter", tmp_path / "merged.dllm", base=model_path, licence="apache-2.0")
    merged = DllmEngine.from_model_file(tmp_path / "merged.dllm")
    runtime = DllmEngine.from_model_file(model_path, adapter=tmp_path / "adapter")
    plain = DllmEngine.from_model_file(model_path)
    assert runtime.system_fingerprint == merged.system_fingerprint != plain.system_fingerprint
    options = SamplingOptions(temperature=0.8, seed=4)
    assert runtime.complete("Once upon a time", 16, options) == merged.complete("Once upon a time", 16, options)
    tokens = runtime.tokenizer.encode("The quick brown fox")
    assert runtime.model.forward(tokens).tobytes() == merged.model.forward(tokens).tobytes()
    assert runtime.model.forward(tokens).tobytes() != plain.model.forward(tokens).tobytes()


def test_cli_lora_finetune(model_path, tmp_path, capsys, monkeypatch, request):  # noqa: F811
    # --model and --adapter set these in-process; monkeypatch restores them and the engine is reset at the end.
    monkeypatch.setenv(ADAPTER_ENVIRONMENT_VARIABLE, "")
    monkeypatch.setenv(MODEL_ENVIRONMENT_VARIABLE, "")
    request.addfinalizer(default_engine.cache_clear)
    lines = [json.dumps({"text": f"{i} plus {i} is {2 * i}."}) for i in range(8)]
    (tmp_path / "data.jsonl").write_text("\n".join(lines), encoding="utf-8")
    common = ["finetune", str(model_path), "--data", str(tmp_path / "data.jsonl"), "--steps", "3"]
    common += ["--batch-size", "2", "--sequence-length", "16", "--learning-rate", "1e-2", "--lora-rank", "2"]
    checkpoint = ["--checkpoint", str(tmp_path / "run.dllmckpt")]
    assert cli([*common, "--adapter-output", str(tmp_path / "a"), "-o", str(tmp_path / "a.dllm"), *checkpoint]) == 0
    assert cli([*common, "--adapter-output", str(tmp_path / "b")]) == 0
    resumed = ["--resume", str(tmp_path / "run.dllmckpt"), "--adapter-output", str(tmp_path / "c")]
    assert cli([*common, *resumed]) == 0
    capsys.readouterr()
    weights = [(tmp_path / d / "adapter_model.safetensors").read_bytes() for d in "abc"]
    assert weights[0] == weights[1] == weights[2]
    assert cli(["--model", str(model_path), "--adapter", str(tmp_path / "a"), "info"]) == 0
    runtime = capsys.readouterr().out
    monkeypatch.setenv(ADAPTER_ENVIRONMENT_VARIABLE, "")
    assert cli(["--model", str(tmp_path / "a.dllm"), "info"]) == 0
    assert runtime.splitlines()[1] == capsys.readouterr().out.splitlines()[1]  # the system_fingerprint line
    assert cli(["finetune", str(model_path), "--data", str(tmp_path / "data.jsonl")]) == 1
    assert cli([*common[:-2], "--adapter-output", str(tmp_path / "d")]) == 1  # not a LoRA run


def test_config_validation(base):
    with pytest.raises(AdapterError, match="targets"):
        LoraConfig(2, 2.0, ("embed",))
    with pytest.raises(AdapterError, match="rank"):
        LoraConfig(0, 2.0)
    assert LoraConfig(4, 8.0, rslora=True).scale == 4.0
    config = base.config
    model = Transformer(config, merged_weights(config, base.tensors, random_adapters(config, LORA), LORA))
    assert model.forward(TOKENS).shape == (config.vocabulary_size,)
