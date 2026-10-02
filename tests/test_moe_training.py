"""Fine-tuning mixture-of-experts models (Phase 43): the routing backward kernel, the router load-balancing loss,
LoRA adapters on the experts in the PEFT format, distillation and ROME edits of the routed expert.

The gradients of whole models are checked against finite differences of the float64 reference in
``test_training.py`` (families ``mixtral``, ``olmoe`` and ``qwen3_moe``), LoRA and DPO runs in ``test_lora.py`` and
``test_preference.py``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from model_fixtures import tiny_config, write_hf_checkpoint
from test_experts import model_path  # noqa: F401 - fixture

from etalii_dllm import lora as lora_module
from etalii_dllm import numerics
from etalii_dllm.cli import main
from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import import_model
from etalii_dllm.importing.safetensors import SafetensorsFile
from etalii_dllm.interpret import trace
from etalii_dllm.interpret.editing import EditRequest, rome, subject_position, write_edited_model
from etalii_dllm.lora import AdapterError, LoraConfig
from etalii_dllm.modelfile import ModelFile, tensor_order
from etalii_dllm.numerics import moe_route, moe_route_backward
from etalii_dllm.training import AdamWConfig, FineTuner, RunConfig, TrainingData
from etalii_dllm.training.backprop import DecoderGradients

TOKENS = [1, 17, 42, 5, 63, 0, 9, 9, 30]
TARGETS = [17, 42, 5, 63, 0, 9, 9, 30, 2]


def gaussian(seed: int, *shape: int) -> np.ndarray:
    return numerics.fill_gaussian(seed, math.prod(shape)).reshape(shape)


def data(sequence_length: int = 8) -> TrainingData:
    documents = ["the quick brown fox jumps over the lazy dog", "pack my box with five dozen liquor jugs"]
    return TrainingData.from_documents(documents, lambda text: list(text.encode()), sequence_length, 2)


# -- the routing backward kernel ----------------------------------------------------------------------------------


def routed_value(logits: np.ndarray, indices: np.ndarray, dweights, extra, normalize: bool) -> float:
    """float64: sum(dweights * routing weights) + sum(extra * probabilities), for a fixed choice of experts."""
    shifted = np.exp(logits - logits.max(-1, keepdims=True))
    probabilities = shifted / shifted.sum(-1, keepdims=True)
    chosen = np.take_along_axis(probabilities, indices, -1)
    if normalize:
        chosen = chosen / chosen.sum(-1, keepdims=True)
    return float((chosen * dweights).sum() + (probabilities * extra).sum())


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("with_extra", [False, True])
def test_routing_backward_matches_numeric_gradient(normalize, with_extra):
    logits = gaussian(1, 6, 8)
    indices, weights = moe_route(logits, 3, normalize)
    dweights = gaussian(2, *weights.shape)
    extra = gaussian(3, *logits.shape) if with_extra else np.zeros_like(logits)
    dlogits = moe_route_backward(logits, indices, dweights, normalize, extra if with_extra else None)
    assert dlogits.shape == logits.shape and dlogits.dtype == np.float32
    base = logits.astype(np.float64)
    eps = 1e-5
    for row in range(logits.shape[0]):
        for expert in range(logits.shape[1]):
            up, down = base.copy(), base.copy()
            up[row, expert] += eps
            down[row, expert] -= eps
            numeric = (
                routed_value(up, indices, dweights, extra, normalize)
                - routed_value(down, indices, dweights, extra, normalize)
            ) / (2 * eps)
            assert abs(float(dlogits[row, expert]) - numeric) < 1e-6, (row, expert)


def test_routing_backward_is_one_row_at_a_time_and_validates():
    logits = gaussian(4, 5, 4)
    indices, weights = moe_route(logits, 2, True)
    dweights = gaussian(5, *weights.shape)
    full = moe_route_backward(logits, indices, dweights, True)
    for row in range(5):
        alone = moe_route_backward(logits[row : row + 1], indices[row : row + 1], dweights[row : row + 1], True)
        assert alone.tobytes() == full[row : row + 1].tobytes()
    with pytest.raises(ValueError, match="shape of indices"):
        moe_route_backward(logits, indices, dweights[:, :1], True)
    with pytest.raises(ValueError, match="shape of logits"):
        moe_route_backward(logits, indices, dweights, True, np.zeros((5, 3), np.float32))
    with pytest.raises(ValueError, match="indices must be"):
        moe_route_backward(logits, np.zeros((5, 5), np.int64), np.zeros((5, 5), np.float32), True)
    with pytest.raises(ValueError, match="out of range"):
        moe_route_backward(logits, indices + 4, dweights, True)


# -- gradients of mixture-of-experts layers -----------------------------------------------------------------------


def test_experts_no_token_reaches_get_zero_gradients(model_path):  # noqa: F811
    model = ModelFile(model_path)
    config = model.config
    _, grads = DecoderGradients(config).loss_and_gradients(model.tensors, [5], [7])
    engine = DllmEngine.from_model_file(model_path)
    routed = trace(engine.model, [5], attention=False, logits=False).experts
    for layer in range(config.layers):
        if not config.is_sparse(layer):
            continue
        for expert in range(config.experts):
            gradient = grads[f"layers.{layer}.mlp.experts.{expert}.down.weight"]
            assert bool(np.any(gradient)) == (expert in routed[layer, 0].tolist())


# -- the router load-balancing loss -------------------------------------------------------------------------------


def transformers_router_loss(engine: DllmEngine, tokens: list[int]) -> float:
    """``load_balancing_loss_func`` of transformers in float64, from the traced residual streams."""
    model, config = engine.model, engine.model.config
    traced = trace(model, tokens, attention=False, logits=False)
    probabilities, chosen = [], []
    for layer in range(config.layers):
        if not config.is_sparse(layer):
            continue
        x = traced.middle[layer].astype(np.float64)
        weight = model.tensors[f"layers.{layer}.mlp_norm.weight"].numpy().astype(np.float64)
        h = x / np.sqrt((x * x).mean(-1, keepdims=True) + config.rms_norm_eps) * weight
        logits = h @ model.tensors[f"layers.{layer}.mlp.router.weight"].numpy().astype(np.float64).T
        shifted = np.exp(logits - logits.max(-1, keepdims=True))
        probabilities.append(shifted / shifted.sum(-1, keepdims=True))
        chosen.append(traced.experts[layer])
    routing = np.concatenate(probabilities)
    mask = np.eye(config.experts)[np.concatenate(chosen)]  # [T, k, E]
    return float((mask.mean(0) * routing.mean(0)[None]).sum() * config.experts)


def test_router_loss_is_the_transformers_formula(model_path):  # noqa: F811
    model = ModelFile(model_path)
    engine = DllmEngine.from_model_file(model_path)
    gradients = DecoderGradients(model.config)
    loss = gradients.router_loss(model.tensors, TOKENS)
    assert loss == pytest.approx(transformers_router_loss(engine, TOKENS), rel=1e-5)
    ce, router, _ = gradients.losses_and_gradients(model.tensors, TOKENS, TARGETS)
    assert router == loss and ce == gradients.loss_and_gradients(model.tensors, TOKENS, TARGETS)[0]


def test_router_loss_gradient_matches_finite_differences(model_path):  # noqa: F811
    model = ModelFile(model_path)
    config = model.config
    gradients = DecoderGradients(config)
    weights = {name: np.array(values, dtype=np.float32) for name, values in model.tensors.items()}
    # scale 0: only the load-balancing loss has a gradient.
    _, _, grads = gradients.losses_and_gradients(weights, TOKENS, TARGETS, scale=0.0, router_scale=1.0)
    layer = max(index for index in range(config.layers) if config.is_sparse(index))
    random = numerics.DeterministicRandom(3)
    for name in (f"layers.{layer}.mlp.router.weight", "layers.0.attention.q.weight"):
        index = np.unravel_index(random.next_u64() % weights[name].size, weights[name].shape)
        eps = 1e-2

        def at(delta, name=name, index=index):
            changed = dict(weights)
            changed[name] = weights[name].copy()
            changed[name][index] += np.float32(delta)
            return gradients.router_loss(changed, TOKENS)

        numeric = (at(eps) - at(-eps)) / (2 * eps)
        assert float(grads[name][index]) == pytest.approx(numeric, rel=5e-2, abs=1e-4), name
    assert not np.any(grads["final_norm.weight"])


def test_router_loss_runs_are_reproducible(model_path, tmp_path):  # noqa: F811
    model = ModelFile(model_path)
    run = RunConfig(4, 2, 8, 1, AdamWConfig(1e-2), router_aux_loss=0.5)
    first = FineTuner.from_model_file(model, data(), run)
    first.train()
    plain = FineTuner.from_model_file(model, data(), RunConfig(4, 2, 8, 1, AdamWConfig(1e-2)))
    plain.train()
    assert first.losses[0] > plain.losses[0]  # the load-balancing term is positive
    resumed = FineTuner.from_model_file(model, data(), run)
    resumed.train(until=2)
    resumed.save_checkpoint(tmp_path / "middle.dllmckpt")
    resumed = FineTuner.load_checkpoint(tmp_path / "middle.dllmckpt", data())
    assert resumed.run == run
    resumed.train()
    assert resumed.losses == first.losses
    assert resumed.export(tmp_path / "b.dllm") == first.export(tmp_path / "a.dllm")
    assert ModelFile(tmp_path / "a.dllm").fine_tuning["run"]["router_aux_loss"] == 0.5
    assert "router_aux_loss" not in RunConfig(4).to_dict()
    assert RunConfig.from_dict(run.to_dict()) == run


def test_router_loss_settings_are_checked(model_path, tmp_path):  # noqa: F811
    for value in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="at least 0"):
            RunConfig(4, router_aux_loss=value)
    with pytest.raises(ValueError, match="not to DPO"):
        RunConfig(4, objective="dpo", router_aux_loss=0.1)
    write_hf_checkpoint(tmp_path / "dense", tiny_config("llama"))
    import_model(tmp_path / "dense", tmp_path / "dense.dllm")
    with pytest.raises(ValueError, match="only mixture-of-experts"):
        FineTuner.from_model_file(ModelFile(tmp_path / "dense.dllm"), data(), RunConfig(2, 2, 8, router_aux_loss=0.1))
    dense = DecoderGradients(ModelFile(tmp_path / "dense.dllm").config)
    assert dense.router_loss(ModelFile(tmp_path / "dense.dllm").tensors, TOKENS) == 0.0


def test_cli_router_loss_receipt_replays(model_path, tmp_path, capsys, monkeypatch):  # noqa: F811
    monkeypatch.delenv("DLLM_MODEL", raising=False)
    text = tmp_path / "data.txt"
    text.write_text("the quick brown fox jumps over the lazy dog\n" * 4, encoding="utf-8")
    receipt = tmp_path / "train.json"
    args = ["finetune", str(model_path), "--data", str(text), "-o", str(tmp_path / "out.dllm"), "--steps", "2"]
    args += ["--batch-size", "2", "--sequence-length", "8", "--router-aux-loss", "0.02", "--receipt", str(receipt)]
    assert main(args) == 0
    assert json.loads(receipt.read_text(encoding="utf-8"))["run"]["router_aux_loss"] == 0.02
    capsys.readouterr()
    assert main(["replay", str(receipt), "--base", str(model_path)]) == 0
    assert "training again gave the same weights" in capsys.readouterr().out
    assert main([*args[:6], "--router-aux-loss", "-1"]) == 1
    assert "at least 0" in capsys.readouterr().err


# -- LoRA adapters on the experts ---------------------------------------------------------------------------------


def test_lora_adapts_every_expert_and_writes_peft_names(model_path, tmp_path):  # noqa: F811
    model = ModelFile(model_path)
    config = model.config
    lora = LoraConfig(2, 4.0, ("q", "gate", "down"))
    names = lora_module.target_weights(config, lora)
    sparse = [layer for layer in range(config.layers) if config.is_sparse(layer)]
    experts = range(config.experts)
    expected = {
        f"layers.{layer}.mlp.experts.{e}.{t}.weight" for layer in sparse for e in experts for t in ("gate", "down")
    }
    assert {name for name in names if ".experts." in name} == expected
    assert not any("router" in name for name in names)
    adapters = lora_module.init_adapters(config, lora, 1)
    for name in adapters:
        adapters[name] = adapters[name] + np.float32(0.01)
    lora_module.write_peft(tmp_path, adapters, lora, config=config)
    settings = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    keys = SafetensorsFile(tmp_path / "adapter_model.safetensors").names()
    if config.family == "mixtral":
        assert settings["target_modules"] == ["q_proj", "w1", "w2"]
        assert f"base_model.model.model.layers.{sparse[0]}.block_sparse_moe.experts.1.w2.lora_B.weight" in keys
    else:
        assert settings["target_modules"] == ["down_proj", "gate_proj", "q_proj"]
        assert f"base_model.model.model.layers.{sparse[0]}.mlp.experts.1.down_proj.lora_B.weight" in keys
    read, read_adapters = lora_module.read_peft(tmp_path, config)
    assert read == lora and list(read_adapters) == tensor_order(adapters)
    assert all(read_adapters[name].tobytes() == adapters[name].tobytes() for name in adapters)
    merged = lora_module.merged_weights(config, model.tensors, adapters, lora)
    name = f"layers.{sparse[0]}.mlp.experts.0.gate.weight"
    assert merged[name].tobytes() != np.asarray(model.tensors[name]).tobytes()


def write_one_tensor(directory, key: str, rank: int = 2) -> None:
    from test_lora_errors import write_raw_safetensors

    directory.mkdir(exist_ok=True)
    config = {"peft_type": "LORA", "r": rank, "lora_alpha": 4, "target_modules": ["w1"]}
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    write_raw_safetensors(directory / "adapter_model.safetensors", {key: ("F32", np.zeros((rank, 16), np.float32))})


def test_expert_adapters_must_fit_the_model(model_path, tmp_path):  # noqa: F811
    config = ModelFile(model_path).config
    sparse = next(layer for layer in range(config.layers) if config.is_sparse(layer))
    parent = "block_sparse_moe" if config.family == "mixtral" else "mlp"
    gate = "w1" if config.family == "mixtral" else "gate_proj"
    other = "mlp" if config.family == "mixtral" else "block_sparse_moe"
    cases = {
        f"model.layers.{sparse}.{parent}.experts.{config.experts}.{gate}.lora_A.weight": "does not fit",
        f"model.layers.{sparse}.{other}.experts.0.{gate}.lora_A.weight": "does not fit",
        f"model.layers.{sparse}.{parent}.experts.0.router.lora_A.weight": "unsupported",
        f"model.layers.{sparse}.mlp.gate_proj.lora_A.weight": "does not fit",  # a sparse layer has no single MLP
    }
    for index, (key, message) in enumerate(cases.items()):
        directory = tmp_path / str(index)
        write_one_tensor(directory, key)
        with pytest.raises(AdapterError, match=message):
            lora_module.read_peft(directory, config)
    if config.dense_layers:
        dense = config.dense_layers[0]
        write_one_tensor(tmp_path / "dense", f"model.layers.{dense}.mlp.experts.0.gate_proj.lora_A.weight")
        with pytest.raises(AdapterError, match="does not fit"):
            lora_module.read_peft(tmp_path / "dense", config)


# -- distillation -------------------------------------------------------------------------------------------------


def test_distilling_into_a_mixture_of_experts_student(model_path, tmp_path, capsys, monkeypatch):  # noqa: F811
    monkeypatch.delenv("DLLM_MODEL", raising=False)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("ab\nba\n", encoding="utf-8")
    out, receipt = tmp_path / "student.dllm", tmp_path / "student.train.json"
    args = ["distill", str(model_path), "--teacher", str(model_path), "--prompts", str(prompts), "-o", str(out)]
    args += ["--steps", "2", "--batch-size", "1", "--sequence-length", "8", "--teacher-max-tokens", "4"]
    assert main([*args, "--lora-rank", "2", "--receipt", str(receipt)]) == 0
    capsys.readouterr()
    assert ModelFile(out).fine_tuning["distillation"]["teacher"] == ModelFile(model_path).fingerprint
    assert main(["replay", str(receipt), "--base", str(model_path), "--teacher", str(model_path)]) == 0
    assert "verified" in capsys.readouterr().out


# -- model editing ------------------------------------------------------------------------------------------------


def test_rome_edits_the_routed_expert(model_path, tmp_path):  # noqa: F811
    engine = DllmEngine.from_model_file(model_path)
    model, tokenizer, config = engine.model, engine.tokenizer, engine.model.config
    index = next(layer for layer in range(config.layers) if config.is_sparse(layer))
    request = EditRequest("so the cat", "cat", " mat")  # the subject ends the prompt: the last layer can edit it
    corpus = ["hello there, general", "the end of the story", "a b c d e f g"]
    result = rome(model, tokenizer, request, layer=index + 1, corpus=corpus, steps=20)
    record = result.record
    tokens, position = subject_position(tokenizer, request.prompt, request.subject)
    traced = trace(model, tokens)
    expert = int(traced.experts[index, position, 0])
    assert record["expert"] == expert
    assert record["routing_weight"] == float(traced.expert_weights[index, position, 0])
    name = f"layers.{index}.mlp.experts.{expert}.down.weight"
    changed = [n for n in result.weights if result.weights[n].tobytes() != model.tensors[n].numpy().tobytes()]
    assert changed == [name]
    # The expert's output for the key changes by delta / weight, so the layer's output changes by delta.
    key = model.mlp_activation(traced.middle[index], index, expert)[position]
    before = numerics.linear(key[None], model.tensors[name].numpy()).numpy()[0]
    after = numerics.linear(key[None], result.weights[name]).numpy()[0]
    moved = float(np.sqrt(numerics.sum_squares(after - before))) * record["routing_weight"]
    assert moved == pytest.approx(record["delta_norm"], rel=1e-3)
    again = rome(model, tokenizer, request, layer=index + 1, corpus=corpus, steps=20)
    assert again.record == record
    base = ModelFile(model_path)
    assert write_edited_model(base, result, tmp_path / "a.dllm") == write_edited_model(base, again, tmp_path / "b.dllm")
    assert ModelFile(tmp_path / "a.dllm").edits == [record]
    if config.dense_layers:  # a dense layer of a mixture-of-experts model edits its one MLP
        dense = rome(model, tokenizer, request, layer=config.dense_layers[0] + 1, corpus=corpus, steps=2)
        assert "expert" not in dense.record


def test_mlp_activation_checks_the_layer(model_path):  # noqa: F811
    engine = DllmEngine.from_model_file(model_path)
    model, config = engine.model, engine.model.config
    sparse = next(layer for layer in range(config.layers) if config.is_sparse(layer))
    middle = trace(model, TOKENS).middle[sparse]
    with pytest.raises(ValueError, match="name one of its"):
        model.mlp_activation(middle, sparse)
    with pytest.raises(ValueError, match="name one of its"):
        model.mlp_activation(middle, sparse, config.experts)
    if config.dense_layers:
        dense = config.dense_layers[0]
        with pytest.raises(ValueError, match="has no experts"):
            model.mlp_activation(middle, dense, 0)
        assert model.mlp_activation(middle, dense).shape == (len(TOKENS), config.intermediate_size)


def test_edit_command_names_the_expert(model_path, tmp_path, capsys, monkeypatch):  # noqa: F811
    monkeypatch.delenv("DLLM_MODEL", raising=False)
    config = ModelFile(model_path).config
    layer = next(index for index in range(config.layers) if config.is_sparse(index)) + 1
    arguments = ["edit", str(model_path), "--prompt", "so the cat", "--subject", "cat", "--target", " mat"]
    assert main([*arguments, "--layer", str(layer), "--steps", "2", "-o", str(tmp_path / "edited.dllm")]) == 0
    output = capsys.readouterr().out
    assert f"edited:             layer {layer}, expert " in output and "(weight 0." in output
