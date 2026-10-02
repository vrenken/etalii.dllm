"""LoRA fine-tuning on a quantised base (Phase 45): the frozen matrices stay Q8_0/Q4_0 in memory, and the run is
exactly LoRA on the dequantised base, so it gives the bits of a float run on those weights; checkpoints, receipts,
merged exports, adapter imports, DPO and distillation all carry the quantisation.

The tiny models here have a hidden size of 32, so their attention and MLP matrices fill whole 32-value blocks.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from golden_values import QLORA_FINETUNE_FINGERPRINT
from model_fixtures import tiny_config, write_hf_checkpoint
from test_model_building import BYTE_LEVEL_TOKENIZER
from test_preference import RECORDS
from test_training import ascii_data

from etalii_dllm.cli import main
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.lora import LoraConfig, QuantizedBase, dequantized_weights, quantizable
from etalii_dllm.modelfile import ModelFile, data_fingerprint
from etalii_dllm.numerics import QuantizedWeight
from etalii_dllm.training import AdamWConfig, FineTuner, RunConfig
from etalii_dllm.training.preference import PreferenceData

LORA = LoraConfig(2, 4.0)
RUN = RunConfig(4, 2, 8, 3, AdamWConfig(learning_rate=3e-2), LORA, base_quantize="q4_0")
FAMILIES = ["gemma2", "granite", "llama", "qwen2_moe", "qwen3", "qwen3_moe"]


def _model(directory, family: str):
    config = {**tiny_config(family), "hidden_size": 32, "vocab_size": 264}
    write_hf_checkpoint(directory / "checkpoint", config, tokenizer_json=BYTE_LEVEL_TOKENIZER)
    import_model(directory / "checkpoint", directory / "base.dllm")
    return ModelFile(directory / "base.dllm")


@pytest.fixture(scope="module", params=FAMILIES)
def base(request, tmp_path_factory) -> ModelFile:
    return _model(tmp_path_factory.mktemp(f"qlora-{request.param}"), request.param)


@pytest.fixture(scope="module")
def llama(tmp_path_factory) -> ModelFile:
    return _model(tmp_path_factory.mktemp("qlora-llama"), "llama")


def float_twin(model: ModelFile, data, run: RunConfig) -> FineTuner:
    """The same run as a plain LoRA run on the dequantised base."""
    return FineTuner(
        model.config,
        dequantized_weights(model.tensors, run.base_quantize),
        data,
        dataclasses.replace(run, base_quantize=None),
        base_fingerprint=model.fingerprint,
        metadata=model.header,
    )


# -- the quantised base -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["q8_0", "q4_0"])
def test_the_base_is_its_dequantized_weights(llama, kind):
    frozen = QuantizedBase(llama.tensors, kind)
    eager = dequantized_weights(llama.tensors, kind)
    assert list(frozen) == list(llama.tensors) and len(frozen) == len(eager)
    held = [name for name in llama.tensors if quantizable(name, np.shape(llama.tensors[name]))]
    assert "layers.0.attention.q.weight" in held and "layers.0.mlp.down.weight" in held
    assert not any(name.endswith(("norm.weight", "embedding.weight")) for name in held)
    for name in llama.tensors:
        assert frozen[name].dtype == np.float32 and frozen[name].tobytes() == eager[name].tobytes()
        if name in held:
            assert frozen[name].tobytes() == QuantizedWeight(llama.tensors[name], kind).dequantize().tobytes()
            assert frozen[name].tobytes() != np.asarray(llama.tensors[name]).tobytes()
        else:
            assert frozen[name].tobytes() == np.asarray(llama.tensors[name]).tobytes()
    ratio = frozen.nbytes / frozen.float_nbytes
    assert ratio == pytest.approx(36 / 128 if kind == "q8_0" else 20 / 128)  # per block: 32 or 16 bytes, a scale
    with pytest.raises(ValueError, match="q8_0, q4_0"):
        QuantizedBase(llama.tensors, "q2_k")
    with pytest.raises(ValueError, match="q8_0, q4_0"):
        dequantized_weights(llama.tensors, "f16")


def test_run_settings(llama):
    assert RUN.to_dict()["base_quantize"] == "q4_0" and RunConfig.from_dict(RUN.to_dict()) == RUN
    assert "base_quantize" not in dataclasses.replace(RUN, base_quantize=None).to_dict()
    with pytest.raises(ValueError, match="needs a LoRA run"):
        RunConfig(2, lora=None, base_quantize="q8_0")
    with pytest.raises(ValueError, match="unknown base quantisation"):
        RunConfig(2, lora=LORA, base_quantize="q5_k")


# -- runs ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["q8_0", "q4_0"])
def test_a_quantized_run_is_lora_on_the_dequantized_base(base, kind, tmp_path):
    run = dataclasses.replace(RUN, base_quantize=kind)
    quantized = FineTuner.from_model_file(base, ascii_data(), run)
    assert isinstance(quantized.base, QuantizedBase) and quantized.base.kind == kind
    quantized.train()
    twin = float_twin(base, ascii_data(), run)
    twin.train()
    assert quantized.losses == twin.losses
    assert all(quantized.params[name].tobytes() == twin.params[name].tobytes() for name in twin.params)
    assert data_fingerprint(quantized.weights()) == data_fingerprint(twin.weights())
    fingerprint = quantized.export(tmp_path / "merged.dllm")
    assert fingerprint == data_fingerprint(twin.weights())
    exported = ModelFile(tmp_path / "merged.dllm")
    assert exported.fine_tuning["run"]["base_quantize"] == kind
    assert exported.lineage[-1]["step"] == "fine_tune" and exported.lineage[-1]["base_quantize"] == kind
    if kind == "q4_0":
        assert fingerprint == QLORA_FINETUNE_FINGERPRINT[base.config.family]


def test_checkpoints_resume_bit_for_bit(llama, tmp_path):
    first = FineTuner.from_model_file(llama, ascii_data(), RUN)
    first.train()
    first.save_checkpoint(tmp_path / "first.dllmckpt")
    second = FineTuner.from_model_file(llama, ascii_data(), RUN)
    second.train(until=2)
    second.save_checkpoint(tmp_path / "middle.dllmckpt")
    resumed = FineTuner.load_checkpoint(tmp_path / "middle.dllmckpt", ascii_data(), llama)
    assert resumed.run.base_quantize == "q4_0" and isinstance(resumed.base, QuantizedBase)
    resumed.train()
    resumed.save_checkpoint(tmp_path / "resumed.dllmckpt")
    assert (tmp_path / "resumed.dllmckpt").read_bytes() == (tmp_path / "first.dllmckpt").read_bytes()


def test_dpo_on_a_quantized_base(llama):
    data = PreferenceData.from_records(RECORDS, lambda text: list(text.encode()), 24, 2)
    run = RunConfig(3, 2, 24, 3, AdamWConfig(learning_rate=1e-2), LORA, "dpo", 0.5, base_quantize="q8_0")
    quantized = FineTuner.from_model_file(llama, data, run)
    quantized.train()
    twin = float_twin(llama, data, run)
    assert quantized.reference == twin.reference  # the reference is the dequantised base
    twin.train()
    assert quantized.losses == twin.losses
    assert all(quantized.params[name].tobytes() == twin.params[name].tobytes() for name in twin.params)


# -- the command line: receipts, adapters, imports and distillation -----------------------------------------------


def test_cli_quantized_finetune_replays_and_imports(llama, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DLLM_MODEL", raising=False)
    lines = [json.dumps({"text": f"{i} plus {i} is {2 * i}."}) for i in range(8)]
    (tmp_path / "data.jsonl").write_text("\n".join(lines), encoding="utf-8")
    common = ["finetune", str(llama.path), "--data", str(tmp_path / "data.jsonl"), "--steps", "3", "--batch-size"]
    common += ["2", "--sequence-length", "16", "--learning-rate", "1e-2", "--lora-rank", "2", "--base-quantize"]
    outputs = ["--adapter-output", str(tmp_path / "adapter"), "-o", str(tmp_path / "merged.dllm")]
    assert main([*common, "q4_0", *outputs, "--receipt", str(tmp_path / "run.json")]) == 0
    out = capsys.readouterr().out
    assert "base:               q4_0, " in out and "MiB of matrices (float32: " in out
    merged = ModelFile(tmp_path / "merged.dllm")
    assert json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))["run"]["base_quantize"] == "q4_0"
    assert main(["replay", str(tmp_path / "run.json"), "--base", str(llama.path)]) == 0
    assert "verified" in capsys.readouterr().out

    # The adapter merged into the dequantised base is the exported model, bit for bit.
    (tmp_path / "adapter" / "README.md").write_text("---\nlicense: mit\n---\n", encoding="utf-8")
    again = ["import", str(tmp_path / "adapter"), "--base", str(llama.path), "-o", str(tmp_path / "again.dllm")]
    assert main([*again, "--base-quantize", "q4_0"]) == 0
    assert f"system_fingerprint: {merged.fingerprint}" in capsys.readouterr().out
    assert ModelFile(tmp_path / "again.dllm").adapter["base_quantize"] == "q4_0"
    plain = import_model(tmp_path / "adapter", tmp_path / "plain.dllm", base=llama.path)
    assert plain.fingerprint != merged.fingerprint  # merged into the float base instead
    with pytest.raises(ModelImportError, match="--base-quantize is for merging"):
        import_model(llama.path.parent / "checkpoint", tmp_path / "x.dllm", base_quantize="q8_0")
    full = ["finetune", str(llama.path), "--data", str(tmp_path / "data.jsonl"), "-o", str(tmp_path / "x.dllm")]
    assert main([*full, "--base-quantize", "q8_0"]) == 1
    assert "needs a LoRA run" in capsys.readouterr().err


def test_cli_distills_into_a_quantized_student(llama, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DLLM_MODEL", raising=False)
    (tmp_path / "prompts.txt").write_text("ab\nba\n", encoding="utf-8")
    args = ["distill", str(llama.path), "--teacher", str(llama.path), "--prompts", str(tmp_path / "prompts.txt")]
    args += ["-o", str(tmp_path / "student.dllm"), "--steps", "2", "--batch-size", "1", "--sequence-length", "8"]
    args += ["--teacher-max-tokens", "4", "--lora-rank", "2", "--base-quantize", "q8_0"]
    assert main([*args, "--receipt", str(tmp_path / "student.json")]) == 0
    capsys.readouterr()
    student = ModelFile(tmp_path / "student.dllm")
    assert student.fine_tuning["run"]["base_quantize"] == "q8_0"
    assert student.fine_tuning["distillation"]["teacher"] == llama.fingerprint
    replay = ["replay", str(tmp_path / "student.json"), "--base", str(llama.path), "--teacher", str(llama.path)]
    assert main(replay) == 0
    assert "verified" in capsys.readouterr().out
