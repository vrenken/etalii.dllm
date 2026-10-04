"""Fine-tuning and exporting T5 text-to-text models (Phase 65): the decoder's gradients against transformers'
autograd, golden fine-tuning runs and their resumption, ``dllm finetune``, LoRA under T5ForConditionalGeneration's
module names and the export to safetensors."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from test_t5 import TEXTS
from test_t5_generation import FLAN, write_checkpoint

from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import import_model
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.training.data import TrainingDataError
from etalii_dllm.training.seq2seq_backprop import TextToTextGradients
from etalii_dllm.training.seq2seq_data import TextToTextData, read_text_pairs

SEQUENCE_LENGTH = 24
PAIRS = [
    {"input": "translate: the quick brown fox", "target": TEXTS[1]},
    {"prompt": "summarize: naive cafe", "completion": TEXTS[2]},
    {"messages": [{"role": "user", "content": "same bits?"}, {"role": "assistant", "content": TEXTS[3]}]},
    {"input": TEXTS[0], "target": "a fox"},
]


@pytest.fixture(scope="module")
def t5(tmp_path_factory) -> tuple[Path, DllmEngine]:
    directory = tmp_path_factory.mktemp("t5-tune")
    write_checkpoint(directory / "checkpoint")
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-t5-text")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


@pytest.fixture(scope="module")
def flan(tmp_path_factory) -> tuple[Path, DllmEngine]:
    directory = tmp_path_factory.mktemp("flan-tune")
    write_checkpoint(directory / "checkpoint", **FLAN)
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-flan-t5")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


def write_pairs(path: Path) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in PAIRS), encoding="utf-8")
    return path


def weights_of(engine_or_file) -> dict[str, np.ndarray]:
    return {name: np.asarray(values) for name, values in engine_or_file.tensors.items()}


# Gradients (#387)


def test_training_logits_are_the_served_bits(t5, flan):
    for checkpoint, engine in (t5, flan):
        file = ModelFile(checkpoint.parent / "model.dllm")
        gradients = TextToTextGradients(file.config)
        source = [*engine.tokenizer.encode(TEXTS[0]), 1]
        target = [*engine.tokenizer.encode(TEXTS[1])[:20], 1]  # past the buckets' maximum distance
        logits = gradients.logits(weights_of(file), source, target)
        for step in range(len(target)):
            assert logits[step].tobytes() == engine.model.forward([*source, *target[:step]]).tobytes(), step


def test_gradients_match_transformers(t5, flan):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from etalii_dllm.importing.importer import _t5_text_to_text_name

    for checkpoint, engine in (t5, flan):
        file = ModelFile(checkpoint.parent / "model.dllm")
        source = [*engine.tokenizer.encode(TEXTS[0]), 1]
        target = [*engine.tokenizer.encode(TEXTS[1])[:20], 1]
        loss, ours = TextToTextGradients(file.config).loss_and_gradients(
            weights_of(file), source, target, scale=1 / len(target)
        )
        model = transformers.T5ForConditionalGeneration.from_pretrained(checkpoint).double()
        output = model(input_ids=torch.tensor([source]), labels=torch.tensor([target]))
        output.loss.backward()
        assert abs(loss / len(target) - output.loss.item()) < 1e-5
        tied = file.config.tie_word_embeddings
        theirs = {
            _t5_text_to_text_name(name, tied): p.grad.numpy()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        assert set(ours) == set(theirs) == set(file.config.tensor_shapes())
        for name, expected in theirs.items():
            scale = max(float(np.abs(expected).max()), 1e-3)
            assert np.abs(ours[name] - expected).max() <= 1e-4 * scale, name


def test_gradient_refusals(t5):
    from etalii_dllm.architecture import TransformerConfig

    config = t5[1].model.config
    gradients = TextToTextGradients(config)
    with pytest.raises(ValueError, match="at least one"):
        gradients.loss_and_gradients(weights_of(t5[1].model), [5, 1], [])
    with pytest.raises(ValueError, match="out of range"):
        gradients.forward(weights_of(t5[1].model), [5, 1], [0, config.vocabulary_size])
    with pytest.raises(ValueError, match="at least one"):
        gradients.forward(weights_of(t5[1].model), [5, 1], [])
    encoder = TransformerConfig.from_dict({**config.to_dict(), "decoder_layers": 0})
    with pytest.raises(ValueError, match="not a text-to-text"):
        TextToTextGradients(encoder)


# Data and fine-tuning (#388)


def test_text_pairs(t5, tmp_path):
    engine = t5[1]
    pairs = read_text_pairs(write_pairs(tmp_path / "pairs.jsonl"))
    assert pairs[2] == ("same bits?", TEXTS[3])
    data = TextToTextData.from_pairs(pairs, engine.tokenizer.encode, SEQUENCE_LENGTH, 1)
    assert len(data) == 4 and all(s[-1] == 1 and t[-1] == 1 for s, t in data.examples)
    assert max(len(t) for _, t in data.examples) == SEQUENCE_LENGTH
    again = TextToTextData.from_pairs(pairs, lambda text: [*engine.tokenizer.encode(text), 1], SEQUENCE_LENGTH, 1)
    assert again.fingerprint == data.fingerprint  # a tokenizer that adds </s> itself gives the same examples
    assert data.batch(0, 2, 1) == data.batch(0, 2, 1) and len(data.batch(3, 3, 1)) == 3
    for text, message in (
        ('{"text": "x"}\n', "expected an object"),
        ("not json\n", "invalid JSON"),
        ('{"messages": [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}]}\n', "expected"),
        ("\n", "no examples"),
    ):
        (tmp_path / "bad.jsonl").write_text(text, encoding="utf-8")
        with pytest.raises(TrainingDataError, match=message):
            read_text_pairs(tmp_path / "bad.jsonl")
    with pytest.raises(TrainingDataError, match=r"No such file|Errno"):
        read_text_pairs(tmp_path / "missing.jsonl")
    with pytest.raises(TrainingDataError, match="at least 2"):
        TextToTextData.from_pairs(pairs, engine.tokenizer.encode, 1, 1)
    with pytest.raises(TrainingDataError, match="no examples"):
        TextToTextData.from_pairs([], engine.tokenizer.encode, SEQUENCE_LENGTH, 1)


def test_finetune_runs_are_golden(t5, flan, tmp_path):
    from golden_values import T5_TEXT_FINETUNE_FINGERPRINT

    from etalii_dllm.training import AdamWConfig, FineTuner, RunConfig

    for kind, (checkpoint, engine) in (("t5", t5), ("flan", flan)):
        pairs = write_pairs(tmp_path / "pairs.jsonl")
        data = TextToTextData.from_file(pairs, engine.tokenizer.encode, SEQUENCE_LENGTH, 1)
        run = RunConfig(3, 2, SEQUENCE_LENGTH, 1, AdamWConfig(1e-3))
        model_file = ModelFile(checkpoint.parent / "model.dllm")
        first = FineTuner.from_model_file(model_file, data, run)
        first.train()
        assert first.losses[-1] < first.losses[0]
        second = FineTuner.from_model_file(model_file, data, run)
        second.train(until=1)
        second.save_checkpoint(tmp_path / f"{kind}.dllmckpt")
        resumed = FineTuner.load_checkpoint(tmp_path / f"{kind}.dllmckpt", data)
        resumed.train()
        assert resumed.losses == first.losses
        fingerprint = first.export(tmp_path / f"{kind}.dllm")
        assert resumed.export(tmp_path / f"{kind}-resumed.dllm") == fingerprint
        assert fingerprint == T5_TEXT_FINETUNE_FINGERPRINT[kind]
        tuned = ModelFile(tmp_path / f"{kind}.dllm")
        for name in ("relative_bias.weight", "decoder.relative_bias.weight", "decoder.layers.0.cross.k.weight"):
            assert (np.asarray(tuned.tensors[name]) != np.asarray(model_file.tensors[name])).any(), name
        assert tuned.config == model_file.config
        tuned_engine = DllmEngine.from_model_file(tmp_path / f"{kind}.dllm")
        greedy = SamplingOptions(temperature=0.0)
        assert tuned_engine.model.forward([5, 1]).tobytes() != engine.model.forward([5, 1]).tobytes()
        assert tuned_engine.complete(TEXTS[0], 4, greedy).fingerprint


def test_finetune_refusals(t5, tmp_path):
    from etalii_dllm.training import AdamWConfig, FineTuner, RunConfig
    from etalii_dllm.training.data import TrainingData

    checkpoint, engine = t5
    model_file = ModelFile(checkpoint.parent / "model.dllm")
    data = TextToTextData.from_file(write_pairs(tmp_path / "pairs.jsonl"), engine.tokenizer.encode, SEQUENCE_LENGTH, 1)
    windows = TrainingData.from_documents([TEXTS[0]], engine.tokenizer.encode, SEQUENCE_LENGTH, 1)
    with pytest.raises(ValueError, match="source and target pairs"):
        FineTuner.from_model_file(model_file, windows, RunConfig(1, 1, SEQUENCE_LENGTH))
    with pytest.raises(ValueError, match="language-model objective"):
        FineTuner.from_model_file(model_file, data, RunConfig(1, 1, SEQUENCE_LENGTH, objective="dpo"))
    from test_modelfile import CONFIG as DECODER  # a decoder cannot train on pairs

    with pytest.raises(ValueError, match="train text-to-text models"):
        run = RunConfig(1, 1, SEQUENCE_LENGTH, AdamWConfig(1e-3))
        FineTuner(DECODER, {}, data, run, base_fingerprint="", metadata={})


def test_finetune_command(flan, tmp_path, capsys):
    from etalii_dllm.cli import main as cli

    checkpoint, _ = flan
    path = checkpoint.parent / "model.dllm"
    pairs = write_pairs(tmp_path / "pairs.jsonl")
    command = ["finetune", str(path), "--data", str(pairs), "--steps", "2", "--batch-size", "2"]
    assert cli([*command, "-o", str(tmp_path / "tuned.dllm"), "--receipt", str(tmp_path / "receipt.json")]) == 0
    out = capsys.readouterr().out
    assert "4 examples" in out
    assert ModelFile(tmp_path / "tuned.dllm").config.is_text_to_text
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["data"]["examples"] == 4
    assert cli(["replay", str(tmp_path / "receipt.json"), "--base", str(path)]) == 0
    capsys.readouterr()
    assert cli([*command, "--dpo", "-o", str(tmp_path / "dpo.dllm")]) == 1
    assert "language-model objective" in capsys.readouterr().err


# LoRA (#389)


def test_lora_on_text_to_text(t5, flan, tmp_path, capsys):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from etalii_dllm.cli import main as cli
    from etalii_dllm.importing.safetensors import SafetensorsFile
    from etalii_dllm.lora import AdapterError, LoraConfig, read_peft, target_modules, target_weights

    checkpoint, engine = flan
    path = checkpoint.parent / "model.dllm"
    pairs = write_pairs(tmp_path / "pairs.jsonl")
    common = ["finetune", str(path), "--data", str(pairs), "--steps", "2", "--batch-size", "2"]
    common += ["--learning-rate", "1e-2", "--lora-rank", "2"]
    assert cli([*common, "-o", str(tmp_path / "merged.dllm"), "--adapter-output", str(tmp_path / "adapter")]) == 0
    capsys.readouterr()
    settings = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert settings["task_type"] == "SEQ_2_SEQ_LM"
    stored = {t.name: t.to_float32() for t in SafetensorsFile(tmp_path / "adapter" / "adapter_model.safetensors")}
    assert stored["base_model.model.decoder.block.0.layer.1.EncDecAttention.k.lora_A.weight"].shape == (2, 32)
    assert stored["base_model.model.decoder.block.1.layer.2.DenseReluDense.wi_0.lora_B.weight"].shape == (64, 2)
    model = transformers.T5ForConditionalGeneration.from_pretrained(checkpoint)
    modules = {name for name, _ in model.named_modules()}
    adapted = {key.removeprefix("base_model.model.").rsplit(".lora_", 1)[0] for key in stored}
    config = engine.model.config
    assert adapted <= modules and len(adapted) == config.layers * 7 + config.decoder_layers * 11
    assert {m for m in modules if re.fullmatch(settings["target_modules"], m)} == adapted
    source = [*engine.tokenizer.encode(TEXTS[2]), 1, 7]
    merged = DllmEngine.from_model_file(tmp_path / "merged.dllm").model.forward(source)
    loaded = DllmEngine.from_model_file(path, adapter=tmp_path / "adapter").model.forward(source)
    assert merged.tobytes() == loaded.tobytes() != engine.model.forward(source).tobytes()
    import_model(tmp_path / "adapter", tmp_path / "imported.dllm", base=path, licence="mit")
    assert ModelFile(tmp_path / "imported.dllm").fingerprint == ModelFile(tmp_path / "merged.dllm").fingerprint
    lora, adapters = read_peft(tmp_path / "adapter", config)
    assert lora.targets == ("q", "k", "v", "o", "gate", "up", "down") and len(adapters) == 2 * len(adapted)
    # T5 v1.0 has no gate; q and v adapt both attentions, as PEFT's default T5 targets do
    plain = t5[1].model.config
    with pytest.raises(AdapterError, match="gate"):
        target_weights(plain, LoraConfig(2, 4.0, ("gate",)))
    names = target_weights(plain, LoraConfig(2, 4.0, ("q", "up")))
    assert "decoder.layers.2.cross.q.weight" in names and "decoder.layers.0.mlp.up.weight" in names
    pattern = target_modules(plain, LoraConfig(2, 4.0, ("q", "up")))
    t5_modules = {name for name, _ in transformers.T5ForConditionalGeneration.from_pretrained(t5[0]).named_modules()}
    chosen = {m for m in t5_modules if re.fullmatch(pattern, m)}
    assert len(chosen) == plain.layers * 2 + plain.decoder_layers * 3
    for key in (
        "base_model.model.decoder.block.0.layer.2.DenseReluDense.wi_0.lora_A.weight",
        "base_model.model.decoder.block.9.layer.0.SelfAttention.q.lora_A.weight",
        "base_model.model.encoder.block.0.layer.1.EncDecAttention.q.lora_A.weight",
        "base_model.model.encoder.block.5.layer.0.SelfAttention.q.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
    ):
        from etalii_dllm.importing.safetensors import write_safetensors as write_tensors

        (tmp_path / "bad").mkdir(exist_ok=True)
        (tmp_path / "bad" / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 2}), encoding="utf-8")
        write_tensors(tmp_path / "bad" / "adapter_model.safetensors", {key: np.zeros((2, 32), np.float32)})
        with pytest.raises(AdapterError, match=r"unsupported adapter tensor|does not fit"):
            read_peft(tmp_path / "bad", plain)


# Export (#390)


def test_export_round_trips(t5, flan, tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from etalii_dllm.exporting import ExportError, export_gguf, export_safetensors

    for name, (checkpoint, engine) in (("t5", t5), ("flan", flan)):
        original = ModelFile(checkpoint.parent / "model.dllm")
        export_safetensors(original, tmp_path / name)
        exported = json.loads((tmp_path / name / "config.json").read_text())
        assert exported["architectures"] == ["T5ForConditionalGeneration"]
        import_model(tmp_path / name, tmp_path / f"{name}.dllm", repository=f"example/{name}")
        again = ModelFile(tmp_path / f"{name}.dllm")
        assert again.fingerprint == original.fingerprint and again.config == original.config
        model = transformers.T5ForConditionalGeneration.from_pretrained(tmp_path / name).eval()
        source = [*engine.tokenizer.encode(TEXTS[0]), 1]
        with torch.no_grad():
            # without the cache: transformers sizes it by num_layers, which breaks when the decoder is deeper
            options = {"max_new_tokens": 8, "do_sample": False, "num_beams": 1, "use_cache": False}
            greedy = model.generate(torch.tensor([source]), **options)[0]
        answer = engine.complete(TEXTS[0], 8, SamplingOptions(temperature=0.0)).tokens
        assert greedy.tolist()[1 : 1 + len(answer)] == list(answer)
        with pytest.raises(ExportError, match="safetensors only"):
            export_gguf(original, tmp_path / f"{name}.gguf")
