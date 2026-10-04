"""Preference tuning and evaluation of T5 text-to-text models (#397-#400): preference pairs with the prompt as the
source, DPO gradients against a DPO loss on transformers' ``T5ForConditionalGeneration`` through autograd, golden DPO
runs and their resumption, ``dllm finetune --dpo`` with receipts and replay, LoRA on quantised bases and ``dllm eval``
scoring answers given their source."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_t5 import TEXTS
from test_t5_generation import FLAN, write_checkpoint

from etalii_dllm import evaluation, scoring
from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import import_model
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.numerics import log_softmax
from etalii_dllm.training import AdamWConfig, FineTuner, RunConfig
from etalii_dllm.training.data import TrainingDataError
from etalii_dllm.training.preference import PreferenceData, PreferenceRecord
from etalii_dllm.training.receipt import load_data
from etalii_dllm.training.seq2seq_backprop import TextToTextGradients

pytest.importorskip("tokenizers")
pytest.importorskip("sentencepiece")

SEQUENCE_LENGTH = 16
END = 1
PAIRS = [
    {"prompt": "translate: the quick brown fox", "chosen": TEXTS[1], "rejected": "a fox"},
    {
        "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "same bits?"}],
        "chosen": TEXTS[3],
        "rejected": "no",
    },
    {"prompt": TEXTS[0], "chosen": "a fox", "rejected": TEXTS[2]},
]


@pytest.fixture(scope="module")
def t5(tmp_path_factory) -> tuple[Path, DllmEngine]:
    directory = tmp_path_factory.mktemp("t5-dpo")
    write_checkpoint(directory / "checkpoint")
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-t5-dpo")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


@pytest.fixture(scope="module")
def flan(tmp_path_factory) -> tuple[Path, DllmEngine]:
    directory = tmp_path_factory.mktemp("flan-dpo")
    write_checkpoint(directory / "checkpoint", **FLAN)
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-flan-t5-dpo")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


def write_pairs(path: Path, rows=PAIRS) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def model_file(fixture) -> ModelFile:
    return ModelFile(fixture[0].parent / "model.dllm")


def weights_of(file: ModelFile) -> dict[str, np.ndarray]:
    return {name: np.asarray(values) for name, values in file.tensors.items()}


# Preference pairs (#397)


def test_preference_pairs_use_the_prompt_as_source(t5, tmp_path):
    _, engine = t5
    data = load_data(engine, model_file(t5), write_pairs(tmp_path / "pairs.jsonl"), SEQUENCE_LENGTH, "dpo")
    assert isinstance(data, PreferenceData) and len(data) == 3
    encode = engine.tokenizer.encode
    first = data.pairs[0]
    assert first.prompt == (*encode("translate: the quick brown fox")[: SEQUENCE_LENGTH - 1], END)
    assert first.chosen == (*encode(TEXTS[1])[: SEQUENCE_LENGTH - 1], END) and first.rejected == (*encode("a fox"), END)
    assert data.pairs[1].prompt == (
        *encode("be brief\n\nsame bits?"),
        END,
    )  # messages joined as the engine renders them
    assert all(len(side) <= SEQUENCE_LENGTH for pair in data.pairs for side in (pair.prompt, pair.chosen))
    records = [PreferenceRecord("x", "y", "z")]
    adds_end = PreferenceData.from_text_to_text_records(records, lambda t: [*encode(t), END], SEQUENCE_LENGTH, END)
    assert adds_end == PreferenceData.from_text_to_text_records(records, encode, SEQUENCE_LENGTH, END)
    with pytest.raises(TrainingDataError, match="at least 2"):
        PreferenceData.from_text_to_text_records(records, encode, 1, END)
    (tmp_path / "bad.jsonl").write_text('{"input": "x", "target": "y"}\n', encoding="utf-8")
    with pytest.raises(TrainingDataError, match="'chosen' and 'rejected'"):
        load_data(engine, model_file(t5), tmp_path / "bad.jsonl", SEQUENCE_LENGTH, "dpo")
    with pytest.raises(ValueError, match="or DPO objective"):
        load_data(engine, model_file(t5), tmp_path / "bad.jsonl", SEQUENCE_LENGTH, "embedding")


def test_answer_log_probabilities_are_the_served_scores(flan, tmp_path):
    _, engine = flan
    file = model_file(flan)
    data = load_data(engine, file, write_pairs(tmp_path / "pairs.jsonl"), SEQUENCE_LENGTH, "dpo")
    tuner = FineTuner.from_model_file(file, data, RunConfig(1, 1, SEQUENCE_LENGTH, objective="dpo"))
    gradients = TextToTextGradients(file.config)
    for pair, (chosen, rejected) in zip(data.pairs, tuner.reference, strict=True):
        for answer, expected in ((pair.chosen, chosen), (pair.rejected, rejected)):
            logits = gradients.logits(weights_of(file), pair.prompt, answer)
            served = engine.model.answer_logits(pair.prompt, answer)
            assert logits.tobytes() == served.tobytes()
            logprobs = [float(log_softmax(row)[token]) for row, token in zip(served, answer, strict=True)]
            assert abs(sum(logprobs) - expected) < 1e-4
            score = scoring.score_tokens(engine, answer, source=pair.prompt)
            assert [t.logprob for t in score.tokens] == logprobs


# DPO gradients (#398)


def test_dpo_gradients_match_transformers(t5, flan, tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from etalii_dllm.importing.importer import _t5_text_to_text_name

    beta = 0.5
    for fixture in (t5, flan):
        checkpoint, engine = fixture
        file = model_file(fixture)
        data = load_data(engine, file, write_pairs(tmp_path / "pairs.jsonl", PAIRS[:1]), SEQUENCE_LENGTH, "dpo")
        run = RunConfig(1, 1, SEQUENCE_LENGTH, objective="dpo", beta=beta)
        tuner = FineTuner.from_model_file(file, data, run)
        reference_chosen, reference_rejected = tuner.reference[0]
        # Move the weights off the base so that the loss is not at its symmetric point.
        shifted = {name: (w * np.float32(1.01)).astype(np.float32) for name, w in tuner.params.items()}
        tuner.params = shifted
        loss, ours = tuner._preference_gradients(data)
        model = transformers.T5ForConditionalGeneration.from_pretrained(checkpoint).double()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.mul_(1.01)
        pair = data.pairs[0]

        source = torch.tensor([list(pair.prompt)])
        chosen, rejected = (
            -model(input_ids=source, labels=torch.tensor([list(answer)])).loss * len(answer)
            for answer in (pair.chosen, pair.rejected)
        )
        z = beta * ((chosen - reference_chosen) - (rejected - reference_rejected))
        expected_loss = -torch.nn.functional.logsigmoid(z)
        expected_loss.backward()
        assert abs(loss - expected_loss.item()) < 1e-4
        tied = file.config.tie_word_embeddings
        theirs = {
            _t5_text_to_text_name(name, tied): p.grad.numpy()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        assert set(ours) == set(theirs) == set(file.config.tensor_shapes())
        for name, expected in theirs.items():
            scale = max(float(np.abs(expected).max()), 1e-3)
            assert np.abs(ours[name] - expected).max() <= 2e-3 * scale, name


def test_dpo_refusals(t5, tmp_path):
    from etalii_dllm.training.seq2seq_data import TextToTextData

    _, engine = t5
    file = model_file(t5)
    pairs = load_data(engine, file, write_pairs(tmp_path / "pairs.jsonl"), SEQUENCE_LENGTH, "dpo")
    (tmp_path / "text.jsonl").write_text('{"input": "x", "target": "y"}\n', encoding="utf-8")
    examples = TextToTextData.from_file(tmp_path / "text.jsonl", engine.tokenizer.encode, SEQUENCE_LENGTH, END)
    with pytest.raises(ValueError, match="source and target pairs"):
        FineTuner.from_model_file(file, pairs, RunConfig(1, 1, SEQUENCE_LENGTH))
    with pytest.raises(ValueError, match="DPO run trains on preference pairs"):
        FineTuner.from_model_file(file, examples, RunConfig(1, 1, SEQUENCE_LENGTH, objective="dpo"))


# dllm finetune --dpo (#399)


def test_dpo_runs_are_golden_and_resume(t5, flan, tmp_path):
    from golden_values import T5_DPO_FINGERPRINT

    for kind, fixture in (("t5", t5), ("flan", flan)):
        _, engine = fixture
        file = model_file(fixture)
        data = load_data(engine, file, write_pairs(tmp_path / "pairs.jsonl"), SEQUENCE_LENGTH, "dpo")
        run = RunConfig(3, 2, SEQUENCE_LENGTH, 1, AdamWConfig(1e-3), objective="dpo", beta=0.5)
        first = FineTuner.from_model_file(file, data, run)
        first.train()
        assert first.losses[-1] < first.losses[0]
        second = FineTuner.from_model_file(file, data, run)
        second.train(until=1)
        second.save_checkpoint(tmp_path / f"{kind}.dllmckpt")
        resumed = FineTuner.load_checkpoint(tmp_path / f"{kind}.dllmckpt", data)
        resumed.train()
        assert resumed.losses == first.losses
        fingerprint = first.export(tmp_path / f"{kind}.dllm")
        assert resumed.export(tmp_path / f"{kind}-resumed.dllm") == fingerprint
        assert fingerprint == T5_DPO_FINGERPRINT[kind]


def test_finetune_dpo_command(flan, tmp_path, capsys):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from etalii_dllm.cli import main as cli
    from etalii_dllm.exporting import export_safetensors

    checkpoint, engine = flan
    path = checkpoint.parent / "model.dllm"
    pairs = write_pairs(tmp_path / "pairs.jsonl")
    command = ["finetune", str(path), "--data", str(pairs), "--dpo", "--steps", "2", "--batch-size", "2"]
    command += ["--sequence-length", str(SEQUENCE_LENGTH)]
    receipt = tmp_path / "receipt.json"
    assert cli([*command, "-o", str(tmp_path / "tuned.dllm"), "--receipt", str(receipt)]) == 0
    assert "3 pairs" in capsys.readouterr().out
    recorded = json.loads(receipt.read_text())
    assert recorded["data"]["pairs"] == 3 and recorded["run"]["objective"] == "dpo"
    assert cli(["replay", str(receipt), "--base", str(path)]) == 0
    capsys.readouterr()
    lora = [*command, "--lora-rank", "2", "--learning-rate", "1e-2", "--base-quantize", "q8_0"]
    assert cli([*lora, "-o", str(tmp_path / "lora.dllm"), "--adapter-output", str(tmp_path / "adapter")]) == 0
    capsys.readouterr()
    source = [*engine.tokenizer.encode(TEXTS[2]), END, 7]
    merged = DllmEngine.from_model_file(tmp_path / "lora.dllm").model.forward(source)
    assert merged.tobytes() != engine.model.forward(source).tobytes()
    tuned = ModelFile(tmp_path / "tuned.dllm")
    export_safetensors(tuned, tmp_path / "exported")
    model = transformers.T5ForConditionalGeneration.from_pretrained(tmp_path / "exported")
    assert model.config.architectures == ["T5ForConditionalGeneration"]


# dllm eval (#400)


def test_eval_scores_answers_given_their_source(flan, tmp_path, capsys):
    from etalii_dllm.cli import main as cli

    checkpoint, engine = flan
    encode = engine.tokenizer.encode
    preference = evaluation.evaluate(engine, PAIRS)
    assert preference["kind"] == "preference" and 0.0 <= preference["preference_accuracy"] <= 1.0
    first = preference["results"][0]
    chosen = scoring.score_tokens(engine, [*encode(TEXTS[1]), END], source=encode(PAIRS[0]["prompt"]))
    assert first["log_likelihoods"][0] == chosen.log_likelihood
    joined = preference["results"][1]["log_likelihoods"][1]
    assert (
        joined
        == scoring.score_tokens(engine, [*encode("no"), END], source=encode("be brief\n\nsame bits?")).log_likelihood
    )
    item = {"context": TEXTS[0], "choices": ["a fox", TEXTS[2]], "answer": 0}
    choice = evaluation.evaluate(engine, [item])
    expected = [scoring.score_text(engine, c, source=TEXTS[0]).log_likelihood for c in item["choices"]]
    assert choice["results"][0]["log_likelihoods"] == expected
    assert evaluation.evaluate(engine, [item])["fingerprint"] == choice["fingerprint"]
    with pytest.raises(evaluation.EvaluationError, match="decoder-only"):
        evaluation.evaluate(engine, [{"text": TEXTS[0]}])
    task = write_pairs(tmp_path / "task.jsonl")
    assert cli(["--model", str(checkpoint.parent / "model.dllm"), "eval", str(task), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["fingerprint"] == preference["fingerprint"]
    empty = evaluation.score(engine, [], encode("a fox"))  # an empty source is just </s>
    assert len(empty.logprobs) == len(encode("a fox"))
