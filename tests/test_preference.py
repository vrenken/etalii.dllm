"""Phase 25: reproducible preference tuning. DPO runs are bit-identical across runs and resume bit for bit, the loss
starts at exactly log 2, the step is the documented formula, receipts replay, and ``dllm eval`` scores preferences."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from golden_values import DPO_FINETUNE_FINGERPRINT
from model_fixtures import TINY_LLAMA_CONFIG, tiny_config, write_hf_checkpoint

from etalii_dllm import _kernels, evaluation
from etalii_dllm.cli import main as cli
from etalii_dllm.engine import DllmEngine, default_engine
from etalii_dllm.importing import import_model
from etalii_dllm.lora import LoraConfig
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.training import AdamWConfig, CheckpointError, FineTuner, RunConfig, TrainingDataError
from etalii_dllm.training.data import TrainingData
from etalii_dllm.training.preference import (
    PreferenceData,
    PreferencePair,
    PreferenceRecord,
    log_sigmoid,
    read_pairs,
)

RECORDS = [
    PreferenceRecord("the quick brown", " fox", " cat"),
    PreferenceRecord("pack my box", " with jugs", " of"),
    PreferenceRecord("two plus two is", " four", " five and a lot more words"),
    PreferenceRecord("hello", " there", " go away"),
    PreferenceRecord("the lazy", " dog", " frog"),
]
RUN = RunConfig(5, 2, 24, 3, AdamWConfig(learning_rate=1e-2), objective="dpo", beta=0.5)


def encode(text: str) -> list[int]:
    return [ord(c) % 64 for c in text]


def pairs(sequence_length: int = 24) -> PreferenceData:
    return PreferenceData.from_records(RECORDS, encode, sequence_length, 2)


@pytest.fixture(scope="module", params=["llama", "qwen3"])
def model_file(request, tmp_path_factory) -> ModelFile:
    directory = tmp_path_factory.mktemp(request.param)
    write_hf_checkpoint(directory / "checkpoint", tiny_config(request.param))
    import_model(directory / "checkpoint", directory / "model.dllm")
    return ModelFile(directory / "model.dllm")


# Data


def test_pairs_are_tokenized_separately_and_cut_to_fit():
    data = pairs()
    first = data.pairs[0]
    assert first.prompt == tuple(encode("the quick brown"))
    assert first.chosen == (*encode(" fox"), 2) and first.rejected == (*encode(" cat"), 2)
    long = data.pairs[2]
    assert len(long.prompt) + len(long.rejected) == 25  # cut at the end to sequence_length + 1
    assert long.rejected == tuple(encode(" five and a lot more words"))[: 25 - len(long.prompt)]
    tokens, targets = first.sequence("chosen")
    assert tokens == [*first.prompt, *first.chosen][:-1]
    assert targets == [-1] * (len(first.prompt) - 1) + list(first.chosen)
    assert len(data) == 5
    assert sorted(data.batch(0, 5, 3)) == [0, 1, 2, 3, 4]
    assert data.batch(1, 3, 3) == [*data.epoch_order(3, 0)[3:], data.epoch_order(3, 1)[0]]  # sample 5: epoch 1


def test_fingerprint_is_domain_separated_and_sensitive():
    data = pairs()
    assert data.fingerprint == pairs().fingerprint
    assert data.fingerprint != pairs(25).fingerprint
    swapped = PreferenceData.from_records(
        [PreferenceRecord(r.prompt, r.rejected, r.chosen) for r in RECORDS], encode, 24, 2
    )
    assert swapped.fingerprint != data.fingerprint
    windows = TrainingData((tuple(range(5)),), 4)
    assert windows.fingerprint != PreferenceData((PreferencePair((0,), (1,), (2,)),), 4).fingerprint


def test_shared_order_keeps_the_window_bits():
    """The order helpers were factored out of TrainingData; its batches must be the ones it always drew."""
    from etalii_dllm.numerics import DeterministicRandom

    def old_epoch_order(count: int, seed: int, epoch: int) -> list[int]:
        random = DeterministicRandom((seed + (epoch + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
        order = list(range(count))
        for i in range(count - 1, 0, -1):
            j = random.next_u64() % (i + 1)
            order[i], order[j] = order[j], order[i]
        return order

    data = TrainingData(tuple((i, i + 1) for i in range(7)), 1)
    for epoch in range(3):
        assert data.epoch_order(3, epoch) == old_epoch_order(7, 3, epoch)
    expected = [*old_epoch_order(7, 3, 0)[6:], *old_epoch_order(7, 3, 1)[:2]]
    assert data.batch(2, 3, 3) == [data.windows[i] for i in expected]


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([PreferenceRecord("", " a", " b")], "prompt is empty"),
        ([PreferenceRecord("x" * 13, " a", " b")], "more than the sequence length"),
    ],
)
def test_bad_pairs(records, message):
    with pytest.raises(TrainingDataError, match=message):
        PreferenceData.from_records(records, encode, 12, 2)
    with pytest.raises(TrainingDataError, match="an answer is empty"):
        PreferenceData.from_records([PreferenceRecord("q", "", "b")], encode, 12, None)
    with pytest.raises(TrainingDataError, match="positive"):
        PreferenceData.from_records(RECORDS, encode, 0, 2)


def test_read_pairs(tmp_path):
    path = tmp_path / "pairs.jsonl"
    lines = [
        json.dumps({"prompt": "Q", "chosen": "a", "rejected": "b"}),
        "",
        json.dumps({"messages": [{"role": "user", "content": "Hi"}], "chosen": "c", "rejected": "d"}),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    render = lambda messages: "<" + messages[0]["content"] + ">"  # noqa: E731
    assert read_pairs(path, render) == [PreferenceRecord("Q", "a", "b"), PreferenceRecord("<Hi>", "c", "d")]
    with pytest.raises(TrainingDataError, match="chat template"):
        read_pairs(path, None)
    for line, message in [
        ("{", "invalid JSON"),
        ('{"prompt": "Q", "chosen": "a"}', "'chosen' and 'rejected'"),
        ('{"chosen": "a", "rejected": "b"}', "'prompt' or 'messages'"),
    ]:
        path.write_text(line, encoding="utf-8")
        with pytest.raises(TrainingDataError, match=message):
            read_pairs(path, render)
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(TrainingDataError, match="no preference pairs"):
        read_pairs(path, render)
    with pytest.raises(TrainingDataError):
        read_pairs(tmp_path / "missing.jsonl", render)


def test_log_sigmoid():
    assert log_sigmoid(0.0) == -_kernels.log(2.0)
    for z in (-800.0, -3.5, -1e-3, 1e-3, 2.0, 800.0):
        assert math.isclose(log_sigmoid(z), -math.log1p(math.exp(-z)) if z > -700 else z, rel_tol=1e-12, abs_tol=1e-300)
    assert log_sigmoid(800.0) == 0.0 and log_sigmoid(-800.0) == -800.0


def test_run_config():
    assert "objective" not in RunConfig(1).to_dict()  # language-model runs keep their settings and receipts
    assert RunConfig.from_dict(RUN.to_dict()) == RUN
    assert RUN.to_dict()["objective"] == "dpo" and RUN.to_dict()["beta"] == 0.5
    with pytest.raises(ValueError, match="objective"):
        RunConfig(1, objective="ppo")
    for beta in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="beta"):
            RunConfig(1, objective="dpo", beta=beta)


# Training


def test_step_zero_loss_is_log_two_and_the_step_is_the_formula(model_file):
    data = pairs()
    tuner = FineTuner.from_model_file(model_file, data, RUN)
    weights = {name: values.copy() for name, values in tuner.params.items()}
    assert tuner.reference == [
        (
            tuner.log_probability(model_file.tensors, p, "chosen"),
            tuner.log_probability(model_file.tensors, p, "rejected"),
        )
        for p in data.pairs
    ]
    loss, gradients = tuner._preference_gradients(data)
    assert loss == _kernels.log(2.0)  # the model is its own reference: z = 0 exactly

    expected: dict[str, np.ndarray] = {}
    weight = np.float32(RUN.beta * 0.5 / RUN.batch_size)  # 1 - sigmoid(0) = 0.5
    for index in data.batch(0, RUN.batch_size, RUN.seed):
        for which, sign in (("chosen", weight), ("rejected", -weight)):
            _, pair_gradients = tuner._gradients.loss_and_gradients(weights, *data.pairs[index].sequence(which))
            for name, gradient in pair_gradients.items():
                expected[name] = expected[name] + gradient * sign if name in expected else gradient * sign
    assert set(gradients) == set(expected)
    for name in expected:
        assert np.array_equal(gradients[name], expected[name])


def test_dpo_moves_toward_the_chosen_answers(model_file):
    data = pairs()
    tuner = FineTuner.from_model_file(model_file, data, RunConfig(12, 5, 24, 0, AdamWConfig(3e-2), objective="dpo"))
    results = tuner.train()
    assert results[-1].loss < results[0].loss
    weights = tuner.weights()
    for pair, (reference_chosen, reference_rejected) in zip(data.pairs, tuner.reference, strict=True):
        margin = tuner.log_probability(weights, pair, "chosen") - tuner.log_probability(weights, pair, "rejected")
        assert margin > reference_chosen - reference_rejected


def test_dpo_runs_are_byte_identical_and_resume_bit_for_bit(model_file, tmp_path):
    data = pairs()
    first = FineTuner.from_model_file(model_file, data, RUN)
    first.train()
    first_ckpt = first.save_checkpoint(tmp_path / "first.dllmckpt")
    raw = (tmp_path / "first.dllmckpt").read_bytes()
    header = json.loads(raw[20 : 20 + int.from_bytes(raw[12:20], "little")])
    assert header["reference"][0] == [first.reference[0][0].hex(), first.reference[0][1].hex()]

    second = FineTuner.from_model_file(model_file, data, RUN)
    second.train(until=2)
    second.save_checkpoint(tmp_path / "middle.dllmckpt")
    resumed = FineTuner.load_checkpoint(tmp_path / "middle.dllmckpt", data)
    assert resumed.reference == first.reference  # restored, not recomputed from the half-trained weights
    resumed.train()
    assert resumed.save_checkpoint(tmp_path / "resumed.dllmckpt") == first_ckpt
    assert (tmp_path / "resumed.dllmckpt").read_bytes() == (tmp_path / "first.dllmckpt").read_bytes()
    assert resumed.losses == first.losses

    fingerprint = first.export(tmp_path / "first.dllm")
    assert resumed.export(tmp_path / "resumed.dllm") == fingerprint
    assert fingerprint == DPO_FINETUNE_FINGERPRINT[model_file.config.family]
    tuned = ModelFile(tmp_path / "first.dllm")
    assert tuned.fine_tuning["run"]["objective"] == "dpo"
    assert tuned.fine_tuning["data_fingerprint"] == data.fingerprint
    assert tuned.lineage[-1]["objective"] == "dpo"

    with pytest.raises(CheckpointError, match="different data"):
        FineTuner.load_checkpoint(tmp_path / "middle.dllmckpt", pairs(25))


def test_dpo_with_lora(model_file, tmp_path):
    run = RunConfig(4, 2, 24, 1, AdamWConfig(1e-2), LoraConfig(2, 2.0), objective="dpo")
    data = pairs()
    first = FineTuner.from_model_file(model_file, data, run)
    assert first.losses == [] and first.reference is not None
    first.train()
    assert first.losses[0] == _kernels.log(2.0)
    second = FineTuner.from_model_file(model_file, data, run)
    second.train(until=1)
    second.save_checkpoint(tmp_path / "lora.dllmckpt")
    resumed = FineTuner.load_checkpoint(tmp_path / "lora.dllmckpt", data, model_file)
    resumed.train()
    assert resumed.losses == first.losses
    assert resumed.export(tmp_path / "a.dllm") == first.export(tmp_path / "b.dllm")


def test_objective_and_data_must_match(model_file):
    with pytest.raises(ValueError, match="preference pairs"):
        FineTuner.from_model_file(model_file, pairs(), RunConfig(1, 1, 24))
    with pytest.raises(ValueError, match="preference pairs"):
        FineTuner.from_model_file(model_file, TrainingData(((1, 2),), 24), RUN)
    with pytest.raises(ValueError, match="one reference pair"):
        FineTuner(
            model_file.config,
            model_file.tensors,
            pairs(),
            RUN,
            base_fingerprint=model_file.fingerprint,
            metadata=model_file.header,
            reference=[(0.0, 0.0)],
        )


# Command line, receipts and evaluation


@pytest.fixture
def chat_model(tmp_path):
    from test_bpe import smollm2_style
    from test_chat_template import SMOLLM2

    pytest.importorskip("tokenizers")
    reference = smollm2_style()
    config = {**TINY_LLAMA_CONFIG, "vocab_size": reference.get_vocab_size(), "eos_token_id": 2}
    write_hf_checkpoint(tmp_path / "checkpoint", config, tokenizer_json=json.loads(reference.to_str()))
    tokenizer_config = {"chat_template": SMOLLM2, "eos_token": "<|im_end|>", "bos_token": None}
    (tmp_path / "checkpoint" / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    import_model(tmp_path / "checkpoint", tmp_path / "base.dllm")
    return tmp_path / "base.dllm"


def preference_lines(offset: int = 0) -> list[str]:
    lines = []
    for i in range(offset, offset + 6):
        question = {"role": "user", "content": f"What is {i} plus {i}?"}
        lines.append(json.dumps({"messages": [question], "chosen": f"It is {2 * i}.", "rejected": "No idea."}))
    lines.append(json.dumps({"prompt": "Say hi:", "chosen": " hi", "rejected": " bye"}))
    return lines


def test_cli_dpo_receipt_replay_and_eval(chat_model, tmp_path, capsys, monkeypatch):
    data = tmp_path / "pairs.jsonl"
    data.write_text("\n".join(preference_lines()), encoding="utf-8")
    common = ["finetune", str(chat_model), "--data", str(data), "--dpo", "--beta", "0.2"]
    common += ["--steps", "3", "--batch-size", "2", "--sequence-length", "128", "--learning-rate", "1e-2"]
    checkpoint = ["--checkpoint", str(tmp_path / "run.dllmckpt"), "--checkpoint-every", "1"]
    receipt = tmp_path / "train.json"
    assert cli([*common, "-o", str(tmp_path / "a.dllm"), *checkpoint, "--receipt", str(receipt)]) == 0
    first = capsys.readouterr().out
    assert "7 pairs of up to 128 tokens" in first
    assert f"loss {math.log(2):.6f}" in first.splitlines()[1]
    assert cli([*common, "-o", str(tmp_path / "b.dllm")]) == 0
    capsys.readouterr()
    assert (tmp_path / "a.dllm").read_bytes() == (tmp_path / "b.dllm").read_bytes()
    assert cli([*common, "-o", str(tmp_path / "c.dllm"), "--resume", str(tmp_path / "run.dllmckpt")]) == 0
    capsys.readouterr()
    assert (tmp_path / "c.dllm").read_bytes() == (tmp_path / "a.dllm").read_bytes()

    recorded = json.loads(receipt.read_text())
    assert recorded["data"]["pairs"] == 7 and "windows" not in recorded["data"]
    assert recorded["run"]["objective"] == "dpo" and recorded["run"]["beta"] == 0.2
    assert cli(["replay", str(receipt), "--base", str(chat_model)]) == 0
    assert "training again gave the same weights" in capsys.readouterr().out
    other = tmp_path / "other.jsonl"
    other.write_text("\n".join(preference_lines(1)), encoding="utf-8")
    assert cli(["replay", str(receipt), "--base", str(chat_model), "--data", str(other), "--json"]) == 1
    outcome = json.loads(capsys.readouterr().out)
    assert any("training data differ" in reason for reason in outcome["reasons"])

    # A pair file is also an evaluation task; the tuned model prefers the chosen answers more than the base does.
    monkeypatch.setenv("DLLM_MODEL", str(chat_model))  # --model sets it; restored when the test ends
    default_engine.cache_clear()
    reports = []
    for model in (chat_model, tmp_path / "a.dllm"):
        assert cli(["--model", str(model), "eval", str(data), "--json"]) == 0
        reports.append(json.loads(capsys.readouterr().out))
    base, tuned = reports
    assert base["kind"] == "preference" and base["items"] == 7
    assert tuned["mean_margin"] > base["mean_margin"]
    assert 0.0 <= base["preference_accuracy"] <= 1.0
    row = tuned["results"][0]
    assert row["margin"] == row["log_likelihoods"][0] - row["log_likelihoods"][1]
    assert row["preferred"] == (row["margin"] > 0)

    assert cli([*common, "--teacher", str(chat_model), "-o", str(tmp_path / "x.dllm")]) == 1
    assert "--dpo trains on preference pairs" in capsys.readouterr().err
    data.write_text('{"prompt": "Q"}', encoding="utf-8")
    assert cli([*common, "-o", str(tmp_path / "x.dllm")]) == 1
    assert "'chosen' and 'rejected'" in capsys.readouterr().err
    default_engine.cache_clear()


def test_preference_evaluation_is_deterministic_and_checked():
    engine = DllmEngine.create_default()
    items = [
        {"prompt": "the cat", "chosen": " sat", "rejected": " sat"},
        {"prompt": "hello", "chosen": " world", "rejected": " there friend"},
    ]
    report = evaluation.evaluate(engine, items)
    assert report == evaluation.evaluate(engine, items)
    assert report["kind"] == "preference"
    assert report["results"][0]["preferred"] is False  # a tie is a miss
    assert report["mean_margin"] == (report["results"][0]["margin"] + report["results"][1]["margin"]) / 2
    if engine.chat_template is None:
        with pytest.raises(evaluation.EvaluationError, match="chat template"):
            evaluation.evaluate(
                engine, [{"messages": [{"role": "user", "content": "x"}], "chosen": "a", "rejected": "b"}]
            )
    with pytest.raises(evaluation.EvaluationError, match="mixes"):
        evaluation.evaluate(engine, [items[0], {"text": "abc"}])
