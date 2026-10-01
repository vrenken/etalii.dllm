"""Validation in the training stack: bad training data, run settings, AdamW hyperparameters and checkpoint files are
refused with a precise reason; gradient clipping, data without a separator and licence attribution follow their
documented behaviour."""

from __future__ import annotations

import dataclasses
import json
import struct

import numpy as np
import pytest
from test_training import RUN, TARGETS, TOKENS, ascii_data, gaussian, model_file  # noqa: F401 - fixture

from etalii_dllm import numerics
from etalii_dllm.importing import import_model
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.training import (
    AdamWConfig,
    CheckpointError,
    DecoderGradients,
    FineTuner,
    RunConfig,
    TrainingData,
    TrainingDataError,
    read_documents,
)
from etalii_dllm.training.optimizer import AdamW, global_norm
from etalii_dllm.training.trainer import CHECKPOINT_MAGIC, CHECKPOINT_VERSION

# Training data


def test_unreadable_data_files(tmp_path):
    with pytest.raises(TrainingDataError, match=r"missing\.txt"):
        read_documents(tmp_path / "missing.txt", None)
    (tmp_path / "latin1.txt").write_bytes("caf\xe9".encode("latin-1"))
    with pytest.raises(TrainingDataError, match=r"latin1\.txt: .*utf-8"):
        read_documents(tmp_path / "latin1.txt", None)
    with pytest.raises(TrainingDataError):
        read_documents(tmp_path, None)  # a directory


def test_invalid_json_lines_name_the_line(tmp_path):
    (tmp_path / "data.jsonl").write_text('{"text": "fine"}\n\n{"text": "broken"\n', encoding="utf-8")
    with pytest.raises(TrainingDataError, match=r"data.jsonl:3: invalid JSON \("):
        read_documents(tmp_path / "data.jsonl", None)


@pytest.mark.parametrize("line", ['["text"]', '"text"', '{"text": 5}', '{"messages": "hi"}', "null"])
def test_records_need_text_or_messages(tmp_path, line):
    (tmp_path / "data.jsonl").write_text(line + "\n", encoding="utf-8")
    with pytest.raises(TrainingDataError, match=r"data.jsonl:1: expected an object with 'text' or 'messages'"):
        read_documents(tmp_path / "data.jsonl", None)


def test_chat_records_need_a_template(tmp_path):
    (tmp_path / "chat.jsonl").write_text('{"messages": []}\n', encoding="utf-8")
    with pytest.raises(TrainingDataError, match=r"chat.jsonl:1: chat records need a model with a chat template"):
        read_documents(tmp_path / "chat.jsonl", None)


def test_empty_and_blank_files_have_no_documents(tmp_path):
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    (tmp_path / "blank.JSONL").write_text("\n  \n\t\n", encoding="utf-8")
    assert read_documents(tmp_path / "empty.txt", None) == []
    assert read_documents(tmp_path / "blank.JSONL", None) == []  # the suffix is case-insensitive


def encode(text: str) -> list[int]:
    return [ord(c) % 64 for c in text]


@pytest.mark.parametrize("length", [0, -3])
def test_sequence_length_must_be_positive(length):
    with pytest.raises(TrainingDataError, match="sequence_length must be positive"):
        TrainingData.from_documents(["abc"], encode, length, 2)


@pytest.mark.parametrize(
    ("documents", "separator"), [([], 2), ([], None), (["", ""], None), (["a"], None), (["", "x"], None)]
)
def test_too_little_data_is_refused(documents, separator):
    with pytest.raises(TrainingDataError, match="fewer than two tokens"):
        TrainingData.from_documents(documents, encode, 4, separator)


def test_documents_without_a_separator_are_concatenated():
    data = TrainingData.from_documents(["abc", "de"], encode, 2, None)
    assert data.windows == (tuple(encode("abc")), tuple(encode("cde")))
    # Two tokens are the smallest usable stream (one input, one target).
    assert TrainingData.from_documents(["a"], encode, 4, 2).windows == ((ord("a") % 64, 2),)


def test_windows_never_shorter_than_two_tokens():
    for length in range(1, 12):
        for size in range(2, 30):
            data = TrainingData.from_documents(["x" * (size - 1)], encode, length, 2)
            assert all(2 <= len(w) <= length + 1 for w in data.windows)
            assert sum(len(w) - 1 for w in data.windows) == size - 1  # every target exactly once


# AdamW


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"schedule": "linear"}, "unknown schedule 'linear'"),
        ({"beta1": 1.0}, r"betas must be in \[0, 1\)"),
        ({"beta1": -0.1}, r"betas must be in \[0, 1\)"),
        ({"beta2": 1.5}, r"betas must be in \[0, 1\)"),
        ({"learning_rate": -1e-3}, "must not be negative"),
        ({"max_grad_norm": -1.0}, "must not be negative"),
        ({"warmup_steps": -1}, "must not be negative"),
    ],
)
def test_bad_adamw_hyperparameters(settings, message):
    with pytest.raises(ValueError, match=message):
        AdamWConfig(**settings)


def test_gradients_above_the_norm_limit_are_clipped():
    shapes = {"a.weight": (4, 6), "a.bias": (6,)}
    grads = {"a.weight": gaussian(1, 4, 6, scale=3.0), "a.bias": gaussian(2, 6, scale=3.0)}
    config = AdamWConfig(learning_rate=1e-2, max_grad_norm=0.5)
    params = {"a.weight": gaussian(3, 4, 6), "a.bias": gaussian(4, 6)}
    clipped = {name: values.copy() for name, values in params.items()}
    norm = AdamW(config, shapes).step(clipped, grads, 1, 1e-2)
    assert norm == global_norm(grads) > 0.5  # the norm before clipping is reported

    # The clipped step is the kernel run with scale = max_norm / (norm + 1e-6).
    scale = 0.5 / (norm + 1e-6)
    for name in ("a.bias", "a.weight"):
        expected, m, v = params[name].copy(), np.zeros(shapes[name], np.float32), np.zeros(shapes[name], np.float32)
        decay = 0.01 if expected.ndim >= 2 else 0.0
        numerics.adamw_step(expected, grads[name], m, v, 1e-2, 0.9, 0.999, 1e-8, decay, 1 - 0.9, 1 - 0.999, scale)
        assert clipped[name].tobytes() == expected.tobytes()

    # A limit of 0 disables clipping, and so does a norm below the limit: both give the unscaled step.
    unclipped = {name: values.copy() for name, values in params.items()}
    AdamW(dataclasses.replace(config, max_grad_norm=0.0), shapes).step(unclipped, grads, 1, 1e-2)
    loose = {name: values.copy() for name, values in params.items()}
    AdamW(dataclasses.replace(config, max_grad_norm=1e6), shapes).step(loose, grads, 1, 1e-2)
    assert all(unclipped[n].tobytes() == loose[n].tobytes() != clipped[n].tobytes() for n in params)


def test_learning_rate_schedule_edges():
    config = AdamWConfig(learning_rate=1.0, warmup_steps=4)
    assert config.learning_rate_at(4, 4) == 1.0  # warmup covers the whole run
    assert config.learning_rate_at(5, 4) == 1.0  # past the planned end: no division by zero


# Decoder gradients


def test_decoder_gradients_validate_their_input(model_file):  # noqa: F811
    config = model_file.config
    with pytest.raises(ValueError, match="Hugging Face rotary layout"):
        DecoderGradients(dataclasses.replace(config, rope_interleaved=True))
    gradients = DecoderGradients(config)
    with pytest.raises(ValueError, match="non-empty and equally long"):
        gradients.loss_and_gradients(model_file.tensors, TOKENS, TARGETS[:-1])
    with pytest.raises(ValueError, match="non-empty and equally long"):
        gradients.loss_and_gradients(model_file.tensors, [], [])
    for bad in (config.vocabulary_size, -1):
        with pytest.raises(ValueError, match="token id out of range"):
            gradients.loss_and_gradients(model_file.tensors, [*TOKENS[:-1], bad], TARGETS)
        with pytest.raises(ValueError, match="token id out of range"):
            gradients.logits(model_file.tensors, [bad])


# Runs


@pytest.mark.parametrize(("steps", "batch_size", "sequence_length"), [(0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 8, 8)])
def test_run_settings_must_be_positive(steps, batch_size, sequence_length):
    with pytest.raises(ValueError, match="steps, batch_size and sequence_length must be positive"):
        RunConfig(steps, batch_size, sequence_length)


def test_run_must_fit_the_model_and_data(model_file):  # noqa: F811
    with pytest.raises(ValueError, match="windowed for a different sequence length"):
        FineTuner.from_model_file(model_file, ascii_data(4), RUN)
    too_long = model_file.config.context_length + 1
    with pytest.raises(ValueError, match=rf"exceeds the model's context length \({too_long - 1}\)"):
        FineTuner.from_model_file(model_file, ascii_data(too_long), dataclasses.replace(RUN, sequence_length=too_long))
    tensors = dict(model_file.tensors)
    del tensors["final_norm.weight"]
    with pytest.raises(ValueError, match="parameters do not match the architecture"):
        FineTuner(model_file.config, tensors, ascii_data(), RUN, base_fingerprint="", metadata={})
    extra = {**model_file.tensors, "unexpected.weight": np.zeros(2, np.float32)}
    with pytest.raises(ValueError, match="parameters do not match the architecture"):
        FineTuner(model_file.config, extra, ascii_data(), RUN, base_fingerprint="", metadata={})


def test_a_finished_run_takes_no_more_steps(model_file):  # noqa: F811
    tuner = FineTuner.from_model_file(model_file, ascii_data(), RunConfig(1, 1, 8))
    assert [r.step for r in tuner.train()] == [1]
    assert tuner.train() == []  # train() stops at the end
    with pytest.raises(RuntimeError, match="already completed all of its steps"):
        tuner.train_step()
    assert tuner.step == 1 and len(tuner.losses) == 1


def test_only_lora_runs_export_adapters(model_file, tmp_path):  # noqa: F811
    tuner = FineTuner.from_model_file(model_file, ascii_data(), RUN)
    with pytest.raises(ValueError, match="only LoRA runs have an adapter to export"):
        tuner.export_adapter(tmp_path / "adapter")
    assert not (tmp_path / "adapter").exists()


def export_licence(model_file, path, licence) -> dict | None:  # noqa: F811
    metadata = {**model_file.header, "licence": licence}
    tuner = FineTuner(
        model_file.config, model_file.tensors, ascii_data(), RunConfig(1, 1, 8),
        base_fingerprint=model_file.fingerprint, metadata=metadata,
    )  # fmt: skip
    tuner.train()
    tuner.export(path)
    return ModelFile(path).header.get("licence")


def test_export_keeps_a_licence_without_attribution(model_file, tmp_path):  # noqa: F811
    assert export_licence(model_file, tmp_path / "none.dllm", None) is None
    assert export_licence(model_file, tmp_path / "mit.dllm", {"spdx": "MIT"}) == {"spdx": "MIT"}


def test_export_appends_to_a_custom_attribution(model_file, tmp_path):  # noqa: F811
    licence = export_licence(model_file, tmp_path / "tuned.dllm", {"spdx": "MIT", "attribution": "Made by Example."})
    assert licence == {
        "spdx": "MIT",
        "attribution": "Made by Example. Fine-tuned with EtAlii.Dllm (1 steps); modified weights.",
    }


# Checkpoints


@pytest.fixture
def checkpoint(model_file, tmp_path):  # noqa: F811
    tuner = FineTuner.from_model_file(model_file, ascii_data(), RUN)
    tuner.train(until=1)
    path = tmp_path / "run.dllmckpt"
    tuner.save_checkpoint(path)
    return path


def rewrite(path, magic=CHECKPOINT_MAGIC, version=CHECKPOINT_VERSION, header: bytes | None = None) -> None:
    raw = path.read_bytes()
    _, _, length = struct.unpack_from("<8sIQ", raw)
    header = raw[20 : 20 + length] if header is None else header
    path.write_bytes(struct.pack("<8sIQ", magic, version, len(header)) + header + raw[20 + length :])


def test_short_files_are_not_checkpoints(tmp_path):
    for content in (b"", b"DLLMCKPT", b"DLLMCKPT\x01\x00\x00\x00\x00"):
        (tmp_path / "short.dllmckpt").write_bytes(content)
        with pytest.raises(CheckpointError, match="not a checkpoint file"):
            FineTuner.load_checkpoint(tmp_path / "short.dllmckpt", ascii_data())


def test_a_model_file_is_not_a_checkpoint(model_file):  # noqa: F811
    with pytest.raises(CheckpointError, match="not a checkpoint file"):
        FineTuner.load_checkpoint(model_file.path, ascii_data())


def test_other_checkpoint_versions_are_refused(checkpoint):
    rewrite(checkpoint, version=CHECKPOINT_VERSION + 1)
    with pytest.raises(CheckpointError, match=f"unsupported checkpoint version {CHECKPOINT_VERSION + 1}"):
        FineTuner.load_checkpoint(checkpoint, ascii_data())


@pytest.mark.parametrize("header", [b"{not json", b"\xff\xfe{}", b'{"step": '])
def test_checkpoint_header_must_be_json(checkpoint, header):
    rewrite(checkpoint, header=header)
    with pytest.raises(CheckpointError, match="header is not valid JSON"):
        FineTuner.load_checkpoint(checkpoint, ascii_data())


def test_truncated_checkpoint_is_corrupt(checkpoint):
    checkpoint.write_bytes(checkpoint.read_bytes()[:-64])
    with pytest.raises(CheckpointError, match="does not match the fingerprint"):
        FineTuner.load_checkpoint(checkpoint, ascii_data())


def test_lora_checkpoint_needs_its_own_base(model_file, tmp_path, tmp_path_factory):  # noqa: F811
    from model_fixtures import tiny_config, write_hf_checkpoint

    from etalii_dllm.lora import LoraConfig

    run = dataclasses.replace(RUN, lora=LoraConfig(2, 4.0))
    tuner = FineTuner.from_model_file(model_file, ascii_data(), run)
    tuner.train(until=1)
    tuner.save_checkpoint(tmp_path / "lora.dllmckpt")
    with pytest.raises(CheckpointError, match="needs the base model it was trained from"):
        FineTuner.load_checkpoint(tmp_path / "lora.dllmckpt", ascii_data())

    other_family = "qwen3" if model_file.config.family != "qwen3" else "llama"
    directory = tmp_path_factory.mktemp("other-base")
    write_hf_checkpoint(directory / "checkpoint", tiny_config(other_family))
    import_model(directory / "checkpoint", directory / "other.dllm")
    other = ModelFile(directory / "other.dllm")
    assert other.fingerprint != model_file.fingerprint
    with pytest.raises(CheckpointError, match="trained from a different base model"):
        FineTuner.load_checkpoint(tmp_path / "lora.dllmckpt", ascii_data(), base=other)

    resumed = FineTuner.load_checkpoint(tmp_path / "lora.dllmckpt", ascii_data(), base=model_file)
    assert resumed.step == 1 and resumed.losses == tuner.losses
    assert all(resumed.params[n].tobytes() == tuner.params[n].tobytes() for n in tuner.params)


def test_checkpoint_header_is_canonical_json(checkpoint):
    raw = checkpoint.read_bytes()
    _, _, length = struct.unpack_from("<8sIQ", raw)
    header = json.loads(raw[20 : 20 + length])
    assert header["format"] == "dllm-checkpoint" and header["step"] == 1
    assert (20 + length) % 64 == 0  # tensor data starts aligned
