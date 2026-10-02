"""Phase 3: gradient kernels against float64 references, decoder gradients against finite differences, AdamW, the
data order, and bit-for-bit reproducible fine-tuning runs, checkpoints and resumption."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from golden_values import FINETUNE_FINGERPRINT, GRADIENT_FINGERPRINT
from model_fixtures import TINY_LLAMA_CONFIG, tiny_config, write_hf_checkpoint
from test_transformer import reference_logits

from etalii_dllm import numerics
from etalii_dllm.cli import main as cli
from etalii_dllm.importing import import_model
from etalii_dllm.modelfile import ModelFile, tensor_order
from etalii_dllm.numerics import fill_gaussian
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
from etalii_dllm.transformer import Transformer

TOKENS = [1, 17, 42, 5, 63, 0, 9, 9, 30]
FAMILIES = ["gemma2", "gemma3", "granite", "llama", "mistral", "olmo2", "phi3", "qwen2", "qwen3"]
TARGETS = [*TOKENS[1:], 2]


def gaussian(seed: int, *shape: int, scale: float = 1.0) -> np.ndarray:
    return fill_gaussian(seed, int(np.prod(shape))).reshape(shape) * np.float32(scale)


def numeric_gradient(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central differences of a float64 scalar function, element by element."""
    grad = np.zeros_like(x, dtype=np.float64)
    for index in np.ndindex(x.shape):
        plus, minus = x.astype(np.float64), x.astype(np.float64)
        plus[index] += eps
        minus[index] -= eps
        grad[index] = (f(plus) - f(minus)) / (2 * eps)
    return grad


def close(ours, reference, tolerance: float = 1e-5) -> None:
    reference = np.asarray(reference, dtype=np.float64)
    np.testing.assert_allclose(
        np.asarray(ours, dtype=np.float64), reference, rtol=0, atol=tolerance * max(1.0, float(np.abs(reference).max()))
    )


# Kernels


def test_linear_backward_matches_float64():
    x, w, dy = gaussian(1, 5, 7), gaussian(2, 3, 7), gaussian(4, 5, 3)
    dx, dw, db = numerics.linear_backward(x, w, dy, with_bias=True)
    x64, w64, dy64 = (a.astype(np.float64) for a in (x, w, dy))
    close(dx, dy64 @ w64)
    close(dw, dy64.T @ x64)
    close(db, dy64.sum(0))
    assert numerics.linear_backward(x, w, dy)[2] is None


def test_linear_backward_rows_do_not_affect_each_other():
    x, w, dy = gaussian(1, 6, 7), gaussian(2, 3, 7), gaussian(4, 6, 3)
    dx_all = numerics.linear_backward(x, w, dy)[0]
    dx_one = numerics.linear_backward(x[2:3], w, dy[2:3])[0]
    assert dx_all[2:3] == dx_one


def test_rms_norm_backward_matches_numeric_gradient():
    x, w, dy = gaussian(5, 3, 8), gaussian(6, 8), gaussian(7, 3, 8)
    eps = 1e-5
    dx, dw = numerics.rms_norm_backward(x, w, dy, eps)

    def forward(xv, wv):
        return xv / np.sqrt(np.mean(xv * xv, -1, keepdims=True) + eps) * wv

    close(dx, numeric_gradient(lambda v: float((forward(v, w) * dy).sum()), x), 1e-6)
    close(dw, numeric_gradient(lambda v: float((forward(x.astype(np.float64), v) * dy).sum()), w), 1e-6)


def test_unit_offset_rms_norm_backward_matches_numeric_gradient():
    x, w, dy = gaussian(5, 3, 8), gaussian(6, 8, scale=0.1), gaussian(7, 3, 8)
    eps = 1e-6
    dx, dw = numerics.rms_norm_backward(x, w, dy, eps, add_unit_offset=True)

    def forward(xv, wv):
        return xv / np.sqrt(np.mean(xv * xv, -1, keepdims=True) + eps) * (1 + wv)

    close(dx, numeric_gradient(lambda v: float((forward(v, w) * dy).sum()), x), 1e-6)
    close(dw, numeric_gradient(lambda v: float((forward(x.astype(np.float64), v) * dy).sum()), w), 1e-6)
    _, plain_dw = numerics.rms_norm_backward(x, w, dy, eps)
    assert dw.numpy().tobytes() == plain_dw.numpy().tobytes()  # d(1 + w)/dw = 1


def test_gelu_tanh_backward_matches_numeric_gradient():
    x, dy = gaussian(8, 20, scale=4.0), gaussian(9, 20)

    def gelu(v):
        return 0.5 * v * (1 + np.tanh(np.sqrt(2 / np.pi) * (v + 0.044715 * v**3)))

    close(numerics.gelu_tanh_backward(x, dy), numeric_gradient(lambda v: float((gelu(v) * dy).sum()), x), 1e-7)


def test_softcap_backward_matches_numeric_gradient():
    x, dy = gaussian(8, 20, scale=4.0), gaussian(9, 20)
    reference = numeric_gradient(lambda v: float((3.0 * np.tanh(v / 3.0) * dy).sum()), x)
    close(numerics.softcap_backward(x, dy, 3.0), reference, 1e-7)
    with pytest.raises(ValueError, match="positive"):
        numerics.softcap_backward(x, dy, 0.0)
    q = gaussian(12, 2, 2, 4)
    with pytest.raises(ValueError, match="softcap must be positive"):
        numerics.attention_backward(q, q, q, q, softcap=0.0)


def test_silu_backward_matches_numeric_gradient():
    x, dy = gaussian(8, 20, scale=4.0), gaussian(9, 20)
    reference = numeric_gradient(lambda v: float((v / (1 + np.exp(-v)) * dy).sum()), x)
    close(numerics.silu_backward(x, dy), reference, 1e-7)


@pytest.mark.parametrize("interleaved", [False, True])
def test_rope_inverse_is_the_transpose(interleaved):
    x, dy = gaussian(10, 5, 2, 8), gaussian(11, 5, 2, 8)
    positions, inv_freq = np.arange(3, 8), numerics.rope_inv_freq(8, 10000.0)
    rotated = numerics.rope(x, positions, inv_freq, interleaved=interleaved).numpy().astype(np.float64)
    back = numerics.rope(dy, positions, inv_freq, interleaved=interleaved, inverse=True).numpy().astype(np.float64)
    # <R x, dy> == <x, R^T dy>
    assert abs((rotated * dy).sum() - (x.astype(np.float64) * back).sum()) < 1e-5
    restored = numerics.rope(rotated.astype(np.float32), positions, inv_freq, interleaved=interleaved, inverse=True)
    close(restored, x, 1e-6)


def reference_attention(q, k, v, scale, window=None, softcap=None):
    q_len, heads, _ = q.shape
    group = heads // k.shape[1]
    out = np.zeros((q_len, heads, v.shape[2]))
    offset = k.shape[0] - q_len
    for h in range(heads):
        scores = q[:, h] @ k[:, h // group].T * scale
        if softcap is not None:
            scores = softcap * np.tanh(scores / softcap)
        scores = scores + np.triu(np.full(scores.shape, -np.inf), offset + 1)
        if window is not None:
            scores = scores + np.tril(np.full(scores.shape, -np.inf), offset - window)
        p = np.exp(scores - scores.max(-1, keepdims=True))
        out[:, h] = (p / p.sum(-1, keepdims=True)) @ v[:, h // group]
    return out


@pytest.mark.parametrize(
    ("q_len", "window", "softcap"), [(4, None, None), (2, None, None), (4, 2, None), (2, 1, None), (4, None, 0.5)]
)
def test_attention_backward_matches_numeric_gradient(q_len, window, softcap):
    q, k, v = gaussian(12, q_len, 4, 6), gaussian(13, 4, 2, 6), gaussian(14, 4, 2, 5)
    dout = gaussian(15, q_len, 4, 5)
    scale = 1 / np.sqrt(6)
    dq, dk, dv = numerics.attention_backward(q, k, v, dout, window=window, softcap=softcap)
    q64, k64, v64 = (a.astype(np.float64) for a in (q, k, v))

    def loss(a, b, c):
        return float((reference_attention(a, b, c, scale, window, softcap) * dout).sum())

    close(dq, numeric_gradient(lambda a: loss(a, k64, v64), q), 1e-6)
    close(dk, numeric_gradient(lambda a: loss(q64, a, v64), k), 1e-6)
    close(dv, numeric_gradient(lambda a: loss(q64, k64, a), v), 1e-6)


def test_cross_entropy_matches_float64():
    logits = gaussian(16, 4, 11, scale=3.0)
    targets = np.array([3, -1, 10, 0])
    loss, dlogits = numerics.cross_entropy(logits, targets, scale=0.25)
    l64 = logits.astype(np.float64)
    lse = np.log(np.exp(l64 - l64.max(1, keepdims=True)).sum(1)) + l64.max(1)
    kept = targets >= 0
    assert abs(loss - float((lse - l64[np.arange(4), targets])[kept].sum())) < 1e-12
    probs = np.exp(l64 - lse[:, None])
    probs[np.arange(4)[kept], targets[kept]] -= 1
    probs[~kept] = 0
    close(dlogits, probs * 0.25, 1e-7)


def test_embedding_backward_sums_rows_per_token():
    dy, tokens = gaussian(17, 6, 5), np.array([3, 1, 3, 0, 3, 1])
    expected = np.zeros((4, 5))
    np.add.at(expected, tokens, dy.astype(np.float64))
    close(numerics.embedding_backward(dy, tokens, 4), expected, 1e-7)
    with pytest.raises(ValueError):
        numerics.embedding_backward(dy, np.array([4, 0, 0, 0, 0, 0]), 4)


def test_adamw_step_is_the_documented_formula():
    param, grad = gaussian(18, 50), gaussian(19, 50)
    m, v = gaussian(20, 50, scale=0.1), np.abs(gaussian(21, 50, scale=0.1))
    p64, g64, m64, v64 = (a.astype(np.float64) for a in (param, grad, m, v))
    lr, b1, b2, eps, wd, c1, c2, s = 1e-3, 0.9, 0.999, 1e-8, 0.01, 1 - 0.9**3, 1 - 0.999**3, 0.5
    g = g64 * s
    m_new = (b1 * m64 + (1 - b1) * g).astype(np.float32)
    v_new = (b2 * v64 + (1 - b2) * g * g).astype(np.float32)
    expected = (p64 * (1 - lr * wd) - lr * (m_new / c1) / (np.sqrt(v_new / c2) + eps)).astype(np.float32)
    param, m, v = param.copy(), m.copy(), v.copy()
    numerics.adamw_step(param, grad, m, v, lr, b1, b2, eps, wd, c1, c2, s)
    assert param.tobytes() == expected.tobytes()
    assert m.tobytes() == m_new.tobytes() and v.tobytes() == v_new.tobytes()


# Decoder gradients


@pytest.fixture(scope="module", params=FAMILIES)
def model_file(request, tmp_path_factory) -> ModelFile:
    directory = tmp_path_factory.mktemp(request.param)
    config = tiny_config(request.param)
    write_hf_checkpoint(directory / "checkpoint", config)
    import_model(directory / "checkpoint", directory / "model.dllm")
    return ModelFile(directory / "model.dllm")


def reference_loss(config, w, tokens, targets) -> float:
    """float64 NumPy forward pass (test_transformer's yardstick) and summed next-token cross-entropy."""
    logits = reference_logits(config, w, tokens, every_position=True)
    top = logits.max(-1, keepdims=True)
    lse = (np.log(np.exp(logits - top).sum(-1, keepdims=True)) + top)[:, 0]
    return float(sum(lse[t] - logits[t, target] for t, target in enumerate(targets) if target >= 0))


def test_training_forward_matches_the_decoder(model_file):
    gradients = DecoderGradients(model_file.config)
    decoder = Transformer(model_file.config, model_file.tensors)
    logits = gradients.logits(model_file.tensors, TOKENS)
    for length in (1, 4, len(TOKENS)):
        assert logits[length - 1].tobytes() == decoder.forward(TOKENS[:length]).tobytes()


def test_decoder_gradients_match_finite_differences(model_file):
    config = model_file.config
    loss, grads = DecoderGradients(config).loss_and_gradients(model_file.tensors, TOKENS, TARGETS)
    w64 = {name: np.asarray(values, dtype=np.float64) for name, values in model_file.tensors.items()}
    assert abs(loss - reference_loss(config, w64, TOKENS, TARGETS)) < 1e-4
    assert set(grads) == set(config.tensor_shapes())
    random = numerics.DeterministicRandom(5)
    for name in tensor_order(grads):
        grad = grads[name]
        assert grad.shape == config.tensor_shapes()[name] and grad.dtype == np.float32
        for _ in range(3):
            index = np.unravel_index(random.next_u64() % grad.size, grad.shape)
            eps = 1e-6

            def at(delta, name=name, index=index):
                changed = dict(w64)
                changed[name] = w64[name].copy()
                changed[name][index] += delta
                return reference_loss(config, changed, TOKENS, TARGETS)

            numeric = (at(eps) - at(-eps)) / (2 * eps)
            assert abs(float(grad[index]) - numeric) < 2e-4 * max(1.0, abs(numeric)), (name, index)


def test_ignored_targets_and_scale(model_file):
    gradients = DecoderGradients(model_file.config)
    loss, grads = gradients.loss_and_gradients(model_file.tensors, TOKENS, TARGETS, scale=0.5)
    masked = [-1] * 4 + TARGETS[4:]
    masked_loss, _ = gradients.loss_and_gradients(model_file.tensors, TOKENS, masked)
    full_loss, full = gradients.loss_and_gradients(model_file.tensors, TOKENS, TARGETS)
    assert loss == full_loss and masked_loss < full_loss
    close(grads["final_norm.weight"], full["final_norm.weight"] * 0.5, 1e-7)


def test_gradients_are_bit_identical_across_runs_and_threads(model_file):
    gradients = DecoderGradients(model_file.config)

    def run(_):
        loss, grads = gradients.loss_and_gradients(model_file.tensors, TOKENS, TARGETS)
        return numerics.fingerprint(np.concatenate([grads[name].ravel() for name in tensor_order(grads)])), loss

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, range(8)))
    assert len(set(results)) == 1
    assert results[0][0] == GRADIENT_FINGERPRINT[model_file.config.family]


# Data order


def ascii_data(sequence_length: int = 8) -> TrainingData:
    documents = ["the quick brown fox jumps over the lazy dog", "pack my box with five dozen liquor jugs"]
    return TrainingData.from_documents(documents, lambda text: [ord(c) % 64 for c in text], sequence_length, 2)


def test_windows_overlap_by_one_and_cover_the_stream():
    data = ascii_data()
    stream = [ord(c) % 64 for c in "the quick brown fox jumps over the lazy dog"] + [2]
    assert data.windows[0] == tuple(stream[:9])
    assert data.windows[1][0] == data.windows[0][-1]
    assert all(2 <= len(w) <= 9 for w in data.windows)


def test_epoch_order_is_a_seeded_permutation():
    data = ascii_data(4)
    first = data.epoch_order(7, 0)
    assert sorted(first) == list(range(len(data)))
    assert first == data.epoch_order(7, 0)
    assert first != data.epoch_order(7, 1) and first != data.epoch_order(8, 0)
    size = 5
    samples = [w for step in range(2 * len(data) // size + 1) for w in data.batch(step, size, 7)]
    assert samples[: len(data)] == [data.windows[i] for i in first]


def test_read_documents(tmp_path):
    (tmp_path / "a.jsonl").write_text(
        '{"text": "one"}\n\n{"messages": [{"role": "user", "content": "hi"}]}\n', encoding="utf-8"
    )
    assert read_documents(tmp_path / "a.jsonl", lambda m: "<" + m[0]["content"] + ">") == ["one", "<hi>"]
    with pytest.raises(TrainingDataError):
        read_documents(tmp_path / "a.jsonl", None)
    (tmp_path / "b.jsonl").write_text('{"other": 1}\n', encoding="utf-8")
    with pytest.raises(TrainingDataError):
        read_documents(tmp_path / "b.jsonl", None)
    (tmp_path / "c.txt").write_text("plain text", encoding="utf-8")
    assert read_documents(tmp_path / "c.txt", None) == ["plain text"]


def test_learning_rate_schedule():
    config = AdamWConfig(learning_rate=1.0, warmup_steps=2, min_learning_rate=0.1)
    assert [config.learning_rate_at(s, 6) for s in (1, 2)] == [0.5, 1.0]
    assert config.learning_rate_at(6, 6) == pytest.approx(0.1)
    assert AdamWConfig(schedule="constant").learning_rate_at(50, 100) == 1e-4


# Fine-tuning runs

RUN = RunConfig(steps=6, batch_size=3, sequence_length=8, seed=3, optimizer=AdamWConfig(learning_rate=1e-2))


def test_loss_decreases(model_file):
    tuner = FineTuner.from_model_file(model_file, ascii_data(), RunConfig(20, 4, 8, 1, AdamWConfig(1e-1)))
    results = tuner.train()
    # Gemma 2's tiny logit cap (0.05) keeps every logit near zero, so its loss cannot fall far below log(64).
    margin = 0.01 if model_file.config.logits_softcap else 0.1
    assert np.mean([r.loss for r in results[-4:]]) < np.mean([r.loss for r in results[:4]]) - margin


def test_runs_are_byte_identical_and_resume_bit_for_bit(model_file, tmp_path):
    data = ascii_data()
    first = FineTuner.from_model_file(model_file, data, RUN)
    first.train()
    first_ckpt = first.save_checkpoint(tmp_path / "first.dllmckpt")

    second = FineTuner.from_model_file(model_file, data, RUN)
    second.train(until=2)
    second.save_checkpoint(tmp_path / "middle.dllmckpt")
    resumed = FineTuner.load_checkpoint(tmp_path / "middle.dllmckpt", data)
    assert resumed.step == 2
    resumed.train()
    assert resumed.save_checkpoint(tmp_path / "resumed.dllmckpt") == first_ckpt
    assert (tmp_path / "resumed.dllmckpt").read_bytes() == (tmp_path / "first.dllmckpt").read_bytes()
    assert resumed.losses == first.losses

    fingerprint = first.export(tmp_path / "first.dllm")
    assert resumed.export(tmp_path / "resumed.dllm") == fingerprint
    assert (tmp_path / "first.dllm").read_bytes() == (tmp_path / "resumed.dllm").read_bytes()
    assert fingerprint == FINETUNE_FINGERPRINT[model_file.config.family]

    tuned = ModelFile(tmp_path / "first.dllm")
    assert tuned.fine_tuning["base_fingerprint"] == model_file.fingerprint
    assert tuned.fine_tuning["data_fingerprint"] == data.fingerprint
    assert tuned.fine_tuning["steps_completed"] == 6
    assert "modified weights" in tuned.licence["attribution"]
    assert "otherwise unmodified" not in tuned.licence["attribution"]
    assert tuned.fingerprint != model_file.fingerprint
    imported, tuned_step = tuned.lineage
    assert imported["step"] == "import" and imported["output"] == model_file.fingerprint
    assert tuned_step["step"] == "fine_tune" and tuned_step["input"] == model_file.fingerprint
    assert (tuned_step["data"], tuned_step["steps"], tuned_step["output"]) == (data.fingerprint, 6, fingerprint)
    assert model_file.lineage == [{**imported, "output": model_file.fingerprint}]


def test_checkpoint_rejects_other_data_and_corruption(model_file, tmp_path):
    tuner = FineTuner.from_model_file(model_file, ascii_data(), RUN)
    tuner.train(until=1)
    path = tmp_path / "run.dllmckpt"
    tuner.save_checkpoint(path)
    with pytest.raises(CheckpointError):
        FineTuner.load_checkpoint(path, ascii_data(9))
    raw = bytearray(path.read_bytes())
    raw[-8] ^= 1
    path.write_bytes(bytes(raw))
    with pytest.raises(CheckpointError):
        FineTuner.load_checkpoint(path, ascii_data())


def test_cli_finetune_and_resume(tmp_path, capsys, monkeypatch):
    from test_bpe import smollm2_style
    from test_chat_template import SMOLLM2

    pytest.importorskip("tokenizers")
    reference = smollm2_style()
    config = {**TINY_LLAMA_CONFIG, "vocab_size": reference.get_vocab_size(), "eos_token_id": 2}
    write_hf_checkpoint(tmp_path / "checkpoint", config, tokenizer_json=json.loads(reference.to_str()))
    tokenizer_config = {"chat_template": SMOLLM2, "eos_token": "<|im_end|>", "bos_token": None}
    (tmp_path / "checkpoint" / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    import_model(tmp_path / "checkpoint", tmp_path / "base.dllm")

    def conversation(i: int) -> str:
        question = {"role": "user", "content": f"What is {i} plus {i}?"}
        return json.dumps({"messages": [question, {"role": "assistant", "content": f"{i} plus {i} is {2 * i}."}]})

    lines = [conversation(i) for i in range(8)]
    (tmp_path / "data.jsonl").write_text("\n".join(lines), encoding="utf-8")
    common = ["finetune", str(tmp_path / "base.dllm"), "--data", str(tmp_path / "data.jsonl")]
    common += ["--steps", "4", "--batch-size", "2", "--sequence-length", "16", "--learning-rate", "1e-2"]

    checkpoint = ["--checkpoint", str(tmp_path / "a.dllmckpt"), "--checkpoint-every", "2"]
    assert cli([*common, "-o", str(tmp_path / "a.dllm"), *checkpoint]) == 0
    first = capsys.readouterr().out
    assert "step     4/4" in first
    assert cli([*common, "-o", str(tmp_path / "b.dllm")]) == 0
    second = capsys.readouterr().out
    assert [line for line in second.splitlines() if line.startswith("step")] == [
        line for line in first.splitlines() if line.startswith("step")
    ]
    assert (tmp_path / "a.dllm").read_bytes() == (tmp_path / "b.dllm").read_bytes()
    # Resuming the finished run trains nothing and exports the same model.
    assert cli([*common, "-o", str(tmp_path / "c.dllm"), "--resume", str(tmp_path / "a.dllmckpt")]) == 0
    capsys.readouterr()
    assert (tmp_path / "c.dllm").read_bytes() == (tmp_path / "a.dllm").read_bytes()

    assert cli(["inspect", str(tmp_path / "a.dllm")]) == 0
    assert "fine-tuned:         4 steps" in capsys.readouterr().out
    assert cli(["--model", str(tmp_path / "a.dllm"), "chat", "What is 3 plus 3?", "--max-tokens", "4"]) == 0
    assert cli([*common, "-o", str(tmp_path / "missing.dllm"), "--resume", str(tmp_path / "nope")]) == 1

    # A training receipt: training again gives the same weights; other data or an edited receipt does not verify.
    receipt = tmp_path / "train.json"
    assert cli([*common, "--adapter-output", str(tmp_path / "x"), "--lora-rank", "2", "--receipt", str(receipt)]) == 0
    assert "receipt:            " in capsys.readouterr().out
    recorded = json.loads(receipt.read_text())
    assert recorded["output"]["steps"] == 4 and len(recorded["losses"]) == 4 and recorded["run"]["lora"]["rank"] == 2
    assert cli(["replay", str(receipt), "--base", str(tmp_path / "base.dllm")]) == 0
    assert "training again gave the same weights" in capsys.readouterr().out
    other = tmp_path / "other.jsonl"
    other.write_text("\n".join(conversation(i) for i in range(1, 9)), encoding="utf-8")
    assert cli(["replay", str(receipt), "--base", str(tmp_path / "base.dllm"), "--data", str(other), "--json"]) == 1
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["diverged_at"] == 1 and any("different training data" in r for r in outcome["reasons"])
    receipt.write_text(json.dumps({**recorded, "engine": "0.0.1"}))
    assert cli(["replay", str(receipt), "--base", str(tmp_path / "a.dllm")]) == 1
    output = capsys.readouterr().out
    assert "the receipt was edited" in output and "a different base model" in output and "note:" in output
    monkeypatch.delenv("DLLM_MODEL", raising=False)  # the chat above set it
    assert cli(["replay", str(receipt)]) == 2
    assert "needs the base model" in capsys.readouterr().err
    receipt.write_text(json.dumps({**recorded, "training_receipt": "dllm-train/0"}))
    assert cli(["replay", str(receipt), "--base", str(tmp_path / "base.dllm")]) == 2


def test_layouts_without_a_backward_pass_are_refused():
    """Interleaved rotary halves (imports convert them) have no gradient code."""
    import dataclasses

    from etalii_dllm.importing.importer import hf_config

    config = hf_config(tiny_config("llama"))
    with pytest.raises(ValueError, match="rotary layout"):
        DecoderGradients(dataclasses.replace(config, rope_interleaved=True))
