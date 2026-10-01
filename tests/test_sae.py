"""Sparse autoencoders: gradients against finite differences, reproducible training (byte-identical files), unit
decoder columns, the feature report, steering with a feature and the ``dllm sae`` commands."""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm import engine as engine_module
from etalii_dllm import numerics
from etalii_dllm.cli import main
from etalii_dllm.engine import DllmEngine, default_engine
from etalii_dllm.interpret.sae import (
    SaeConfig,
    SparseAutoencoder,
    collect_activations,
    feature_report,
    train_sae,
)

CORPUS = ["hello there, general", "the end of the story", "a cat sat on the mat", "hello hello the cat"]


@pytest.fixture(scope="module")
def engine(model_path):  # noqa: F811
    return DllmEngine.from_model_file(model_path)


@pytest.fixture(scope="module")
def activations(engine):
    return collect_activations(engine.model, engine.tokenizer, CORPUS, 1, outlier=0)


def test_collect_activations(engine, activations):
    total = sum(len(engine.tokenizer.encode(text)) for text in CORPUS)
    assert activations.values.shape == (total, engine.model.config.hidden_size)
    assert activations.positions[0] == (0, 0) and len(activations.positions) == total
    filtered = collect_activations(engine.model, engine.tokenizer, CORPUS, 1, outlier=1.0)
    assert 0 < filtered.values.shape[0] < total and len(filtered.positions) == filtered.values.shape[0]
    with pytest.raises(ValueError, match="layer"):
        collect_activations(engine.model, engine.tokenizer, CORPUS, 0)
    with pytest.raises(ValueError, match="no tokens"):
        collect_activations(engine.model, engine.tokenizer, [""], 1)


def test_gradients_match_finite_differences(activations):
    sae = SparseAutoencoder.initial(activations, SaeConfig(features=24, l1=0.3))
    sae.params["encoder.bias"] = (numerics.fill_gaussian(9, 24) * np.float32(0.5)).astype(np.float32)
    x = sae.scale(activations.values[:6])
    loss, _, gradients = sae.loss_and_gradients(x)
    for name, index in (
        ("encoder.weight", (3, 2)),
        ("decoder.weight", (5, 7)),
        ("decoder.bias", (4,)),
        ("encoder.bias", (2,)),
    ):
        original = sae.params[name][index]
        values = []
        for sign in (1, -1):
            sae.params[name][index] = original + sign * np.float32(2e-3)
            values.append(sae.loss_and_gradients(x)[0])
        sae.params[name][index] = original
        numeric = (values[0] - values[1]) / 4e-3
        assert numeric == pytest.approx(float(gradients[name][index]), rel=5e-2, abs=2e-3), name
    assert loss > 0


def test_training_is_reproducible(tmp_path, engine, activations):
    config = SaeConfig(features=32, steps=12, batch_size=8, learning_rate=1e-2, l1=0.5, seed=3)
    seen = []
    first = train_sae(activations, config, "fp", seen.append)
    second = train_sae(activations, config, "fp")
    assert [s.step for s in seen] == list(range(1, 13)) and seen[-1].loss < seen[0].loss
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    first.save(a)
    second.save(b)
    assert a.read_bytes() == b.read_bytes()
    norms = numerics.linear(np.ones((1, 16), np.float32), (first.params["decoder.weight"] ** 2).T.copy()).numpy()
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    loaded = SparseAutoencoder.load(a)
    assert loaded.layer == 1 and loaded.features == 32 and loaded.hidden_size == 16 and loaded.config == config
    assert loaded.model_fingerprint == "fp" and loaded.history["rows"] == activations.values.shape[0]
    assert loaded.encode(activations.values).tobytes() == first.encode(activations.values).tobytes()
    f = loaded.encode(activations.values)
    assert (f >= 0).all() and loaded.decode(f).shape == activations.values.shape
    other = train_sae(activations, SaeConfig(features=32, steps=12, batch_size=8, learning_rate=1e-2, l1=0.5, seed=4))
    assert other.params["decoder.weight"].tobytes() != first.params["decoder.weight"].tobytes()


def test_feature_report_and_steering(engine, activations):
    sae = train_sae(activations, SaeConfig(features=16, steps=6, batch_size=8, learning_rate=1e-2, l1=0.1))
    report = feature_report(sae, activations, top=3, count=4)
    assert len(report) == 4
    maxima = [feature.max_activation for feature in report]
    assert maxima == sorted(maxima, reverse=True)
    for feature in report:
        activations_shown = [e.activation for e in feature.examples]
        assert activations_shown == sorted(activations_shown, reverse=True) and len(activations_shown) <= 3
        assert 0 <= feature.frequency <= 1
    assert [f.index for f in feature_report(sae, activations, [2, 0])] == [2, 0]
    with pytest.raises(ValueError, match="feature"):
        feature_report(sae, activations, [16])
    elsewhere = collect_activations(engine.model, engine.tokenizer, CORPUS, 2, outlier=0)
    with pytest.raises(ValueError, match="layer"):
        feature_report(sae, elsewhere)
    vector = sae.steering_vector(2, strength=3.0)
    assert vector.layer == 1 and vector.strength == 3.0 and vector.origin["sae_feature"] == 2
    expected = sae.params["decoder.weight"][:, 2] / np.float32(sae.input_scale)
    assert vector.vector.tobytes() == expected.astype(np.float32).tobytes()
    with pytest.raises(ValueError):
        sae.steering_vector(16)


def test_bad_settings_and_files(tmp_path):
    with pytest.raises(ValueError):
        SaeConfig(features=0)
    with pytest.raises(ValueError):
        SaeConfig(learning_rate=0)
    (tmp_path / "junk.safetensors").write_bytes(b"junk")
    with pytest.raises(ValueError):
        SparseAutoencoder.load(tmp_path / "junk.safetensors")
    from etalii_dllm.importing.safetensors import write_safetensors

    write_safetensors(tmp_path / "other.safetensors", {"x": np.zeros(2, np.float32)}, {"format": "other"})
    with pytest.raises(ValueError, match="not a sparse autoencoder"):
        SparseAutoencoder.load(tmp_path / "other.safetensors")


def test_sae_commands(tmp_path, model_path, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, "")
    monkeypatch.delenv(engine_module.MODEL_ENVIRONMENT_VARIABLE)
    default_engine.cache_clear()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(CORPUS) + "\n", encoding="utf-8")
    sae, vector = tmp_path / "sae.safetensors", tmp_path / "v.json"
    model = ["--model", str(model_path)]
    arguments = ["sae", "train", "--corpus", str(corpus), "--features", "16", "--steps", "4", "--batch-size", "8"]
    assert main([*model, *arguments, "-o", str(sae)]) == 0
    captured = capsys.readouterr()
    assert "wrote:" in captured.out and "step      4" in captured.err
    assert main([*model, "sae", "features", str(sae), "--corpus", str(corpus), "--count", "2"]) == 0
    assert capsys.readouterr().out.startswith("feature ")
    assert main([*model, "sae", "features", str(sae), "--corpus", str(corpus), "--feature", "3", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["feature"] == 3
    assert main([*model, "sae", "steer", str(sae), "--feature", "3", "-o", str(vector)]) == 0
    assert json.loads(vector.read_text(encoding="utf-8"))["origin"]["sae_feature"] == 3
    capsys.readouterr()
    assert main([*model, "sae", "steer", str(tmp_path / "none.safetensors"), "--feature", "3", "-o", str(vector)]) == 1
    assert capsys.readouterr().err.startswith("dllm sae: ")
    default_engine.cache_clear()
