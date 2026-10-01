"""Steering vectors and ROME model edits: a steered or edited model is reproducible bit for bit, keeps the KV cache
and batch invariance, gets its own fingerprint, and the edit does what it says (``W' k = W k + delta``)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from model_fixtures import tiny_config, write_hf_checkpoint
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm import engine as engine_module
from etalii_dllm import modelfile, numerics
from etalii_dllm.cli import main
from etalii_dllm.engine import DllmEngine, default_engine
from etalii_dllm.importing import import_model
from etalii_dllm.interpret import trace
from etalii_dllm.interpret.editing import (
    EditRequest,
    key_covariance,
    rome,
    subject_position,
    write_edited_model,
)
from etalii_dllm.interpret.steering import SteeringVector, build_steering_vector, mean_activation
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.training.backprop import DecoderGradients
from etalii_dllm.transformer import Transformer, steered_fingerprint

PROMPT = [1, 17, 42, 5, 63, 0, 9, 9, 30]


@pytest.fixture
def isolated(monkeypatch):
    for name in (
        engine_module.MODEL_ENVIRONMENT_VARIABLE,
        engine_module.STEER_ENVIRONMENT_VARIABLE,
        engine_module.STEER_STRENGTH_ENVIRONMENT_VARIABLE,
    ):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    default_engine.cache_clear()
    yield monkeypatch
    default_engine.cache_clear()


def gaussian(seed: int, n: int) -> np.ndarray:
    return numerics.fill_gaussian(seed, n)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Transformer:
    directory = tmp_path_factory.mktemp("steer")
    write_hf_checkpoint(directory / "checkpoint", tiny_config("llama"))
    import_model(directory / "checkpoint", directory / "model.dllm")
    return Transformer.from_file(directory / "model.dllm")


def steered(model: Transformer, steering: dict[int, np.ndarray]) -> Transformer:
    return Transformer(model.config, model.tensors, weights_fingerprint=model.weights_fingerprint, steering=steering)


# -- steering ---------------------------------------------------------------------------------------------------


def test_steering_changes_the_output_and_the_fingerprint(tiny):
    zero = steered(tiny, {0: np.zeros(tiny.config.hidden_size, dtype=np.float32)})
    assert zero.forward(PROMPT).tobytes() == tiny.forward(PROMPT).tobytes()
    assert zero.weights_fingerprint not in ("", tiny.weights_fingerprint)
    vector = gaussian(3, tiny.config.hidden_size)
    model = steered(tiny, {0: vector})
    assert model.forward(PROMPT).tobytes() != tiny.forward(PROMPT).tobytes()
    assert model.weights_fingerprint == steered(tiny, {0: vector.copy()}).weights_fingerprint
    assert model.weights_fingerprint != steered(tiny, {1: vector}).weights_fingerprint
    assert steered_fingerprint("abc", None) == "abc"
    # The traced residual stream carries the vector.
    plain, pushed = trace(tiny, PROMPT), trace(model, PROMPT)
    expected = plain.middle[0] + plain.mlp_output[0] + vector
    assert expected.tobytes() == pushed.residual[1].tobytes()


def test_steered_models_keep_the_cache_and_batch_invariance(tiny):
    model = steered(tiny, {1: gaussian(4, tiny.config.hidden_size)})
    full = [model.forward(PROMPT[: i + 1]).tobytes() for i in range(len(PROMPT))]
    cache = model.new_cache()
    assert [model.forward_cached(PROMPT[: i + 1], cache).tobytes() for i in range(len(PROMPT))] == full
    batch = model.forward_batch([PROMPT, PROMPT[:3]])
    assert batch[0].tobytes() == full[-1] and batch[1].tobytes() == full[2]


def test_steering_rejects_bad_vectors(tiny):
    with pytest.raises(ValueError, match="steering"):
        steered(tiny, {tiny.config.layers: np.zeros(tiny.config.hidden_size, dtype=np.float32)})
    with pytest.raises(ValueError, match="steering"):
        steered(tiny, {0: np.zeros(3, dtype=np.float32)})


def test_steering_vector_files_round_trip(tmp_path, model_path):  # noqa: F811
    engine = DllmEngine.from_model_file(model_path)
    model, tokenizer = engine.model, engine.tokenizer
    vector = build_steering_vector(model, tokenizer, ["hello there", "hello"], ["the end"], 1, strength=2.5)
    expected = mean_activation(model, tokenizer, ["hello there", "hello"], 1) - mean_activation(
        model, tokenizer, ["the end"], 1
    )
    assert vector.vector.tobytes() == expected.tobytes()
    assert vector.origin == {"positive": ["hello there", "hello"], "negative": ["the end"]}
    path = tmp_path / "v.json"
    vector.save(path)
    loaded = SteeringVector.load(path)
    assert loaded.vector.tobytes() == vector.vector.tobytes() and loaded.strength == 2.5 and loaded.layer == 1
    assert loaded.scaled().tobytes() == (vector.vector * np.float32(2.5)).tobytes()
    assert list(loaded.for_model(model)) == [0]
    assert loaded.for_model(model, 1.0)[0].tobytes() == vector.vector.tobytes()
    with pytest.raises(ValueError, match="between"):
        SteeringVector(9, vector.vector).for_model(model)
    with pytest.raises(ValueError, match="values"):
        SteeringVector(1, vector.vector[:3]).for_model(model)
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not a steering vector"):
        SteeringVector.load(tmp_path / "bad.json")
    with pytest.raises(ValueError):
        SteeringVector.load(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="layer"):
        build_steering_vector(model, tokenizer, ["a"], ["b"], 0)
    with pytest.raises(ValueError, match="at least one"):
        mean_activation(model, tokenizer, [], 1)
    with pytest.raises(ValueError, match="no tokens"):
        mean_activation(model, tokenizer, [""], 1)
    with pytest.raises(ValueError, match="unsteered"):
        build_steering_vector(steered(model, loaded.for_model(model)), tokenizer, ["a"], ["b"], 1)


def test_engines_and_commands_apply_steering(tmp_path, model_path, capsys, isolated):  # noqa: F811
    path = tmp_path / "v.json"
    arguments = ["--model", str(model_path)]
    assert main([*arguments, "steer", "--positive", "hello there", "--negative", "the end", "-o", str(path)]) == 0
    assert "wrote:" in capsys.readouterr().out
    plain = DllmEngine.from_model_file(model_path)
    engine = DllmEngine.from_model_file(model_path, steer=path, steer_strength=3.0)
    assert engine.system_fingerprint != plain.system_fingerprint
    assert (
        engine.system_fingerprint
        == DllmEngine.from_model_file(model_path, steer=path, steer_strength=3.0).system_fingerprint
    )
    assert main([*arguments, "--steer", str(path), "--steer-strength", "3", "info"]) == 0
    assert engine.system_fingerprint in capsys.readouterr().out

    isolated.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(model_path))
    isolated.setenv(engine_module.STEER_ENVIRONMENT_VARIABLE, str(path))
    isolated.setenv(engine_module.STEER_STRENGTH_ENVIRONMENT_VARIABLE, "3")
    default_engine.cache_clear()
    assert default_engine().system_fingerprint == engine.system_fingerprint
    isolated.setenv(engine_module.STEER_STRENGTH_ENVIRONMENT_VARIABLE, "strong")
    default_engine.cache_clear()
    with pytest.raises(ValueError, match="expected a number"):
        default_engine()
    for name in (engine_module.STEER_ENVIRONMENT_VARIABLE, engine_module.STEER_STRENGTH_ENVIRONMENT_VARIABLE):
        isolated.delenv(name)
    default_engine.cache_clear()

    lines = tmp_path / "positive.txt"
    lines.write_text("hello\n\nhello there\n", encoding="utf-8")
    assert (
        main([*arguments, "steer", "--positive-file", str(lines), "--negative", "x", "--layer", "2", "-o", str(path)])
        == 0
    )
    assert json.loads(path.read_text(encoding="utf-8"))["origin"]["positive"] == ["hello", "hello there"]
    capsys.readouterr()
    assert main([*arguments, "steer", "--positive", "a", "-o", str(path)]) == 1
    assert "at least one" in capsys.readouterr().err
    assert (
        main([*arguments, "steer", "--positive-file", str(tmp_path / "none.txt"), "--negative", "x", "-o", str(path)])
        == 1
    )


def test_a_steering_file_must_fit_the_model(tmp_path, model_path):  # noqa: F811
    path = tmp_path / "v.json"
    SteeringVector(1, np.zeros(5, dtype=np.float32)).save(path)
    with pytest.raises(ValueError, match="does not fit"):
        DllmEngine.from_model_file(model_path, steer=path)


# -- editing ----------------------------------------------------------------------------------------------------


def test_residual_gradient_matches_finite_differences(tiny):
    weights = {name: tensor.numpy() for name, tensor in tiny.tensors.items()}
    gradients = DecoderGradients(tiny.config)
    tokens, targets = PROMPT[:-1], [-1, -1, -1, -1, 5, 6, 7, 8]
    zero = np.zeros((len(tokens), tiny.config.hidden_size), dtype=np.float32)
    loss, gradient = gradients.residual_gradient(weights, tokens, targets, 0, zero)
    full, _ = gradients.loss_and_gradients(weights, tokens, targets)
    assert loss == pytest.approx(full, rel=1e-6)
    assert gradient.shape == zero.shape
    for position, dimension in ((2, 3), (4, 0), (6, 11)):
        step = np.zeros_like(zero)
        step[position, dimension] = 1e-2
        up, _ = gradients.residual_gradient(weights, tokens, targets, 0, step)
        down, _ = gradients.residual_gradient(weights, tokens, targets, 0, -step)
        assert (up - down) / 2e-2 == pytest.approx(float(gradient[position, dimension]), rel=2e-2, abs=1e-4)
    last, _ = gradients.residual_gradient(weights, tokens, targets, tiny.config.layers - 1, zero)
    assert last == pytest.approx(full, rel=1e-6)
    with pytest.raises(ValueError):
        gradients.residual_gradient(weights, tokens, targets, tiny.config.layers, zero)
    with pytest.raises(ValueError):
        gradients.residual_gradient(weights, tokens, targets[:-1], 0, zero)


def test_subject_position(model_path):  # noqa: F811
    tokenizer = DllmEngine.from_model_file(model_path).tokenizer
    tokens, position = subject_position(tokenizer, "the cat sat on the mat", "cat")
    assert tokenizer.decode(tokens[: position + 1]).endswith("cat")
    assert not tokenizer.decode(tokens[:position]).endswith("cat")
    with pytest.raises(ValueError, match="does not contain"):
        subject_position(tokenizer, "the cat", "dog")


def test_rome_edit(tmp_path, model_path):  # noqa: F811
    engine = DllmEngine.from_model_file(model_path)
    model, tokenizer = engine.model, engine.tokenizer
    request = EditRequest("the cat sat on the", "cat", " mat")
    corpus = ["hello there, general", "the end of the story", "a b c d e f g"]
    result = rome(model, tokenizer, request, layer=1, corpus=corpus, steps=20, contexts=["so "])
    changed = [
        name for name in result.weights if result.weights[name].tobytes() != model.tensors[name].numpy().tobytes()
    ]
    assert changed == ["layers.0.mlp.down.weight"]
    assert result.probability_after > result.probability_before
    record = result.record
    assert record["layer"] == 1 and record["contexts"] == ["so "] and record["covariance"]["texts"] == 3
    # The edited projection maps the key to the old value plus delta (up to float32 rounding).
    keys = []
    for prefix in ("", "so "):
        tokens, position = subject_position(tokenizer, prefix + request.prompt, request.subject)
        keys.append(trace(model, tokens).mlp_activation[0][position])
    key = numerics.column_mean(np.stack(keys))
    before = numerics.linear(key[None], model.tensors["layers.0.mlp.down.weight"].numpy()).numpy()[0]
    after = numerics.linear(key[None], result.weights["layers.0.mlp.down.weight"]).numpy()[0]
    assert float(np.sqrt(numerics.sum_squares(after - before))) == pytest.approx(record["delta_norm"], rel=1e-3)

    again = rome(model, tokenizer, request, layer=1, corpus=corpus, steps=20, contexts=["so "])
    assert again.record == record
    base = ModelFile(model_path)
    first, second = tmp_path / "a.dllm", tmp_path / "b.dllm"
    assert write_edited_model(base, result, first) == write_edited_model(base, again, second)
    assert first.read_bytes() == second.read_bytes()
    edited = ModelFile(first)
    assert edited.edits == [record]
    with pytest.raises(ValueError, match="other weights"):
        write_edited_model(edited, result, tmp_path / "wrong.dllm")
    edited_engine = DllmEngine.from_model_file(first)
    second_request = EditRequest("the dog sat on the", "dog", " rug")
    follow = rome(edited_engine.model, tokenizer, second_request, layer=1, corpus=corpus, steps=5)
    twice = tmp_path / "c.dllm"
    write_edited_model(edited, follow, twice)
    assert len(ModelFile(twice).edits) == 2
    assert ModelFile(twice).licence == edited.licence
    steps = ModelFile(twice).lineage
    assert [step["step"] for step in steps] == ["import", "edit", "edit"]
    assert steps[1]["input"] == base.fingerprint and steps[2]["input"] == edited.fingerprint
    assert modelfile.lineage_problems(steps) == []


def test_rome_rejects_bad_requests(model_path, tmp_path_factory):  # noqa: F811
    engine = DllmEngine.from_model_file(model_path)
    model, tokenizer = engine.model, engine.tokenizer
    request = EditRequest("the cat sat", "cat", " mat")
    with pytest.raises(ValueError, match="layer"):
        rome(model, tokenizer, request, layer=0)
    with pytest.raises(ValueError, match="steps"):
        rome(model, tokenizer, request, steps=0)
    with pytest.raises(ValueError, match="target"):
        rome(model, tokenizer, EditRequest("the cat sat", "cat", ""))
    with pytest.raises(ValueError, match="corpus"):
        rome(model, tokenizer, request, corpus=[""], steps=1)
    with pytest.raises(ValueError, match="unsteered"):
        rome(steered(model, {0: np.zeros(model.config.hidden_size, dtype=np.float32)}), tokenizer, request)
    directory = tmp_path_factory.mktemp("olmo")
    write_hf_checkpoint(directory / "checkpoint", tiny_config("olmo2"))
    import_model(directory / "checkpoint", directory / "model.dllm")
    with pytest.raises(ValueError, match="fine-tuning support"):
        rome(Transformer.from_file(directory / "model.dllm"), tokenizer, request)


def test_key_covariance_is_symmetric(model_path):  # noqa: F811
    engine = DllmEngine.from_model_file(model_path)
    covariance = key_covariance(engine.model, engine.tokenizer, ["hello there", "the end"], 0)
    assert covariance.tobytes() == covariance.T.copy().tobytes()
    keys = np.concatenate(
        [trace(engine.model, engine.tokenizer.encode(t)).mlp_activation[0] for t in ("hello there", "the end")]
    ).astype(np.float64)
    np.testing.assert_allclose(covariance, keys.T @ keys / keys.shape[0], rtol=1e-5, atol=1e-7)


def test_edit_command(tmp_path, model_path, capsys, isolated):  # noqa: F811
    output = tmp_path / "edited.dllm"
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello there\nthe end of the story\n", encoding="utf-8")
    arguments = ["edit", str(model_path), "--prompt", "the cat sat on the", "--subject", "cat", "--target", " mat"]
    assert main([*arguments, "--corpus", str(corpus), "--steps", "5", "-o", str(output)]) == 0
    out = capsys.readouterr().out
    assert "p(target):" in out and output.exists()
    assert main(["inspect", str(output)]) == 0
    assert "edited:             rome at layer 1" in capsys.readouterr().out
    assert main([*arguments[:-2], "--target", " mat", "--subject", "dog", "-o", str(output)]) == 1
    assert capsys.readouterr().err.startswith("dllm edit: ")
    assert main(["edit", str(tmp_path / "missing.dllm"), *arguments[2:], "-o", str(output)]) == 1
