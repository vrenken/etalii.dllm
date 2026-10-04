"""Editing and sparse autoencoders for T5 text-to-text models (#407-#410): the gradient of the encoder's residual
stream against transformers' autograd and finite differences, ROME edits of encoder MLPs with byte-identical files,
sparse autoencoders on the decoder's residual stream and their features as decoder steering vectors, the embedding
explorer, and ``dllm edit|sae|neighbours`` on T5 models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_encoders import encoder_path  # noqa: F401 - fixture
from test_t5 import TEXTS
from test_t5_generation import FLAN, write_checkpoint

from etalii_dllm import numerics
from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import import_model
from etalii_dllm.interpret import trace_text_to_text
from etalii_dllm.interpret.editing import EditRequest, rome, subject_position, write_edited_model
from etalii_dllm.interpret.embeddings import neighbours
from etalii_dllm.interpret.sae import SaeConfig, SparseAutoencoder, collect_activations, feature_report, train_sae
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.seq2seq import TextToText
from etalii_dllm.training.encoder_backprop import EncoderGradients
from etalii_dllm.training.seq2seq_backprop import TextToTextGradients

pytest.importorskip("tokenizers")
pytest.importorskip("sentencepiece")

END = 1
REQUEST = EditRequest("The quick brown fox jumps over the lazy dog", "fox", "dog")


@pytest.fixture(scope="module")
def t5(tmp_path_factory) -> tuple[Path, DllmEngine]:
    directory = tmp_path_factory.mktemp("t5-editing")
    write_checkpoint(directory / "checkpoint")
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-t5-editing")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


@pytest.fixture(scope="module")
def flan(tmp_path_factory) -> tuple[Path, DllmEngine]:
    directory = tmp_path_factory.mktemp("flan-editing")
    write_checkpoint(directory / "checkpoint", **FLAN)
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-flan-t5-editing")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


def weights_of(model: TextToText) -> dict[str, np.ndarray]:
    return {name: tensor.numpy() for name, tensor in model.tensors.items()}


# The encoder's residual gradient (#407)


def test_encoder_residual_gradient_matches_autograd(t5, flan):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    for checkpoint, engine in (t5, flan):
        model = engine.model
        gradients = TextToTextGradients(model.config)
        weights = weights_of(model)
        source = [*engine.tokenizer.encode(TEXTS[0]), END]
        target = engine.tokenizer.encode(TEXTS[1])[:5]
        delta = (numerics.fill_gaussian(5, len(source) * 32).reshape(len(source), 32) * 0.1).astype(np.float32)
        reference = transformers.T5ForConditionalGeneration.from_pretrained(checkpoint).eval().double()
        for layer in range(model.config.layers):
            loss, gradient = gradients.residual_gradient(weights, source, target, layer, delta)
            added = torch.tensor(delta, dtype=torch.float64, requires_grad=True)

            def hook(module, inputs, output, added=added):
                return (output[0] + added, *output[1:])

            handle = reference.encoder.block[layer].register_forward_hook(hook)
            output = reference(input_ids=torch.tensor([source]), decoder_input_ids=torch.tensor([[0, *target[:-1]]]))
            expected = torch.nn.functional.cross_entropy(output.logits[0], torch.tensor(target), reduction="sum")
            expected.backward()
            handle.remove()
            assert loss == pytest.approx(expected.item(), rel=1e-5)
            np.testing.assert_allclose(gradient, added.grad.numpy(), atol=2e-5, rtol=1e-4)


def test_encoder_residual_gradient_matches_finite_differences(flan):
    _, engine = flan
    model = engine.model
    gradients = TextToTextGradients(model.config)
    weights = weights_of(model)
    source = [*engine.tokenizer.encode(TEXTS[3]), END]
    target = engine.tokenizer.encode(TEXTS[1])[:4]
    zero = np.zeros((len(source), model.config.hidden_size), dtype=np.float32)
    loss, gradient = gradients.residual_gradient(weights, source, target, 0, zero)
    full, _ = gradients.loss_and_gradients(weights, source, target)
    assert loss == full  # a zero delta changes no bit
    for position, dimension in ((1, 3), (4, 0), (len(source) - 1, 17)):
        step = np.zeros_like(zero)
        step[position, dimension] = 1e-2
        up, _ = gradients.residual_gradient(weights, source, target, 0, step)
        down, _ = gradients.residual_gradient(weights, source, target, 0, -step)
        assert (up - down) / 2e-2 == pytest.approx(float(gradient[position, dimension]), rel=2e-2, abs=1e-4)
    with pytest.raises(ValueError, match="layer must be between 0 and 1"):
        gradients.residual_gradient(weights, source, target, 2, zero)
    with pytest.raises(ValueError, match="target"):
        gradients.residual_gradient(weights, source, [], 0, zero)


def test_only_t5_encoders_take_a_residual_delta(encoder_path):  # noqa: F811 - fixture
    file = ModelFile(encoder_path)
    gradients = EncoderGradients(file.config)
    weights = {name: np.asarray(values) for name, values in file.tensors.items()}
    zero = np.zeros((3, file.config.hidden_size), dtype=np.float32)
    with pytest.raises(ValueError, match="only T5 encoders"):
        gradients.encode(weights, [2, 5, 3], add=(0, zero))
    with pytest.raises(ValueError, match="defined for T5"):
        gradients.residual_gradient(weights, gradients.encode(weights, [2, 5, 3]), zero, 0)


# ROME (#407)


def test_rome_edits_an_encoder_mlp(t5, flan, tmp_path, cli_environment):
    from golden_values import T5_EDITING_FINGERPRINT

    for name, (checkpoint, engine) in (("t5", t5), ("flan", flan)):
        model, tokenizer = engine.model, engine.tokenizer
        result = rome(model, tokenizer, REQUEST, layer=2, steps=20, corpus=TEXTS, contexts=["so "])
        changed = [n for n in result.weights if result.weights[n].tobytes() != model.tensors[n].numpy().tobytes()]
        assert changed == ["layers.1.mlp.down.weight"]
        record = result.record
        assert record["stack"] == "encoder" and record["layer"] == 2 and record["contexts"] == ["so "]
        assert result.probability_after > result.probability_before
        # the edited projection maps the key to the old value plus delta (up to float32 rounding)
        encoder = TextToTextGradients(model.config).encoder
        keys = []
        for prefix in ("", "so "):
            tokens, position = subject_position(tokenizer, prefix + REQUEST.prompt, REQUEST.subject)
            keys.append(encoder.encode(weights_of(model), [*tokens, END]).layers[1]["inner"][position])
        key = numerics.column_mean(np.stack(keys))
        before = numerics.linear(key[None], model.tensors["layers.1.mlp.down.weight"].numpy()).numpy()[0]
        after = numerics.linear(key[None], result.weights["layers.1.mlp.down.weight"]).numpy()[0]
        assert float(np.sqrt(numerics.sum_squares(after - before))) == pytest.approx(record["delta_norm"], rel=1e-3)
        again = rome(model, tokenizer, REQUEST, layer=2, steps=20, corpus=TEXTS, contexts=["so "])
        assert again.record == record
        base = ModelFile(checkpoint.parent / "model.dllm")
        first, second = tmp_path / f"{name}-a.dllm", tmp_path / f"{name}-b.dllm"
        fingerprint = write_edited_model(base, result, first)
        assert fingerprint == write_edited_model(base, again, second) == T5_EDITING_FINGERPRINT[f"edit_{name}"]
        assert first.read_bytes() == second.read_bytes()
        edited = ModelFile(first)
        assert edited.edits == [record] and [s["step"] for s in edited.lineage] == ["import", "edit"]
        served = DllmEngine.from_model_file(first).model
        source = [*tokenizer.encode(REQUEST.prompt), END]
        logits = served.answer_logits(source, tokenizer.encode(REQUEST.target))
        probability = 1.0
        for row, token in zip(logits, tokenizer.encode(REQUEST.target), strict=True):
            probability *= float(numerics.softmax(row)[token])
        assert probability == result.probability_after


def test_rome_refusals(flan):
    checkpoint, engine = flan
    model, tokenizer = engine.model, engine.tokenizer
    for options, message in (
        ({"layer": 0}, "encoder layers"),
        ({"layer": 3}, "encoder layers"),
        ({"steps": 0}, "steps"),
        ({"expert": "shared"}, "no experts"),
    ):
        with pytest.raises(ValueError, match=message):
            rome(model, tokenizer, REQUEST, **options)
    with pytest.raises(ValueError, match="no tokens"):
        rome(model, tokenizer, EditRequest(REQUEST.prompt, REQUEST.subject, ""))
    with pytest.raises(ValueError, match="must not contain </s>"):
        rome(model, tokenizer, EditRequest(REQUEST.prompt, REQUEST.subject, "a </s>"))
    with pytest.raises(ValueError, match="does not contain"):
        rome(model, tokenizer, EditRequest(REQUEST.prompt, "cat", "dog"))
    tensors = ModelFile(checkpoint.parent / "model.dllm").tensors
    steered = TextToText(model.config, tensors, steering={0: np.zeros(model.config.hidden_size)})
    with pytest.raises(ValueError, match="unsteered"):
        rome(steered, tokenizer, REQUEST)


def test_edit_command(flan, tmp_path, capsys, cli_environment):
    from etalii_dllm.cli import main

    checkpoint, _ = flan
    path = str(checkpoint.parent / "model.dllm")
    output = tmp_path / "edited.dllm"
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(TEXTS) + "\n", encoding="utf-8")
    arguments = ["edit", path, "--prompt", REQUEST.prompt, "--subject", "fox", "--target", "dog", "--layer", "2"]
    assert main([*arguments, "--corpus", str(corpus), "--steps", "5", "-o", str(output)]) == 0
    out = capsys.readouterr().out
    assert "edited:             encoder layer 2, 5 steps" in out and "p(target):" in out
    assert main(["inspect", str(output)]) == 0
    assert "edited:             rome at encoder layer 2" in capsys.readouterr().out
    assert main([*arguments, "--expert", "shared", "-o", str(output)]) == 1
    assert "no experts" in capsys.readouterr().err


# Sparse autoencoders (#408)


def test_sparse_autoencoders_on_the_decoder(flan, tmp_path):
    from golden_values import T5_EDITING_FINGERPRINT

    _, engine = flan
    model, tokenizer = engine.model, engine.tokenizer
    activations = collect_activations(model, tokenizer, TEXTS, 1, outlier=0)
    counts = [len(tokenizer.encode(text)) for text in TEXTS]
    assert activations.values.shape == (sum(counts), model.config.hidden_size)
    assert activations.positions[: counts[0]] == [(0, p) for p in range(counts[0])]
    first = trace_text_to_text(model, [END, *tokenizer.encode(TEXTS[0])], attention=False, logits=False)
    assert activations.values[: counts[0]].tobytes() == first.residual[1][1:].tobytes()
    config = SaeConfig(features=16, steps=8, batch_size=8, learning_rate=1e-2, l1=0.5, seed=2)
    sae = train_sae(activations, config, model.weights_fingerprint)
    sae.save(tmp_path / "a.safetensors")
    train_sae(activations, config, model.weights_fingerprint).save(tmp_path / "b.safetensors")
    data = (tmp_path / "a.safetensors").read_bytes()
    assert data == (tmp_path / "b.safetensors").read_bytes()
    assert hashlib.sha256(data).hexdigest() == T5_EDITING_FINGERPRINT["sae"]
    report = feature_report(sae, activations, top=2, count=3)
    assert len(report) == 3 and all(e.position < counts[e.text] for f in report for e in f.examples)
    vector = sae.steering_vector(1, strength=2.0)
    assert vector.layer == 1 and vector.for_model(model)[0].tobytes() == vector.scaled().tobytes()
    vector.save(tmp_path / "feature.json")
    steered = DllmEngine.from_model_file(flan[0].parent / "model.dllm", steer=tmp_path / "feature.json")
    assert steered.system_fingerprint != engine.system_fingerprint
    with pytest.raises(ValueError, match="layer must be between 1 and 2"):
        collect_activations(model, tokenizer, TEXTS, 3)
    with pytest.raises(ValueError, match="contains </s>"):
        collect_activations(model, tokenizer, ["a </s> b"], 1)
    assert SparseAutoencoder.load(tmp_path / "a.safetensors").layer == 1


def test_sae_commands(flan, tmp_path, capsys, cli_environment):
    from etalii_dllm.cli import main

    checkpoint, _ = flan
    model = ["--model", str(checkpoint.parent / "model.dllm")]
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(TEXTS) + "\n", encoding="utf-8")
    sae, vector = tmp_path / "sae.safetensors", tmp_path / "v.json"
    arguments = ["sae", "train", "--corpus", str(corpus), "--features", "8", "--steps", "3", "--batch-size", "4"]
    assert main([*model, *arguments, "-o", str(sae)]) == 0
    assert SparseAutoencoder.load(sae).layer == 1  # half of the two decoder layers
    capsys.readouterr()
    assert main([*model, "sae", "features", str(sae), "--corpus", str(corpus), "--feature", "2", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["feature"] == 2
    assert main([*model, "sae", "steer", str(sae), "--feature", "2", "-o", str(vector)]) == 0
    capsys.readouterr()
    assert main([*model, "--steer", str(vector), "generate", "--prompt", TEXTS[0], "--max-tokens", "3"]) == 0


# The embedding explorer (#409)


def test_neighbours_on_text_to_text_models(t5, flan, capsys, cli_environment):
    from etalii_dllm.cli import main

    for checkpoint, engine in (t5, flan):
        model = engine.model
        assert neighbours(model, engine.tokenizer, "fox", 5).neighbours
        output = neighbours(model, engine.tokenizer, "fox - dog", 4, "output")
        head = "token_embedding.weight" if model.config.tie_word_embeddings else "lm_head.weight"
        assert output.space == "output" and len(output.neighbours) == 4
        path = str(checkpoint.parent / "model.dllm")
        assert main(["--model", path, "neighbours", "fox", "--top-k", "3", "--json"]) == 0
        result = json.loads(capsys.readouterr().out)
        expected = neighbours(model, engine.tokenizer, "fox", 3)
        assert [n["id"] for n in result["neighbours"]] == [n.token for n in expected.neighbours]
        assert head in model.tensors
    assert main(["--model", str(flan[0].parent / "model.dllm"), "experts", "--prompt", "x"]) == 1
    assert "needs a decoder-only model" in capsys.readouterr().err
