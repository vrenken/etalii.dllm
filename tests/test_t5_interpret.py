"""Interpretability of T5 text-to-text models (#402-#405): the biased attention probabilities kernel, traces of the
encoder and decoder that change no bit (against transformers' ``output_hidden_states``/``output_attentions``), the
logit lens over decoder layers, self- and cross-attention maps and steering vectors on the decoder, in the library
and through ``dllm lens|attention|steer``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_encoders import encoder_path  # noqa: F401 - fixture
from test_engine_import import model_path  # noqa: F401 - fixture
from test_t5 import TEXTS
from test_t5_generation import FLAN, write_checkpoint

from etalii_dllm import numerics
from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import import_model
from etalii_dllm.interpret import TextToTextTrace, logit_lens, top_k, trace_text_to_text
from etalii_dllm.interpret.steering import SteeringVector, build_steering_vector, mean_activation
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.seq2seq import TextToText

pytest.importorskip("tokenizers")
pytest.importorskip("sentencepiece")

END = 1


@pytest.fixture(scope="module")
def t5(tmp_path_factory) -> tuple[Path, DllmEngine]:
    directory = tmp_path_factory.mktemp("t5-interpret")
    write_checkpoint(directory / "checkpoint")
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-t5-interpret")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


@pytest.fixture(scope="module")
def flan(tmp_path_factory) -> tuple[Path, DllmEngine]:
    directory = tmp_path_factory.mktemp("flan-interpret")
    write_checkpoint(directory / "checkpoint", **FLAN)
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-flan-t5-interpret")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


def sequence(engine: DllmEngine, source: str = TEXTS[0], answer: str = TEXTS[1], length: int = 6):
    return [*engine.tokenizer.encode(source), END], engine.tokenizer.encode(answer)[:length]


def gaussian(seed: int, *shape: int) -> np.ndarray:
    return numerics.fill_gaussian(seed, int(np.prod(shape))).reshape(shape)


# The biased attention probabilities kernel (#404)


def test_biased_attention_weights_are_the_probabilities_biased_attention_applies():
    q, k, v, bias = gaussian(1, 5, 4, 8), gaussian(2, 7, 4, 8), gaussian(3, 7, 4, 6), gaussian(4, 4, 5, 7)
    weights = numerics.biased_attention_weights(q, k, bias, 0.3).numpy()
    assert weights.shape == (5, 4, 7)
    out = numerics.biased_attention(q, k, v, bias, 0.3).numpy()
    mixed = np.einsum("thj,jhd->thd", weights.astype(np.float64), v.astype(np.float64))
    np.testing.assert_allclose(mixed, out, atol=1e-6)
    scores = np.einsum("thd,jhd->htj", q.astype(np.float64), k.astype(np.float64))
    scores = ((scores + bias) * 0.3).transpose(1, 0, 2)
    reference = np.exp(scores - scores.max(-1, keepdims=True))
    reference /= reference.sum(-1, keepdims=True)
    np.testing.assert_allclose(weights, reference, atol=1e-7)
    # a zero bias gives exactly the bits of the unbiased, unmasked kernel
    zero = np.zeros_like(bias)
    plain = numerics.attention_weights(q, k, scale=0.3, causal=False).numpy()
    assert numerics.biased_attention_weights(q, k, zero, 0.3).numpy().tobytes() == plain.tobytes()
    results = set()
    try:
        for threads in (1, 3, 8):
            numerics.set_threads(threads)
            results.add(numerics.biased_attention_weights(q, k, bias, 0.3).numpy().tobytes())
    finally:
        numerics.set_threads(0)
    assert len(results) == 1
    assert numerics.biased_attention_weights(q, k[:0], bias[:, :, :0], 1.0).numpy().shape == (5, 4, 0)
    with pytest.raises(ValueError, match="heads"):
        numerics.biased_attention_weights(q, k[:, :2], bias, 1.0)
    with pytest.raises(ValueError, match="bias"):
        numerics.biased_attention_weights(q, k, bias[:, :4], 1.0)
    with pytest.raises(ValueError, match="head_dim"):
        numerics.biased_attention_weights(q, k[:, :, :4], bias, 1.0)
    with pytest.raises(ValueError, match="length, heads"):
        numerics.biased_attention_weights(q[0], k, bias, 1.0)


# Tracing (#402)


def test_tracing_changes_no_bit_and_is_golden(t5, flan):
    from golden_values import T5_INTERPRET_FINGERPRINT

    for name, (_, engine) in (("t5", t5), ("flan", flan)):
        model = engine.model
        source, answer = sequence(engine)
        recorded = trace_text_to_text(model, source + answer)
        assert isinstance(recorded, TextToTextTrace)
        assert recorded.source == tuple(source) and recorded.tokens == (0, *answer)
        assert recorded.logits is not None
        for step in range(len(answer) + 1):
            assert recorded.logits[step].tobytes() == model.forward(source + answer[:step]).tobytes()
        assert recorded.encoder_hidden.tobytes() == model.encoder_states(source).tobytes()
        assert recorded.fingerprint() == T5_INTERPRET_FINGERPRINT[f"trace_{name}"]
        config = model.config
        n, s = len(answer) + 1, len(source)
        assert recorded.residual.shape == (config.decoder_layers + 1, n, config.hidden_size)
        assert recorded.encoder_residual.shape == (config.layers + 1, s, config.hidden_size)
        assert recorded.attention is not None and recorded.cross_attention is not None
        assert recorded.attention.shape == (config.decoder_layers, config.heads, n, n)
        assert recorded.cross_attention.shape == (config.decoder_layers, config.heads, n, s)
        assert np.all(np.triu(recorded.attention, 1) == 0)
        # the blocks add up: residual + attention + cross-attention + MLP is the next layer's input
        for layer in range(config.decoder_layers):
            middle = recorded.residual[layer] + recorded.attention_output[layer]
            assert middle.tobytes() == recorded.middle[layer].tobytes()
            cross = recorded.middle[layer] + recorded.cross_attention_output[layer]
            assert cross.tobytes() == recorded.cross_middle[layer].tobytes()
            after = recorded.cross_middle[layer] + recorded.mlp_output[layer]
            assert after.tobytes() == recorded.residual[layer + 1].tobytes()
        assert recorded.hidden.tobytes() == model.final_norm(recorded.residual[-1]).tobytes()
        lean = trace_text_to_text(model, source + answer, attention=False, logits=False)
        assert lean.attention is lean.cross_attention is lean.encoder_attention is lean.logits is None
        assert lean.residual.tobytes() == recorded.residual.tobytes()
        assert dict(lean.arrays()).keys() < dict(recorded.arrays()).keys()


def test_traces_match_transformers(t5, flan):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    for checkpoint, engine in (t5, flan):
        source, answer = sequence(engine, length=20)  # past the buckets' maximum distance
        recorded = trace_text_to_text(engine.model, source + answer)
        model = transformers.T5ForConditionalGeneration.from_pretrained(checkpoint, attn_implementation="eager")
        with torch.no_grad():
            output = model.eval()(
                input_ids=torch.tensor([source]),
                decoder_input_ids=torch.tensor([[0, *answer]]),
                output_attentions=True,
                output_hidden_states=True,
            )
        layers = engine.model.config.decoder_layers
        for layer in range(layers):
            np.testing.assert_allclose(recorded.residual[layer], output.decoder_hidden_states[layer][0], atol=2e-4)
            np.testing.assert_allclose(recorded.attention[layer], output.decoder_attentions[layer][0], atol=1e-5)
            np.testing.assert_allclose(recorded.cross_attention[layer], output.cross_attentions[layer][0], atol=1e-5)
        np.testing.assert_allclose(recorded.hidden, output.decoder_hidden_states[layers][0], atol=1e-4)
        for layer in range(engine.model.config.layers):
            np.testing.assert_allclose(
                recorded.encoder_residual[layer], output.encoder_hidden_states[layer][0], atol=2e-4
            )
            np.testing.assert_allclose(
                recorded.encoder_attention[layer], output.encoder_attentions[layer][0], atol=1e-5
            )
        np.testing.assert_allclose(recorded.encoder_hidden, output.encoder_last_hidden_state[0], atol=1e-4)


def test_traces_are_thread_invariant(flan):
    _, engine = flan
    source, answer = sequence(engine, TEXTS[2], TEXTS[3])
    fingerprints = set()
    try:
        for threads in (1, 3, 8):
            numerics.set_threads(threads)
            fingerprints.add(trace_text_to_text(engine.model, source + answer).fingerprint())
    finally:
        numerics.set_threads(0)
    assert len(fingerprints) == 1


def test_trace_refusals(flan, encoder_path):  # noqa: F811 - fixture
    _, engine = flan
    with pytest.raises(ValueError, match="ending with </s>"):
        trace_text_to_text(engine.model, [5, 6])
    with pytest.raises(ValueError, match="out of range"):
        trace_text_to_text(engine.model, [5, END, engine.model.vocabulary_size])
    with pytest.raises(ValueError, match="one sequence"):
        model = engine.model
        model._steps([0, 0], [model.new_cache(), model.new_cache()], recorder=object())  # type: ignore[arg-type]
    bert = DllmEngine.from_model_file(encoder_path).model
    with pytest.raises(ValueError, match="only T5 encoders"):
        bert.hidden_states([2, 5, 3], recorder=object())  # type: ignore[attr-defined]


# The logit lens (#403)


def test_logit_lens_over_decoder_layers(t5):
    _, engine = t5
    model = engine.model
    source, answer = sequence(engine)
    lens = logit_lens(model, source + answer, top=3)
    assert lens.tokens == (0, *answer) and lens.layers == model.config.decoder_layers
    for position in range(len(answer) + 1):
        probabilities = numerics.softmax(model.forward(source + answer[:position]))
        last = lens.predictions[-1][position]
        assert [p.token for p in last] == top_k(probabilities, 3)
        assert last[0].probability == float(probabilities[last[0].token])
    again = logit_lens(model, source + answer, top=3, recorded=trace_text_to_text(model, source + answer))
    assert again == lens


def test_lens_command(t5, capsys, cli_environment, tmp_path):
    from etalii_dllm.cli import main

    checkpoint, engine = t5
    path = str(checkpoint.parent / "model.dllm")
    arguments = ["--model", path, "lens", "--prompt", TEXTS[0], "--answer", TEXTS[1][:20], "--top-k", "2"]
    assert main([*arguments, "--json", "--html", str(tmp_path / "lens.html")]) == 0
    result = json.loads(capsys.readouterr().out)
    source, answer = sequence(engine, answer=TEXTS[1][:20], length=100)
    assert [token["id"] for token in result["tokens"]] == [0, *answer]
    assert len(result["layers"]) == engine.model.config.decoder_layers + 1
    lens = logit_lens(engine.model, source + answer, 2)
    assert result["layers"][-1][-1][0]["id"] == lens.predictions[-1][-1][0].token
    assert "Logit lens" in (tmp_path / "lens.html").read_text(encoding="utf-8")
    assert main([*arguments, "--position", "0"]) == 0
    assert "position 0 ('<pad>')" in capsys.readouterr().out
    assert main(["--model", path, "lens", "--prompt", TEXTS[0], "--chat"]) == 0
    assert "position 0" in capsys.readouterr().out
    assert main([*arguments, "--answer", "a </s> b"]) == 1
    assert "must not contain </s>" in capsys.readouterr().err


# Attention maps (#404)


def test_attention_command(flan, capsys, cli_environment, tmp_path):
    from etalii_dllm.cli import main

    checkpoint, engine = flan
    path = str(checkpoint.parent / "model.dllm")
    source, answer = sequence(engine, TEXTS[0], TEXTS[1][:20], 100)
    recorded = trace_text_to_text(engine.model, source + answer)
    arguments = ["--model", path, "attention", "--prompt", TEXTS[0], "--answer", TEXTS[1][:20]]
    assert main([*arguments, "--json", "--layer", "2", "--head", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [token["id"] for token in result["tokens"]] == [0, *answer]
    assert np.asarray(result["attention"]["2"]["1"], dtype=np.float32).tobytes() == recorded.attention[1, 1].tobytes()
    assert main([*arguments, "--json", "--cross", "--html", str(tmp_path / "cross.html")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [token["id"] for token in result["source"]] == source
    maps = np.asarray(result["cross_attention"]["1"]["3"], dtype=np.float32)
    assert maps.tobytes() == recorded.cross_attention[0, 3].tobytes()
    page = (tmp_path / "cross.html").read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "Cross-attention" in page
    assert main([*arguments, "--cross", "--layer", "1", "--head", "0", "--top-k", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("layer 1 head 0:") and out.count("->") == len(answer) + 1
    assert main([*arguments, "--html", str(tmp_path / "self.html"), "--layer", "1", "--head", "0"]) == 0
    assert "Attention: " in (tmp_path / "self.html").read_text(encoding="utf-8")
    capsys.readouterr()
    assert main([*arguments, "--layer", "3"]) == 1
    assert "--layer must be between 1 and 2" in capsys.readouterr().err
    assert main(["--model", path, "experts", "--prompt", "x"]) == 1
    assert "needs a decoder-only model" in capsys.readouterr().err


def test_decoder_only_models_refuse_the_text_to_text_options(model_path, capsys, cli_environment):  # noqa: F811
    from etalii_dllm.cli import main

    path = str(model_path)
    assert main(["--model", path, "lens", "--prompt", "hello", "--answer", "x"]) == 1
    assert "--answer is for text-to-text models" in capsys.readouterr().err
    assert main(["--model", path, "attention", "--prompt", "hello", "--cross"]) == 1
    assert "--cross is for text-to-text models" in capsys.readouterr().err


# Steering (#405)


def test_steering_vectors_on_the_decoder(flan, tmp_path):
    from golden_values import T5_INTERPRET_FINGERPRINT

    checkpoint, engine = flan
    model = engine.model
    vector = build_steering_vector(model, engine.tokenizer, [TEXTS[1], TEXTS[3]], [TEXTS[2]], 1, 3.0)
    assert vector.layer == 1 and vector.vector.shape == (model.config.hidden_size,)
    assert vector.model_fingerprint == model.weights_fingerprint
    positive = mean_activation(model, engine.tokenizer, [TEXTS[1], TEXTS[3]], 1)
    tokens = engine.tokenizer.encode(TEXTS[1])
    recorded = trace_text_to_text(model, [END, *tokens], attention=False, logits=False)
    first = numerics.column_mean(recorded.residual[1][1:])
    assert first.shape == positive.shape
    path = tmp_path / "steer.json"
    vector.save(path)
    assert SteeringVector.load(path).vector.tobytes() == vector.vector.tobytes()
    file = checkpoint.parent / "model.dllm"
    steered = DllmEngine.from_model_file(file, steer=path)
    assert isinstance(steered.model, TextToText)
    assert steered.system_fingerprint != engine.system_fingerprint
    assert steered.model.steering[0].tobytes() == vector.scaled().tobytes()
    source, answer = sequence(engine)
    assert steered.model.forward(source + answer).tobytes() != model.forward(source + answer).tobytes()
    # the vector is added after its layer, so the steered trace's next residual is the sum, bit for bit
    plain = trace_text_to_text(model, source + answer, attention=False, logits=False)
    shifted = trace_text_to_text(steered.model, source + answer, attention=False, logits=False)
    assert shifted.residual[1].tobytes() == (plain.residual[1] + vector.scaled()).tobytes()
    assert shifted.residual[0].tobytes() == plain.residual[0].tobytes()
    result = steered.complete(TEXTS[0], 8, SamplingOptions(temperature=0.8, seed=7))
    assert result.fingerprint == steered.complete(TEXTS[0], 8, SamplingOptions(temperature=0.8, seed=7)).fingerprint
    assert result.fingerprint == T5_INTERPRET_FINGERPRINT["steered"]
    again = DllmEngine.from_model_file(file, steer=path, steer_strength=3.0)
    assert again.system_fingerprint == steered.system_fingerprint
    weaker = DllmEngine.from_model_file(file, steer=path, steer_strength=0.5)
    assert weaker.system_fingerprint != steered.system_fingerprint
    assert vector.for_model(model)[0].tobytes() == vector.scaled().tobytes()


def test_steering_refusals(flan, tmp_path):
    checkpoint, engine = flan
    model = engine.model
    with pytest.raises(ValueError, match="layer must be between 1 and 2"):
        build_steering_vector(model, engine.tokenizer, ["a"], ["b"], 3)
    with pytest.raises(ValueError, match="contains </s>"):
        mean_activation(model, engine.tokenizer, ["a </s> b"], 1)
    hidden = model.config.hidden_size
    with pytest.raises(ValueError, match=r"decoder layer in 0\.\.1"):
        TextToText(model.config, ModelFile(checkpoint.parent / "model.dllm").tensors, steering={2: np.zeros(hidden)})
    deep = SteeringVector(3, np.zeros(hidden, dtype=np.float32))
    with pytest.raises(ValueError, match="between 1 and 2"):
        deep.for_model(model)
    deep.save(tmp_path / "deep.json")
    with pytest.raises(ValueError, match="does not fit"):
        DllmEngine.from_model_file(checkpoint.parent / "model.dllm", steer=tmp_path / "deep.json")
    steered = TextToText(
        model.config, ModelFile(checkpoint.parent / "model.dllm").tensors, steering={0: np.zeros(hidden)}
    )
    with pytest.raises(ValueError, match="unsteered"):
        build_steering_vector(steered, engine.tokenizer, ["a"], ["b"], 1)


def test_steer_command(flan, capsys, cli_environment, tmp_path):
    from etalii_dllm.cli import main

    checkpoint, engine = flan
    path = str(checkpoint.parent / "model.dllm")
    output = tmp_path / "vector.json"
    arguments = ["--model", path, "steer", "--positive", TEXTS[1], "--negative", TEXTS[2], "-o", str(output)]
    assert main(arguments) == 0
    assert "layer:    1 of 2" in capsys.readouterr().out
    expected = build_steering_vector(engine.model, engine.tokenizer, [TEXTS[1]], [TEXTS[2]], 1)
    assert SteeringVector.load(output).vector.tobytes() == expected.vector.tobytes()
    assert main(["--model", path, "--steer", str(output), "generate", "--prompt", TEXTS[0], "--max-tokens", "4"]) == 0
    assert capsys.readouterr().out
    assert main([*arguments, "--layer", "5"]) == 1
    assert "layer must be between 1 and 2" in capsys.readouterr().err
