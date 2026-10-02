"""Phase 29: exact guided decoding. Classifier-free guidance, contrastive decoding and ensembles combine the logits
with the documented float32 operations, match the independent reference implementation, replay from receipts and
reach every front end."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from golden_values import GUIDANCE_FINGERPRINTS
from model_fixtures import TINY_LLAMA_CONFIG, write_hf_checkpoint
from test_chat_template import SMOLLM2
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm import engine as engine_module
from etalii_dllm import guidance, receipts, reference
from etalii_dllm.chat import ChatMessage
from etalii_dllm.cli import main
from etalii_dllm.engine import ChatRequest, DllmEngine, default_engine
from etalii_dllm.importing import import_model
from etalii_dllm.numerics import fill_gaussian, log_softmax, softmax
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.server.app import app

PROMPT = "Once upon a time"
STORY = [ChatMessage("user", "Tell me a story.")]


@pytest.fixture(scope="module")
def amateur_path(tmp_path_factory):
    """A smaller model (one layer, other weights) with the tiny model's tokenizer."""
    from test_bpe import smollm2_style

    tokenizer = smollm2_style()
    directory = tmp_path_factory.mktemp("amateur")
    config = {**TINY_LLAMA_CONFIG, "vocab_size": tokenizer.get_vocab_size(), "eos_token_id": 2,
              "num_hidden_layers": 1}  # fmt: skip
    write_hf_checkpoint(directory / "checkpoint", config, tokenizer_json=json.loads(tokenizer.to_str()))
    tokenizer_config = {"chat_template": SMOLLM2, "eos_token": "<|im_end|>", "bos_token": None}
    (directory / "checkpoint" / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    import_model(directory / "checkpoint", directory / "amateur.dllm", repository="example/amateur")
    return directory / "amateur.dllm"


@pytest.fixture(scope="module")
def tiny(model_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(model_path)


@pytest.fixture(scope="module")
def contrasting(model_path, amateur_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(model_path, contrast_model=amateur_path)


@pytest.fixture(scope="module")
def ensembled(model_path, amateur_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(model_path, ensemble=[(amateur_path, 0.5)], ensemble_weight=2.0)


@pytest.fixture
def served(model_path, amateur_path, monkeypatch):  # noqa: F811
    for name in dir(engine_module):  # earlier tests leave runtime settings (an auditor, a response cache) behind
        if name.endswith("_ENVIRONMENT_VARIABLE"):
            monkeypatch.delenv(getattr(engine_module, name), raising=False)
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(model_path))
    monkeypatch.setenv(engine_module.CONTRAST_MODEL_ENVIRONMENT_VARIABLE, str(amateur_path))
    default_engine.cache_clear()
    yield default_engine()
    default_engine.cache_clear()


# The combination rules


def test_combinations_are_the_documented_float32_operations():
    logits, other = fill_gaussian(1, 300) * np.float32(4.0), fill_gaussian(2, 300) * np.float32(4.0)
    expected = (other.astype(np.float64) + np.float64(np.float32(1.75)) * (logits.astype(np.float64) - other)).astype(
        np.float32
    )  # each operation rounds: compare with the explicit spelling, and with the reference
    explicit = np.float32(other + np.float32(np.float32(1.75) * np.float32(logits - other)))
    assert np.array_equal(guidance.guided(logits, other, 1.75), explicit)
    assert guidance.guided(logits, other, 1.75).dtype == np.float32 and expected.shape == explicit.shape
    assert np.array_equal(guidance.guided(logits, other, 1.0), other + (logits - other))
    contrasted = guidance.contrasted(logits, other, 0.1, 0.5)
    p = softmax(logits)
    kept = p.astype(np.float64) >= 0.1 * float(p.max())
    assert np.all(np.isneginf(contrasted[~kept])) and kept.sum() >= 1
    assert np.array_equal(contrasted[kept], (np.float32(1.5) * logits - np.float32(0.5) * other)[kept])
    mixed = guidance.ensembled([(logits, 2.0), (other, 1.0)])
    spelled = (2 / 3) * log_softmax(logits).astype(np.float64) + (1 / 3) * log_softmax(other).astype(np.float64)
    assert np.array_equal(mixed, spelled.astype(np.float32))
    for ours, theirs in [
        (guidance.guided(logits, other, 1.75), reference.guided(logits, other, 1.75)),
        (contrasted, reference.contrasted(logits, other, 0.1, 0.5)),
        (mixed, reference.ensembled([(logits, 2.0), (other, 1.0)])),
    ]:
        assert ours.tobytes() == theirs.tobytes()


def test_options_validate_and_record_only_when_set():
    assert "negative_prompt" not in SamplingOptions(temperature=0.5).record()
    options = SamplingOptions(negative_prompt="no", guidance_scale=2.0)
    assert SamplingOptions.from_record(json.loads(json.dumps(options.record()))) == options
    contrast = SamplingOptions(contrast_beta=0.5, contrast_alpha=0.2)
    assert SamplingOptions.from_record(contrast.record()) == contrast
    for bad, message in [
        ({"guidance_scale": float("inf")}, "guidance_scale"),
        ({"contrast_beta": -1.0}, "contrast_beta"),
        ({"contrast_alpha": 2.0}, "contrast_alpha"),
        ({"negative_prompt": "x", "contrast_beta": 0.5}, "not both"),
    ]:
        with pytest.raises(ValueError, match=message):
            SamplingOptions(**bad)


# Generation


def test_negative_prompts_guide_raw_and_chat_answers(tiny):
    options = SamplingOptions(temperature=0.7, seed=3, negative_prompt="It was a dark night", guidance_scale=3.0)
    plain = tiny.complete(PROMPT, 12, SamplingOptions(temperature=0.7, seed=3))
    guided = tiny.complete(PROMPT, 12, options)
    assert guided.tokens != plain.tokens and guided.tokens == tiny.complete(PROMPT, 12, options).tokens
    assert [s.text for s in tiny.complete_stream(PROMPT, 12, options)] and guided.fingerprint == (
        tiny.complete_stream(PROMPT, 12, options).result().fingerprint
    )
    # Scale 1 decodes from the request's own logits (n + (l - n) rounds back to l for these values or close to it).
    chat = tiny.chat_completion(ChatRequest(STORY, 10, options, top_logprobs=0))
    assert (
        chat.content != tiny.chat_completion(ChatRequest(STORY, 10, SamplingOptions(temperature=0.7, seed=3))).content
    )
    assert chat.fingerprint == GUIDANCE_FINGERPRINTS["guided"]
    receipt = chat.receipt
    assert receipt["request"]["options"]["negative_prompt"] == "It was a dark night"
    assert receipts.verify(tiny, receipt).ok
    with pytest.raises(ValueError, match="a user message"):
        tiny.chat_completion(ChatRequest([ChatMessage("system", "Hi")], 4, options))
    with pytest.raises(ValueError, match="cannot roll"):
        tiny.chat_completion(ChatRequest(STORY, 4, options, context_overflow="roll"))
    assert tiny.complete(PROMPT, 4, SamplingOptions(negative_prompt="")).tokens  # an empty negative prompt works


def test_guided_answers_match_the_reference(tiny):
    options = SamplingOptions(temperature=0.8, seed=5, negative_prompt="It was a dark night", guidance_scale=2.5)
    answer = tiny.complete(PROMPT, 8, options)
    twin = reference.ReferenceTransformer.from_engine_model(tiny.model)
    negative = reference.ReferenceTransformer.from_engine_model(tiny.model)
    negative_tokens = tiny.tokenizer.encode("It was a dark night")
    fed = [0]

    def guide(logits, generated):
        unfed = [*negative_tokens, *generated] if fed[0] == 0 and negative.length == 0 else generated[fed[0] :]
        fed[0] = len(generated)
        return reference.guided(logits, negative.forward(unfed), 2.5)

    tokens, _ = twin.generate(tiny.tokenizer.encode(PROMPT), 8, reference.sampler(options), sorted(tiny.stop_tokens),
                              guide=guide)  # fmt: skip
    assert list(answer.tokens) == tokens


def test_contrastive_decoding(tiny, contrasting, amateur_path):
    assert contrasting.system_fingerprint != tiny.system_fingerprint
    options = SamplingOptions(temperature=0.6, seed=2, contrast_beta=0.8)
    plain = contrasting.complete(PROMPT, 12, SamplingOptions(temperature=0.6, seed=2))
    assert plain.tokens == tiny.complete(PROMPT, 12, SamplingOptions(temperature=0.6, seed=2)).tokens
    contrasted = contrasting.complete(PROMPT, 12, options)
    assert contrasted.tokens != plain.tokens
    amateur = DllmEngine.from_model_file(amateur_path).model
    twin = reference.ReferenceTransformer.from_engine_model(tiny.model)
    small = reference.ReferenceTransformer.from_engine_model(amateur)
    context = tiny.tokenizer.encode(PROMPT)
    fed = [0]

    def guide(logits, generated):
        unfed = [*context, *generated] if small.length == 0 else generated[fed[0] :]
        fed[0] = len(generated)
        return reference.contrasted(logits, small.forward(unfed), 0.1, 0.8)

    tokens, _ = twin.generate(context, 12, reference.sampler(options), sorted(tiny.stop_tokens), guide=guide)
    assert list(contrasted.tokens) == tokens
    chat = contrasting.chat_completion(ChatRequest(STORY, 10, options))
    assert chat.fingerprint == GUIDANCE_FINGERPRINTS["contrasted"]
    assert receipts.verify(contrasting, chat.receipt).ok
    with pytest.raises(ValueError, match="needs a contrast model"):
        tiny.complete(PROMPT, 4, options)


def test_ensembles(tiny, ensembled, amateur_path):
    assert ensembled.system_fingerprint not in (tiny.system_fingerprint,)
    answer = ensembled.complete(PROMPT, 12, SamplingOptions(temperature=0.6, seed=2))
    amateur = DllmEngine.from_model_file(amateur_path).model
    twin = reference.ReferenceTransformer.from_engine_model(tiny.model)
    member = reference.ReferenceTransformer.from_engine_model(amateur)
    context = tiny.tokenizer.encode(PROMPT)
    fed = [0]

    def guide(logits, generated):
        unfed = [*context, *generated] if member.length == 0 else generated[fed[0] :]
        fed[0] = len(generated)
        return reference.ensembled([(logits, 2.0), (member.forward(unfed), 0.5)])

    options = SamplingOptions(temperature=0.6, seed=2)
    tokens, _ = twin.generate(context, 12, reference.sampler(options), sorted(tiny.stop_tokens), guide=guide)
    assert list(answer.tokens) == tokens
    chat = ensembled.chat_completion(ChatRequest(STORY, 10, options))
    assert chat.fingerprint == GUIDANCE_FINGERPRINTS["ensembled"]
    assert receipts.verify(ensembled, chat.receipt).ok
    with pytest.raises(ValueError, match="ensemble"):
        ensembled.complete(PROMPT, 4, SamplingOptions(negative_prompt="no"))


def test_engines_check_their_companions(model_path, amateur_path, tmp_path):  # noqa: F811
    tiny = DllmEngine.from_model_file(model_path)
    other = DllmEngine.create_default()
    with pytest.raises(ValueError, match="vocabulary"):
        DllmEngine(tiny.model, tiny.tokenizer, "fp_x", contrast_model=other.model)
    with pytest.raises(ValueError, match="positive"):
        DllmEngine(tiny.model, tiny.tokenizer, "fp_x", ensemble=[(tiny.model, 0.0)])
    original = engine_module._vocabulary
    try:
        engine_module._vocabulary = lambda tokenizer: id(tokenizer)  # every header differs
        with pytest.raises(ValueError, match="contrast model's tokenizer differs"):
            DllmEngine.from_model_file(model_path, contrast_model=amateur_path)
    finally:
        engine_module._vocabulary = original


def test_a_full_guide_context_ends_the_answer(tiny):
    window = tiny.model.config.context_length
    long_negative = "x " * (window - 4)
    result = tiny.complete("Hi", 40, SamplingOptions(negative_prompt=long_negative))
    assert result.finish_reason == "length" and len(result.tokens) < 40


# Front ends


def test_front_ends(served, capsys, monkeypatch):
    client = TestClient(app)
    body = {"messages": [{"role": "user", "content": "Tell me a story."}], "max_tokens": 10, "temperature": 0.7,
            "seed": 3}  # fmt: skip
    plain = client.post("/v1/chat/completions", json=body).json()
    guided_body = {**body, "guidance": {"negative_prompt": "It was a dark night", "scale": 3.0}}
    guided = client.post("/v1/chat/completions", json=guided_body).json()
    expected = served.chat_completion(
        ChatRequest(STORY, 10, SamplingOptions(temperature=0.7, seed=3, negative_prompt="It was a dark night",
                                               guidance_scale=3.0))
    )  # fmt: skip
    assert guided["id"] != plain["id"] and guided["choices"][0]["message"]["content"] == expected.content
    contrast_body = {**body, "contrast": {"beta": 0.8}}
    contrasted = client.post("/v1/chat/completions", json=contrast_body).json()
    expected = served.chat_completion(
        ChatRequest(STORY, 10, SamplingOptions(temperature=0.7, seed=3, contrast_beta=0.8))
    )
    assert contrasted["choices"][0]["message"]["content"] == expected.content
    anthropic = client.post("/v1/messages", json={**contrast_body, "max_tokens": 10}).json()
    assert anthropic["content"][0]["text"] == expected.content
    both = client.post("/v1/chat/completions", json={**guided_body, "contrast": {"beta": 0.5}})
    assert both.status_code == 400 and "not both" in both.json()["error"]["message"]

    completion = client.post(
        "/v1/completions",
        json={
            "prompt": PROMPT,
            "max_tokens": 8,
            "temperature": 0.7,
            "seed": 3,
            "contrast": {"beta": 0.8, "alpha": 0.2},
        },
    ).json()
    options = SamplingOptions(temperature=0.7, seed=3, contrast_beta=0.8, contrast_alpha=0.2)
    assert completion["choices"][0]["text"] == served.complete(PROMPT, 8, options).text  # fmt: skip

    args = ["generate", "--prompt", PROMPT, "--max-tokens", "8", "--temperature", "0.7", "--seed", "3"]
    assert main([*args, "--negative-prompt", "It was a dark night", "--guidance-scale", "3"]) == 0
    options = SamplingOptions(temperature=0.7, seed=3, negative_prompt="It was a dark night", guidance_scale=3.0)
    assert capsys.readouterr().out == served.complete(PROMPT, 8, options).text + "\n"
    assert main([*args, "--contrast", "0.8"]) == 0
    contrasting = SamplingOptions(temperature=0.7, seed=3, contrast_beta=0.8)
    assert capsys.readouterr().out == served.complete(PROMPT, 8, contrasting).text + "\n"


def test_runtime_settings(monkeypatch, model_path, amateur_path):  # noqa: F811
    for name in dir(engine_module):
        if name.endswith("_ENVIRONMENT_VARIABLE"):
            monkeypatch.setenv(getattr(engine_module, name), "")  # so that monkeypatch restores them all at the end
            monkeypatch.delenv(getattr(engine_module, name))
    engine_module.use_model_file(model_path, ensemble_models=[f"{amateur_path}=0.5"], ensemble_weight=2.0,
                                 contrast_model=amateur_path)  # fmt: skip
    try:
        assert engine_module.configured_ensemble() == [(str(amateur_path), 0.5)]
        assert engine_module.configured_ensemble_weight() == 2.0
        engine = default_engine()
        assert engine.ensemble and engine.contrast_model is not None
        monkeypatch.setenv(engine_module.ENSEMBLE_MODELS_ENVIRONMENT_VARIABLE, "a=b.dllm")
        assert engine_module.configured_ensemble() == [("a=b.dllm", 1.0)]
        monkeypatch.setenv(engine_module.ENSEMBLE_MODELS_ENVIRONMENT_VARIABLE, "plain.dllm")
        assert engine_module.configured_ensemble() == [("plain.dllm", 1.0)]
        monkeypatch.setenv(engine_module.ENSEMBLE_WEIGHT_ENVIRONMENT_VARIABLE, "heavy")
        with pytest.raises(ValueError, match="DLLM_ENSEMBLE_WEIGHT"):
            engine_module.configured_ensemble_weight()
    finally:
        default_engine.cache_clear()
