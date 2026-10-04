"""T5 text-to-text generation (#382-#385): the decoder's one-directional buckets, logits against transformers'
``T5ForConditionalGeneration`` (T5 v1.0 with a tied head, v1.1/Flan-T5 with a gated MLP and its own head), the
import, generation in the front ends, goldens and the reference implementation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from model_fixtures import write_safetensors
from test_t5 import TEXTS, t5_tokenizer

from etalii_dllm import numerics
from etalii_dllm.encoder import t5_relative_buckets
from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.sampling import SamplingOptions

pytest.importorskip("tokenizers")
pytest.importorskip("sentencepiece")

CONFIG = {
    "model_type": "t5",
    "architectures": ["T5ForConditionalGeneration"],
    "d_model": 32,
    "d_kv": 8,
    "d_ff": 64,
    "num_layers": 2,
    "num_decoder_layers": 3,
    "num_heads": 4,
    "relative_attention_num_buckets": 8,
    "relative_attention_max_distance": 16,
    "layer_norm_epsilon": 1e-6,
    "feed_forward_proj": "relu",
    "is_encoder_decoder": True,
    "tie_word_embeddings": True,
    "dropout_rate": 0.0,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "decoder_start_token_id": 0,
}
FLAN = {"feed_forward_proj": "gated-gelu", "tie_word_embeddings": False, "num_decoder_layers": 2}


def full_weights(config: dict, seed: int = 23) -> dict[str, np.ndarray]:
    """Deterministic float32 weights under transformers' ``T5ForConditionalGeneration`` names."""
    hidden, inner, size = config["d_model"], config["d_kv"] * config["num_heads"], config["d_ff"]
    counter = iter(range(seed * 1000, seed * 1000 + 5000))

    def matrix(rows: int, columns: int, scale: float = 0.2) -> np.ndarray:
        values = numerics.fill_gaussian(next(counter), rows * columns).reshape(rows, columns)
        return (values * np.float32(scale)).astype(np.float32)

    def vector(count: int) -> np.ndarray:
        return (np.float32(1.0) + numerics.fill_gaussian(next(counter), count) * np.float32(0.1)).astype(np.float32)

    buckets, heads = config["relative_attention_num_buckets"], config["num_heads"]
    gated = config["feed_forward_proj"].startswith("gated-")
    mlp = ("wi_0", "wi_1") if gated else ("wi",)
    weights = {"shared.weight": matrix(config["vocab_size"], hidden, 1.0)}
    for stack, layers in (("encoder", config["num_layers"]), ("decoder", config["num_decoder_layers"])):
        weights[f"{stack}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = matrix(buckets, heads, 1.0)
        blocks = ("SelfAttention",) if stack == "encoder" else ("SelfAttention", "EncDecAttention")
        for i in range(layers):
            p = f"{stack}.block.{i}.layer."
            for j, block in enumerate(blocks):
                for name in "qkv":
                    weights[p + f"{j}.{block}.{name}.weight"] = matrix(inner, hidden)
                weights[p + f"{j}.{block}.o.weight"] = matrix(hidden, inner)
                weights[p + f"{j}.layer_norm.weight"] = vector(hidden)
            j = len(blocks)
            for name in mlp:
                weights[p + f"{j}.DenseReluDense.{name}.weight"] = matrix(size, hidden)
            weights[p + f"{j}.DenseReluDense.wo.weight"] = matrix(hidden, size)
            weights[p + f"{j}.layer_norm.weight"] = vector(hidden)
        weights[f"{stack}.final_layer_norm.weight"] = vector(hidden)
    if not config.get("tie_word_embeddings", True):
        weights["lm_head.weight"] = matrix(config["vocab_size"], hidden, 0.5)
    return weights


def write_checkpoint(directory: Path, **changes) -> dict:
    reference = t5_tokenizer()
    directory.mkdir(parents=True, exist_ok=True)
    config = {**CONFIG, "vocab_size": reference.get_vocab_size() + 4, **changes}
    weights = full_weights(config)
    write_safetensors(directory / "model.safetensors", {n: ("F32", v) for n, v in weights.items()}, {"format": "pt"})
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "tokenizer.json").write_text(reference.to_str(), encoding="utf-8")
    tokenizer_config = {"eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>", "model_max_length": 512}
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    (directory / "README.md").write_bytes(b"---\nlicense: apache-2.0\n---\n\n# Tiny T5\n")
    return config


@pytest.fixture(scope="module")
def t5(tmp_path_factory) -> tuple[Path, DllmEngine]:
    """T5 v1.0: ReLU, the LM head tied to the shared embedding (scaled by d_model^-0.5), 3 decoder layers."""
    directory = tmp_path_factory.mktemp("t5-text")
    write_checkpoint(directory / "checkpoint")
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-t5-text")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


@pytest.fixture(scope="module")
def flan(tmp_path_factory) -> tuple[Path, DllmEngine]:
    """Flan-T5 / T5 v1.1: gated tanh GELU, its own LM head."""
    directory = tmp_path_factory.mktemp("flan")
    write_checkpoint(directory / "checkpoint", **FLAN)
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-flan-t5")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


def transformers_logits(directory: Path, source: list[int], answer: list[int]) -> np.ndarray:
    """transformers' logits ``[len(answer) + 1, vocabulary]`` for the decoder inputs ``[0, *answer]``."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.T5ForConditionalGeneration.from_pretrained(directory).eval()
    with torch.no_grad():
        output = model(input_ids=torch.tensor([source]), decoder_input_ids=torch.tensor([[0, *answer]]))
    return output.logits[0].double().numpy()


# The decoder (#382)


@pytest.mark.parametrize(("buckets", "distance"), [(32, 128), (8, 16), (16, 64), (30, 100)])
def test_decoder_buckets_match_transformers(buckets, distance):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.t5.modeling_t5 import T5Attention

    distances = np.arange(-6000, 6001)
    expected = T5Attention._relative_position_bucket(
        torch.tensor(distances), bidirectional=False, num_buckets=buckets, max_distance=distance
    ).numpy()
    ours = t5_relative_buckets(distances, buckets, distance, bidirectional=False)
    assert (ours == expected).all()
    assert (ours[distances >= 0] == 0).all()  # keys at or after the query share bucket 0


def test_logits_match_transformers(t5, flan):
    for checkpoint, engine in (t5, flan):
        model = engine.model
        source = [*engine.tokenizer.encode(TEXTS[0]), 1]
        answer = engine.tokenizer.encode(TEXTS[1])[:20]  # past the buckets' maximum distance
        expected = transformers_logits(checkpoint, source, answer)
        for step in range(len(answer) + 1):
            ours = model.forward([*source, *answer[:step]])
            np.testing.assert_allclose(ours, expected[step], atol=2e-4, rtol=1e-4)


def test_cache_gives_the_bits_of_a_recompute(flan):
    _, engine = flan
    model = engine.model
    source = [*engine.tokenizer.encode(TEXTS[3]), 1]
    answer = engine.tokenizer.encode(TEXTS[2])
    cache = model.new_cache()
    for step in range(len(answer) + 1):
        tokens = [*source, *answer[:step]]
        assert model.forward_cached(tokens, cache).tobytes() == model.forward(tokens).tobytes()
    # another source resets the cache; a shorter answer recomputes it
    other = [*engine.tokenizer.encode(TEXTS[1]), 1]
    assert model.forward_cached(other, cache).tobytes() == model.forward(other).tobytes()
    assert model.forward_cached([*other, 5, 6], cache).tobytes() == model.forward([*other, 5, 6]).tobytes()
    assert model.forward_cached([*other, 5], cache).tobytes() == model.forward([*other, 5]).tobytes()
    with pytest.raises(ValueError, match="ending with </s>"):
        model.forward([5, 6])
    with pytest.raises(ValueError, match="out of range"):
        model.forward([1, model.vocabulary_size])


def test_thread_invariance(flan):
    _, engine = flan
    tokens = [*engine.tokenizer.encode(TEXTS[0]), 1, 7, 8, 9]
    results = set()
    try:
        for threads in (1, 3, 8):
            numerics.set_threads(threads)
            results.add(engine.model.forward(tokens).tobytes())
    finally:
        numerics.set_threads(0)
    assert len(results) == 1


# Import (#383)


def test_import_records_the_text_to_text_model(t5, flan):
    for checkpoint, _ in (t5, flan):
        file = ModelFile(checkpoint.parent / "model.dllm")
        config = file.config
        assert config.family == "t5" and config.is_text_to_text and not config.is_encoder
        assert file.embedding is None
    assert ModelFile(t5[0].parent / "model.dllm").config.decoder_layers == 3
    flan_file = ModelFile(flan[0].parent / "model.dllm")
    assert not flan_file.config.tie_word_embeddings and "lm_head.weight" in flan_file.tensors
    assert flan_file.config.gated_mlp and flan_file.config.activation == "gelu_tanh"


def test_import_refusals(tmp_path):
    write_checkpoint(tmp_path / "c", decoder_start_token_id=5)
    with pytest.raises(ModelImportError, match="decoder starts"):
        import_model(tmp_path / "c", tmp_path / "c.dllm")
    write_checkpoint(tmp_path / "d")
    (tmp_path / "d" / "tokenizer.json").unlink()
    with pytest.raises(ModelImportError, match=r"tokenizer\.json"):
        import_model(tmp_path / "d", tmp_path / "d.dllm")
    from etalii_dllm.architecture import TransformerConfig

    with pytest.raises(ValueError, match="decoder_layers"):
        TransformerConfig(
            family="bert", vocabulary_size=8, hidden_size=8, intermediate_size=8, layers=1, heads=2, kv_heads=2,
            head_dim=4, context_length=8, rms_norm_eps=1e-6, rope_theta=0.0, activation="gelu",
            type_vocabulary_size=2, decoder_layers=1,
        )  # fmt: skip


def test_quantised_text_to_text(t5, tmp_path):
    checkpoint, engine = t5
    quantized = DllmEngine.from_model_file(checkpoint.parent / "model.dllm", quantize="q8_0")
    assert quantized.system_fingerprint != engine.system_fingerprint
    tokens = [*engine.tokenizer.encode(TEXTS[0]), 1, 4]
    a, b = quantized.model.forward(tokens), engine.model.forward(tokens)
    assert a.tobytes() != b.tobytes()
    assert np.abs(a - b).max() < 0.5


# Generation (#384)


def test_generation_is_golden_and_matches_transformers(flan):
    from golden_values import T5_GENERATION_FINGERPRINT

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    checkpoint, engine = flan
    result = engine.complete(TEXTS[0], 12, SamplingOptions(temperature=0.0))
    tokens = list(result.tokens)
    model = transformers.T5ForConditionalGeneration.from_pretrained(checkpoint).eval()
    source = [*engine.tokenizer.encode(TEXTS[0]), 1]
    with torch.no_grad():
        greedy = model.generate(torch.tensor([source]), max_new_tokens=12, do_sample=False, num_beams=1)[0].tolist()
    assert greedy[0] == 0 and greedy[1 : 1 + len(tokens)] == tokens
    sampled = engine.complete(TEXTS[0], 16, SamplingOptions(temperature=0.8, seed=7))
    again = engine.complete(TEXTS[0], 16, SamplingOptions(temperature=0.8, seed=7))
    assert sampled.text == again.text and sampled.fingerprint == again.fingerprint
    assert (result.fingerprint, sampled.fingerprint) == T5_GENERATION_FINGERPRINT


def test_generation_in_the_front_ends(flan, monkeypatch, capsys):
    from fastapi.testclient import TestClient

    from etalii_dllm import engine as engine_module
    from etalii_dllm.cli import main as cli

    for name in [n for n in __import__("os").environ if n.startswith("DLLM_")]:
        monkeypatch.delenv(name)
    checkpoint, engine = flan
    path = checkpoint.parent / "model.dllm"
    expected = engine.complete(TEXTS[3], 10, SamplingOptions(temperature=0.0)).text
    assert cli(["--model", str(path), "generate", "--prompt", TEXTS[3], "--max-tokens", "10"]) == 0
    assert expected in capsys.readouterr().out
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(path))
    engine_module.default_engine.cache_clear()
    from etalii_dllm.server.app import app

    try:
        client = TestClient(app)
        body = {"prompt": TEXTS[3], "max_tokens": 10, "temperature": 0}
        assert client.post("/v1/completions", json=body).json()["choices"][0]["text"] == expected
        chat = {"messages": [{"role": "user", "content": TEXTS[3]}], "max_tokens": 10, "temperature": 0}
        assert client.post("/v1/chat/completions", json=chat).json()["choices"][0]["message"]["content"] == expected
    finally:
        engine_module.default_engine.cache_clear()


def test_text_to_text_refusals(flan, tmp_path):
    checkpoint, engine = flan
    path = checkpoint.parent / "model.dllm"
    with pytest.raises(ValueError, match="text-to-text"):
        DllmEngine.from_model_file(path, speculate=4)
    with pytest.raises(ValueError, match="cannot roll"):
        engine.complete_stream(TEXTS[0], 4, SamplingOptions(), overflow="roll")
    with pytest.raises(ValueError, match="hidden states"):
        engine.embed(TEXTS[0])


def test_training_lora_and_export_refusals(flan, tmp_path, capsys):
    from etalii_dllm.cli import main as cli
    from etalii_dllm.exporting import export_gguf, export_safetensors
    from etalii_dllm.lora import AdapterError, LoraConfig, target_weights
    from etalii_dllm.transformer import Transformer

    checkpoint, _ = flan
    path = checkpoint.parent / "model.dllm"
    model = ModelFile(path)
    with pytest.raises(ValueError, match="text-to-text"):
        export_safetensors(model, tmp_path / "out")
    with pytest.raises(ValueError, match="text-to-text"):
        export_gguf(model, tmp_path / "out.gguf")
    with pytest.raises(AdapterError, match="text-to-text"):
        target_weights(model.config, LoraConfig(rank=2, alpha=4.0))
    with pytest.raises(ValueError, match="text-to-text"):
        Transformer(model.config, model.tensors)
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"text": TEXTS[0]}) + "\n", encoding="utf-8")
    assert cli(["finetune", str(path), "--data", str(data), "--output", str(tmp_path / "x.dllm")]) == 1
    assert "text-to-text" in capsys.readouterr().err


# The reference implementation and verify (#385)


def test_reference_implementation_and_verify(t5, flan):
    from etalii_dllm import reference, verify

    for _, engine in (t5, flan):
        twin = reference.ReferenceTextToText.from_engine_model(engine.model)
        source = [*engine.tokenizer.encode(TEXTS[2]), 1]
        for answer in ([], [5], [5, 17, 9]):
            assert engine.model.forward([*source, *answer]).tobytes() == twin.forward(source, answer).tobytes()
        check = verify.check_reference(engine, max_tokens=6)
        assert check.equal, check.results
        assert set(check.results) == {"logits", "greedy", "sampled", "controlled", "modern", "adaptive"}
        report = verify.run(engine)
        assert {"logits", "greedy", "sampled"} <= set(report.parts)


@pytest.mark.parametrize("distance", [0, -1, -3, -4, -7, -12, -40, 5])
def test_reference_decoder_buckets(distance):
    from etalii_dllm import reference

    expected = t5_relative_buckets(np.array([distance]), 8, 16, bidirectional=False)[0]
    assert reference.t5_decoder_bucket(distance, 8, 16) == expected


def test_merging_text_to_text_models(flan, tmp_path):
    from etalii_dllm.merging import merge_models

    checkpoint, engine = flan
    path = checkpoint.parent / "model.dllm"
    merge_models([path, path], tmp_path / "merged.dllm", "slerp", t=0.5)
    merged = DllmEngine.from_model_file(tmp_path / "merged.dllm")
    assert getattr(merged.model, "text_to_text", False)
    options = SamplingOptions(temperature=0.0)
    assert merged.complete(TEXTS[0], 6, options).tokens == engine.complete(TEXTS[0], 6, options).tokens
