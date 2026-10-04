"""T5 encoder embedders (#372-#375): T5's exact relative attention bias, the t5 encoder family against transformers'
``T5EncoderModel``, sentence-t5/GTR-T5 style embedders with a Dense projection, and the reference implementation."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from model_fixtures import write_safetensors

from etalii_dllm import numerics
from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.encoder import t5_relative_buckets
from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.importing.importer import t5_config
from etalii_dllm.modelfile import ModelFile

tokenizers = pytest.importorskip("tokenizers")
pytest.importorskip("sentencepiece")

TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Hello, world! How are you?",
    "naïve café façade",
    "Deterministic models give the same bits on every machine, whatever the load.",
]
MAX_SEQ_LENGTH = 48
T5_CONFIG = {
    "model_type": "t5",
    "architectures": ["T5EncoderModel"],
    "d_model": 32,
    "d_kv": 8,
    "d_ff": 64,
    "num_layers": 2,
    "num_heads": 4,
    "relative_attention_num_buckets": 8,
    "relative_attention_max_distance": 16,
    "layer_norm_epsilon": 1e-6,
    "feed_forward_proj": "relu",
    "is_encoder_decoder": True,
    "dropout_rate": 0.0,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "decoder_start_token_id": 0,
}
MODULES = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
    {"idx": 3, "name": "3", "path": "3_Normalize", "type": "sentence_transformers.models.Normalize"},
]
DENSE = {
    "identity": "torch.nn.modules.linear.Identity",
    "tanh": "torch.nn.modules.activation.Tanh",
}


def t5_tokenizer():
    """A Unigram tokenizer laid out as transformers converts T5's ``spiece.model``: ``<pad>``, ``</s>``, ``<unk>``
    first, the Precompiled/space-collapsing normaliser, WhitespaceSplit and Metaspace, and ``$A </s>``."""
    from test_deberta import frozen_pieces
    from test_unigram import sentencepiece_model
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

    _, charsmap = sentencepiece_model()
    vocab = [("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), *frozen_pieces(300)]
    tokenizer = Tokenizer(models.Unigram(vocab, 2, False))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Precompiled(charsmap), normalizers.Replace(Regex(" {2,}"), " ")]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")]
    )
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A:0 </s>:0", pair="$A:0 </s>:0 $B:0 </s>:0", special_tokens=[("</s>", 1)]
    )
    tokenizer.add_special_tokens([AddedToken(t, special=True) for t in ("<pad>", "</s>", "<unk>")])
    return tokenizer


def t5_weights(config: dict, seed: int = 17) -> dict[str, np.ndarray]:
    """Float32 weights keyed by transformers' ``T5EncoderModel`` names."""
    hidden, inner, size = config["d_model"], config["d_kv"] * config["num_heads"], config["d_ff"]
    counter = iter(range(seed * 1000, seed * 1000 + 1000))

    def matrix(rows: int, columns: int, scale: float = 0.2) -> np.ndarray:
        values = numerics.fill_gaussian(next(counter), rows * columns).reshape(rows, columns)
        return (values * np.float32(scale)).astype(np.float32)

    def vector(size: int) -> np.ndarray:
        return (np.float32(1.0) + numerics.fill_gaussian(next(counter), size) * np.float32(0.1)).astype(np.float32)

    weights = {
        "shared.weight": matrix(config["vocab_size"], hidden, 1.0),
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": matrix(
            config["relative_attention_num_buckets"], config["num_heads"], 1.0
        ),
        "encoder.final_layer_norm.weight": vector(hidden),
    }
    gated = config["feed_forward_proj"].startswith("gated-")
    for i in range(config["num_layers"]):
        p = f"encoder.block.{i}.layer."
        for name in "qkv":
            weights[p + f"0.SelfAttention.{name}.weight"] = matrix(inner, hidden)
        weights[p + "0.SelfAttention.o.weight"] = matrix(hidden, inner)
        weights[p + "0.layer_norm.weight"] = vector(hidden)
        weights[p + "1.layer_norm.weight"] = vector(hidden)
        for name in ("wi_0", "wi_1") if gated else ("wi",):
            weights[p + f"1.DenseReluDense.{name}.weight"] = matrix(size, hidden)
        weights[p + "1.DenseReluDense.wo.weight"] = matrix(hidden, size)
    return weights


def write_t5_checkpoint(
    directory: Path, *, dense: str | None = None, dense_bias: bool = False, dense_size: int = 24, **changes
) -> dict:
    """A T5 encoder checkpoint as sentence-transformers stores sentence-t5 and GTR-T5: mean pooling, then (with
    ``dense``, its activation) a Dense projection, then Normalize."""
    reference = t5_tokenizer()
    directory.mkdir(parents=True, exist_ok=True)
    config = {**T5_CONFIG, "vocab_size": reference.get_vocab_size() + 4, **changes}
    weights = t5_weights(config)
    write_safetensors(directory / "model.safetensors", {n: ("F32", v) for n, v in weights.items()}, {"format": "pt"})
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "tokenizer.json").write_text(reference.to_str(), encoding="utf-8")
    tokenizer_config = {
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_max_length": MAX_SEQ_LENGTH,
        "tokenizer_class": "T5Tokenizer",
    }
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    (directory / "README.md").write_bytes(b"---\nlicense: apache-2.0\n---\n\n# Tiny T5\n")
    modules = MODULES if dense else [MODULES[0], MODULES[1], {**MODULES[3], "idx": 2, "path": "2_Normalize"}]
    (directory / "modules.json").write_text(json.dumps(modules), encoding="utf-8")
    (directory / "1_Pooling").mkdir(exist_ok=True)
    settings = {"word_embedding_dimension": config["d_model"], "pooling_mode_mean_tokens": True}
    (directory / "1_Pooling" / "config.json").write_text(json.dumps(settings), encoding="utf-8")
    (directory / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": MAX_SEQ_LENGTH}), "utf-8")
    if dense:
        (directory / "2_Dense").mkdir(exist_ok=True)
        hidden = config["d_model"]
        dense_config = {
            "in_features": hidden,
            "out_features": dense_size,
            "bias": dense_bias,
            "activation_function": DENSE[dense],
        }
        (directory / "2_Dense" / "config.json").write_text(json.dumps(dense_config), encoding="utf-8")
        values = numerics.fill_gaussian(91, dense_size * hidden).reshape(dense_size, hidden) * np.float32(0.3)
        tensors = {"linear.weight": ("F32", values.astype(np.float32))}
        if dense_bias:
            tensors["linear.bias"] = ("F32", numerics.fill_gaussian(92, dense_size) * np.float32(0.1))
        write_safetensors(directory / "2_Dense" / "model.safetensors", tensors, {"format": "pt"})
    return config


@pytest.fixture(scope="module")
def embedder(tmp_path_factory) -> tuple[Path, DllmEngine]:
    """T5 v1.0 (ReLU) with mean pooling and Normalize."""
    directory = tmp_path_factory.mktemp("t5")
    write_t5_checkpoint(directory / "checkpoint")
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-t5")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


@pytest.fixture(scope="module")
def sentence_t5(tmp_path_factory) -> tuple[Path, DllmEngine]:
    """T5 v1.1 (gated tanh GELU) with mean pooling, an identity Dense projection and Normalize, as sentence-t5."""
    directory = tmp_path_factory.mktemp("sentence-t5")
    write_t5_checkpoint(directory / "checkpoint", dense="identity", feed_forward_proj="gated-gelu")
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-sentence-t5")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


def transformers_states(directory: Path, tokens: list[int]) -> np.ndarray:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.T5EncoderModel.from_pretrained(directory).eval()
    with torch.no_grad():
        return model(torch.tensor([tokens])).last_hidden_state[0].double().numpy()


def long_tokens(engine: DllmEngine) -> list[int]:
    """A sequence far longer than the buckets' maximum distance, so every bucket is used."""
    tokens: list[int] = []
    for text in TEXTS:
        tokens += engine.tokenizer.encode(text)
    return [*tokens[:63], 1]


# The relative attention bias (#372)


@pytest.mark.parametrize(("buckets", "distance"), [(32, 128), (32, 256), (64, 512), (8, 16), (16, 64), (30, 100)])
def test_relative_buckets_match_transformers(buckets, distance):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.t5.modeling_t5 import T5Attention

    distances = np.arange(-6000, 6001)
    expected = T5Attention._relative_position_bucket(torch.tensor(distances), True, buckets, distance).numpy()
    ours = t5_relative_buckets(distances, buckets, distance)
    assert (ours == expected).all()
    from etalii_dllm import reference

    assert [reference.t5_relative_bucket(int(d), buckets, distance) for d in distances[::7]] == list(ours[::7])


def test_bias_and_bucket_refusals():
    from etalii_dllm.encoder import t5_bias

    with pytest.raises(ValueError, match="at least four"):
        t5_relative_buckets([1, 2], 3, 16)
    config = t5_config({**T5_CONFIG, "vocab_size": 50})
    table = numerics.fill_gaussian(3, 8 * 4).reshape(8, 4)
    bias = t5_bias(config, table, 5)
    assert bias.shape == (4, 5, 5)
    assert bias[2, 1, 3] == table[t5_relative_buckets([2], 8, 16)[0], 2]  # key 3 after query 1: distance +2


# The encoder family (#373)


def test_config_mapping():
    config = t5_config({**T5_CONFIG, "vocab_size": 50})
    assert config.family == "t5" and config.is_encoder and config.activation == "relu" and not config.gated_mlp
    assert (config.position_buckets, config.max_relative_positions, config.head_dim) == (8, 16, 8)
    assert TransformerConfig.from_dict(config.to_dict()) == config
    gated = t5_config({**T5_CONFIG, "vocab_size": 50, "feed_forward_proj": "gated-gelu"})
    assert gated.gated_mlp and gated.activation == "gelu_tanh"
    assert "mlp.gate.weight" in "".join(gated.tensor_shapes())
    assert t5_config({**T5_CONFIG, "vocab_size": 50, "feed_forward_proj": "gated-silu"}).activation == "silu"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"feed_forward_proj": "gated-swish"}, "feed_forward_proj"),
        ({"relative_attention_num_buckets": 1}, "position_buckets"),
        ({"is_encoder_decoder": False, "is_decoder": True}, "decoder"),
    ],
)
def test_config_refusals(change, message):
    with pytest.raises(ModelImportError, match=message):
        t5_config({**T5_CONFIG, "vocab_size": 50, **change})
    with pytest.raises(ModelImportError, match="context-length"):
        t5_config({**T5_CONFIG, "vocab_size": 50}, context_length=64)


def test_config_validation():
    config = t5_config({**T5_CONFIG, "vocab_size": 50})
    for change, message in (
        ({"type_vocabulary_size": 2}, "token types"),
        ({"projection_bias": True}, "projection bias needs"),
        ({"family": "bert", "type_vocabulary_size": 2, "gated_mlp": True, "activation": "gelu"}, "gated_mlp"),
        ({"family": "llama", "projection_size": 4}, "encoders only"),
    ):
        values = {**config.to_dict(), **change}
        if values["family"] != "t5":
            values.pop("position_buckets"), values.pop("max_relative_positions")
        with pytest.raises(ValueError, match=message):
            TransformerConfig.from_dict(values)


def test_import_records_the_encoder(embedder):
    checkpoint, engine = embedder
    file = ModelFile(checkpoint.parent / "model.dllm")
    assert file.config.family == "t5" and file.embedding["pooling"] == "mean"
    assert "projection" not in file.embedding and not file.config.projection_size
    with pytest.raises(ValueError, match="cannot generate"):
        engine.model.forward([1, 2])


def test_states_match_transformers(embedder, sentence_t5):
    for checkpoint, engine in (embedder, sentence_t5):
        for tokens in (engine.embedding_tokens(TEXTS[0]), long_tokens(engine)):
            np.testing.assert_allclose(
                engine.model.hidden_states(tokens), transformers_states(checkpoint, tokens), atol=3e-5
            )


def test_tokens_match_transformers(embedder):
    transformers = pytest.importorskip("transformers")
    checkpoint, engine = embedder
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    for text in TEXTS:
        assert engine.embedding_tokens(text) == tokenizer(text)["input_ids"]
        assert engine.embedding_tokens(text)[-1] == 1  # </s>


def test_thread_invariance_and_goldens(embedder, sentence_t5):
    from golden_values import T5_EMBEDDING_FINGERPRINT

    fixed = [5, 17, 40, 9, 200, 5, 77, 12, 1] * 6
    fingerprints = []
    for _, engine in (embedder, sentence_t5):
        tokens = long_tokens(engine)
        try:
            results = set()
            for threads in (1, 2, 5):
                numerics.set_threads(threads)
                results.add(engine.model.hidden_states(tokens).tobytes())
        finally:
            numerics.set_threads(0)
        assert len(results) == 1
        fingerprints.append(numerics.fingerprint(engine.model.hidden_states(fixed)))
    assert fingerprints == list(T5_EMBEDDING_FINGERPRINT)


def test_quantised_t5_runs(tmp_path, embedder):
    checkpoint, engine = embedder
    quantised = DllmEngine.from_model_file(checkpoint.parent / "model.dllm", quantize="q8_0")
    tokens = engine.embedding_tokens(TEXTS[1])
    a, b = quantised.embed(TEXTS[1]).vector, engine.embed(TEXTS[1]).vector
    assert a.tobytes() != b.tobytes() and float(np.dot(a.astype(np.float64), b)) > 0.99
    assert quantised.model.hidden_states(tokens).shape == (len(tokens), 32)


def test_checkpoint_variants(tmp_path):
    """A T5ForConditionalGeneration checkpoint drops its decoder; a checkpoint that only stores
    encoder.embed_tokens imports the same embedding."""
    from etalii_dllm.importing.safetensors import SafetensorsFile

    write_t5_checkpoint(tmp_path / "plain")
    stored = {t.name: t.to_float32() for t in SafetensorsFile(tmp_path / "plain" / "model.safetensors")}
    import_model(tmp_path / "plain", tmp_path / "plain.dllm")
    expected = ModelFile(tmp_path / "plain.dllm").fingerprint
    full = dict(stored)
    full["decoder.final_layer_norm.weight"] = stored["encoder.final_layer_norm.weight"]
    full["lm_head.weight"] = stored["shared.weight"]
    only = {("encoder.embed_tokens.weight" if n == "shared.weight" else n): v for n, v in stored.items()}
    for name, tensors in (("full", full), ("only", only)):
        shutil.copytree(tmp_path / "plain", tmp_path / name, ignore=shutil.ignore_patterns("model.safetensors"))
        write_safetensors(tmp_path / name / "model.safetensors", {n: ("F32", v) for n, v in tensors.items()}, {})
        import_model(tmp_path / name, tmp_path / f"{name}.dllm")
        assert ModelFile(tmp_path / f"{name}.dllm").fingerprint == expected
    from etalii_dllm.importing.importer import _t5_name

    with pytest.raises(ModelImportError, match="unexpected tensor"):
        _t5_name("encoder.block.0.layer.2.stray.weight")
    (tmp_path / "only" / "tokenizer.json").unlink()
    with pytest.raises(ModelImportError, match=r"no tokenizer\.json"):
        import_model(tmp_path / "only", tmp_path / "none.dllm")


# Sentence-T5 and GTR-T5 embedders (#374)


def sentence_transformers_vector(directory: Path, tokens: list[int], activation: str) -> np.ndarray:
    """sentence-transformers' module chain in torch: mean pooling, Dense (``linear`` then the activation),
    Normalize."""
    torch = pytest.importorskip("torch")
    from etalii_dllm.importing.safetensors import SafetensorsFile

    states = torch.tensor(transformers_states(directory, tokens))
    pooled = states.mean(dim=0)
    dense = {
        t.name: torch.tensor(t.to_float32(), dtype=torch.float64)
        for t in SafetensorsFile(directory / "2_Dense" / "model.safetensors")
    }
    projected = dense["linear.weight"] @ pooled + dense.get("linear.bias", 0)
    if activation == "tanh":
        projected = torch.tanh(projected)
    return torch.nn.functional.normalize(projected, dim=0).numpy()


def test_sentence_t5_embeddings(embedder, sentence_t5):
    checkpoint, engine = sentence_t5
    file = ModelFile(checkpoint.parent / "model.dllm")
    assert file.config.projection_size == 24 and not file.config.projection_bias
    assert file.embedding["projection"] == "identity"
    for text in TEXTS:
        vector = engine.embed(text).vector
        assert vector.shape == (24,)
        expected = sentence_transformers_vector(checkpoint, engine.embedding_tokens(text), "identity")
        np.testing.assert_allclose(vector, expected, atol=3e-5)
    assert engine.embed(TEXTS[0], dimensions=8).vector.shape == (8,)
    with pytest.raises(ValueError, match="no projection"):
        embedder[1].model.project([0.0] * 32, "tanh")


def test_tanh_projection_with_bias(tmp_path):
    from golden_values import T5_SENTENCE_FINGERPRINT

    write_t5_checkpoint(tmp_path / "checkpoint", dense="tanh", dense_bias=True, dense_size=16)
    import_model(tmp_path / "checkpoint", tmp_path / "tanh.dllm")
    engine = DllmEngine.from_model_file(tmp_path / "tanh.dllm")
    assert engine.model.config.projection_bias and engine.embedding["projection"] == "tanh"
    vector = engine.embed(TEXTS[3]).vector
    expected = sentence_transformers_vector(tmp_path / "checkpoint", engine.embedding_tokens(TEXTS[3]), "tanh")
    np.testing.assert_allclose(vector, expected, atol=3e-5)
    assert numerics.fingerprint(vector) == T5_SENTENCE_FINGERPRINT


def test_dense_refusals(tmp_path):
    write_t5_checkpoint(tmp_path / "c", dense="identity")
    dense = tmp_path / "c" / "2_Dense" / "config.json"
    original = json.loads(dense.read_text())
    for change, message in (
        ({"activation_function": "torch.nn.modules.activation.ReLU"}, "Dense activation"),
        ({"in_features": 7}, "does not read"),
        ({"bias": True}, "disagree"),
    ):
        dense.write_text(json.dumps({**original, **change}))
        with pytest.raises(ModelImportError, match=message):
            import_model(tmp_path / "c", tmp_path / "c.dllm")
    dense.write_text(json.dumps(original))
    modules = tmp_path / "c" / "modules.json"
    swapped = [MODULES[0], MODULES[2], MODULES[1], MODULES[3]]
    modules.write_text(json.dumps(swapped))
    with pytest.raises(ModelImportError, match="must follow the pooling"):
        import_model(tmp_path / "c", tmp_path / "c.dllm")
    modules.write_text(
        json.dumps([*MODULES, {"idx": 4, "path": "4", "type": "sentence_transformers.models.LayerNorm"}])
    )
    with pytest.raises(ModelImportError, match="one Dense"):
        import_model(tmp_path / "c", tmp_path / "c.dllm")
    modules.write_text(json.dumps(MODULES))
    (tmp_path / "c" / "2_Dense" / "model.safetensors").rename(tmp_path / "c" / "2_Dense" / "pytorch_model.bin")
    with pytest.raises(ModelImportError, match=r"model\.safetensors"):
        import_model(tmp_path / "c", tmp_path / "c.dllm")


def test_t5_in_every_embedding_front_end(sentence_t5, monkeypatch):
    from fastapi.testclient import TestClient

    from etalii_dllm import engine as engine_module
    from etalii_dllm.retrieval import build_index

    checkpoint, engine = sentence_t5
    index = build_index(engine, [(f"doc{i}.txt", text) for i, text in enumerate(TEXTS)], chunk_tokens=8)
    first, again = index.search(engine, TEXTS[1], 2), index.search(engine, TEXTS[1], 2)
    assert [(h.chunk.source, h.score) for h in first] == [(h.chunk.source, h.score) for h in again]
    for name in [name for name in os.environ if name.startswith("DLLM_")]:
        monkeypatch.delenv(name)  # front-end tests that call use_model_file leave these set
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(checkpoint.parent / "model.dllm"))
    engine_module.default_engine.cache_clear()
    from etalii_dllm.server.app import app

    try:
        response = TestClient(app).post("/v1/embeddings", json={"input": TEXTS[2]}).json()
        expected = engine.embed(TEXTS[2]).vector
        assert np.asarray(response["data"][0]["embedding"], np.float32).tobytes() == expected.tobytes()
    finally:
        engine_module.default_engine.cache_clear()


# Verification and refusals (#375)


def test_reference_implementation_and_verify(embedder, sentence_t5):
    from etalii_dllm import reference, verify

    for _, engine in (embedder, sentence_t5):
        twin = reference.ReferenceEncoder.from_engine(engine)
        for tokens in (engine.embedding_tokens(TEXTS[3]), long_tokens(engine)):
            assert engine.model.hidden_states(tokens).tobytes() == twin.hidden_states(tokens).tobytes()
        assert verify.check_reference(engine).equal


def test_fine_tuning_lora_and_export_are_refused(embedder, sentence_t5, tmp_path):
    from etalii_dllm.encoder_export import export_encoder_gguf, export_encoder_safetensors
    from etalii_dllm.lora import AdapterError, LoraConfig, target_weights
    from etalii_dllm.training.encoder_backprop import EncoderGradients

    checkpoint, _ = embedder
    file = ModelFile(checkpoint.parent / "model.dllm")
    with pytest.raises(AdapterError, match="T5"):
        target_weights(file.config, LoraConfig(rank=2, alpha=4.0))
    with pytest.raises(ValueError, match="T5"):
        export_encoder_safetensors(file, tmp_path / "out")
    with pytest.raises(ValueError, match="T5"):
        export_encoder_gguf(file, tmp_path / "out.gguf")
    with pytest.raises(ValueError, match="fine-tuning T5"):
        EncoderGradients(file.config)
    checkpoint, _ = sentence_t5
    projected = ModelFile(checkpoint.parent / "model.dllm")
    import dataclasses

    bert_like = dataclasses.replace(projected.config, family="bert", type_vocabulary_size=2, gated_mlp=False,
                                    activation="gelu", position_buckets=0, max_relative_positions=0)  # fmt: skip
    with pytest.raises(ValueError, match="Dense projection"):
        EncoderGradients(bert_like)
