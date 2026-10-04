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
        add_dense(directory, config["d_model"], dense, dense_bias, dense_size)
    return config


def add_dense(directory: Path, hidden: int, activation: str, bias: bool = False, size: int = 24) -> None:
    """A sentence-transformers ``Dense`` module in ``2_Dense`` (``modules.json`` must list it)."""
    (directory / "2_Dense").mkdir(exist_ok=True)
    dense_config = {"in_features": hidden, "out_features": size, "bias": bias, "activation_function": DENSE[activation]}
    (directory / "2_Dense" / "config.json").write_text(json.dumps(dense_config), encoding="utf-8")
    values = numerics.fill_gaussian(91, size * hidden).reshape(size, hidden) * np.float32(0.3)
    tensors = {"linear.weight": ("F32", values.astype(np.float32))}
    if bias:
        tensors["linear.bias"] = ("F32", numerics.fill_gaussian(92, size) * np.float32(0.1))
    write_safetensors(directory / "2_Dense" / "model.safetensors", tensors, {"format": "pt"})


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


# Gradients (#377)


def t5_autograd(directory: Path, tokens: list[int], upstream: np.ndarray) -> dict[str, np.ndarray]:
    """transformers' float64 gradients of ``sum(upstream * last_hidden_state)`` under our names."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from etalii_dllm.importing.importer import _t5_name

    model = transformers.T5EncoderModel.from_pretrained(directory).double()
    states = model(torch.tensor([tokens])).last_hidden_state[0]
    (states * torch.tensor(upstream, dtype=torch.float64)).sum().backward()
    return {_t5_name(name): p.grad.numpy() for name, p in model.named_parameters() if p.grad is not None}


def assert_close_gradients(ours: dict[str, np.ndarray], theirs: dict[str, np.ndarray]) -> None:
    assert set(ours) == set(theirs)
    for name, expected in theirs.items():
        scale = max(float(np.abs(expected).max()), 1e-3)
        assert np.abs(ours[name] - expected).max() <= 2e-4 * scale, name


def check_t5_gradients(directory: Path, engine: DllmEngine, tokens: list[int]) -> None:
    from etalii_dllm.training.encoder_backprop import EncoderGradients

    model = engine.model
    weights = {name: np.asarray(values, dtype=np.float32) for name, values in model.tensors.items()}
    gradients = EncoderGradients(model.config)
    result = gradients.encode(weights, tokens)
    assert result.states.tobytes() == model.hidden_states(tokens).tobytes()
    upstream = numerics.fill_gaussian(4, len(tokens) * 32).reshape(len(tokens), 32)
    ours = gradients.gradients(weights, result, upstream)
    again = gradients.gradients(weights, gradients.encode(weights, tokens), upstream)
    assert all(ours[name].tobytes() == values.tobytes() for name, values in again.items())
    theirs = t5_autograd(directory, tokens, upstream)
    theirs = {name: values for name, values in theirs.items() if not name.startswith("projection.")}
    assert_close_gradients(ours, theirs)


def test_gradients_match_autograd(embedder, sentence_t5):
    for checkpoint, engine in (embedder, sentence_t5):
        check_t5_gradients(checkpoint, engine, long_tokens(engine))  # every bucket, the far ones included


@pytest.mark.parametrize("feed_forward_proj", ["silu", "gelu", "gated-relu"])
def test_gradients_of_other_activations(tmp_path, feed_forward_proj):
    write_t5_checkpoint(tmp_path / "checkpoint", feed_forward_proj=feed_forward_proj)
    import_model(tmp_path / "checkpoint", tmp_path / "model.dllm")
    engine = DllmEngine.from_model_file(tmp_path / "model.dllm")
    check_t5_gradients(tmp_path / "checkpoint", engine, engine.embedding_tokens(TEXTS[0]))


def chain_autograd(
    directory: Path, family: str, tokens: list[int], upstream: np.ndarray, activation: str
) -> dict[str, np.ndarray]:
    """sentence-transformers' chain in float64 torch (the encoder, mean pooling, Dense, Normalize): the gradients of
    ``upstream . vector`` under our names."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from etalii_dllm.importing.importer import _bert_name, _t5_name
    from etalii_dllm.importing.safetensors import SafetensorsFile

    if family == "t5":
        model = transformers.T5EncoderModel.from_pretrained(directory).double()
        rename = _t5_name
    else:
        model = transformers.BertModel.from_pretrained(directory, add_pooling_layer=False).double()
        rename = _bert_name
    dense = {
        t.name: torch.tensor(t.to_float32(), dtype=torch.float64, requires_grad=True)
        for t in SafetensorsFile(directory / "2_Dense" / "model.safetensors")
    }
    pooled = model(torch.tensor([tokens])).last_hidden_state[0].mean(dim=0)
    projected = dense["linear.weight"] @ pooled + dense.get("linear.bias", 0)
    if activation == "tanh":
        projected = torch.tanh(projected)
    vector = torch.nn.functional.normalize(projected, dim=0)
    (vector * torch.tensor(upstream, dtype=torch.float64)).sum().backward()
    grads = {rename(name): p.grad.numpy() for name, p in model.named_parameters() if p.grad is not None}
    grads |= {f"projection.{name.split('.')[1]}": p.grad.numpy() for name, p in dense.items()}
    return grads


def check_chain(directory: Path, engine: DllmEngine, text: str) -> None:
    from etalii_dllm.training.encoder_backprop import EncoderGradients, sentence_backward, sentence_vector

    model, settings = engine.model, engine.embedding
    weights = {name: np.asarray(values, dtype=np.float32) for name, values in model.tensors.items()}
    tokens = engine.embedding_tokens(text)
    gradients = EncoderGradients(model.config)
    result = gradients.encode(weights, tokens)
    vector = sentence_vector(weights, result.states, "mean", settings["projection"])
    assert vector.normalized.tobytes() == engine.embed(text).vector.tobytes()  # what the engine serves
    upstream = numerics.fill_gaussian(6, len(vector.normalized))
    dstates, ours = sentence_backward(weights, vector, upstream, len(tokens), "mean", settings["projection"])
    ours |= gradients.gradients(weights, result, dstates)
    theirs = chain_autograd(directory, model.config.family, tokens, upstream, settings["projection"])
    assert_close_gradients(ours, theirs)


def test_projection_gradients_match_autograd(sentence_t5, tmp_path):
    checkpoint, engine = sentence_t5
    check_chain(checkpoint, engine, TEXTS[0])
    write_t5_checkpoint(tmp_path / "tanh", dense="tanh", dense_bias=True, dense_size=16)
    import_model(tmp_path / "tanh", tmp_path / "tanh.dllm")
    check_chain(tmp_path / "tanh", DllmEngine.from_model_file(tmp_path / "tanh.dllm"), TEXTS[1])
    from etalii_dllm.training.encoder_backprop import sentence_vector

    with pytest.raises(ValueError, match="projection activation"):
        sentence_vector(engine.model.tensors, np.ones((2, 32), np.float32), "mean", "relu")


def bert_with_dense(directory: Path) -> Path:
    """A BERT embedder with a tanh Dense module with a bias after its mean pooling (as LaBSE has), and Normalize."""
    from test_encoders import write_bert_checkpoint

    config, _ = write_bert_checkpoint(directory)
    add_dense(directory, config["hidden_size"], "tanh", bias=True, size=16)
    modules = [MODULES[0], MODULES[1], MODULES[2], MODULES[3]]
    (directory / "modules.json").write_text(json.dumps(modules), encoding="utf-8")
    return directory


def test_bert_projection_gradients(tmp_path):
    bert_with_dense(tmp_path / "bert")
    import_model(tmp_path / "bert", tmp_path / "bert.dllm")
    engine = DllmEngine.from_model_file(tmp_path / "bert.dllm")
    assert engine.model.config.projection_size == 16 and engine.embedding["projection"] == "tanh"
    check_chain(tmp_path / "bert", engine, "the quick brown fox")


# dllm finetune (#378)


def write_pairs(path: Path) -> Path:
    rows = [
        {"anchor": "quick fox", "positive": TEXTS[0], "negative": TEXTS[2]},
        {"anchor": "how are you", "positive": TEXTS[1]},
        {"anchor": "naive cafe", "positive": TEXTS[2], "negative": TEXTS[1]},
        {"anchor": "same bits", "positive": TEXTS[3]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_finetune_runs_are_golden(embedder, sentence_t5, tmp_path):
    from golden_values import T5_FINETUNE_FINGERPRINT

    from etalii_dllm.training import AdamWConfig, FineTuner, RunConfig
    from etalii_dllm.training.encoder_data import EncoderData, read_examples

    for kind, (checkpoint, engine) in (("t5", embedder), ("sentence-t5", sentence_t5)):
        data = EncoderData.from_records(
            read_examples(write_pairs(tmp_path / "pairs.jsonl"), "embedding"), engine, "embedding", MAX_SEQ_LENGTH
        )
        run = RunConfig(3, 2, MAX_SEQ_LENGTH, 1, AdamWConfig(1e-3), objective="embedding")
        model_file = ModelFile(checkpoint.parent / "model.dllm")
        first = FineTuner.from_model_file(model_file, data, run)
        first.train()
        second = FineTuner.from_model_file(model_file, data, run)
        second.train(until=1)
        second.save_checkpoint(tmp_path / f"{kind}.dllmckpt")
        resumed = FineTuner.load_checkpoint(tmp_path / f"{kind}.dllmckpt", data)
        resumed.train()
        assert resumed.losses == first.losses
        fingerprint = first.export(tmp_path / f"{kind}.dllm")
        assert resumed.export(tmp_path / f"{kind}-resumed.dllm") == fingerprint
        assert fingerprint == T5_FINETUNE_FINGERPRINT[kind]
        tuned = ModelFile(tmp_path / f"{kind}.dllm")
        trained = ["relative_bias.weight", *(["projection.weight"] if kind == "sentence-t5" else [])]
        for name in trained:  # the bucket table and the Dense projection train
            assert (np.asarray(tuned.tensors[name]) != np.asarray(model_file.tensors[name])).any(), name
        assert tuned.embedding == model_file.embedding
        tuned_engine = DllmEngine.from_model_file(tmp_path / f"{kind}.dllm")
        assert tuned_engine.embed(TEXTS[0]).vector.tobytes() != engine.embed(TEXTS[0]).vector.tobytes()


def test_finetune_command(sentence_t5, tmp_path, capsys):
    from etalii_dllm.cli import main as cli

    checkpoint, _ = sentence_t5
    pairs = write_pairs(tmp_path / "pairs.jsonl")
    command = ["finetune", str(checkpoint.parent / "model.dllm"), "--data", str(pairs), "--steps", "2"]
    command += ["--batch-size", "2", "-o", str(tmp_path / "tuned.dllm")]
    assert cli(command) == 0
    assert "4 examples" in capsys.readouterr().out
    assert ModelFile(tmp_path / "tuned.dllm").config.family == "t5"


# LoRA (#379)


def test_lora_on_t5(embedder, sentence_t5, tmp_path, capsys):
    import re

    torch = pytest.importorskip("torch")  # noqa: F841
    transformers = pytest.importorskip("transformers")
    from etalii_dllm.cli import main as cli
    from etalii_dllm.importing.safetensors import SafetensorsFile
    from etalii_dllm.lora import AdapterError, LoraConfig, read_peft, target_modules, target_weights

    checkpoint, engine = sentence_t5
    path = checkpoint.parent / "model.dllm"
    pairs = write_pairs(tmp_path / "pairs.jsonl")
    common = ["finetune", str(path), "--data", str(pairs), "--steps", "2", "--batch-size", "2"]
    common += ["--learning-rate", "1e-2", "--lora-rank", "2"]
    assert cli([*common, "-o", str(tmp_path / "merged.dllm"), "--adapter-output", str(tmp_path / "adapter")]) == 0
    capsys.readouterr()
    settings = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert settings["task_type"] == "FEATURE_EXTRACTION"
    stored = {t.name: t.to_float32() for t in SafetensorsFile(tmp_path / "adapter" / "adapter_model.safetensors")}
    assert stored["base_model.model.encoder.block.0.layer.0.SelfAttention.q.lora_A.weight"].shape == (2, 32)
    assert stored["base_model.model.encoder.block.1.layer.1.DenseReluDense.wi_0.lora_B.weight"].shape == (64, 2)
    modules = {name for name, _ in transformers.T5EncoderModel.from_pretrained(checkpoint).named_modules()}
    adapted = {key.removeprefix("base_model.model.").rsplit(".lora_", 1)[0] for key in stored}
    assert adapted <= modules and len(adapted) == 2 * 7
    assert {m for m in modules if re.fullmatch(settings["target_modules"], m)} == adapted
    merged = DllmEngine.from_model_file(tmp_path / "merged.dllm").embed("hello there").vector
    loaded = DllmEngine.from_model_file(path, adapter=tmp_path / "adapter").embed("hello there").vector
    assert merged.tobytes() == loaded.tobytes()
    assert merged.tobytes() != engine.embed("hello there").vector.tobytes()
    import_model(tmp_path / "adapter", tmp_path / "imported.dllm", base=path, licence="mit")
    assert ModelFile(tmp_path / "imported.dllm").fingerprint == ModelFile(tmp_path / "merged.dllm").fingerprint
    lora, adapters = read_peft(tmp_path / "adapter", engine.model.config)
    assert lora.targets == ("q", "k", "v", "o", "gate", "up", "down") and len(adapters) == 2 * 7 * 2
    # T5 v1.0 has one MLP input, wi; it has no gate to adapt
    config = embedder[1].model.config
    pattern = r".*encoder\.block\.\d+\.(?:layer\.0\.SelfAttention\.q|layer\.1\.DenseReluDense\.wi)"
    assert target_modules(config, LoraConfig(2, 4.0, ("q", "up"))) == pattern
    with pytest.raises(AdapterError, match="gate"):
        target_weights(config, LoraConfig(2, 4.0, ("gate",)))
    # a module T5 does not have, or a block the model does not have
    for key in (
        "base_model.model.encoder.block.0.layer.1.DenseReluDense.wi_0.lora_A.weight",
        "base_model.model.encoder.block.9.layer.0.SelfAttention.q.lora_A.weight",
        "base_model.model.encoder.layer.0.attention.self.query.lora_A.weight",
    ):
        from etalii_dllm.importing.safetensors import write_safetensors as write_tensors

        (tmp_path / "bad").mkdir(exist_ok=True)
        (tmp_path / "bad" / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 2}), encoding="utf-8")
        write_tensors(tmp_path / "bad" / "adapter_model.safetensors", {key: np.zeros((2, 32), np.float32)})
        with pytest.raises(AdapterError, match=r"unsupported adapter tensor|does not fit"):
            read_peft(tmp_path / "bad", config)


# Export (#380)


def test_export_round_trips(embedder, sentence_t5, tmp_path):
    from etalii_dllm.exporting import ExportError, export_gguf, export_safetensors

    for name, (checkpoint, engine) in (("t5", embedder), ("sentence-t5", sentence_t5)):
        original = ModelFile(checkpoint.parent / "model.dllm")
        export_safetensors(original, tmp_path / name)
        exported = json.loads((tmp_path / name / "config.json").read_text())
        assert exported["architectures"] == ["T5EncoderModel"]
        import_model(tmp_path / name, tmp_path / f"{name}.dllm", repository=f"example/{name}")
        again = ModelFile(tmp_path / f"{name}.dllm")
        assert again.fingerprint == original.fingerprint and again.config == original.config
        assert again.embedding == original.embedding
        tokens = long_tokens(engine)
        expected = transformers_states(tmp_path / name, tokens)
        np.testing.assert_allclose(engine.model.hidden_states(tokens), expected, atol=3e-5)
        with pytest.raises(ExportError, match="T5"):
            export_gguf(original, tmp_path / f"{name}.gguf")
    modules = json.loads((tmp_path / "sentence-t5" / "modules.json").read_text())
    assert [m["path"] for m in modules] == ["", "1_Pooling", "2_Dense", "3_Normalize"]
    assert (tmp_path / "sentence-t5" / "3_Normalize").is_dir()
    tokens = sentence_t5[1].embedding_tokens(TEXTS[3])
    expected = sentence_transformers_vector(tmp_path / "sentence-t5", tokens, "identity")
    np.testing.assert_allclose(sentence_t5[1].embed(TEXTS[3]).vector, expected, atol=3e-5)


def test_export_refusals_and_bert_projection(tmp_path):
    from etalii_dllm.exporting import ExportError, export_gguf, export_safetensors

    bert_with_dense(tmp_path / "bert")
    import_model(tmp_path / "bert", tmp_path / "bert.dllm")
    original = ModelFile(tmp_path / "bert.dllm")
    export_safetensors(original, tmp_path / "out")
    import_model(tmp_path / "out", tmp_path / "again.dllm")
    assert ModelFile(tmp_path / "again.dllm").fingerprint == original.fingerprint
    dense = json.loads((tmp_path / "out" / "2_Dense" / "config.json").read_text())
    assert dense == {"activation_function": DENSE["tanh"], "bias": True, "in_features": 32, "out_features": 16}
    with pytest.raises(ExportError, match="Dense projection"):
        export_gguf(original, tmp_path / "bert.gguf")
    from etalii_dllm.encoder_export import t5_tensor_name

    write_t5_checkpoint(tmp_path / "t5")
    import_model(tmp_path / "t5", tmp_path / "t5.dllm")
    with pytest.raises(ExportError, match="unexpected encoder tensor"):
        t5_tensor_name("layers.0.mlp.experts.0.up.weight", ModelFile(tmp_path / "t5.dllm").config)
    # T5's gated MLP is the tanh GELU's only: a gated erf GELU or SiLU has no feed_forward_proj
    for activation in ("gelu", "silu"):
        write_t5_checkpoint(tmp_path / activation, feed_forward_proj=f"gated-{activation}")
        import_model(tmp_path / activation, tmp_path / f"{activation}.dllm")
        model = ModelFile(tmp_path / f"{activation}.dllm")
        if model.config.activation == "gelu_tanh":  # transformers' gated-gelu is the tanh GELU
            export_safetensors(model, tmp_path / f"{activation}-out")
        else:
            with pytest.raises(ExportError, match="gated"):
                export_safetensors(model, tmp_path / f"{activation}-out")
