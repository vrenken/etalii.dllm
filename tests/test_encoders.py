"""BERT-style encoders (issues #338 and #339): the layer_norm kernel, the encoder against transformers' BertModel,
sentence-transformers imports (CLS and mean pooling, max_seq_length truncation), thread invariance, Q8_0, the
served embeddings and the refusals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from golden_values import ENCODER_EMBEDDINGS
from model_fixtures import write_safetensors

from etalii_dllm import engine as engine_module
from etalii_dllm import numerics
from etalii_dllm.architecture import ENCODER_ONLY, TransformerConfig
from etalii_dllm.chat import ChatMessage
from etalii_dllm.encoder import Encoder
from etalii_dllm.engine import DllmEngine, default_engine
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.importing.importer import bert_config
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.transformer import Transformer

tokenizers = pytest.importorskip("tokenizers")

BERT_CONFIG = {
    "model_type": "bert",
    "architectures": ["BertModel"],
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "max_position_embeddings": 40,
    "type_vocab_size": 2,
    "layer_norm_eps": 1e-12,
    "hidden_act": "gelu",
    "position_embedding_type": "absolute",
}
MODULES = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
]
TEXTS = ["The quick brown fox.", "Hello, world! How are you?", "naïve café façade"]
MAX_SEQ_LENGTH = 12


def bert_weights(config: dict, seed: int = 5) -> dict[str, np.ndarray]:
    """Float32 weights for a BERT config, keyed by Hugging Face ``BertModel`` name."""
    hidden, inner = config["hidden_size"], config["intermediate_size"]
    counter = iter(range(seed * 1000, seed * 1000 + 1000))

    def gaussian(*shape: int) -> np.ndarray:
        return numerics.fill_gaussian(next(counter), int(np.prod(shape))).reshape(shape)

    def matrix(rows: int, columns: int) -> np.ndarray:
        return (gaussian(rows, columns) * np.float32(0.2)).astype(np.float32)

    def vector(size: int, centre: float = 0.0) -> np.ndarray:
        return (np.float32(centre) + gaussian(size) * np.float32(0.1)).astype(np.float32)

    weights = {
        "embeddings.word_embeddings.weight": matrix(config["vocab_size"], hidden),
        "embeddings.position_embeddings.weight": matrix(config["max_position_embeddings"], hidden),
        "embeddings.token_type_embeddings.weight": matrix(config["type_vocab_size"], hidden),
        "embeddings.LayerNorm.weight": vector(hidden, 1.0),
        "embeddings.LayerNorm.bias": vector(hidden),
    }
    for i in range(config["num_hidden_layers"]):
        p = f"encoder.layer.{i}."
        for name in ("attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"):
            weights[p + name + ".weight"] = matrix(hidden, hidden)
            weights[p + name + ".bias"] = vector(hidden)
        weights[p + "intermediate.dense.weight"] = matrix(inner, hidden)
        weights[p + "intermediate.dense.bias"] = vector(inner)
        weights[p + "output.dense.weight"] = matrix(hidden, inner)
        weights[p + "output.dense.bias"] = vector(hidden)
        for name in ("attention.output.LayerNorm", "output.LayerNorm"):
            weights[p + name + ".weight"] = vector(hidden, 1.0)
            weights[p + name + ".bias"] = vector(hidden)
    weights["pooler.dense.weight"] = matrix(hidden, hidden)  # not used by embeddings
    weights["pooler.dense.bias"] = vector(hidden)
    return weights


def write_bert_checkpoint(
    directory: Path,
    pooling: str | None = "pooling_mode_mean_tokens",
    *,
    prefix: str = "",
    config: dict | None = None,
    legacy_norm_names: bool = False,
) -> tuple[dict, dict[str, np.ndarray]]:
    """A BERT checkpoint with a WordPiece tokenizer; with ``pooling``, a sentence-transformers encoder."""
    from test_wordpiece import bert_tokenizer

    reference, _ = bert_tokenizer()
    directory.mkdir(parents=True, exist_ok=True)
    config = {**BERT_CONFIG, "vocab_size": reference.get_vocab_size(), **(config or {})}
    weights = bert_weights(config)
    stored = {}
    for name, values in weights.items():
        if legacy_norm_names and "LayerNorm" in name:
            name = name.replace(".weight", ".gamma").replace(".bias", ".beta")
        stored[prefix + name] = ("F32", values)
    write_safetensors(directory / "model.safetensors", stored, {"format": "pt"})
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "tokenizer.json").write_text(reference.to_str(), encoding="utf-8")
    tokenizer_config = {"cls_token": "[CLS]", "sep_token": "[SEP]", "unk_token": "[UNK]", "do_lower_case": True}
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    (directory / "README.md").write_bytes(b"---\nlicense: apache-2.0\n---\n\n# Tiny encoder\n")
    if pooling is not None:
        (directory / "modules.json").write_text(json.dumps(MODULES), encoding="utf-8")
        (directory / "1_Pooling").mkdir(exist_ok=True)
        settings = {"word_embedding_dimension": config["hidden_size"], pooling: True}
        (directory / "1_Pooling" / "config.json").write_text(json.dumps(settings), encoding="utf-8")
        limits = {"max_seq_length": MAX_SEQ_LENGTH, "do_lower_case": False}
        (directory / "sentence_bert_config.json").write_text(json.dumps(limits), encoding="utf-8")
    return config, weights


@pytest.fixture(scope="module")
def encoder_path(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("encoder")
    write_bert_checkpoint(directory / "checkpoint")
    import_model(directory / "checkpoint", directory / "encoder.dllm", repository="example/tiny-encoder")
    return directory / "encoder.dllm"


@pytest.fixture(scope="module")
def engine(encoder_path) -> DllmEngine:
    return DllmEngine.from_model_file(encoder_path)


def transformers_states(directory: Path, tokens: list[int]) -> np.ndarray:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = json.loads((directory / "config.json").read_text())
    model = transformers.BertModel(transformers.BertConfig(**config), add_pooling_layer=False).eval()
    from etalii_dllm.importing.safetensors import open_checkpoint

    state = {name: torch.from_numpy(t.to_float32().copy()) for name, t in open_checkpoint(directory).items()}
    state = {name: value for name, value in state.items() if not name.startswith("pooler.")}
    model.load_state_dict(state, strict=False)
    with torch.no_grad():
        output = model(torch.tensor([tokens]), token_type_ids=torch.zeros((1, len(tokens)), dtype=torch.long))
    return output.last_hidden_state[0].double().numpy()


def test_layer_norm_kernel_matches_its_definition():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((5, 37)) * 3 + 1).astype(np.float32)
    w = rng.standard_normal(37).astype(np.float32)
    b = rng.standard_normal(37).astype(np.float32)
    got = numerics.layer_norm(x, w, b, 1e-5).numpy()
    xd = x.astype(np.float64)
    mean = np.array([sum(float(v) for v in row) for row in xd]) / 37
    variance = np.array([sum((float(v) - m) ** 2 for v in row) for row, m in zip(xd, mean, strict=True)]) / 37
    expected = ((xd - mean[:, None]) * (1.0 / np.sqrt(variance + 1e-5))[:, None] * w + b).astype(np.float32)
    assert got.tobytes() == expected.tobytes()
    assert numerics.layer_norm(x[0], w, b).shape == (37,)
    with pytest.raises(ValueError, match="weight and bias"):
        numerics.layer_norm(x, w[:3], b)


def test_import_records_the_encoder(encoder_path):
    file = ModelFile(encoder_path)
    assert file.config.family == "bert" and file.config.is_encoder
    assert file.config.type_vocabulary_size == 2 and file.config.activation == "gelu"
    assert file.embedding == {
        "pooling": "mean",
        "normalize": True,
        "prompts": {},
        "default_prompt_name": None,
        "max_tokens": MAX_SEQ_LENGTH,
    }
    assert "pooler.dense.weight" not in file.tensors and "position_embedding.weight" in file.tensors
    assert file.config.to_dict()["type_vocabulary_size"] == 2
    assert TransformerConfig.from_dict(file.config.to_dict()) == file.config


def test_states_match_transformers(tmp_path, engine):
    write_bert_checkpoint(tmp_path / "checkpoint")
    for text in TEXTS:
        tokens = engine.tokenizer.encode(text, add_special_tokens=True)
        expected = transformers_states(tmp_path / "checkpoint", tokens)
        np.testing.assert_allclose(engine.model.hidden_states(tokens), expected, rtol=1e-4, atol=1e-5)


def test_mean_and_cls_pooling(tmp_path, encoder_path, engine):
    tokens = engine.tokenizer.encode(TEXTS[0], add_special_tokens=True)
    states = engine.model.hidden_states(tokens).astype(np.float64)
    mean = states.mean(axis=0)
    np.testing.assert_allclose(engine.embed(TEXTS[0]).vector, mean / np.linalg.norm(mean), rtol=1e-5, atol=1e-6)
    write_bert_checkpoint(tmp_path / "cls", "pooling_mode_cls_token", prefix="bert.", legacy_norm_names=True)
    import_model(tmp_path / "cls", tmp_path / "cls.dllm")
    cls = DllmEngine.from_model_file(tmp_path / "cls.dllm")
    first = cls.model.hidden_states(tokens)[0].astype(np.float64)
    np.testing.assert_allclose(cls.embed(TEXTS[0]).vector, first / np.linalg.norm(first), rtol=1e-5, atol=1e-6)
    # The bert. prefix and the old gamma/beta norm names map to the same tensors.
    assert ModelFile(tmp_path / "cls.dllm").fingerprint == ModelFile(encoder_path).fingerprint


def test_long_inputs_are_truncated_like_sentence_transformers(engine):
    text = "the quick brown fox jumps over the lazy dog " * 4
    embedding = engine.embed(text)
    assert embedding.tokens == MAX_SEQ_LENGTH
    tokens = engine.tokenizer.encode(text, add_special_tokens=True)
    kept = [*tokens[: MAX_SEQ_LENGTH - 1], tokens[-1]]  # [CLS] words... [SEP]
    states = engine.model.hidden_states(kept).astype(np.float64).mean(axis=0)
    np.testing.assert_allclose(embedding.vector, states / np.linalg.norm(states), rtol=1e-5, atol=1e-6)


def test_bare_bert_pools_the_mean_over_every_position(tmp_path):
    write_bert_checkpoint(tmp_path / "bare", None)
    import_model(tmp_path / "bare", tmp_path / "bare.dllm")
    assert ModelFile(tmp_path / "bare.dllm").embedding["max_tokens"] == BERT_CONFIG["max_position_embeddings"]
    bare = DllmEngine.from_model_file(tmp_path / "bare.dllm")
    assert bare.embed(TEXTS[1]).tokens == len(bare.tokenizer.encode(TEXTS[1])) + 2


def test_embeddings_are_golden_and_thread_invariant(engine):
    def digest() -> str:
        vectors = [engine.embed(text).vector for text in TEXTS]
        return hashlib.sha256(b"".join(v.astype("<f4").tobytes() for v in vectors)).hexdigest()

    first = digest()
    try:
        for threads in (1, 3, 8):
            numerics.set_threads(threads)
            assert digest() == first
    finally:
        numerics.set_threads(0)
    assert first == ENCODER_EMBEDDINGS


def test_quantized_encoder(encoder_path, engine):
    quantized = DllmEngine.from_model_file(encoder_path, quantize="q8_0")
    assert quantized.system_fingerprint != engine.system_fingerprint
    a, b = quantized.embed(TEXTS[0]).vector, engine.embed(TEXTS[0]).vector
    assert a.tobytes() != b.tobytes()
    assert float(np.dot(a.astype(np.float64), b)) > 0.99


def test_served_embeddings(encoder_path, engine, monkeypatch):
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(encoder_path))
    default_engine.cache_clear()
    from etalii_dllm.server.app import app

    try:
        client = TestClient(app)
        response = client.post("/v1/embeddings", json={"input": TEXTS[0]}).json()
        expected = engine.embed(TEXTS[0]).vector
        assert np.asarray(response["data"][0]["embedding"], np.float32).tobytes() == expected.tobytes()
        chat = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert chat.status_code == 400 and "encoder" in chat.text
    finally:
        default_engine.cache_clear()


def test_encoders_do_not_generate(encoder_path, engine):
    with pytest.raises(ValueError, match="encoder"):
        engine.chat([ChatMessage("user", "hi")], 4, SamplingOptions())
    file = ModelFile(encoder_path)
    with pytest.raises(ValueError, match="encoder"):
        Transformer(file.config, file.tensors)
    with pytest.raises(ValueError, match="need a decoder"):
        DllmEngine.from_model_file(encoder_path, speculate=4)
    with pytest.raises(ValueError, match="CPU"):
        Encoder(file.config, file.tensors, device="cuda")
    with pytest.raises(ValueError, match="exceed"):
        engine.model.hidden_states(list(range(41)))
    with pytest.raises(ValueError, match="at least one"):
        engine.model.hidden_states([])
    with pytest.raises(ValueError, match="out of range"):
        engine.model.hidden_states([10_000])
    assert ENCODER_ONLY.startswith("an encoder")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"position_embedding_type": "relative_key"}, "position embeddings"),
        ({"hidden_act": "relu"}, "activation 'relu'"),
        ({"hidden_size": 30}, "multiple of num_attention_heads"),
        ({"type_vocab_size": 0}, "token types"),
    ],
)
def test_unsupported_bert_configs(change, message):
    with pytest.raises(ModelImportError, match=message):
        bert_config({**BERT_CONFIG, "vocab_size": 50, **change})


def test_import_refusals(tmp_path):
    with pytest.raises(ModelImportError, match="context-length"):
        bert_config({**BERT_CONFIG, "vocab_size": 50}, context_length=128)
    write_bert_checkpoint(tmp_path / "odd")
    write_safetensors(tmp_path / "odd" / "model.safetensors", {"encoder.mystery": ("F32", np.zeros(2, np.float32))})
    with pytest.raises(ModelImportError, match="unexpected tensor"):
        import_model(tmp_path / "odd", tmp_path / "odd.dllm")
    integers = {"embeddings.LayerNorm.bias": ("I32", np.zeros(2, np.int32))}
    write_safetensors(tmp_path / "odd" / "model.safetensors", integers)
    with pytest.raises(ModelImportError, match="dtype"):
        import_model(tmp_path / "odd", tmp_path / "odd.dllm")


def test_architecture_validation():
    base = bert_config({**BERT_CONFIG, "vocab_size": 50})
    with pytest.raises(ValueError, match="grouped-query"):
        TransformerConfig.from_dict({**base.to_dict(), "kv_heads": 2})
    with pytest.raises(ValueError, match="for encoders only"):
        TransformerConfig.from_dict({**base.to_dict(), "family": "llama"})


def test_the_reference_implementation_gives_the_kernels_bits(engine):
    from etalii_dllm import reference

    twin = reference.ReferenceEncoder.from_engine(engine)
    for text in TEXTS:
        tokens = engine.embedding_tokens(text)
        assert twin.hidden_states(tokens).tobytes() == engine.model.hidden_states(tokens).tobytes()
        assert twin.embed(tokens).tobytes() == engine.embed(text).vector.tobytes()
    x = numerics.fill_gaussian(3, 3 * 50).reshape(3, 50) * np.float32(5)
    w, b = numerics.fill_gaussian(4, 50), numerics.fill_gaussian(5, 50)
    assert reference.layer_norm(x, w, b, 1e-6).tobytes() == numerics.layer_norm(x, w, b, 1e-6).numpy().tobytes()


def test_reference_pooling_variants(encoder_path):
    from etalii_dllm import reference

    file = ModelFile(encoder_path)
    tokens = [2, 7, 9, 3]
    states = Encoder(file.config, file.tensors).hidden_states(tokens)
    last = reference.ReferenceEncoder(
        file.config, file.tensors, embedding={"pooling": "last_token", "normalize": False}
    )
    assert last.embed(tokens).tobytes() == states[-1].tobytes()


def test_verify_an_encoder(encoder_path, capsys, monkeypatch):
    from etalii_dllm import verify
    from etalii_dllm.cli import main

    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, "")
    assert main(["--model", str(encoder_path), "verify", "--json", "--reference"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert sorted(report["parts"]) == ["embeddings", "kernels", "tokenizer", "unicode"]
    assert report["reference"] == {"equal": True, "states": "equal", "embeddings": "equal"}
    assert report["parts"]["kernels"] == verify.REFERENCE["kernels"]


def test_embed_command(encoder_path, engine, capsys, monkeypatch):
    from etalii_dllm.cli import main

    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, "")
    assert main(["--model", str(encoder_path), "embed", TEXTS[0], "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert np.asarray(result["embedding"], np.float32).tobytes() == engine.embed(TEXTS[0]).vector.tobytes()
    assert result["tokens"] == len(engine.embedding_tokens(TEXTS[0]))
    assert result["fingerprint"] == numerics.fingerprint(engine.embed(TEXTS[0]).vector)
    assert main(["--model", str(encoder_path), "embed", TEXTS[0], "--dimensions", "4"]) == 0
    captured = capsys.readouterr()
    assert len(captured.out.split()) == 4 and "4 dimensions" in captured.err
    assert main(["--model", str(encoder_path), "embed", TEXTS[0], "--dimensions", "999"]) == 2
    assert main(["--model", str(encoder_path), "inspect", str(encoder_path)]) == 0
    assert "bert" in capsys.readouterr().out


def test_encoders_train_with_encoder_objectives(encoder_path):
    from etalii_dllm.lora import LoraConfig, target_weights
    from etalii_dllm.training import RunConfig
    from etalii_dllm.training.trainer import FineTuner

    config = ModelFile(encoder_path).config
    assert target_weights(config, LoraConfig(4, 8.0, ("q", "v")))[:2] == [
        "layers.0.attention.q.weight",
        "layers.0.attention.v.weight",
    ]
    with pytest.raises(ValueError, match="an encoder trains with the embedding or classifier objective"):
        FineTuner(config, {}, None, RunConfig(1), base_fingerprint="", metadata={})  # type: ignore[arg-type]


def test_an_index_built_with_an_encoder(engine):
    from etalii_dllm.retrieval import build_index

    documents = [("fox.txt", "The quick brown fox jumps."), ("greeting.txt", "Hello, world! How are you?")]
    index = build_index(engine, documents, chunk_tokens=8)
    assert sorted({chunk.source for chunk in index.chunks}) == ["fox.txt", "greeting.txt"]
    hits = index.search(engine, "hello world", 2)  # a random tiny model ranks arbitrarily, but always the same
    assert [(h.chunk.source, h.score) for h in hits] == [
        (h.chunk.source, h.score) for h in index.search(engine, "hello world", 2)
    ]
