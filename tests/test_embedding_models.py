"""Embedding models imported from sentence-transformers: the pooling settings in ``model.dllm``, unprefixed
checkpoints without an LM head, last-token pooling with the tokenizer's special tokens, query prompts
(``input_type``), the hub download of the pooling module and the import errors."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from model_fixtures import TINY_LLAMA_CONFIG, to_bf16_bits, write_hf_checkpoint, write_safetensors

from etalii_dllm import engine as engine_module
from etalii_dllm.engine import DllmEngine, default_engine
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.modelfile import ModelFile

tokenizers = pytest.importorskip("tokenizers")

QUERY = "Instruct: find passages about cats\nQuery:"
MODULES = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
]


def write_embedding_checkpoint(directory: Path, pooling: dict | None = None, modules: list | None = None) -> None:
    """A sentence-transformers style checkpoint: tensor names without ``model.``, no LM head, a tokenizer that
    appends ``<|endoftext|>``, a pooling module and query/document prompts."""
    from test_bpe import smollm2_style
    from tokenizers import processors

    reference = smollm2_style()
    reference.post_processor = processors.TemplateProcessing(
        single="$A <|endoftext|>", special_tokens=[("<|endoftext|>", reference.token_to_id("<|endoftext|>"))]
    )
    config = {**TINY_LLAMA_CONFIG, "vocab_size": reference.get_vocab_size(), "tie_word_embeddings": False}
    weights = write_hf_checkpoint(directory, config, tokenizer_json=json.loads(reference.to_str()))
    write_safetensors(
        directory / "model.safetensors",
        {
            name.removeprefix("model."): ("BF16", to_bf16_bits(values))
            for name, values in weights.items()
            if name != "lm_head.weight"
        },
        {"format": "pt"},
    )
    tokenizer_config = {"eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>", "padding_side": "left"}
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    (directory / "modules.json").write_text(json.dumps(modules or MODULES), encoding="utf-8")
    (directory / "1_Pooling").mkdir(exist_ok=True)
    settings = pooling or {"word_embedding_dimension": 16, "pooling_mode_lasttoken": True, "include_prompt": True}
    (directory / "1_Pooling" / "config.json").write_text(json.dumps(settings), encoding="utf-8")
    prompts = {"prompts": {"query": QUERY, "document": ""}, "default_prompt_name": None}
    (directory / "config_sentence_transformers.json").write_text(json.dumps(prompts), encoding="utf-8")


@pytest.fixture(scope="module")
def embedding_path(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("embedding")
    write_embedding_checkpoint(directory / "checkpoint")
    import_model(directory / "checkpoint", directory / "embed.dllm", repository="example/tiny-embedding")
    return directory / "embed.dllm"


@pytest.fixture(scope="module")
def engine(embedding_path) -> DllmEngine:
    return DllmEngine.from_model_file(embedding_path)


def _normalised(vector: np.ndarray) -> np.ndarray:
    return vector / np.sqrt(np.sum(vector.astype(np.float64) ** 2))


def test_import_records_the_pooling(embedding_path):
    file = ModelFile(embedding_path)
    assert file.embedding == {
        "pooling": "last_token",
        "normalize": True,
        "prompts": {"document": "", "query": QUERY},
        "default_prompt_name": None,
    }
    assert file.config.tie_word_embeddings  # no LM head in the checkpoint: the embedding is reused


def test_last_token_pooling_with_prompts(engine):
    tokens = engine.tokenizer.encode("a cat", add_special_tokens=True)
    assert tokens[-1] == engine.tokenizer.encode("<|endoftext|>")[0]
    states = engine.model.hidden_states(tokens)
    document = engine.embed("a cat")
    assert document.tokens == len(tokens)
    np.testing.assert_allclose(document.vector, _normalised(states[-1]), rtol=1e-6, atol=1e-7)
    assert engine.embed("a cat", input_type="document").vector.tobytes() == document.vector.tobytes()
    query = engine.embed("a cat", input_type="query")
    expected = engine.model.hidden_states(engine.tokenizer.encode(QUERY + "a cat", add_special_tokens=True))[-1]
    np.testing.assert_allclose(query.vector, _normalised(expected), rtol=1e-6, atol=1e-7)
    assert query.vector.tobytes() == engine.embed("a cat", input_type="query").vector.tobytes()
    with pytest.raises(ValueError, match="input_type"):
        engine.embed("a cat", input_type="passage")
    assert engine.embed(tokens).vector.tobytes() == document.vector.tobytes()  # token ids are used as given
    assert engine.embed("a cat", dimensions=4).vector.shape == (4,)


def test_hidden_states_match_the_causal_checkpoint(tmp_path, engine):
    """Dropping the ``model.`` prefix and the LM head changes nothing about the decoder itself."""
    write_embedding_checkpoint(tmp_path / "embedding")
    causal = tmp_path / "causal"
    causal.mkdir()
    weights = {}
    from etalii_dllm.importing.safetensors import SafetensorsFile

    file = SafetensorsFile(tmp_path / "embedding" / "model.safetensors")
    for name in file.names():
        weights["model." + name] = ("BF16", to_bf16_bits(file[name].to_float32()))
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "README.md"):
        (causal / name).write_bytes((tmp_path / "embedding" / name).read_bytes())
    config = json.loads((causal / "config.json").read_text(encoding="utf-8"))
    (causal / "config.json").write_text(json.dumps({**config, "tie_word_embeddings": True}), encoding="utf-8")
    write_safetensors(causal / "model.safetensors", weights, {"format": "pt"})
    import_model(causal, tmp_path / "causal.dllm")
    other = DllmEngine.from_model_file(tmp_path / "causal.dllm")
    assert other.embedding is None
    tokens = engine.tokenizer.encode("hello there", add_special_tokens=True)
    assert other.model.hidden_states(tokens).tobytes() == engine.model.hidden_states(tokens).tobytes()


def test_mean_pooling_without_normalisation(tmp_path):
    modules = [m for m in MODULES if "Normalize" not in m["type"]]
    pooling = {"pooling_mode_mean_tokens": True, "pooling_mode_lasttoken": False}
    write_embedding_checkpoint(tmp_path / "checkpoint", pooling, modules)
    import_model(tmp_path / "checkpoint", tmp_path / "mean.dllm")
    engine = DllmEngine.from_model_file(tmp_path / "mean.dllm")
    assert engine.embedding is not None and engine.embedding["pooling"] == "mean"
    tokens = engine.tokenizer.encode("a cat", add_special_tokens=True)
    states = engine.model.hidden_states(tokens).astype(np.float64)
    np.testing.assert_allclose(engine.embed("a cat").vector, states.mean(axis=0), rtol=1e-5, atol=1e-6)
    truncated = engine.embed("a cat", dimensions=3).vector  # truncation always normalises again
    assert float(np.sum(truncated.astype(np.float64) ** 2)) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    ("pooling", "modules", "message"),
    [
        ({"pooling_mode_max_tokens": True}, None, "only mean, CLS or last-token"),
        ({"pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": True}, None, "only mean, CLS or last-token"),
        ({"pooling_mode_mean_tokens": True, "include_prompt": False}, None, "leaves out the prompt"),
        (None, [MODULES[0]], "no pooling module"),
    ],
)
def test_unsupported_pooling_is_refused(tmp_path, pooling, modules, message):
    write_embedding_checkpoint(tmp_path / "checkpoint", pooling, modules)
    with pytest.raises(ModelImportError, match=message):
        import_model(tmp_path / "checkpoint", tmp_path / "out.dllm")


def test_hub_downloads_the_pooling_module(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    write_embedding_checkpoint(checkpoint)
    commit = "0123456789abcdef0123456789abcdef01234567"
    files = [
        *sorted(p.name for p in checkpoint.iterdir() if p.is_file()),
        "1_Pooling/config.json",
        "onnx/model.onnx",
    ]

    def opener(url: str):
        if "/api/models/" in url:
            return io.BytesIO(json.dumps({"sha": commit, "siblings": [{"rfilename": f} for f in files]}).encode())
        return io.BytesIO((checkpoint / url.split(f"/resolve/{commit}/", 1)[1]).read_bytes())

    import_model("hf:org/embed", tmp_path / "out.dllm", cache=tmp_path / "cache", opener=opener)
    assert (tmp_path / "cache" / "org" / "embed" / commit / "1_Pooling" / "config.json").exists()
    assert not (tmp_path / "cache" / "org" / "embed" / commit / "onnx").exists()
    assert ModelFile(tmp_path / "out.dllm").embedding["pooling"] == "last_token"


def test_server_input_type(embedding_path, engine, monkeypatch):
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(embedding_path))
    default_engine.cache_clear()
    from etalii_dllm.server.app import app

    try:
        client = TestClient(app)
        response = client.post("/v1/embeddings", json={"input": "a cat", "input_type": "query"}).json()
        expected = engine.embed("a cat", input_type="query").vector
        assert np.asarray(response["data"][0]["embedding"], np.float32).tobytes() == expected.tobytes()
        assert client.post("/v1/embeddings", json={"input": "a cat", "input_type": "x"}).status_code == 400
    finally:
        default_engine.cache_clear()
