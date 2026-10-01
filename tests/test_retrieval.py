"""Retrieval: chunking, reading documents, reproducible index files, exact search with a total order, grounding chats
(engine, CLI, server and the MCP ``search_documents`` tool) and the errors."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_embedding_models import embedding_path  # noqa: F401 - fixture
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm import engine as engine_module
from etalii_dllm import mcp_server, retrieval
from etalii_dllm.chat import ChatMessage
from etalii_dllm.cli import main
from etalii_dllm.engine import ChatRequest, DllmEngine, default_engine
from etalii_dllm.retrieval import Chunk, Index, RetrievalError, Retriever, build_index, chunk_text, read_documents

DOCUMENTS = [
    ("animals.md", "The cat sat on the mat.\n\nDogs bark at the mailman every morning."),
    ("cities.txt", "Paris is the capital of France.\n\n\nBerlin is the capital of Germany."),
]


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture(scope="module")
def embedder(embedding_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(embedding_path)


@pytest.fixture(scope="module")
def index(embedder, embedding_path) -> Index:  # noqa: F811
    return build_index(embedder, DOCUMENTS, chunk_tokens=30, model_path=embedding_path)


@pytest.fixture
def index_file(tmp_path, index):
    index.save(tmp_path / "docs.index")
    return tmp_path / "docs.index"


@pytest.fixture
def clean_environment(monkeypatch):
    for name in ("MODEL", "INDEX", "INDEX_TOP", "EMBEDDING_MODEL"):
        monkeypatch.delenv(getattr(engine_module, f"{name}_ENVIRONMENT_VARIABLE"), raising=False)
    default_engine.cache_clear()
    yield monkeypatch
    for name in ("MODEL", "INDEX", "INDEX_TOP", "EMBEDDING_MODEL"):  # the CLI sets them directly
        os.environ.pop(getattr(engine_module, f"{name}_ENVIRONMENT_VARIABLE"), None)
    default_engine.cache_clear()


def test_chunking():
    text = "one two three\n\nfour five\n \nsix seven eight nine ten eleven\n\n  \n"
    spans = chunk_text(text, words, 5)
    assert [text[a:b] for a, b in spans] == ["one two three\n\nfour five", "six seven eight nine ten", "eleven"]
    assert [text[a:b] for a, b in chunk_text(text, words, 100)] == [text.strip()]
    long = "a verylongword b"
    assert [long[a:b] for a, b in chunk_text(long, len, 4)] == ["a", "verylongword", "b"]
    # Only ASCII white space separates: a no-break space keeps words together.
    assert len(chunk_text("x\N{NO-BREAK SPACE}y z", words, 1)) == 2
    assert chunk_text("", words) == [] and chunk_text(" \n\n ", words) == []
    with pytest.raises(ValueError):
        chunk_text("x", words, 0)


def test_read_documents(tmp_path):
    (tmp_path / "docs" / "sub").mkdir(parents=True)
    (tmp_path / "docs" / "b.md").write_bytes(b"line one\r\nline two\rthree")
    (tmp_path / "docs" / "sub" / "a.txt").write_text("nested", encoding="utf-8")
    (tmp_path / "docs" / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "single.rst").write_text("single", encoding="utf-8")
    documents = read_documents([tmp_path / "docs", tmp_path / "single.rst"])
    assert documents == [("b.md", "line one\nline two\nthree"), ("single.rst", "single"), ("sub/a.txt", "nested")]
    with pytest.raises(FileNotFoundError):
        read_documents([tmp_path / "missing"])
    with pytest.raises(ValueError, match="two documents"):
        read_documents([tmp_path / "single.rst", tmp_path / "single.rst"])


def test_index_files_are_reproducible(tmp_path, embedder, embedding_path, index):  # noqa: F811
    assert [c.source for c in index.chunks] == ["animals.md", "animals.md", "cities.txt", "cities.txt"]
    assert index.chunks[3] == Chunk("cities.txt", 34, 67, "Berlin is the capital of Germany.")
    assert index.vectors.shape == (4, embedder.model.config.hidden_size) and index.dimensions == 16
    np.testing.assert_array_equal(index.vectors[0], embedder.embed(index.chunks[0].text, input_type="document").vector)
    again = build_index(embedder, DOCUMENTS, chunk_tokens=30, model_path=embedding_path)
    index.save(tmp_path / "a.index")
    again.save(tmp_path / "b.index")
    assert (tmp_path / "a.index").read_bytes() == (tmp_path / "b.index").read_bytes()
    loaded = Index.load(tmp_path / "a.index")
    assert loaded.fingerprint == index.fingerprint and loaded.chunks == index.chunks
    assert loaded.model == {"fingerprint": embedder.model.weights_fingerprint, "id": "example/tiny-embedding",
                            "path": str(embedding_path)}  # fmt: skip
    other = build_index(embedder, DOCUMENTS, chunk_tokens=100)
    assert len(other.chunks) == 2 and other.fingerprint != index.fingerprint
    empty = build_index(embedder, [])
    empty.save(tmp_path / "empty.index")
    assert Index.load(tmp_path / "empty.index").chunks == [] and empty.search(embedder, "cats") == []


def test_bad_index_files(tmp_path, index):
    from etalii_dllm.importing.safetensors import write_safetensors

    with pytest.raises(RetrievalError, match="fingerprint"):
        Index(index.chunks, index.vectors * 2, index.model, index.chunk_tokens, index.fingerprint)
    with pytest.raises(RetrievalError, match="one vector per chunk"):
        Index(index.chunks[:1], index.vectors, index.model)
    (tmp_path / "junk.index").write_bytes(b"junk")
    with pytest.raises(RetrievalError, match="not a dllm index"):
        Index.load(tmp_path / "junk.index")
    write_safetensors(tmp_path / "other.index", {"x": np.zeros(2, np.float32)}, {"format": "other"})
    with pytest.raises(RetrievalError, match="not a dllm index"):
        Index.load(tmp_path / "other.index")
    write_safetensors(
        tmp_path / "v2.index",
        {"vectors": index.vectors},
        {"format": "dllm-index", "index": json.dumps({"version": 99})},
    )
    with pytest.raises(RetrievalError, match="version 99"):
        Index.load(tmp_path / "v2.index")


def test_search(embedder, model_path, index):  # noqa: F811
    hits = index.search(embedder, "Which city is the capital of France?", top=4)
    assert [h.rank for h in hits] == [1, 2, 3, 4] and sorted(h.index for h in hits) == [0, 1, 2, 3]
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    expected = embedder.embed("Which city is the capital of France?", input_type="query").vector
    from etalii_dllm.numerics import cosine_similarity

    assert hits[0].score == float(cosine_similarity(index.vectors, expected)[hits[0].index])
    assert [h.index for h in index.search(embedder, "Which city is the capital of France?", top=2)] == [
        h.index for h in hits[:2]
    ]
    # Ties go to the earlier chunk.
    twins = Index([index.chunks[0], index.chunks[0]], np.stack([index.vectors[0]] * 2), index.model)
    assert [h.index for h in twins.search(embedder, "cat")] == [0, 1]
    with pytest.raises(ValueError, match="top"):
        index.search(embedder, "cat", top=0)
    with pytest.raises(ValueError, match="empty"):
        index.search(embedder, "  ")
    with pytest.raises(RetrievalError, match="built with"):
        index.search(DllmEngine.from_model_file(model_path), "cat")


def test_grounding(embedder, index):
    retriever = Retriever(index, embedder, top=2)
    question = [ChatMessage("user", "Tell me about Paris.")]
    grounded, hits = retriever.ground(question)
    assert len(hits) == 2 and [m.role for m in grounded] == ["system", "user"]
    assert grounded[0].content.startswith(retrieval.GROUNDING_INSTRUCTION)
    assert f"[1] ({hits[0].chunk.source})\n{hits[0].chunk.text}" in grounded[0].content
    with_system = [ChatMessage("system", "Be brief."), *question]
    merged, _ = retriever.ground(with_system)
    assert merged[0].content.startswith("Be brief.\n\n" + retrieval.GROUNDING_INSTRUCTION) and len(merged) == 2
    assert retriever.ground([ChatMessage("system", "")])[0] == [ChatMessage("system", "")]
    empty = Retriever(build_index(embedder, []), embedder)
    assert empty.ground(question) == (question, [])
    assert retriever.fingerprint != Retriever(index, embedder, top=3).fingerprint
    with pytest.raises(ValueError):
        Retriever(index, embedder, top=0)


def test_grounded_engine(tmp_path, model_path, embedding_path, index, index_file):  # noqa: F811
    plain = DllmEngine.from_model_file(model_path)
    grounded = DllmEngine.from_model_file(model_path, index=index_file, index_top=1)
    assert grounded.retriever is not None and grounded.retriever.top == 1
    assert grounded.system_fingerprint != plain.system_fingerprint
    assert (
        grounded.system_fingerprint
        == DllmEngine.from_model_file(model_path, index=index_file, index_top=1).system_fingerprint
    )
    request = ChatRequest([ChatMessage("user", "Where is Paris?")], 6)
    first = grounded.chat_completion(request)
    assert first.content == grounded.chat_completion(request).content
    messages, _ = grounded.retriever.ground(request.messages)
    expected = plain.chat_completion(ChatRequest(messages, 6))
    assert first.content == expected.content
    # An index without a recorded model needs --embedding-model.
    unnamed = Index(index.chunks, index.vectors, {**index.model, "path": None}, index.chunk_tokens)
    unnamed.save(tmp_path / "unnamed.index")
    with pytest.raises(RetrievalError, match="embedding model"):
        DllmEngine.from_model_file(model_path, index=tmp_path / "unnamed.index")
    named = DllmEngine.from_model_file(model_path, index=tmp_path / "unnamed.index", embedding_model=embedding_path)
    assert named.retriever is not None


def test_environment_and_front_ends(model_path, embedding_path, index_file, clean_environment, capsys):  # noqa: F811
    monkeypatch = clean_environment
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(model_path))
    monkeypatch.setenv(engine_module.INDEX_ENVIRONMENT_VARIABLE, str(index_file))
    monkeypatch.setenv(engine_module.INDEX_TOP_ENVIRONMENT_VARIABLE, "2")
    engine = default_engine()
    assert engine.retriever is not None and engine.retriever.top == 2
    hits = json.loads(mcp_server.search_documents("Paris", top=1))
    assert len(hits) == 1 and set(hits[0]) == {"rank", "score", "source", "start", "end", "text"}
    client = TestClient(__import__("etalii_dllm.server.app", fromlist=["app"]).app)
    body = {"messages": [{"role": "user", "content": "Where is Paris?"}], "max_tokens": 4}
    response = client.post("/v1/chat/completions", json=body).json()
    assert response["system_fingerprint"] == engine.system_fingerprint
    monkeypatch.setenv(engine_module.INDEX_TOP_ENVIRONMENT_VARIABLE, "zero")
    with pytest.raises(ValueError, match="DLLM_INDEX_TOP"):
        engine_module.configured_index_top()
    monkeypatch.delenv(engine_module.INDEX_TOP_ENVIRONMENT_VARIABLE)
    assert engine_module.configured_index_top() == retrieval.DEFAULT_TOP
    monkeypatch.delenv(engine_module.INDEX_ENVIRONMENT_VARIABLE)
    default_engine.cache_clear()
    with pytest.raises(ValueError, match="no document index"):
        mcp_server.search_documents("Paris")
    # The CLI flags set the same variables.
    assert main(["--model", str(model_path), "--index", str(index_file), "--index-top", "1", "info"]) == 0
    assert capsys.readouterr().out and default_engine().retriever is not None
    assert main(["--embedding-model", str(embedding_path), "info"]) == 0
    assert engine_module.os.environ[engine_module.EMBEDDING_MODEL_ENVIRONMENT_VARIABLE] == str(embedding_path)


def test_index_commands(tmp_path, embedding_path, model_path, clean_environment, capsys):  # noqa: F811
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in DOCUMENTS:
        (docs / name).write_text(text, encoding="utf-8")
    output = tmp_path / "docs.index"
    model = ["--model", str(embedding_path)]
    assert main([*model, "index", "build", str(docs), "-o", str(output), "--chunk-tokens", "30"]) == 0
    captured = capsys.readouterr()
    assert "4 chunks" in captured.out and "chunk      4  cities.txt:34" in captured.err
    assert Index.load(output).model["path"] == str(embedding_path.resolve())
    default_engine.cache_clear()
    # Search finds the embedding model through the index.
    assert main(["index", "search", str(output), "the capital of France", "--top", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("1. ") and "\n2. " in out
    assert main([*model, "index", "search", str(output), "cats", "--json", "--top", "1"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["rank"] == 1
    assert main(["--model", str(model_path), "index", "search", str(output), "cats"]) == 1
    assert "built with" in capsys.readouterr().err
    os.environ.pop(engine_module.MODEL_ENVIRONMENT_VARIABLE)  # set by the CLI, not by monkeypatch
    default_engine.cache_clear()
    assert main(["index", "search", str(tmp_path / "missing.index"), "cats"]) == 1
    assert capsys.readouterr().err.startswith("dllm index: ")
    assert main([*model, "index", "build", str(tmp_path / "nothing"), "-o", str(output)]) == 1
