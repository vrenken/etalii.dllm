"""Phase 26: exact hybrid search and reranking. BM25 equals a float64 transcription of the formula, hybrid ranking is
reciprocal rank fusion with a total order, the language-model reranker scores sigmoid(yes - no), and every ranking
has a golden fingerprint, so it is the same on every machine."""

from __future__ import annotations

import hashlib
import json
import math
import os

import pytest
from fastapi.testclient import TestClient
from golden_values import SEARCH_FINGERPRINTS
from test_embedding_models import embedding_path  # noqa: F401 - fixture
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm import engine as engine_module
from etalii_dllm import mcp_server, retrieval
from etalii_dllm.chat import ChatMessage
from etalii_dllm.cli import main
from etalii_dllm.engine import DllmEngine, default_engine
from etalii_dllm.reranking import JUDGE_SYSTEM, Reranker
from etalii_dllm.retrieval import Chunk, Index, LexicalStatistics, Retriever, build_index, terms

DOCUMENTS = [
    ("animals.md", "The cat sat on the mat.\n\nDogs bark at the mailman every morning."),
    ("cities.txt", "Paris is the capital of France.\n\n\nBerlin is the capital of Germany."),
    ("more.md", "A cat and another CAT chase the dog.\n\nThe capital city has many cats."),
]
VARIABLES = ("MODEL", "INDEX", "INDEX_TOP", "EMBEDDING_MODEL", "INDEX_MODE", "RERANK_MODEL")


def ranking_fingerprint(hits) -> str:
    rows = [[hit.index, float(hit.score).hex()] for hit in hits]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


@pytest.fixture(scope="module")
def embedder(embedding_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(embedding_path)


@pytest.fixture(scope="module")
def index(embedder, embedding_path) -> Index:  # noqa: F811
    return build_index(embedder, DOCUMENTS, chunk_tokens=30, model_path=embedding_path)


@pytest.fixture(scope="module")
def chat_engine(model_path) -> DllmEngine:  # noqa: F811
    return DllmEngine.from_model_file(model_path)


@pytest.fixture
def clean_environment(monkeypatch):
    for name in VARIABLES:
        monkeypatch.delenv(getattr(engine_module, f"{name}_ENVIRONMENT_VARIABLE"), raising=False)
    default_engine.cache_clear()
    yield monkeypatch
    for name in VARIABLES:  # the CLI sets them directly
        os.environ.pop(getattr(engine_module, f"{name}_ENVIRONMENT_VARIABLE"), None)
    default_engine.cache_clear()


# Lexical search


def test_terms_are_pinned_unicode_words():
    assert terms("The CAT's  café, n°5 — Straße!") == ["the", "cat", "s", "café", "n", "5", "straße"]
    assert terms("\ufb01ne \uff21\uff22\uff23") == ["fine", "abc"]  # NFKC: ligature and full-width letters
    assert terms("naïve") == terms("naïve") == ["naïve"]  # combining marks join the word
    assert terms("...") == [] and terms("") == []


def test_bm25_equals_the_formula():
    texts = ["the cat sat", "cat cat dog", "a dog", "nothing here at all"]
    statistics = LexicalStatistics(texts)
    lengths = [3, 3, 2, 4]
    average = sum(lengths) / 4
    frequencies = {"cat": 2, "dog": 2}

    def expected(text: str, length: int, query_terms: list[str]) -> float:
        total = 0.0
        for term in query_terms:
            tf = text.split().count(term)
            if tf:
                df = frequencies[term]
                idf = math.log(1 + (4 - df + 0.5) / (df + 0.5))
                total += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / average))
        return total

    scores = statistics.scores("Cat? dog, cat and unknown")
    for score, text, length in zip(scores, texts, lengths, strict=True):
        assert math.isclose(score, expected(text, length, ["cat", "dog"]), rel_tol=1e-14)
    assert scores[3] == 0.0
    assert statistics.idf("unknown") == pytest.approx(math.log(1 + 4.5 / 0.5))
    assert LexicalStatistics([]).scores("cat") == []


def test_lexical_and_hybrid_search(embedder, index):
    lexical = index.search(None, "cat capital", top=10, mode="lexical")
    assert all(hit.score > 0 for hit in lexical)
    assert {hit.chunk.source for hit in lexical} == {"animals.md", "cities.txt", "more.md"}
    scores = [hit.score for hit in lexical]
    assert scores == sorted(scores, reverse=True)
    assert index.search(embedder, "zebra", mode="lexical") == []
    assert index.lexical is index.lexical  # computed once

    dense = index.search(embedder, "cat capital", top=len(index.chunks))
    hybrid = index.search(embedder, "cat capital", top=len(index.chunks), mode="hybrid")
    dense_rank = {hit.index: hit.rank for hit in dense}
    lexical_rank = {hit.index: hit.rank for hit in index.search(None, "cat capital", top=99, mode="lexical")}
    for hit in hybrid:
        fused = 1.0 / (retrieval.RRF_K + dense_rank[hit.index])
        if hit.index in lexical_rank:
            fused += 1.0 / (retrieval.RRF_K + lexical_rank[hit.index])
        assert hit.score == fused
    assert [h.score for h in hybrid] == sorted((h.score for h in hybrid), reverse=True)

    # Ties go to the earlier chunk in every mode.
    twin = Chunk("a.md", 0, 7, "cat cat")
    twins = Index([twin, twin], index.vectors[:1].repeat(2, axis=0), index.model)
    for mode in retrieval.MODES:
        assert [h.index for h in twins.search(embedder, "cat", mode=mode)] == [0, 1]
    with pytest.raises(ValueError, match="search mode"):
        index.search(embedder, "cat", mode="fuzzy")
    with pytest.raises(ValueError, match="embedding model"):
        index.search(None, "cat", mode="hybrid")


def test_golden_rankings(embedder, index, chat_engine):
    assert ranking_fingerprint(index.search(None, "the capital cat", 6, "lexical")) == SEARCH_FINGERPRINTS["lexical"]
    assert ranking_fingerprint(index.search(embedder, "the capital cat", 6, "hybrid")) == SEARCH_FINGERPRINTS["hybrid"]
    reranked = Retriever(index, embedder, 2, "hybrid", Reranker(chat_engine)).search("Where is Paris?")
    assert ranking_fingerprint(reranked) == SEARCH_FINGERPRINTS["reranked"]


# Reranking


def test_reranker_is_sigmoid_of_yes_minus_no(chat_engine):
    reranker = Reranker(chat_engine)
    prompt = reranker.prompt("Where is Paris?", "Paris is in France.")
    assert JUDGE_SYSTEM in prompt and "<Query>: Where is Paris?\n<Document>: Paris is in France." in prompt
    assert prompt == chat_engine.render_chat(
        [
            ChatMessage("system", JUDGE_SYSTEM),
            ChatMessage(
                "user", f"<Instruct>: {reranker.instruction}\n<Query>: Where is Paris?\n<Document>: Paris is in France."
            ),
        ]
    )
    logits = chat_engine.model.forward(chat_engine.tokenizer.encode(prompt))
    judgement = reranker.judge("Where is Paris?", "Paris is in France.")
    from etalii_dllm import _kernels

    assert judgement.score == _kernels.sigmoid(float(logits[reranker.yes]) - float(logits[reranker.no]))
    assert judgement.tokens == len(chat_engine.tokenizer.encode(prompt))
    documents = ["Berlin is in Germany.", "Paris is in France.", "Berlin is in Germany."]
    ranked = reranker.rerank("Where is Paris?", documents)
    assert sorted(i for i, _ in ranked) == [0, 1, 2]
    assert [s for _, s in ranked] == sorted((s for _, s in ranked), reverse=True)
    assert [i for i, _ in ranked].index(0) < [i for i, _ in ranked].index(2)  # equal documents: earlier first
    assert ranked == reranker.rerank("Where is Paris?", documents)
    assert reranker.judge("q", "d", "another instruction").score != reranker.judge("q", "d").score
    assert reranker.fingerprint != Reranker(chat_engine, "another").fingerprint
    with pytest.raises(ValueError, match="empty"):
        reranker.rerank(" ", documents)


def test_reranker_needs_distinct_yes_and_no(chat_engine):
    class Same:
        def encode(self, text: str) -> list[int]:
            return [7]

    class Fake:
        tokenizer = Same()
        model = chat_engine.model
        system_fingerprint = "fp"

    with pytest.raises(ValueError, match="'yes' from 'no'"):
        Reranker(Fake())  # type: ignore[arg-type]


def test_retriever_modes_and_fingerprints(embedder, index, chat_engine):
    dense = Retriever(index, embedder, 2)
    assert dense.fingerprint == hashlib.sha256(f"{index.fingerprint}|2".encode()).hexdigest()  # unchanged
    hybrid = Retriever(index, embedder, 2, "hybrid")
    reranking = Retriever(index, embedder, 2, "hybrid", Reranker(chat_engine))
    assert len({dense.fingerprint, hybrid.fingerprint, reranking.fingerprint}) == 3
    hits = reranking.search("Where is Paris?")
    candidates = hybrid.search("Where is Paris?", retrieval.RERANK_DEPTH * 2)
    expected = Reranker(chat_engine).rerank("Where is Paris?", [h.chunk.text for h in candidates])[:2]
    assert [(h.index, h.score) for h in hits] == [(candidates[i].index, s) for i, s in expected]
    assert [h.rank for h in hits] == [1, 2]
    lexical = Retriever(index, None, 2, "lexical")
    assert [h.chunk.source for h in lexical.search("Berlin Germany")] == ["cities.txt"]
    with pytest.raises(ValueError, match="embedding model"):
        Retriever(index, None, 2, "dense")
    with pytest.raises(ValueError, match="search mode"):
        Retriever(index, embedder, 2, "fuzzy")


# Front ends


def test_engine_environment_and_mcp(tmp_path, model_path, embedding_path, index, clean_environment):  # noqa: F811
    index.save(tmp_path / "docs.index")
    monkeypatch = clean_environment
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(model_path))
    monkeypatch.setenv(engine_module.INDEX_ENVIRONMENT_VARIABLE, str(tmp_path / "docs.index"))
    monkeypatch.setenv(engine_module.INDEX_MODE_ENVIRONMENT_VARIABLE, "lexical")
    engine = default_engine()
    assert engine.retriever is not None and engine.retriever.mode == "lexical" and engine.retriever.embedder is None
    rows = json.loads(mcp_server.search_documents("Berlin Germany", top=3))
    assert [row["source"] for row in rows] == ["cities.txt"]
    monkeypatch.setenv(engine_module.RERANK_MODEL_ENVIRONMENT_VARIABLE, str(model_path))
    default_engine.cache_clear()
    reranked = default_engine()
    assert reranked.retriever is not None and reranked.retriever.reranker is not None
    assert reranked.system_fingerprint != engine.system_fingerprint
    monkeypatch.setenv(engine_module.INDEX_MODE_ENVIRONMENT_VARIABLE, "fuzzy")
    with pytest.raises(ValueError, match="DLLM_INDEX_MODE"):
        engine_module.configured_index_mode()
    monkeypatch.delenv(engine_module.INDEX_MODE_ENVIRONMENT_VARIABLE)
    assert engine_module.configured_index_mode() == "dense"


def test_rerank_endpoint(model_path, clean_environment):  # noqa: F811
    clean_environment.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(model_path))
    default_engine.cache_clear()
    client = TestClient(__import__("etalii_dllm.server.app", fromlist=["app"]).app)
    body = {"query": "Where is Paris?", "documents": ["Berlin is in Germany.", {"text": "Paris is in France."}]}
    first = client.post("/v1/rerank", json=body).json()
    assert first == client.post("/rerank", json=body).json()
    assert first["id"].startswith("rerank-") and first["model"] == default_engine().model.id
    expected = Reranker(default_engine()).judgements(
        "Where is Paris?", ["Berlin is in Germany.", "Paris is in France."]
    )
    assert [(r["index"], r["relevance_score"]) for r in first["results"]] == [(i, j.score) for i, j in expected]
    assert first["results"][0]["document"]["text"] in {"Berlin is in Germany.", "Paris is in France."}
    assert first["usage"]["total_tokens"] == sum(j.tokens for _, j in expected)
    short = client.post("/v1/rerank", json={**body, "top_n": 1, "return_documents": False}).json()
    assert len(short["results"]) == 1 and "document" not in short["results"][0] and short["id"] != first["id"]
    for bad, message in [({**body, "documents": []}, "must not be empty"), ({**body, "top_n": 0}, "top_n")]:
        response = client.post("/v1/rerank", json=bad)
        assert response.status_code == 400 and message in response.json()["error"]["message"]
    assert client.post("/v1/rerank", json={**body, "query": " "}).status_code == 400


def test_commands(tmp_path, model_path, embedding_path, index, clean_environment, capsys):  # noqa: F811
    index.save(tmp_path / "docs.index")
    model = ["--model", str(model_path)]
    assert main([*model, "rerank", "Where is Paris?", "Berlin is in Germany.", "Paris is in France.", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert sorted(row["index"] for row in rows) == [0, 1] and [row["rank"] for row in rows] == [1, 2]
    (tmp_path / "docs.txt").write_text("Berlin is in Germany.\n\nParis is in France.\n", encoding="utf-8")
    assert main([*model, "rerank", "Where is Paris?", "--file", str(tmp_path / "docs.txt"), "--top", "1"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("1. ") and "\n2. " not in out
    assert main([*model, "rerank", "Where is Paris?"]) == 1
    assert "give documents" in capsys.readouterr().err
    assert main([*model, "rerank", "q", "d", "--top", "0"]) == 1
    capsys.readouterr()

    os.environ.pop(engine_module.MODEL_ENVIRONMENT_VARIABLE)
    default_engine.cache_clear()
    search = ["index", "search", str(tmp_path / "docs.index"), "Berlin Germany", "--json"]
    assert main([*search, "--mode", "lexical"]) == 0
    assert [row["source"] for row in json.loads(capsys.readouterr().out)] == ["cities.txt"]
    os.environ.pop(engine_module.MODEL_ENVIRONMENT_VARIABLE)
    default_engine.cache_clear()
    assert main(["--rerank-model", str(model_path), *search, "--mode", "hybrid", "--top", "2"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 2 and all(0.0 < row["score"] < 1.0 for row in rows)
