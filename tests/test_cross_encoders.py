"""BERT cross-encoders (issues #342-#345): sentence-pair encoding with token types against ``tokenizers``, the
sequence-classification head against transformers' ``BertForSequenceClassification``, ``engine.classify``,
reranking with a cross-encoder in ``dllm rerank``, ``/v1/rerank`` and hybrid search, golden scores, the reference
implementation and the refusals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from model_fixtures import write_safetensors
from test_encoders import BERT_CONFIG, bert_weights

from etalii_dllm import numerics
from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import ModelImportError, import_model

tokenizers = pytest.importorskip("tokenizers")

QUERY = "how are you"
DOCUMENTS = ["Hello, world! How are you?", "The quick brown fox jumps over the lazy dog.", "naïve café façade"]
PAIRS = [(QUERY, document) for document in DOCUMENTS] + [("the lazy dog", DOCUMENTS[1] * 3)]
MAX_LENGTH = 24


def write_cross_encoder(
    directory: Path, labels: int = 1, *, activation: str | None = None, max_length: int | None = MAX_LENGTH
) -> dict:
    """A tiny ``BertForSequenceClassification`` checkpoint with a WordPiece tokenizer."""
    from test_wordpiece import bert_tokenizer

    reference, _ = bert_tokenizer()
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        **BERT_CONFIG,
        "architectures": ["BertForSequenceClassification"],
        "vocab_size": reference.get_vocab_size(),
        "id2label": {str(i): f"LABEL_{i}" for i in range(labels)},
        "label2id": {f"LABEL_{i}": i for i in range(labels)},
    }
    if activation is not None:
        config["sentence_transformers"] = {"activation_fn": activation}
    weights = bert_weights(config, seed=7)
    hidden = config["hidden_size"]
    weights["classifier.weight"] = (numerics.fill_gaussian(91, labels * hidden).reshape(labels, hidden) * 0.3).astype(
        np.float32
    )
    weights["classifier.bias"] = np.full(labels, 0.25, dtype=np.float32)
    stored = {("" if name.startswith("classifier") else "bert.") + name: ("F32", v) for name, v in weights.items()}
    write_safetensors(directory / "model.safetensors", stored, {"format": "pt"})
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "tokenizer.json").write_text(reference.to_str(), encoding="utf-8")
    tokenizer_config = {"cls_token": "[CLS]", "sep_token": "[SEP]", "unk_token": "[UNK]", "do_lower_case": True}
    if max_length is not None:
        tokenizer_config["model_max_length"] = max_length
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    (directory / "README.md").write_bytes(b"---\nlicense: apache-2.0\n---\n\n# Tiny cross-encoder\n")
    return config


@pytest.fixture(scope="module")
def cross_path(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("cross")
    write_cross_encoder(directory / "checkpoint")
    import_model(directory / "checkpoint", directory / "cross.dllm", repository="example/tiny-cross-encoder")
    return directory / "cross.dllm"


@pytest.fixture(scope="module")
def cross(cross_path) -> DllmEngine:
    return DllmEngine.from_model_file(cross_path)


def transformers_logits(directory: Path, pairs: list[tuple[str, str]], max_length: int = MAX_LENGTH) -> np.ndarray:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(directory).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(directory)
    rows = []
    for query, document in pairs:
        batch = tokenizer(query, document, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            rows.append(model(**batch).logits[0].double().numpy())
    return np.stack(rows)


def test_classifier_matches_transformers(tmp_path, cross):
    write_cross_encoder(tmp_path)
    expected = transformers_logits(tmp_path, PAIRS)
    actual = np.array([cross.classify(query, document).logits for query, document in PAIRS])
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_classification_tokens_match_transformers(tmp_path, cross):
    transformers = pytest.importorskip("transformers")
    write_cross_encoder(tmp_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tmp_path)
    for query, document in PAIRS:
        batch = tokenizer(query, document, truncation=True, max_length=MAX_LENGTH)
        assert cross.classification_tokens(query, document) == (batch["input_ids"], batch["token_type_ids"])
    single = tokenizer(DOCUMENTS[1] * 3, truncation=True, max_length=MAX_LENGTH)["input_ids"]
    assert cross.classification_tokens(DOCUMENTS[1] * 3) == (single, [0] * len(single))


# Pair encoding (#342)


@pytest.mark.parametrize("processor", ["template", "bert", "none"])
def test_pair_encoding_matches_tokenizers(processor):
    from test_wordpiece import TEXTS, bert_tokenizer
    from tokenizers import processors

    from etalii_dllm.bpe import BpeTokenizer

    reference, _ = bert_tokenizer()
    if processor == "bert":
        reference.post_processor = processors.BertProcessing(("[SEP]", 3), ("[CLS]", 2))
    elif processor == "none":
        reference.post_processor = None
    ours = BpeTokenizer(json.loads(reference.to_str()))
    texts = [*TEXTS[:9], "[CLS] added [SEP] tokens [MASK]", " ".join(["hello"] * 12)]
    for first in texts:
        for second in texts:
            reference.no_truncation()
            expected = reference.encode(first, second)
            assert ours.encode_pair(first, second) == (expected.ids, expected.type_ids)
            for limit in (3, 4, 5, 6, 7, 9, 12, 20):
                reference.enable_truncation(limit, strategy="longest_first")
                expected = reference.encode(first, second)
                assert ours.encode_pair(first, second, max_tokens=limit) == (expected.ids, expected.type_ids), limit


def test_truncation_follows_tokenizers_word_limit():
    """tokenizers tokenizes each text of a truncated pair only up to the word that reaches max_length tokens, so a
    pair of long texts keeps fewer tokens of the first than the plain longest-first formula would."""
    from test_wordpiece import bert_tokenizer

    reference, ours = bert_tokenizer()
    first, second = " ".join(["hello"] * 8), " ".join(["world"] * 6)
    reference.enable_truncation(6, strategy="longest_first")
    expected = reference.encode(first, second)
    assert ours.encode_pair(first, second, max_tokens=6) == (expected.ids, expected.type_ids)
    assert expected.tokens == ["[CLS]", "hello", "[SEP]", "world", "world", "[SEP]"]
    with pytest.raises(ValueError, match="no room"):
        ours.encode_pair(first, second, max_tokens=2)


def test_pair_templates_are_checked():
    from test_wordpiece import bert_tokenizer

    from etalii_dllm.bpe import BpeTokenizer, TokenizerError

    reference, _ = bert_tokenizer()
    spec = json.loads(reference.to_str())
    del spec["post_processor"]["pair"]
    with pytest.raises(TokenizerError, match="no template for a pair"):
        BpeTokenizer(spec).encode_pair("a", "b")
    twice = {"type": "Sequence", "processors": [spec["post_processor"], spec["post_processor"]]}
    with pytest.raises(TokenizerError, match="more than one template"):
        BpeTokenizer({**spec, "post_processor": twice})
    once = {"type": "Sequence", "processors": [{"type": "ByteLevel"}, json.loads(reference.to_str())["post_processor"]]}
    assert BpeTokenizer({**spec, "post_processor": once}).encode("hello", add_special_tokens=True)[0] == 2


# The classification head (#343)


def test_import_records_the_classifier(cross_path, cross):
    from etalii_dllm.modelfile import ModelFile

    file = ModelFile(cross_path)
    assert file.config.classifier_labels == 1 and file.embedding is None
    assert file.classifier == {"labels": ["LABEL_0"], "activation": "none", "max_tokens": MAX_LENGTH}
    assert {"pooler.weight", "classifier.weight"} <= set(file.tensors)
    assert cross.classifier == file.classifier
    verdict = cross.classify(QUERY, DOCUMENTS[0])
    assert verdict.labels == ("LABEL_0",) and verdict.scores == verdict.logits
    assert verdict.tokens == len(cross.classification_tokens(QUERY, DOCUMENTS[0])[0])


def test_scores_are_golden_and_thread_invariant(cross, cross_path):
    from golden_values import CROSS_ENCODER_SCORES

    scores = np.array([cross.classify(q, d).logits for q, d in PAIRS], dtype=np.float32)
    assert numerics.fingerprint(scores) == CROSS_ENCODER_SCORES
    for threads in (1, 3):
        numerics.set_threads(threads)
        try:
            again = np.array([cross.classify(q, d).logits for q, d in PAIRS], dtype=np.float32)
        finally:
            numerics.set_threads(0)
        assert again.tobytes() == scores.tobytes()
    quantized = DllmEngine.from_model_file(cross_path, quantize="q8_0")
    close = np.array([quantized.classify(q, d).logits for q, d in PAIRS], dtype=np.float32)
    np.testing.assert_allclose(close, scores, atol=0.05)
    assert quantized.system_fingerprint != cross.system_fingerprint


def test_sigmoid_activation_and_several_labels(tmp_path):
    from etalii_dllm.reranking import Reranker

    write_cross_encoder(tmp_path / "sigmoid", activation="torch.nn.modules.activation.Sigmoid", max_length=None)
    import_model(tmp_path / "sigmoid", tmp_path / "sigmoid.dllm")
    engine = DllmEngine.from_model_file(tmp_path / "sigmoid.dllm")
    assert engine.classifier is not None and engine.classifier["max_tokens"] == BERT_CONFIG["max_position_embeddings"]
    verdict = engine.classify(QUERY, DOCUMENTS[0])
    assert verdict.scores == (numerics.sigmoid(verdict.logits[0]),)
    assert Reranker(engine).judge(QUERY, DOCUMENTS[0]).score == verdict.scores[0]
    write_cross_encoder(tmp_path / "three", labels=3)
    import_model(tmp_path / "three", tmp_path / "three.dllm")
    three = DllmEngine.from_model_file(tmp_path / "three.dllm")
    assert len(three.classify("one text").logits) == 3 and three.classify("one text").labels[2] == "LABEL_2"
    with pytest.raises(ValueError, match="several labels"):
        Reranker(three)


def test_import_refusals(tmp_path):
    write_cross_encoder(tmp_path / "odd", activation="torch.nn.modules.activation.Tanh")
    with pytest.raises(ModelImportError, match="activation"):
        import_model(tmp_path / "odd", tmp_path / "odd.dllm")
    config = write_cross_encoder(tmp_path / "labels", labels=2)
    config["id2label"] = {"0": "only"}
    config["num_labels"] = 1
    (tmp_path / "labels" / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ModelImportError, match="2 labels"):
        import_model(tmp_path / "labels", tmp_path / "labels.dllm")


def test_classifier_refusals(cross, tmp_path):
    from etalii_dllm.architecture import TransformerConfig

    with pytest.raises(ValueError, match="cross-encoder, which scores pairs"):
        cross.embed("hello")
    with pytest.raises(ValueError, match="token types"):
        cross.model.hidden_states([2, 5, 3], [0, 2, 0])  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="for encoders only"):
        TransformerConfig.from_dict({**cross.model.config.to_dict(), "family": "llama", "type_vocabulary_size": 0})
    with pytest.raises(ValueError, match="must not be negative"):
        TransformerConfig.from_dict({**cross.model.config.to_dict(), "classifier_labels": -1})


def test_embedders_have_no_classifier(tmp_path):
    from test_encoders import write_bert_checkpoint

    write_bert_checkpoint(tmp_path / "checkpoint")
    import_model(tmp_path / "checkpoint", tmp_path / "encoder.dllm")
    embedder = DllmEngine.from_model_file(tmp_path / "encoder.dllm")
    assert embedder.classifier is None and embedder.model.config.classifier_labels == 0
    with pytest.raises(ValueError, match="no classification head"):
        embedder.classify("hello")
    with pytest.raises(ValueError, match="no classification head"):
        embedder.model.classify([2, 3])  # type: ignore[attr-defined]


# Reranking (#344)


def test_reranker_scores_pairs(cross):
    from etalii_dllm.reranking import Reranker

    reranker = Reranker(cross)
    assert reranker.cross_encoder
    expected = [cross.classify(QUERY, d).scores[0] for d in DOCUMENTS]
    order = sorted(range(len(DOCUMENTS)), key=lambda i: (-expected[i], i))
    assert reranker.rerank(QUERY, DOCUMENTS) == [(i, expected[i]) for i in order]
    assert reranker.rerank(QUERY, DOCUMENTS, "an instruction a cross-encoder ignores") == reranker.rerank(
        QUERY, DOCUMENTS
    )
    assert reranker.fingerprint == Reranker(cross, "other").fingerprint  # the instruction does not apply
    with pytest.raises(ValueError, match="query"):
        reranker.rerank(" ", DOCUMENTS)


def test_rerank_command_and_endpoint(cross_path, cross, monkeypatch, capsys):
    from fastapi.testclient import TestClient

    from etalii_dllm import engine as engine_module
    from etalii_dllm.cli import main
    from etalii_dllm.engine import default_engine
    from etalii_dllm.reranking import Reranker

    expected = Reranker(cross).judgements(QUERY, DOCUMENTS)
    # The CLI sets the model variable directly; setting it here first makes the test's undo remove it again.
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(cross_path))
    default_engine.cache_clear()
    try:
        assert main(["--model", str(cross_path), "rerank", QUERY, *DOCUMENTS, "--json"]) == 0
        rows = json.loads(capsys.readouterr().out)
        assert [(row["index"], row["score"]) for row in rows] == [(i, j.score) for i, j in expected]
        client = TestClient(__import__("etalii_dllm.server.app", fromlist=["app"]).app)
        body = client.post("/v1/rerank", json={"query": QUERY, "documents": DOCUMENTS}).json()
        assert [(r["index"], r["relevance_score"]) for r in body["results"]] == [(i, j.score) for i, j in expected]
        assert body["usage"]["total_tokens"] == sum(j.tokens for _, j in expected)
    finally:
        default_engine.cache_clear()


def test_hybrid_search_reranked_by_a_cross_encoder(tmp_path, cross):
    from test_encoders import write_bert_checkpoint

    from etalii_dllm import retrieval
    from etalii_dllm.reranking import Reranker
    from etalii_dllm.retrieval import Retriever, build_index

    write_bert_checkpoint(tmp_path / "checkpoint")
    import_model(tmp_path / "checkpoint", tmp_path / "encoder.dllm")
    embedder = DllmEngine.from_model_file(tmp_path / "encoder.dllm")
    documents = [("a.md", DOCUMENTS[0]), ("b.md", DOCUMENTS[1]), ("c.md", DOCUMENTS[2])]
    index = build_index(embedder, documents, chunk_tokens=8, model_path=tmp_path / "encoder.dllm")
    reranking = Retriever(index, embedder, 2, "hybrid", Reranker(cross))
    hybrid = Retriever(index, embedder, 2, "hybrid")
    candidates = hybrid.search(QUERY, retrieval.RERANK_DEPTH * 2)
    expected = Reranker(cross).rerank(QUERY, [h.chunk.text for h in candidates])[:2]
    assert [(h.index, h.score) for h in reranking.search(QUERY)] == [(candidates[i].index, s) for i, s in expected]
    assert reranking.fingerprint != hybrid.fingerprint


# Verification (#345)


def test_reference_implementation_and_verify(cross):
    from etalii_dllm import reference, verify

    twin = reference.ReferenceEncoder.from_engine(cross)
    for query, document in PAIRS:
        tokens, types = cross.classification_tokens(query, document)
        mine = np.asarray(cross.classify(query, document).logits, dtype=np.float32)
        assert mine.tobytes() == twin.classify(tokens, types).tobytes()
    report = verify.run(cross)
    assert "scores" in report.parts and "embeddings" not in report.parts and "logits" not in report.parts
    check = verify.check_reference(cross)
    assert check.equal and set(check.results) == {"states", "scores"}
