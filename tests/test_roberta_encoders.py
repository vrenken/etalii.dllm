"""RoBERTa and XLM-RoBERTa encoders (issues #348-#350): positions counted from past the padding token, the imports
(sentence-transformers embedders with byte-level BPE and Unigram tokenizers, XLM-RoBERTa cross-encoders) against
transformers, the reference implementation and dllm verify --reference, and the refusals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from model_fixtures import write_safetensors
from test_encoders import MODULES, bert_weights

from etalii_dllm import numerics
from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.importing.importer import bert_config
from etalii_dllm.modelfile import ModelFile

tokenizers = pytest.importorskip("tokenizers")
pytest.importorskip("sentencepiece")

ROOT = Path(__file__).resolve().parent.parent
TEXTS = ["The quick brown fox.", "Hello, world! How are you?", "naïve café façade", "日本語 and Ελληνικά"]
MAX_SEQ_LENGTH = 14
POSITIONS = 40
ROBERTA_CONFIG = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "max_position_embeddings": POSITIONS + 2,
    "type_vocab_size": 1,
    "layer_norm_eps": 1e-5,
    "hidden_act": "gelu",
    "position_embedding_type": "absolute",
    "pad_token_id": 1,
    "bos_token_id": 0,
    "eos_token_id": 2,
}


def roberta_bpe_tokenizer():
    """A byte-level BPE tokenizer laid out like RoBERTa's (``<s>``, ``<pad>``, ``</s>``, ``<unk>`` first)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train([str(ROOT / "README.md")], trainer)
    tokenizer.post_processor = processors.RobertaProcessing(("</s>", 2), ("<s>", 0))
    return tokenizer


def write_roberta_checkpoint(
    directory: Path,
    model_type: str = "xlm-roberta",
    *,
    pooling: str | None = "pooling_mode_mean_tokens",
    labels: int = 0,
    prefix: str = "",
) -> dict:
    """A RoBERTa (byte-level BPE) or XLM-RoBERTa (Unigram) checkpoint: a sentence-transformers embedder, or with
    ``labels`` a ``XLMRobertaForSequenceClassification`` cross-encoder."""
    if model_type == "roberta":
        reference = roberta_bpe_tokenizer()
    else:
        from test_unigram import xlmr_tokenizer

        reference, _ = xlmr_tokenizer()
    directory.mkdir(parents=True, exist_ok=True)
    architecture = "XLMRoberta" if model_type == "xlm-roberta" else "Roberta"
    config = {
        **ROBERTA_CONFIG,
        "model_type": model_type,
        "architectures": [f"{architecture}{'ForSequenceClassification' if labels else 'Model'}"],
        "vocab_size": reference.get_vocab_size(),
    }
    if labels:
        config["id2label"] = {str(i): f"LABEL_{i}" for i in range(labels)}
        config["label2id"] = {f"LABEL_{i}": i for i in range(labels)}
    weights = bert_weights(config, seed=9)
    hidden = config["hidden_size"]
    if labels:
        dense, dense_bias = weights.pop("pooler.dense.weight"), weights.pop("pooler.dense.bias")
        stored = {"roberta." + name: ("F32", values) for name, values in weights.items()}
        out = (numerics.fill_gaussian(77, labels * hidden).reshape(labels, hidden) * 0.3).astype(np.float32)
        stored |= {
            "classifier.dense.weight": ("F32", dense),
            "classifier.dense.bias": ("F32", dense_bias),
            "classifier.out_proj.weight": ("F32", out),
            "classifier.out_proj.bias": ("F32", np.full(labels, 0.125, dtype=np.float32)),
        }
    else:
        stored = {prefix + name: ("F32", values) for name, values in weights.items()}
    write_safetensors(directory / "model.safetensors", stored, {"format": "pt"})
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "tokenizer.json").write_text(reference.to_str(), encoding="utf-8")
    tokenizer_config = {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>"}
    tokenizer_config["model_max_length"] = MAX_SEQ_LENGTH
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    (directory / "README.md").write_bytes(b"---\nlicense: apache-2.0\n---\n\n# Tiny RoBERTa\n")
    if pooling is not None and not labels:
        (directory / "modules.json").write_text(json.dumps(MODULES), encoding="utf-8")
        (directory / "1_Pooling").mkdir(exist_ok=True)
        settings = {"word_embedding_dimension": hidden, pooling: True}
        (directory / "1_Pooling" / "config.json").write_text(json.dumps(settings), encoding="utf-8")
        limits = {"max_seq_length": MAX_SEQ_LENGTH, "do_lower_case": False}
        (directory / "sentence_bert_config.json").write_text(json.dumps(limits), encoding="utf-8")
    return config


@pytest.fixture(scope="module", params=["xlm-roberta", "roberta"])
def embedder(request, tmp_path_factory) -> tuple[Path, DllmEngine]:
    directory = tmp_path_factory.mktemp(request.param)
    write_roberta_checkpoint(directory / "checkpoint", request.param)
    import_model(directory / "checkpoint", directory / "model.dllm", repository=f"example/tiny-{request.param}")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


@pytest.fixture(scope="module")
def cross(tmp_path_factory) -> tuple[Path, DllmEngine]:
    directory = tmp_path_factory.mktemp("xlmr-cross")
    write_roberta_checkpoint(directory / "checkpoint", labels=1)
    import_model(directory / "checkpoint", directory / "cross.dllm", repository="example/tiny-xlmr-cross-encoder")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "cross.dllm")


def transformers_model(directory: Path, classification: bool = False):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    kind = transformers.AutoModelForSequenceClassification if classification else transformers.AutoModel
    return kind.from_pretrained(directory).eval()


def transformers_states(directory: Path, tokens: list[int]) -> np.ndarray:
    torch = pytest.importorskip("torch")
    model = transformers_model(directory)
    with torch.no_grad():
        return model(torch.tensor([tokens])).last_hidden_state[0].double().numpy()


# Positions (#348)


def test_position_ids_skip_padding():
    config = bert_config({**ROBERTA_CONFIG, "model_type": "roberta", "vocab_size": 50})
    assert config.padding_index == 1 and config.context_length == POSITIONS
    assert config.tensor_shapes()["position_embedding.weight"] == (POSITIONS + 2, 32)
    assert config.position_ids([0, 5, 1, 7, 2]) == [2, 3, 1, 4, 5]
    plain = bert_config({**ROBERTA_CONFIG, "model_type": "bert", "vocab_size": 50})
    assert plain.padding_index is None and plain.position_ids([0, 1, 2]) == [0, 1, 2]
    assert "padding_index" not in plain.to_dict() and config.to_dict()["padding_index"] == 1
    assert TransformerConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="padding_index must not be negative"):
        TransformerConfig.from_dict({**config.to_dict(), "padding_index": -1})
    with pytest.raises(ValueError, match="padding positions are for bert"):
        TransformerConfig.from_dict({**config.to_dict(), "family": "llama", "type_vocabulary_size": 0})
    with pytest.raises(ModelImportError, match="no positions past the padding"):
        bert_config({**ROBERTA_CONFIG, "model_type": "roberta", "vocab_size": 50, "max_position_embeddings": 2})


def test_import_records_the_encoder(embedder):
    _, engine = embedder
    config = engine.model.config
    assert config.family == "bert" and config.padding_index == 1 and config.type_vocabulary_size == 1
    assert config.context_length == POSITIONS and config.rms_norm_eps == 1e-5
    assert engine.embedding is not None and engine.embedding["max_tokens"] == MAX_SEQ_LENGTH
    assert engine.embedding["pooling"] == "mean"


def test_states_match_transformers(embedder):
    checkpoint, engine = embedder
    for text in TEXTS:
        tokens = engine.tokenizer.encode(text, add_special_tokens=True)
        expected = transformers_states(checkpoint, tokens)
        np.testing.assert_allclose(engine.model.hidden_states(tokens), expected, rtol=1e-4, atol=1e-5)
    padded = [0, 7, 1, 1, 9, 2]  # padding tokens inside the input keep their own position
    np.testing.assert_allclose(
        engine.model.hidden_states(padded), transformers_states(checkpoint, padded), rtol=1e-4, atol=1e-5
    )


def test_embeddings_match_mean_pooling(embedder):
    checkpoint, engine = embedder
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    for text in [*TEXTS, " ".join(TEXTS * 3)]:
        tokens = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"]
        assert engine.embedding_tokens(text) == tokens
        states = transformers_states(checkpoint, tokens)
        mean = states.mean(axis=0)
        np.testing.assert_allclose(engine.embed(text).vector, mean / np.linalg.norm(mean), rtol=1e-4, atol=1e-5)


def test_thread_invariance(embedder):
    _, engine = embedder
    tokens = engine.embedding_tokens(TEXTS[1])
    states = engine.model.hidden_states(tokens)
    for threads in (1, 3):
        numerics.set_threads(threads)
        try:
            assert engine.model.hidden_states(tokens).tobytes() == states.tobytes()
        finally:
            numerics.set_threads(0)


def test_bare_and_prefixed_checkpoints(tmp_path):
    write_roberta_checkpoint(tmp_path / "prefixed", "roberta", prefix="roberta.", pooling=None)
    import_model(tmp_path / "prefixed", tmp_path / "prefixed.dllm")
    engine = DllmEngine.from_model_file(tmp_path / "prefixed.dllm")
    assert engine.embedding is not None and engine.embedding["max_tokens"] == POSITIONS
    assert engine.embed(TEXTS[0]).vector.shape == (32,)


# Cross-encoders


def test_cross_encoder_matches_transformers(cross):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    checkpoint, engine = cross
    file = ModelFile(Path(checkpoint).parent / "cross.dllm")
    assert file.classifier == {"labels": ["LABEL_0"], "activation": "none", "max_tokens": MAX_SEQ_LENGTH}
    assert {"pooler.weight", "classifier.weight"} <= set(file.tensors)
    model = transformers_model(checkpoint, classification=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    for query, document in [(TEXTS[0], TEXTS[1]), (TEXTS[2], TEXTS[3] * 4), (TEXTS[3], TEXTS[0])]:
        batch = tokenizer(query, document, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        tokens, types = engine.classification_tokens(query, document)
        assert tokens == batch["input_ids"][0].tolist() and types == [0] * len(tokens)
        with torch.no_grad():
            expected = model(input_ids=batch["input_ids"]).logits[0].double().numpy()
        np.testing.assert_allclose(engine.classify(query, document).logits, expected, atol=1e-5)


def test_reranker_with_an_xlm_roberta_cross_encoder(cross):
    from etalii_dllm.reranking import Reranker

    _, engine = cross
    reranker = Reranker(engine)
    assert reranker.cross_encoder
    expected = [engine.classify(TEXTS[0], d).scores[0] for d in TEXTS]
    assert sorted(i for i, _ in reranker.rerank(TEXTS[0], TEXTS)) == [0, 1, 2, 3]
    assert dict(reranker.rerank(TEXTS[0], TEXTS)) == dict(enumerate(expected))


# Verification (#350)


def test_reference_implementation_and_verify(embedder, cross):
    from etalii_dllm import reference, verify

    _, engine = embedder
    twin = reference.ReferenceEncoder.from_engine(engine)
    for tokens in (engine.embedding_tokens(TEXTS[3]), [0, 7, 1, 1, 9, 2]):
        assert engine.model.hidden_states(tokens).tobytes() == twin.hidden_states(tokens).tobytes()
    assert verify.check_reference(engine).equal
    _, scorer = cross
    twin = reference.ReferenceEncoder.from_engine(scorer)
    tokens, types = scorer.classification_tokens(TEXTS[0], TEXTS[1])
    mine = np.asarray(scorer.classify(TEXTS[0], TEXTS[1]).logits, dtype=np.float32)
    assert mine.tobytes() == twin.classify(tokens, types).tobytes()
    assert verify.check_reference(scorer).equal
