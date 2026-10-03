"""DeBERTa-v2/v3 encoders (issues #362-#365): log-bucketed relative positions and the biased attention kernel against
their definitions, states, embeddings and cross-encoders against transformers' ``DebertaV2Model``, thread invariance,
golden scores, the reference implementation and dllm verify --reference, and the refusals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from model_fixtures import write_safetensors
from test_encoders import MODULES

from etalii_dllm import numerics
from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.encoder import relative_buckets, relative_index
from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.importing.importer import deberta_config
from etalii_dllm.modelfile import ModelFile

tokenizers = pytest.importorskip("tokenizers")
pytest.importorskip("sentencepiece")

TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Hello, world! How are you?",
    "naïve café façade",
    "Deterministic models give the same bits on every machine, whatever the load.",
]
MAX_SEQ_LENGTH = 40
POSITIONS = 64
DEBERTA_CONFIG = {
    "model_type": "deberta-v2",
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "max_position_embeddings": POSITIONS,
    "type_vocab_size": 0,
    "layer_norm_eps": 1e-7,
    "hidden_act": "gelu",
    "relative_attention": True,
    "position_buckets": 8,
    "max_relative_positions": 32,
    "pos_att_type": ["p2c", "c2p"],
    "norm_rel_ebd": "layer_norm",
    "share_att_key": True,
    "position_biased_input": False,
    "pad_token_id": 0,
    "pooler_hidden_size": 32,
    "pooler_hidden_act": "gelu",
    "pooler_dropout": 0,
}


def frozen_pieces(count: int) -> list[tuple[str, float]]:
    """At most ``count`` Unigram pieces written out from :data:`TEXTS` (the printable ASCII and the texts'
    characters, then their most frequent two- and three-character pieces), so the vocabulary never changes with a
    trained model."""
    from collections import Counter

    corpus = " ".join(TEXTS).split()
    characters = sorted({chr(c) for c in range(33, 127)} | {c for word in corpus for c in word})
    grams: Counter[str] = Counter()
    for word in corpus:
        marked = "▁" + word
        grams.update(marked[i : i + n] for n in (2, 3) for i in range(len(marked) - n + 1))
    chosen = sorted(grams, key=lambda gram: (-grams[gram], gram))[: count - 1 - len(characters)]
    pieces = [("▁", -2.0)] + [(gram, -5.0 - 0.001 * rank) for rank, gram in enumerate(chosen)]
    return pieces + [(c, -10.0 - 0.001 * rank) for rank, c in enumerate(characters)]


def deberta_tokenizer():
    """A Unigram tokenizer laid out as transformers converts DeBERTa-v3's ``spm.model``: ``[PAD]``, ``[CLS]``,
    ``[SEP]``, ``[UNK]`` first, the Strip/Precompiled/space-collapsing normaliser, Metaspace and the
    ``[CLS] A [SEP] B [SEP]`` template with the second text of type 1."""
    from test_unigram import sentencepiece_model
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

    _, charsmap = sentencepiece_model()  # nmt_nfkc's precompiled map does not depend on the training text
    vocab = [("[PAD]", 0.0), ("[CLS]", 0.0), ("[SEP]", 0.0), ("[UNK]", 0.0), *frozen_pieces(397), ("[MASK]", 0.0)]
    tokenizer = Tokenizer(models.Unigram(vocab, 3, False))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Strip(), normalizers.Precompiled(charsmap), normalizers.Replace(Regex(" {2,}"), " ")]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")]
    )
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    tokenizer.add_special_tokens([AddedToken(t, special=True) for t in ("[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]")])
    return tokenizer


def deberta_weights(config: dict, seed: int = 13, labels: int = 0) -> dict[str, np.ndarray]:
    """Float32 weights keyed by transformers' ``DebertaV2Model`` names (plus the pooler and classifier)."""
    hidden, inner = config["hidden_size"], config["intermediate_size"]
    counter = iter(range(seed * 1000, seed * 1000 + 1000))

    def gaussian(*shape: int) -> np.ndarray:
        return numerics.fill_gaussian(next(counter), int(np.prod(shape))).reshape(shape)

    def matrix(rows: int, columns: int, scale: float = 0.2) -> np.ndarray:
        return (gaussian(rows, columns) * np.float32(scale)).astype(np.float32)

    def vector(size: int, centre: float = 0.0) -> np.ndarray:
        return (np.float32(centre) + gaussian(size) * np.float32(0.1)).astype(np.float32)

    span = config["position_buckets"] if config["position_buckets"] > 0 else config["max_relative_positions"]
    weights = {
        "embeddings.word_embeddings.weight": matrix(config["vocab_size"], hidden, 1.0),
        "embeddings.LayerNorm.weight": vector(hidden, 1.0),
        "embeddings.LayerNorm.bias": vector(hidden),
        "encoder.rel_embeddings.weight": matrix(2 * span, hidden, 1.0),
        "encoder.LayerNorm.weight": vector(hidden, 1.0),
        "encoder.LayerNorm.bias": vector(hidden),
    }
    if config["type_vocab_size"]:
        weights["embeddings.token_type_embeddings.weight"] = matrix(config["type_vocab_size"], hidden, 1.0)
    for i in range(config["num_hidden_layers"]):
        p = f"encoder.layer.{i}."
        for name in ("query_proj", "key_proj", "value_proj"):
            weights[p + f"attention.self.{name}.weight"] = matrix(hidden, hidden)
            weights[p + f"attention.self.{name}.bias"] = vector(hidden)
        weights[p + "attention.output.dense.weight"] = matrix(hidden, hidden)
        weights[p + "attention.output.dense.bias"] = vector(hidden)
        weights[p + "intermediate.dense.weight"] = matrix(inner, hidden)
        weights[p + "intermediate.dense.bias"] = vector(inner)
        weights[p + "output.dense.weight"] = matrix(hidden, inner)
        weights[p + "output.dense.bias"] = vector(hidden)
        for name in ("attention.output.LayerNorm", "output.LayerNorm"):
            weights[p + name + ".weight"] = vector(hidden, 1.0)
            weights[p + name + ".bias"] = vector(hidden)
    if labels:
        weights["pooler.dense.weight"] = matrix(hidden, hidden, 0.3)
        weights["pooler.dense.bias"] = vector(hidden)
        weights["classifier.weight"] = matrix(labels, hidden, 0.3)
        weights["classifier.bias"] = np.full(labels, 0.125, dtype=np.float32)
    return weights


def write_deberta_checkpoint(
    directory: Path, *, labels: int = 0, types: int = 0, pooling: str = "pooling_mode_mean_tokens", **changes
) -> dict:
    """A DeBERTa-v3 style checkpoint: a sentence-transformers embedder, or with ``labels`` a
    ``DebertaV2ForSequenceClassification`` cross-encoder (``deberta.`` prefix)."""
    reference = deberta_tokenizer()
    directory.mkdir(parents=True, exist_ok=True)
    kind = "ForSequenceClassification" if labels else "Model"
    config = {
        **DEBERTA_CONFIG,
        "architectures": [f"DebertaV2{kind}"],
        "vocab_size": reference.get_vocab_size(),
        "type_vocab_size": types,
        **changes,
    }
    weights = deberta_weights(config, labels=labels)
    if labels:
        config["id2label"] = {str(i): f"LABEL_{i}" for i in range(labels)}
        config["label2id"] = {f"LABEL_{i}": i for i in range(labels)}
        weights = {
            (name if name.startswith(("pooler.", "classifier.")) else "deberta." + name): values
            for name, values in weights.items()
        }
    write_safetensors(directory / "model.safetensors", {n: ("F32", v) for n, v in weights.items()}, {"format": "pt"})
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "tokenizer.json").write_text(reference.to_str(), encoding="utf-8")
    tokenizer_config = {
        "bos_token": "[CLS]",
        "eos_token": "[SEP]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "mask_token": "[MASK]",
        "model_max_length": MAX_SEQ_LENGTH,
        "tokenizer_class": "DebertaV2Tokenizer",
    }
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    (directory / "README.md").write_bytes(b"---\nlicense: mit\n---\n\n# Tiny DeBERTa\n")
    if not labels:
        (directory / "modules.json").write_text(json.dumps(MODULES), encoding="utf-8")
        (directory / "1_Pooling").mkdir(exist_ok=True)
        settings = {"word_embedding_dimension": config["hidden_size"], pooling: True}
        (directory / "1_Pooling" / "config.json").write_text(json.dumps(settings), encoding="utf-8")
        limits = {"max_seq_length": MAX_SEQ_LENGTH, "do_lower_case": False}
        (directory / "sentence_bert_config.json").write_text(json.dumps(limits), encoding="utf-8")
    return config


@pytest.fixture(scope="module")
def embedder(tmp_path_factory) -> tuple[Path, DllmEngine]:
    directory = tmp_path_factory.mktemp("deberta")
    write_deberta_checkpoint(directory / "checkpoint")
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-deberta")
    return directory / "checkpoint", DllmEngine.from_model_file(directory / "model.dllm")


@pytest.fixture(scope="module")
def cross(tmp_path_factory) -> tuple[Path, DllmEngine]:
    directory = tmp_path_factory.mktemp("deberta-cross")
    write_deberta_checkpoint(directory / "checkpoint", labels=3, types=2)
    import_model(directory / "checkpoint", directory / "cross.dllm", repository="example/tiny-deberta-nli")
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


def long_tokens(engine: DllmEngine) -> list[int]:
    """A sequence longer than the buckets' exact range and the table span, so the logarithmic buckets and the
    clamping both matter."""
    tokens = [1]
    for text in TEXTS:
        tokens += engine.tokenizer.encode(text)
    return [*tokens[: POSITIONS - 1], 2]


# The kernel and the buckets (#362)


@pytest.mark.parametrize(("buckets", "maximum"), [(256, 512), (256, 1024), (8, 32), (64, 128), (0, 64), (10, 40)])
def test_relative_buckets_match_transformers(buckets, maximum):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position

    distances = np.arange(-4096, 4097, dtype=np.int64)
    mine = relative_buckets(distances, buckets, maximum)
    if buckets <= 0:
        assert np.array_equal(mine, distances)
        return
    expected = make_log_bucket_position(torch.tensor(distances), buckets, maximum).long().numpy()
    assert np.array_equal(mine, expected)


def test_relative_index_is_clamped_to_the_table():
    config = deberta_config({**DEBERTA_CONFIG, "vocab_size": 50})
    assert config.relative_span == 8 and config.position_buckets == 8 and config.max_relative_positions == 32
    index = relative_index(config, 60)
    assert index.shape == (60, 60) and index.min() == 0 and index.max() == 15
    assert index[5, 5] == 8 and index[6, 5] == 9 and index[5, 6] == 7  # distance i - j, offset by the span
    assert all(index[i, j] + index[j, i] == 16 for i in range(60) for j in range(60) if 0 < index[i, j] < 15)


def test_biased_attention_matches_its_definition():
    q = numerics.fill_gaussian(1, 7 * 3 * 5).reshape(7, 3, 5)
    k = numerics.fill_gaussian(2, 9 * 3 * 5).reshape(9, 3, 5)
    v = numerics.fill_gaussian(3, 9 * 3 * 4).reshape(9, 3, 4)
    bias = numerics.fill_gaussian(4, 3 * 7 * 9).reshape(3, 7, 9)
    out = numerics.biased_attention(q, k, v, bias, 0.3).numpy()
    q64, k64, v64, b64 = (a.astype(np.float64) for a in (q, k, v, bias))
    scores = (np.einsum("thd,jhd->htj", q64, k64) + b64) * 0.3
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = np.einsum("htj,jhd->thd", p / p.sum(axis=-1, keepdims=True), v64)
    np.testing.assert_allclose(out, expected, atol=1e-6)
    zero = numerics.biased_attention(q, k, v, np.zeros_like(bias), 0.3).numpy()
    assert zero.tobytes() == numerics.attention(q, k, v, scale=0.3, causal=False).numpy().tobytes()
    from etalii_dllm import reference

    assert out.tobytes() == reference.biased_attention(q, k, v, bias, 0.3).tobytes()
    with pytest.raises(ValueError, match="bias must be"):
        numerics.biased_attention(q, k, v, bias[:, :6], 0.3)
    with pytest.raises(ValueError, match="same heads"):
        numerics.biased_attention(q, k[:, :2], v, bias, 0.3)


def test_biased_attention_is_thread_invariant():
    q = numerics.fill_gaussian(5, 33 * 4 * 8).reshape(33, 4, 8)
    k = numerics.fill_gaussian(6, 33 * 4 * 8).reshape(33, 4, 8)
    v = numerics.fill_gaussian(7, 33 * 4 * 8).reshape(33, 4, 8)
    bias = numerics.fill_gaussian(8, 4 * 33 * 33).reshape(4, 33, 33)
    try:
        results = []
        for threads in (1, 3, 8):
            numerics.set_threads(threads)
            results.append(numerics.biased_attention(q, k, v, bias, 0.25).numpy().tobytes())
    finally:
        numerics.set_threads(0)
    assert len(set(results)) == 1


# Embedders (#363)


def test_config_mapping():
    config = deberta_config({**DEBERTA_CONFIG, "vocab_size": 50, "pos_att_type": "p2c|c2p"})
    assert config.family == "deberta" and config.is_encoder and config.type_vocabulary_size == 0
    assert config.context_length == POSITIONS and config.rms_norm_eps == 1e-7
    again = TransformerConfig.from_dict(config.to_dict())
    assert again == config and "padding_index" not in config.to_dict()
    plain = {"position_buckets": -1, "max_relative_positions": -1}
    unbucketed = deberta_config({**DEBERTA_CONFIG, "vocab_size": 50, **plain})
    assert unbucketed.position_buckets == 0 and unbucketed.relative_span == POSITIONS
    assert "relative_embedding.weight" in unbucketed.tensor_shapes()
    assert unbucketed.tensor_shapes()["relative_embedding.weight"] == (2 * POSITIONS, 32)
    shapes = config.tensor_shapes()
    assert "position_embedding.weight" not in shapes and "token_type_embedding.weight" not in shapes


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"relative_attention": False}, "relative_attention"),
        ({"position_biased_input": True}, "position_biased_input"),
        ({"share_att_key": False}, "share_att_key"),
        ({"pos_att_type": ["c2p"]}, "pos_att_type"),
        ({"norm_rel_ebd": "none"}, "norm_rel_ebd"),
        ({"conv_kernel_size": 3}, "conv_kernel_size"),
        ({"embedding_size": 16}, "embedding_size"),
        ({"z_steps": 2}, "z_steps"),
        ({"hidden_act": "relu"}, "activation 'relu'"),
        ({"pooler_hidden_act": "tanh"}, "pooler activation"),
        ({"num_attention_heads": 5}, "times the head size"),
        ({"num_hidden_layers": 0}, "layers must be positive"),
    ],
)
def test_config_refusals(change, message):
    with pytest.raises(ModelImportError, match=message):
        deberta_config({**DEBERTA_CONFIG, "vocab_size": 50, **change})


def test_config_validation():
    with pytest.raises(ModelImportError, match="--context-length"):
        deberta_config({**DEBERTA_CONFIG, "vocab_size": 50}, context_length=128)
    base = deberta_config({**DEBERTA_CONFIG, "vocab_size": 50}).to_dict()
    with pytest.raises(ValueError, match="only deberta"):
        TransformerConfig.from_dict({**base, "max_relative_positions": 0})
    with pytest.raises(ValueError, match="only deberta"):
        TransformerConfig.from_dict({**base, "family": "bert", "type_vocabulary_size": 2})
    with pytest.raises(ValueError, match="padding positions"):
        TransformerConfig.from_dict({**base, "padding_index": 0})
    with pytest.raises(ValueError, match="for bert"):
        TransformerConfig.from_dict({**base, "family": "llama", "activation": "silu"})


def test_import_records_the_encoder(embedder):
    checkpoint, _ = embedder
    file = ModelFile(checkpoint.parent / "model.dllm")
    assert file.config.family == "deberta" and file.embedding is not None
    assert file.embedding["pooling"] == "mean" and file.embedding["max_tokens"] == MAX_SEQ_LENGTH
    assert {"relative_embedding.weight", "relative_norm.bias"} <= set(file.tensors)
    assert not any(name.startswith(("pooler", "classifier")) for name in file.tensors)


def test_states_match_transformers(embedder):
    checkpoint, engine = embedder
    for tokens in (engine.embedding_tokens(TEXTS[0]), long_tokens(engine), [1, 2]):
        mine = engine.model.hidden_states(tokens)
        np.testing.assert_allclose(mine, transformers_states(checkpoint, tokens), atol=2e-5)


def test_embeddings_match_mean_pooling(embedder):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    checkpoint, engine = embedder
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    model = transformers_model(checkpoint)
    for text in TEXTS:
        batch = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        assert engine.embedding_tokens(text) == batch["input_ids"][0].tolist()
        with torch.no_grad():
            states = model(**batch).last_hidden_state[0].double().numpy()
        vector = states.mean(axis=0)
        np.testing.assert_allclose(engine.embed(text).vector, vector / np.linalg.norm(vector), atol=2e-5)


def test_thread_invariance_and_goldens(embedder):
    from golden_values import DEBERTA_EMBEDDING_FINGERPRINT

    _, engine = embedder
    tokens = long_tokens(engine)
    try:
        results = set()
        for threads in (1, 2, 5):
            numerics.set_threads(threads)
            results.add(engine.model.hidden_states(tokens).tobytes())
    finally:
        numerics.set_threads(0)
    assert len(results) == 1
    fixed = [1, 17, 40, 9, 300, 5, 77, 12, 2] * 6
    assert numerics.fingerprint(engine.model.hidden_states(fixed)) == DEBERTA_EMBEDDING_FINGERPRINT


def test_quantised_deberta_runs(tmp_path, embedder):
    checkpoint, engine = embedder
    quantised = DllmEngine.from_model_file(checkpoint.parent / "model.dllm", quantize="q8_0")
    tokens = engine.embedding_tokens(TEXTS[1])
    np.testing.assert_allclose(quantised.model.hidden_states(tokens), engine.model.hidden_states(tokens), atol=0.1)
    assert quantised.model.weights_fingerprint != engine.model.weights_fingerprint


def test_an_index_built_with_deberta(embedder):
    from etalii_dllm.retrieval import build_index

    _, engine = embedder
    index = build_index(engine, [(f"doc{i}.txt", text) for i, text in enumerate(TEXTS)], chunk_tokens=8)
    assert sorted({chunk.source for chunk in index.chunks}) == [f"doc{i}.txt" for i in range(len(TEXTS))]
    first, again = index.search(engine, TEXTS[1], 2), index.search(engine, TEXTS[1], 2)
    assert [(h.chunk.source, h.score) for h in first] == [(h.chunk.source, h.score) for h in again]


# Cross-encoders (#364)


def test_cross_encoder_matches_transformers(cross):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    checkpoint, engine = cross
    file = ModelFile(checkpoint.parent / "cross.dllm")
    assert file.classifier is not None and file.classifier["labels"] == ["LABEL_0", "LABEL_1", "LABEL_2"]
    assert file.config.type_vocabulary_size == 2
    model = transformers_model(checkpoint, classification=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    for query, document in [(TEXTS[0], TEXTS[1]), (TEXTS[2], TEXTS[3] * 3), (TEXTS[3], TEXTS[0])]:
        batch = tokenizer(query, document, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        tokens, types = engine.classification_tokens(query, document)
        assert tokens == batch["input_ids"][0].tolist() and types == batch["token_type_ids"][0].tolist()
        with torch.no_grad():
            expected = model(**batch).logits[0].double().numpy()
        np.testing.assert_allclose(engine.classify(query, document).logits, expected, atol=2e-5)


def test_single_label_reranker(tmp_path):
    from etalii_dllm.reranking import Reranker

    write_deberta_checkpoint(tmp_path / "checkpoint", labels=1)
    import_model(tmp_path / "checkpoint", tmp_path / "reranker.dllm")
    engine = DllmEngine.from_model_file(tmp_path / "reranker.dllm")
    reranker = Reranker(engine)
    assert reranker.cross_encoder
    expected = [engine.classify(TEXTS[0], d).scores[0] for d in TEXTS]
    assert dict(reranker.rerank(TEXTS[0], TEXTS)) == dict(enumerate(expected))
    from golden_values import DEBERTA_CLASSIFIER_FINGERPRINT

    fixed = [1, 17, 40, 9, 2, 300, 5, 77, 12, 2]
    logits = engine.model.classify(fixed, [0] * 5 + [1] * 5)
    assert numerics.fingerprint(logits) == DEBERTA_CLASSIFIER_FINGERPRINT


# Verification (#365)


def test_reference_implementation_and_verify(embedder, cross):
    from etalii_dllm import reference, verify

    _, engine = embedder
    twin = reference.ReferenceEncoder.from_engine(engine)
    for tokens in (engine.embedding_tokens(TEXTS[3]), long_tokens(engine)):
        assert engine.model.hidden_states(tokens).tobytes() == twin.hidden_states(tokens).tobytes()
    assert verify.check_reference(engine).equal
    _, scorer = cross
    twin = reference.ReferenceEncoder.from_engine(scorer)
    tokens, types = scorer.classification_tokens(TEXTS[0], TEXTS[1])
    mine = np.asarray(scorer.classify(TEXTS[0], TEXTS[1]).logits, dtype=np.float32)
    assert mine.tobytes() == twin.classify(tokens, types).tobytes()
    assert verify.check_reference(scorer).equal


def test_import_refusals(tmp_path):
    write_deberta_checkpoint(tmp_path / "spm")
    (tmp_path / "spm" / "tokenizer.json").unlink()
    with pytest.raises(ModelImportError, match=r"no tokenizer\.json"):
        import_model(tmp_path / "spm", tmp_path / "spm.dllm")
    from etalii_dllm.importing.importer import _deberta_name

    with pytest.raises(ModelImportError, match="unexpected tensor"):
        _deberta_name("stray.weight")

    assert _deberta_name("deberta.embeddings.position_embeddings.weight") is None
    assert _deberta_name("lm_predictions.lm_head.dense.weight") is None
    assert _deberta_name("pooler.dense.weight") is None
    assert _deberta_name("pooler.dense.weight", classifier=True) == "pooler.weight"
    assert _deberta_name("deberta.encoder.layer.1.attention.self.key_proj.bias") == "layers.1.attention.k.bias"


# Fine-tuning, LoRA and export (#367-#370)


def test_biased_attention_backward_matches_numeric_gradient():
    q = numerics.fill_gaussian(11, 5 * 2 * 4).reshape(5, 2, 4).astype(np.float64)
    k = numerics.fill_gaussian(12, 6 * 2 * 4).reshape(6, 2, 4).astype(np.float64)
    v = numerics.fill_gaussian(13, 6 * 2 * 3).reshape(6, 2, 3).astype(np.float64)
    bias = numerics.fill_gaussian(14, 2 * 5 * 6).reshape(2, 5, 6).astype(np.float64)
    dout = numerics.fill_gaussian(15, 5 * 2 * 3).reshape(5, 2, 3).astype(np.float64)
    scale = 0.4

    def loss(q, k, v, bias):
        scores = (np.einsum("thd,jhd->htj", q, k) + bias) * scale
        p = np.exp(scores - scores.max(axis=-1, keepdims=True))
        out = np.einsum("htj,jhd->thd", p / p.sum(axis=-1, keepdims=True), v)
        return float((out * dout).sum())

    grads = [t.numpy() for t in numerics.biased_attention_backward(q, k, v, bias, dout, scale)]
    inputs = [q, k, v, bias]
    for index, (value, grad) in enumerate(zip(inputs, grads, strict=True)):
        assert grad.shape == value.shape
        numeric = np.zeros_like(value)
        for position in np.ndindex(value.shape):
            plus, minus = [a.copy() for a in inputs], [a.copy() for a in inputs]
            plus[index][position] += 1e-6
            minus[index][position] -= 1e-6
            numeric[position] = (loss(*plus) - loss(*minus)) / 2e-6
        np.testing.assert_allclose(grad, numeric, atol=2e-5)
    # a zero bias gives attention_backward's dq, dk and dv
    zero = numerics.biased_attention_backward(q, k, v, np.zeros_like(bias), dout, scale)
    plain = numerics.attention_backward(q, k, v, dout, scale=scale, causal=False, q_offset=0)
    assert all(a.numpy().tobytes() == b.numpy().tobytes() for a, b in zip(zero[:3], plain, strict=True))
    with pytest.raises(ValueError, match="dout must be"):
        numerics.biased_attention_backward(q, k, v, bias, dout[:4], scale)
    with pytest.raises(ValueError, match="bias must be"):
        numerics.biased_attention_backward(q, k, v, bias[:1], dout, scale)


def test_biased_attention_backward_is_thread_invariant():
    q = numerics.fill_gaussian(5, 33 * 4 * 8).reshape(33, 4, 8)
    bias = numerics.fill_gaussian(8, 4 * 33 * 33).reshape(4, 33, 33)
    try:
        results = set()
        for threads in (1, 3, 8):
            numerics.set_threads(threads)
            grads = numerics.biased_attention_backward(q, q, q, bias, q, 0.25)
            results.add(b"".join(g.numpy().tobytes() for g in grads))
    finally:
        numerics.set_threads(0)
    assert len(results) == 1


def deberta_autograd(directory: Path, tokens: list[int], upstream: np.ndarray, labels: int, types=None):
    """transformers' float64 gradients of ``sum(upstream * output)`` under our names."""
    torch = pytest.importorskip("torch")
    from etalii_dllm.importing.importer import _deberta_name

    model = transformers_model(directory, classification=bool(labels)).double()
    extra = {} if types is None else {"token_type_ids": torch.tensor([types])}
    output = model(torch.tensor([tokens]), **extra)
    values = output.logits[0] if labels else output.last_hidden_state[0]
    (values * torch.tensor(upstream, dtype=torch.float64)).sum().backward()
    return {
        _deberta_name(name, bool(labels)): parameter.grad.numpy()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def check_gradients(directory: Path, engine: DllmEngine, tokens: list[int], types=None) -> None:
    from etalii_dllm.training.encoder_backprop import EncoderGradients

    model = engine.model
    weights = {name: np.asarray(values, dtype=np.float32) for name, values in model.tensors.items()}
    config = model.config
    gradients = EncoderGradients(config)
    if config.classifier_labels:
        result = gradients.classify(weights, tokens, types)
        assert result.logits.tobytes() == model.classify(tokens, types).tobytes()
        upstream = np.linspace(0.75, -0.5, config.classifier_labels).astype(np.float32)
        ours = gradients.gradients(weights, result, dlogits=upstream)
    else:
        result = gradients.encode(weights, tokens)
        assert result.states.tobytes() == model.hidden_states(tokens).tobytes()
        upstream = numerics.fill_gaussian(4, len(tokens) * 32).reshape(len(tokens), 32)
        ours = gradients.gradients(weights, result, upstream)
    theirs = deberta_autograd(directory, tokens, upstream, config.classifier_labels, types)
    assert set(ours) == set(theirs)
    for name, expected in theirs.items():
        scale = max(float(np.abs(expected).max()), 1e-3)
        assert np.abs(ours[name] - expected).max() <= 2e-4 * scale, name


def test_gradients_match_autograd(embedder, cross):
    checkpoint, engine = embedder
    tokens = long_tokens(engine)  # far buckets, clamped rows and table rows no pair reads
    check_gradients(checkpoint, engine, tokens)
    checkpoint, scorer = cross
    tokens, types = scorer.classification_tokens(TEXTS[0], TEXTS[1])
    check_gradients(checkpoint, scorer, tokens, types)


def test_tanh_gelu_deberta_matches_transformers(tmp_path):
    torch = pytest.importorskip("torch")
    write_deberta_checkpoint(tmp_path / "checkpoint", labels=2, hidden_act="gelu_new", pooler_hidden_act="gelu_new")
    import_model(tmp_path / "checkpoint", tmp_path / "tanh.dllm")
    engine = DllmEngine.from_model_file(tmp_path / "tanh.dllm")
    assert engine.model.config.activation == "gelu_tanh"
    tokens = [1, 17, 40, 9, 2, 300, 5, 77, 12, 2]
    with torch.no_grad():
        expected = transformers_model(tmp_path / "checkpoint", classification=True)(torch.tensor([tokens])).logits[0]
    np.testing.assert_allclose(engine.model.classify(tokens), expected.numpy(), atol=2e-5)
    from etalii_dllm import reference

    twin = reference.ReferenceEncoder.from_engine(engine)
    assert twin.classify(tokens, None).tobytes() == engine.model.classify(tokens).tobytes()
    check_gradients(tmp_path / "checkpoint", engine, tokens)


def write_pairs(path: Path) -> Path:
    rows = [
        {"anchor": "quick fox", "positive": TEXTS[0], "negative": TEXTS[2]},
        {"anchor": "how are you", "positive": TEXTS[1]},
        {"anchor": "naive cafe", "positive": TEXTS[2], "negative": TEXTS[1]},
        {"anchor": "same bits", "positive": TEXTS[3]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def write_labels(path: Path) -> Path:
    rows = [
        {"text": "quick fox", "pair": TEXTS[0], "label": 1},
        {"text": "quick fox", "pair": TEXTS[1], "label": 0},
        {"text": "how are you", "pair": TEXTS[1], "label": 2},
        {"text": "same bits", "pair": TEXTS[3], "label": 1},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def model_path(checkpoint: Path, engine: DllmEngine) -> Path:
    return checkpoint.parent / ("cross.dllm" if engine.model.config.classifier_labels else "model.dllm")


def test_finetune_runs_are_golden(embedder, cross, tmp_path):
    from golden_values import DEBERTA_FINETUNE_FINGERPRINT

    from etalii_dllm.training import AdamWConfig, FineTuner, RunConfig
    from etalii_dllm.training.encoder_data import EncoderData, read_examples

    for kind, (checkpoint, engine) in (("embedding", embedder), ("classifier", cross)):
        data_file = write_pairs(tmp_path / "pairs.jsonl") if kind == "embedding" else write_labels(tmp_path / "l.jsonl")
        data = EncoderData.from_records(read_examples(data_file, kind), engine, kind, MAX_SEQ_LENGTH)
        run = RunConfig(3, 2, MAX_SEQ_LENGTH, 1, AdamWConfig(1e-3), objective=kind)
        model_file = ModelFile(model_path(checkpoint, engine))
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
        assert fingerprint == DEBERTA_FINETUNE_FINGERPRINT[kind]
        table = "relative_embedding.weight"
        tuned = np.asarray(ModelFile(tmp_path / f"{kind}.dllm").tensors[table])
        assert (tuned != np.asarray(model_file.tensors[table])).any()  # the position table trains


def test_lora_on_deberta(embedder, cross, tmp_path, capsys):
    import re

    from etalii_dllm.cli import main as cli
    from etalii_dllm.importing.safetensors import SafetensorsFile
    from etalii_dllm.lora import AdapterError, peft_key, read_peft, target_modules

    checkpoint, engine = embedder
    path = model_path(checkpoint, engine)
    config = engine.model.config
    pairs = write_pairs(tmp_path / "pairs.jsonl")
    common = ["finetune", str(path), "--data", str(pairs), "--steps", "2", "--batch-size", "2"]
    common += ["--learning-rate", "1e-2", "--lora-rank", "2"]
    assert cli([*common, "-o", str(tmp_path / "merged.dllm"), "--adapter-output", str(tmp_path / "adapter")]) == 0
    capsys.readouterr()
    settings = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert settings["task_type"] == "FEATURE_EXTRACTION"
    stored = {t.name: t.to_float32() for t in SafetensorsFile(tmp_path / "adapter" / "adapter_model.safetensors")}
    assert stored["base_model.model.encoder.layer.0.attention.self.query_proj.lora_A.weight"].shape == (2, 32)
    assert stored["base_model.model.encoder.layer.1.intermediate.dense.lora_B.weight"].shape == (64, 2)
    modules = {name for name, _ in transformers_model(checkpoint).named_modules()}
    adapted = {key.removeprefix("base_model.model.").rsplit(".lora_", 1)[0] for key in stored}
    assert adapted <= modules
    assert {m for m in modules if re.fullmatch(settings["target_modules"], m)} == adapted
    merged = DllmEngine.from_model_file(tmp_path / "merged.dllm").embed("hello there").vector
    loaded = DllmEngine.from_model_file(path, adapter=tmp_path / "adapter").embed("hello there").vector
    assert merged.tobytes() == loaded.tobytes()
    assert merged.tobytes() != engine.embed("hello there").vector.tobytes()
    import_model(tmp_path / "adapter", tmp_path / "imported.dllm", base=path, licence="mit")
    assert ModelFile(tmp_path / "imported.dllm").fingerprint == ModelFile(tmp_path / "merged.dllm").fingerprint
    # BERT's module names are not DeBERTa's
    from etalii_dllm.importing.safetensors import write_safetensors as write_tensors

    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 2}), encoding="utf-8")
    key = "base_model.model.encoder.layer.0.attention.self.query.lora_A.weight"
    write_tensors(tmp_path / "bad" / "adapter_model.safetensors", {key: np.zeros((2, 32), np.float32)})
    with pytest.raises(AdapterError, match="unsupported adapter tensor"):
        read_peft(tmp_path / "bad", config)
    # a cross-encoder's adapter sits under deberta.; the head stays frozen
    checkpoint, scorer = cross
    assert peft_key("layers.1.attention.v.weight.lora_b", scorer.model.config) == (
        "base_model.model.deberta.encoder.layer.1.attention.self.value_proj.lora_B.weight"
    )
    from etalii_dllm.lora import LoraConfig

    assert target_modules(config, LoraConfig(2, 2.0, ("q", "k"))) == (
        r".*encoder\.layer\.\d+\.(?:attention\.self\.query_proj|attention\.self\.key_proj)"
    )
    labels = write_labels(tmp_path / "labels.jsonl")
    args = ["finetune", str(model_path(checkpoint, scorer)), "--data", str(labels), "--steps", "1"]
    assert cli([*args, "--lora-rank", "2", "--adapter-output", str(tmp_path / "cross-adapter")]) == 0
    capsys.readouterr()
    assert json.loads((tmp_path / "cross-adapter" / "adapter_config.json").read_text())["task_type"] == "SEQ_CLS"
    _, adapters = read_peft(tmp_path / "cross-adapter", scorer.model.config)
    assert "layers.0.attention.q.weight.lora_a" in adapters


def test_export_round_trips(embedder, cross, tmp_path):
    from etalii_dllm.exporting import ExportError, export_gguf, export_safetensors

    torch = pytest.importorskip("torch")
    for name, (checkpoint, engine) in (("embedder", embedder), ("cross", cross)):
        original = ModelFile(model_path(checkpoint, engine))
        export_safetensors(original, tmp_path / name)
        exported = json.loads((tmp_path / name / "config.json").read_text())
        assert exported["model_type"] == "deberta-v2" and exported["position_buckets"] == 8
        import_model(tmp_path / name, tmp_path / f"{name}.dllm", repository=f"example/{name}")
        again = ModelFile(tmp_path / f"{name}.dllm")
        assert again.fingerprint == original.fingerprint and again.config == original.config
        assert (again.embedding, again.classifier) == (original.embedding, original.classifier)
        model = transformers_model(tmp_path / name, classification=name == "cross")
        tokens = long_tokens(engine)
        types = [0] * 32 + [1] * 32 if name == "cross" else [0] * 64
        with torch.no_grad():
            output = model(input_ids=torch.tensor([tokens]), token_type_ids=torch.tensor([types]))
        if name == "cross":
            np.testing.assert_allclose(engine.model.classify(tokens, types), output.logits[0].numpy(), atol=2e-5)
        else:
            np.testing.assert_allclose(
                engine.model.hidden_states(tokens), output.last_hidden_state[0].numpy(), atol=2e-5
            )
        with pytest.raises(ExportError, match="GGUF"):
            export_gguf(original, tmp_path / f"{name}.gguf")
    # a table reaching every distance (no buckets) is written without position_buckets
    write_deberta_checkpoint(tmp_path / "plain", position_buckets=-1)
    import_model(tmp_path / "plain", tmp_path / "plain.dllm")
    export_safetensors(ModelFile(tmp_path / "plain.dllm"), tmp_path / "plain-out")
    assert json.loads((tmp_path / "plain-out" / "config.json").read_text())["position_buckets"] == -1
