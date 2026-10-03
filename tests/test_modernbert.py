"""ModernBERT encoders (issues #357-#359): bidirectional local attention in the kernels, the ModernBERT forward pass
against transformers, sentence-transformers embedders and sequence-classification cross-encoders imported from
Hugging Face, the reference implementation and dllm verify --reference, and the refusals."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from model_fixtures import write_safetensors
from test_encoders import MODULES

from etalii_dllm import numerics, reference
from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.importing.importer import modernbert_config

tokenizers = pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parent.parent
TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Hello, world! How are you?",
    "naïve café façade",
]
MAX_SEQ_LENGTH = 24
PATHS: dict[int, Path] = {}
"""The model file of each engine the fixtures serve."""
MODERNBERT_CONFIG = {
    "model_type": "modernbert",
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 48,
    "max_position_embeddings": 64,
    "global_attn_every_n_layers": 3,
    "local_attention": 8,
    "global_rope_theta": 160000.0,
    "local_rope_theta": 10000.0,
    "norm_eps": 1e-5,
    "norm_bias": False,
    "attention_bias": False,
    "mlp_bias": False,
    "classifier_bias": False,
    "hidden_activation": "gelu",
    "classifier_activation": "gelu",
    "pad_token_id": 3,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "cls_token_id": 1,
    "sep_token_id": 2,
}


MERGE_WORDS = ("Ġthe", "Ġand", "Ġquick", "Ġfox", "Ġmodel", "Ġare", "Ġyou", "Ġcafe", "Ġhow", "Ġlazy", "Ġdog", "ing")


def modernbert_tokenizer():
    """A byte-level BPE tokenizer laid out like ModernBERT's, with its ``[CLS] $A [SEP]`` template. The vocabulary is
    written out rather than trained: `tokenizers`' BpeTrainer is not deterministic, and the goldens depend on it."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

    tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", *sorted(pre_tokenizers.ByteLevel.alphabet())]
    merges: list[tuple[str, str]] = []
    for word in MERGE_WORDS:
        left = word[0]
        for character in word[1:]:
            if left + character not in tokens:
                merges.append((left, character))
                tokens.append(left + character)
            left += character
    tokens += [f"[unused{i}]" for i in range(320 - len(tokens))]
    tokenizer = Tokenizer(models.BPE({token: i for i, token in enumerate(tokens)}, merges, unk_token="[UNK]"))
    tokenizer.add_special_tokens(["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:0 [SEP]:0",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    return tokenizer


def modernbert_weights(config: dict, seed: int = 11, labels: int = 0) -> dict[str, np.ndarray]:
    """Float32 weights for a ModernBERT config, keyed by Hugging Face ``ModernBertModel`` name (plus the head and
    classifier with ``labels``)."""
    hidden, inner = config["hidden_size"], config["intermediate_size"]
    counter = iter(range(seed * 1000, seed * 1000 + 1000))

    def gaussian(*shape: int) -> np.ndarray:
        return numerics.fill_gaussian(next(counter), int(np.prod(shape))).reshape(shape)

    def matrix(rows: int, columns: int, scale: float = 0.2) -> np.ndarray:
        return (gaussian(rows, columns) * np.float32(scale)).astype(np.float32)

    def norm(size: int) -> np.ndarray:
        return (np.float32(1.0) + gaussian(size) * np.float32(0.1)).astype(np.float32)

    weights = {
        "embeddings.tok_embeddings.weight": matrix(config["vocab_size"], hidden, 1.0),
        "embeddings.norm.weight": norm(hidden),
    }
    for i in range(config["num_hidden_layers"]):
        p = f"layers.{i}."
        if i:
            weights[p + "attn_norm.weight"] = norm(hidden)
        weights[p + "attn.Wqkv.weight"] = matrix(3 * hidden, hidden)
        weights[p + "attn.Wo.weight"] = matrix(hidden, hidden)
        weights[p + "mlp_norm.weight"] = norm(hidden)
        weights[p + "mlp.Wi.weight"] = matrix(2 * inner, hidden)
        weights[p + "mlp.Wo.weight"] = matrix(hidden, inner)
    weights["final_norm.weight"] = norm(hidden)
    if labels:
        weights = {"model." + name: values for name, values in weights.items()}
        weights["head.dense.weight"] = matrix(hidden, hidden)
        weights["head.norm.weight"] = norm(hidden)
        weights["classifier.weight"] = matrix(labels, hidden, 0.3)
        weights["classifier.bias"] = np.full(labels, 0.125, dtype=np.float32)
    return weights


def write_modernbert_checkpoint(
    directory: Path,
    *,
    pooling: str | None = "pooling_mode_mean_tokens",
    labels: int = 0,
    classifier_pooling: str = "mean",
    overrides: dict | None = None,
) -> dict:
    """A ModernBERT checkpoint: a sentence-transformers embedder, or with ``labels`` a
    ``ModernBertForSequenceClassification`` cross-encoder."""
    tokenizer = modernbert_tokenizer()
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        **MODERNBERT_CONFIG,
        "architectures": ["ModernBertForSequenceClassification" if labels else "ModernBertModel"],
        "vocab_size": tokenizer.get_vocab_size(),
        "classifier_pooling": classifier_pooling,
        **(overrides or {}),
    }
    if labels:
        config["id2label"] = {str(i): f"LABEL_{i}" for i in range(labels)}
        config["label2id"] = {f"LABEL_{i}": i for i in range(labels)}
    weights = modernbert_weights(config, labels=labels)
    write_safetensors(
        directory / "model.safetensors", {name: ("F32", values) for name, values in weights.items()}, {"format": "pt"}
    )
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "tokenizer.json").write_text(tokenizer.to_str(), encoding="utf-8")
    tokenizer_config = {
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "mask_token": "[MASK]",
        "model_max_length": MAX_SEQ_LENGTH,
    }
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    (directory / "README.md").write_bytes(b"---\nlicense: apache-2.0\n---\n\n# Tiny ModernBERT\n")
    if pooling is not None and not labels:
        (directory / "modules.json").write_text(json.dumps(MODULES), encoding="utf-8")
        (directory / "1_Pooling").mkdir(exist_ok=True)
        settings = {"word_embedding_dimension": config["hidden_size"], pooling: True}
        (directory / "1_Pooling" / "config.json").write_text(json.dumps(settings), encoding="utf-8")
        limits = {"max_seq_length": MAX_SEQ_LENGTH, "do_lower_case": False}
        (directory / "sentence_bert_config.json").write_text(json.dumps(limits), encoding="utf-8")
    return config


@pytest.fixture(scope="module")
def embedder(tmp_path_factory) -> tuple[Path, DllmEngine]:
    directory = tmp_path_factory.mktemp("modernbert")
    write_modernbert_checkpoint(directory / "checkpoint")
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-modernbert")
    engine = DllmEngine.from_model_file(directory / "model.dllm")
    PATHS[id(engine)] = directory / "model.dllm"
    return directory / "checkpoint", engine


@pytest.fixture(scope="module", params=["mean", "cls"])
def cross(request, tmp_path_factory) -> tuple[Path, DllmEngine]:
    directory = tmp_path_factory.mktemp(f"modernbert-cross-{request.param}")
    write_modernbert_checkpoint(directory / "checkpoint", labels=1, classifier_pooling=request.param)
    import_model(directory / "checkpoint", directory / "cross.dllm", repository="example/tiny-modernbert-reranker")
    engine = DllmEngine.from_model_file(directory / "cross.dllm")
    PATHS[id(engine)] = directory / "cross.dllm"
    return directory / "checkpoint", engine


def transformers_model(directory: Path, classification: bool = False):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    kind = transformers.AutoModelForSequenceClassification if classification else transformers.AutoModel
    return kind.from_pretrained(directory, attn_implementation="eager", dtype="float32").eval()


# Bidirectional local attention


def masked_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, window: int) -> np.ndarray:
    """Attention in float64 over the keys closer than ``window`` to each query on either side."""
    scores = np.einsum("thd,jhd->htj", q.astype(np.float64), k.astype(np.float64)) / np.sqrt(q.shape[2])
    index = np.arange(len(q))
    visible = np.abs(index[:, None] - index[None, :]) < window
    scores = np.where(visible[None], scores, -np.inf)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("htj,jhd->thd", p, v.astype(np.float64))


@pytest.mark.parametrize("window", [1, 2, 3, 9, 20])
def test_bidirectional_local_attention(window):
    q, k, v = (numerics.fill_gaussian(seed, 9 * 2 * 4).reshape(9, 2, 4) for seed in (1, 2, 3))
    out = numerics.attention(q, k, v, causal=False, window=window).numpy()
    assert np.allclose(out, masked_attention(q, k, v, window), atol=1e-6)
    assert np.array_equal(out, reference.attention(q, k, v, causal=False, window=window))
    scalar = numerics._kernels.attention(q, k, v, 0.5, False, 0, window, 0.0, True)
    assert np.array_equal(out, scalar)
    if window >= 9:
        assert np.array_equal(out, numerics.attention(q, k, v, causal=False).numpy())
    weights = numerics.attention_weights(q, k, causal=False, window=window).numpy()
    index = np.arange(9)
    hidden = np.abs(index[:, None] - index[None, :]) >= window
    assert not weights.transpose(1, 0, 2)[:, hidden].any()


def test_bidirectional_local_attention_gradients():
    q, k, v, dout = (numerics.fill_gaussian(seed, 7 * 2 * 4).reshape(7, 2, 4) for seed in (4, 5, 6, 7))
    dq, dk, dv = (t.numpy() for t in numerics.attention_backward(q, k, v, dout, causal=False, window=3))
    step = 1e-3
    for array, grad in ((q, dq), (k, dk), (v, dv)):
        for index in [(0, 0, 0), (3, 1, 2), (6, 0, 3)]:
            plus, minus = array.copy(), array.copy()
            plus[index] += step
            minus[index] -= step
            args = {id(q): 0, id(k): 1, id(v): 2}[id(array)]

            def loss(changed: np.ndarray, args: int = args) -> float:
                inputs = [q, k, v]
                inputs[args] = changed
                out = masked_attention(*inputs, window=3)
                return float((out * dout.astype(np.float64)).sum())

            numeric = (loss(plus) - loss(minus)) / (2 * step)
            assert grad[index] == pytest.approx(numeric, rel=2e-3, abs=2e-4)


# The ModernBERT encoder


def test_config_maps_both_layouts():
    config = modernbert_config({**MODERNBERT_CONFIG, "vocab_size": 300})
    assert config.family == "modernbert" and config.is_encoder
    assert config.sliding_window == 5 and config.sliding_window_layers == (1, 2)
    assert (config.rope_theta, config.local_rope_theta) == (160000.0, 10000.0)
    assert [config.window(i) for i in range(4)] == [None, 5, 5, None]
    v5 = {
        key: value
        for key, value in MODERNBERT_CONFIG.items()
        if key not in ("global_attn_every_n_layers", "global_rope_theta", "local_rope_theta")
    }
    v5 |= {
        "vocab_size": 300,
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    }
    assert modernbert_config(v5) == config
    shapes = config.tensor_shapes()
    assert "layers.0.attention_norm.weight" not in shapes and "layers.1.attention_norm.weight" in shapes
    assert not any(name.endswith(".bias") for name in shapes)
    assert TransformerConfig.from_dict(config.to_dict()) == config
    every = modernbert_config({**MODERNBERT_CONFIG, "vocab_size": 300, "global_attn_every_n_layers": 1})
    assert every.sliding_window is None and every.local_rope_theta is None


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"norm_bias": True}, "norm_bias"),
        ({"attention_bias": True}, "attention_bias"),
        ({"hidden_activation": "relu"}, "activation"),
        ({"classifier_activation": "gelu_pytorch_tanh"}, "classifier activation"),
        ({"layer_types": ["full_attention"]}, "layer_types"),
        ({"rope_scaling": {"rope_type": "linear", "factor": 2.0}}, "RoPE scaling"),
        ({"classifier_pooling": "max"}, "classifier pooling"),
        ({"hidden_size": 30}, "head_dim"),
    ],
)
def test_config_refusals(override, message):
    with pytest.raises(ModelImportError, match=message):
        modernbert_config({**MODERNBERT_CONFIG, "vocab_size": 300, **override})
    with pytest.raises(ModelImportError, match="context-length"):
        modernbert_config({**MODERNBERT_CONFIG, "vocab_size": 300}, context_length=128)


def test_config_validation():
    config = modernbert_config({**MODERNBERT_CONFIG, "vocab_size": 300})
    values = config.to_dict()
    for change, message in (
        ({"type_vocabulary_size": 2}, "token types"),
        ({"classifier_pooling": "max"}, "classifier_pooling"),
        ({"family": "llama", "activation": "gelu_tanh", "classifier_pooling": "cls"}, "encoders only"),
        ({"family": "bert", "type_vocabulary_size": 2, "classifier_pooling": "cls"}, "modernbert only"),
    ):
        with pytest.raises(ValueError, match=message):
            TransformerConfig.from_dict({**values, **change})


def test_states_match_transformers(embedder):
    checkpoint, engine = embedder
    model = transformers_model(checkpoint)
    import torch

    tokenizer = modernbert_tokenizer()
    tokenizer.enable_truncation(MAX_SEQ_LENGTH)
    for text in TEXTS:
        ids = tokenizer.encode(text).ids
        assert engine.embedding_tokens(text) == ids
        with torch.no_grad():
            expected = model(input_ids=torch.tensor([ids])).last_hidden_state[0].numpy()
        ours = engine.model.hidden_states(ids)
        assert len(ids) > 2 * 4  # longer than the local window, so the local layers really are local
        assert np.allclose(ours, expected, atol=2e-5), np.abs(ours - expected).max()


def test_embeddings_match_mean_pooling(embedder):
    checkpoint, engine = embedder
    model = transformers_model(checkpoint)
    import torch

    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    for text in [*TEXTS, " ".join(TEXTS * 3)]:
        ids = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"]
        assert engine.embedding_tokens(text) == ids
        with torch.no_grad():
            states = model(input_ids=torch.tensor([ids])).last_hidden_state[0].numpy().astype(np.float64)
        mean = states.mean(axis=0)
        vector = engine.embed(text).vector
        np.testing.assert_allclose(vector, mean / np.linalg.norm(mean), rtol=1e-4, atol=1e-5)
        assert engine.embed(text).vector.tobytes() == vector.tobytes()


def test_thread_invariance(embedder):
    _, engine = embedder
    ids = engine.embedding_tokens(TEXTS[0])
    states = engine.model.hidden_states(ids)
    for threads in (1, 3):
        numerics.set_threads(threads)
        try:
            assert engine.model.hidden_states(ids).tobytes() == states.tobytes()
        finally:
            numerics.set_threads(0)


def test_reference_implementation_and_verify(embedder, cross):
    from etalii_dllm import verify

    _, engine = embedder
    twin = reference.ReferenceEncoder.from_engine(engine)
    for tokens in (engine.embedding_tokens(TEXTS[0]), [1, 7, 3, 3, 9, 2]):
        assert engine.model.hidden_states(tokens).tobytes() == twin.hidden_states(tokens).tobytes()
    assert verify.check_reference(engine).equal
    _, scorer = cross
    assert verify.check_reference(scorer).equal


def test_cross_encoder_matches_transformers(cross):
    checkpoint, engine = cross
    model = transformers_model(checkpoint, classification=True)
    import torch

    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    query = "Where does the fox jump?"
    for document in [*TEXTS, " ".join(TEXTS * 2)]:
        encoded = tokenizer(query, document, truncation="longest_first", max_length=MAX_SEQ_LENGTH)
        tokens, _ = engine.classification_tokens(query, document)
        assert tokens == encoded["input_ids"]
        with torch.no_grad():
            expected = model(input_ids=torch.tensor([tokens])).logits[0].numpy()
        verdict = engine.classify(query, document)
        np.testing.assert_allclose(verdict.logits, expected, rtol=1e-4, atol=2e-5)
        twin = reference.ReferenceEncoder.from_engine(engine)
        assert np.asarray(verdict.logits, dtype=np.float32).tobytes() == twin.classify(tokens).tobytes()


def test_reranker_with_a_modernbert_cross_encoder(cross):
    from etalii_dllm.reranking import Reranker

    _, engine = cross
    reranker = Reranker(engine)
    assert reranker.cross_encoder
    expected = [engine.classify("fox", d).scores[0] for d in TEXTS]
    order = sorted(range(len(TEXTS)), key=lambda i: (-expected[i], i))
    assert reranker.rerank("fox", TEXTS) == [(i, expected[i]) for i in order]


# Fine-tuning, LoRA and export (#360)


def our_names(name: str, values: np.ndarray, config) -> dict[str, np.ndarray]:
    """transformers' ModernBERT parameter (or its gradient) under our names, the fused ones split."""
    from etalii_dllm.importing.importer import _modernbert_tensors

    class Stored:
        def __init__(self) -> None:
            self.name, self.dtype, self.shape = name, "F32", tuple(values.shape)

        def to_float32(self) -> np.ndarray:
            return np.asarray(values, dtype=np.float32)

    return {
        ours: np.asarray(source.load(), dtype=np.float64)
        for ours, source in _modernbert_tensors(Stored(), config, bool(config.classifier_labels)).items()
    }


def modernbert_autograd(directory: Path, tokens: list[int], upstream: np.ndarray, config):
    """transformers' float64 gradients of ``sum(upstream * output)`` under our names."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    kind = transformers.AutoModelForSequenceClassification if config.classifier_labels else transformers.AutoModel
    model = kind.from_pretrained(directory, attn_implementation="eager").double().eval()
    output = model(torch.tensor([tokens]))
    values = output.logits[0] if config.classifier_labels else output.last_hidden_state[0]
    (values * torch.tensor(upstream, dtype=torch.float64)).sum().backward()
    grads: dict[str, np.ndarray] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grads |= our_names(name, parameter.grad.numpy(), config)
    return values.detach().numpy(), grads


def test_gradients_match_autograd(embedder, cross):
    from etalii_dllm.training.encoder_backprop import EncoderGradients

    for directory, engine in (embedder, cross):
        model = engine.model
        weights = {name: np.asarray(values, dtype=np.float32) for name, values in model.tensors.items()}
        config = model.config
        tokens = engine.embedding_tokens(TEXTS[0]) if not config.classifier_labels else [1, 9, 40, 5, 77, 2, 51, 2]
        gradients = EncoderGradients(config)
        if config.classifier_labels:
            result = gradients.classify(weights, tokens)
            assert result.logits.tobytes() == model.classify(tokens).tobytes()
            upstream = np.array([0.75], dtype=np.float32)
            ours = gradients.gradients(weights, result, dlogits=upstream)
        else:
            result = gradients.encode(weights, tokens)
            assert result.states.tobytes() == model.hidden_states(tokens).tobytes()
            upstream = numerics.fill_gaussian(4, len(tokens) * 32).reshape(len(tokens), 32)
            ours = gradients.gradients(weights, result, upstream)
        _, theirs = modernbert_autograd(directory, tokens, upstream, config)
        assert set(ours) == set(theirs)
        for name, expected in theirs.items():
            scale = max(float(np.abs(expected).max()), 1e-3)
            assert np.abs(ours[name] - expected).max() <= 2e-4 * scale, name


def write_pairs(path: Path) -> Path:
    rows = [
        {"anchor": "quick fox", "positive": TEXTS[0], "negative": TEXTS[2]},
        {"anchor": "how are you", "positive": TEXTS[1]},
        {"anchor": "naive cafe", "positive": TEXTS[2], "negative": TEXTS[1]},
        {"anchor": "a lazy dog", "positive": "over the lazy dog"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def write_labels(path: Path) -> Path:
    rows = [
        {"text": "quick fox", "pair": TEXTS[0], "label": 1},
        {"text": "quick fox", "pair": TEXTS[1], "label": 0},
        {"text": "how are you", "pair": TEXTS[1], "label": 1},
        {"text": "naive cafe", "pair": TEXTS[0], "label": 0.25},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def model_path(engine: DllmEngine) -> Path:
    return PATHS[id(engine)]


def test_finetune_runs_are_golden(embedder, cross, tmp_path):
    from golden_values import MODERNBERT_FINETUNE_FINGERPRINT

    from etalii_dllm.modelfile import ModelFile
    from etalii_dllm.training import AdamWConfig, FineTuner, RunConfig
    from etalii_dllm.training.encoder_data import EncoderData, read_examples

    for kind, (_, engine) in (("embedding", embedder), ("classifier", cross)):
        if kind == "classifier" and engine.model.config.classifier_pooling != "mean":
            continue
        data_file = write_pairs(tmp_path / "pairs.jsonl") if kind == "embedding" else write_labels(tmp_path / "l.jsonl")
        data = EncoderData.from_records(read_examples(data_file, kind), engine, kind, MAX_SEQ_LENGTH)
        run = RunConfig(3, 2, MAX_SEQ_LENGTH, 1, AdamWConfig(1e-3), objective=kind)
        model_file = ModelFile(model_path(engine))
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
        assert fingerprint == MODERNBERT_FINETUNE_FINGERPRINT[kind]
        assert first.losses[-1] < first.losses[0] or kind == "classifier"


def test_lora_on_modernbert(embedder, cross, tmp_path, capsys):
    import re

    from etalii_dllm.cli import main as cli
    from etalii_dllm.lora import AdapterError, LoraConfig, peft_key, read_peft, target_modules, target_weights
    from etalii_dllm.modelfile import ModelFile

    transformers = pytest.importorskip("transformers")
    directory, engine = embedder
    path = model_path(engine)
    config = engine.model.config
    pairs = write_pairs(tmp_path / "pairs.jsonl")
    common = ["finetune", str(path), "--data", str(pairs), "--steps", "2", "--batch-size", "2"]
    common += ["--learning-rate", "1e-2", "--lora-rank", "2"]
    assert cli([*common, "-o", str(tmp_path / "merged.dllm"), "--adapter-output", str(tmp_path / "adapter")]) == 0
    capsys.readouterr()
    settings = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert settings["task_type"] == "FEATURE_EXTRACTION"
    from etalii_dllm.importing.safetensors import SafetensorsFile

    stored = {
        tensor.name: tensor.to_float32()
        for tensor in SafetensorsFile(tmp_path / "adapter" / "adapter_model.safetensors")
    }
    assert stored["base_model.model.layers.0.attn.Wqkv.lora_A.weight"].shape == (2, 32)
    assert stored["base_model.model.layers.0.attn.Wqkv.lora_B.weight"].shape == (96, 2)
    assert stored["base_model.model.layers.2.mlp.Wi.lora_B.weight"].shape == (96, 2)
    modules = {name for name, _ in transformers.AutoModel.from_pretrained(directory).named_modules()}
    adapted = {key.removeprefix("base_model.model.").rsplit(".lora_", 1)[0] for key in stored}
    assert adapted <= modules
    assert {m for m in modules if re.fullmatch(settings["target_modules"], m)} == adapted
    # PEFT's fused module: W_qkv + scale * B @ A, its rows q, k, v in order
    fused = stored["base_model.model.layers.0.attn.Wqkv.lora_B.weight"].astype(np.float64)
    fused = fused @ stored["base_model.model.layers.0.attn.Wqkv.lora_A.weight"].astype(np.float64)
    tuned, base = ModelFile(tmp_path / "merged.dllm").tensors, engine.model.tensors
    for index, part in enumerate("qkv"):
        name = f"layers.0.attention.{part}.weight"
        delta = np.asarray(tuned[name], dtype=np.float64) - np.asarray(base[name], dtype=np.float64)
        np.testing.assert_allclose(delta, fused[32 * index : 32 * (index + 1)] * settings["lora_alpha"] / 2, atol=1e-6)
    lora, adapters = read_peft(tmp_path / "adapter", config)
    assert lora.rank == 2 and "layers.0.attention.k.weight.lora_a" not in adapters
    assert len(adapters) == 4 * (4 + 7)  # per layer: four A (one per module), seven B (one per part)
    merged = DllmEngine.from_model_file(tmp_path / "merged.dllm").embed("hello there").vector
    loaded = DllmEngine.from_model_file(path, adapter=tmp_path / "adapter").embed("hello there").vector
    assert merged.tobytes() == loaded.tobytes()
    import_model(tmp_path / "adapter", tmp_path / "imported.dllm", base=path, licence="apache-2.0")
    assert ModelFile(tmp_path / "imported.dllm").fingerprint == ModelFile(tmp_path / "merged.dllm").fingerprint
    # q, k and v are one module, as are gate and up
    with pytest.raises(AdapterError, match="fuses q, k and v"):
        target_weights(config, LoraConfig(2, 2.0, ("q",)))
    with pytest.raises(AdapterError, match="fuses gate and up"):
        target_weights(config, LoraConfig(2, 2.0, ("gate", "o")))
    assert target_modules(config, LoraConfig(2, 2.0, ("o", "down"))) == r".*layers\.\d+\.(?:attn\.Wo|mlp\.Wo)"
    # a cross-encoder's adapter sits under model.; the head stays frozen
    _, scorer = cross
    assert peft_key("layers.1.mlp.up.weight.lora_b", scorer.model.config) == (
        "base_model.model.model.layers.1.mlp.Wi.lora_B.weight"
    )
    labels = write_labels(tmp_path / "labels.jsonl")
    args = ["finetune", str(model_path(scorer)), "--data", str(labels), "--steps", "1", "--lora-rank", "2"]
    assert cli([*args, "--adapter-output", str(tmp_path / "cross-adapter")]) == 0
    capsys.readouterr()
    assert json.loads((tmp_path / "cross-adapter" / "adapter_config.json").read_text())["task_type"] == "SEQ_CLS"
    read_peft(tmp_path / "cross-adapter", scorer.model.config)
    with pytest.raises(AdapterError, match="unsupported adapter tensor"):
        read_peft(tmp_path / "cross-adapter", config)  # an embedder's modules have no model. prefix


def test_export_round_trips(embedder, cross, tmp_path):
    from etalii_dllm.exporting import ExportError, export_gguf, export_safetensors
    from etalii_dllm.modelfile import ModelFile

    torch = pytest.importorskip("torch")
    for name, (_, engine) in (("embedder", embedder), ("cross", cross)):
        path = model_path(engine)
        original = ModelFile(path)
        export_safetensors(original, tmp_path / name)
        exported = json.loads((tmp_path / name / "config.json").read_text())
        assert exported["model_type"] == "modernbert" and exported["global_attn_every_n_layers"] == 3
        import_model(tmp_path / name, tmp_path / f"{name}.dllm", repository=f"example/{name}")
        again = ModelFile(tmp_path / f"{name}.dllm")
        assert again.fingerprint == original.fingerprint and again.config == original.config
        assert (again.embedding, again.classifier) == (original.embedding, original.classifier)
        model = transformers_model(tmp_path / name, classification=name == "cross")
        tokens = engine.embedding_tokens(TEXTS[1])
        with torch.no_grad():
            output = model(input_ids=torch.tensor([tokens]))
        if name == "cross":
            np.testing.assert_allclose(engine.model.classify(tokens), output.logits[0].numpy(), atol=2e-5)
        else:
            np.testing.assert_allclose(
                engine.model.hidden_states(tokens), output.last_hidden_state[0].numpy(), atol=2e-5
            )
        with pytest.raises(ExportError, match="GGUF"):
            export_gguf(original, tmp_path / f"{name}.gguf")


def test_modernbert_refusals(embedder, cross, tmp_path):
    from etalii_dllm.encoder_export import hf_encoder_config
    from etalii_dllm.importing.safetensors import write_safetensors as write_tensors
    from etalii_dllm.lora import AdapterError, read_peft
    from etalii_dllm.modelfile import ModelFile
    from etalii_dllm.training.encoder_backprop import EncoderGradients

    _, engine = embedder
    config = engine.model.config
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 2}), encoding="utf-8")
    for key, shape, message in (
        ("base_model.model.layers.9.attn.Wqkv.lora_A.weight", (2, 32), "does not fit the model"),
        ("base_model.model.layers.0.attn.Wq.lora_A.weight", (2, 32), "unsupported adapter tensor"),
        ("base_model.model.layers.0.attn.Wqkv.lora_B.weight", (95, 2), "has shape"),
    ):
        write_tensors(tmp_path / "adapter_model.safetensors", {key: np.zeros(shape, np.float32)})
        with pytest.raises(AdapterError, match=message):
            read_peft(tmp_path, config)
    weights = {name: np.asarray(values, dtype=np.float32) for name, values in engine.model.tensors.items()}
    gradients = EncoderGradients(config)
    result = gradients.encode(weights, [1, 5, 2])
    with pytest.raises(ValueError, match="classification head"):
        gradients.gradients(weights, result, dlogits=[1.0])
    for change, message in (
        ({"activation": "silu"}, "gelu or gelu_tanh"),
        ({"family": "bert"}, "bert needs token types"),
        ({"kv_heads": 2}, "grouped-query"),
    ):
        with pytest.raises(ValueError, match=message):
            TransformerConfig.from_dict({**config.to_dict(), **change})
    # a layer pattern that is not "global every n layers" is written as layer_types, all-global as every 1
    model = ModelFile(PATHS[id(engine)])
    for layers, expected in (((0, 3), "layer_types"), (None, "global_attn_every_n_layers")):
        import dataclasses

        changed = dataclasses.replace(
            config,
            sliding_window=None if layers is None else config.sliding_window,
            sliding_window_layers=layers,
            local_rope_theta=None if layers is None else config.local_rope_theta,
        )

        stand_in = SimpleNamespace(config=changed, classifier=None, tokenizer={"tokenizer_config": {"pad_token": {}}})
        document = hf_encoder_config(stand_in)  # type: ignore[arg-type]
        assert expected in document and document["pad_token_id"] == 0
    assert hf_encoder_config(model)["pad_token_id"] == 3


def test_checkpoint_tensor_refusals():
    from etalii_dllm.importing.importer import _modernbert_tensors

    config = modernbert_config({**MODERNBERT_CONFIG, "vocab_size": 300})

    def stored(name: str, shape: tuple[int, ...] = (96, 32), dtype: str = "F32") -> SimpleNamespace:
        return SimpleNamespace(name=name, shape=shape, dtype=dtype, to_float32=lambda: np.zeros(shape, np.float32))

    assert _modernbert_tensors(stored("decoder.bias", (300,)), config, False) == {}
    assert _modernbert_tensors(stored("head.dense.weight", (32, 32)), config, False) == {}
    for tensor, message in (
        (stored("layers.0.attn.Wqkv.weight", dtype="I8"), "dtype"),
        (stored("embeddings.position_embeddings.weight"), "unexpected tensor"),
        (stored("layers.0.attn.Wq.weight"), "unexpected tensor"),
        (stored("layers.0.attn.Wqkv.weight", (95, 32)), "does not split"),
    ):
        with pytest.raises(ModelImportError, match=message):
            _modernbert_tensors(tensor, config, False)
    with pytest.raises(ModelImportError, match="layers must be positive"):
        modernbert_config({**MODERNBERT_CONFIG, "vocab_size": 300, "num_hidden_layers": 0})
