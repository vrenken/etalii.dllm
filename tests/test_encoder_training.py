"""Fine-tuning encoders (issues #352-#354): the LayerNorm and GELU backward kernels, encoder gradients against
torch autograd and finite differences, sentence-vector pooling, the contrastive and cross-encoder objectives, runs
that resume and replay bit for bit, and LoRA adapters for encoders in the PEFT format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from golden_values import ENCODER_FINETUNE_FINGERPRINT
from test_cross_encoders import write_cross_encoder
from test_encoders import write_bert_checkpoint
from test_roberta_encoders import write_roberta_checkpoint

from etalii_dllm import numerics
from etalii_dllm.cli import main as cli
from etalii_dllm.encoder import Encoder
from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.importing.importer import _bert_name
from etalii_dllm.lora import AdapterError, LoraConfig, peft_key, read_peft, target_modules, target_weights
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.training import AdamWConfig, FineTuner, RunConfig, TrainingDataError
from etalii_dllm.training.encoder_backprop import EncoderGradients, pool, pool_backward
from etalii_dllm.training.encoder_data import EncoderData, read_examples
from etalii_dllm.training.receipt import make_receipt, verify

tokenizers = pytest.importorskip("tokenizers")


@pytest.fixture(scope="module")
def embedder(tmp_path_factory) -> tuple[Path, Path]:
    directory = tmp_path_factory.mktemp("embedder")
    write_bert_checkpoint(directory / "checkpoint")
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-encoder")
    return directory / "checkpoint", directory / "model.dllm"


@pytest.fixture(scope="module")
def cross(tmp_path_factory) -> tuple[Path, Path]:
    directory = tmp_path_factory.mktemp("cross")
    write_cross_encoder(directory / "checkpoint")
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-cross-encoder")
    return directory / "checkpoint", directory / "model.dllm"


@pytest.fixture(scope="module")
def roberta_cross(tmp_path_factory) -> tuple[Path, Path]:
    pytest.importorskip("sentencepiece")
    directory = tmp_path_factory.mktemp("xlmr-cross")
    write_roberta_checkpoint(directory / "checkpoint", labels=2)
    import_model(directory / "checkpoint", directory / "model.dllm", repository="example/tiny-xlmr-cross-encoder")
    return directory / "checkpoint", directory / "model.dllm"


def weights_of(path: Path) -> tuple[ModelFile, dict[str, np.ndarray]]:
    model = ModelFile(path)
    return model, {name: np.asarray(values, dtype=np.float32) for name, values in model.tensors.items()}


# Kernels


def test_layer_norm_backward_matches_autograd():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((6, 24)) * 3 + 1).astype(np.float32)
    w, b, dy = (rng.standard_normal(s).astype(np.float32) for s in (24, 24, (6, 24)))
    dx, dw, db = numerics.layer_norm_backward(x, w, dy, 1e-5)
    xt, wt, bt = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (x, w, b))
    torch.nn.functional.layer_norm(xt, (24,), wt, bt, 1e-5).backward(torch.tensor(dy, dtype=torch.float64))
    for ours, theirs in ((dx, xt), (dw, wt), (db, bt)):
        assert np.allclose(ours.numpy(), theirs.grad.numpy(), rtol=1e-5, atol=1e-6)
    # one rounding per output: the bits do not depend on how many rows come along
    single = numerics.layer_norm_backward(x[2:3], w, dy[2:3], 1e-5)[0].numpy()
    assert single.tobytes() == dx.numpy()[2:3].tobytes()
    with pytest.raises(ValueError, match="dy must have the shape"):
        numerics.layer_norm_backward(x, w, dy[:2], 1e-5)


def test_gelu_backward_matches_autograd():
    torch = pytest.importorskip("torch")
    x = np.linspace(-9, 9, 2001, dtype=np.float32)
    dy = np.linspace(-1, 2, 2001, dtype=np.float32)
    ours = numerics.gelu_backward(x, dy).numpy()
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    torch.nn.functional.gelu(xt).backward(torch.tensor(dy, dtype=torch.float64))
    assert np.allclose(ours, xt.grad.numpy(), rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="same size"):
        numerics.gelu_backward(x, dy[:3])


# Gradients


def autograd(directory: Path, tokens: list[int], types: list[int], upstream: np.ndarray, classification: bool):
    """transformers' gradients (float64) of ``sum(upstream * output)``, keyed by our weight names."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    kind = transformers.AutoModelForSequenceClassification if classification else transformers.AutoModel
    model = kind.from_pretrained(directory).double().eval()
    output = model(torch.tensor([tokens]), token_type_ids=torch.tensor([types]))
    values = output.logits[0] if classification else output.last_hidden_state[0]
    (values * torch.tensor(upstream, dtype=torch.float64)).sum().backward()
    grads = {}
    for name, parameter in model.named_parameters():
        ours = _bert_name(name, classification)
        if ours is not None and parameter.grad is not None:
            grads[ours] = parameter.grad.numpy()
    return values.detach().numpy(), grads


def compare(ours: dict[str, np.ndarray], theirs: dict[str, np.ndarray]) -> None:
    for name, expected in theirs.items():
        scale = max(float(np.abs(expected).max()), 1e-3)
        assert np.abs(ours[name] - expected).max() <= 2e-4 * scale, name


def test_encoder_forward_is_the_served_encoder(embedder, cross, roberta_cross):
    for _, path in (embedder, cross, roberta_cross):
        model, weights = weights_of(path)
        served = Encoder(model.config, model.tensors)
        gradients = EncoderGradients(model.config)
        tokens = [0, 5, 9, 1, 1, 12, 2] if model.config.padding_index is not None else [2, 7, 30, 11, 3, 9, 3]
        types = [0] * 4 + [1] * 3 if model.config.type_vocabulary_size > 1 else [0] * 7
        states = gradients.hidden_states(weights, tokens, types)
        assert states.tobytes() == served.hidden_states(tokens, types).tobytes()
        if model.config.classifier_labels:
            logits = gradients.classify(weights, tokens, types).logits
            assert logits.tobytes() == served.classify(tokens, types).tobytes()


def test_embedder_gradients_match_autograd(embedder):
    directory, path = embedder
    model, weights = weights_of(path)
    tokens, types = [2, 7, 30, 11, 3, 9, 3], [0, 0, 0, 0, 1, 1, 1]
    upstream = numerics.fill_gaussian(4, 7 * 32).reshape(7, 32)
    gradients = EncoderGradients(model.config)
    result = gradients.encode(weights, tokens, types)
    ours = gradients.gradients(weights, result, upstream)
    states, theirs = autograd(directory, tokens, types, upstream, classification=False)
    assert np.allclose(result.states, states, atol=1e-4)
    assert set(theirs) == set(ours)
    compare(ours, theirs)


@pytest.mark.parametrize("which", ["bert", "xlm-roberta"])
def test_cross_encoder_gradients_match_autograd(which, cross, roberta_cross):
    directory, path = cross if which == "bert" else roberta_cross
    model, weights = weights_of(path)
    if which == "bert":
        tokens, types = [2, 7, 30, 11, 3, 9, 3], [0, 0, 0, 0, 1, 1, 1]
    else:  # padding inside the sequence: RoBERTa's positions skip it
        tokens, types = [0, 17, 3, 1, 1, 41, 2, 2, 77, 2], [0] * 10
    labels = model.config.classifier_labels
    upstream = np.linspace(-1.0, 1.5, labels, dtype=np.float32)
    gradients = EncoderGradients(model.config)
    result = gradients.classify(weights, tokens, types)
    ours = gradients.gradients(weights, result, dlogits=upstream)
    logits, theirs = autograd(directory, tokens, types, upstream, classification=True)
    assert np.allclose(result.logits, logits, atol=1e-5)
    assert set(theirs) == set(ours)
    compare(ours, theirs)


def test_gradients_match_finite_differences(cross):
    _, path = cross
    model, weights = weights_of(path)
    tokens, types = [2, 7, 30, 11, 3], [0, 0, 0, 1, 1]
    gradients = EncoderGradients(model.config)

    def loss(values: dict[str, np.ndarray]) -> float:
        return float(np.float64(gradients.classify(values, tokens, types).logits[0]))

    grads = gradients.gradients(weights, gradients.classify(weights, tokens, types), dlogits=[1.0])
    for name, index in (
        ("layers.1.mlp_norm.weight", (3,)),
        ("layers.0.attention.k.weight", (4, 2)),
        ("layers.0.mlp.up.bias", (5,)),
        ("embedding_norm.bias", (1,)),
        ("position_embedding.weight", (2, 6)),
        ("pooler.weight", (0, 1)),
    ):
        step = 1e-2
        plus, minus = dict(weights), dict(weights)
        plus[name], minus[name] = weights[name].copy(), weights[name].copy()
        plus[name][index] += step
        minus[name][index] -= step
        estimate = (loss(plus) - loss(minus)) / (2 * step)
        assert abs(estimate - grads[name][index]) <= 2e-3 + 2e-2 * abs(estimate), name


def test_gradients_are_reproducible_and_thread_invariant(cross):
    _, path = cross
    model, weights = weights_of(path)
    gradients = EncoderGradients(model.config)
    tokens, types = [2, 7, 30, 11, 3, 9, 3], [0, 0, 0, 0, 1, 1, 1]

    def run() -> bytes:
        grads = gradients.gradients(weights, gradients.classify(weights, tokens, types), dlogits=[0.5])
        return b"".join(grads[name].tobytes() for name in sorted(grads))

    reference = run()
    for threads in (1, 3):
        numerics.set_threads(threads)
        try:
            assert run() == reference
        finally:
            numerics.set_threads(0)


def test_pooling_gradients(embedder):
    _, path = embedder
    model, weights = weights_of(path)
    gradients = EncoderGradients(model.config)
    states = gradients.hidden_states(weights, [2, 7, 30, 11, 3])
    direction = numerics.fill_gaussian(8, 32)
    for mode in ("mean", "cls", "last_token"):
        normalized, _, norm = pool(states, mode)
        assert abs(float(np.sqrt(np.sum(normalized.astype(np.float64) ** 2))) - 1.0) < 1e-6
        dstates = pool_backward(normalized, norm, direction, len(states), mode)
        step = 1e-3
        for position, column in ((0, 3), (4, 7), (2, 1)):
            plus, minus = states.copy(), states.copy()
            plus[position, column] += step
            minus[position, column] -= step
            estimate = (
                float(np.dot(pool(plus, mode)[0], direction)) - float(np.dot(pool(minus, mode)[0], direction))
            ) / (2 * step)
            assert abs(estimate - dstates[position, column]) < 2e-3, (mode, position)
    with pytest.raises(ValueError, match="unknown pooling"):
        pool(states, "max")
    with pytest.raises(ValueError, match="zero sentence vector"):
        pool(np.zeros((2, 4), dtype=np.float32), "mean")


def test_gradient_refusals(embedder, cross):
    from model_fixtures import TINY_LLAMA_CONFIG

    from etalii_dllm.importing.importer import hf_config

    _, path = embedder
    model, weights = weights_of(path)
    gradients = EncoderGradients(model.config)
    with pytest.raises(ValueError, match="is a decoder"):
        EncoderGradients(hf_config(TINY_LLAMA_CONFIG))
    with pytest.raises(ValueError, match="no classification head"):
        gradients.classify(weights, [2, 3])
    result = gradients.encode(weights, [2, 3])
    with pytest.raises(ValueError, match="states' shape"):
        gradients.gradients(weights, result, np.zeros((3, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="needs a pass that ran the classification head"):
        gradients.gradients(weights, result, dlogits=[1.0])
    with pytest.raises(ValueError, match="this pass did not run the classification head"):
        _ = result.logits
    for tokens, types, message in (
        ([], None, "at least one token"),
        ([2] * 100, None, "exceed"),
        ([10**6], None, "out of range"),
        ([2, 3], [0, 5], "token types"),
    ):
        with pytest.raises(ValueError, match=message):
            gradients.encode(weights, tokens, types)


# Training runs (#353)

TEXTS = [
    ("how are you", "Hello, world! How are you?"),
    ("quick brown fox", "The quick brown fox jumps over the lazy dog."),
    ("naive cafe", "naïve café façade"),
    ("lazy dog", "over the lazy dog"),
    ("hello world", "world, hello!"),
]


def write_pairs(path: Path, negatives: bool = True) -> Path:
    lines = []
    for i, (anchor, positive) in enumerate(TEXTS):
        record = {"anchor": anchor, "positive": positive}
        if negatives and i % 2:
            record["negative"] = TEXTS[(i + 1) % len(TEXTS)][1]
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_labels(path: Path, labels: int = 1) -> Path:
    lines = []
    for i, (query, document) in enumerate(TEXTS):
        other = TEXTS[(i + 2) % len(TEXTS)][1]
        right, wrong = (1, 0) if labels == 1 else (i % labels, (i + 1) % labels)
        lines.append(json.dumps({"text": query, "pair": document, "label": right}))
        lines.append(json.dumps({"text": query, "pair": other, "label": wrong if labels > 1 else 0.25}))
    lines.append(json.dumps({"text": "a lone text", "label": 0 if labels > 1 else 1}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def encoder_data(path: Path, data_file: Path, objective: str, sequence_length: int | None = None) -> EncoderData:
    engine = DllmEngine.from_model_file(path)
    limit = sequence_length or (12 if objective == "embedding" else 24)  # the models' own limits
    return EncoderData.from_records(read_examples(data_file, objective), engine, objective, limit)


RUN = RunConfig(4, 3, 12, 1, AdamWConfig(1e-3), objective="embedding")


def torch_embeddings(directory: Path, data: EncoderData, indices: list[int]):
    """transformers' (float64) sentence vectors of a step's texts in candidate order, with the model."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.AutoModel.from_pretrained(directory).double().eval()

    def vector(tokens: tuple[int, ...]):
        states = model(torch.tensor([tokens])).last_hidden_state[0]
        mean = states.mean(dim=0)
        return mean / mean.norm()

    examples = [data.examples[i] for i in indices]
    anchors = torch.stack([vector(e.texts[0]) for e in examples])
    candidates = [vector(e.texts[1]) for e in examples] + [vector(e.texts[2]) for e in examples if len(e.texts) > 2]
    return model, anchors, torch.stack(candidates)


def test_contrastive_loss_matches_sentence_transformers(embedder, tmp_path):
    torch = pytest.importorskip("torch")
    directory, path = embedder
    data = encoder_data(path, write_pairs(tmp_path / "pairs.jsonl"), "embedding")
    model_file = ModelFile(path)
    tuner = FineTuner.from_model_file(model_file, data, RUN)
    loss, gradients = tuner._embedding_gradients(data)
    indices = data.batch(0, RUN.batch_size, RUN.seed)
    model, anchors, candidates = torch_embeddings(directory, data, indices)
    # sentence-transformers' MultipleNegativesRankingLoss: cross-entropy of scale * cos_sim, labels 0..n-1
    scores = RUN.similarity_scale * anchors @ candidates.T
    expected = torch.nn.functional.cross_entropy(scores, torch.arange(len(indices)))
    assert abs(loss - expected.item()) < 1e-5
    expected.backward()
    theirs = {}
    for name, parameter in model.named_parameters():
        ours = _bert_name(name)
        if ours is not None and parameter.grad is not None:
            theirs[ours] = parameter.grad.numpy()
    compare(gradients, theirs)
    assert set(gradients) == set(model_file.config.tensor_shapes())


@pytest.mark.parametrize("which", ["bert", "xlm-roberta"])
def test_classifier_losses_match_torch(which, cross, roberta_cross, tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    directory, path = cross if which == "bert" else roberta_cross
    labels = ModelFile(path).config.classifier_labels
    data = encoder_data(path, write_labels(tmp_path / "labels.jsonl", labels), "classifier")
    run = RunConfig(3, 4, data.sequence_length, 2, AdamWConfig(1e-3), objective="classifier")
    tuner = FineTuner.from_model_file(ModelFile(path), data, run)
    loss, gradients = tuner._classifier_gradients(data)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(directory).double().eval()
    examples = [data.examples[i] for i in data.batch(0, run.batch_size, run.seed)]
    logits = torch.cat(
        [model(torch.tensor([e.texts[0]]), token_type_ids=torch.tensor([e.types[0]])).logits for e in examples]
    )
    if labels == 1:  # sentence-transformers' CrossEncoder: BCEWithLogitsLoss for one label
        targets = torch.tensor([e.label for e in examples], dtype=torch.float64)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], targets)
    else:
        expected = torch.nn.functional.cross_entropy(logits, torch.tensor([int(e.label) for e in examples]))
    assert abs(loss - expected.item()) < 1e-5
    expected.backward()
    theirs = {}
    for name, parameter in model.named_parameters():
        ours = _bert_name(name, True)
        if ours is not None and parameter.grad is not None:
            theirs[ours] = parameter.grad.numpy()
    compare(gradients, theirs)


def test_runs_resume_bit_for_bit_and_are_golden(embedder, cross, tmp_path):
    for kind, (_, path) in (("embedding", embedder), ("classifier", cross)):
        model_file = ModelFile(path)
        if kind == "embedding":
            data, run = encoder_data(path, write_pairs(tmp_path / "pairs.jsonl"), kind), RUN
        else:
            data = encoder_data(path, write_labels(tmp_path / "labels.jsonl"), kind)
            run = RunConfig(4, 3, data.sequence_length, 1, AdamWConfig(1e-3), objective="classifier")
        first = FineTuner.from_model_file(model_file, data, run)
        first.train()
        first_checkpoint = first.save_checkpoint(tmp_path / f"{kind}.dllmckpt")
        second = FineTuner.from_model_file(model_file, data, run)
        second.train(until=2)
        second.save_checkpoint(tmp_path / f"{kind}-middle.dllmckpt")
        resumed = FineTuner.load_checkpoint(tmp_path / f"{kind}-middle.dllmckpt", data)
        resumed.train()
        assert resumed.save_checkpoint(tmp_path / f"{kind}-resumed.dllmckpt") == first_checkpoint
        assert resumed.losses == first.losses
        fingerprint = first.export(tmp_path / f"{kind}.dllm")
        assert resumed.export(tmp_path / f"{kind}-resumed.dllm") == fingerprint
        assert fingerprint == ENCODER_FINETUNE_FINGERPRINT[kind]
        tuned = ModelFile(tmp_path / f"{kind}.dllm")
        assert (tuned.embedding, tuned.classifier) == (model_file.embedding, model_file.classifier)
        assert tuned.fine_tuning["run"]["objective"] == kind
        assert ("similarity_scale" in tuned.fine_tuning["run"]) == (kind == "embedding")
        engine = DllmEngine.from_model_file(tmp_path / f"{kind}.dllm")
        if kind == "embedding":
            assert engine.embed("hello").vector.shape == (32,)
        else:
            assert len(engine.classify("how are you", "Hello").logits) == 1
        # a training receipt replays to the same weights
        data_file = tmp_path / ("pairs.jsonl" if kind == "embedding" else "labels.jsonl")
        receipt = make_receipt(first, data_file)
        assert receipt["data"]["examples"] == len(data)
        assert verify(receipt, path).ok


def test_the_loss_falls(cross, tmp_path):
    _, path = cross
    data = encoder_data(path, write_labels(tmp_path / "labels.jsonl"), "classifier")
    run = RunConfig(24, 11, data.sequence_length, 0, AdamWConfig(3e-3, weight_decay=0.0), objective="classifier")
    results = FineTuner.from_model_file(ModelFile(path), data, run).train()
    assert results[-1].loss < results[0].loss - 0.1


def test_pooling_follows_the_model(tmp_path):
    write_bert_checkpoint(tmp_path / "cls", pooling="pooling_mode_cls_token")
    import_model(tmp_path / "cls", tmp_path / "cls.dllm")
    data = encoder_data(tmp_path / "cls.dllm", write_pairs(tmp_path / "pairs.jsonl", negatives=False), "embedding")
    tuner = FineTuner.from_model_file(ModelFile(tmp_path / "cls.dllm"), data, RUN)
    assert tuner.pooling == "cls"
    assert tuner.train_step().loss > 0


def test_cli_finetunes_encoders(embedder, cross, tmp_path, capsys):
    _, path = embedder
    pairs = write_pairs(tmp_path / "pairs.jsonl")
    common = ["finetune", str(path), "--data", str(pairs), "--steps", "2", "--batch-size", "2"]
    assert cli([*common, "-o", str(tmp_path / "a.dllm"), "--receipt", str(tmp_path / "a.json")]) == 0
    out = capsys.readouterr().out
    assert "5 examples of up to 12 tokens" in out  # the model's own limit (max_seq_length)
    assert cli(["replay", str(tmp_path / "a.json"), "--base", str(path)]) == 0
    assert "training again gave the same weights" in capsys.readouterr().out
    labels = write_labels(tmp_path / "labels.jsonl")
    wrong = ["finetune", str(path), "--data", str(labels), "--objective", "classifier", "-o", str(tmp_path / "b.dllm")]
    assert cli(wrong) == 1
    assert "no classification head" in capsys.readouterr().err
    assert cli([*common, "-o", str(tmp_path / "b.dllm"), "--dpo"]) == 1
    assert "an encoder trains with the embedding or classifier objective" in capsys.readouterr().err
    assert cli([*common, "-o", str(tmp_path / "b.dllm"), "--dpo", "--objective", "lm"]) == 1
    assert "different objectives" in capsys.readouterr().err
    _, cross_path = cross
    args = ["finetune", str(cross_path), "--data", str(labels), "--steps", "1", "-o", str(tmp_path / "c.dllm")]
    assert cli([*args, "--sequence-length", "10"]) == 0
    assert "11 examples of up to 10 tokens" in capsys.readouterr().out


def test_encoder_run_refusals(embedder, cross, tmp_path):
    from model_fixtures import TINY_LLAMA_CONFIG

    from etalii_dllm.importing.importer import hf_config
    from etalii_dllm.training.data import TrainingData

    _, path = embedder
    _, cross_path = cross
    model_file = ModelFile(path)
    pairs = encoder_data(path, write_pairs(tmp_path / "pairs.jsonl"), "embedding")
    labels = encoder_data(cross_path, write_labels(tmp_path / "labels.jsonl"), "classifier")
    windows = TrainingData(((1, 2, 3),), 12)
    with pytest.raises(ValueError, match="similarity scale"):
        RunConfig(1, similarity_scale=0.0)
    with pytest.raises(ValueError, match="unknown training objective"):
        RunConfig(1, objective="mlm")
    cases = [
        (model_file, windows, RunConfig(1, 2, 12), "an encoder trains with"),
        (model_file, labels, RunConfig(1, 2, labels.sequence_length, objective="embedding"), "trains on embedding"),
        (model_file, labels, RunConfig(1, 2, labels.sequence_length, objective="classifier"), "no classification head"),
        (ModelFile(cross_path), pairs, RunConfig(1, 2, 12, objective="embedding"), "is a cross-encoder"),
    ]
    for model, data, run, message in cases:
        with pytest.raises(ValueError, match=message):
            FineTuner.from_model_file(model, data, run)
    decoder = hf_config(TINY_LLAMA_CONFIG)
    shapes = decoder.tensor_shapes()
    with pytest.raises(ValueError, match="trains encoders; this model is a decoder"):
        FineTuner(decoder, shapes, pairs, RunConfig(1, 2, 12, objective="embedding"), base_fingerprint="", metadata={})
    with pytest.raises(ValueError, match="encoder examples train encoders"):
        FineTuner(decoder, shapes, pairs, RunConfig(1, 2, 12), base_fingerprint="", metadata={})


def test_encoder_data_refusals(embedder, cross, roberta_cross, tmp_path):
    _, path = embedder
    for objective, line, message in (
        ("embedding", '{"anchor": "a"}', "expected 'anchor' and 'positive'"),
        ("embedding", '{"anchor": "a", "positive": "b", "negative": 3}', "expected 'anchor' and 'positive'"),
        ("embedding", "[1]", "expected a JSON object"),
        ("embedding", "{nope", "invalid JSON"),
        ("classifier", '{"text": "a", "label": "yes"}', "the label must be a number"),
        ("classifier", '{"text": "a", "label": true}', "the label must be a number"),
        ("classifier", '{"pair": "a", "label": 1}', "expected a 'text'"),
    ):
        (tmp_path / "bad.jsonl").write_text(line + "\n", encoding="utf-8")
        with pytest.raises(TrainingDataError, match=message):
            read_examples(tmp_path / "bad.jsonl", objective)
    (tmp_path / "empty.jsonl").write_text("\n", encoding="utf-8")
    with pytest.raises(TrainingDataError, match="no examples"):
        read_examples(tmp_path / "empty.jsonl", "embedding")
    with pytest.raises(TrainingDataError, match="unknown encoder objective"):
        read_examples(tmp_path / "empty.jsonl", "lm")
    with pytest.raises(TrainingDataError):
        read_examples(tmp_path / "missing.jsonl", "embedding")
    engine = DllmEngine.from_model_file(cross[1])
    for record, message in (
        ({"text": "a", "label": 2}, "between 0 and 1"),
        ({"text": "a", "label": -0.5}, "between 0 and 1"),
    ):
        with pytest.raises(TrainingDataError, match=message):
            EncoderData.from_records([record], engine, "classifier", 12)
    two = DllmEngine.from_model_file(roberta_cross[1])
    for label in (2, 0.5, -1):
        with pytest.raises(TrainingDataError, match="label index below 2"):
            EncoderData.from_records([{"text": "a", "label": label}], two, "classifier", 12)
    with pytest.raises(TrainingDataError, match="positive"):
        EncoderData.from_records([], engine, "classifier", 0)
    # long texts are cut to the sequence length with their special tokens kept
    embedder_engine = DllmEngine.from_model_file(path)
    long = EncoderData.from_records([{"anchor": "word " * 50, "positive": "b"}], embedder_engine, "embedding", 6)
    first = long.examples[0].texts[0]
    assert len(first) == 6 and first[0] == embedder_engine.embedding_tokens("x")[0]
    pair = EncoderData.from_records([{"text": "word " * 40, "pair": "doc " * 40, "label": 1}], engine, "classifier", 9)
    assert len(pair.examples[0].texts[0]) == 9
    assert long.fingerprint != pair.fingerprint and len(long.epoch_order(0, 0)) == 1


# LoRA (#354)


def test_lora_on_encoders(embedder, cross, tmp_path, capsys):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    directory, path = embedder
    pairs = write_pairs(tmp_path / "pairs.jsonl")
    common = ["finetune", str(path), "--data", str(pairs), "--steps", "3", "--batch-size", "3"]
    common += ["--learning-rate", "1e-2", "--lora-rank", "2"]
    assert cli([*common, "-o", str(tmp_path / "merged.dllm"), "--adapter-output", str(tmp_path / "adapter")]) == 0
    capsys.readouterr()
    settings = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert settings["task_type"] == "FEATURE_EXTRACTION"
    # every key names a module transformers' model has, and the regex PEFT matches picks exactly those
    import re

    modules = {name for name, _ in transformers.AutoModel.from_pretrained(directory).named_modules()}
    lora, adapters = read_peft(tmp_path / "adapter", ModelFile(path).config)
    assert lora.rank == 2 and len(adapters) == 2 * 6 * 2
    keys = [peft_key(name, ModelFile(path).config) for name in adapters]
    adapted = {key.removeprefix("base_model.model.").rsplit(".lora_", 1)[0] for key in keys}
    assert adapted <= modules
    assert {m for m in modules if re.fullmatch(settings["target_modules"], m)} == adapted
    # the merged model, the adapter applied at load time and the adapter imported onto the base agree
    merged = DllmEngine.from_model_file(tmp_path / "merged.dllm").embed("hello there").vector
    loaded = DllmEngine.from_model_file(path, adapter=tmp_path / "adapter").embed("hello there").vector
    assert merged.tobytes() == loaded.tobytes()
    import_model(tmp_path / "adapter", tmp_path / "imported.dllm", base=path, licence="apache-2.0")
    imported = ModelFile(tmp_path / "imported.dllm")
    assert imported.fingerprint == ModelFile(tmp_path / "merged.dllm").fingerprint
    assert imported.embedding == ModelFile(path).embedding
    assert torch is not None

    # a cross-encoder's adapter sits under bert.; the head stays frozen
    _, cross_path = cross
    config = ModelFile(cross_path).config
    assert peft_key("layers.1.mlp.down.weight.lora_b", config) == (
        "base_model.model.bert.encoder.layer.1.output.dense.lora_B.weight"
    )
    assert target_weights(config, LoraConfig(2, 2.0, ("gate", "q"))) == [
        "layers.0.attention.q.weight",
        "layers.1.attention.q.weight",
    ]
    with pytest.raises(AdapterError, match="no gate projection"):
        target_weights(config, LoraConfig(2, 2.0, ("gate",)))
    assert "output\\.dense" in target_modules(config, LoraConfig(2, 2.0, ("down",)))
    labels = write_labels(tmp_path / "labels.jsonl")
    args = ["finetune", str(cross_path), "--data", str(labels), "--steps", "1", "--lora-rank", "2"]
    assert cli([*args, "--adapter-output", str(tmp_path / "cross-adapter")]) == 0
    capsys.readouterr()
    assert json.loads((tmp_path / "cross-adapter" / "adapter_config.json").read_text())["task_type"] == "SEQ_CLS"
    read_peft(tmp_path / "cross-adapter", config)


def test_encoder_adapter_refusals(embedder, tmp_path):
    from etalii_dllm.importing.safetensors import write_safetensors

    _, path = embedder
    config = ModelFile(path).config
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 2}), encoding="utf-8")
    for key, message in (
        ("base_model.model.encoder.layer.0.attention.self.query.lora_C.weight", "unsupported adapter tensor"),
        ("base_model.model.encoder.layer.0.pooler.dense.lora_A.weight", "unsupported adapter tensor"),
        ("base_model.model.encoder.layer.9.attention.self.query.lora_A.weight", "does not fit the model"),
    ):
        write_safetensors(tmp_path / "adapter_model.safetensors", {key: np.zeros((2, 32), np.float32)})
        with pytest.raises(AdapterError, match=message):
            read_peft(tmp_path, config)
    with pytest.raises(ModelImportError):
        import_model(tmp_path, tmp_path / "x.dllm", base=path, licence="apache-2.0")
