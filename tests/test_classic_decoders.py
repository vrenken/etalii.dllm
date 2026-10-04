"""The classic decoders (#422-#425): Phi-1/1.5/2, GPT-NeoX (Pythia) and GPT-2 on the exact kernels.

LayerNorms with biases, biases on every projection, plain GELU MLPs, parallel attention and MLP (one shared norm for
Phi, a norm each for GPT-NeoX), partial rotary embeddings, learned absolute positions (GPT-2) and Phi's biased LM
head. Tiny checkpoints built with transformers' own model classes check the logits; the cache, batching and threads
change no bit, the reference implementation gives the kernels' bits, and the generations are golden.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from golden_values import CLASSIC_GENERATION_FINGERPRINTS
from model_fixtures import MODEL_CARD, write_safetensors

from etalii_dllm import numerics
from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.engine import ChatRequest, DllmEngine
from etalii_dllm.importing import ModelImportError, import_model
from etalii_dllm.importing.importer import classic_config
from etalii_dllm.interpret.lens import logit_lens
from etalii_dllm.modelfile import ModelFile
from etalii_dllm.numerics import fill_gaussian, fingerprint
from etalii_dllm.reference import ReferenceTransformer
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.transformer import LayerHook, Transformer

pytest.importorskip("tokenizers")

TEXT = "the quick brown fox jumps over the lazy dog, 12 times"

PHI = {
    "architectures": ["PhiForCausalLM"],
    "model_type": "phi",
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "hidden_act": "gelu_new",
    "max_position_embeddings": 128,
    "layer_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "partial_rotary_factor": 0.5,
    "qk_layernorm": False,
    "resid_pdrop": 0.0,
    "embd_pdrop": 0.0,
    "attention_dropout": 0.0,
}
"""Phi-2's ``config.json`` layout (transformers v4 names), tiny, with grouped-query attention."""

NEOX = {
    "architectures": ["GPTNeoXForCausalLM"],
    "model_type": "gpt_neox",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "hidden_act": "gelu",
    "max_position_embeddings": 128,
    "layer_norm_eps": 1e-5,
    "rotary_emb_base": 10000,
    "rotary_pct": 0.5,
    "use_parallel_residual": True,
    "tie_word_embeddings": False,
}
"""Pythia's ``config.json`` layout, tiny."""

GPT2 = {
    "architectures": ["GPT2LMHeadModel"],
    "model_type": "gpt2",
    "activation_function": "gelu_new",
    "n_embd": 32,
    "n_head": 4,
    "n_layer": 2,
    "n_positions": 64,
    "n_ctx": 64,
    "layer_norm_epsilon": 1e-5,
    "attn_pdrop": 0.0,
    "embd_pdrop": 0.0,
    "resid_pdrop": 0.0,
}
"""GPT-2's ``config.json`` layout, tiny."""

CONFIGS = {"phi": PHI, "gpt_neox": NEOX, "sequential_neox": {**NEOX, "use_parallel_residual": False}, "gpt2": GPT2}


def tokenizer_json() -> dict:
    from test_bpe import smollm2_style

    return json.loads(smollm2_style().to_str())


def write_checkpoint(directory: Path, raw: dict, seed: int = 5) -> None:
    """A checkpoint of transformers' model for ``raw`` with deterministic float32 weights: norm weights near 1,
    everything else small Gaussians, so the logits are far from flat."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizer = tokenizer_json()
    vocabulary = len(tokenizer["model"]["vocab"]) + len(tokenizer.get("added_tokens", []))
    raw = {**raw, "vocab_size": vocabulary, "bos_token_id": 1, "eos_token_id": 2}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    config = transformers.AutoConfig.from_pretrained(directory)
    model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float32).eval()
    tensors = {}
    for index, (name, parameter) in enumerate(sorted(model.named_parameters())):
        values = fill_gaussian(seed * 1000 + index, parameter.numel()).reshape(tuple(parameter.shape))
        if name.endswith(("layernorm.weight", "layer_norm.weight", "ln_1.weight", "ln_2.weight", "ln_f.weight")):
            values = np.float32(1) + values * np.float32(0.1)
        else:
            values = values * np.float32(0.2)
        parameter.data.copy_(torch.from_numpy(values))
        tensors["embed_out.weight" if name == "lm_head.weight" and raw["model_type"] == "gpt_neox" else name] = (
            "F32",
            values,
        )
    write_safetensors(directory / "model.safetensors", tensors, {"format": "pt"})
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    template = "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"
    tokenizer_config = {"chat_template": template, "eos_token": "<|im_end|>", "bos_token": None}
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    (directory / "README.md").write_bytes(MODEL_CARD.encode())


def transformers_logits(directory: Path, tokens: list[int]) -> np.ndarray:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()
    with torch.no_grad():
        return model(torch.tensor([tokens])).logits[0].numpy()


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict[str, tuple[Path, Path]]:
    """(checkpoint, model file) of each tiny classic decoder."""
    out = {}
    for name, raw in CONFIGS.items():
        directory = tmp_path_factory.mktemp(name)
        write_checkpoint(directory / "checkpoint", raw)
        import_model(directory / "checkpoint", directory / "model.dllm", repository=f"example/tiny-{name}")
        out[name] = (directory / "checkpoint", directory / "model.dllm")
    return out


def tokens_of(path: Path, text: str = TEXT) -> list[int]:
    return DllmEngine.from_model_file(path).tokenizer.encode(text)


# The layouts (#422-#424)


def test_import_records_the_layouts(built):
    expected = {
        "phi": ("phi", "shared", True, False, "gelu_tanh", 4),
        "gpt_neox": ("gpt_neox", "separate", False, False, "gelu", 4),
        "sequential_neox": ("gpt_neox", None, False, False, "gelu", 4),
        "gpt2": ("gpt2", None, False, True, "gelu_tanh", None),
    }
    for name, (_, path) in built.items():
        config = ModelFile(path).config
        family, parallel, head_bias, absolute, activation, rotary = expected[name]
        assert (config.family, config.parallel_residual, config.lm_head_bias) == (family, parallel, head_bias)
        assert (config.absolute_positions, config.activation, config.rotary_dim) == (absolute, activation, rotary)
        assert config.layer_norm and config.linear_bias and config.plain_mlp and config.attention_bias
        assert TransformerConfig.from_dict(config.to_dict()) == config
    phi = ModelFile(built["phi"][1]).config
    assert phi.kv_heads == 2 and "layers.0.mlp_norm.weight" not in phi.tensor_shapes()
    gpt2 = ModelFile(built["gpt2"][1]).config
    assert gpt2.tie_word_embeddings and gpt2.context_length == 64
    assert gpt2.tensor_shapes()["position_embedding.weight"] == (64, 32)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_logits_match_transformers(built, name):
    checkpoint, path = built[name]
    model = Transformer.from_file(path)
    tokens = tokens_of(path)
    expected = transformers_logits(checkpoint, tokens)
    ours = model.forward_cached_last(tokens, model.new_cache(), len(tokens))
    np.testing.assert_allclose(ours, expected, atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_cache_batching_and_threads_change_no_bit(built, name):
    _, path = built[name]
    model = Transformer.from_file(path)
    tokens = tokens_of(path)
    expected = model.forward(tokens)
    cache = model.new_cache()
    for end in range(1, len(tokens) + 1):  # one token at a time
        stepped = model.forward_cached(tokens[:end], cache)
    assert stepped.tobytes() == expected.tobytes()
    other = tokens_of(path, "a lazy dog")
    batched = model.forward_batch([tokens, other])
    assert batched[0].tobytes() == expected.tobytes()
    assert batched[1].tobytes() == model.forward(other).tobytes()
    try:
        for threads in (1, 3):
            numerics.set_threads(threads)
            assert model.forward(tokens).tobytes() == expected.tobytes()
    finally:
        numerics.set_threads(0)


def test_generation_is_golden_and_matches_transformers(built):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    for name, (checkpoint, path) in built.items():
        engine = DllmEngine.from_model_file(path)
        prompt = engine.tokenizer.encode(TEXT)
        greedy = engine.complete(TEXT, 12, SamplingOptions())
        model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).eval()
        with torch.no_grad():
            expected = model.generate(
                torch.tensor([prompt]), max_new_tokens=12, do_sample=False, eos_token_id=None, pad_token_id=0
            )[0].tolist()[len(prompt) :]
        assert list(greedy.tokens[: len(expected)]) == expected[: len(greedy.tokens)], name
        sampled = engine.complete(TEXT, 16, SamplingOptions(temperature=0.9, seed=3))
        digest = fingerprint(np.asarray([*greedy.tokens, -1, *sampled.tokens], dtype=np.int64), np.int64)
        assert digest == CLASSIC_GENERATION_FINGERPRINTS[name], (name, digest)


def test_gpt2_positions_end_the_context(built):
    _, path = built["gpt2"]
    model = Transformer.from_file(path)
    assert model.forward(list(range(3, 67))[:64]).shape == (model.vocabulary_size,)
    with pytest.raises(ValueError, match="64 position embeddings"):
        model.forward([5] * 65)
    engine = DllmEngine.from_model_file(path)
    assert engine._generator.context_length == 64


# Across the engine (#425)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_reference_implementation_gives_the_kernels_bits(built, name):
    _, path = built[name]
    for quantize in (None, "q8_0"):
        model = Transformer.from_file(path, quantize=quantize)
        reference = ReferenceTransformer.from_engine_model(model)
        tokens = tokens_of(path, "the lazy dog")
        assert reference.forward(tokens[:-1]).tobytes() == model.forward(tokens[:-1]).tobytes()
        assert reference.forward(tokens[-1:]).tobytes() == model.forward(tokens).tobytes()


@pytest.mark.parametrize("quantize", ["q8_0", "q4_0"])
def test_quantised_classic_decoders(built, quantize):
    for name, (_, path) in built.items():
        plain = Transformer.from_file(path)
        quantised = Transformer.from_file(path, quantize=quantize)
        tokens = tokens_of(path)
        logits = quantised.forward(tokens)
        assert logits.tobytes() == Transformer.from_file(path, quantize=quantize).forward(tokens).tobytes()
        assert float(np.mean(np.abs(logits - plain.forward(tokens)))) < 0.5, name
        assert quantised.weights_fingerprint != plain.weights_fingerprint


class Recorder(LayerHook):
    wants_attention = True

    def __init__(self) -> None:
        self.points: dict[tuple[int, str], np.ndarray] = {}
        self.activations: dict[int, np.ndarray] = {}
        self.attention_maps: dict[int, np.ndarray] = {}

    def residual(self, layer, point, x):
        self.points[layer, point] = x
        return None

    def attention(self, layer, probabilities):
        self.attention_maps[layer] = probabilities

    def mlp_activation(self, layer, activation):
        self.activations[layer] = activation


@pytest.mark.parametrize("name", list(CONFIGS))
def test_traces_change_no_bit(built, name):
    _, path = built[name]
    model = Transformer.from_file(path)
    tokens = tokens_of(path)
    recorder = Recorder()
    hidden = model.run_hooked(tokens, recorder)
    assert hidden.tobytes() == model.hidden_states(tokens).tobytes()
    assert model.logits_from_hidden(hidden[-1:])[0].tobytes() == model.forward(tokens).tobytes()
    config = model.config
    for layer in range(config.layers):
        middle = recorder.points[layer, "middle"]
        if config.parallel_residual is not None:  # the MLP reads the layer's input
            assert middle.tobytes() == recorder.points[layer, "input"].tobytes()
        keys = model.mlp_activation(middle, layer)
        assert keys.tobytes() == recorder.activations[layer].tobytes()
        assert recorder.attention_maps[layer].shape == (len(tokens), config.heads, len(tokens))
    final = recorder.points[config.layers - 1, "output"]
    assert model.final_norm(final).tobytes() == hidden.tobytes()
    lens = logit_lens(model, tokens)
    assert len(lens.predictions) == config.layers + 1 and len(lens.predictions[0]) == len(tokens)


def test_steering_moves_the_output(built):
    for name, (_, path) in built.items():
        plain = Transformer.from_file(path)
        vector = fill_gaussian(9, plain.config.hidden_size)
        steered = Transformer.from_file(path, steering={0: vector})
        tokens = tokens_of(path)
        assert steered.forward(tokens).tobytes() != plain.forward(tokens).tobytes(), name
        assert steered.weights_fingerprint != plain.weights_fingerprint
        reference = ReferenceTransformer.from_engine_model(steered)
        assert reference.forward(tokens).tobytes() == steered.forward(tokens).tobytes()


def test_steering_vectors_and_sae_activations(built):
    from etalii_dllm.interpret.sae import collect_activations
    from etalii_dllm.interpret.steering import build_steering_vector

    for name, (_, path) in built.items():
        engine = DllmEngine.from_model_file(path)
        vector = build_steering_vector(engine.model, engine.tokenizer, ["a quick fox"], ["a lazy dog"], 1)
        assert vector.vector.shape == (engine.model.config.hidden_size,) and np.any(vector.vector), name
        activations = collect_activations(engine.model, engine.tokenizer, [TEXT, "a lazy dog"], 2, outlier=0)
        assert activations.values.shape[1] == engine.model.config.hidden_size


def test_speculation_and_the_prompt_cache_change_no_bit(built):
    for name, (_, path) in built.items():
        plain = DllmEngine.from_model_file(path, prompt_cache=0)
        fast = DllmEngine.from_model_file(path, speculate=4, prompt_cache=2)
        for options in (SamplingOptions(), SamplingOptions(temperature=0.8, seed=5)):
            request = ChatRequest((), 14, options, prompt=TEXT + " " + TEXT, top_logprobs=1)
            expected = plain.chat_completion(request)
            assert dataclasses.replace(fast.chat_completion(request), cached_tokens=0) == expected, name
            assert dataclasses.replace(fast.chat_completion(request), cached_tokens=0) == expected, name


# Refusals


def test_what_does_not_support_them_yet_refuses(built, tmp_path):
    _, path = built["phi"]
    with pytest.raises(ValueError, match="CPU only"):
        Transformer.from_file(path, device="cuda")
    from etalii_dllm.exporting import ExportError, export_gguf, export_safetensors
    from etalii_dllm.lora import AdapterError, LoraConfig, read_peft, target_weights
    from etalii_dllm.training.backprop import DecoderGradients

    config = ModelFile(path).config
    with pytest.raises(ValueError, match="fine-tuning phi decoders"):
        DecoderGradients(config)
    from etalii_dllm.interpret.editing import EditRequest, rome

    engine = DllmEngine.from_model_file(path)
    with pytest.raises(ValueError, match="model editing needs the fine-tuning support"):
        rome(engine.model, engine.tokenizer, EditRequest("the quick brown fox", "fox", " dog"))
    with pytest.raises(AdapterError, match="LoRA adapters for phi"):
        target_weights(config, LoraConfig(rank=2, alpha=4.0))
    with pytest.raises(AdapterError, match="LoRA adapters for phi"):
        read_peft(tmp_path, config)
    model = ModelFile(path)
    with pytest.raises(ExportError, match="phi"):
        export_safetensors(model, tmp_path / "out")
    with pytest.raises(ExportError, match="phi"):
        export_gguf(model, tmp_path / "out.gguf")


def test_import_refusals(tmp_path):
    base = {**PHI, "vocab_size": 64}
    with pytest.raises(ModelImportError, match="qk_layernorm"):
        classic_config({**base, "qk_layernorm": True})
    with pytest.raises(ModelImportError, match="activation 'relu'"):
        classic_config({**base, "hidden_act": "relu"})
    with pytest.raises(ModelImportError, match="attention biases"):
        classic_config({**NEOX, "vocab_size": 64, "attention_bias": False})
    gpt2 = {**GPT2, "vocab_size": 64}
    for change in (
        {"scale_attn_weights": False},
        {"scale_attn_by_inverse_layer_idx": True},
        {"add_cross_attention": True},
    ):
        with pytest.raises(ModelImportError, match="GPT-2 without scaled attention"):
            classic_config({**gpt2, **change})
    with pytest.raises(ModelImportError, match="64 tokens or fewer"):
        classic_config(gpt2, context_length=65)
    assert classic_config(gpt2, context_length=32).context_length == 32
    assert classic_config({**gpt2, "eos_token_id": 2}, {"eos_token_id": [2, 7]}).eos_token_ids == (2, 7)
    assert classic_config(base, context_length=512).rope_scaling["rope_type"] == "yarn"
    v5 = {
        **NEOX,
        "vocab_size": 64,
        "rope_parameters": {"rope_type": "default", "rope_theta": 500.0, "partial_rotary_factor": 0.25},
    }
    config = classic_config(v5)
    assert (config.rope_theta, config.rotary_dim) == (500.0, 2)
    assert classic_config({**NEOX, "vocab_size": 64, "rotary_pct": 1.0}).rotary_dim is None
    with pytest.raises(ModelImportError, match="multiple of kv_heads"):
        classic_config({**base, "num_key_value_heads": 3})


def test_checkpoint_refusals_and_variants(built, tmp_path):
    from safetensors.numpy import load_file, save_file

    def variant(name: str, edit) -> Path:
        checkpoint = built[name][0]
        directory = tmp_path / f"{name}-{len(list(tmp_path.iterdir()))}"
        directory.mkdir()
        for file in checkpoint.iterdir():
            (directory / file.name).write_bytes(file.read_bytes())
        weights = load_file(directory / "model.safetensors")
        edit(weights)
        save_file(weights, directory / "model.safetensors", {"format": "pt"})
        return directory

    def add(name: str, values: np.ndarray):
        return lambda weights: weights.__setitem__(name, values)

    with pytest.raises(ModelImportError, match=r"unexpected tensor 'model\.extra\.weight'"):
        import_model(variant("phi", add("model.extra.weight", np.zeros(2, np.float32))), tmp_path / "x.dllm")
    with pytest.raises(ModelImportError, match=r"unexpected tensor 'model\.layers\.0\.self_attn\.rotate\.weight'"):
        import_model(
            variant("phi", add("model.layers.0.self_attn.rotate.weight", np.zeros(2, np.float32))), tmp_path / "x.dllm"
        )
    with pytest.raises(ModelImportError, match="does not split"):
        bad = add("gpt_neox.layers.0.attention.query_key_value.weight", np.zeros((90, 32), np.float32))
        import_model(variant("gpt_neox", bad), tmp_path / "x.dllm")
    with pytest.raises(ModelImportError, match="dtype I32"):
        import_model(variant("gpt2", add("wpe.weight", np.zeros((64, 32), np.int32))), tmp_path / "x.dllm")
    with pytest.raises(ModelImportError, match="lm_head differs"):
        import_model(variant("gpt2", add("lm_head.weight", np.zeros((1, 1), np.float32) + 1)), tmp_path / "x.dllm")

    def legacy(weights):  # the original GPT-2 checkpoints: no "transformer." prefix, mask buffers, the head stored
        for name in list(weights):
            weights[name.removeprefix("transformer.")] = weights.pop(name)
        weights["h.0.attn.bias"] = np.ones((1, 1, 64, 64), np.float32)
        weights["lm_head.weight"] = weights["wte.weight"]

    path = tmp_path / "legacy.dllm"
    import_model(variant("gpt2", legacy), path)
    tokens = tokens_of(path)
    assert (
        Transformer.from_file(path).forward(tokens).tobytes()
        == Transformer.from_file(built["gpt2"][1]).forward(tokens).tobytes()
    )
    short = tmp_path / "short.dllm"
    import_model(built["gpt2"][0], short, context_length=16)
    assert ModelFile(short).config.tensor_shapes()["position_embedding.weight"] == (16, 32)
    assert (
        Transformer.from_file(short).forward(tokens[:16]).tobytes()
        == Transformer.from_file(built["gpt2"][1]).forward(tokens[:16]).tobytes()
    )


def test_config_validation():
    base = classic_config({**PHI, "vocab_size": 64})
    with pytest.raises(ValueError, match="needs LayerNorms"):
        dataclasses.replace(base, layer_norm=False)
    with pytest.raises(ValueError, match="needs LayerNorms"):
        dataclasses.replace(base, activation="silu")
    with pytest.raises(ValueError, match="parallel_residual"):
        dataclasses.replace(base, parallel_residual="both")
    with pytest.raises(ValueError, match="do not rotate"):
        dataclasses.replace(base, absolute_positions=True)
    with pytest.raises(ValueError, match="no experts"):
        dataclasses.replace(base, qk_norm=True)
    with pytest.raises(ValueError, match="are for gpt2, gpt_neox, phi"):
        dataclasses.replace(base, family="llama")
    with pytest.raises(ValueError, match="are for encoders only"):  # a gelu MLP without plain_mlp
        dataclasses.replace(
            classic_config({**NEOX, "vocab_size": 64}),
            family="llama",
            layer_norm=False,
            linear_bias=False,
            lm_head_bias=False,
            plain_mlp=False,
            parallel_residual=None,
        )
    assert not dataclasses.replace(
        base,
        family="llama",
        layer_norm=False,
        linear_bias=False,
        lm_head_bias=False,
        plain_mlp=False,
        parallel_residual=None,
        activation="silu",
    ).cpu_only
