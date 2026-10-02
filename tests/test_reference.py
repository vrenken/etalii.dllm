"""The independent reference implementation (issue #161) and ``dllm verify --reference`` (issue #162).

``etalii_dllm.reference`` shares no code with the C++ kernels; these tests require the two to give the same bits.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from test_model_building import BYTE_LEVEL_TOKENIZER, _import

from etalii_dllm import engine as engine_module
from etalii_dllm import numerics, reference, verify
from etalii_dllm.architecture import TransformerConfig
from etalii_dllm.cli import main
from etalii_dllm.engine import DllmEngine, default_engine
from etalii_dllm.sampling import Sampler, SamplingOptions
from etalii_dllm.transformer import Transformer


@pytest.fixture
def isolated(monkeypatch):
    """``main`` writes ``--model`` to the environment; register it so monkeypatch restores it."""
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, "")
    monkeypatch.delenv(engine_module.MODEL_ENVIRONMENT_VARIABLE)
    default_engine.cache_clear()
    yield
    default_engine.cache_clear()


def _bits(a) -> bytes:
    values = np.asarray(a.numpy() if hasattr(a, "numpy") else a)
    return values.dtype.str.encode() + str(values.shape).encode() + values.tobytes()


def _same(a, b) -> bool:
    return _bits(a) == _bits(b)


def _gaussian(seed: int, *shape: int) -> np.ndarray:
    return numerics.fill_gaussian(seed, math.prod(shape)).reshape(shape)


SPECIAL = np.array(
    [0.0, -0.0, 1e-310, -1e-310, 5e-324, 709.78, 709.7, -745.2, -745.1, -708.5, -740.0, 710.0, -746.0, math.inf,
     -math.inf, math.nan, 1.0, -1.0, 2.5, -2.5, 6.0, -6.0, 27.3, 22.0, 0.125, 1e6, -1e6, 3e5, 7.1e15, -7.1e15, 1e16,
     2.0**54 + 4, 1.5e19, -1e20, 1e300, -1e300]
)  # fmt: skip
ARGUMENTS = np.concatenate(
    [_gaussian(3, 4000) * s for s in (1e-3, 0.3, 3.0, 50.0)] + [np.linspace(-800, 800, 2001), SPECIAL]
).astype(np.float64)


def _scalar_bits(f, xs) -> np.ndarray:
    values = np.array([f(float(x)) for x in xs])
    return np.where(np.isnan(values), np.nan, values).view(np.uint64)


@pytest.mark.parametrize("name", ["exp", "sin", "cos", "tanh", "erf", "atan"])
def test_transcendentals_are_the_kernels_bits(name):
    expected = _scalar_bits(getattr(numerics, name), ARGUMENTS)
    actual = getattr(reference, name)(ARGUMENTS)
    assert np.array_equal(np.where(np.isnan(actual), np.nan, actual).view(np.uint64), expected)


def test_log_and_acos_are_the_kernels_bits():
    for f, g, xs in (
        (numerics.log, reference.log, np.concatenate([np.abs(ARGUMENTS), ARGUMENTS])),
        (numerics.acos, reference.acos, np.concatenate([np.linspace(-1.2, 1.2, 4001), _gaussian(4, 1000) / 4])),
    ):
        actual = g(xs)
        assert np.array_equal(np.where(np.isnan(actual), np.nan, actual).view(np.uint64), _scalar_bits(f, xs))


def test_random_numbers_and_sampling_are_the_kernels_bits():
    a, b = numerics.DeterministicRandom(42), reference.Random(42)
    assert [a.next_u64() for _ in range(200)] == [b.next_u64() for _ in range(200)]
    assert [a.next_double() for _ in range(50)] == [b.next_double() for _ in range(50)]
    assert _same(numerics.fill_gaussian(5, 500), reference.fill_gaussian(5, 500))
    logits = [_gaussian(20 + i, 300) * 4 for i in range(40)]
    for options in (
        SamplingOptions(),
        SamplingOptions(temperature=0.8, seed=7),
        SamplingOptions(temperature=1.3, top_k=20, seed=2**64 + 5),
        SamplingOptions(temperature=0.5, top_p=0.9, top_k=50, seed=11),
    ):
        ours = Sampler(options)
        theirs = reference.Sampler(options.temperature, options.top_k, options.top_p, options.seed)
        assert [ours.sample(x) for x in logits] == [theirs.sample(x) for x in logits]


def test_kernels_are_the_kernels_bits():
    x, w, bias = _gaussian(1, 7, 96), _gaussian(2, 50, 96), _gaussian(3, 50)
    assert _same(numerics.linear(x, w, bias), reference.linear(x, w, bias))
    assert _same(numerics.linear(x[None], w), reference.linear(x[None], w))
    for kind in ("q8_0", "q4_0"):
        expected = numerics.linear(x, numerics.QuantizedWeight(w, kind), bias)
        assert _same(expected, reference.linear(x, reference.Weight(w, kind), bias))
    assert _same(numerics.matmul(x, np.ascontiguousarray(w.T)), reference.matmul(x, w.T))
    for unit in (False, True):
        assert _same(numerics.rms_norm(x, _gaussian(7, 96), 1e-6, add_unit_offset=unit),
                     reference.rms_norm(x, _gaussian(7, 96), 1e-6, unit))  # fmt: skip
    assert _same(numerics.rms_norm(x, None, 1e-6), reference.rms_norm(x, None, 1e-6))
    wide = np.concatenate([_gaussian(9, 4000) * 6, SPECIAL[np.abs(SPECIAL) < 1e30].astype(np.float32) / 1e3])
    up = _gaussian(10, wide.size)
    assert _same(numerics.silu(wide), reference.silu(wide))
    assert _same(numerics.gelu(wide), reference.gelu(wide))
    assert _same(numerics.gelu(wide, approximate="tanh"), reference.gelu(wide, "tanh"))
    for activation in ("silu", "gelu_tanh"):
        assert _same(numerics.swiglu(wide, up, activation), reference.swiglu(wide, up, activation))
    assert _same(numerics.softcap(wide, 3.0), reference.softcap(wide, 3.0))
    assert _same(numerics.softmax(wide), reference.softmax(wide))
    assert _same(numerics.log_softmax(wide), reference.log_softmax(wide))
    assert numerics.argmax(np.array([1, 3, 3, 2], np.float32)) == reference.argmax(np.array([1, 3, 3, 2])) == 1


@pytest.mark.parametrize(
    "options",
    [{}, {"window": 3}, {"softcap": 0.5}, {"q_offset": 3}, {"scale": 0.3, "window": 5, "softcap": 2.0},
     {"causal": False}],
)  # fmt: skip
def test_attention_is_the_kernels_bits(options):
    q, k, v = _gaussian(4, 9, 4, 16), _gaussian(5, 12, 2, 16), _gaussian(6, 12, 2, 16)
    assert _same(numerics.attention(q, k, v, **options), reference.attention(q, k, v, **options))


@pytest.mark.parametrize(
    "scaling",
    [
        None,
        {"rope_type": "linear", "factor": 2.0},
        {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
         "original_max_position_embeddings": 16},
        {"type": "longrope", "short_factor": [1.0, 2.0, 3.0, 4.0]},
    ],
)  # fmt: skip
def test_rope_is_the_kernels_bits(scaling):
    inv_freq = numerics.rope_inv_freq(16, 10000.0, rotary_dim=8, scaling=scaling)
    assert _same(inv_freq, reference.rope_inv_freq(16, 10000.0, rotary_dim=8, scaling=scaling))
    x, positions = _gaussian(4, 9, 4, 16), np.arange(9) * 97
    for interleaved in (False, True):
        expected = numerics.rope(x, positions, inv_freq, interleaved=interleaved)
        assert _same(expected, reference.rope(x, positions, inv_freq, interleaved))


def test_reference_rejects_an_unknown_rope_scaling():
    with pytest.raises(ValueError, match="unsupported rope scaling"):
        reference.rope_inv_freq(16, scaling={"rope_type": "dynamic"})


BASE = {
    "family": "llama",
    "vocabulary_size": 96,
    "hidden_size": 64,
    "intermediate_size": 96,
    "layers": 2,
    "heads": 4,
    "kv_heads": 2,
    "head_dim": 16,
    "context_length": 128,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}
GEMMA = {"norm_placement": "sandwich", "norm_unit_offset": True, "activation": "gelu_tanh"}
VARIANTS = {
    "llama": {"rope_scaling": {"rope_type": "llama3", "factor": 4.0, "original_max_position_embeddings": 16}},
    "qwen2": {"family": "qwen2", "attention_bias": True, "tie_word_embeddings": False},
    "qwen3": {"family": "qwen3", "qk_norm": True},
    "mistral": {"family": "mistral", "sliding_window": 3, "tie_word_embeddings": False},
    "olmo2": {"family": "olmo2", "norm_placement": "post", "qk_norm": True, "qk_norm_scope": "all"},
    "granite": {"family": "granite", "embedding_multiplier": 12.0, "attention_multiplier": 0.125,
                "residual_multiplier": 0.22, "logits_scaling": 8.0},
    "phi3": {"family": "phi3", "rotary_dim": 8, "sliding_window": 3, "tie_word_embeddings": False,
             "rope_scaling": {"rope_type": "longrope", "short_factor": [4.0, 1.0, 2.0, 1.5], "attention_factor": 1.2}},
    "gemma3": {"family": "gemma3", **GEMMA, "qk_norm": True, "embedding_multiplier": 8.0, "sliding_window": 3,
               "sliding_window_layers": (0,), "local_rope_theta": 100.0,
               "rope_scaling": {"rope_type": "linear", "factor": 2.0}},
    "gemma2": {"family": "gemma2", **GEMMA, "sliding_window": 3, "sliding_window_layers": (0,),
               "attention_softcap": 0.5, "logits_softcap": 2.0, "attention_multiplier": 0.3},
}  # fmt: skip


def _model(variant: str, quantize: str | None = None, steering=None) -> Transformer:
    config = TransformerConfig.from_dict({**BASE, **VARIANTS[variant]})
    tensors = {}
    for index, (name, shape) in enumerate(sorted(config.tensor_shapes().items())):
        values = _gaussian(100 + index, *shape) * np.float32(0.3)
        if name.endswith("norm.weight") and not config.norm_unit_offset:
            values = values + np.float32(1.0)
        tensors[name] = values
    return Transformer(config, tensors, quantize=quantize, steering=steering)


@pytest.mark.parametrize("quantize", [None, "q8_0", "q4_0"])
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_the_reference_decoder_gives_the_engines_logits(variant, quantize):
    model = _model(variant, quantize)
    twin = reference.ReferenceTransformer.from_engine_model(model)
    prompt = [5, 17, 3, 90, 41, 41, 8]
    cache = model.new_cache()
    assert _same(model.forward_cached(prompt, cache).reshape(-1), twin.forward(prompt))
    for token in (12, 77, 3):  # decoding one token at a time against the caches, past the sliding window
        prompt = [*prompt, token]
        assert _same(model.forward_cached(prompt, cache).reshape(-1), twin.forward([token]))


def test_the_reference_decoder_applies_steering():
    vector = _gaussian(99, 64)
    model = _model("llama", steering={1: vector})
    twin = reference.ReferenceTransformer.from_engine_model(model)
    assert _same(np.asarray(model.forward([1, 2, 3])).reshape(-1), twin.forward([1, 2, 3]))


def test_reference_generation_matches_the_engine(tmp_path):
    engine = DllmEngine.from_model_file(_import(tmp_path, tokenizer=BYTE_LEVEL_TOKENIZER, vocabulary=264))
    check = verify.check_reference(engine, max_tokens=6)
    assert check.equal, check.results
    assert check.as_dict() == {
        "equal": True,
        "logits": "equal",
        "greedy": "equal",
        "sampled": "equal",
        "controlled": "equal",
        "rolled": "equal",
        "budgeted": "equal",
        "guided": "equal",
        "scored": "equal",
    }


def test_reference_check_reports_where_the_bits_part(tmp_path, monkeypatch):
    engine = DllmEngine.from_model_file(_import(tmp_path, tokenizer=BYTE_LEVEL_TOKENIZER, vocabulary=264))
    original = reference.ReferenceTransformer.forward

    def off_by_one_bit(self, tokens):
        logits = original(self, tokens).copy()
        logits.view(np.uint32)[3] ^= 1
        return logits

    monkeypatch.setattr(reference.ReferenceTransformer, "forward", off_by_one_bit)
    check = verify.check_reference(engine, max_tokens=4)
    assert not check.equal
    assert check.results["logits"] == "1 of 264 differ, first at token id 3"
    assert verify._first_difference([1, 2, 3], [1, 2, 4]) == "differ from token 2: engine [3], reference [4]"


def test_verify_reference_on_the_command_line(tmp_path, capsys, isolated):
    path = _import(tmp_path, tokenizer=BYTE_LEVEL_TOKENIZER, vocabulary=264)
    assert main(["--model", str(path), "verify", "--reference"]) == 0
    out = capsys.readouterr().out
    assert "against the reference implementation:" in out and "greedy:             equal" in out
    assert main(["--model", str(path), "verify", "--reference", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["reference"]["equal"] is True


def test_verify_reference_needs_a_model_file(capsys, isolated):
    assert main(["verify", "--reference"]) == 2
    assert "--reference needs a model file" in capsys.readouterr().err
