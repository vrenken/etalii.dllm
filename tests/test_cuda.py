"""The CUDA backend (issue #31): the GPU gives the CPU's bits, run after run.

Every GPU kernel is compared with its CPU kernel bit for bit on shapes that exercise the edges (single rows, odd
sizes, grouped-query heads, causal offsets, partial and interleaved RoPE, Q8_0), the Phase 1 golden kernel hashes are
reproduced on the GPU, repeated and concurrent runs are checked, and whole models (float and Q8_0) give the golden
logits and the same generations as on the CPU. These tests need an NVIDIA GPU and NVRTC (see etalii_dllm.cuda);
elsewhere they are skipped, except the ones that check the error paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from golden_values import KERNEL_FINGERPRINTS, TINY_LOGITS_FINGERPRINT
from model_fixtures import TINY_LLAMA_CONFIG, tiny_config, write_hf_checkpoint

from etalii_dllm import _kernels, cuda, numerics
from etalii_dllm import engine as engine_module
from etalii_dllm.engine import DllmEngine
from etalii_dllm.importing import import_model
from etalii_dllm.numerics import CudaQuantizedWeight, CudaWeight, QuantizedWeight, fingerprint
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.tensor import Tensor
from etalii_dllm.transformer import Transformer

gpu = pytest.mark.skipif(not cuda.available(), reason="needs an NVIDIA GPU and NVRTC")

REPEATS = 20


def gaussian(seed: int, *shape: int) -> np.ndarray:
    return numerics.fill_gaussian(seed, math.prod(shape)).reshape(shape)


def same_bits(a: Tensor | np.ndarray, b: Tensor | np.ndarray) -> bool:
    x = a.numpy() if isinstance(a, Tensor) else a
    y = b.numpy() if isinstance(b, Tensor) else b
    return x.shape == y.shape and x.tobytes() == y.tobytes()


# -- without a GPU -----------------------------------------------------------------------------------------------


def test_unknown_devices_are_rejected():
    with pytest.raises(ValueError):
        numerics.rms_norm(gaussian(1, 2, 4), device="tpu")
    with pytest.raises(ValueError):
        cuda.check_device("metal")


def test_configured_device(monkeypatch):
    monkeypatch.delenv(engine_module.DEVICE_ENVIRONMENT_VARIABLE, raising=False)
    assert engine_module.configured_device() == "cpu"
    monkeypatch.setenv(engine_module.DEVICE_ENVIRONMENT_VARIABLE, "CUDA")
    assert engine_module.configured_device() == "cuda"
    monkeypatch.setenv(engine_module.DEVICE_ENVIRONMENT_VARIABLE, "gpu0")
    with pytest.raises(ValueError):
        engine_module.configured_device()


@pytest.mark.skipif(cuda.available(), reason="checks the message on machines without CUDA")
def test_missing_cuda_is_reported_clearly():
    with pytest.raises(cuda.CudaUnavailableError):
        cuda.initialize()
    with pytest.raises(cuda.CudaUnavailableError):
        CudaWeight(gaussian(1, 4, 4))


# -- NVRTC without a GPU: the [cuda] install route --------------------------------------------------------------

# CI sets DLLM_REQUIRE_NVRTC=1 after installing the [cuda] extra: a missing or broken NVRTC fails, never skips.
nvrtc = pytest.mark.skipif(
    cuda.find_nvrtc() is None and not os.environ.get("DLLM_REQUIRE_NVRTC"), reason="needs NVRTC (the [cuda] extra)"
)


@nvrtc
@pytest.mark.parametrize("arch", [50, 61, 70, 75, 80, 86, 89, 90])
def test_kernels_compile_with_nvrtc(arch):
    """The embedded kernels.cu + math.hpp compile for every architecture from Maxwell to Hopper, reproducibly."""
    first = cuda.compile_kernels(arch)
    assert first.compiler.startswith("NVRTC ")
    assert first.architecture in (f"sm_{arch}", f"compute_{arch}")
    assert len(first.image) > 0
    second = cuda.compile_kernels(arch)
    assert hashlib.sha256(second.image).digest() == hashlib.sha256(first.image).digest()


@nvrtc
def test_nvrtc_errors_are_reported_clearly(tmp_path):
    with pytest.raises(cuda.CudaUnavailableError, match=r"cannot compile for compute capability 1\.0"):
        cuda.compile_kernels(10)
    with pytest.raises(cuda.CudaUnavailableError, match="cannot load NVRTC"):
        cuda.compile_kernels(86, nvrtc=tmp_path / "missing-nvrtc")


def test_compile_without_nvrtc_is_reported_clearly(monkeypatch):
    monkeypatch.setattr(cuda, "find_nvrtc", lambda: None)
    with pytest.raises(cuda.CudaUnavailableError, match="NVRTC not found"):
        cuda.compile_kernels(86)


# -- kernels -----------------------------------------------------------------------------------------------------


def gpu_kernel_outputs() -> dict[str, Tensor]:
    """test_kernels.kernel_outputs() with every kernel on the GPU (matmul has no GPU version: the models use linear)."""
    x = gaussian(20, 8, 64)
    w = gaussian(21, 48, 64)
    q, k, v = gaussian(22, 8, 4, 16), gaussian(23, 8, 2, 16), gaussian(24, 8, 2, 16)
    return {
        "linear": numerics.linear(x, CudaWeight(w), gaussian(25, 48)),
        "rms_norm": numerics.rms_norm(x, gaussian(26, 64), 1e-6, device="cuda"),
        "silu": numerics.silu(x, device="cuda"),
        "gelu": numerics.gelu(x, device="cuda"),
        "gelu_tanh": numerics.gelu(x, approximate="tanh", device="cuda"),
        "rope": numerics.rope(q, np.arange(8) * 97, numerics.rope_inv_freq(16, 10000.0), device="cuda"),
        "attention": numerics.attention(q, k, v, device="cuda"),
        "linear_q8": numerics.linear(x, CudaQuantizedWeight(w), gaussian(25, 48)),
    }


@gpu
def test_gpu_reproduces_the_golden_kernel_hashes():
    fingerprints = {name: tensor.fingerprint() for name, tensor in gpu_kernel_outputs().items()}
    assert fingerprints == {name: KERNEL_FINGERPRINTS[name] for name in fingerprints}


@gpu
def test_repeated_gpu_runs_are_identical():
    first = {name: tensor.fingerprint() for name, tensor in gpu_kernel_outputs().items()}
    for _ in range(REPEATS):
        assert {name: tensor.fingerprint() for name, tensor in gpu_kernel_outputs().items()} == first


@gpu
@pytest.mark.parametrize(("rows", "inputs", "outputs"), [(1, 1, 1), (1, 576, 1536), (3, 7, 130), (37, 129, 150)])
def test_linear_matches_cpu(rows, inputs, outputs):
    x, w, b = gaussian(1, rows, inputs), gaussian(2, outputs, inputs), gaussian(3, outputs)
    weight = CudaWeight(w)
    assert same_bits(numerics.linear(x, weight, b), numerics.linear(x, w, b))
    assert same_bits(numerics.linear(x, weight), _kernels.linear_reference(x, w, None))
    # Rows do not depend on their batch.
    assert same_bits(numerics.linear(x[-1:], weight).numpy()[0], numerics.linear(x, weight).numpy()[-1])


@gpu
@pytest.mark.parametrize(("rows", "inputs", "outputs"), [(1, 32, 1), (2, 64, 48), (9, 896, 130)])
def test_quantized_linear_matches_cpu(rows, inputs, outputs):
    x, w, b = gaussian(4, rows, inputs) * 3, gaussian(5, outputs, inputs), gaussian(6, outputs)
    x[0, :32] = 0.0  # an all-zero activation block has scale 0
    quantized = QuantizedWeight(w)
    on_gpu = CudaQuantizedWeight(quantized)
    assert same_bits(numerics.linear(x, on_gpu, b), numerics.linear(x, quantized, b))
    assert same_bits(numerics.linear(x, on_gpu), numerics.linear(x, quantized))


@gpu
def test_norm_and_activations_match_cpu():
    x = gaussian(7, 5, 300) * 4
    weight = gaussian(8, 300)
    for offset in (False, True):
        cpu = numerics.rms_norm(x, weight, 1e-5, add_unit_offset=offset)
        assert same_bits(numerics.rms_norm(x, weight, 1e-5, add_unit_offset=offset, device="cuda"), cpu)
    assert same_bits(numerics.rms_norm(x, None, device="cuda"), numerics.rms_norm(x, None))
    wide = np.concatenate([gaussian(9, 5000) * 8, np.array([0.0, -0.0, 30.0, -30.0, 1e-30, -800.0], np.float32)])
    assert same_bits(numerics.silu(wide, device="cuda"), numerics.silu(wide))
    assert same_bits(numerics.gelu(wide, device="cuda"), numerics.gelu(wide))
    assert same_bits(numerics.gelu(wide, approximate="tanh", device="cuda"), numerics.gelu(wide, approximate="tanh"))


@gpu
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_rope_matches_cpu(interleaved, inverse):
    x = gaussian(10, 11, 3, 24)
    positions = np.array([0, 1, 2, 7, 100, 1000, 4095, 32767, 131071, 5, 3])
    for rotary in (24, 16):
        freqs = numerics.rope_inv_freq(24, 500000.0, rotary_dim=rotary)
        cpu = numerics.rope(x, positions, freqs, interleaved=interleaved, inverse=inverse)
        on_gpu = numerics.rope(x, positions, freqs, interleaved=interleaved, inverse=inverse, device="cuda")
        assert same_bits(on_gpu, cpu)


@gpu
@pytest.mark.parametrize(("q_len", "kv_len", "q_heads", "kv_heads"), [(1, 1, 1, 1), (1, 300, 9, 3), (6, 20, 8, 2)])
def test_attention_matches_cpu(q_len, kv_len, q_heads, kv_heads):
    q = gaussian(11, q_len, q_heads, 16) * 2
    k = gaussian(12, kv_len, kv_heads, 16) * 2
    v = gaussian(13, kv_len, kv_heads, 24)
    for causal in (True, False):
        cpu = numerics.attention(q, k, v, causal=causal)
        assert same_bits(numerics.attention(q, k, v, causal=causal, device="cuda"), cpu)
    for window in (1, 4):
        cpu = numerics.attention(q, k, v, window=window)
        assert same_bits(numerics.attention(q, k, v, window=window, device="cuda"), cpu)
    if kv_len >= q_len:
        offset = kv_len - q_len
        # Prefill and one-at-a-time decoding give the same rows, as on the CPU.
        full = numerics.attention(q, k, v, q_offset=offset, device="cuda").numpy()
        for t in range(q_len):
            row = numerics.attention(q[t : t + 1], k[: offset + t + 1], v[: offset + t + 1], device="cuda")
            assert same_bits(row.numpy()[0], full[t])


@gpu
def test_concurrent_gpu_calls_match_serial_ones():
    x, w = gaussian(14, 4, 256), gaussian(15, 512, 256)
    weight = CudaWeight(w)
    expected = numerics.linear(x, w).fingerprint()
    q, k, v = gaussian(16, 3, 4, 16), gaussian(17, 30, 2, 16), gaussian(18, 30, 2, 16)
    attended = numerics.attention(q, k, v).fingerprint()

    def work(i: int) -> tuple[str, str]:
        return numerics.linear(x, weight).fingerprint(), numerics.attention(q, k, v, device="cuda").fingerprint()

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert set(pool.map(work, range(64))) == {(expected, attended)}


# -- models ------------------------------------------------------------------------------------------------------

PROMPT = [1, 17, 42, 5, 63, 0, 9, 9, 30]


@pytest.fixture(
    scope="module", params=["gemma2", "gemma3", "granite", "llama", "mistral", "olmo2", "phi3", "qwen2", "qwen3"]
)
def tiny_model_path(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(f"cuda-{request.param}")
    config = tiny_config(request.param)
    write_hf_checkpoint(directory / "checkpoint", config)
    import_model(directory / "checkpoint", directory / "model.dllm")
    return directory / "model.dllm"


@gpu
def test_gpu_model_gives_the_golden_logits(tiny_model_path):
    model = Transformer.from_file(tiny_model_path, device="cuda")
    assert model.device == "cuda"
    assert fingerprint(model.forward(PROMPT)) == TINY_LOGITS_FINGERPRINT[model.config.family]
    # The KV cache and repeated runs change nothing.
    cache = model.new_cache()
    stepwise = [model.forward_cached(PROMPT[: i + 1], cache) for i in range(len(PROMPT))]
    assert fingerprint(stepwise[-1]) == TINY_LOGITS_FINGERPRINT[model.config.family]
    assert {fingerprint(model.forward(PROMPT)) for _ in range(5)} == {TINY_LOGITS_FINGERPRINT[model.config.family]}


@gpu
def test_gpu_quantized_model_matches_cpu(tiny_model_path):
    cpu = Transformer.from_file(tiny_model_path, quantize="q8_0")
    on_gpu = Transformer.from_file(tiny_model_path, quantize="q8_0", device="cuda")
    assert on_gpu.weights_fingerprint == cpu.weights_fingerprint
    sequences = [PROMPT, PROMPT[:3], [4, 4, 4, 4]]
    assert [fingerprint(x) for x in on_gpu.forward_batch(sequences)] == [
        fingerprint(x) for x in cpu.forward_batch(sequences)
    ]


# A model with the chat template and tokenizer of SmolLM2, big enough for Q8_0.
CHAT_CONFIG = {
    **TINY_LLAMA_CONFIG,
    "hidden_size": 64,
    "intermediate_size": 160,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
}


@pytest.fixture(scope="module")
def chat_model_path(tmp_path_factory):
    pytest.importorskip("tokenizers")
    from test_bpe import smollm2_style
    from test_chat_template import SMOLLM2

    reference = smollm2_style()
    directory = tmp_path_factory.mktemp("cuda-chat")
    config = {**CHAT_CONFIG, "vocab_size": reference.get_vocab_size(), "eos_token_id": 2}
    write_hf_checkpoint(directory / "checkpoint", config, tokenizer_json=json.loads(reference.to_str()))
    tokenizer_config = {"chat_template": SMOLLM2, "eos_token": "<|im_end|>", "bos_token": None}
    (directory / "checkpoint" / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    import_model(directory / "checkpoint", directory / "chat.dllm", repository="example/cuda-chat")
    return directory / "chat.dllm"


@gpu
@pytest.mark.parametrize("quantize", [None, "q8_0"])
def test_gpu_generation_equals_cpu_generation(chat_model_path, quantize):
    cpu = DllmEngine.from_model_file(chat_model_path, quantize=quantize)
    on_gpu = DllmEngine.from_model_file(chat_model_path, quantize=quantize, device="cuda")
    assert on_gpu.system_fingerprint == cpu.system_fingerprint
    for options in (SamplingOptions(), SamplingOptions(temperature=0.8, seed=3)):
        expected = cpu.complete("The device never matters", 24, options).fingerprint
        assert {on_gpu.complete("The device never matters", 24, options).fingerprint for _ in range(3)} == {expected}
