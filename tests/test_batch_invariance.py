"""Phase 6: the performance work must not change a single bit.

- Every SIMD code path and every thread count gives the bits of the plain scalar statement of each kernel.
- A row's result does not depend on which other rows (other tokens, other requests) are computed with it.
- Batched, concurrent and threaded model runs, and concurrent server requests under load, reproduce a lone request.
- Q8_0 quantisation matches a straightforward exact reference.
"""

from __future__ import annotations

import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient
from model_fixtures import TINY_LLAMA_CONFIG, write_hf_checkpoint

from etalii_dllm import _kernels, numerics
from etalii_dllm import engine as engine_module
from etalii_dllm.engine import DllmEngine, default_engine
from etalii_dllm.importing import import_model
from etalii_dllm.numerics import PackedWeight, QuantizedWeight, fingerprint
from etalii_dllm.sampling import SamplingOptions
from etalii_dllm.transformer import Transformer

THREAD_COUNTS = (1, 2, 3, 8)
ISAS = _kernels.supported_isas()

# Big enough for Q8_0 (dimensions multiple of 32) and for several 16-output panels, still fast.
CONFIG = {
    **TINY_LLAMA_CONFIG,
    "vocab_size": 96,
    "hidden_size": 64,
    "intermediate_size": 160,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
}


def gaussian(seed: int, *shape: int) -> np.ndarray:
    return numerics.fill_gaussian(seed, math.prod(shape)).reshape(shape)


@pytest.fixture(autouse=True)
def restore_kernel_settings():
    yield
    numerics.set_threads(0)
    _kernels.set_isa("best")


def every_setting():
    for isa in ISAS:
        _kernels.set_isa(isa)
        for count in THREAD_COUNTS:
            numerics.set_threads(count)
            yield isa, count


# -- kernels ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("rows", "inputs", "outputs"), [(1, 64, 48), (3, 37, 19), (70, 96, 33), (130, 5, 17)])
def test_linear_is_identical_on_every_code_path(rows, inputs, outputs):
    # Magnitudes from 1e-30 to 1e20 make any change in rounding order visible.
    exponents = np.clip((gaussian(2, rows, inputs) * 12).round(), -30, 20)
    x = (gaussian(1, rows, inputs) * np.float32(10.0) ** exponents).astype(np.float32)
    w = gaussian(3, outputs, inputs)
    b = gaussian(4, outputs)
    expected = _kernels.linear_reference(np.ascontiguousarray(x, dtype=np.float32), w, b).tobytes()
    packed = PackedWeight(w)
    for setting in every_setting():
        assert numerics.linear(x, w, b).numpy().tobytes() == expected, setting
        assert numerics.linear(x, packed, b).numpy().tobytes() == expected, setting


def test_attention_is_identical_on_every_code_path():
    q, k, v = gaussian(5, 9, 8, 16), gaussian(6, 40, 2, 16), gaussian(7, 40, 2, 24)
    expected = None
    for setting in every_setting():
        out = numerics.attention(q, k, v, q_offset=31).fingerprint()
        expected = expected or out
        assert out == expected, setting


def test_quantized_linear_is_identical_on_every_code_path():
    weight = QuantizedWeight(gaussian(8, 50, 96))
    x = gaussian(9, 67, 96)
    expected = None
    for setting in every_setting():
        out = numerics.linear(x, weight, gaussian(10, 50)).fingerprint()
        expected = expected or out
        assert out == expected, setting


def test_activations_are_identical_for_every_thread_count():
    """Elementwise kernels run over fixed chunks on the pool; the fused gated activation rounds the activation
    before the multiply, exactly as the two separate steps do."""
    gate, up = gaussian(11, 7, 5003) * np.float32(4.0), gaussian(12, 7, 5003)  # several chunks and a ragged end
    numerics.set_threads(1)
    expected = {
        "silu": numerics.silu(gate).numpy().tobytes(),
        "gelu": numerics.gelu(gate).numpy().tobytes(),
        "gelu_tanh": numerics.gelu(gate, approximate="tanh").numpy().tobytes(),
        "softcap": numerics.softcap(gate, 3.0).numpy().tobytes(),
        "swiglu": (numerics.silu(gate).numpy() * up).tobytes(),
        "geglu": (numerics.gelu(gate, approximate="tanh").numpy() * up).tobytes(),
    }
    single = np.array([numerics.silu(np.array([v], np.float32)).numpy()[0] for v in gate.reshape(-1)[:64]])
    assert single.tobytes() == numerics.silu(gate).numpy().reshape(-1)[:64].tobytes()
    for setting in every_setting():
        actual = {
            "silu": numerics.silu(gate).numpy().tobytes(),
            "gelu": numerics.gelu(gate).numpy().tobytes(),
            "gelu_tanh": numerics.gelu(gate, approximate="tanh").numpy().tobytes(),
            "softcap": numerics.softcap(gate, 3.0).numpy().tobytes(),
            "swiglu": numerics.swiglu(gate, up).numpy().tobytes(),
            "geglu": numerics.swiglu(gate, up, "gelu_tanh").numpy().tobytes(),
        }
        assert actual == expected, setting
    with pytest.raises(ValueError):
        numerics.swiglu(gate, up, "relu")
    with pytest.raises(ValueError):
        numerics.swiglu(gate, up[:, :5])


def test_linear_backward_is_identical_for_every_thread_count():
    x, w, dy = gaussian(23, 37, 48), gaussian(24, 40, 48), gaussian(25, 37, 40)
    expected = None
    for count in THREAD_COUNTS:
        numerics.set_threads(count)
        out = [t.fingerprint() for t in numerics.linear_backward(x, w, dy, with_bias=True)]
        expected = expected or out
        assert out == expected, count


def test_a_row_does_not_depend_on_its_batch():
    w, x = PackedWeight(gaussian(11, 40, 64)), gaussian(12, 90, 64)
    q8 = QuantizedWeight(gaussian(11, 40, 64))
    for weight in (w, q8):
        alone = [numerics.linear(x[i : i + 1], weight).numpy() for i in range(len(x))]
        for start, stop in [(0, 90), (5, 6), (3, 71), (64, 90), (1, 66)]:
            batch = numerics.linear(x[start:stop], weight).numpy()
            for i in range(start, stop):
                assert batch[i - start].tobytes() == alone[i].tobytes()


def test_kernels_called_from_many_threads_at_once():
    w, x = PackedWeight(gaussian(13, 200, 128)), gaussian(14, 33, 128)
    q, k, v = gaussian(15, 4, 8, 16), gaussian(16, 64, 2, 16), gaussian(17, 64, 2, 16)
    expected = (numerics.linear(x, w).fingerprint(), numerics.attention(q, k, v).fingerprint())

    def work(_):
        return [(numerics.linear(x, w).fingerprint(), numerics.attention(q, k, v).fingerprint()) for _ in range(20)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        for results in pool.map(work, range(16)):
            assert all(result == expected for result in results)


# -- Q8_0 --------------------------------------------------------------------------------------------------------


def quantize_reference(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocks = x.astype(np.float32).reshape(-1, 32)
    d = np.max(np.abs(blocks), axis=1) / np.float32(127)
    inverse = np.where(d != 0, np.float32(1) / np.where(d != 0, d, np.float32(1)), np.float32(0)).astype(np.float32)
    q = np.clip(np.rint(blocks * inverse[:, None]), -127, 127)  # np.rint rounds half to even
    return q.astype(np.int8).reshape(x.shape), d.astype(np.float32).reshape(*x.shape[:-1], -1)


def test_quantization_matches_reference_and_rounds_half_to_even():
    w = gaussian(18, 12, 96)
    w[0, :32] = 0.0
    w[1, :32] = np.arange(32, dtype=np.float32) - 16.5  # d = 1: exact halves round to even
    values, scales = _kernels.quantize_q8_0(w)
    expected_values, expected_scales = quantize_reference(w)
    assert values.tobytes() == expected_values.tobytes()
    assert scales.tobytes() == expected_scales.tobytes()
    halves = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5] + [0.0] * 25 + [127.0], dtype=np.float32)
    values, scales = _kernels.quantize_q8_0(halves.reshape(1, 32))
    assert scales[0, 0] == 1.0
    assert list(values[0, :6]) == [-2, -2, 0, 0, 2, 2]


def test_quantized_linear_matches_exact_reference():
    w, x, b = gaussian(19, 5, 64), gaussian(20, 3, 64), gaussian(21, 5)
    wq, ws = _kernels.quantize_q8_0(w)
    xq, xs = _kernels.quantize_q8_0(x)
    expected = np.empty((3, 5), dtype=np.float32)
    for r in range(3):
        for n in range(5):
            acc = 0.0  # Python floats are IEEE doubles: the kernel's order, spelled out
            for block in range(2):
                part = slice(32 * block, 32 * block + 32)
                isum = int(np.dot(xq[r, part].astype(np.int64), wq[n, part].astype(np.int64)))
                acc += (float(xs[r, block]) * float(ws[n, block])) * float(isum)
            expected[r, n] = np.float32(acc + float(b[n]))
    assert numerics.linear(x, QuantizedWeight(w), b).numpy().tobytes() == expected.tobytes()
    approx = numerics.linear(x, w, b).numpy()
    assert np.abs(expected - approx).max() < 0.05 * np.abs(approx).max()


def test_quantized_weight_rejects_unsupported_shapes():
    with pytest.raises(ValueError):
        QuantizedWeight(gaussian(22, 4, 40))
    with pytest.raises(ValueError):
        QuantizedWeight(gaussian(22, 4, 64), "q4_0")


# -- model -------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    directory = tmp_path_factory.mktemp("batch")
    write_hf_checkpoint(directory / "checkpoint", CONFIG)
    import_model(directory / "checkpoint", directory / "model.dllm")
    return directory / "model.dllm"


@pytest.fixture(scope="module", params=[None, "q8_0"])
def model(request, model_path) -> Transformer:
    return Transformer.from_file(model_path, quantize=request.param)


SEQUENCES = [[1, 5, 9, 33, 70], [4], [90, 91, 92, 93, 94, 95, 1, 2, 3, 4, 5, 6, 7, 8], [60, 61], [1, 5, 9, 11]]


def test_quantization_changes_the_fingerprint(model_path):
    plain = Transformer.from_file(model_path)
    quantized = Transformer.from_file(model_path, quantize="q8_0")
    assert quantized.weights_fingerprint != plain.weights_fingerprint
    assert all(isinstance(quantized._w[f"layers.0.mlp.{m}.weight"], QuantizedWeight) for m in ("gate", "up", "down"))
    difference = np.abs(quantized.forward(SEQUENCES[0]) - plain.forward(SEQUENCES[0])).max()
    assert 0 < difference < 0.1 * np.abs(plain.forward(SEQUENCES[0])).max()


def test_forward_batch_matches_lone_forward(model):
    alone = [fingerprint(model.forward(tokens)) for tokens in SEQUENCES]
    assert [fingerprint(logits) for logits in model.forward_batch(SEQUENCES)] == alone
    assert [fingerprint(logits) for logits in model.forward_batch(SEQUENCES[::-1])] == alone[::-1]


def test_forward_batch_with_caches_at_different_positions(model):
    caches = [model.new_cache() for _ in SEQUENCES]
    for tokens, cache in zip(SEQUENCES, caches, strict=True):
        model.forward_cached(tokens[: len(tokens) // 2 + 1], cache)  # prefill different lengths first
    extended = [[*tokens, 7, 8] for tokens in SEQUENCES]
    expected = [fingerprint(model.forward(tokens)) for tokens in extended]
    assert [fingerprint(logits) for logits in model.forward_batch(extended, caches)] == expected
    # Every sequence then decodes one token at a time, all in one batch per step.
    for step in range(3):
        extended = [[*tokens, 10 + step + i] for i, tokens in enumerate(extended)]
        expected = [fingerprint(model.forward(tokens)) for tokens in extended]
        assert [fingerprint(logits) for logits in model.forward_batch(extended, caches)] == expected


def test_forward_batch_rejects_shared_caches(model):
    cache = model.new_cache()
    with pytest.raises(ValueError):
        model.forward_batch([[1], [2]], [cache, cache])


def test_concurrent_forwards_under_changing_thread_counts(model):
    work = [SEQUENCES[i % len(SEQUENCES)][: 1 + i % 5] for i in range(48)]
    expected = [fingerprint(model.forward(tokens)) for tokens in work]
    stop = threading.Event()

    def churn():  # resizes the kernel thread pool while requests are running
        count = 0
        while not stop.is_set():
            numerics.set_threads(1 + count % 4)
            count += 1

    changer = threading.Thread(target=churn)
    changer.start()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda tokens: fingerprint(model.forward(tokens)), work))
    finally:
        stop.set()
        changer.join()
    assert results == expected


# -- served --------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chat_model_path(tmp_path_factory):
    tokenizers = pytest.importorskip("tokenizers")
    del tokenizers
    from test_bpe import smollm2_style
    from test_chat_template import SMOLLM2

    reference = smollm2_style()
    directory = tmp_path_factory.mktemp("served")
    config = {**CONFIG, "vocab_size": reference.get_vocab_size(), "eos_token_id": 2}
    write_hf_checkpoint(directory / "checkpoint", config, tokenizer_json=json.loads(reference.to_str()))
    tokenizer_config = {"chat_template": SMOLLM2, "eos_token": "<|im_end|>", "bos_token": None}
    (directory / "checkpoint" / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    import_model(directory / "checkpoint", directory / "chat.dllm", repository="example/batch-chat")
    return directory / "chat.dllm"


@pytest.fixture(params=[None, "q8_0"])
def served(request, chat_model_path, monkeypatch):
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, str(chat_model_path))
    if request.param:
        monkeypatch.setenv(engine_module.QUANTIZE_ENVIRONMENT_VARIABLE, request.param)
    else:
        monkeypatch.delenv(engine_module.QUANTIZE_ENVIRONMENT_VARIABLE, raising=False)
    default_engine.cache_clear()
    yield default_engine()
    default_engine.cache_clear()


def requests() -> list[dict]:
    bodies = []
    for i in range(12):
        body = {
            "model": "x",
            "messages": [{"role": "user", "content": f"Request {i}: count to {i % 4 + 2}."}],
            "max_tokens": 6 + i % 5,
            "temperature": 0.0 if i % 3 == 0 else 0.9,
            "seed": i,
            "stream": i % 2 == 1,
        }
        bodies.append(body)
    return bodies


def _uncounted(response: bytes) -> bytes:
    return re.sub(rb'"cached_tokens":\d+', b'"cached_tokens":0', response)


def test_concurrent_server_requests_match_lone_requests(served):
    from etalii_dllm.server.app import app

    client = TestClient(app)
    bodies = requests()
    alone = [client.post("/v1/chat/completions", json=body).content for body in bodies]
    load = bodies * 3  # 36 requests in flight on 12 workers, streamed and not, mixed samplers
    with ThreadPoolExecutor(max_workers=12) as pool:
        responses = list(pool.map(lambda body: client.post("/v1/chat/completions", json=body).content, load))
    # The prompt cache makes the cache counters in usage depend on earlier requests; every other byte may not.
    assert [_uncounted(r) for r in responses] == [_uncounted(r) for r in alone] * 3
    fingerprints = {json.loads(response)["system_fingerprint"] for response in alone if not response.startswith(b"d")}
    assert fingerprints == {served.system_fingerprint}


def test_thread_count_does_not_change_generation(chat_model_path):
    engine = DllmEngine.from_model_file(chat_model_path)
    options = SamplingOptions(temperature=0.8, seed=3)
    results = []
    for count in (1, 2, 4):
        numerics.set_threads(count)
        results.append(engine.complete("Threads never matter", 16, options).fingerprint)
    assert len(set(results)) == 1
