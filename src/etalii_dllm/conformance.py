"""Conformance vectors: files that let any implementation prove it gives EtAlii.Dllm's bits (``dllm conformance``).

``write(directory)`` writes fixed inputs and this build's exact outputs for every kernel of the specification
(``docs/specification.md``): the transcendentals, the linear layers (float32, Q8_0, Q4_0) and quantisation, RMSNorm,
the activations, softmax, RoPE, attention, the random number generator, the sampler and two small decoders. Every
array is a raw little-endian file; ``manifest.json`` (canonical JSON: sorted keys, no whitespace) lists each case's
kernel, parameters and arrays with their dtype, shape and SHA-256. A port to another language reads the inputs, runs
its own kernels and compares the outputs bit for bit (any NaN matches any NaN: payloads are not specified).

``check(directory)`` does that for this build's compiled kernels (``implementation="kernels"``) or for the
independent reference implementation (``"reference"``). The inputs come from the portable random number generator
and the outputs from the portable kernels, so the manifest is the same on every machine; its SHA-256 is a golden value
(``tests/golden_values.py``).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "etalii-dllm-conformance"
VERSION = 1
IMPLEMENTATIONS = ("kernels", "reference")

Arrays = dict[str, np.ndarray]


def _gaussian(seed: int, *shape: int, scale: float = 1.0) -> np.ndarray:
    from etalii_dllm import numerics  # the generator is part of the specification (the "random" cases)

    return numerics.fill_gaussian(seed, math.prod(shape)).reshape(shape) * np.float32(scale)


_SPECIAL = [0.0, -0.0, 5e-324, -5e-324, 1e-310, 2.2250738585072014e-308, 1.0, -1.0, 0.125, -0.125, 0.5, 2.5, -2.5]
_SPECIAL += [6.0, -6.0, 22.0, 27.3, 709.78, 709.79, -745.2, -745.3, 1e6, 3e5, -1.6e6, 7.1e15, 1e19, -1e300]
_SPECIAL += [math.inf, -math.inf, math.nan]


def _arguments(seed: int) -> np.ndarray:
    parts = [_gaussian(seed, 256, scale=s).astype(np.float64) for s in (1e-3, 0.3, 3.0, 40.0)]
    return np.concatenate([*parts, np.linspace(-750.0, 750.0, 301), np.array(_SPECIAL)])


DECODERS: dict[str, dict[str, Any]] = {
    "llama": {
        "family": "llama", "vocabulary_size": 96, "hidden_size": 64, "intermediate_size": 96, "layers": 2,
        "heads": 4, "kv_heads": 2, "head_dim": 16, "context_length": 64, "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0, "tie_word_embeddings": True, "attention_bias": True,
        "rope_scaling": {"rope_type": "llama3", "factor": 4.0, "original_max_position_embeddings": 16},
    },
    "gemma2": {
        "family": "gemma2", "vocabulary_size": 96, "hidden_size": 64, "intermediate_size": 96, "layers": 2,
        "heads": 4, "kv_heads": 1, "head_dim": 16, "context_length": 64, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "tie_word_embeddings": True, "norm_placement": "sandwich", "norm_unit_offset": True,
        "activation": "gelu_tanh", "embedding_multiplier": 8.0, "sliding_window": 3, "sliding_window_layers": [0],
        "attention_softcap": 0.5, "logits_softcap": 2.0, "attention_multiplier": 0.3,
    },
    "qwen3_moe": {
        "family": "qwen3_moe", "vocabulary_size": 96, "hidden_size": 64, "intermediate_size": 96, "layers": 2,
        "heads": 4, "kv_heads": 2, "head_dim": 16, "context_length": 64, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "tie_word_embeddings": False, "qk_norm": True, "experts": 8, "experts_per_token": 3,
        "expert_intermediate_size": 32, "normalize_expert_weights": True, "dense_layers": [0],
    },
    "qwen2_moe": {
        "family": "qwen2_moe", "vocabulary_size": 96, "hidden_size": 64, "intermediate_size": 96, "layers": 2,
        "heads": 4, "kv_heads": 2, "head_dim": 16, "context_length": 64, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "tie_word_embeddings": False, "attention_bias": True, "experts": 4,
        "experts_per_token": 2, "expert_intermediate_size": 32, "shared_expert_intermediate_size": 48,
        "shared_expert_gate": True,
    },
}  # fmt: skip
"""Four small decoders that between them use every architecture option the kernels see: biases, GQA and MQA, llama3
RoPE scaling, sandwich (1 + w) norms, GELU, scaled embeddings, a sliding window, both soft-caps, QK-norm, a mixture
of experts next to a dense layer and a gated shared expert."""


def _decoder_tensors(name: str) -> Arrays:
    from etalii_dllm.architecture import TransformerConfig

    config = TransformerConfig.from_dict(DECODERS[name])
    tensors = {}
    for index, (tensor, shape) in enumerate(sorted(config.tensor_shapes().items())):
        values = _gaussian(500 + index, *shape, scale=0.3)
        if tensor.endswith("norm.weight") and not config.norm_unit_offset:
            values = values + np.float32(1.0)
        tensors["tensor." + tensor] = values
    return tensors


def cases() -> list[tuple[str, str, dict[str, Any], Arrays]]:
    """Every case: (name, kernel, parameters, inputs)."""
    items: list[tuple[str, str, dict[str, Any], Arrays]] = []
    for i, name in enumerate(("exp", "log", "sin", "cos", "tanh", "erf", "atan")):
        items.append((name, name, {}, {"x": _arguments(10 + i)}))
    items.append(("acos", "acos", {}, {"x": np.concatenate([np.linspace(-1.1, 1.1, 441), _arguments(17) / 50])}))
    x, w, bias = _gaussian(20, 5, 96), _gaussian(21, 40, 96), _gaussian(22, 40)
    items.append(("linear", "linear", {"quantize": None}, {"x": x, "weight": w, "bias": bias}))
    items.append(("linear-no-bias", "linear", {"quantize": None}, {"x": x, "weight": w}))
    for kind in ("q8_0", "q4_0"):
        items.append((f"quantize-{kind}", "quantize", {"kind": kind}, {"x": w}))
        items.append((f"linear-{kind}", "linear", {"quantize": kind}, {"x": x, "weight": w, "bias": bias}))
    items.append(("matmul", "matmul", {}, {"a": x, "b": _gaussian(23, 96, 24)}))
    for unit in (False, True):
        params = {"eps": 1e-6, "add_unit_offset": unit}
        items.append((f"rms-norm-unit-{str(unit).lower()}", "rms_norm", params, {"x": x, "weight": _gaussian(24, 96)}))
    wide = np.concatenate([_gaussian(25, 2000, scale=6.0), np.array([0.0, -0.0, 1e-40, 30.0, -30.0], np.float32)])
    for name in ("silu", "gelu", "gelu_tanh", "softmax", "log_softmax"):
        items.append((name, name, {}, {"x": wide}))
    for activation in ("silu", "gelu_tanh"):
        params = {"activation": activation}
        items.append((f"swiglu-{activation}", "swiglu", params, {"gate": wide, "up": _gaussian(26, wide.size)}))
    items.append(("softcap", "softcap", {"cap": 3.0}, {"x": wide}))
    items.append(("sigmoid", "sigmoid", {}, {"x": wide}))
    router = _gaussian(34, 40, 16, scale=3.0)
    router[0, :] = 1.0  # all experts tied
    router[1, [2, 5, 9]] = 50.0  # a tie for first place
    router[2, :] = -np.inf
    router[2, [7, 3]] = 0.0  # only two experts can win
    for k, normalize in ((1, False), (2, True), (4, False), (4, True)):
        name = f"moe-route-top{k}" + ("-normalized" if normalize else "")
        items.append((name, "moe_route", {"k": k, "normalize": normalize}, {"logits": router}))
    scalings: list[tuple[str, dict[str, Any] | None]] = [
        ("default", None),
        ("linear", {"rope_type": "linear", "factor": 2.0}),
        ("llama3", {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 16}),
        ("longrope", {"rope_type": "longrope", "short_factor": [1.0, 2.0, 3.0, 4.0]}),
        ("longrope-long", {"rope_type": "longrope", "short_factor": [1.0] * 4, "long_factor": [1.5, 2.0, 3.0, 4.0],
                           "factor_set": "long"}),
        ("yarn", {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64}),
    ]  # fmt: skip
    for name, scaling in scalings:
        params = {"head_dim": 16, "theta": 10000.0, "rotary_dim": 8, "scaling": scaling}
        items.append((f"rope-inv-freq-{name}", "rope_inv_freq", params, {}))
    from etalii_dllm import reference

    inv_freq = reference.rope_inv_freq(16, 10000.0, rotary_dim=8)
    rope_inputs = {"x": _gaussian(27, 9, 4, 16), "positions": np.arange(9, dtype=np.int64) * 97, "inv_freq": inv_freq}
    for interleaved in (False, True):
        items.append((f"rope-interleaved-{str(interleaved).lower()}", "rope", {"interleaved": interleaved},
                      rope_inputs))  # fmt: skip
    qkv = {"q": _gaussian(28, 9, 4, 16), "k": _gaussian(29, 12, 2, 16), "v": _gaussian(30, 12, 2, 16)}
    attention_cases = {
        "attention-causal": {"scale": 0.25, "causal": True, "q_offset": 3, "window": None, "softcap": None},
        "attention-window": {"scale": 0.25, "causal": True, "q_offset": 3, "window": 3, "softcap": None},
        "attention-softcap": {"scale": 0.3, "causal": True, "q_offset": 3, "window": 5, "softcap": 2.0},
        "attention-full": {"scale": 0.25, "causal": False, "q_offset": 0, "window": None, "softcap": None},
    }
    for name, params in attention_cases.items():
        items.append((name, "attention", params, qkv))
    items.append(("random", "random", {"seed": 42, "count": 64}, {}))
    items.append(("random-large-seed", "random", {"seed": 2**64 - 1, "count": 16}, {}))
    logits = _gaussian(31, 24, 300, scale=4.0)
    for name, options in {
        "sample-greedy": {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "seed": 0},
        "sample-temperature": {"temperature": 0.8, "top_k": 0, "top_p": 1.0, "seed": 7},
        "sample-top-k-top-p": {"temperature": 1.3, "top_k": 50, "top_p": 0.9, "seed": 11},
    }.items():
        items.append((name, "sample", options, {"logits": logits}))
    history = np.array([int(np.argmax(row)) for row in logits[::3]], dtype=np.int64)
    for name, options in {
        "sample-penalties": {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "seed": 5, "min_p": 0.05,
                             "repetition_penalty": 1.3, "repeat_last_n": 8, "frequency_penalty": 0.5,
                             "presence_penalty": 0.25, "logit_bias": [[3, 2.5], [200, -1.0]]},
        "sample-penalties-greedy": {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "seed": 0,
                                    "repetition_penalty": 1.5, "repeat_last_n": -1, "frequency_penalty": 0.3,
                                    "logit_bias": [[7, 1.0]]},
        "sample-watermark": {"temperature": 0.9, "top_k": 0, "top_p": 1.0, "seed": 9, "watermark_key": "conformance",
                             "watermark_gamma": 0.3, "watermark_delta": 3.0},
        "sample-watermark-greedy": {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "seed": 0, "watermark_key": "k",
                                    "watermark_delta": 4.0, "logit_bias": [[3, 1.0]]},
    }.items():  # fmt: skip
        items.append((name, "sample_controls", options, {"logits": logits, "prompt": history}))
    other = _gaussian(32, 24, 300, scale=4.0)
    items.append(("guided", "guided", {"scale": 1.75}, {"logits": logits, "other": other}))
    items.append(("contrasted", "contrasted", {"alpha": 0.1, "beta": 0.5}, {"logits": logits, "other": other}))
    members = np.stack([logits, other, _gaussian(33, 24, 300, scale=2.0)])
    items.append(("ensembled", "ensembled", {"weights": [1.0, 0.5, 0.25]}, {"members": members}))
    tokens = np.array([5, 17, 3, 90, 41, 41, 8, 12, 77], dtype=np.int64)
    for name in DECODERS:
        for quantize in (None, "q8_0"):
            params = {"config": DECODERS[name], "quantize": quantize}
            label = f"decoder-{name}" + (f"-{quantize}" if quantize else "")
            items.append((label, "decoder", params, {"tokens": tokens, **_decoder_tensors(name)}))
    return items


# -- the two implementations --------------------------------------------------------------------------------------


def _kernels(kernel: str, params: Mapping[str, Any], inputs: Arrays) -> Arrays:
    from etalii_dllm import numerics
    from etalii_dllm.sampling import Sampler, SamplingOptions

    def array(value: Any) -> np.ndarray:
        return np.asarray(value.numpy() if hasattr(value, "numpy") else value)

    if kernel in ("exp", "log", "sin", "cos", "tanh", "erf", "atan", "acos"):
        f = getattr(numerics, kernel)
        return {"y": np.array([f(float(v)) for v in inputs["x"]], dtype=np.float64)}
    if kernel == "linear":
        weight = inputs["weight"]
        if params["quantize"]:
            weight = numerics.QuantizedWeight(weight, params["quantize"])
        return {"y": array(numerics.linear(inputs["x"], weight, inputs.get("bias")))}
    if kernel == "quantize":
        quantized = numerics.QuantizedWeight(inputs["x"], params["kind"])
        return {"values": array(quantized.int8_values()), "scales": array(quantized.scales)}
    if kernel == "matmul":
        return {"y": array(numerics.matmul(inputs["a"], np.ascontiguousarray(inputs["b"])))}
    if kernel == "rms_norm":
        normed = numerics.rms_norm(
            inputs["x"], inputs["weight"], params["eps"], add_unit_offset=params["add_unit_offset"]
        )
        return {"y": array(normed)}
    if kernel in ("silu", "softmax", "log_softmax"):
        return {"y": array(getattr(numerics, kernel)(inputs["x"]))}
    if kernel == "gelu":
        return {"y": array(numerics.gelu(inputs["x"]))}
    if kernel == "gelu_tanh":
        return {"y": array(numerics.gelu(inputs["x"], approximate="tanh"))}
    if kernel == "swiglu":
        return {"y": array(numerics.swiglu(inputs["gate"], inputs["up"], params["activation"]))}
    if kernel == "softcap":
        return {"y": array(numerics.softcap(inputs["x"], params["cap"]))}
    if kernel == "sigmoid":
        return {"y": array(numerics.sigmoid_elementwise(inputs["x"]))}
    if kernel == "moe_route":
        experts, weights = numerics.moe_route(inputs["logits"], params["k"], params["normalize"])
        return {"experts": experts, "weights": weights}
    if kernel == "rope_inv_freq":
        values = numerics.rope_inv_freq(
            params["head_dim"], params["theta"], rotary_dim=params["rotary_dim"], scaling=params["scaling"]
        )
        return {"y": array(values)}
    if kernel == "rope":
        rotated = numerics.rope(inputs["x"], inputs["positions"], inputs["inv_freq"], interleaved=params["interleaved"])
        return {"y": array(rotated)}
    if kernel == "attention":
        out = numerics.attention(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            scale=params["scale"],
            causal=params["causal"],
            q_offset=params["q_offset"],
            window=params["window"],
            softcap=params["softcap"],
        )
        return {"y": array(out)}
    if kernel == "random":
        generator = numerics.DeterministicRandom(params["seed"])
        u64 = np.array([generator.next_u64() for _ in range(params["count"])], dtype=np.uint64)
        doubles = np.array([generator.next_double() for _ in range(params["count"])], dtype=np.float64)
        gaussians = np.array([generator.next_gaussian() for _ in range(params["count"])], dtype=np.float32)
        return {"u64": u64, "doubles": doubles, "gaussians": gaussians}
    if kernel == "sample":
        sampler = Sampler(SamplingOptions(params["temperature"], params["top_k"], params["top_p"], params["seed"]))
        return {"tokens": np.array([sampler.sample(row) for row in inputs["logits"]], dtype=np.int64)}
    if kernel == "sample_controls":
        sampler = Sampler(SamplingOptions.from_record(params), [int(t) for t in inputs["prompt"]])
        tokens = []
        for row in inputs["logits"]:
            tokens.append(sampler.sample(row))
            sampler.accept(tokens[-1])
        return {"tokens": np.array(tokens, dtype=np.int64)}
    if kernel in ("guided", "contrasted", "ensembled"):
        from etalii_dllm import guidance

        if kernel == "guided":
            rows = [
                guidance.guided(a, b, params["scale"]) for a, b in zip(inputs["logits"], inputs["other"], strict=True)
            ]
        elif kernel == "contrasted":
            rows = [guidance.contrasted(a, b, params["alpha"], params["beta"])
                    for a, b in zip(inputs["logits"], inputs["other"], strict=True)]  # fmt: skip
        else:
            stack = inputs["members"]
            rows = [guidance.ensembled([(stack[m][i], w) for m, w in enumerate(params["weights"])])
                    for i in range(stack.shape[1])]  # fmt: skip
        return {"y": np.stack(rows).astype(np.float32)}
    if kernel == "decoder":
        from etalii_dllm.architecture import TransformerConfig
        from etalii_dllm.transformer import Transformer

        config = TransformerConfig.from_dict(params["config"])
        tensors = {name[len("tensor.") :]: v for name, v in inputs.items() if name.startswith("tensor.")}
        model = Transformer(config, tensors, quantize=params["quantize"])
        cache = model.new_cache()
        tokens = [int(t) for t in inputs["tokens"]]
        rows = [array(model.forward_cached(tokens[: i + 1], cache)).reshape(-1) for i in range(len(tokens))]
        return {"logits": np.stack(rows)}
    raise ValueError(f"unknown kernel {kernel!r}")


def _reference(kernel: str, params: Mapping[str, Any], inputs: Arrays) -> Arrays:
    from etalii_dllm import reference as r

    if kernel in ("exp", "log", "sin", "cos", "tanh", "erf", "atan", "acos"):
        return {"y": getattr(r, kernel)(inputs["x"])}
    if kernel == "linear":
        weight = r.Weight(inputs["weight"], params["quantize"])
        return {"y": r.linear(inputs["x"], weight, inputs.get("bias"))}
    if kernel == "quantize":
        values, scales = r.quantize(inputs["x"], params["kind"])
        return {"values": values, "scales": scales}
    if kernel == "matmul":
        return {"y": r.matmul(inputs["a"], inputs["b"])}
    if kernel == "rms_norm":
        return {"y": r.rms_norm(inputs["x"], inputs["weight"], params["eps"], params["add_unit_offset"])}
    if kernel in ("silu", "gelu", "softmax", "log_softmax"):
        return {"y": getattr(r, kernel)(inputs["x"])}
    if kernel == "gelu_tanh":
        return {"y": r.gelu(inputs["x"], "tanh")}
    if kernel == "swiglu":
        return {"y": r.swiglu(inputs["gate"], inputs["up"], params["activation"])}
    if kernel == "softcap":
        return {"y": r.softcap(inputs["x"], params["cap"])}
    if kernel == "sigmoid":
        return {"y": r.sigmoid_float(inputs["x"])}
    if kernel == "moe_route":
        chosen, weights = r.moe_route(inputs["logits"], params["k"], params["normalize"])
        return {"experts": np.array(chosen, dtype=np.int64), "weights": weights}
    if kernel == "rope_inv_freq":
        values = r.rope_inv_freq(
            params["head_dim"], params["theta"], rotary_dim=params["rotary_dim"], scaling=params["scaling"]
        )
        return {"y": values}
    if kernel == "rope":
        return {"y": r.rope(inputs["x"], inputs["positions"], inputs["inv_freq"], params["interleaved"])}
    if kernel == "attention":
        out = r.attention(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            scale=params["scale"],
            causal=params["causal"],
            q_offset=params["q_offset"],
            window=params["window"],
            softcap=params["softcap"],
        )
        return {"y": out}
    if kernel == "random":
        generator = r.Random(params["seed"])
        u64 = np.array([generator.next_u64() for _ in range(params["count"])], dtype=np.uint64)
        doubles = np.array([generator.next_double() for _ in range(params["count"])], dtype=np.float64)
        gaussians = np.array([generator.next_gaussian() for _ in range(params["count"])], dtype=np.float32)
        return {"u64": u64, "doubles": doubles, "gaussians": gaussians}
    if kernel == "sample":
        sampler = r.Sampler(params["temperature"], params["top_k"], params["top_p"], params["seed"])
        return {"tokens": np.array([sampler.sample(row) for row in inputs["logits"]], dtype=np.int64)}
    if kernel == "sample_controls":
        controls = {k: v for k, v in params.items() if k not in ("temperature", "top_k", "top_p", "seed")}
        controls["logit_bias"] = {int(t): float(b) for t, b in controls.get("logit_bias", [])}
        sampler = r.Sampler(params["temperature"], params["top_k"], params["top_p"], params["seed"], **controls)
        sampler.begin([int(t) for t in inputs["prompt"]])
        tokens = []
        for row in inputs["logits"]:
            tokens.append(sampler.sample(row))
            sampler.accept(tokens[-1])
        return {"tokens": np.array(tokens, dtype=np.int64)}
    if kernel == "guided":
        pairs = zip(inputs["logits"], inputs["other"], strict=True)
        return {"y": np.stack([r.guided(a, b, params["scale"]) for a, b in pairs])}
    if kernel == "contrasted":
        pairs = zip(inputs["logits"], inputs["other"], strict=True)
        return {"y": np.stack([r.contrasted(a, b, params["alpha"], params["beta"]) for a, b in pairs])}
    if kernel == "ensembled":
        stack = inputs["members"]
        weights = params["weights"]
        rows = [r.ensembled([(stack[m][i], w) for m, w in enumerate(weights)]) for i in range(stack.shape[1])]
        return {"y": np.stack(rows)}
    if kernel == "decoder":
        from etalii_dllm.architecture import TransformerConfig

        config = TransformerConfig.from_dict(params["config"])
        tensors = {name[len("tensor.") :]: v for name, v in inputs.items() if name.startswith("tensor.")}
        model = r.ReferenceTransformer(config, tensors, quantize=params["quantize"])
        return {"logits": np.stack([model.forward([int(t)]) for t in inputs["tokens"]])}
    raise ValueError(f"unknown kernel {kernel!r}")


RUNNERS: dict[str, Callable[[str, Mapping[str, Any], Arrays], Arrays]] = {
    "kernels": _kernels,
    "reference": _reference,
}


# -- files ---------------------------------------------------------------------------------------------------------


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _little_endian(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=values.dtype.newbyteorder("<"))


_PORTABLE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/")
"""Characters every file system accepts in a name (``:`` is a stream separator on Windows)."""


def _save(directory: Path, case: str, key: str, values: np.ndarray) -> dict[str, Any]:
    data = _little_endian(np.asarray(values))
    if data.dtype.kind == "f":  # NaN payloads and signs are not specified (they differ between CPUs): write one NaN
        data = np.where(np.isnan(data), data.dtype.type(math.nan), data).astype(data.dtype)
    relative = f"{case}/{key}.bin"
    if not set(relative) <= _PORTABLE:
        raise ValueError(f"{relative}: not a portable file name")
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data.tobytes()
    path.write_bytes(raw)
    return {
        "file": relative,
        "dtype": data.dtype.str,
        "shape": list(data.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def write(directory: str | Path) -> tuple[int, str]:
    """Writes the vectors (inputs and this build's outputs) to ``directory``; returns the number of cases and the
    manifest's SHA-256."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, kernel, params, inputs in cases():
        outputs = _kernels(kernel, params, inputs)
        entries.append(
            {
                "name": name,
                "kernel": kernel,
                "params": params,
                "inputs": {key: _save(directory, name, key, value) for key, value in sorted(inputs.items())},
                "outputs": {key: _save(directory, name, key, value) for key, value in sorted(outputs.items())},
            }
        )
    manifest = _canonical({"format": FORMAT, "version": VERSION, "cases": entries})
    (directory / "manifest.json").write_bytes(manifest)
    return len(entries), hashlib.sha256(manifest).hexdigest()


def _load(directory: Path, entry: Mapping[str, Any]) -> np.ndarray:
    raw = (directory / entry["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"{entry['file']}: the file does not match its SHA-256 in the manifest")
    return np.frombuffer(raw, dtype=np.dtype(entry["dtype"])).reshape(entry["shape"]).copy()


def same_bits(expected: np.ndarray, actual: np.ndarray) -> bool:
    """Equal shapes and bits; for floats any NaN equals any NaN."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.shape != actual.shape:
        return False
    if expected.dtype.kind == "f":
        if actual.dtype.kind != "f" or expected.dtype.itemsize != actual.dtype.itemsize:
            return False
        a = _little_endian(expected).astype(expected.dtype.newbyteorder("="))
        b = actual.astype(a.dtype)
        both_nan = np.isnan(a) & np.isnan(b)
        unsigned = np.dtype(f"u{a.dtype.itemsize}")
        return bool(np.all(both_nan | (a.view(unsigned) == b.view(unsigned))))
    return bool(np.array_equal(expected, actual.astype(expected.dtype)))


@dataclass
class CheckResult:
    manifest_sha256: str
    passed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed


def check(directory: str | Path, implementation: str = "kernels") -> CheckResult:
    """Runs every case of the vectors in ``directory`` with ``implementation`` and compares the outputs."""
    if implementation not in RUNNERS:
        raise ValueError(f"unknown implementation {implementation!r}; choose from {', '.join(IMPLEMENTATIONS)}")
    directory = Path(directory)
    raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise ValueError(f"{directory}: not {FORMAT} version {VERSION} vectors")
    result = CheckResult(hashlib.sha256(raw).hexdigest())
    for case in manifest["cases"]:
        name = case["name"]
        try:
            inputs = {key: _load(directory, entry) for key, entry in case["inputs"].items()}
            expected = {key: _load(directory, entry) for key, entry in case["outputs"].items()}
            actual = RUNNERS[implementation](case["kernel"], case["params"], inputs)
        except (ValueError, KeyError, OSError) as error:
            result.failed[name] = str(error)
            continue
        wrong = [key for key in expected if key not in actual or not same_bits(expected[key], actual[key])]
        if wrong:
            result.failed[name] = f"different bits in {', '.join(wrong)}"
        else:
            result.passed.append(name)
    return result
