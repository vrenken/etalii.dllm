"""The CUDA backend: the deterministic kernels on an NVIDIA GPU, with the CPU's bits.

Nothing CUDA is needed to install the package. :func:`initialize` loads the NVIDIA driver and NVRTC (the CUDA
runtime compiler) at run time and compiles ``cpp/cuda/kernels.cu`` for the GPU that is present. Every kernel runs
the documented CPU evaluation order (docs/kernels.md) with multiply-add contraction disabled, so the GPU reproduces
the CPU results bit for bit and a model's ``system_fingerprint`` does not depend on the device.

NVRTC is looked up in this order: ``$DLLM_NVRTC`` (the library's full path), the ``nvidia-cuda-nvrtc`` wheels
(``pip install "etalii-dllm[cuda]"``), PyTorch's copy, ``$CUDA_PATH``/``$CUDA_HOME``, ``/usr/local/cuda``, then the
system search path. ``$DLLM_CUDA_DEVICE`` picks the GPU (default 0).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from etalii_dllm import _kernels

NVRTC_ENVIRONMENT_VARIABLE = "DLLM_NVRTC"
DEVICE_ENVIRONMENT_VARIABLE = "DLLM_CUDA_DEVICE"

NVRTC_MISSING = (
    'NVRTC not found: pip install "etalii-dllm[cuda]", install the CUDA toolkit, or set '
    f"{NVRTC_ENVIRONMENT_VARIABLE} to the nvrtc library"
)

DEVICES = ("cpu", "cuda")
"""Devices a model can run on."""


class CudaUnavailableError(RuntimeError):
    """The CUDA backend cannot run here; the message says why."""


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    compiler: str
    architecture: str


_lock = threading.Lock()


def _library_patterns() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(NVRTC library globs, NVRTC builtins globs) for this platform."""
    if sys.platform == "win32":
        return ("nvrtc64_*.dll",), ("nvrtc-builtins64_*.dll",)
    if sys.platform == "darwin":
        return (), ()
    return ("libnvrtc.so.*", "libnvrtc.so"), ("libnvrtc-builtins.so*",)


def _search_directories() -> list[Path]:
    directories: list[Path] = []
    for entry in sys.path:
        root = Path(entry)
        nvidia = root / "nvidia"
        if nvidia.is_dir():
            directories += sorted(p for p in nvidia.glob("*/bin") if p.is_dir())
            directories += sorted(p for p in nvidia.glob("*/bin/x86_64") if p.is_dir())
            directories += sorted(p for p in nvidia.glob("*/lib") if p.is_dir())
        if (root / "torch" / "lib").is_dir():
            directories.append(root / "torch" / "lib")
    for variable in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(variable):
            base = Path(os.environ[variable])
            directories += [base / "bin", base / "bin" / "x64", base / "lib64", base / "lib"]
    directories += [Path("/usr/local/cuda/lib64"), Path("/usr/lib/x86_64-linux-gnu")]
    return directories


def find_nvrtc() -> Path | None:
    """The NVRTC library the backend would load, or ``None``."""
    configured = os.environ.get(NVRTC_ENVIRONMENT_VARIABLE)
    if configured:
        return Path(configured)
    patterns, _ = _library_patterns()
    for directory in _search_directories():
        # Newest version first; the names sort by version (nvrtc64_120_0 < nvrtc64_130_0).
        found = sorted({p for pattern in patterns for p in directory.glob(pattern)}, reverse=True)
        if found:
            return found[0]
    name = ctypes.util.find_library("nvrtc64_120_0" if sys.platform == "win32" else "nvrtc")
    return Path(name) if name else None


def _preload_builtins(nvrtc: Path) -> None:
    """NVRTC loads its builtins library by name at compile time; loading it first from NVRTC's own directory makes
    that work without changing the search path."""
    _, patterns = _library_patterns()
    for pattern in patterns:
        for candidate in sorted(nvrtc.parent.glob(pattern), reverse=True):
            try:
                ctypes.CDLL(str(candidate))
            except OSError:
                continue
            return


def initialize(device: int | None = None) -> DeviceInfo:
    """Starts the backend on ``device`` (default ``$DLLM_CUDA_DEVICE``, else 0) and returns what it runs on.
    Raises :class:`CudaUnavailableError` with the reason when there is no usable GPU, driver or NVRTC."""
    if device is None:
        device = int(os.environ.get(DEVICE_ENVIRONMENT_VARIABLE, "0") or 0)
    with _lock:
        current = info()
        if current is not None:
            if current.index != device:
                raise CudaUnavailableError(f"the CUDA backend already runs on device {current.index}")
            return current
        if sys.platform == "darwin":
            raise CudaUnavailableError("CUDA is not available on macOS")
        if _kernels.cuda_device_count() == 0:
            raise CudaUnavailableError("no CUDA device found (is the NVIDIA driver installed?)")
        nvrtc = find_nvrtc()
        if nvrtc is None:
            raise CudaUnavailableError(NVRTC_MISSING)
        _preload_builtins(nvrtc)
        try:
            _kernels.cuda_initialize(str(nvrtc), device)
        except RuntimeError as error:
            raise CudaUnavailableError(str(error)) from error
        result = info()
        assert result is not None
        return result


@dataclass(frozen=True)
class CompiledKernels:
    compiler: str
    """NVRTC's version, e.g. ``NVRTC 12.9``."""
    architecture: str
    """``sm_XX`` for a cubin, ``compute_XX`` for PTX the driver finishes compiling."""
    image: bytes


def compile_kernels(arch: int, nvrtc: Path | str | None = None) -> CompiledKernels:
    """Compiles the GPU kernels exactly as :func:`initialize` would for compute capability ``arch`` (e.g. 86 for
    8.6), without a GPU or driver. Checks the ``[cuda]`` install and the embedded kernel sources on any machine.
    Raises :class:`CudaUnavailableError` when NVRTC is missing or cannot compile them."""
    path = Path(nvrtc) if nvrtc is not None else find_nvrtc()
    if path is None:
        raise CudaUnavailableError(NVRTC_MISSING)
    _preload_builtins(path)
    try:
        compiler, architecture, image = _kernels.cuda_compile(str(path), arch)
    except RuntimeError as error:
        raise CudaUnavailableError(str(error)) from error
    return CompiledKernels(compiler, architecture, image)


def info() -> DeviceInfo | None:
    """The device the backend runs on, or ``None`` before :func:`initialize`."""
    details = _kernels.cuda_info()
    return None if details is None else DeviceInfo(*details)


def available() -> bool:
    """Whether :func:`initialize` succeeds (it is tried once and then remembered)."""
    try:
        initialize()
    except CudaUnavailableError:
        return False
    return True


def check_device(device: str) -> str:
    """Validates a device name and starts the backend when it is ``"cuda"``."""
    if device not in DEVICES:
        raise ValueError(f"unknown device {device!r}; supported: {', '.join(DEVICES)}")
    if device == "cuda":
        initialize()
    return device


# -- tensors and operations on the device ------------------------------------------------------------------------


class CudaTensor:
    """A float32 tensor in GPU memory: a :class:`_kernels.CudaArray` plus a shape. Operations on it queue kernels
    without waiting; :meth:`numpy` waits for the result and copies it back."""

    __slots__ = ("array", "shape")

    def __init__(self, array: _kernels.CudaArray, shape: Sequence[int]) -> None:
        self.array = array
        self.shape = tuple(int(d) for d in shape)
        if math.prod(self.shape) * 4 != array.bytes:
            raise ValueError(f"shape {self.shape} does not match {array.bytes} bytes")

    @staticmethod
    def upload(values: npt.ArrayLike) -> CudaTensor:
        array = np.ascontiguousarray(values, dtype=np.float32)
        return CudaTensor(_kernels.cuda_upload(array), array.shape)

    @staticmethod
    def empty(shape: Sequence[int]) -> CudaTensor:
        return CudaTensor(_kernels.cuda_empty(4 * math.prod(shape)), shape)

    def numpy(self) -> np.ndarray:
        return _kernels.cuda_download(self.array, list(self.shape))

    @property
    def rows(self) -> int:
        """Leading dimensions flattened (the last dimension is the feature dimension)."""
        return math.prod(self.shape[:-1])

    def reshape(self, *shape: int) -> CudaTensor:
        return CudaTensor(self.array, shape)

    def __getitem__(self, rows: slice) -> CudaTensor:
        """Rows ``start:stop`` of the first dimension, as a view."""
        start, stop, step = rows.indices(self.shape[0])
        if step != 1:
            raise ValueError("only contiguous row ranges")
        stop = max(stop, start)
        row_bytes = 4 * math.prod(self.shape[1:])
        return CudaTensor(
            self.array.view(start * row_bytes, (stop - start) * row_bytes), (stop - start, *self.shape[1:])
        )

    def copy_to(self, destination: CudaTensor) -> None:
        _kernels.cuda_copy(self.array, destination.array)


def upload_raw(values: np.ndarray) -> _kernels.CudaArray:
    """Copies any C-contiguous array (int64 positions, float64 frequencies) to the GPU."""
    return _kernels.cuda_upload(np.ascontiguousarray(values))


class CudaWeight:
    """A linear weight ``[out, in]`` resident on the GPU. :func:`etalii_dllm.numerics.linear` and :func:`linear`
    with it give the bits of the CPU kernel."""

    def __init__(self, weight: npt.ArrayLike) -> None:
        w = np.ascontiguousarray(weight.numpy() if hasattr(weight, "numpy") else weight, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError("weight must be [out_features, in_features]")
        initialize()
        self.out_features, self.in_features = int(w.shape[0]), int(w.shape[1])
        self.array = _kernels.cuda_upload_linear(w)

    @property
    def shape(self) -> tuple[int, int]:
        return self.out_features, self.in_features


class CudaQuantizedWeight:
    """A Q8_0 :class:`etalii_dllm.numerics.QuantizedWeight` resident on the GPU; linear with it gives the bits of
    the CPU Q8_0 kernel."""

    def __init__(self, weight: Any, kind: str = "q8_0") -> None:
        from etalii_dllm.numerics import QuantizedWeight

        quantized = weight if isinstance(weight, QuantizedWeight) else QuantizedWeight(weight, kind)
        initialize()
        self.kind = quantized.kind
        self.out_features, self.in_features = quantized.out_features, quantized.in_features
        self.array = _kernels.cuda_upload_q8(quantized.values, quantized.scales)

    @property
    def shape(self) -> tuple[int, int]:
        return self.out_features, self.in_features


def linear(x: CudaTensor, weight: CudaWeight | CudaQuantizedWeight, bias: CudaTensor | None = None) -> CudaTensor:
    rows, in_features = x.rows, x.shape[-1]
    if in_features != weight.in_features:
        raise ValueError("weight in_features does not match the last dimension of x")
    kernel = _kernels.cuda_linear_q8 if isinstance(weight, CudaQuantizedWeight) else _kernels.cuda_linear
    out = kernel(x.array, rows, in_features, weight.array, weight.out_features, None if bias is None else bias.array)
    return CudaTensor(out, (*x.shape[:-1], weight.out_features))


def rms_norm(x: CudaTensor, weight: CudaTensor | None, eps: float, *, add_unit_offset: bool = False) -> CudaTensor:
    w = None if weight is None else weight.array
    return CudaTensor(_kernels.cuda_rms_norm(x.array, x.rows, x.shape[-1], w, eps, add_unit_offset), x.shape)


ACTIVATIONS = {"silu": 0, "gelu": 1, "gelu_tanh": 2}


def activation(x: CudaTensor, kind: str) -> CudaTensor:
    return CudaTensor(_kernels.cuda_activation(x.array, ACTIVATIONS[kind]), x.shape)


def swiglu(gate: CudaTensor, up: CudaTensor) -> CudaTensor:
    """``silu(gate) * up`` with the SiLU rounded to float32 first, as the CPU decoder computes it."""
    return CudaTensor(_kernels.cuda_swiglu(gate.array, up.array), gate.shape)


def add(a: CudaTensor, b: CudaTensor) -> CudaTensor:
    return CudaTensor(_kernels.cuda_add(a.array, b.array), a.shape)


def rope(
    x: CudaTensor,
    positions: _kernels.CudaArray,
    inv_freq: _kernels.CudaArray,
    *,
    interleaved: bool = False,
    inverse: bool = False,
) -> CudaTensor:
    """``x[tokens, heads, head_dim]``; ``positions`` (int64) and ``inv_freq`` (float64) from :func:`upload_raw`."""
    tokens, heads, head_dim = x.shape
    out = _kernels.cuda_rope(x.array, positions, inv_freq, tokens, heads, head_dim, interleaved, inverse)
    return CudaTensor(out, x.shape)


def attention(
    q: CudaTensor,
    k: CudaTensor,
    v: CudaTensor,
    *,
    kv_len: int,
    scale: float,
    causal: bool,
    q_offset: int,
    window: int | None = None,
) -> CudaTensor:
    """Attention of ``q[q_len, q_heads, d]`` over the first ``kv_len`` rows of ``k`` and ``v`` (a KV cache may be
    longer); ``window`` as in :func:`etalii_dllm.numerics.attention`."""
    q_len, q_heads, head_dim = q.shape
    kv_heads, value_dim = k.shape[1], v.shape[2]
    out = _kernels.cuda_attention(
        q.array,
        k.array,
        v.array,
        q_len,
        kv_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        scale,
        causal,
        q_offset,
        window or 0,
    )
    return CudaTensor(out, (q_len, q_heads, value_dim))
