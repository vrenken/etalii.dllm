"""AdamW with deterministic state updates, and the learning-rate schedule.

The update itself is the C++ kernel ``adamw_step`` (``cpp/include/dllm/grad.hpp``): one element at a time, in
double, rounding the moments to float32 before they are used, so the stored state is exactly what the next step
reads. Tensors are updated in the natural order of their names; the global gradient norm sums each tensor's squares
(index order, double) in that same order. Bias corrections ``1 - beta^step`` come from repeated multiplication, not
``pow``, and the cosine schedule uses the portable ``dllm`` cosine, so no libm function is involved.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np

from etalii_dllm.modelfile import tensor_order
from etalii_dllm.numerics import FloatArray, adamw_step, cos, sum_squares


@dataclass(frozen=True)
class AdamWConfig:
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    """Decoupled decay, applied to matrices (projections and embeddings) only, not to norm weights or biases."""
    max_grad_norm: float = 1.0
    """Clip the global gradient norm to this value; 0 disables clipping."""
    warmup_steps: int = 0
    schedule: str = "cosine"
    """``"constant"`` or ``"cosine"`` (decay to ``min_learning_rate`` at the last step), after linear warmup."""
    min_learning_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.schedule not in ("constant", "cosine"):
            raise ValueError(f"unknown schedule {self.schedule!r}")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("betas must be in [0, 1)")
        if self.learning_rate < 0 or self.max_grad_norm < 0 or self.warmup_steps < 0:
            raise ValueError("learning rate, gradient norm and warmup must not be negative")

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)

    def learning_rate_at(self, step: int, total_steps: int) -> float:
        """Learning rate of 1-based ``step`` out of ``total_steps``."""
        if self.warmup_steps and step <= self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        if self.schedule == "constant" or total_steps <= self.warmup_steps:
            return self.learning_rate
        progress = (step - self.warmup_steps) / (total_steps - self.warmup_steps)
        span = self.learning_rate - self.min_learning_rate
        return self.min_learning_rate + 0.5 * span * (1.0 + cos(math.pi * progress))


def _power(base: float, exponent: int) -> float:
    result = 1.0
    for _ in range(exponent):
        result *= base
    return result


def global_norm(gradients: Mapping[str, FloatArray]) -> float:
    """``sqrt`` of the sum of squares of every gradient, tensors in natural name order."""
    total = 0.0
    for name in tensor_order(gradients):
        total += sum_squares(gradients[name])
    return math.sqrt(total)


class AdamW:
    """Optimizer state (first and second moments, float32) for a set of named parameters."""

    def __init__(self, config: AdamWConfig, shapes: Mapping[str, tuple[int, ...]]) -> None:
        self.config = config
        self.m = {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}
        self.v = {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}

    def step(
        self, params: Mapping[str, FloatArray], gradients: Mapping[str, FloatArray], step: int, learning_rate: float
    ) -> float:
        """Applies update number ``step`` (1-based) in place and returns the gradient norm before clipping."""
        config = self.config
        norm = global_norm(gradients)
        scale = 1.0
        if config.max_grad_norm > 0 and norm > config.max_grad_norm:
            scale = config.max_grad_norm / (norm + 1e-6)
        correction1 = 1.0 - _power(config.beta1, step)
        correction2 = 1.0 - _power(config.beta2, step)
        for name in tensor_order(params):
            decay = config.weight_decay if params[name].ndim >= 2 else 0.0
            adamw_step(
                params[name],
                np.ascontiguousarray(gradients[name], dtype=np.float32),
                self.m[name],
                self.v[name],
                learning_rate,
                config.beta1,
                config.beta2,
                config.eps,
                decay,
                correction1,
                correction2,
                scale,
            )
        return norm

    def state(self) -> Iterable[tuple[str, FloatArray]]:
        for name in tensor_order(self.m):
            yield f"adam_m/{name}", self.m[name]
            yield f"adam_v/{name}", self.v[name]
