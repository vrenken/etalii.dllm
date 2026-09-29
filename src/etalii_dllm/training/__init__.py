"""Deterministic fine-tuning (roadmap Phase 3): gradients, AdamW, fixed data order and reproducible checkpoints.

See ``docs/training.md``.
"""

from __future__ import annotations

from etalii_dllm.lora import LoraConfig
from etalii_dllm.training.backprop import DecoderGradients
from etalii_dllm.training.data import TrainingData, TrainingDataError, read_documents
from etalii_dllm.training.optimizer import AdamW, AdamWConfig
from etalii_dllm.training.trainer import CheckpointError, FineTuner, RunConfig, StepResult

__all__ = [
    "AdamW",
    "AdamWConfig",
    "CheckpointError",
    "DecoderGradients",
    "FineTuner",
    "LoraConfig",
    "RunConfig",
    "StepResult",
    "TrainingData",
    "TrainingDataError",
    "read_documents",
]
