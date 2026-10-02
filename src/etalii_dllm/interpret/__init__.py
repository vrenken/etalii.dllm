"""Interpretability and editing tools that run on the deterministic kernels (``docs/interpretability.md``).

Because every forward pass is bit-exact, an observation made with these tools can be reproduced exactly by anyone
with the same model file, on any machine.
"""

from __future__ import annotations

from etalii_dllm.interpret.embeddings import Neighbour, Neighbourhood, embedding_matrix, neighbours, token_text
from etalii_dllm.interpret.experts import Routing, routing
from etalii_dllm.interpret.lens import Lens, Prediction, logit_lens, top_k
from etalii_dllm.interpret.trace import Trace, trace

__all__ = [
    "Lens",
    "Neighbour",
    "Neighbourhood",
    "Prediction",
    "Routing",
    "Trace",
    "embedding_matrix",
    "logit_lens",
    "neighbours",
    "routing",
    "token_text",
    "top_k",
    "trace",
]
