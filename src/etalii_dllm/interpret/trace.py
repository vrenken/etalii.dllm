"""Activation tracing: every intermediate of one forward pass, captured without changing a bit of it.

:func:`trace` runs the decoder with a :class:`~etalii_dllm.transformer.LayerHook` that copies what it is shown. The
hook never returns a replacement, so the traced pass computes exactly the bits of an untraced one: the last row of
``Trace.logits`` equals ``Transformer.forward(tokens)`` and ``Trace.hidden`` equals ``hidden_states(tokens)``. Each
array is produced by the deterministic kernels, so a trace is the same bits on every run, thread count, SIMD path
and machine (``tests/test_interpret.py``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from etalii_dllm.transformer import LayerHook, Transformer


@dataclass(frozen=True)
class Trace:
    """The activations of one forward pass over ``tokens`` (``n`` positions, ``L`` layers)."""

    tokens: tuple[int, ...]
    residual: np.ndarray
    """``[L + 1, n, hidden]``: the residual stream entering each layer (row 0: the embeddings) and, last, leaving the
    final layer."""
    middle: np.ndarray
    """``[L, n, hidden]``: the residual stream after each layer's attention block."""
    attention_output: np.ndarray
    """``[L, n, hidden]``: what each attention block adds to the residual stream."""
    mlp_activation: np.ndarray
    """``[L, n, intermediate]``: each MLP's hidden activation ``act(gate) * up``."""
    mlp_output: np.ndarray
    """``[L, n, hidden]``: what each MLP block adds to the residual stream."""
    attention: np.ndarray | None
    """``[L, heads, n, n]``: attention probabilities (query position, key position), or ``None`` when not traced."""
    hidden: np.ndarray
    """``[n, hidden]``: the final-norm hidden states."""
    logits: np.ndarray | None
    """``[n, vocabulary]``: the next-token logits after every position, or ``None`` when not traced."""

    def arrays(self) -> Iterator[tuple[str, np.ndarray]]:
        """The captured arrays by name, in a fixed order."""
        for name in ("residual", "middle", "attention_output", "mlp_activation", "mlp_output", "attention"):
            values = getattr(self, name)
            if values is not None:
                yield name, values
        yield "hidden", self.hidden
        if self.logits is not None:
            yield "logits", self.logits

    def fingerprint(self) -> str:
        """SHA-256 over the tokens and the exact bits of every array; equal fingerprints mean identical traces."""
        digest = hashlib.sha256(np.asarray(self.tokens, dtype="<i8").tobytes())
        for name, values in self.arrays():
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(values, dtype="<f4").tobytes())
        return digest.hexdigest()


class _Recorder(LayerHook):
    def __init__(self, layers: int, with_attention: bool) -> None:
        self.layers = layers
        self.wants_attention = with_attention
        self.streams: list[np.ndarray] = []
        self.middle: list[np.ndarray] = []
        self.attention_maps: list[np.ndarray] = []
        self.attention_outputs: list[np.ndarray] = []
        self.mlp_activations: list[np.ndarray] = []
        self.mlp_outputs: list[np.ndarray] = []

    def residual(self, layer: int, point: str, x: np.ndarray) -> None:  # type: ignore[override]
        if point == "middle":
            self.middle.append(x)
        elif point == "input" or layer == self.layers - 1:
            self.streams.append(x)

    def attention(self, layer: int, probabilities: np.ndarray) -> None:
        self.attention_maps.append(np.ascontiguousarray(probabilities.transpose(1, 0, 2)))

    def attention_output(self, layer: int, out: np.ndarray) -> None:
        self.attention_outputs.append(out)

    def mlp_activation(self, layer: int, activation: np.ndarray) -> None:
        self.mlp_activations.append(activation)

    def mlp_output(self, layer: int, out: np.ndarray) -> None:
        self.mlp_outputs.append(out)


def trace(model: Transformer, tokens: Sequence[int], *, attention: bool = True, logits: bool = True) -> Trace:
    """Runs ``tokens`` through ``model`` (on the CPU) and returns every intermediate activation. ``attention=False``
    skips the attention probabilities, which take a second pass over the keys; ``logits=False`` skips the LM head
    over every position (the largest matrix of a small model)."""
    recorder = _Recorder(model.config.layers, attention)
    hidden = model.run_hooked(tokens, recorder)
    return Trace(
        tokens=tuple(int(t) for t in tokens),
        residual=np.stack(recorder.streams),
        middle=np.stack(recorder.middle),
        attention_output=np.stack(recorder.attention_outputs),
        mlp_activation=np.stack(recorder.mlp_activations),
        mlp_output=np.stack(recorder.mlp_outputs),
        attention=np.stack(recorder.attention_maps) if attention else None,
        hidden=hidden,
        logits=model.logits_from_hidden(hidden) if logits else None,
    )
