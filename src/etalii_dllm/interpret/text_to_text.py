"""Activation tracing of T5 text-to-text models (#402): every intermediate of the encoder over the source and of the
decoder over the answer, captured without changing a bit.

The source goes through the encoder once and the decoder reads its start token, then the answer, one position at a
time, exactly as :meth:`etalii_dllm.seq2seq.TextToText.answer_logits` does, with a recorder that copies what it is
shown. The recorder never replaces anything, so ``logits[i]`` is the bits of ``TextToText.forward`` for the source
and ``answer[:i]``; the last row is the next token's logits after the whole answer. The attention probabilities come
from :func:`etalii_dllm.numerics.biased_attention_weights`, the probabilities ``biased_attention`` applies, bit for
bit.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from etalii_dllm.seq2seq import DECODER_START, TextToText


@dataclass(frozen=True)
class TextToTextTrace:
    """The activations of a T5 model over ``source`` (``s`` positions, ``E`` encoder layers) and the decoder inputs
    ``tokens`` (the start token, then the answer: ``n`` positions, ``L`` decoder layers)."""

    source: tuple[int, ...]
    tokens: tuple[int, ...]
    """The decoder's inputs: its start token, then the answer."""
    encoder_residual: np.ndarray
    """``[E + 1, s, hidden]``: the encoder's residual stream entering each layer and, last, leaving the final one."""
    encoder_attention: np.ndarray | None
    """``[E, heads, s, s]``: the encoder's bidirectional attention probabilities, or ``None`` when not traced."""
    encoder_hidden: np.ndarray
    """``[s, hidden]``: the encoder's final-norm states, what cross-attention reads."""
    residual: np.ndarray
    """``[L + 1, n, hidden]``: the decoder's residual stream entering each layer (row 0: the embeddings) and, last,
    leaving the final layer (after a steering vector, when the model has one)."""
    middle: np.ndarray
    """``[L, n, hidden]``: the decoder's residual stream after each self-attention block."""
    cross_middle: np.ndarray
    """``[L, n, hidden]``: the decoder's residual stream after each cross-attention block."""
    attention_output: np.ndarray
    """``[L, n, hidden]``: what each self-attention block adds."""
    cross_attention_output: np.ndarray
    """``[L, n, hidden]``: what each cross-attention block adds."""
    mlp_activation: np.ndarray
    """``[L, n, intermediate]``: each decoder MLP's hidden activation."""
    mlp_output: np.ndarray
    """``[L, n, hidden]``: what each decoder MLP adds."""
    attention: np.ndarray | None
    """``[L, heads, n, n]``: decoder self-attention probabilities (query, key; 0 above the diagonal)."""
    cross_attention: np.ndarray | None
    """``[L, heads, n, s]``: cross-attention probabilities over the source positions."""
    hidden: np.ndarray
    """``[n, hidden]``: the decoder's final-norm states."""
    logits: np.ndarray | None
    """``[n, vocabulary]``: the next answer token's logits after every decoder position, or ``None``."""

    def arrays(self) -> Iterator[tuple[str, np.ndarray]]:
        """The captured arrays by name, in a fixed order."""
        names = (
            "encoder_residual",
            "encoder_attention",
            "encoder_hidden",
            "residual",
            "middle",
            "cross_middle",
            "attention_output",
            "cross_attention_output",
            "mlp_activation",
            "mlp_output",
            "attention",
            "cross_attention",
            "hidden",
            "logits",
        )
        for name in names:
            values = getattr(self, name)
            if values is not None:
                yield name, values

    def fingerprint(self) -> str:
        """SHA-256 over the source, the decoder inputs and the exact bits of every array."""
        digest = hashlib.sha256(np.asarray(self.source, dtype="<i8").tobytes())
        digest.update(b"tokens")
        digest.update(np.asarray(self.tokens, dtype="<i8").tobytes())
        for name, values in self.arrays():
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(values, dtype="<f4").tobytes())
        return digest.hexdigest()


class _Recorder:
    """Collects what one encoder pass or the decoder steps show it, per part and layer, in arrival order."""

    def __init__(self, wants_attention: bool) -> None:
        self.wants_attention = wants_attention
        self.parts: dict[str, dict[int, list[np.ndarray]]] = {}

    def record(self, part: str, layer: int, values: np.ndarray) -> None:
        self.parts.setdefault(part, {}).setdefault(layer, []).append(np.array(values, dtype=np.float32))

    def stacked(self, part: str) -> np.ndarray:
        """``[layers, rows, ...]``: each layer's arrays joined along their first axis (the positions)."""
        layers = self.parts[part]
        return np.stack([np.concatenate(layers[layer]) for layer in sorted(layers)])


def _square(rows: list[np.ndarray]) -> np.ndarray:
    """Decoder self-attention rows ``[1, heads, t + 1]`` (one per position) as ``[heads, n, n]``, zero-padded."""
    n = len(rows)
    out = np.zeros((rows[0].shape[1], n, n), dtype=np.float32)
    for t, row in enumerate(rows):
        out[:, t, : t + 1] = row[0]
    return out


def trace_text_to_text(
    model: TextToText, tokens: Sequence[int], *, attention: bool = True, logits: bool = True
) -> TextToTextTrace:
    """Runs ``tokens`` (the source, ending with ``</s>``, then the answer so far) through ``model`` and returns every
    intermediate activation of the encoder and the decoder."""
    source, answer = model.split(tokens)
    if any(not 0 <= token < model.vocabulary_size for token in answer):
        raise ValueError("token id out of range")
    encoder = _Recorder(attention)
    decoder = _Recorder(attention)
    cache = model.new_cache()
    model._encode(source, cache, encoder)
    inputs = [DECODER_START, *answer]
    rows = [model._steps([token], [cache], decoder)[0] for token in inputs]
    final = decoder.stacked("residual")[-1]
    hidden = model.final_norm(final)
    self_attention = cross_attention = encoder_attention = None
    if attention:
        maps = decoder.parts["attention"]
        self_attention = np.stack([_square(maps[layer]) for layer in sorted(maps)])
        cross_attention = np.ascontiguousarray(decoder.stacked("cross_attention").transpose(0, 2, 1, 3))
        encoder_attention = np.ascontiguousarray(encoder.stacked("attention").transpose(0, 2, 1, 3))
    return TextToTextTrace(
        source=tuple(int(t) for t in source),
        tokens=tuple(int(t) for t in inputs),
        encoder_residual=encoder.stacked("residual"),
        encoder_attention=encoder_attention,
        encoder_hidden=model.encoder_states(source),
        residual=decoder.stacked("residual"),
        middle=decoder.stacked("middle"),
        cross_middle=decoder.stacked("cross_middle"),
        attention_output=decoder.stacked("attention_output"),
        cross_attention_output=decoder.stacked("cross_attention_output"),
        mlp_activation=decoder.stacked("mlp_activation"),
        mlp_output=decoder.stacked("mlp_output"),
        attention=self_attention,
        cross_attention=cross_attention,
        hidden=hidden,
        logits=np.stack(rows) if logits else None,
    )
