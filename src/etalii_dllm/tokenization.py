"""Tokenizers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class Tokenizer(Protocol):
    @property
    def vocabulary_size(self) -> int: ...

    @property
    def end_of_sequence(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: Iterable[int]) -> str: ...

    def decode_bytes(self, tokens: Iterable[int]) -> bytes:
        """The exact bytes of ``tokens``, special tokens left out. Concatenating the bytes of single tokens gives
        the bytes of the sequence (true for byte-level tokenizers), which streaming and constrained decoding rely
        on."""
        ...


class ByteTokenizer:
    """One token per UTF-8 byte plus an end-of-sequence token. A placeholder until the BPE tokenizer of imported
    models lands, but already lossless and deterministic."""

    vocabulary_size = 257
    end_of_sequence = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: Iterable[int]) -> str:
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")

    def decode_bytes(self, tokens: Iterable[int]) -> bytes:
        return bytes(t for t in tokens if 0 <= t < 256)
