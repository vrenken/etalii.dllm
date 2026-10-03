"""Unigram tokenizers (XLM-RoBERTa, multilingual MiniLM, T5-style vocabularies), as the ``tokenizers`` library runs
them (issue #347).

Two pieces plug into :class:`etalii_dllm.bpe.BpeTokenizer`:

- :class:`Precompiled`, SentencePiece's normaliser: a ``precompiled_charsmap`` holding a double-array trie over
  UTF-8 bytes whose values point into a table of replacement strings. ``tokenizers`` looks up every extended grapheme
  cluster shorter than six bytes as a whole and takes the shortest match in the trie (even one that covers only the
  start of the cluster, which then replaces all of it); otherwise each character is looked up on its own. The
  clusters come from :func:`etalii_dllm.unicode.graphemes`, the tables of ``tokenizers``' own segmentation crate.
- :class:`Unigram`, the model: the segmentation of each word whose pieces' scores (log probabilities) have the
  highest sum, found by ``tokenizers``' Viterbi pass over the characters. A character no piece covers on its own
  is the unknown piece, scored ten below the lowest score; neighbouring unknown pieces are fused, and with
  ``byte_fallback`` an unknown piece becomes its ``<0xAB>`` byte pieces. Ties keep the path found first: the
  earliest start for each end position, as the reference compares with a strict ``>``. Scores are summed in
  float64, left to right, exactly as the reference sums them.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

from etalii_dllm import unicode

UNKNOWN_PENALTY = 10.0
"""How far below the lowest piece score an unknown character scores (``K_UNK_PENALTY``)."""


class Precompiled:
    """SentencePiece's ``precompiled_charsmap`` normaliser (the ``spm_precompiled`` crate ``tokenizers`` uses)."""

    def __init__(self, charsmap: bytes) -> None:
        if len(charsmap) < 4:
            raise ValueError("the precompiled charsmap is too short")
        (trie_size,) = struct.unpack_from("<I", charsmap)
        units = trie_size // 4
        if len(charsmap) < 4 + 4 * units or not units:
            raise ValueError("the precompiled charsmap's trie is truncated")
        self._array = struct.unpack_from(f"<{units}I", charsmap, 4)
        self._normalized = charsmap[4 + 4 * units :]
        try:
            self._normalized.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("the precompiled charsmap's replacements are not UTF-8") from error
        self._cache: dict[str, str | None] = {}

    def _first_match(self, key: bytes) -> int | None:
        """The value of the shortest key prefix in the trie (Darts' ``common_prefix_search``, first result)."""
        array = self._array
        try:
            unit = array[0]
            position = (unit >> 10) << ((unit & (1 << 9)) >> 6)
            for byte in key:
                if byte == 0:
                    return None
                position ^= byte
                unit = array[position]
                if unit & ((1 << 31) | 0xFF) != byte:
                    return None
                position ^= (unit >> 10) << ((unit & (1 << 9)) >> 6)
                if (unit >> 8) & 1:
                    return array[position] & ((1 << 31) - 1)
        except IndexError:
            return None
        return None

    def transform(self, chunk: str) -> str | None:
        """The replacement of ``chunk``'s shortest prefix in the trie, or None when no prefix is in it."""
        if chunk in self._cache:
            return self._cache[chunk]
        index = self._first_match(chunk.encode("utf-8"))
        result: str | None = None
        if index is not None and index <= len(self._normalized):
            end = self._normalized.find(b"\0", index)
            result = self._normalized[index : len(self._normalized) if end < 0 else end].decode("utf-8")
        if len(self._cache) < 100_000:
            self._cache[chunk] = result
        return result

    def __call__(self, text: str) -> str:
        out: list[str] = []
        for cluster in unicode.graphemes(text):
            if len(cluster.encode("utf-8")) < 6:
                replacement = self.transform(cluster)
                if replacement is not None:
                    out.append(replacement)
                    continue
            for character in cluster:
                replacement = self.transform(character)
                out.append(character if replacement is None else replacement)
        return "".join(out)


class Unigram:
    """A Unigram model: pieces with scores, the unknown piece's id and whether unknown pieces fall back to bytes."""

    def __init__(self, vocab: Sequence[tuple[str, float]], unknown: int | None, byte_fallback: bool) -> None:
        if unknown is not None and not 0 <= unknown < len(vocab):
            raise ValueError("the Unigram unk_id is outside the vocabulary")
        self.pieces = [str(piece) for piece, _ in vocab]
        self.scores = [float(score) for _, score in vocab]
        self.ids = {piece: index for index, piece in enumerate(self.pieces)}  # a repeated piece: the last id
        self.unknown = unknown
        self.byte_fallback = byte_fallback
        self._longest = max((len(piece) for piece in self.pieces), default=0)
        self._unknown_score = min(self.scores, default=0.0) - UNKNOWN_PENALTY

    def segment(self, word: str) -> list[str]:
        """The best segmentation of ``word`` into pieces (unknown runs fused), as ``Unigram::encode_optimized``."""
        size = len(word)
        if not size:
            return []
        best = [0.0] * (size + 1)
        starts: list[int | None] = [None] * (size + 1)
        chosen = [0] * (size + 1)
        ids, scores = self.ids, self.scores
        for start in range(size):
            so_far = best[start]
            single = False
            for end in range(start + 1, min(size, start + self._longest) + 1):
                index = ids.get(word[start:end])
                if index is None:
                    continue
                candidate = scores[index] + so_far
                if starts[end] is None or candidate > best[end]:
                    best[end], starts[end], chosen[end] = candidate, start, index
                single = single or end == start + 1
            if not single:
                candidate = self._unknown_score + so_far
                if starts[start + 1] is None or candidate > best[start + 1]:
                    if self.unknown is None:
                        raise ValueError(f"character {word[start]!r} is not in the vocabulary and there is no unk_id")
                    best[start + 1], starts[start + 1], chosen[start + 1] = candidate, start, self.unknown
        pieces: list[str] = []
        unknown_run: list[str] = []
        end = size
        while end > 0:
            start = starts[end]
            assert start is not None
            if chosen[end] == self.unknown:
                unknown_run.append(word[start:end])
            else:
                if unknown_run:
                    pieces.append("".join(reversed(unknown_run)))
                    unknown_run = []
                pieces.append(word[start:end])
            end = start
        if unknown_run:
            pieces.append("".join(reversed(unknown_run)))
        pieces.reverse()
        return pieces

    def tokenize(self, word: str) -> list[int]:
        """Token ids of ``word``: each piece's id, an unknown run as its byte pieces (``byte_fallback``, when all of
        them exist) or the unknown id."""
        out: list[int] = []
        for piece in self.segment(word):
            index = self.ids.get(piece)
            if index is not None:
                out.append(index)
                continue
            if self.byte_fallback:
                fallback = [self.ids.get(f"<0x{byte:02X}>") for byte in piece.encode("utf-8")]
                if all(token is not None for token in fallback):
                    out.extend(token for token in fallback if token is not None)
                    continue
            if self.unknown is None:
                raise ValueError(f"piece {piece!r} is not in the vocabulary and there is no unk_id")
            out.append(self.unknown)
        return out
