"""BPE tokenizer driven by a Hugging Face ``tokenizer.json``.

Two families are supported: byte-level BPE (GPT-2, SmolLM2, Qwen2, Llama 3) and the SentencePiece-style BPE that
``transformers`` converts Llama 2, TinyLlama, Mistral and Phi-3 tokenizers to (spaces become ``▁``, unknown
characters fall back to ``<0xAB>`` byte tokens).

The pipeline mirrors the ``tokenizers`` library: added tokens are split out first, the rest is normalised,
pre-tokenised (``Split``, ``Digits``, ``ByteLevel``, ``Metaspace``), mapped to byte-level characters (byte-level
BPE only) and merged with the same priority rule (lowest merge rank first, leftmost on ties). Tests compare the
output with the reference library.

Determinism: no sets or dict iteration decide an outcome, and nothing depends on the locale or ``PYTHONHASHSEED``.
Normalisation, lower-casing and the Unicode classes (``\\p{L}``, ...) in the regular expressions come from
:mod:`etalii_dllm.unicode`, pinned to one Unicode version, so the installed Python and ``regex`` versions do not
change how text is tokenized.
Unsupported components fail at load time instead of tokenizing differently from the reference.
"""

from __future__ import annotations

import copy as _copy
import heapq
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import regex

from etalii_dllm import unicode

GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


class TokenizerError(ValueError):
    """The tokenizer description uses a component this implementation does not support, or is invalid."""


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable character table."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    characters = list(printable)
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            characters.append(256 + extra)
            extra += 1
    return {byte: chr(character) for byte, character in zip(printable, characters, strict=True)}


# --- normalisers -------------------------------------------------------------------------------------------------


def _pattern(spec: Mapping[str, Any]) -> regex.Pattern[str]:
    """A ``tokenizers`` pattern: ``{"String": ...}`` (literal) or ``{"Regex": ...}``."""
    if "Regex" in spec:
        return unicode.compile(spec["Regex"])
    return regex.compile(regex.escape(spec["String"]))


def _normalizer(spec: Mapping[str, Any] | None) -> Callable[[str], str]:
    if spec is None:
        return lambda text: text
    kind = spec.get("type")
    if kind in ("NFC", "NFD", "NFKC", "NFKD"):
        form = kind
        return lambda text: unicode.normalize(form, text)
    if kind == "Lowercase":
        return unicode.lower
    if kind == "Prepend":
        prefix = spec["prepend"]
        return lambda text: prefix + text if text else text
    if kind == "Replace":
        pattern, content = _pattern(spec["pattern"]), spec["content"]
        return lambda text: pattern.sub(lambda _: content, text)
    if kind == "Strip":
        left, right = bool(spec.get("strip_left", True)), bool(spec.get("strip_right", True))

        def strip(text: str) -> str:
            if left:
                text = text.lstrip()
            return text.rstrip() if right else text

        return strip
    if kind == "Sequence":
        steps = [_normalizer(s) for s in spec["normalizers"]]

        def run(text: str) -> str:
            for step in steps:
                text = step(text)
            return text

        return run
    raise TokenizerError(f"normalizer {kind!r} is not supported")


# --- pre-tokenisers ----------------------------------------------------------------------------------------------

PreTokenizer = Callable[[list[str], bool], list[str]]
"""Maps pieces to pieces. The flag says whether the first piece starts at the beginning of the input text (not
after an added token), which ``Metaspace`` with ``prepend_scheme: first`` needs."""


def _split_by(pattern: regex.Pattern[str], text: str, behavior: str, invert: bool) -> list[str]:
    """``tokenizers``' ``Split`` with a regex: matches are the delimiters (or, inverted, the content)."""
    pieces: list[tuple[str, bool]] = []  # (text, is_match)
    position = 0
    for match in pattern.finditer(text):
        if match.start() == match.end():
            continue
        if match.start() > position:
            pieces.append((text[position : match.start()], False))
        pieces.append((match.group(), True))
        position = match.end()
    if position < len(text):
        pieces.append((text[position:], False))
    if invert:
        pieces = [(piece, not is_match) for piece, is_match in pieces]

    if behavior == "Isolated":
        return [piece for piece, _ in pieces]
    if behavior == "Removed":
        return [piece for piece, is_match in pieces if not is_match]
    if behavior == "MergedWithPrevious":
        merged: list[str] = []
        for piece, is_match in pieces:
            if is_match and merged:
                merged[-1] += piece
            else:
                merged.append(piece)
        return merged
    if behavior == "MergedWithNext":
        merged = []
        pending = ""
        for piece, is_match in pieces:
            if is_match:
                if pending:
                    merged.append(pending)
                pending = piece
            else:
                merged.append(pending + piece)
                pending = ""
        if pending:
            merged.append(pending)
        return merged
    if behavior == "Contiguous":
        merged = []
        previous: bool | None = None
        for piece, is_match in pieces:
            if is_match == previous:  # tokenizers merges runs of either kind (with invert, runs of delimiters)
                merged[-1] += piece
            else:
                merged.append(piece)
            previous = is_match
        return merged
    raise TokenizerError(f"split behaviour {behavior!r} is not supported")


def _pre_tokenizer(spec: Mapping[str, Any] | None) -> tuple[PreTokenizer, bool]:
    """Returns the pre-tokeniser and whether it includes the byte-level mapping."""
    if spec is None:
        return (lambda pieces, _: pieces), False
    kind = spec.get("type")
    if kind == "Sequence":
        steps = [_pre_tokenizer(s) for s in spec["pretokenizers"]]

        def run(pieces: list[str], at_start: bool) -> list[str]:
            for step, _ in steps:
                pieces = step(pieces, at_start)
            return pieces

        return run, any(byte_level for _, byte_level in steps)
    if kind == "Split":
        pattern = _pattern(spec["pattern"])
        behavior, invert = spec.get("behavior", "Isolated"), bool(spec.get("invert", False))
        return (lambda pieces, _: [p for piece in pieces for p in _split_by(pattern, piece, behavior, invert)]), False
    if kind == "Digits":
        digits = unicode.compile(r"\p{Nd}" if spec.get("individual_digits") else r"\p{Nd}+")
        return (lambda pieces, _: [p for piece in pieces for p in _split_by(digits, piece, "Isolated", False)]), False
    if kind == "Metaspace":
        replacement = spec.get("replacement", "▁")
        scheme = _prepend_scheme(spec)
        delimiter = regex.compile(regex.escape(replacement)) if spec.get("split", True) else None

        def metaspace(pieces: list[str], at_start: bool) -> list[str]:
            out = []
            for i, piece in enumerate(pieces):
                piece = piece.replace(" ", replacement)
                prepend = scheme == "always" or (scheme == "first" and at_start and i == 0)
                if prepend and piece and not piece.startswith(replacement):
                    piece = replacement + piece
                out.extend(_split_by(delimiter, piece, "MergedWithNext", False) if delimiter else [piece])
            return out

        return metaspace, False
    if kind == "ByteLevel":
        add_prefix_space = bool(spec.get("add_prefix_space", False))
        gpt2 = unicode.compile(GPT2_PATTERN) if spec.get("use_regex", True) else None

        def byte_level(pieces: list[str], _: bool) -> list[str]:
            out = []
            for piece in pieces:
                if add_prefix_space and not piece.startswith(" "):
                    piece = " " + piece
                out.extend(_split_by(gpt2, piece, "Isolated", False) if gpt2 else [piece])
            return out

        return byte_level, True
    raise TokenizerError(f"pre-tokenizer {kind!r} is not supported")


def _prepend_scheme(spec: Mapping[str, Any]) -> str:
    """``Metaspace``'s prepend scheme; files written before ``prepend_scheme`` existed use ``add_prefix_space``."""
    if "prepend_scheme" in spec:
        scheme = str(spec["prepend_scheme"])
    else:
        scheme = "always" if spec.get("add_prefix_space", True) else "never"
    if scheme not in ("always", "first", "never"):
        raise TokenizerError(f"Metaspace prepend_scheme {scheme!r} is not supported")
    return scheme


# --- decoders ----------------------------------------------------------------------------------------------------

_BYTE_TOKEN = regex.compile(r"<0x([0-9A-F]{2})>")


@dataclass(frozen=True)
class SentencePieceDecoding:
    """What the SentencePiece-style decoder chain does: ``replacement`` becomes a space, ``<0xAB>`` tokens become
    their byte (``ByteFallback``), and one leading space of the whole text is dropped (``Strip`` after ``Fuse``) or,
    for ``Metaspace`` with a prepend scheme, every replacement character of the first token."""

    replacement: str
    byte_fallback: bool
    strip_leading_space: bool
    strip_first_token: bool = False


def _sentencepiece_decoding(spec: Mapping[str, Any]) -> SentencePieceDecoding:
    steps = spec["decoders"] if spec.get("type") == "Sequence" else [spec]
    replacement: str | None = None
    byte_fallback = strip = fused = first_token = False
    for step in steps:
        kind = step.get("type")
        if kind == "Replace" and "String" in step["pattern"] and step["content"] == " " and replacement is None:
            replacement = step["pattern"]["String"]
        elif kind == "Metaspace" and replacement is None:
            replacement = step.get("replacement", "▁")
            first_token = _prepend_scheme(step) != "never"
            strip = strip or first_token
        elif kind == "ByteFallback":
            byte_fallback = True
        elif kind == "Fuse":
            fused = True
        elif kind == "Strip" and step.get("content") == " " and int(step.get("stop", 0)) == 0:
            # Before Fuse, Strip would act on every token; after it, on the start of the text.
            if int(step.get("start", 0)) > 1 or (int(step.get("start", 0)) and not fused):
                raise TokenizerError(f"Strip decoder {step!r} is not supported")
            strip = strip or int(step.get("start", 0)) == 1
        else:
            raise TokenizerError(f"decoder {kind!r} ({step!r}) is not supported")
    if replacement is None:
        raise TokenizerError("a SentencePiece-style decoder must map the replacement character back to a space")
    return SentencePieceDecoding(replacement, byte_fallback, strip, first_token)


# --- BPE model ---------------------------------------------------------------------------------------------------


def _merge(symbols: list[int], merges: Mapping[tuple[int, int], tuple[int, int]]) -> list[int]:
    """Applies merges exactly like ``tokenizers``' ``Word::merge_all``: a heap ordered by (rank, position); stale
    entries are skipped when popped."""
    count = len(symbols)
    ids = list(symbols)
    previous = list(range(-1, count - 1))
    following = [*range(1, count), -1]
    alive = [True] * count
    heap: list[tuple[int, int, int]] = []
    for i in range(count - 1):
        merge = merges.get((ids[i], ids[i + 1]))
        if merge is not None:
            heap.append((merge[0], i, merge[1]))
    heapq.heapify(heap)
    while heap:
        _, position, new_id = heapq.heappop(heap)
        if not alive[position] or following[position] == -1:
            continue
        right = following[position]
        current = merges.get((ids[position], ids[right]))
        if current is None or current[1] != new_id:
            continue
        ids[position] = new_id
        alive[right] = False
        following[position] = following[right]
        if following[right] != -1:
            previous[following[right]] = position
        if previous[position] != -1:
            left = previous[position]
            merge = merges.get((ids[left], new_id))
            if merge is not None:
                heapq.heappush(heap, (merge[0], left, merge[1]))
        if following[position] != -1:
            merge = merges.get((new_id, ids[following[position]]))
            if merge is not None:
                heapq.heappush(heap, (merge[0], position, merge[1]))
    return [token for token, keep in zip(ids, alive, strict=True) if keep]


@dataclass(frozen=True)
class AddedToken:
    id: int
    content: str
    special: bool
    lstrip: bool = False
    rstrip: bool = False


class BpeTokenizer:
    """Encodes and decodes like the Hugging Face tokenizer described by ``spec`` (a parsed ``tokenizer.json``)."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        *,
        end_of_sequence: str | int | None = None,
        begin_of_sequence: str | int | None = None,
    ) -> None:
        model = spec.get("model") or {}
        if model.get("type") != "BPE":
            raise TokenizerError(f"model {model.get('type')!r} is not supported (only BPE)")
        for option in ("continuing_subword_prefix", "end_of_word_suffix"):
            if model.get(option):
                raise TokenizerError(f"BPE option {option} is not supported")
        if model.get("dropout"):
            raise TokenizerError("BPE dropout is not deterministic")

        self._vocab: dict[str, int] = {str(k): int(v) for k, v in model["vocab"].items()}
        self._merges: dict[tuple[int, int], tuple[int, int]] = {}
        ranks = model.get("merge_ranks")  # SentencePiece models from GGUF: equal scores share a rank
        for index, merge in enumerate(model.get("merges", [])):
            rank = index if ranks is None else int(ranks[index])
            left, right = merge.split(" ", 1) if isinstance(merge, str) else merge
            try:
                pair = (self._vocab[left], self._vocab[right])
                merged = self._vocab[left + right]
            except KeyError as error:
                raise TokenizerError(f"merge {left!r} {right!r} refers to an unknown token") from error
            self._merges.setdefault(pair, (rank, merged))
        self._ignore_merges = bool(model.get("ignore_merges", False))
        self._byte_fallback = bool(model.get("byte_fallback", False))
        self._fuse_unk = bool(model.get("fuse_unk", False))
        unk = model.get("unk_token")
        self._unk = self._vocab.get(unk) if unk is not None else None

        self._normalize = _normalizer(spec.get("normalizer"))
        self._pre_tokenize, self._byte_level = _pre_tokenizer(spec.get("pre_tokenizer"))
        decoder = spec.get("decoder") or {}
        self._sentencepiece: SentencePieceDecoding | None = None
        if self._byte_level:
            if decoder.get("type") != "ByteLevel":
                raise TokenizerError(f"decoder {decoder.get('type')!r} is not supported with a ByteLevel pre-tokenizer")
        elif decoder.get("type") == "ByteLevel":
            raise TokenizerError("a ByteLevel decoder needs a ByteLevel pre-tokenizer (byte-level BPE)")
        else:
            self._sentencepiece = _sentencepiece_decoding(decoder)
        self.strips_leading_space = self._sentencepiece is not None and self._sentencepiece.strip_leading_space
        """Whether decoding drops one leading space of the text (SentencePiece-style tokenizers). ``decode`` does it;
        callers that concatenate ``decode_bytes`` of single tokens, as streaming does, drop it themselves."""
        self._post = self._post_processor(spec.get("post_processor"))

        self._added = [
            AddedToken(int(t["id"]), t["content"], bool(t.get("special")), bool(t.get("lstrip")), bool(t.get("rstrip")))
            for t in spec.get("added_tokens", [])
        ]
        for token in self._added:
            if token.content not in self._vocab:
                self._vocab[token.content] = token.id
        # Longest first, then by content: regex alternation then finds the leftmost-longest added token.
        ordered = sorted(self._added, key=lambda t: (-len(t.content), t.content, t.id))
        self._added_by_content = {t.content: t for t in ordered}
        self._added_pattern = regex.compile("|".join(regex.escape(t.content) for t in ordered)) if ordered else None
        self._special_ids = frozenset(t.id for t in self._added if t.special)
        self._shown: frozenset[int] = frozenset()
        """Special tokens that decoding keeps (tool call markers, :meth:`showing`)."""

        self._id_to_token: dict[int, str] = {}
        for token, index in sorted(self._vocab.items(), key=lambda item: (item[1], item[0])):
            self._id_to_token.setdefault(index, token)
        self._byte_tokens = {
            index: int(match.group(1), 16)
            for token, index in self._vocab.items()
            if (match := _BYTE_TOKEN.fullmatch(token)) is not None
        }
        self._byte_encoder = bytes_to_unicode()
        self._byte_decoder = {character: byte for byte, character in self._byte_encoder.items()}
        self._cache: dict[str, tuple[int, ...]] = {}

        self.vocabulary_size = max(self._id_to_token) + 1
        self.end_of_sequence = self._resolve(end_of_sequence)
        self.begin_of_sequence = self._resolve(begin_of_sequence)

    def _resolve(self, token: str | int | None) -> int:
        if token is None:
            return -1
        if isinstance(token, int):
            return token
        if token not in self._vocab:
            raise TokenizerError(f"unknown token {token!r}")
        return self._vocab[token]

    def _post_processor(self, spec: Mapping[str, Any] | None) -> Callable[[list[int]], list[int]]:
        if spec is None or spec.get("type") == "ByteLevel":
            return lambda ids: ids
        if spec.get("type") == "Sequence":
            steps = [self._post_processor(s) for s in spec["processors"]]

            def run(ids: list[int]) -> list[int]:
                for step in steps:
                    ids = step(ids)
                return ids

            return run
        if spec.get("type") == "TemplateProcessing":
            template = spec["single"]
            special = {name: entry["ids"] for name, entry in spec.get("special_tokens", {}).items()}

            def apply(ids: list[int]) -> list[int]:
                out: list[int] = []
                for item in template:
                    if "Sequence" in item:
                        out.extend(ids)
                    else:
                        out.extend(special[item["SpecialToken"]["id"]])
                return out

            return apply
        raise TokenizerError(f"post-processor {spec.get('type')!r} is not supported")

    @property
    def special_ids(self) -> frozenset[int]:
        """Ids of the special (control) tokens, which :meth:`decode` skips."""
        return self._special_ids

    def showing(self, contents: Iterable[str]) -> BpeTokenizer:
        """A copy whose decoding keeps the special tokens ``contents`` as text (Mistral's ``[TOOL_CALLS]`` and
        Granite's ``<|tool_call|>`` mark tool calls, so the engine must see them); other contents are ignored."""
        wanted = set(contents)
        shown = frozenset(t.id for t in self._added if t.special and t.content in wanted)
        if shown == self._shown:
            return self
        copy = _copy.copy(self)
        copy._shown = shown
        return copy

    def token_to_id(self, token: str) -> int | None:
        return self._vocab.get(token)

    def id_to_token(self, index: int) -> str | None:
        return self._id_to_token.get(index)

    # -- encoding --

    def _split_added(self, text: str) -> list[tuple[str, AddedToken | None]]:
        if self._added_pattern is None:
            return [(text, None)]
        parts: list[tuple[str, AddedToken | None]] = []
        position = 0
        for match in self._added_pattern.finditer(text):
            token = self._added_by_content[match.group()]
            before = text[position : match.start()]
            if token.lstrip:
                before = before.rstrip()
            if before:
                parts.append((before, None))
            parts.append((match.group(), token))
            position = match.end()
            if token.rstrip:
                while position < len(text) and text[position].isspace():
                    position += 1
        if position < len(text):
            parts.append((text[position:], None))
        return parts

    def _word(self, piece: str) -> tuple[int, ...]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        if self._ignore_merges and piece in self._vocab:
            result: tuple[int, ...] = (self._vocab[piece],)
        else:
            symbols: list[int] = []
            previous_unknown = False
            for character in piece:
                index = self._vocab.get(character)
                if index is not None:
                    symbols.append(index)
                    previous_unknown = False
                    continue
                if self._byte_fallback:
                    fallback = [self._vocab.get(f"<0x{b:02X}>") for b in character.encode("utf-8")]
                    if all(token is not None for token in fallback):
                        symbols.extend(token for token in fallback if token is not None)
                        previous_unknown = False
                        continue
                if self._unk is None:
                    raise TokenizerError(f"character {character!r} is not in the vocabulary")
                if not (self._fuse_unk and previous_unknown):
                    symbols.append(self._unk)
                previous_unknown = True
            result = tuple(_merge(symbols, self._merges))
        if len(self._cache) < 100_000 and len(piece) <= 256:
            self._cache[piece] = result
        return result

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Token ids for ``text``. ``add_special_tokens`` applies the post-processor (e.g. a BOS token), as the
        ``tokenizers`` default does; chat prompts rendered from a template already contain their special tokens."""
        ids: list[int] = []
        for i, (part, added) in enumerate(self._split_added(text)):
            if added is not None:
                ids.append(added.id)
                continue
            for piece in self._pre_tokenize([self._normalize(part)], i == 0):
                if piece:
                    if self._byte_level:
                        piece = "".join(self._byte_encoder[b] for b in piece.encode("utf-8"))
                    ids.extend(self._word(piece))
        return self._post(ids) if add_special_tokens else ids

    # -- decoding --

    def decode_bytes(self, tokens: Iterable[int], *, skip_special_tokens: bool = True) -> bytes:
        """The exact bytes of ``tokens`` (useful for streaming, where a token may end mid-character). Concatenating
        the bytes of single tokens gives the bytes of the sequence; see :attr:`strips_leading_space` for the one
        difference to :meth:`decode`."""
        out = bytearray()
        sentencepiece = self._sentencepiece
        for token in tokens:
            if skip_special_tokens and token in self._special_ids and token not in self._shown:
                continue
            text = self._id_to_token.get(token)
            if text is None:
                continue
            if sentencepiece is not None:
                byte = self._byte_tokens.get(token) if sentencepiece.byte_fallback else None
                if byte is not None:
                    out.append(byte)
                else:
                    out.extend(text.replace(sentencepiece.replacement, " ").encode("utf-8"))
                continue
            if text in self._added_by_content:
                out.extend(text.encode("utf-8"))
                continue
            for character in text:
                byte = self._byte_decoder.get(character)
                out.extend(character.encode("utf-8") if byte is None else (byte,))
        return bytes(out)

    def decode(self, tokens: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        tokens = [t for t in tokens if not (skip_special_tokens and t in self._special_ids and t not in self._shown)]
        sentencepiece = self._sentencepiece
        if (
            sentencepiece is not None
            and sentencepiece.strip_first_token
            and tokens
            and not (sentencepiece.byte_fallback and tokens[0] in self._byte_tokens)
        ):
            # tokenizers' Metaspace decoder drops every replacement character of the first token, not only one.
            first = (self._id_to_token.get(tokens[0]) or "").replace(sentencepiece.replacement, "")
            rest = self.decode_bytes(tokens[1:], skip_special_tokens=False)
            return (first.encode("utf-8") + rest).decode("utf-8", errors="replace")
        data = self.decode_bytes(tokens, skip_special_tokens=skip_special_tokens)
        if self.strips_leading_space and data.startswith(b" "):
            data = data[1:]
        return data.decode("utf-8", errors="replace")


# --- GGUF tokenizers ---------------------------------------------------------------------------------------------

# Llama 3 and Qwen2 split words like GPT-2 but also isolate newlines; they differ in how many digits form a piece.
_WORDS = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}%s| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_LLAMA3_PATTERN = _WORDS % "{1,3}"
_QWEN2_PATTERN = _WORDS % ""
_BYTE_LEVEL = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}
_BYTE_LEVEL_NO_REGEX = {**_BYTE_LEVEL, "use_regex": False}


def _split(pattern: str) -> dict[str, Any]:
    return {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False}


# ``tokenizer.ggml.pre`` -> (normalizer, pre-tokenizer, ignore_merges), matching the Hugging Face tokenizer the GGUF
# was converted from (llama.cpp hard-codes the same splits in llama-vocab.cpp).
_GGUF_PRE: dict[str, tuple[dict[str, Any] | None, dict[str, Any], bool]] = {
    "gpt-2": (None, _BYTE_LEVEL, False),
    "smollm": (
        None,
        {"type": "Sequence", "pretokenizers": [{"type": "Digits", "individual_digits": True}, _BYTE_LEVEL]},
        False,
    ),
    "qwen2": (
        {"type": "NFC"},
        {"type": "Sequence", "pretokenizers": [_split(_QWEN2_PATTERN), _BYTE_LEVEL_NO_REGEX]},
        False,
    ),
    "llama-bpe": (
        None,
        {"type": "Sequence", "pretokenizers": [_split(_LLAMA3_PATTERN), _BYTE_LEVEL_NO_REGEX]},
        True,
    ),
}

GGUF_PRE_TOKENIZERS = tuple(_GGUF_PRE)
"""The ``tokenizer.ggml.pre`` values this reader understands."""


def without_offsets(value: Any) -> Any:
    """A normalizer or pre-tokenizer description without ``trim_offsets``, which only moves offsets, never ids."""
    if isinstance(value, dict):
        return {k: without_offsets(v) for k, v in value.items() if k != "trim_offsets"}
    if isinstance(value, list):
        return [without_offsets(v) for v in value]
    return value


def gguf_pre_tokenizer(pre: str) -> tuple[Any, Any]:
    """``(normalizer, pre-tokenizer)`` of a ``tokenizer.ggml.pre`` value, without ``trim_offsets``."""
    normalizer, pre_tokenizer, _ = _GGUF_PRE[pre]
    return without_offsets(normalizer), without_offsets(pre_tokenizer)


SENTENCEPIECE_DECODER = {
    "type": "Sequence",
    "decoders": [
        {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
        {"type": "Strip", "content": " ", "start": 1, "stop": 0},
    ],
}
"""The decoder of a SentencePiece model with a space prefix: ``▁`` back to spaces, ``<0xAB>`` byte tokens, the
prefix's leading space dropped (without a prefix, the ``Strip`` step is left out)."""


def sentencepiece_normalizer(add_space_prefix: bool) -> dict[str, Any]:
    """SentencePiece's whitespace handling: a ``▁`` in front of every text (and after every special token) when
    ``add_space_prefix``, and every space escaped as ``▁``."""
    replace = {"type": "Replace", "pattern": {"String": " "}, "content": "\u2581"}
    if not add_space_prefix:
        return replace
    return {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "\u2581"}, replace]}


def sentencepiece_merges(
    tokens: Sequence[str], scores: Sequence[float], types: Sequence[int]
) -> tuple[list[list[str]], list[int]]:
    """The merges of a SentencePiece BPE model, in SentencePiece's own order: every split of a normal piece into two
    normal pieces, ranked by the merged piece's score (highest first). Equal scores share a rank, so the merge further
    left wins, as SentencePiece and llama.cpp break ties by position. Returns the merges and their ranks."""
    normal: dict[str, int] = {}
    for index, (token, kind) in enumerate(zip(tokens, types, strict=True)):
        if kind == 1:
            normal.setdefault(token, index)
    values = [float(scores[index]) for index in normal.values()]
    if any(value != value for value in values):
        raise TokenizerError("a SentencePiece score is not a number")
    rank_of = {value: rank for rank, value in enumerate(sorted(set(values), reverse=True))}
    found: list[tuple[int, int, int, str, str]] = []
    for piece, index in normal.items():
        for cut in range(1, len(piece)):
            left, right = piece[:cut], piece[cut:]
            if left in normal and right in normal:
                found.append((rank_of[float(scores[index])], index, normal[left], left, right))
    found.sort()
    return [[left, right] for *_, left, right in found], [rank for rank, *_ in found]


def _sentencepiece_spec(metadata: Mapping[str, Any]) -> dict[str, Any]:
    tokens: list[str] = list(metadata["tokenizer.ggml.tokens"])
    scores: list[float] = list(metadata.get("tokenizer.ggml.scores", [0.0] * len(tokens)))
    types: list[int] = list(metadata.get("tokenizer.ggml.token_type", [1] * len(tokens)))
    if len(scores) != len(tokens) or len(types) != len(tokens):
        raise TokenizerError("a SentencePiece vocabulary needs one score and one type per token")
    merges, ranks = sentencepiece_merges(tokens, scores, types)
    prefix = bool(metadata.get("tokenizer.ggml.add_space_prefix", True))
    unknown = metadata.get("tokenizer.ggml.unknown_token_id")
    if unknown is None:
        unknown = next((i for i, kind in enumerate(types) if kind == 2), None)
    added = [
        {"id": i, "content": token, "special": kind == 3, "lstrip": False, "rstrip": False, "normalized": False,
         "single_word": False}
        for i, (token, kind) in enumerate(zip(tokens, types, strict=True))
        if kind in (3, 4)
    ]  # fmt: skip
    return {
        "added_tokens": added,
        "normalizer": sentencepiece_normalizer(prefix),
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": SENTENCEPIECE_DECODER
        if prefix
        else {**SENTENCEPIECE_DECODER, "decoders": SENTENCEPIECE_DECODER["decoders"][:3]},
        "model": {
            "type": "BPE",
            "vocab": {token: i for i, token in reversed(list(enumerate(tokens)))},
            "merges": merges,
            "merge_ranks": ranks,
            "unk_token": None if unknown is None else tokens[int(unknown)],
            "fuse_unk": True,
            "byte_fallback": 6 in types,
        },
    }


def spec_from_gguf(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """A ``tokenizer.json``-shaped description from GGUF ``tokenizer.ggml.*`` metadata: byte-level BPE (``gpt2``) or
    SentencePiece BPE (``llama``, :func:`sentencepiece_merges`)."""
    model = metadata.get("tokenizer.ggml.model")
    if model == "llama":
        return _sentencepiece_spec(metadata)
    if model != "gpt2":
        raise TokenizerError(
            f"GGUF tokenizer model {model!r} is not supported (gpt2-style BPE and llama SentencePiece)"
        )
    pre = metadata.get("tokenizer.ggml.pre", "gpt-2")
    if pre not in _GGUF_PRE:
        raise TokenizerError(f"GGUF pre-tokenizer {pre!r} is not supported (supported: {', '.join(sorted(_GGUF_PRE))})")
    normalizer, pre_tokenizer, ignore_merges = _GGUF_PRE[pre]
    tokens: list[str] = list(metadata["tokenizer.ggml.tokens"])
    types: list[int] = list(metadata.get("tokenizer.ggml.token_type", [1] * len(tokens)))
    added = [
        {"id": i, "content": token, "special": kind == 3, "lstrip": False, "rstrip": False, "normalized": kind != 3,
         "single_word": False}
        for i, (token, kind) in enumerate(zip(tokens, types, strict=True))
        if kind in (3, 4)
    ]  # fmt: skip
    return {
        "added_tokens": added,
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "vocab": {token: i for i, token in enumerate(tokens)},
            "merges": list(metadata.get("tokenizer.ggml.merges", [])),
            "ignore_merges": ignore_merges,
        },
    }


def from_model_header(tokenizer: Mapping[str, Any]) -> BpeTokenizer:
    """The tokenizer stored in a ``model.dllm`` header by ``dllm import``."""
    if tokenizer.get("format") == "huggingface":
        config = tokenizer.get("tokenizer_config") or {}
        return BpeTokenizer(
            tokenizer["tokenizer_json"],
            end_of_sequence=special_token_text(config.get("eos_token")),
            begin_of_sequence=special_token_text(config.get("bos_token")),
        )
    if tokenizer.get("format") == "gguf":
        eos = tokenizer.get("tokenizer.ggml.eos_token_id")
        bos = tokenizer.get("tokenizer.ggml.bos_token_id")
        return BpeTokenizer(spec_from_gguf(tokenizer), end_of_sequence=eos, begin_of_sequence=bos)
    raise TokenizerError(f"unknown tokenizer format {tokenizer.get('format')!r}")


def special_token_text(value: Any) -> str | None:
    """``tokenizer_config.json`` stores special tokens as strings or as AddedToken dicts."""
    if isinstance(value, dict):
        return value.get("content")
    return value
