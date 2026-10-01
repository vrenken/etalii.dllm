"""Deterministic retrieval: an exact vector index of text chunks, built and searched with an embedding model.

Building is reproducible byte for byte: files are read in sorted path order with their line endings normalised,
split into chunks at fixed rules (paragraphs packed up to ``chunk_tokens`` tokens, long paragraphs split at word
boundaries), and every chunk is embedded on its own (as a ``document`` when the model has that prompt). Search is
exact, no approximate nearest neighbours: the query is embedded (as a ``query``), compared with every chunk by the
``cosine_similarity`` kernel, and ranked by a total order (higher score first, ties on the earlier chunk). So the
same index, model and query give the same passages on every machine, and the index fingerprint names all of it.

The index file is safetensors: the vectors as ``vectors [chunks, dim]`` and the chunks, settings and embedding model
in the ``dllm-index`` metadata (``docs/retrieval.md``).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from etalii_dllm.numerics import cosine_similarity

if TYPE_CHECKING:
    from etalii_dllm.engine import DllmEngine

INDEX_FORMAT = "dllm-index"
INDEX_VERSION = 1
DEFAULT_CHUNK_TOKENS = 256
# Files `dllm index build` reads from a directory; other files are skipped.
# White space for chunking: ASCII only, so chunks do not depend on the Python version's Unicode tables.
WHITESPACE = " \t\n\r\f\v"
TEXT_SUFFIXES = (".md", ".markdown", ".txt", ".rst", ".html", ".htm", ".json", ".py", ".csv", ".yaml", ".yml")


class RetrievalError(ValueError):
    """A bad index file or an index used with the wrong embedding model."""


@dataclass(frozen=True)
class Chunk:
    source: str
    """The file the text came from, relative to the directory given to ``build`` (``/`` separated)."""
    start: int
    """Character offsets of the text in the (line-ending normalised) file."""
    end: int
    text: str

    def to_json(self) -> dict[str, Any]:
        return {"source": self.source, "start": self.start, "end": self.end, "text": self.text}


@dataclass(frozen=True)
class Hit:
    rank: int
    score: float
    chunk: Chunk
    index: int


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray
    model: dict[str, Any]
    """The embedding model: ``fingerprint`` (its weights), ``id`` and the ``path`` it was loaded from, if known."""
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    fingerprint: str = field(default="")

    def __post_init__(self) -> None:
        self.vectors = np.ascontiguousarray(self.vectors, dtype=np.float32)
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.chunks):
            raise RetrievalError("the index needs one vector per chunk")
        expected = self._fingerprint()
        if self.fingerprint and self.fingerprint != expected:
            raise RetrievalError("the index fingerprint does not match its contents")
        self.fingerprint = expected

    def _settings(self) -> dict[str, Any]:
        return {
            "format": INDEX_FORMAT,
            "version": INDEX_VERSION,
            "model": self.model,
            "chunk_tokens": self.chunk_tokens,
            "chunks": [chunk.to_json() for chunk in self.chunks],
        }

    def _fingerprint(self) -> str:
        digest = hashlib.sha256(json.dumps(self._settings(), sort_keys=True, separators=(",", ":")).encode())
        digest.update(np.asarray(self.vectors.shape, dtype="<i8").tobytes())
        digest.update(self.vectors.astype("<f4").tobytes())
        return digest.hexdigest()

    @property
    def dimensions(self) -> int:
        return int(self.vectors.shape[1])

    def save(self, path: str | Path) -> None:
        from etalii_dllm.importing.safetensors import write_safetensors

        settings = {**self._settings(), "fingerprint": self.fingerprint}
        metadata = {"format": INDEX_FORMAT, "index": json.dumps(settings, sort_keys=True, separators=(",", ":"))}
        write_safetensors(path, {"vectors": self.vectors}, metadata)

    @staticmethod
    def load(path: str | Path) -> Index:
        from etalii_dllm.importing.safetensors import SafetensorsError, SafetensorsFile

        try:
            file = SafetensorsFile(path)
            if file.metadata.get("format") != INDEX_FORMAT:
                raise RetrievalError(f"{path}: not a dllm index")
            settings = json.loads(file.metadata["index"])
            vectors = file["vectors"].to_float32()
        except (OSError, SafetensorsError, KeyError, json.JSONDecodeError) as problem:
            raise RetrievalError(f"{path}: not a dllm index ({problem})") from problem
        if settings.get("version") != INDEX_VERSION:
            raise RetrievalError(f"{path}: index version {settings.get('version')} is not supported")
        chunks = [Chunk(c["source"], int(c["start"]), int(c["end"]), c["text"]) for c in settings["chunks"]]
        return Index(chunks, vectors, settings["model"], int(settings["chunk_tokens"]), settings["fingerprint"])

    def check_model(self, engine: DllmEngine) -> None:
        fingerprint = getattr(engine.model, "weights_fingerprint", None)
        if fingerprint != self.model.get("fingerprint"):
            raise RetrievalError(
                f"the index was built with {self.model.get('id')} ({str(self.model.get('fingerprint'))[:12]}), "
                f"not {engine.model.id} ({str(fingerprint)[:12]})"
            )

    def search(self, engine: DllmEngine, query: str, top: int = 5) -> list[Hit]:
        """The ``top`` chunks closest to ``query``, best first (ties on the earlier chunk)."""
        if top < 1:
            raise ValueError("top must be at least 1")
        if not query.strip():
            raise ValueError("the query must not be empty")
        self.check_model(engine)
        if not self.chunks:
            return []
        vector = engine.embed(query, input_type=_prompt(engine, "query")).vector
        scores = cosine_similarity(self.vectors, vector)
        order = np.lexsort((np.arange(len(self.chunks)), -scores.astype(np.float64)))[:top]
        return [Hit(rank + 1, float(scores[i]), self.chunks[i], int(i)) for rank, i in enumerate(order)]


def _prompt(engine: DllmEngine, name: str) -> str | None:
    prompts = (engine.embedding or {}).get("prompts") or {}
    return name if name in prompts else None


_PARAGRAPH_BREAK = re.compile(r"\n[ \t\r\f\v]*\n")


def _paragraphs(text: str) -> list[tuple[int, int]]:
    """Spans of the paragraphs of ``text`` (separated by blank lines), without surrounding white space."""
    spans = []
    position = 0
    for match in [*_PARAGRAPH_BREAK.finditer(text), None]:
        stop = len(text) if match is None else match.start()
        block = text[position:stop]
        stripped = block.strip(WHITESPACE)
        if stripped:
            start = position + block.index(stripped)
            spans.append((start, start + len(stripped)))
        if match is not None:
            position = match.end()
    return spans


def _words(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans = []
    position = start
    while position < end:
        while position < end and text[position] in WHITESPACE:
            position += 1
        if position >= end:
            break
        stop = position
        while stop < end and text[stop] not in WHITESPACE:
            stop += 1
        spans.append((position, stop))
        position = stop
    return spans


def chunk_text(
    text: str, count: Callable[[str], int], chunk_tokens: int = DEFAULT_CHUNK_TOKENS
) -> list[tuple[int, int]]:
    """Character spans of the chunks of ``text``: paragraphs packed in order while the chunk stays within
    ``chunk_tokens`` tokens (``count`` counts them); a paragraph that alone is longer is split at word boundaries
    (a single word longer than that becomes a chunk of its own)."""
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be at least 1")
    pieces: list[tuple[int, int]] = []
    for start, end in _paragraphs(text):
        if count(text[start:end]) <= chunk_tokens:
            pieces.append((start, end))
            continue
        current: tuple[int, int] | None = None
        for word in _words(text, start, end):
            if current is not None and count(text[current[0] : word[1]]) <= chunk_tokens:
                current = (current[0], word[1])
            else:
                if current is not None:
                    pieces.append(current)
                current = word
        if current is not None:
            pieces.append(current)
    chunks: list[tuple[int, int]] = []
    for piece in pieces:
        if chunks and count(text[chunks[-1][0] : piece[1]]) <= chunk_tokens:
            chunks[-1] = (chunks[-1][0], piece[1])
        else:
            chunks.append(piece)
    return chunks


def read_documents(paths: Sequence[str | Path]) -> list[tuple[str, str]]:
    """``(source, text)`` of every file under ``paths`` (directories: the files with a ``TEXT_SUFFIXES`` suffix,
    recursively), sorted by source; line endings normalised to ``\\n``. Sources are relative to the directory given,
    or the file name for a file given directly."""
    documents: dict[str, str] = {}
    for given in paths:
        root = Path(given)
        if root.is_dir():
            files = [(p.relative_to(root).as_posix(), p) for p in root.rglob("*") if p.is_file()]
            files = [(name, p) for name, p in files if p.suffix.lower() in TEXT_SUFFIXES]
        elif root.is_file():
            files = [(root.name, root)]
        else:
            raise FileNotFoundError(f"{given}: no such file or directory")
        for name, path in files:
            if name in documents:
                raise ValueError(f"two documents are called {name}")
            raw = path.read_bytes().decode("utf-8", errors="replace")
            documents[name] = raw.replace("\r\n", "\n").replace("\r", "\n")
    return sorted(documents.items(), key=lambda item: item[0].encode("utf-8"))


def build_index(
    engine: DllmEngine,
    documents: Iterable[tuple[str, str]],
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    model_path: str | Path | None = None,
    progress: Callable[[int, Chunk], None] | None = None,
) -> Index:
    """Chunks and embeds ``documents`` (``(source, text)`` pairs, in the order given) with ``engine``."""
    fingerprint = getattr(engine.model, "weights_fingerprint", None)
    if fingerprint is None:
        raise ValueError(f"model {engine.model.id} has no weights fingerprint")
    prompt = _prompt(engine, "document")
    chunks: list[Chunk] = []
    vectors: list[np.ndarray] = []
    for source, text in documents:
        for start, end in chunk_text(text, engine.count_tokens, chunk_tokens):
            chunk = Chunk(source, start, end, text[start:end])
            vectors.append(engine.embed(chunk.text, input_type=prompt).vector)
            chunks.append(chunk)
            if progress is not None:
                progress(len(chunks), chunk)
    dimensions = vectors[0].shape[0] if vectors else 0
    model = {"fingerprint": fingerprint, "id": engine.model.id, "path": str(model_path) if model_path else None}
    matrix = np.stack(vectors) if vectors else np.zeros((0, dimensions), np.float32)
    return Index(chunks, matrix, model, chunk_tokens)


DEFAULT_TOP = 3
GROUNDING_INSTRUCTION = (
    "Answer using the passages below when they are relevant, and say which passage you used by its number. "
    "If they do not contain the answer, say so."
)


class Retriever:
    """Grounds chats in an index: the last user message is the query, and the ``top`` passages found are added to
    the system message (:meth:`ground`). Equal requests get equal passages, so answers stay reproducible; the
    :attr:`fingerprint` (index, embedding model and ``top``) goes into the chat engine's ``system_fingerprint``."""

    def __init__(self, index: Index, embedder: DllmEngine, top: int = DEFAULT_TOP) -> None:
        if top < 1:
            raise ValueError("top must be at least 1")
        index.check_model(embedder)
        self.index = index
        self.embedder = embedder
        self.top = top
        self.fingerprint = hashlib.sha256(f"{index.fingerprint}|{top}".encode()).hexdigest()

    @staticmethod
    def open(path: str | Path, top: int = DEFAULT_TOP, embedding_model: str | Path | None = None) -> Retriever:
        """The index at ``path`` with its embedding model (``embedding_model``, else the path the index records)."""
        from etalii_dllm.engine import DllmEngine

        index = Index.load(path)
        model_path = embedding_model or index.model.get("path")
        if not model_path:
            raise RetrievalError(f"{path}: the index does not record its embedding model; name it")
        return Retriever(index, DllmEngine.from_model_file(model_path, prompt_cache=0), top)

    def search(self, query: str, top: int | None = None) -> list[Hit]:
        return self.index.search(self.embedder, query, top or self.top)

    def ground(self, messages: Sequence[Any]) -> tuple[list[Any], list[Hit]]:
        """``messages`` with the passages for the last user message added to the system message (a new first
        message when there is none), and the passages; unchanged when there is no user text to search for."""
        from etalii_dllm.chat import ChatMessage

        query = next((m.content for m in reversed(messages) if m.role == "user" and m.content.strip()), None)
        if query is None:
            return list(messages), []
        hits = self.search(query)
        if not hits:
            return list(messages), []
        passages = "\n\n".join(f"[{h.rank}] ({h.chunk.source})\n{h.chunk.text}" for h in hits)
        context = f"{GROUNDING_INSTRUCTION}\n\n{passages}"
        grounded = list(messages)
        if grounded and grounded[0].role == "system":
            first = grounded[0]
            grounded[0] = ChatMessage("system", f"{first.content}\n\n{context}" if first.content else context)
        else:
            grounded.insert(0, ChatMessage("system", context))
        return grounded, hits
