"""Deterministic retrieval: an exact vector index of text chunks, built and searched with an embedding model.

Building is reproducible byte for byte: files are read in sorted path order with their line endings normalised,
split into chunks at fixed rules (paragraphs packed up to ``chunk_tokens`` tokens, long paragraphs split at word
boundaries), and every chunk is embedded on its own (as a ``document`` when the model has that prompt). Search is
exact, no approximate nearest neighbours: the query is embedded (as a ``query``), compared with every chunk by the
``cosine_similarity`` kernel, and ranked by a total order (higher score first, ties on the earlier chunk). So the
same index, model and query give the same passages on every machine, and the index fingerprint names all of it.

Lexical search (``mode="lexical"``) scores chunks with Okapi BM25 over terms from :func:`terms` (NFKC, lower case and
runs of letters, marks and digits, all from the pinned Unicode tables), in double, with the query's distinct terms in
order of first appearance and the portable ``log`` kernel for the IDF. The statistics are a pure function of the chunk
texts, so every index has them. ``mode="hybrid"`` fuses the dense and lexical rankings with reciprocal rank fusion
(``1 / (RRF_K + rank)`` from each ranking a chunk is in, dense first). Every ranking breaks ties on the earlier chunk.

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

from etalii_dllm import unicode
from etalii_dllm.numerics import cosine_similarity, log

if TYPE_CHECKING:
    from etalii_dllm.engine import DllmEngine
    from etalii_dllm.reranking import Reranker

INDEX_FORMAT = "dllm-index"
INDEX_VERSION = 1
DEFAULT_CHUNK_TOKENS = 256
# Files `dllm index build` reads from a directory; other files are skipped.
# White space for chunking: ASCII only, so chunks do not depend on the Python version's Unicode tables.
WHITESPACE = " \t\n\r\f\v"
TEXT_SUFFIXES = (".md", ".markdown", ".txt", ".rst", ".html", ".htm", ".json", ".py", ".csv", ".yaml", ".yml")
MODES = ("dense", "lexical", "hybrid")
"""How :meth:`Index.search` ranks: embeddings (the default), BM25, or both fused."""
BM25_K1 = 1.2
BM25_B = 0.75
RRF_K = 60
"""Reciprocal rank fusion constant (the value of Cormack et al. 2009)."""


def terms(text: str) -> list[str]:
    """The lexical terms of ``text``: NFKC-normalised, lower-cased, then the maximal runs of letters, marks and
    digits (Unicode categories ``L*``, ``M*``, ``N*`` of the pinned tables), in order."""
    text = unicode.lower(unicode.normalize("NFKC", text))
    found: list[str] = []
    start = -1
    for position, character in enumerate(text):
        word = character.isalnum() if character.isascii() else unicode.category(character)[0] in "LMN"
        if word and start < 0:
            start = position
        elif not word and start >= 0:
            found.append(text[start:position])
            start = -1
    if start >= 0:
        found.append(text[start:])
    return found


class LexicalStatistics:
    """BM25 statistics of a list of texts: term counts per text, document frequencies and lengths."""

    def __init__(self, texts: Sequence[str]) -> None:
        self.counts: list[dict[str, int]] = []
        self.lengths: list[int] = []
        self.frequencies: dict[str, int] = {}
        for text in texts:
            counts: dict[str, int] = {}
            found = terms(text)
            for term in found:
                counts[term] = counts.get(term, 0) + 1
            for term in counts:
                self.frequencies[term] = self.frequencies.get(term, 0) + 1
            self.counts.append(counts)
            self.lengths.append(len(found))
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0

    def idf(self, term: str) -> float:
        frequency = self.frequencies.get(term, 0)
        return log(1.0 + (len(self.counts) - frequency + 0.5) / (frequency + 0.5))

    def scores(self, query: str) -> list[float]:
        """The BM25 score of every text for ``query``: per text, the query's distinct terms in order of first
        appearance, each adding ``idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))``."""
        distinct = list(dict.fromkeys(terms(query)))
        weights = [(term, self.idf(term)) for term in distinct if term in self.frequencies]
        result = []
        for counts, length in zip(self.counts, self.lengths, strict=True):
            norm = BM25_K1 * (1.0 - BM25_B + BM25_B * length / self.average_length) if self.average_length else 0.0
            total = 0.0
            for term, idf in weights:
                frequency = counts.get(term, 0)
                if frequency:
                    total += idf * (frequency * (BM25_K1 + 1.0)) / (frequency + norm)
            result.append(total)
        return result


def _ranking(scores: Sequence[float]) -> list[int]:
    """Indices by score, highest first, ties on the lower index (a total order)."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


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
    _lexical: LexicalStatistics | None = field(default=None, init=False, repr=False, compare=False)

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

    @property
    def lexical(self) -> LexicalStatistics:
        """BM25 statistics of the chunks, computed once."""
        if self._lexical is None:
            self._lexical = LexicalStatistics([chunk.text for chunk in self.chunks])
        return self._lexical

    def search(self, engine: DllmEngine | None, query: str, top: int = 5, mode: str = "dense") -> list[Hit]:
        """The ``top`` chunks for ``query``, best first (ties on the earlier chunk), ranked by ``mode`` (see the
        module docstring). Lexical search returns only chunks sharing a term with the query and needs no
        ``engine``."""
        if top < 1:
            raise ValueError("top must be at least 1")
        if mode not in MODES:
            raise ValueError(f"unknown search mode {mode!r}; expected one of {', '.join(MODES)}")
        if not query.strip():
            raise ValueError("the query must not be empty")
        if mode != "lexical":
            if engine is None:
                raise ValueError(f"{mode} search needs the embedding model")
            self.check_model(engine)
        if not self.chunks:
            return []
        if mode == "lexical":
            scores = self.lexical.scores(query)
            order = [i for i in _ranking(scores) if scores[i] > 0.0]
        else:
            assert engine is not None
            vector = engine.embed(query, input_type=_prompt(engine, "query")).vector
            scores = [float(x) for x in cosine_similarity(self.vectors, vector)]
            order = _ranking(scores)
            if mode == "hybrid":
                fused = [0.0] * len(self.chunks)
                lexical = self.lexical.scores(query)
                for ranking in (order, [i for i in _ranking(lexical) if lexical[i] > 0.0]):
                    for rank, i in enumerate(ranking, 1):
                        fused[i] += 1.0 / (RRF_K + rank)
                scores = fused
                order = _ranking(scores)
        return [Hit(rank + 1, scores[i], self.chunks[i], i) for rank, i in enumerate(order[:top])]


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
RERANK_DEPTH = 4
"""With a reranker, grounding reranks this many times ``top`` hits."""
GROUNDING_INSTRUCTION = (
    "Answer using the passages below when they are relevant, and say which passage you used by its number. "
    "If they do not contain the answer, say so."
)


class Retriever:
    """Grounds chats in an index: the last user message is the query, and the ``top`` passages found are added to
    the system message (:meth:`ground`). Equal requests get equal passages, so answers stay reproducible; the
    :attr:`fingerprint` (index, embedding model, ``top``, and the search mode and reranker when they are not the
    defaults) goes into the chat engine's ``system_fingerprint``.

    With a ``reranker`` (:class:`etalii_dllm.reranking.Reranker`), the first ``RERANK_DEPTH * top`` hits are scored
    again by the reranking model and the best ``top`` of them, in its order, are the passages."""

    def __init__(
        self,
        index: Index,
        embedder: DllmEngine | None,
        top: int = DEFAULT_TOP,
        mode: str = "dense",
        reranker: Reranker | None = None,
    ) -> None:
        if top < 1:
            raise ValueError("top must be at least 1")
        if mode not in MODES:
            raise ValueError(f"unknown search mode {mode!r}; expected one of {', '.join(MODES)}")
        if embedder is not None:
            index.check_model(embedder)
        elif mode != "lexical":
            raise ValueError(f"{mode} search needs the embedding model")
        self.index = index
        self.embedder = embedder
        self.top = top
        self.mode = mode
        self.reranker = reranker
        key = f"{index.fingerprint}|{top}"  # the dense default keeps the fingerprints it always had
        if mode != "dense":
            key += f"|{mode}"
        if reranker is not None:
            key += f"|rerank:{reranker.fingerprint}"
        self.fingerprint = hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def open(
        path: str | Path,
        top: int = DEFAULT_TOP,
        embedding_model: str | Path | None = None,
        mode: str = "dense",
        rerank_model: str | Path | None = None,
    ) -> Retriever:
        """The index at ``path`` with its embedding model (``embedding_model``, else the path the index records; not
        loaded for lexical search) and, with ``rerank_model``, a reranker."""
        from etalii_dllm.engine import DllmEngine

        index = Index.load(path)
        embedder = None
        if mode != "lexical":
            model_path = embedding_model or index.model.get("path")
            if not model_path:
                raise RetrievalError(f"{path}: the index does not record its embedding model; name it")
            embedder = DllmEngine.from_model_file(model_path, prompt_cache=0)
        reranker = None
        if rerank_model:
            from etalii_dllm.reranking import Reranker

            reranker = Reranker(DllmEngine.from_model_file(rerank_model, prompt_cache=0))
        return Retriever(index, embedder, top, mode, reranker)

    def search(self, query: str, top: int | None = None) -> list[Hit]:
        top = top or self.top
        if self.reranker is None:
            return self.index.search(self.embedder, query, top, self.mode)
        candidates = self.index.search(self.embedder, query, RERANK_DEPTH * top, self.mode)
        ranked = self.reranker.rerank(query, [hit.chunk.text for hit in candidates])
        return [
            Hit(rank, score, candidates[i].chunk, candidates[i].index)
            for rank, (i, score) in enumerate(ranked[:top], 1)
        ]

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
