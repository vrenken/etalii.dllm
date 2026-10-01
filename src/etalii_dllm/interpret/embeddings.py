"""The embedding explorer: nearest tokens, vector arithmetic (analogies) and word clouds.

Similarities come from the ``cosine_similarity`` kernel (double sums in a fixed order), vector arithmetic is
elementwise float32 in the order written, multi-token words are the ``column_mean`` of their rows, and rankings
break ties on the token id; so every list is the same on every machine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from etalii_dllm.interpret.lens import top_k
from etalii_dllm.numerics import column_mean, cosine_similarity
from etalii_dllm.tokenization import Tokenizer
from etalii_dllm.transformer import Transformer

SPACES = ("input", "output")


def embedding_matrix(model: Transformer, space: str = "input") -> np.ndarray:
    """``[vocabulary, hidden]`` token vectors: the input embeddings, or the LM head rows (``"output"``; the same
    matrix for models with tied embeddings)."""
    if space not in SPACES:
        raise ValueError(f"space must be one of {', '.join(SPACES)}")
    name = "token_embedding.weight"
    if space == "output" and not model.config.tie_word_embeddings:
        name = "lm_head.weight"
    return model.tensors[name].numpy()


def token_text(tokenizer: Tokenizer, token: int) -> str:
    """The text of one token as it appears inside a sequence (a leading space kept)."""
    decode_bytes = getattr(tokenizer, "decode_bytes", None)
    try:
        data = decode_bytes([token], skip_special_tokens=False) if decode_bytes else None
    except TypeError:
        data = decode_bytes([token])  # type: ignore[misc]
    if data is None:
        return tokenizer.decode([token])
    return data.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class Term:
    sign: int
    text: str
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class Neighbour:
    token: int
    text: str
    similarity: float


@dataclass(frozen=True)
class Neighbourhood:
    expression: str
    terms: tuple[Term, ...]
    space: str
    neighbours: list[Neighbour]


def parse_expression(expression: str) -> list[tuple[int, str]]:
    """``"king - man + woman"`` as ``[(1, " king"), (-1, " man"), (1, " woman")]``. Operators need spaces around
    them (``"x-ray"`` is one word). A bare word gets a leading space, as words inside a sentence have; text in
    quotes is taken exactly as written (``'"Paris"'`` has none)."""
    parts = re.split(r"\s+([+-])\s+", expression.strip())
    terms: list[tuple[int, str]] = []
    sign = 1
    for index, part in enumerate(parts):
        if index % 2:
            sign = 1 if part == "+" else -1
            continue
        if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'":
            text = part[1:-1]
        else:
            text = f" {part}" if part else part
        if not text:
            raise ValueError(f"empty term in {expression!r}")
        terms.append((sign, text))
    return terms


def neighbours(
    model: Transformer, tokenizer: Tokenizer, expression: str, top: int = 20, space: str = "input"
) -> Neighbourhood:
    """The ``top`` tokens most similar (cosine) to ``expression``, a word or a sum and difference of words; the
    tokens of the expression itself and special tokens are left out."""
    if top < 1:
        raise ValueError("top must be at least 1")
    matrix = embedding_matrix(model, space)
    vector = np.zeros(matrix.shape[1], dtype=np.float32)
    terms = []
    for sign, text in parse_expression(expression):
        tokens = tuple(tokenizer.encode(text))
        if not tokens:
            raise ValueError(f"{text!r} has no tokens")
        terms.append(Term(sign, text, tokens))
        rows = matrix[np.asarray(tokens, dtype=np.int64)]
        term = rows[0].copy() if len(tokens) == 1 else column_mean(rows)
        vector = vector + term if sign > 0 else vector - term
    similarities = cosine_similarity(matrix, vector)
    excluded = {t for term in terms for t in term.tokens} | set(getattr(tokenizer, "special_ids", ()))
    ranked = [i for i in top_k(similarities, top + len(excluded)) if i not in excluded][:top]
    return Neighbourhood(
        expression,
        tuple(terms),
        space,
        [Neighbour(i, token_text(tokenizer, i), float(similarities[i])) for i in ranked],
    )
