"""Reranking with a language model (``dllm rerank``, ``POST /v1/rerank``, ``--rerank-model``; ``docs/retrieval.md``).

A chat model judges each document for a query, the recipe of the Qwen3-Reranker models: a fixed system message asks
whether the document meets the query, the user message holds the instruction, the query and the document, and the
prompt is the model's own chat template with the generation prompt (thinking switched off for thinking models). The
score is the probability of ``yes`` against ``no`` as the next token, ``sigmoid(logit(yes) - logit(no))`` in double
with the portable kernel, where ``yes`` and ``no`` are the first token of each word. Documents are ranked by a total
order: the higher score first, the earlier document on a tie. Scores are a pure function of the weights and the text,
so the ranking is the same on every machine.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from etalii_dllm.chat import ChatMessage
from etalii_dllm.numerics import sigmoid

if TYPE_CHECKING:
    from etalii_dllm.engine import DllmEngine

JUDGE_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the "
    'answer can only be "yes" or "no".'
)
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


@dataclass(frozen=True)
class Judgement:
    score: float
    """The probability (double) that the document is relevant."""
    tokens: int
    """Tokens of the prompt the model read."""


class Reranker:
    """Scores documents for a query with ``engine``'s model (see the module docstring)."""

    def __init__(self, engine: DllmEngine, instruction: str = DEFAULT_INSTRUCTION) -> None:
        self.engine = engine
        self.instruction = instruction
        yes, no = engine.tokenizer.encode("yes"), engine.tokenizer.encode("no")
        if not yes or not no or yes[0] == no[0]:
            raise ValueError(f"model {engine.model.id} cannot tell 'yes' from 'no' by their first token")
        self.yes, self.no = yes[0], no[0]
        key = f"{engine.system_fingerprint}|{instruction}"
        self.fingerprint = hashlib.sha256(key.encode()).hexdigest()
        """Names the model and instruction, so equal fingerprints give equal scores."""

    def prompt(self, query: str, document: str, instruction: str | None = None) -> str:
        user = f"<Instruct>: {instruction or self.instruction}\n<Query>: {query}\n<Document>: {document}"
        messages = [ChatMessage("system", JUDGE_SYSTEM), ChatMessage("user", user)]
        return self.engine.render_chat(messages, thinking=False)

    def judge(self, query: str, document: str, instruction: str | None = None) -> Judgement:
        tokens = self.engine.tokenizer.encode(self.prompt(query, document, instruction))
        logits = self.engine.model.forward(tokens)
        return Judgement(sigmoid(float(logits[self.yes]) - float(logits[self.no])), len(tokens))

    def rerank(self, query: str, documents: Sequence[str], instruction: str | None = None) -> list[tuple[int, float]]:
        """``(index, score)`` of every document, best first (ties on the lower index)."""
        return [(i, j.score) for i, j in self.judgements(query, documents, instruction)]

    def judgements(
        self, query: str, documents: Sequence[str], instruction: str | None = None
    ) -> list[tuple[int, Judgement]]:
        """``(index, judgement)`` of every document, best first (ties on the lower index)."""
        if not query.strip():
            raise ValueError("the query must not be empty")
        judged = [self.judge(query, document, instruction) for document in documents]
        order = sorted(range(len(judged)), key=lambda i: (-judged[i].score, i))
        return [(i, judged[i]) for i in order]
