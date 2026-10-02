"""Model Context Protocol server over stdio.

Lets MCP clients (Claude Code, Claude Desktop, IDEs) call the deterministic model as a tool, e.g.::

    claude mcp add dllm -- dllm-mcp --model smollm2-135m.dllm

Besides the tools it offers resources (the model card, the chat template, the determinism guarantee) and prompts
(ready-made tasks for the ``chat`` tool). Everything it returns is derived from the model file and the request, so
the same server gives the same answers.
"""

from __future__ import annotations

import json
import os
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from etalii_dllm import receipts
from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import (
    MODEL_ENVIRONMENT_VARIABLE,
    ChatRequest,
    ResponseFormat,
    add_runtime_arguments,
    default_engine,
    use_model_file,
)
from etalii_dllm.sampling import SamplingOptions

server = MCPServer("dllm")

_DETERMINISTIC = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)


@server.tool(name="generate", annotations=_DETERMINISTIC)
def generate(prompt: str, max_tokens: int = 64, temperature: float = 0.0, seed: int = 0) -> str:
    """Generates a continuation of the prompt with the EtAlii deterministic LLM.

    The same arguments always return the same text.
    """
    options = SamplingOptions(temperature=temperature, seed=seed)
    return default_engine().complete(prompt, max_tokens, options).text


@server.tool(name="chat", annotations=_DETERMINISTIC)
def chat(
    messages: list[dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 0,
    json_schema: dict[str, Any] | None = None,
    receipt: bool = False,
    previous_receipt: str = "",
) -> str:
    """Answers a conversation with the EtAlii deterministic LLM, using the model's own chat template.

    ``messages`` is a list of {"role": "system" | "user" | "assistant", "content": "..."}. With ``json_schema`` the
    answer is JSON valid under that schema (constrained decoding). The same arguments always return the same text.
    With ``receipt`` the result is JSON {"content", "receipt"}: the receipt lets anyone re-run and check the answer
    (``verify_receipt``, ``dllm replay``). ``previous_receipt``, the receipt id of the previous turn, chains the
    receipts of a conversation.
    """
    conversation = [ChatMessage(m.get("role", "user"), m.get("content", "")) for m in messages]
    response_format = ResponseFormat("json_schema", json_schema) if json_schema is not None else ResponseFormat()
    request = ChatRequest(
        conversation,
        max_tokens,
        SamplingOptions(temperature=temperature, seed=seed),
        response_format=response_format,
        previous_receipt=previous_receipt or None,
    )
    result = default_engine().chat_completion(request)
    if receipt:
        return json.dumps({"content": result.content, "receipt": result.receipt}, ensure_ascii=False, indent=2)
    return result.content


@server.tool(name="verify_receipt", annotations=_DETERMINISTIC)
def verify_receipt(receipt: dict[str, Any]) -> str:
    """Re-runs the request a generation receipt records and checks that the output is the same, bit for bit.

    Returns JSON {"ok", "reasons", "notes", "receipt"}: ``reasons`` says why a receipt does not verify (it was
    edited, made with other weights, or the output differs), ``receipt`` is the one the replay produced.
    """
    return json.dumps(receipts.verify(default_engine(), receipt).to_json(), ensure_ascii=False, indent=2)


@server.tool(name="search_documents", annotations=_DETERMINISTIC)
def search_documents(query: str, top: int = 5) -> str:
    """Finds the passages of the server's document index (``--index``) closest to the query, best first.

    Returns a JSON list of {"rank", "score", "source", "start", "end", "text"}. The search is exact, so the same
    query always returns the same passages.
    """
    retriever = default_engine().retriever
    if retriever is None:
        raise ValueError("this server has no document index; start it with --index (dllm index build)")
    hits = retriever.search(query, top)
    rows = [{"rank": h.rank, "score": h.score, **h.chunk.to_json()} for h in hits]
    return json.dumps(rows, ensure_ascii=False, indent=2)


@server.tool(name="model_info", annotations=_DETERMINISTIC)
def model_info() -> str:
    """Returns the model id and the system fingerprint that identifies the exact weights."""
    engine = default_engine()
    return f"model: {engine.model.id}\nsystem_fingerprint: {engine.system_fingerprint}"


DETERMINISM = """\
# Determinism

EtAlii.Dllm produces bit-identical output on every run and every supported machine: the same weights (identified by
the system fingerprint), the same prompt or conversation and the same options (max_tokens, temperature, seed, schema)
always give the same text, regardless of load, batching or thread scheduling.

- Temperature 0 is greedy decoding; with a temperature above 0 the seed selects the one reproducible sample.
- Every reduction (matmul, softmax, norms, attention) runs in a fixed, documented order with double accumulation.
- Results are the same on every supported machine (x86-64 and arm64 CPUs, every SIMD path, NVIDIA GPUs).
- A chat answer can carry a receipt (``receipt: true``) that anyone can re-run and check (``verify_receipt``).
- Ids and fingerprints in the answers are derived from the request and the weights, never from a clock.
"""

FALLBACK_TEMPLATE = """\
This model has no chat template of its own. Conversations use the engine's fixed format: every message is
rendered as "<|role|>\\n" + content + "\\n" and the prompt ends with "<|assistant|>\\n".
"""


@server.resource(
    "dllm://model",
    name="model",
    title="Model card",
    description="The served model: id, system fingerprint, architecture, source and licence.",
    mime_type="application/json",
)
def model_card() -> str:
    engine = default_engine()
    card: dict[str, Any] = {
        "id": engine.model.id,
        "system_fingerprint": engine.system_fingerprint,
        "vocabulary_size": engine.model.vocabulary_size,
        "chat_template": engine.chat_template is not None,
    }
    path = os.environ.get(MODEL_ENVIRONMENT_VARIABLE)
    if path:
        from etalii_dllm.modelfile import ModelFile

        file = ModelFile(path, verify=False)
        card["architecture"] = file.config.to_dict()
        card["source"] = {k: v for k, v in file.source.items() if k != "files"}
        card["licence"] = {k: v for k, v in file.licence.items() if k != "text"}
        if file.fine_tuning:
            card["fine_tuning"] = file.fine_tuning
    return json.dumps(card, sort_keys=True, ensure_ascii=False, indent=2)


@server.resource(
    "dllm://model/chat-template",
    name="chat-template",
    title="Chat template",
    description="The Jinja chat template the model's conversations are rendered with.",
    mime_type="text/plain",
)
def chat_template() -> str:
    template = default_engine().chat_template
    return template.source if template is not None else FALLBACK_TEMPLATE


@server.resource(
    "dllm://determinism",
    name="determinism",
    title="Determinism guarantee",
    description="What makes the answers reproducible, and what may differ between machines.",
    mime_type="text/markdown",
)
def determinism() -> str:
    return DETERMINISM


@server.prompt(name="summarize", title="Summarize a text")
def summarize(text: str, max_words: str = "60") -> str:
    """Asks for a short summary of a text; send the prompt's message to the ``chat`` tool."""
    return f"Summarize the following text in at most {max_words} words.\n\n{text}"


@server.prompt(name="translate", title="Translate a text")
def translate(text: str, language: str) -> str:
    """Asks for a translation of a text into another language; send the prompt's message to the ``chat`` tool."""
    return f"Translate the following text into {language}. Answer with the translation only.\n\n{text}"


@server.prompt(name="extract_json", title="Extract fields as JSON")
def extract_json(text: str, fields: str) -> str:
    """Asks for the given comma-separated fields of a text as a JSON object; pair it with the ``chat`` tool's
    ``json_schema`` to guarantee the shape."""
    names = ", ".join(name.strip() for name in fields.split(",") if name.strip())
    return (
        f"Extract these fields from the text below and answer with one JSON object with exactly these keys: "
        f"{names}. Use null for a field the text does not mention.\n\n{text}"
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="dllm-mcp", description="EtAlii.Dllm MCP server (stdio)")
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    use_model_file(
        args.model,
        args.quantize,
        args.threads,
        args.device,
        args.prompt_cache,
        args.adapter,
        steer=args.steer,
        steer_strength=args.steer_strength,
        index=args.index,
        index_top=args.index_top,
        index_mode=args.index_mode,
        rerank_model=args.rerank_model,
        embedding_model=args.embedding_model,
        speculate=args.speculate,
        draft_model=args.draft_model,
        prompt_cache_dir=args.prompt_cache_dir,
        sign_key=args.sign_key,
        response_cache=args.response_cache,
        audit_every=args.audit_every,
    )
    server.run("stdio")
