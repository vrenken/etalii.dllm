"""Model Context Protocol server over stdio.

Lets MCP clients (Claude Code, Claude Desktop, IDEs) call the deterministic model as a tool, e.g.::

    claude mcp add dllm -- dllm-mcp --model smollm2-135m.dllm
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import ChatRequest, ResponseFormat, default_engine, use_model_file
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
) -> str:
    """Answers a conversation with the EtAlii deterministic LLM, using the model's own chat template.

    ``messages`` is a list of {"role": "system" | "user" | "assistant", "content": "..."}. With ``json_schema`` the
    answer is JSON valid under that schema (constrained decoding). The same arguments always return the same text.
    """
    conversation = [ChatMessage(m.get("role", "user"), m.get("content", "")) for m in messages]
    response_format = ResponseFormat("json_schema", json_schema) if json_schema is not None else ResponseFormat()
    request = ChatRequest(
        conversation, max_tokens, SamplingOptions(temperature=temperature, seed=seed), response_format=response_format
    )
    return default_engine().chat_completion(request).content


@server.tool(name="model_info", annotations=_DETERMINISTIC)
def model_info() -> str:
    """Returns the model id and the system fingerprint that identifies the exact weights."""
    engine = default_engine()
    return f"model: {engine.model.id}\nsystem_fingerprint: {engine.system_fingerprint}"


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="dllm-mcp", description="EtAlii.Dllm MCP server (stdio)")
    parser.add_argument("--model", help="model.dllm file to serve (default: $DLLM_MODEL, else the placeholder model)")
    use_model_file(parser.parse_args(argv).model)
    server.run("stdio")
