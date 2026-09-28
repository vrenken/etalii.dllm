"""Model Context Protocol server over stdio.

Lets MCP clients (Claude Code, Claude Desktop, IDEs) call the deterministic model as a tool, e.g.::

    claude mcp add dllm -- dllm-mcp
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from etalii_dllm.engine import default_engine
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


@server.tool(name="model_info", annotations=_DETERMINISTIC)
def model_info() -> str:
    """Returns the model id and the system fingerprint that identifies the exact weights."""
    engine = default_engine()
    return f"model: {engine.model.id}\nsystem_fingerprint: {engine.system_fingerprint}"


def main() -> None:
    server.run("stdio")
