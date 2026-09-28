import json
import os
import subprocess
import sys

MESSAGES = [
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
    },
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "generate", "arguments": {"prompt": "Hi", "max_tokens": 8, "temperature": 0.5, "seed": 3}},
    },
    {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "chat",
            "arguments": {
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 40,
                "json_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
            },
        },
    },
    {"jsonrpc": "2.0", "id": 5, "method": "resources/list"},
    {"jsonrpc": "2.0", "id": 6, "method": "resources/read", "params": {"uri": "dllm://model"}},
    {"jsonrpc": "2.0", "id": 7, "method": "prompts/list"},
    {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "prompts/get",
        "params": {"name": "translate", "arguments": {"text": "Good morning", "language": "Dutch"}},
    },
]


def run_session() -> dict[int, dict]:
    """Sends each request and waits for its response before the next, so the server never sees EOF mid-request."""
    process = subprocess.Popen(
        [sys.executable, "-c", "from etalii_dllm.mcp_server import main; main()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert process.stdin is not None and process.stdout is not None
    responses: dict[int, dict] = {}
    try:
        for message in MESSAGES:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
            if "id" not in message:
                continue
            while message["id"] not in responses:
                line = process.stdout.readline()
                assert line, "MCP server exited before answering"
                response = json.loads(line)
                if "id" in response:
                    responses[response["id"]] = response
    finally:
        process.stdin.close()
        process.wait(timeout=30)
    return responses


def test_mcp_server_lists_and_runs_tools_deterministically():
    first = run_session()
    tools = {t["name"]: t for t in first[2]["result"]["tools"]}
    assert set(tools) == {"chat", "generate", "model_info"}
    assert tools["generate"]["annotations"]["idempotentHint"] is True

    second = run_session()
    assert first[3]["result"]["content"] == second[3]["result"]["content"]
    assert first[4]["result"]["content"] == second[4]["result"]["content"]
    answer = json.loads(first[4]["result"]["content"][0]["text"])
    assert isinstance(answer["ok"], bool)


def test_mcp_server_offers_resources_and_prompts():
    responses = run_session()
    uris = [r["uri"] for r in responses[5]["result"]["resources"]]
    assert uris == ["dllm://model", "dllm://model/chat-template", "dllm://determinism"]
    card = json.loads(responses[6]["result"]["contents"][0]["text"])
    assert card["id"] == "dllm-bigram-257-42" and card["system_fingerprint"].startswith("fp_")

    prompts = {p["name"]: p for p in responses[7]["result"]["prompts"]}
    assert set(prompts) == {"summarize", "translate", "extract_json"}
    assert {a["name"]: a["required"] for a in prompts["translate"]["arguments"]} == {"text": True, "language": True}
    (message,) = responses[8]["result"]["messages"]
    assert message["role"] == "user"
    assert message["content"]["text"].startswith("Translate the following text into Dutch.")
    assert message["content"]["text"].endswith("Good morning")
