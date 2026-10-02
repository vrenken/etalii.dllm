"""The ``dllm`` command line: error exits (non-zero status and a ``dllm <command>:`` message on stderr), JSON schema
sources, LoRA fine-tuning to an adapter and back, and how MCP tool rounds are reported."""

from __future__ import annotations

import json

import pytest
from model_fixtures import TINY_LLAMA_CONFIG, write_hf_checkpoint
from test_engine_import import model_path  # noqa: F401 - fixture

from etalii_dllm import cuda, engine, mcp_host
from etalii_dllm.chat import ToolCall
from etalii_dllm.cli import main
from etalii_dllm.engine import Finished, TextDelta, ToolCallEvent, default_engine
from etalii_dllm.importing import import_model
from etalii_dllm.tools import Tool

RUNTIME_VARIABLES = tuple(
    getattr(engine, name) for name in sorted(vars(engine)) if name.endswith("_ENVIRONMENT_VARIABLE")
)
"""Every runtime option variable (``main`` sets some of them, other tests may leave them set)."""


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """``main`` writes the runtime options to the environment; register every variable so monkeypatch restores it
    (setting then deleting records the original state, even when the variable was unset)."""
    for name in RUNTIME_VARIABLES:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    default_engine.cache_clear()
    yield
    default_engine.cache_clear()


def test_inspect_reports_unreadable_files(tmp_path, capsys):
    assert main(["inspect", str(tmp_path / "missing.dllm")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("dllm inspect: ")
    (tmp_path / "junk.dllm").write_bytes(b"not a model file")
    assert main(["inspect", str(tmp_path / "junk.dllm")]) == 1
    assert capsys.readouterr().err.startswith("dllm inspect: ")


def test_chat_json_schema_from_a_file_or_inline(tmp_path, capsys):
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    arguments = ["chat", "Answer.", "--max-tokens", "32"]
    assert main([*arguments, "--json-schema", str(path)]) == 0
    from_file = capsys.readouterr()
    assert main([*arguments, "--json-schema", json.dumps(schema)]) == 0
    inline = capsys.readouterr()
    assert from_file == inline
    assert isinstance(json.loads(from_file.out)["ok"], bool)
    assert "fingerprint: " in from_file.err


@pytest.mark.parametrize("source", ["missing.json", "{not json"])
def test_chat_json_schema_errors(tmp_path, capsys, source):
    value = str(tmp_path / source) if source.endswith(".json") else source
    assert main(["chat", "Hi", "--json-schema", value]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("dllm chat: --json-schema: ")


def test_chat_reports_unsupported_schemas(capsys):
    assert main(["chat", "Hi", "--json-schema", '{"type": "string", "uniqueItems": true}']) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("dllm chat: ") and "--json-schema" not in captured.err


@pytest.mark.skipif(cuda.available(), reason="needs a machine without a usable CUDA GPU")
def test_cuda_device_without_a_gpu(model_path, capsys):  # noqa: F811
    assert main(["--model", str(model_path), "--device", "cuda", "info"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("dllm: --device cuda: ")


def test_lora_finetune_adapter_round_trip(tmp_path, capsys):
    """A base without a chat template trains on plain text; its LoRA adapter imports back onto the base and
    ``inspect`` reports it."""
    from test_bpe import smollm2_style

    pytest.importorskip("tokenizers")
    reference = smollm2_style()
    config = {**TINY_LLAMA_CONFIG, "vocab_size": reference.get_vocab_size(), "eos_token_id": 2}
    write_hf_checkpoint(tmp_path / "checkpoint", config, tokenizer_json=json.loads(reference.to_str()))
    tokenizer_config = {"eos_token": "<|im_end|>", "bos_token": None}  # and no chat template
    (tmp_path / "checkpoint" / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    import_model(tmp_path / "checkpoint", tmp_path / "base.dllm")
    (tmp_path / "data.txt").write_text("the quick brown fox jumps over the lazy dog\n" * 4, encoding="utf-8")
    arguments = ["finetune", str(tmp_path / "base.dllm"), "--data", str(tmp_path / "data.txt"), "--steps", "2"]
    arguments += ["--batch-size", "2", "--sequence-length", "8", "--lora-rank", "2", "--lora-targets", "q,v"]
    assert main([*arguments, "--adapter-output", str(tmp_path / "adapter")]) == 0
    out = capsys.readouterr().out
    assert "step     2/2" in out and f"adapter:            {tmp_path / 'adapter'}" in out
    assert (tmp_path / "adapter" / "adapter_model.safetensors").is_file()

    (tmp_path / "adapter" / "README.md").write_text("---\nlicense: mit\n---\n", encoding="utf-8")
    merged = tmp_path / "merged.dllm"
    assert main(["import", str(tmp_path / "adapter"), "-o", str(merged), "--base", str(tmp_path / "base.dllm")]) == 0
    capsys.readouterr()
    assert main(["inspect", str(merged), "--no-verify"]) == 0
    lines = capsys.readouterr().out.splitlines()
    (adapter,) = [line for line in lines if line.startswith("adapter:")]
    assert "LoRA rank 2 alpha 2.0 on q,v, merged into " in adapter
    assert "chat template:      no" in lines


def test_finetune_needs_an_output(tmp_path, capsys):
    assert main(["finetune", str(tmp_path / "base.dllm"), "--data", str(tmp_path / "data.txt")]) == 1
    assert "pass -o/--output, --adapter-output, or both" in capsys.readouterr().err


def test_chat_with_mcp_reports_tool_rounds(monkeypatch, capsys):
    """Tool calls and their results (errors marked) go to stderr; only the answer goes to stdout."""
    weather = ToolCall("call_1", "get_weather", '{"city": "Rome"}')
    events = [
        TextDelta("Let me check. "),
        ToolCallEvent(0, weather),
        mcp_host.ToolResult(weather, "fixture", "24 C, sun", False),
        ToolCallEvent(1, ToolCall("call_2", "fail", "{}")),
        mcp_host.ToolResult(ToolCall("call_2", "fail", "{}"), "fixture", "out of order", True),
        TextDelta("Sunny."),
        Finished("stop", None, 5, "abc"),
    ]

    class Host:
        def __init__(self, servers):
            self.tools = [Tool("get_weather", "", {}), Tool("fail", "", {})]
            seen.append([server.name for server in servers])

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    async def chat(engine, request, host, max_rounds):
        seen.append(max_rounds)
        for event in events:
            yield event

    seen: list = []
    monkeypatch.setattr(mcp_host, "McpHost", Host)
    monkeypatch.setattr(mcp_host, "chat", chat)
    assert main(["chat", "Weather?", "--mcp-server", "fixture=python -V", "--max-tool-rounds", "3"]) == 0
    captured = capsys.readouterr()
    assert seen == [["fixture"], 3]
    assert captured.out == "Let me check. Sunny.\n"
    assert captured.err.splitlines() == [
        "tools: get_weather, fail",
        "",
        '-> get_weather({"city": "Rome"})',
        "<- result: 24 C, sun",
        "",
        "-> fail({})",
        "<- error: out of order",
        "fingerprint: abc  tokens: 5  finish: stop",
    ]
