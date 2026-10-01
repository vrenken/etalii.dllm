"""Built-in deterministic tools (Phase 16): exact arithmetic, read-only files sorted by name, document search, served
over MCP so an agent run with them repeats bit for bit and replays offline."""

from __future__ import annotations

import json
from fractions import Fraction
from types import SimpleNamespace

import anyio
import pytest

from etalii_dllm import builtin_tools
from etalii_dllm.builtin_tools import Files, ToolError, calculate, decimal, evaluate
from etalii_dllm.chat import ToolCall
from etalii_dllm.cli import main
from etalii_dllm.mcp_host import McpHost
from etalii_dllm.retrieval import Chunk, Hit


def call(specs, name: str, arguments: dict, engine=None):
    async def go():
        async with McpHost({"tools": builtin_tools.server(specs, engine)}) as host:
            return [t.name for t in host.tools], await host.call(ToolCall("c1", name, json.dumps(arguments)))

    return anyio.run(go)


def test_calculator_is_exact():
    assert evaluate("0.1 + 0.2") == Fraction(3, 10)
    assert evaluate("2 ** -3") == Fraction(1, 8)
    assert evaluate("-7 % 3") == 2 and evaluate("7 // 2") == 3 and evaluate("+(1.5 + 2.25) * 4") == 15
    assert evaluate("1e3") == 1000
    assert calculate("1/7 * 7") == "1"
    assert calculate("0.1+0.2") == "3/10 = 0.3"
    assert calculate("1/3") == "1/3 ≈ 0.333333333333333333333333333333"
    assert calculate("-2/3") == "-2/3 ≈ -0.666666666666666666666666666666"
    assert decimal(Fraction(1, 8), 2) == "≈ 0.12"


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("1/0", "division by zero"),
        ("5 % 0", "division by zero"),
        ("0 ** -1", "division by zero"),
        ("2 ** 0.5", "integer exponents"),
        ("2 ** 2000", "limited"),
        ("10 ** 1000 * 10 ** 1000 * 10 ** 1000 * 10 ** 1000 * 10 ** 1000", "too large"),
        ("99999 ** 1024", "too large"),
        ("__import__('os')", "unsupported"),
        ("True + 1", "unsupported"),
        ("1 +", "not an arithmetic expression"),
        ("1" * 1001, "longer than"),
    ],
)
def test_calculator_errors(expression, message):
    with pytest.raises(ToolError, match=message):
        evaluate(expression)


def test_files_are_sorted_and_confined(tmp_path):
    (tmp_path / "b.txt").write_text("one\ntwo\nthree\n")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "deep.md").write_text("x")
    (tmp_path / "empty").mkdir()
    files = Files(tmp_path)
    assert files.list_files() == "a/\nb.txt  (14 bytes)\nempty/"
    assert files.list_files("a") == "a/deep.md  (1 bytes)"
    assert files.list_files("empty") == "(empty)"
    assert files.read_file("b.txt") == "b.txt: lines 1-3 of 3\none\ntwo\nthree"
    assert files.read_file("b.txt", 2, 1) == "b.txt: lines 2-2 of 3\ntwo"
    assert files.read_file("b.txt", 9) == "b.txt: no lines from 9 (the file has 3)"
    for bad, message in [("../x", "outside"), ("b.txt/..", None), ("missing", "not a file")]:
        with pytest.raises(ToolError, match=message):
            files.read_file(bad)
    with pytest.raises(ToolError, match="not a directory"):
        files.list_files("b.txt")
    with pytest.raises(ToolError, match="at least 1"):
        files.read_file("b.txt", 0)
    with pytest.raises(ToolError, match="not a directory"):
        Files(tmp_path / "b.txt")


def test_parse():
    assert builtin_tools.parse(["calculator", "files=docs"]) == {"calculator": None, "files": "docs"}
    with pytest.raises(ToolError, match="unknown built-in tool"):
        builtin_tools.parse(["shell"])
    with pytest.raises(ToolError, match="needs a directory"):
        builtin_tools.parse(["files"])


def test_the_server_offers_the_chosen_tools(tmp_path):
    (tmp_path / "note.txt").write_text("hello")
    names, result = call(["calculator", f"files={tmp_path}"], "calculate", {"expression": "6 * 7"})
    assert names == ["calculate", "list_files", "read_file"]
    assert (result.content, result.is_error) == ("42", False)
    _, listed = call([f"files={tmp_path}"], "list_files", {})
    assert listed.content == "note.txt  (5 bytes)"
    _, failed = call(["calculator"], "calculate", {"expression": "1/0"})
    assert failed.is_error and "division by zero" in failed.content


def test_search_documents():
    chunk = Chunk("cities.txt", 0, 31, "Paris is the capital of France.")
    retriever = SimpleNamespace(search=lambda query, top: [Hit(1, 0.5, chunk, 0)][:top])
    names, result = call(["documents"], "search_documents", {"query": "France"}, SimpleNamespace(retriever=retriever))
    assert names == ["search_documents"]
    assert json.loads(result.content) == [{"rank": 1, "score": 0.5, **chunk.to_json()}]
    _, missing = call(["documents"], "search_documents", {"query": "x"}, SimpleNamespace(retriever=None))
    assert missing.is_error and "no document index" in missing.content


def test_chat_with_built_in_tools_and_replay(tmp_path, capsys):
    transcript = tmp_path / "run.json"
    arguments = ["chat", "What is 12.5 times 8?", "--max-tokens", "24", "--tool", "calculator"]
    assert main([*arguments, "--max-tool-rounds", "2", "--transcript", str(transcript)]) == 0
    first = capsys.readouterr()
    assert "tools: calculate" in first.err
    recorded = json.loads(transcript.read_text())
    assert recorded["tools"][0]["server"] == "tools"
    assert main(["replay", str(transcript)]) == 0
    assert "verified: every round" in capsys.readouterr().out
    assert main(["chat", "Hi", "--tool", "shell"]) == 1
    assert "unknown built-in tool" in capsys.readouterr().err
    assert main(["chat", "Hi", "--tool", "calculator", "--mcp-server", "tools=python -V"]) == 1
    assert "must be unique" in capsys.readouterr().err


def test_stdio_entry_point(monkeypatch):
    served = []
    monkeypatch.setattr(builtin_tools.MCPServer, "run", lambda self, transport: served.append((self, transport)))
    builtin_tools.main([])
    builtin_tools.main(["--tool", "calculator", "--tool", "documents"])
    assert [transport for _, transport in served] == ["stdio", "stdio"]
    with pytest.raises(SystemExit):
        builtin_tools.main(["--tool", "files"])
