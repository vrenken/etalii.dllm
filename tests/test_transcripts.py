"""Agent transcripts (Phase 16): an MCP tool loop recorded byte for byte, replayed offline with the recorded tool
results, and a report of the first round where an edited or different run diverges."""

from __future__ import annotations

import json

import anyio
import pytest
from test_mcp_host import ask, fixture_server

from etalii_dllm import mcp_host, transcripts
from etalii_dllm.engine import DllmEngine
from etalii_dllm.mcp_host import McpHost
from etalii_dllm.tools import ToolChoice


def record(request=None, max_rounds: int = 3) -> dict:
    request = request or ask(ToolChoice("named", "get_weather"))
    engine = DllmEngine.create_default()

    async def go() -> dict:
        async with McpHost({"fixture": fixture_server()}) as host:
            recorder = transcripts.Recorder(engine, request, host.tools, max_rounds, host.servers)
            async for event in mcp_host.chat(engine, request, host, max_rounds):
                recorder.add(event)
            return recorder.transcript()

    return anyio.run(go)


def test_a_run_gives_the_same_transcript_and_replays_offline():
    transcript = record()
    assert record() == transcript
    assert transcript["transcript"] == transcripts.FORMAT
    assert transcript["id"] == transcripts.transcript_id(transcript)
    rounds = transcript["rounds"]
    assert len(rounds) >= 2 and rounds[0]["tool_calls"][0]["name"] == "get_weather"
    result = rounds[0]["results"][0]
    assert result["server"] == "fixture" and result["content"] in ("12 C, rain", "24 C, sun")
    assert {t["name"]: t["server"] for t in transcript["tools"]}["add"] == "fixture"
    assert json.loads(json.dumps(transcript)) == transcript

    outcome = transcripts.replay(DllmEngine.create_default(), transcript)
    assert outcome.ok and outcome.diverged_at is None and outcome.reasons == () and outcome.notes == ()
    assert outcome.transcript == transcript
    assert outcome.to_json()["ok"] is True


def test_an_edited_tool_result_shows_where_the_run_diverges():
    transcript = json.loads(json.dumps(record()))
    transcript["rounds"][0]["results"][0]["content"] = "-40 C, blizzard"
    outcome = transcripts.replay(DllmEngine.create_default(), transcript)
    assert not outcome.ok and outcome.diverged_at == 1
    assert outcome.reasons[0].startswith("the transcript was edited")
    assert outcome.reasons[-1] == "round 1 diverges: its conversation differs from the transcript"

    # A consistent forgery (id recomputed) is still caught by the replay.
    transcript["id"] = transcripts.transcript_id(transcript)
    assert transcripts.replay(DllmEngine.create_default(), transcript).diverged_at == 1


def test_an_edited_answer_is_caught():
    transcript = json.loads(json.dumps(record()))
    transcript["rounds"][-1]["content"] += " (edited)"
    transcript["id"] = transcripts.transcript_id(transcript)
    reasons = transcripts.replay(DllmEngine.create_default(), transcript).reasons
    assert reasons == (f"round {len(transcript['rounds']) - 1}: the recorded text or tool calls do not match the "
                       "round's receipt",)  # fmt: skip

    forged = json.loads(json.dumps(record()))
    forged["rounds"][0]["receipt"]["output"]["tokens"] = "0" * 64
    forged["id"] = transcripts.transcript_id(forged)
    outcome = transcripts.replay(DllmEngine.create_default(), forged)
    assert outcome.diverged_at == 0
    assert outcome.reasons == ("round 0 diverges: the model's answer differs from the transcript",)


def test_round_counts_and_other_settings():
    transcript = json.loads(json.dumps(record()))
    extra = json.loads(json.dumps(transcript["rounds"][-1]))
    transcript["rounds"].append(extra)
    transcript["engine"] = "0.0.1"
    transcript["system_fingerprint"] = "fp_other"
    transcript["id"] = transcripts.transcript_id(transcript)
    outcome = transcripts.replay(DllmEngine.create_default(), transcript)
    assert outcome.notes[0].startswith("made by engine version 0.0.1")
    assert outcome.reasons[0].startswith("different weights or settings")
    assert outcome.diverged_at == len(transcript["rounds"]) - 1
    assert outcome.reasons[-1].startswith(f"the run has {len(transcript['rounds']) - 1} rounds")

    with pytest.raises(ValueError, match="dllm-agent/1"):
        transcripts.replay(DllmEngine.create_default(), {"transcript": "other"})


def test_missing_recorded_results_become_error_results():
    transcript = json.loads(json.dumps(record()))
    transcript["rounds"][0]["results"] = []
    outcome = transcripts.replay(DllmEngine.create_default(), transcript)
    assert outcome.diverged_at == 1
    assert outcome.transcript["rounds"][0]["results"][0]["content"] == "no recorded result for this call"


def test_chat_and_replay_commands(tmp_path, capsys, monkeypatch):
    from etalii_dllm.cli import main
    from etalii_dllm.engine import default_engine

    monkeypatch.delenv("DLLM_MODEL", raising=False)
    default_engine.cache_clear()
    real_host = mcp_host.McpHost
    monkeypatch.setattr(mcp_host, "McpHost", lambda servers: real_host({"fixture": fixture_server()}))
    path = tmp_path / "run.json"
    arguments = ["chat", "What is the weather in Rome?", "--max-tokens", "160", "--mcp-server", "fixture=unused"]
    assert main([*arguments, "--max-tool-rounds", "3", "--transcript", str(path)]) == 0
    transcript = json.loads(path.read_text(encoding="utf-8"))
    assert f"transcript: {path} ({transcript['id']})" in capsys.readouterr().err

    assert main(["replay", str(path)]) == 0
    out = capsys.readouterr().out
    assert f"({len(transcript['rounds'])} rounds)" in out
    assert "verified: every round gave the same output, bit for bit" in out
    assert main(["replay", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["diverged_at"] is None

    transcript["engine"] = "0.0.1"
    transcript["rounds"][-1]["content"] += "!"
    path.write_text(json.dumps(transcript), encoding="utf-8")
    assert main(["replay", str(path)]) == 1
    out = capsys.readouterr().out
    assert "note:               made by engine version 0.0.1" in out and "NOT verified" in out
    default_engine.cache_clear()
