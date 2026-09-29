"""``dllm verify`` (issue #100): a fingerprint that is the same on every machine."""

from __future__ import annotations

import json

import pytest
from golden_values import VERIFY_FINGERPRINT

from etalii_dllm import _kernels, verify
from etalii_dllm import engine as engine_module
from etalii_dllm.cli import main as cli
from etalii_dllm.engine import DllmEngine


def test_release_reference_is_current():
    assert verify.kernels_fingerprint() == verify.REFERENCE["kernels"]
    assert verify.unicode_fingerprint() == verify.REFERENCE["unicode"]


@pytest.mark.parametrize("isa", _kernels.supported_isas())
def test_placeholder_model_report_is_golden_on_every_simd_path(isa):
    try:
        _kernels.set_isa(isa)
        report = verify.run(DllmEngine.create_default())
    finally:
        _kernels.set_isa("best")
    assert report.mismatches == []
    assert report.fingerprint == VERIFY_FINGERPRINT


def test_cli(capsys, monkeypatch):
    monkeypatch.setenv(engine_module.MODEL_ENVIRONMENT_VARIABLE, "")
    assert cli(["verify", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verify"] == VERIFY_FINGERPRINT
    assert sorted(report["parts"]) == ["greedy", "kernels", "logits", "sampled", "tokenizer", "unicode"]
    assert report["environment"]["unicode tables"].startswith("15.1.0")
    assert cli(["verify"]) == 0
    assert f"verify:             {VERIFY_FINGERPRINT}" in capsys.readouterr().out
