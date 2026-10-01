"""Conformance vectors (issue #163): written by this build, passed by both implementations, the same everywhere."""

from __future__ import annotations

import json

import numpy as np
import pytest
from golden_values import CONFORMANCE_MANIFEST_SHA256

from etalii_dllm import conformance
from etalii_dllm.cli import main


@pytest.fixture(scope="module")
def vectors(tmp_path_factory):
    directory = tmp_path_factory.mktemp("conformance")
    count, digest = conformance.write(directory)
    return directory, count, digest


def test_the_vectors_are_the_golden_ones(vectors):
    directory, count, digest = vectors
    assert digest == CONFORMANCE_MANIFEST_SHA256
    manifest = json.loads((directory / "manifest.json").read_bytes())
    assert manifest["format"] == conformance.FORMAT and len(manifest["cases"]) == count
    kernels = {case["kernel"] for case in manifest["cases"]}
    assert {"exp", "linear", "quantize", "attention", "rope", "random", "sample", "decoder"} <= kernels


@pytest.mark.parametrize("implementation", conformance.IMPLEMENTATIONS)
def test_both_implementations_pass(vectors, implementation):
    directory, count, digest = vectors
    result = conformance.check(directory, implementation)
    assert result.ok, result.failed
    assert len(result.passed) == count and result.manifest_sha256 == digest


def test_a_wrong_output_fails_its_case(vectors, tmp_path):
    directory, _, _ = vectors
    manifest = json.loads((directory / "manifest.json").read_bytes())
    case = next(c for c in manifest["cases"] if c["name"] == "softmax")
    entry = case["outputs"]["y"]
    values = np.frombuffer((directory / entry["file"]).read_bytes(), dtype=entry["dtype"]).copy()
    values.view(np.uint32)[7] ^= 1
    copy = tmp_path / "copy"
    for path in directory.rglob("*"):
        if path.is_file():
            target = copy / path.relative_to(directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (copy / entry["file"]).write_bytes(values.tobytes())
    result = conformance.check(copy)
    assert "SHA-256" in result.failed["softmax"]  # a changed file no longer matches the manifest
    import hashlib

    entry["sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
    (copy / "manifest.json").write_text(json.dumps(manifest))
    result = conformance.check(copy, "reference")
    assert result.failed == {"softmax": "different bits in y"}


def test_file_names_are_portable(tmp_path):
    with pytest.raises(ValueError, match="not a portable file name"):
        conformance._save(tmp_path, "case", "tensor:x", np.zeros(1, np.float32))


def test_nans_match_any_nan_and_shapes_must_agree():
    assert conformance.same_bits(np.array([np.nan, 1.0]), np.array([-np.nan, 1.0]))
    assert not conformance.same_bits(np.array([0.0]), np.array([-0.0]))
    assert not conformance.same_bits(np.zeros(2, np.float32), np.zeros(3, np.float32))
    assert not conformance.same_bits(np.zeros(2, np.float32), np.zeros(2, np.float64))
    assert conformance.same_bits(np.array([1, 2], np.int64), np.array([1, 2], np.int32))


def test_conformance_on_the_command_line(tmp_path, capsys):
    assert main(["conformance", "write", str(tmp_path / "v")]) == 0
    assert CONFORMANCE_MANIFEST_SHA256 in capsys.readouterr().out
    assert main(["conformance", "check", str(tmp_path / "v"), "--implementation", "reference"]) == 0
    assert "0 failed (reference)" in capsys.readouterr().out
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "manifest.json").write_text('{"format": "other"}')
    assert main(["conformance", "check", str(tmp_path / "bad")]) == 2
    assert "not etalii-dllm-conformance" in capsys.readouterr().err
    with pytest.raises(ValueError, match="unknown implementation"):
        conformance.check(tmp_path / "v", "other")
    with pytest.raises(ValueError, match="unknown kernel"):
        conformance.RUNNERS["kernels"]("nope", {}, {})
    with pytest.raises(ValueError, match="unknown kernel"):
        conformance.RUNNERS["reference"]("nope", {}, {})
