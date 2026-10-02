"""Reproducible model building (Phase 19): exact merges, exports back to safetensors and GGUF, and distillation that
replays."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
from model_fixtures import tiny_config, write_gguf, write_hf_checkpoint

from etalii_dllm import exporting, merging, numerics
from etalii_dllm.bpe import bytes_to_unicode, from_model_header
from etalii_dllm.cli import main
from etalii_dllm.importing import import_model
from etalii_dllm.modelfile import ModelFile, TensorSource, file_sha256, lineage_problems, write_model_file

# A byte-level BPE tokenizer (as GGUF needs): every byte, a few merges, and an end-of-sequence token.
_BYTES = bytes_to_unicode()
_VOCAB = {_BYTES[b]: b for b in range(256)}
_MERGES = [f"{_BYTES[ord('h')]} {_BYTES[ord('e')]}", f"{_BYTES[ord(' ')]} {_BYTES[ord('t')]}"]
_VOCAB |= {_BYTES[ord("h")] + _BYTES[ord("e")]: 256, _BYTES[ord(" ")] + _BYTES[ord("t")]: 257}
BYTE_LEVEL_TOKENIZER = {
    "version": "1.0",
    "added_tokens": [{"id": 258, "content": "</s>", "special": True, "lstrip": False, "rstrip": False,
                      "normalized": False, "single_word": False}],
    "normalizer": None,
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
    "post_processor": None,
    "decoder": {"type": "ByteLevel"},
    "model": {"type": "BPE", "vocab": _VOCAB, "merges": _MERGES},
}  # fmt: skip


def _import(tmp_path, family="llama", name="base", tokenizer=None, vocabulary=None):
    config = tiny_config(family)
    if vocabulary:
        config["vocab_size"] = vocabulary
    write_hf_checkpoint(tmp_path / f"{name}-checkpoint", config, tokenizer_json=tokenizer)
    import_model(tmp_path / f"{name}-checkpoint", tmp_path / f"{name}.dllm")
    return tmp_path / f"{name}.dllm"


def _variant(path, out, seed):
    """The model at ``path`` with every tensor nudged by seeded noise (a stand-in for a fine-tune)."""
    model = ModelFile(path)
    tensors = {}
    for index, (name, values) in enumerate(sorted(model.tensors.items())):
        noise = numerics.fill_gaussian(seed * 1000 + index, values.size).reshape(values.shape)
        tensors[name] = (values + 0.05 * noise).astype(np.float32)
    sources = {name: TensorSource(values.shape, lambda values=values: values) for name, values in tensors.items()}
    write_model_file(out, model.config, sources, model.header)
    return out


# -- merges ---------------------------------------------------------------------------------------------------------


def test_linear_and_slerp_merges_are_exact(tmp_path):
    a = _import(tmp_path)
    b = _variant(a, tmp_path / "b.dllm", 1)
    first, second = ModelFile(a), ModelFile(b)
    assert merging.merge_models([a, a], tmp_path / "same.dllm") == first.fingerprint  # (a + a) / 2 is a, exactly

    fingerprint = merging.merge_models([a, b], tmp_path / "mean.dllm", weights=[3, 1])
    merged = ModelFile(tmp_path / "mean.dllm")
    for name, values in merged.tensors.items():
        expected = first.tensors[name].astype(np.float64) * 0.75 + second.tensors[name].astype(np.float64) * 0.25
        assert np.array_equal(values, expected.astype(np.float32))
    assert merging.merge_models([a, b], tmp_path / "again.dllm", weights=[3, 1]) == fingerprint
    assert file_sha256(tmp_path / "mean.dllm") == file_sha256(tmp_path / "again.dllm")

    assert merging.merge_models([a, b], tmp_path / "t0.dllm", "slerp", t=0.0) == first.fingerprint
    assert merging.merge_models([a, b], tmp_path / "t1.dllm", "slerp", t=1.0) == second.fingerprint
    merging.merge_models([a, b], tmp_path / "half.dllm", "slerp")
    middle = ModelFile(tmp_path / "half.dllm")
    assert middle.fingerprint not in (first.fingerprint, second.fingerprint)

    # The record: the merge section, the lineage continuing the first input's, and the licence.
    assert merged.header["merge"]["method"] == "linear" and merged.header["merge"]["weights"] == [0.75, 0.25]
    assert [item["fingerprint"] for item in merged.header["merge"]["inputs"]] == [first.fingerprint, second.fingerprint]
    assert merged.lineage[-1]["step"] == "merge" and merged.lineage[-1]["input"] == first.fingerprint
    assert merged.lineage[-1]["output"] == fingerprint and not lineage_problems(merged.lineage)
    assert "Merged with EtAlii.Dllm (linear); modified weights." in merged.licence["attribution"]
    assert "otherwise unmodified" not in merged.licence["attribution"]


def test_slerp_of_vectors():
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    half = merging.slerp(a, b, 0.5)
    assert half == pytest.approx([math.sqrt(0.5), math.sqrt(0.5), 0, 0], abs=1e-7)
    assert np.array_equal(merging.slerp(a, a * 2, 0.5), np.array([1.5, 0, 0, 0], dtype=np.float32))  # parallel
    zero = np.zeros(4, dtype=np.float32)
    assert np.array_equal(merging.slerp(zero, a, 0.25), np.array([0.25, 0, 0, 0], dtype=np.float32))


def test_ties_merges(tmp_path):
    base = _import(tmp_path)
    b, c = _variant(base, tmp_path / "b.dllm", 1), _variant(base, tmp_path / "c.dllm", 2)
    # One model, nothing trimmed: base + (b - base) is b.
    same = merging.merge_models([b, b], tmp_path / "bb.dllm", "ties", base=base, density=1.0)
    assert same == ModelFile(b).fingerprint
    merging.merge_models([b, c], tmp_path / "ties.dllm", "ties", base=base, density=0.5, weights=[1, 2])
    merged = ModelFile(tmp_path / "ties.dllm")
    assert merged.header["merge"]["base"]["fingerprint"] == ModelFile(base).fingerprint
    assert merged.header["merge"]["density"] == 0.5

    origin = np.array([0.0, 0.0, 0.0, 0.0])
    first = np.array([1.0, -2.0, 0.5, 3.0])
    second = np.array([-3.0, -1.0, 0.25, 0.0])
    result = merging.ties([first, second], origin, [1.0, 1.0], 0.5)
    # Trimmed: first keeps 3.0 and -2.0, second -3.0 and -1.0; signs: +0, -, 0, +; agreeing means.
    assert result.tolist() == [-3.0, -1.5, 0.0, 3.0]
    equal = merging._trim(np.array([1.0, -1.0, 1.0, 0.5]), 0.5)  # ties keep the earlier positions
    assert equal.tolist() == [1.0, -1.0, 0.0, 0.0]


def test_merge_errors(tmp_path):
    a = _import(tmp_path)
    b = _variant(a, tmp_path / "b.dllm", 1)
    other = _import(tmp_path, "qwen2", "other")
    for kwargs, message in [
        ({"method": "average"}, "unknown merge method"),
        ({"method": "slerp", "t": 2.0}, "0 <= t <= 1"),
        ({"method": "ties"}, "needs the --base"),
        ({"method": "ties", "base": a, "density": 0.0}, "0 < density <= 1"),
        ({"weights": [1.0]}, "1 weights for 2 models"),
        ({"weights": [1.0, -1.0]}, "add up to 0"),
    ]:
        with pytest.raises(merging.MergeError, match=message):
            merging.merge_models([a, b], tmp_path / "x.dllm", **kwargs)
    with pytest.raises(merging.MergeError, match="at least two"):
        merging.merge_models([a], tmp_path / "x.dllm")
    with pytest.raises(merging.MergeError, match="exactly two"):
        merging.merge_models([a, b, a], tmp_path / "x.dllm", "slerp")
    with pytest.raises(merging.MergeError, match="another architecture"):
        merging.merge_models([a, other], tmp_path / "x.dllm")
    retokenized = _import(tmp_path, name="retokenized", tokenizer=BYTE_LEVEL_TOKENIZER, vocabulary=264)
    config_a = ModelFile(a).header["architecture"]
    assert ModelFile(retokenized).header["architecture"] != config_a
    sources = {n: TensorSource(v.shape, lambda v=v: v) for n, v in ModelFile(a).tensors.items()}
    write_model_file(tmp_path / "t.dllm", ModelFile(a).config, sources, ModelFile(retokenized).header)
    with pytest.raises(merging.MergeError, match="another tokenizer"):
        merging.merge_models([a, tmp_path / "t.dllm"], tmp_path / "x.dllm")


def test_cli_merge(tmp_path, capsys):
    a = _import(tmp_path)
    b = _variant(a, tmp_path / "b.dllm", 1)
    out = tmp_path / "merged.dllm"
    assert main(["merge", str(a), str(b), "-o", str(out), "--method", "slerp", "--t", "0.3"]) == 0
    assert "file_sha256:" in capsys.readouterr().out
    assert main(["inspect", str(out)]) == 0
    text = capsys.readouterr().out
    assert "merged:             slerp of" in text and "lineage 1:          merge" in text
    assert main(["merge", str(a), str(b), "-o", str(out), "--weights", "1,x"]) == 2
    assert main(["merge", str(a), str(b), "-o", str(out), "--method", "ties"]) == 2


# -- exports -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["llama", "mistral", "qwen2", "qwen3", "olmo2", "granite"])
def test_safetensors_export_imports_back_to_the_same_model(tmp_path, family):
    path = _import(tmp_path, family)
    files = exporting.export_model(path, tmp_path / "hf", "safetensors")
    assert [f.name for f in files] == sorted(f.name for f in files) and (tmp_path / "hf" / "LICENSE").exists()
    again = import_model(tmp_path / "hf", tmp_path / "again.dllm")
    original = ModelFile(path)
    assert again.fingerprint == original.fingerprint and again.config == original.config
    reimported = ModelFile(tmp_path / "again.dllm")
    assert reimported.chat_template == original.chat_template
    assert reimported.licence["spdx"] == original.licence["spdx"]
    exporting.export_model(path, tmp_path / "hf2", "safetensors")
    assert [file_sha256(f) for f in files] == [file_sha256(tmp_path / "hf2" / f.name) for f in files]


def test_gguf_export_imports_back(tmp_path):
    pytest.importorskip("gguf")
    for family in ("llama", "qwen2", "qwen3"):
        path = _import(tmp_path, family, family, tokenizer=BYTE_LEVEL_TOKENIZER, vocabulary=264)
        out = tmp_path / f"{family}.gguf"
        exporting.export_model(path, out, "gguf")
        again = import_model(out, tmp_path / f"{family}-again.dllm")
        original = ModelFile(path)
        assert again.fingerprint == original.fingerprint
        assert again.config == dataclasses.replace(
            original.config, rms_norm_eps=float(np.float32(original.config.rms_norm_eps))
        )
        reimported = ModelFile(tmp_path / f"{family}-again.dllm")
        text = "hello there, the 12 tests\n</s>"
        tokens = from_model_header(original.tokenizer).encode(text)
        assert from_model_header(reimported.tokenizer).encode(text) == tokens
        assert reimported.chat_template == original.chat_template
        # A GGUF import exports again to the same bytes, and to safetensors.
        exporting.export_model(tmp_path / f"{family}-again.dllm", tmp_path / f"{family}-2.gguf", "gguf")
        assert import_model(tmp_path / f"{family}-2.gguf", tmp_path / "x.dllm").fingerprint == original.fingerprint
    import gguf

    reader = gguf.GGUFReader(str(tmp_path / "llama.gguf"))
    assert reader.fields["general.architecture"].contents() == "llama"
    assert reader.fields["tokenizer.ggml.pre"].contents() == "gpt-2"
    assert len(reader.fields["tokenizer.ggml.tokens"].contents()) == 264  # padded to the embedding


def test_gguf_imports_export_to_safetensors(tmp_path):
    pytest.importorskip("gguf")
    write_gguf(tmp_path / "tiny.gguf")
    import_model(tmp_path / "tiny.gguf", tmp_path / "tiny.dllm")
    exporting.export_model(tmp_path / "tiny.dllm", tmp_path / "hf", "safetensors")
    assert json.loads((tmp_path / "hf" / "tokenizer_config.json").read_text())["eos_token"]
    back = import_model(tmp_path / "hf", tmp_path / "back.dllm")
    assert back.fingerprint == ModelFile(tmp_path / "tiny.dllm").fingerprint


def test_export_errors(tmp_path, capsys):
    path = _import(tmp_path)
    for family in ("phi3",):
        with pytest.raises(exporting.ExportError, match="not supported"):
            exporting.export_model(_import(tmp_path, family, family), tmp_path / "x", "safetensors")
    config = ModelFile(path).config
    with pytest.raises(exporting.ExportError, match="cannot be described exactly"):
        exporting.hf_config_json(dataclasses.replace(config, qk_norm=True))
    with pytest.raises(exporting.ExportError, match="cannot be described"):
        exporting.hf_config_json(dataclasses.replace(config, rope_scaling={"rope_type": "dynamic", "factor": 2.0}))
    with pytest.raises(exporting.ExportError, match="unknown export format"):
        exporting.export_model(path, tmp_path / "x", "onnx")
    with pytest.raises(exporting.ExportError, match="only byte-level BPE"):
        exporting.export_model(path, tmp_path / "x.gguf", "gguf")
    with pytest.raises(exporting.ExportError, match="to GGUF is not supported"):
        exporting.export_model(_import(tmp_path, "mistral", "mistral"), tmp_path / "x.gguf", "gguf")
    byte_level = _import(tmp_path, name="bytes", tokenizer=BYTE_LEVEL_TOKENIZER, vocabulary=264)
    model = ModelFile(byte_level)
    for change, message in [
        ({"sliding_window": 3}, "sliding windows"),
        ({"rope_scaling": {"rope_type": "llama3", "factor": 8.0}}, "RoPE scaling 'llama3'"),
        ({"qk_norm": True}, "normalisation"),
    ]:
        model.config = dataclasses.replace(model.config, **change)
        with pytest.raises(exporting.ExportError, match=message):
            exporting.export_gguf(model, tmp_path / "x.gguf")
        model.config = ModelFile(byte_level).config
    model.header = {**model.header, "tokenizer": None}
    with pytest.raises(exporting.ExportError, match="no tokenizer"):
        exporting.export_gguf(model, tmp_path / "x.gguf")
    prefix = {**BYTE_LEVEL_TOKENIZER["pre_tokenizer"], "add_prefix_space": True}
    odd = _import(tmp_path, name="odd", tokenizer={**BYTE_LEVEL_TOKENIZER, "pre_tokenizer": prefix}, vocabulary=264)
    with pytest.raises(exporting.ExportError, match="no GGUF equivalent"):
        exporting.export_model(odd, tmp_path / "x.gguf", "gguf")

    assert main(["export", str(byte_level), "--format", "gguf", "-o", str(tmp_path / "cli.gguf")]) == 0
    assert "cli.gguf" in capsys.readouterr().out
    assert main(["export", str(path), "--format", "gguf", "-o", str(tmp_path / "cli.gguf")]) == 2


# -- distillation --------------------------------------------------------------------------------------------------


def test_distillation_replays(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DLLM_MODEL", raising=False)
    teacher = _import(tmp_path, name="teacher", tokenizer=BYTE_LEVEL_TOKENIZER, vocabulary=264)
    student = _variant(teacher, tmp_path / "student.dllm", 3)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text('ab\n\n{"messages": [{"role": "user", "content": "ba"}]}\n', encoding="utf-8")
    out, receipt = tmp_path / "distilled.dllm", tmp_path / "distilled.train.json"
    args = ["distill", str(student), "--teacher", str(teacher), "--prompts", str(prompts), "-o", str(out)]
    args += ["--steps", "2", "--batch-size", "1", "--sequence-length", "8", "--teacher-max-tokens", "4"]
    assert main([*args, "--receipt", str(receipt)]) == 0
    assert "teacher:            2 answers" in capsys.readouterr().out
    data = (tmp_path / "distilled.dllm.distill.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["messages"][-1]["role"] for line in data] == ["assistant", "assistant"]
    model = ModelFile(out)
    distillation = model.fine_tuning["distillation"]
    assert distillation["teacher"] == ModelFile(teacher).fingerprint and len(distillation["receipts"]) == 2
    assert model.lineage[-1]["teacher"] == ModelFile(teacher).fingerprint

    assert main(["replay", str(receipt), "--base", str(student), "--teacher", str(teacher)]) == 0
    assert "verified" in capsys.readouterr().out
    assert main(["replay", str(receipt), "--base", str(student)]) == 0
    assert "not regenerated" in capsys.readouterr().out
    assert main(["replay", str(receipt), "--base", str(student), "--teacher", str(student), "--json"]) == 1
    reasons = json.loads(capsys.readouterr().out)["reasons"]
    assert any("a different teacher" in r for r in reasons) and any("answers differ" in r for r in reasons)
    other = tmp_path / "other.txt"
    other.write_text("ab\nba\n", encoding="utf-8")
    replay = ["replay", str(receipt), "--base", str(student), "--teacher", str(teacher)]
    assert main([*replay, "--prompts", str(other)]) == 1
    assert "different prompts" in capsys.readouterr().out

    assert main(["distill", str(student), "-o", str(out)]) == 1
    assert main(["finetune", str(student), "-o", str(out)]) == 1
    empty = tmp_path / "empty.txt"
    empty.write_text("\n", encoding="utf-8")
    assert main([*args[:5], str(empty), "-o", str(out)]) == 1
    bad = tmp_path / "bad.txt"
    bad.write_text('{"text": "x"}\n', encoding="utf-8")
    assert main([*args[:5], str(bad), "-o", str(out)]) == 1
