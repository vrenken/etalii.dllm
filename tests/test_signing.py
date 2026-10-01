"""Signatures (Phase 17): deterministic Ed25519 signatures on receipts, chains, transcripts and model files, checked
by ``dllm replay --trust`` and ``dllm inspect --trust``."""

from __future__ import annotations

import json

import pytest

from etalii_dllm import engine as engine_module
from etalii_dllm import signing, transcripts
from etalii_dllm.chat import ChatMessage
from etalii_dllm.cli import main
from etalii_dllm.engine import ChatRequest, DllmEngine, default_engine, use_model_file
from etalii_dllm.receipts import receipt_id

pytest.importorskip("cryptography")


@pytest.fixture
def keys(tmp_path):
    first, second = tmp_path / "first.key", tmp_path / "second.key"
    return signing.generate_key(first), first, signing.generate_key(second), second


@pytest.fixture
def clean(monkeypatch):
    for name in ("MODEL_ENVIRONMENT_VARIABLE", "SIGN_KEY_ENVIRONMENT_VARIABLE"):
        monkeypatch.delenv(getattr(engine_module, name), raising=False)
    default_engine.cache_clear()
    yield monkeypatch
    monkeypatch.delenv(engine_module.SIGN_KEY_ENVIRONMENT_VARIABLE, raising=False)
    default_engine.cache_clear()


REQUEST = ChatRequest([ChatMessage("user", "Hi")], 6)


def test_signatures_are_deterministic_and_leave_the_id_alone(keys):
    public, path, other, _ = keys
    signer = signing.Signer.load(path)
    assert signer.public_key == public == (path.parent / "first.key.pub").read_text().strip()
    receipt = DllmEngine.create_default().chat_completion(REQUEST).receipt
    signed = signer.sign(receipt)
    assert signer.sign(receipt) == signed and signer.sign(signed) == signed  # Ed25519 is deterministic
    assert receipt_id(signed) == signed["id"] == receipt["id"]
    assert signing.signature_problem(signed, [public]) is None
    assert "not a trusted key" in signing.signature_problem(signed, [other])
    assert signing.signature_problem(receipt, [public]) == "it is not signed"
    assert "does not match" in signing.signature_problem({**signed, "model": "other"}, [public])
    odd = {**signed, "signature": {**signed["signature"], "algorithm": "rsa"}}
    assert "unknown signature algorithm" in signing.signature_problem(odd, [public])
    broken = {**signed, "signature": {**signed["signature"], "value": "zz"}}
    assert "does not match" in signing.signature_problem(broken, [public])


def test_keys(tmp_path, keys):
    public, path, _, _ = keys
    assert signing.read_public_key(str(path) + ".pub") == public == signing.read_public_key(public.upper())
    with pytest.raises(signing.SigningError, match="not an Ed25519 public key"):
        signing.read_public_key("nope")
    (tmp_path / "bad.key").write_text("not a key")
    with pytest.raises(signing.SigningError, match="not a private key"):
        signing.Signer.load(tmp_path / "bad.key")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    other = ec.generate_private_key(ec.SECP256R1()).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    (tmp_path / "ec.key").write_bytes(other)
    with pytest.raises(signing.SigningError, match="not an Ed25519 key"):
        signing.Signer.load(tmp_path / "ec.key")


def test_engines_sign_receipts_and_transcripts(keys, clean):
    public, path, _, _ = keys
    use_model_file(None, sign_key=path)
    engine = default_engine()
    receipt = engine.chat_completion(REQUEST).receipt
    assert signing.signature_problem(receipt, [public]) is None
    recorder = transcripts.Recorder(engine, REQUEST, [], 1)
    transcript = recorder.transcript()
    assert transcripts.transcript_id(transcript) == transcript["id"]
    assert signing.signature_problem(transcript, [public]) is None


def test_cli_sign_and_trust(tmp_path, keys, clean, capsys):
    public, path, other, other_path = keys
    receipt_path = tmp_path / "receipt.json"
    assert main(["--sign-key", str(path), "chat", "Hi", "--max-tokens", "6", "--receipt", str(receipt_path)]) == 0
    capsys.readouterr()
    assert main(["replay", str(receipt_path), "--trust", public]) == 0
    assert main(["replay", str(receipt_path), "--trust", other]) == 1
    assert "not a trusted key" in capsys.readouterr().err
    assert main(["replay", str(receipt_path), "--trust", "nope"]) == 2

    # Re-signing with another key replaces the signature; a chain gets one per receipt.
    assert main(["sign", str(receipt_path), "--key", str(other_path), "-o", str(tmp_path / "resigned.json")]) == 0
    assert main(["replay", str(tmp_path / "resigned.json"), "--trust", str(other_path) + ".pub", "--json"]) == 0
    chain = tmp_path / "chain.json"
    chain.write_text(json.dumps([json.loads(receipt_path.read_text())]))
    assert main(["replay", str(chain), "--trust", other]) == 1
    assert "turn 0: it is signed by" in capsys.readouterr().err
    assert main(["sign", str(chain), "--key", str(other_path)]) == 0
    assert main(["replay", str(chain), "--trust", other]) == 0
    capsys.readouterr()

    # Model files get a detached signature over their SHA-256.
    from model_fixtures import write_hf_checkpoint

    from etalii_dllm.importing import import_model

    write_hf_checkpoint(tmp_path / "tiny")
    model = tmp_path / "tiny.dllm"
    import_model(tmp_path / "tiny", model)
    assert main(["inspect", str(model), "--trust", public]) == 1
    assert "it is not signed" in capsys.readouterr().out
    assert main(["sign", str(model), "--key", str(path)]) == 0
    assert main(["inspect", str(model), "--trust", public]) == 0
    assert "valid, by a trusted key" in capsys.readouterr().out
    assert main(["inspect", str(model), "--trust", other]) == 1
    assert main(["inspect", str(model), "--trust", "nope"]) == 1
    signature = json.loads((tmp_path / "tiny.dllm.sig").read_text())
    (tmp_path / "tiny.dllm.sig").write_text(json.dumps({**signature, "file_sha256": "0" * 64}))
    assert "for another file" in signing.model_signature_problem(model, [public])
    (tmp_path / "tiny.dllm.sig").write_text("{")
    assert signing.model_signature_problem(model, [public]) == "tiny.dllm.sig is not a signature file"

    assert main(["sign", "--keygen", str(tmp_path / "third.key")]) == 0
    assert (tmp_path / "third.key.pub").exists()
    assert main(["sign", str(receipt_path)]) == 2
    (tmp_path / "scalar.json").write_text("3")
    assert main(["sign", str(tmp_path / "scalar.json"), "--key", str(path)]) == 2
