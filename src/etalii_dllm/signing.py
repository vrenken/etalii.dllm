"""Signatures: who vouches for a receipt, a transcript or a model file.

A receipt proves that the weights give the answer; it does not say who ran the request. An Ed25519 signature does,
and Ed25519 signatures are deterministic (RFC 8032): the same key signing the same content gives the same bytes, so
a signed receipt is as reproducible as an unsigned one.

- JSON documents (generation receipts, training receipts, agent transcripts) carry the signature inline::

      "signature": {"algorithm": "ed25519", "key": "<public key, 64 hex digits>", "value": "<128 hex digits>"}

  over the canonical JSON of the document without its ``signature``. The documents' ids leave the signature out, so
  signing a receipt does not change its id.
- A model file gets a detached ``<file>.sig`` (JSON ``{"file_sha256", "signature"}``), the signature being over the
  file's SHA-256 (:func:`etalii_dllm.modelfile.file_sha256`).

Keys: :func:`generate_key` writes a private key file (PKCS #8 PEM) and ``<file>.pub`` with the public key in hex.
Signing needs the ``cryptography`` package (``pip install "etalii-dllm[sign]"``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALGORITHM = "ed25519"


class SigningError(ValueError):
    """A key or signature that cannot be used."""


def _crypto() -> Any:
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError:  # pragma: no cover - the dev extra installs it
        raise SigningError('signing needs the cryptography package: pip install "etalii-dllm[sign]"') from None
    return ed25519


def _canonical(document: Mapping[str, Any]) -> bytes:
    body = {k: v for k, v in document.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class Signer:
    """An Ed25519 private key."""

    key: Any

    @staticmethod
    def load(path: str | Path) -> Signer:
        from cryptography.hazmat.primitives import serialization

        ed25519 = _crypto()
        try:
            key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
        except (ValueError, TypeError) as error:
            raise SigningError(f"{path}: not a private key ({error})") from None
        if not isinstance(key, ed25519.Ed25519PrivateKey):
            raise SigningError(f"{path}: not an Ed25519 key")
        return Signer(key)

    @property
    def public_key(self) -> str:
        from cryptography.hazmat.primitives import serialization

        raw = self.key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return raw.hex()

    def signature(self, content: bytes) -> dict[str, str]:
        return {"algorithm": ALGORITHM, "key": self.public_key, "value": self.key.sign(content).hex()}

    def sign(self, document: Mapping[str, Any]) -> dict[str, Any]:
        """``document`` with a ``signature`` over everything else in it."""
        return {**document, "signature": self.signature(_canonical(document))}

    def sign_model(self, path: str | Path) -> dict[str, Any]:
        """The detached signature of a model file (write it to ``<file>.sig``)."""
        from etalii_dllm.modelfile import file_sha256

        digest = file_sha256(path)
        return {"file_sha256": digest, "signature": self.signature(digest.encode())}


def generate_key(path: str | Path) -> str:
    """Writes a new private key to ``path`` and its public key to ``<path>.pub``; returns the public key. This is
    the one place that uses the operating system's entropy: a key must be secret."""
    from cryptography.hazmat.primitives import serialization

    ed25519 = _crypto()
    key = ed25519.Ed25519PrivateKey.generate()
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    path = Path(path)
    path.write_bytes(pem)
    public = Signer(key).public_key
    Path(f"{path}.pub").write_text(public + "\n", encoding="utf-8")
    return public


def read_public_key(value: str) -> str:
    """A public key given as 64 hex digits or as a ``.pub`` file holding them."""
    text = value.strip()
    if not _is_key(text) and Path(value).is_file():
        text = Path(value).read_text(encoding="utf-8").strip()
    if not _is_key(text):
        raise SigningError(f"{value}: not an Ed25519 public key (64 hex digits, or a .pub file)")
    return text.lower()


def _is_key(text: str) -> bool:
    return len(text) == 64 and all(c in "0123456789abcdefABCDEF" for c in text)


def _check(signature: Any, content: bytes, trusted: Iterable[str]) -> str | None:
    if not isinstance(signature, Mapping):
        return "it is not signed"
    if signature.get("algorithm") != ALGORITHM:
        return f"unknown signature algorithm {signature.get('algorithm')!r}"
    key = str(signature.get("key", "")).lower()
    trusted = list(trusted)
    if key not in trusted:
        return f"it is signed by {key}, which is not a trusted key"
    ed25519 = _crypto()
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(key)).verify(
            bytes.fromhex(str(signature.get("value", ""))), content
        )
    except Exception:  # InvalidSignature, or a malformed key or value
        return "the signature does not match the content"
    return None


def signature_problem(document: Mapping[str, Any], trusted: Iterable[str]) -> str | None:
    """Why ``document``'s signature is not a valid one by a ``trusted`` key, else ``None``."""
    return _check(document.get("signature"), _canonical(document), trusted)


def model_signature_problem(path: str | Path, trusted: Iterable[str]) -> str | None:
    """Why the model file ``path`` lacks a valid detached signature (``<path>.sig``) by a ``trusted`` key."""
    from etalii_dllm.modelfile import file_sha256

    sig = Path(f"{path}.sig")
    if not sig.is_file():
        return f"it is not signed (no {sig.name})"
    try:
        detached = json.loads(sig.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return f"{sig.name} is not a signature file"
    digest = file_sha256(path)
    if detached.get("file_sha256") != digest:
        return f"{sig.name} is for another file (the file was changed)"
    return _check(detached.get("signature"), digest.encode(), trusted)
