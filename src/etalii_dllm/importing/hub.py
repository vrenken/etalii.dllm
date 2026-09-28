"""Downloads a model snapshot from the Hugging Face Hub, pinned to a commit.

Only the files an import needs are fetched. The revision (branch, tag or commit) is resolved to a commit hash
first and every file is fetched from that commit, so the import records exactly what it converted. Uses only the
standard library; ``HF_TOKEN`` is sent when set (needed for gated repositories).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

ENDPOINT = "https://huggingface.co"

# Files an import reads. Weights in other formats (.bin, .pt, .onnx) are never downloaded.
WANTED = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "README.md",
    "LICENSE*",
    "LICENCE*",
    "*.safetensors",
    "model.safetensors.index.json",
)

_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

Opener = Callable[[str], BinaryIO]


@dataclass(frozen=True)
class Snapshot:
    repository: str
    revision: str  # commit hash
    directory: Path
    files: tuple[str, ...]


def parse_reference(reference: str) -> tuple[str, str]:
    """``hf:org/name`` or ``hf:org/name@revision`` to ``(repository, revision)``; the revision defaults to main."""
    if not reference.startswith("hf:"):
        raise ValueError(f"not a Hugging Face reference: {reference!r}")
    repository, _, revision = reference[3:].partition("@")
    if not _REPO.match(repository):
        raise ValueError(f"invalid repository name {repository!r}")
    return repository, revision or "main"


def _default_opener(url: str) -> BinaryIO:
    request = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(request, timeout=60)


def download(repository: str, revision: str, cache: str | Path, opener: Opener | None = None) -> Snapshot:
    """Downloads the wanted files of ``repository`` at ``revision`` into ``cache/<org>/<name>/<commit>/``.
    Files already in the cache are not downloaded again (the directory is keyed by commit, so they cannot be
    stale)."""
    opener = opener or _default_opener
    quoted_repo = urllib.parse.quote(repository, safe="/")
    api = f"{ENDPOINT}/api/models/{quoted_repo}/revision/{urllib.parse.quote(revision, safe='')}"
    with opener(api) as response:
        info = json.load(response)
    commit = info.get("sha")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"{repository}@{revision}: the Hub did not return a commit hash")
    names = sorted(
        sibling["rfilename"]
        for sibling in info.get("siblings", [])
        if "/" not in sibling["rfilename"] and any(fnmatch.fnmatch(sibling["rfilename"], p) for p in WANTED)
    )
    directory = Path(cache) / repository / commit
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = directory / name
        if target.exists():
            continue
        partial = target.with_name(target.name + ".partial")
        url = f"{ENDPOINT}/{quoted_repo}/resolve/{commit}/{urllib.parse.quote(name)}"
        with opener(url) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=16 * 1024 * 1024)
        partial.replace(target)
    return Snapshot(repository, commit, directory, tuple(names))
