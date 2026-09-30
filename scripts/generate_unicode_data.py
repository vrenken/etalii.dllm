"""Generates ``src/etalii_dllm/unicode_data.json.gz``, the Unicode tables the tokenizer uses (issue #98).

Run it with the Python whose ``unicodedata`` has the Unicode version to pin (3.13: Unicode 15.1)::

    python3.13 scripts/generate_unicode_data.py

Everything is read from :mod:`unicodedata` and ``str.lower`` of that interpreter; ``etalii_dllm.unicode`` then gives
the same results on every Python version. The output is canonical JSON, so regenerating it with the same Python gives
the same bytes (gzip with mtime 0).
"""

from __future__ import annotations

import gzip
import json
import sys
import unicodedata
from pathlib import Path

MAX = 0x110000
HANGUL_S_BASE, HANGUL_S_COUNT = 0xAC00, 11172
OUTPUT = Path(__file__).resolve().parent.parent / "src" / "etalii_dllm" / "unicode_data.json.gz"


def ranges(value_of) -> list[list]:
    """[[start, end, value], ...] for maximal runs of equal, truthy values (end inclusive)."""
    out: list[list] = []
    for cp in range(MAX):
        value = value_of(cp)
        if value and out and out[-1][1] == cp - 1 and out[-1][2] == value:
            out[-1][1] = cp
        elif value:
            out.append([cp, cp, value])
    return out


def is_surrogate(cp: int) -> bool:
    return 0xD800 <= cp <= 0xDFFF


def main() -> None:
    categories = ranges(lambda cp: (c := unicodedata.category(chr(cp))) != "Cn" and c)
    combining = ranges(lambda cp: unicodedata.combining(chr(cp)))
    decompositions: dict[str, list] = {}
    exclusions: list[int] = []
    for cp in range(MAX):
        if is_surrogate(cp) or HANGUL_S_BASE <= cp < HANGUL_S_BASE + HANGUL_S_COUNT:
            continue
        text = unicodedata.decomposition(chr(cp))
        if not text:
            continue
        parts = text.split()
        tag = parts.pop(0) if parts[0].startswith("<") else ""
        decompositions[str(cp)] = [tag, [int(p, 16) for p in parts]]
        if not tag and unicodedata.normalize("NFC", chr(cp)) != chr(cp):
            exclusions.append(cp)
    lower: dict[str, list[int]] = {}
    for cp in range(MAX):
        if is_surrogate(cp):
            continue
        lowered = chr(cp).lower()
        if lowered != chr(cp):
            lower[str(cp)] = [ord(c) for c in lowered]

    # Final sigma (Python's handle_capital_sigma): Case_Ignorable and Cased, recovered from str.lower itself.
    def final(text: str, index: int) -> bool:
        return text.lower()[index] == "\u03c2"

    ignorable, cased = [], []
    for cp in range(MAX):
        if is_surrogate(cp):
            continue
        c = chr(cp)
        after_cased = final("A" + c + "\u03a3", 2)  # ignorable or cased
        not_followed = final("A\u03a3" + c, 1)  # ignorable or not cased
        if after_cased and not_followed:
            ignorable.append(cp)
        elif after_cased:
            cased.append(cp)
    whitespace = [cp for cp in range(MAX) if not is_surrogate(cp) and chr(cp).isspace()]

    def compact(values: list[int]) -> list[list[int]]:
        out: list[list[int]] = []
        for v in values:
            if out and out[-1][1] == v - 1:
                out[-1][1] = v
            else:
                out.append([v, v])
        return out

    data = {
        "unicode_version": unicodedata.unidata_version,
        "categories": categories,
        "combining": combining,
        "decompositions": decompositions,
        "composition_exclusions": exclusions,
        "lower": lower,
        "case_ignorable": compact(ignorable),
        "cased": compact(cased),
        "whitespace": compact(whitespace),
    }
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    with open(OUTPUT, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as f:
        f.write(payload)
    print(f"{OUTPUT.name}: Unicode {data['unicode_version']} from Python {sys.version.split()[0]}", file=sys.stderr)


if __name__ == "__main__":
    main()
