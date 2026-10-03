"""Generates ``src/etalii_dllm/grapheme_data.json.gz``, the grapheme cluster tables of the Unigram normalizer (#347).

``tokenizers`` splits text into extended grapheme clusters before SentencePiece's precompiled normalizer looks them
up, with the Rust ``unicode-segmentation`` crate (1.13.3 in tokenizers 0.23, Unicode 17.0). To split exactly where
it does, ``etalii_dllm.unicode.graphemes`` uses that crate's own tables, read from its ``src/tables.rs``::

    curl -sSL -o seg.crate https://static.crates.io/crates/unicode-segmentation/unicode-segmentation-1.13.3.crate
    tar xzf seg.crate
    python scripts/generate_grapheme_data.py unicode-segmentation-1.13.3/src/tables.rs

The output is canonical JSON gzipped with mtime 0, so the same crate gives the same bytes.
"""

from __future__ import annotations

import gzip
import itertools
import json
import re
import sys
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "src" / "etalii_dllm" / "grapheme_data.json.gz"
CHAR = r"'\\u\{([0-9a-fA-F]+)\}'"


def _section(source: str, start: str) -> str:
    begin = source.index(start)
    return source[begin : source.index("];", begin)]


def main(tables: Path) -> None:
    source = tables.read_text(encoding="utf-8")
    version = re.search(r"pub const UNICODE_VERSION: \(u64, u64, u64\) = \((\d+), (\d+), (\d+)\);", source)
    assert version is not None
    categories = [
        [int(lo, 16), int(hi, 16), name]
        for lo, hi, name in re.findall(
            rf"\(\s*{CHAR},\s*{CHAR},\s*GC_(\w+)\s*\)", _section(source, "const grapheme_cat_table")
        )
    ]
    extend = [
        [int(lo, 16), int(hi, 16)]
        for lo, hi in re.findall(rf"\(\s*{CHAR},\s*{CHAR}\s*\)", _section(source, "const InCB_Extend_table"))
    ]
    linker_source = source[source.index("pub fn is_incb_linker") :]
    linker_source = linker_source[: linker_source.index("\n}")]
    linkers = [int(cp, 16) for cp in re.findall(r"'\\u\{([0-9a-fA-F]+)\}'", linker_source)]
    for table in (categories, extend):
        assert all(a[0] <= a[1] < b[0] for a, b in itertools.pairwise(table)), "ranges must be sorted"
    data = {
        "source": "unicode-segmentation 1.13.3",
        "unicode_version": ".".join(version.groups()),
        "categories": categories,
        "incb_extend": extend,
        "incb_linkers": linkers,
    }
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    OUTPUT.write_bytes(gzip.compress(raw, mtime=0))
    print(f"{OUTPUT}: {len(categories)} category ranges, {len(extend)} InCB extend ranges, {len(linkers)} linkers")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
