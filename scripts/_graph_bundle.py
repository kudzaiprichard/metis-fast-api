"""
Shared format definitions for the Metis clinical knowledge graph bundle.

The bundle is a gzipped JSON Lines file. Every line is one JSON object with a
``kind`` discriminator, written and read in this order:

    manifest    exactly one, first line — counts, provenance, similarity rule
    constraint  schema DDL (uniqueness constraints)
    index       schema DDL (range indexes not owned by a constraint)
    node        one per node, identified by its business key
    rel         one per relationship, endpoints referenced by business key

Nodes are addressed by business key (``patient_id``, ``drug_name``, ...) rather
than Neo4j's internal element IDs, which are not stable across databases. The
exporter derives the key for each label from the uniqueness constraints present
in the source database, so no label-to-key mapping is hardcoded here.

SIMILAR_TO is deliberately not carried in the bundle — see SIMILARITY_RULE.
"""

from __future__ import annotations

import gzip
import io
import json
import math
from typing import Any, Dict, Iterator, TextIO

FORMAT_VERSION = 1

#: Relationship types excluded from the bundle because they are derived data
#: that ``import_graph.py --with-similarity`` can rebuild exactly.
DERIVED_REL_TYPES = ("SIMILAR_TO",)

#: The exact rule that produced the 10,616,965 SIMILAR_TO edges in the source
#: database. Recovered from the data itself and verified to reproduce both the
#: edge count and the full similarity-score range bit-for-bit.
#:
#: An edge a -> b exists iff all of the following hold:
#:   a.patient_id < b.patient_id      (one direction per unordered pair)
#:   a.age_group      == b.age_group
#:   a.hba1c_severity == b.hba1c_severity
#:   |Δage| <= 10, |Δhba1c_baseline| < 2.0,
#:   |Δc_peptide| < 0.5, |Δdiabetes_duration| < 5.0
#:
#: and carries:
#:   similarity_score = 1 - (|Δage|/100 + |Δhba1c|/20
#:                           + |Δc_peptide|/5 + |Δduration|/50) / 4
#:   matched_features = ["age", "hba1c", "c_peptide", "diabetes_duration"]
SIMILARITY_RULE = {
    "tolerances": {
        "age": 10.0,
        "hba1c_baseline": 2.0,
        "c_peptide": 0.5,
        "diabetes_duration": 5.0,
    },
    "denominators": {
        "age": 100.0,
        "hba1c_baseline": 20.0,
        "c_peptide": 5.0,
        "diabetes_duration": 50.0,
    },
    "must_share": ["age_group", "hba1c_severity"],
    "matched_features": ["age", "hba1c", "c_peptide", "diabetes_duration"],
    "expected_edges": 10616965,
}

# ── Non-finite float encoding ──────────────────────────────────────────
# Outcome.adverse_events holds NaN for 14,276 nodes and the API's Cypher
# branches on `toString(o.adverse_events) = 'NaN'`, so these values are load
# bearing and must survive the round trip. Bare NaN/Infinity tokens are not
# valid JSON, so they travel as a tagged object instead — the bundle stays
# parseable by any conforming JSON reader.

_SPECIAL_KEY = "$float"
_SPECIAL_DECODE = {
    "NaN": float("nan"),
    "Infinity": float("inf"),
    "-Infinity": float("-inf"),
}


def encode_value(value: Any) -> Any:
    """Convert a Neo4j property value into a strictly-JSON-safe value."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            token = "NaN"
        else:
            token = "Infinity" if value > 0 else "-Infinity"
        return {_SPECIAL_KEY: token}
    if isinstance(value, list):
        return [encode_value(v) for v in value]
    return value


def decode_value(value: Any) -> Any:
    """Inverse of :func:`encode_value`."""
    if isinstance(value, dict) and _SPECIAL_KEY in value and len(value) == 1:
        return _SPECIAL_DECODE[value[_SPECIAL_KEY]]
    if isinstance(value, list):
        return [decode_value(v) for v in value]
    return value


def encode_props(props: Dict[str, Any]) -> Dict[str, Any]:
    return {k: encode_value(v) for k, v in props.items()}


def decode_props(props: Dict[str, Any]) -> Dict[str, Any]:
    return {k: decode_value(v) for k, v in props.items()}


# ── Line IO ────────────────────────────────────────────────────────────


def open_write(path) -> TextIO:
    """Open the bundle for writing. mtime=0 keeps re-exports byte-stable, so an
    unchanged graph does not show up as a diff in version control."""
    raw = gzip.GzipFile(filename=str(path), mode="wb", compresslevel=9, mtime=0)
    return io.TextIOWrapper(raw, encoding="utf-8", write_through=True)


def open_read(path) -> TextIO:
    return gzip.open(path, "rt", encoding="utf-8")


def write_record(fh: TextIO, record: Dict[str, Any]) -> None:
    # allow_nan=False makes an un-encoded non-finite float a hard error here
    # rather than a corrupt bundle discovered at import time.
    fh.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
    fh.write("\n")


def read_records(fh: TextIO) -> Iterator[Dict[str, Any]]:
    for line in fh:
        line = line.strip()
        if line:
            yield json.loads(line)
