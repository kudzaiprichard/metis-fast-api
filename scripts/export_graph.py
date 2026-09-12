"""
Export the Neo4j clinical knowledge graph to a portable bundle.

Reads the graph this API is configured to talk to and writes every node,
relationship, constraint and index into a single gzipped JSON Lines file that
``import_graph.py`` can replay into an empty database.

Usage (from the fast_api directory):

    python -m scripts.export_graph
    python -m scripts.export_graph --output data/metis_graph.jsonl.gz
    python -m scripts.export_graph --include-similarity   # see note below

SIMILAR_TO is skipped by default. It accounts for 10,616,965 of the graph's
10,703,215 relationships, no query in this codebase traverses it, and it is
fully derived from Patient properties — ``import_graph.py --with-similarity``
rebuilds it exactly. Passing --include-similarity writes it out verbatim
instead, producing a file far too large to commit.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._graph_bundle import (  # noqa: E402
    DERIVED_REL_TYPES,
    FORMAT_VERSION,
    SIMILARITY_RULE,
    encode_props,
    open_read,
    open_write,
    write_record,
)

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "metis_graph.jsonl.gz"

PROGRESS_EVERY = 5_000

#: Relationship types larger than this are written in stream order instead of
#: being buffered and sorted. Only --include-similarity reaches it.
SORT_LIMIT = 2_000_000


def _load_settings() -> tuple[str, str, str]:
    from src.configs import neo4j as neo4j_config

    return neo4j_config.uri, neo4j_config.username, neo4j_config.password


def _node_keys(session) -> Dict[str, str]:
    """Map each label to the property backing its uniqueness constraint."""
    keys: Dict[str, str] = {}
    for record in session.run(
        "SHOW CONSTRAINTS YIELD type, labelsOrTypes, properties, entityType "
        "WHERE type = 'UNIQUENESS' AND entityType = 'NODE' "
        "RETURN labelsOrTypes, properties"
    ):
        labels = record["labelsOrTypes"]
        props = record["properties"]
        if len(labels) == 1 and len(props) == 1:
            keys[labels[0]] = props[0]
    return keys


def _labels(session) -> List[str]:
    return sorted(r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label"))


def _rel_types(session) -> List[str]:
    return sorted(
        r["relationshipType"]
        for r in session.run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )
    )


def _export_schema(session, fh) -> tuple[int, int]:
    constraints = 0
    for record in session.run(
        "SHOW CONSTRAINTS YIELD name, createStatement RETURN name, createStatement ORDER BY name"
    ):
        write_record(
            fh,
            {"kind": "constraint", "name": record["name"], "cypher": record["createStatement"]},
        )
        constraints += 1

    indexes = 0
    # LOOKUP indexes exist in every database by default, and an index owned by
    # a constraint is created implicitly with that constraint — replaying
    # either one would fail on import.
    for record in session.run(
        "SHOW INDEXES YIELD name, type, createStatement, owningConstraint "
        "WHERE type <> 'LOOKUP' AND owningConstraint IS NULL "
        "RETURN name, createStatement ORDER BY name"
    ):
        write_record(
            fh, {"kind": "index", "name": record["name"], "cypher": record["createStatement"]}
        )
        indexes += 1

    return constraints, indexes


def _export_nodes(session, fh, labels: List[str], keys: Dict[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    for label in labels:
        key = keys.get(label)
        if key is None:
            raise SystemExit(
                f"Label '{label}' has no single-property uniqueness constraint, so its "
                f"nodes cannot be given a stable identity in the bundle. Add one, or "
                f"remove the label, then re-run."
            )

        # Streamed rather than SKIP-paged: SKIP re-scans everything it skips,
        # which turns a large label into quadratic work. ORDER BY rides the
        # constraint's index, so the output stays deterministic for free.
        exported = 0
        result = session.run(
            f"MATCH (n:`{label}`) RETURN properties(n) AS props ORDER BY n.`{key}`"
        )
        for record in result:
            props = record["props"]
            write_record(
                fh,
                {
                    "kind": "node",
                    "label": label,
                    "key": [key, props[key]],
                    "props": encode_props(props),
                },
            )
            exported += 1
            if exported % PROGRESS_EVERY == 0:
                print(f"  {label}: {exported} nodes", end="\r", flush=True)

        counts[label] = exported
        print(f"  {label}: {exported} nodes      ")

    return counts


def _key_expression(alias: str, keys: Dict[str, str]) -> str:
    """Cypher that reads whichever key property a node happens to carry.

    Endpoint labels vary per relationship type, so the key property cannot be
    named statically. coalesce over every known key resolves it in one pass and
    keeps full node properties off the wire.
    """
    distinct = sorted(set(keys.values()))
    return f"coalesce({', '.join(f'{alias}.`{k}`' for k in distinct)})"


def _export_relationships(
    session, fh, rel_types: List[str], keys: Dict[str, str]
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    start_key = _key_expression("a", keys)
    end_key = _key_expression("b", keys)

    for rel_type in rel_types:
        rows = []
        streamed = 0
        result = session.run(
            f"MATCH (a)-[r:`{rel_type}`]->(b) "
            f"RETURN labels(a)[0] AS a_label, {start_key} AS a_key, "
            f"       labels(b)[0] AS b_label, {end_key} AS b_key, "
            f"       properties(r) AS props"
        )
        for record in result:
            rows.append(
                {
                    "kind": "rel",
                    "type": rel_type,
                    "start": [record["a_label"], record["a_key"]],
                    "end": [record["b_label"], record["b_key"]],
                    "props": encode_props(record["props"]),
                }
            )
            streamed += 1
            if streamed % PROGRESS_EVERY == 0:
                print(f"  {rel_type}: {streamed} relationships", end="\r", flush=True)

        # Relationships have no natural index to order by, so they are sorted
        # here to keep re-exports byte-identical. Skipped past SORT_LIMIT, where
        # holding the whole type in memory costs more than a stable diff is
        # worth — only --include-similarity gets anywhere near that.
        if len(rows) <= SORT_LIMIT:
            rows.sort(key=lambda r: (r["start"][0], str(r["start"][1]),
                                     r["end"][0], str(r["end"][1])))
        for row in rows:
            write_record(fh, row)

        counts[rel_type] = len(rows)
        print(f"  {rel_type}: {len(rows)} relationships      ")

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the Metis Neo4j graph to a bundle.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Bundle path.")
    parser.add_argument("--database", default="neo4j", help="Source database name.")
    parser.add_argument(
        "--include-similarity",
        action="store_true",
        help="Write SIMILAR_TO verbatim instead of leaving it to be regenerated.",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Omit exported_at, making the whole file byte-identical between runs "
             "when the graph is unchanged.",
    )
    args = parser.parse_args()

    uri, username, password = _load_settings()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    excluded = () if args.include_similarity else DERIVED_REL_TYPES

    print(f"Exporting from {uri} (database: {args.database})")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=args.database) as session:
            keys = _node_keys(session)
            labels = _labels(session)
            rel_types = [t for t in _rel_types(session) if t not in excluded]

            # The manifest is the first line, but its counts are only known
            # once everything is written — so the body goes to a temp file and
            # the two are stitched together at the end.
            tmp = args.output.with_suffix(args.output.suffix + ".part")
            with open_write(tmp) as body:
                print("Schema...")
                constraint_count, index_count = _export_schema(session, body)
                print(f"  {constraint_count} constraints, {index_count} indexes")

                print("Nodes...")
                node_counts = _export_nodes(session, body, labels, keys)

                print("Relationships...")
                rel_counts = _export_relationships(session, body, rel_types, keys)

            manifest = {
                "kind": "manifest",
                "format_version": FORMAT_VERSION,
                "exported_at": None
                if args.no_timestamp
                else datetime.now(timezone.utc).isoformat(),
                "source_uri": uri,
                "source_database": args.database,
                "node_keys": keys,
                "counts": {
                    "constraints": constraint_count,
                    "indexes": index_count,
                    "nodes": node_counts,
                    "relationships": rel_counts,
                    "total_nodes": sum(node_counts.values()),
                    "total_relationships": sum(rel_counts.values()),
                },
                "excluded_relationship_types": list(excluded),
                "similarity_rule": SIMILARITY_RULE if excluded else None,
            }

            with open_write(args.output) as out:
                write_record(out, manifest)
                with open_read(tmp) as body_in:
                    for line in body_in:
                        out.write(line)
            tmp.unlink()
    finally:
        driver.close()

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print()
    print(f"Wrote {args.output}  ({size_mb:.1f} MB)")
    print(f"  {manifest['counts']['total_nodes']} nodes, "
          f"{manifest['counts']['total_relationships']} relationships")
    if excluded:
        print(f"  excluded (regenerable): {', '.join(excluded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
