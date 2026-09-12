"""
Import the Metis clinical knowledge graph into a Neo4j database.

Run this once, before starting the API, against an empty database:

    python -m scripts.import_graph

Options:

    --with-similarity   also rebuild the 10.6M SIMILAR_TO relationships
    --drop-existing     wipe the target database first
    --input PATH        read a bundle from somewhere other than the default
    --database NAME     target database (default: neo4j)

Nothing in this codebase queries SIMILAR_TO — the similar-patients endpoints
compute similarity in Cypher at request time — so the default import produces a
fully functional graph. --with-similarity is for working with those edges
directly; it adds roughly 10.6M relationships and takes considerably longer.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._graph_bundle import (  # noqa: E402
    SIMILARITY_RULE,
    decode_props,
    open_read,
    read_records,
)

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "data" / "metis_graph.jsonl.gz"

NODE_BATCH = 5_000
REL_BATCH = 10_000
SIMILARITY_BATCH = 50_000


def _load_settings() -> tuple[str, str, str]:
    from src.configs import neo4j as neo4j_config

    return neo4j_config.uri, neo4j_config.username, neo4j_config.password


def _batched(items: Iterable[Any], size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _drop_existing(session) -> None:
    print("Dropping existing data...")
    for record in session.run("SHOW CONSTRAINTS YIELD name RETURN name"):
        session.run(f"DROP CONSTRAINT `{record['name']}` IF EXISTS")
    for record in session.run(
        "SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name"
    ):
        session.run(f"DROP INDEX `{record['name']}` IF EXISTS")

    # Delete in batches so a large graph doesn't blow the transaction memory
    # limit the way a single DETACH DELETE would.
    while True:
        deleted = session.run(
            "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS c"
        ).single()["c"]
        if deleted == 0:
            break
        print(f"  deleted {deleted} nodes", end="\r", flush=True)
    print("  database cleared      ")


def _assert_empty(session) -> None:
    count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    if count:
        raise SystemExit(
            f"Target database already contains {count} nodes. Re-run with "
            f"--drop-existing to replace them, or point --database at an empty one."
        )


def _create_schema(session, records: List[Dict[str, Any]]) -> None:
    # Constraints and indexes go in first: the uniqueness constraints back the
    # MERGE lookups below, turning node creation from a scan into an index hit.
    for record in records:
        cypher = record["cypher"]
        # createStatement has no IF NOT EXISTS, so make replays idempotent.
        for verb in ("CREATE CONSTRAINT ", "CREATE RANGE INDEX ", "CREATE INDEX ",
                     "CREATE TEXT INDEX ", "CREATE POINT INDEX ", "CREATE FULLTEXT INDEX "):
            if cypher.startswith(verb):
                head, _, tail = cypher.partition(" FOR ")
                cypher = f"{head} IF NOT EXISTS FOR {tail}"
                break
        session.run(cypher)
    print(f"  {len(records)} schema statements applied")


def _create_nodes(session, nodes: List[Dict[str, Any]]) -> int:
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for node in nodes:
        by_label.setdefault(node["label"], []).append(node)

    total = 0
    for label, group in sorted(by_label.items()):
        key = group[0]["key"][0]
        written = 0
        for batch in _batched(group, NODE_BATCH):
            session.run(
                f"UNWIND $rows AS row "
                f"MERGE (n:`{label}` {{`{key}`: row.key}}) "
                f"SET n = row.props",
                rows=[{"key": n["key"][1], "props": decode_props(n["props"])} for n in batch],
            )
            written += len(batch)
            print(f"  {label}: {written}/{len(group)}", end="\r", flush=True)
        print(f"  {label}: {written} nodes      ")
        total += written
    return total


def _create_relationships(
    session, rels: List[Dict[str, Any]], node_keys: Dict[str, str]
) -> int:
    # Group by (type, start label, end label) so each batch is a single
    # statement with static labels — Cypher cannot parameterise those.
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for rel in rels:
        grouped.setdefault((rel["type"], rel["start"][0], rel["end"][0]), []).append(rel)

    total = 0
    for (rel_type, start_label, end_label), group in sorted(grouped.items()):
        start_key = node_keys[start_label]
        end_key = node_keys[end_label]
        written = 0
        for batch in _batched(group, REL_BATCH):
            session.run(
                f"UNWIND $rows AS row "
                f"MATCH (a:`{start_label}` {{`{start_key}`: row.start}}) "
                f"MATCH (b:`{end_label}` {{`{end_key}`: row.end}}) "
                f"CREATE (a)-[r:`{rel_type}`]->(b) "
                f"SET r = row.props",
                rows=[
                    {
                        "start": r["start"][1],
                        "end": r["end"][1],
                        "props": decode_props(r["props"]),
                    }
                    for r in batch
                ],
            )
            written += len(batch)
            print(f"  {rel_type}: {written}/{len(group)}", end="\r", flush=True)
        print(f"  {rel_type} ({start_label}->{end_label}): {written} relationships      ")
        total += written
    return total


def _rebuild_similarity(session) -> int:
    """
    Recreate SIMILAR_TO from Patient properties.

    The rule in SIMILARITY_RULE was recovered from the source database and
    verified to reproduce its edge count (10,616,965) and full similarity-score
    range exactly, so this is a faithful reconstruction rather than an
    approximation. Pair generation happens here in Python — comparing every
    patient against every other inside Cypher would exhaust the server's
    transaction memory long before it finished.
    """
    try:
        import numpy as np
    except ImportError:
        raise SystemExit(
            "--with-similarity needs numpy (pip install numpy). Without it the "
            "20,000 x 20,000 pair comparison is impractically slow."
        )

    tol = SIMILARITY_RULE["tolerances"]
    den = SIMILARITY_RULE["denominators"]
    matched = SIMILARITY_RULE["matched_features"]

    rows = session.run(
        "MATCH (p:Patient) RETURN p.patient_id AS pid, p.age AS age, "
        "p.hba1c_baseline AS hba1c, p.c_peptide AS c_peptide, "
        "p.diabetes_duration AS duration, p.age_group AS age_group, "
        "p.hba1c_severity AS hba1c_severity ORDER BY p.patient_id"
    ).data()

    n = len(rows)
    if n == 0:
        print("  no Patient nodes; nothing to build")
        return 0
    print(f"  comparing {n} patients pairwise...")

    pid = [r["pid"] for r in rows]
    age = np.array([r["age"] for r in rows], dtype=np.float64)
    hba1c = np.array([r["hba1c"] for r in rows], dtype=np.float64)
    c_peptide = np.array([r["c_peptide"] for r in rows], dtype=np.float64)
    duration = np.array([r["duration"] for r in rows], dtype=np.float64)

    def codes(field: str) -> "np.ndarray":
        lookup = {v: i for i, v in enumerate(sorted({r[field] for r in rows}))}
        return np.array([lookup[r[field]] for r in rows], dtype=np.int32)

    age_group = codes("age_group")
    severity = codes("hba1c_severity")
    index = np.arange(n)

    total = 0
    pending: List[Dict[str, Any]] = []
    block = 512

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        session.run(
            "UNWIND $rows AS row "
            "MATCH (a:Patient {patient_id: row.a}) "
            "MATCH (b:Patient {patient_id: row.b}) "
            "CREATE (a)-[r:SIMILAR_TO]->(b) "
            "SET r.similarity_score = row.score, r.matched_features = $matched",
            rows=pending,
            matched=matched,
        )
        pending = []

    for start in range(0, n, block):
        stop = min(start + block, n)
        d_age = np.abs(age[start:stop, None] - age[None, :])
        d_hba1c = np.abs(hba1c[start:stop, None] - hba1c[None, :])
        d_pep = np.abs(c_peptide[start:stop, None] - c_peptide[None, :])
        d_dur = np.abs(duration[start:stop, None] - duration[None, :])

        keep = (
            (d_age <= tol["age"])
            & (d_hba1c < tol["hba1c_baseline"])
            & (d_pep < tol["c_peptide"])
            & (d_dur < tol["diabetes_duration"])
            & (age_group[start:stop, None] == age_group[None, :])
            & (severity[start:stop, None] == severity[None, :])
            # One edge per unordered pair, oriented low patient_id -> high.
            & (index[start:stop, None] < index[None, :])
        )
        if not keep.any():
            continue

        score = 1.0 - (
            d_age / den["age"]
            + d_hba1c / den["hba1c_baseline"]
            + d_pep / den["c_peptide"]
            + d_dur / den["diabetes_duration"]
        ) / 4.0

        left, right = np.nonzero(keep)
        for li, ri, sc in zip(left, right, score[keep]):
            pending.append({"a": pid[start + li], "b": pid[ri], "score": float(sc)})
            if len(pending) >= SIMILARITY_BATCH:
                total += len(pending)
                flush()
                print(f"  SIMILAR_TO: {total}", end="\r", flush=True)

    total += len(pending)
    flush()
    print(f"  SIMILAR_TO: {total} relationships      ")

    expected = SIMILARITY_RULE["expected_edges"]
    if total != expected:
        print(
            f"  note: built {total} edges, the source graph had {expected}. "
            f"This is expected if the Patient set differs from the bundled one."
        )
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Import the Metis graph bundle into Neo4j.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Bundle path.")
    parser.add_argument("--database", default="neo4j", help="Target database name.")
    parser.add_argument(
        "--drop-existing", action="store_true", help="Wipe the target database first."
    )
    parser.add_argument(
        "--with-similarity",
        action="store_true",
        help="Also rebuild the ~10.6M derived SIMILAR_TO relationships (slow).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Bundle not found: {args.input}")

    print(f"Reading {args.input}")
    manifest: Dict[str, Any] | None = None
    schema: List[Dict[str, Any]] = []
    nodes: List[Dict[str, Any]] = []
    rels: List[Dict[str, Any]] = []

    with open_read(args.input) as fh:
        for record in read_records(fh):
            kind = record["kind"]
            if kind == "manifest":
                manifest = record
            elif kind in ("constraint", "index"):
                schema.append(record)
            elif kind == "node":
                nodes.append(record)
            elif kind == "rel":
                rels.append(record)
            else:
                raise SystemExit(f"Unrecognised record kind in bundle: {kind!r}")

    if manifest is None:
        raise SystemExit("Bundle has no manifest line; it may be truncated or corrupt.")

    stamp = manifest.get("exported_at")
    print(
        f"  bundle{f' exported {stamp}' if stamp else ''} — "
        f"{len(nodes)} nodes, {len(rels)} relationships"
    )

    uri, username, password = _load_settings()
    print(f"Importing into {uri} (database: {args.database})")

    started = time.time()
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=args.database) as session:
            if args.drop_existing:
                _drop_existing(session)
            else:
                _assert_empty(session)

            print("Schema...")
            _create_schema(session, schema)

            print("Nodes...")
            node_total = _create_nodes(session, nodes)

            print("Relationships...")
            rel_total = _create_relationships(session, rels, manifest["node_keys"])

            similarity_total = 0
            if args.with_similarity:
                print("Rebuilding derived SIMILAR_TO relationships...")
                similarity_total = _rebuild_similarity(session)

            final_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            final_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    finally:
        driver.close()

    elapsed = time.time() - started
    print()
    print(f"Import complete in {elapsed:.1f}s")
    print(f"  nodes:         {node_total} imported, {final_nodes} in database")
    print(f"  relationships: {rel_total} imported"
          + (f" + {similarity_total} regenerated" if args.with_similarity else "")
          + f", {final_rels} in database")

    if not args.with_similarity:
        print()
        print("SIMILAR_TO was not built. The API does not query it, so every endpoint")
        print("works as-is; re-run with --with-similarity if you need those edges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
