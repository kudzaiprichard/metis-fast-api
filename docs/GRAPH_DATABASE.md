# Clinical Knowledge Graph — Setup

The Metis API reads a Neo4j graph of de-identified patient cases, treatments,
outcomes and clinical reference data. That graph ships with this repository as a
portable bundle, so a new environment does not need to rebuild it from scratch.

```
data/metis_graph.jsonl.gz     the exported graph (3.2 MB)
scripts/import_graph.py       load it into your Neo4j
scripts/export_graph.py       re-export after changing the graph
```

## Quick start

1. Start a Neo4j instance (5.x or 2025.x, Community or Enterprise).

2. Point the API at it, in `.env`:

   ```ini
   NEO4J_URI=neo4j://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your-password
   ```

3. Import the graph:

   ```bash
   python -m scripts.import_graph
   ```

   Takes about 20 seconds and prints a summary:

   ```
   Import complete in 18.6s
     nodes:         40048 imported, 40048 in database
     relationships: 86250 imported, 86250 in database
   ```

4. Start the API. `GET /health` reports the graph's node and relationship counts.

The importer refuses to run against a database that already has nodes. Use
`--drop-existing` to replace its contents.

## What's in the bundle

40,048 nodes and 86,250 relationships, plus 7 uniqueness constraints and 10
range indexes.

| Node | Count | Key |
| --- | --- | --- |
| `Patient` | 20,000 | `patient_id` |
| `Outcome` | 20,000 | `outcome_id` |
| `Guideline` | 15 | `guideline_id` |
| `Contraindication` | 12 | `contraindication_id` |
| `DrugInteraction` | 11 | `interaction_id` |
| `Treatment` | 5 | `drug_name` |
| `Comorbidity` | 5 | `condition_name` |

| Relationship | Count | Shape |
| --- | --- | --- |
| `HAS_CONDITION` | 46,212 | `Patient → Comorbidity` |
| `RECEIVED_TREATMENT` | 20,000 | `Patient → Treatment` |
| `RESULTED_IN` | 20,000 | `Treatment → Outcome` |
| `HAS_GUIDELINE` | 15 | `Treatment → Guideline` |
| `CONTRAINDICATED_BY` | 12 | `Treatment → Contraindication` |
| `INTERACTS_WITH` | 11 | `Treatment → DrugInteraction` |

## The SIMILAR_TO relationships

The source graph also held **10,616,965 `SIMILAR_TO` relationships** between
Patients — 99.2% of all its relationships. They are not in the bundle, for two
reasons:

- **Nothing queries them.** The similar-patient endpoints compute similarity in
  Cypher at request time from Patient properties (see
  `src/shared/neo4j/neo4j_graph_database.py`). No code path traverses a stored
  `SIMILAR_TO` edge.
- **They are fully derived.** Every one of them can be recomputed from the
  Patient nodes that *are* in the bundle.

So the default import produces a graph on which every API endpoint behaves
identically, at 3.2 MB instead of a couple hundred megabytes.

If you want the edges materialised anyway — to run your own graph algorithms
over them, say — rebuild them:

```bash
python -m scripts.import_graph --with-similarity   # needs numpy
```

### The rule

An edge `a → b` exists exactly when all of these hold:

```
a.patient_id     <  b.patient_id          (one edge per unordered pair)
a.age_group      == b.age_group
a.hba1c_severity == b.hba1c_severity
|Δage|                <= 10
|Δhba1c_baseline|     <  2.0
|Δc_peptide|          <  0.5
|Δdiabetes_duration|  <  5.0
```

and carries:

```
similarity_score = 1 - ( |Δage|/100 + |Δhba1c_baseline|/20
                       + |Δc_peptide|/5 + |Δdiabetes_duration|/50 ) / 4
matched_features = ["age", "hba1c", "c_peptide", "diabetes_duration"]
```

This rule was recovered from the source graph rather than from a generator
script, so it was checked against it: it reproduces the edge count exactly
(10,616,965) and the full similarity-score range to the last digit
(0.9018939617724214 … 0.9999875323616085). It lives in
`scripts/_graph_bundle.py` as `SIMILARITY_RULE`.

## Re-exporting

After changing the graph, regenerate the bundle:

```bash
python -m scripts.export_graph
```

It reads whichever database `NEO4J_URI` points at and rewrites
`data/metis_graph.jsonl.gz`. Nodes are written in key order and relationships
are sorted by endpoint, so the graph content is byte-stable between runs — the
only thing that changes for an unchanged graph is the manifest's `exported_at`.
Pass `--no-timestamp` to drop that too and get a fully identical file.

`--include-similarity` writes `SIMILAR_TO` out verbatim instead of leaving it to
be regenerated. Expect a file too large to commit.

Every node label needs a single-property uniqueness constraint, since the bundle
identifies nodes by business key rather than by Neo4j's internal element IDs
(which are not stable across databases). The exporter stops with an error if a
label lacks one.

## Bundle format

Gzipped [JSON Lines](https://jsonlines.org/). One JSON object per line, each
tagged with a `kind`:

```jsonc
{"kind":"manifest","format_version":1,"exported_at":"...","counts":{...}}
{"kind":"constraint","name":"patient_id_unique","cypher":"CREATE CONSTRAINT ..."}
{"kind":"index","name":"patient_age_group","cypher":"CREATE RANGE INDEX ..."}
{"kind":"node","label":"Patient","key":["patient_id","P000001"],"props":{...}}
{"kind":"rel","type":"RECEIVED_TREATMENT","start":["Patient","P000001"],
 "end":["Treatment","GLP-1"],"props":{"dose":"Standard"}}
```

It's plain text, so you can inspect it without any tooling:

```bash
zcat data/metis_graph.jsonl.gz | head -1 | python -m json.tool
```

One wrinkle worth knowing: 14,276 `Outcome` nodes have `adverse_events` set to
NaN, and the API's Cypher branches on `toString(o.adverse_events) = 'NaN'`, so
those values have to survive the round trip. Bare `NaN` is not valid JSON, so
non-finite floats travel as `{"$float":"NaN"}` and are decoded back on import.
The bundle stays readable by any conforming JSON parser.

## A note on comorbidity ordering

The graph queries build comorbidity lists with `collect(DISTINCT c.condition_name)`
and no `ORDER BY`, so Cypher returns them in storage order. A freshly imported
database stores relationships in a different physical order than the one they
were exported from, which means a patient's `comorbidities` list can come back as
`["CKD", "Hypertension", "NAFLD"]` where the original returned
`["Hypertension", "CKD", "NAFLD"]`.

Same members, different order — no data is lost, and this was verified across 350
query comparisons between the source and a freshly imported database. It is a
pre-existing property of the queries rather than something the import introduces,
but it is worth knowing if you ever assert on list order in a test. Adding
`ORDER BY` inside the `collect` would make it deterministic.

## Troubleshooting

**`Target database already contains N nodes`** — the importer will not write
into a non-empty database. Re-run with `--drop-existing`, or point `--database`
at an empty one.

**`Failed to connect to Neo4j`** on API startup — the graph connection is
established during startup and the app will not boot without it. Check that
Neo4j is running and that the `NEO4J_*` values in `.env` are right.

**`--with-similarity needs numpy`** — `pip install numpy`. It is only needed for
that flag, not for a normal import.

**Import into a non-default database** — both scripts take `--database NAME`
(requires Neo4j Enterprise for multi-database support).
