# Metis — FastAPI Backend (`metis-fast-api`)

> **Repository:** <https://github.com/kudzaiprichard/metis-fast-api>
> **Role in Metis:** the clinical-workflow backend. Wraps the diabetes contextual-bandit
> model from [`diabetes-bandits`](https://github.com/kudzaiprichard/diabetes-bandits)
> and serves the [`metis_ui`](https://github.com/kudzaiprichard/metis_ui) desktop client.

A modular FastAPI backend for **clinical decision support in Type 2 Diabetes treatment selection**.

At its core is a **Neural Thompson Sampling contextual bandit** that recommends one of five
diabetes therapies (Metformin, GLP-1, SGLT-2, DPP-4, Insulin) given a 16-feature clinical
context. Around the model the service provides a full clinical workflow: patient + medical
record CRUD, persisted prediction history with doctor decisions, similar-case retrieval against
a Neo4j historical dataset, and a CSV-driven bandit simulator that streams per-step events over
Server-Sent Events.

### Why this repo exists as a separate component

The bandit model and its inference façade live in
[`diabetes-bandits`](https://github.com/kudzaiprichard/diabetes-bandits) — that repo is
research-shaped (notebooks, OPE, training CLI, synthetic data oracle). This repo is the
**production HTTP/persistence layer** that wraps it: PostgreSQL, JWT auth, Alembic
migrations, Neo4j similar-case retrieval, doctor decisions, and the multi-tenant
SSE-streamed simulator. Splitting them keeps the model artefact pipeline reproducible
and the API service deployable, with `bandits/inference/` vendored verbatim under
`src/shared/inference/` as the seam between the two.

---

## Table of Contents

1. [What this service does](#what-this-service-does)
2. [Tech stack](#tech-stack)
3. [Quick start](#quick-start)
4. [Configuration](#configuration)
5. [Project structure](#project-structure)
6. [Architecture](#architecture)
7. [Request lifecycle](#request-lifecycle)
8. [Inference engine](#inference-engine)
9. [Modules and routes](#modules-and-routes)
10. [Database schema](#database-schema)
11. [Migrations](#migrations)
12. [Response envelope and errors](#response-envelope-and-errors)
13. [Simulation streaming model](#simulation-streaming-model)
14. [Adding a new module](#adding-a-new-module)
15. [Operational notes](#operational-notes)

---

## What this service does

```mermaid
flowchart LR
    Doctor[Doctor / Admin] -->|REST + JWT| API[FastAPI app]

    subgraph Backend["Metis backend"]
        API --> Auth[Auth module]
        API --> Patients[Patients module]
        API --> Inference[Inference router]
        API --> Predictions[Predictions module]
        API --> SimulationsMod[Simulations module]
        API --> SimilarMod[Similar Patients module]
    end

    Patients --> PG[(PostgreSQL)]
    Predictions --> PG
    SimulationsMod --> PG
    Auth --> PG

    Inference --> Engine[InferenceEngine<br/>shared.inference]
    Predictions --> Engine
    SimulationsMod --> Engine

    Engine --> Torch[(neural_thompson.pt<br/>feature_pipeline_scaled.joblib)]
    Engine -. optional .-> Gemini[(Gemini LLM)]
    SimilarMod --> Neo[(Neo4j historical cohort)]
```

The four primary use cases are:

1. **Predict + explain** — Given a stored medical record, run the bandit, persist the structured
   recommendation plus a Gemini-generated clinical narrative, then let the doctor record an
   `ACCEPTED` or `OVERRIDDEN` decision (`/api/v1/predictions`).
2. **Stateless inference** — One-shot or batch prediction endpoints that bypass persistence
   (`/api/v1/inference`).
3. **Find similar historical patients** — Match the current patient's profile against a Neo4j
   cohort using normalised feature distance + comorbidity Jaccard overlap, return tabular or
   graph-shaped results (`/api/v1/similar-patients`).
4. **Simulate the bandit on a CSV cohort** — Upload a CSV (100–50,000 rows × 16 features),
   replay it through a per-simulation engine, and stream every step over SSE
   (`/api/v1/simulations`).

---

## Tech stack

| Concern | Choice | Notes |
|---|---|---|
| Web framework | **FastAPI** | App is built via factory in `src/core/factory.py` |
| Async runtime | **uvicorn** | Entry point `main.py` calls `uvicorn.run("src.core.factory:create_app", factory=True)` |
| ORM | **SQLAlchemy 2.x** (async) | `asyncpg` driver, async sessions |
| RDBMS | **PostgreSQL** | UUID primary keys via `gen_random_uuid()`, JSONB for variable payloads |
| Migrations | **Alembic** | Async env, models imported in `alembic/env.py` |
| Graph DB | **Neo4j** (5.x driver) | Sync driver wrapped via `asyncio.to_thread` in services |
| ML | **PyTorch** + scikit-learn | Loaded once at startup; engine is the only consumer |
| Auth | **PyJWT** + **bcrypt** | Tokens persisted; revocation enforced on every request |
| LLM | **google-genai** (Gemini) | Optional; only the predict-with-explanation paths require it |
| Streaming | **sse-starlette** | Raw ASGI logging middleware leaves SSE untouched |
| Validation | **Pydantic v2** | Used at the HTTP boundary and inside the inference engine |
| Logging | stdlib `logging` + **loguru** | `loguru` is used inside `shared/inference` |

The full module-version pinning lives in the project's virtualenv; there is no
`requirements.txt` or `pyproject.toml` checked in at the repo root, so install the deps from
the imports below or freeze the local venv.

**Imported (non-stdlib) packages at runtime:** `fastapi`, `uvicorn`, `sqlalchemy[asyncio]`,
`asyncpg`, `alembic`, `pydantic[email]`, `python-dotenv`, `pyyaml`, `pyjwt`, `bcrypt`, `torch`,
`numpy`, `pandas`, `joblib`, `scikit-learn`, `loguru`, `google-genai`, `sse-starlette`,
`neo4j`.

---

## Quick start

### Prerequisites

- Python 3.11+
- PostgreSQL with a database you can connect to
- Neo4j 5.x (the similar-patient endpoints require a populated graph; everything else still
  works if Neo4j fails to connect — see `lifespan.py`)
- A trained NeuralThompson checkpoint and a fitted FeaturePipeline `.joblib`
- Optional: a Gemini API key if you want LLM explanations

### Install

```bash
git clone <repo-url>
cd fast_api
python -m venv venv
# Windows
venv\Scripts\activate
# *nix
source venv/bin/activate

pip install fastapi uvicorn "sqlalchemy[asyncio]" asyncpg alembic \
            "pydantic[email]" python-dotenv pyyaml pyjwt bcrypt \
            torch numpy pandas joblib scikit-learn loguru \
            google-genai sse-starlette neo4j
```

### Configure

Create `.env` at the project root. Required vars (defaulted ones in the YAML can be omitted):

```dotenv
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/metis
JWT_SECRET_KEY=replace-me-with-a-long-random-string
GEMINI_API_KEY=...                    # optional but expected by current YAML
NEO4J_PASSWORD=...

# Model artefacts
MODEL_PATH=/absolute/path/to/models
MODEL_FILE=neural_thompson.pt
PIPELINE_FILE=feature_pipeline_scaled.joblib
```

> **Note.** `application.yaml` currently marks `GEMINI_API_KEY` and `NEO4J_PASSWORD` as
> `required` — startup will exit with a clear "required but not set" error if either is
> missing. Set them to placeholder values during local bring-up if you don't have them yet.

### Migrate

```bash
# create the database first
psql -c "CREATE DATABASE metis;"

# apply migrations
alembic upgrade head
```

### Run

```bash
python main.py
```

The app boots at `http://127.0.0.1:8000`. Swagger UI: `/docs`. Health: `/health`.

---

## Configuration

All configuration is centralised in `src/configs/application.yaml`. Each leaf value uses a
**pipe format** that fuses the env-var binding, the type, and the required-ness:

```yaml
pool_size: "${DB_POOL_SIZE:5} | int"          # env var with default → int
url:       "${DATABASE_URL} | str | required"  # required env var → str
debug:     "${DEBUG:false} | bool"             # bool ('true'/'1'/'yes'/'on' are true)
origins:   "${CORS_ORIGINS:*} | list"          # comma-separated list
```

`src/configs/loader.py` resolves the templates against `os.environ` (loaded from `.env` via
`python-dotenv`), casts to the declared type, collects every error, and **fails fast** on the
first start if any required value is missing or any cast fails. Unknown types raise
`ConfigError` with a list of issues.

`src/configs/__init__.py` triggers the load on import and exposes each top-level YAML section
as a `SimpleNamespace`:

```python
from src.configs import application, database, security, server, logging, model, gemini, neo4j

database.url
security.jwt.secret_key
neo4j.uri
```

`src/configs/generate.py` writes a `__init__.pyi` stub after every successful load so IDEs get
autocomplete on `database.url`, `security.jwt.algorithm`, etc. The stub is in `.gitignore`
(`*.pyi`) and is regenerated each boot.

You can also reload at runtime via `from src.configs import reload_config; reload_config()`.

### Sections currently defined

| Section | Purpose |
|---|---|
| `application` | name, version, debug, environment |
| `database` | URL, pool sizing, echo, pre-ping, recycle |
| `security.jwt` | secret, algorithm, access TTL (min), refresh TTL (days) |
| `security.admin` | seeded admin email/username/password |
| `security.token_cleanup_interval_seconds` | background purge cadence |
| `server.cors` | origins, methods, headers, credentials |
| `logging` | level, format, file path |
| `model` | model dir, model file, pipeline file |
| `gemini` | api key, model name, temperature |
| `neo4j` | uri, username, password |

The inference engine has its **own** config layer (`BANDITS_*` env vars, see
[Inference engine](#inference-engine)). The bridge that translates Metis YAML config into an
`InferenceConfig` lives in `src/shared/inference_bootstrap.py` and is the single seam where
the two systems meet.

---

## Project structure

```
fast_api/
├── main.py                      # uvicorn entry — runs src.core.factory:create_app
├── alembic.ini
├── alembic/                     # Async migrations (4 revisions, see § Migrations)
│   ├── env.py
│   └── versions/
├── docs/
│   └── API_INTEGRATION.md       # Frontend-facing endpoint reference
├── src/
│   ├── configs/                 # YAML + env-var loader, .pyi stub generator
│   ├── core/
│   │   ├── factory.py           # FastAPI app factory + router wiring
│   │   ├── lifespan.py          # Startup / shutdown lifecycle
│   │   └── middleware.py        # CORS + raw-ASGI request logger (SSE-safe)
│   ├── shared/
│   │   ├── database/            # async engine, BaseModel, BaseRepository, get_db
│   │   ├── exceptions/          # AppException hierarchy + global handlers
│   │   ├── neo4j/               # Singleton Neo4j driver + query layer
│   │   ├── inference/           # Treated as a frozen third-party library
│   │   ├── inference_bootstrap.py  # Bridge: app config → InferenceConfig + engine
│   │   └── responses/           # ApiResponse, PaginatedResponse, ErrorDetail
│   └── modules/
│       ├── auth/                # Authentication + admin user management
│       ├── patients/            # Patient + MedicalRecord CRUD + similar-case search
│       ├── models/              # Stateless inference router
│       ├── predictions/         # Persisted predictions + doctor decisions
│       └── simulations/         # CSV-driven bandit simulator + SSE streaming
└── logs/
```

Every business module follows the same three-layer split:

```
modules/<name>/
├── domain/
│   ├── models/         # SQLAlchemy ORM + enums
│   ├── repositories/   # Data access (extend BaseRepository)
│   └── services/       # Business logic — accept primitives, return domain objects
├── internal/           # Module-private helpers (mappers, runners, seeders, …)
└── presentation/
    ├── controllers/    # FastAPI route handlers
    ├── dependencies.py # Service factories + auth/role dependencies
    └── dtos/           # Pydantic request/response models
```

---

## Architecture

### Layered modules

```mermaid
flowchart TB
    Client([HTTP client])

    subgraph Pres["Presentation"]
        C[Controller<br/>FastAPI router]
        D[DTOs<br/>Pydantic v2]
        Dep[dependencies.py<br/>service factories + auth guards]
    end

    subgraph Domain["Domain"]
        S[Service<br/>business logic]
        R[Repository<br/>extends BaseRepository]
        M[ORM model<br/>extends BaseModel]
    end

    subgraph Internal["Internal"]
        I[mappers / runners / seeders]
    end

    subgraph SharedLayer["src/shared"]
        DB[(async SQLAlchemy)]
        Eng[InferenceEngine]
        N4[Neo4j driver]
        Resp[ApiResponse / ErrorDetail]
        Exc[AppException + handlers]
    end

    Client --> C
    C --> D
    C --> Dep
    Dep --> S
    S --> R
    S --> I
    S --> Eng
    S --> N4
    R --> M
    M --> DB
    C --> Resp
    S -. raises .-> Exc
    Exc --> Resp
```

Rules of thumb the codebase follows:

- **Controllers convert** between DTOs and primitives. They never call repositories directly.
- **Services accept primitives**, return ORM objects, and raise `AppException` subclasses for
  any business or input failure.
- **Repositories extend `BaseRepository[T]`** which provides `get_by_id`, `get_one`, `exists`,
  `paginate`, `create`, `create_many`, `update`, `delete` and uniform filter/order helpers.
- **The shared inference engine is a frozen library**. Modules consume it through the
  bootstrap module and the public façade in `src/shared/inference/__init__.py` only.

### Startup lifecycle

`src/core/lifespan.py` runs the following on app startup, in order. Failures are logged but do
not crash the process — the engine and Neo4j independently degrade.

```mermaid
sequenceDiagram
    autonumber
    participant U as uvicorn
    participant L as lifespan()
    participant Cfg as configs
    participant Log as logging
    participant Eng as InferenceEngine
    participant Neo as Neo4j
    participant Seed as admin_seeder
    participant DB as Postgres
    participant T1 as token_cleanup task
    participant T2 as registry sweep task

    U->>Cfg: import src.configs (loads YAML + .env, generates .pyi)
    U->>L: enter lifespan
    L->>Log: configure file + stream handlers
    L->>Eng: build_inference_engine() → app.state.engine
    L->>Neo: connect(uri, user, pw)
    L->>Seed: seed_admin() if no ADMIN exists
    L->>DB: UPDATE simulations SET status='FAILED' WHERE status='RUNNING'
    L->>T1: asyncio.create_task(start_token_cleanup)
    L->>T2: asyncio.create_task(_periodic_registry_sweep)
    L-->>U: yield (app serves traffic)

    Note over L: shutdown
    L->>T1: cancel + await
    L->>T2: cancel + await
    L->>Neo: close()
    L->>DB: engine.dispose()
```

The "recover orphaned simulations" step is important: if the process dies mid-simulation, the
in-memory registry is gone but the DB still says `RUNNING`. On next boot those rows are
flipped to `FAILED` with `error_message="Server restarted during simulation"` so the UI never
sees a phantom-running simulation.

### Middleware

`src/core/middleware.py` registers, in order:

1. **CORS** (`fastapi.middleware.cors`) — origins/methods/headers/credentials from the YAML.
2. **`RequestLoggingMiddleware`** — a *raw ASGI* middleware (not `BaseHTTPMiddleware`) that
   logs `method path — status (elapsed)`. The raw form is deliberate: `BaseHTTPMiddleware`
   buffers the response body and would break SSE streaming.

---

## Request lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant FA as FastAPI
    participant MW as RequestLoggingMiddleware
    participant Route as Controller
    participant Dep as Depends(get_db)
    participant Auth as Depends(get_current_user)
    participant Svc as Service
    participant Repo as Repository
    participant DB as Postgres

    C->>FA: HTTP request
    FA->>MW: ASGI scope
    MW->>Route: forward
    Route->>Dep: open AsyncSession + begin transaction
    Route->>Auth: verify Bearer JWT (decode + DB revocation check)
    Auth-->>Route: User
    Route->>Svc: call(primitives)
    Svc->>Repo: queries / writes
    Repo->>DB: SQL
    DB-->>Repo: rows
    Repo-->>Svc: ORM objects
    Svc-->>Route: domain object (or raises AppException)
    Route-->>FA: ApiResponse / PaginatedResponse
    FA-->>MW: response
    MW-->>C: HTTP response
    Note over Dep: on success: commit · on exception: rollback
```

`get_db` (in `src/shared/database/dependencies.py`) wraps the request in `async with
session.begin()`, so every controller is **transactional by default**. A read-only variant
`get_db_readonly` is also available but is not currently used by any controller.

---

## Inference engine

`src/shared/inference/` is the prediction stack and is **deliberately treated as a frozen
third-party library** by the rest of the codebase — only `src/shared/inference_bootstrap.py`
crosses the boundary. The engine ships its own configuration, schemas, errors, streaming
helpers, and LLM client adapters.

### Public surface

Importable from `src.shared.inference`:

- `InferenceEngine` — the façade
- `InferenceConfig` — config loader (env prefix `BANDITS_`, optional YAML via
  `BANDITS_CONFIG_FILE`)
- `PatientInput`, `LearningRecord`, `PredictionResult`, `LearningAck`, `Treatment`
- `LearningStream`, `LearningStepEvent`, `LearningSession`, `AsyncLearningSession`
- Errors: `InferenceError`, `ConfigurationError`, `ValidationError`, `ModelError`,
  `ExplanationError`
- `StubClient` for deterministic tests/demos

### What the engine knows

**Treatments (5, fixed ordering — also the on-the-wire arm indexing contract):**
`Metformin`, `GLP-1`, `SGLT-2`, `DPP-4`, `Insulin`.

**Context features (16, also fixed ordering):**
`age`, `bmi`, `hba1c_baseline`, `egfr`, `diabetes_duration`, `fasting_glucose`, `c_peptide`,
`cvd`, `ckd`, `nafld`, `hypertension`, `bp_systolic`, `ldl`, `hdl`, `triglycerides`, `alt`.

`PatientInput` additionally accepts four optional structured safety flags
(`medullary_thyroid_history`, `men2_history`, `pancreatitis_history`, `type1_suspicion`) which
default to `0`, plus audit-only fields (`gender`, `ethnicity`, `patient_id`) that are
**never** passed into the feature pipeline (G-15 fairness posture). All numeric fields have
explicit value ranges enforced by Pydantic.

### Construction and lifecycle

```mermaid
flowchart LR
    YAML[application.yaml<br/>+ .env]
    AppCfg[src.configs.model<br/>src.configs.gemini]
    Bootstrap[inference_bootstrap.<br/>build_inference_config]
    BanditsEnv[BANDITS_* env vars]
    InfCfg[InferenceConfig]
    FromCfg[InferenceEngine.from_config]
    Pipeline[FeaturePipeline.load<br/>feature_pipeline_scaled.joblib]
    Model[NeuralThompson.load<br/>neural_thompson.pt]
    Drift[DriftMonitor]
    Extractor[ExplainabilityExtractor]
    Engine[InferenceEngine<br/>app.state.engine]

    YAML --> AppCfg
    AppCfg --> Bootstrap
    BanditsEnv -.-> Bootstrap
    Bootstrap --> InfCfg
    InfCfg --> FromCfg
    FromCfg --> Pipeline
    FromCfg --> Model
    FromCfg --> Extractor
    FromCfg --> Drift
    Pipeline --> Engine
    Model --> Engine
    Extractor --> Engine
    Drift --> Engine
```

The engine is built once during `lifespan()` and stored on `app.state.engine`. The
`models` module's `get_inference_engine` dependency reads it back and raises a 503 if the
engine never initialised or reports `ready=False`. The `predictions` module uses the same
dependency. `simulations` is different: it builds a **per-simulation** engine via
`InferenceEngine.from_config(build_inference_config())` and optionally calls
`model.reset_posterior()` so each simulation starts from the prior without disturbing the
app-wide engine.

### Predicting

```mermaid
flowchart LR
    PI[PatientInput]
    Pipe[FeaturePipeline<br/>transform_single]
    NB[NeuralThompson]
    EX[ExplainabilityExtractor.extract]
    Gate[Safety gate<br/>+ contraindications]
    Result[PredictionResult]
    LLM[LLMExplainer<br/>Gemini or Stub]
    Drift[DriftMonitor.observe]

    PI --> Pipe --> NB --> EX --> Gate --> Result
    Result -. explain=True/'require' .-> LLM
    LLM --> Result
    NB -. context norm .-> Drift
```

Three things flow into a `PredictionResult`:

1. **Decision payload** — `recommended`, `recommended_idx`, `confidence_pct`,
   `confidence_label` (`HIGH`/`MODERATE`/`LOW`), `posterior_means`, `win_rates` (Thompson-draw
   win rates), `runner_up`, `runner_up_win_rate`, `mean_gap`.
2. **Safety envelope** — `safety_status` (`CLEAR` / `WARNING` / `CONTRAINDICATION_FOUND`),
   `safety_findings`, `excluded_treatments`, an optional `override`.
3. **Engine extras** — `model_top_treatment` (the bandit's choice *before* safety override),
   `attribution`, `contrast`, `uncertainty_drivers`, plus an optional structured
   `explanation` block from the LLM.

`apredict` is the async variant (just `asyncio.to_thread`). Controllers always use
`apredict`. `predict_batch` accepts a DataFrame or iterable of dicts and returns a list of
`PredictionResult` — invalid rows come back as `accepted=False` sentinels rather than raising.

The engine is thread-safe for reads. Writes (`update`, `update_many`, `ingest_csv`) serialise
on a single `RLock`; predictions never acquire the write lock so the worst-case view is one
update stale (by design — see the engine docstring).

### Continuous learning

The engine also exposes online-update paths: `update`/`aupdate` for one record, `update_many`
for an iterator, `ingest_csv` for a file, plus context-managed `learning_session` and the
richer `learning_stream` (used by the simulator) which emits one
`LearningStepEvent` per step with the Thompson sample, post-update posterior, running
aggregates, drift signals, and patient context. The events serialise straight to SSE,
WebSocket, or console via `to_sse()` / `to_ws()` / `to_console_line()`.

### LLM explanations

Provider is selected by `BANDITS_LLM_PROVIDER` (or `gemini`/`stub`/`none` via the bootstrap):

- `none` (default in `InferenceConfig`): `explain=True` raises `ConfigurationError` if the
  caller passes `"require"`, otherwise returns `explanation=None`.
- `gemini`: requires an API key — pulled from `BANDITS_LLM_API_KEY`, then `GEMINI_API_KEY`.
  `inference_bootstrap.py` automatically enables `gemini` whenever `gemini.api_key` is set
  in YAML.
- `stub`: deterministic local generator for testing.

Errors are surfaced as `ExplanationError`. The global error handler maps the special case
"API key" / "API_KEY_INVALID" / "LLM generate call failed" sub-strings to a 503 with code
`LLM_UNAVAILABLE` so the UI can show "AI explanation unavailable" rather than a 500.

---

## Modules and routes

All routes are mounted in `src/core/factory.py::_register_routers`. There is exactly one
router per module file, and each one declares its auth guards as router-level dependencies.

| Prefix | Router | Auth | Brief |
|---|---|---|---|
| `/health` | inline in factory | none | engine readiness + snapshot |
| `/api/v1/auth` | `auth_router` | mixed | register/login/refresh/logout/me |
| `/api/v1/users` | `user_router` | `ADMIN` | user management |
| `/api/v1/patients` | `patient_router` | `ADMIN` or `DOCTOR` | patient + medical record CRUD |
| `/api/v1/similar-patients` | `similar_patients_router` | `DOCTOR` | Neo4j cohort search |
| `/api/v1/inference` | `inference_router` | `ADMIN` or `DOCTOR` | stateless predict |
| `/api/v1/predictions` | `prediction_router` | `ADMIN` or `DOCTOR` | persisted predictions + doctor decisions |
| `/api/v1/simulations` | `simulation_router` | `ADMIN` | bandit simulator + SSE |

Below is what each module actually exposes. The frontend-facing reference with full request /
response examples lives in `docs/API_INTEGRATION.md`.

### Auth

`POST /register` · `POST /login` · `POST /refresh` · `POST /logout` · `GET /me` · `PATCH /me`.

JWT access + refresh tokens are **persisted in the `tokens` table** and re-checked against the
DB on every authenticated request — that is, `verify_token` decodes the JWT *and* asserts the
token row is unrevoked. Login revokes every prior token for the user before issuing a new
pair; `refresh` revokes the used refresh token then mints both anew. `logout` revokes every
token belonging to the calling user. Passwords are bcrypt-hashed
(`internal/password_hasher.py`).

A background task (`internal/token_cleanup.py`) sleeps for
`security.token_cleanup_interval_seconds` and deletes expired rows on each tick.

A default admin (`security.admin.email/username/password` from YAML) is seeded on startup if
no `ADMIN` row exists.

### Users (admin)

`GET /` (paginated) · `GET /{id}` · `POST /` · `PATCH /{id}` · `DELETE /{id}`. Filters: `role`
(regex `^(ADMIN|DOCTOR)$`), `is_active`. The router itself is gated by `require_admin`.

### Patients

CRUD on `Patient` and `MedicalRecord`:

- `POST /`, `GET /` (paginated), `GET /{id}`, `PATCH /{id}`, `DELETE /{id}` — patients.
- `POST /{patient_id}/medical-records` — add a record (16 features + 4 comorbidities + notes).
- `GET /{patient_id}/medical-records` — skip/limit pagination.
- `GET /{patient_id}/medical-records/{record_id}` — fetch one.

### Similar Patients (doctor only)

- `POST /search` — tabular results with body-driven filters (`patient_id` *or*
  `medical_record_id`, `treatment_filter`, `min_similarity`, `limit`) plus query-param
  pagination. The service slices the full Neo4j result client-side so total counts are
  accurate.
- `POST /search/graph` — graph-shaped results for visualisation. Adds the patient's `gender`
  to the profile (graph view renders the "49F" pill).
- `GET /{case_id}` — full detail of a single historical case.

The Cypher query computes a weighted score: `0.7 · clinical_similarity + 0.3 ·
comorbidity_jaccard`, filtered by age group + Hba1c severity + bounded distance on
`hba1c_baseline`/`c_peptide`. If Neo4j is not connected, every endpoint returns a 503 with
`code=NEO4J_UNAVAILABLE`.

### Inference (stateless)

- `POST /predict` — full `PredictionResult`, no LLM, no DB writes.
- `POST /predict-with-explanation` — same but with a Gemini explanation
  (`explain="require"`). 503 if the LLM is misconfigured.
- `POST /predict-batch` — list of `PatientInput` (max 50, declared on the DTO); failures come
  back per-row as rejected sentinels.

### Predictions (persisted)

- `POST /` — given a `(patient_id, medical_record_id)`, fetch the record, build a
  `PatientInput`, call `engine.apredict(..., explain="require")`, map the result onto a
  `Prediction` row and save it. The mapper is in `predictions/internal/prediction_mapper.py`
  and explains the JSONB layout choices in detail.
- `PATCH /{id}/decision` — record `ACCEPTED` / `OVERRIDDEN`. Override requires
  `final_treatment`.
- `GET /{id}` — single prediction with full payload.
- `GET /patient/{patient_id}` — paginated history per patient.

### Simulations (admin)

- `POST /` — multipart upload of a CSV (`.csv`, UTF-8, ≤20 MB, 100–50,000 rows). Form fields
  set `initial_epsilon`, `epsilon_decay`, `min_epsilon`, `random_seed`, `reset_posterior`.
  Returns the new simulation row immediately; the run starts as a background asyncio task.
- `GET /{id}/stream` — Server-Sent Events. If the simulation is still running and registered
  in the in-memory registry, the connection is **live**; otherwise the controller falls back
  to **DB replay** in 500-step chunks. Supports `?last_step=N` for client-side reconnects.
- `POST /{id}/cancel` — flips a flag and cancels the asyncio task. The runner sees the flag,
  flushes whatever is buffered, marks `CANCELLED`, and emits a final SSE complete frame.
- `GET /` — list (paginated) · `GET /{id}` · `GET /{id}/steps` (paginated) · `DELETE /{id}`.

---

## Database schema

All entities inherit from `BaseModel` (`src/shared/database/base_model.py`):

- `id: UUID` — server-generated via `gen_random_uuid()`.
- `created_at: TIMESTAMPTZ` — `DEFAULT now()`.
- `updated_at: TIMESTAMPTZ` — `DEFAULT now()` and `ON UPDATE now()`.

```mermaid
erDiagram
    USERS ||--o{ TOKENS : owns
    USERS ||--o{ PREDICTIONS : "creates (created_by)"
    PATIENTS ||--o{ MEDICAL_RECORDS : has
    PATIENTS ||--o{ PREDICTIONS : about
    MEDICAL_RECORDS ||--o{ PREDICTIONS : "snapshot for"
    SIMULATIONS ||--o{ SIMULATION_STEPS : produced

    USERS {
        UUID id PK
        string email UK
        string username UK
        string first_name
        string last_name
        string password_hash
        enum role "ADMIN | DOCTOR"
        bool is_active
    }
    TOKENS {
        UUID id PK
        UUID user_id FK
        text token UK
        enum token_type "ACCESS | REFRESH"
        bool is_revoked
        timestamptz expires_at
    }
    PATIENTS {
        UUID id PK
        string first_name
        string last_name
        date date_of_birth
        string gender
        string email "nullable, UK"
        string phone "nullable"
        string address "nullable"
    }
    MEDICAL_RECORDS {
        UUID id PK
        UUID patient_id FK
        int age
        float bmi
        float hba1c_baseline
        float egfr
        float diabetes_duration
        float fasting_glucose
        float c_peptide
        int cvd
        int ckd
        int nafld
        int hypertension
        float bp_systolic
        float ldl
        float hdl
        float triglycerides
        float alt
        string notes "nullable"
    }
    PREDICTIONS {
        UUID id PK
        UUID medical_record_id FK
        UUID patient_id FK
        UUID created_by FK
        string recommended_treatment
        int recommended_idx
        int confidence_pct
        string confidence_label
        float mean_gap
        string runner_up
        float runner_up_win_rate
        jsonb win_rates
        jsonb posterior_means
        string safety_status
        jsonb safety_details
        jsonb fairness "engine extras"
        text explanation_summary
        text explanation_runner_up
        text explanation_confidence
        text explanation_safety
        text explanation_monitoring
        text explanation_disclaimer
        enum doctor_decision "PENDING | ACCEPTED | OVERRIDDEN"
        string final_treatment "nullable"
        text doctor_notes "nullable"
    }
    SIMULATIONS {
        UUID id PK
        float initial_epsilon
        float epsilon_decay
        float min_epsilon
        int random_seed
        bool reset_posterior
        string dataset_filename
        int dataset_row_count
        enum status "PENDING | RUNNING | COMPLETED | FAILED | CANCELLED"
        int current_step
        string error_message "nullable"
        float final_accuracy "nullable"
        float final_cumulative_reward "nullable"
        float final_cumulative_regret "nullable"
        float mean_reward "nullable"
        float mean_regret "nullable"
        float thompson_exploration_rate "nullable"
        json treatment_counts "nullable"
        json confidence_distribution "nullable"
        json safety_distribution "nullable"
    }
    SIMULATION_STEPS {
        UUID id PK
        UUID simulation_id FK
        int step_number
        float epsilon
        json patient_context
        json oracle_rewards
        string optimal_treatment
        float optimal_reward
        string selected_treatment
        int selected_idx
        json posterior_means
        json win_rates
        int confidence_pct
        string confidence_label
        json sampled_values
        string runner_up
        float runner_up_winrate
        float mean_gap
        bool thompson_explored
        bool epsilon_explored
        string posterior_mean_best
        float observed_reward
        float instantaneous_regret
        bool matched_oracle
        string safety_status
        json safety_contraindications
        json safety_warnings
        float cumulative_reward
        float cumulative_regret
        float running_accuracy
        json treatment_counts
        json running_estimates
    }
```

Note the deliberate naming oddity in `predictions`: the `fairness` JSONB column is currently
**reused** as the engine-extras envelope (`model_top_treatment`, `attribution`, `contrast`,
`uncertainty_drivers`) for new rows, while still readable for legacy rows. The mapper
docstring (`predictions/internal/prediction_mapper.py`) documents this as a deferred rename.

Cascade behaviour:

- `tokens.user_id` → `users.id` `ON DELETE CASCADE`.
- `medical_records.patient_id` → `patients.id` `ON DELETE CASCADE`.
- `predictions.medical_record_id` and `predictions.patient_id` → `ON DELETE CASCADE`.
- `predictions.created_by` → `users.id` (no cascade — predictions are kept).
- `simulation_steps.simulation_id` → `simulations.id` `ON DELETE CASCADE`.

`Patient.medical_records` uses `lazy="selectin"` so listing patients still pulls each
patient's records efficiently; `Simulation.steps` uses `lazy="noload"` because steps tables
get *very* big and are paged through dedicated endpoints.

---

## Migrations

Alembic is configured for **async**: `alembic/env.py` builds an `async_engine_from_config`,
imports every model module so SQLAlchemy's metadata is complete, and pulls the DB URL from
`src.configs.database.url`.

Linear chain (oldest → newest):

```
e1bb89a5ff07  create users and tokens tables
3c9a6c24a2c1  create patients and medical_records tables
29ddf149906c  create predictions table
818587a0958c  add simulations and simulation_steps tables
```

```bash
alembic upgrade head           # apply
alembic downgrade -1           # roll one back
alembic revision --autogenerate -m "your change"
```

> **Note.** The `MODEL_PATH` in the example `.env` is currently a Windows-absolute path.
> Migrations don't depend on it, but the inference engine does — set it to a real directory
> on your machine before booting the app.

---

## Response envelope and errors

Every endpoint returns the same envelope. `value` and `error` are mutually exclusive
(enforced by a model validator on `ApiResponse`).

```jsonc
// success
{
  "success": true,
  "message": "Patient created",
  "value": { /* domain DTO */ }
}

// failure
{
  "success": false,
  "message": "This email is already registered",
  "error": {
    "title": "Registration Failed",
    "code": "EMAIL_EXISTS",
    "status": 409,
    "fieldErrors": { "email": ["Email already registered"] }
  }
}

// paginated
{
  "success": true,
  "value": [ /* items */ ],
  "pagination": { "page": 1, "total": 42, "pageSize": 20, "totalPages": 3 }
}
```

`ErrorDetail` supports a builder for accumulating field errors:

```python
err = ErrorDetail.builder("Update Failed", "USERNAME_EXISTS", 409)
err.add_field_error("username", "Username already taken")
raise ConflictException(message="...", error_detail=err.build())
```

### Exception hierarchy

```
AppException (500 default)
├── BadRequestException             400  BAD_REQUEST
├── ValidationException             400  VALIDATION_ERROR
├── AuthenticationException         401  AUTH_FAILED
├── AuthorizationException          403  FORBIDDEN
├── NotFoundException               404  NOT_FOUND
├── ConflictException               409  CONFLICT
├── InternalServerException         500  INTERNAL_ERROR
└── ServiceUnavailableException     503  SERVICE_UNAVAILABLE
```

Global handlers in `src/shared/exceptions/error_handlers.py` translate:

| Exception | HTTP | Code |
|---|---|---|
| `AppException` and subclasses | from `error_detail.status` | from `error_detail.code` |
| `RequestValidationError` (FastAPI) / `ValidationError` (Pydantic) | 400 | `VALIDATION_ERROR` |
| `inference.ValidationError` | 422 | `INFERENCE_VALIDATION_ERROR` |
| `inference.ConfigurationError` | 503 | `INFERENCE_CONFIGURATION_ERROR` |
| `inference.ModelError` | 500 | `INFERENCE_MODEL_ERROR` |
| `inference.ExplanationError` | 500 (or 503 if API-key flavour) | `INFERENCE_EXPLANATION_ERROR` / `LLM_UNAVAILABLE` |
| `inference.InferenceError` (catch-all) | 500 | `INFERENCE_ERROR` |
| `StarletteHTTPException` | echoed | derived from `detail` |
| Unhandled `Exception` | 500 | `INTERNAL_ERROR` |

---

## Simulation streaming model

The simulator is the most behaviourally rich piece of the codebase. It marries an in-memory
pub/sub registry with durable DB writes so a client can join late, drop, reconnect, and
still get a complete frame-by-frame replay.

```mermaid
sequenceDiagram
    autonumber
    participant Admin as Admin (POST /simulations)
    participant Ctrl as Controller
    participant Svc as SimulationService
    participant Run as run_simulation()
    participant Reg as SimulationRegistry (in-memory)
    participant Eng as per-sim InferenceEngine
    participant DB as Postgres
    participant SSE as Live SSE client
    participant Late as Reconnecting client

    Admin->>Ctrl: CSV + epsilon params
    Ctrl->>Ctrl: validate file size + UTF-8 + headers + rows (PatientInput)
    Ctrl->>Svc: create_simulation()
    Svc->>DB: INSERT simulation (status=PENDING)
    Ctrl->>Run: asyncio.create_task(run_simulation, ...)
    Ctrl-->>Admin: 200 + Simulation row

    Run->>Reg: register(sim_id)
    Run->>Eng: from_config + (optional) reset_posterior()
    Run->>DB: UPDATE status=RUNNING
    loop n_patients
        Run->>Run: epsilon decay, oracle_rewards (noise=False)
        Run->>Eng: stream.astep(patient, oracle_vector)
        Eng-->>Run: LearningStepEvent
        Run->>Reg: publish(step) → live subscribers
        Run->>DB: buffer; flush every 100 steps
    end
    Run->>DB: save final aggregates + status=COMPLETED
    Run->>Reg: publish(complete) + cleanup

    SSE->>Ctrl: GET /{id}/stream
    Ctrl->>Reg: subscribe(sim_id)
    Reg-->>Ctrl: queue (with replayed history)
    loop forever
        Reg-->>Ctrl: step/complete/error/ping
        Ctrl-->>SSE: SSE frame
    end

    Late->>Ctrl: GET /{id}/stream?last_step=N
    Ctrl->>Reg: subscribe → None (sim already complete)
    Ctrl->>DB: paged step replay (chunks of 500)
    Ctrl-->>Late: step frames + final complete frame
```

Important properties of the runner (`simulations/internal/simulation_runner.py`):

- **Per-simulation engine** isolates the posterior; `reset_posterior=True` reverts to the
  prior while keeping the trained backbone.
- **Action selection is owned by `LearningStream`** (Thompson draw → argmax → observe →
  online update). The legacy ε-greedy hook was dropped — `epsilon` is still computed and
  persisted for the UI's decay chart, but `epsilon_explored` is always `False`. Exploration
  is reported via the event's `explored` flag (`thompson_explored` in the row).
- **Oracle rewards** are noise-free; the previous noisy-observation / noise-free-regret split
  was collapsed because `LearningStream.astep` derives both signals from the single oracle
  vector you pass in.
- **Batched writes** every 100 steps. Up to 3 consecutive flush failures abort the run; the
  final flush retries 3× with exponential backoff before raising.
- **Registry caps**: 20 SSE subscribers per simulation, 5,000 events of in-memory replay
  history (later joiners fall back to DB replay), 1-hour TTL swept every 10 minutes plus once
  on every new registration.
- **Process-local only.** Documented explicitly — horizontal scaling needs Redis pub/sub or
  similar to replace the registry.

The reward oracle (`simulations/internal/reward_oracle.py`) is a **simulation-owned** synthetic
ground truth — vendored from the original ML notebook so behaviour matches exactly. It is
deliberately not imported from `shared/inference`: the engine accepts oracle vectors as input,
so the simulator owns the arm ordering and noise model.

---

## Adding a new module

1. Create `src/modules/<name>/` with `domain/`, `internal/`, `presentation/`.
2. Define the ORM model (`domain/models/<entity>.py`) extending `BaseModel`.
3. Add a repository (`domain/repositories/...`) extending `BaseRepository[Entity]`.
4. Write the service (`domain/services/...`) — accept primitives, return entities, raise
   `AppException` subclasses.
5. Add request / response DTOs (`presentation/dtos/`).
6. Add a controller (`presentation/controllers/...`) with an `APIRouter`. Use module-level
   `dependencies=[Depends(...)]` for the auth/role gate.
7. Wire a service factory in `presentation/dependencies.py`.
8. Re-export the router from the module's `__init__.py`.
9. Register it in `src/core/factory.py::_register_routers` and `alembic/env.py` (so
   `--autogenerate` sees the new model).
10. `alembic revision --autogenerate -m "..."` then `alembic upgrade head`.

---

## Operational notes

- **Logs** stream to stdout *and* to `logs/metis.log` (`logging.file_path` from YAML).
- **Engine readiness** is reported by `GET /health`:

  ```jsonc
  { "status": "ok", "engine": { "ready": true, "snapshot": { /* model paths, versions, etc */ } } }
  ```

  `engine.snapshot()` returns model/pipeline versions (sha1 of path+mtime+size), feature
  names, online-retraining flags and drift settings — useful for sanity-checking which model
  is loaded.
- **CORS origins** are read from `CORS_ORIGINS` as a comma-separated list. The shipped `.env`
  includes the typical Vite / Next.js / JetBrains live-server ports.
- **The `*.pyi` config stub is git-ignored** and regenerated on every boot. Do not edit it.
- **JWTs are revocable** because every check hits `tokens` — there's no asymmetric perf
  benefit, but it lets `logout` and `login` reliably invalidate stolen tokens.
- **The model artefacts are not versioned** (`.gitignore` excludes `*.pt`, `*.joblib`,
  `*.pkl`, `*.h5`, `*.onnx`). Distribute them out-of-band and set `MODEL_PATH` accordingly.

For a per-endpoint reference (request bodies, response shapes, error codes, SSE frame layout)
see [`docs/API_INTEGRATION.md`](docs/API_INTEGRATION.md).

---

## How this repo connects to the rest of Metis

```mermaid
flowchart LR
    UI["metis_ui<br/>Next.js desktop app"]
    API["metis-fast-api<br/>(this repo)"]
    Bandits["diabetes-bandits<br/>research + inference façade"]
    Artefacts[("models/<br/>neural_thompson.pt<br/>feature_pipeline.joblib")]
    PG[("PostgreSQL")]
    Neo[("Neo4j historical cohort")]
    LLM[("Gemini")]

    UI -- "REST + JWT (camelCase responses)" --> API
    UI -- "SSE: GET /simulations/:id/stream" --> API
    Bandits -- "produces" --> Artefacts
    Bandits -- "vendored: bandits/inference/<br/>→ src/shared/inference/" --> API
    API -- "loads at startup" --> Artefacts
    API --> PG
    API --> Neo
    API -. "explanations (optional)" .-> LLM
```

### What flows in and out

| Other repo | Direction | What is exchanged |
|---|---|---|
| [`metis_ui`](https://github.com/kudzaiprichard/metis_ui) | inbound HTTP + SSE | Bearer-JWT REST calls (`/api/v1/auth`, `/patients`, `/predictions`, `/inference`, `/similar-patients`, `/simulations`); SSE stream of `LearningStepEvent` frames during simulations; multipart CSV upload to start a simulation. The UI relies on the `{success, value, error}` envelope and the machine-readable `error.code` taxonomy described in [Response envelope and errors](#response-envelope-and-errors). |
| [`diabetes-bandits`](https://github.com/kudzaiprichard/diabetes-bandits) | inbound import (vendored) | The contents of `bandits/inference/` are vendored as `src/shared/inference/` and treated as a frozen library — `src/shared/inference_bootstrap.py` is the only seam that translates Metis YAML config into `InferenceConfig`. Trained `neural_thompson.pt` + `feature_pipeline_scaled.joblib` are produced by `python -m src.cli train` in that repo and loaded here at startup via `MODEL_PATH` / `MODEL_FILE` / `PIPELINE_FILE`. |

### Keeping the boundary stable

- The **wire-contract constants** (5 treatments + 16 features, in fixed order) are owned by
  `bandits/inference/_internal/constants.py` and re-exported from
  `src/shared/inference/_internal/constants.py`. Both copies must change together — the UI's
  CSV validator (`features/bandit-demo/lib/csv-validation.ts` in `metis_ui`) mirrors the
  same list byte-for-byte.
- When upgrading the bandits version, copy the whole `inference/` tree across, **diff
  `src/shared/inference/__init__.py` against the upstream** to confirm the public surface
  hasn't drifted, run Alembic + tests, and re-deploy.

---

## Related Repositories

| Repo | Role | One-line description |
|---|---|---|
| [`metis-fast-api`](https://github.com/kudzaiprichard/metis-fast-api) | Backend (this repo) | FastAPI service — auth, patients, predictions, similar-cases, bandit simulator + SSE |
| [`diabetes-bandits`](https://github.com/kudzaiprichard/diabetes-bandits) | ML research + inference | Neural-Thompson contextual bandit, training CLI, OPE, and the inference façade vendored here as `src/shared/inference/` |
| [`metis_ui`](https://github.com/kudzaiprichard/metis_ui) | Desktop client | Next.js 16 / React 19 UI — clinician + admin surfaces; consumes this API over REST + SSE |
