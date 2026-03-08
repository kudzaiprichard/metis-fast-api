# Metis

A modular FastAPI clinical decision support system for Type 2 Diabetes treatment selection. Uses a Contextual Bandit (Neural Thompson Sampling) to recommend personalised treatments, with async SQLAlchemy, JWT authentication, role-based access control, and a config system driven by YAML and environment variables.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Architecture & Design](#architecture--design)
- [Module Structure](#module-structure)
- [Auth Module](#auth-module)
- [Patients Module](#patients-module)
- [Models Module](#models-module)
- [Predictions Module](#predictions-module)
- [Adding a New Module](#adding-a-new-module)
- [Shared Layer](#shared-layer)
- [Database Migrations](#database-migrations-alembic)
- [API Response Format](#api-response-format)

---

## Project Structure

```
├── main.py
├── .env
├── requirements.txt
├── src/
│   ├── configs/
│   │   ├── __init__.py            # Auto-loads config on import
│   │   ├── application.yaml       # Central config file
│   │   ├── loader.py              # YAML + env var resolver
│   │   └── generate.py            # .pyi stub generator for IDE support
│   ├── core/
│   │   ├── __init__.py
│   │   ├── factory.py             # FastAPI app factory
│   │   ├── lifespan.py            # Startup/shutdown lifecycle
│   │   └── middleware.py          # CORS, request logging
│   ├── shared/
│   │   ├── __init__.py
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── engine.py          # Async SQLAlchemy engine + session
│   │   │   ├── base_model.py      # Base model (UUID pk, timestamps)
│   │   │   ├── repository.py      # Generic CRUD repository
│   │   │   └── dependencies.py    # get_db, get_db_readonly
│   │   ├── responses/
│   │   │   ├── __init__.py
│   │   │   └── api_response.py    # ApiResponse, PaginatedResponse, ErrorDetail
│   │   └── exceptions/
│   │       ├── __init__.py
│   │       ├── exceptions.py      # AppException hierarchy
│   │       └── error_handlers.py  # Global FastAPI error handlers
│   └── modules/
│       ├── auth/                  # Authentication & user management
│       ├── patients/              # Patient records & medical data
│       ├── models/                # ML inference (stateless)
│       └── predictions/           # Clinical workflow & decision tracking
```

---

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL

### Installation

```bash
git clone <repo-url>
cd metis
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Key Dependencies

```
fastapi
uvicorn
sqlalchemy[asyncio]
asyncpg
pydantic[email]
python-dotenv
pyyaml
PyJWT
bcrypt
torch
joblib
scikit-learn
numpy
pandas
loguru
google-genai
```

### Environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required variables:
- `DATABASE_URL` — PostgreSQL async connection string
- `JWT_SECRET_KEY` — Secret key for JWT signing
- `GEMINI_API_KEY` — Google Gemini API key for LLM explanations

### Database

```sql
CREATE DATABASE metisdb;
```

```bash
alembic upgrade head
```

### ML Model Files

Place your trained model files in the configured model directory (default: `models/`):
- `neural_thompson.pt` — trained NeuralThompson model
- `feature_pipeline_scaled.joblib` — fitted FeaturePipeline

### Run

```bash
python main.py
```

The app starts at `http://127.0.0.1:8000`. Swagger docs at `/docs`.

On startup:
1. Load and validate all configuration
2. Generate IDE stubs for config autocomplete
3. Initialize the database connection pool
4. Seed a default admin user (if none exists)
5. Start the background token cleanup task

---

## Configuration

All configuration lives in `src/configs/application.yaml` using a pipe-delimited format:

```yaml
pool_pre_ping: "true | bool"                          # Static value
pool_size: "${DB_POOL_SIZE:5} | int"                   # Env var with default
url: "${DATABASE_URL} | str | required"                # Required env var
```

Supported types: `str`, `int`, `float`, `bool`, `list` (comma-separated).

Access config anywhere:

```python
from src.configs import database, security, application, model, gemini

database.url
security.jwt.secret_key
model.path
gemini.api_key
```

---

## Architecture & Design

### Principles

- **Modular** — Each feature is a self-contained module under `src/modules/`
- **Layered** — Domain → Internal → Presentation
- **Dependency injection** — Services receive repositories via constructor; FastAPI `Depends()` wires everything
- **Clean boundaries** — Services accept plain arguments and return domain objects
- **Consistent responses** — All endpoints return `ApiResponse` or `PaginatedResponse`

### Request Flow

```
HTTP Request
  → Controller (converts DTO → args)
    → Service (business logic, raises exceptions)
      → Repository (data access)
    ← Service returns domain object
  ← Controller (converts domain → response DTO)
  ← ApiResponse wrapper
```

### Transaction Management

The `get_db` dependency wraps each request in a single transaction. Auto-commits on success, rolls back on exception.

---

## Module Structure

Every module follows the same three-layer structure:

```
src/modules/<module_name>/
├── __init__.py
├── domain/
│   ├── models/        # SQLAlchemy models and enums
│   ├── repositories/  # Data access (extends BaseRepository)
│   └── services/      # Business logic
├── internal/          # Module-specific utilities
└── presentation/
    ├── dependencies.py
    ├── dtos/          # Request/response Pydantic models
    └── controllers/   # FastAPI route handlers
```

---

## Auth Module

Handles authentication, authorization, and user management.

### Features
- JWT access + refresh tokens stored in database
- Tokens verified against DB on every request (revocation check)
- Login revokes all previous tokens, refresh rotates both
- Role-based access via `require_role()` dependency
- Background task purges expired tokens
- Default admin seeded on startup

### Endpoints

**Auth** (`/api/v1/auth`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | /register | No | Register new user |
| POST | /login | No | Login, get tokens |
| POST | /refresh | No | Refresh token pair |
| POST | /logout | Bearer | Revoke all tokens |
| GET | /me | Bearer | Get current profile |
| PATCH | /me | Bearer | Update profile |

**User Management** (`/api/v1/users`) — Admin only

| Method | Path | Description |
|--------|------|-------------|
| GET | / | List users (paginated) |
| GET | /{user_id} | Get user by ID |
| POST | / | Create user |
| PATCH | /{user_id} | Update user |
| DELETE | /{user_id} | Delete user |

### Using Auth in Other Modules

```python
from src.modules.auth.presentation.dependencies import get_current_user, require_role, require_admin
from src.modules.auth.domain.models.enums import Role

@router.get("/resource")
async def get_resource(user: User = Depends(get_current_user)): ...

@router.post("/resource")
async def create_resource(user: User = Depends(require_role(Role.DOCTOR))): ...

@router.delete("/resource/{id}")
async def delete_resource(_admin: User = Depends(require_admin)): ...
```

---

## Patients Module

Manages patient records and time-series medical data.

A patient has demographics (name, DOB, gender, contact info) and many medical records over time. Each medical record captures 16 clinical features at a point in time: age, BMI, HbA1c, eGFR, diabetes duration, fasting glucose, C-peptide, CVD, CKD, NAFLD, hypertension, BP systolic, LDL, HDL, triglycerides, ALT.

Medical data changes between visits — the patient doesn't. That's why they're separate entities.

### Entities
- `Patient` — demographics, contact info
- `MedicalRecord` — 16 clinical features, linked to patient, timestamped

### Endpoints (`/api/v1/patients`) — ADMIN, DOCTOR

| Method | Path | Description |
|--------|------|-------------|
| POST | / | Register patient |
| GET | / | List patients (paginated) |
| GET | /{id} | Get patient with medical history |
| PATCH | /{id} | Update patient demographics |
| DELETE | /{id} | Delete patient |
| POST | /{id}/medical-records | Add medical record |
| GET | /{id}/medical-records | List medical records |
| GET | /{id}/medical-records/{record_id} | Get single record |

---

## Models Module

Stateless ML inference utility. Loads the NeuralThompson model and FeaturePipeline from disk, runs predictions, returns results. No database, no state between requests.

### What It Does
1. Loads model + pipeline fresh per request from configured paths
2. Transforms patient features via FeaturePipeline
3. Runs ExplainabilityExtractor → prediction + confidence + safety + fairness
4. Runs LLMExplainer → Gemini-powered clinical explanation
5. Returns the full payload

### Internal Files
- `constants.py` — treatment constants (TREATMENTS, N_TREATMENTS, IDX_TO_TREATMENT, CONTEXT_FEATURES)
- `neural_bandit.py` — NeuralThompson model (from ML project, imports updated)
- `feature_engineering.py` — FeaturePipeline (from ML project, imports updated)
- `explainability.py` — ExplainabilityExtractor + safety checks + fairness (from ML project, imports updated)
- `llm_explainer.py` — LLMExplainer + Gemini prompt (from ML project, imports updated)
- `model_loader.py` — loads pipeline, model, extractor from config paths
- `inference_engine.py` — orchestrates: predict(), predict_with_explanation(), explain(), predict_batch()

### Endpoints (`/api/v1/inference`) — ADMIN, DOCTOR

| Method | Path | Description |
|--------|------|-------------|
| POST | /predict | Prediction + confidence + safety (instant) |
| POST | /predict-with-explanation | Prediction + Gemini explanation |
| POST | /explain | Standalone explanation from existing payload |
| POST | /predict-batch | Batch predictions, max 50 |

These endpoints are for **standalone testing** with raw features. The clinical workflow uses the predictions module instead.

---

## Predictions Module

The core clinical workflow. Connects patients, ML model, and persistence.

### How It Works
1. Doctor selects a patient and medical record
2. `POST /api/v1/predictions` with medical record ID and patient ID
3. Service fetches the medical record from the database
4. `medical_record_mapper` converts the MedicalRecord to a 16-feature context dict
5. `inference_engine.predict_with_explanation()` runs prediction + Gemini explanation
6. Full result stored in the `Prediction` table (recommendation, confidence, safety, explanation, all structured)
7. Doctor reviews and calls `PATCH /decision` to accept or override

### Entities
- `Prediction` — linked to medical record, patient, and doctor. Stores model recommendation (treatment, confidence, win rates, posterior means), safety details, fairness, LLM explanation (6 structured text fields), and doctor's final decision.
- `DoctorDecision` enum — PENDING, ACCEPTED, OVERRIDDEN

### Key Design Decisions
- Explanation is **always generated** with the prediction — no separate call, no extra Gemini cost on history retrieval
- Explanation stored as 6 structured text columns, not JSONB — queryable and type-safe
- Safety details and fairness use JSONB since they have variable-length lists
- Doctor's final decision stored alongside the recommendation — captures acceptance rate data for model improvement

### Endpoints (`/api/v1/predictions`) — ADMIN, DOCTOR

| Method | Path | Description |
|--------|------|-------------|
| POST | / | Run prediction + explanation on medical record |
| GET | /{id} | Get prediction with full details |
| PATCH | /{id}/decision | Doctor accepts or overrides |
| GET | /patient/{patient_id} | Prediction history (paginated) |

### Database Relationships

```
patients (1) ──→ (many) medical_records (1) ──→ (many) predictions
                                                         ↑
                                                    users (doctor)
```

---

## Adding a New Module

### 1. Create the directory structure

```
src/modules/<module_name>/
├── __init__.py
├── domain/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── <model>.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── <model>_repository.py
│   └── services/
│       ├── __init__.py
│       └── <module>_service.py
├── internal/
│   └── __init__.py
└── presentation/
    ├── __init__.py
    ├── dependencies.py
    ├── dtos/
    │   ├── __init__.py
    │   ├── requests.py
    │   └── responses.py
    └── controllers/
        ├── __init__.py
        └── <module>_controller.py
```

### 2. Create your model

```python
from src.shared.database import BaseModel
class Appointment(BaseModel):
    __tablename__ = "appointments"
    title: Mapped[str] = mapped_column(String(255), nullable=False)
```

### 3. Create the repository

```python
from src.shared.database import BaseRepository
class AppointmentRepository(BaseRepository[Appointment]):
    def __init__(self, session):
        super().__init__(Appointment, session)
```

### 4. Create the service

```python
class AppointmentService:
    def __init__(self, repo: AppointmentRepository):
        self.repo = repo
    async def get_appointment(self, id: UUID) -> Appointment:
        appointment = await self.repo.get_by_id(id)
        if not appointment:
            raise NotFoundException(message="Appointment not found")
        return appointment
```

### 5. Create dependencies

```python
def get_appointment_service(session = Depends(get_db)):
    return AppointmentService(AppointmentRepository(session))
```

### 6. Register the router in `src/core/factory.py`

```python
from src.modules.appointments import appointment_router
app.include_router(appointment_router, prefix="/api/v1/appointments", tags=["Appointments"])
```

---

## Shared Layer

The shared layer provides reusable building blocks. It never imports from any module.

**Database** — `BaseModel` (UUID pk, created_at, updated_at), `BaseRepository` (CRUD, pagination, filtering), `get_db` / `get_db_readonly` session dependencies.

**Responses** — `ApiResponse` for single results, `PaginatedResponse` for lists, `ErrorDetail` with builder pattern for structured errors.

**Exceptions** — `AppException` base class with subclasses: `NotFoundException`, `ValidationException`, `AuthenticationException`, `AuthorizationException`, `ConflictException`, `BadRequestException`, `InternalServerException`, `ServiceUnavailableException`. All automatically handled by global error handlers.

---

## Database Migrations (Alembic)

Alembic tracks and applies database schema changes. Migration files live in `alembic/versions/` and should always be committed to git.

### Common Commands

```bash
alembic upgrade head                                    # Apply all pending migrations
alembic revision --autogenerate -m "describe change"    # Generate migration from model changes
alembic downgrade -1                                    # Rollback one migration
alembic history                                         # View migration history
alembic current                                         # See current state
```

### Workflow

1. Make changes to SQLAlchemy models
2. Import the model in `alembic/env.py`
3. Run `alembic revision --autogenerate -m "describe change"`
4. Review the generated migration file
5. Run `alembic upgrade head`

### Adding Models from New Modules

Add imports to `alembic/env.py`:

```python
from src.modules.auth.domain.models.user import User
from src.modules.auth.domain.models.token import Token
from src.modules.patients.domain.models.patient import Patient
from src.modules.patients.domain.models.medical_record import MedicalRecord
from src.modules.predictions.domain.models.prediction import Prediction
```

---

## API Response Format

### Success

```json
{
  "success": true,
  "message": "Operation successful",
  "value": { ... }
}
```

### Paginated

```json
{
  "success": true,
  "value": [ ... ],
  "pagination": {
    "page": 1,
    "total": 45,
    "pageSize": 20,
    "totalPages": 3
  }
}
```

### Error

```json
{
  "success": false,
  "message": "Human-friendly message",
  "error": {
    "title": "Not Found",
    "code": "USER_NOT_FOUND",
    "status": 404,
    "details": ["No user found with id ..."]
  }
}
```

### Validation Error

```json
{
  "success": false,
  "message": "Please check your input and try again",
  "error": {
    "title": "Validation Failed",
    "code": "VALIDATION_ERROR",
    "status": 400,
    "fieldErrors": {
      "body.email": ["value is not a valid email address"],
      "body.password": ["String should have at least 8 characters"]
    }
  }
}
```