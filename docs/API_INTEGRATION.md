# API Integration Guide

A self-contained reference for integrating a frontend with the Metis API. Every endpoint, header, parameter, payload, status code, enum, SSE frame, and error shape the frontend needs to handle is documented below.

---

## 1. Conventions

### 1.1 Base URL

All endpoints are served under:

```
http://<host>:<port>
```

The default local server runs on `http://127.0.0.1:8000`. Health check aside, every business endpoint lives beneath `/api/v1/...`.

### 1.2 Content types

- All JSON request bodies **must** be sent with `Content-Type: application/json`.
- The **start simulation** endpoint is the single exception — it takes `multipart/form-data` because it uploads a CSV file.
- All responses are JSON, except the **simulation stream** endpoint which uses `text/event-stream` (Server-Sent Events).

### 1.3 Authentication

Authentication uses HTTP bearer tokens:

```
Authorization: Bearer <accessToken>
```

- **Unauthenticated endpoints** (no header required):
  - `GET /health`
  - `POST /api/v1/auth/register`
  - `POST /api/v1/auth/login`
  - `POST /api/v1/auth/refresh`
- **Every other endpoint** requires a valid bearer access token.
- Some endpoints also require a specific user role. Role requirements are noted per-endpoint. Roles are: `ADMIN` and `DOCTOR`.

If the token is missing, malformed, expired, or revoked, the API returns `401` with `code: "AUTH_FAILED"`.
If the token is valid but the user's role is insufficient, the API returns `403` with `code: "INSUFFICIENT_ROLE"`.

### 1.4 Response envelope

**Every JSON response** (success or failure) uses this envelope:

```jsonc
{
  "success": true,            // boolean — true on 2xx, false on errors
  "message": "Login successful", // optional human-readable message
  "value":   { /* ... */ },   // present on success; null/absent on error
  "error":   { /* ... */ },   // present on error; null/absent on success
  "pagination": { /* ... */ } // present only on paginated list endpoints
}
```

`value` and `error` are mutually exclusive. `value` may be an object, an array, `null`, or omitted entirely (for delete/logout-style endpoints).

### 1.5 Error envelope (`error`)

```jsonc
{
  "title":  "Validation Failed",       // short title
  "code":   "VALIDATION_ERROR",        // stable machine-readable code
  "status": 400,                        // mirrors the HTTP status
  "details": [                          // optional array of human-readable strings
    "email is not a valid email address"
  ],
  "fieldErrors": {                      // optional map of field → [error, ...]
    "body.email": ["value is not a valid email address"]
  }
}
```

#### Known error codes

| Code | HTTP | Meaning |
|---|---|---|
| `VALIDATION_ERROR` | 400 | Request body / query / path failed schema validation. `fieldErrors` lists offending fields. |
| `BAD_REQUEST` | 400 | Generic bad request (business rule). |
| `INVALID_FILE_TYPE` | 400 | Simulation upload was not a `.csv` file. |
| `FILE_TOO_LARGE` | 400 | Simulation CSV exceeds 20 MB. |
| `INVALID_ENCODING` | 400 | Simulation CSV was not UTF-8. |
| `CSV_VALIDATION_FAILED` | 400 | Simulation CSV headers or rows failed validation. `details` is a line-per-row list (capped at 20). |
| `AUTH_FAILED` | 401 | Missing, expired, or invalid token. |
| `FORBIDDEN` | 403 | Authenticated but not allowed to do this action. |
| `INSUFFICIENT_ROLE` | 403 | Role gate failed (e.g., endpoint requires `ADMIN`). |
| `NOT_FOUND` | 404 | Resource does not exist. |
| `CONFLICT` | 409 | Resource already exists, or state transition conflict (e.g., cancelling a non-running simulation — `SIMULATION_NOT_RUNNING`). |
| `SIMULATION_NOT_RUNNING` | 409 | Attempted to cancel a simulation that is not in `RUNNING`. |
| `INFERENCE_VALIDATION_ERROR` | 422 | Inference engine rejected the patient payload. `fieldErrors` contains engine-level field paths. |
| `INFERENCE_MODEL_ERROR` | 500 | Model failed during inference. |
| `INFERENCE_EXPLANATION_ERROR` | 500 | LLM explanation step failed (only possible with `/predict-with-explanation`). |
| `INFERENCE_ERROR` | 500 | Unclassified inference failure. |
| `INFERENCE_CONFIGURATION_ERROR` | 503 | Engine is misconfigured or not yet ready. |
| `SERVICE_UNAVAILABLE` | 503 | Transient downstream failure. |
| `INTERNAL_ERROR` | 500 | Unexpected server error. |
| `APP_ERROR` | 500 | Generic application error. |

Any HTTP-level fallback (e.g., 404/405 from the router) is mapped to an error envelope as well; the `code` is the detail string uppercased with spaces replaced by underscores.

### 1.6 Pagination

Paginated list endpoints accept two query parameters:

| Name | Type | Default | Range | Notes |
|---|---|---|---|---|
| `page` | integer | `1` | ≥ 1 | 1-indexed page number. |
| `pageSize` | integer | `20` | 1–100 | Items per page. |

The response envelope adds a `pagination` block:

```jsonc
{
  "success": true,
  "value": [ /* array of items */ ],
  "pagination": {
    "page": 1,
    "total": 57,
    "pageSize": 20,
    "totalPages": 3
  }
}
```

Two endpoints use a different, legacy shape:

- `GET /api/v1/patients/{patientId}/medical-records` uses `skip` (0+) and `limit` (1–100, default 50); the response is a plain `value: [...]` array (no `pagination` block).
- `GET /api/v1/simulations/{id}/stream` accepts a `last_step` cursor (see §10).

### 1.7 Field naming

Responses use **camelCase** for multi-word fields (`firstName`, `medicalRecordId`, `winRates`, etc.). Request bodies generally use **snake_case** (`first_name`, `medical_record_id`). A few request DTOs also accept camelCase aliases; the safe choice is snake_case everywhere for requests.

### 1.8 Timestamps

All timestamps are ISO-8601 strings in UTC, e.g. `2026-04-20T09:12:34.123456+00:00`.
Dates (without time) are ISO-8601 date strings, e.g. `1982-05-17`.

### 1.9 Identifiers

All resource IDs are UUID v4 strings unless explicitly noted otherwise. Similar-patient `caseId` and graph node/edge IDs are opaque strings, not UUIDs.

---

## 2. Enums reference

### 2.1 `Role`

- `ADMIN`
- `DOCTOR`

### 2.2 `Gender` (patient demographic field)

- `male`
- `female`
- `other`

### 2.3 `Treatment` (5 drug classes)

- `Metformin`
- `GLP-1`
- `SGLT-2`
- `DPP-4`
- `Insulin`

The integer index of a treatment (`selectedIdx`, `recommendedIdx`, `oracleOptimalIdx`, etc.) corresponds to its position in this list:

| Index | Treatment |
|---|---|
| 0 | Metformin |
| 1 | GLP-1 |
| 2 | SGLT-2 |
| 3 | DPP-4 |
| 4 | Insulin |

### 2.4 `ConfidenceLabel`

- `HIGH`
- `MODERATE`
- `LOW`

### 2.5 `SafetyStatus`

- `CLEAR` — no safety issues for the recommended treatment.
- `WARNING` — one or more non-blocking safety warnings.
- `CONTRAINDICATION_FOUND` — one or more contraindications; the recommended treatment was downgraded (check `override`).

### 2.6 `DoctorDecision`

- `PENDING` — default when a prediction is first created.
- `ACCEPTED` — doctor accepted the model's recommendation.
- `OVERRIDDEN` — doctor overrode with `finalTreatment`.

### 2.7 `SimulationStatus`

- `PENDING`
- `RUNNING`
- `COMPLETED`
- `FAILED`
- `CANCELLED`

### 2.8 `Phase` (simulation step phase, SSE only)

- `Early`
- `Mid`
- `Late`

### 2.9 Binary flags (patient medical fields)

`cvd`, `ckd`, `nafld`, `hypertension`, `medullary_thyroid_history`, `men2_history`, `pancreatitis_history`, `type1_suspicion` are strict `0` or `1` integers, not booleans.

---

## 3. Health

### `GET /health`

- **Auth:** none.
- **Query / body:** none.
- **Response 200:** plain JSON (no envelope):
  ```json
  {
    "status": "ok",
    "engine": {
      "ready": true,
      "snapshot": { /* opaque engine state */ }
    }
  }
  ```
  `engine.ready` is `false` while the inference engine is still loading; `engine.snapshot` is `null` in that case.

---

## 4. Auth — `/api/v1/auth`

Public endpoints for sign-in, sign-out, token refresh, and the current user's profile.

### 4.1 `POST /api/v1/auth/register`

- **Auth:** none.
- **Status:** `201 Created`.
- **Request body:**
  ```json
  {
    "email": "doctor@example.com",
    "username": "dr_smith",
    "first_name": "Alice",
    "last_name": "Smith",
    "password": "at-least-8-chars"
  }
  ```
  | Field | Type | Rules |
  |---|---|---|
  | `email` | string | Valid email. |
  | `username` | string | 3–100 chars. |
  | `first_name` | string | 1–100 chars. |
  | `last_name` | string | 1–100 chars. |
  | `password` | string | 8–128 chars. |
- **Response 201 `value`:** `AuthResponse` (see §4.6).
- **Errors:** `400 VALIDATION_ERROR`, `409 CONFLICT` (email/username taken).

### 4.2 `POST /api/v1/auth/login`

- **Auth:** none.
- **Request body:**
  ```json
  { "email": "doctor@example.com", "password": "..." }
  ```
- **Response 200 `value`:** `AuthResponse`.
- **Errors:** `400 VALIDATION_ERROR`, `401 AUTH_FAILED` (wrong credentials or inactive user).

### 4.3 `POST /api/v1/auth/refresh`

- **Auth:** none (the refresh token itself is the credential).
- **Request body:**
  ```json
  { "refresh_token": "<refreshToken>" }
  ```
- **Response 200 `value`:** `TokenResponse` (see §4.6).
- **Errors:** `401 AUTH_FAILED` (refresh token expired or revoked).

### 4.4 `POST /api/v1/auth/logout`

- **Auth:** bearer access token.
- **Request body:** none.
- **Response 200 `value`:** `null`, `message: "Logged out successfully"`.
- **Errors:** `401 AUTH_FAILED`.

### 4.5 `GET /api/v1/auth/me`  •  `PATCH /api/v1/auth/me`

- **Auth:** bearer access token.
- `GET` response 200 `value`: `UserResponse`.
- `PATCH` body (all optional, same rules as register):
  ```json
  { "first_name": "Alice", "last_name": "Smith-Jones", "username": "alices" }
  ```
- `PATCH` response 200 `value`: `UserResponse`, `message: "Profile updated"`.

### 4.6 Shared auth payloads

**`UserResponse`**
```jsonc
{
  "id":        "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "email":     "doctor@example.com",
  "username":  "dr_smith",
  "firstName": "Alice",
  "lastName":  "Smith",
  "role":      "DOCTOR",             // Role enum
  "isActive":  true,
  "createdAt": "2026-04-20T09:12:34.123456+00:00",
  "updatedAt": "2026-04-20T09:12:34.123456+00:00"
}
```

**`TokenResponse`**
```json
{ "accessToken": "eyJhbGc...", "refreshToken": "eyJhbGc..." }
```

**`AuthResponse`**
```json
{ "user": { /* UserResponse */ }, "tokens": { /* TokenResponse */ } }
```

---

## 5. User management — `/api/v1/users`

**All endpoints require the `ADMIN` role.** Responses use `UserResponse` (§4.6).

### 5.1 `GET /api/v1/users`

- **Query params:**
  | Name | Type | Notes |
  |---|---|---|
  | `page` | integer | Pagination (§1.6). |
  | `pageSize` | integer | Pagination (§1.6). |
  | `role` | string | Optional, `ADMIN` or `DOCTOR`. |
  | `is_active` | boolean | Optional. |
- **Response 200:** paginated `UserResponse[]`.

### 5.2 `GET /api/v1/users/{userId}`

- **Path:** `userId` (UUID).
- **Response 200 `value`:** `UserResponse`.
- **Errors:** `404 NOT_FOUND`.

### 5.3 `POST /api/v1/users`

- **Status:** `201 Created`.
- **Body:**
  ```json
  {
    "email": "doc@example.com",
    "username": "new_doc",
    "first_name": "Bob",
    "last_name": "Jones",
    "password": "at-least-8-chars",
    "role": "DOCTOR"
  }
  ```
  `role` must be `ADMIN` or `DOCTOR`.
- **Response 201 `value`:** `UserResponse`, `message: "User created"`.

### 5.4 `PATCH /api/v1/users/{userId}`

- **Body (all optional):**
  ```json
  {
    "first_name": "Bob",
    "last_name": "Jones",
    "username": "bobj",
    "role": "ADMIN",
    "is_active": false
  }
  ```
- **Response 200 `value`:** `UserResponse`, `message: "User updated"`.

### 5.5 `DELETE /api/v1/users/{userId}`

- **Response 200 `value`:** `null`, `message: "User deleted"`.

---

## 6. Patients — `/api/v1/patients`

**All endpoints require `ADMIN` or `DOCTOR`.**

### 6.1 `POST /api/v1/patients`

- **Status:** `201 Created`.
- **Body:**
  ```json
  {
    "first_name":    "Jane",
    "last_name":     "Doe",
    "date_of_birth": "1972-03-14",
    "gender":        "female",
    "email":         "jane.doe@example.com",
    "phone":         "+1-555-0123",
    "address":       "123 Main St"
  }
  ```
  `email`, `phone`, `address` are optional. `gender` must be `male | female | other`.
- **Response 201 `value`:** `PatientResponse` (see §6.7).

### 6.2 `GET /api/v1/patients`

- **Query params:** `page`, `pageSize` (§1.6).
- **Response 200:** paginated `PatientResponse[]`.

### 6.3 `GET /api/v1/patients/{patientId}`

- **Response 200 `value`:** `PatientDetailResponse` (includes `medicalRecords` array, §6.7).
- **Errors:** `404 NOT_FOUND`.

### 6.4 `PATCH /api/v1/patients/{patientId}`

- **Body:** same fields as `POST`, all optional.
- **Response 200 `value`:** `PatientResponse`, `message: "Patient updated"`.

### 6.5 `DELETE /api/v1/patients/{patientId}`

- **Response 200 `value`:** `null`, `message: "Patient deleted"`.

### 6.6 Medical records

#### `POST /api/v1/patients/{patientId}/medical-records`

- **Status:** `201 Created`.
- **Body — all fields required except `notes`:**
  ```json
  {
    "age": 52,
    "bmi": 31.4,
    "hba1c_baseline": 8.1,
    "egfr": 82.0,
    "diabetes_duration": 5.0,
    "fasting_glucose": 175.0,
    "c_peptide": 1.4,
    "cvd": 0,
    "ckd": 0,
    "nafld": 1,
    "hypertension": 1,
    "bp_systolic": 138.0,
    "ldl": 120.0,
    "hdl": 42.0,
    "triglycerides": 210.0,
    "alt": 34.0,
    "notes": "Optional free text, up to 1000 chars"
  }
  ```
  **Validation ranges:**

  | Field | Type | Min | Max |
  |---|---|---|---|
  | `age` | int | 18 | 120 |
  | `bmi` | float | 10 | 80 |
  | `hba1c_baseline` | float | 3 | 20 |
  | `egfr` | float | 5 | 200 |
  | `diabetes_duration` | float | 0 | 60 |
  | `fasting_glucose` | float | 50 | 500 |
  | `c_peptide` | float | 0 | 10 |
  | `cvd`, `ckd`, `nafld`, `hypertension` | int | 0 | 1 |
  | `bp_systolic` | float | 60 | 250 |
  | `ldl` | float | 20 | 400 |
  | `hdl` | float | 10 | 150 |
  | `triglycerides` | float | 30 | 800 |
  | `alt` | float | 5 | 500 |
  | `notes` | string? | — | 1000 chars |

- **Response 201 `value`:** `MedicalRecordResponse`, `message: "Medical record added"`.

#### `GET /api/v1/patients/{patientId}/medical-records`

- **Query params:**
  | Name | Type | Default | Range |
  |---|---|---|---|
  | `skip` | integer | 0 | ≥ 0 |
  | `limit` | integer | 50 | 1–100 |
- **Response 200 `value`:** plain array `MedicalRecordResponse[]` (no `pagination` block).

#### `GET /api/v1/patients/{patientId}/medical-records/{recordId}`

- **Response 200 `value`:** `MedicalRecordResponse`.
- **Errors:** `404 NOT_FOUND`.

### 6.7 Shared patient payloads

**`MedicalRecordResponse`**
```jsonc
{
  "id":                 "uuid",
  "patientId":          "uuid",
  "age":                52,
  "bmi":                31.4,
  "hba1cBaseline":      8.1,
  "egfr":               82.0,
  "diabetesDuration":   5.0,
  "fastingGlucose":     175.0,
  "cPeptide":           1.4,
  "cvd":                0,
  "ckd":                0,
  "nafld":              1,
  "hypertension":       1,
  "bpSystolic":         138.0,
  "ldl":                120.0,
  "hdl":                42.0,
  "triglycerides":      210.0,
  "alt":                34.0,
  "notes":              "Optional",
  "createdAt":          "2026-04-20T09:12:34.123456+00:00",
  "updatedAt":          "2026-04-20T09:12:34.123456+00:00"
}
```

**`PatientResponse`**
```jsonc
{
  "id":          "uuid",
  "firstName":   "Jane",
  "lastName":    "Doe",
  "dateOfBirth": "1972-03-14",
  "gender":      "female",
  "email":       "jane.doe@example.com",
  "phone":       "+1-555-0123",
  "address":     "123 Main St",
  "createdAt":   "...",
  "updatedAt":   "..."
}
```

**`PatientDetailResponse`** — same as `PatientResponse`, plus:
```jsonc
{
  "medicalRecords": [ /* MedicalRecordResponse[] */ ]
}
```

---

## 7. Similar Patients — `/api/v1/similar-patients`

Graph-database–backed similar-case search. **All endpoints require the `DOCTOR` role.**

### 7.1 `POST /api/v1/similar-patients/search`

Returns similar cases in tabular format (paginated).

- **Query params:** `page`, `pageSize` (§1.6).
- **Body:**
  ```json
  {
    "patient_id":        "uuid",            // optional, uses that patient's latest medical record
    "medical_record_id": "uuid",            // optional, uses this specific record
    "limit":             20,                // max 1–100, default 20; total Neo4j match cap
    "treatment_filter":  "GLP-1",           // optional Treatment enum value
    "min_similarity":    0.5                // default 0.5, range 0.0–1.0
  }
  ```
  Exactly one of `patient_id` / `medical_record_id` must be provided (either is fine, both is fine, **neither** fails with `VALIDATION_ERROR`).
- **Response 200:** paginated `SimilarPatientCaseResponse[]` (see §7.4).

### 7.2 `POST /api/v1/similar-patients/search/graph`

Returns similar cases formatted as a graph for visualization.

- **Body:**
  ```json
  {
    "patient_id":        "uuid",
    "medical_record_id": "uuid",
    "limit":             5,              // range 1–20, default 5
    "treatment_filter":  "Metformin"
  }
  ```
  Same "at least one ID" rule as §7.1.
- **Response 200 `value`:** `SimilarPatientsGraphResponse` (see §7.4).

### 7.3 `GET /api/v1/similar-patients/{caseId}`

- **Path:** `caseId` — opaque string (**not** a UUID).
- **Response 200 `value`:** `SimilarPatientDetailResponse` (see §7.4).

### 7.4 Similar-patient payloads

**`SimilarPatientCaseResponse`** (tabular result)
```jsonc
{
  "caseId":                "CASE-00123",
  "similarityScore":        0.87,
  "clinicalSimilarity":     0.82,
  "comorbiditySimilarity":  0.91,
  "profile": {
    "age":               54,
    "gender":            "male",
    "ethnicity":         "caucasian",
    "hba1cBaseline":     8.4,
    "cPeptide":          1.6,
    "bmi":               33.1,
    "egfr":              79.0,
    "diabetesDuration":  6.0,
    "bpSystolic":        142,
    "fastingGlucose":    170.0
  },
  "comorbidities":   ["hypertension", "nafld"],
  "treatmentGiven":  "Metformin + GLP-1",
  "drugClass":       "GLP-1",
  "outcome": {
    "hba1cReduction":   1.4,
    "hba1cFollowup":    7.0,
    "timeToTarget":     "6 months",
    "adverseEvents":    "none",
    "outcomeCategory":  "good",
    "success":          true
  }
}
```

**`SimilarPatientsGraphResponse`** (graph visualization)
```jsonc
{
  "patientId": "uuid-or-record-id",
  "nodes": [
    {
      "id":    "node-1",
      "type":  "query_patient",      // opaque string — type tag used by renderer
      "label": "Index patient",
      "data":  { /* arbitrary renderer payload */ },
      "style": { "color": "#1e90ff", "size": "lg", "shape": "circle" }
    }
  ],
  "edges": [
    {
      "id":     "edge-1",
      "source": "node-1",
      "target": "node-2",
      "type":   "similar_to",
      "label":  "0.87",
      "data":   { /* arbitrary */ },
      "style":  { "width": 2, "color": "#888" }
    }
  ],
  "metadata": {
    "queryPatient":    { /* arbitrary keys */ },
    "filtersApplied":  { "treatment": "Metformin", "limit": 5 },
    "resultsFound":    5,
    "similarityRange": { "min": 0.62, "max": 0.91 }    // nullable
  }
}
```

**`SimilarPatientDetailResponse`** (case drill-down)
```jsonc
{
  "patientId":   "CASE-00123",
  "demographics": {
    "age":       54,
    "gender":    "male",
    "ethnicity": "caucasian",
    "ageGroup":  "50-59"
  },
  "clinicalFeatures": {
    "hba1cBaseline":       8.4,
    "diabetesDuration":    6.0,
    "fastingGlucose":      170.0,
    "cPeptide":            1.6,
    "egfr":                79.0,
    "bmi":                 33.1,
    "bpSystolic":          142,
    "bpDiastolic":         88,
    "alt":                 31.0,
    "ldl":                 118.0,
    "hdl":                 41.0,
    "triglycerides":       205.0,
    "previousPrediabetes": true
  },
  "clinicalCategories": {
    "bmiCategory":    "obese-class-1",
    "hba1cSeverity":  "high",
    "kidneyFunction": "normal"
  },
  "comorbidities": ["hypertension", "nafld"],
  "treatment": {                         // nullable
    "drugName":      "Liraglutide",
    "drugClass":     "GLP-1",
    "costCategory":  "mid",
    "evidenceLevel": "A"
  },
  "outcome": {                           // nullable, same shape as §7.4 outcome
    "hba1cReduction":  1.4,
    "hba1cFollowup":   7.0,
    "timeToTarget":    "6 months",
    "adverseEvents":   "none",
    "outcomeCategory": "good",
    "success":         true
  }
}
```

---

## 8. Inference — `/api/v1/inference`

Raw ML inference over a single in-memory patient payload (no DB persistence). **Require `ADMIN` or `DOCTOR`.**

### 8.1 `PatientInput` request shape (shared by all three endpoints)

```jsonc
{
  // 16 clinical features — required
  "age":                 62,       // int,    18..110
  "bmi":                 34.2,     // float,  10..80
  "hba1c_baseline":      8.9,      // float,  4..20
  "egfr":                85.0,     // float,  0..200
  "diabetes_duration":   6.0,      // float,  0..80
  "fasting_glucose":     180.0,    // float,  40..600
  "c_peptide":           1.4,      // float,  0..10
  "bp_systolic":         140.0,    // float,  60..260
  "ldl":                 120.0,    // float,  20..400
  "hdl":                 45.0,     // float,  10..150
  "triglycerides":       200.0,    // float,  10..2000
  "alt":                 30.0,     // float,  0..1000
  "cvd":                 1,        // 0 or 1
  "ckd":                 0,
  "nafld":               1,
  "hypertension":        1,

  // Optional safety flags (default 0)
  "medullary_thyroid_history": 0,
  "men2_history":              0,
  "pancreatitis_history":      0,
  "type1_suspicion":           0,

  // Optional audit-only fields — do NOT influence the model
  "gender":     "M",               // "M" | "F" | "Other" | null
  "ethnicity":  "south-asian",     // free text, nullable
  "patient_id": "PID-0001"         // opaque string, echoed back in the response
}
```

Note the difference from `POST /patients/{id}/medical-records`: the inference input uses **wider** validation ranges (e.g., `hba1c_baseline` 4–20 here vs. 3–20 there) and adds the four optional safety flags. Use the inference ranges when calling these endpoints.

### 8.2 `POST /api/v1/inference/predict`

Run the model (no LLM explanation). Response is fast.

- **Body:** `PatientInput`.
- **Response 200 `value`:** `PredictionResult` (see §8.5). `explanation` is `null`.

### 8.3 `POST /api/v1/inference/predict-with-explanation`

Run the model **and** produce the LLM explanation. Slower; may fail with `INFERENCE_EXPLANATION_ERROR (500)`.

- **Body:** `PatientInput`.
- **Response 200 `value`:** `PredictionResult` with `explanation` populated.

### 8.4 `POST /api/v1/inference/predict-batch`

Run the model for many patients in one call.

- **Body:**
  ```json
  { "patients": [ /* PatientInput[], 1..50 items */ ] }
  ```
- **Response 200 `value`:** array of `PredictionResult[]`. **Rows that fail per-patient validation are returned with `accepted: false` and `validation_errors` populated — they do not abort the batch.**

### 8.5 `PredictionResult` payload

```jsonc
{
  "accepted":           true,
  "validation_errors":  [],                    // non-empty when accepted=false
  "patient_id":         "PID-0001",            // echoed from input, nullable

  "recommended":        "GLP-1",               // Treatment enum, nullable on reject
  "recommended_idx":    1,                     // int 0..4, nullable
  "model_top_treatment": "GLP-1",              // model argmax before safety override

  "confidence_pct":     78,                    // 0..100
  "confidence_label":   "HIGH",                // ConfidenceLabel enum

  "posterior_means":    { "Metformin": 0.1, "GLP-1": 1.9, "SGLT-2": 1.3, "DPP-4": 0.4, "Insulin": 0.0 },
  "win_rates":          { "Metformin": 0.02, "GLP-1": 0.78, "SGLT-2": 0.15, "DPP-4": 0.04, "Insulin": 0.01 },

  "runner_up":          "SGLT-2",
  "runner_up_win_rate": 0.15,
  "mean_gap":           0.6,

  "safety_status":      "CLEAR",                // SafetyStatus enum
  "safety_findings":    [ /* findings for the recommended arm, see below */ ],
  "excluded_treatments": { /* treatmentName -> findings */ },
  "override":           null,                   // null unless safety override kicked in

  "explanation":        null,                   // structured LLM output; see below

  "attribution":        { /* feature attributions */ },   // nullable
  "contrast":           { /* counterfactual contrast */ },// nullable
  "uncertainty_drivers":[ /* top uncertainty contributors */ ],  // nullable

  "model_version":      "2026.04-a1",
  "pipeline_version":   "1.2.0",
  "generated_at":       "2026-04-20T09:12:34.123456+00:00"
}
```

**`validation_errors`** — each entry is a Pydantic error dict:
```jsonc
{ "loc": ["bmi"], "msg": "Input should be less than or equal to 80", "type": "less_than_equal" }
```

**`safety_findings`** / **`excluded_treatments[*]`** — each finding is an object produced by the engine's safety layer. Treat it as structured but schema-flexible:
```jsonc
{
  "rule":          "egfr_low_metformin",
  "severity":      "contraindication",   // "contraindication" | "warning"
  "treatment":     "Metformin",
  "details":       "Metformin contraindicated for eGFR < 30",
  "feature":       "egfr",
  "feature_value": 27.0
}
```

**`override`** — present only when the safety layer replaced the model's top choice. Shape:
```jsonc
{
  "from": "Metformin",
  "to":   "GLP-1",
  "reason": "contraindication",
  "findings": [ /* finding[] */ ]
}
```

**`explanation`** — populated only by `/predict-with-explanation`. Known keys (all strings unless noted) from the predictions module's `ExplanationResponse` contract:
```jsonc
{
  "summary":     "Your model recommends GLP-1 with high confidence...",
  "runner_up":   "SGLT-2 is the next best option because...",
  "confidence":  "The model is confident based on...",
  "safety":      "No safety issues detected.",
  "monitoring":  "Monitor HbA1c in 3 months.",
  "disclaimer":  "This is a decision support tool, not a substitute for clinical judgment."
}
```
(For legacy inference responses, `explanation` may carry different keys such as `recommendation_summary`; treat it as a `Record<string, string>` in the FE.)

---

## 9. Predictions — `/api/v1/predictions`

DB-persisted predictions tied to a `patient` + `medical_record` + the authenticated doctor. **Require `ADMIN` or `DOCTOR`.**

### 9.1 `POST /api/v1/predictions`

Run inference + explanation against an existing medical record and save the result.

- **Status:** `201 Created`.
- **Body:**
  ```json
  {
    "medical_record_id": "uuid",
    "patient_id":        "uuid"
  }
  ```
- **Response 201 `value`:** `PredictionResponse` (see §9.5), `message: "Prediction created"`.
- **Errors:** `404 NOT_FOUND` (unknown IDs), `500 INFERENCE_MODEL_ERROR`, `500 INFERENCE_EXPLANATION_ERROR`.

### 9.2 `GET /api/v1/predictions/{predictionId}`

- **Response 200 `value`:** `PredictionResponse`.

### 9.3 `PATCH /api/v1/predictions/{predictionId}/decision`

Record the doctor's final decision.

- **Body:**
  ```json
  {
    "decision":        "ACCEPTED",            // required: "ACCEPTED" | "OVERRIDDEN"
    "final_treatment": "GLP-1",               // optional; one of the 5 Treatment values
    "doctor_notes":    "Preferred GLP-1 due to weight loss goal"   // optional, ≤1000 chars
  }
  ```
  - If `decision: "ACCEPTED"`, the system uses the model's `recommendedTreatment` as the final; `final_treatment` may be omitted.
  - If `decision: "OVERRIDDEN"`, `final_treatment` should be set to one of the 5 treatments.
- **Response 200 `value`:** `PredictionResponse`, `message: "Decision recorded"`.

### 9.4 `GET /api/v1/predictions/patient/{patientId}`

Prediction history for a patient.

- **Query params:** `page`, `pageSize` (§1.6).
- **Response 200:** paginated `PredictionResponse[]`.

### 9.5 `PredictionResponse` payload

```jsonc
{
  "id":               "uuid",
  "medicalRecordId":  "uuid",
  "patientId":        "uuid",
  "createdBy":        "uuid",          // doctor user id

  // Model recommendation
  "recommendedTreatment": "GLP-1",
  "recommendedIdx":       1,
  "confidencePct":        78,
  "confidenceLabel":      "HIGH",      // ConfidenceLabel enum
  "meanGap":              0.6,
  "runnerUp":             "SGLT-2",
  "runnerUpWinRate":      0.15,
  "winRates":             { "Metformin": 0.02, "GLP-1": 0.78, "SGLT-2": 0.15, "DPP-4": 0.04, "Insulin": 0.01 },
  "posteriorMeans":       { "Metformin": 0.10, "GLP-1": 1.9,  "SGLT-2": 1.3,  "DPP-4": 0.4,  "Insulin": 0.0  },

  // Safety — DUAL SHAPE, see below
  "safetyStatus":  "CLEAR",            // SafetyStatus enum
  "safetyDetails": { /* see §9.6 */ },

  // Fairness — DUAL SHAPE, see below
  "fairness":      { /* see §9.7 */ },

  // LLM explanation (§8.5 `explanation`)
  "explanation": {
    "summary":    "...",
    "runnerUp":   "...",
    "confidence": "...",
    "safety":     "...",
    "monitoring": "...",
    "disclaimer": "..."
  },

  // Doctor decision
  "doctorDecision": "PENDING",         // DoctorDecision enum
  "finalTreatment": null,              // string | null — Treatment value when set
  "doctorNotes":    null,              // string | null

  "createdAt": "...",
  "updatedAt": "..."
}
```

### 9.6 `safetyDetails` — dual-shape JSONB field

`safetyDetails` is a JSON object whose shape depends on when the row was created. **The frontend must handle both shapes.** Use the presence of the "new" keys to branch.

- **New shape (current engine):**
  ```jsonc
  {
    "safety_findings":     [ /* finding[] as in §8.5 */ ],
    "excluded_treatments": { "Metformin": [ /* finding[] */ ] },
    "override": {           // may be null
      "from": "Metformin",
      "to":   "GLP-1",
      "reason": "contraindication",
      "findings": [ /* finding[] */ ]
    }
  }
  ```
- **Legacy shape (older rows):**
  ```jsonc
  {
    "recommended_contraindications": [ /* finding[] */ ],
    "recommended_warnings":          [ /* finding[] */ ],
    "all_warnings":                  [ /* finding[] */ ]
  }
  ```

Detection rule: if `safety_findings` is present → new shape, else → legacy shape.

### 9.7 `fairness` — dual-shape JSONB field

Same dual-shape pattern as `safetyDetails`.

- **New shape:**
  ```jsonc
  {
    "model_top_treatment": "GLP-1",
    "attribution":         { /* feature → contribution map */ },
    "contrast":            { /* counterfactual contrast block */ },
    "uncertainty_drivers": [
      { "feature": "egfr", "contribution": 0.32 }
    ]
  }
  ```
- **Legacy shape:**
  ```jsonc
  {
    "decision_features": { /* legacy feature breakdown */ }
    // plus other legacy keys
  }
  ```

Detection rule: if `model_top_treatment` or `attribution` is present → new shape, else → legacy shape.

If the engine did not produce any fairness payload, `fairness` is an empty object `{}`.

---

## 10. Simulations — `/api/v1/simulations`

Bandit-learning simulations on a user-supplied CSV. **All endpoints require the `ADMIN` role.**

### 10.1 `POST /api/v1/simulations`

Start a new simulation. Uploads a CSV and kicks off a background task.

- **Content-Type:** `multipart/form-data`.
- **Form fields:**
  | Name | Type | Default | Range | Notes |
  |---|---|---|---|---|
  | `file` | file | — | — | Required. Must be `.csv`, UTF-8, ≤ 20 MB, 100–50,000 data rows, containing all 16 context columns (see below). |
  | `initial_epsilon` | float | `0.3` | 0.0–1.0 | |
  | `epsilon_decay` | float | `0.997` | 0.9–1.0 | |
  | `min_epsilon` | float | `0.01` | 0.0–1.0 | |
  | `random_seed` | int | `42` | 0–999,999 | |
  | `reset_posterior` | bool | `true` | — | |
- **Required CSV columns:** `age, bmi, hba1c_baseline, egfr, diabetes_duration, fasting_glucose, c_peptide, cvd, ckd, nafld, hypertension, bp_systolic, ldl, hdl, triglycerides, alt`. Each row must satisfy the same validation ranges as §8.1 (inference `PatientInput`). Rows with any invalid value cause the upload to fail with `CSV_VALIDATION_FAILED` — the first 20 per-row errors are listed under `error.details`.
- **Response 200 `value`:** `SimulationResponse` (see §10.7), `message: "Simulation started with N patients"`. The simulation is created in `RUNNING` status (or will transition there immediately) and runs asynchronously.
- **Errors:** `400 INVALID_FILE_TYPE`, `400 FILE_TOO_LARGE`, `400 INVALID_ENCODING`, `400 CSV_VALIDATION_FAILED`.

### 10.2 `GET /api/v1/simulations/{simulationId}/stream`

Live SSE stream of simulation steps. Clients may reconnect.

- **Content-Type:** `text/event-stream`.
- **Query params:**
  | Name | Type | Default | Notes |
  |---|---|---|---|
  | `last_step` | integer | `0` | Cursor for reconnection — only steps **greater than** this number are emitted. |
- **Behavior:**
  - If the simulation is actively running in-process, the stream delivers live events from the in-memory queue.
  - If it is already completed/failed/cancelled (or this process does not own it), the server replays steps from the DB and then emits a final `complete` event.
  - Every 30 s of idleness the server sends a `ping` event with empty data (connection keep-alive).
- **Event types:**

  #### `step` event
  Emitted for each simulation step. The `data` field is a JSON string:
  ```jsonc
  {
    "step":              42,
    "totalSteps":        500,
    "selectedIdx":       1,
    "selectedTreatment": "GLP-1",         // Treatment enum
    "explored":          false,            // Thompson explored (selectedIdx ≠ posterior argmax)
    "observedReward":    1.12,             // rounded to 4dp
    "epsilon":           0.2865,           // current decayed epsilon value
    "runningEstimates":  { "Metformin": 0.3, "GLP-1": 1.7, "SGLT-2": 1.1, "DPP-4": 0.4, "Insulin": 0.0 },
    "runningAccuracy":   0.7738,           // 0..1, rounded to 4dp
    "cumulativeReward":  55.7234,
    "cumulativeRegret":  4.1108,
    "treatmentCounts":   { "Metformin": 5, "GLP-1": 22, "SGLT-2": 10, "DPP-4": 3, "Insulin": 2 }
  }
  ```

  #### `complete` event (success)
  ```jsonc
  {
    "status":                   "COMPLETED",
    "final_accuracy":            0.78,
    "final_cumulative_reward":   520.3,
    "final_cumulative_regret":   41.2,
    "mean_reward":               1.04,
    "mean_regret":               0.08,
    "thompson_exploration_rate": 0.12,
    "treatment_counts":          { /* Treatment -> count */ },
    "confidence_distribution":   { "HIGH": 310, "MODERATE": 170, "LOW": 20 },
    "safety_distribution":       { "CLEAR": 480, "WARNING": 15, "CONTRAINDICATION_FOUND": 5 }
  }
  ```
  During DB replay (simulation already completed when you connected), the final `complete` event contains only `status` if the simulation is not `COMPLETED` (e.g. `CANCELLED` or `FAILED`).

  #### `complete` event (cancelled mid-run)
  ```json
  { "status": "CANCELLED", "cancelled_at_step": 217 }
  ```
  Or simply `{ "status": "CANCELLED" }` when the asyncio task is cancelled externally.

  #### `error` event
  Emitted and the stream terminates when the server-side simulation fails or the client's queue overflows.
  ```json
  { "error": "Subscriber queue full — connection dropped" }
  ```
  ```json
  { "error": "<exception message>" }
  ```

  #### `ping` event
  Emitted every 30 s of idleness with empty `data`. Ignore or use as a keep-alive signal.

- **Reconnection strategy:** on reconnect, pass the last successfully-processed step number as `?last_step=<n>` to avoid re-playing already-handled rows. The server drops any step ≤ `last_step`.

### 10.3 `POST /api/v1/simulations/{simulationId}/cancel`

- **Status:** `202 Accepted` (the cancellation is asynchronous).
- **Body:** none.
- **Response 202 `value`:** `null`, `message: "Simulation cancellation requested. Status will transition to CANCELLED asynchronously — poll GET /simulations/{id} or watch the SSE stream."`
- **Errors:** `409 SIMULATION_NOT_RUNNING` if the simulation is not in `RUNNING`.

### 10.4 `GET /api/v1/simulations`

- **Query params:** `page`, `pageSize` (§1.6).
- **Response 200:** paginated `SimulationResponse[]`.

### 10.5 `GET /api/v1/simulations/{simulationId}`

- **Response 200 `value`:** `SimulationResponse`, `message: "Simulation retrieved"`.

### 10.6 `GET /api/v1/simulations/{simulationId}/steps`

Paginated step drill-down (full details — much richer than the SSE frame).

- **Query params:** `page`, `pageSize` (§1.6).
- **Response 200:** paginated `SimulationStepResponse[]` (see §10.8), `message: "Retrieved N steps"`.

### 10.7 `SimulationResponse` payload

```jsonc
{
  "id":          "uuid",
  "status":      "RUNNING",              // SimulationStatus enum
  "currentStep": 217,                    // progress counter
  "errorMessage": null,                  // string | null (set if status=FAILED)

  "config": {
    "initialEpsilon":   0.3,
    "epsilonDecay":     0.997,
    "minEpsilon":       0.01,
    "randomSeed":       42,
    "resetPosterior":   true,
    "datasetFilename":  "cohort_2026_q1.csv",
    "datasetRowCount":  500
  },

  "aggregates": {                        // all nullable until simulation completes
    "finalAccuracy":           0.78,
    "finalCumulativeReward":   520.3,
    "finalCumulativeRegret":   41.2,
    "meanReward":              1.04,
    "meanRegret":              0.08,
    "thompsonExplorationRate": 0.12,
    "treatmentCounts":         { /* Treatment -> count */ },
    "confidenceDistribution":  { "HIGH": 310, "MODERATE": 170, "LOW": 20 },
    "safetyDistribution":      { "CLEAR": 480, "WARNING": 15, "CONTRAINDICATION_FOUND": 5 }
  },

  "createdAt": "...",
  "updatedAt": "..."
}
```

### 10.8 `SimulationStepResponse` payload (full REST shape)

One entry per step. All per-treatment dicts are keyed by `Treatment` enum value.

```jsonc
{
  "step":    42,
  "epsilon": 0.2865,

  "patient": {                              // the 16 context features, plain number map
    "age": 54, "bmi": 31.4, "hba1c_baseline": 8.1,
    "egfr": 82, "diabetes_duration": 5, "fasting_glucose": 175,
    "c_peptide": 1.4, "cvd": 0, "ckd": 0, "nafld": 1,
    "hypertension": 1, "bp_systolic": 138, "ldl": 120, "hdl": 42,
    "triglycerides": 210, "alt": 34
  },

  "oracle": {
    "rewards":          { "Metformin": 0.9, "GLP-1": 1.7, "SGLT-2": 1.3, "DPP-4": 0.4, "Insulin": 0.2 },
    "optimalTreatment": "GLP-1",
    "optimalReward":    1.7
  },

  "decision": {
    "selectedTreatment": "GLP-1",
    "selectedIdx":       1,
    "posteriorMeans":    { /* Treatment -> float */ },
    "winRates":          { /* Treatment -> float */ },
    "confidencePct":     78,
    "confidenceLabel":   "HIGH",             // ConfidenceLabel enum
    "sampledValues":     { /* Treatment -> float (Thompson draws) */ },
    "runnerUp":          "SGLT-2",
    "runnerUpWinrate":   0.15,
    "meanGap":           0.6
  },

  "exploration": {
    "thompsonExplored":  true,               // Thompson chose non-argmax arm
    "epsilonExplored":   false,              // Always false — ε-greedy is not currently used
    "posteriorMeanBest": "GLP-1"             // argmax of posteriorMeans
  },

  "outcome": {
    "observedReward":      1.7,
    "instantaneousRegret": 0.0,
    "matchedOracle":       true
  },

  "safety": {
    "status":             "CLEAR",            // SafetyStatus enum
    "contraindications":  [],                 // always [] in the current runner
    "warnings":           []                  // always [] in the current runner
  },

  "aggregates": {
    "cumulativeReward":  55.7234,
    "cumulativeRegret":  4.1108,
    "runningAccuracy":   0.7738,
    "treatmentCounts":   { /* Treatment -> int */ },
    "runningEstimates":  { /* Treatment -> float, per-arm observed mean reward */ }
  }
}
```

### 10.9 `DELETE /api/v1/simulations/{simulationId}`

- **Response 200 `value`:** `null`, `message: "Simulation deleted"`. Deletes the simulation record and all its steps.

---

## 11. Example requests and responses

### 11.1 Login

Request:
```http
POST /api/v1/auth/login
Content-Type: application/json

{ "email": "dr_smith@example.com", "password": "hunter2hunter2" }
```

Response `200 OK`:
```json
{
  "success": true,
  "message": "Login successful",
  "value": {
    "user": {
      "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "email": "dr_smith@example.com",
      "username": "dr_smith",
      "firstName": "Alice",
      "lastName": "Smith",
      "role": "DOCTOR",
      "isActive": true,
      "createdAt": "2026-04-01T10:00:00+00:00",
      "updatedAt": "2026-04-01T10:00:00+00:00"
    },
    "tokens": {
      "accessToken":  "eyJhbGciOi...",
      "refreshToken": "eyJhbGciOi..."
    }
  }
}
```

### 11.2 Validation failure

Request:
```http
POST /api/v1/auth/register
Content-Type: application/json

{ "email": "not-an-email", "username": "x", "first_name": "", "last_name": "Smith", "password": "short" }
```

Response `400 Bad Request`:
```json
{
  "success": false,
  "message": "Please check your input and try again",
  "error": {
    "title": "Validation Failed",
    "code": "VALIDATION_ERROR",
    "status": 400,
    "fieldErrors": {
      "body.email":      ["value is not a valid email address"],
      "body.username":   ["String should have at least 3 characters"],
      "body.first_name": ["String should have at least 1 character"],
      "body.password":   ["String should have at least 8 characters"]
    }
  }
}
```

### 11.3 Paginated list

Request:
```http
GET /api/v1/patients?page=2&pageSize=10
Authorization: Bearer <accessToken>
```

Response `200 OK`:
```json
{
  "success": true,
  "value": [ /* PatientResponse[] up to 10 items */ ],
  "pagination": { "page": 2, "total": 57, "pageSize": 10, "totalPages": 6 }
}
```

### 11.4 Live simulation stream (SSE)

Request:
```http
GET /api/v1/simulations/3fa85f64-5717-4562-b3fc-2c963f66afa6/stream?last_step=0
Authorization: Bearer <accessToken>
Accept: text/event-stream
```

Server streams:
```
event: step
data: {"step":1,"totalSteps":500,"selectedIdx":1,"selectedTreatment":"GLP-1","explored":false,"observedReward":1.7,"epsilon":0.2991,"runningEstimates":{"Metformin":0.0,"GLP-1":1.7,"SGLT-2":0.0,"DPP-4":0.0,"Insulin":0.0},"runningAccuracy":1.0,"cumulativeReward":1.7,"cumulativeRegret":0.0,"treatmentCounts":{"Metformin":0,"GLP-1":1,"SGLT-2":0,"DPP-4":0,"Insulin":0}}

event: step
data: {"step":2, ...}

event: ping
data: 

event: complete
data: {"status":"COMPLETED","final_accuracy":0.78,"final_cumulative_reward":520.3,"final_cumulative_regret":41.2,"mean_reward":1.04,"mean_regret":0.08,"thompson_exploration_rate":0.12,"treatment_counts":{"Metformin":100,"GLP-1":220,"SGLT-2":140,"DPP-4":25,"Insulin":15},"confidence_distribution":{"HIGH":310,"MODERATE":170,"LOW":20},"safety_distribution":{"CLEAR":480,"WARNING":15,"CONTRAINDICATION_FOUND":5}}
```

Close the connection on `complete` or `error`.

### 11.5 Start a simulation (multipart)

```http
POST /api/v1/simulations
Authorization: Bearer <accessToken>
Content-Type: multipart/form-data; boundary=---FormBoundary

-----FormBoundary
Content-Disposition: form-data; name="file"; filename="cohort.csv"
Content-Type: text/csv

age,bmi,hba1c_baseline,egfr,diabetes_duration,fasting_glucose,c_peptide,cvd,ckd,nafld,hypertension,bp_systolic,ldl,hdl,triglycerides,alt
52,31.4,8.1,82,5,175,1.4,0,0,1,1,138,120,42,210,34
...
-----FormBoundary
Content-Disposition: form-data; name="initial_epsilon"

0.3
-----FormBoundary
Content-Disposition: form-data; name="reset_posterior"

true
-----FormBoundary--
```

---

## 12. Quick endpoint index

| # | Method | Path | Auth | Role |
|---|---|---|---|---|
| 1 | GET | `/health` | — | — |
| 2 | POST | `/api/v1/auth/register` | — | — |
| 3 | POST | `/api/v1/auth/login` | — | — |
| 4 | POST | `/api/v1/auth/refresh` | — | — |
| 5 | POST | `/api/v1/auth/logout` | Bearer | any |
| 6 | GET | `/api/v1/auth/me` | Bearer | any |
| 7 | PATCH | `/api/v1/auth/me` | Bearer | any |
| 8 | GET | `/api/v1/users` | Bearer | ADMIN |
| 9 | GET | `/api/v1/users/{userId}` | Bearer | ADMIN |
| 10 | POST | `/api/v1/users` | Bearer | ADMIN |
| 11 | PATCH | `/api/v1/users/{userId}` | Bearer | ADMIN |
| 12 | DELETE | `/api/v1/users/{userId}` | Bearer | ADMIN |
| 13 | POST | `/api/v1/patients` | Bearer | ADMIN/DOCTOR |
| 14 | GET | `/api/v1/patients` | Bearer | ADMIN/DOCTOR |
| 15 | GET | `/api/v1/patients/{patientId}` | Bearer | ADMIN/DOCTOR |
| 16 | PATCH | `/api/v1/patients/{patientId}` | Bearer | ADMIN/DOCTOR |
| 17 | DELETE | `/api/v1/patients/{patientId}` | Bearer | ADMIN/DOCTOR |
| 18 | POST | `/api/v1/patients/{patientId}/medical-records` | Bearer | ADMIN/DOCTOR |
| 19 | GET | `/api/v1/patients/{patientId}/medical-records` | Bearer | ADMIN/DOCTOR |
| 20 | GET | `/api/v1/patients/{patientId}/medical-records/{recordId}` | Bearer | ADMIN/DOCTOR |
| 21 | POST | `/api/v1/similar-patients/search` | Bearer | DOCTOR |
| 22 | POST | `/api/v1/similar-patients/search/graph` | Bearer | DOCTOR |
| 23 | GET | `/api/v1/similar-patients/{caseId}` | Bearer | DOCTOR |
| 24 | POST | `/api/v1/inference/predict` | Bearer | ADMIN/DOCTOR |
| 25 | POST | `/api/v1/inference/predict-with-explanation` | Bearer | ADMIN/DOCTOR |
| 26 | POST | `/api/v1/inference/predict-batch` | Bearer | ADMIN/DOCTOR |
| 27 | POST | `/api/v1/predictions` | Bearer | ADMIN/DOCTOR |
| 28 | GET | `/api/v1/predictions/{predictionId}` | Bearer | ADMIN/DOCTOR |
| 29 | PATCH | `/api/v1/predictions/{predictionId}/decision` | Bearer | ADMIN/DOCTOR |
| 30 | GET | `/api/v1/predictions/patient/{patientId}` | Bearer | ADMIN/DOCTOR |
| 31 | POST | `/api/v1/simulations` | Bearer | ADMIN |
| 32 | GET | `/api/v1/simulations` | Bearer | ADMIN |
| 33 | GET | `/api/v1/simulations/{simulationId}` | Bearer | ADMIN |
| 34 | GET | `/api/v1/simulations/{simulationId}/stream` (SSE) | Bearer | ADMIN |
| 35 | GET | `/api/v1/simulations/{simulationId}/steps` | Bearer | ADMIN |
| 36 | POST | `/api/v1/simulations/{simulationId}/cancel` | Bearer | ADMIN |
| 37 | DELETE | `/api/v1/simulations/{simulationId}` | Bearer | ADMIN |
