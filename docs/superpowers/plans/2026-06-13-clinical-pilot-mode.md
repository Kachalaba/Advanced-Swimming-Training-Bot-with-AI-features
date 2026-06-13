# Clinical Pilot Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first clinical rehabilitation workflow that lets one specialist create a patient and episode, conduct a quality-gated live or uploaded visit, finalize it against persisted biomechanics, compare progress, and export a bilingual report.

**Architecture:** Add an additive SQLite clinical repository beside the existing athlete/session store, expose it through a focused FastAPI router, and build a Next.js clinical workspace with small route-level components. Existing rehabilitation analyzers, saved `TrainingSession` reports, progress math, and print/PDF handoff remain the measurement and reporting foundations.

**Tech Stack:** Python 3.11, SQLite, FastAPI, Pydantic, pytest, Next.js 16, React 19, TypeScript, Tailwind CSS, Vitest, Testing Library.

---

## File Map

### Backend and persistence

- Create `video_analysis/clinical_repository.py`
  - clinical dataclasses, additive schema initialization, first-migration backup,
    patient/episode/visit persistence, and finalization rules
- Create `backend/app/api/clinical.py`
  - Pydantic request/response contracts and `/api/clinical` routes
- Modify `backend/app/main.py`
  - register the clinical router
- Modify `backend/app/api/rehabilitation.py`
  - allow save-by-athlete-ID while preserving the athlete-name fallback
- Modify `video_analysis/athlete_database.py`
  - add an identity-safe `save_analysis_to_athlete` helper so clinical saves
    cannot attach to the wrong same-named person
- Create `tests/unit/test_clinical_repository.py`
- Create `tests/unit/test_clinical_api.py`
- Modify `tests/unit/test_rehabilitation_api.py`
- Modify `tests/unit/test_athlete_database.py`

### Frontend domain and API

- Create `frontend/lib/clinical.ts`
  - clinical types, API functions, status guards, and comparison adapters
- Create `frontend/lib/clinicalCopy.ts`
  - symmetric UK/EN strings for the clinical workspace and wizard
- Create `frontend/lib/captureReadiness.ts`
  - pure readiness evaluation from pose, camera, confidence, and frame coverage
- Create `frontend/lib/clinical.test.ts`
- Create `frontend/lib/captureReadiness.test.ts`

### Frontend workspace

- Create `frontend/app/rehabilitation/clinical/page.tsx`
  - patient roster and create-patient flow
- Create `frontend/app/rehabilitation/clinical/page.test.tsx`
- Create `frontend/app/rehabilitation/clinical/patients/[patientId]/page.tsx`
  - patient profile, active episode, progress, visit timeline
- Create `frontend/app/rehabilitation/clinical/patients/[patientId]/page.test.tsx`
- Create `frontend/components/rehabilitation/clinical/PatientForm.tsx`
- Create `frontend/components/rehabilitation/clinical/EpisodeForm.tsx`
- Create `frontend/components/rehabilitation/clinical/ClinicalVisitTimeline.tsx`

### Visit wizard and handoff

- Create `frontend/app/rehabilitation/clinical/episodes/[episodeId]/visits/new/page.tsx`
- Create `frontend/app/rehabilitation/clinical/episodes/[episodeId]/visits/new/page.test.tsx`
- Create `frontend/components/rehabilitation/clinical/CaptureReadinessPanel.tsx`
- Create `frontend/components/rehabilitation/clinical/VisitReview.tsx`
- Modify `frontend/components/rehabilitation/LiveRehabWorkspace.tsx`
  - accept explicit save identity and report/save callbacks
- Modify `frontend/components/rehabilitation/RehabUploader.tsx`
  - accept explicit save identity and report/save callbacks
- Modify `frontend/components/rehabilitation/LiveRehabWorkspace.test.tsx`
- Modify `frontend/components/rehabilitation/RehabUploader.test.tsx`
- Modify `frontend/lib/rehabilitation.ts`
  - accept `athleteId` on save requests
- Modify `frontend/components/rehabilitation/rehabHandoff.ts`
  - add optional clinical context and comparison
- Modify `frontend/components/rehabilitation/ClinicalReport.tsx`
  - render patient, episode, quality, comparison, and specialist observation
- Modify `frontend/components/rehabilitation/ClinicalReport.test.tsx`
- Modify `frontend/app/rehabilitation/page.tsx`
  - add the clinical workspace entry action
- Modify `frontend/app/rehabilitation/page.test.tsx`

### Delivery

- Modify `README.md`
  - document Clinical Pilot Mode and the local-data limitation
- Create `docs/clinical-pilot.md`
  - pilot walkthrough and recovery/backup notes

---

### Task 1: Add additive clinical SQLite schema and backup

**Files:**
- Create: `video_analysis/clinical_repository.py`
- Create: `tests/unit/test_clinical_repository.py`

- [ ] **Step 1: Write the failing schema and backup tests**

```python
def test_initialization_backs_up_existing_database_before_adding_tables(tmp_path):
    path = tmp_path / "athletes.db"
    sqlite3.connect(path).execute("CREATE TABLE athletes (id INTEGER PRIMARY KEY, name TEXT)")

    ClinicalRepository(path)

    tables = _table_names(path)
    assert {"patient_profiles", "rehab_episodes", "clinical_visits"} <= tables
    backups = list(tmp_path.glob("athletes.pre-clinical-*.db"))
    assert len(backups) == 1
    assert _table_names(backups[0]) == {"athletes"}


def test_initialization_is_idempotent_and_does_not_create_second_backup(tmp_path):
    path = _existing_athlete_database(tmp_path)
    ClinicalRepository(path)
    ClinicalRepository(path)
    assert len(list(tmp_path.glob("athletes.pre-clinical-*.db"))) == 1
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/unit/test_clinical_repository.py -v
```

Expected: collection fails because `video_analysis.clinical_repository` does
not exist.

- [ ] **Step 3: Implement dataclasses and additive initialization**

Use these public types:

```python
AffectedSide = Literal["left", "right", "bilateral", "unspecified"]
EpisodeStatus = Literal["active", "completed", "archived"]
CaptureSource = Literal["live", "upload"]
CaptureQuality = Literal[
    "acceptable", "accepted_with_warning", "repeat_required"
]
VisitStatus = Literal["draft", "finalized"]

@dataclass(frozen=True)
class PatientProfile:
    id: int | None
    athlete_id: int
    display_name: str
    affected_side: AffectedSide
    clinical_context: str
    precautions: str
    status: Literal["active", "archived"]
    created_at: str
    updated_at: str

@dataclass(frozen=True)
class RehabEpisode:
    id: int | None
    patient_profile_id: int
    title: str
    protocol: str
    functional_goal: str
    target_left_rom: float | None
    target_right_rom: float | None
    status: EpisodeStatus
    started_at: str
    completed_at: str | None
    created_at: str
    updated_at: str

@dataclass(frozen=True)
class ClinicalVisit:
    id: int | None
    rehab_episode_id: int
    training_session_id: int | None
    visited_at: str
    capture_source: CaptureSource
    pre_session_note: str
    specialist_observation: str
    capture_quality: CaptureQuality | None
    capture_quality_details: str
    warning_acknowledged: bool
    status: VisitStatus
    created_at: str
    updated_at: str
```

`ClinicalRepository.__init__` must:

1. use the provided path or `ATHLETE_DB_PATH`
2. detect whether all three clinical tables exist
3. if an existing non-empty DB lacks them, copy it once to
   `<stem>.pre-clinical-<UTC timestamp>.db`
4. create tables and indexes in one transaction
5. leave current `athletes` and `sessions` rows untouched

- [ ] **Step 4: Run the focused tests and verify GREEN**

```bash
.venv/bin/pytest tests/unit/test_clinical_repository.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add video_analysis/clinical_repository.py tests/unit/test_clinical_repository.py
git commit -m "feat: add clinical pilot persistence schema"
```

---

### Task 2: Implement clinical repository rules

**Files:**
- Modify: `video_analysis/clinical_repository.py`
- Modify: `tests/unit/test_clinical_repository.py`

- [ ] **Step 1: Write failing CRUD and domain-rule tests**

Cover these behaviors with real temporary SQLite databases:

```python
def test_patient_episode_and_draft_visit_round_trip(tmp_path):
    repository = _repository_with_athlete(tmp_path, athlete_id=7)
    patient = repository.create_patient(
        athlete_id=7,
        display_name="Patient A",
        affected_side="right",
        clinical_context="Post-operative shoulder mobility",
        precautions="Stop on sharp pain",
    )
    episode = repository.create_episode(
        patient_profile_id=patient.id,
        title="Right shoulder recovery",
        protocol="shoulder_flexion",
        functional_goal="Reach an overhead shelf",
        target_left_rom=150,
        target_right_rom=140,
    )
    visit = repository.create_visit(
        rehab_episode_id=episode.id,
        capture_source="live",
        pre_session_note="No pain at rest",
    )
    assert repository.get_patient(patient.id).active_episodes == [episode]
    assert repository.get_visit(visit.id).status == "draft"


def test_rejects_second_active_episode_for_same_patient_and_protocol(tmp_path):
    repository, patient = _patient_repository(tmp_path)
    repository.create_episode(patient.id, "First", "knee_extension", "Goal")
    with pytest.raises(ClinicalConflictError):
        repository.create_episode(patient.id, "Second", "knee_extension", "Goal")


def test_finalize_requires_session_quality_and_observation(tmp_path):
    repository, visit = _draft_visit(tmp_path)
    with pytest.raises(ClinicalValidationError):
        repository.finalize_visit(visit.id)
    repository.update_visit(
        visit.id,
        training_session_id=31,
        capture_quality="accepted_with_warning",
        capture_quality_details="Camera tilt",
        warning_acknowledged=True,
        specialist_observation="ROM remains asymmetric.",
    )
    assert repository.finalize_visit(visit.id).status == "finalized"


def test_repeat_required_visit_cannot_be_finalized(tmp_path):
    repository, visit = _draft_visit(tmp_path)
    repository.update_visit(
        visit.id,
        training_session_id=31,
        capture_quality="repeat_required",
        specialist_observation="Repeat capture.",
    )
    with pytest.raises(ClinicalValidationError):
        repository.finalize_visit(visit.id)
```

- [ ] **Step 2: Run tests and verify expected failures**

```bash
.venv/bin/pytest tests/unit/test_clinical_repository.py -v
```

Expected: failures for missing CRUD and finalization methods.

- [ ] **Step 3: Implement repository operations**

Add:

```python
create_patient(
    athlete_id: int,
    display_name: str,
    affected_side: AffectedSide,
    clinical_context: str,
    precautions: str,
) -> PatientProfile
update_patient(
    patient_id: int,
    *,
    display_name: str,
    affected_side: AffectedSide,
    clinical_context: str,
    precautions: str,
) -> PatientProfile
archive_patient(patient_id: int) -> PatientProfile
list_patients(include_archived: bool = False) -> list[PatientProfile]
get_patient(patient_id: int) -> PatientProfile
create_episode(
    patient_profile_id: int,
    title: str,
    protocol: str,
    functional_goal: str,
    target_left_rom: float | None = None,
    target_right_rom: float | None = None,
) -> RehabEpisode
update_episode(
    episode_id: int,
    *,
    title: str,
    functional_goal: str,
    target_left_rom: float | None,
    target_right_rom: float | None,
    status: EpisodeStatus,
) -> RehabEpisode
archive_episode(episode_id: int) -> RehabEpisode
get_episode(episode_id: int) -> RehabEpisode
create_visit(
    rehab_episode_id: int,
    capture_source: CaptureSource,
    pre_session_note: str,
) -> ClinicalVisit
update_visit(
    visit_id: int,
    *,
    training_session_id: int | None = None,
    specialist_observation: str | None = None,
    capture_quality: CaptureQuality | None = None,
    capture_quality_details: str | None = None,
    warning_acknowledged: bool | None = None,
) -> ClinicalVisit
get_visit(visit_id: int) -> ClinicalVisit
list_episode_visits(episode_id: int) -> list[ClinicalVisit]
finalize_visit(visit_id: int) -> ClinicalVisit
```

Use `ClinicalNotFoundError`, `ClinicalConflictError`, and
`ClinicalValidationError` for API-safe rule failures. Validate protocol against
the same five rehabilitation protocol IDs already supported by the analyzer.

- [ ] **Step 4: Run repository tests and all database tests**

```bash
.venv/bin/pytest tests/unit/test_clinical_repository.py tests/unit/test_athlete_database.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add video_analysis/clinical_repository.py tests/unit/test_clinical_repository.py
git commit -m "feat: add clinical patient episode and visit rules"
```

---

### Task 3: Expose the clinical FastAPI contract

**Files:**
- Create: `backend/app/api/clinical.py`
- Create: `tests/unit/test_clinical_api.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Write failing API contract tests**

Create a fake repository and test:

```python
def test_patient_roster_and_detail_contract(monkeypatch):
    client = _client(monkeypatch, FakeClinicalRepository())
    created = client.post("/api/clinical/patients", json={
        "athleteId": 3,
        "displayName": "Patient A",
        "affectedSide": "right",
        "clinicalContext": "Shoulder mobility",
        "precautions": "Stop on sharp pain",
    })
    assert created.status_code == 201
    assert created.json()["affectedSide"] == "right"
    assert client.get("/api/clinical/patients").json()[0]["id"] == 11


def test_episode_and_visit_finalization_contract(monkeypatch):
    client = _client(monkeypatch, FakeClinicalRepository())
    episode = client.post("/api/clinical/patients/11/episodes", json={
        "title": "Shoulder recovery",
        "protocol": "shoulder_flexion",
        "functionalGoal": "Reach overhead",
        "targetLeftRom": 150,
        "targetRightRom": 140,
    })
    visit = client.post("/api/clinical/episodes/21/visits", json={
        "captureSource": "live",
        "preSessionNote": "Stable",
    })
    finalized = client.post("/api/clinical/visits/31/finalize")
    assert episode.status_code == 201
    assert visit.status_code == 201
    assert finalized.json()["status"] == "finalized"


def test_episode_detail_contains_only_finalized_usable_progress(monkeypatch):
    client = _client(monkeypatch, FakeClinicalRepository())
    payload = client.get("/api/clinical/episodes/21").json()
    assert [visit["id"] for visit in payload["visits"]] == [31, 32, 33]
    assert [observation["visitId"] for observation in payload["progress"]] == [32]
    assert payload["progress"][0]["trainingSessionId"] == 91
```

Also assert mappings:

- repository not found -> `404`
- conflict -> `409`
- validation -> `422`
- archive endpoints return the archived resource

- [ ] **Step 2: Run API tests and verify RED**

```bash
.venv/bin/pytest tests/unit/test_clinical_api.py -v
```

Expected: import failure for `backend.app.api.clinical`.

- [ ] **Step 3: Implement the router**

Use `alias_generator` or explicit aliases so JSON is camelCase while Python
remains snake_case. Routes:

```text
GET   /api/clinical/patients
POST  /api/clinical/patients
GET   /api/clinical/patients/{patient_id}
PATCH /api/clinical/patients/{patient_id}
POST  /api/clinical/patients/{patient_id}/archive
POST  /api/clinical/patients/{patient_id}/episodes
GET   /api/clinical/episodes/{episode_id}
PATCH /api/clinical/episodes/{episode_id}
POST  /api/clinical/episodes/{episode_id}/archive
POST  /api/clinical/episodes/{episode_id}/visits
GET   /api/clinical/visits/{visit_id}
PATCH /api/clinical/visits/{visit_id}
POST  /api/clinical/visits/{visit_id}/finalize
```

The episode detail response joins visits to compatible persisted rehabilitation
sessions and returns a `progress` array. Include an observation only when the
visit is finalized, its capture quality is `acceptable` or
`accepted_with_warning`, and the stored report protocol matches the episode.
Reuse the finite-number and malformed-report safeguards from the current
rehabilitation progress endpoint.

Register:

```python
from app.api import clinical
app.include_router(clinical.router, prefix="/api/clinical")
```

- [ ] **Step 4: Run API and unit suites**

```bash
.venv/bin/pytest tests/unit/test_clinical_api.py tests/unit/test_clinical_repository.py -v
.venv/bin/ruff check backend/app/api/clinical.py video_analysis/clinical_repository.py tests/unit/test_clinical_api.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/clinical.py backend/app/main.py tests/unit/test_clinical_api.py
git commit -m "feat: expose clinical pilot API"
```

---

### Task 4: Save rehabilitation analysis by athlete ID

**Files:**
- Modify: `video_analysis/athlete_database.py`
- Modify: `backend/app/api/rehabilitation.py`
- Modify: `tests/unit/test_athlete_database.py`
- Modify: `tests/unit/test_rehabilitation_api.py`

- [ ] **Step 1: Write failing identity-safe save tests**

```python
def test_save_analysis_to_existing_athlete_id(tmp_path):
    database = AthleteDatabase(tmp_path / "athletes.db")
    athlete_id = database.add_athlete(Athlete(name="Patient A"))
    session_id = save_analysis_to_athlete(
        athlete_id=athlete_id,
        session_type="rehab",
        analysis={"rehab_analysis": _report()},
        database=database,
    )
    assert database.get_sessions(athlete_id)[0].id == session_id


def test_live_save_prefers_athlete_id(monkeypatch):
    response = client.post(
        "/api/analysis/rehabilitation/live/session-123/save",
        json={"athlete_id": 7, "athlete_name": "Fallback"},
    )
    assert response.status_code == 200
    assert saved["athlete_id"] == 7
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
.venv/bin/pytest tests/unit/test_athlete_database.py tests/unit/test_rehabilitation_api.py -v
```

Expected: missing helper and request field failures.

- [ ] **Step 3: Implement backward-compatible save**

Update the request:

```python
class SaveRehabRequest(BaseModel):
    athlete_id: int | None = Field(default=None, ge=1)
    athlete_name: str = Field(default="Athlete", min_length=1, max_length=120)
```

When `athlete_id` is present, verify the athlete and use
`save_analysis_to_athlete`. Otherwise retain `save_analysis_to_db`.

- [ ] **Step 4: Run all affected backend tests**

```bash
.venv/bin/pytest tests/unit/test_athlete_database.py tests/unit/test_rehabilitation_api.py tests/unit/test_clinical_api.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add video_analysis/athlete_database.py backend/app/api/rehabilitation.py tests/unit/test_athlete_database.py tests/unit/test_rehabilitation_api.py
git commit -m "feat: save rehabilitation sessions by athlete id"
```

---

### Task 5: Add frontend clinical domain, API, and bilingual copy

**Files:**
- Create: `frontend/lib/clinical.ts`
- Create: `frontend/lib/clinical.test.ts`
- Create: `frontend/lib/clinicalCopy.ts`
- Modify: `frontend/lib/rehabilitation.ts`

- [ ] **Step 1: Write failing domain and API tests**

```typescript
it("exposes only finalized acceptable visits as progress observations", () => {
  expect(progressVisits([
    visit({ status: "draft" }),
    visit({ status: "finalized", captureQuality: "repeat_required" }),
    visit({ status: "finalized", captureQuality: "acceptable", id: 3 }),
  ])).toEqual([expect.objectContaining({ id: 3 })]);
});

it("sends athleteId when a clinical analysis is saved", async () => {
  fetchMock.mockResolvedValue(jsonResponse({ session_id: 91 }));
  await saveLiveRehabSession("live-1", { athleteId: 7, athleteName: "Patient A" });
  expect(fetchMock).toHaveBeenCalledWith(
    expect.stringContaining("/live/live-1/save"),
    expect.objectContaining({ body: JSON.stringify({
      athlete_id: 7,
      athlete_name: "Patient A",
    }) }),
  );
});
```

- [ ] **Step 2: Run frontend tests and verify RED**

```bash
cd frontend && npm test -- lib/clinical.test.ts
```

Expected: missing module and signature failures.

- [ ] **Step 3: Implement types and API client**

Define:

```typescript
type AffectedSide = "left" | "right" | "bilateral" | "unspecified";
type PatientStatus = "active" | "archived";
type EpisodeStatus = "active" | "completed" | "archived";
type CaptureSource = "live" | "upload";
type CaptureQuality =
  | "acceptable"
  | "accepted_with_warning"
  | "repeat_required";

type PatientProfile = {
  id: number;
  athleteId: number;
  displayName: string;
  affectedSide: AffectedSide;
  clinicalContext: string;
  precautions: string;
  status: PatientStatus;
  createdAt: string;
  updatedAt: string;
};

type RehabEpisode = {
  id: number;
  patientProfileId: number;
  title: string;
  protocol: RehabProtocol;
  functionalGoal: string;
  targetLeftRom: number | null;
  targetRightRom: number | null;
  status: EpisodeStatus;
  startedAt: string;
  completedAt: string | null;
  createdAt: string;
  updatedAt: string;
};

type ClinicalVisit = {
  id: number;
  rehabEpisodeId: number;
  trainingSessionId: number | null;
  visitedAt: string;
  captureSource: CaptureSource;
  preSessionNote: string;
  specialistObservation: string;
  captureQuality: CaptureQuality | null;
  captureQualityDetails: string;
  warningAcknowledged: boolean;
  status: "draft" | "finalized";
  createdAt: string;
  updatedAt: string;
};

type ClinicalProgressObservation = RehabProgressSession & {
  visitId: number;
  trainingSessionId: number;
  captureQuality: Exclude<CaptureQuality, "repeat_required">;
  captureQualityDetails: string;
};

type PatientDetail = PatientProfile & {
  athlete: Athlete;
  episodes: RehabEpisode[];
  latestVisit: ClinicalVisit | null;
};

type EpisodeDetail = RehabEpisode & {
  patient: PatientProfile;
  athlete: Athlete;
  visits: ClinicalVisit[];
  progress: ClinicalProgressObservation[];
};
```

Export a `clinicalApi` object with functions matching every backend route.
Keep copy in one symmetric `clinicalCopy.uk` / `clinicalCopy.en` object and
reuse `RehabLocale`.

- [ ] **Step 4: Run tests and typecheck**

```bash
cd frontend && npm test -- lib/clinical.test.ts && npm run typecheck
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/clinical.ts frontend/lib/clinical.test.ts frontend/lib/clinicalCopy.ts frontend/lib/rehabilitation.ts
git commit -m "feat: add clinical pilot frontend domain"
```

---

### Task 6: Build the specialist patient workspace

**Files:**
- Create: `frontend/components/rehabilitation/clinical/PatientForm.tsx`
- Create: `frontend/app/rehabilitation/clinical/page.tsx`
- Create: `frontend/app/rehabilitation/clinical/page.test.tsx`
- Modify: `frontend/app/rehabilitation/page.tsx`
- Modify: `frontend/app/rehabilitation/page.test.tsx`

- [ ] **Step 1: Write failing workspace tests**

Test:

```typescript
it("renders active patients and creates a patient from an existing athlete", async () => {
  clinicalApi.listPatients.mockResolvedValue([patient()]);
  api.listAthletes.mockResolvedValue([athlete()]);
  render(<ClinicalWorkspacePage />);
  expect(await screen.findByText("Patient A")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /new patient/i }));
  await user.selectOptions(screen.getByLabelText(/athlete/i), "7");
  await user.type(screen.getByLabelText(/clinical context/i), "Shoulder mobility");
  await user.click(screen.getByRole("button", { name: /create patient/i }));
  expect(clinicalApi.createPatient).toHaveBeenCalledWith(
    expect.objectContaining({ athleteId: 7, affectedSide: "unspecified" }),
  );
});

it("links the rehabilitation page to the clinical workspace", () => {
  render(<RehabilitationPage />);
  expect(screen.getByRole("link", { name: /clinical workspace/i }))
    .toHaveAttribute("href", "/rehabilitation/clinical");
});
```

- [ ] **Step 2: Run tests and verify RED**

```bash
cd frontend && npm test -- app/rehabilitation/clinical/page.test.tsx app/rehabilitation/page.test.tsx
```

Expected: missing page/form and missing link failures.

- [ ] **Step 3: Implement the workspace**

Use:

- dark clinical header with local-data and research-prototype badges
- active/archive segmented filter
- patient cards with affected side, active episode, latest date and latest
  compatible metrics
- empty state leading to `PatientForm`
- modal or inline form with athlete, display name, affected side, context, and
  precautions
- locale from the existing rehabilitation storage/event mechanism

Do not fetch each patient's detail in a client loop. The roster API response
must include the compact active-episode/latest-visit summary.

- [ ] **Step 4: Run tests, typecheck, and build**

```bash
cd frontend
npm test -- app/rehabilitation/clinical/page.test.tsx app/rehabilitation/page.test.tsx
npm run typecheck
npm run build
```

Expected: pass and `/rehabilitation/clinical` appears in the route list.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/rehabilitation/clinical/PatientForm.tsx frontend/app/rehabilitation/clinical frontend/app/rehabilitation/page.tsx frontend/app/rehabilitation/page.test.tsx
git commit -m "feat: add clinical patient workspace"
```

---

### Task 7: Build patient detail and episode setup

**Files:**
- Create: `frontend/components/rehabilitation/clinical/EpisodeForm.tsx`
- Create: `frontend/components/rehabilitation/clinical/ClinicalVisitTimeline.tsx`
- Create: `frontend/app/rehabilitation/clinical/patients/[patientId]/page.tsx`
- Create: `frontend/app/rehabilitation/clinical/patients/[patientId]/page.test.tsx`

- [ ] **Step 1: Write failing patient-detail tests**

Cover:

```typescript
it("shows clinical context, active episode, progress, and finalized visits", async () => {
  clinicalApi.getPatient.mockResolvedValue(patientDetail());
  render(<PatientDetailPage />);
  expect(await screen.findByText("Reach an overhead shelf")).toBeInTheDocument();
  expect(screen.getByText("142°")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /new visit/i }))
    .toHaveAttribute("href", "/rehabilitation/clinical/episodes/21/visits/new");
});

it("creates an episode when the patient has no active course", async () => {
  clinicalApi.getPatient.mockResolvedValue(patientDetail({ episodes: [] }));
  render(<PatientDetailPage />);
  await user.click(screen.getByRole("button", { name: /start episode/i }));
  await user.selectOptions(screen.getByLabelText(/protocol/i), "shoulder_flexion");
  await user.type(screen.getByLabelText(/functional goal/i), "Reach overhead");
  await user.click(screen.getByRole("button", { name: /create episode/i }));
  expect(clinicalApi.createEpisode).toHaveBeenCalled();
});
```

- [ ] **Step 2: Run and verify RED**

```bash
cd frontend && npm test -- "app/rehabilitation/clinical/patients/[patientId]/page.test.tsx"
```

Expected: missing route and components.

- [ ] **Step 3: Implement patient detail**

Reuse `RehabProgressChart`, `compareRehabProgress`, and protocol copy. Do not
duplicate comparison math. Show:

- profile and precautions
- active episode goal and configured targets
- baseline/current comparison
- episode-filtered progress
- visit timeline with draft, finalized, and warning badges
- episode creation and archive actions

- [ ] **Step 4: Run focused and progress tests**

```bash
cd frontend
npm test -- "app/rehabilitation/clinical/patients/[patientId]/page.test.tsx" lib/rehabProgress.test.ts components/rehabilitation/RehabProgressChart.test.tsx
npm run typecheck
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/rehabilitation/clinical frontend/app/rehabilitation/clinical/patients
git commit -m "feat: add clinical patient episode view"
```

---

### Task 8: Implement capture-readiness evaluation

**Files:**
- Create: `frontend/lib/captureReadiness.ts`
- Create: `frontend/lib/captureReadiness.test.ts`
- Create: `frontend/components/rehabilitation/clinical/CaptureReadinessPanel.tsx`

- [ ] **Step 1: Write failing readiness tests**

```typescript
it("blocks when target landmarks are missing", () => {
  expect(evaluateCaptureReadiness({
    protocol: "shoulder_flexion",
    poseDetected: true,
    landmarks: { left_shoulder: point() },
    cameraLevel: levelCamera(),
    confidence: 0.92,
  })).toMatchObject({ state: "blocked", code: "target_landmarks_missing" });
});

it("allows explicit acknowledgement for a tilted camera warning", () => {
  expect(evaluateCaptureReadiness({
    protocol: "shoulder_flexion",
    poseDetected: true,
    landmarks: completeShoulderLandmarks(),
    cameraLevel: { ...levelCamera(), status: "adjust", angle_deg: 4.5 },
    confidence: 0.91,
  })).toMatchObject({ state: "warning", code: "camera_adjust" });
});

it("returns ready for complete stable input", () => {
  expect(evaluateCaptureReadiness(readyInput())).toEqual({
    state: "ready",
    code: "ready",
    issues: [],
  });
});
```

- [ ] **Step 2: Run and verify RED**

```bash
cd frontend && npm test -- lib/captureReadiness.test.ts
```

Expected: missing module failure.

- [ ] **Step 3: Implement pure readiness rules**

Map each protocol to required landmarks. Hard-block:

- no pose
- missing target landmarks
- confidence below `0.45`

Warn:

- camera status `adjust` or `recalibrate`
- confidence from `0.45` through `0.69`
- incomplete optional contralateral landmarks

Ready:

- target landmarks present
- camera level or relatively calibrated
- confidence at least `0.70`

Render the result in `CaptureReadinessPanel` with a checklist and one explicit
warning-acknowledgement control. Keep thresholds in this module, not JSX.

- [ ] **Step 4: Run focused tests and typecheck**

```bash
cd frontend && npm test -- lib/captureReadiness.test.ts && npm run typecheck
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/captureReadiness.ts frontend/lib/captureReadiness.test.ts frontend/components/rehabilitation/clinical/CaptureReadinessPanel.tsx
git commit -m "feat: add clinical capture readiness"
```

---

### Task 9: Make live and upload analysis clinical-aware

**Files:**
- Modify: `frontend/components/rehabilitation/LiveRehabWorkspace.tsx`
- Modify: `frontend/components/rehabilitation/RehabUploader.tsx`
- Modify: `frontend/components/rehabilitation/LiveRehabWorkspace.test.tsx`
- Modify: `frontend/components/rehabilitation/RehabUploader.test.tsx`

- [ ] **Step 1: Write failing integration-prop tests**

Add props:

```typescript
type ClinicalSaveTarget = {
  athleteId: number;
  athleteName: string;
};

type AnalysisCallbacks = {
  saveTarget?: ClinicalSaveTarget;
  onAnalysisUpdate?: (snapshot: RehabAnalysisSnapshot) => void;
  onSessionSaved?: (sessionId: number) => void;
};
```

Test that live and upload:

- save with the explicit athlete ID
- emit the latest snapshot
- emit the saved session ID exactly once
- preserve current default behavior when props are absent

- [ ] **Step 2: Run and verify RED**

```bash
cd frontend && npm test -- components/rehabilitation/LiveRehabWorkspace.test.tsx components/rehabilitation/RehabUploader.test.tsx
```

Expected: prop and save payload assertion failures.

- [ ] **Step 3: Implement the minimal prop integration**

Do not add clinical routing or visit API calls inside the analysis components.
They only expose measurement and save events to the wizard owner.

- [ ] **Step 4: Run affected component tests**

```bash
cd frontend && npm test -- components/rehabilitation/LiveRehabWorkspace.test.tsx components/rehabilitation/RehabUploader.test.tsx app/rehabilitation/page.test.tsx
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/rehabilitation/LiveRehabWorkspace.tsx frontend/components/rehabilitation/RehabUploader.tsx frontend/components/rehabilitation/LiveRehabWorkspace.test.tsx frontend/components/rehabilitation/RehabUploader.test.tsx
git commit -m "feat: connect rehab analysis to clinical visits"
```

---

### Task 10: Build the five-stage clinical visit wizard

**Files:**
- Create: `frontend/components/rehabilitation/clinical/VisitReview.tsx`
- Create: `frontend/app/rehabilitation/clinical/episodes/[episodeId]/visits/new/page.tsx`
- Create: `frontend/app/rehabilitation/clinical/episodes/[episodeId]/visits/new/page.test.tsx`

- [ ] **Step 1: Write failing wizard tests**

Test the complete route state machine:

```typescript
it("creates a draft and advances through context, readiness, analysis, review, and summary", async () => {
  clinicalApi.getEpisode.mockResolvedValue(episodeDetail());
  clinicalApi.createVisit.mockResolvedValue(draftVisit());
  render(<NewClinicalVisitPage />);

  await user.type(screen.getByLabelText(/pre-session status/i), "No pain at rest");
  await user.click(screen.getByRole("button", { name: /continue/i }));
  expect(clinicalApi.createVisit).toHaveBeenCalledWith(21, {
    captureSource: "live",
    preSessionNote: "No pain at rest",
  });

  publishReadyCapture();
  await user.click(screen.getByRole("button", { name: /start analysis/i }));
  publishSavedSession(91);
  await user.click(screen.getByRole("button", { name: /review visit/i }));
  expect(screen.getByText(/baseline comparison/i)).toBeInTheDocument();
});

it("does not start analysis while readiness is blocked", async () => {
  renderReadyWizard();
  publishBlockedCapture();
  expect(screen.getByRole("button", { name: /start analysis/i })).toBeDisabled();
});

it("requires acknowledgement before proceeding with a readiness warning", async () => {
  renderReadyWizard();
  publishWarningCapture();
  expect(screen.getByRole("button", { name: /start analysis/i })).toBeDisabled();
  await user.click(screen.getByLabelText(/acknowledge warning/i));
  expect(screen.getByRole("button", { name: /start analysis/i })).toBeEnabled();
});
```

- [ ] **Step 2: Run and verify RED**

```bash
cd frontend && npm test -- "app/rehabilitation/clinical/episodes/[episodeId]/visits/new/page.test.tsx"
```

Expected: missing route failure.

- [ ] **Step 3: Implement the wizard**

Use a reducer with:

```typescript
type VisitStep = "context" | "readiness" | "analysis" | "review" | "summary";
```

Rules:

- create one draft when leaving context
- keep patient, episode, protocol, and source immutable during the draft
- live readiness consumes camera/pose updates before analysis
- upload readiness validates file presence and explains that pose quality is
  confirmed during analysis
- update draft with the returned `trainingSessionId`
- require capture quality and non-empty specialist observation
- finalize only through `clinicalApi.finalizeVisit`
- retry API failures without creating duplicate drafts

Keep the route file focused by delegating readiness and review UI to components.

- [ ] **Step 4: Run wizard, analysis component, and type tests**

```bash
cd frontend
npm test -- "app/rehabilitation/clinical/episodes/[episodeId]/visits/new/page.test.tsx" components/rehabilitation/LiveRehabWorkspace.test.tsx components/rehabilitation/RehabUploader.test.tsx
npm run typecheck
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/app/rehabilitation/clinical/episodes frontend/components/rehabilitation/clinical/VisitReview.tsx
git commit -m "feat: add clinical visit wizard"
```

---

### Task 11: Extend clinical handoff and visit comparison

**Files:**
- Modify: `frontend/components/rehabilitation/rehabHandoff.ts`
- Modify: `frontend/components/rehabilitation/rehabHandoff.test.ts`
- Modify: `frontend/components/rehabilitation/ClinicalReport.tsx`
- Modify: `frontend/components/rehabilitation/ClinicalReport.test.tsx`
- Modify: `frontend/components/rehabilitation/clinical/VisitReview.tsx`

- [ ] **Step 1: Write failing handoff/report tests**

```typescript
it("renders clinical identity, episode goal, comparison, quality, and specialist observation", () => {
  render(<ClinicalReport handoff={clinicalHandoff()} onClose={vi.fn()} />);
  expect(screen.getByText("Patient A")).toBeInTheDocument();
  expect(screen.getByText("Reach an overhead shelf")).toBeInTheDocument();
  expect(screen.getByText(/baseline.*\\+18°/i)).toBeInTheDocument();
  expect(screen.getByText(/accepted with warning/i)).toBeInTheDocument();
  expect(screen.getByText("ROM remains asymmetric.")).toBeInTheDocument();
});
```

Add optional fields:

```typescript
clinical?: {
  patientName: string;
  episodeTitle: string;
  functionalGoal: string;
  captureSource: "live" | "upload";
  captureQuality: CaptureQuality;
  qualityDetails: string;
  specialistObservation: string;
  baselineDelta: RehabDelta | null;
  previousDelta: RehabDelta | null;
};
```

- [ ] **Step 2: Run and verify RED**

```bash
cd frontend && npm test -- components/rehabilitation/rehabHandoff.test.ts components/rehabilitation/ClinicalReport.test.tsx
```

Expected: missing clinical context assertions.

- [ ] **Step 3: Implement optional clinical report sections**

Preserve demo and non-clinical reports. Render clinical context only when
`handoff.clinical` exists. Move patient and specialist notes out of ephemeral
report-local state for clinical reports; retain the editable local-only fields
for legacy handoffs.

- [ ] **Step 4: Run all handoff tests**

```bash
cd frontend && npm test -- components/rehabilitation/rehabHandoff.test.ts components/rehabilitation/ClinicalReport.test.tsx components/rehabilitation/RehabPresentationMode.test.tsx
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/rehabilitation/rehabHandoff.ts frontend/components/rehabilitation/rehabHandoff.test.ts frontend/components/rehabilitation/ClinicalReport.tsx frontend/components/rehabilitation/ClinicalReport.test.tsx frontend/components/rehabilitation/clinical/VisitReview.tsx
git commit -m "feat: add clinical visit handoff report"
```

---

### Task 12: Verify the complete pilot and document operation

**Files:**
- Modify: `README.md`
- Create: `docs/clinical-pilot.md`

- [ ] **Step 1: Add the end-to-end browser test checklist to the documentation**

Document:

```text
1. Create Patient A from an existing athlete.
2. Create a shoulder-flexion episode with bilateral ROM targets.
3. Start a live visit and deny camera permission; verify upload fallback.
4. Grant camera permission and verify blocked/warning/ready states.
5. Save a real analysis to Patient A.
6. Add a specialist observation and finalize.
7. Confirm the visit appears in patient progress.
8. Open UK and EN reports and invoke print preview.
9. Repeat at 1440px desktop and 768px tablet.
```

Include the backup filename pattern and recovery command without instructing
users to edit SQLite manually.

- [ ] **Step 2: Run the complete local verification**

```bash
.venv/bin/pytest tests/unit/ -v
.venv/bin/ruff check .
.venv/bin/black --check .
.venv/bin/isort --check-only .
.venv/bin/mypy video_analysis --ignore-missing-imports --no-strict-optional
cd frontend && npm test
cd frontend && npm run typecheck
cd frontend && npm run build
git diff --check
```

Expected:

- all backend unit tests pass
- all frontend tests pass
- no lint, format, type, build, or whitespace failures

- [ ] **Step 3: Run browser QA against the live stack**

Use the in-app browser at:

```text
http://127.0.0.1:3000/rehabilitation/clinical
```

Verify the documented flow with real local API responses. Inspect browser
console errors, network failures, overflow at 1440x900 and 768x1024, focus
return from overlays, UK/EN switching, and print layout.

- [ ] **Step 4: Fix every browser-discovered defect through TDD**

For each defect:

1. add a focused failing Vitest or pytest regression test
2. verify the expected failure
3. implement the smallest fix
4. rerun the focused and affected suites

- [ ] **Step 5: Update product documentation**

Add Clinical Pilot Mode to `README.md` with:

- local specialist workflow
- patient/episode/visit model
- capture-quality gate
- bilingual clinical report
- explicit research-prototype and local-data statements

- [ ] **Step 6: Run final verification again**

Repeat every command from Step 2 after all QA fixes. Confirm
`data/athletes.db` and `.superpowers/` remain outside staged changes.

- [ ] **Step 7: Commit**

```bash
git add README.md docs/clinical-pilot.md
git commit -m "docs: add clinical pilot workflow"
```

- [ ] **Step 8: Publish for review**

```bash
git push -u origin codex/clinical-pilot-mode
gh pr create --draft --base main --head codex/clinical-pilot-mode \
  --title "[codex] Add Clinical Pilot Mode" \
  --body-file /tmp/clinical-pilot-pr.md
gh pr checks --watch --interval 10
```

The PR body must summarize the clinical workflow, additive database behavior,
identity-safe rehabilitation saves, browser QA, and all test counts.
