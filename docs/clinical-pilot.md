# Clinical Pilot Mode

Clinical Pilot Mode is a local-first workflow for a single rehabilitation
specialist. It organizes SPRINT AI measurements into patient profiles,
rehabilitation episodes, quality-controlled visits, progress comparisons, and
bilingual handoff reports.

## Scope

- Route: `http://127.0.0.1:3000/rehabilitation/clinical`
- Languages: Ukrainian and English
- Capture: live camera or uploaded video
- Storage: local SQLite only
- Intended use: movement observation and professional discussion
- Not included: diagnosis, treatment prescription, prognosis, authentication,
  cloud synchronization, or concurrent multi-user work

## Workflow

1. Create or select an athlete in the existing history.
2. Open the specialist workspace and create a clinical profile linked to that
   athlete ID.
3. Create an episode with one immutable measurement protocol, a functional
   goal, and optional left/right ROM targets.
4. Start a visit, choose live or upload, and record the pre-session status.
5. Resolve the readiness state:
   - `Ready`: target landmarks, camera level, and confidence are usable.
   - `Acknowledgement required`: usable with a documented limitation.
   - `Analysis blocked`: pose, target landmarks, or confidence are inadequate.
6. Save the completed analysis to the linked athlete.
7. Add the specialist observation and classify measurement quality.
8. Finalize the visit. Only finalized usable visits enter progress charts.
9. Open the Clinical Handoff Pack, switch UK/EN if needed, and use the browser
   print dialog to export PDF.

## Data Model

- Patient profile: display identity, affected side, clinical context, and
  precautions linked to one existing athlete ID.
- Rehabilitation episode: protocol, functional goal, ROM targets, and status.
- Clinical visit: source, pre-session note, linked analysis session, quality,
  specialist observation, acknowledgement, and finalization status.

The clinical tables are additive. Existing athlete and training-session rows
are not rewritten.

## Backup And Recovery

Before the first clinical schema migration of a non-empty database, SPRINT AI
creates one timestamped sibling copy:

```text
athletes.pre-clinical-YYYYMMDDTHHMMSSffffffZ.db
```

To restore the newest automatic backup for the default local database, stop
the application first and run:

```bash
DB_PATH="${ATHLETE_DB_PATH:-data/athletes.db}"
BACKUP="$(ls -1t data/athletes.pre-clinical-*.db | head -1)"
cp "$DB_PATH" "${DB_PATH}.before-recovery-$(date -u +%Y%m%dT%H%M%SZ)"
cp "$BACKUP" "$DB_PATH"
```

For a custom `ATHLETE_DB_PATH`, use the matching `*.pre-clinical-*.db` file in
the same directory. Do not edit the SQLite tables manually.

## Demonstration Checklist

1. Create Patient A from an existing athlete.
2. Create a shoulder-flexion episode with bilateral ROM targets.
3. Start a live visit and deny camera permission; verify the upload option
   remains available.
4. Grant camera permission and demonstrate blocked, warning, and ready states.
5. Save a real analysis to Patient A.
6. Add a specialist observation and finalize the visit.
7. Confirm the visit appears in patient progress.
8. Open Ukrainian and English reports and invoke print preview.
9. Repeat at 1440 x 900 desktop and 768 x 1024 tablet sizes.

## Pilot Safety Notes

- Use coded or test identities when presenting outside the care team.
- Treat a warning measurement as limited evidence and document the limitation.
- Repeat any measurement classified as `Repeat required`.
- Interpret ROM and symmetry together with the patient context and specialist
  assessment.
- Back up the local database before moving it to another computer.
