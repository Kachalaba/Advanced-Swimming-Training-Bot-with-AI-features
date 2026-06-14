# Changelog

## [Unreleased]
### Added
- Added a quality-gated cycling web workflow with upload, SSE progress,
  annotated video, cadence and joint-geometry evidence, athlete save, and
  persisted history.
- Added a visible swimming waterline baseline contract with surface position,
  confidence, frame coverage, and temporal drift evidence.
- Added normalized athlete sport overviews for swimming, running, cycling, and
  dryland without inventing data for empty or malformed sessions.
- Added persisted running analysis, stable athlete selection, idempotent saves,
  and durable copies of annotated running/swimming videos.
- Added real running and swimming history to the Next.js sport landing pages.
- Added frontend and API tests for sport history, running save, and honest empty
  states.
- Extracted the static Streamlit theme into `assets/styles.css`, loaded once by
  `app.py` (only the light/dark `:root` variables remain dynamic).

### Changed
- Cycling scoring now excludes unavailable joints instead of treating missing
  evidence as a failed measurement.
- Swimming and cycling services share one MediaPipe processing lock so cached
  pose inference cannot overlap across concurrent jobs.
- Declared Next.js + FastAPI as the primary product and Streamlit as the frozen
  legacy/demo shell.
- Replaced fabricated cycling and dryland athlete metrics with truthful
  capability and web-readiness states.
- `StrokeAnalyzer` and `ExerciseAnalyzer` now inherit `BaseAnalyzer`, removing
  duplicated `_get_point` / `_smooth` helpers (per the architecture guideline).
- Refreshed `CLAUDE.md`, `AGENTS.md`, and `ARCHITECTURE.md` to match the current
  8-tab Streamlit shell, the FastAPI + Next.js web app, and the four CI workflows.

### Fixed
- Removed the production build dependency on downloading Google Fonts so the
  local pilot can build and start without external font access.
- Corrected cycling top/bottom knee-angle semantics, made extrema robust to
  outliers, and reset temporal state between clips.
- `StrokeAnalyzer` keypoint lookup now resolves the canonical integer-indexed
  MediaPipe format (`{11: {"x", "y"}}`) via the inherited `_get_point`, which the
  previous duplicate silently dropped.

### Removed
- Deleted the stray root-level `test_simple.py`, a leftover manual script from the
  retired Telegram-bot project (referenced non-existent `bot.py`/`handlers/`).

## [1.0.0] - 2025-10-15
### Added
- Documented self-documentation roadmap in `REPORT.md` для оновлення інструкцій українською.
- Added українські гайдлайни: оновлений README з шильдами, `SETUP.md`, `ARCHITECTURE.md`, `OPERATIONS.md`, демо-ілюстрації.
- Added `/ping` healthcheck handler, pytest-based contract tests, and CI workflow exporting coverage.
- Added тестовые фабрики и фейковые клиенты Sheets/Telegram для unit-тестов.
- Added `SECURITY_NOTES.md` documenting hardening rules and next steps.
- Planned тестовую инфраструктуру (фабрики, фейки, pytest/CI) и задокументировано в `REPORT.md`.
- Planned CI/CD pipeline rollout: pre-commit hooks, strict mypy, GitHub Actions раздельные пайплайны и обновление Makefile/README.
- Added `.pre-commit-config.yaml`, dev-зависимости (black, isort, ruff, mypy, pre-commit) и README-бейджі CI/CD.
- Added GitHub Actions workflows `lint.yml`, `tests.yml`, `docker.yml` (buildx + semver-теги) вместо монолитного `ci.yml`/`docker-publish.yml`.
- Technical audit report summarised in `REPORT_AUDIT.md` and `REPORT.md`.
- Added architecture migration plan (`ARCH_PLAN.md`) and domain/application/infrastructure skeleton.
- Repository map and storage migration roadmap documented in `REPORT.md`.
- Defined storage layer contracts (`AthletesRepo`, `CoachesRepo`, `ResultsRepo`, `RecordsRepo`) and `Storage` facade.
- Added Google Sheets storage implementation with configurable backend selection via `.env`.
- Introduced Postgres storage layer (SQLAlchemy models, repositories) and updated dependencies.
- Added Alembic configuration and initial migration for Postgres schema.
- Added migration tooling (`Makefile` targets) and batch import script from Sheets to Postgres.
- Added `sprint_bot.domain.analytics` with canonical swim metrics and dedicated tests.
- Added onboarding scenario tests (`tests/test_onboarding_flow.py`) and UX playbook с mermaid-діаграммами в `docs/UX.md`.
- Added async export module with `/export_csv`, `/export_xlsx`, `/export_graphs`, caching, tests, and docs (`docs/reports.md`).

### Changed
- Reused domain analytics across handlers, reports and notifications to remove duplicated formulas and improve consistency.
- Замінили демонстраційні PNG (UX-плейбук, README) на Mermaid-діаграми, щоб уникнути зберігання бінарних файлів у репозиторії.
- Reworked Makefile (`format`, `lint`, `test`, `build`, `run`), ужесточён `mypy` (`strict` для `sprint_bot.domain` и `services`), типизированы сервисы (`base`, `stats_service`, `user_service`).
- Hardened observability by masking chat/user identifiers in logs & Sentry, enforced client timeouts, and added docker healthcheck.

### Fixed
- Suppressed unused exception binding in Google Sheets storage to satisfy `ruff` static checks.
