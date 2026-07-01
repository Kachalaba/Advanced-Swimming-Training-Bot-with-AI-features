"""
Service layer and interface contracts for AI coaching and analysis.

Defines:
- ``AICoachProtocol``   — interface any AI coach implementation must satisfy
- ``AnalysisService``   — coordinates analyzers, AI coach, and persistence
                         for a single analysis session

The service layer decouples page code (Streamlit UI) from business logic
so pages only call ``AnalysisService`` and never import individual analyzers
or database helpers directly.

Example usage in a page::

    from video_analysis.service_layer import AnalysisService
    service = AnalysisService(athlete_id=1, fps=30.0)
    report  = service.run_swimming_analysis(keypoints_list)
    service.save(report)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interfaces / Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class AICoachProtocol(Protocol):
    """Minimal interface every AI coaching implementation must satisfy.

    Both ``AICoach`` and ``AIChat`` already implement these methods;
    this protocol makes it explicit and enables type-checking.
    """

    def chat(self, user_message: str) -> str:
        """Return an AI-generated coaching response to *user_message*."""
        ...

    def set_context(self, analysis_results: Dict) -> None:
        """Load analysis results as context for subsequent ``chat`` calls."""
        ...


# ---------------------------------------------------------------------------
# Shared result container
# ---------------------------------------------------------------------------


@dataclass
class AnalysisReport:
    """Unified container returned by ``AnalysisService.run_*`` methods."""

    sport: str  # "swimming" | "running" | "cycling" | "dryland"
    athlete_id: Optional[int] = None
    athlete_name: str = "Unknown"
    fps: float = 30.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    ai_coaching: Optional[Dict] = None  # Result from AICoach.analyze / AIChat.chat
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if report contains metrics and no fatal errors."""
        return bool(self.metrics) and not self.errors


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class AnalysisService:
    """Coordinates sport-specific analyzers, AI coaching, and DB persistence.

    Instantiate once per analysis session (or use ``analyzer_factory`` for
    cached instances). All heavy objects (analyzers, AI client) are lazy-loaded
    on first use.

    Args:
        athlete_id:   Database row ID of the current athlete (optional).
        athlete_name: Display name used in coaching prompts.
        fps:          Frame rate of the source video.
        ai_coach:     Optional pre-built coach implementing ``AICoachProtocol``.
                      If *None*, the service creates an ``AICoach`` from env vars.
    """

    def __init__(
        self,
        athlete_id: Optional[int] = None,
        athlete_name: str = "Unknown",
        fps: float = 30.0,
        ai_coach: Optional[AICoachProtocol] = None,
    ) -> None:
        self.athlete_id = athlete_id
        self.athlete_name = athlete_name
        self.fps = fps
        self._ai_coach = ai_coach
        self._stroke_analyzer = None
        self._running_analyzer = None
        self._cycling_analyzer = None

    # ------------------------------------------------------------------
    # Lazy accessor helpers
    # ------------------------------------------------------------------

    def _get_stroke_analyzer(self):
        if self._stroke_analyzer is None:
            from video_analysis.analyzer_factory import get_stroke_analyzer

            self._stroke_analyzer = get_stroke_analyzer(fps=self.fps)
        return self._stroke_analyzer

    def _get_running_analyzer(self):
        if self._running_analyzer is None:
            from video_analysis.analyzer_factory import get_running_analyzer

            self._running_analyzer = get_running_analyzer(fps=self.fps)
        return self._running_analyzer

    def _get_cycling_analyzer(self):
        if self._cycling_analyzer is None:
            from video_analysis.analyzer_factory import get_cycling_analyzer

            self._cycling_analyzer = get_cycling_analyzer(fps=self.fps)
        return self._cycling_analyzer

    def _get_ai_coach(self) -> AICoachProtocol:
        if self._ai_coach is None:
            from video_analysis.ai_chat import AIChat

            # AIChat handles a missing API key / SDK internally by switching
            # to its keyword-based fallback, so no try/except is needed here.
            self._ai_coach = AIChat(athlete_name=self.athlete_name)
        return self._ai_coach

    # ------------------------------------------------------------------
    # Public analysis methods
    # ------------------------------------------------------------------

    def run_swimming_analysis(self, keypoints_list: List[Dict]) -> AnalysisReport:
        """Run stroke + AI coaching for a swimming session.

        Args:
            keypoints_list: Per-frame keypoint dicts from SwimmingPoseAnalyzer.

        Returns:
            AnalysisReport with ``metrics`` populated from stroke analysis
            and ``ai_coaching`` from the AI coach.
        """
        report = AnalysisReport(
            sport="swimming",
            athlete_id=self.athlete_id,
            athlete_name=self.athlete_name,
            fps=self.fps,
        )
        try:
            analyzer = self._get_stroke_analyzer()
            stroke_result = analyzer.analyze(keypoints_list, fps=self.fps)
            report.metrics["stroke_analysis"] = stroke_result

            coach = self._get_ai_coach()
            coach.set_context(report.metrics)
            report.ai_coaching = {"response": coach.chat("Проаналізуй результати плавання")}
        except Exception as exc:
            logger.error("Swimming analysis failed: %s", exc, exc_info=True)
            report.errors.append(str(exc))
        return report

    def run_running_analysis(self, keypoints_list: List[Dict]) -> AnalysisReport:
        """Run running biomechanics + AI coaching."""
        report = AnalysisReport(
            sport="running",
            athlete_id=self.athlete_id,
            athlete_name=self.athlete_name,
            fps=self.fps,
        )
        try:
            analyzer = self._get_running_analyzer()
            run_result = analyzer.analyze(keypoints_list, fps=self.fps)
            report.metrics["running_analysis"] = run_result

            coach = self._get_ai_coach()
            coach.set_context(report.metrics)
            report.ai_coaching = {"response": coach.chat("Проаналізуй результати бігу")}
        except Exception as exc:
            logger.error("Running analysis failed: %s", exc, exc_info=True)
            report.errors.append(str(exc))
        return report

    def run_cycling_analysis(self, keypoints_list: List[Dict]) -> AnalysisReport:
        """Run cycling biomechanics + AI coaching."""
        report = AnalysisReport(
            sport="cycling",
            athlete_id=self.athlete_id,
            athlete_name=self.athlete_name,
            fps=self.fps,
        )
        try:
            analyzer = self._get_cycling_analyzer()
            cycling_result = analyzer.analyze(keypoints_list, fps=self.fps)
            report.metrics["cycling_analysis"] = cycling_result

            coach = self._get_ai_coach()
            coach.set_context(report.metrics)
            report.ai_coaching = {"response": coach.chat("Проаналізуй результати велосипеду")}
        except Exception as exc:
            logger.error("Cycling analysis failed: %s", exc, exc_info=True)
            report.errors.append(str(exc))
        return report

    def save(self, report: AnalysisReport) -> bool:
        """Persist report to the athlete database.

        Returns True on success, False if persistence is unavailable or fails.
        """
        if not report.ok or report.athlete_id is None:
            return False
        try:
            from video_analysis.athlete_database import save_analysis_to_db

            save_analysis_to_db(
                athlete_name=report.athlete_name,
                session_type=report.sport,
                analysis=report.metrics,
            )
            return True
        except Exception as exc:
            logger.error("Failed to save analysis: %s", exc)
            return False
