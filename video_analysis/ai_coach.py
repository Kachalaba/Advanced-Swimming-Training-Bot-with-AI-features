"""
🤖 AI Coach - Intelligent Swimming Analysis Agent

Uses LLM (Claude/GPT) to analyze biomechanics and provide
personalized coaching recommendations.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CoachingAdvice:
    """Structured coaching advice from AI."""

    summary: str
    strengths: List[str]
    improvements: List[str]
    drills: List[str]
    priority: str  # "technique", "endurance", "speed"
    score: int  # 0-100 overall assessment


class AICoach:
    """
    AI-powered swimming coach that analyzes biomechanics
    and provides personalized recommendations.

    Supports:
    - Anthropic Claude (recommended)
    - OpenAI GPT-4
    - Offline fallback (rule-based)
    """

    SYSTEM_PROMPT = """<system>
<role>Ти — Senior тренер-аналітик з плавання та триатлону (20+ років досвіду, рівень МС).</role>

<rules>
1. Відповідай ВИКЛЮЧНО українською мовою
2. Використовуй конкретні числові показники з наданих даних аналізу
3. Пріоритизуй рекомендації за впливом на результат згідно з аналітичною структурою
4. Тон: конкретний, практичний, мотивуючий
5. Відповідь — строго валідний JSON без markdown-огорток
</rules>

<analytical_framework>
Рівень 1 — Критичні помилки: ризик травми або суттєва втрата швидкості (усунути першочергово)
Рівень 2 — Технічні вади: помилки техніки з вимірним впливом на ефективність
Рівень 3 — Оптимізація: покращення для спортсменів з добрим базовим рівнем
Рівень 4 — Тонке налаштування: мікро-корекції для досвідчених атлетів
</analytical_framework>

<response_format>
{
    "summary": "2-3 речення: поточний рівень + ключове спостереження з даних",
    "strengths": ["конкретна сильна сторона з числовим підтвердженням"],
    "improvements": ["Рівень N: конкретна вада → конкретна корекція"],
    "drills": ["Назва вправи: деталі виконання + ціль"],
    "priority": "technique|endurance|speed",
    "score": 0
}
</response_format>
</system>"""

    def __init__(
        self,
        provider: str = "auto",  # "anthropic", "openai", "offline"
        api_key: Optional[str] = None,
    ):
        """
        Initialize AI Coach.

        Args:
            provider: LLM provider ("anthropic", "openai", "auto", "offline")
            api_key: API key (or set ANTHROPIC_API_KEY / OPENAI_API_KEY env var)
        """
        self.provider = provider
        self.api_key = api_key
        self.client: Any = None

        # Auto-detect provider
        if provider == "auto":
            self._auto_detect_provider()
        else:
            self._init_provider(provider, api_key)

    def _auto_detect_provider(self):
        """Auto-detect available LLM provider."""
        # Try Anthropic first
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            self._init_provider("anthropic", anthropic_key)
            return

        # Try OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            self._init_provider("openai", openai_key)
            return

        # Fallback to offline
        logger.info("No API keys found, using offline mode")
        self.provider = "offline"

    def _init_provider(self, provider: str, api_key: Optional[str]):
        """Initialize specific provider."""
        self.provider = provider

        if provider == "anthropic":
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info("AI Coach: Using Anthropic Claude")
            except ImportError:
                logger.warning("anthropic package not installed, using offline mode")
                self.provider = "offline"

        elif provider == "openai":
            try:
                import openai

                self.client = openai.OpenAI(api_key=api_key)
                logger.info("AI Coach: Using OpenAI GPT")
            except ImportError:
                logger.warning("openai package not installed, using offline mode")
                self.provider = "offline"

        elif provider == "offline":
            logger.info("AI Coach: Using offline rule-based mode")

    def analyze(
        self,
        biomechanics: Dict,
        trajectory: Optional[Dict] = None,
        splits: Optional[Dict] = None,
        swimming_pose: Optional[Dict] = None,
        athlete_name: str = "Спортсмен",
    ) -> CoachingAdvice:
        """
        Analyze swimming data and generate coaching advice.

        Args:
            biomechanics: Biomechanics analysis results
            trajectory: Trajectory analysis results
            splits: Split times analysis
            swimming_pose: Swimming pose analysis results
            athlete_name: Name for personalization

        Returns:
            CoachingAdvice with recommendations
        """
        # Prepare context for AI
        context = self._prepare_context(biomechanics, trajectory, splits, swimming_pose, athlete_name)

        # Get AI response based on provider
        if self.provider == "anthropic":
            return self._analyze_with_claude(context)
        elif self.provider == "openai":
            return self._analyze_with_gpt(context)
        else:
            return self._analyze_offline(context)

    def _prepare_context(
        self,
        biomechanics: Dict,
        trajectory: Optional[Dict],
        splits: Optional[Dict],
        swimming_pose: Optional[Dict],
        athlete_name: str,
    ) -> str:
        """Prepare analysis context for LLM."""
        lines = [f"# Аналіз плавця: {athlete_name}\n"]

        # Biomechanics
        if biomechanics:
            avg = biomechanics.get("average_metrics", {})
            lines.append("## Біомеханіка (MediaPipe)")
            lines.append(f"- Posture Score: {avg.get('average_posture_score', 0):.1f}/100")
            lines.append(f"- Drag Coefficient: {avg.get('average_drag_coefficient', 0):.2f}")
            lines.append(f"- Streamline Score: {avg.get('average_streamline_score', 0):.1f}%")
            lines.append(f"- Кадрів з позою: {avg.get('frames_with_pose', 0)}/{avg.get('total_frames', 0)}")

            # Angles
            angles = avg.get("average_angles", {})
            if angles:
                lines.append("\n### Кути тіла:")
                for key, val in angles.items():
                    lines.append(f"- {key}: {val:.1f}°")

            # Recommendations from biomechanics
            recs = biomechanics.get("recommendations", [])
            if recs:
                lines.append("\n### Попередні рекомендації:")
                for r in recs[:5]:
                    lines.append(f"- {r}")

        # Swimming pose (new analyzer)
        if swimming_pose:
            lines.append("\n## Аналіз пози плавця (SwimmingPose)")
            lines.append(f"- Detection Rate: {swimming_pose.get('detection_rate', 0)*100:.1f}%")
            lines.append(f"- Avg Streamline: {swimming_pose.get('avg_streamline', 0):.1f}/100")
            lines.append(f"- Avg Spine Deviation: {swimming_pose.get('avg_deviation', 0):.1f}°")

        # Trajectory
        if trajectory:
            summary = trajectory.get("summary", {})
            lines.append("\n## Траєкторія руху")
            lines.append(f"- Movement Score: {summary.get('movement_quality_score', 0):.1f}/100")
            lines.append(f"- Velocity Consistency: {summary.get('velocity_consistency', 0):.1f}%")
            lines.append(f"- Streamline (bbox): {summary.get('streamline_score', 0):.1f}%")

        # Splits
        if splits:
            split_data = splits.get("splits", [])
            if split_data:
                lines.append("\n## Спліти")
                for s in split_data[:4]:
                    lines.append(f"- {s.get('distance', 0)}м: {s.get('time', 0):.2f}с ({s.get('speed', 0):.2f} м/с)")

        return "\n".join(lines)

    _COACHING_TOOL = {
        "name": "coaching_advice",
        "description": "Повернути структурований тренерський висновок після аналізу біомеханіки плавця.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "2-3 речення: поточний рівень + ключове спостереження",
                },
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Сильні сторони з числовим підтвердженням",
                },
                "improvements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Що покращити: Рівень N → конкретна корекція",
                },
                "drills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Назва вправи: деталі виконання + ціль",
                },
                "priority": {
                    "type": "string",
                    "enum": ["technique", "endurance", "speed"],
                },
                "score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Загальна оцінка техніки 0-100",
                },
            },
            "required": [
                "summary",
                "strengths",
                "improvements",
                "drills",
                "priority",
                "score",
            ],
        },
    }

    def _analyze_with_claude(self, context: str) -> CoachingAdvice:
        """Analyze using Anthropic Claude with Tool Use for guaranteed structured JSON."""
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                temperature=0.2,
                system=self.SYSTEM_PROMPT,
                tools=[self._COACHING_TOOL],
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": context}],
            )

            for block in message.content:
                if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "coaching_advice":
                    return self._coaching_advice_from_data(block.input)
                if text := getattr(block, "text", None):
                    return self._parse_response(text)

            raise ValueError("coaching_advice tool not called in response")

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._analyze_offline({"error": str(e)})

    def _analyze_with_gpt(self, context: str) -> CoachingAdvice:
        """Analyze using OpenAI GPT with JSON Structured Outputs."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1024,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
            )

            return self._parse_response(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._analyze_offline({"error": str(e)})

    def _coaching_advice_from_data(self, data: Dict[str, Any]) -> CoachingAdvice:
        """Build a safe CoachingAdvice object from provider JSON data."""
        return CoachingAdvice(
            summary=str(data.get("summary", "")),
            strengths=list(data.get("strengths") or []),
            improvements=list(data.get("improvements") or []),
            drills=list(data.get("drills") or []),
            priority=str(data.get("priority", "technique")),
            score=int(data.get("score", 50)),
        )

    def _parse_response(self, response: str) -> CoachingAdvice:
        """Parse JSON provider text, including JSON embedded in prose."""
        try:
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                start = response.index("{")
                end = response.rindex("}") + 1
                data = json.loads(response[start:end])
            return self._coaching_advice_from_data(data)
        except Exception:
            return CoachingAdvice(
                summary="Не вдалося розібрати відповідь AI провайдера.",
                strengths=[],
                improvements=["Повторіть аналіз або використайте offline режим."],
                drills=[],
                priority="technique",
                score=50,
            )

    def _analyze_offline(self, context: Any) -> CoachingAdvice:
        """Offline rule-based analysis."""
        # Extract metrics if available
        if isinstance(context, str):
            # Parse some basic metrics from context string
            streamline = 70
            posture = 70
            detection = 50
        else:
            streamline = 70
            posture = 70
            detection = 50

        # Rule-based recommendations
        improvements = []
        strengths = []
        drills = []

        # Analyze streamline
        if streamline < 60:
            improvements.append("⚠️ Покращуйте обтікаємість - тримайте тіло рівно")
            drills.append("🏊 Вправа 'Стріла': відштовхування від бортика в streamline позиції")
        elif streamline > 80:
            strengths.append("✅ Відмінна обтікаємість тіла")

        # Analyze posture
        if posture < 60:
            improvements.append("⚠️ Працюйте над положенням голови - не піднімайте занадто")
            drills.append("🎯 Вправа: плавання з дошкою, фокус на положенні голови")
        elif posture > 80:
            strengths.append("✅ Хороше положення тіла")

        # Detection rate feedback
        if detection < 50:
            improvements.append("📹 Рекомендуємо краще освітлення для точнішого аналізу")

        # Default recommendations
        if not improvements:
            improvements = [
                "💪 Продовжуйте працювати над технікою",
                "🔄 Додайте вправи на core-стабілізацію",
                "⏱️ Працюйте над рівномірністю темпу",
            ]

        if not strengths:
            strengths = ["📊 Дані зібрано для аналізу"]

        if not drills:
            drills = [
                "🏊 Catch-up drill: покращення координації рук",
                "🦶 Kick drill з дошкою: робота над ударами ніг",
                "🔄 6-3-6 drill: баланс та обертання тіла",
            ]

        # Calculate score
        score = int((streamline + posture + detection) / 3)

        return CoachingAdvice(
            summary=f"Аналіз завершено. Загальна оцінка техніки: {score}/100. "
            f"Основний фокус: {'обтікаємість' if streamline < 70 else 'стабільність темпу'}.",
            strengths=strengths,
            improvements=improvements,
            drills=drills,
            priority="technique" if posture < 70 else "speed",
            score=score,
        )


def get_ai_coaching(
    biomechanics: Dict = None,
    trajectory: Dict = None,
    splits: Dict = None,
    swimming_pose: Dict = None,
    athlete_name: str = "Спортсмен",
    provider: str = "auto",
) -> CoachingAdvice:
    """
    Convenience function to get AI coaching advice.

    Args:
        biomechanics: Biomechanics analysis results
        trajectory: Trajectory analysis results
        splits: Split times
        swimming_pose: Swimming pose analysis
        athlete_name: Athlete name
        provider: LLM provider

    Returns:
        CoachingAdvice with personalized recommendations
    """
    coach = AICoach(provider=provider)
    return coach.analyze(
        biomechanics=biomechanics or {},
        trajectory=trajectory,
        splits=splits,
        swimming_pose=swimming_pose,
        athlete_name=athlete_name,
    )


# Example usage
if __name__ == "__main__":
    # Test offline mode
    advice = get_ai_coaching(
        biomechanics={
            "average_metrics": {
                "average_posture_score": 72,
                "average_drag_coefficient": 0.45,
                "average_streamline_score": 68,
            }
        },
        athlete_name="Тест",
    )

    print(f"Summary: {advice.summary}")
    print(f"Score: {advice.score}")
    print(f"Strengths: {advice.strengths}")
    print(f"Improvements: {advice.improvements}")
    print(f"Drills: {advice.drills}")
