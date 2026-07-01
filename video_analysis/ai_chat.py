"""
🤖 AI Chat & Training Plan Generator

Features:
- Interactive chat about swimming/dryland technique (LLM-powered if ANTHROPIC_API_KEY set)
- Text-to-Speech (TTS) for recommendations
- Automatic training plan generation
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Claude model for coaching chat. claude-3-5-sonnet-20241022 was retired by
# Anthropic (Oct 2025) and now returns 404, silently degrading the chat to the
# keyword fallback. Overridable without a code change via env var.
CLAUDE_MODEL = os.environ.get("SPRINT_AI_CLAUDE_MODEL", "claude-sonnet-5")


@dataclass
class ChatMessage:
    """Chat message."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: str = ""


@dataclass
class TrainingPlan:
    """Weekly training plan."""

    athlete_name: str
    goal: str
    level: str
    weeks: int
    sessions: List[Dict] = field(default_factory=list)
    notes: str = ""


# ============================================================================
# KNOWLEDGE BASE
# ============================================================================

SWIMMING_KNOWLEDGE = {
    "freestyle": {
        "catch": "Catch - це перша фаза гребка. Рука входить у воду перед плечем, пальці направлені вперед і вниз. Тримайте лікоть високо (high elbow catch).",
        "pull": "Pull - тягнете воду назад. Рука рухається під тілом, лікоть згинається до 90°. Уявіть що тягнете за канат.",
        "push": "Push - завершальна фаза під водою. Рука випрямляється біля стегна, долоня направлена назад.",
        "recovery": "Recovery - рука виходить з води ліктем вперед, розслаблена. Рухається над водою до входу.",
        "body_roll": "Body roll (обертання тіла) оптимальне 30-50°. Допомагає ефективному гребку та диханню.",
        "breathing": "Дихання - поворот голови в бік під час recovery, один очний яблук залишається у воді.",
        "kick": "Удари ногами від стегна, коліна трохи зігнуті. 6-beat kick для спринту, 2-beat для дистанції.",
    },
    "drills": {
        "catch_up": "Catch-up drill: одна рука чекає попереду поки інша завершить повний цикл.",
        "fingertip_drag": "Fingertip drag: проводьте пальцями по воді під час recovery для високого ліктя.",
        "fist_drill": "Fist drill: плавайте із стиснутими кулаками для відчуття forearm.",
        "single_arm": "Single arm drill: гребок однією рукою, інша вздовж тіла.",
        "catch_up_pause": "Catch-up з паузою: пауза 2с коли руки зустрічаються попереду.",
    },
    "common_errors": {
        "dropped_elbow": "Опущений лікоть - втрата потужності. Тримайте лікоть вище кисті під час catch.",
        "crossover": "Перехрест центральної лінії - рука заходить за центр. Входьте на ширині плеча.",
        "flat_body": "Плоске тіло - відсутній body roll. Обертайтеся 30-50° в кожен бік.",
        "head_lift": "Підняття голови - порушує баланс. Тримайте голову нейтрально, дивіться на дно.",
        "scissor_kick": "Ножиці ногами - широкий удар. Тримайте ноги близько, удар від стегна.",
    },
}

DRYLAND_KNOWLEDGE = {
    "exercises": {
        "lat_pulldown": "Lat pulldown імітує фазу pull. Тягніть до грудей, стискайте лопатки.",
        "tricep_extension": "Tricep extension для сили push фази. Повна амплітуда руху.",
        "band_pull": "Резинка для імітації гребка. Тягніть до стегна, контролюйте повернення.",
        "plank": "Планка для core stability. Тримайте тіло рівно, не провисайте.",
        "flutter_kicks": "Flutter kicks лежачи для сили ніг. Швидкі малі рухи.",
    },
    "stretching": {
        "shoulder": "Розтяжка плечей: рука поперек грудей, притисніть іншою рукою.",
        "tricep": "Розтяжка трицепса: рука за головою, притисніть лікоть вниз.",
        "hip_flexor": "Розтяжка hip flexor: випад вперед, заднє коліно на підлозі.",
        "ankle": "Розтяжка гомілковостопу: сидячи, тягніть носок на себе.",
    },
}


# ============================================================================
# AI CHAT
# ============================================================================

_SYSTEM_PROMPT = """<system>
<role>Ти — Senior тренер-аналітик з плавання та триатлону (20+ років досвіду, рівень МС).</role>

<rules>
1. Відповідай ВИКЛЮЧНО українською мовою
2. Будь конкретним та практичним — посилайся на числові показники з контексту спортсмена
3. Тон: професійний, мотивуючий, без зайвих слів
4. Відповідь — не більше 300 слів
5. Якщо є дані аналізу — аналізуй їх; не давай загальних порад без опори на дані
</rules>

<analytical_framework>
Рівень 1 — Критичні помилки: ризик травми або суттєва втрата швидкості (усунути першочергово)
Рівень 2 — Технічні вади: помилки техніки з вимірним впливом на ефективність
Рівень 3 — Оптимізація: покращення для спортсменів з добрим базовим рівнем
Рівень 4 — Тонке налаштування: мікро-корекції для досвідчених атлетів
</analytical_framework>
</system>"""


class AIChat:
    """AI Chat for swimming and dryland questions.

    Uses Anthropic Claude API when ANTHROPIC_API_KEY is set;
    falls back to keyword-based responses otherwise.
    """

    def __init__(self, athlete_data: Dict = None, athlete_name: str = "Спортсмен"):
        self.athlete_data = athlete_data or {}
        self.athlete_name = athlete_name
        self.history: List[ChatMessage] = []
        self.context = ""
        self._client = None
        self._use_llm = False

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=api_key)
                self._use_llm = True
            except ImportError:
                logger.warning("anthropic package not installed, using keyword fallback")

    def set_context(self, analysis_results: Dict):
        """Serialize full analysis_results to compact JSON, stripping heavy coordinate arrays."""
        self.athlete_data = analysis_results

        def _is_heavy(v) -> bool:
            return isinstance(v, (bytes, bytearray)) or (isinstance(v, (list, tuple)) and len(v) > 10)

        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items() if not _is_heavy(v)}
            if hasattr(obj, "__dataclass_fields__"):
                return {f: _clean(getattr(obj, f)) for f in obj.__dataclass_fields__ if not _is_heavy(getattr(obj, f))}
            if isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            return None

        sections: Dict = {}
        for key, val in analysis_results.items():
            if key == "recent_sessions" and isinstance(val, list):
                sections[key] = [
                    {
                        k: v
                        for k, v in (s.items() if isinstance(s, dict) else vars(s).items())
                        if k in ("date", "type", "ai_score")
                    }
                    for s in val[-5:]
                ]
            else:
                cleaned = _clean(val)
                if cleaned is not None:
                    sections[key] = cleaned

        self.context = json.dumps(sections, ensure_ascii=False, separators=(",", ":"))

    def chat(self, user_message: str) -> str:
        """Process user message and return response."""
        self.history.append(ChatMessage(role="user", content=user_message, timestamp=datetime.now().isoformat()))
        if self._use_llm:
            response = self._llm_response(user_message)
        else:
            response = self._keyword_response(user_message.lower())
        self.history.append(ChatMessage(role="assistant", content=response, timestamp=datetime.now().isoformat()))
        return response

    def _build_system(self) -> str:
        """Build system prompt with optional athlete context embedded as JSON."""
        system = _SYSTEM_PROMPT
        ctx_parts = []
        if self.athlete_name and self.athlete_name != "Спортсмен":
            ctx_parts.append(f"Спортсмен: {self.athlete_name}")
        if self.context:
            ctx_parts.append(f"Дані аналізу (JSON): {self.context}")
        if ctx_parts:
            system += "\n\n<athlete_context>\n" + "\n".join(ctx_parts) + "\n</athlete_context>"
        return system

    def _llm_response(self, user_message: str) -> str:
        """Call Claude API and return response."""
        try:
            # Build conversation history (last 10 messages)
            messages = []
            for msg in self.history[-10:]:
                messages.append({"role": msg.role, "content": msg.content})

            # No sampling params: claude-sonnet-5 rejects non-default
            # temperature/top_p with a 400.
            result = self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=512,
                system=self._build_system(),
                messages=messages,
            )
            return result.content[0].text
        except Exception:
            logger.warning("LLM call failed, falling back to keyword mode", exc_info=True)
            return self._keyword_response(user_message.lower())

    def _keyword_response(self, query: str) -> str:
        """Keyword-based fallback response."""
        for topic, info in SWIMMING_KNOWLEDGE["freestyle"].items():
            if topic in query:
                return f"🏊 **{topic.upper()}**\n\n{info}"
        for drill, info in SWIMMING_KNOWLEDGE["drills"].items():
            if drill.replace("_", " ") in query or drill in query:
                return f"🏋️ **Вправа: {drill.replace('_', ' ').title()}**\n\n{info}"
        for error, info in SWIMMING_KNOWLEDGE["common_errors"].items():
            if error.replace("_", " ") in query or error in query:
                return f"⚠️ **Помилка: {error.replace('_', ' ').title()}**\n\n{info}"
        for ex, info in DRYLAND_KNOWLEDGE["exercises"].items():
            if ex.replace("_", " ") in query or ex in query:
                return f"🏋️ **{ex.replace('_', ' ').title()}**\n\n{info}"
        if any(w in query for w in ["привіт", "привет", "hello", "hi"]):
            return "👋 Привіт! Я AI тренер. Можу відповісти на питання про техніку плавання та суходільні вправи."
        if any(w in query for w in ["що покращити", "рекомендац", "порад"]):
            if self.context:
                return f"📊 **На основі вашого аналізу** ({self.context}):\n\n" + self._get_recommendations()
            return "📊 Спочатку проведіть аналіз відео, щоб я міг дати персоналізовані рекомендації."
        if "план" in query or "тренуван" in query:
            return "📅 Для плану тренувань перейдіть в розділ **Автоплан**."
        if any(w in query for w in ["фаз", "гребок", "stroke"]):
            return "🏊 **Фази гребка**: 1. Catch 2. Pull 3. Push 4. Recovery. Запитайте про конкретну фазу!"
        if any(w in query for w in ["помилк", "error", "проблем"]):
            errors = list(SWIMMING_KNOWLEDGE["common_errors"].keys())
            return "⚠️ **Типові помилки**:\n" + "\n".join(f"• {e.replace('_', ' ').title()}" for e in errors)
        return "🤔 Спробуйте запитати про: catch, pull, body roll, dropped elbow, або напишіть вашу проблему!"

    def _get_recommendations(self) -> str:
        recs = []
        stroke = self.athlete_data.get("stroke_analysis")
        if stroke:
            if getattr(stroke, "symmetry_score", 100) < 80:
                recs.append("• **Симетрія**: single arm drill")
            roll = getattr(stroke, "avg_body_roll", 40)
            if roll < 30:
                recs.append("• **Body roll**: збільшіть до 30-50°")
            elif roll > 50:
                recs.append("• **Body roll**: зменшіть до 30-50°")
        pose = self.athlete_data.get("swimming_pose", {})
        if pose.get("avg_streamline", 70) < 70:
            recs.append("• **Streamline**: витягуйтесь максимально під час recovery")
        return "\n".join(recs) if recs else "• Загалом техніка гарна! Продовжуйте в тому ж дусі."


# ============================================================================
# TRAINING PLAN GENERATOR
# ============================================================================


def generate_training_plan(
    athlete_name: str,
    level: str = "intermediate",  # beginner, intermediate, advanced
    goal: str = "general",  # general, speed, endurance, technique
    sessions_per_week: int = 4,
    weeks: int = 4,
) -> TrainingPlan:
    """Generate personalized training plan."""

    plan = TrainingPlan(
        athlete_name=athlete_name,
        goal=goal,
        level=level,
        weeks=weeks,
    )

    # Templates based on level and goal
    templates = {
        "beginner": {
            "general": [
                {
                    "day": "Пн",
                    "type": "Плавання",
                    "duration": 45,
                    "focus": "Техніка",
                    "workout": "200м розминка | 4x50м catch-up drill | 4x50м fingertip drag | 200м спокійно",
                },
                {
                    "day": "Ср",
                    "type": "Суходіл",
                    "duration": 30,
                    "focus": "Core + Плечі",
                    "workout": "Планка 3x30с | Lat pulldown 3x12 | Flutter kicks 3x20 | Розтяжка",
                },
                {
                    "day": "Пт",
                    "type": "Плавання",
                    "duration": 45,
                    "focus": "Витривалість",
                    "workout": "200м розминка | 8x50м (відпочинок 20с) | 200м заминка",
                },
            ],
        },
        "intermediate": {
            "general": [
                {
                    "day": "Пн",
                    "type": "Плавання",
                    "duration": 60,
                    "focus": "Техніка + Швидкість",
                    "workout": "400м розминка | 4x100м техніка | 8x25м спринт | 200м заминка",
                },
                {
                    "day": "Вт",
                    "type": "Суходіл",
                    "duration": 45,
                    "focus": "Сила",
                    "workout": "Lat pulldown 4x10 | Tricep ext 3x12 | Band pull 3x15 | Core circuit",
                },
                {
                    "day": "Чт",
                    "type": "Плавання",
                    "duration": 60,
                    "focus": "Витривалість",
                    "workout": "300м розминка | 10x100м (відпочинок 15с) | 200м заминка",
                },
                {
                    "day": "Сб",
                    "type": "Плавання",
                    "duration": 45,
                    "focus": "Відновлення",
                    "workout": "500м спокійно різними стилями | Drills на вибір | Розтяжка у воді",
                },
            ],
            "speed": [
                {
                    "day": "Пн",
                    "type": "Плавання",
                    "duration": 60,
                    "focus": "Швидкісна витривалість",
                    "workout": "400м розминка | 12x50м (80% max, відпочинок 30с) | 200м заминка",
                },
                {
                    "day": "Вт",
                    "type": "Суходіл",
                    "duration": 45,
                    "focus": "Вибухова сила",
                    "workout": "Medicine ball throws 4x8 | Box jumps 4x6 | Band sprints 3x10",
                },
                {
                    "day": "Чт",
                    "type": "Плавання",
                    "duration": 60,
                    "focus": "Спринт",
                    "workout": "300м розминка | 16x25м max (відпочинок 45с) | 200м заминка",
                },
                {
                    "day": "Сб",
                    "type": "Плавання",
                    "duration": 50,
                    "focus": "Темпова робота",
                    "workout": "400м розминка | 4x200м race pace | 200м заминка",
                },
            ],
        },
        "advanced": {
            "general": [
                {
                    "day": "Пн",
                    "type": "Плавання",
                    "duration": 90,
                    "focus": "Техніка + Швидкість",
                    "workout": "600м розминка | 6x100м drill | 10x50м sprint | 400м заминка",
                },
                {
                    "day": "Вт",
                    "type": "Суходіл",
                    "duration": 60,
                    "focus": "Сила максимальна",
                    "workout": "Bench pull 5x5 | Lat pulldown 4x8 | Core 3 раунди",
                },
                {
                    "day": "Ср",
                    "type": "Плавання",
                    "duration": 75,
                    "focus": "Поріг",
                    "workout": "400м розминка | 5x400м threshold | 400м заминка",
                },
                {
                    "day": "Чт",
                    "type": "Суходіл",
                    "duration": 45,
                    "focus": "Відновлення",
                    "workout": "Легка йога | Foam rolling | Розтяжка 30хв",
                },
                {
                    "day": "Пт",
                    "type": "Плавання",
                    "duration": 90,
                    "focus": "Спринт",
                    "workout": "500м розминка | 20x25м all-out | 4x100м easy | 300м заминка",
                },
                {
                    "day": "Сб",
                    "type": "Плавання",
                    "duration": 60,
                    "focus": "Дистанційна",
                    "workout": "2000м безперервно помірний темп",
                },
            ],
        },
    }

    # Get template
    level_templates = templates.get(level, templates["intermediate"])
    goal_sessions = level_templates.get(goal, level_templates.get("general", []))

    # Generate weeks
    for week_num in range(1, weeks + 1):
        for session in goal_sessions[:sessions_per_week]:
            plan.sessions.append(
                {
                    "week": week_num,
                    **session,
                    "intensity": ("Помірна" if week_num % 4 == 0 else "Стандартна"),  # Deload every 4th week
                }
            )

    plan.notes = (
        f"План на {weeks} тижнів для {level} рівня. Мета: {goal}. " f"Кожен 4-й тиждень - полегшений для відновлення."
    )

    return plan


# ============================================================================
# TEXT-TO-SPEECH
# ============================================================================


def text_to_speech(text: str, output_path: str = None) -> Optional[str]:
    """Convert text to speech using pyttsx3 or gTTS."""
    try:
        # Try pyttsx3 first (offline)
        import pyttsx3

        engine = pyttsx3.init()

        if output_path:
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            return output_path
        else:
            engine.say(text)
            engine.runAndWait()
            return None

    except ImportError:
        try:
            # Fallback to gTTS (online)
            import tempfile

            from gtts import gTTS

            tts = gTTS(text=text, lang="uk")

            if output_path:
                tts.save(output_path)
                return output_path
            else:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    tts.save(f.name)
                    return f.name

        except ImportError:
            logger.warning("No TTS library available. Install pyttsx3 or gTTS.")
            return None
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None


# Convenience function
def get_ai_chat(athlete_data: Dict = None) -> AIChat:
    """Get AI chat instance."""
    return AIChat(athlete_data)
