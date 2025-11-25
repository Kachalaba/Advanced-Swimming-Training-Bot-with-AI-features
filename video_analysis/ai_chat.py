"""
🤖 AI Chat & Training Plan Generator

Features:
- Interactive chat about swimming/dryland technique
- Text-to-Speech (TTS) for recommendations
- Automatic training plan generation
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


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
    }
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
    }
}


# ============================================================================
# AI CHAT
# ============================================================================

class AIChat:
    """AI Chat for swimming and dryland questions."""
    
    def __init__(self, athlete_data: Dict = None):
        self.athlete_data = athlete_data or {}
        self.history: List[ChatMessage] = []
        self.context = ""
    
    def set_context(self, analysis_results: Dict):
        """Set context from analysis results."""
        self.athlete_data = analysis_results
        
        # Build context string
        parts = []
        
        if "stroke_analysis" in analysis_results:
            stroke = analysis_results["stroke_analysis"]
            parts.append(f"Stroke rate: {getattr(stroke, 'stroke_rate', 'N/A')}/min")
            parts.append(f"Symmetry: {getattr(stroke, 'symmetry_score', 'N/A')}%")
            parts.append(f"Body roll: {getattr(stroke, 'avg_body_roll', 'N/A')}°")
        
        if "swimming_pose" in analysis_results:
            pose = analysis_results["swimming_pose"]
            parts.append(f"Streamline: {pose.get('avg_streamline', 'N/A')}/100")
        
        self.context = " | ".join(parts)
    
    def chat(self, user_message: str) -> str:
        """Process user message and return response."""
        self.history.append(ChatMessage(
            role="user",
            content=user_message,
            timestamp=datetime.now().isoformat()
        ))
        
        response = self._generate_response(user_message.lower())
        
        self.history.append(ChatMessage(
            role="assistant",
            content=response,
            timestamp=datetime.now().isoformat()
        ))
        
        return response
    
    def _generate_response(self, query: str) -> str:
        """Generate response based on query."""
        
        # Check swimming technique questions
        for topic, info in SWIMMING_KNOWLEDGE["freestyle"].items():
            if topic in query:
                return f"🏊 **{topic.upper()}**\n\n{info}"
        
        # Check drills
        for drill, info in SWIMMING_KNOWLEDGE["drills"].items():
            if drill.replace("_", " ") in query or drill in query:
                return f"🏋️ **Вправа: {drill.replace('_', ' ').title()}**\n\n{info}"
        
        # Check errors
        for error, info in SWIMMING_KNOWLEDGE["common_errors"].items():
            if error.replace("_", " ") in query or error in query:
                return f"⚠️ **Помилка: {error.replace('_', ' ').title()}**\n\n{info}"
        
        # Check dryland
        for ex, info in DRYLAND_KNOWLEDGE["exercises"].items():
            if ex.replace("_", " ") in query or ex in query:
                return f"🏋️ **{ex.replace('_', ' ').title()}**\n\n{info}"
        
        # General questions
        if any(w in query for w in ["привіт", "привет", "hello", "hi"]):
            return "👋 Привіт! Я AI тренер. Можу відповісти на питання про техніку плавання та суходільні вправи. Спитайте про catch, pull, push, recovery, body roll, або конкретні помилки!"
        
        if any(w in query for w in ["що покращити", "рекомендац", "порад"]):
            if self.context:
                return f"📊 **На основі вашого аналізу** ({self.context}):\n\n" + self._get_recommendations()
            return "📊 Спочатку проведіть аналіз відео, щоб я міг дати персоналізовані рекомендації."
        
        if "план" in query or "тренуван" in query:
            return "📅 Для створення плану тренувань перейдіть в розділ **Автоплан** або запитайте конкретно: 'створи план на тиждень для початківця'"
        
        if any(w in query for w in ["фаз", "гребок", "stroke"]):
            return """🏊 **Фази гребка (Freestyle)**:

1. **Catch** - вхід руки у воду, високий лікоть
2. **Pull** - тягнення води під тілом  
3. **Push** - завершення біля стегна
4. **Recovery** - рух над водою

Запитайте про конкретну фазу для детальнішої інформації!"""
        
        if any(w in query for w in ["помилк", "error", "проблем"]):
            errors = list(SWIMMING_KNOWLEDGE["common_errors"].keys())
            return f"""⚠️ **Типові помилки**:

{chr(10).join(f'• {e.replace("_", " ").title()}' for e in errors)}

Запитайте про конкретну помилку для порад як виправити!"""
        
        # Default response
        return """🤔 Не зовсім зрозумів питання. Спробуйте запитати про:

• **Техніку**: catch, pull, push, recovery, body roll
• **Помилки**: dropped elbow, crossover, flat body
• **Вправи**: catch-up drill, fingertip drag, fist drill
• **Рекомендації**: "що покращити", "дай поради"
• **План тренувань**: "створи план"

Або просто опишіть вашу проблему!"""
    
    def _get_recommendations(self) -> str:
        """Get recommendations based on athlete data."""
        recs = []
        
        stroke = self.athlete_data.get("stroke_analysis")
        if stroke:
            symmetry = getattr(stroke, 'symmetry_score', 100)
            body_roll = getattr(stroke, 'avg_body_roll', 40)
            
            if symmetry < 80:
                recs.append("• **Симетрія рук**: Працюйте над рівномірним гребком обома руками. Спробуйте single arm drill.")
            
            if body_roll < 30:
                recs.append("• **Body roll**: Збільшіть обертання тіла до 30-50°. Це покращить потужність гребка.")
            elif body_roll > 50:
                recs.append("• **Body roll**: Зменшіть обертання тіла до 30-50°. Занадто великий roll неефективний.")
        
        pose = self.athlete_data.get("swimming_pose", {})
        streamline = pose.get("avg_streamline", 70)
        
        if streamline < 70:
            recs.append("• **Streamline**: Покращіть обтічність тіла. Витягуйтесь максимально під час recovery.")
        
        if not recs:
            recs.append("• Загалом техніка гарна! Продовжуйте працювати над стабільністю.")
        
        return "\n".join(recs)


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
                {"day": "Пн", "type": "Плавання", "duration": 45, "focus": "Техніка", 
                 "workout": "200м розминка | 4x50м catch-up drill | 4x50м fingertip drag | 200м спокійно"},
                {"day": "Ср", "type": "Суходіл", "duration": 30, "focus": "Core + Плечі",
                 "workout": "Планка 3x30с | Lat pulldown 3x12 | Flutter kicks 3x20 | Розтяжка"},
                {"day": "Пт", "type": "Плавання", "duration": 45, "focus": "Витривалість",
                 "workout": "200м розминка | 8x50м (відпочинок 20с) | 200м заминка"},
            ],
        },
        "intermediate": {
            "general": [
                {"day": "Пн", "type": "Плавання", "duration": 60, "focus": "Техніка + Швидкість",
                 "workout": "400м розминка | 4x100м техніка | 8x25м спринт | 200м заминка"},
                {"day": "Вт", "type": "Суходіл", "duration": 45, "focus": "Сила",
                 "workout": "Lat pulldown 4x10 | Tricep ext 3x12 | Band pull 3x15 | Core circuit"},
                {"day": "Чт", "type": "Плавання", "duration": 60, "focus": "Витривалість",
                 "workout": "300м розминка | 10x100м (відпочинок 15с) | 200м заминка"},
                {"day": "Сб", "type": "Плавання", "duration": 45, "focus": "Відновлення",
                 "workout": "500м спокійно різними стилями | Drills на вибір | Розтяжка у воді"},
            ],
            "speed": [
                {"day": "Пн", "type": "Плавання", "duration": 60, "focus": "Швидкісна витривалість",
                 "workout": "400м розминка | 12x50м (80% max, відпочинок 30с) | 200м заминка"},
                {"day": "Вт", "type": "Суходіл", "duration": 45, "focus": "Вибухова сила",
                 "workout": "Medicine ball throws 4x8 | Box jumps 4x6 | Band sprints 3x10"},
                {"day": "Чт", "type": "Плавання", "duration": 60, "focus": "Спринт",
                 "workout": "300м розминка | 16x25м max (відпочинок 45с) | 200м заминка"},
                {"day": "Сб", "type": "Плавання", "duration": 50, "focus": "Темпова робота",
                 "workout": "400м розминка | 4x200м race pace | 200м заминка"},
            ],
        },
        "advanced": {
            "general": [
                {"day": "Пн", "type": "Плавання", "duration": 90, "focus": "Техніка + Швидкість",
                 "workout": "600м розминка | 6x100м drill | 10x50м sprint | 400м заминка"},
                {"day": "Вт", "type": "Суходіл", "duration": 60, "focus": "Сила максимальна",
                 "workout": "Bench pull 5x5 | Lat pulldown 4x8 | Core 3 раунди"},
                {"day": "Ср", "type": "Плавання", "duration": 75, "focus": "Поріг",
                 "workout": "400м розминка | 5x400м threshold | 400м заминка"},
                {"day": "Чт", "type": "Суходіл", "duration": 45, "focus": "Відновлення",
                 "workout": "Легка йога | Foam rolling | Розтяжка 30хв"},
                {"day": "Пт", "type": "Плавання", "duration": 90, "focus": "Спринт",
                 "workout": "500м розминка | 20x25м all-out | 4x100м easy | 300м заминка"},
                {"day": "Сб", "type": "Плавання", "duration": 60, "focus": "Дистанційна",
                 "workout": "2000м безперервно помірний темп"},
            ],
        },
    }
    
    # Get template
    level_templates = templates.get(level, templates["intermediate"])
    goal_sessions = level_templates.get(goal, level_templates.get("general", []))
    
    # Generate weeks
    for week_num in range(1, weeks + 1):
        for session in goal_sessions[:sessions_per_week]:
            plan.sessions.append({
                "week": week_num,
                **session,
                "intensity": "Помірна" if week_num % 4 == 0 else "Стандартна"  # Deload every 4th week
            })
    
    plan.notes = f"План на {weeks} тижнів для {level} рівня. Мета: {goal}. " \
                 f"Кожен 4-й тиждень - полегшений для відновлення."
    
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
            from gtts import gTTS
            import tempfile
            
            tts = gTTS(text=text, lang='uk')
            
            if output_path:
                tts.save(output_path)
                return output_path
            else:
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
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
