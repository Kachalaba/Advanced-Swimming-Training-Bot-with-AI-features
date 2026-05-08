"""
Sprint AI — UI translations (Ukrainian / English).

Usage:
    from i18n.translations import t
    st.markdown(t("tab_swimming"))
"""

import streamlit as st

TRANSLATIONS: dict = {
    "uk": {
        # App header
        "tagline": "Професійний аналіз спортсменів",
        # Main tab labels
        "tab_swimming": "🏊 ПЛАВАННЯ",
        "tab_running": "🏃 БІГ",
        "tab_cycling": "🚴 ВЕЛОСИПЕД",
        "tab_dryland": "🏋️ СУХОДІЛ",
        "tab_history": "📊 ІСТОРІЯ",
        "tab_ai": "🤖 AI АСИСТЕНТ",
        "tab_tools": "🎬 ІНСТРУМЕНТИ",
        # Swimming page
        "swim_title": "Аналіз техніки плавання",
        "swim_settings": "⚙️ Налаштування аналізу",
        "swim_athlete": "👤 Ім'я спортсмена",
        "swim_pool": "🏊 Басейн",
        "swim_fps": "🎬 FPS",
        "swim_method": "🔬 Метод",
        "swim_upload_title": "Завантаження відео",
        "swim_analyze_btn": "🏊 АНАЛІЗУВАТИ ПЛАВАННЯ",
        "swim_tracking": "🎯 Трекінг плавця",
        "swim_tracking_desc": "Налаштування для стабільного відстеження потрібного плавця",
        # Running page
        "run_title": "Аналіз техніки бігу",
        "run_settings": "⚙️ Налаштування аналізу",
        "run_athlete": "👤 Ім'я спортсмена",
        "run_fps": "🎬 FPS відео",
        "run_type": "🏃 Тип бігу",
        "run_upload_title": "Завантаження відео",
        "run_analyze_btn": "🏃 АНАЛІЗУВАТИ БІГ",
        "run_tracking": "🎯 Трекінг персони",
        # Cycling page
        "cycle_title": "Аналіз техніки велосипеду",
        "cycle_settings": "⚙️ Налаштування аналізу",
        "cycle_athlete": "👤 Ім'я спортсмена",
        "cycle_fps": "🎬 FPS",
        "cycle_bike": "🚴 Тип велосипеду",
        "cycle_upload_title": "Завантаження відео",
        "cycle_analyze_btn": "🚴 АНАЛІЗУВАТИ ВЕЛОСИПЕД",
        "cycle_tracking": "🎯 Трекінг персони",
        # Dryland page
        "dry_title": "Аналіз суходільних вправ",
        "dry_settings": "⚙️ Налаштування аналізу",
        "dry_athlete": "👤 Ім'я спортсмена",
        "dry_fps": "🎬 FPS",
        "dry_exercise": "🏋️ Тип вправи",
        "dry_upload_title": "Завантаження відео",
        "dry_analyze_btn": "🏋️ АНАЛІЗУВАТИ ВПРАВУ",
        # History page
        "hist_title": "📊 Історія тренувань",
        "hist_select_athlete": "👤 Оберіть спортсмена",
        "hist_delete_athlete": "🗑️ Видалити спортсмена",
        "hist_stats": "### 📈 Загальна статистика",
        "hist_by_type": "### 📊 По типам тренувань",
        "hist_chart_title": "### 📈 Динаміка метрик",
        "hist_sessions": "### 📋 Історія сесій",
        "hist_csv_btn": "⬇️ Завантажити CSV",
        "hist_compare": "### ⚖️ Порівняти сесії",
        # AI assistant page
        "ai_title": "🤖 AI Асистент",
        "ai_chat_tab": "💬 Чат",
        "ai_plan_tab": "📅 Автоплан",
        "ai_athlete_label": "👤 Ім'я спортсмена (для контексту)",
        "ai_send_btn": "📤 Надіслати",
        "ai_clear_btn": "🗑️ Очистити",
        "ai_speak_btn": "🔊 Озвучити",
        "ai_quick_title": "### ⚡ Швидкі питання",
        # Common
        "common_upload_hint": "Перетягніть файл або оберіть",
        "common_upload_types": "MP4, MOV, AVI до 200 МБ",
        "common_detail_analysis": "⚡ Детальний аналіз (5-10 хв)",
        "common_fast_analysis": "⏱️ Швидкий аналіз (1-3 хв)",
        # Tracking
        "track_who": "🎯 Кого аналізувати?",
        "track_options": ["Найбільший у кадрі (авто)", "Найближчий до центру", "Вибрати за номером"],
        "track_person_num": "Номер (зліва направо)",
        "track_max_lost": "Макс. кадрів без детекції",
        "track_bbox_pad": "Відступ bbox (%)",
        # Analysis progress status messages
        "status_extract_frames": "🎞️ Витягуємо кадри з відео...",
        "status_extract_frames_short": "🎬 Витягуємо кадри...",
        "status_detect_swimmer": "👁️ Детекція плавця (YOLO + 🌊 підводна)...",
        "status_detect_runner": "🎯 Детекція бігуна...",
        "status_detect_cyclist": "🎯 Детекція велосипедиста...",
        "status_detect_person": "🎯 Детекція людини...",
        "status_biomech": "🔬 Аналіз біомеханіки (pose)...",
        "status_biomech_stable": "🦴 Аналіз біомеханіки (стабільний трекінг)...",
        "status_biomech_short": "🦴 Аналіз біомеханіки...",
        "status_biomech_viz": "🦴 Візуалізація біомеханіки...",
        "status_trajectory": "📍 Аналіз траєкторії (bbox)...",
        "status_splits": "⏱️ Аналіз сплітів і швидкості...",
        "status_swim_pose": "🏊 Аналіз пози плавця (rotation + spine)...",
        "status_stroke": "🏊 Аналіз гребка (фази, симетрія, body roll)...",
        "status_running": "🏃 Аналіз техніки бігу...",
        "status_cycling": "🚴 Аналіз техніки педалювання...",
        "status_reps": "🔄 Підрахунок повторень...",
        "status_annotated_video": "🎬 Створення анотованого відео...",
        "status_skeleton_video": "🎬 Створення відео зі скелетом...",
        "status_effects_video": "🎬 Створення відео з ефектами...",
        "status_charts": "📊 Генерація графіків...",
        "status_reports": "📊 Генерація звітів...",
        "status_ai_coach": "🤖 AI тренер аналізує результати...",
        "status_ai_coach_short": "🤖 AI тренер аналізує...",
        "status_done": "✅ Аналіз завершено!",
        # Result section labels
        "result_stroke_analysis": "Аналіз гребка",
        "result_swim_pose": "Поза плавця",
        "result_metrics": "Метрики",
        "result_improvements": "Покращення",
        "result_regressions": "Регресії",
        "result_exercise_stats": "Статистика вправи",
        "result_ai_chat_athlete": "Контекст спортсмена",
    },
    "en": {
        # App header
        "tagline": "Professional athlete analysis",
        # Main tab labels
        "tab_swimming": "🏊 SWIMMING",
        "tab_running": "🏃 RUNNING",
        "tab_cycling": "🚴 CYCLING",
        "tab_dryland": "🏋️ DRYLAND",
        "tab_history": "📊 HISTORY",
        "tab_ai": "🤖 AI ASSISTANT",
        "tab_tools": "🎬 TOOLS",
        # Swimming page
        "swim_title": "Swimming technique analysis",
        "swim_settings": "⚙️ Analysis settings",
        "swim_athlete": "👤 Athlete name",
        "swim_pool": "🏊 Pool",
        "swim_fps": "🎬 FPS",
        "swim_method": "🔬 Method",
        "swim_upload_title": "Upload video",
        "swim_analyze_btn": "🏊 ANALYZE SWIMMING",
        "swim_tracking": "🎯 Swimmer tracking",
        "swim_tracking_desc": "Settings for stable tracking of the target swimmer",
        # Running page
        "run_title": "Running technique analysis",
        "run_settings": "⚙️ Analysis settings",
        "run_athlete": "👤 Athlete name",
        "run_fps": "🎬 Video FPS",
        "run_type": "🏃 Run type",
        "run_upload_title": "Upload video",
        "run_analyze_btn": "🏃 ANALYZE RUNNING",
        "run_tracking": "🎯 Person tracking",
        # Cycling page
        "cycle_title": "Cycling technique analysis",
        "cycle_settings": "⚙️ Analysis settings",
        "cycle_athlete": "👤 Athlete name",
        "cycle_fps": "🎬 FPS",
        "cycle_bike": "🚴 Bike type",
        "cycle_upload_title": "Upload video",
        "cycle_analyze_btn": "🚴 ANALYZE CYCLING",
        "cycle_tracking": "🎯 Person tracking",
        # Dryland page
        "dry_title": "Dryland exercise analysis",
        "dry_settings": "⚙️ Analysis settings",
        "dry_athlete": "👤 Athlete name",
        "dry_fps": "🎬 FPS",
        "dry_exercise": "🏋️ Exercise type",
        "dry_upload_title": "Upload video",
        "dry_analyze_btn": "🏋️ ANALYZE EXERCISE",
        # History page
        "hist_title": "📊 Training history",
        "hist_select_athlete": "👤 Select athlete",
        "hist_delete_athlete": "🗑️ Delete athlete",
        "hist_stats": "### 📈 Overall statistics",
        "hist_by_type": "### 📊 By training type",
        "hist_chart_title": "### 📈 Metric trends",
        "hist_sessions": "### 📋 Session history",
        "hist_csv_btn": "⬇️ Download CSV",
        "hist_compare": "### ⚖️ Compare sessions",
        # AI assistant page
        "ai_title": "🤖 AI Assistant",
        "ai_chat_tab": "💬 Chat",
        "ai_plan_tab": "📅 Auto-plan",
        "ai_athlete_label": "👤 Athlete name (for context)",
        "ai_send_btn": "📤 Send",
        "ai_clear_btn": "🗑️ Clear",
        "ai_speak_btn": "🔊 Speak",
        "ai_quick_title": "### ⚡ Quick questions",
        # Common
        "common_upload_hint": "Drag & drop or browse",
        "common_upload_types": "MP4, MOV, AVI up to 200 MB",
        "common_detail_analysis": "⚡ Detailed analysis (5-10 min)",
        "common_fast_analysis": "⏱️ Quick analysis (1-3 min)",
        # Tracking
        "track_who": "🎯 Who to analyze?",
        "track_options": ["Largest in frame (auto)", "Closest to center", "Select by number"],
        "track_person_num": "Number (left to right)",
        "track_max_lost": "Max frames without detection",
        "track_bbox_pad": "BBox padding (%)",
        # Analysis progress status messages
        "status_extract_frames": "🎞️ Extracting frames from video...",
        "status_extract_frames_short": "🎬 Extracting frames...",
        "status_detect_swimmer": "👁️ Detecting swimmer (YOLO + 🌊 underwater)...",
        "status_detect_runner": "🎯 Detecting runner...",
        "status_detect_cyclist": "🎯 Detecting cyclist...",
        "status_detect_person": "🎯 Detecting person...",
        "status_biomech": "🔬 Biomechanics analysis (pose)...",
        "status_biomech_stable": "🦴 Biomechanics analysis (stable tracking)...",
        "status_biomech_short": "🦴 Biomechanics analysis...",
        "status_biomech_viz": "🦴 Biomechanics visualization...",
        "status_trajectory": "📍 Trajectory analysis (bbox)...",
        "status_splits": "⏱️ Splits & speed analysis...",
        "status_swim_pose": "🏊 Swimming pose analysis (rotation + spine)...",
        "status_stroke": "🏊 Stroke analysis (phases, symmetry, body roll)...",
        "status_running": "🏃 Running technique analysis...",
        "status_cycling": "🚴 Pedaling technique analysis...",
        "status_reps": "🔄 Counting repetitions...",
        "status_annotated_video": "🎬 Generating annotated video...",
        "status_skeleton_video": "🎬 Generating skeleton video...",
        "status_effects_video": "🎬 Generating effects video...",
        "status_charts": "📊 Generating charts...",
        "status_reports": "📊 Generating reports...",
        "status_ai_coach": "🤖 AI coach analyzing results...",
        "status_ai_coach_short": "🤖 AI coach analyzing...",
        "status_done": "✅ Analysis complete!",
        # Result section labels
        "result_stroke_analysis": "Stroke analysis",
        "result_swim_pose": "Swimming pose",
        "result_metrics": "Metrics",
        "result_improvements": "Improvements",
        "result_regressions": "Regressions",
        "result_exercise_stats": "Exercise statistics",
        "result_ai_chat_athlete": "Athlete context",
    },
}


def t(key: str, **fmt) -> str:
    """Return translated string for *key* in the active language.

    Falls back to Ukrainian, then to *key* itself if no translation exists.
    Optional keyword arguments are interpolated via str.format_map.

    Example::
        t("status_extract_frames")           # plain key
        t("result_score", score=87)          # with interpolation
    """
    lang = st.session_state.get("lang", "uk")
    val = TRANSLATIONS.get(lang, TRANSLATIONS["uk"]).get(key)
    if val is None:
        val = TRANSLATIONS["uk"].get(key, key)
    if fmt and isinstance(val, str):
        try:
            val = val.format_map(fmt)
        except (KeyError, ValueError):
            pass
    return val


def get_lang() -> str:
    """Return the current UI language code ('uk' or 'en')."""
    return st.session_state.get("lang", "uk")
