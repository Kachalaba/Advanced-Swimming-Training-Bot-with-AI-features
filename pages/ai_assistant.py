"""
AI assistant page module.
"""

import logging

import streamlit as st

from video_analysis.ai_chat import AIChat, generate_training_plan, text_to_speech

logger = logging.getLogger(__name__)


def render_ai_tab():
    """Render AI assistant tab with chat and training plan."""

    st.markdown('<div class="section-title">🤖 AI Асистент</div>', unsafe_allow_html=True)

    # Sub-tabs
    ai_tab1, ai_tab2 = st.tabs(["💬 Чат", "📅 Автоплан"])

    # ========================================================================
    # CHAT TAB
    # ========================================================================
    with ai_tab1:
        st.markdown("### 💬 Запитайте про техніку")

        # Athlete name picker for context
        import os

        athlete_name_chat = st.text_input(
            "👤 Ім'я спортсмена (для контексту)",
            value="Спортсмен",
            key="ai_athlete_name",
        )

        # LLM status indicator
        if os.environ.get("ANTHROPIC_API_KEY"):
            st.markdown(
                '<div class="status-success">🤖 Claude AI активний — відповіді на базі LLM</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-info">💡 Режим без API ключа — відповіді на базі бази знань. Додайте ANTHROPIC_API_KEY для Claude AI.</div>',
                unsafe_allow_html=True,
            )

        # Initialize chat in session state (reset if athlete changed)
        if "ai_chat" not in st.session_state or st.session_state.get("ai_chat_athlete") != athlete_name_chat:
            st.session_state.ai_chat = AIChat(athlete_name=athlete_name_chat)
            st.session_state.ai_chat_athlete = athlete_name_chat
            st.session_state.chat_history = []
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                <div style="background: rgba(59,130,246,0.2); border-radius: 12px; padding: 0.75rem; margin: 0.5rem 0; text-align: right;">
                    <strong>Ви:</strong> {msg["content"]}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div style="background: rgba(16,185,129,0.2); border-radius: 12px; padding: 0.75rem; margin: 0.5rem 0;">
                    <strong>🤖 AI:</strong> {msg["content"]}
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Chat input
        user_input = st.text_input(
            "Ваше питання:",
            key="ai_chat_input",
            placeholder="Наприклад: Як покращити body roll?",
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("📤 Надіслати", type="primary", use_container_width=True):
                if user_input:
                    response = st.session_state.ai_chat.chat(user_input)
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()

        with col2:
            if st.button("🗑️ Очистити", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.ai_chat = AIChat()
                st.rerun()

        with col3:
            # TTS button
            if st.session_state.chat_history:
                last_response = (
                    st.session_state.chat_history[-1]["content"]
                    if st.session_state.chat_history[-1]["role"] == "assistant"
                    else ""
                )
                if last_response and st.button("🔊 Озвучити", use_container_width=True):
                    try:
                        # Simplified text for TTS
                        clean_text = last_response.replace("**", "").replace("•", "").replace("#", "")
                        audio_file = text_to_speech(clean_text, "temp_speech.mp3")
                        if audio_file:
                            st.audio(audio_file)
                    except (OSError, RuntimeError, ImportError) as e:
                        logger.warning("TTS error", exc_info=True)
                        st.warning(f"TTS недоступний: {e}")

        # Quick questions
        st.markdown("### ⚡ Швидкі питання")
        quick_cols = st.columns(4)
        quick_questions = [
            "Що таке catch?",
            "Типові помилки",
            "Як покращити body roll?",
            "Вправи для плечей",
        ]
        for i, (col, question) in enumerate(zip(quick_cols, quick_questions)):
            with col:
                if st.button(question, key=f"quick_{i}", use_container_width=True):
                    response = st.session_state.ai_chat.chat(question)
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()

    # ========================================================================
    # TRAINING PLAN TAB
    # ========================================================================
    with ai_tab2:
        st.markdown("### 📅 Генератор плану тренувань")

        col1, col2 = st.columns(2)

        with col1:
            plan_name = st.text_input("👤 Ім'я спортсмена", value="Спортсмен", key="plan_name")
            plan_level = st.selectbox(
                "📊 Рівень",
                ["beginner", "intermediate", "advanced"],
                format_func=lambda x: {
                    "beginner": "🌱 Початківець",
                    "intermediate": "📈 Середній",
                    "advanced": "🏆 Просунутий",
                }[x],
                key="plan_level",
            )

        with col2:
            plan_goal = st.selectbox(
                "🎯 Мета",
                ["general", "speed", "endurance", "technique"],
                format_func=lambda x: {
                    "general": "🎯 Загальна підготовка",
                    "speed": "⚡ Швидкість",
                    "endurance": "🏃 Витривалість",
                    "technique": "🎓 Техніка",
                }[x],
                key="plan_goal",
            )
            plan_weeks = st.slider("📆 Тижнів", 1, 12, 4, key="plan_weeks")

        sessions_per_week = st.slider("🏊 Тренувань на тиждень", 2, 6, 4, key="plan_sessions")

        if st.button("📋 Згенерувати план", type="primary", use_container_width=True):
            plan = generate_training_plan(
                athlete_name=plan_name,
                level=plan_level,
                goal=plan_goal,
                sessions_per_week=sessions_per_week,
                weeks=plan_weeks,
            )

            st.success(f"✅ План створено: {plan.notes}")

            # Display plan by weeks
            for week in range(1, plan_weeks + 1):
                with st.expander(f"📅 Тиждень {week}", expanded=(week == 1)):
                    week_sessions = [s for s in plan.sessions if s["week"] == week]

                    for session in week_sessions:
                        type_icon = "🏊" if session["type"] == "Плавання" else "🏋️"
                        st.markdown(
                            f"""
                        <div style="background: rgba(59,130,246,0.1); border-radius: 8px; padding: 0.75rem; margin: 0.5rem 0;">
                            <strong>{type_icon} {session['day']} - {session['type']}</strong> ({session['duration']} хв)<br>
                            <span style="color: #60a5fa;">Фокус: {session['focus']}</span><br>
                            <span style="color: #94a3b8;">{session['workout']}</span>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
