"""
History/progress page module.
"""

import streamlit as st

from video_analysis.athlete_database import get_database


def render_history_tab():
    """Render athlete history and progress tab."""

    st.markdown('<div class="section-title">📊 Історія тренувань</div>', unsafe_allow_html=True)

    db = get_database()
    athletes = db.get_all_athletes()

    if not athletes:
        st.info("👤 Поки немає збережених спортсменів. Проведіть аналіз відео щоб створити першу запис.")
        return

    # Athlete selector
    col1, col2 = st.columns([2, 1])
    with col1:
        athlete_names = [a.name for a in athletes]
        selected_name = st.selectbox("👤 Оберіть спортсмена", athlete_names, key="history_athlete")

    with col2:
        if st.button("🗑️ Видалити спортсмена", type="secondary"):
            athlete = db.get_athlete(name=selected_name)
            if athlete:
                db.delete_athlete(athlete.id)
                st.rerun()

    athlete = db.get_athlete(name=selected_name)
    if not athlete:
        return

    # Athlete stats
    stats = db.get_athlete_stats(athlete.id)

    st.markdown("### 📈 Загальна статистика")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Сесій", stats["total_sessions"])
    with col2:
        st.metric("Середня оцінка", f"{stats['avg_score']}/100")
    with col3:
        st.metric("Найкраща оцінка", f"{stats['best_score']}/100")
    with col4:
        st.metric("Загальний час", f"{stats['total_time_min']:.0f} хв")

    # Sessions by type
    if stats["by_type"]:
        st.markdown("### 📊 По типам тренувань")
        for stype, sdata in stats["by_type"].items():
            type_icon = "🏊" if stype == "swimming" else "🏋️"
            st.markdown(
                f"""
            <div style="background: rgba(59,130,246,0.1); border-radius: 8px; padding: 0.5rem 1rem; margin: 0.5rem 0;">
                <strong>{type_icon} {stype.capitalize()}</strong>: {sdata['count']} сесій |
                Середня: {sdata['avg_score']}/100 | Найкраща: {sdata['best_score']}/100
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Multi-metric progress chart
    st.markdown("### 📈 Динаміка метрик")

    import pandas as pd

    _METRIC_LABELS = {
        "ai_score": "AI Score",
        "avg_speed": "Швидкість (м/с)",
        "stroke_rate": "Темп гребків/хв",
        "symmetry_score": "Симетрія (%)",
        "streamline_score": "Обтічність",
        "stability_score": "Стабільність (%)",
        "duration_sec": "Тривалість (с)",
    }
    _PERIOD_DAYS = {"7 днів": 7, "30 днів": 30, "90 днів": 90, "Весь час": None}
    _SESSION_TYPES = {
        "Всі": "all",
        "Swimming": "swimming",
        "Running": "running",
        "Cycling": "cycling",
        "Dryland": "dryland",
    }

    _mcol1, _mcol2, _mcol3 = st.columns(3)
    with _mcol1:
        selected_metric = st.selectbox(
            "📊 Метрика",
            list(_METRIC_LABELS.keys()),
            format_func=lambda k: _METRIC_LABELS[k],
            key="hist_metric",
        )
    with _mcol2:
        selected_period_label = st.radio("📅 Період", list(_PERIOD_DAYS.keys()), horizontal=True, key="hist_period")
    with _mcol3:
        selected_type_label = st.selectbox("🏊 Тип", list(_SESSION_TYPES.keys()), key="hist_stype")

    progress_data = db.get_progress(
        athlete.id,
        metric=selected_metric,
        session_type=_SESSION_TYPES[selected_type_label],
        limit=200,
        days=_PERIOD_DAYS[selected_period_label],
    )

    if progress_data:
        df_prog = pd.DataFrame(progress_data)
        df_prog["date"] = pd.to_datetime(df_prog["date"])
        df_prog = df_prog.dropna(subset=["value"])
        if not df_prog.empty:
            try:
                import plotly.express as px

                fig = px.line(
                    df_prog,
                    x="date",
                    y="value",
                    title=f"{_METRIC_LABELS[selected_metric]} — {athlete.name}",
                    markers=True,
                    color_discrete_sequence=["#3b82f6"],
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#94a3b8",
                    xaxis_title="Дата",
                    yaxis_title=_METRIC_LABELS[selected_metric],
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.line_chart(df_prog.set_index("date")["value"])
        else:
            st.info("Немає даних для вибраної метрики/фільтрів")
    else:
        st.info("Недостатньо даних для графіка прогресу")

    # Session history
    st.markdown("### 📋 Історія сесій")
    sessions = db.get_sessions(athlete.id, limit=100)

    if sessions:
        # CSV export
        import pandas as pd

        _csv_rows = []
        for s in sessions:
            _csv_rows.append(
                {
                    "date": s.date[:10],
                    "type": s.session_type,
                    "ai_score": s.ai_score,
                    "distance_m": s.distance_m,
                    "duration_sec": s.duration_sec,
                    "avg_speed": s.avg_speed,
                    "stroke_rate": s.stroke_rate,
                    "symmetry_score": s.symmetry_score,
                    "ai_summary": s.ai_summary or "",
                }
            )
        _csv_bytes = pd.DataFrame(_csv_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Завантажити CSV",
            _csv_bytes,
            f"{selected_name}_history.csv",
            "text/csv",
            key="csv_export",
        )

        for session in sessions:
            type_icon = "🏊" if session.session_type == "swimming" else "🏋️"
            with st.expander(f"{type_icon} {session.date[:10]} - Оцінка: {session.ai_score}/100"):
                col1, col2 = st.columns(2)

                with col1:
                    if session.session_type == "swimming":
                        st.write(f"**Дистанція:** {session.distance_m:.0f} м")
                        st.write(f"**Час:** {session.duration_sec:.0f} с")
                        st.write(f"**Швидкість:** {session.avg_speed:.2f} м/с")
                        st.write(f"**Гребків/хв:** {session.stroke_rate:.0f}")
                        st.write(f"**Симетрія:** {session.symmetry_score:.0f}%")
                    else:
                        st.write(f"**Вправа:** {session.exercise_type}")
                        st.write(f"**Повторень:** {session.reps}")
                        st.write(f"**Темп:** {session.avg_tempo:.1f} с/повт")
                        st.write(f"**Стабільність:** {session.stability_score:.0f}%")

                with col2:
                    if session.ai_summary:
                        st.write(f"**AI резюме:** {session.ai_summary}")

                if st.button("🗑️ Видалити", key=f"del_session_{session.id}"):
                    db.delete_session(session.id)
                    st.rerun()
    else:
        st.info("Поки немає записаних сесій")

    # Compare sessions
    if len(sessions) >= 2:
        st.markdown("### ⚖️ Порівняти сесії")
        col1, col2, col3 = st.columns([2, 2, 1])

        session_options = {f"{s.date[:10]} (#{s.id})": s.id for s in sessions}

        with col1:
            s1_label = st.selectbox("Сесія 1", list(session_options.keys()), key="compare_s1")
        with col2:
            s2_label = st.selectbox("Сесія 2", list(session_options.keys()), index=1, key="compare_s2")
        with col3:
            if st.button("Порівняти"):
                comparison = db.compare_sessions(session_options[s1_label], session_options[s2_label])

                if comparison.get("improvements"):
                    st.success("**Покращення:** " + ", ".join(comparison["improvements"]))
                if comparison.get("regressions"):
                    st.warning("**Погіршення:** " + ", ".join(comparison["regressions"]))
                if not comparison.get("improvements") and not comparison.get("regressions"):
                    st.info("Результати приблизно однакові")
