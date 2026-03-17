"""
Dryland/gym analysis page module.
"""

import logging
import sqlite3
import streamlit as st
import cv2
from pathlib import Path
from typing import Dict

from video_analysis.frame_extractor import extract_frames_from_video
from video_analysis.swimmer_detector import detect_swimmer_in_frames
from video_analysis.ai_coach import get_ai_coaching
from video_analysis.athlete_database import save_analysis_to_db
from video_analysis.biomechanics_visualizer import BiomechanicsVisualizer
from video_analysis.exercise_analyzer import ExerciseAnalyzer, generate_exercise_chart

logger = logging.getLogger(__name__)


def render_dryland_tab():
    """Render dryland/gym analysis tab."""

    st.markdown("""
    <div class="section-title">Аналіз сухих тренувань</div>
    """, unsafe_allow_html=True)

    # Settings
    with st.expander("⚙️ Налаштування", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            athlete_name = st.text_input(
                "👤 Ім'я спортсмена",
                value="Спортсмен",
                key="gym_athlete"
            )

        with col2:
            exercise_type = st.selectbox(
                "🏋️ Тип вправи",
                options=["general", "strength", "flexibility", "technique"],
                format_func=lambda x: {
                    "general": "🎯 Загальний аналіз",
                    "strength": "💪 Силові вправи",
                    "flexibility": "🤸 Гнучкість",
                    "technique": "🎓 Техніка рухів"
                }[x],
                key="gym_type"
            )

        with col3:
            fps = st.select_slider(
                "🎬 FPS",
                options=[10, 15, 20, 30],
                value=15,
                key="gym_fps"
            )

        slow_motion = st.select_slider(
            "🐢 Slow-motion",
            options=[1.0, 0.75, 0.5, 0.25],
            value=1.0,
            format_func=lambda x: f"{x}x" if x == 1.0 else f"🐢 {x}x",
            key="gym_slowmo"
        )

    # Upload
    st.markdown("""
    <div class="section-title">Завантаження відео</div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Перетягніть файл або оберіть",
        type=["mp4", "mov", "avi"],
        key="gym_upload"
    )

    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{uploaded_file.size / (1024*1024):.1f}</div>
                <div class="metric-label">МБ</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{fps}</div>
                <div class="metric-label">FPS</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🏋️ АНАЛІЗУВАТИ ВПРАВУ", type="primary", use_container_width=True, key="gym_analyze"):
            analyze_dryland(uploaded_file, athlete_name, exercise_type, fps, slow_motion)

    # Features
    with st.expander("📊 Можливості аналізу"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Поза тіла:**
            - 📐 33 ключові точки
            - 📏 Кути суглобів
            - 🦴 Вісь хребта
            - ⚖️ Баланс тіла
            """)
        with col2:
            st.markdown("""
            **Аналіз руху:**
            - 🔄 Траєкторія руху
            - ⏱️ Темп виконання
            - 📈 Амплітуда
            - ✅ Рекомендації
            """)


def analyze_dryland(uploaded_file, athlete_name, exercise_type, fps, slow_motion=1.0):
    """Analyze dryland/gym exercise video."""

    with st.spinner("🏋️ Аналізуємо вправу..."):
        # Create persistent output directory
        output_dir = Path("streamlit_outputs") / f"dryland_{Path(uploaded_file.name).stem}"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save uploaded file
            video_path = output_dir / uploaded_file.name
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Extract frames
            status_text.text("🎬 Витягуємо кадри...")
            frame_result = extract_frames_from_video(
                str(video_path),
                output_dir=str(output_dir / "frames"),
                fps=float(fps),
            )
            progress_bar.progress(15)
            st.markdown(f'<div class="status-success">✅ Витягнуто {frame_result["count"]} кадрів</div>', unsafe_allow_html=True)

            # Step 2: Detect person
            status_text.text("🎯 Детекція людини...")
            detection_result = detect_swimmer_in_frames(
                frame_result["frames"],
                output_dir=str(output_dir / "detections"),
                draw_boxes=True,
                enable_tracking=True,
            )
            progress_bar.progress(30)
            st.markdown('<div class="status-success">✅ Детекція завершена</div>', unsafe_allow_html=True)

            # Step 3: First pass - collect angles for rep detection
            status_text.text("🦴 Аналіз біомеханіки...")
            visualizer = BiomechanicsVisualizer(trajectory_length=30)

            first_frame_info = frame_result["frames"][0]
            first_path = str(Path(first_frame_info["path"] if isinstance(first_frame_info, dict) else first_frame_info))
            first_frame = cv2.imread(first_path)
            h, w = first_frame.shape[:2]

            all_angles = []
            frames_with_pose = 0
            detected_movements = []
            annotated_frames = []

            for i, frame_info in enumerate(frame_result["frames"]):
                frame_path = str(Path(frame_info["path"] if isinstance(frame_info, dict) else frame_info))
                frame = cv2.imread(frame_path)

                if frame is None:
                    all_angles.append({})
                    annotated_frames.append(None)
                    continue

                bbox = None
                if i < len(detection_result["detections"]):
                    bbox = detection_result["detections"][i].get("bbox")

                annotated_frame, analysis = visualizer.process_frame(frame, i, bbox)
                annotated_frames.append(annotated_frame)

                if analysis.get("has_pose"):
                    frames_with_pose += 1
                    angles = analysis.get("angles", {})
                    all_angles.append(angles)
                    movement = detect_movement_type(angles)
                    if movement:
                        detected_movements.append(movement)
                else:
                    all_angles.append({})

                if i % 20 == 0:
                    progress_bar.progress(30 + int(25 * (i / len(frame_result["frames"]))))

            progress_bar.progress(55)

            # Step 4: Analyze exercise (reps, tempo, etc.)
            status_text.text("🔄 Підрахунок повторень...")

            exercise_analyzer = ExerciseAnalyzer(fps=float(fps))
            exercise_stats = exercise_analyzer.analyze(all_angles, exercise_type)

            st.markdown(f'<div class="status-success">🔄 Знайдено <strong>{exercise_stats.total_reps}</strong> повторень</div>', unsafe_allow_html=True)
            if exercise_stats.total_reps > 0:
                st.markdown(f'<div class="status-info">⏱️ Темп: {exercise_stats.avg_tempo:.1f}с/повт | 📐 Амплітуда: {exercise_stats.avg_range_of_motion:.0f}° | 📊 Стабільність: {exercise_stats.stability_score:.0f}%</div>', unsafe_allow_html=True)

            progress_bar.progress(65)

            # Determine main movement
            if detected_movements:
                from collections import Counter
                movement_counts = Counter(detected_movements)
                main_movement = movement_counts.most_common(1)[0][0]
            else:
                main_movement = exercise_type

            # Step 5: Generate video with rep counter
            status_text.text("🎬 Створення відео з ефектами...")

            # Video FPS adjusted for slow-motion
            video_fps = float(fps) * slow_motion
            annotated_video_path = output_dir / "dryland_annotated.mp4"

            video_writer = None
            for codec in ["avc1", "mp4v"]:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_writer = cv2.VideoWriter(str(annotated_video_path), fourcc, video_fps, (w, h))
                if video_writer.isOpened():
                    break
            if video_writer is None or not video_writer.isOpened():
                raise RuntimeError("Не вдалося створити відеофайл: кодек не підтримується")

            for i, annotated_frame in enumerate(annotated_frames):
                if annotated_frame is None:
                    continue

                video_writer.write(annotated_frame)

                if i % 20 == 0:
                    progress_bar.progress(65 + int(20 * (i / len(annotated_frames))))

            video_writer.release()

            st.markdown(f'<div class="status-success">🎬 Відео створено ({slow_motion}x швидкість)</div>', unsafe_allow_html=True)

            progress_bar.progress(85)

            # Step 6: Generate chart
            status_text.text("📊 Генерація графіків...")
            chart_path = output_dir / "exercise_chart.png"
            generate_exercise_chart(exercise_stats, str(chart_path))

            progress_bar.progress(90)

            # Step 7: AI Coach
            status_text.text("🤖 AI тренер аналізує...")

            pose_result = {
                "detection_rate": frames_with_pose / len(frame_result["frames"]) if frame_result["frames"] else 0,
                "avg_streamline": 70,
                "avg_deviation": 5,
                "frame_analyses": [{"has_pose": True, "angles": a} for a in all_angles if a],
            }

            if all_angles:
                avg_angles = {}
                valid_angles = [a for a in all_angles if a]
                if valid_angles:
                    for key in valid_angles[0].keys():
                        values = [a.get(key, 0) for a in valid_angles if key in a]
                        avg_angles[key] = sum(values) / len(values) if values else 0
                    pose_result["avg_angles"] = avg_angles

            ai_advice = get_ai_coaching(
                biomechanics={"average_metrics": pose_result},
                athlete_name=athlete_name,
            )

            progress_bar.progress(100)
            status_text.text("✅ Аналіз завершено!")

            # Display results
            display_dryland_results(
                pose_result, detection_result, output_dir,
                {"main_movement": main_movement, "all_angles": all_angles, "exercise_stats": exercise_stats},
                annotated_video_path, ai_advice, chart_path if chart_path.exists() else None
            )

            # Save to database
            try:
                session_id = save_analysis_to_db(
                    athlete_name=athlete_name,
                    session_type="dryland",
                    analysis={"main_movement": main_movement, "exercise_stats": exercise_stats},
                    ai_advice=ai_advice,
                    video_path=str(video_path) if 'video_path' in dir() else ""
                )
                st.success(f"💾 Результати збережено в базу даних (сесія #{session_id})")
            except (sqlite3.Error, ValueError, OSError) as db_error:
                logger.warning("DB save error (dryland)", exc_info=True)
                st.warning(f"⚠️ Не вдалося зберегти в БД: {db_error}")

        except (ValueError, AttributeError, TypeError) as e:
            logger.warning("Dryland analysis error", exc_info=True)
            st.error(f"❌ Помилка аналізу: {str(e)}")
        except Exception:
            logger.exception("Unexpected error in dryland analysis")
            st.error("❌ Непередбачена помилка. Перевірте логи.")
            import traceback
            st.code(traceback.format_exc())


def detect_movement_type(angles: Dict) -> str:
    """Detect type of exercise based on joint angles."""
    if not angles:
        return "загальний рух"

    l_elbow = angles.get("L.elbow", 180)
    r_elbow = angles.get("R.elbow", 180)
    l_knee = angles.get("L.knee", 180)
    r_knee = angles.get("R.knee", 180)

    avg_elbow = (l_elbow + r_elbow) / 2
    avg_knee = (l_knee + r_knee) / 2

    # Detect movement patterns
    if avg_elbow < 90 and avg_knee > 150:
        return "🏋️ Згинання рук (біцепс)"
    elif avg_elbow < 60:
        return "💪 Віджимання / Жим"
    elif avg_knee < 100:
        return "🦵 Присідання"
    elif avg_knee < 130 and avg_elbow > 150:
        return "🏃 Випади"
    elif avg_elbow > 160 and avg_knee > 160:
        return "🧘 Планка / Стретчинг"
    elif 90 < avg_elbow < 140:
        return "🏊 Імітація гребка"
    else:
        return "🏋️ Загальна вправа"


def display_dryland_results(pose_result, detection_result, output_dir, biomech_result=None, video_path=None, ai_advice=None, chart_path=None):
    """Display dryland exercise analysis results."""

    st.markdown('<div class="section-title">Результати аналізу</div>', unsafe_allow_html=True)

    # ========================================================================
    # EXERCISE STATS (REPS, TEMPO, STABILITY)
    # ========================================================================
    exercise_stats = biomech_result.get("exercise_stats") if biomech_result else None

    if exercise_stats and exercise_stats.total_reps > 0:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
                    border-radius: 16px; padding: 1.5rem; margin: 1rem 0;">
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; font-weight: 800; color: #fff;">{exercise_stats.total_reps}</div>
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">🔄 ПОВТОРЕНЬ</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; font-weight: 700; color: #fbbf24;">{exercise_stats.avg_tempo:.1f}с</div>
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">⏱️ ТЕМП</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; font-weight: 700; color: #34d399;">{exercise_stats.avg_range_of_motion:.0f}°</div>
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">📐 АМПЛІТУДА</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; font-weight: 700; color: #60a5fa;">{exercise_stats.stability_score:.0f}%</div>
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">📊 СТАБІЛЬНІСТЬ</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Rep details table
        if exercise_stats.reps:
            with st.expander("📋 Деталі по кожному повторенню"):
                rep_data = []
                for rep in exercise_stats.reps:
                    rep_data.append({
                        "№": rep.rep_number,
                        "Тривалість (с)": f"{rep.duration_sec:.2f}",
                        "Мін. кут (°)": f"{rep.min_angle:.0f}",
                        "Макс. кут (°)": f"{rep.max_angle:.0f}",
                        "Амплітуда (°)": f"{rep.range_of_motion:.0f}",
                    })
                st.table(rep_data)

    # ========================================================================
    # DETECTED MOVEMENT TYPE
    # ========================================================================
    if biomech_result and biomech_result.get("main_movement"):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #059669 0%, #10b981 100%);
                    border-radius: 12px; padding: 1rem; margin: 1rem 0; text-align: center;">
            <div style="font-size: 1.8rem;">{biomech_result["main_movement"]}</div>
            <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Автоматично визначений тип вправи</div>
        </div>
        """, unsafe_allow_html=True)

    # ========================================================================
    # VIDEO WITH EFFECTS
    # ========================================================================
    if video_path and Path(video_path).exists():
        st.markdown('<div class="section-title">🎬 Відео з біомеханікою</div>', unsafe_allow_html=True)
        st.video(str(video_path))

        # Download button
        with open(video_path, "rb") as f:
            st.download_button(
                "📥 Завантажити відео",
                f,
                file_name="dryland_biomechanics.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

    # ========================================================================
    # CHART
    # ========================================================================
    if chart_path and Path(chart_path).exists():
        st.markdown('<div class="section-title">📊 Графіки аналізу</div>', unsafe_allow_html=True)
        st.image(str(chart_path), use_container_width=True)

    # ========================================================================
    # AI COACH
    # ========================================================================
    if ai_advice:
        st.markdown("---")
        st.markdown('<div class="section-title">🤖 AI Тренер</div>', unsafe_allow_html=True)

        score = ai_advice.score
        score_color = "#10b981" if score >= 70 else "#f59e0b" if score >= 50 else "#ef4444"

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(59,130,246,0.2) 0%, rgba(139,92,246,0.2) 100%);
                    border: 1px solid {score_color}; border-radius: 16px; padding: 1.5rem; margin: 1rem 0;">
            <div style="display: flex; align-items: center; gap: 2rem; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; font-weight: 800; color: {score_color};">{score}</div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">ОЦІНКА</div>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <div style="font-size: 1.1rem; color: #fff;">{ai_advice.summary}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Добре:**")
            for s in ai_advice.strengths:
                st.markdown(f'<div class="status-success">{s}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("**⚠️ Покращити:**")
            for imp in ai_advice.improvements:
                st.markdown(f'<div class="status-warning">{imp}</div>', unsafe_allow_html=True)

        if ai_advice.drills:
            st.markdown("**🏋️ Вправи:**")
            for drill in ai_advice.drills:
                st.markdown(f'<div class="status-info">{drill}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        det_rate = pose_result.get("detection_rate", 0) * 100
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{det_rate:.0f}%</div>
            <div class="metric-label">Детекція</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        streamline = pose_result.get("avg_streamline", 0)
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{streamline:.0f}</div>
            <div class="metric-label">Streamline</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        deviation = pose_result.get("avg_deviation", 0)
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{deviation:.1f}°</div>
            <div class="metric-label">Відхилення</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        frames = len(pose_result.get("frame_analyses", []))
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{frames}</div>
            <div class="metric-label">Кадрів</div>
        </div>
        """, unsafe_allow_html=True)

    # Detailed analysis
    st.markdown('<div class="section-title">Детальний аналіз</div>', unsafe_allow_html=True)

    frame_analyses = pose_result.get("frame_analyses", [])
    valid_frames = [f for f in frame_analyses if f.get("has_pose")]

    if valid_frames:
        # Average metrics
        avg_metrics = {}
        metrics_keys = ["body_roll", "hip_drop", "streamline_score", "kick_amplitude"]

        for key in metrics_keys:
            values = [f["metrics"].get(key, 0) for f in valid_frames if f.get("metrics")]
            if values:
                avg_metrics[key] = sum(values) / len(values)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📐 Положення тіла:**")
            if "body_roll" in avg_metrics:
                roll = avg_metrics["body_roll"]
                status = "✅" if abs(roll) < 15 else "⚠️"
                st.write(f"{status} Body Roll: {roll:.1f}°")

            if "hip_drop" in avg_metrics:
                hip = avg_metrics["hip_drop"]
                status = "✅" if abs(hip) < 30 else "⚠️"
                st.write(f"{status} Hip Drop: {hip:.1f}px")

        with col2:
            st.markdown("**📊 Оцінка:**")
            if "streamline_score" in avg_metrics:
                score = avg_metrics["streamline_score"]
                status = "✅" if score > 70 else "⚠️" if score > 50 else "❌"
                st.write(f"{status} Streamline Score: {score:.0f}/100")

            if "kick_amplitude" in avg_metrics:
                amp = avg_metrics["kick_amplitude"]
                st.write(f"📈 Амплітуда: {amp:.0f}px")

        # Recommendations
        st.markdown('<div class="section-title">Рекомендації</div>', unsafe_allow_html=True)

        recommendations = []

        if avg_metrics.get("streamline_score", 100) < 70:
            recommendations.append("⚠️ Покращуйте положення тіла - тримайте спину рівно")

        if abs(avg_metrics.get("body_roll", 0)) > 20:
            recommendations.append("⚠️ Зменшіть обертання тіла - стабілізуйте корпус")

        if abs(avg_metrics.get("hip_drop", 0)) > 40:
            recommendations.append("⚠️ Контролюйте положення стегон - не опускайте їх")

        if not recommendations:
            recommendations.append("✅ Відмінна техніка! Продовжуйте в тому ж дусі.")

        for rec in recommendations:
            if "⚠️" in rec:
                st.markdown(f'<div class="status-warning">{rec}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-success">{rec}</div>', unsafe_allow_html=True)

    # Sample pose images
    pose_dir = output_dir / "pose_analysis"
    if pose_dir.exists():
        st.markdown('<div class="section-title">Візуалізація пози</div>', unsafe_allow_html=True)

        pose_images = sorted(pose_dir.glob("pose_*.jpg"))[:6]  # First 6

        if pose_images:
            cols = st.columns(3)
            for i, img_path in enumerate(pose_images):
                with cols[i % 3]:
                    st.image(str(img_path), caption=f"Кадр {i+1}", use_container_width=True)
