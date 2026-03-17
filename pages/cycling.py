"""
Cycling analysis page module.
"""

import logging
import sqlite3
import streamlit as st
import cv2
from pathlib import Path

from video_analysis.frame_extractor import extract_frames_from_video
from video_analysis.swimmer_detector import detect_swimmer_in_frames
from video_analysis.ai_coach import get_ai_coaching
from video_analysis.athlete_database import save_analysis_to_db
from video_analysis.biomechanics_visualizer import BiomechanicsVisualizer
from video_analysis.cycling_analyzer import CyclingAnalyzer, CyclingAnalysis, generate_cycling_chart

logger = logging.getLogger(__name__)


def render_cycling_tab():
    """Render cycling analysis tab."""

    st.markdown('<div class="section-title">🚴 Аналіз техніки велосипеда</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Налаштування", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            athlete_name = st.text_input("👤 Ім'я спортсмена", value="Спортсмен", key="bike_athlete")
        with col2:
            fps = st.number_input("🎬 FPS відео", min_value=15, max_value=240, value=30, key="bike_fps")
        with col3:
            bike_type = st.selectbox("🚴 Тип їзди",
                ["Шосе", "Триатлон TT", "MTB", "Трек"],
                key="bike_type")

    uploaded_file = st.file_uploader(
        "📹 Завантажте відео (вид збоку на тренажері або дорозі)",
        type=["mp4", "mov", "avi", "mkv"],
        key="bike_upload"
    )

    if uploaded_file:
        st.video(uploaded_file)

        if st.button("🚴 АНАЛІЗУВАТИ ВЕЛОСИПЕД", type="primary", use_container_width=True, key="bike_analyze"):
            analyze_cycling(uploaded_file, athlete_name, fps, bike_type)

    with st.expander("📊 Можливості аналізу"):
        st.markdown("""
        | Метрика | Опис |
        |---------|------|
        | **Cadence** | Обертів на хвилину (оптимум 80-100) |
        | **Knee Angle** | Кут коліна вгорі/внизу педалювання |
        | **Hip Angle** | Кут нахилу корпусу (аеро позиція) |
        | **Stability** | Стабільність верхньої частини тіла |
        | **L/R Balance** | Баланс лівої/правої ноги |
        | **Saddle Height** | Оцінка висоти сідла |
        """)


def analyze_cycling(uploaded_file, athlete_name, fps, bike_type):
    """Analyze cycling video - full pipeline like dryland."""

    with st.spinner("🚴 Аналізуємо велосипед..."):
        # Create persistent output directory
        output_dir = Path("streamlit_outputs") / f"cycling_{Path(uploaded_file.name).stem}"
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
            status_text.text("🎯 Детекція велосипедиста...")
            detection_result = detect_swimmer_in_frames(
                frame_result["frames"],
                output_dir=str(output_dir / "detections"),
                draw_boxes=True,
                enable_tracking=True,
            )
            progress_bar.progress(30)
            st.markdown('<div class="status-success">✅ Детекція завершена</div>', unsafe_allow_html=True)

            # Step 3: Biomechanics analysis with skeleton visualization
            status_text.text("🦴 Аналіз біомеханіки...")
            visualizer = BiomechanicsVisualizer(trajectory_length=30)

            first_frame_info = frame_result["frames"][0]
            first_path = str(Path(first_frame_info["path"] if isinstance(first_frame_info, dict) else first_frame_info))
            first_frame = cv2.imread(first_path)
            h, w = first_frame.shape[:2]

            keypoints_list = []
            annotated_frames = []
            frames_with_pose = 0

            for i, frame_info in enumerate(frame_result["frames"]):
                frame_path = str(Path(frame_info["path"] if isinstance(frame_info, dict) else frame_info))
                frame = cv2.imread(frame_path)

                if frame is None:
                    keypoints_list.append({})
                    annotated_frames.append(None)
                    continue

                bbox = None
                if i < len(detection_result["detections"]):
                    bbox = detection_result["detections"][i].get("bbox")

                # Get annotated frame with skeleton
                annotated_frame, analysis_data = visualizer.process_frame(frame, i, bbox)
                annotated_frames.append(annotated_frame)

                # Extract keypoints for cycling analysis
                kps = {}
                if analysis_data.get("has_pose") and analysis_data.get("keypoints"):
                    frames_with_pose += 1
                    raw_kps = analysis_data.get("keypoints", {})
                    landmark_map = {
                        "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
                        "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
                        "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
                        "left_ankle": 27, "right_ankle": 28,
                    }
                    for name, idx in landmark_map.items():
                        if name in raw_kps:
                            kps[name] = raw_kps[name]
                        elif str(idx) in raw_kps:
                            kps[name] = raw_kps[str(idx)]

                keypoints_list.append(kps)

                if i % 20 == 0:
                    progress_bar.progress(30 + int(25 * (i / len(frame_result["frames"]))))

            progress_bar.progress(55)
            st.markdown(f'<div class="status-success">✅ Поза виявлена на {frames_with_pose}/{len(frame_result["frames"])} кадрах</div>', unsafe_allow_html=True)

            # Step 4: Cycling-specific analysis
            status_text.text("🚴 Аналіз техніки педалювання...")
            analyzer = CyclingAnalyzer(fps=float(fps))
            cycling_analysis = analyzer.analyze(keypoints_list, fps=float(fps))

            st.markdown(f'<div class="status-success">🚴 Cadence: {cycling_analysis.cadence:.0f} RPM | Bike Fit: {cycling_analysis.bike_fit_score:.0f}</div>', unsafe_allow_html=True)

            progress_bar.progress(65)

            # Step 5: Generate video with skeleton
            status_text.text("🎬 Створення відео зі скелетом...")

            annotated_video_path = output_dir / "cycling_annotated.mp4"

            video_writer = None
            for codec in ["avc1", "mp4v"]:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_writer = cv2.VideoWriter(str(annotated_video_path), fourcc, float(fps), (w, h))
                if video_writer.isOpened():
                    break
            if video_writer is None or not video_writer.isOpened():
                raise RuntimeError("Не вдалося створити відеофайл: кодек не підтримується")

            for i, annotated_frame in enumerate(annotated_frames):
                if annotated_frame is None:
                    continue

                # Add cycling metrics overlay
                cv2.putText(annotated_frame, f"Cadence: {cycling_analysis.cadence:.0f} RPM",
                           (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(annotated_frame, f"Knee Range: {cycling_analysis.knee_range:.0f} deg",
                           (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(annotated_frame, f"Bike Fit: {cycling_analysis.bike_fit_score:.0f}/100",
                           (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                video_writer.write(annotated_frame)

                if i % 20 == 0:
                    progress_bar.progress(65 + int(15 * (i / len(annotated_frames))))

            video_writer.release()
            st.markdown('<div class="status-success">🎬 Відео зі скелетом створено</div>', unsafe_allow_html=True)

            progress_bar.progress(80)

            # Step 6: Generate chart
            status_text.text("📊 Генерація графіків...")
            chart_path = output_dir / "cycling_chart.png"
            generate_cycling_chart(cycling_analysis, str(chart_path))

            progress_bar.progress(90)

            # Step 7: AI Coach
            status_text.text("🤖 AI тренер аналізує...")
            ai_advice = get_ai_coaching(
                biomechanics={"cycling": {
                    "cadence": cycling_analysis.cadence,
                    "knee_range": cycling_analysis.knee_range,
                    "hip_angle": cycling_analysis.avg_hip_angle,
                    "stability": cycling_analysis.upper_body_stability,
                    "bike_fit": cycling_analysis.bike_fit_score,
                    "pedal_smoothness": cycling_analysis.pedal_smoothness
                }},
                athlete_name=athlete_name,
            )

            progress_bar.progress(100)
            status_text.text("✅ Аналіз завершено!")

            # Display results
            display_cycling_results(cycling_analysis, ai_advice, chart_path, bike_type, annotated_video_path)

            # Save to database
            try:
                session_id = save_analysis_to_db(
                    athlete_name=athlete_name,
                    session_type="cycling",
                    analysis={
                        "cadence": cycling_analysis.cadence,
                        "knee_range": cycling_analysis.knee_range,
                        "bike_fit": cycling_analysis.bike_fit_score,
                        "efficiency": cycling_analysis.efficiency_score
                    },
                    ai_advice=ai_advice,
                    video_path=str(video_path)
                )
                st.success(f"💾 Результати збережено в базу даних (сесія #{session_id})")
            except (sqlite3.Error, ValueError, OSError) as db_error:
                logger.warning("DB save error (cycling)", exc_info=True)
                st.warning(f"⚠️ Не вдалося зберегти в БД: {db_error}")

        except (ValueError, AttributeError, TypeError) as e:
            logger.warning("Cycling analysis error", exc_info=True)
            st.error(f"❌ Помилка аналізу: {e}")
        except Exception as e:
            logger.exception("Unexpected error in cycling analysis")
            st.error("❌ Непередбачена помилка. Перевірте логи.")
            import traceback
            st.code(traceback.format_exc())


def display_cycling_results(analysis: CyclingAnalysis, ai_advice, chart_path, bike_type, annotated_video_path=None):
    """Display cycling analysis results."""

    st.markdown("---")
    st.markdown('<div class="success-box" style="text-align: center;">🚴 Аналіз велосипеда завершено!</div>', unsafe_allow_html=True)

    # Show annotated video with skeleton
    if annotated_video_path and Path(annotated_video_path).exists():
        st.markdown("### 🎬 Відео зі скелетом")
        st.video(str(annotated_video_path))

    # Main metrics
    st.markdown("### 📊 Ключові метрики")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cadence_color = "#10b981" if 80 <= analysis.cadence <= 100 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {cadence_color};">{analysis.cadence:.0f}</div>
            <div class="metric-label">Cadence (RPM)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: #3b82f6;">{analysis.knee_range:.0f}°</div>
            <div class="metric-label">Knee Range</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: #8b5cf6;">{analysis.avg_hip_angle:.0f}°</div>
            <div class="metric-label">Hip Angle</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        stab_color = "#10b981" if analysis.upper_body_stability >= 70 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {stab_color};">{analysis.upper_body_stability:.0f}%</div>
            <div class="metric-label">Stability</div>
        </div>
        """, unsafe_allow_html=True)

    # NEW: Advanced metrics
    st.markdown("### ⚙️ Техніка педалювання")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ank_color = "#10b981" if analysis.ankling_score >= 80 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {ank_color};">{analysis.ankling_score:.0f}</div>
            <div class="metric-label">Ankling Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        ds_color = "#10b981" if analysis.dead_spot_score >= 80 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {ds_color};">{analysis.dead_spot_score:.0f}</div>
            <div class="metric-label">Dead Spot Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        smooth_color = "#10b981" if analysis.pedal_smoothness >= 80 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {smooth_color};">{analysis.pedal_smoothness:.0f}</div>
            <div class="metric-label">Smoothness</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        rock_color = "#10b981" if not analysis.rock_detected else "#ef4444"
        rock_text = "❌ ТАК" if analysis.rock_detected else "✅ НІ"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {rock_color};">{rock_text}</div>
            <div class="metric-label">Rock/Sway</div>
        </div>
        """, unsafe_allow_html=True)

    # Bike fit scores
    st.markdown("### 🔧 Bike Fit")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        saddle_color = "#10b981" if analysis.saddle_height_score >= 80 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {saddle_color};">{analysis.saddle_height_score:.0f}</div>
            <div class="metric-label">Saddle Height</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        aero_color = "#10b981" if analysis.aero_score >= 80 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {aero_color};">{analysis.aero_score:.0f}</div>
            <div class="metric-label">Aero Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        fit_color = "#10b981" if analysis.bike_fit_score >= 80 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {fit_color};">{analysis.bike_fit_score:.0f}</div>
            <div class="metric-label">Overall Fit</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        balance_delta = abs(analysis.left_right_balance - 50)
        balance_color = "#10b981" if balance_delta < 5 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {balance_color};">{analysis.left_right_balance:.0f}%</div>
            <div class="metric-label">L/R Balance</div>
        </div>
        """, unsafe_allow_html=True)

    # Chart
    if chart_path.exists():
        st.image(str(chart_path), use_container_width=True)

    # AI advice
    if ai_advice:
        st.markdown("### 🤖 AI Тренер")
        score_color = "#10b981" if ai_advice.score >= 70 else "#f59e0b" if ai_advice.score >= 50 else "#ef4444"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(255,165,0,0.2), rgba(139,92,246,0.2));
                    border-radius: 12px; padding: 1rem; border: 1px solid {score_color};">
            <div style="font-size: 2rem; font-weight: bold; color: {score_color};">{ai_advice.score}/100</div>
            <div>{ai_advice.summary}</div>
        </div>
        """, unsafe_allow_html=True)
