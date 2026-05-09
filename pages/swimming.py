"""
Swimming analysis page module.
"""

import logging
import sqlite3
import streamlit as st
import tempfile
from pathlib import Path

from video_analysis.frame_extractor import extract_frames_from_video
from video_analysis.swimmer_detector import detect_swimmer_in_frames
from video_analysis.split_analyzer import analyze_swimming_video
from video_analysis.biomechanics_analyzer import analyze_biomechanics
from video_analysis.trajectory_analyzer import analyze_trajectory
from video_analysis.report_generator import ReportGenerator
from video_analysis.video_overlay import VideoOverlayGenerator
from video_analysis.swimming_pose_analyzer import analyze_swimming_pose
from video_analysis.ai_coach import get_ai_coaching
from video_analysis.biomechanics_visualizer import visualize_biomechanics
from video_analysis.stroke_analyzer import generate_stroke_chart
from video_analysis.analyzer_factory import get_stroke_analyzer
from video_analysis.athlete_database import save_analysis_to_db

logger = logging.getLogger(__name__)


def render_swimming_tab():
    """Render swimming analysis tab."""

    st.markdown("""
    <div class="section-title">Аналіз техніки плавання</div>
    """, unsafe_allow_html=True)

    # Settings in expander
    with st.expander("⚙️ Налаштування аналізу", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            athlete_name = st.text_input(
                "👤 Ім'я спортсмена",
                value="Спортсмен",
                key="swim_athlete"
            )

        with col2:
            pool_length = st.selectbox(
                "🏊 Басейн",
                options=[25, 50],
                index=0,
                format_func=lambda x: f"{x}м",
                key="swim_pool"
            )

        with col3:
            fps = st.select_slider(
                "🎬 FPS",
                options=[5, 10, 15, 20, 30, 60],
                value=15,
                key="swim_fps"
            )

        col4, col5 = st.columns(2)

        with col4:
            analysis_method = st.selectbox(
                "🔬 Метод",
                options=["hybrid", "pose", "trajectory"],
                format_func=lambda x: {
                    "hybrid": "🎯 Гібридний",
                    "pose": "🔬 Поза",
                    "trajectory": "📍 Траєкторія"
                }[x],
                key="swim_method"
            )

        with col5:
            # FPS info
            if fps >= 30:
                st.markdown('<div class="status-warning">⚡ Детальний аналіз (5-10 хв)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-info">⏱️ Швидкий аналіз (1-3 хв)</div>', unsafe_allow_html=True)

    with st.expander("🎯 Трекінг плавця", expanded=False):
        st.markdown("Налаштування для стабільного відстеження потрібного плавця")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            swim_target = st.selectbox(
                "🎯 Кого аналізувати?",
                ["Найбільший у кадрі (авто)", "Найближчий до центру", "Вибрати за номером"],
                key="swim_target_person"
            )
            if swim_target == "Вибрати за номером":
                swim_person_num = st.number_input("Номер (зліва направо)", min_value=1, max_value=10, value=1, key="swim_person_num")
            else:
                swim_person_num = 1
        with col_t2:
            swim_max_lost = st.slider("Макс. кадрів без детекції", min_value=1, max_value=60, value=15, key="swim_max_lost")
            st.caption("При втраті плавця (під водою) використовуватимуться дані попереднього кадру")

    # Upload area
    st.markdown("""
    <div class="section-title">Завантаження відео</div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Перетягніть файл або оберіть",
        type=["mp4", "mov", "avi"],
        key="swim_upload",
        help="MP4, MOV, AVI до 200 МБ"
    )

    if uploaded_file:
        # File info
        col1, col2, col3 = st.columns(3)
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
        with col3:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{pool_length}м</div>
                <div class="metric-label">Басейн</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🏊 АНАЛІЗУВАТИ ПЛАВАННЯ", type="primary", use_container_width=True, key="swim_analyze"):
            analyze_video(
                uploaded_file, athlete_name, pool_length, fps, analysis_method,
                target_person=swim_target,
                person_number=swim_person_num,
                max_lost_frames=swim_max_lost,
            )

    # Features list
    with st.expander("📊 Можливості аналізу"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Детекція:**
            - 🎯 YOLO детекція плавця
            - 🔄 Velocity Tracking
            - 🌊 Підводна детекція
            - 📍 Сегментація тіла
            """)
        with col2:
            st.markdown("""
            **Біомеханіка:**
            - 📐 33 точки тіла
            - 📏 Вісь хребта
            - 💧 Гідродинаміка
            - ⏱️ Точні спліти
            """)



def _select_target_swimmer(detections, target_person="Найбільший у кадрі (авто)", person_number=1, max_lost_frames=15):
    """Filter detections to keep only the target swimmer across all frames using IoU tracking."""
    if not detections:
        return detections

    def _iou(a, b):
        if not a or not b:
            return 0.0
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _centroid_dist(a, b):
        if not a or not b:
            return float("inf")
        ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
        cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5

    def _pick_initial(all_dets, strategy, num):
        if not all_dets:
            return None
        sorted_dets = sorted(all_dets, key=lambda d: (
            (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
        ), reverse=True)
        if strategy == "Вибрати за номером":
            idx = min(num - 1, len(sorted_dets) - 1)
            return sorted_dets[idx]["bbox"]
        if strategy == "Найближчий до центру":
            # pick the one with bbox center closest to frame center
            # use confidence-based sort as proxy when no frame size available
            best = min(all_dets, key=lambda d: (
                ((d["bbox"][0] + d["bbox"][2]) / 2 - 320) ** 2 +
                ((d["bbox"][1] + d["bbox"][3]) / 2 - 240) ** 2
            ))
            return best["bbox"]
        # Default: largest
        return sorted_dets[0]["bbox"]

    result = []
    target_bbox = None
    lost_count = 0

    for det in detections:
        all_dets = det.get("all_detections") or []
        # Filter out entries with None bbox
        all_dets = [d for d in all_dets if d.get("bbox")]

        if not all_dets:
            if target_bbox and lost_count < max_lost_frames:
                lost_count += 1
                result.append(det)  # keep as-is (no bbox update)
            else:
                target_bbox = None
                result.append(det)
            continue

        if target_bbox is None:
            # First valid frame — pick initial target
            target_bbox = _pick_initial(all_dets, target_person, person_number)
            lost_count = 0
        else:
            # Match by IoU then centroid
            best_match = max(all_dets, key=lambda d: _iou(target_bbox, d["bbox"]))
            if _iou(target_bbox, best_match["bbox"]) > 0:
                target_bbox = best_match["bbox"]
                lost_count = 0
            else:
                # IoU = 0, fall back to centroid distance
                best_match = min(all_dets, key=lambda d: _centroid_dist(target_bbox, d["bbox"]))
                dist = _centroid_dist(target_bbox, best_match["bbox"])
                if dist < 200:  # pixels threshold
                    target_bbox = best_match["bbox"]
                    lost_count = 0
                else:
                    lost_count += 1

        if target_bbox and lost_count == 0:
            # Update detection to use target swimmer's bbox
            cx = (target_bbox[0] + target_bbox[2]) // 2
            cy = (target_bbox[1] + target_bbox[3]) // 2
            new_det = dict(det)
            new_det["bbox"] = target_bbox
            new_det["center"] = (cx, cy)
            result.append(new_det)
        else:
            result.append(det)

    return result


def analyze_video(uploaded_file, athlete_name, pool_length, fps, analysis_method,
                  target_person="Найбільший у кадрі (авто)", person_number=1, max_lost_frames=15):
    """Run video analysis pipeline."""

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save uploaded video
        video_path = temp_path / uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Create output directory
        output_dir = Path("streamlit_outputs") / Path(uploaded_file.name).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Крок 1: Витягуємо кадри
            status_text.text("🎞️ Витягуємо кадри з відео...")
            progress_bar.progress(10)

            frames_dir = output_dir / "frames"
            frame_result = extract_frames_from_video(
                str(video_path),
                output_dir=str(frames_dir),
                fps=fps,
            )

            st.markdown(f'<div class="success-box">✅ Витягнуто {frame_result["count"]} кадрів (з timestamp)</div>', unsafe_allow_html=True)
            progress_bar.progress(25)

            # Крок 2: Детекція плавця
            status_text.text("👁️ Детекція плавця (YOLO + 🌊 підводна)...")

            detections_dir = output_dir / "detections"
            detection_result = detect_swimmer_in_frames(
                frame_result["frames"],
                output_dir=str(detections_dir),
                draw_boxes=True,
                enable_tracking=True,  # CRITICAL for swimmer tracking!
            )

            st.markdown('<div class="success-box">✅ Детекція завершена (Velocity Tracking + 🌊 підводна детекція)</div>', unsafe_allow_html=True)

            # Filter to target swimmer if not default auto-tracking
            if target_person != "Найбільший у кадрі (авто)" or person_number != 1:
                detection_result["detections"] = _select_target_swimmer(
                    detection_result["detections"],
                    target_person=target_person,
                    person_number=person_number,
                    max_lost_frames=max_lost_frames,
                )

            progress_bar.progress(40)

            # Step 3: Biomechanics/Trajectory
            biomechanics_result = None
            trajectory_result = None

            if analysis_method in ["pose", "hybrid"]:
                status_text.text("🔬 Аналіз біомеханіки (pose)...")
                biomechanics_dir = output_dir / "biomechanics"
                biomechanics_result = analyze_biomechanics(
                    frame_result["frames"],
                    detection_result["detections"],
                    output_dir=str(biomechanics_dir),
                )
                st.markdown('<div class="success-box">✅ Біомеханічний аналіз (pose) завершено</div>', unsafe_allow_html=True)

                # NEW: Swimming-specific pose analysis with rotation compensation
                status_text.text("🏊 Аналіз пози плавця (rotation + spine)...")
                swimming_pose_dir = output_dir / "swimming_pose"
                swimming_pose_result = analyze_swimming_pose(
                    frame_result["frames"],
                    detection_result["detections"],
                    output_dir=str(swimming_pose_dir),
                )
                biomechanics_result["swimming_pose"] = swimming_pose_result
                st.markdown(f'<div class="success-box">✅ Pose: detection rate {swimming_pose_result["detection_rate"]*100:.0f}%, streamline {swimming_pose_result["avg_streamline"]:.0f}/100</div>', unsafe_allow_html=True)

                # NEW: Stroke analysis (phases, rate, symmetry, body roll)
                status_text.text("🏊 Аналіз гребка (фази, симетрія, body roll)...")
                stroke_analyzer = get_stroke_analyzer(fps=float(fps))

                # Extract keypoints from swimming_pose_result
                keypoints_list = []
                for frame_data in swimming_pose_result.get("frame_analyses", []):
                    kps = frame_data.get("keypoints", {})
                    keypoints_list.append(kps)

                stroke_analysis = stroke_analyzer.analyze(keypoints_list, fps=float(fps))
                biomechanics_result["stroke_analysis"] = stroke_analysis

                # Generate stroke chart
                stroke_chart_path = output_dir / "stroke_chart.png"
                generate_stroke_chart(stroke_analysis, str(stroke_chart_path))

                st.markdown(f'<div class="success-box">🏊 Гребки: {stroke_analysis.total_strokes} | Темп: {stroke_analysis.stroke_rate}/хв | Симетрія: {stroke_analysis.symmetry_score:.0f}% | Body Roll: {stroke_analysis.avg_body_roll:.1f}°</div>', unsafe_allow_html=True)

                # NEW: Advanced biomechanics visualization (skeleton + angles + trajectories)
                status_text.text("🦴 Візуалізація біомеханіки...")
                biomech_viz_dir = output_dir / "biomech_viz"
                biomech_viz_result = visualize_biomechanics(
                    frame_result["frames"],
                    detection_result["detections"],
                    output_dir=str(biomech_viz_dir),
                )
                biomechanics_result["visualization"] = biomech_viz_result
                st.markdown(f'<div class="success-box">🦴 Візуалізація: {biomech_viz_result["with_pose"]}/{biomech_viz_result["total"]} кадрів з скелетом</div>', unsafe_allow_html=True)

            if analysis_method in ["trajectory", "hybrid"]:
                status_text.text("📍 Аналіз траєкторії (bbox)...")
                trajectory_dir = output_dir / "trajectory"
                trajectory_result = analyze_trajectory(
                    detection_result["detections"],
                    fps=max(1, int(fps)),
                    pool_length=pool_length,
                    output_dir=str(trajectory_dir),
                )
                st.markdown('<div class="success-box">✅ Аналіз траєкторії (bbox) завершено</div>', unsafe_allow_html=True)

            progress_bar.progress(60)

            # Крок 4: Аналіз сплітів
            status_text.text("⏱️ Аналіз сплітів і швидкості...")

            analysis = analyze_swimming_video(
                detection_result["detections"],
                pool_length=pool_length,
                fps=max(1.0, float(fps)),  # Используем точный float, не округлённый
                output_path=str(output_dir / "analysis.json"),
            )
            analysis["biomechanics"] = biomechanics_result
            analysis["trajectory"] = trajectory_result
            analysis["analysis_method"] = analysis_method

            st.markdown('<div class="success-box">✅ Аналіз сплітів завершено (за реальним timestamp)</div>', unsafe_allow_html=True)
            progress_bar.progress(75)

            # Крок 5: Генерація звітів
            status_text.text("📊 Генерація звітів...")

            reports_dir = output_dir / "reports"
            generator = ReportGenerator(output_dir=str(reports_dir))
            generator.generate_complete_report(
                analysis,
                athlete_name=athlete_name,
            )

            progress_bar.progress(85)

            # Крок 6: Створюємо анотоване відео
            status_text.text("🎬 Створення анотованого відео...")

            video_fps = max(10.0, float(fps))
            overlay_generator = VideoOverlayGenerator(
                output_dir=str(output_dir),
                fps=video_fps,
            )
            overlay_generator.generate_annotated_video(
                frame_result["frames"],
                detection_result["detections"],
                analysis=analysis,
                output_path=str(output_dir / "annotated_video.mp4"),
            )

            progress_bar.progress(95)

            # Крок 7: AI Coach аналіз
            status_text.text("🤖 AI тренер аналізує результати...")

            swimming_pose_data = biomechanics_result.get("swimming_pose") if biomechanics_result else None
            ai_advice = get_ai_coaching(
                biomechanics=biomechanics_result,
                trajectory=trajectory_result,
                splits=analysis,
                swimming_pose=swimming_pose_data,
                athlete_name=athlete_name,
            )
            analysis["ai_coaching"] = {
                "summary": ai_advice.summary,
                "strengths": ai_advice.strengths,
                "improvements": ai_advice.improvements,
                "drills": ai_advice.drills,
                "score": ai_advice.score,
                "priority": ai_advice.priority,
            }

            st.markdown(f'<div class="success-box">🤖 AI Coach: оцінка {ai_advice.score}/100</div>', unsafe_allow_html=True)

            progress_bar.progress(100)
            status_text.text("✅ Аналіз завершено!")

            # Відображаємо результати
            display_results(analysis, biomechanics_result, trajectory_result, output_dir, ai_advice)

            # Save to database
            try:
                session_id = save_analysis_to_db(
                    athlete_name=athlete_name,
                    session_type="swimming",
                    analysis={"summary": analysis.get("summary", {}), "biomechanics": biomechanics_result},
                    ai_advice=ai_advice,
                    video_path=str(video_path) if 'video_path' in dir() else ""
                )
                st.success(f"💾 Результати збережено в базу даних (сесія #{session_id})")
            except (sqlite3.Error, ValueError, OSError) as db_error:
                logger.warning("DB save error (swimming)", exc_info=True)
                st.warning(f"⚠️ Не вдалося зберегти в БД: {db_error}")

        except (ValueError, AttributeError, TypeError) as e:
            logger.warning("Swimming analysis error", exc_info=True)
            st.error(f"❌ Помилка аналізу: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error in swimming analysis")
            st.error("❌ Непередбачена помилка. Перевірте логи.")
            st.exception(e)


def display_results(analysis, biomechanics, trajectory, output_dir, ai_advice=None):
    """Відображаємо результати аналізу."""

    st.markdown("---")
    st.markdown('<div class="success-box" style="text-align: center; font-size: 1.3rem;">🎉 Аналіз успішно завершено!</div>', unsafe_allow_html=True)

    # ========================================================================
    # AI COACH SECTION (якщо є)
    # ========================================================================
    if ai_advice:
        st.markdown('<div class="section-title">🤖 AI Тренер</div>', unsafe_allow_html=True)

        # Score card
        score = ai_advice.score
        score_color = "#10b981" if score >= 70 else "#f59e0b" if score >= 50 else "#ef4444"

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(59,130,246,0.2) 0%, rgba(139,92,246,0.2) 100%);
                    border: 1px solid {score_color}; border-radius: 16px; padding: 1.5rem; margin: 1rem 0;">
            <div style="display: flex; align-items: center; gap: 2rem;">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; font-weight: 800; color: {score_color};">{score}</div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">ЗАГАЛЬНА ОЦІНКА</div>
                </div>
                <div style="flex: 1;">
                    <div style="font-size: 1.1rem; color: #fff; margin-bottom: 0.5rem;">{ai_advice.summary}</div>
                    <div style="color: #94a3b8;">Пріоритет: <strong style="color: {score_color};">{ai_advice.priority.upper()}</strong></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Strengths & Improvements
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✅ Сильні сторони")
            for s in ai_advice.strengths:
                st.markdown(f'<div class="status-success">{s}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("### ⚠️ Що покращити")
            for imp in ai_advice.improvements:
                st.markdown(f'<div class="status-warning">{imp}</div>', unsafe_allow_html=True)

        # Drills
        if ai_advice.drills:
            st.markdown("### 🏊 Рекомендовані вправи")
            for drill in ai_advice.drills:
                st.markdown(f'<div class="status-info">{drill}</div>', unsafe_allow_html=True)

        st.markdown("---")

    # Вкладки для різних результатів
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Основні метрики",
        "🏊 Гребки",
        "🔬 Біомеханіка",
        "⏱️ Спліти",
        "📹 Відео",
        "📥 Завантажити"
    ])

    with tab1:
        display_main_metrics(analysis, output_dir)

    with tab2:
        display_stroke_analysis(biomechanics, output_dir)

    with tab3:
        display_biomechanics(biomechanics, trajectory)

    with tab4:
        display_splits(analysis)

    with tab5:
        display_video(output_dir)

    with tab6:
        display_downloads(output_dir)


def display_stroke_analysis(biomechanics, output_dir):
    """Display stroke analysis results."""

    stroke_analysis = biomechanics.get("stroke_analysis") if biomechanics else None

    if not stroke_analysis or stroke_analysis.total_strokes == 0:
        st.info("🏊 Аналіз гребків недоступний. Переконайтеся що плавець добре видно на відео.")
        return

    st.subheader("🏊 Аналіз гребка")

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: #00d9ff;">{stroke_analysis.total_strokes}</div>
            <div class="metric-label">Гребків</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: #10b981;">{stroke_analysis.stroke_rate}</div>
            <div class="metric-label">Гребків/хв</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        sym_color = "#10b981" if stroke_analysis.symmetry_score >= 80 else "#f59e0b" if stroke_analysis.symmetry_score >= 60 else "#ef4444"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {sym_color};">{stroke_analysis.symmetry_score:.0f}%</div>
            <div class="metric-label">Симетрія</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        roll_color = "#10b981" if 30 <= stroke_analysis.avg_body_roll <= 50 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {roll_color};">{stroke_analysis.avg_body_roll:.1f}°</div>
            <div class="metric-label">Body Roll</div>
        </div>
        """, unsafe_allow_html=True)

    # Symmetry details
    st.markdown("### ⚖️ Симетрія рук")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
                    border-radius: 12px; padding: 1rem; text-align: center;">
            <div style="font-size: 2rem; font-weight: 700;">{stroke_analysis.left_strokes}</div>
            <div style="color: rgba(0,0,0,0.7);">Ліва рука</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
                    border-radius: 12px; padding: 1rem; text-align: center;">
            <div style="font-size: 2rem; font-weight: 700;">{stroke_analysis.right_strokes}</div>
            <div style="color: rgba(0,0,0,0.7);">Права рука</div>
        </div>
        """, unsafe_allow_html=True)

    # Body roll details
    st.markdown("### 📐 Body Roll")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ліво", f"{stroke_analysis.body_roll_left:.1f}°")
    with col2:
        st.metric("Середній", f"{stroke_analysis.avg_body_roll:.1f}°")
    with col3:
        st.metric("Право", f"{stroke_analysis.body_roll_right:.1f}°")

    # Optimal range info
    if 30 <= stroke_analysis.avg_body_roll <= 50:
        st.success("✅ Body roll в оптимальному діапазоні (30-50°)")
    elif stroke_analysis.avg_body_roll < 30:
        st.warning("⚠️ Body roll занадто малий. Рекомендовано 30-50° для ефективного гребка")
    else:
        st.warning("⚠️ Body roll занадто великий. Рекомендовано 30-50°")

    # Phase distribution
    if stroke_analysis.phases_distribution:
        st.markdown("### 🔄 Розподіл фаз гребка")
        phases_data = []
        for phase, pct in stroke_analysis.phases_distribution.items():
            if phase != "Unknown":
                phases_data.append({"Фаза": phase, "Відсоток": f"{pct:.1f}%"})
        if phases_data:
            st.table(phases_data)

    # Chart
    chart_path = output_dir / "stroke_chart.png"
    if chart_path.exists():
        st.markdown("### 📊 Графіки аналізу")
        st.image(str(chart_path), use_column_width=True)


def display_main_metrics(analysis, output_dir):
    """Display main swimming metrics."""

    st.subheader("🏊 Основные показатели")

    summary = analysis.get("summary", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Дистанция",
            f"{summary.get('total_distance_m', 0):.1f} м",
            help="Общая проплытая дистанция"
        )

    with col2:
        st.metric(
            "Время",
            f"{summary.get('total_time_s', 0):.1f} с",
            help="Общее время заплыва"
        )

    with col3:
        st.metric(
            "Средняя скорость",
            f"{summary.get('average_speed_mps', 0):.2f} м/с",
            help="Средняя скорость движения"
        )

    with col4:
        st.metric(
            "Темп на 100м",
            f"{summary.get('average_pace_per_100m', 0):.1f} с",
            help="Темп в секундах на 100 метров"
        )

    # Speed chart
    st.subheader("📈 График скорости")
    chart_path = Path(output_dir) / "reports" / "speed_chart.png"
    if chart_path.exists():
        st.image(str(chart_path), use_column_width=True)


def display_biomechanics(biomechanics, trajectory):
    """Display biomechanics and trajectory results."""

    st.subheader("🔬 Биомеханика и анализ движения")

    # Check what data is available
    has_pose = biomechanics and biomechanics.get("average_metrics", {}).get("frames_with_pose", 0) > 0
    has_trajectory = trajectory and trajectory.get("summary", {})

    if not has_pose and not has_trajectory:
        st.warning("⚠️ Данные недоступны")
        return

    # Pose-based biomechanics
    if has_pose:
        st.markdown("### 🔬 Анализ позы (MediaPipe)")
        avg_metrics = biomechanics.get("average_metrics", {})

        # Main metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            posture = avg_metrics.get("average_posture_score", 0)
            st.metric(
                "Оценка позы",
                f"{posture:.1f}/100",
                delta=f"{posture - 70:.1f}" if posture > 0 else None,
                help="Общая оценка положения тела (70+ хорошо)"
            )

        with col2:
            drag = avg_metrics.get("average_drag_coefficient", 0)
            st.metric(
                "Коэффициент сопротивления",
                f"{drag:.2f}",
                delta=f"{0.5 - drag:.2f}" if drag > 0 else None,
                delta_color="inverse",
                help="Cd: чем меньше, тем лучше (0.4-0.5 отлично)"
            )

        with col3:
            streamline = avg_metrics.get("average_streamline_score", 0)
            st.metric(
                "Обтекаемость",
                f"{streamline:.0f}%",
                delta=f"{streamline - 70:.0f}%" if streamline > 0 else None,
                help="Качество streamline позиции (70%+ хорошо)"
            )

        # Angles
        st.subheader("📐 Углы тела")
        angles = avg_metrics.get("average_angles", {})

        if angles:
            col1, col2 = st.columns(2)

            with col1:
                if "head_elevation" in angles:
                    st.write(f"**Голова:** {angles['head_elevation']:.1f}°")
                if "left_elbow" in angles and "right_elbow" in angles:
                    avg_elbow = (angles['left_elbow'] + angles['right_elbow']) / 2
                    st.write(f"**Локти (ср.):** {avg_elbow:.1f}°")

            with col2:
                if "body_streamline" in angles:
                    st.write(f"**Обтекаемость тела:** {angles['body_streamline']:.1f}°")
                if "left_knee" in angles and "right_knee" in angles:
                    avg_knee = (angles['left_knee'] + angles['right_knee']) / 2
                    st.write(f"**Колени (ср.):** {avg_knee:.1f}°")

    # Trajectory-based analysis
    if has_trajectory:
        st.markdown("---")
        st.markdown("### 📍 Анализ траектории (bbox-based)")

        traj_summary = trajectory.get("summary", {})

        col1, col2, col3 = st.columns(3)

        with col1:
            movement_score = traj_summary.get("movement_quality_score", 0)
            st.metric(
                "Качество движения",
                f"{movement_score:.1f}/100",
                delta=f"{movement_score - 70:.1f}" if movement_score > 0 else None,
                help="Общее качество движения (70+ хорошо)"
            )

        with col2:
            traj_streamline = traj_summary.get("streamline_score", 0)
            st.metric(
                "Обтекаемость (bbox)",
                f"{traj_streamline:.0f}%",
                delta=f"{traj_streamline - 70:.0f}%" if traj_streamline > 0 else None,
                help="По форме bounding box"
            )

        with col3:
            velocity_cons = traj_summary.get("velocity_consistency", 0)
            st.metric(
                "Стабильность скорости",
                f"{velocity_cons:.0f}%",
                delta=f"{velocity_cons - 70:.0f}%" if velocity_cons > 0 else None,
                help="Постоянство темпа"
            )

        # Velocity info
        vel_profile = trajectory.get("velocity_profile", {})
        if vel_profile:
            st.write(f"**Ср. скорость:** {vel_profile.get('avg_velocity', 0):.1f} пикс/с")
            st.write(f"**Макс. скорость:** {vel_profile.get('max_velocity', 0):.1f} пикс/с")

    # Recommendations
    st.markdown("---")
    st.subheader("💡 Рекомендации")

    # Show pose recommendations
    if has_pose:
        recommendations = biomechanics.get("recommendations", [])
        if recommendations:
            st.markdown("**Рекомендации (pose):**")
            for rec in recommendations:
                if "⚠️" in rec:
                    st.warning(rec)
                elif "✅" in rec:
                    st.success(rec)
                else:
                    st.info(rec)

    # Show trajectory recommendations
    if has_trajectory:
        traj_recs = trajectory.get("recommendations", [])
        if traj_recs:
            if has_pose:
                st.markdown("---")
            st.markdown("**Рекомендации (trajectory):**")
            for rec in traj_recs:
                if "⚠️" in rec:
                    st.warning(rec)
                elif "✅" in rec:
                    st.success(rec)
                else:
                    st.info(rec)


def display_splits(analysis):
    """Display split times."""

    st.subheader("⏱️ Сплит-таймы")

    splits = analysis.get("splits", [])

    if not splits:
        st.warning("⚠️ Сплиты не обнаружены")
        return

    # Splits table
    import pandas as pd

    splits_data = []
    for split in splits:
        splits_data.append({
            "Сплит": split["split_number"],
            "Время (с)": f"{split['time_seconds']:.2f}",
            "Дистанция (м)": f"{split['distance_meters']:.1f}",
            "Скорость (м/с)": f"{split['speed_mps']:.2f}",
            "Темп /100м (с)": f"{split['pace_per_100m']:.1f}",
        })

    df = pd.DataFrame(splits_data)
    st.dataframe(df, use_container_width=True)

    # Wall touches
    wall_touches = analysis.get("wall_touches", {})
    if wall_touches:
        st.write(f"**Касаний стенки:** {wall_touches.get('count', 0)}")


def display_video(output_dir):
    """Display annotated video."""

    st.subheader("🎬 Аннотированное видео")

    video_path = output_dir / "annotated_video.mp4"

    if video_path.exists():
        st.video(str(video_path))
        st.success("✅ Видео с детекцией, осями тела и метриками")
    else:
        st.warning("⚠️ Видео не найдено")


def display_downloads(output_dir):
    """Display download links."""

    st.subheader("📥 Скачать результаты")

    col1, col2 = st.columns(2)

    with col1:
        # Summary athlete
        athlete_summary = output_dir / "reports" / "summary_athlete.txt"
        if athlete_summary.exists():
            with open(athlete_summary, "rb") as f:
                st.download_button(
                    "📄 Резюме для атлета",
                    f,
                    file_name="summary_athlete.txt",
                    mime="text/plain"
                )

        # Analysis JSON
        analysis_json = output_dir / "analysis.json"
        if analysis_json.exists():
            with open(analysis_json, "rb") as f:
                st.download_button(
                    "📊 Анализ (JSON)",
                    f,
                    file_name="analysis.json",
                    mime="application/json"
                )

        # Speed chart
        speed_chart = output_dir / "reports" / "speed_chart.png"
        if speed_chart.exists():
            with open(speed_chart, "rb") as f:
                st.download_button(
                    "📈 График скорости",
                    f,
                    file_name="speed_chart.png",
                    mime="image/png"
                )

    with col2:
        # Summary coach
        coach_summary = output_dir / "reports" / "summary_coach.txt"
        if coach_summary.exists():
            with open(coach_summary, "rb") as f:
                st.download_button(
                    "📄 Резюме для тренера",
                    f,
                    file_name="summary_coach.txt",
                    mime="text/plain"
                )

        # Biomechanics JSON
        biomech_json = output_dir / "biomechanics" / "biomechanics.json"
        if biomech_json.exists():
            with open(biomech_json, "rb") as f:
                st.download_button(
                    "🔬 Биомеханика (JSON)",
                    f,
                    file_name="biomechanics.json",
                    mime="application/json"
                )

        # Annotated video
        video_path = output_dir / "annotated_video.mp4"
        if video_path.exists():
            with open(video_path, "rb") as f:
                st.download_button(
                    "🎬 Аннотированное видео",
                    f,
                    file_name="annotated_video.mp4",
                    mime="video/mp4"
                )

    # Info about output directory
    st.info(f"📁 Все файлы также сохранены в: `{output_dir}`")
