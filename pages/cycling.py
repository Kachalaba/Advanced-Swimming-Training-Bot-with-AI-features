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
from video_analysis.cycling_analyzer import CyclingAnalyzer, CyclingAnalysis, generate_cycling_chart

logger = logging.getLogger(__name__)


@st.cache_resource
def _load_mediapipe_pose_cycling():
    """Load and cache MediaPipe Pose model for cycling (loaded once per session)."""
    import mediapipe as mp
    from video_analysis.constants import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        enable_segmentation=False,
    )


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

    with st.expander("🎯 Трекінг персони", expanded=False):
        st.markdown("Налаштування для стабільного відстеження велосипедиста у кадрі")
        col1, col2 = st.columns(2)
        with col1:
            target_person = st.selectbox(
                "🎯 Кого аналізувати?",
                ["Найбільший у кадрі (авто)", "Найближчий до центру", "Вибрати за номером"],
                key="bike_target_person"
            )
            if target_person == "Вибрати за номером":
                person_number = st.number_input("Номер (зліва направо)", min_value=1, max_value=10, value=1, key="bike_person_num")
            else:
                person_number = 1
        with col2:
            bbox_padding = st.slider("Відступ навколо bbox (%)", min_value=0, max_value=50, value=20, key="bike_bbox_pad")
            max_lost_frames = st.slider("Макс. кадрів інтерполяції", min_value=1, max_value=30, value=10, key="bike_max_lost")

    uploaded_file = st.file_uploader(
        "📹 Завантажте відео (вид збоку на тренажері або дорозі)",
        type=["mp4", "mov", "avi", "mkv"],
        key="bike_upload"
    )

    if uploaded_file:
        st.video(uploaded_file)

        if st.button("🚴 АНАЛІЗУВАТИ ВЕЛОСИПЕД", type="primary", use_container_width=True, key="bike_analyze"):
            analyze_cycling(
                uploaded_file, athlete_name, fps, bike_type,
                target_person=target_person,
                person_number=person_number,
                bbox_padding_pct=bbox_padding,
                max_lost_frames=max_lost_frames,
            )

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


def analyze_cycling(uploaded_file, athlete_name, fps, bike_type,
                    target_person="Найбільший у кадрі (авто)",
                    person_number=1, bbox_padding_pct=20, max_lost_frames=10):
    """Analyze cycling video with stable person tracking."""

    with st.spinner("🚴 Аналізуємо велосипед..."):
        output_dir = Path("streamlit_outputs") / f"cycling_{Path(uploaded_file.name).stem}"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
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

            # Step 3: Pose detection with stable IoU tracking
            status_text.text("🦴 Аналіз біомеханіки (стабільний трекінг)...")

            first_frame_info = frame_result["frames"][0]
            first_path = str(Path(first_frame_info["path"] if isinstance(first_frame_info, dict) else first_frame_info))
            first_frame = cv2.imread(first_path)
            h, w = first_frame.shape[:2]

            CYCLING_LANDMARK_MAP = {
                0: "nose", 11: "left_shoulder", 12: "right_shoulder",
                13: "left_elbow", 14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
                23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
                27: "left_ankle", 28: "right_ankle",
            }
            MIN_LANDMARK_VISIBILITY = 0.5

            pose = _load_mediapipe_pose_cycling()

            keypoints_list = []
            annotated_frames = []
            frames_with_pose = 0

            prev_keypoints = {}
            lost_frame_count = {}
            SMOOTHING_FACTOR = 0.5

            def smooth_kps(current, prev, factor=SMOOTHING_FACTOR):
                if not prev:
                    return current
                return {
                    name: (
                        prev[name][0] * factor + current[name][0] * (1 - factor),
                        prev[name][1] * factor + current[name][1] * (1 - factor),
                    ) if name in prev else current[name]
                    for name in current
                }

            def pad_bbox(bbox, pct, iw, ih):
                x1, y1, x2, y2 = bbox[:4]
                pw = (x2 - x1) * pct / 100
                ph = (y2 - y1) * pct / 100
                return [max(0, x1 - pw), max(0, y1 - ph), min(iw, x2 + pw), min(ih, y2 + ph)]

            def calc_iou(b1, b2):
                ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                return inter / ((b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter + 1e-6)

            def centroid_dist(b1, b2):
                c1 = ((b1[0]+b1[2])/2, (b1[1]+b1[3])/2)
                c2 = ((b2[0]+b2[2])/2, (b2[1]+b2[3])/2)
                return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5

            def match_tracks(curr_bboxes, prev_bboxes_dict, iou_thr=0.2):
                matches = {}
                used = set()
                for ci, cb in sorted(enumerate(curr_bboxes), key=lambda x: (x[1][0]+x[1][2])/2):
                    best_iou, best_id = 0, None
                    for tid, pb in prev_bboxes_dict.items():
                        if tid in used:
                            continue
                        iou = calc_iou(cb, pb)
                        if iou > best_iou and iou >= iou_thr:
                            best_iou, best_id = iou, tid
                    if best_id is None:
                        best_dist, max_d = float("inf"), max(w, h) * 0.3
                        for tid, pb in prev_bboxes_dict.items():
                            if tid in used:
                                continue
                            d = centroid_dist(cb, pb)
                            if d < best_dist and d < max_d:
                                best_dist, best_id = d, tid
                    if best_id is not None:
                        matches[ci] = best_id
                        used.add(best_id)
                return matches

            COLORS = [(255, 165, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
            CONNECTIONS = [
                ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"),
                ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
                ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
                ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
                ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
            ]

            def draw_skeleton(frame, kps, tid=0, faded=False, is_target=False):
                if not kps:
                    return
                color = COLORS[tid % len(COLORS)]
                if faded:
                    color = tuple(c // 2 for c in color)
                thick = 3 if is_target else 2
                for s, e in CONNECTIONS:
                    if s in kps and e in kps:
                        cv2.line(frame, (int(kps[s][0]), int(kps[s][1])),
                                 (int(kps[e][0]), int(kps[e][1])), color, thick)
                for name, (x, y) in kps.items():
                    r = 6 if is_target else 4
                    if "knee" in name or "ankle" in name or "hip" in name:
                        cv2.circle(frame, (int(x), int(y)), r + 2, (0, 0, 255), -1)
                    else:
                        cv2.circle(frame, (int(x), int(y)), r, color, -1)
                if "nose" in kps:
                    label = f"#{tid+1}" + (" [T]" if is_target else "")
                    cv2.putText(frame, label, (int(kps["nose"][0])-25, int(kps["nose"][1])-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            prev_bboxes = {}
            track_id_counter = 0
            target_track_id = None

            for i, frame_info in enumerate(frame_result["frames"]):
                frame_path = str(Path(frame_info["path"] if isinstance(frame_info, dict) else frame_info))
                frame = cv2.imread(frame_path)
                if frame is None:
                    keypoints_list.append({})
                    annotated_frames.append(None)
                    continue

                annotated = frame.copy()
                current_bboxes = []
                if i < len(detection_result["detections"]):
                    det = detection_result["detections"][i]
                    bboxes = det.get("all_boxes", [det.get("bbox")] if det.get("bbox") else [])
                    current_bboxes = sorted([b for b in bboxes if b], key=lambda b: (b[0]+b[2])/2)

                if prev_bboxes:
                    matches = match_tracks(current_bboxes, prev_bboxes)
                else:
                    matches = {idx: idx for idx in range(len(current_bboxes))}
                    track_id_counter = len(current_bboxes)

                # Select target on first frame
                if i == 0 and current_bboxes and target_track_id is None:
                    if target_person == "Найбільший у кадрі (авто)":
                        target_track_id = max(range(len(current_bboxes)),
                                              key=lambda idx: (current_bboxes[idx][2]-current_bboxes[idx][0])*(current_bboxes[idx][3]-current_bboxes[idx][1]))
                    elif target_person == "Найближчий до центру":
                        target_track_id = min(range(len(current_bboxes)),
                                              key=lambda idx: ((current_bboxes[idx][0]+current_bboxes[idx][2])/2 - w/2)**2 + ((current_bboxes[idx][1]+current_bboxes[idx][3])/2 - h/2)**2)
                    else:
                        target_track_id = min(person_number - 1, len(current_bboxes) - 1)

                new_prev = {}
                target_kps = {}

                for ci, bbox in enumerate(current_bboxes):
                    tid = matches.get(ci, track_id_counter)
                    if ci not in matches:
                        track_id_counter += 1
                    new_prev[tid] = bbox

                    pb = pad_bbox(bbox, bbox_padding_pct, w, h)
                    x1, y1, x2, y2 = int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])
                    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if crop.size == 0:
                        continue

                    results = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    kps = {}
                    if results.pose_landmarks:
                        frames_with_pose += 1
                        ch, cw = crop.shape[:2]
                        for idx, name in CYCLING_LANDMARK_MAP.items():
                            lm = results.pose_landmarks.landmark[idx]
                            if lm.visibility >= MIN_LANDMARK_VISIBILITY:
                                kps[name] = (x1 + lm.x * cw, y1 + lm.y * ch)
                            elif tid in prev_keypoints and name in prev_keypoints[tid]:
                                kps[name] = prev_keypoints[tid][name]
                        kps = smooth_kps(kps, prev_keypoints.get(tid, {}))
                        prev_keypoints[tid] = kps
                        lost_frame_count[tid] = 0
                        draw_skeleton(annotated, kps, tid, is_target=(tid == target_track_id))
                    elif tid in prev_keypoints and lost_frame_count.get(tid, 0) < max_lost_frames:
                        kps = prev_keypoints[tid]
                        lost_frame_count[tid] = lost_frame_count.get(tid, 0) + 1
                        draw_skeleton(annotated, kps, tid, faded=True, is_target=(tid == target_track_id))

                    if tid == target_track_id and kps:
                        target_kps = kps

                prev_bboxes = new_prev

                # Fallback: full frame if target lost
                if not target_kps and (target_track_id is None or lost_frame_count.get(target_track_id, 0) <= max_lost_frames):
                    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.pose_landmarks:
                        frames_with_pose += 1
                        tid = target_track_id if target_track_id is not None else 0
                        kps = {}
                        for idx, name in CYCLING_LANDMARK_MAP.items():
                            lm = results.pose_landmarks.landmark[idx]
                            if lm.visibility >= MIN_LANDMARK_VISIBILITY:
                                kps[name] = (lm.x * w, lm.y * h)
                            elif tid in prev_keypoints and name in prev_keypoints[tid]:
                                kps[name] = prev_keypoints[tid][name]
                        kps = smooth_kps(kps, prev_keypoints.get(tid, {}))
                        prev_keypoints[tid] = kps
                        target_kps = kps
                        if target_track_id is None:
                            target_track_id = 0
                        lost_frame_count[target_track_id] = 0
                        draw_skeleton(annotated, kps, target_track_id, is_target=True)
                    elif target_track_id in prev_keypoints and lost_frame_count.get(target_track_id, 0) <= max_lost_frames:
                        target_kps = prev_keypoints[target_track_id]
                        lost_frame_count[target_track_id] = lost_frame_count.get(target_track_id, 0) + 1
                        draw_skeleton(annotated, target_kps, target_track_id, faded=True, is_target=True)

                keypoints_list.append(target_kps)
                annotated_frames.append(annotated)

                if i % 20 == 0:
                    progress_bar.progress(30 + int(25 * (i / len(frame_result["frames"]))))

            # NOTE: do NOT call pose.close() — cached via @st.cache_resource
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
        except Exception:
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
