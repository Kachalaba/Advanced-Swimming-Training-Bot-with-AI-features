"""
Running analysis page module.
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
from video_analysis.running_analyzer import RunningAnalysis, generate_running_chart

logger = logging.getLogger(__name__)


@st.cache_resource
def _load_mediapipe_pose():
    """Load and cache MediaPipe Pose model (loaded once per Streamlit session)."""
    import mediapipe as mp
    from video_analysis.constants import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        enable_segmentation=False,
    )


def render_running_tab():
    """Render running analysis tab."""

    st.markdown('<div class="section-title">🏃 Аналіз техніки бігу</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Налаштування", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            athlete_name = st.text_input("👤 Ім'я спортсмена", value="Спортсмен", key="run_athlete")
        with col2:
            fps = st.number_input("🎬 FPS відео", min_value=15, max_value=240, value=30, key="run_fps")
        with col3:
            run_type = st.selectbox("🏃 Тип бігу",
                ["Спринт", "Середня дистанція", "Марафон", "Трейл"],
                key="run_type")

    with st.expander("🎯 Трекінг персони", expanded=False):
        st.markdown("Налаштування для стабільного відстеження бігуна у кадрі")
        col1, col2 = st.columns(2)
        with col1:
            target_person = st.selectbox(
                "🎯 Кого аналізувати?",
                ["Найбільший у кадрі (авто)", "Найближчий до центру", "Вибрати за номером"],
                key="run_target_person"
            )
            if target_person == "Вибрати за номером":
                person_number = st.number_input("Номер бігуна (зліва направо)", min_value=1, max_value=10, value=1, key="run_person_num")
            else:
                person_number = 1
        with col2:
            bbox_padding = st.slider("Відступ навколо bbox (%)", min_value=0, max_value=50, value=20, key="run_bbox_pad")
            max_lost_frames = st.slider("Макс. кадрів інтерполяції", min_value=1, max_value=30, value=10, key="run_max_lost")

    uploaded_file = st.file_uploader(
        "📹 Завантажте відео бігу (збоку)",
        type=["mp4", "mov", "avi", "mkv"],
        key="run_upload"
    )

    if uploaded_file:
        st.video(uploaded_file)

        if st.button("🏃 АНАЛІЗУВАТИ БІГ", type="primary", use_container_width=True, key="run_analyze"):
            analyze_running(
                uploaded_file, athlete_name, fps, run_type,
                target_person=target_person,
                person_number=person_number,
                bbox_padding_pct=bbox_padding,
                max_lost_frames=max_lost_frames,
            )

    with st.expander("📊 Можливості аналізу"):
        st.markdown("""
        | Метрика | Опис |
        |---------|------|
        | **Cadence** | Кроків за хвилину (оптимум 170-190) |
        | **Knee Lift** | Підйом коліна в градусах |
        | **Forward Lean** | Нахил корпусу (оптимум 8-15°) |
        | **Arm Symmetry** | Симетрія маху руками |
        | **Vertical Oscillation** | Вертикальні коливання |
        | **Ground Contact** | Час контакту з землею |
        """)


def analyze_running(uploaded_file, athlete_name, fps, run_type,
                     target_person="Найбільший у кадрі (авто)",
                     person_number=1, bbox_padding_pct=20, max_lost_frames=10):
    """Analyze running video with stable person tracking."""

    with st.spinner("🏃 Аналізуємо біг..."):
        # Create persistent output directory
        output_dir = Path("streamlit_outputs") / f"running_{Path(uploaded_file.name).stem}"
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
            status_text.text("🎯 Детекція бігуна...")
            detection_result = detect_swimmer_in_frames(
                frame_result["frames"],
                output_dir=str(output_dir / "detections"),
                draw_boxes=True,
                enable_tracking=True,
            )
            progress_bar.progress(30)
            st.markdown('<div class="status-success">✅ Детекція завершена</div>', unsafe_allow_html=True)

            # Step 3: Pose detection with stable person tracking
            status_text.text("🦴 Аналіз біомеханіки (стабільний трекінг)...")

            first_frame_info = frame_result["frames"][0]
            first_path = str(Path(first_frame_info["path"] if isinstance(first_frame_info, dict) else first_frame_info))
            first_frame = cv2.imread(first_path)
            h, w = first_frame.shape[:2]

            # Full 33-point landmark mapping
            FULL_LANDMARK_MAP = {
                0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
                4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
                7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
                11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
                15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky",
                19: "left_index", 20: "right_index", 21: "left_thumb", 22: "right_thumb",
                23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
                27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
                31: "left_toe", 32: "right_toe",
            }

            # Minimum landmark confidence to accept a detection
            MIN_LANDMARK_VISIBILITY = 0.5

            keypoints_list = []
            annotated_frames = []
            frames_with_pose = 0
            persons_detected = set()

            import numpy as np
            from collections import deque

            # Per-keypoint history buffer for median smoothing (rejects outlier frames)
            MEDIAN_WIN = 5           # frames in median window
            kp_history = {}          # name -> deque[(x, y)]

            prev_keypoints   = {}    # track_id -> last good kps
            lost_frame_count = {}    # track_id -> consecutive lost frames

            def median_smooth(name, x, y):
                """Push (x, y) into rolling window and return median position."""
                if name not in kp_history:
                    kp_history[name] = deque(maxlen=MEDIAN_WIN)
                kp_history[name].append((x, y))
                arr = np.array(kp_history[name])
                return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))

            def pad_bbox(bbox, pad_pct, img_w, img_h):
                """Add padding around bbox for better MediaPipe detection."""
                x1, y1, x2, y2 = bbox[:4]
                bw = x2 - x1
                bh = y2 - y1
                pad_x = bw * pad_pct / 100
                pad_y = bh * pad_pct / 100
                return [
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_y),
                    min(img_w, x2 + pad_x),
                    min(img_h, y2 + pad_y),
                ]

            def bbox_area(bbox):
                """Calculate bbox area."""
                return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            def bbox_center_dist_to_frame_center(bbox, fw, fh):
                """Distance from bbox center to frame center."""
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                return ((cx - fw / 2) ** 2 + (cy - fh / 2) ** 2) ** 0.5

            def filter_landmarks_by_confidence(landmarks, landmark_map, min_vis):
                """Only keep landmarks with sufficient visibility/confidence."""
                kps = {}
                for idx, name in landmark_map.items():
                    lm = landmarks.landmark[idx]
                    if lm.visibility >= min_vis:
                        kps[name] = lm
                return kps

            # Skeleton connections for drawing
            # ---- Premium minimal skeleton drawing ----
            # Left limbs: cool cyan (BGR)
            _C_LEFT   = (220, 180, 40)   # warm gold-cyan
            # Right limbs: coral (BGR)
            _C_RIGHT  = (80,  80, 220)   # coral/red
            # Torso: silver
            _C_TORSO  = (210, 210, 210)

            _LEFT_CONN = [
                ("left_shoulder", "left_elbow"),
                ("left_elbow",    "left_wrist"),
                ("left_hip",      "left_knee"),
                ("left_knee",     "left_ankle"),
                ("left_ankle",    "left_heel"),
                ("left_ankle",    "left_toe"),
            ]
            _RIGHT_CONN = [
                ("right_shoulder", "right_elbow"),
                ("right_elbow",    "right_wrist"),
                ("right_hip",      "right_knee"),
                ("right_knee",     "right_ankle"),
                ("right_ankle",    "right_heel"),
                ("right_ankle",    "right_toe"),
            ]
            _TORSO_CONN = [
                ("left_shoulder",  "right_shoulder"),
                ("left_shoulder",  "left_hip"),
                ("right_shoulder", "right_hip"),
                ("left_hip",       "right_hip"),
            ]

            def _dim(color, alpha):
                return tuple(max(0, int(c * alpha)) for c in color)

            def draw_skeleton_smooth(frame, kps, person_idx=0, faded=False, is_target=False):
                """Draw premium minimal skeleton — thin AA lines, left/right colour coding."""
                if not kps:
                    return

                # Non-target persons drawn very faint
                alpha   = 0.85 if is_target else 0.25
                thick   = 2    if is_target else 1
                j_r     = 3    if is_target else 2   # joint radius

                groups = [
                    (_LEFT_CONN,  _dim(_C_LEFT,  alpha)),
                    (_RIGHT_CONN, _dim(_C_RIGHT, alpha)),
                    (_TORSO_CONN, _dim(_C_TORSO, alpha)),
                ]

                for connections, color in groups:
                    for a_name, b_name in connections:
                        if a_name in kps and b_name in kps:
                            pa = (int(kps[a_name][0]), int(kps[a_name][1]))
                            pb = (int(kps[b_name][0]), int(kps[b_name][1]))
                            cv2.line(frame, pa, pb, color, thick, cv2.LINE_AA)

                # Joints — small filled circles
                for name, (x, y) in kps.items():
                    pt = (int(x), int(y))
                    if "left" in name:
                        jc = _dim(_C_LEFT, alpha)
                    elif "right" in name:
                        jc = _dim(_C_RIGHT, alpha)
                    else:
                        jc = _dim(_C_TORSO, alpha)
                    cv2.circle(frame, pt, j_r, jc, -1, cv2.LINE_AA)
                    if is_target:
                        # Thin bright rim for clarity
                        rim = tuple(min(c + 60, 255) for c in jc)
                        cv2.circle(frame, pt, j_r + 1, rim, 1, cv2.LINE_AA)

                # Head dot only (no text labels)
                if "nose" in kps and is_target:
                    np_ = (int(kps["nose"][0]), int(kps["nose"][1]))
                    cv2.circle(frame, np_, 4, (240, 240, 240), -1, cv2.LINE_AA)

            # ---- Stable tracking with IoU + centroid fallback ----
            prev_bboxes = {}
            track_id_counter = 0
            target_track_id = None  # Will be set after first frame

            def calculate_iou(box1, box2):
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                return inter_area / (box1_area + box2_area - inter_area + 1e-6)

            def centroid_distance(box1, box2):
                """Euclidean distance between bbox centers."""
                c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
                c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
                return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

            def match_tracks(current_bboxes, prev_bboxes, iou_threshold=0.2):
                """Match detections to tracks using IoU with centroid distance fallback."""
                matches = {}
                used_tracks = set()
                sorted_current = sorted(enumerate(current_bboxes),
                                        key=lambda x: (x[1][0] + x[1][2]) / 2)

                for curr_idx, curr_box in sorted_current:
                    best_iou = 0
                    best_track = None

                    for track_id, prev_box in prev_bboxes.items():
                        if track_id in used_tracks:
                            continue
                        iou = calculate_iou(curr_box, prev_box)
                        if iou > best_iou and iou >= iou_threshold:
                            best_iou = iou
                            best_track = track_id

                    # Fallback: if no IoU match, use centroid distance
                    if best_track is None:
                        best_dist = float("inf")
                        max_dist = max(w, h) * 0.3  # max 30% of frame diagonal
                        for track_id, prev_box in prev_bboxes.items():
                            if track_id in used_tracks:
                                continue
                            dist = centroid_distance(curr_box, prev_box)
                            if dist < best_dist and dist < max_dist:
                                best_dist = dist
                                best_track = track_id

                    if best_track is not None:
                        matches[curr_idx] = best_track
                        used_tracks.add(best_track)

                return matches

            # ----------------------------------------------------------------
            # HIGH-QUALITY POSE DETECTION
            # Strategy:
            #   • Run MediaPipe on the FULL FRAME every frame (not crops)
            #     → landmark coords are always in absolute frame space,
            #       completely independent of YOLO bbox jitter
            #   • MediaPipe video-mode (static_image_mode=False) + smooth_landmarks=True
            #     → built-in Kalman stabilisation between sequential frames
            #   • YOLO bbox tracked with IoU → used ONLY to verify the detected
            #     pose belongs to the target person (not to crop or rescale)
            #   • Median filter over a rolling window of MEDIAN_WIN frames
            #     → rejects any single-frame outlier/misdetection
            #   • Fresh Pose instance per analysis → no state bleeding between
            #     different video uploads in the same session
            # ----------------------------------------------------------------
            import mediapipe as _mp
            pose_hq = _mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=2,          # most accurate model
                smooth_landmarks=True,       # MediaPipe built-in Kalman
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.7,
            )

            # YOLO-based target bbox tracking (IoU + centroid)
            target_bbox   = None   # current frame's confirmed YOLO bbox

            # ---- Main frame processing loop ----
            for i, frame_info in enumerate(frame_result["frames"]):
                frame_path = str(Path(frame_info["path"] if isinstance(frame_info, dict) else frame_info))
                frame = cv2.imread(frame_path)

                if frame is None:
                    keypoints_list.append([{}])
                    annotated_frames.append(None)
                    continue

                annotated_frame = frame.copy()
                current_bboxes = []

                # ---- Step A: update YOLO target bbox ----
                if i < len(detection_result["detections"]):
                    det = detection_result["detections"][i]
                    bboxes = det.get("all_boxes", [det.get("bbox")] if det.get("bbox") else [])
                    current_bboxes = sorted([b for b in bboxes if b], key=lambda b: (b[0] + b[2]) / 2)

                if current_bboxes:
                    if prev_bboxes:
                        matches = match_tracks(current_bboxes, prev_bboxes)
                    else:
                        matches = {idx: idx for idx in range(len(current_bboxes))}
                        track_id_counter = len(current_bboxes)

                    # First frame: choose which track is the target
                    if i == 0 and target_track_id is None:
                        if target_person == "Найбільший у кадрі (авто)":
                            target_track_id = max(range(len(current_bboxes)),
                                                  key=lambda k: bbox_area(current_bboxes[k]))
                        elif target_person == "Найближчий до центру":
                            target_track_id = min(range(len(current_bboxes)),
                                                  key=lambda k: bbox_center_dist_to_frame_center(
                                                      current_bboxes[k], w, h))
                        elif target_person == "Вибрати за номером":
                            target_track_id = min(person_number - 1, len(current_bboxes) - 1)
                        else:
                            target_track_id = 0

                    new_prev_bboxes = {}
                    for curr_idx, bbox in enumerate(current_bboxes):
                        track_id = matches.get(curr_idx, track_id_counter)
                        if track_id not in [m for m in matches.values()]:
                            track_id_counter += 1
                        persons_detected.add(track_id)
                        new_prev_bboxes[track_id] = bbox
                        if track_id == target_track_id:
                            target_bbox = bbox

                    prev_bboxes = new_prev_bboxes

                # ---- Step B: full-frame MediaPipe ----
                rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_result = pose_hq.process(rgb_full)

                kps = {}
                if mp_result.pose_landmarks:
                    raw_kps = {}
                    for idx, name in FULL_LANDMARK_MAP.items():
                        lm = mp_result.pose_landmarks.landmark[idx]
                        if lm.visibility >= MIN_LANDMARK_VISIBILITY:
                            raw_kps[name] = (lm.x * w, lm.y * h)

                    # ---- Step C: verify pose belongs to TARGET person ----
                    # Check if torso centroid is inside (or near) the YOLO target bbox
                    pose_ok = True
                    if target_bbox and raw_kps:
                        torso = [raw_kps[n] for n in
                                 ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
                                 if n in raw_kps]
                        if torso:
                            cx = sum(p[0] for p in torso) / len(torso)
                            cy = sum(p[1] for p in torso) / len(torso)
                            tx1, ty1, tx2, ty2 = target_bbox
                            margin = max(tx2 - tx1, ty2 - ty1) * 0.6  # generous tolerance
                            if not (tx1 - margin <= cx <= tx2 + margin and
                                    ty1 - margin <= cy <= ty2 + margin):
                                pose_ok = False  # MediaPipe detected a different person

                    # ---- Step D: median-smooth keypoints ----
                    if pose_ok and raw_kps:
                        for name, (x, y) in raw_kps.items():
                            kps[name] = median_smooth(name, x, y)
                        frames_with_pose += 1

                # ---- Step E: fallback to previous if detection lost ----
                tid = target_track_id if target_track_id is not None else 0
                is_faded = False
                if kps:
                    prev_keypoints[tid] = kps
                    lost_frame_count[tid] = 0
                elif tid in prev_keypoints:
                    lost_count = lost_frame_count.get(tid, 0) + 1
                    lost_frame_count[tid] = lost_count
                    if lost_count <= max_lost_frames:
                        kps = prev_keypoints[tid]
                        is_faded = True

                if kps:
                    draw_skeleton_smooth(annotated_frame, kps, 0, faded=is_faded, is_target=True)

                keypoints_list.append([kps] if kps else [{}])
                annotated_frames.append(annotated_frame)

                if i % 20 == 0:
                    progress_bar.progress(30 + int(25 * (i / len(frame_result["frames"]))))

            pose_hq.close()   # release fresh instance
            progress_bar.progress(55)

            num_persons = len(persons_detected) if persons_detected else 1
            st.markdown(f'<div class="status-success">✅ Виявлено {num_persons} бігун(ів) | Поза на {frames_with_pose} кадрах</div>', unsafe_allow_html=True)

            # Step 4: Running-specific analysis (multi-person)
            status_text.text("🏃 Аналіз техніки бігу...")
            from video_analysis.running_analyzer import RunningAnalyzer, MultiPersonRunningAnalysis
            analyzer = RunningAnalyzer(fps=float(fps), multi_person=True)

            # Convert to format analyzer expects
            if num_persons == 1:
                # Single person - flatten to list of dicts
                single_kps = [frame_kps[0] if frame_kps else {} for frame_kps in keypoints_list]
                running_analysis = analyzer.analyze(single_kps, fps=float(fps))
            else:
                # Multi-person
                running_analysis = analyzer.analyze(keypoints_list, fps=float(fps))
                if isinstance(running_analysis, MultiPersonRunningAnalysis):
                    # Use best runner for display, but save all
                    main_analysis = running_analysis.get_best_runner()
                    st.markdown(f'<div class="status-info">📊 Аналізуємо {running_analysis.person_count} бігунів</div>', unsafe_allow_html=True)
                    running_analysis = main_analysis if main_analysis else analyzer._empty_analysis()

            st.markdown(f'<div class="status-success">🏃 Cadence: {running_analysis.cadence:.0f} spm | Foot Strike: {running_analysis.foot_strike_type}</div>', unsafe_allow_html=True)

            progress_bar.progress(65)

            # Step 5: Generate video with skeleton
            status_text.text("🎬 Створення відео зі скелетом...")

            annotated_video_path = output_dir / "running_annotated.mp4"

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

                # Add running metrics overlay
                cv2.putText(annotated_frame, f"Cadence: {running_analysis.cadence:.0f} spm",
                           (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Foot Strike: {running_analysis.foot_strike_type}",
                           (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                video_writer.write(annotated_frame)

                if i % 20 == 0:
                    progress_bar.progress(65 + int(15 * (i / len(annotated_frames))))

            video_writer.release()
            st.markdown('<div class="status-success">🎬 Відео зі скелетом створено</div>', unsafe_allow_html=True)

            progress_bar.progress(80)

            # Step 6: Generate chart
            status_text.text("📊 Генерація графіків...")
            chart_path = output_dir / "running_chart.png"
            generate_running_chart(running_analysis, str(chart_path))

            progress_bar.progress(90)

            # Step 7: AI Coach
            status_text.text("🤖 AI тренер аналізує...")
            ai_advice = get_ai_coaching(
                biomechanics={"running": {
                    "cadence": running_analysis.cadence,
                    "knee_lift": running_analysis.avg_knee_lift,
                    "forward_lean": running_analysis.forward_lean,
                    "arm_symmetry": running_analysis.arm_symmetry,
                    "foot_strike": running_analysis.foot_strike_type,
                    "injury_risk": running_analysis.injury_risk_score
                }},
                athlete_name=athlete_name,
            )

            progress_bar.progress(100)
            status_text.text("✅ Аналіз завершено!")

            # Display results
            display_running_results(running_analysis, ai_advice, chart_path, run_type, annotated_video_path)

            # Save to database
            try:
                session_id = save_analysis_to_db(
                    athlete_name=athlete_name,
                    session_type="running",
                    analysis={
                        "cadence": running_analysis.cadence,
                        "foot_strike": running_analysis.foot_strike_type,
                        "efficiency": running_analysis.efficiency_score,
                        "injury_risk": running_analysis.injury_risk_score
                    },
                    ai_advice=ai_advice,
                    video_path=str(video_path)
                )
                st.success(f"💾 Результати збережено в базу даних (сесія #{session_id})")
            except (sqlite3.Error, ValueError, OSError) as db_error:
                logger.warning("DB save error (running)", exc_info=True)
                st.warning(f"⚠️ Не вдалося зберегти в БД: {db_error}")

        except (ValueError, AttributeError, TypeError) as e:
            logger.warning("Running analysis error", exc_info=True)
            st.error(f"❌ Помилка аналізу: {e}")
        except Exception:
            logger.exception("Unexpected error in running analysis")
            st.error("❌ Непередбачена помилка. Перевірте логи.")
            import traceback
            st.code(traceback.format_exc())


def display_running_results(analysis: RunningAnalysis, ai_advice, chart_path, run_type, annotated_video_path=None):
    """Display running analysis results."""

    st.markdown("---")
    st.markdown('<div class="success-box" style="text-align: center;">🏃 Аналіз бігу завершено!</div>', unsafe_allow_html=True)

    # Show annotated video with skeleton
    if annotated_video_path and Path(annotated_video_path).exists():
        st.markdown("### 🎬 Відео зі скелетом")
        st.video(str(annotated_video_path))

    # Main metrics row 1
    st.markdown("### 📊 Ключові метрики")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cadence_color = "#10b981" if 170 <= analysis.cadence <= 190 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {cadence_color};">{analysis.cadence:.0f}</div>
            <div class="metric-label">Cadence (spm)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: #3b82f6;">{analysis.avg_knee_lift:.0f}°</div>
            <div class="metric-label">Knee Lift</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        lean_color = "#10b981" if 8 <= analysis.forward_lean <= 15 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {lean_color};">{analysis.forward_lean:.1f}°</div>
            <div class="metric-label">Forward Lean</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        sym_color = "#10b981" if analysis.arm_symmetry >= 80 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {sym_color};">{analysis.arm_symmetry:.0f}%</div>
            <div class="metric-label">Arm Symmetry</div>
        </div>
        """, unsafe_allow_html=True)

    # NEW: Foot Strike & Injury Prevention
    st.markdown("### 🦶 Foot Strike & Травмопрофілактика")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Foot strike type with color coding
        fs_color = "#10b981" if analysis.foot_strike_type == "midfoot" else "#f59e0b" if analysis.foot_strike_type == "forefoot" else "#ef4444"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {fs_color};">{analysis.foot_strike_type.upper()}</div>
            <div class="metric-label">Foot Strike</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        over_color = "#10b981" if not analysis.overstriding_detected else "#ef4444"
        over_text = "❌ ТАК" if analysis.overstriding_detected else "✅ НІ"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {over_color};">{over_text}</div>
            <div class="metric-label">Overstriding</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        hip_color = "#10b981" if analysis.hip_drop_score >= 80 else "#f59e0b" if analysis.hip_drop_score >= 60 else "#ef4444"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {hip_color};">{analysis.hip_drop_score:.0f}</div>
            <div class="metric-label">Hip Stability</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        risk_color = "#10b981" if analysis.injury_risk_score < 30 else "#f59e0b" if analysis.injury_risk_score < 60 else "#ef4444"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {risk_color};">{analysis.injury_risk_score:.0f}</div>
            <div class="metric-label">Injury Risk</div>
        </div>
        """, unsafe_allow_html=True)

    # Contact time & efficiency
    st.markdown("### ⏱️ Contact Time & Efficiency")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ct_color = "#10b981" if analysis.avg_contact_time_ms < 250 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {ct_color};">{analysis.avg_contact_time_ms:.0f}</div>
            <div class="metric-label">Contact Time (ms)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: #8b5cf6;">{analysis.bounce_score:.0f}</div>
            <div class="metric-label">Bounce Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        eff_color = "#10b981" if analysis.efficiency_score >= 80 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {eff_color};">{analysis.efficiency_score:.0f}</div>
            <div class="metric-label">Efficiency</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        cross_color = "#10b981" if not analysis.arm_crossover_detected else "#f59e0b"
        cross_text = "❌ ТАК" if analysis.arm_crossover_detected else "✅ НІ"
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value" style="color: {cross_color};">{cross_text}</div>
            <div class="metric-label">Arm Crossover</div>
        </div>
        """, unsafe_allow_html=True)

    # Steps
    st.markdown("### 👟 Кроки")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всього кроків", analysis.total_steps)
    with col2:
        st.metric("Ліва нога", analysis.left_steps)
    with col3:
        st.metric("Права нога", analysis.right_steps)

    # Chart
    if chart_path.exists():
        st.image(str(chart_path), use_container_width=True)

    # AI advice
    if ai_advice:
        st.markdown("### 🤖 AI Тренер")
        score_color = "#10b981" if ai_advice.score >= 70 else "#f59e0b" if ai_advice.score >= 50 else "#ef4444"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(139,92,246,0.2));
                    border-radius: 12px; padding: 1rem; border: 1px solid {score_color};">
            <div style="font-size: 2rem; font-weight: bold; color: {score_color};">{ai_advice.score}/100</div>
            <div>{ai_advice.summary}</div>
        </div>
        """, unsafe_allow_html=True)
