"""
🏊 SPRINT AI - Професійний аналіз спортсменів
Плавання • Суходіл • AI-біомеханіка
"""

import streamlit as st
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json
import sys
import cv2
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent))

from video_analysis.frame_extractor import extract_frames_from_video
from video_analysis.swimmer_detector import detect_swimmer_in_frames
from video_analysis.split_analyzer import analyze_swimming_video
from video_analysis.biomechanics_analyzer import analyze_biomechanics
from video_analysis.trajectory_analyzer import analyze_trajectory
from video_analysis.report_generator import ReportGenerator
from video_analysis.video_overlay import VideoOverlayGenerator
from video_analysis.swimming_pose_analyzer import SwimmingPoseAnalyzer, analyze_swimming_pose
from video_analysis.ai_coach import AICoach, get_ai_coaching
from video_analysis.biomechanics_visualizer import BiomechanicsVisualizer, visualize_biomechanics
from video_analysis.exercise_analyzer import ExerciseAnalyzer, ExerciseStats, generate_exercise_chart
from video_analysis.stroke_analyzer import StrokeAnalyzer, StrokeAnalysis, generate_stroke_chart
from video_analysis.athlete_database import (
    AthleteDatabase, Athlete, TrainingSession, 
    get_database, save_analysis_to_db
)
from video_analysis.ai_chat import AIChat, generate_training_plan, text_to_speech
from video_analysis.video_tools import (
    create_side_by_side, extract_highlight, find_highlights,
    create_zoom_video, create_tracked_zoom, get_video_info
)
from video_analysis.running_analyzer import RunningAnalyzer, RunningAnalysis, generate_running_chart
from video_analysis.cycling_analyzer import CyclingAnalyzer, CyclingAnalysis, generate_cycling_chart

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="SPRINT AI • Аналіз спортсменів",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# PREMIUM CSS
# ============================================================================
st.markdown("""
<style>
    /* === PREMIUM DARK THEME === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a24;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-cyan: #06b6d4;
        --accent-green: #10b981;
        --accent-orange: #f59e0b;
        --text-primary: #ffffff;
        --text-secondary: #94a3b8;
        --border-color: #2d2d3a;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    }
    
    /* === HEADER === */
    .premium-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 1rem;
    }
    
    .logo-text {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -2px;
        margin-bottom: 0.5rem;
    }
    
    .tagline {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 400;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    
    /* === TAB NAVIGATION === */
    .tab-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-card);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: var(--text-secondary);
        font-weight: 600;
        padding: 12px 32px;
        font-size: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
        color: white !important;
    }
    
    /* === CARDS === */
    .glass-card {
        background: rgba(26, 26, 36, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-item {
        background: linear-gradient(135deg, rgba(59,130,246,0.1) 0%, rgba(139,92,246,0.1) 100%);
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-item:hover {
        transform: translateY(-4px);
        border-color: var(--accent-blue);
        box-shadow: 0 20px 40px rgba(59,130,246,0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* === BUTTONS === */
    .stButton>button {
        background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(59,130,246,0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(59,130,246,0.5);
    }
    
    /* === STATUS BOXES === */
    .status-success {
        background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(6,182,212,0.15) 100%);
        border: 1px solid var(--accent-green);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        color: var(--accent-green);
        font-weight: 500;
    }
    
    .status-info {
        background: linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(139,92,246,0.15) 100%);
        border: 1px solid var(--accent-blue);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        color: var(--accent-blue);
        font-weight: 500;
    }
    
    .status-warning {
        background: linear-gradient(135deg, rgba(245,158,11,0.15) 0%, rgba(249,115,22,0.15) 100%);
        border: 1px solid var(--accent-orange);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        color: var(--accent-orange);
        font-weight: 500;
    }
    
    /* === UPLOAD AREA === */
    .upload-zone {
        border: 2px dashed var(--border-color);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: rgba(26, 26, 36, 0.5);
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: var(--accent-blue);
        background: rgba(59,130,246,0.05);
    }
    
    /* === SECTION HEADERS === */
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-title::before {
        content: '';
        width: 4px;
        height: 24px;
        background: linear-gradient(180deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
        border-radius: 2px;
    }
    
    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: var(--bg-card);
        border-right: 1px solid var(--border-color);
    }
    
    /* === INPUTS === */
    .stTextInput>div>div>input, .stSelectbox>div>div, .stSlider {
        background: var(--bg-card) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    /* === EXPANDER === */
    .streamlit-expanderHeader {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
    }
    
    /* === HIDE STREAMLIT BRANDING === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* === LEGACY COMPAT === */
    .success-box { 
        background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(6,182,212,0.15) 100%);
        border: 1px solid var(--accent-green);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, rgba(245,158,11,0.15) 0%, rgba(249,115,22,0.15) 100%);
        border: 1px solid var(--accent-orange);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(139,92,246,0.15) 100%);
        border: 1px solid var(--accent-blue);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="premium-header">
    <div class="logo-text">⚡ SPRINT AI</div>
    <div class="tagline">Професійний аналіз спортсменів</div>
</div>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit app with tabs."""
    
    # ========================================================================
    # MAIN TABS
    # ========================================================================
    tab_swimming, tab_running, tab_cycling, tab_dryland, tab_history, tab_ai, tab_tools = st.tabs([
        "🏊 ПЛАВАННЯ",
        "🏃 БІГ",
        "🚴 ВЕЛОСИПЕД",
        "🏋️ СУХОДІЛ",
        "📊 ІСТОРІЯ",
        "🤖 AI АСИСТЕНТ",
        "🎬 ІНСТРУМЕНТИ"
    ])
    
    # ========================================================================
    # TAB 1: SWIMMING
    # ========================================================================
    with tab_swimming:
        render_swimming_tab()
    
    # ========================================================================
    # TAB 2: RUNNING
    # ========================================================================
    with tab_running:
        render_running_tab()
    
    # ========================================================================
    # TAB 3: CYCLING
    # ========================================================================
    with tab_cycling:
        render_cycling_tab()
    
    # ========================================================================
    # TAB 4: DRYLAND
    # ========================================================================
    with tab_dryland:
        render_dryland_tab()
    
    # ========================================================================
    # TAB 5: HISTORY
    # ========================================================================
    with tab_history:
        render_history_tab()
    
    # ========================================================================
    # TAB 6: AI ASSISTANT
    # ========================================================================
    with tab_ai:
        render_ai_tab()
    
    # ========================================================================
    # TAB 7: VIDEO TOOLS
    # ========================================================================
    with tab_tools:
        render_tools_tab()


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
    
    uploaded_file = st.file_uploader(
        "📹 Завантажте відео бігу (збоку)",
        type=["mp4", "mov", "avi", "mkv"],
        key="run_upload"
    )
    
    if uploaded_file:
        st.video(uploaded_file)
        
        if st.button("🏃 АНАЛІЗУВАТИ БІГ", type="primary", use_container_width=True, key="run_analyze"):
            analyze_running(uploaded_file, athlete_name, fps, run_type)
    
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


def analyze_running(uploaded_file, athlete_name, fps, run_type):
    """Analyze running video - full pipeline like dryland."""
    
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
            
            # Step 3: Multi-person pose detection with MediaPipe
            status_text.text("🦴 Аналіз біомеханіки (multi-person)...")
            import mediapipe as mp
            
            first_frame_info = frame_result["frames"][0]
            first_path = first_frame_info["path"] if isinstance(first_frame_info, dict) else first_frame_info
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
            
            # Initialize MediaPipe Pose
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            keypoints_list = []  # List of dicts for single person OR list of lists for multi
            annotated_frames = []
            frames_with_pose = 0
            persons_detected = set()
            
            # Use YOLO detections for multi-person
            for i, frame_info in enumerate(frame_result["frames"]):
                frame_path = frame_info["path"] if isinstance(frame_info, dict) else frame_info
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    keypoints_list.append([])
                    annotated_frames.append(None)
                    continue
                
                annotated_frame = frame.copy()
                frame_persons_kps = []
                
                # Get all person detections for this frame
                if i < len(detection_result["detections"]):
                    det = detection_result["detections"][i]
                    bboxes = det.get("all_boxes", [det.get("bbox")] if det.get("bbox") else [])
                    
                    for person_idx, bbox in enumerate(bboxes):
                        if bbox is None:
                            continue
                        
                        persons_detected.add(person_idx)
                        x1, y1, x2, y2 = [int(c) for c in bbox[:4]]
                        
                        # Crop and analyze person
                        person_crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                        if person_crop.size == 0:
                            continue
                        
                        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                        results = pose.process(rgb_crop)
                        
                        kps = {}
                        if results.pose_landmarks:
                            frames_with_pose += 1
                            crop_h, crop_w = person_crop.shape[:2]
                            
                            for idx, name in FULL_LANDMARK_MAP.items():
                                lm = results.pose_landmarks.landmark[idx]
                                # Convert to full frame coordinates
                                px = x1 + lm.x * crop_w
                                py = y1 + lm.y * crop_h
                                kps[name] = (px, py)
                            
                            # Draw skeleton on annotated frame
                            mp_drawing.draw_landmarks(
                                annotated_frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)],
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                            )
                            
                            # Draw person ID
                            cv2.putText(annotated_frame, f"Runner #{person_idx+1}", 
                                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        frame_persons_kps.append(kps)
                
                # If no YOLO detections, try full frame
                if not frame_persons_kps:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)
                    
                    if results.pose_landmarks:
                        frames_with_pose += 1
                        kps = {}
                        for idx, name in FULL_LANDMARK_MAP.items():
                            lm = results.pose_landmarks.landmark[idx]
                            kps[name] = (lm.x * w, lm.y * h)
                        frame_persons_kps.append(kps)
                        
                        mp_drawing.draw_landmarks(
                            annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                        )
                
                keypoints_list.append(frame_persons_kps)
                annotated_frames.append(annotated_frame)
                
                if i % 20 == 0:
                    progress_bar.progress(30 + int(25 * (i / len(frame_result["frames"]))))
            
            pose.close()
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
            
            for codec in ["avc1", "mp4v"]:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_writer = cv2.VideoWriter(str(annotated_video_path), fourcc, float(fps), (w, h))
                if video_writer.isOpened():
                    break
            
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
            except Exception as db_error:
                st.warning(f"⚠️ Не вдалося зберегти в БД: {db_error}")
                
        except Exception as e:
            st.error(f"❌ Помилка: {e}")
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
            first_path = first_frame_info["path"] if isinstance(first_frame_info, dict) else first_frame_info
            first_frame = cv2.imread(first_path)
            h, w = first_frame.shape[:2]
            
            keypoints_list = []
            annotated_frames = []
            frames_with_pose = 0
            
            for i, frame_info in enumerate(frame_result["frames"]):
                frame_path = frame_info["path"] if isinstance(frame_info, dict) else frame_info
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
            
            for codec in ["avc1", "mp4v"]:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_writer = cv2.VideoWriter(str(annotated_video_path), fourcc, float(fps), (w, h))
                if video_writer.isOpened():
                    break
            
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
            except Exception as db_error:
                st.warning(f"⚠️ Не вдалося зберегти в БД: {db_error}")
                
        except Exception as e:
            st.error(f"❌ Помилка: {e}")
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


def render_tools_tab():
    """Render video tools tab."""
    
    st.markdown('<div class="section-title">🎬 Відео інструменти</div>', unsafe_allow_html=True)
    
    tool_tab1, tool_tab2, tool_tab3 = st.tabs([
        "⚖️ Side-by-Side",
        "✂️ Highlight",
        "🔍 Zoom"
    ])
    
    # ========================================================================
    # SIDE-BY-SIDE
    # ========================================================================
    with tool_tab1:
        st.markdown("### ⚖️ Порівняння відео Side-by-Side")
        st.markdown("Завантажте два відео для порівняння техніки")
        
        col1, col2 = st.columns(2)
        
        with col1:
            video1 = st.file_uploader("📹 Відео 1", type=["mp4", "mov", "avi"], key="sbs_video1")
            label1 = st.text_input("Підпис 1", value="До", key="sbs_label1")
        
        with col2:
            video2 = st.file_uploader("📹 Відео 2", type=["mp4", "mov", "avi"], key="sbs_video2")
            label2 = st.text_input("Підпис 2", value="Після", key="sbs_label2")
        
        if video1 and video2:
            if st.button("🎬 Створити порівняння", type="primary", use_container_width=True):
                with st.spinner("Створюємо відео..."):
                    output_dir = Path("streamlit_outputs/video_tools")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save uploaded files
                    v1_path = output_dir / f"temp_v1_{video1.name}"
                    v2_path = output_dir / f"temp_v2_{video2.name}"
                    
                    with open(v1_path, "wb") as f:
                        f.write(video1.read())
                    with open(v2_path, "wb") as f:
                        f.write(video2.read())
                    
                    output_path = output_dir / "side_by_side.mp4"
                    result = create_side_by_side(
                        str(v1_path), str(v2_path), str(output_path),
                        labels=(label1, label2)
                    )
                    
                    if result:
                        st.success("✅ Відео створено!")
                        st.video(str(output_path))
                        
                        with open(output_path, "rb") as f:
                            st.download_button(
                                "📥 Завантажити",
                                f,
                                file_name="comparison.mp4",
                                mime="video/mp4"
                            )
                    else:
                        st.error("❌ Помилка створення відео")
    
    # ========================================================================
    # HIGHLIGHT EXTRACTION
    # ========================================================================
    with tool_tab2:
        st.markdown("### ✂️ Вирізати фрагмент")
        st.markdown("Виріжте важливий момент з відео")
        
        video_hl = st.file_uploader("📹 Відео", type=["mp4", "mov", "avi"], key="hl_video")
        
        if video_hl:
            output_dir = Path("streamlit_outputs/video_tools")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            temp_path = output_dir / f"temp_{video_hl.name}"
            with open(temp_path, "wb") as f:
                f.write(video_hl.read())
            
            info = get_video_info(str(temp_path))
            if info:
                st.info(f"📊 Відео: {info.duration_sec:.1f}с, {info.fps:.0f} FPS, {info.width}x{info.height}")
                
                col1, col2 = st.columns(2)
                with col1:
                    start_sec = st.number_input("⏱️ Початок (сек)", min_value=0.0, 
                                               max_value=info.duration_sec, value=0.0, step=0.5)
                with col2:
                    end_sec = st.number_input("⏱️ Кінець (сек)", min_value=0.0,
                                             max_value=info.duration_sec, value=min(5.0, info.duration_sec), step=0.5)
                
                col1, col2 = st.columns(2)
                with col1:
                    slow_factor = st.select_slider("🐢 Швидкість", 
                                                  options=[0.25, 0.5, 0.75, 1.0],
                                                  value=1.0,
                                                  format_func=lambda x: f"{x}x")
                with col2:
                    add_text = st.text_input("📝 Текст на відео", value="", key="hl_text")
                
                if st.button("✂️ Вирізати", type="primary", use_container_width=True):
                    with st.spinner("Вирізаємо фрагмент..."):
                        output_path = output_dir / "highlight.mp4"
                        result = extract_highlight(
                            str(temp_path), str(output_path),
                            start_sec, end_sec,
                            add_text=add_text if add_text else None,
                            slow_factor=slow_factor
                        )
                        
                        if result:
                            st.success(f"✅ Фрагмент вирізано ({end_sec - start_sec:.1f}с)")
                            st.video(str(output_path))
                            
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    "📥 Завантажити",
                                    f,
                                    file_name="highlight.mp4",
                                    mime="video/mp4"
                                )
    
    # ========================================================================
    # ZOOM
    # ========================================================================
    with tool_tab3:
        st.markdown("### 🔍 Zoom відео")
        st.markdown("Збільште частину відео для детального аналізу")
        
        video_zoom = st.file_uploader("📹 Відео", type=["mp4", "mov", "avi"], key="zoom_video")
        
        if video_zoom:
            output_dir = Path("streamlit_outputs/video_tools")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            temp_path = output_dir / f"temp_zoom_{video_zoom.name}"
            with open(temp_path, "wb") as f:
                f.write(video_zoom.read())
            
            info = get_video_info(str(temp_path))
            if info:
                st.info(f"📊 Відео: {info.width}x{info.height}")
                
                zoom_type = st.radio("Тип zoom", ["📍 Фіксована область", "🎯 Трекінг об'єкта"], horizontal=True)
                
                zoom_factor = st.slider("🔍 Zoom", min_value=1.5, max_value=4.0, value=2.0, step=0.5)
                
                if zoom_type == "📍 Фіксована область":
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.slider("X позиція", 0, info.width, info.width // 2)
                        w = st.slider("Ширина", 100, info.width // 2, info.width // 4)
                    with col2:
                        y = st.slider("Y позиція", 0, info.height, info.height // 2)
                        h = st.slider("Висота", 100, info.height // 2, info.height // 4)
                    
                    if st.button("🔍 Створити Zoom", type="primary", use_container_width=True):
                        with st.spinner("Створюємо zoom відео..."):
                            output_path = output_dir / "zoomed.mp4"
                            result = create_zoom_video(
                                str(temp_path), str(output_path),
                                region=(x, y, w, h),
                                zoom_factor=zoom_factor
                            )
                            
                            if result:
                                st.success("✅ Zoom відео створено!")
                                st.video(str(output_path))
                                
                                with open(output_path, "rb") as f:
                                    st.download_button("📥 Завантажити", f, 
                                                      file_name="zoomed.mp4", mime="video/mp4")
                else:
                    st.info("🎯 Трекінг автоматично слідкує за виявленим спортсменом. Спочатку проведіть аналіз відео.")
                    
                    if st.button("🔍 Створити Tracking Zoom", type="primary", use_container_width=True):
                        with st.spinner("Виявляємо та zoom..."):
                            # Quick detection for tracking
                            from video_analysis.frame_extractor import extract_frames_from_video
                            from video_analysis.swimmer_detector import detect_swimmer_in_frames
                            
                            frames_dir = output_dir / "temp_frames"
                            frame_result = extract_frames_from_video(str(temp_path), str(frames_dir), fps=10)
                            detection_result = detect_swimmer_in_frames(frame_result["frames"], str(frames_dir))
                            
                            output_path = output_dir / "tracked_zoom.mp4"
                            result = create_tracked_zoom(
                                str(temp_path), str(output_path),
                                detection_result["detections"],
                                zoom_factor=zoom_factor
                            )
                            
                            if result:
                                st.success("✅ Tracking zoom створено!")
                                st.video(str(output_path))
                                
                                with open(output_path, "rb") as f:
                                    st.download_button("📥 Завантажити", f,
                                                      file_name="tracked_zoom.mp4", mime="video/mp4")


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
        st.markdown("""
        <div style="background: rgba(59,130,246,0.1); border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            Я можу відповісти на питання про:
            <ul>
                <li>🏊 Техніку плавання (catch, pull, push, recovery, body roll)</li>
                <li>⚠️ Типові помилки та як їх виправити</li>
                <li>🏋️ Суходільні вправи для плавців</li>
                <li>💡 Рекомендації на основі вашого аналізу</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize chat in session state
        if "ai_chat" not in st.session_state:
            st.session_state.ai_chat = AIChat()
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="background: rgba(59,130,246,0.2); border-radius: 12px; padding: 0.75rem; margin: 0.5rem 0; text-align: right;">
                    <strong>Ви:</strong> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(16,185,129,0.2); border-radius: 12px; padding: 0.75rem; margin: 0.5rem 0;">
                    <strong>🤖 AI:</strong> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input("Ваше питання:", key="ai_chat_input", placeholder="Наприклад: Як покращити body roll?")
        
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
                last_response = st.session_state.chat_history[-1]["content"] if st.session_state.chat_history[-1]["role"] == "assistant" else ""
                if last_response and st.button("🔊 Озвучити", use_container_width=True):
                    try:
                        # Simplified text for TTS
                        clean_text = last_response.replace("**", "").replace("•", "").replace("#", "")
                        audio_file = text_to_speech(clean_text, "temp_speech.mp3")
                        if audio_file:
                            st.audio(audio_file)
                    except Exception as e:
                        st.warning(f"TTS недоступний: {e}")
        
        # Quick questions
        st.markdown("### ⚡ Швидкі питання")
        quick_cols = st.columns(4)
        quick_questions = [
            "Що таке catch?",
            "Типові помилки",
            "Як покращити body roll?",
            "Вправи для плечей"
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
            plan_level = st.selectbox("📊 Рівень", ["beginner", "intermediate", "advanced"], 
                                     format_func=lambda x: {"beginner": "🌱 Початківець", 
                                                           "intermediate": "📈 Середній", 
                                                           "advanced": "🏆 Просунутий"}[x],
                                     key="plan_level")
        
        with col2:
            plan_goal = st.selectbox("🎯 Мета", ["general", "speed", "endurance", "technique"],
                                    format_func=lambda x: {"general": "🎯 Загальна підготовка",
                                                          "speed": "⚡ Швидкість",
                                                          "endurance": "🏃 Витривалість",
                                                          "technique": "🎓 Техніка"}[x],
                                    key="plan_goal")
            plan_weeks = st.slider("📆 Тижнів", 1, 12, 4, key="plan_weeks")
        
        sessions_per_week = st.slider("🏊 Тренувань на тиждень", 2, 6, 4, key="plan_sessions")
        
        if st.button("📋 Згенерувати план", type="primary", use_container_width=True):
            plan = generate_training_plan(
                athlete_name=plan_name,
                level=plan_level,
                goal=plan_goal,
                sessions_per_week=sessions_per_week,
                weeks=plan_weeks
            )
            
            st.success(f"✅ План створено: {plan.notes}")
            
            # Display plan by weeks
            for week in range(1, plan_weeks + 1):
                with st.expander(f"📅 Тиждень {week}", expanded=(week == 1)):
                    week_sessions = [s for s in plan.sessions if s["week"] == week]
                    
                    for session in week_sessions:
                        type_icon = "🏊" if session["type"] == "Плавання" else "🏋️"
                        st.markdown(f"""
                        <div style="background: rgba(59,130,246,0.1); border-radius: 8px; padding: 0.75rem; margin: 0.5rem 0;">
                            <strong>{type_icon} {session['day']} - {session['type']}</strong> ({session['duration']} хв)<br>
                            <span style="color: #60a5fa;">Фокус: {session['focus']}</span><br>
                            <span style="color: #94a3b8;">{session['workout']}</span>
                        </div>
                        """, unsafe_allow_html=True)


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
            st.markdown(f"""
            <div style="background: rgba(59,130,246,0.1); border-radius: 8px; padding: 0.5rem 1rem; margin: 0.5rem 0;">
                <strong>{type_icon} {stype.capitalize()}</strong>: {sdata['count']} сесій | 
                Середня: {sdata['avg_score']}/100 | Найкраща: {sdata['best_score']}/100
            </div>
            """, unsafe_allow_html=True)
    
    # Progress chart
    st.markdown("### 📈 Прогрес AI оцінки")
    progress_data = db.get_progress(athlete.id, "ai_score")
    
    if progress_data:
        import pandas as pd
        df = pd.DataFrame(progress_data)
        df['date'] = pd.to_datetime(df['date'])
        st.line_chart(df.set_index('date')['value'])
    else:
        st.info("Недостатньо даних для графіка прогресу")
    
    # Session history
    st.markdown("### 📋 Історія сесій")
    sessions = db.get_sessions(athlete.id, limit=20)
    
    if sessions:
        for session in sessions:
            type_icon = "🏊" if session.session_type == "swimming" else "🏋️"
            score_color = "#10b981" if session.ai_score >= 70 else "#f59e0b" if session.ai_score >= 50 else "#ef4444"
            
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
                
                if st.button(f"🗑️ Видалити", key=f"del_session_{session.id}"):
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
            analyze_video(uploaded_file, athlete_name, pool_length, fps, analysis_method)
    
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
            first_path = first_frame_info["path"] if isinstance(first_frame_info, dict) else first_frame_info
            first_frame = cv2.imread(first_path)
            h, w = first_frame.shape[:2]
            
            all_angles = []
            frames_with_pose = 0
            detected_movements = []
            annotated_frames = []
            
            for i, frame_info in enumerate(frame_result["frames"]):
                frame_path = frame_info["path"] if isinstance(frame_info, dict) else frame_info
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
            
            for codec in ["avc1", "mp4v"]:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_writer = cv2.VideoWriter(str(annotated_video_path), fourcc, video_fps, (w, h))
                if video_writer.isOpened():
                    break
            
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
            except Exception as db_error:
                st.warning(f"⚠️ Не вдалося зберегти в БД: {db_error}")
            
        except Exception as e:
            st.error(f"❌ Помилка: {str(e)}")
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


def analyze_video(uploaded_file, athlete_name, pool_length, fps, analysis_method):
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
                stroke_analyzer = StrokeAnalyzer(fps=float(fps))
                
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
            report_files = generator.generate_complete_report(
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
            annotated_video_path = overlay_generator.generate_annotated_video(
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
            except Exception as db_error:
                st.warning(f"⚠️ Не вдалося зберегти в БД: {db_error}")
            
        except Exception as e:
            st.error(f"❌ Помилка при аналізі: {str(e)}")
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
        st.image(str(chart_path), use_container_width=True)


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
        st.image(str(chart_path), use_container_width=True)


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


if __name__ == "__main__":
    main()
