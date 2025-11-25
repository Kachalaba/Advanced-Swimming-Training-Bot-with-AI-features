"""
🏊 SPRINT AI - Професійний аналіз спортсменів
Плавання • Суходіл • AI-біомеханіка
"""

import streamlit as st
import tempfile
import shutil
from pathlib import Path
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
    tab_swimming, tab_dryland = st.tabs([
        "🏊 ПЛАВАННЯ",
        "🏋️ СУХОДІЛ"
    ])
    
    # ========================================================================
    # TAB 1: SWIMMING
    # ========================================================================
    with tab_swimming:
        render_swimming_tab()
    
    # ========================================================================
    # TAB 2: DRYLAND
    # ========================================================================
    with tab_dryland:
        render_dryland_tab()


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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Основні метрики",
        "🔬 Біомеханіка",
        "⏱️ Спліти",
        "📹 Відео",
        "📥 Завантажити"
    ])
    
    with tab1:
        display_main_metrics(analysis, output_dir)
    
    with tab2:
        display_biomechanics(biomechanics, trajectory)
    
    with tab3:
        display_splits(analysis)
    
    with tab4:
        display_video(output_dir)
    
    with tab5:
        display_downloads(output_dir)


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
