"""Веб-додаток для аналізу відео плавців."""

import streamlit as st
import tempfile
import shutil
from pathlib import Path
import json
import sys

# Додаємо проект до path
sys.path.insert(0, str(Path(__file__).parent))

from video_analysis.frame_extractor import extract_frames_from_video
from video_analysis.swimmer_detector import detect_swimmer_in_frames
from video_analysis.split_analyzer import analyze_swimming_video
from video_analysis.biomechanics_analyzer import analyze_biomechanics
from video_analysis.trajectory_analyzer import analyze_trajectory
from video_analysis.report_generator import ReportGenerator
from video_analysis.video_overlay import VideoOverlayGenerator

# Налаштування сторінки
st.set_page_config(
    page_title="Аналіз відео плавання",
    page_icon="🏊‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Стильний CSS
st.markdown("""
<style>
    /* Градієнт для заголовка */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    
    /* Підзаголовок */
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Карточки метрик */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Успішний блок */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(40,167,69,0.2);
    }
    
    /* Попередження */
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(255,193,7,0.2);
    }
    
    /* Інфо блок */
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid #17a2b8;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(23,162,184,0.2);
    }
    
    /* Кнопки */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
    }
    
    /* Сайдбар */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Заголовки розділів */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #495057;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit app."""
    
    # Заголовок
    st.markdown('<h1 class="main-header">🏊‍♂️ Аналіз Відео Плавання</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">✨ Професійний аналіз техніки плавання з AI • Velocity Tracking • Підводна детекція</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Бокова панель
    with st.sidebar:
        st.header("⚙️ Налаштування")
        
        athlete_name = st.text_input(
            "👤 Ім'я спортсмена",
            value="Спортсмен",
            help="Ім'я буде використано в звітах"
        )
        
        pool_length = st.slider(
            "🏊 Довжина басейну (метри)",
            min_value=25,
            max_value=50,
            value=25,
            step=5,
            help="Оберіть довжину басейну: 25м або 50м"
        )
        
        fps = st.slider(
            "🎬 Частота кадрів (FPS)",
            min_value=1.0,
            max_value=60.0,
            value=10.0,
            step=1.0,
            help="Кількість кадрів за секунду для аналізу. Рекомендовано: 10-15 FPS. Повна розкадровка: 30-60 FPS (кожен кадр відео, тривала обробка!)"
        )
        
        # Попередження для високих FPS
        if fps >= 30:
            st.markdown(f'<div class="warning-box">⚠️ FPS {fps:.0f} - повна розкадровка! Обробка займе 5-10 хвилин для 30 сек відео. Рекомендовано для фінального аналізу.</div>', unsafe_allow_html=True)
        elif fps >= 20:
            st.markdown(f'<div class="info-box">ℹ️ FPS {fps:.0f} - висока деталізація. Обробка займе 3-5 хвилин.</div>', unsafe_allow_html=True)
        
        analysis_method = st.selectbox(
            "🔬 Метод аналізу",
            options=["hybrid", "pose", "trajectory"],
            index=0,
            format_func=lambda x: {
                "hybrid": "🎯 Гібридний (поза + траєкторія)",
                "pose": "🔬 Тільки поза (MediaPipe)",
                "trajectory": "📍 Тільки траєкторія (bbox)"
            }[x],
            help="Гібридний: обидва методи. Pose: детальна біомеханіка. Trajectory: працює на всіх кадрах"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Що аналізується:")
        st.markdown("""
        - ✅ Детекція плавця (YOLO)
        - ✅ Трекінг рухів (Velocity Prediction)
        - ✅ Біомеханіка (33 точки тіла)
        - ✅ Гідродинаміка (опір)
        - ✅ Сплітай (за реальним timestamp)
        - ✅ Швидкість і темп
        - ✅ Підводна детекція 🌊
        - ✅ Рекомендації з техніки
        """)
    
    # Основний контент
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">📹 Завантажте відео</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Перетягніть файл або натисніть Browse",
            type=["mp4", "mov", "avi"],
            help="Підтримуються формати: MP4, MOV, AVI. Макс. 60 секунд."
        )
    
    with col2:
        if uploaded_file:
            st.markdown('<div class="success-box">✅ Відео завантажено!</div>', unsafe_allow_html=True)
            st.markdown("**📄 Деталі файлу:**")
            file_details = {
                "📝 Назва": uploaded_file.name,
                "💾 Розмір": f"{uploaded_file.size / (1024*1024):.2f} МБ",
                "📦 Тип": uploaded_file.type
            }
            for key, value in file_details.items():
                st.text(f"{key}: {value}")
    
    # Кнопка аналізу
    if uploaded_file:
        st.markdown("---")
        st.markdown('<h2 class="section-header">🚀 Запустити аналіз</h2>', unsafe_allow_html=True)
        
        if st.button("🏊‍♂️ Аналізувати відео", type="primary", use_container_width=True):
            analyze_video(uploaded_file, athlete_name, pool_length, fps, analysis_method)


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
            
            progress_bar.progress(100)
            status_text.text("✅ Аналіз завершено!")
            
            # Відображаємо результати
            display_results(analysis, biomechanics_result, trajectory_result, output_dir)
            
        except Exception as e:
            st.error(f"❌ Помилка при аналізі: {str(e)}")
            st.exception(e)


def display_results(analysis, biomechanics, trajectory, output_dir):
    """Відображаємо результати аналізу."""
    
    st.markdown("---")
    st.markdown('<div class="success-box" style="text-align: center; font-size: 1.3rem;">🎉 Аналіз успішно завершено!</div>', unsafe_allow_html=True)
    
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


if __name__ == "__main__":
    main()
