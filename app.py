"""
🏊 SPRINT AI - Професійний аналіз спортсменів
Плавання • Суходіл • AI-біомеханіка
"""

import logging
import sqlite3
import streamlit as st
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json
import sys
import cv2
from typing import Dict

logger = logging.getLogger(__name__)

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

# ============================================================================
# PAGE MODULE IMPORTS
# ============================================================================
from pages.swimming import render_swimming_tab
from pages.running import render_running_tab
from pages.cycling import render_cycling_tab
from pages.dryland import render_dryland_tab
from pages.history import render_history_tab
from pages.ai_assistant import render_ai_tab
from pages.tools import render_tools_tab


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


if __name__ == "__main__":
    main()
