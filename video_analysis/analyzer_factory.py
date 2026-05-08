"""
Cached factory functions for heavy analyzer objects.

Uses ``@st.cache_resource`` so Streamlit reuses the same instances
across reruns instead of recreating them (and reloading model weights)
on every user interaction.

Usage in page files:
    from video_analysis.analyzer_factory import get_stroke_analyzer
    analyzer = get_stroke_analyzer(fps=30.0)
"""

from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)


@st.cache_resource
def get_stroke_analyzer(fps: float = 30.0):
    """Return a cached StrokeAnalyzer instance."""
    from video_analysis.stroke_analyzer import StrokeAnalyzer  # lazy import
    logger.info("Creating StrokeAnalyzer (fps=%.1f)", fps)
    return StrokeAnalyzer(fps=fps)


@st.cache_resource
def get_running_analyzer(fps: float = 30.0, multi_person: bool = True):
    """Return a cached RunningAnalyzer instance."""
    from video_analysis.running_analyzer import RunningAnalyzer
    logger.info("Creating RunningAnalyzer (fps=%.1f)", fps)
    return RunningAnalyzer(fps=fps, multi_person=multi_person)


@st.cache_resource
def get_cycling_analyzer(fps: float = 30.0):
    """Return a cached CyclingAnalyzer instance."""
    from video_analysis.cycling_analyzer import CyclingAnalyzer
    logger.info("Creating CyclingAnalyzer (fps=%.1f)", fps)
    return CyclingAnalyzer(fps=fps)


@st.cache_resource
def get_biomechanics_visualizer(trajectory_length: int = 30):
    """Return a cached BiomechanicsVisualizer instance."""
    from video_analysis.biomechanics_visualizer import BiomechanicsVisualizer
    logger.info("Creating BiomechanicsVisualizer")
    return BiomechanicsVisualizer(trajectory_length=trajectory_length)


@st.cache_resource
def get_exercise_analyzer(fps: float = 30.0):
    """Return a cached ExerciseAnalyzer instance."""
    from video_analysis.exercise_analyzer import ExerciseAnalyzer
    logger.info("Creating ExerciseAnalyzer (fps=%.1f)", fps)
    return ExerciseAnalyzer(fps=fps)


@st.cache_resource
def get_biomechanics_analyzer():
    """Return a cached BiomechanicsAnalyzer instance."""
    from video_analysis.biomechanics_analyzer import BiomechanicsAnalyzer
    logger.info("Creating BiomechanicsAnalyzer")
    return BiomechanicsAnalyzer()
