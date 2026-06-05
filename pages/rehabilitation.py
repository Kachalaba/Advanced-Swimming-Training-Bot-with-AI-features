"""Rehabilitation and kinesiotherapy analysis page."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pandas as pd
import streamlit as st

from i18n.translations import t
from video_analysis.ai_coach import get_ai_coaching
from video_analysis.analyzer_factory import get_biomechanics_visualizer, get_rehab_analyzer
from video_analysis.athlete_database import save_analysis_to_db
from video_analysis.constants import REHAB_PROTOCOLS
from video_analysis.frame_extractor import extract_frames_from_video
from video_analysis.swimmer_detector import detect_swimmer_in_frames

logger = logging.getLogger(__name__)


def render_rehabilitation_tab() -> None:
    """Render the rehabilitation analysis tab."""
    st.markdown(
        f'<div class="section-title">{t("rehab_title")}</div>',
        unsafe_allow_html=True,
    )

    with st.expander(t("rehab_settings"), expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            athlete_name = st.text_input(
                t("rehab_athlete"),
                value=t("rehab_default_athlete"),
                key="rehab_athlete",
            )
        with col2:
            protocol = st.selectbox(
                t("rehab_protocol"),
                options=list(REHAB_PROTOCOLS),
                format_func=lambda value: t(f"rehab_protocol_{value}"),
                key="rehab_protocol",
            )
        with col3:
            fps = st.select_slider(
                t("rehab_fps"),
                options=[10, 15, 20, 30],
                value=15,
                key="rehab_fps",
            )
        input_mode = st.radio(
            t("rehab_input_mode"),
            options=["upload", "live"],
            format_func=lambda value: t(f"rehab_input_{value}"),
            horizontal=True,
            key="rehab_input_mode",
        )

    st.caption(t("rehab_disclaimer"))

    if input_mode == "upload":
        st.markdown(
            f'<div class="section-title">{t("rehab_upload_title")}</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            t("common_upload_hint"),
            type=["mp4", "mov", "avi"],
            key="rehab_upload",
        )
        if uploaded_file and st.button(
            t("rehab_analyze_btn"),
            type="primary",
            use_container_width=True,
            key="rehab_analyze",
        ):
            analyze_rehabilitation(
                uploaded_file,
                athlete_name,
                protocol,
                float(fps),
            )
    else:
        render_live_rehabilitation(athlete_name, protocol, float(fps))

    with st.expander(t("rehab_features")):
        st.markdown(t("rehab_features_body"))


def analyze_rehabilitation(
    uploaded_file: Any,
    athlete_name: str,
    protocol: str,
    fps: float,
) -> None:
    """Run the complete rehabilitation video-analysis workflow."""
    with st.spinner(t("rehab_spinner")):
        output_dir = Path("streamlit_outputs") / f"rehab_{Path(uploaded_file.name).stem}"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            video_path = output_dir / uploaded_file.name
            with open(video_path, "wb") as video_file:
                video_file.write(uploaded_file.read())

            progress = st.progress(0)
            status = st.empty()
            status.text(t("status_extract_frames_short"))
            frame_result = extract_frames_from_video(
                str(video_path),
                output_dir=str(output_dir / "frames"),
                fps=fps,
            )
            frames = frame_result.get("frames", [])
            if not frames:
                raise ValueError(t("rehab_no_frames"))
            progress.progress(20)

            status.text(t("status_detect_person"))
            detection_result = detect_swimmer_in_frames(
                frames,
                output_dir=str(output_dir / "detections"),
                draw_boxes=True,
                enable_tracking=True,
            )
            progress.progress(35)

            status.text(t("status_biomech_short"))
            visualizer = get_biomechanics_visualizer(trajectory_length=30)
            keypoint_frames: List[Dict] = []
            annotated_frames = []
            frame_size = None

            for index, frame_info in enumerate(frames):
                frame_path = Path(frame_info["path"] if isinstance(frame_info, dict) else frame_info)
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    keypoint_frames.append({})
                    annotated_frames.append(None)
                    continue
                if frame_size is None:
                    frame_size = (frame.shape[1], frame.shape[0])

                bbox = None
                detections = detection_result.get("detections", [])
                if index < len(detections):
                    bbox = detections[index].get("bbox")
                annotated, pose_data = visualizer.process_frame(frame, index, bbox)
                annotated_frames.append(annotated)
                keypoint_frames.append(pose_data.get("keypoints", {}) if pose_data.get("has_pose") else {})
                if index % 20 == 0:
                    progress.progress(35 + int(30 * index / len(frames)))

            status.text(t("rehab_status_rom"))
            analyzer = get_rehab_analyzer(fps=fps)
            report = analyzer.analyze(keypoint_frames, protocol=protocol)
            progress.progress(70)

            status.text(t("rehab_status_video"))
            annotated_video_path = output_dir / "rehab_annotated.mp4"
            if frame_size:
                _write_annotated_video(
                    annotated_frames,
                    annotated_video_path,
                    fps,
                    frame_size,
                )
            progress.progress(85)

            status.text(t("status_ai_coach_short"))
            ai_advice = get_ai_coaching(
                biomechanics={
                    "average_metrics": {
                        "rehab_completion_score": report["completion_score"],
                        "rehab_symmetry_score": report["symmetry"]["score"],
                        "rehab_correct_reps": report["total_correct_reps"],
                    }
                },
                athlete_name=athlete_name,
            )
            progress.progress(95)

            display_rehabilitation_results(
                report,
                annotated_video_path if annotated_video_path.exists() else None,
                ai_advice,
            )
            _save_rehabilitation_session(
                athlete_name,
                report,
                ai_advice,
                video_path,
            )
            progress.progress(100)
            status.text(t("status_done"))
        except (ValueError, AttributeError, TypeError) as exc:
            logger.warning("Rehabilitation analysis error", exc_info=True)
            st.error(t("rehab_analysis_error", error=str(exc)))
        except Exception:
            logger.exception("Unexpected rehabilitation analysis error")
            st.error(t("rehab_unexpected_error"))


def render_live_rehabilitation(
    athlete_name: str,
    protocol: str,
    fps: float,
) -> None:
    """Render a browser-webcam WebRTC stream with rolling ROM analysis."""
    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer

        from video_analysis.live_rehab import LiveRehabProcessor
    except ImportError:
        st.error(t("rehab_live_dependency_error"))
        return

    st.markdown(
        f'<div class="section-title">{t("rehab_live_title")}</div>',
        unsafe_allow_html=True,
    )
    st.info(t("rehab_live_help"))

    analyzer = get_rehab_analyzer(fps=fps)
    visualizer = get_biomechanics_visualizer(trajectory_length=30)
    labels = {
        "collecting": t("rehab_live_collecting"),
        "rom": t("rehab_live_rom"),
        "reps": t("rehab_live_reps"),
        "asymmetry": t("rehab_live_asymmetry"),
    }
    translations = {
        "start": t("rehab_webrtc_start"),
        "stop": t("rehab_webrtc_stop"),
        "select_device": t("rehab_webrtc_select_device"),
        "media_api_not_available": t("rehab_webrtc_media_unavailable"),
        "device_ask_permission": t("rehab_webrtc_ask_permission"),
        "device_not_available": t("rehab_webrtc_device_unavailable"),
        "device_access_denied": t("rehab_webrtc_access_denied"),
    }

    context = webrtc_streamer(
        key=f"rehab-live-{protocol}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": fps},
            },
            "audio": False,
        },
        video_processor_factory=lambda: LiveRehabProcessor(
            analyzer=analyzer,
            visualizer=visualizer,
            protocol=protocol,
            labels=labels,
        ),
        async_processing=True,
        sendback_audio=False,
        video_html_attrs={
            "autoPlay": True,
            "controls": False,
            "muted": True,
        },
        translations=translations,
    )

    processor = context.video_processor
    if processor is None:
        return

    report = processor.get_report()
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            t("rehab_live_show_report"),
            use_container_width=True,
            key=f"rehab_live_report_{protocol}",
        ):
            if report:
                display_rehabilitation_results(report, None, None)
            else:
                st.warning(t("rehab_live_not_ready"))
    with col2:
        if st.button(
            t("rehab_live_save"),
            use_container_width=True,
            key=f"rehab_live_save_{protocol}",
        ):
            if report:
                ai_advice = get_ai_coaching(
                    biomechanics={
                        "average_metrics": {
                            "rehab_completion_score": report["completion_score"],
                            "rehab_symmetry_score": report["symmetry"]["score"],
                            "rehab_correct_reps": report["total_correct_reps"],
                        }
                    },
                    athlete_name=athlete_name,
                )
                _save_rehabilitation_session(
                    athlete_name,
                    report,
                    ai_advice,
                    None,
                )
            else:
                st.warning(t("rehab_live_not_ready"))


def _write_annotated_video(
    frames: List[Any],
    output_path: Path,
    fps: float,
    frame_size: Tuple[int, int],
) -> None:
    writer = None
    for codec in ("avc1", "mp4v"):
        candidate = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()
    if writer is None:
        return
    for frame in frames:
        if frame is not None:
            writer.write(frame)
    writer.release()


def display_rehabilitation_results(
    report: Dict[str, Any],
    video_path: Optional[Path],
    ai_advice: Any,
) -> None:
    """Display ROM metrics, charts, feedback, video, and coaching."""
    st.markdown(
        f'<div class="section-title">{t("rehab_results")}</div>',
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("rehab_metric_reps"), report["total_correct_reps"])
    col2.metric(t("rehab_metric_completion"), f'{report["completion_score"]:.1f}%')
    col3.metric(t("rehab_metric_symmetry"), f'{report["symmetry"]["score"]:.1f}%')
    col4.metric(
        t("rehab_metric_asymmetry"),
        f'{report["symmetry"]["asymmetry_index"]:.1f}%',
    )

    rows = []
    for joint_name, joint in report["joint_metrics"].items():
        for side in ("left", "right"):
            metrics = joint[side]
            rows.append(
                {
                    t("rehab_joint"): t(f"rehab_joint_{joint_name}"),
                    t("rehab_side"): (t("rehab_left") if side == "left" else t("rehab_right")),
                    t("rehab_min_angle"): metrics["min_angle"],
                    t("rehab_max_angle"): metrics["max_angle"],
                    t("rehab_average_angle"): metrics["average_angle"],
                    t("rehab_rom"): metrics["rom"],
                    t("rehab_target"): metrics["target_rom"],
                    t("rehab_deficit"): metrics["deficit_deg"],
                    t("rehab_correct_reps"): metrics["correct_reps"],
                }
            )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    history = report["angle_history"]
    if history["left"] or history["right"]:
        st.markdown(f"### {t('rehab_angle_chart')}")
        st.line_chart(
            pd.DataFrame(
                {
                    t("rehab_left"): pd.Series(history["left"]),
                    t("rehab_right"): pd.Series(history["right"]),
                }
            )
        )

    rep_history = report["rep_rom_history"]
    if rep_history["left"] or rep_history["right"]:
        st.markdown(f"### {t('rehab_rep_chart')}")
        st.bar_chart(
            pd.DataFrame(
                {
                    t("rehab_left"): pd.Series(rep_history["left"]),
                    t("rehab_right"): pd.Series(rep_history["right"]),
                }
            )
        )

    st.markdown(f"### {t('rehab_feedback')}")
    for feedback in report["feedback"]:
        code = feedback["code"]
        if code == "rom_deficit":
            side = t("rehab_left") if feedback["side"] == "left" else t("rehab_right")
            message = t(
                "rehab_feedback_rom_deficit",
                side=side,
                value=feedback["value"],
            )
        else:
            message = t(f"rehab_feedback_{code}", value=feedback.get("value", 0))
        st.info(message)

    if video_path and video_path.exists():
        st.markdown(f"### {t('rehab_video_title')}")
        st.video(str(video_path))
        with open(video_path, "rb") as video_file:
            st.download_button(
                t("rehab_download_video"),
                video_file,
                file_name="rehabilitation_analysis.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

    if ai_advice:
        st.markdown(f"### {t('rehab_ai_title')}")
        st.metric(t("rehab_ai_score"), ai_advice.score)
        st.write(ai_advice.summary)
        for improvement in ai_advice.improvements:
            st.warning(improvement)


def _save_rehabilitation_session(
    athlete_name: str,
    report: Dict[str, Any],
    ai_advice: Any,
    video_path: Optional[Path],
) -> None:
    try:
        session_id = save_analysis_to_db(
            athlete_name=athlete_name,
            session_type="rehab",
            analysis={"rehab_analysis": report},
            ai_advice=ai_advice,
            video_path=str(video_path) if video_path else "",
        )
        st.success(t("rehab_saved", session_id=session_id))
    except (sqlite3.Error, ValueError, OSError) as exc:
        logger.warning("DB save error (rehab)", exc_info=True)
        st.warning(t("rehab_save_error", error=str(exc)))
