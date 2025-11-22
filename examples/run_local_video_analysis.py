"""Run local swimming video analysis end-to-end.

This script demonstrates how to process an mp4 locally without Telegram
or external services. It extracts frames, runs swimmer detection, computes
splits/stroke metrics, and generates charts/PDF reports.
"""

from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from video_analysis.frame_extractor import extract_frames_from_video
from video_analysis.report_generator import ReportGenerator
from video_analysis.split_analyzer import analyze_swimming_video
from video_analysis.swimmer_detector import detect_swimmer_in_frames
from video_analysis.video_overlay import VideoOverlayGenerator
from video_analysis.biomechanics_analyzer import analyze_biomechanics

LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / "bot.log"


def configure_logging() -> None:
    """Configure logging to file (INFO) and stderr (WARNING+)."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local swimming video analysis")
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input mp4 file",
    )
    parser.add_argument(
        "--output",
        default="./run_outputs",
        help="Directory to store frames/detections/reports",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second to sample from the video",
    )
    parser.add_argument(
        "--pool-length",
        type=float,
        default=25.0,
        help="Pool length in meters",
    )
    parser.add_argument(
        "--athlete",
        default="Атлет",
        help="Name to place in the generated PDF report",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_root = Path(args.output)
    frames_dir = output_root / "frames"
    detections_dir = output_root / "detections"
    reports_dir = output_root / "reports"

    logging.info("Starting analysis for %s", video_path)

    frame_result = extract_frames_from_video(
        str(video_path),
        output_dir=str(frames_dir),
        fps=max(1, int(args.fps)),
    )
    logging.info("Extracted %s frames", frame_result["count"])

    detection_result = detect_swimmer_in_frames(
        frame_result["frames"],
        output_dir=str(detections_dir),
    )
    logging.info("Detections saved to %s", detection_result["output_dir"])

    # Biomechanics and hydrodynamics analysis
    biomechanics_dir = output_root / "biomechanics"
    biomechanics_result = analyze_biomechanics(
        frame_result["frames"],
        detection_result["detections"],
        output_dir=str(biomechanics_dir),
    )
    logging.info("Biomechanics analysis saved to %s", biomechanics_dir)
    
    # Display key biomechanics metrics
    avg_metrics = biomechanics_result.get("average_metrics", {})
    if avg_metrics:
        logging.info(
            "Biomechanics: Posture=%.1f/100, Drag Cd=%.2f, Streamline=%.0f%%",
            avg_metrics.get("average_posture_score", 0),
            avg_metrics.get("average_drag_coefficient", 0),
            avg_metrics.get("average_streamline_score", 0),
        )

    analysis = analyze_swimming_video(
        detection_result["detections"],
        pool_length=args.pool_length,
        fps=max(1, int(args.fps)),
        output_path=str(output_root / "analysis.json"),
    )
    # Add biomechanics to main analysis
    analysis["biomechanics"] = biomechanics_result
    logging.info(
        "Analysis complete: %s m in %s s",
        analysis["summary"]["total_distance_m"],
        analysis["summary"]["total_time_s"],
    )

    generator = ReportGenerator(output_dir=str(reports_dir))
    report_files = generator.generate_complete_report(
        analysis,
        athlete_name=args.athlete,
    )
    logging.info("Reports generated in %s", reports_dir)

    # Use higher fps for smoother video (10 fps minimum)
    video_fps = max(10.0, float(args.fps))
    overlay_generator = VideoOverlayGenerator(
        output_dir=str(output_root),
        fps=video_fps,
    )
    annotated_video_path = overlay_generator.generate_annotated_video(
        frame_result["frames"],
        detection_result["detections"],
        analysis=analysis,
        output_path=str(output_root / "annotated_video.mp4"),
    )
    logging.info("Annotated video saved to %s", annotated_video_path)

    print("\n✅ Локальный анализ выполнен")
    print(f"Видео: {video_path}")
    print(f"Кадры: {frames_dir}")
    print(f"Детекции: {detections_dir}")
    print(f"Биомеханика: {biomechanics_dir}")
    print(f"Отчёты: {reports_dir}")
    print(f"Аннотированное видео: {annotated_video_path}")
    print(f"Логи: {LOG_PATH}")
    
    # Display biomechanics summary
    if avg_metrics and biomechanics_result.get("recommendations"):
        print("\n🔬 Биомеханика и гидродинамика:")
        print(f"  Оценка позы: {avg_metrics.get('average_posture_score', 0):.1f}/100")
        print(f"  Коэффициент сопротивления: {avg_metrics.get('average_drag_coefficient', 0):.2f}")
        print(f"  Обтекаемость: {avg_metrics.get('average_streamline_score', 0):.0f}%")
        print("\n📋 Рекомендации:")
        for rec in biomechanics_result["recommendations"]:
            print(f"  {rec}")


if __name__ == "__main__":
    main()
