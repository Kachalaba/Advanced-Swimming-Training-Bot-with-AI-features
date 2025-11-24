from pathlib import Path

from video_analysis.report_generator import ReportGenerator


def test_generate_text_summary_and_chart(tmp_path: Path):
    generator = ReportGenerator(output_dir=str(tmp_path))
    analysis = {
        "summary": {
            "total_distance_m": 50,
            "total_time_s": 40.0,
            "average_speed_mps": 1.25,
            "average_pace_per_100m": 80.0,
        },
        "splits": [
            {"split_number": 1, "speed_mps": 1.3, "time_seconds": 20.0},
            {"split_number": 2, "speed_mps": 1.2, "time_seconds": 20.0},
        ],
        "stroke_rate_spm": 45,
    }

    summary = generator.generate_text_summary(analysis)

    assert summary["athlete"].startswith("🏊‍♂️")
    assert "Средняя скорость: 1.25" in summary["coach"]

    chart_path = generator.generate_speed_chart(analysis, output_path=str(tmp_path / "chart.png"))
    assert Path(chart_path).exists()
