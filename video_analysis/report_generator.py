"""Generate training reports with charts and insights."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from fpdf import FPDF
except ImportError:  # pragma: no cover - optional dependency for PDF export
    FPDF = None

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


class ReportGenerator:
    """Generate swimming analysis reports."""

    def __init__(self, output_dir: str = "./reports"):
        """Initialize report generator.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_speed_chart(
        self,
        analysis: Dict,
        output_path: str = None,
    ) -> str:
        """Generate speed progression chart.

        Args:
            analysis: Analysis results
            output_path: Output PNG path

        Returns:
            Path to saved chart
        """
        if output_path is None:
            output_path = str(self.output_dir / "speed_chart.png")

        splits = analysis.get("splits", [])
        if not splits:
            logger.warning("No splits data for chart")
            return ""

        # Extract data
        split_numbers = [s["split_number"] for s in splits]
        speeds = [s["speed_mps"] for s in splits]

        # Create chart
        plt.figure(figsize=(12, 6))

        # Speed line
        plt.plot(split_numbers, speeds, marker="o", linewidth=2, markersize=8, label="Speed")

        # Average line
        avg_speed = analysis["summary"]["average_speed_mps"]
        plt.axhline(
            y=avg_speed,
            color="r",
            linestyle="--",
            label=f"Average: {avg_speed:.2f} m/s",
        )

        # Styling
        plt.xlabel("Split Number", fontsize=12, fontweight="bold")
        plt.ylabel("Speed (m/s)", fontsize=12, fontweight="bold")
        plt.title("Swimming Speed by Split", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved speed chart: {output_path}")
        return output_path

    def generate_text_summary(
        self,
        analysis: Dict,
    ) -> Dict[str, str]:
        """Generate text summary for coach and athlete.

        Args:
            analysis: Analysis results

        Returns:
            Dict with 'coach' and 'athlete' summaries
        """
        summary = analysis.get("summary", {})
        splits = analysis.get("splits", [])
        stroke_rate = analysis.get("stroke_rate_spm")

        # Calculate performance metrics
        total_dist = summary.get("total_distance_m", 0)
        total_time = summary.get("total_time_s", 0)
        avg_speed = summary.get("average_speed_mps", 0)
        avg_pace = summary.get("average_pace_per_100m", 0)

        # Find best/worst splits
        if splits:
            best_split = max(splits, key=lambda x: x["speed_mps"])
            worst_split = min(splits, key=lambda x: x["speed_mps"])
        else:
            best_split = worst_split = None

        # Athlete summary (motivational, simple)
        athlete_summary = f"""
🏊‍♂️ ТВОИ РЕЗУЛЬТАТЫ

📊 Общее:
• Дистанция: {total_dist} м
• Время: {total_time:.2f} сек
• Средняя скорость: {avg_speed:.2f} м/с
• Темп: {avg_pace:.2f} сек/100м
{"• Частота гребков: " + str(stroke_rate) + " в минуту" if stroke_rate else ""}

🎯 Лучший отрезок:
{"• Отрезок " + str(best_split['split_number']) + ": " + str(best_split['speed_mps']) + " м/с" if best_split else "• Нет данных"}

💪 Над чем поработать:
{"• Отрезок " + str(worst_split['split_number']) + ": " + str(worst_split['speed_mps']) + " м/с" if worst_split else "• Нет данных"}
{"• Поработай над выносливостью на финише" if splits and len(splits) > 2 and splits[-1]['speed_mps'] < splits[0]['speed_mps'] else ""}

✨ Продолжай тренироваться! Прогресс виден!
        """.strip()

        # Coach summary (detailed, technical)
        coach_summary = f"""
📋 ТЕХНИЧЕСКИЙ ОТЧЁТ

📊 Основные показатели:
• Дистанция: {total_dist} м
• Общее время: {total_time:.2f} сек
• Средняя скорость: {avg_speed:.2f} м/с
• Средний темп: {avg_pace:.2f} сек/100м
{"• Частота гребков: " + str(stroke_rate) + " SPM" if stroke_rate else ""}

🔍 Детали по отрезкам:
"""

        for split in splits:
            coach_summary += f"\n• Отрезок {split['split_number']}: {split['time_seconds']}с, {split['speed_mps']} м/с"

        if best_split and worst_split:
            speed_variance = best_split["speed_mps"] - worst_split["speed_mps"]
            coach_summary += f"""

📈 Анализ:
• Лучший отрезок: #{best_split['split_number']} ({best_split['speed_mps']:.2f} м/с)
• Худший отрезок: #{worst_split['split_number']} ({worst_split['speed_mps']:.2f} м/с)
• Разброс скорости: {speed_variance:.2f} м/с

💡 Рекомендации:
"""
            if speed_variance > 0.3:
                coach_summary += "\n• Высокий разброс скорости - работать над равномерностью темпа"
            if splits and splits[-1]["speed_mps"] < avg_speed * 0.9:
                coach_summary += "\n• Падение скорости на финише - увеличить выносливость"
            if stroke_rate and stroke_rate < 40:
                coach_summary += "\n• Низкая частота гребков - работать над темпом"
            elif stroke_rate and stroke_rate > 70:
                coach_summary += "\n• Высокая частота гребков - работать над длиной гребка"

        return {
            "athlete": athlete_summary,
            "coach": coach_summary,
        }

    def generate_pdf_report(
        self,
        analysis: Dict,
        athlete_name: str = "Атлет",
        chart_path: str = None,
        output_path: str = None,
    ) -> str:
        """Generate PDF report.

        Args:
            analysis: Analysis results
            athlete_name: Athlete name
            chart_path: Path to speed chart
            output_path: Output PDF path

        Returns:
            Path to saved PDF
        """
        if output_path is None:
            output_path = str(self.output_dir / "report.pdf")

        if FPDF is None:
            raise RuntimeError("PDF generation requires the optional 'fpdf2' package. Install it to enable exports.")

        # Generate summaries
        summaries = self.generate_text_summary(analysis)

        # Create PDF (skip if font not available on macOS)
        try:
            pdf = FPDF()
            pdf.add_page()

            # Try to add Unicode font support
            # On macOS, DejaVu fonts are in different location
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS
                "/Library/Fonts/Arial Unicode.ttf",  # macOS alternative
            ]

            font_loaded = False
            for font_path in font_paths:
                if Path(font_path).exists():
                    pdf.add_font("CustomFont", "", font_path, uni=True)
                    pdf.set_font("CustomFont", "", 16)
                    font_loaded = True
                    break

            if not font_loaded:
                # Fallback to built-in font (no Cyrillic support)
                logger.warning("Unicode font not found, using built-in font (no Cyrillic)")
                pdf.set_font("Arial", "", 16)

            # Title
            pdf.cell(0, 10, f"Video Analysis Report - {athlete_name}", ln=True, align="C")
            pdf.ln(10)

            # Athlete summary (convert to ASCII if needed)
            if font_loaded:
                pdf.set_font("CustomFont", "", 12)
                for line in summaries["athlete"].split("\n"):
                    pdf.multi_cell(0, 6, line)
            else:
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(0, 6, "Report generated. See text files for details.")

            pdf.ln(10)

            # Add chart if available
            if chart_path and Path(chart_path).exists():
                pdf.image(chart_path, x=10, w=190)

            # Save
            pdf.output(output_path)
            logger.info(f"Saved PDF report: {output_path}")

            return output_path

        except Exception as e:
            logger.warning(f"Could not generate PDF: {e}")
            logger.info("Skipping PDF generation, text reports still available")
            return ""

    def generate_complete_report(
        self,
        analysis: Dict,
        athlete_name: str = "Атлет",
    ) -> Dict[str, str]:
        """Generate complete report package.

        Args:
            analysis: Analysis results
            athlete_name: Athlete name

        Returns:
            Dict with paths to all generated files
        """
        # Generate chart
        chart_path = self.generate_speed_chart(analysis)

        # Generate summaries
        summaries = self.generate_text_summary(analysis)

        # Generate PDF
        pdf_path = self.generate_pdf_report(
            analysis,
            athlete_name=athlete_name,
            chart_path=chart_path,
        )

        # Save text summaries
        athlete_summary_path = str(self.output_dir / "summary_athlete.txt")
        coach_summary_path = str(self.output_dir / "summary_coach.txt")

        with open(athlete_summary_path, "w", encoding="utf-8") as f:
            f.write(summaries["athlete"])

        with open(coach_summary_path, "w", encoding="utf-8") as f:
            f.write(summaries["coach"])

        return {
            "pdf": pdf_path,
            "chart": chart_path,
            "athlete_summary": athlete_summary_path,
            "coach_summary": coach_summary_path,
        }


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    # Load analysis
    with open("analysis.json") as f:
        analysis = json.load(f)

    # Generate report
    generator = ReportGenerator()
    report_files = generator.generate_complete_report(
        analysis,
        athlete_name="Иван Петров",
    )

    print("Report generated:")
    for key, path in report_files.items():
        print(f"  {key}: {path}")
