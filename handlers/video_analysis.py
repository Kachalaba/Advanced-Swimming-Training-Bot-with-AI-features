"""Video analysis handler for Telegram bot."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from datetime import datetime

from aiogram import F, Router, types
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.filters import Command

from video_analysis.frame_extractor import extract_frames_from_video
from video_analysis.swimmer_detector import detect_swimmer_in_frames
from video_analysis.split_analyzer import analyze_swimming_video
from video_analysis.report_generator import ReportGenerator

logger = logging.getLogger(__name__)
router = Router()


class VideoAnalysisStates(StatesGroup):
    """FSM states for video analysis."""
    waiting_for_video = State()
    processing = State()


@router.message(Command("analyze_video"))
async def cmd_analyze_video(message: types.Message, state: FSMContext):
    """Start video analysis flow."""
    
    await state.set_state(VideoAnalysisStates.waiting_for_video)
    
    help_text = (
        "🎥 <b>Анализ видео тренировки</b>\n\n"
        "Отправь мне видео с тренировки (до 60 секунд).\n\n"
        "📊 Я проанализирую:\n"
        "• 🏊‍♂️ Скорость на разных отрезках\n"
        "• ⏱️ Сплит-таймы\n"
        "• 💪 Частоту гребков\n"
        "• 📈 График прогресса\n\n"
        "📹 Требования к видео:\n"
        "• Формат: MP4, MOV\n"
        "• Длительность: до 60 сек\n"
        "• Качество: чем выше, тем лучше\n"
        "• Ракурс: сбоку или сверху\n\n"
        "⏳ Анализ займёт 1-2 минуты."
    )
    
    await message.answer(help_text)


@router.message(VideoAnalysisStates.waiting_for_video, F.video)
async def process_video(message: types.Message, state: FSMContext):
    """Process uploaded video."""
    
    await state.set_state(VideoAnalysisStates.processing)
    
    # Send processing message
    processing_msg = await message.answer(
        "🔄 <b>Анализирую видео...</b>\n\n"
        "⏳ Это займёт 1-2 минуты.\n"
        "Пожалуйста, подожди..."
    )
    
    try:
        # Create temp directory for this analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_id = message.from_user.id
        temp_dir = Path(f"./temp/video_analysis/{user_id}_{timestamp}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Download video
        video_file = await message.bot.download(
            message.video.file_id,
            destination=temp_dir / "input_video.mp4"
        )
        
        video_path = str(temp_dir / "input_video.mp4")
        
        # Update status
        await processing_msg.edit_text(
            "🔄 <b>Анализ видео...</b>\n\n"
            "✅ Видео загружено\n"
            "🔄 Извлекаю кадры..."
        )
        
        # Step 1: Extract frames
        frames_dir = str(temp_dir / "frames")
        frames_result = await asyncio.to_thread(
            extract_frames_from_video,
            video_path,
            output_dir=frames_dir,
            fps=2,  # 2 frames per second
        )
        
        frame_paths = frames_result["frames"]
        logger.info(f"Extracted {len(frame_paths)} frames")
        
        # Update status
        await processing_msg.edit_text(
            "🔄 <b>Анализ видео...</b>\n\n"
            "✅ Видео загружено\n"
            "✅ Кадры извлечены ({count})\n"
            "🔄 Определяю пловца...".format(count=len(frame_paths))
        )
        
        # Step 2: Detect swimmer
        detections_dir = str(temp_dir / "detections")
        detection_result = await asyncio.to_thread(
            detect_swimmer_in_frames,
            frame_paths,
            output_dir=detections_dir,
            draw_boxes=True,
        )
        
        detections = detection_result["detections"]
        logger.info(f"Detected swimmer in {len(detections)} frames")
        
        # Update status
        await processing_msg.edit_text(
            "🔄 <b>Анализ видео...</b>\n\n"
            "✅ Видео загружено\n"
            "✅ Кадры извлечены ({frames})\n"
            "✅ Пловец определён\n"
            "🔄 Анализирую технику...".format(frames=len(frame_paths))
        )
        
        # Step 3: Analyze splits
        analysis_path = str(temp_dir / "analysis.json")
        analysis = await asyncio.to_thread(
            analyze_swimming_video,
            detections,
            pool_length=25.0,
            fps=2.0,
            output_path=analysis_path,
        )
        
        logger.info(f"Analysis complete: {analysis['summary']}")
        
        # Update status
        await processing_msg.edit_text(
            "🔄 <b>Анализ видео...</b>\n\n"
            "✅ Видео загружено\n"
            "✅ Кадры извлечены ({frames})\n"
            "✅ Пловец определён\n"
            "✅ Техника проанализирована\n"
            "🔄 Генерирую отчёт...".format(frames=len(frame_paths))
        )
        
        # Step 4: Generate report
        reports_dir = str(temp_dir / "reports")
        generator = ReportGenerator(output_dir=reports_dir)
        
        athlete_name = message.from_user.full_name or "Атлет"
        report_files = await asyncio.to_thread(
            generator.generate_complete_report,
            analysis,
            athlete_name=athlete_name,
        )
        
        logger.info(f"Report generated: {report_files}")
        
        # Send results
        await processing_msg.edit_text(
            "✅ <b>Анализ завершён!</b>\n\n"
            "📊 Отправляю результаты..."
        )
        
        # Send athlete summary
        with open(report_files["athlete_summary"], "r", encoding="utf-8") as f:
            athlete_text = f.read()
        
        await message.answer(athlete_text)
        
        # Send chart
        if Path(report_files["chart"]).exists():
            chart_file = types.FSInputFile(report_files["chart"])
            await message.answer_photo(
                chart_file,
                caption="📈 График твоей скорости по отрезкам"
            )
        
        # Send PDF
        if Path(report_files["pdf"]).exists():
            pdf_file = types.FSInputFile(report_files["pdf"])
            await message.answer_document(
                pdf_file,
                caption="📄 Полный отчёт с анализом"
            )
        
        # Send sample detection image
        if detection_result["detected_images"]:
            sample_image = detection_result["detected_images"][0]
            if Path(sample_image).exists():
                img_file = types.FSInputFile(sample_image)
                await message.answer_photo(
                    img_file,
                    caption="🎯 Пример детекции пловца"
                )
        
        # Clean up
        await processing_msg.delete()
        
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Clear state
        await state.clear()
        
        logger.info(f"Video analysis complete for user {user_id}")
        
    except Exception as e:
        logger.exception(f"Error processing video: {e}")
        
        await processing_msg.edit_text(
            "❌ <b>Ошибка при анализе видео</b>\n\n"
            f"Причина: {str(e)}\n\n"
            "Попробуй другое видео или повтори позже."
        )
        
        # Clean up
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        await state.clear()


@router.message(VideoAnalysisStates.waiting_for_video)
async def wrong_content_type(message: types.Message):
    """Handle wrong content type."""
    
    await message.answer(
        "⚠️ Пожалуйста, отправь <b>видео</b>.\n\n"
        "Или отмени командой /cancel"
    )


@router.message(Command("cancel"))
async def cmd_cancel(message: types.Message, state: FSMContext):
    """Cancel current operation."""
    
    current_state = await state.get_state()
    if current_state is None:
        await message.answer("Нет активных операций.")
        return
    
    await state.clear()
    await message.answer("✅ Операция отменена.")
