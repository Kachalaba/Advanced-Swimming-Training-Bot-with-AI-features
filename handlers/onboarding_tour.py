"""Interactive onboarding tour for new users.

Guides new users through key features in 5 simple steps.
"""

from __future__ import annotations

from aiogram import F, Router, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

router = Router()


class OnboardingTour(StatesGroup):
    """FSM states for onboarding tour."""

    welcome = State()
    step_add_result = State()
    step_records = State()
    step_progress = State()
    step_complete = State()


@router.message(Command("tour"))
async def cmd_start_tour(message: types.Message, state: FSMContext):
    """Start the interactive onboarding tour."""

    await state.set_state(OnboardingTour.welcome)

    welcome_text = (
        "👋 <b>Вітаємо в Sprint-Bot!</b>\n\n"
        "Давайте швидко познайомимося з основними функціями. "
        "Це займе лише <b>2 хвилини</b>.\n"
        "━━━━━━━━━━━━━━━━━\n\n"
        "📱 Sprint-Bot допоможе вам:\n"
        "• 📊 Відслідковувати результати\n"
        "• 🏆 Бити персональні рекорди\n"
        "• 📈 Аналізувати прогрес\n"
        "• 🎯 Досягати цілей\n\n"
        "Готові почати?"
    )

    keyboard = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="🚀 Почати тур", callback_data="tour:start"
                )
            ],
            [
                types.InlineKeyboardButton(
                    text="⏭️ Пропустити", callback_data="tour:skip"
                )
            ],
        ]
    )

    await message.answer(welcome_text, reply_markup=keyboard)


@router.callback_query(F.data == "tour:start")
async def tour_step_1_add_result(callback: types.CallbackQuery, state: FSMContext):
    """Step 1: How to add results."""

    await state.set_state(OnboardingTour.step_add_result)

    step1_text = (
        "📊 <b>Крок 1 з 4: Додавання результатів</b>\n\n"
        "Це найголовніша функція! Просто надішліть команду:\n"
        "<code>/addresult</code>\n\n"
        "Бот запитає:\n"
        "1️⃣ Стиль плавання (кроль, брас, батерфляй...)\n"
        "2️⃣ Дистанцію (50м, 100м, 200м...)\n"
        "3️⃣ Ваш час\n\n"
        "✨ І все! Бот автоматично:\n"
        "• 💾 Збереже результат\n"
        "• 📊 Розрахує спліти\n"
        "• 🏆 Перевірить чи це новий рекорд\n"
        "• 📈 Покаже прогрес\n\n"
        "━━━━━━━━━━━━━━━━━\n"
        "💡 <i>Спробуйте прямо зараз!</i>"
    )

    keyboard = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="✅ Зрозуміло, далі", callback_data="tour:step2"
                )
            ],
            [types.InlineKeyboardButton(text="🔙 Назад", callback_data="tour:start")],
        ]
    )

    await callback.message.edit_text(step1_text, reply_markup=keyboard)
    await callback.answer()


@router.callback_query(F.data == "tour:step2")
async def tour_step_2_records(callback: types.CallbackQuery, state: FSMContext):
    """Step 2: Personal records."""

    await state.set_state(OnboardingTour.step_records)

    step2_text = (
        "🏆 <b>Крок 2 з 4: Персональні рекорди</b>\n\n"
        "Бот автоматично відслідковує всі ваші рекорди!\n\n"
        "Команда: <code>/records</code>\n\n"
        "Ви побачите:\n"
        "• 🥇 Найкращі результати по кожній дистанції\n"
        "• 📅 Дату встановлення рекорду\n"
        "• 📊 Порівняння з попередніми результатами\n"
        "• 📈 Графік покращень\n\n"
        "🎉 Коли ви б'єте рекорд - бот одразу сповіщає!\n\n"
        "━━━━━━━━━━━━━━━━━\n"
        "💡 <i>Рекорди - це ваша мотивація!</i>"
    )

    keyboard = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [types.InlineKeyboardButton(text="✅ Далі", callback_data="tour:step3")],
            [types.InlineKeyboardButton(text="🔙 Назад", callback_data="tour:start")],
        ]
    )

    await callback.message.edit_text(step2_text, reply_markup=keyboard)
    await callback.answer()


@router.callback_query(F.data == "tour:step3")
async def tour_step_3_progress(callback: types.CallbackQuery, state: FSMContext):
    """Step 3: Progress tracking."""

    await state.set_state(OnboardingTour.step_progress)

    step3_text = (
        "📈 <b>Крок 3 з 4: Прогрес та аналітика</b>\n\n"
        "Дізнайтесь як ви просуваєтесь!\n\n"
        "Команда: <code>/progress</code>\n\n"
        "Ви отримаєте:\n"
        "• 📊 Графіки покращень\n"
        "• 📉 Динаміку за період\n"
        "• 🎯 Порівняння з цілями\n"
        "• 💪 Зони зростання\n"
        "• 🔥 Streak (серії тренувань)\n\n"
        "📅 Можна вибрати період:\n"
        "• Тиждень / Місяць / Рік / Весь час\n\n"
        "━━━━━━━━━━━━━━━━━\n"
        "💡 <i>Бачте свій прогрес - залишайтесь мотивованими!</i>"
    )

    keyboard = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [types.InlineKeyboardButton(text="✅ Далі", callback_data="tour:step4")],
            [types.InlineKeyboardButton(text="🔙 Назад", callback_data="tour:step2")],
        ]
    )

    await callback.message.edit_text(step3_text, reply_markup=keyboard)
    await callback.answer()


@router.callback_query(F.data == "tour:step4")
async def tour_step_4_complete(callback: types.CallbackQuery, state: FSMContext):
    """Step 4: Tour complete."""

    await state.set_state(OnboardingTour.step_complete)

    complete_text = (
        "🎉 <b>Готово! Ви познайомились зі Sprint-Bot!</b>\n\n"
        "Тепер ви знаєте основи:\n"
        "✅ Як додавати результати\n"
        "✅ Де дивитись рекорди\n"
        "✅ Як відслідковувати прогрес\n\n"
        "━━━━━━━━━━━━━━━━━\n\n"
        "<b>🚀 Корисні команди:</b>\n"
        "• <code>/addresult</code> - додати результат\n"
        "• <code>/records</code> - переглянути рекорди\n"
        "• <code>/progress</code> - прогрес\n"
        "• <code>/menu</code> - головне меню\n"
        "• <code>/help</code> - повна довідка\n\n"
        "━━━━━━━━━━━━━━━━━\n\n"
        "💪 <b>Почніть з додавання вашого першого результату!</b>\n\n"
        "💡 <i>Успіхів у тренуваннях! Ми завжди поруч!</i>"
    )

    keyboard = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="🏊‍♂️ Додати результат", callback_data="quick:addresult"
                )
            ],
            [
                types.InlineKeyboardButton(
                    text="📱 Головне меню", callback_data="menu:main"
                )
            ],
        ]
    )

    await callback.message.edit_text(complete_text, reply_markup=keyboard)
    await callback.answer("🎉 Тур завершено!")
    await state.clear()


@router.callback_query(F.data == "tour:skip")
async def tour_skip(callback: types.CallbackQuery, state: FSMContext):
    """Skip the tour."""

    await callback.message.edit_text(
        "Ви завжди можете повернутись до туру командою <code>/tour</code>\n\n"
        "Для довідки використайте: <code>/help</code>"
    )
    await callback.answer("Тур пропущено")
    await state.clear()


# Quick action handlers
@router.callback_query(F.data == "quick:addresult")
async def quick_add_result(callback: types.CallbackQuery):
    """Quick action: add result."""
    await callback.message.answer(
        "🏊‍♂️ Відмінно! Використайте команду:\n<code>/addresult</code>"
    )
    await callback.answer()


@router.callback_query(F.data == "menu:main")
async def quick_menu(callback: types.CallbackQuery):
    """Quick action: open main menu."""
    await callback.message.answer("📱 Головне меню:\n<code>/menu</code>")
    await callback.answer()
