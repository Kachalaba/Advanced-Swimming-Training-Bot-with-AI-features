"""Smart contextual help system for Sprint-Bot.

Provides intelligent suggestions based on user activity and context.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from i18n import t


class ContextualHelp:
    """Provide contextual suggestions based on user state and activity."""

    @staticmethod
    async def get_suggestion(
        user_id: int,
        total_results: int = 0,
        days_since_last_result: int = 999,
        last_progress_check_days: int = 999,
        unchecked_prs: int = 0,
        has_completed_onboarding: bool = True,
    ) -> Optional[str]:
        """Get contextual suggestion for user based on their activity.
        
        Args:
            user_id: Telegram user ID
            total_results: Total number of results user has added
            days_since_last_result: Days since last result was added
            last_progress_check_days: Days since last progress check
            unchecked_prs: Number of unchecked personal records
            has_completed_onboarding: Whether user completed onboarding
            
        Returns:
            Suggestion message or None if no suggestion needed
        """
        
        # New user - no results yet
        if total_results == 0:
            return t("help.suggestion.first_result")
        
        # User has new PRs to check
        if unchecked_prs > 0:
            return t("help.suggestion.new_prs", count=unchecked_prs)
        
        # User hasn't checked progress in a while
        if last_progress_check_days > 7:
            return t("help.suggestion.check_progress", days=last_progress_check_days)
        
        # User hasn't added results recently (low activity)
        if days_since_last_result > 3:
            return t("help.suggestion.add_result", days=days_since_last_result)
        
        # User is active - encourage consistency
        if total_results >= 10 and days_since_last_result <= 1:
            return t("help.suggestion.keep_going")
        
        # No suggestion needed
        return None

    @staticmethod
    def get_motivational_message(pr_count: int = 0, improvement_percent: float = 0.0) -> str:
        """Get motivational message based on performance.
        
        Args:
            pr_count: Number of personal records this week
            improvement_percent: Percentage improvement
            
        Returns:
            Motivational message
        """
        
        if pr_count >= 3:
            return "🔥 Ви в ударі! Продовжуйте в тому ж дусі!"
        
        if improvement_percent >= 5.0:
            return f"📈 Відмінний прогрес (+{improvement_percent:.1f}%)! Ви рухаєтесь вперед!"
        
        if improvement_percent >= 2.0:
            return "💪 Хороший прогрес! Тримайте темп!"
        
        if pr_count > 0:
            return "🎯 Новий рекорд! Кожен крок наближає до мети!"
        
        return "👊 Продовжуйте тренуватись! Результати не за горами!"

    @staticmethod
    def get_quick_action_suggestion(role: str, context: dict) -> Optional[str]:
        """Get quick action suggestion based on user role and context.
        
        Args:
            role: User role (athlete, trainer, admin)
            context: Current context data
            
        Returns:
            Quick action suggestion or None
        """
        
        if role == "athlete":
            # Athlete quick actions
            if context.get("has_training_today"):
                return "📝 У вас тренування сьогодні. Не забудьте додати результати після!"
            
            if context.get("competition_soon"):
                days = context.get("days_to_competition", 0)
                return f"🏊 До змагань залишилось {days} днів. Переглянути прогрес: /progress"
        
        elif role == "trainer":
            # Trainer quick actions
            pending_count = context.get("pending_reviews", 0)
            if pending_count > 0:
                return f"👥 У вас {pending_count} нових результатів для перегляду"
            
            if context.get("team_summary_ready"):
                return "📊 Готовий тижневий звіт команди. Переглянути?"
        
        elif role == "admin":
            # Admin quick actions
            if context.get("system_alerts"):
                return "⚠️ Є системні сповіщення. Перевірте адмін-панель"
        
        return None


def format_suggestion_message(suggestion: str, add_separator: bool = True) -> str:
    """Format suggestion message with emoji and separators.
    
    Args:
        suggestion: Suggestion text
        add_separator: Whether to add visual separator
        
    Returns:
        Formatted suggestion message
    """
    
    if add_separator:
        return f"\n━━━━━━━━━━━━━━━━━\n💡 {suggestion}"
    
    return f"💡 {suggestion}"
