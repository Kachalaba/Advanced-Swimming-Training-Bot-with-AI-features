"""Rate limiting middleware to prevent spam and abuse."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

from i18n import t
from utils.logger import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseMiddleware):
    """Middleware that limits the rate of requests per user.
    
    Tracks user requests over a sliding time window and blocks
    users who exceed the configured rate.
    
    Attributes:
        rate: Maximum number of requests allowed
        per: Time window in seconds
        
    Example:
        >>> from aiogram import Dispatcher
        >>> dp = Dispatcher()
        >>> # Allow 10 requests per minute
        >>> dp.message.middleware(RateLimitMiddleware(rate=10, per=60))
    """

    def __init__(
        self,
        rate: int = 10,  # requests
        per: int = 60,  # seconds
    ):
        """Initialize rate limiter.
        
        Args:
            rate: Maximum number of requests in time window
            per: Time window duration in seconds
        """
        super().__init__()
        self.rate = rate
        self.per = timedelta(seconds=per)
        self._user_requests: Dict[int, list[datetime]] = defaultdict(list)
        self._warned_users: set[int] = set()

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        """Check rate limit before processing event."""
        
        # Only rate limit messages and callbacks
        if not isinstance(event, (Message, CallbackQuery)):
            return await handler(event, data)

        # Get user ID
        user_id = self._get_user_id(event)
        if user_id is None:
            return await handler(event, data)

        # Check if user is admin (admins bypass rate limiting)
        role_service = data.get("role_service")
        if role_service:
            from role_service import ROLE_ADMIN
            
            role = await role_service.get_role(user_id)
            if role == ROLE_ADMIN:
                return await handler(event, data)

        # Check rate limit
        now = datetime.now()
        
        # Cleanup old requests outside time window
        cutoff = now - self.per
        self._user_requests[user_id] = [
            ts for ts in self._user_requests[user_id] if ts > cutoff
        ]

        # Check if user exceeded rate limit
        request_count = len(self._user_requests[user_id])
        
        if request_count >= self.rate:
            # Rate limit exceeded
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "user_id": user_id,
                    "request_count": request_count,
                    "rate_limit": self.rate,
                    "window_seconds": self.per.total_seconds(),
                },
            )
            
            # Send warning only once per time window
            if user_id not in self._warned_users:
                await self._send_rate_limit_message(event)
                self._warned_users.add(user_id)
            
            return  # Block request

        # Reset warning flag if user is within limits
        if user_id in self._warned_users:
            self._warned_users.remove(user_id)

        # Record this request
        self._user_requests[user_id].append(now)

        # Process event
        return await handler(event, data)

    def _get_user_id(self, event: TelegramObject) -> int | None:
        """Extract user ID from event."""
        
        if isinstance(event, Message):
            return event.from_user.id if event.from_user else None
        
        if isinstance(event, CallbackQuery):
            return event.from_user.id if event.from_user else None
        
        return None

    async def _send_rate_limit_message(self, event: TelegramObject) -> None:
        """Send rate limit warning to user."""
        
        message_text = t(
            "error.rate_limit",
            rate=self.rate,
            seconds=int(self.per.total_seconds()),
        )
        
        if isinstance(event, Message):
            await event.answer(message_text)
        
        elif isinstance(event, CallbackQuery):
            await event.answer(message_text, show_alert=True)


class CommandRateLimitMiddleware(RateLimitMiddleware):
    """Rate limiter specifically for commands.
    
    More strict limits for commands to prevent command spam.
    
    Example:
        >>> dp.message.middleware(CommandRateLimitMiddleware(rate=5, per=60))
    """

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        """Only rate limit command messages."""
        
        # Only apply to messages that are commands
        if isinstance(event, Message):
            if not (event.text and event.text.startswith("/")):
                return await handler(event, data)
        
        return await super().__call__(handler, event, data)
