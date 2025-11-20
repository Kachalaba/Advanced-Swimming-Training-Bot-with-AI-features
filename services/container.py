"""Dependency Injection Container for Sprint Bot services."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aiogram import Bot

if TYPE_CHECKING:
    from backup_service import BackupService
    from chat_service import ChatService
    from notifications import NotificationService
    from role_service import RoleService
    from services.audit_service import AuditService
    from services.io_service import IOService
    from services.query_service import QueryService
    from services.stats_service import StatsService
    from services.turn_service import TurnService
    from services.user_service import UserService
    from template_service import TemplateService


@dataclass
class ServiceContainer:
    """Central container for all bot services with lifecycle management.
    
    Example:
        >>> admin_ids = (123456, 789012)
        >>> container = await ServiceContainer.create(admin_ids)
        >>> try:
        >>>     # Use services
        >>>     users = await container.role_service.list_users()
        >>> finally:
        >>>     await container.shutdown()
    """

    bot: Bot
    role_service: RoleService
    chat_service: ChatService
    template_service: TemplateService
    notification_service: NotificationService
    backup_service: BackupService
    stats_service: StatsService
    query_service: QueryService
    io_service: IOService
    audit_service: AuditService
    user_service: UserService
    turn_service: TurnService

    @classmethod
    async def create(
        cls, 
        admin_ids: tuple[int, ...],
        db_path: str | Path | None = None,
    ) -> "ServiceContainer":
        """Create and initialize all services.
        
        Args:
            admin_ids: Telegram IDs of administrators
            db_path: Optional custom database path
            
        Returns:
            Initialized ServiceContainer
            
        Raises:
            RuntimeError: If any service fails to initialize
        """
        from backup_service import BackupService
        from chat_service import DB_PATH, ChatService
        from notifications import NotificationService
        from role_service import RoleService
        from services import get_bot
        from services.audit_service import AuditService
        from services.io_service import IOService
        from services.query_service import QueryService
        from services.stats_service import StatsService
        from services.turn_service import TurnService
        from services.user_service import UserService
        from template_service import TemplateService

        # Core services
        bot = get_bot()
        turn_service = TurnService()
        
        # Initialize role service first (many depend on it)
        role_service = RoleService()
        await role_service.init(admin_ids=admin_ids)
        
        # Initialize chat service
        chat_service = ChatService(db_path or DB_PATH)
        await chat_service.init()
        
        # Initialize audit service (needed for template and io services)
        audit_service = AuditService()
        await audit_service.init()
        
        # Services that depend on audit
        template_service = TemplateService(audit_service=audit_service)
        await template_service.init()
        
        io_service = IOService(audit_service=audit_service)
        await io_service.init()
        
        # Initialize remaining services
        user_service = UserService()
        await user_service.init()
        
        query_service = QueryService()
        await query_service.init()
        
        stats_service = StatsService()
        await stats_service.init()
        
        # Notification service (no async init)
        notification_service = NotificationService(bot=bot)
        
        # Backup service configuration
        backup_db_path = Path(os.getenv("CHAT_DB_PATH", DB_PATH))
        backup_service = BackupService(
            bot=bot,
            db_path=backup_db_path,
            bucket_name=os.getenv("S3_BACKUP_BUCKET", ""),
            backup_prefix=os.getenv("S3_BACKUP_PREFIX", "sprint-bot/backups/"),
            interval=cls._parse_backup_interval(),
            admin_chat_ids=admin_ids,
            storage_class=os.getenv("S3_STORAGE_CLASS") or None,
            endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,
        )

        return cls(
            bot=bot,
            role_service=role_service,
            chat_service=chat_service,
            template_service=template_service,
            notification_service=notification_service,
            backup_service=backup_service,
            stats_service=stats_service,
            query_service=query_service,
            io_service=io_service,
            audit_service=audit_service,
            user_service=user_service,
            turn_service=turn_service,
        )

    async def shutdown(self) -> None:
        """Gracefully shutdown all services in reverse dependency order."""
        
        # Stop services with active tasks first
        await self.notification_service.shutdown()
        await self.backup_service.shutdown()
        
        # No cleanup needed for other services currently,
        # but this provides a hook for future cleanup logic
        
    def as_dict(self) -> dict:
        """Export services as dictionary for dispatcher injection.
        
        Returns:
            Dictionary mapping service names to instances
        """
        return {
            "bot": self.bot,
            "role_service": self.role_service,
            "chat_service": self.chat_service,
            "template_service": self.template_service,
            "notifications": self.notification_service,
            "backup_service": self.backup_service,
            "stats_service": self.stats_service,
            "query_service": self.query_service,
            "io_service": self.io_service,
            "audit_service": self.audit_service,
            "user_service": self.user_service,
            "turn_service": self.turn_service,
        }

    @staticmethod
    def _parse_backup_interval():
        """Parse backup interval from environment."""
        from datetime import timedelta
        
        default_hours = 6.0
        value = os.getenv("BACKUP_INTERVAL_HOURS")
        
        if not value:
            return timedelta(hours=default_hours)
        
        try:
            hours = float(value)
            if hours <= 0:
                return timedelta(hours=default_hours)
            return timedelta(hours=hours)
        except ValueError:
            return timedelta(hours=default_hours)
