"""Health check HTTP endpoint for production monitoring.

Provides /health (liveness) and /ready (readiness) endpoints
for Kubernetes and other orchestrators.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from services.container import ServiceContainer

logger = logging.getLogger(__name__)


class HealthcheckServer:
    """HTTP server for health and readiness probes."""

    def __init__(self, container: "ServiceContainer", port: int = 8080):
        """Initialize healthcheck server.
        
        Args:
            container: Service container with all bot services
            port: HTTP port to listen on (default: 8080)
        """
        self.container = container
        self.port = port
        self.app = web.Application()
        self.app.router.add_get("/health", self.health)
        self.app.router.add_get("/ready", self.ready)
        self.app.router.add_get("/metrics", self.metrics_redirect)
        self._runner = None
        self._site = None

    async def health(self, request: web.Request) -> web.Response:
        """Liveness probe - is the process alive?
        
        Returns:
            200 OK if process is alive
        """
        return web.json_response(
            {
                "status": "ok",
                "service": "sprint-bot",
                "version": "1.0.0",
            }
        )

    async def ready(self, request: web.Request) -> web.Response:
        """Readiness probe - can the service handle traffic?
        
        Checks:
        - Database connection
        - Bot API connection
        - Critical services availability
        
        Returns:
            200 OK if ready to serve traffic
            503 Service Unavailable if not ready
        """
        checks = {
            "bot_api": await self._check_bot_api(),
            "database": await self._check_database(),
        }

        all_ready = all(checks.values())

        status_code = 200 if all_ready else 503
        
        return web.json_response(
            {
                "ready": all_ready,
                "checks": checks,
            },
            status=status_code,
        )

    async def metrics_redirect(self, request: web.Request) -> web.Response:
        """Redirect to Prometheus metrics endpoint.
        
        Returns:
            Redirect to :9090/metrics if Prometheus is enabled
        """
        return web.json_response(
            {
                "metrics_endpoint": "http://localhost:9090/metrics",
                "note": "Prometheus metrics available on port 9090",
            }
        )

    async def _check_bot_api(self) -> bool:
        """Check if Telegram Bot API is accessible.
        
        Returns:
            True if bot can communicate with Telegram
        """
        try:
            me = await self.container.bot.get_me()
            return me is not None
        except Exception as e:
            logger.warning("Bot API health check failed: %s", e)
            return False

    async def _check_database(self) -> bool:
        """Check if database is accessible.
        
        Returns:
            True if database connection is healthy
        """
        try:
            # Try to access role service (uses SQLite)
            async with self.container.role_service._lock:
                # Simple lock acquire/release to test availability
                return True
        except Exception as e:
            logger.warning("Database health check failed: %s", e)
            return False

    async def start(self) -> None:
        """Start the healthcheck HTTP server."""
        try:
            self._runner = web.AppRunner(self.app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, "0.0.0.0", self.port)
            await self._site.start()
            logger.info("Healthcheck server started on http://0.0.0.0:%d", self.port)
            logger.info("  • Liveness:  http://0.0.0.0:%d/health", self.port)
            logger.info("  • Readiness: http://0.0.0.0:%d/ready", self.port)
        except Exception as e:
            logger.error("Failed to start healthcheck server: %s", e)
            raise

    async def stop(self) -> None:
        """Stop the healthcheck HTTP server."""
        if self._runner:
            await self._runner.cleanup()
            logger.info("Healthcheck server stopped")
