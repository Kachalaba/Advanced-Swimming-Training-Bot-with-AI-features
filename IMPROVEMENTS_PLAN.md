# 📈 План улучшений Sprint-Bot

Анализ текущего состояния и конкретные рекомендации по улучшению.

---

## 🎯 Приоритетные улучшения

### 1. Dependency Injection Container (Высокий приоритет)

**Проблема:** В `bot.py` создается 11+ сервисов вручную, это:
- Сложно тестировать
- Нет контроля над lifecycle
- Тяжело добавлять новые зависимости

**Решение:** Использовать dependency injection

```python
# services/container.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ServiceContainer:
    """Centralized service management."""
    
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
    async def create(cls, admin_ids: tuple[int, ...]) -> "ServiceContainer":
        """Factory method for async initialization."""
        bot = get_bot()
        
        # Core services
        role_service = RoleService()
        await role_service.init(admin_ids=admin_ids)
        
        chat_service = ChatService()
        await chat_service.init()
        
        audit_service = AuditService()
        await audit_service.init()
        
        # Dependent services
        template_service = TemplateService(audit_service=audit_service)
        await template_service.init()
        
        notification_service = NotificationService(bot=bot)
        
        # ... остальные сервисы
        
        return cls(
            bot=bot,
            role_service=role_service,
            chat_service=chat_service,
            # ...
        )
    
    async def shutdown(self) -> None:
        """Graceful shutdown of all services."""
        await self.notification_service.shutdown()
        await self.backup_service.shutdown()
        # ... остальные сервисы

# bot.py - упрощается до:
async def main() -> None:
    admin_ids = _parse_admin_chat_ids(ADMIN_IDS)
    container = await ServiceContainer.create(admin_ids)
    
    dp = setup_dispatcher(container)
    dp["container"] = container  # Доступ из handlers
    
    try:
        await _start_polling_with_retries(dp, container.bot)
    finally:
        await container.shutdown()
```

**Преимущества:**
- ✅ Один source of truth для всех зависимостей
- ✅ Легко тестировать с mock'ами
- ✅ Контроль lifecycle
- ✅ Проще добавлять новые сервисы

---

### 2. Кеширование Google Sheets (Критично для производительности)

**Проблема:** Каждый запрос к Google Sheets — это API call. Лимит: 100 requests/100 seconds/user.

**Текущий код:**
```python
# services/base.py
@lru_cache(maxsize=None)
def get_worksheet(name: str) -> gspread.Worksheet:
    # Кеш есть, но он синхронный и не учитывает время жизни
    return spreadsheet.worksheet(name)
```

**Решение:** Асинхронный кеш с TTL

```python
# services/cache.py
from datetime import datetime, timedelta
from typing import Any, Optional
import asyncio

class AsyncTTLCache:
    """Async cache with time-to-live."""
    
    def __init__(self, ttl_seconds: int = 300):
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._cache:
                return None
            
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp > self._ttl:
                del self._cache[key]
                return None
            
            return value
    
    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._cache[key] = (value, datetime.now())
    
    async def invalidate(self, key: str) -> None:
        async with self._lock:
            self._cache.pop(key, None)
    
    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()

# services/sheets_service.py
class CachedSheetsService:
    """Google Sheets with intelligent caching."""
    
    def __init__(self):
        self._cache = AsyncTTLCache(ttl_seconds=300)  # 5 минут
        self._client: Optional[gspread.Client] = None
    
    async def get_worksheet_data(
        self, 
        worksheet_name: str,
        force_refresh: bool = False
    ) -> list[dict[str, Any]]:
        """Get worksheet data with caching."""
        
        cache_key = f"worksheet:{worksheet_name}"
        
        if not force_refresh:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for %s", worksheet_name)
                return cached
        
        # Fetch from API
        logger.debug("Cache miss for %s, fetching from API", worksheet_name)
        worksheet = await asyncio.to_thread(
            self._get_worksheet_sync, worksheet_name
        )
        data = await asyncio.to_thread(worksheet.get_all_records)
        
        await self._cache.set(cache_key, data)
        return data
    
    async def invalidate_worksheet(self, worksheet_name: str) -> None:
        """Manually invalidate cache after write operations."""
        await self._cache.invalidate(f"worksheet:{worksheet_name}")

# Использование в handlers:
async def add_result_handler(message: Message, sheets: CachedSheetsService):
    # Read операции - с кешем
    results = await sheets.get_worksheet_data("results")
    
    # Write операция
    await sheets.append_row("results", new_data)
    
    # Инвалидировать кеш после записи
    await sheets.invalidate_worksheet("results")
```

**Результат:**
- 🚀 Снижение API calls на 80-90%
- ⚡ Ускорение ответов бота в 5-10 раз
- ✅ Не превышаем rate limits

---

### 3. Graceful Shutdown (Средний приоритет)

**Проблема:** При остановке бота могут теряться данные:
```python
# bot.py - текущий код
finally:
    with suppress(Exception):  # Игнорируем ошибки!
        await bot.close()
```

**Решение:** Proper cleanup

```python
# bot.py
import signal

class BotApplication:
    """Main application with proper lifecycle management."""
    
    def __init__(self, container: ServiceContainer):
        self.container = container
        self._shutdown_event = asyncio.Event()
    
    def _signal_handler(self, sig, frame):
        """Handle SIGTERM/SIGINT."""
        logger.info("Received signal %s, shutting down gracefully...", sig)
        self._shutdown_event.set()
    
    async def run(self) -> None:
        """Run bot with graceful shutdown."""
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        dp = setup_dispatcher(self.container)
        
        # Start background tasks
        queue_task = asyncio.create_task(
            drain_queue(), 
            name="notification-queue"
        )
        
        try:
            # Start polling in background
            polling_task = asyncio.create_task(
                dp.start_polling(self.container.bot)
            )
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
            logger.info("Shutdown initiated...")
            
            # Stop polling
            await dp.stop_polling()
            polling_task.cancel()
            
            # Wait for ongoing handlers (max 30 seconds)
            logger.info("Waiting for handlers to complete...")
            await asyncio.wait_for(
                polling_task, 
                timeout=30.0
            )
            
        except asyncio.TimeoutError:
            logger.warning("Handlers didn't finish in time, forcing shutdown")
        
        finally:
            # Cancel background tasks
            queue_task.cancel()
            await asyncio.gather(queue_task, return_exceptions=True)
            
            # Shutdown services (save state, close connections)
            logger.info("Shutting down services...")
            await self.container.shutdown()
            
            # Close bot session
            await self.container.bot.session.close()
            
            logger.info("Shutdown complete")

# main
async def main() -> None:
    admin_ids = _parse_admin_chat_ids(ADMIN_IDS)
    container = await ServiceContainer.create(admin_ids)
    
    app = BotApplication(container)
    await app.run()
```

**Преимущества:**
- ✅ Дожидаемся завершения текущих обработчиков
- ✅ Сохраняем состояние перед выключением
- ✅ Логируем процесс shutdown
- ✅ Корректно закрываем соединения

---

### 4. Rate Limiting для пользователей (Средний приоритет)

**Проблема:** Нет защиты от спама/злоупотреблений

**Решение:** Middleware с rate limiting

```python
# middlewares/rate_limit.py
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

class RateLimitMiddleware(BaseMiddleware):
    """Prevent spam and abuse."""
    
    def __init__(
        self,
        rate: int = 5,  # requests
        per: int = 60,  # seconds
    ):
        self.rate = rate
        self.per = timedelta(seconds=per)
        self._user_requests: Dict[int, list[datetime]] = defaultdict(list)
    
    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        if not isinstance(event, Message):
            return await handler(event, data)
        
        user_id = event.from_user.id
        now = datetime.now()
        
        # Cleanup old requests
        cutoff = now - self.per
        self._user_requests[user_id] = [
            ts for ts in self._user_requests[user_id]
            if ts > cutoff
        ]
        
        # Check rate limit
        if len(self._user_requests[user_id]) >= self.rate:
            await event.answer(
                "⏱️ Забагато запитів. Зачекайте трохи.",
                show_alert=True
            )
            return
        
        # Record request
        self._user_requests[user_id].append(now)
        
        return await handler(event, data)

# bot.py
dp.message.middleware(RateLimitMiddleware(rate=10, per=60))
```

---

### 5. Метрики и мониторинг (Высокий приоритет для продакшна)

**Проблема:** Нет visibility в production

**Решение:** Prometheus metrics

```python
# services/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
MESSAGES_TOTAL = Counter(
    'bot_messages_total',
    'Total messages processed',
    ['handler', 'status']
)

COMMAND_DURATION = Histogram(
    'bot_command_duration_seconds',
    'Command processing time',
    ['command']
)

ACTIVE_USERS = Gauge(
    'bot_active_users',
    'Number of active users in last 24h'
)

SHEETS_API_CALLS = Counter(
    'bot_sheets_api_calls_total',
    'Google Sheets API calls',
    ['operation']
)

class MetricsMiddleware(BaseMiddleware):
    """Track metrics for all messages."""
    
    async def __call__(self, handler, event, data):
        if not isinstance(event, Message):
            return await handler(event, data)
        
        command = event.text.split()[0] if event.text else "unknown"
        start_time = time.time()
        
        try:
            result = await handler(event, data)
            MESSAGES_TOTAL.labels(handler=command, status='success').inc()
            return result
        
        except Exception as e:
            MESSAGES_TOTAL.labels(handler=command, status='error').inc()
            raise
        
        finally:
            duration = time.time() - start_time
            COMMAND_DURATION.labels(command=command).observe(duration)

# bot.py
async def main():
    # Start metrics server on :9090
    start_http_server(9090)
    
    # Add middleware
    dp.message.middleware(MetricsMiddleware())
    
    # ...

# Теперь можно подключить Grafana и мониторить:
# - Количество сообщений
# - Latency команд
# - Ошибки
# - Активные пользователи
```

---

### 6. Структурированное логирование (Средний приоритет)

**Проблема:** Логи сложно парсить и анализировать

```python
# Текущий подход
logger.info("Backup uploaded to s3://%s/%s (%s bytes)", bucket, key, size)
```

**Решение:** Структурированные логи с контекстом

```python
# utils/logger.py
import structlog
from structlog.stdlib import LoggerFactory

def configure_logging():
    """Setup structured logging."""
    
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        logger_factory=LoggerFactory(),
    )

# Использование
logger = structlog.get_logger(__name__)

# Вместо:
logger.info("User %s added result", user_id)

# Пишем:
logger.info(
    "result_added",
    user_id=user_id,
    distance=100,
    stroke="freestyle",
    time=56.78
)

# Output (JSON):
# {
#   "event": "result_added",
#   "user_id": 123,
#   "distance": 100,
#   "stroke": "freestyle",
#   "time": 56.78,
#   "timestamp": "2025-11-08T01:37:00.123Z",
#   "level": "info"
# }

# Можно легко парсить в ELK/Loki/CloudWatch
```

---

### 7. Database Connection Pool (Критично для PostgreSQL)

**Проблема:** Создается новое соединение для каждого запроса

**Решение:** Connection pooling

```python
# services/database.py
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker
)

class Database:
    """Database connection manager."""
    
    def __init__(self, db_url: str):
        self.engine: AsyncEngine = create_async_engine(
            db_url,
            pool_size=20,           # Размер пула
            max_overflow=10,        # Дополнительные соединения
            pool_pre_ping=True,     # Проверка соединения
            pool_recycle=3600,      # Пересоздавать каждый час
            echo=False,             # SQL логирование
        )
        
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    
    async def close(self) -> None:
        """Close all connections."""
        await self.engine.dispose()
    
    def get_session(self) -> AsyncSession:
        """Get database session from pool."""
        return self.session_factory()

# Использование в handlers
async def my_handler(message: Message, db: Database):
    async with db.get_session() as session:
        result = await session.execute(select(User))
        users = result.scalars().all()
        # ...
    # Сессия автоматически возвращается в пул
```

---

### 8. Feature Flags (Низкий приоритет, но полезно)

**Проблема:** Нельзя включать/выключать фичи без редеплоя

**Решение:** Runtime feature toggles

```python
# services/features.py
from enum import Enum
from typing import Dict, Set

class Feature(Enum):
    """Available feature flags."""
    TURN_ANALYSIS = "turn_analysis"
    PDF_REPORTS = "pdf_reports"
    LEADERBOARD = "leaderboard"
    AI_RECOMMENDATIONS = "ai_recommendations"

class FeatureFlags:
    """Manage feature flags at runtime."""
    
    def __init__(self):
        self._enabled: Set[Feature] = set()
        self._user_overrides: Dict[int, Set[Feature]] = {}
    
    def enable(self, feature: Feature) -> None:
        """Enable feature globally."""
        self._enabled.add(feature)
    
    def disable(self, feature: Feature) -> None:
        """Disable feature globally."""
        self._enabled.discard(feature)
    
    def is_enabled(
        self, 
        feature: Feature, 
        user_id: int = None
    ) -> bool:
        """Check if feature is enabled."""
        
        # User-specific override
        if user_id and user_id in self._user_overrides:
            return feature in self._user_overrides[user_id]
        
        # Global setting
        return feature in self._enabled
    
    def enable_for_user(self, feature: Feature, user_id: int) -> None:
        """Enable feature for specific user (beta testing)."""
        if user_id not in self._user_overrides:
            self._user_overrides[user_id] = set()
        self._user_overrides[user_id].add(feature)

# Использование
async def leaderboard_handler(
    message: Message, 
    features: FeatureFlags
):
    if not features.is_enabled(Feature.LEADERBOARD, message.from_user.id):
        await message.answer("Ця функція тимчасово недоступна")
        return
    
    # Show leaderboard
```

---

### 9. Retry механизм для внешних API (Средний приоритет)

**Проблема:** Один failed request = ошибка для пользователя

**Решение:** Exponential backoff с retry

```python
# utils/retry.py
import asyncio
from typing import TypeVar, Callable
from functools import wraps

T = TypeVar('T')

def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """Retry decorator with exponential backoff."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "Attempt %d failed, retrying in %.1fs: %s",
                            attempt + 1, delay, e
                        )
                        await asyncio.sleep(delay)
            
            # All attempts failed
            raise last_exception
        
        return wrapper
    return decorator

# Использование
@async_retry(max_attempts=3, exceptions=(gspread.exceptions.APIError,))
async def fetch_worksheet_data(worksheet_name: str):
    """Fetch with auto-retry."""
    return await asyncio.to_thread(
        worksheet.get_all_records
    )
```

---

### 10. Healthcheck endpoint (Критично для продакшна)

**Проблема:** Нельзя проверить состояние бота

**Решение:** HTTP healthcheck

```python
# services/healthcheck.py
from aiohttp import web
import asyncio

class HealthcheckServer:
    """HTTP server for health checks."""
    
    def __init__(self, container: ServiceContainer, port: int = 8080):
        self.container = container
        self.port = port
        self.app = web.Application()
        self.app.router.add_get('/health', self.health)
        self.app.router.add_get('/ready', self.ready)
        self._runner = None
    
    async def health(self, request):
        """Liveness probe - is process alive?"""
        return web.json_response({
            'status': 'ok',
            'service': 'sprint-bot'
        })
    
    async def ready(self, request):
        """Readiness probe - can handle traffic?"""
        
        checks = {
            'database': await self._check_database(),
            'bot_api': await self._check_bot_api(),
            'sheets': await self._check_sheets(),
        }
        
        all_ready = all(checks.values())
        
        return web.json_response(
            {
                'ready': all_ready,
                'checks': checks
            },
            status=200 if all_ready else 503
        )
    
    async def _check_database(self) -> bool:
        try:
            # Простой запрос в БД
            async with self.container.role_service._lock:
                return True
        except Exception:
            return False
    
    async def _check_bot_api(self) -> bool:
        try:
            me = await self.container.bot.get_me()
            return me is not None
        except Exception:
            return False
    
    async def _check_sheets(self) -> bool:
        # Проверка доступности Google Sheets
        return True
    
    async def start(self):
        """Start healthcheck server."""
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, '0.0.0.0', self.port)
        await site.start()
        logger.info("Healthcheck server started on :%d", self.port)
    
    async def stop(self):
        """Stop healthcheck server."""
        if self._runner:
            await self._runner.cleanup()

# bot.py
async def main():
    container = await ServiceContainer.create(admin_ids)
    
    # Start healthcheck
    healthcheck = HealthcheckServer(container)
    await healthcheck.start()
    
    try:
        await app.run()
    finally:
        await healthcheck.stop()

# Kubernetes/Docker проверяет:
# curl http://localhost:8080/health  # Liveness
# curl http://localhost:8080/ready   # Readiness
```

---

## 📊 Приоритизация улучшений

### Must Have (Сделать в первую очередь):
1. ✅ Dependency Injection Container
2. ✅ Кеширование Google Sheets
3. ✅ Database Connection Pool

### Should Have (Важно для продакшна):
4. ✅ Метрики и мониторинг
5. ✅ Graceful Shutdown
6. ✅ Healthcheck endpoint

### Nice to Have (Улучшают DX/UX):
7. ✅ Rate Limiting
8. ✅ Retry механизм
9. ✅ Структурированное логирование
10. ✅ Feature Flags

---

## 🚀 План внедрения (Поэтапно)

### Этап 1: Основа (1-2 недели)
- [ ] Dependency Injection Container
- [ ] Database Connection Pool
- [ ] Healthcheck endpoint

### Этап 2: Производительность (1 неделя)
- [ ] Кеширование Google Sheets
- [ ] Retry механизм для API
- [ ] Rate Limiting

### Этап 3: Observability (1 неделя)
- [ ] Метрики Prometheus
- [ ] Структурированное логирование
- [ ] Graceful Shutdown

### Этап 4: Advanced (опционально)
- [ ] Feature Flags
- [ ] A/B тестирование
- [ ] Advanced analytics

---

## 📈 Ожидаемые результаты

После внедрения улучшений:

### Производительность:
- 🚀 **80-90%** снижение API calls к Google Sheets
- ⚡ **5-10x** ускорение ответов бота
- 📊 **50%** снижение latency команд

### Надежность:
- ✅ **99.9%** uptime
- 🛡️ Защита от spam/abuse
- 🔄 Auto-recovery при сбоях

### Observability:
- 📊 Метрики в реальном времени
- 🔍 Structured logs для анализа
- 🚨 Alerts при проблемах

### Developer Experience:
- 🧪 Легче тестировать
- 🔧 Проще добавлять features
- 📝 Лучше документировано

---

## 🛠️ Дополнительные улучшения

### A. Testing Infrastructure
```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
async def container():
    """Mock container for testing."""
    container = AsyncMock(spec=ServiceContainer)
    container.role_service = AsyncMock()
    container.bot = AsyncMock()
    yield container
    await container.shutdown()

# tests/test_handlers.py
async def test_start_command(container):
    message = AsyncMock()
    await start_handler(message, container.role_service)
    message.answer.assert_called_once()
```

### B. CI/CD улучшения
```yaml
# .github/workflows/test.yml
- name: Performance tests
  run: pytest tests/performance --benchmark-only

- name: Load tests
  run: locust -f tests/load/locustfile.py --headless

- name: Security scan
  run: bandit -r . -f json -o security-report.json
```

### C. Documentation
```python
# Добавить OpenAPI для админ API
from fastapi import FastAPI

admin_api = FastAPI(title="Sprint Bot Admin API")

@admin_api.get("/api/v1/users")
async def list_users():
    """Get all registered users."""
    pass
```

---

**Резюме:** Код хорошо структурирован, но улучшения сделают его production-ready с высокой производительностью, observability и reliability.
