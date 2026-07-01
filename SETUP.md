# SETUP.md — запуск SPRINT AI з нуля

Документ описує повний шлях розгортання SPRINT AI: від клонування репозиторію до запуску веб-стека в Docker. Усі приклади наведені для macOS/Linux; для Windows використовуйте PowerShell або WSL.

## Попередні вимоги

| Компонент | Версія | Примітки |
| --- | --- | --- |
| Python | 3.11+ | Рекомендується встановити через `pyenv` або `asdf` |
| pip | 23+ | Оновіть `python -m pip install --upgrade pip` |
| Node.js | 22+ | Потрібен лише для розробки frontend (Next.js) |
| Docker | 24+ | Необов'язково для локального запуску без контейнерів |
| Git | 2.40+ | Для клонування репозиторію |

Опційно знадобиться **Anthropic API ключ** (для AI-тренера та чату) — без нього застосунок працює в offline/rule-based режимі.

## Швидкий старт (Streamlit-оболонка)

1. Клонуйте репозиторій і перейдіть у каталог:
   ```bash
   git clone https://github.com/Kachalaba/Advanced-Swimming-Training-Bot-with-AI-features.git
   cd Advanced-Swimming-Training-Bot-with-AI-features
   ```
2. Створіть та активуйте віртуальне середовище (опційно, але бажано):
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
3. Встановіть залежності та перевірте тестовий набір:
   ```bash
   pip install -r requirements.txt
   pytest tests/unit -q
   ```
4. Скопіюйте та заповніть конфіг:
   ```bash
   cp .env.example .env
   ```
   Після цього відредагуйте `.env`, додавши за потреби `ANTHROPIC_API_KEY` та інші параметри.
5. Запустіть застосунок:
   ```bash
   python3 -m streamlit run app.py
   ```
6. Відкрийте http://localhost:8501 — має з'явитися інтерфейс з вкладками аналізу.

## Docker сценарій (production-стек: FastAPI + Next.js)

1. Переконайтеся, що `.env` налаштований (див. попередній розділ).
2. Запустіть збірку та піднімання сервісів:
   ```bash
   docker compose up --build
   ```
3. Backend доступний на http://localhost:8000 (health: `/api/health`), frontend — на http://localhost:3000. Логи:
   ```bash
   docker compose logs -f backend frontend
   ```
4. Для оновлення образу використовуйте `docker compose pull && docker compose up -d`.

## Середовище та змінні

| Змінна | Опис |
| --- | --- |
| `ANTHROPIC_API_KEY` | Claude API ключ для AI-тренера та чату (опційно) |
| `OPENAI_API_KEY` | OpenAI fallback для AI-тренера (опційно) |
| `SPRINT_AI_CLAUDE_MODEL` | Модель Claude (за замовчуванням `claude-sonnet-5`) |
| `ATHLETE_DB_PATH` | Шлях до бази атлетів (SQLite) |
| `CORS_ORIGINS` | Дозволені origin-и для FastAPI backend |
| `MAX_VIDEO_UPLOAD_MB` | Ліміт розміру відео для завантаження (за замовчуванням 512) |
| `NEXT_PUBLIC_BACKEND_URL` | URL backend для Next.js frontend |
| `SENTRY_DSN` | DSN для трекінгу помилок у Sentry (опційно) |
| `PORT` | Порт Streamlit Docker-образу |

Не зберігайте готовий `.env` у репозиторії; використовуйте секрети CI/CD або менеджер паролів.

## Перевірка якості

```bash
make format    # isort + black
make lint      # ruff + black + isort + mypy
make test      # pytest
make test-cov  # pytest з покриттям (coverage.xml)
```

Frontend:

```bash
cd frontend
npm ci
npm run typecheck && npm run lint && npm run test && npm run build
```

## Типові проблеми

- **`ModuleNotFoundError` під час запуску.** Переконайтеся, що активоване віртуальне середовище і виконаний `pip install -r requirements.txt`.
- **Docker контейнер завершується.** Перевірте лог `docker compose logs -f backend` і доступність тому `./data`.
- **AI-тренер відповідає шаблонними порадами.** Перевірте, що `ANTHROPIC_API_KEY` заданий і пакет `anthropic` встановлений — інакше працює offline-режим.
- **Тести не запускаються.** Використайте `pytest tests/unit -q`, переконайтеся в сумісності версії Python і наявності dev-залежностей.

Після виконання всіх кроків застосунок готовий до роботи, а CI/CD pipeline гарантує стабільність при подальших змінах.
