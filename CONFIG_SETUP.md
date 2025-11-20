# Налаштування Sprint-Bot після виправлень

## ✅ Виконані виправлення

### 1. Виправлено застарілий datetime API
- Замінено всі `datetime.utcnow()` на `datetime.now(timezone.utc)` для сумісності з Python 3.12+
- Оновлено файли:
  - `backup_service.py`
  - `chat_service.py`
  - `notifications.py`
  - `handlers/menu.py`
  - `handlers/common.py`
  - `sprint_bot/infrastructure/storage/google_sheets.py`
  - `sprint_bot/domain/models/entities.py`

### 2. Виправлено відсутні імпорти
- Додано `from __future__ import annotations` у `handlers/error_handler.py`
- Додано `timezone` імпорт у всіх необхідних файлах

### 3. Покращено обробку помилок
- Виправлено логіку `getattr` у `role_service.py` (тепер коректно працює з ID = 0)
- Додано логування помилок у `handlers/common.py`

## 🚀 Наступні кроки для запуску бота

### Крок 1: Створіть файл `.env`

Скопіюйте `.env.example` у `.env` та заповніть:

```bash
cp .env.example .env
```

Мінімально необхідні параметри:

```env
# ОБОВ'ЯЗКОВО
BOT_TOKEN="your_telegram_bot_token_from_@BotFather"

# Необхідно для адміністрування
ADMIN_IDS="your_telegram_id"

# Для використання PostgreSQL (рекомендовано)
STORAGE_BACKEND="postgres"
DB_URL="postgresql+asyncpg://postgres:password@localhost:5432/sprint_bot"

# АБО для використання Google Sheets
STORAGE_BACKEND="sheets"
SPREADSHEET_KEY="your_google_spreadsheet_key"
GOOGLE_APPLICATION_CREDENTIALS="creds/service-account.json"
```

### Крок 2: Налаштуйте Google Sheets (якщо використовуєте)

Якщо ви використовуєте `STORAGE_BACKEND="sheets"`:

1. Створіть Service Account у Google Cloud Console:
   - Перейдіть на https://console.cloud.google.com/
   - Створіть новий проект або виберіть існуючий
   - Увімкніть Google Sheets API
   - Створіть Service Account
   - Завантажте JSON ключ

2. Збережіть JSON ключ як `creds.json` у кореневій директорії проекту:
   ```bash
   mkdir -p creds
   mv ~/Downloads/your-service-account-key.json creds/service-account.json
   ```

3. Надайте доступ Service Account до вашої Google Таблиці:
   - Відкрийте вашу Google Таблицю
   - Натисніть "Share"
   - Додайте email Service Account (виглядає як `name@project-id.iam.gserviceaccount.com`)
   - Надайте права на редагування

### Крок 3: Встановіть залежності

```bash
pip install -r requirements.txt
```

### Крок 4: Запустіть міграції БД (для PostgreSQL)

Якщо використовуєте PostgreSQL:

```bash
# Встановіть PostgreSQL якщо ще не встановлений
# macOS: brew install postgresql
# Ubuntu: sudo apt-get install postgresql

# Створіть базу даних
createdb sprint_bot

# Запустіть міграції
make migrate
# або
alembic upgrade head
```

### Крок 5: Запустіть бота

```bash
# Локально
python bot.py

# Або через Docker
make run
# або
docker compose up --build
```

## 🧪 Перевірка якості коду

Перед запуском рекомендується перевірити код:

```bash
# Форматування коду
make format

# Перевірка лінтером
make lint

# Запуск тестів
make test
```

## 📋 Отримання Telegram Bot Token

1. Відкрийте Telegram та знайдіть [@BotFather](https://t.me/botfather)
2. Надішліть `/newbot`
3. Слідуйте інструкціям:
   - Введіть ім'я бота (наприклад, "My Sprint Bot")
   - Введіть username бота (має закінчуватись на `bot`, наприклад, `my_sprint_bot`)
4. Скопіюйте токен який надав BotFather
5. Вставте токен у `.env` файл як `BOT_TOKEN`

## 📋 Отримання вашого Telegram ID

Для налаштування як адміністратор:

1. Відкрийте [@userinfobot](https://t.me/userinfobot) у Telegram
2. Надішліть `/start`
3. Скопіюйте ваш ID
4. Вставте у `.env` файл як `ADMIN_IDS`

## ⚙️ Додаткові налаштування (опціонально)

### Sentry для моніторингу помилок

```env
SENTRY_DSN="https://your-sentry-dsn@sentry.io/project-id"
ENV="production"
```

### S3 бекапи (для резервного копіювання)

```env
S3_BACKUP_BUCKET="sprint-bot-backups"
S3_ACCESS_KEY="your-access-key"
S3_SECRET_KEY="your-secret-key"
S3_BACKUP_PREFIX="sprint-bot/backups/"
BACKUP_INTERVAL_HOURS="6"
```

### Тихі години (нічний режим)

```env
QUIET_HOURS="22:00-07:00"
QUIET_HOURS_TZ="Europe/Kyiv"
QUIET_QUEUE_INTERVAL="60"
```

## 🐛 Вирішення проблем

### Помилка "BOT_TOKEN not set"
- Переконайтесь що `.env` файл існує у кореневій директорії
- Перевірте що `BOT_TOKEN` не порожній та без лапок

### Помилка "creds.json not found" (для Google Sheets)
- Переконайтесь що файл існує за шляхом `creds/service-account.json`
- Або вкажіть інший шлях у `GOOGLE_APPLICATION_CREDENTIALS`

### Помилка підключення до PostgreSQL
- Переконайтесь що PostgreSQL запущений: `pg_isready`
- Перевірте credentials у `DB_URL`
- Створіть базу даних: `createdb sprint_bot`

### Помилка "Spreadsheet not found"
- Перевірте `SPREADSHEET_KEY` у `.env`
- Переконайтесь що Service Account має доступ до таблиці

## 📚 Додаткова документація

- [SETUP.md](SETUP.md) - Детальна інструкція з встановлення
- [ARCHITECTURE.md](ARCHITECTURE.md) - Архітектура проекту
- [OPERATIONS.md](OPERATIONS.md) - Операційні процедури
- [SECURITY_NOTES.md](SECURITY_NOTES.md) - Безпека
