# 🚀 Швидкий старт Sprint-Bot

## ✅ Виправлення застосовано

Всі критичні помилки виправлені! Бот готовий до запуску.

---

## 📦 Крок 1: Встановіть залежності

```bash
cd /Users/nikita/Downloads/Sprint-Bot-main
pip install -r requirements.txt
```

---

## ⚙️ Крок 2: Створіть .env файл

```bash
# Скопіюйте приклад
cp .env.example .env

# Відредагуйте файл
nano .env  # або vim .env, або code .env
```

### Мінімальна конфігурація:

```env
# Отримайте токен у @BotFather в Telegram
BOT_TOKEN="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"

# Ваш Telegram ID (отримайте у @userinfobot)
ADMIN_IDS="123456789"

# База даних (оберіть один варіант)
STORAGE_BACKEND="postgres"
DB_URL="postgresql+asyncpg://postgres:password@localhost:5432/sprint_bot"
```

---

## 🗄️ Крок 3: Налаштуйте базу даних

### Варіант A: PostgreSQL (рекомендовано)

```bash
# macOS
brew install postgresql
brew services start postgresql

# Linux (Ubuntu/Debian)
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql

# Створіть базу даних
createdb sprint_bot

# Запустіть міграції
alembic upgrade head
```

### Варіант B: Google Sheets

```bash
# 1. Створіть Service Account у Google Cloud Console
# 2. Завантажте JSON ключ
# 3. Збережіть як creds/service-account.json

mkdir -p creds
mv ~/Downloads/your-service-key.json creds/service-account.json

# У .env використовуйте:
STORAGE_BACKEND="sheets"
SPREADSHEET_KEY="your_spreadsheet_key_from_url"
GOOGLE_APPLICATION_CREDENTIALS="creds/service-account.json"
```

---

## ▶️ Крок 4: Запустіть бота

```bash
python bot.py
```

Якщо все налаштовано правильно, ви побачите:

```
[SprintBot] starting…
Sentry DSN not provided; Sentry disabled
Chat database initialised at data/chat.db
Backup service started (interval: 6:00:00, bucket: ...)
INFO:aiogram:Start polling...
```

---

## 🧪 Крок 5: Протестуйте бота

1. Відкрийте Telegram
2. Знайдіть вашого бота за username
3. Надішліть `/start`
4. Якщо ви адміністратор (ваш ID у ADMIN_IDS), ви побачите додаткові кнопки

---

## 🎯 Готово!

Ваш Sprint-Bot працює! 

### Наступні дії:

- 📖 Прочитайте [FIXES_APPLIED.md](FIXES_APPLIED.md) щоб дізнатись що було виправлено
- ⚙️ Прочитайте [CONFIG_SETUP.md](CONFIG_SETUP.md) для детальніших налаштувань
- 📚 Прочитайте [README.md](README.md) щоб дізнатись про функції бота
- 🏗️ Прочитайте [ARCHITECTURE.md](ARCHITECTURE.md) щоб зрозуміти структуру

---

## 🆘 Проблеми?

### "ModuleNotFoundError: No module named 'aiogram'"
```bash
pip install -r requirements.txt
```

### "RuntimeError: BOT_TOKEN environment variable must be set"
```bash
# Переконайтесь що .env існує та містить BOT_TOKEN
cat .env | grep BOT_TOKEN
```

### "Unable to connect to database"
```bash
# PostgreSQL
pg_isready
createdb sprint_bot

# або використовуйте Google Sheets
```

### "creds.json not found"
```bash
# Для Google Sheets
ls -la creds/service-account.json
# Файл має існувати та бути валідним JSON
```

---

## 🐳 Docker (альтернатива)

```bash
# Спершу налаштуйте .env
cp .env.example .env
nano .env

# Запустіть через Docker
docker compose up --build
```

---

**Потрібна допомога?** Відкрийте issue на GitHub або прочитайте детальну документацію.
