# SPRINT AI — Triathlon Video Analysis Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![CI: Lint](https://github.com/Kachalaba/Advanced-Swimming-Training-Bot-with-AI-features/actions/workflows/lint.yml/badge.svg)](https://github.com/Kachalaba/Advanced-Swimming-Training-Bot-with-AI-features/actions/workflows/lint.yml)
[![CI: Tests](https://github.com/Kachalaba/Advanced-Swimming-Training-Bot-with-AI-features/actions/workflows/tests.yml/badge.svg)](https://github.com/Kachalaba/Advanced-Swimming-Training-Bot-with-AI-features/actions/workflows/tests.yml)
[![CI: Docker](https://github.com/Kachalaba/Advanced-Swimming-Training-Bot-with-AI-features/actions/workflows/docker.yml/badge.svg)](https://github.com/Kachalaba/Advanced-Swimming-Training-Bot-with-AI-features/actions/workflows/docker.yml)

**Професійний AI-інструмент для тренерів з тріатлону: аналіз відео у реальному часі для плавання, бігу та велосипеда**

<p align="center">
  <img src="https://img.shields.io/badge/Swimming-40+_metrics-00D9FF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Running-30+_metrics-10B981?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Cycling-35+_metrics-F59E0B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dryland-AI_coaching-8B5CF6?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-EN%20%2F%20UA-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Theme-Light%20%2F%20Dark-gray?style=for-the-badge" />
  <img src="https://img.shields.io/badge/AI-Claude%20API-blueviolet?style=for-the-badge" />
</p>

---

## Для кого цей інструмент?

| Користувач | Використання |
|---------------|-----------------|
| **Тренери з тріатлону** | Аналіз техніки всіх 3 дисциплін + AI-асистент |
| **Тренери з плавання** | Детальний аналіз гребка, body roll, дихання |
| **Тренери з легкої атлетики** | Foot strike, cadence, травмопрофілактика |
| **Bike fitters** | Bike fit аналіз, посадка, педалювання |
| **Фітнес-тренери** | Аналіз вправ суходолу + база спортсменів |

---

## Можливості платформи

### Плавання (40+ метрик)

| Категорія | Метрики |
|-----------|---------|
| **Гребок** | Фази (Catch/Pull/Push/Recovery), Stroke Rate, DPS, SWOLF |
| **Техніка рук** | Hand Entry Angle (опт. 40°), High Elbow Catch Score |
| **Тіло** | Body Roll (опт. 30-50°), Head Stability, Streamline Score |
| **Дихання** | Pattern Detection (bilateral/2/3/4), Regularity |
| **Ноги** | Kick Frequency, Amplitude, Symmetry |
| **Симетрія** | L/R Balance, Phase Distribution |

### Біг (30+ метрик)

| Категорія | Метрики |
|-----------|---------|
| **Cadence** | Steps/min (опт. 170-190), Ground Contact Time |
| **Foot Strike** | Type (heel/midfoot/forefoot), Angle, Score |
| **Overstriding** | Detection, Distance Ahead, Risk Score |
| **Hip Drop** | Left/Right degrees, Trendelenburg Score |
| **Arms** | Symmetry, Crossover Detection, Swing Range |
| **Efficiency** | Bounce Score, Overall Efficiency, **Injury Risk Score** |

### Велосипед (35+ метрик)

| Категорія | Метрики |
|-----------|---------|
| **Cadence** | RPM (опт. 80-100), Power Phase % |
| **Knee** | Angle Top/Bottom, Range of Motion |
| **Bike Fit** | Saddle Height Score, Aero Score, Stack Score |
| **Ankling** | Ankle Angle Top/Bottom, Ankling Score |
| **Dead Spots** | Top/Bottom ms, Dead Spot Score |
| **Stability** | Lateral Sway, Vertical Bounce, Rock Detection |
| **Efficiency** | Pedal Smoothness, Torque Effectiveness |

### Суходіл

| Можливість | Опис |
|------------|------|
| **Детекція вправ** | Присідання, випади, планка, віджимання |
| **Біомеханіка** | Кути суглобів, траєкторія руху |
| **AI Coaching** | Рекомендації щодо техніки в реальному часі |

---

## Додаткові інструменти

### База даних спортсменів
- Профілі атлетів (рівень, спеціалізація)
- Історія всіх тренувань
- Графіки прогресу з фільтрами за типом та часовим проміжком
- Порівняння сесій та CSV-експорт метрик

### AI Асистент (Claude API)
- Контекстно-залежний чат (знає дані спортсмена)
- Генератор персоналізованих тренувальних планів
- TTS озвучення (pyttsx3/gTTS)
- База знань: drills, типові помилки техніки

### Відео інструменти
- **Side-by-Side** — порівняння двох відео
- **Highlights** — вирізка + slow-motion
- **Zoom** — фіксований або tracking zoom

---

## Швидкий старт

### 1. Встановлення

```bash
git clone https://github.com/Kachalaba/Advanced-Swimming-Training-Bot-with-AI-features.git
cd Advanced-Swimming-Training-Bot-with-AI-features

python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# або: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Запуск

```bash
python3 -m streamlit run app.py
```

Відкрийте: **http://localhost:8501**

---

## Архітектура проекту

```
sprint-ai/
├── app.py                          # Streamlit UI (7 вкладок)
├── video_analysis/
│   ├── # === CORE DETECTION ===
│   ├── frame_extractor.py          # Витягування кадрів з відео
│   ├── swimmer_detector.py         # YOLO детекція + IoU Tracking
│   ├── swimming_pose_analyzer.py   # MediaPipe поза (33 точки)
│   │
│   ├── # === SPORT-SPECIFIC ANALYZERS ===
│   ├── stroke_analyzer.py          # Аналіз гребка (40+ метрик)
│   ├── running_analyzer.py         # Аналіз бігу (30+ метрик)
│   ├── cycling_analyzer.py         # Аналіз велосипеда (35+ метрик)
│   ├── exercise_analyzer.py        # Аналіз вправ суходолу
│   │
│   ├── # === BIOMECHANICS ===
│   ├── biomechanics_analyzer.py    # Загальна біомеханіка
│   ├── biomechanics_visualizer.py  # Преміум візуалізація скелету
│   ├── trajectory_analyzer.py      # Траєкторія руху
│   ├── split_analyzer.py           # Спліти та темп
│   │
│   ├── # === AI & DATABASE ===
│   ├── ai_coach.py                 # AI рекомендації
│   ├── ai_chat.py                  # Claude API чат + TTS + Плани
│   ├── athlete_database.py         # SQLite/PostgreSQL база спортсменів
│   │
│   ├── # === OUTPUT ===
│   ├── video_overlay.py            # Анотоване відео
│   ├── video_tools.py              # Side-by-side, Zoom, Highlights
│   └── report_generator.py         # PDF/JSON звіти
│
├── i18n/                           # Локалізація EN/UA
├── data/
│   └── athletes.db                 # SQLite база даних
├── tests/                          # pytest + pytest-asyncio
├── .github/workflows/              # CI: lint, tests, docker
└── requirements.txt                # Залежності
```

---

## Приклад результатів

### Плавання
```json
{
  "stroke_analysis": {
    "total_strokes": 24,
    "stroke_rate": 58.5,
    "dps": 2.08,
    "swolf": 42.3,
    "symmetry_score": 94.2,
    "body_roll": 38.5,
    "high_elbow_score": 87.0,
    "breathing_pattern": "bilateral/3"
  }
}
```

### Біг
```json
{
  "running_analysis": {
    "cadence": 176,
    "foot_strike_type": "midfoot",
    "foot_strike_score": 95.0,
    "overstriding_detected": false,
    "hip_drop_score": 88.0,
    "efficiency_score": 82.5,
    "injury_risk_score": 15
  }
}
```

### Велосипед
```json
{
  "cycling_analysis": {
    "cadence": 92,
    "knee_range": 78.5,
    "saddle_height_score": 95.0,
    "aero_score": 85.0,
    "ankling_score": 80.0,
    "pedal_smoothness": 88.0,
    "bike_fit_score": 90.0
  }
}
```

---

## Інтерфейс

### 7 основних вкладок:

| Вкладка | Призначення |
|---------|-------------|
| **Плавання** | Аналіз техніки плавання |
| **Біг** | Аналіз техніки бігу |
| **Велосипед** | Bike fit та педалювання |
| **Суходіл** | Аналіз вправ |
| **Історія** | Прогрес спортсменів + графіки + CSV |
| **AI Асистент** | Claude API чат та генератор планів |
| **Інструменти** | Відео утиліти |

---

## Технології

| Технологія | Використання |
|------------|--------------|
| **Python 3.8+** | Основна мова |
| **YOLOv8** | Детекція спортсмена + stable IoU tracking |
| **MediaPipe** | Pose Estimation (33 keypoints) + EMA smoothing |
| **OpenCV** | Обробка відео |
| **Streamlit** | Веб-інтерфейс (light/dark theme) |
| **Claude API** | AI-асистент з контекстом спортсмена |
| **SQLite / PostgreSQL** | База даних спортсменів |
| **SQLAlchemy + Alembic** | ORM та міграції |
| **Matplotlib / Seaborn** | Графіки та візуалізація |
| **ReportLab / FPDF2** | PDF звіти |
| **pyttsx3 / gTTS** | Text-to-Speech |
| **Sentry SDK** | Логування помилок та моніторинг |
| **Docker** | Контейнеризація |
| **GitHub Actions** | CI/CD: lint, tests, docker build |

---

## Roadmap

- [x] Плавання: фази гребка, body roll, симетрія (40+ метрик)
- [x] Біг: foot strike, overstriding, hip drop (30+ метрик)
- [x] Велосипед: bike fit, ankling, dead spots (35+ метрик)
- [x] База даних спортсменів (SQLite / PostgreSQL)
- [x] AI чат (Claude API) з контекстом спортсмена
- [x] Генератор тренувальних планів
- [x] Відео інструменти (side-by-side, zoom, highlights)
- [x] PDF / JSON звіти
- [x] Локалізація EN / UA з перемикачем мови
- [x] Light / Dark theme toggle
- [x] Графіки прогресу з фільтрами та CSV-експортом
- [x] Стабільний IoU tracking (плавання, велосипед)
- [x] Преміум скелет: EMA smoothing, triple-guard displacement
- [x] GitHub Actions CI/CD (lint, tests, docker)
- [ ] Календар тренувань
- [ ] Інтеграція Garmin / Strava

---

## Документація

- [Посібник користувача](docs/USER_GUIDE.md)
- [Архітектура](ARCHITECTURE.md)
- [Історія змін](CHANGELOG.md)
- [Налаштування](SETUP.md)

---

## Внесок

```bash
# 1. Fork репозиторію
# 2. Створіть branch
git checkout -b feature/amazing-feature

# 3. Commit
git commit -m 'Add amazing feature'

# 4. Push
git push origin feature/amazing-feature

# 5. Відкрийте Pull Request
```

---

## Ліцензія

MIT License — див. [LICENSE](LICENSE)

---

<p align="center">
  <b>Створено для тренерів та спортсменів</b>
  <br>
  Swimming | Running | Cycling
</p>
