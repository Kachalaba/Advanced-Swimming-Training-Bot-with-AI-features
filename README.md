# � SPRINT AI — Triathlon Video Analysis Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)

**🇺🇦 Професійний AI-інструмент для тренерів з тріатлону: плавання, біг, велосипед**

<p align="center">
  <img src="https://img.shields.io/badge/🏊_Swimming-00D9FF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/🏃_Running-10B981?style=for-the-badge" />
  <img src="https://img.shields.io/badge/🚴_Cycling-F59E0B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/🏋️_Dryland-8B5CF6?style=for-the-badge" />
</p>

---

## 🎯 Для кого цей інструмент?

| 👤 Користувач | 💡 Використання |
|---------------|-----------------|
| **Тренери з тріатлону** | Аналіз техніки всіх 3 дисциплін |
| **Тренери з плавання** | Детальний аналіз гребка, body roll, дихання |
| **Тренери з легкої атлетики** | Foot strike, cadence, травмопрофілактика |
| **Bike fitters** | Bike fit аналіз, посадка, педалювання |
| **Фітнес-тренери** | Аналіз вправ суходолу |

---

## ✨ Можливості платформи

### 🏊 Плавання (40+ метрик)

| Категорія | Метрики |
|-----------|---------|
| **Гребок** | Фази (Catch/Pull/Push/Recovery), Stroke Rate, DPS, SWOLF |
| **Техніка рук** | Hand Entry Angle (опт. 40°), High Elbow Catch Score |
| **Тіло** | Body Roll (опт. 30-50°), Head Stability, Streamline Score |
| **Дихання** | Pattern Detection (bilateral/2/3/4), Regularity |
| **Ноги** | Kick Frequency, Amplitude, Symmetry |
| **Симетрія** | L/R Balance, Phase Distribution |

### 🏃 Біг (30+ метрик)

| Категорія | Метрики |
|-----------|---------|
| **Cadence** | Steps/min (опт. 170-190), Ground Contact Time |
| **Foot Strike** | Type (heel/midfoot/forefoot), Angle, Score |
| **Overstriding** | Detection, Distance Ahead, Risk Score |
| **Hip Drop** | Left/Right degrees, Trendelenburg Score |
| **Arms** | Symmetry, Crossover Detection, Swing Range |
| **Efficiency** | Bounce Score, Overall Efficiency, **Injury Risk Score** |

### 🚴 Велосипед (35+ метрик)

| Категорія | Метрики |
|-----------|---------|
| **Cadence** | RPM (опт. 80-100), Power Phase % |
| **Knee** | Angle Top/Bottom, Range of Motion |
| **Bike Fit** | Saddle Height Score, Aero Score, Stack Score |
| **Ankling** | Ankle Angle Top/Bottom, Ankling Score |
| **Dead Spots** | Top/Bottom ms, Dead Spot Score |
| **Stability** | Lateral Sway, Vertical Bounce, Rock Detection |
| **Efficiency** | Pedal Smoothness, Torque Effectiveness |

### 🏋️ Суходіл

| Можливість | Опис |
|------------|------|
| **Детекція вправ** | Присідання, випади, планка, віджимання |
| **Біомеханіка** | Кути суглобів, траєкторія руху |
| **AI Coaching** | Рекомендації щодо техніки |

---

## 🛠️ Додаткові інструменти

### 📊 База даних спортсменів
- Профілі атлетів (рівень, спеціалізація)
- Історія всіх тренувань
- Графіки прогресу
- Порівняння сесій

### 🤖 AI Асистент
- Чат про техніку плавання/бігу/велосипеду
- Генератор тренувальних планів
- TTS озвучення (pyttsx3/gTTS)
- База знань: drills, типові помилки

### � Відео інструменти
- **Side-by-Side** — порівняння двох відео
- **Highlights** — вирізка + slow-motion
- **Zoom** — фіксований або tracking zoom

---

## 🚀 Швидкий старт

### 1. Встановлення

```bash
# Клонуємо репозиторій
git clone https://github.com/Kachalaba/Advanced-Swimming-Training-Bot-with-AI-features.git
cd Advanced-Swimming-Training-Bot-with-AI-features

# Створюємо віртуальне середовище
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# або: venv\Scripts\activate  # Windows

# Встановлюємо залежності
pip install -r requirements.txt
```

### 2. Запуск

```bash
python3 -m streamlit run app.py
```

Відкрийте: **http://localhost:8501**

---

## 📁 Архітектура проекту

```
sprint-ai/
├── app.py                          # 🎨 Streamlit UI (7 вкладок)
├── video_analysis/
│   ├── # === CORE DETECTION ===
│   ├── frame_extractor.py          # Витягування кадрів з відео
│   ├── swimmer_detector.py         # YOLO детекція + Velocity Tracking
│   ├── swimming_pose_analyzer.py   # MediaPipe поза (33 точки)
│   │
│   ├── # === SPORT-SPECIFIC ANALYZERS ===
│   ├── stroke_analyzer.py          # 🏊 Аналіз гребка (40+ метрик)
│   ├── running_analyzer.py         # 🏃 Аналіз бігу (30+ метрик)
│   ├── cycling_analyzer.py         # 🚴 Аналіз велосипеда (35+ метрик)
│   ├── exercise_analyzer.py        # 🏋️ Аналіз вправ суходолу
│   │
│   ├── # === BIOMECHANICS ===
│   ├── biomechanics_analyzer.py    # Загальна біомеханіка
│   ├── biomechanics_visualizer.py  # Візуалізація скелету
│   ├── trajectory_analyzer.py      # Траєкторія руху
│   ├── split_analyzer.py           # Спліти та темп
│   │
│   ├── # === AI & DATABASE ===
│   ├── ai_coach.py                 # AI рекомендації
│   ├── ai_chat.py                  # Чат + TTS + План тренувань
│   ├── athlete_database.py         # SQLite база спортсменів
│   │
│   ├── # === OUTPUT ===
│   ├── video_overlay.py            # Анотоване відео
│   ├── video_tools.py              # Side-by-side, Zoom, Highlights
│   └── report_generator.py         # PDF/JSON звіти
│
├── data/
│   └── athletes.db                 # 💾 SQLite база даних
├── streamlit_outputs/              # � Результати аналізу
├── requirements.txt                # 📦 Залежності
└── yolov8n.pt                      # 🎯 YOLO модель
```

---

## 📊 Приклад результатів

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

## �️ Інтерфейс

### 7 основних вкладок:

| Вкладка | Призначення |
|---------|-------------|
| 🏊 **Плавання** | Аналіз техніки плавання |
| 🏃 **Біг** | Аналіз техніки бігу |
| 🚴 **Велосипед** | Bike fit та педалювання |
| 🏋️ **Суходіл** | Аналіз вправ |
| 📊 **Історія** | База даних сесій |
| 🤖 **AI Асистент** | Чат та генератор планів |
| 🎬 **Інструменти** | Відео утиліти |

---

## � Технології

| Технологія | Використання |
|------------|--------------|
| **Python 3.8+** | Основна мова |
| **YOLOv8** | Детекція спортсмена |
| **MediaPipe** | Pose Estimation (33 keypoints) |
| **OpenCV** | Обробка відео |
| **Streamlit** | Веб-інтерфейс |
| **SQLite** | База даних спортсменів |
| **Matplotlib** | Графіки та візуалізація |
| **pyttsx3/gTTS** | Text-to-Speech |

---

## 📈 Roadmap

- [x] Плавання: фази гребка, body roll, симетрія
- [x] Біг: foot strike, overstriding, hip drop
- [x] Велосипед: bike fit, ankling, dead spots
- [x] База даних спортсменів
- [x] AI чат + генератор планів
- [x] Відео інструменти
- [ ] Таймер/секундомір для тренувань
- [ ] Календар тренувань
- [ ] PDF звіти
- [ ] Інтеграція Garmin/Strava

---

## 📚 Документація

- [📖 Посібник користувача](docs/USER_GUIDE.md)
- [🏗️ Архітектура](ARCHITECTURE.md)
- [📝 Історія змін](CHANGELOG.md)
- [⚙️ Налаштування](SETUP.md)

---

## 🤝 Внесок

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

## 📄 Ліцензія

MIT License — див. [LICENSE](LICENSE)

---

<p align="center">
  <b>Створено з ❤️ для тренерів та спортсменів</b>
  <br>
  🏊 🏃 🚴
</p>
