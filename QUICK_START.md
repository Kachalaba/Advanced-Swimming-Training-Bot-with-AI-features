# ⚡ Быстрый старт - Шпаргалка

## 🎨 Streamlit (САМЫЙ ПРОСТОЙ)

```bash
cd /Users/nikita/Downloads/Sprint-Bot-main
python3 -m streamlit run app.py
```

Откроется: `http://localhost:8501`

---

## 💻 CLI (Командная строка)

### 🎯 Гибридный (РЕКОМЕНДУЮ)

```bash
cd /Users/nikita/Downloads/Sprint-Bot-main

PYTHONPATH=$PWD python3 examples/run_local_video_analysis.py \
    --video test_videos/твоё_видео.mp4 \
    --output ./results \
    --athlete "Имя" \
    --analysis-method hybrid \
    --fps 3.0
```

---

### 🔬 Только поза

```bash
PYTHONPATH=$PWD python3 examples/run_local_video_analysis.py \
    --video видео.mp4 \
    --analysis-method pose
```

---

### 📍 Только траектория

```bash
PYTHONPATH=$PWD python3 examples/run_local_video_analysis.py \
    --video видео.mp4 \
    --analysis-method trajectory
```

---

## 📊 Просмотр результатов

```bash
# Открыть папку
open results/

# Видео
open results/annotated_video.mp4

# Биомеханика
cat results/biomechanics/biomechanics.json

# Траектория
cat results/trajectory/trajectory_analysis.json

# Отчёты
open results/reports/
```

---

## 🔍 Полезные команды

### Проверить детекцию:
```bash
cat results/biomechanics/biomechanics.json | python3 -c "import json, sys; d=json.load(sys.stdin); print(f\"Pose: {d['average_metrics']['frames_with_pose']}/{d['average_metrics']['total_frames']} ({d['average_metrics']['frames_with_pose']/d['average_metrics']['total_frames']*100:.1f}%)\")"
```

### Посмотреть рекомендации:
```bash
cat results/biomechanics/biomechanics.json | python3 -c "import json, sys; [print(r) for r in json.load(sys.stdin)['recommendations']]"
```

---

## 🆘 Помощь

```bash
python3 examples/run_local_video_analysis.py --help
```

---

## 📁 Где результаты?

```
results/
├── frames/           # Кадры
├── detections/       # Детекция
├── biomechanics/     # Pose анализ
├── trajectory/       # Bbox анализ
├── reports/          # Отчёты
└── annotated_video.mp4
```

---

**Всё готово! Запускай!** 🚀
