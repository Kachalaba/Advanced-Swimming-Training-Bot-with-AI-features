#!/bin/bash

# Quick start script for Streamlit app

echo "🏊 Swimming Video Analysis - Streamlit App"
echo "=========================================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Streamlit не установлен. Установка..."
    pip3 install streamlit
    echo "✅ Streamlit установлен!"
    echo ""
fi

# Check if mediapipe is installed
if ! python3 -c "import mediapipe" &> /dev/null; then
    echo "📦 MediaPipe не установлен. Установка..."
    pip3 install mediapipe
    echo "✅ MediaPipe установлен!"
    echo ""
fi

echo "🚀 Запуск приложения..."
echo "📱 Откроется в браузере: http://localhost:8501"
echo ""
echo "⚠️  Для остановки нажмите Ctrl+C"
echo ""

# Run streamlit
streamlit run app.py
