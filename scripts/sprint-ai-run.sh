#!/usr/bin/env bash
# Sprint AI — local launcher for the Streamlit app.
# Used by the desktop "Sprint AI.app", or run directly from a terminal:
#     bash scripts/sprint-ai-run.sh
set -euo pipefail

# --- locate the project root -------------------------------------------------
# The desktop .app bakes the path into SPRINT_AI_HOME; otherwise derive it from
# this script's location so the launcher works wherever the repo is cloned.
if [ -n "${SPRINT_AI_HOME:-}" ] && [ -f "${SPRINT_AI_HOME}/app.py" ]; then
  ROOT="${SPRINT_AI_HOME}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

echo "🏊  Sprint AI"
echo "📂  $ROOT"
echo ""

# --- pick a Python / virtualenv ---------------------------------------------
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
elif [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi
PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "❌  Python 3 not found. Install Python 3.10+ and try again."
  read -r -p "Press Enter to close…" _ || true
  exit 1
fi

# --- make sure Streamlit is available ---------------------------------------
if ! "$PY" -c "import streamlit" >/dev/null 2>&1; then
  echo "📦  Streamlit is not installed — installing…"
  if ! "$PY" -m pip install --quiet streamlit; then
    echo "❌  Could not install Streamlit. See SETUP.md (a virtualenv is recommended)."
    read -r -p "Press Enter to close…" _ || true
    exit 1
  fi
fi

# Heavy CV deps power the actual analysis (~2 GB). Warn, but don't auto-install.
if ! "$PY" -c "import mediapipe, cv2" >/dev/null 2>&1; then
  echo "⚠️   Analysis dependencies (mediapipe / opencv) are not fully installed."
  echo "     For full functionality run:  $PY -m pip install -r requirements.txt"
  echo ""
fi

# --- open the browser shortly after the server boots ------------------------
URL="http://localhost:8501"
(
  sleep 4
  if command -v open >/dev/null 2>&1; then
    open "$URL"            # macOS
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL"        # Linux
  fi
) >/dev/null 2>&1 &

echo "🚀  Starting… it will open in your browser at: $URL"
echo "🛑  To stop: close this window or press Ctrl+C."
echo ""

exec "$PY" -m streamlit run app.py \
  --server.port=8501 \
  --server.headless=true \
  --browser.gatherUsageStats=false
