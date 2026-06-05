#!/usr/bin/env bash
# Sprint AI — local launcher for the Next.js + FastAPI app (docker compose).
# Used by the desktop "Sprint AI.app", or run directly from a terminal:
#     bash scripts/sprint-ai-run.sh
set -euo pipefail

# --- locate the project root -------------------------------------------------
# The desktop .app bakes the path into SPRINT_AI_HOME; otherwise derive it from
# this script's location so the launcher works wherever the repo is cloned.
if [ -n "${SPRINT_AI_HOME:-}" ] && [ -f "${SPRINT_AI_HOME}/docker-compose.yml" ]; then
  ROOT="${SPRINT_AI_HOME}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$ROOT"

echo "🏊  Sprint AI"
echo "📂  $ROOT"
echo ""

# --- check Docker is available -----------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "❌  Docker not found. Install Docker Desktop and try again."
  read -r -p "Press Enter to close…" _ || true
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "❌  Docker daemon is not running. Start Docker Desktop and try again."
  read -r -p "Press Enter to close…" _ || true
  exit 1
fi

# --- stop containers on exit -------------------------------------------------
cleanup() {
  echo ""
  echo "🛑  Stopping containers…"
  docker compose down
}
trap cleanup EXIT INT TERM

# --- start containers in the background so we can poll the port --------------
URL="http://localhost:3000"
echo "🚀  Starting containers (this may take a minute on first launch)…"
docker compose up &
COMPOSE_PID=$!

# --- wait for port 3000 to become available (up to 60 s) --------------------
echo "⏳  Waiting for $URL to become available (up to 60 s)…"
TIMEOUT=60
ELAPSED=0
until curl -sf "$URL" >/dev/null 2>&1; do
  if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
    echo "❌  Timed out waiting for $URL after ${TIMEOUT}s."
    echo "   Check 'docker compose logs' for errors."
    exit 1
  fi
  sleep 2
  ELAPSED=$((ELAPSED + 2))
done

echo "✅  App is ready at $URL"
echo ""

# --- open the browser --------------------------------------------------------
if command -v open >/dev/null 2>&1; then
  open "$URL"           # macOS
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL"       # Linux
fi

echo "🌐  Opened: $URL"
echo "🛑  To stop: close this window or press Ctrl+C."
echo ""

# Keep running until docker compose stops or user interrupts
wait "$COMPOSE_PID"
