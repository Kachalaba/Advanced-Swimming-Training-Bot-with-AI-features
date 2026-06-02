#!/usr/bin/env sh
set -eu

exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8443}" \
  --server.headless=true \
  --browser.gatherUsageStats=false
