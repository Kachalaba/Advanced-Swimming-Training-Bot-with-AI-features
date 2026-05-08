# syntax=docker/dockerfile:1.7
# Multi-stage build: builder installs all deps; runtime image omits dev tools.

# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time system deps (needed to compile some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Install runtime-only packages (skip dev/test tools) into an isolated prefix
RUN pip install --no-cache-dir --prefix=/install \
        $(grep -vE '^\s*(#|pytest|factory_boy|Faker|black|isort|ruff|mypy|pre-commit)' requirements.txt | tr '\n' ' ')

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels from builder stage
COPY --from=builder /install /usr/local

# Copy application source (excludes .dockerignore entries)
COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8443

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8443/_stcore/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
