# syntax=docker/dockerfile:1.7
# Multi-stage build: builder installs all deps; runtime image omits dev tools.

# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ARG TARGETARCH
WORKDIR /build

# Install build-time system deps (needed to compile some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Install runtime-only packages (skip dev/test tools, strip comments) into an
# isolated prefix. Use CPU-only torch wheels on amd64; older torch CPU wheels
# are not published with the +cpu suffix for arm64.
RUN sed -E 's/[[:space:]]+#.*$//' requirements.txt \
    | grep -vE '^\s*#' \
    | grep -vE '^\s*$' \
    | grep -vE '^(pytest|pytest-asyncio|pytest-cov|factory_boy|Faker|black|isort|ruff|mypy|pre-commit)([=<>!~ ]|$)' \
    > /tmp/runtime-requirements.txt \
    && if [ "${TARGETARCH:-amd64}" = "amd64" ]; then \
        sed -i -E 's/^torch==2\.1\.2$/torch==2.1.2+cpu/' /tmp/runtime-requirements.txt \
        && sed -i -E 's/^torchvision==0\.16\.2$/torchvision==0.16.2+cpu/' /tmp/runtime-requirements.txt; \
    fi \
    && pip install --no-cache-dir --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/runtime-requirements.txt

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
