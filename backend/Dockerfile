FROM python:3.11-slim

WORKDIR /srv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps required by opencv (libgl, libglib) and ffmpeg for video I/O.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /srv/requirements.txt
RUN pip install -r /srv/requirements.txt

COPY backend/app /srv/app

# The pipeline service computes REPO_ROOT = parents[3] from
# /srv/app/services/running.py which resolves to "/".  Put the shared
# analyzers there so `import video_analysis` works inside the container
# while keeping the same code path for local dev (where parents[3] is
# the repo root).
COPY video_analysis /video_analysis

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
