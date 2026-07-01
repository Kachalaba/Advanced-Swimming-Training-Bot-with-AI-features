from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import analysis, athletes, clinical, health, rehabilitation, tools
from app.core.config import settings

# video_analysis resolves the same way as in app.api.* (PYTHONPATH=/ in the
# container image, repo root on sys.path in local dev).
from video_analysis.monitoring import init_sentry

init_sentry()

app = FastAPI(
    title="SPRINT AI Backend",
    version="0.1.0",
    description="REST API for the SPRINT AI triathlon analysis platform.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(athletes.router, prefix="/api/athletes")
app.include_router(clinical.router, prefix="/api/clinical")
app.include_router(analysis.router, prefix="/api/analysis")
app.include_router(rehabilitation.router, prefix="/api/analysis")
app.include_router(tools.router, prefix="/api/tools")
