from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import analysis, athletes, health
from app.core.config import settings

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
app.include_router(analysis.router, prefix="/api/analysis")
