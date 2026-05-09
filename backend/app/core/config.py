import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    cors_origins: list[str] = field(
        default_factory=lambda: os.environ.get(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
    )
    athlete_db_path: str = os.environ.get(
        "ATHLETE_DB_PATH", "data/athletes_orm.db"
    )


settings = Settings()
