from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _ruta_raiz_proyecto() -> Path:
    return Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Configuración validada por entorno para el microservicio activofijo-ai."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    port: int = Field(default=8000, ge=1, le=65535)
    env: str = Field(default="development")
    ai_internal_api_key: str = Field(default="")
    core_internal_url: str = Field(default="http://localhost:3000")

    llm_provider: str = Field(default="groq")
    llm_base_url: str = Field(default="https://api.groq.com/openai/v1")
    llm_api_key: str = Field(default="")
    llm_model: str = Field(default="llama-3.3-70b-versatile")
    llm_timeout_seconds: float = Field(default=25.0, gt=0)
    llm_max_tokens: int = Field(default=1024, gt=0, le=8192)

    models_dir: Path = Field(default_factory=lambda: _ruta_raiz_proyecto() / "models")


@lru_cache
def get_settings() -> Settings:
    """Devuelve la instancia cacheada de configuración del entorno."""
    return Settings()

