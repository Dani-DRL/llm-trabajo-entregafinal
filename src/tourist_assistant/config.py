"""Configuración del proyecto: el único módulo que lee variables de entorno o .env"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_provider: str = Field("openai", description="openai | ollama")
    llm_model: str = "gpt-4o-mini"
    embed_model: str = "embeddinggemma"
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str | None = None

    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(1024, gt=0)
    num_ctx: int = Field(8192, gt=0, description="Ventana de contexto de Ollama (tokens)")
    max_history_tokens: int = Field(2000, gt=0, description="Presupuesto de tokens del historial")

    pdf_path: Path = PROJECT_ROOT / "data" / "TENERIFE.pdf"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    log_dir: Path = PROJECT_ROOT / "logs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
