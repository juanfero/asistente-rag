"""Configuración centralizada del proyecto (variables de entorno y `.env`)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parámetros del asistente; cada campo se lee de la variable en MAYÚSCULAS homónima."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM: Gemini vía endpoint compatible con OpenAI (ADR-010)
    gemini_api_key: SecretStr | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-3.1-flash-lite"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=700, ge=1)

    # Embeddings
    # e5-small: entrenado para recuperar pasajes, 512 tokens (ADR-009)
    embedding_model: str = "intfloat/multilingual-e5-small"
    # Prefijos que e5 espera; solo se aplican al codificar (vacíos = sin prefijo)
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "

    # Datos y vector store
    docs_dir: Path = Path("data/docs")
    chroma_dir: Path = Path("data/chroma")
    chroma_collection: str = "documentos"

    # Chunking y recuperación
    # 800/120: 0 % de chunks sobre 512 tokens con e5-small (ADR-004, ADR-009)
    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=120, ge=0)
    top_k: int = Field(default=4, ge=1)
    # Provisional: se calibra en M7 (ADR-005) como filtro de ruido, no como abstención
    min_score: float = Field(default=0.80, ge=0.0, le=1.0)

    # Interfaces y logging
    api_url: str = "http://localhost:8000"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        """Exige que el solapamiento sea menor que el tamaño del chunk."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) debe ser menor que "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Devuelve la instancia única (cacheada) de `Settings`."""
    return Settings()
