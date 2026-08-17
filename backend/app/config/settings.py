"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the Medical Report Simplifier backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="AI Medical Report Simplifier")
    app_version: str = Field(default="0.1.0")
    environment: Literal["local", "development", "staging", "production"] = Field(
        default="local"
    )
    api_v1_prefix: str = Field(default="/api/v1")
    log_level: str = Field(default="INFO")
    debug: bool = Field(default=False)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    max_upload_size_mb: int = Field(default=25, ge=1)
    pdf_text_min_chars_per_page: int = Field(default=40, ge=0)
    ocr_image_dpi: int = Field(default=300, ge=72)
    scispacy_model_name: str = Field(default="en_core_sci_sm")
    entity_default_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    modernbert_model_name: str = Field(default="answerdotai/ModernBERT-base")
    modernbert_max_length: int = Field(default=512, ge=32)
    difficult_term_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    clinical_context_model_name: str = Field(
        default="emilyalsentzer/Bio_ClinicalBERT"
    )
    clinical_context_max_length: int = Field(default=512, ge=32)
    semantic_match_threshold: float = Field(default=0.35, ge=-1.0, le=1.0)
    qwen3_model_name: str = Field(default="Qwen/Qwen3-0.6B")
    qwen3_max_new_tokens: int = Field(default=768, ge=64)
    qwen3_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    qwen3_prompt_template_path: str = Field(
        default="app/prompts/qwen3_simplification_prompt.txt"
    )
    granite_guardian_model_name: str = Field(
        default="ibm-granite/granite-guardian-3.0-2b"
    )
    granite_guardian_max_new_tokens: int = Field(default=256, ge=32)
    validation_failure_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    bertscore_model_type: str = Field(default="microsoft/deberta-xlarge-mnli")
    semantic_similarity_model_name: str = Field(default="all-MiniLM-L6-v2")
    max_simplification_regeneration_attempts: int = Field(default=1, ge=0, le=3)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings for dependency injection."""
    return Settings()
