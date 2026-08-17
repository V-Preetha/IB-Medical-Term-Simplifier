"""Configuration for the medical report SFT data generator."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - requirements.txt installs this.
    def load_dotenv() -> bool:
        """Allow imports before optional runtime dependencies are installed."""
        return False


load_dotenv()

# Change this one value to "openai", "openrouter", or "claude".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()

SYSTEM_MESSAGE: Final[str] = """\
You are an expert medical communication assistant.

Your task is to simplify medical reports while preserving every clinical fact.

Rules:
- Never hallucinate.
- Never add or remove diagnoses.
- Never change medications.
- Never change laboratory values.
- Preserve uncertainty (possible, suspected, likely, cannot rule out).
- Expand abbreviations only when certain.
- Do not provide medical advice.
- Use simple language.
- Return ONLY valid JSON."""

INSTRUCTION_VARIANTS: Final[tuple[str, ...]] = (
    "Simplify this medical report.",
    "Explain this medical report to a patient.",
    "Rewrite this report in simple English.",
    "Explain this report in three readability levels.",
    "Convert this report into patient-friendly language.",
    "Help a patient understand this report.",
    "Explain this clinical report clearly.",
    "Simplify this report without changing any medical facts.",
    "Make this report understandable for a non-medical person.",
    "Explain this medical report using plain language.",
)


@dataclass(frozen=True)
class ProviderSettings:
    """Resolved API settings for the selected provider."""

    provider: str
    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    input_csv: Path
    output_jsonl: Path
    checkpoint_file: Path
    failed_csv: Path
    log_file: Path
    provider: ProviderSettings
    max_retries: int
    initial_backoff_seconds: float
    request_timeout_seconds: float
    max_output_tokens: int
    temperature: float
    samples_per_report: int = 5
    enable_self_check: bool = False
    random_seed: int = 42


def _environment_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of true/false, yes/no, on/off, or 1/0."
    )


def _provider_settings(provider: str) -> ProviderSettings:
    """Build provider settings and fail early when credentials are missing."""
    providers = {
        "openai": {
            "key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4.1-mini",
        },
        "openrouter": {
            "key_env": "OPENROUTER_API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "openai/gpt-4.1-mini",
        },
        "claude": {
            "key_env": "ANTHROPIC_API_KEY",
            "base_url": "https://api.anthropic.com/v1",
            "model": "claude-haiku-4-5-20251001",
        },
    }
    if provider not in providers:
        supported = ", ".join(sorted(providers))
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider}'. Choose one of: {supported}."
        )

    defaults = providers[provider]
    key_env = defaults["key_env"]
    api_key = os.getenv(key_env, "").strip()
    if not api_key:
        raise ValueError(
            f"{key_env} is required when LLM_PROVIDER={provider!r}."
        )

    prefix = provider.upper()
    base_url = os.getenv(
        f"{prefix}_BASE_URL", defaults["base_url"]
    ).rstrip("/")
    model = os.getenv(f"{prefix}_MODEL", defaults["model"]).strip()
    if not base_url or not model:
        raise ValueError(f"{prefix}_BASE_URL and {prefix}_MODEL cannot be empty.")

    return ProviderSettings(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )


def load_settings() -> Settings:
    """Load and validate all runtime settings."""
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    initial_backoff = float(os.getenv("INITIAL_BACKOFF_SECONDS", "1"))
    timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
    max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "3000"))
    temperature = float(os.getenv("TEMPERATURE", "0.1"))
    samples_per_report = int(os.getenv("SAMPLES_PER_REPORT", "5"))
    enable_self_check = _environment_bool("ENABLE_SELF_CHECK", False)
    random_seed = int(os.getenv("RANDOM_SEED", "42"))

    if max_retries < 0:
        raise ValueError("MAX_RETRIES must be zero or greater.")
    if initial_backoff < 0:
        raise ValueError("INITIAL_BACKOFF_SECONDS must be zero or greater.")
    if timeout <= 0 or max_tokens <= 0:
        raise ValueError(
            "REQUEST_TIMEOUT_SECONDS and MAX_OUTPUT_TOKENS must be positive."
        )
    if not 0 <= temperature <= 2:
        raise ValueError("TEMPERATURE must be between 0 and 2.")
    if samples_per_report <= 0:
        raise ValueError("SAMPLES_PER_REPORT must be positive.")

    return Settings(
        input_csv=Path(os.getenv("INPUT_CSV", "reports.csv")),
        output_jsonl=Path(
            os.getenv("OUTPUT_JSONL", "medical_lora_dataset.jsonl")
        ),
        checkpoint_file=Path(
            os.getenv("CHECKPOINT_FILE", "checkpoint.json")
        ),
        failed_csv=Path(os.getenv("FAILED_CSV", "failed_reports.csv")),
        log_file=Path(os.getenv("LOG_FILE", "pipeline.log")),
        provider=_provider_settings(LLM_PROVIDER),
        max_retries=max_retries,
        initial_backoff_seconds=initial_backoff,
        request_timeout_seconds=timeout,
        max_output_tokens=max_tokens,
        temperature=temperature,
        samples_per_report=samples_per_report,
        enable_self_check=enable_self_check,
        random_seed=random_seed,
    )
