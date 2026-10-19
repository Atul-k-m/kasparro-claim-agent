"""Runtime configuration. Secrets come only from the environment / .env (never committed)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATASET_VERSION = "long-brand-answers-50-v4"
FACT_VERSION = "speaker-catalogue-v4"
EVALUATOR_VERSION = "sentence-spans-and-labels-v4"
AGENT_VERSIONS = ("v1", "v2")
LABELS = ("supported", "contradicted", "unknown")

# OpenAI-compatible endpoints; the provider is a setting, so swapping needs only .env changes.
PROVIDERS = {
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1"),
    "openai": ("OPENAI_API_KEY", None),
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"),
}


@dataclass(frozen=True)
class Settings:
    provider: str = "groq"
    model: str = "openai/gpt-oss-120b"
    request_timeout_s: float = 90.0
    max_turns: int = 3
    temperature: float = 0.0
    max_output_tokens: int = 3500
    rate_limit_retries: int = 6
    data_dir: Path = ROOT
    reports_dir: Path = ROOT / "reports"
    database_url: str | None = None
    extra: dict = field(default_factory=dict)

    def public(self) -> dict:
        """Settings safe to persist in run metadata (no secrets)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "request_timeout_s": self.request_timeout_s,
            "max_turns": self.max_turns,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "rate_limit_retries": self.rate_limit_retries,
            **self.extra,
        }


def get_settings() -> Settings:
    return Settings(
        provider=os.getenv("LLM_PROVIDER", "groq"),
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
        request_timeout_s=float(os.getenv("LLM_TIMEOUT_S", "90")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "3500")),
        rate_limit_retries=int(os.getenv("LLM_RATE_LIMIT_RETRIES", "6")),
        data_dir=Path(os.getenv("DATA_DIR", str(ROOT))),
        database_url=os.getenv("DATABASE_URL") or None,
        extra={"reasoning_effort": os.getenv("LLM_REASONING_EFFORT", "low")},
    )
