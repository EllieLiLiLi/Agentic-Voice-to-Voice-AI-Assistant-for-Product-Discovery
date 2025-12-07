"""Global configuration helpers for the project.

This module centralizes environment variable parsing and shared settings
for models, data paths, and service endpoints. Configuration should be
simple to swap between providers (OpenAI, Anthropic, etc.) via env vars.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from a local .env file if present.
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM provider selection and models."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class AudioConfig:
    """Configuration for ASR and TTS components."""

    asr_model: str = "whisper-small"
    tts_voice: str = "alloy"
    sample_rate: int = 16_000


@dataclass
class DataPaths:
    """Paths for raw data, processed artifacts, and vector indexes."""

    root: Path = Path("data")
    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    indexes: Path = Path("data/indexes")


@dataclass
class AppConfig:
    """Top-level configuration bundle for the application."""

    llm: LLMConfig = LLMConfig()
    audio: AudioConfig = AudioConfig()
    data_paths: DataPaths = DataPaths()
    tracing_enabled: bool = False


def get_config() -> AppConfig:
    """Return an instantiated configuration with environment overrides.

    TODO: Replace manual environment lookups with a more robust solution
    (e.g., Pydantic) once requirements are finalized.
    """

    llm_config = LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    audio_config = AudioConfig(
        asr_model=os.getenv("ASR_MODEL", "whisper-small"),
        tts_voice=os.getenv("TTS_VOICE", "alloy"),
        sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", 16_000)),
    )

    data_paths = DataPaths()
    return AppConfig(
        llm=llm_config,
        audio=audio_config,
        data_paths=data_paths,
        tracing_enabled=os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
    )


# Ensure os is imported after type hints to avoid circular imports.
import os  # noqa: E402
