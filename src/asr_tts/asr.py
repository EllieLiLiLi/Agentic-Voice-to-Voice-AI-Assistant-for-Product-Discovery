"""Automatic speech recognition (ASR) interface."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


class ASRClient:
    """Thin wrapper around the chosen ASR provider (e.g., Whisper)."""

    def __init__(self, model: str, sample_rate: int) -> None:
        self.model = model
        self.sample_rate = sample_rate

    def transcribe(self, audio_path: Path) -> Optional[str]:
        """Transcribe an audio file into text.

        TODO: Implement provider-specific logic and streaming/fragment-based
        transcription to reduce latency.
        """

        # Placeholder implementation.
        return None
