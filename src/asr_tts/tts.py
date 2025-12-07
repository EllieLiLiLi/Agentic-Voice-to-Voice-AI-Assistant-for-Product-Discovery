"""Text-to-speech (TTS) interface."""
from __future__ import annotations

from pathlib import Path


class TTSClient:
    """Thin wrapper around the chosen TTS provider."""

    def __init__(self, voice: str, sample_rate: int) -> None:
        self.voice = voice
        self.sample_rate = sample_rate

    def synthesize(self, text: str, output_path: Path) -> Path:
        """Convert text to speech and save to ``output_path``.

        TODO: Implement provider-specific synthesis and streaming playback.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # TODO: Write actual audio bytes.
        output_path.touch()
        return output_path
