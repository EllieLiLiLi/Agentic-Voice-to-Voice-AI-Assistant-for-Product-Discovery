"""Streamlit UI for the agentic voice-to-voice assistant."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import streamlit as st

from src.asr_tts.asr import ASRClient
from src.asr_tts.tts import TTSClient
from src.config import get_config


def _init_clients() -> tuple[ASRClient, TTSClient]:
    cfg = get_config()
    return ASRClient(cfg.audio.asr_model, cfg.audio.sample_rate), TTSClient(
        cfg.audio.tts_voice, cfg.audio.sample_rate
    )


def render_audio_recorder() -> Optional[bytes]:
    """Render mic capture widget and return recorded audio bytes."""

    audio_bytes = st.audio_input("Record a query", key="mic")
    return audio_bytes


def app() -> None:
    """Compose the Streamlit layout for the assistant."""

    st.set_page_config(page_title="Voice Product Assistant", layout="wide")
    st.title("Agentic Voice-to-Voice Product Discovery")

    asr_client, tts_client = _init_clients()

    with st.sidebar:
        st.header("Controls")
        top_k = st.slider("Top-k products", min_value=1, max_value=10, value=5)
        st.write("TODO: Add model selection and tracing toggles")

    audio_bytes = render_audio_recorder()
    transcript: Optional[str] = None
    if audio_bytes:
        # TODO: Save bytes to temp file for ASR model.
        transcript = asr_client.transcribe(Path("/tmp/recording.wav"))

    st.subheader("Transcript")
    st.write(transcript or "Waiting for audio...")

    st.subheader("Agent step log")
    st.write("TODO: display LangGraph events and MCP calls")

    st.subheader("Comparison table")
    st.write("TODO: render top-k retrieved products")

    if st.button("Play TTS"):
        if transcript:
            audio_path = tts_client.synthesize(transcript, Path("/tmp/output.wav"))
            with open(audio_path, "rb") as f:
                st.audio(io.BytesIO(f.read()), format="audio/wav")
        else:
            st.warning("No transcript available for TTS")


if __name__ == "__main__":  # pragma: no cover
    app()
