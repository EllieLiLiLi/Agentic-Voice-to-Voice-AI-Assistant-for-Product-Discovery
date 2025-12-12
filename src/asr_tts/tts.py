# backend/tts.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def synthesize_speech(text: str, voice: str = "alloy") -> bytes:
    """
    TTS: use OpenAI TTS to synthesize speech from text.
    Returns raw audio bytes (mp3).
    """
    if not text or text.strip() == "":
        raise ValueError("Input text for TTS is empty.")

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",   # or "tts-1" / "gpt-4o-realtime-tts" depending on docs
        voice=voice,
        input=text
    )
    # response is a streamed object; use .read() or .to_bytes()
    audio_bytes = response.read()  # with new client this returns the full bytes
    return audio_bytes
