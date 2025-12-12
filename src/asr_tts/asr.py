# backend/asr.py
import io
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcribe_audio(file_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    ASR: use OpenAI Whisper API to transcribe audio to text.
    file_bytes: raw bytes of uploaded audio
    """
    # Wrap bytes as a file-like object
    audio_file = io.BytesIO(file_bytes)
    audio_file.name = filename  # OpenAI client expects a name

    transcript = client.audio.transcriptions.create(
        model="whisper-1",   # or other Whisper model if available
        file=audio_file,
        language=None,       # let Whisper auto-detect; or "en" / "zh"
        response_format="json"
    )
    # transcript.text is the recognized content
    return transcript.text
