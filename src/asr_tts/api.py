# backend/api.py
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from asr import transcribe_audio
from tts import synthesize_speech

app = FastAPI(title="ASR + TTS Service")

# CORS: allow your frontend origin in dev; for now allow all
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # in production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/asr")
async def asr_endpoint(file: UploadFile = File(...)):
    """
    Upload an audio file and get back a transcription.
    """
    try:
        file_bytes = await file.read()
        text = transcribe_audio(file_bytes, filename=file.filename)
        return JSONResponse({"text": text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR failed: {e}")


@app.post("/api/tts")
async def tts_endpoint(payload: dict):
    """
    Provide text and optional voice, get back an MP3 audio stream.
    Body example: { "text": "...", "voice": "alloy" }
    """
    text = payload.get("text", "")
    voice = payload.get("voice", "alloy")
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' is required.")

    try:
        audio_bytes = synthesize_speech(text, voice=voice)
        audio_stream = io.BytesIO(audio_bytes)

        return StreamingResponse(
            audio_stream,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": 'inline; filename="speech.mp3"'
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")
