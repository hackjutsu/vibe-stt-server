import base64
import logging
import os
import time
from functools import lru_cache
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from faster_whisper import WhisperModel

logger = logging.getLogger("vibe_stt_server")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

EXPECTED_SAMPLE_RATE = 16000


class Settings(BaseModel):
    model_name: str = Field(default_factory=lambda: os.getenv("WHISPER_MODEL", "large-v3-turbo"))
    device: str = Field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "auto"))
    compute_type: str = Field(default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "auto"))
    cpu_threads: int = Field(default_factory=lambda: int(os.getenv("WHISPER_CPU_THREADS", "0")))
    num_workers: int = Field(default_factory=lambda: int(os.getenv("WHISPER_NUM_WORKERS", "1")))
    default_beam_size: int = Field(default_factory=lambda: int(os.getenv("WHISPER_BEAM_SIZE", "5")))
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_model() -> WhisperModel:
    settings = get_settings()
    logger.info(
        "Loading Whisper model %s (device=%s, compute_type=%s, cpu_threads=%s, num_workers=%s)",
        settings.model_name,
        settings.device,
        settings.compute_type,
        settings.cpu_threads,
        settings.num_workers,
    )
    return WhisperModel(
        model_size_or_path=settings.model_name,
        device=settings.device,
        compute_type=settings.compute_type,
        cpu_threads=settings.cpu_threads,
        num_workers=settings.num_workers,
    )


class TranscribeRequest(BaseModel):
    audio: str = Field(..., description="Base64-encoded mono 16 kHz int16 PCM audio")
    sample_rate: int = Field(..., description="Sample rate of the provided audio (expected 16000)")
    language: Optional[str] = Field(None, description="Language code, e.g., 'en'")
    initial_prompt: Optional[str] = Field(None, description="Optional initial prompt to guide decoding")
    beam_size: Optional[int] = Field(None, description="Beam size for decoding")


class TranscriptionInfo(BaseModel):
    language: Optional[str]
    duration_ms: Optional[float]
    decode_time_ms: float
    num_samples: int


class TranscribeResponse(BaseModel):
    text: str
    info: Optional[TranscriptionInfo]


app = FastAPI(title="Vibe STT Server", version="0.1.0")


@app.on_event("startup")
async def _warm_model() -> None:
    # Trigger lazy load at startup to avoid first-request latency.
    get_model()


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/info")
async def info() -> JSONResponse:
    settings = get_settings()
    return JSONResponse(
        {
            "model": settings.model_name,
            "device": settings.device,
            "compute_type": settings.compute_type,
            "sample_rate": EXPECTED_SAMPLE_RATE,
            "num_workers": settings.num_workers,
            "cpu_threads": settings.cpu_threads,
            "default_beam_size": settings.default_beam_size,
        }
    )


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest) -> TranscribeResponse:
    if request.sample_rate != EXPECTED_SAMPLE_RATE:
        raise HTTPException(status_code=400, detail=f"sample_rate must be {EXPECTED_SAMPLE_RATE}")

    try:
        audio_bytes = base64.b64decode(request.audio)
    except Exception as exc:  # broad catch to include binascii errors
        logger.exception("Failed to decode base64 audio")
        raise HTTPException(status_code=400, detail="Invalid base64 audio") from exc

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    if audio_np.size == 0:
        raise HTTPException(status_code=400, detail="Audio payload is empty")

    audio = audio_np.astype(np.float32) / 32768.0

    settings = get_settings()
    model = get_model()

    beam_size = request.beam_size or settings.default_beam_size

    start = time.perf_counter()
    segments, model_info = model.transcribe(
        audio=audio,
        beam_size=beam_size,
        language=request.language,
        initial_prompt=request.initial_prompt,
    )
    decode_time_ms = (time.perf_counter() - start) * 1000

    text = "".join(segment.text for segment in segments).strip()

    info = TranscriptionInfo(
        language=getattr(model_info, "language", None),
        duration_ms=(getattr(model_info, "duration", None) or 0) * 1000,
        decode_time_ms=decode_time_ms,
        num_samples=int(audio_np.size),
    )

    logger.info("Transcribed %d samples in %.2f ms", audio_np.size, decode_time_ms)

    return TranscribeResponse(text=text, info=info)


if __name__ == "__main__":
    settings = get_settings()
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
