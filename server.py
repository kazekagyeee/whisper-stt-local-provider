"""
Whisper STT Local Provider
OpenAI-compatible STT server using faster-whisper

Model: faster-whisper (ctranslate2-based, much faster than OpenAI whisper)
Languages: multilingual (ru, en, etc.)
Inference: CPU or CUDA (auto-detected)
"""

import os
import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logging.warning("faster-whisper not installed, STT will be unavailable")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config from env ──────────────────────────────────────────────────────────
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large-v3
DEFAULT_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ru")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")  # auto, float16, int8, float32
DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # auto, cpu, cuda

# ── Model loading ─────────────────────────────────────────────────────────────
whisper_model = None
model_info = {}

def load_model():
    """Lazy load the Whisper model."""
    global whisper_model, model_info
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError("faster-whisper is not installed. Run: pip install faster-whisper")
    
    if whisper_model is None:
        logger.info(f"Loading faster-whisper model: {DEFAULT_MODEL}")
        logger.info(f"Device: {DEVICE}, Compute type: {COMPUTE_TYPE}")
        
        try:
            whisper_model = WhisperModel(
                DEFAULT_MODEL,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                download_root=os.getenv("MODEL_CACHE_DIR", None),
            )
            model_info = {
                "model": DEFAULT_MODEL,
                "device": DEVICE,
                "compute_type": COMPUTE_TYPE,
                "language": DEFAULT_LANGUAGE,
            }
            logger.info(f"Model loaded successfully: {DEFAULT_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Whisper STT on {HOST}:{PORT} | model={DEFAULT_MODEL} | lang={DEFAULT_LANGUAGE}")
    try:
        load_model()
    except Exception as e:
        logger.warning(f"Model not loaded yet: {e}")
    yield
    logger.info("Server stopped")


app = FastAPI(title="Whisper STT Local Provider", lifespan=lifespan)


# ── OpenAI-compatible /v1/audio/transcriptions ────────────────────────────────

class TranscriptionRequest(BaseModel):
    model: str = "whisper-1"
    language: Optional[str] = DEFAULT_LANGUAGE
    prompt: Optional[str] = None
    temperature: Optional[float] = 0.0
    response_format: str = "json"  # json, text, srt, verbose_json
    timestamp_granularities: Optional[list[str]] = None


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = DEFAULT_LANGUAGE,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    response_format: str = "json",
):
    """
    OpenAI-compatible /v1/audio/transcriptions endpoint.
    Accepts audio file upload (wav, mp3, ogg, m4a, etc.)
    """
    if not FASTER_WHISPER_AVAILABLE or whisper_model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(503, f"Model not available: {e}")

    # Read audio data
    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(400, "Empty audio file")

    # Save to temp file (faster-whisper reads from path)
    import tempfile
    import numpy as np
    import wave
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    try:
        # Transcribe
        segments, info = whisper_model.transcribe(
            tmp_path,
            language=language if language else None,
            initial_prompt=prompt,
            temperature=temperature,
            vad_filter=True,  # voice activity detection
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        # Collect segments
        if response_format == "text":
            text = "".join([seg.text for seg in segments])
            return Response(content=text, media_type="text/plain")
        
        elif response_format == "verbose_json":
            return JSONResponse({
                "text": "".join([seg.text for seg in segments]),
                "language": info.language,
                "duration": info.duration,
                "segments": [
                    {
                        "id": i,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                    }
                    for i, seg in enumerate(segments)
                ],
            })
        
        else:  # json (default)
            text = "".join([seg.text for seg in segments])
            return JSONResponse({
                "text": text,
            })

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(500, f"Transcription failed: {e}")
    finally:
        # Cleanup
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Simple POST endpoint (custom API) ────────────────────────────────────────

@app.post("/transcribe")
async def transcribe_post(
    file: UploadFile = File(...),
    language: Optional[str] = DEFAULT_LANGUAGE,
    prompt: Optional[str] = None,
):
    """Custom /transcribe endpoint with simpler response."""
    if not FASTER_WHISPER_AVAILABLE or whisper_model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(503, f"Model not available: {e}")

    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(400, "Empty audio file")

    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    try:
        segments, info = whisper_model.transcribe(
            tmp_path,
            language=language if language else None,
            initial_prompt=prompt,
            vad_filter=True,
        )

        return {
            "text": "".join([seg.text for seg in segments]),
            "language": info.language,
            "duration": round(info.duration, 2) if info.duration else None,
            "segments": [
                {"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text}
                for seg in segments
            ],
        }
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(500, f"Transcription failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Utility endpoints ─────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "whisper-1",
            "object": "model",
            "created": 1700000000,
            "owned_by": "openai",
        }]
    }


@app.get("/models/available")
async def available_models():
    """List available whisper model sizes."""
    return {
        "models": [
            {"id": "tiny", "params": "~39M", "vram": "~1GB", "realtime_factor": "~10x"},
            {"id": "base", "params": "~74M", "vram": "~1GB", "realtime_factor": "~7x"},
            {"id": "small", "params": "~244M", "vram": "~2GB", "realtime_factor": "~5x"},
            {"id": "medium", "params": "~769M", "vram": "~5GB", "realtime_factor": "~2x"},
            {"id": "large-v3", "params": "~1550M", "vram": "~6GB", "realtime_factor": "~1x"},
        ],
        "current": {
            "model": DEFAULT_MODEL,
            "language": DEFAULT_LANGUAGE,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
        },
        "status": "loaded" if whisper_model is not None else "not_loaded",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok" if whisper_model is not None else "model_not_loaded",
        "model": model_info.get("model", DEFAULT_MODEL),
        "language": DEFAULT_LANGUAGE,
        "device": DEVICE,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
