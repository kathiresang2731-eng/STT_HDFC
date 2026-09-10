"""
FastAPI Backend Server for VoiceBot STT & Translation Pipeline
Provides REST API endpoints for uploading audio files, running high-accuracy
transcription and translation with faster-whisper and Silero VAD, and retrieving history.
"""

import os
import sys
import time
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure translate.py can be imported
CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import translate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("stt-api")

app = FastAPI(
    title="VoiceBot Speech-to-Text & Translation API",
    description="REST API for faster-whisper + Silero VAD transcription and translation",
    version="1.0.0",
)

# Enable CORS for frontend development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRANSCRIPTS_DIR = CURRENT_DIR / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Cache loaded translators by model size
_TRANSLATORS: Dict[str, translate.SpeechTranslator] = {}


def get_translator_for_model(model_size: str = "medium") -> translate.SpeechTranslator:
    """Returns or creates a SpeechTranslator for the specified model size."""
    size = (model_size or "medium").lower().strip()
    if size not in ["tiny", "base", "small", "medium", "large-v3"]:
        size = "medium"
    
    if size not in _TRANSLATORS:
        logger.info("Initializing SpeechTranslator instance for model '%s'...", size)
        _TRANSLATORS[size] = translate.SpeechTranslator(model_size=size)
    return _TRANSLATORS[size]


@app.get("/api/health")
def health_check():
    """Health check endpoint showing active configuration."""
    return {
        "status": "healthy",
        "default_model": translate.MODEL_SIZE,
        "device": translate.DEVICE,
        "compute_type": translate.COMPUTE_TYPE,
        "ffmpeg_available": translate.FFMPEG_EXE is not None,
        "cached_models": list(_TRANSLATORS.keys()),
    }


@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    model_size: Optional[str] = Form("medium"),
    force_language: Optional[str] = Form(None),
):
    """
    Receives an audio file upload, performs dual-pass STT & translation,
    and returns comprehensive structured results.
    """
    start_time = time.perf_counter()
    filename = file.filename or "uploaded_audio.wav"
    logger.info("Received audio upload: '%s' (model: %s, forced_lang: %s)", filename, model_size, force_language)

    # Read uploaded bytes into memory
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # Choose translator based on user's model selection
        translator = get_translator_for_model(model_size or "medium")

        # Process audio using translate pipeline
        result = translator.process_audio_source(content, filename=filename)

        # Generate report text
        model_info = {
            "model_size": translator.model_size,
            "device": translator.device,
            "compute_type": translator.compute_type,
            "beam_size": translate.BEAM_SIZE,
        }
        report_text = translate.generate_report(result, model_info)
        result["report_text"] = report_text

        # Automatically save transcript report to disk
        save_stem = Path(filename).stem
        report_path = TRANSCRIPTS_DIR / f"{save_stem}.txt"
        translate.save_text_safe(report_path, report_text)
        result["saved_report_path"] = str(report_path)

        logger.info(
            "Completed '%s' in %.2fs. Detected: %s (confidence: %.2f%%)",
            filename,
            time.perf_counter() - start_time,
            result.get("detected_language"),
            (result.get("language_confidence") or 0.0) * 100,
        )

        return JSONResponse(content=result)

    except Exception as exc:
        logger.exception("Error processing audio '%s': %s", filename, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/history")
def list_history():
    """Lists recent transcript reports from the transcripts directory."""
    if not TRANSCRIPTS_DIR.exists():
        return []

    items = []
    for txt_file in sorted(TRANSCRIPTS_DIR.glob("*.txt"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            stat = txt_file.stat()
            # Read to parse summary info
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            lang = "Unknown"
            status = "UNKNOWN"
            for line in content.splitlines():
                if "Detected Language" in line and ":" in line:
                    lang = line.split(":", 1)[1].strip()
                elif "Overall Status" in line and ":" in line:
                    status = line.split(":", 1)[1].strip()

            items.append({
                "filename": txt_file.name,
                "stem": txt_file.stem,
                "size_bytes": stat.st_size,
                "modified_time": stat.st_mtime,
                "detected_language": lang,
                "status": status,
            })
        except Exception:
            continue

    return items


@app.get("/api/history/{filename}")
def get_history_report(filename: str):
    """Retrieves the full report text for a specific historical transcription."""
    file_path = (TRANSCRIPTS_DIR / filename).resolve()
    # Security check: ensure path is inside TRANSCRIPTS_DIR
    if not str(file_path).startswith(str(TRANSCRIPTS_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Access denied.")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found.")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return {"filename": filename, "report_text": content}


@app.get("/api/samples")
def list_demo_samples():
    """Lists available demo audio samples from the testing directories."""
    sample_dirs = [
        Path(r"C:\Users\Personal\Downloads\Today_testing"),
        Path(r"C:\Users\Personal\Downloads\audio"),
        translate.AUDIO_FOLDER,
    ]
    seen = set()
    samples = []
    for s_dir in sample_dirs:
        if s_dir.exists() and s_dir.is_dir():
            for f in sorted(s_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in translate.AUDIO_EXTENSIONS and f.name not in seen:
                    seen.add(f.name)
                    # Label based on dialect
                    dialect_label = "Telugu" if "3" in f.stem else ("Tamil" if "tamil" in f.stem.lower() else "English")
                    samples.append({
                        "filename": f.name,
                        "stem": f.stem,
                        "size_bytes": f.stat().st_size,
                        "label": f"{dialect_label} ({f.name})",
                    })
    return samples


@app.get("/api/samples/{filename}")
def get_sample_audio(filename: str):
    """Serves the sample audio file for instant browser playback."""
    search_dirs = [
        Path(r"C:\Users\Personal\Downloads\Today_testing"),
        Path(r"C:\Users\Personal\Downloads\audio"),
        translate.AUDIO_FOLDER,
    ]
    for s_dir in search_dirs:
        candidate = s_dir / filename
        if candidate.exists():
            media_type = "audio/ogg" if candidate.suffix.lower() == ".ogg" else ("audio/wav" if candidate.suffix.lower() == ".wav" else "audio/mpeg")
            return FileResponse(candidate, media_type=media_type)
    raise HTTPException(status_code=404, detail="Sample audio not found.")




# Mount built React frontend if available
FRONTEND_DIST = CURRENT_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static_frontend")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting VoiceBot STT API server on http://localhost:8000...")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)

