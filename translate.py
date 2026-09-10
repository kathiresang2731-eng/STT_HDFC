"""
High-Accuracy Speech-to-Text (STT) & Translation Pipeline
Powered by faster-whisper and native Silero Voice Activity Detection (VAD).

Supports:
- Batch directory processing
- Single file / File upload processing (via stream, bytes, or file path)
- Native Silero VAD sentence/phrase boundary detection (no mid-word splitting)
- Dual-pass transcription and timestamped English translation
- Memory-safe CPU execution preventing MKL allocation crashes
- Comprehensive analysis report generation
"""

import io
import os
import sys
import time
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union, BinaryIO, List, Dict, Any

# Prevent Intel MKL / OpenMP memory exhaustion and crashes on Windows CPU
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions

# ==============================================================================
# CONFIGURATION
# ==============================================================================

AUDIO_FOLDER = Path(r"C:\Users\Personal\Downloads\audio")
OUTPUT_FOLDER = Path("transcripts")

# Model options: "tiny", "base", "small", "medium", "large-v3"
# "small" is highly recommended for 8GB RAM systems: fast (~10-15s), ~500MB RAM.
# "medium" is supported with memory guards (cpu_threads=2, int8).
MODEL_SIZE = "medium"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CPU_THREADS = 2
NUM_WORKERS = 1

BEAM_SIZE = 5

# Voice Activity Detection (Silero VAD) configuration
VAD_FILTER = True
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_DURATION_MS = 500   # Min silence to split natural sentences
VAD_SPEECH_PAD_MS = 300            # Padding around speech to avoid clipping

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".flac",
    ".mp4",
    ".aac",
    ".webm",
}

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("whisper-stt")


# ==============================================================================
# FFMPEG & ENVIRONMENT DISCOVERY
# ==============================================================================

def setup_ffmpeg_environment() -> tuple[Optional[Path], Optional[Path]]:
    """
    Dynamically finds FFmpeg and FFprobe binaries and ensures their directory
    is added to os.environ['PATH'] so faster-whisper and subprocesses find it.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    if ffmpeg and ffprobe:
        ffmpeg_path = Path(ffmpeg).resolve()
        ffprobe_path = Path(ffprobe).resolve()
        bin_dir = str(ffmpeg_path.parent)
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        return ffmpeg_path, ffprobe_path

    # Search common Windows package locations
    search_roots = [
        Path(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")),
        Path(os.path.expandvars(r"%PROGRAMDATA%\chocolatey\bin")),
        Path(os.path.expandvars(r"%USERPROFILE%\scoop\shims")),
        Path(os.path.expandvars(r"C:\Program Files\ffmpeg\bin")),
    ]

    for root in search_roots:
        if root.exists():
            for ffmpeg_candidate in root.glob("**/ffmpeg.exe"):
                ffprobe_candidate = ffmpeg_candidate.parent / "ffprobe.exe"
                if ffprobe_candidate.exists():
                    bin_dir = str(ffmpeg_candidate.parent.resolve())
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                    logger.info("Found FFmpeg at: %s", bin_dir)
                    return ffmpeg_candidate.resolve(), ffprobe_candidate.resolve()

    logger.warning("FFmpeg / FFprobe not found automatically. Please ensure it is in PATH.")
    return None, None


FFMPEG_EXE, FFPROBE_EXE = setup_ffmpeg_environment()


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def format_duration(seconds: Optional[float]) -> str:
    """Formats float seconds into HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "Unknown"
    secs_int = int(seconds)
    hours, remainder = divmod(secs_int, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp(seconds: float) -> str:
    """Formats float seconds into HH:MM:SS.mmm."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3600000
    milliseconds %= 3600000
    minutes = milliseconds // 60000
    milliseconds %= 60000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def save_text_safe(path: Path, text: str, max_retries: int = 3):
    """Saves text with atomic rename and Windows permission retry support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f".tmp_{int(time.time() * 1000)}")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")

        for attempt in range(max_retries):
            try:
                temp_path.replace(path)
                return
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.2)
                else:
                    # Direct write fallback
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(text.rstrip() + "\n")
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def get_audio_duration(audio_path: Path) -> float:
    """Extracts audio duration with fallback from container format to audio stream."""
    if not FFPROBE_EXE or not FFPROBE_EXE.exists():
        return 0.0

    # 1. Try container format duration
    cmd_format = [
        str(FFPROBE_EXE),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        res = subprocess.run(cmd_format, capture_output=True, text=True, check=True)
        val = res.stdout.strip()
        if val and val != "N/A":
            return float(val)
    except Exception:
        pass

    # 2. Fallback to audio stream duration
    cmd_stream = [
        str(FFPROBE_EXE),
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        res = subprocess.run(cmd_stream, capture_output=True, text=True, check=True)
        val = res.stdout.strip()
        if val and val != "N/A":
            return float(val)
    except Exception:
        pass

    return 0.0


def is_already_successfully_processed(output_path: Path) -> bool:
    """Checks if output exists and contains a successful run (does not skip failed runs)."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        return False
    try:
        with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if "Overall Status         : SUCCESS" in content:
                return True
    except Exception:
        return False
    return False


def get_audio_files(folder: Path) -> List[Path]:
    """Scans folder for supported audio files."""
    if not folder.exists():
        raise FileNotFoundError(f"Audio folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Audio path is not a directory: {folder}")

    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )


# ==============================================================================
# SPEECH TRANSLATOR PIPELINE
# ==============================================================================

class SpeechTranslator:
    """
    End-to-end Speech-to-Text and Translation engine with native Silero VAD,
    memory optimization, and multi-format audio input support.
    """

    def __init__(
        self,
        model_size: str = MODEL_SIZE,
        device: str = DEVICE,
        compute_type: str = COMPUTE_TYPE,
        cpu_threads: int = CPU_THREADS,
        num_workers: int = NUM_WORKERS,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self._model: Optional[WhisperModel] = None

    def get_model(self) -> WhisperModel:
        """Lazily initializes and caches the WhisperModel instance."""
        if self._model is None:
            logger.info(
                "Loading Whisper model '%s' on %s (%s, cpu_threads=%d)...",
                self.model_size,
                self.device,
                self.compute_type,
                self.cpu_threads,
            )
            start_t = time.perf_counter()
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=self.num_workers,
            )
            logger.info("Whisper model loaded in %.2fs.", time.perf_counter() - start_t)
        return self._model

    def process_audio_source(
        self,
        audio_source: Union[str, Path, BinaryIO, bytes],
        filename: str = "audio_file",
    ) -> Dict[str, Any]:
        """
        Transcribes and translates an audio source (file path, file-like object, or bytes).
        Returns a rich structured dictionary suitable for APIs, UI rendering, and report creation.
        """
        temp_audio_file = None
        start_time = time.perf_counter()

        try:
            # Handle in-memory bytes or stream uploads by writing to temporary file
            if isinstance(audio_source, (bytes, bytearray)):
                temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_audio_file.write(audio_source)
                temp_audio_file.flush()
                temp_audio_file.close()
                target_path = Path(temp_audio_file.name)
            elif isinstance(audio_source, (io.BufferedIOBase, io.RawIOBase)) or hasattr(audio_source, "read"):
                temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                shutil.copyfileobj(audio_source, temp_audio_file)
                temp_audio_file.flush()
                temp_audio_file.close()
                target_path = Path(temp_audio_file.name)
            else:
                target_path = Path(audio_source)
                if not target_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {target_path}")

            audio_duration = get_audio_duration(target_path)
            model = self.get_model()

            vad_params = dict(
                threshold=VAD_THRESHOLD,
                min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
                speech_pad_ms=VAD_SPEECH_PAD_MS,
            )

            # ------------------------------------------------------------------
            # PASS 1: Native VAD Transcription & Robust Language Detection
            # ------------------------------------------------------------------
            logger.info("Transcribing '%s' with native Silero VAD...", filename)
            transcribe_segments, info = model.transcribe(
                str(target_path),
                task="transcribe",
                beam_size=BEAM_SIZE,
                vad_filter=VAD_FILTER,
                vad_parameters=vad_params,
                hallucination_silence_threshold=2.0,
                no_speech_threshold=0.6,
                repetition_penalty=1.1,
                condition_on_previous_text=False,
                temperature=0.0,
                language_detection_segments=2,
            )

            transcription_segments = []
            for seg in transcribe_segments:
                txt = seg.text.strip()
                if txt:
                    transcription_segments.append({
                        "id": seg.id,
                        "start": seg.start,
                        "end": seg.end,
                        "text": txt,
                        "avg_logprob": getattr(seg, "avg_logprob", 0.0),
                        "no_speech_prob": getattr(seg, "no_speech_prob", 0.0),
                    })

            detected_language = getattr(info, "language", "unknown")
            language_confidence = getattr(info, "language_probability", 0.0)
            if audio_duration == 0.0 and getattr(info, "duration", 0.0):
                audio_duration = float(info.duration)

            # --------------------------------------------------------------
            # Language‑specific handling: enforce Telugu if auto‑detected
            # --------------------------------------------------------------
            forced_rerun = False
            if detected_language == "te":
                # Heuristic: if all transcription text is ASCII, assume mis‑detected and re‑run with forced Telugu
                if all(all(ord(ch) < 128 for ch in seg["text"]) for seg in transcription_segments):
                    logger.info("Detected Telugu but transcription appears English; re‑running with forced language='te'.")
                    # Use a smaller model for the forced run to reduce memory pressure
                    try:
                        small_model = WhisperModel(
                            "small",
                            device=self.device,
                            compute_type=self.compute_type,
                            cpu_threads=1,
                            num_workers=1,
                        )
                        forced_segments, _ = small_model.transcribe(
                            str(target_path),
                            task="transcribe",
                            language="te",
                            beam_size=BEAM_SIZE,
                            vad_filter=VAD_FILTER,
                            vad_parameters=vad_params,
                            hallucination_silence_threshold=2.0,
                            no_speech_threshold=0.6,
                            repetition_penalty=1.1,
                            condition_on_previous_text=False,
                            temperature=0.0,
                        )
                        transcription_segments = []
                        for seg in forced_segments:
                            txt = seg.text.strip()
                            if txt:
                                transcription_segments.append({
                                    "id": seg.id,
                                    "start": seg.start,
                                    "end": seg.end,
                                    "text": txt,
                                    "avg_logprob": getattr(seg, "avg_logprob", 0.0),
                                    "no_speech_prob": getattr(seg, "no_speech_prob", 0.0),
                                })
                        forced_rerun = True
                    except Exception as e:
                        logger.error("Forced Telugu re‑run failed: %s", e)
                    finally:
                        # Ensure small model resources are released
                        del small_model

            full_transcription = " ".join(s["text"] for s in transcription_segments).strip()
            logger.info(
                "Transcription complete. Language: %s (confidence: %.2f%%). Segments: %d",
                detected_language,
                (language_confidence or 0.0) * 100,
                len(transcription_segments),
            )

            # ------------------------------------------------------------------
            # PASS 2: Segment-Level English Translation (if source is not English)
            # ------------------------------------------------------------------
            translation_segments = []
            if detected_language.lower() != "en" and transcription_segments:
                logger.info("Translating '%s' (%s -> en)...", filename, detected_language)
                translate_segs, _ = model.transcribe(
                    str(target_path),
                    task="translate",
                    language=detected_language,
                    beam_size=BEAM_SIZE,
                    vad_filter=VAD_FILTER,
                    vad_parameters=vad_params,
                    hallucination_silence_threshold=2.0,
                    no_speech_threshold=0.6,
                    repetition_penalty=1.1,
                    condition_on_previous_text=False,
                    temperature=0.0,
                )
                for seg in translate_segs:
                    txt = seg.text.strip()
                    if txt:
                        translation_segments.append({
                            "id": seg.id,
                            "start": seg.start,
                            "end": seg.end,
                            "text": txt,
                        })
                full_translation = " ".join(s["text"] for s in translation_segments).strip()
            elif detected_language.lower() == "en":
                translation_segments = [
                    {"id": s["id"], "start": s["start"], "end": s["end"], "text": s["text"]}
                    for s in transcription_segments
                ]
                full_translation = full_transcription
            else:
                full_translation = ""

            processing_time = time.perf_counter() - start_time

            return {
                "filename": filename,
                "file_path": str(target_path),
                "file_size_bytes": target_path.stat().st_size if target_path.exists() else 0,
                "audio_duration_seconds": audio_duration,
                "detected_language": detected_language,
                "language_confidence": language_confidence,
                "full_transcription": full_transcription,
                "full_translation": full_translation,
                "transcription_segments": transcription_segments,
                "translation_segments": translation_segments,
                "processing_time_seconds": processing_time,
                "status": "SUCCESS",
                "error": "",
                "forced_rerun": forced_rerun,
            }

        except Exception as exc:
            logger.exception("Failed processing '%s': %s", filename, exc)
            return {
                "filename": filename,
                "file_path": str(audio_source) if isinstance(audio_source, (str, Path)) else filename,
                "file_size_bytes": 0,
                "audio_duration_seconds": 0.0,
                "detected_language": "unknown",
                "language_confidence": 0.0,
                "full_transcription": "",
                "full_translation": "",
                "transcription_segments": [],
                "translation_segments": [],
                "processing_time_seconds": time.perf_counter() - start_time,
                "status": "FAILED",
                "error": str(exc),
            }

        finally:
            if temp_audio_file:
                try:
                    os.unlink(temp_audio_file.name)
                except Exception:
                    pass


# ==============================================================================
# REPORT GENERATOR
# ==============================================================================

def generate_report(result: Dict[str, Any], model_info: Dict[str, Any]) -> str:
    """Formats the transcription & translation results into a clean, comprehensive text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("WHISPER STT & TRANSLATION ANALYSIS")
    lines.append("=" * 80)

    lines.append("")
    lines.append("[AUDIO INFORMATION]")
    lines.append("-" * 80)
    lines.append(f"File Name              : {result['filename']}")
    lines.append(f"File Path              : {result['file_path']}")
    size_mb = result["file_size_bytes"] / (1024 * 1024) if result["file_size_bytes"] else 0.0
    lines.append(f"File Size              : {size_mb:.2f} MB")
    lines.append(f"Audio Duration         : {format_duration(result['audio_duration_seconds'])}")
    lines.append(f"Speech Segmentation    : Native Silero VAD (Sentence & Pause Boundaries)")
    lines.append(f"Total Speech Segments  : {len(result['transcription_segments'])}")

    lines.append("")
    lines.append("[MODEL INFORMATION]")
    lines.append("-" * 80)
    lines.append(f"Model                  : {model_info.get('model_size', MODEL_SIZE)}")
    lines.append(f"Device                 : {model_info.get('device', DEVICE)}")
    lines.append(f"Compute Type           : {model_info.get('compute_type', COMPUTE_TYPE)}")
    lines.append(f"Beam Size              : {model_info.get('beam_size', BEAM_SIZE)}")
    lines.append(f"VAD Enabled            : {VAD_FILTER}")
    lines.append(f"VAD Min Silence        : {VAD_MIN_SILENCE_DURATION_MS} ms")
    lines.append(f"VAD Speech Pad         : {VAD_SPEECH_PAD_MS} ms")

    lines.append("")
    lines.append("[LANGUAGE DETECTION]")
    lines.append("-" * 80)
    lines.append(f"Detected Language      : {result['detected_language']}")
    conf = result.get("language_confidence")
    lines.append(f"Language Confidence    : {f'{conf:.4f}' if conf is not None else 'Unknown'}")

    lines.append("")
    lines.append("[FULL TRANSCRIPTS]")
    lines.append("-" * 80)
    lines.append("Original Transcription:")
    lines.append(result["full_transcription"] if result["full_transcription"] else "[No speech detected]")
    lines.append("")
    lines.append("English Translation:")
    lines.append(result["full_translation"] if result["full_translation"] else "[No speech detected]")

    lines.append("")
    lines.append("[SEGMENT DETAILS]")
    lines.append("-" * 80)

    t_segs = result.get("transcription_segments", [])
    tr_segs = result.get("translation_segments", [])

    if not t_segs:
        lines.append("[No speech segments detected]")
    else:
        for i, seg in enumerate(t_segs, 1):
            lines.append("")
            lines.append(f"SEGMENT {i:04d}")
            lines.append(f"Time                   : {format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
            lines.append(f"Original Text          : {seg['text']}")

            # Match translation segment if available
            trans_text = ""
            if i - 1 < len(tr_segs):
                trans_text = tr_segs[i - 1]["text"]
            elif result["detected_language"].lower() == "en":
                trans_text = seg["text"]

            if trans_text:
                lines.append(f"English Translation    : {trans_text}")

    lines.append("")
    lines.append("[PROCESSING INFORMATION]")
    lines.append("-" * 80)
    lines.append(f"Processing Time        : {result['processing_time_seconds']:.2f} seconds")
    lines.append(f"Overall Status         : {result['status']}")
    if result.get("error"):
        lines.append(f"Error                  : {result['error']}")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


# ==============================================================================
# MODULAR FILE UPLOAD & DIRECT INGESTION
# ==============================================================================

_DEFAULT_TRANSLATOR: Optional[SpeechTranslator] = None


def get_default_translator() -> SpeechTranslator:
    """Returns a shared SpeechTranslator instance."""
    global _DEFAULT_TRANSLATOR
    if _DEFAULT_TRANSLATOR is None:
        _DEFAULT_TRANSLATOR = SpeechTranslator()
    return _DEFAULT_TRANSLATOR


def process_upload(
    audio_file: Union[str, Path, BinaryIO, bytes],
    filename: Optional[str] = None,
    save_transcript_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Entry point for file uploads (compatible with FastAPI, Flask, Streamlit, etc.).
    
    Args:
        audio_file: A file path, open binary file handle, or raw bytes.
        filename: Optional filename for display/logging.
        save_transcript_dir: Optional folder to save the .txt report.

    Returns:
        dict: Complete structured result and formatted text report.
    """
    if filename is None:
        if isinstance(audio_file, (str, Path)):
            filename = Path(audio_file).name
        elif hasattr(audio_file, "name"):
            filename = Path(audio_file.name).name
        else:
            filename = "uploaded_audio.wav"

    translator = get_default_translator()
    result = translator.process_audio_source(audio_file, filename=filename)

    model_info = {
        "model_size": translator.model_size,
        "device": translator.device,
        "compute_type": translator.compute_type,
        "beam_size": BEAM_SIZE,
    }
    report_text = generate_report(result, model_info)
    result["report_text"] = report_text

    if save_transcript_dir:
        save_folder = Path(save_transcript_dir)
        save_folder.mkdir(parents=True, exist_ok=True)
        report_path = save_folder / f"{Path(filename).stem}.txt"
        save_text_safe(report_path, report_text)
        result["saved_report_path"] = str(report_path)

    return result


# ==============================================================================
# BATCH DIRECTORY RUNNER
# ==============================================================================

def run_batch():
    """Processes all audio files in AUDIO_FOLDER and saves structured reports to OUTPUT_FOLDER."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("WHISPER STT & TRANSLATION BATCH PROCESSING")
    logger.info("=" * 80)

    try:
        audio_files = get_audio_files(AUDIO_FOLDER)
    except Exception as exc:
        logger.error("Audio folder error: %s", exc)
        return

    if not audio_files:
        logger.warning("No supported audio files found in: %s", AUDIO_FOLDER)
        return

    logger.info("Found %d audio file(s) in: %s", len(audio_files), AUDIO_FOLDER)
    translator = SpeechTranslator()

    # Preload model
    translator.get_model()

    model_info = {
        "model_size": translator.model_size,
        "device": translator.device,
        "compute_type": translator.compute_type,
        "beam_size": BEAM_SIZE,
    }

    statuses = []

    for idx, audio_path in enumerate(audio_files, 1):
        output_path = OUTPUT_FOLDER / f"{audio_path.stem}.txt"

        if is_already_successfully_processed(output_path):
            logger.info("[%d/%d] Skipping %s (already completed successfully).", idx, len(audio_files), audio_path.name)
            statuses.append("SKIPPED")
            continue

        logger.info("[%d/%d] Processing %s...", idx, len(audio_files), audio_path.name)
        result = translator.process_audio_source(audio_path, filename=audio_path.name)

        report = generate_report(result, model_info)
        save_text_safe(output_path, report)
        logger.info("Saved report: %s", output_path)

        statuses.append(result["status"])

    logger.info("")
    logger.info("=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info("Total Files : %d", len(audio_files))
    logger.info("Success     : %d", statuses.count("SUCCESS"))
    logger.info("Skipped     : %d", statuses.count("SKIPPED"))
    logger.info("Failed      : %d", statuses.count("FAILED"))
    logger.info("Output      : %s", OUTPUT_FOLDER.resolve())


if __name__ == "__main__":
    run_batch()