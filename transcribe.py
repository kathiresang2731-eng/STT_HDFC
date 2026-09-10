"""
Bulk audio transcription + translation using faster-whisper.

For every audio file in AUDIO_FOLDER, this produces TWO output files:
  1. <name>.txt      -> transcript in the ORIGINAL spoken language
  2. <name>_en.txt   -> ENGLISH translation (Whisper's built-in translate task,
                        works from any supported source language into English)

Install first:
    pip install faster-whisper
"""

import os
from pathlib import Path
from faster_whisper import WhisperModel

# ----- CONFIGURE THESE -----
AUDIO_FOLDER = r"C:\Users\Personal\Downloads\Today_testing"
OUTPUT_FOLDER = "transcripts"
MODEL_SIZE = "medium"            # tiny/base/small/medium/large-v3
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
BEAM_SIZE = 5
# ----------------------------

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4", ".aac"}


def run_task(model, audio_path, task):
    """Runs the model once with the given task ('transcribe' or 'translate')."""
    segments, info = model.transcribe(str(audio_path), beam_size=BEAM_SIZE, task=task)
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip(), info


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"Loading model '{MODEL_SIZE}'..... on {DEVICE}...... ({COMPUTE_TYPE})...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    print("entered Into audio folder...")
    audio_files = sorted(
        
        f for f in Path(AUDIO_FOLDER).iterdir()
        if f.suffix.lower() in AUDIO_EXTENSIONS
    )
    print("come out from the audio folder")

    if not audio_files:
        print(f"No audio files found in '{AUDIO_FOLDER}'.")
        return

    print(f"Found {len(audio_files)} audio file(s). Starting transcription + translation...\n")

    for i, audio_path in enumerate(audio_files, 1):
        transcript_path = Path(OUTPUT_FOLDER) / f"{audio_path.stem}.txt"
        translation_path = Path(OUTPUT_FOLDER) / f"{audio_path.stem}_en.txt"

        if transcript_path.exists() and translation_path.exists():
            print(f"[{i}/{len(audio_files)}] Skipping '{audio_path.name}' (already done)")
            continue

        print(f"[{i}/{len(audio_files)}] Processing '{audio_path.name}'...")

        try:
            # Pass 1: native-language transcript
            transcript_text, info = run_task(model, audio_path, task="transcribe")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            print(f"    -> Transcript saved '{transcript_path.name}' (language: {info.language})")

            # Pass 2: English translation (skip re-running if source is already English)
            if info.language == "en":
                translation_text = transcript_text
            else:
                translation_text, _ = run_task(model, audio_path, task="translate")

            with open(translation_path, "w", encoding="utf-8") as f:
                f.write(translation_text)
            print(f"    -> English translation saved '{translation_path.name}'")

        except Exception as e:
            print(f"    !! Failed to process '{audio_path.name}': {e}")

    print("\nDone. Transcripts + translations saved to:", OUTPUT_FOLDER)

if __name__ == "__main__":
    main()