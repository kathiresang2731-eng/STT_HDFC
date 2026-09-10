"""
VoiceBot STT & Translation Studio Launcher
Starts the FastAPI server (which serves the compiled React + Tailwind UI)
and opens the application in your default web browser.
"""

import sys
import time
import subprocess
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
FRONTEND_DIR = BASE_DIR / "frontend"

def main():
    print("=" * 70)
    print(" VoiceBot Speech-to-Text & Translation Studio")
    print("=" * 70)

    is_dev = "--dev" in sys.argv

    # 1. Start FastAPI backend (serves API & compiled React UI)
    print("\n[1/2] Starting Studio server on http://localhost:8000...")
    backend_cmd = [
        sys.executable,
        "-m", "uvicorn",
        "api_server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
    ]
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(BASE_DIR))

    frontend_proc = None
    if is_dev:
        print("[Dev Mode] Starting Vite development server on http://localhost:5173...")
        frontend_cmd = ["cmd", "/c", "npm", "run", "dev"]
        frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(FRONTEND_DIR))
        target_url = "http://localhost:5173"
    else:
        target_url = "http://localhost:8000"

    # Wait for server to initialize
    time.sleep(2.5)

    # 2. Open browser
    print(f"[2/2] Opening Studio in browser at {target_url}...")
    try:
        webbrowser.open(target_url)
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

    print("\n" + "=" * 70)
    print(f" VoiceBot STT Studio is LIVE at: {target_url}")
    print(" API Documentation:              http://localhost:8000/docs")
    print(" Press Ctrl+C in this window to stop the server.")
    print("=" * 70 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        if frontend_proc:
            frontend_proc.terminate()
        backend_proc.terminate()
        print("Studio stopped.")

if __name__ == "__main__":
    main()

