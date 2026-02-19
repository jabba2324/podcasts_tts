"""
RunPod Serverless TTS Handler — Chatterbox

Input schema:
  {
    "input": {
      "text": "Hello, world!",          # required
      "audio_prompt_url": "<url>",      # optional – voice clone reference (≈10 s clip)
      "exaggeration": 0.5,              # optional – expressiveness 0.0–1.0 (default 0.5)
      "cfg_weight": 0.5,                # optional – pacing/guidance weight (default 0.5)
      "mp3_bitrate": "128k"             # optional – output bitrate (default "128k")
    }
  }

Output schema:
  {
    "audio_base64": "<base64-encoded MP3>",
    "sample_rate": <int>,
    "format": "mp3"
  }
"""

import base64
import io
import os
import tempfile
import urllib.request

import runpod
import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS
from pydub import AudioSegment

# ---------------------------------------------------------------------------
# Model initialisation (runs once per cold start)
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] Loading ChatterboxTTS on {DEVICE} …")
model = ChatterboxTTS.from_pretrained(device=DEVICE)
print(f"[init] Model ready  (sample_rate={model.sr})")


# ---------------------------------------------------------------------------
# Helper: download a URL to a temp file and return its path
# ---------------------------------------------------------------------------
def _fetch_audio_prompt(url: str) -> str:
    suffix = os.path.splitext(url.split("?")[0])[-1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def handler(event: dict) -> dict:
    job_input = event.get("input", {})

    text: str = job_input.get("text", "").strip()
    if not text:
        return {"error": "input.text is required and must not be empty"}

    exaggeration: float = float(job_input.get("exaggeration", 0.5))
    cfg_weight: float = float(job_input.get("cfg_weight", 0.5))
    mp3_bitrate: str = job_input.get("mp3_bitrate", "128k")

    # Optional voice-clone reference audio
    prompt_path: str | None = None
    prompt_url: str | None = job_input.get("audio_prompt_url")
    try:
        if prompt_url:
            prompt_path = _fetch_audio_prompt(prompt_url)

        wav: torch.Tensor = model.generate(
            text,
            audio_prompt_path=prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
    finally:
        if prompt_path and os.path.exists(prompt_path):
            os.unlink(prompt_path)

    # Convert tensor → WAV bytes → MP3 bytes
    wav_buf = io.BytesIO()
    torchaudio.save(wav_buf, wav, model.sr, format="wav")
    wav_buf.seek(0)

    mp3_buf = io.BytesIO()
    AudioSegment.from_wav(wav_buf).export(mp3_buf, format="mp3", bitrate=mp3_bitrate)
    mp3_buf.seek(0)

    return {
        "audio_base64": base64.b64encode(mp3_buf.read()).decode("utf-8"),
        "sample_rate": model.sr,
        "format": "mp3",
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
