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
import re
import tempfile
import unicodedata
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
# Helper: sanitise text for the Chatterbox tokenizer
# ---------------------------------------------------------------------------
def _sanitise_text(text: str) -> str:
    # toJsonString() in n8n double-encodes the string — unwrap it if so
    # e.g. '"Hello, \"world\""' → 'Hello, "world"'
    stripped = text.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        try:
            import json
            text = json.loads(stripped)
        except Exception:
            pass

    # Normalise unicode (e.g. smart quotes → straight quotes)
    text = unicodedata.normalize("NFKC", text)
    # Strip non-printable / control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Helper: download a URL to a temp file and return its path
# ---------------------------------------------------------------------------
def _fetch_audio_prompt(url: str) -> str:
    suffix = os.path.splitext(url.split("?")[0])[-1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Helper: synthesise a single text segment → AudioSegment
# ---------------------------------------------------------------------------
def _synthesise(text: str, prompt_path: str | None, exaggeration: float, cfg_weight: float) -> AudioSegment:
    wav: torch.Tensor = model.generate(
        text,
        audio_prompt_path=prompt_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    buf = io.BytesIO()
    torchaudio.save(buf, wav, model.sr, format="wav")
    buf.seek(0)
    return AudioSegment.from_wav(buf)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def handler(event: dict) -> dict:
    job_input = event.get("input", {})

    text: str = _sanitise_text(job_input.get("text", ""))
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

        # Split on [pause:N] markers — e.g. "[pause:4]" inserts 4 s of silence
        parts = re.split(r"\[pause:(\d+(?:\.\d+)?)\]", text)
        # parts alternates: [text, duration, text, duration, ..., text]

        combined: AudioSegment | None = None
        i = 0
        while i < len(parts):
            segment_text = parts[i].strip()
            if segment_text:
                audio = _synthesise(segment_text, prompt_path, exaggeration, cfg_weight)
                combined = audio if combined is None else combined + audio

            # If there's a pause duration following this segment, append silence
            if i + 1 < len(parts):
                silence_ms = int(float(parts[i + 1]) * 1000)
                combined = (combined or AudioSegment.empty()) + AudioSegment.silent(duration=silence_ms)
                i += 2
            else:
                i += 1

        if combined is None:
            return {"error": "No speakable text found after parsing pause markers"}

    finally:
        if prompt_path and os.path.exists(prompt_path):
            os.unlink(prompt_path)

    mp3_buf = io.BytesIO()
    combined.export(mp3_buf, format="mp3", bitrate=mp3_bitrate)
    mp3_buf.seek(0)

    return {
        "audio_base64": base64.b64encode(mp3_buf.read()).decode("utf-8"),
        "sample_rate": model.sr,
        "format": "mp3",
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
