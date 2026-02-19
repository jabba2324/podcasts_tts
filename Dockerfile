# ── Base ──────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ffmpeg is required by pydub for MP3 encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download Chatterbox weights at build time ─────────────────────────────
# This bakes the HuggingFace model weights into the image so cold starts are fast.
RUN python - <<'EOF'
from chatterbox.tts import ChatterboxTTS
ChatterboxTTS.from_pretrained(device="cpu")
print("Chatterbox weights cached.")
EOF

# ── Application code ──────────────────────────────────────────────────────────
COPY handler.py .

CMD ["python", "-u", "handler.py"]
