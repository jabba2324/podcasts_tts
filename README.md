# podcasts_tts

A RunPod serverless endpoint for text-to-speech using [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI. Accepts text input and returns a base64-encoded MP3, with optional voice cloning from a reference audio clip.

## Features

- High-quality TTS via Chatterbox
- Voice cloning from a URL-hosted reference clip
- MP3 output (configurable bitrate)
- Designed for use with n8n or any HTTP client

## Project Structure

```
.
├── handler.py          # RunPod serverless handler
├── requirements.txt    # Python dependencies
├── Dockerfile          # CUDA-enabled container, pre-bakes model weights
├── test_input.json     # Sample request payload
└── .gitignore
```

## Deployment

### 1. Build and push the Docker image

```bash
docker build -t your-dockerhub-username/podcasts-tts:latest .
docker push your-dockerhub-username/podcasts-tts:latest
```

### 2. Create a RunPod Serverless endpoint

- Go to **Serverless → New Endpoint** in the RunPod console
- Select **Custom Docker Image** and enter `your-dockerhub-username/podcasts-tts:latest`
- Recommended GPU: **RTX 4090 (24 GB VRAM)**
- Container disk: **20 GB minimum**

## API

### Request

```json
{
  "input": {
    "text": "Hello, welcome to the podcast.",
    "audio_prompt_url": "https://your-bucket.s3.amazonaws.com/voices/host.wav",
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "mp3_bitrate": "128k"
  }
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | string | Yes | — | Text to synthesise |
| `audio_prompt_url` | string | No | — | URL to a ~10 s audio clip for voice cloning |
| `exaggeration` | float | No | `0.5` | Expressiveness (0.0 = flat, 1.0 = dramatic) |
| `cfg_weight` | float | No | `0.5` | Pacing / guidance weight |
| `mp3_bitrate` | string | No | `"128k"` | MP3 output bitrate (e.g. `"64k"`, `"192k"`) |

### Response

```json
{
  "audio_base64": "<base64-encoded MP3>",
  "sample_rate": 24000,
  "format": "mp3"
}
```

The `audio_base64` value is a standard base64 string. In n8n, use a **Convert to File** node to turn it into a binary file before passing it to downstream nodes.

## Voice Cloning

Pass `audio_prompt_url` pointing to a clean, ~10 second WAV or MP3 of the target speaker. Chatterbox will match the voice characteristics of that clip. If omitted, the model's default voice is used.

## Local Testing

```bash
pip install -r requirements.txt
python handler.py
```
