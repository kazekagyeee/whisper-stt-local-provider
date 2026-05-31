# Whisper STT Local Provider

OpenAI-compatible Speech-to-Text server using [faster-whisper](https://github.com/cydonia999/faster-whisper).

## Features

- 🎤 OpenAI-compatible `/v1/audio/transcriptions` endpoint
- ⚡ Fast inference with CTranslate2 optimization
- 🖥️ CPU and GPU support (auto-detect)
- 🌐 Multilingual (ru, en, de, es, fr, etc.)
- 🐳 Docker-ready

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/whisper-stt-local-provider.git
cd whisper-stt-local-provider
cp .env.example .env
```

### 2. Run with Docker

```bash
docker-compose up -d
```

### 3. Or run directly

```bash
pip install -r requirements.txt
python server.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port |
| `WHISPER_MODEL` | base | Model size: tiny, base, small, medium, large-v3 |
| `WHISPER_LANGUAGE` | ru | Language code (None = auto-detect) |
| `WHISPER_COMPUTE_TYPE` | auto | float16, int8, float32, auto |
| `WHISPER_DEVICE` | auto | cpu, cuda, auto |

### Model sizes

| Model | Params | VRAM | Speed |
|-------|--------|------|-------|
| tiny | ~39M | ~1GB | ~10x realtime |
| base | ~74M | ~1GB | ~7x realtime |
| small | ~244M | ~2GB | ~5x realtime |
| medium | ~769M | ~5GB | ~2x realtime |
| large-v3 | ~1550M | ~6GB | ~1x realtime |

**Recommended for CPU:** `base` or `small`  
**Recommended for GPU:** `medium` or `large-v3`

## API Usage

### OpenAI-compatible endpoint

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "language=ru"
```

Response:
```json
{
  "text": "Привет, как дела?"
}
```

### Custom endpoint

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=ru"
```

Response:
```json
{
  "text": "Привет, как дела?",
  "language": "ru",
  "duration": 2.5,
  "segments": [
    {"start": 0.0, "end": 2.5, "text": " Привет, как дела?"}
  ]
}
```

### Health check

```bash
curl http://localhost:8000/health
```

### List available models

```bash
curl http://localhost:8000/models/available
```

## Hermes Integration

In your Hermes config, add a speech-to-text provider:

```yaml
providers:
  speech_to_text:
    provider: openai
    api_key: dummy  # Not used for local provider
    base_url: http://your-host:8000/v1
    model: whisper-1
```

## License

MIT
