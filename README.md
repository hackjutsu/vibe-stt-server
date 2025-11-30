# Vibe STT Server

[GitHub Repository](https://github.com/hackjutsu/vibe-stt-server).

FastAPI-based one-shot Whisper transcription service following the [design](./discussions/init_design.md). This is a project for near real-time speech-to-text transcription in headless Ubuntu machine(with Nvidia GPU) to empower [the local voice chatbot project](https://github.com/hackjutsu/vibe-speech). When running in a more powerful machine, the speech-to-text latency is reduced from seconds to <0.5s with the model `large-v3-turbo` compared to running in the local dev machine.

## Endpoints
- `GET /health` → `{ "status": "ok" }`
- `GET /info` → model/device/config info
- `POST /transcribe` → transcribe a single audio session

### POST /transcribe body
```json
{
  "audio": "<base64 mono 16 kHz int16 PCM>",
  "sample_rate": 16000,
  "language": "en",
  "initial_prompt": "optional",
  "beam_size": 5
}
```

Response:
```json
{
  "text": "...",
  "info": {
    "language": "en",
    "duration_ms": 2000,
    "decode_time_ms": 120,
    "num_samples": 32000
  }
}
```
- `language` is optional; when omitted, Whisper auto-detects the language.
- `beam_size` defaults to the server setting if not provided.

### GET /info response
```json
{
  "model": "large-v3-turbo",
  "device": "auto",
  "compute_type": "auto",
  "sample_rate": 16000,
  "num_workers": 1,
  "cpu_threads": 0,
  "default_beam_size": 5
}
```

## Config (env vars)
- `WHISPER_MODEL` (default: `large-v3-turbo`)
- `WHISPER_DEVICE` (default: `auto`) — e.g., `cpu`, `cuda`
- `WHISPER_COMPUTE_TYPE` (default: `auto`) — e.g., `float16`, `int8_float16`
- `WHISPER_CPU_THREADS` (default: `0` for library default)
- `WHISPER_NUM_WORKERS` (default: `1`)
- `WHISPER_BEAM_SIZE` (default: `5`)
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8000`)
- `LOG_LEVEL` (default: `INFO`)

## Run 

### macOS dev, CPU
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000  # for local dev hot-reload
```

### Linux with GPU
Check out this [summary](./discussions/resolve_whisper_linux_gpu_deps.md) on how to resolve dependencies issue with running Nvidia GPU with Ubuntu.

For local development.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

For running service as a detached tmux session
```bash
tmux new -s stt-session -d 'uvicorn app.main:app --host 0.0.0.0 --port 8000'
```
