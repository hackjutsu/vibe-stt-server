# Vibe STT Server

FastAPI-based one-shot Whisper transcription service following `discussions/init_design.md`.

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

## Run (macOS dev, CPU)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

## Run (Windows 11 + NVIDIA GPU via WSL2 Ubuntu)
The easiest way to use CUDA with faster-whisper on Windows is through WSL2 (Linux wheels ship with GPU support).

```powershell
wsl --install Ubuntu    # once, if not already installed
wsl -d Ubuntu
```

Inside the Ubuntu shell:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export WHISPER_DEVICE=cuda
export WHISPER_COMPUTE_TYPE=float16   # or int8_float16 if VRAM is tight
python -m app.main
```

Prereqs:
- Recent NVIDIA Windows driver with WSL2 CUDA support (same requirement as Docker GPU).
- In WSL2, verify GPU is visible before running the server: `nvidia-smi`.

## Run (Windows 11 native, CPU-only)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:WHISPER_DEVICE = "cpu"            # Windows wheels are CPU-only
$env:WHISPER_COMPUTE_TYPE = "auto"
python -m app.main
```

Use your client to POST to `http://<host>:8000/transcribe` with gzip enabled. The server preloads the model at startup to avoid first-request latency.

### Verify GPU is being used (Windows)
1. Ensure `/info` shows `"device": "cuda"` and your `compute_type` (e.g., `float16`).
2. While sending a `/transcribe`, watch GPU activity:
   - PowerShell loop:
     ```powershell
     while ($true) { nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv; Start-Sleep 1; Clear-Host }
     ```
     You should see a `python` process with non-zero `used_gpu_memory`.
   - Or `nvidia-smi dmon` for a short live view.
   - Task Manager → Performance → GPU also shows compute/utilization.
