# whisper.cpp Server Setup

> Server setup is optional. WhisperO runs locally by default using faster-whisper.

WhisperO sends audio to a whisper.cpp HTTP server when backend is set to `server`.
This guide shows two ways to run it: native build or Docker.

## Option 1: Build whisper.cpp from source

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build
cmake --build build -j
```

### Download a model

```bash
bash ./models/download-ggml-model.sh large-v3
```

You can swap `large-v3` for another model name.

### Start the server

```bash
./build/bin/whisper-server \
  --host 0.0.0.0 \
  --port 8080 \
  --model ./models/ggml-large-v3.bin
```

## Option 2: Docker

A Docker setup is handy if you do not want local build tools.

```bash
docker run --rm -it \
  -p 8080:8080 \
  -v $(pwd)/models:/models \
  ghcr.io/ggml-org/whisper.cpp:main \
  /app/whisper-server \
  --host 0.0.0.0 \
  --port 8080 \
  --model /models/ggml-large-v3.bin
```

Before running this, place your model file in `./models`.

## Test the server

### Health check

```bash
curl http://localhost:8080/health
```

Expected response:

```json
{"status":"ok"}
```

### Inference test

```bash
curl -s -X POST http://localhost:8080/inference \
  -F "file=@sample.wav" \
  -F "response_format=text"
```

If you get text back, WhisperO is ready to use.

## Recommended models

- `large-v3`: best accuracy, higher VRAM/CPU cost
- `base`: much faster, lower accuracy

Pick based on your hardware and latency target.

## GPU vs CPU notes

- **GPU** is better for low latency dictation.
- **CPU** works fine for light use, but expect slower turnaround.
- On Apple Silicon, Metal builds are usually fast enough for daily dictation.
- On NVIDIA GPUs, CUDA builds give the best speed.

## Connect WhisperO

Set your server URL with either:

```bash
export WHISPERO_SERVER="http://localhost:8080"
```

or in `~/.whispero/config.json`:

```json
{
  "server": "http://localhost:8080"
}
```
