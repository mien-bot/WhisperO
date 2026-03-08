#!/usr/bin/env python3
"""
WhisperO Benchmark
Records a short audio clip and benchmarks transcription speed.

Usage:
    python benchmark.py                          # local mode (default)
    python benchmark.py --backend server --server http://host:8080
    python benchmark.py --runs 5 --seconds 3
"""

import argparse
import io
import time
import wave

try:
    import sounddevice as sd
except ImportError:
    print("Install sounddevice: pip install sounddevice numpy")
    exit(1)

SAMPLE_RATE = 16000


def record_clip(seconds=5):
    print(f"Recording {seconds}s of audio... speak now!")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="int16", blocking=True)
    print("Done recording.\n")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    return buf


def benchmark_local(audio_buf, runs=10, model_size="large-v3"):
    from whispero.transcribe import transcribe_local, get_model

    print(f"Loading model ({model_size})...")
    get_model(model_size)
    print("Model ready.\n")

    times = []
    for i in range(runs):
        audio_buf.seek(0)
        start = time.perf_counter()
        text = transcribe_local(audio_buf, model_size=model_size)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        if i == 0:
            print(f"Transcription: \"{text}\"\n")
        print(f"  Run {i+1}/{runs}: {elapsed:.0f}ms")

    return times


def benchmark_server(audio_buf, runs=10, server="http://localhost:8080"):
    import requests

    times = []
    for i in range(runs):
        audio_buf.seek(0)
        start = time.perf_counter()
        resp = requests.post(
            f"{server}/inference",
            files={"file": ("audio.wav", audio_buf, "audio/wav")},
            data={"response_format": "text"},
            timeout=60,
        )
        elapsed = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        text = resp.text.strip()
        times.append(elapsed)

        if i == 0:
            print(f"Transcription: \"{text}\"\n")
        print(f"  Run {i+1}/{runs}: {elapsed:.0f}ms")

    return times


def print_results(times, runs):
    print(f"\n{'='*40}")
    print(f"Results ({runs} runs):")
    print(f"  Average: {sum(times)/len(times):.0f}ms")
    print(f"  Min:     {min(times):.0f}ms")
    print(f"  Max:     {max(times):.0f}ms")
    print(f"  Median:  {sorted(times)[len(times)//2]:.0f}ms")
    print(f"{'='*40}")


def main():
    parser = argparse.ArgumentParser(description="WhisperO Benchmark")
    parser.add_argument("--backend", default="local", choices=["local", "server"],
                        help="Backend to benchmark (default: local)")
    parser.add_argument("--model", default="large-v3", help="Model size for local mode")
    parser.add_argument("--server", default="http://localhost:8080", help="Server URL for server mode")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--seconds", type=int, default=5, help="Recording length in seconds")
    args = parser.parse_args()

    if args.backend == "server":
        import requests
        try:
            requests.get(f"{args.server}/health", timeout=5)
            print(f"Server: {args.server} (healthy)\n")
        except Exception:
            print(f"Cannot reach server at {args.server}")
            exit(1)

    audio = record_clip(args.seconds)

    if args.backend == "local":
        times = benchmark_local(audio, args.runs, args.model)
    else:
        times = benchmark_server(audio, args.runs, args.server)

    print_results(times, args.runs)


if __name__ == "__main__":
    main()
