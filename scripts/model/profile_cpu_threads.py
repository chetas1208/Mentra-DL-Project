#!/usr/bin/env python3
"""CPU thread profiling for streaming SpeakerNet inference (Track E).
Simulates the actual deployment workload: repeated 2.0s-context inference
calls at 200ms hop spacing (not single-shot), across 1/2/4
intra_op_num_threads. CUDA forced off -- this profiles the CPU deployment
path, not training.
"""
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import psutil
import soundfile as sf

from research.sherpa_onnx.detector import SherpaOnnxWearerDetector

MODEL_PATH = "models/sherpa-speaker/nemo_en_speakerverification_speakernet.onnx"
CONTEXT_S = 2.0
N_CALLS = 150


def load_wav(path: str):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return np.ascontiguousarray(data[:, 0]), sr


def profile(num_threads: int) -> dict:
    detector = SherpaOnnxWearerDetector(MODEL_PATH, num_threads=num_threads)
    enroll_segments = [load_wav(p) for p in
                        ["evaluation/data/day1_public/2300/enroll_00.wav",
                         "evaluation/data/day1_public/2300/enroll_01.wav",
                         "evaluation/data/day1_public/2300/enroll_02.wav"]]
    detector.enroll(enroll_segments)

    samples, sr = load_wav("evaluation/data/day1_public/2300/test_00.wav")
    context_samples = int(CONTEXT_S * sr)
    if len(samples) < context_samples:
        samples = np.tile(samples, int(np.ceil(context_samples / len(samples))))
    window = samples[:context_samples]

    process = psutil.Process()
    latencies_ms = []
    cpu_before = process.cpu_times()
    t_wall_start = time.perf_counter()

    for _ in range(N_CALLS):
        t0 = time.perf_counter()
        detector.process(window, sr)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    wall_elapsed = time.perf_counter() - t_wall_start
    cpu_after = process.cpu_times()
    cpu_time = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    rss_mb = process.memory_info().rss / 1e6

    latencies_ms.sort()
    n = len(latencies_ms)
    return {
        "num_threads": num_threads,
        "p50_ms": latencies_ms[n // 2],
        "p95_ms": latencies_ms[int(n * 0.95)],
        "p99_ms": latencies_ms[int(n * 0.99)],
        "rtf": (sum(latencies_ms) / 1000) / (N_CALLS * CONTEXT_S),
        "cpu_percent_avg": (cpu_time / wall_elapsed) * 100,
        "rss_mb": rss_mb,
    }


if __name__ == "__main__":
    print(f"profiling SpeakerNet streaming inference, {N_CALLS} calls per thread count, {CONTEXT_S}s context")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} -- CPU only\n")
    results = []
    for threads in [1, 2, 4]:
        r = profile(threads)
        results.append(r)
        print(f"threads={threads}: p50={r['p50_ms']:.2f}ms p95={r['p95_ms']:.2f}ms p99={r['p99_ms']:.2f}ms "
              f"RTF={r['rtf']:.4f} CPU%={r['cpu_percent_avg']:.0f} RSS={r['rss_mb']:.0f}MB")

    print("\n| threads | p50 (ms) | p95 (ms) | p99 (ms) | RTF | CPU% | RSS (MB) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['num_threads']} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} | "
              f"{r['rtf']:.4f} | {r['cpu_percent_avg']:.0f} | {r['rss_mb']:.0f} |")
