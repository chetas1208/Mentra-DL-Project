#!/usr/bin/env python3
"""Exact parameter count for an ONNX model (sums initializer tensor sizes).
More reliable than file-size/4 estimation once quantization/int8 is in play.
"""
import argparse
import os
from pathlib import Path

import onnx


def count_params(path: Path) -> int:
    m = onnx.load(str(path))
    total = 0
    for init in m.graph.initializer:
        n = 1
        for d in init.dims:
            n *= d
        total += n
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_paths", nargs="+", type=Path)
    args = ap.parse_args()
    for p in args.model_paths:
        n = count_params(p)
        size_mb = os.path.getsize(p) / 1e6
        print(f"{p.name:45s} params={n/1e6:7.3f}M  file={size_mb:7.1f}MB")
