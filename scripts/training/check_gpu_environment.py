#!/usr/bin/env python3
"""GPU environment health check (sprint spec section 33/34). Run before any
DDP training -- CUDA_VISIBLE_DEVICES="" workarounds used for CPU parity
work are NOT a training fix, this actually exercises the GPUs."""
import sys

import torch

print(f"torch version: {torch.__version__}")
print(f"torch.version.cuda (build): {torch.version.cuda}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("BLOCKED: torch.cuda.is_available() is False in this process")
    sys.exit(1)

try:
    count = torch.cuda.device_count()
    print(f"GPU count: {count}")
    for i in range(count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as e:
    print(f"BLOCKED: device enumeration failed: {e}")
    sys.exit(1)

failures = []
for i in range(count):
    try:
        x = torch.randn(1024, 1024, device=f"cuda:{i}")
        y = x @ x
        torch.cuda.synchronize(i)
        print(f"  GPU {i}: real matmul OK, result norm={y.norm().item():.2f}")
    except Exception as e:
        print(f"  GPU {i}: FAILED - {e}")
        failures.append(i)

nccl_available = torch.distributed.is_nccl_available()
print(f"NCCL available: {nccl_available}")

if failures:
    print(f"\nBLOCKED: GPUs failed real tensor ops: {failures}")
    sys.exit(1)
if not nccl_available:
    print("\nBLOCKED: NCCL not available -- DDP across GPUs will not work")
    sys.exit(1)

print("\nGPU_TRAINING_READY: all GPUs passed real tensor ops, NCCL available")
