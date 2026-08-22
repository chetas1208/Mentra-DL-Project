#!/usr/bin/env python3
"""DDP smoke test (sprint spec section 36). Run with:
torchrun --nproc_per_node=2 scripts/training/ddp_smoke_test.py
Verifies: correct per-rank GPU assignment, real all-reduce, real backward
through a DDP-wrapped module, clean process exit."""
import os

import torch
import torch.distributed as dist
import torch.nn as nn


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    print(f"[rank {rank}] sees GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")

    # real all-reduce
    t = torch.tensor([float(rank + 1)], device=local_rank)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    assert abs(t.item() - expected) < 1e-6, f"all-reduce mismatch: got {t.item()}, expected {expected}"
    print(f"[rank {rank}] all-reduce OK: {t.item()} == {expected}")

    # real backward through a DDP-wrapped module
    model = nn.Linear(16, 4).to(local_rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    x = torch.randn(8, 16, device=local_rank)
    out = ddp_model(x)
    loss = out.sum()
    loss.backward()
    grad_exists = model.weight.grad is not None and model.weight.grad.abs().sum().item() > 0
    print(f"[rank {rank}] DDP backward OK: grad_exists={grad_exists}")

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] clean exit")


if __name__ == "__main__":
    main()
