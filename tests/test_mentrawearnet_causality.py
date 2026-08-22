#!/usr/bin/env python3
"""Causality test (sprint spec section 24/108).

Generates prefix A, then A+B and A+C with different futures appended.
Compares model outputs on the A-portion between the two runs -- outputs
covering A must be near-identical if the model is truly causal (no future
leakage). This is a diagnostic script (run directly), not wired into a test
runner yet -- no pytest/CI harness exists in this repo.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.models.mentrawearnet import MentraWearNet


def main():
    torch.manual_seed(0)
    model = MentraWearNet()
    model.eval()

    sr = 16000
    a = torch.randn(1, int(1.0 * sr))       # 1.0s prefix, shared
    b = torch.randn(1, int(1.0 * sr))       # 1.0s future #1
    c = torch.randn(1, int(1.0 * sr))       # 1.0s future #2 (different)

    ab = torch.cat([a, b], dim=1)
    ac = torch.cat([a, c], dim=1)

    enr = torch.randn(1, int(3.0 * sr))
    enr_len = torch.tensor([enr.shape[1]])

    with torch.no_grad():
        wearer_emb = model.encode_enrollment(enr, enr_len)

        out_a_only = model.process_with_embedding(a, torch.tensor([a.shape[1]]), wearer_emb)
        out_ab = model.process_with_embedding(ab, torch.tensor([ab.shape[1]]), wearer_emb)
        out_ac = model.process_with_embedding(ac, torch.tensor([ac.shape[1]]), wearer_emb)

    t_a = out_a_only["wearer_logits"].shape[1]
    print(f"frames covering A alone: {t_a}")
    print(f"frames covering A+B: {out_ab['wearer_logits'].shape[1]}, A+C: {out_ac['wearer_logits'].shape[1]}")

    # Compare the first t_a frames of AB vs AC -- these SHOULD correspond
    # to the A region in both. If causal, they should match closely.
    diff_wearer = (out_ab["wearer_logits"][:, :t_a] - out_ac["wearer_logits"][:, :t_a]).abs()
    diff_env = (out_ab["environment_logits"][:, :t_a] - out_ac["environment_logits"][:, :t_a]).abs()

    print()
    print("Per-frame |AB - AC| on the A-region (index 0 = start of A):")
    for i in range(0, t_a, max(1, t_a // 10)):
        print(f"  frame {i:3d}: wearer_diff={diff_wearer[0, i].item():.4f}  env_diff={diff_env[0, i].item():.4f}")

    max_diff = max(diff_wearer.max().item(), diff_env.max().item())
    mean_diff = (diff_wearer.mean().item() + diff_env.mean().item()) / 2
    early_diff = diff_wearer[:, :10].mean().item()  # frames far from the A/B boundary
    late_diff = diff_wearer[:, -10:].mean().item()  # frames right at the A/B boundary

    print()
    print(f"max abs diff over full A-region: {max_diff:.4f}")
    print(f"mean abs diff over full A-region: {mean_diff:.4f}")
    print(f"mean diff, first 10 frames (far from boundary): {early_diff:.4f}")
    print(f"mean diff, last 10 frames (at A/B boundary): {late_diff:.4f}")

    tol = 1e-4
    verdict = "PASS" if max_diff < tol else "FAIL"
    print()
    print(f"CAUSALITY TEST: {verdict} (tolerance={tol})")
    if verdict == "FAIL":
        print("Expected FAIL if the pretrained SpeakerNet backbone uses standard")
        print("(symmetric-padded) convolutions -- those pull in real future audio")
        print("samples within their receptive field, regardless of the custom TCN")
        print("being causal. Check whether the diff grows toward the boundary vs")
        print("staying flat (flat = the custom head is somehow the cause instead).")


if __name__ == "__main__":
    main()
