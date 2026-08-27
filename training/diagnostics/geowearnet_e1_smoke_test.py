"""GeoWearNet-E1 smoke test (Phase 45-46).

Not a training run -- verifies the E1 architecture (training/geowearnet/model.py)
is actually usable before committing to a real training job:
  1. forward+backward pass produces finite gradients on every parameter
  2. parameter count is within the Phase 11 target (<1.5M, hard cap <10M)
  3. the model is genuinely causal: perturbing a future frame must not
     change the logits at any earlier frame (verified numerically, not just
     inferred from the padding code)
  4. batch=1 (the real streaming-inference batch size) does not crash
     BatchNorm in eval mode

Run: .venv/bin/python3 -m training.diagnostics.geowearnet_e1_smoke_test
"""
from __future__ import annotations

import torch

from training.geowearnet.model import GeoWearNetE1, GeoWearNetE1Config


def check_forward_backward(model: GeoWearNetE1, config: GeoWearNetE1Config) -> dict:
    B, T = 4, 200
    log_mel = torch.randn(B, T, config.n_mels, requires_grad=False)
    phys = torch.randn(B, T, config.n_physical_features) if config.use_physical_features else None

    model.train()
    out = model(log_mel, phys)
    loss = out["wearer_logits"].sum() + out["environment_logits"].sum()
    if "four_state_logits" in out:
        loss = loss + out["four_state_logits"].sum()
    loss.backward()

    finite_outputs = all(torch.isfinite(v).all().item() for v in out.values())
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    has_all_grads = all(g is not None for g in grads)
    finite_grads = all(torch.isfinite(g).all().item() for g in grads if g is not None)
    model.zero_grad()

    return {
        "output_shapes": {k: list(v.shape) for k, v in out.items()},
        "finite_outputs": finite_outputs,
        "every_param_has_grad": has_all_grads,
        "finite_grads": finite_grads,
    }


def check_causality(model: GeoWearNetE1, config: GeoWearNetE1Config) -> dict:
    model.eval()
    B, T = 1, 120
    torch.manual_seed(0)
    log_mel = torch.randn(B, T, config.n_mels)
    phys = torch.randn(B, T, config.n_physical_features) if config.use_physical_features else None

    with torch.no_grad():
        out_a = model(log_mel, phys)

    perturb_frame = T - 1  # last frame only
    log_mel_b = log_mel.clone()
    log_mel_b[:, perturb_frame, :] += 5.0
    phys_b = phys.clone() if phys is not None else None
    if phys_b is not None:
        phys_b[:, perturb_frame, :] += 5.0

    with torch.no_grad():
        out_b = model(log_mel_b, phys_b)

    # Every frame strictly before the perturbed one must be byte-identical;
    # the perturbed frame itself (and nothing after it, since it's the last
    # frame here) is allowed to change.
    max_diff_before = 0.0
    for key in ("wearer_logits", "environment_logits"):
        diff = (out_a[key][:, :perturb_frame] - out_b[key][:, :perturb_frame]).abs().max().item()
        max_diff_before = max(max_diff_before, diff)
    changed_at_perturb = (
        (out_a["wearer_logits"][:, perturb_frame] - out_b["wearer_logits"][:, perturb_frame]).abs().item()
    )

    return {
        "max_abs_diff_before_perturbed_frame": max_diff_before,
        "is_causal": max_diff_before < 1e-5,
        "perturbed_frame_did_change": changed_at_perturb > 1e-4,
        "receptive_field_frames": model.receptive_field_frames(),
    }


def check_batch_one_eval(model: GeoWearNetE1, config: GeoWearNetE1Config) -> dict:
    model.eval()
    log_mel = torch.randn(1, 50, config.n_mels)
    phys = torch.randn(1, 50, config.n_physical_features) if config.use_physical_features else None
    try:
        with torch.no_grad():
            out = model(log_mel, phys)
        ok = all(torch.isfinite(v).all().item() for v in out.values())
        return {"ok": ok, "error": None}
    except Exception as e:  # noqa: BLE001 -- smoke test, report any failure honestly
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def main():
    config = GeoWearNetE1Config()
    model = GeoWearNetE1(config)
    n_params = model.count_parameters()

    results = {
        "config": dataclasses_asdict_safe(config),
        "param_count": n_params,
        "param_count_under_1_5m_target": n_params < 1_500_000,
        "param_count_under_10m_hard_cap": n_params < 10_000_000,
        "forward_backward": check_forward_backward(model, config),
        "causality": check_causality(model, config),
        "batch_one_eval": check_batch_one_eval(model, config),
    }

    fb = results["forward_backward"]
    causal = results["causality"]
    b1 = results["batch_one_eval"]
    all_pass = (
        results["param_count_under_10m_hard_cap"]
        and fb["finite_outputs"] and fb["every_param_has_grad"] and fb["finite_grads"]
        and causal["is_causal"] and causal["perturbed_frame_did_change"]
        and b1["ok"]
    )
    results["verdict"] = "E1_SMOKE_TEST_PASS" if all_pass else "E1_SMOKE_TEST_FAIL"

    import json
    print(json.dumps(results, indent=2))
    return results


def dataclasses_asdict_safe(config: GeoWearNetE1Config) -> dict:
    import dataclasses
    return dataclasses.asdict(config)


if __name__ == "__main__":
    main()
