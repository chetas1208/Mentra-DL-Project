"""GeoWearNet E1 training (Workstreams C, K, L, M, N, O).

Production-quality trainer for the enrollment-free GeoWearNet branch. It is a
NEW file and does not touch `training/train.py` (MentraWearNet) in any way.

CHECKPOINT DISCIPLINE (hard project rule, from a real past data-loss incident)
  * every run gets its own unique directory under training/geowearnet/runs/
  * a self-contained log.txt lives INSIDE that directory
  * checkpoints are step-numbered and NEVER overwritten in place
  * `latest.pt` is a convenience copy alongside -- never a replacement for the
    step-numbered history
  * every checkpoint carries model + optimizer + scheduler + global step +
    full config + loss history + simulator version/config + manifest hashes

Distribution
  * torchrun -> DDP across both 3090s
  * plain invocation -> single GPU
  * --device cpu -> CPU debug mode
Note (measured, see docs): for a ~108K model the bottleneck is scene synthesis
in the data loader, not GPU compute, so the campaign runs ONE experiment PER
GPU concurrently rather than DDP over both. DDP is implemented and smoke-tested
because the brief requires it, not because it is the fastest option here.

Usage
  python3 -m training.geowearnet.train --name geowearnet_e1_s1_base \
      --generation S1 --steps 20000 --batch-size 24 --num-workers 6 --device cuda:0
  torchrun --nproc_per_node=2 -m training.geowearnet.train --name ... --ddp
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from . import manifests
from .data import DataConfig, build_dataset, collate
from .model_zoo import ZooConfig, build_model, check_causality
from .simulate_s1 import SIMULATOR_VERSION, simulator_config_summary

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "training/geowearnet/runs"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class TrainConfig:
    name: str = "geowearnet_e1_s1_base"
    # data
    generation: str = "S1"
    condition: str = "train_mix"
    duration_s: float = 8.0
    n_mels: int = 64
    normalization: str = "global"          # Workstream O: none | global | cmvn
    use_physical: bool = True              # Workstream N
    drop_amplitude_features: bool = False  # Workstream N3
    with_noise: bool = True
    # model
    arch: str = "tcn"                      # Workstream R
    variant: str = "ctx680"                # Workstream P/Q
    aux_four_state_head: bool = True
    dropout: float = 0.1
    # loss (Workstream M)
    loss_mode: str = "M1"                  # M0 | M1 | M2
    aux_weight: float = 0.2
    # optim
    steps: int = 20000
    batch_size: int = 24
    lr: float = 3e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 5.0
    amp: bool = False
    # runtime
    num_workers: int = 6
    seed: int = 1234
    val_every: int = 1000
    val_scenes: int = 96
    ckpt_every: int = 2000
    log_every: int = 50
    early_stop_patience: int = 6           # in validations, on the selection metric
    init_from: Optional[str] = None        # Workstream L: curriculum warm start

    def data_cfg(self, split: str) -> DataConfig:
        return DataConfig(
            split=split,
            generation=self.generation,
            condition=self.condition if split == "train" else "train_mix",
            duration_s=self.duration_s,
            n_mels=self.n_mels,
            normalization=self.normalization,
            use_physical=self.use_physical,
            drop_amplitude_features=self.drop_amplitude_features,
            with_noise=self.with_noise,
            virtual_size=10_000_000 if split == "train" else self.val_scenes,
            # Deterministic validation manifest: fixed seed, independent of the
            # training seed, so every run in the campaign validates on the
            # IDENTICAL scenes and results are comparable.
            seed=self.seed if split == "train" else 777_000,
        )


# ---------------------------------------------------------------------------
# distributed helpers
# ---------------------------------------------------------------------------
def ddp_info() -> Dict[str, int]:
    return {
        "rank": int(os.environ.get("RANK", 0)),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
    }


def is_main() -> bool:
    return ddp_info()["rank"] == 0


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC with tie handling; no sklearn dependency in the loop."""
    y = y.astype(np.int8)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def far_frr_curve(y: np.ndarray, s: np.ndarray):
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    n_pos, n_neg = max(int(y.sum()), 1), max(int((1 - y).sum()), 1)
    far = (1 - ys).cumsum() / n_neg
    frr = 1.0 - ys.cumsum() / n_pos
    return far, frr


def frr_at_far(y: np.ndarray, s: np.ndarray, target: float) -> float:
    if y.sum() == 0 or (1 - y).sum() == 0:
        return float("nan")
    far, frr = far_frr_curve(y, s)
    i = int(np.searchsorted(far, target))
    return float(frr[min(i, len(frr) - 1)])


def eer(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0 or (1 - y).sum() == 0:
        return float("nan")
    far, frr = far_frr_curve(y, s)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2)


def compute_metrics(w_logit, e_logit, w_lab, e_lab, state) -> Dict[str, float]:
    w_s, e_s = w_logit, e_logit
    m: Dict[str, float] = {
        "wearer_auroc": _auroc(w_lab, w_s),
        "environment_auroc": _auroc(e_lab, e_s),
        "wearer_eer": eer(w_lab, w_s),
        "wearer_frr_at_far20": frr_at_far(w_lab, w_s, 0.20),
        "wearer_frr_at_far10": frr_at_far(w_lab, w_s, 0.10),
        "wearer_frr_at_far5": frr_at_far(w_lab, w_s, 0.05),
        "wearer_frr_at_far1": frr_at_far(w_lab, w_s, 0.01),
        "wearer_positive_rate": float(w_lab.mean()),
        "environment_positive_rate": float(e_lab.mean()),
    }
    # 10-vs-01: the product-critical discrimination, on solo frames only
    solo = (w_lab + e_lab) == 1
    if solo.sum() > 10 and len(np.unique(w_lab[solo])) == 2:
        m["wearer_vs_env_solo_auroc"] = _auroc(w_lab[solo], (w_s - e_s)[solo])
        m["wearer_vs_env_solo_frr_at_far5"] = frr_at_far(w_lab[solo], (w_s - e_s)[solo], 0.05)
    # overlap F1 at the 0.5 operating point of both heads
    pred_ov = (w_s > 0) & (e_s > 0)
    true_ov = state == 3
    tp = float((pred_ov & true_ov).sum()); fp = float((pred_ov & ~true_ov).sum()); fn = float((~pred_ov & true_ov).sum())
    m["overlap_f1"] = float(2 * tp / max(2 * tp + fp + fn, 1e-9))
    # 4-state confusion from the two independent heads
    pred_state = (w_s > 0).astype(np.int64) + 2 * (e_s > 0).astype(np.int64)
    m["state_accuracy"] = float((pred_state == state).mean())
    conf = np.zeros((4, 4), dtype=np.int64)
    for a, b in zip(state, pred_state):
        conf[a, b] += 1
    m["state_confusion"] = conf.tolist()
    # The campaign's headline product risk: environment wrongly called wearer.
    env_only = state == 2
    if env_only.sum() > 0:
        m["false_wearer_rate_on_env_only"] = float((w_s[env_only] > 0).mean())
    return m


SELECTION_KEYS = ["wearer_vs_env_solo_auroc", "wearer_auroc"]


def selection_metric(m: Dict[str, float]) -> float:
    for k in SELECTION_KEYS:
        v = m.get(k)
        if v is not None and not math.isnan(v):
            return float(v)
    return float("nan")


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
class GeoLoss(nn.Module):
    """Workstream M. M0 = two BCE. M1 = + auxiliary 4-state CE.
    M2 = class-balanced BCE (pos_weight from the measured label rates).
    Deliberately NO speaker-ID / ArcFace term -- GeoWearNet is identity-free."""

    def __init__(self, mode: str, aux_weight: float, pos_weight: Optional[Dict[str, float]] = None):
        super().__init__()
        self.mode = mode
        self.aux_weight = aux_weight
        pw = pos_weight or {}
        self.register_buffer("pw_w", torch.tensor(float(pw.get("wearer", 1.0))))
        self.register_buffer("pw_e", torch.tensor(float(pw.get("environment", 1.0))))

    def forward(self, out, batch) -> Dict[str, torch.Tensor]:
        mask = batch["mask"]
        denom = mask.sum().clamp(min=1.0)
        pw_w = self.pw_w if self.mode == "M2" else None
        pw_e = self.pw_e if self.mode == "M2" else None
        lw = nn.functional.binary_cross_entropy_with_logits(
            out["wearer_logits"], batch["wearer"], reduction="none", pos_weight=pw_w)
        le = nn.functional.binary_cross_entropy_with_logits(
            out["environment_logits"], batch["environment"], reduction="none", pos_weight=pw_e)
        loss_w = (lw * mask).sum() / denom
        loss_e = (le * mask).sum() / denom
        total = loss_w + loss_e
        parts = {"loss_wearer": loss_w, "loss_environment": loss_e}
        if self.mode == "M1" and "four_state_logits" in out:
            ce = nn.functional.cross_entropy(
                out["four_state_logits"].reshape(-1, 4), batch["state"].reshape(-1), reduction="none"
            ).reshape(mask.shape)
            loss_s = (ce * mask).sum() / denom
            total = total + self.aux_weight * loss_s
            parts["loss_state"] = loss_s
        parts["loss"] = total
        return parts


# ---------------------------------------------------------------------------
# run directory
# ---------------------------------------------------------------------------
class Run:
    def __init__(self, cfg: TrainConfig, resume_dir: Optional[Path] = None):
        self.cfg = cfg
        if resume_dir is not None:
            self.dir = resume_dir
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            h = abs(hash(json.dumps(dataclasses.asdict(cfg), sort_keys=True))) % 0xFFFF
            self.dir = RUNS_DIR / f"{cfg.name}_{stamp}_{h:04x}"
        self.ckpt_dir = self.dir / "checkpoints"
        if is_main():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
        self.log_path = self.dir / "log.txt"
        self.metrics_path = self.dir / "metrics.jsonl"

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        if is_main():
            print(line, flush=True)
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    def metric(self, obj: dict) -> None:
        if is_main():
            with open(self.metrics_path, "a") as f:
                f.write(json.dumps(obj) + "\n")

    def save_checkpoint(self, step: int, model, opt, sched, history: List[dict],
                        best: Optional[dict], tag: str = "") -> Optional[Path]:
        """Step-numbered, never overwritten. `latest.pt`/`best.pt` are COPIES."""
        if not is_main():
            return None
        raw = model.module if hasattr(model, "module") else model
        payload = {
            "model_state": raw.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": sched.state_dict() if sched is not None else None,
            "global_step": step,
            "config": dataclasses.asdict(self.cfg),
            "model_config": dataclasses.asdict(raw.config),
            "params": raw.count_parameters(),
            "receptive_field_frames": raw.receptive_field_frames(),
            "context_ms": raw.context_ms(),
            "loss_history_summary": history[-50:],
            "best": best,
            "simulator_version": SIMULATOR_VERSION,
            "simulator_config": simulator_config_summary(
                self.cfg.generation if self.cfg.generation in ("S1", "S2") else "S1"),
            "manifest_hashes": {s: manifests.manifest_hash(s) for s in ("train", "val", "test")},
            "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch_version": torch.__version__,
            "note": "SIMULATED GEOMETRY ONLY -- not validated on real Mentra hardware.",
        }
        p = self.ckpt_dir / f"step_{step:07d}{('_' + tag) if tag else ''}.pt"
        torch.save(payload, p)
        shutil.copyfile(p, self.ckpt_dir / "latest.pt")
        if tag == "best":
            shutil.copyfile(p, self.ckpt_dir / "best.pt")
        return p


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn) -> Dict[str, float]:
    model.eval()
    W, E, WL, EL, ST = [], [], [], [], []
    losses = []
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(b["log_mel"], b["physical"] if b["physical"].shape[-1] else None)
        losses.append(float(loss_fn(out, b)["loss"].item()))
        m = b["mask"].bool()
        W.append(out["wearer_logits"][m].float().cpu().numpy())
        E.append(out["environment_logits"][m].float().cpu().numpy())
        WL.append(b["wearer"][m].cpu().numpy())
        EL.append(b["environment"][m].cpu().numpy())
        ST.append(b["state"][m].cpu().numpy())
    model.train()
    met = compute_metrics(np.concatenate(W), np.concatenate(E), np.concatenate(WL),
                          np.concatenate(EL), np.concatenate(ST))
    met["val_loss"] = float(np.mean(losses))
    met["val_frames"] = int(len(np.concatenate(WL)))
    return met


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None, help="run dir or checkpoint path")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--ddp", action="store_true")
    for f in dataclasses.fields(TrainConfig):
        if f.type in (bool, "bool"):
            ap.add_argument(f"--{f.name.replace('_','-')}", dest=f.name, action="store_true", default=None)
            ap.add_argument(f"--no-{f.name.replace('_','-')}", dest=f.name, action="store_false", default=None)
        else:
            ap.add_argument(f"--{f.name.replace('_','-')}", dest=f.name, type=type(f.default)
                            if f.default is not None and not isinstance(f.default, type(None)) else str,
                            default=None)
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.config:
        cfg = TrainConfig(**{**dataclasses.asdict(cfg), **json.loads(Path(args.config).read_text())})
    resume_ckpt = None
    resume_dir = None
    if args.resume:
        p = Path(args.resume)
        if p.is_dir():
            resume_dir = p
            resume_ckpt = p / "checkpoints/latest.pt"
        else:
            resume_ckpt = p
            resume_dir = p.parent.parent
        saved = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        cfg = TrainConfig(**saved["config"])
    for f in dataclasses.fields(TrainConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)

    info = ddp_info()
    use_ddp = args.ddp and info["world_size"] > 1
    if use_ddp:
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{info['local_rank']}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device if (args.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")

    set_seed(cfg.seed + info["rank"])
    run = Run(cfg, resume_dir=resume_dir)
    run.log(f"=== GeoWearNet training: {cfg.name} ===")
    run.log(f"run_dir={run.dir}")
    run.log(f"device={device} ddp={use_ddp} world_size={info['world_size']} host={platform.node()}")
    run.log(f"config={json.dumps(dataclasses.asdict(cfg))}")
    run.log(f"simulator={SIMULATOR_VERSION}")
    run.log(f"manifest_hashes={ {s: manifests.manifest_hash(s)[:16] for s in ('train','val','test')} }")

    # --- data ---
    train_cfg = cfg.data_cfg("train")
    val_cfg = cfg.data_cfg("val")
    train_ds = build_dataset(train_cfg)
    val_ds = build_dataset(val_cfg)
    n_phys = len(train_ds.feat_idx) if cfg.use_physical else 0
    run.log(f"data: train_speakers={len(train_ds.pool.speakers)} val_speakers={len(val_ds.pool.speakers)} "
            f"n_physical={n_phys} normalization={cfg.normalization}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=collate, pin_memory=(device.type == "cuda"), drop_last=True,
        persistent_workers=cfg.num_workers > 0, prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=min(cfg.num_workers, 4), collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    # --- model ---
    model = build_model(
        cfg.arch, cfg.variant, n_mels=cfg.n_mels, n_physical_features=max(n_phys, 1),
        use_physical_features=cfg.use_physical and n_phys > 0,
        aux_four_state_head=cfg.aux_four_state_head, dropout=cfg.dropout,
    )
    run.log(f"model: arch={cfg.arch} variant={cfg.variant} params={model.count_parameters()} "
            f"rf={model.receptive_field_frames()} frames ({model.context_ms():.0f} ms)")
    caus = check_causality(model)
    run.log(f"causality check (untrained): {caus}")

    if cfg.init_from:  # Workstream L curriculum warm start
        src = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(src["model_state"], strict=False)
        run.log(f"init_from={cfg.init_from} step={src.get('global_step')} "
                f"missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device)
    if use_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[info["local_rank"]])

    # class-balanced weights measured from the actual training distribution
    pos_weight = None
    if cfg.loss_mode == "M2":
        pr = {"wearer": 0.0, "environment": 0.0}
        n = 0
        for i in range(48):
            sc = train_ds.scene_for(900_000 + i)
            pr["wearer"] += float(sc.wearer.mean()); pr["environment"] += float(sc.environment.mean()); n += 1
        pos_weight = {k: (1 - v / n) / max(v / n, 1e-3) for k, v in pr.items()}
        run.log(f"M2 pos_weight (measured over {n} scenes) = {pos_weight}")
    loss_fn = GeoLoss(cfg.loss_mode, cfg.aux_weight, pos_weight).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    def lr_lambda(s: int) -> float:
        if s < cfg.warmup_steps:
            return (s + 1) / max(cfg.warmup_steps, 1)
        p = (s - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    start_step = 0
    history: List[dict] = []
    best: Optional[dict] = None
    if resume_ckpt is not None and Path(resume_ckpt).exists():
        saved = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        raw = model.module if hasattr(model, "module") else model
        raw.load_state_dict(saved["model_state"])
        opt.load_state_dict(saved["optimizer_state"])
        if saved.get("scheduler_state"):
            sched.load_state_dict(saved["scheduler_state"])
        start_step = int(saved["global_step"])
        history = list(saved.get("loss_history_summary") or [])
        best = saved.get("best")
        run.log(f"RESUMED from {resume_ckpt} at step {start_step}")

    # --- graceful shutdown: always leave a checkpoint behind ---
    stop = {"flag": False}
    def _sig(_s, _f):
        stop["flag"] = True
        run.log("signal received -- will checkpoint and exit at the next step boundary")
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _sig)
        except Exception:
            pass

    run.log(f"starting at step {start_step}/{cfg.steps}")
    model.train()
    step = start_step
    t_last = time.perf_counter()
    running: List[float] = []
    bad_vals = 0
    it = iter(train_loader)
    data_wait = 0.0

    while step < cfg.steps and not stop["flag"]:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        data_wait += time.perf_counter() - t0

        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
            out = model(b["log_mel"], b["physical"] if b["physical"].shape[-1] else None)
            parts = loss_fn(out, b)
            loss = parts["loss"]

        if not torch.isfinite(loss):
            run.log(f"NON-FINITE LOSS at step {step}: {parts} -- skipping batch")
            opt.zero_grad(set_to_none=True)
            step += 1
            continue

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        running.append(float(loss.item()))
        step += 1

        if step % cfg.log_every == 0:
            dt = time.perf_counter() - t_last
            sps = cfg.log_every / max(dt, 1e-9)
            rec = {
                "step": step, "loss": float(np.mean(running[-cfg.log_every:])),
                "lr": sched.get_last_lr()[0], "grad_norm": float(gn),
                "steps_per_sec": sps,
                "scenes_per_sec": sps * cfg.batch_size * info["world_size"],
                "data_wait_frac": data_wait / max(dt, 1e-9),
            }
            history.append(rec)
            run.metric({"type": "train", **rec})
            run.log(f"step {step:6d}/{cfg.steps} loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
                    f"gn {rec['grad_norm']:.2f} {sps:.2f} it/s "
                    f"({rec['scenes_per_sec']:.0f} scenes/s) data_wait {rec['data_wait_frac']*100:.0f}%")
            t_last = time.perf_counter()
            data_wait = 0.0

        if step % cfg.val_every == 0 or step >= cfg.steps:
            met = evaluate(model, val_loader, device, loss_fn)
            sel = selection_metric(met)
            run.metric({"type": "val", "step": step, **met})
            run.log(f"VAL step {step}: loss {met['val_loss']:.4f} wearer_auroc {met['wearer_auroc']:.4f} "
                    f"env_auroc {met['environment_auroc']:.4f} solo_auroc "
                    f"{met.get('wearer_vs_env_solo_auroc', float('nan')):.4f} "
                    f"overlap_f1 {met['overlap_f1']:.4f} frr@far5 {met['wearer_frr_at_far5']:.4f} "
                    f"select {sel:.4f}")
            if best is None or (not math.isnan(sel) and sel > best.get("selection", -1)):
                best = {"step": step, "selection": sel, "metrics": {k: v for k, v in met.items()
                                                                    if k != "state_confusion"}}
                run.save_checkpoint(step, model, opt, sched, history, best, tag="best")
                run.log(f"  new best (selection={sel:.4f}) checkpointed")
                bad_vals = 0
            else:
                bad_vals += 1
                run.log(f"  no improvement ({bad_vals}/{cfg.early_stop_patience})")
                if bad_vals >= cfg.early_stop_patience:
                    run.log("EARLY STOP: validation plateau reached")
                    break

        if step % cfg.ckpt_every == 0:
            p = run.save_checkpoint(step, model, opt, sched, history, best)
            run.log(f"  checkpoint -> {p}")

    p = run.save_checkpoint(step, model, opt, sched, history, best, tag="final")
    run.log(f"FINISHED at step {step}. final checkpoint {p}")
    if best:
        run.log(f"BEST: step {best['step']} selection {best['selection']:.4f}")
    summary = {
        "name": cfg.name, "run_dir": str(run.dir), "final_step": step,
        "best": best, "config": dataclasses.asdict(cfg),
        "params": (model.module if hasattr(model, "module") else model).count_parameters(),
        "stopped_early": bad_vals >= cfg.early_stop_patience,
        "interrupted": stop["flag"],
    }
    if is_main():
        (run.dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
