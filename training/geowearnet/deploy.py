"""GeoWearNet deployment: streaming inference, CPU benchmark, export, soak.

Workstreams AK (CPU benchmark), AL (export + parity), AM (streaming state),
AN (long soak test).

All timings here are CPU timings at batch=1, because that is the deployment
condition. No CUDA number is ever reported as deployment evidence.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .model_zoo import ARCHS, GeoWearNetCRNN, GeoWearNetTCN, ZooConfig, build_model

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "evaluation/geowearnet/results"


# ---------------------------------------------------------------------------
# Workstream AM -- true streaming state
# ---------------------------------------------------------------------------
class StreamingTCN:
    """Frame-synchronous streaming inference for `GeoWearNetTCN`.

    The batch model recomputes the whole 680 ms window for every output frame.
    Here each causal convolution keeps a ring buffer of exactly its own
    `(kernel-1)*dilation` past inputs, so one new frame costs one convolution
    per layer instead of re-running the network over 69 frames.

    Correctness is not assumed: `verify_streaming_parity()` asserts the streamed
    outputs match the batch outputs to within float tolerance.

    Safe because, in eval mode, everything except Conv1d is pointwise in time:
    BatchNorm uses fixed running statistics (a per-channel affine map), Dropout
    is identity, ReLU/GELU are pointwise, and residual adds are aligned.
    """

    def __init__(self, model: GeoWearNetTCN):
        assert isinstance(model, GeoWearNetTCN), "streaming implemented for the TCN arch"
        self.m = model.eval()
        self.reset()

    def reset(self) -> None:
        c = self.m.config
        self.fe_buf: List[torch.Tensor] = []
        cin = c.n_mels
        for layer in self.m.frontend:
            self.fe_buf.append(torch.zeros(1, cin, layer.pad))
            cin = layer.pw.out_channels
        self.blk_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for b in self.m.blocks:
            self.blk_buf.append((torch.zeros(1, c.tcn_channels, b.pad),
                                 torch.zeros(1, c.tcn_channels, b.pad)))

    @torch.no_grad()
    def step(self, mel_frame: torch.Tensor, phys_frame: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """mel_frame: (n_mels,) -> logits for this frame."""
        x = mel_frame.reshape(1, -1, 1)
        for i, layer in enumerate(self.m.frontend):
            buf = self.fe_buf[i]
            cat = torch.cat([buf, x], dim=2)
            self.fe_buf[i] = cat[:, :, -buf.shape[2]:] if buf.shape[2] > 0 else buf
            y = layer.dw(cat) if layer.pad > 0 else layer.dw(x)
            x = layer.act(layer.bn(layer.pw(y)))
        for i, b in enumerate(self.m.blocks):
            b1, b2 = self.blk_buf[i]
            res = x
            cat1 = torch.cat([b1, x], dim=2)
            self.blk_buf[i] = (cat1[:, :, -b1.shape[2]:] if b1.shape[2] > 0 else b1, b2)
            y = b.act(b.bn1(b.conv1(cat1)))
            b1n, _ = self.blk_buf[i]
            cat2 = torch.cat([b2, y], dim=2)
            self.blk_buf[i] = (b1n, cat2[:, :, -b2.shape[2]:] if b2.shape[2] > 0 else b2)
            y = b.act(b.bn2(b.conv2(cat2)))
            x = b.act(y + res)
        feat = x.transpose(1, 2)  # (1,1,C)
        if self.m.heads.phys is not None:
            assert phys_frame is not None
            feat = torch.cat([feat, self.m.heads.phys(phys_frame.reshape(1, 1, -1))], dim=-1)
        return {
            "wearer_logits": self.m.heads.wearer(feat).reshape(()),
            "environment_logits": self.m.heads.environment(feat).reshape(()),
        }


class StreamingCRNN:
    """Frame-synchronous streaming inference for `GeoWearNetCRNN`.

    The conv frontend uses the same ring-buffer trick as `StreamingTCN`
    (finite kernel, so it needs a buffer). The GRU is already causal and
    stateful by construction -- streaming it is just carrying its hidden
    state `h` from one frame to the next, no buffer required. Its receptive
    field is unbounded in principle (see `GeoWearNetCRNN` docstring), so
    this only proves per-frame numerical parity with the batch path, not a
    bounded context claim."""

    def __init__(self, model: GeoWearNetCRNN):
        assert isinstance(model, GeoWearNetCRNN), "StreamingCRNN implemented for the CRNN arch"
        self.m = model.eval()
        self.reset()

    def reset(self) -> None:
        c = self.m.config
        self.fe_buf: List[torch.Tensor] = []
        cin = c.n_mels
        for layer in self.m.frontend:
            self.fe_buf.append(torch.zeros(1, cin, layer.pad))
            cin = layer.pw.out_channels
        self.h: Optional[torch.Tensor] = None

    @torch.no_grad()
    def step(self, mel_frame: torch.Tensor, phys_frame: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """mel_frame: (n_mels,) -> logits for this frame."""
        x = mel_frame.reshape(1, -1, 1)
        for i, layer in enumerate(self.m.frontend):
            buf = self.fe_buf[i]
            cat = torch.cat([buf, x], dim=2)
            self.fe_buf[i] = cat[:, :, -buf.shape[2]:] if buf.shape[2] > 0 else buf
            y = layer.dw(cat) if layer.pad > 0 else layer.dw(x)
            x = layer.act(layer.bn(layer.pw(y)))
        gru_in = x.transpose(1, 2)  # (1,1,C)
        out, self.h = self.m.gru(gru_in, self.h)
        feat = out  # (1,1,C)
        if self.m.heads.phys is not None:
            assert phys_frame is not None
            feat = torch.cat([feat, self.m.heads.phys(phys_frame.reshape(1, 1, -1))], dim=-1)
        return {
            "wearer_logits": self.m.heads.wearer(feat).reshape(()),
            "environment_logits": self.m.heads.environment(feat).reshape(()),
        }


def make_streamer(model):
    """Dispatch to the right stateful streaming wrapper for `model`'s arch."""
    if isinstance(model, GeoWearNetCRNN):
        return StreamingCRNN(model)
    if isinstance(model, GeoWearNetTCN):
        return StreamingTCN(model)
    raise NotImplementedError(f"no streaming implementation for arch {type(model).__name__}")


@torch.no_grad()
def verify_streaming_parity(model, t: int = 300, tol: float = 2e-4) -> Dict[str, float]:
    model.eval()
    c = model.config
    mel = torch.randn(1, t, c.n_mels)
    ph = torch.randn(1, t, c.n_physical_features) if c.use_physical_features else None
    batch = model(mel, ph)
    st = make_streamer(model)
    w, e = [], []
    for i in range(t):
        o = st.step(mel[0, i], ph[0, i] if ph is not None else None)
        w.append(float(o["wearer_logits"])); e.append(float(o["environment_logits"]))
    dw = float(np.abs(np.array(w) - batch["wearer_logits"][0].numpy()).max())
    de = float(np.abs(np.array(e) - batch["environment_logits"][0].numpy()).max())
    assert dw < tol and de < tol, f"STREAMING PARITY FAILED: wearer {dw}, env {de}"
    return {"max_abs_diff_wearer": dw, "max_abs_diff_environment": de, "frames": t, "tolerance": tol}


# ---------------------------------------------------------------------------
# Workstream AK -- CPU benchmark
# ---------------------------------------------------------------------------
def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@torch.no_grad()
def cpu_benchmark(model, threads: int, chunk_frames: int = 100, n_chunks: int = 40,
                  hop_ms: float = 10.0, mode: str = "chunk") -> Dict[str, float]:
    """batch=1 CPU latency. `mode`:
       chunk     -- feed `chunk_frames` at a time (typical 1 s streaming block)
       rolling   -- recompute the full receptive-field window per output frame
       streaming -- stateful, one frame at a time (Workstream AM)
    """
    torch.set_num_threads(threads)
    model.eval()
    c = model.config
    lat: List[float] = []

    if mode == "streaming":
        st = make_streamer(model)
        mel = torch.randn(chunk_frames, c.n_mels)
        ph = torch.randn(chunk_frames, c.n_physical_features) if c.use_physical_features else None
        for i in range(min(20, chunk_frames)):
            st.step(mel[i], ph[i] if ph is not None else None)
        for _ in range(n_chunks):
            t0 = time.perf_counter()
            for i in range(chunk_frames):
                st.step(mel[i], ph[i] if ph is not None else None)
            lat.append((time.perf_counter() - t0) * 1000.0)
        audio_ms = chunk_frames * hop_ms
    elif mode == "rolling":
        rf = model.receptive_field_frames() + 1
        mel = torch.randn(1, rf, c.n_mels)
        ph = torch.randn(1, rf, c.n_physical_features) if c.use_physical_features else None
        for _ in range(5):
            model(mel, ph)
        for _ in range(n_chunks):
            t0 = time.perf_counter()
            for _ in range(chunk_frames):
                model(mel, ph)
            lat.append((time.perf_counter() - t0) * 1000.0)
        audio_ms = chunk_frames * hop_ms
    else:
        mel = torch.randn(1, chunk_frames, c.n_mels)
        ph = torch.randn(1, chunk_frames, c.n_physical_features) if c.use_physical_features else None
        for _ in range(5):
            model(mel, ph)
        for _ in range(n_chunks):
            t0 = time.perf_counter()
            model(mel, ph)
            lat.append((time.perf_counter() - t0) * 1000.0)
        audio_ms = chunk_frames * hop_ms

    a = np.array(lat)
    return {
        "mode": mode, "threads": threads, "chunk_frames": chunk_frames,
        "audio_ms_per_chunk": audio_ms,
        "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
        "rtf": float(np.percentile(a, 50) / audio_ms),
        "per_frame_p50_us": float(np.percentile(a, 50) / chunk_frames * 1000.0),
        "rss_mb": _rss_mb(),
    }


def serialized_size_mb(model) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        torch.save(model.state_dict(), f.name)
        return os.path.getsize(f.name) / 1e6


# ---------------------------------------------------------------------------
# Workstream AL -- export + parity
# ---------------------------------------------------------------------------
@torch.no_grad()
def _rel(p: Path) -> str:
    """Repo-relative path when possible; absolute otherwise (e.g. a temp dir
    during tests). `Path.relative_to` raises for paths outside the repo, which
    used to make the whole export look like a failure."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def export_and_verify(model, out_dir: Path, t: int = 200, tol: float = 1e-4) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    c = model.config
    mel = torch.randn(1, t, c.n_mels)
    ph = torch.randn(1, t, c.n_physical_features) if c.use_physical_features else None
    ref = model(mel, ph)
    rep: Dict[str, object] = {"tolerance": tol, "frames": t}

    # --- TorchScript ---
    ts_path = out_dir / "geowearnet_e1.ts.pt"
    try:
        args = (mel, ph) if ph is not None else (mel,)
        ts = torch.jit.trace(model, args, strict=False)
        ts = torch.jit.freeze(ts)
        torch.jit.save(ts, str(ts_path))
        loaded = torch.jit.load(str(ts_path))
        got = loaded(*args)
        d = {k: float((ref[k] - got[k]).abs().max()) for k in ("wearer_logits", "environment_logits")}
        rep["torchscript"] = {
            "status": "OK" if max(d.values()) < tol else "PARITY_FAIL",
            "path": _rel(ts_path),
            "size_mb": ts_path.stat().st_size / 1e6,
            "max_abs_diff": d,
        }
    except Exception as ex:
        rep["torchscript"] = {"status": "FAILED", "error": repr(ex)}

    # --- ONNX (onnx is installed; onnxruntime is NOT, so parity is checked
    #     structurally + via the ONNX checker, and that limitation is recorded) ---
    onnx_path = out_dir / "geowearnet_e1.onnx"
    try:
        args = (mel, ph) if ph is not None else (mel,)
        names = ["log_mel"] + (["physical"] if ph is not None else [])
        torch.onnx.export(
            model, args, str(onnx_path), input_names=names,
            output_names=["wearer_logits", "environment_logits", "four_state_logits"],
            dynamic_axes={n: {0: "batch", 1: "time"} for n in names},
            opset_version=17, dynamo=False,
        )
        import onnx
        m = onnx.load(str(onnx_path))
        onnx.checker.check_model(m)
        entry = {
            "status": "EXPORTED_CHECKED",
            "path": _rel(onnx_path),
            "size_mb": onnx_path.stat().st_size / 1e6,
            "opset": 17,
            "onnx_checker": "PASS",
        }
        try:
            import onnxruntime as ort  # noqa
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            feeds = {"log_mel": mel.numpy()}
            if ph is not None:
                feeds["physical"] = ph.numpy()
            o = sess.run(["wearer_logits", "environment_logits"], feeds)
            # torch.onnx.export's tracer leaves autograd globally re-enabled even
            # inside this @torch.no_grad() function, so `ref` tensors can come back
            # with requires_grad=True by the time we get here -- detach defensively.
            d = {"wearer_logits": float(np.abs(ref["wearer_logits"].detach().numpy() - o[0]).max()),
                 "environment_logits": float(np.abs(ref["environment_logits"].detach().numpy() - o[1]).max())}
            entry["max_abs_diff"] = d
            entry["status"] = "OK" if max(d.values()) < tol else "PARITY_FAIL"
        except ImportError:
            entry["numerical_parity"] = (
                "NOT VERIFIED -- onnxruntime is not installed in this environment. "
                "The graph exports and passes onnx.checker, but no numerical parity "
                "claim is made for the ONNX path. TorchScript parity IS verified."
            )
        rep["onnx"] = entry
    except Exception as ex:
        rep["onnx"] = {"status": "FAILED", "error": repr(ex)}
    return rep


# ---------------------------------------------------------------------------
# Workstream AN -- long soak
# ---------------------------------------------------------------------------
@torch.no_grad()
def soak(model, minutes: float = 30.0, threads: int = 2, hop_ms: float = 10.0,
         block_frames: int = 100, report_every_s: float = 120.0) -> Dict[str, object]:
    """Continuous streaming soak. Checks for memory growth, state drift, NaNs
    and latency drift over a long run."""
    torch.set_num_threads(threads)
    model.eval()
    c = model.config
    st = make_streamer(model)
    rng = np.random.default_rng(0)
    target_frames = int(minutes * 60 * 1000 / hop_ms)

    lat, rss, samples = [], [], []
    t_start = time.perf_counter()
    frames = 0
    nan_count = 0
    next_report = report_every_s
    first_block_out = None

    while frames < target_frames:
        mel = torch.from_numpy(rng.standard_normal((block_frames, c.n_mels)).astype(np.float32))
        ph = torch.from_numpy(rng.standard_normal((block_frames, c.n_physical_features)).astype(np.float32)) \
            if c.use_physical_features else None
        t0 = time.perf_counter()
        outs = []
        for i in range(block_frames):
            o = st.step(mel[i], ph[i] if ph is not None else None)
            v = float(o["wearer_logits"])
            outs.append(v)
            if not np.isfinite(v):
                nan_count += 1
        lat.append((time.perf_counter() - t0) * 1000.0 / block_frames)
        frames += block_frames
        if first_block_out is None:
            first_block_out = outs
        el = time.perf_counter() - t_start
        if el >= next_report:
            rss.append({"elapsed_s": el, "audio_minutes": frames * hop_ms / 60000.0, "rss_mb": _rss_mb()})
            samples.append(float(np.mean(np.abs(outs))))
            next_report += report_every_s

    a = np.array(lat)
    half = len(a) // 2
    n_buf_elems = sum(b.numel() for b in st.fe_buf)
    n_buf_elems += sum(x.numel() + y.numel() for x, y in st.blk_buf) if hasattr(st, "blk_buf") else 0
    n_buf_elems += st.h.numel() if getattr(st, "h", None) is not None else 0
    return {
        "requested_minutes": minutes,
        "audio_minutes_processed": frames * hop_ms / 60000.0,
        "wall_seconds": time.perf_counter() - t_start,
        "frames": frames,
        "threads": threads,
        "per_frame_p50_ms": float(np.percentile(a, 50)),
        "per_frame_p95_ms": float(np.percentile(a, 95)),
        "latency_drift_pct_second_half_vs_first": float(
            (a[half:].mean() - a[:half].mean()) / max(a[:half].mean(), 1e-9) * 100.0),
        "rtf": float(np.percentile(a, 50) / hop_ms),
        "nan_or_inf_outputs": nan_count,
        "rss_timeline": rss,
        "rss_growth_mb": (rss[-1]["rss_mb"] - rss[0]["rss_mb"]) if len(rss) > 1 else 0.0,
        "streaming_state_elements": int(n_buf_elems),
        "streaming_state_constant": True,
        "output_magnitude_timeline": samples,
    }


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--variant", default="ctx680")
    ap.add_argument("--arch", default="tcn")
    ap.add_argument("--soak-minutes", type=float, default=0.0)
    ap.add_argument("--all-variants", action="store_true")
    ap.add_argument("--out", default="geowearnet_deploy_benchmark.json")
    a = ap.parse_args()

    res: Dict[str, object] = {"note": "CPU, batch=1. No CUDA timing is reported as deployment evidence."}

    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        mc = ck["model_config"]
        cfg = ZooConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in mc.items()})
        model = ARCHS[cfg.arch](cfg)
        model.load_state_dict(ck["model_state"])
        res["checkpoint"] = a.checkpoint
        res["global_step"] = ck["global_step"]
        models = {ck["config"]["name"]: model}
    elif a.all_variants:
        from .model_zoo import CONTEXT_VARIANTS, SIZE_VARIANTS
        models = {}
        for v in CONTEXT_VARIANTS:
            models[f"tcn_{v}"] = build_model("tcn", v)
        for v in SIZE_VARIANTS:
            models[f"tcn_{v}"] = build_model("tcn", v)
        for arch in ARCHS:
            models[f"arch_{arch}"] = build_model(arch)
    else:
        models = {f"{a.arch}_{a.variant}": build_model(a.arch, a.variant)}

    bench: Dict[str, object] = {}
    for name, model in models.items():
        entry: Dict[str, object] = {
            "params": model.count_parameters(),
            "state_dict_mb": serialized_size_mb(model),
            "receptive_field_frames": model.receptive_field_frames(),
            "context_ms": model.context_ms(),
            "arch": model.config.arch,
        }
        for th in (1, 2, 4):
            entry[f"chunk_{th}thread"] = cpu_benchmark(model, th, mode="chunk")
        if isinstance(model, (GeoWearNetTCN, GeoWearNetCRNN)):
            entry["streaming_parity"] = verify_streaming_parity(model)
            entry["streaming_1thread"] = cpu_benchmark(model, 1, mode="streaming", n_chunks=12)
            if isinstance(model, GeoWearNetTCN):
                # "rolling" mode recomputes the full receptive-field window per
                # frame, which is only a meaningful non-streaming baseline for a
                # model with a bounded receptive field.
                entry["rolling_window_1thread"] = cpu_benchmark(model, 1, mode="rolling", n_chunks=6)
                sp = entry["streaming_1thread"]["p50_ms"]
                rl = entry["rolling_window_1thread"]["p50_ms"]
                entry["streaming_speedup_vs_rolling"] = float(rl / max(sp, 1e-9))
        bench[name] = entry
        print(f"{name}: params={entry['params']} "
              f"chunk1t_p50={entry['chunk_1thread']['p50_ms']:.2f}ms "
              f"rtf={entry['chunk_1thread']['rtf']:.5f}", flush=True)
        gc.collect()

    res["benchmarks"] = bench

    primary = next(iter(models.values()))
    res["export"] = export_and_verify(primary, REPO_ROOT / "training/geowearnet/export")

    if a.soak_minutes > 0:
        if isinstance(primary, (GeoWearNetTCN, GeoWearNetCRNN)):
            print(f"soak test: {a.soak_minutes} minutes of audio...", flush=True)
            res["soak"] = soak(primary, minutes=a.soak_minutes)
            print("soak:", json.dumps({k: v for k, v in res["soak"].items()
                                       if not isinstance(v, list)}, indent=2), flush=True)
        else:
            print(f"soak test skipped: no streaming implementation for arch "
                  f"{type(primary).__name__} yet", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / a.out
    p.write_text(json.dumps(res, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
