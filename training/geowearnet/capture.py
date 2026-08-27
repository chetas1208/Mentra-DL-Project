"""Real-Mentra research capture pipeline (Workstreams AA, AB, AC, AD).

Builds the engineering NOW so that the moment glasses and subjects exist, a
5-10 person pilot can be recorded the same day. Nothing here needs a trained
model; it is pure data collection and validation.

WHAT THIS IS FOR
  The single most valuable missing evidence in this campaign is: does an
  E1 model trained on simulated geometry transfer to a real Mentra
  microphone at all (Workstream AZ)? That question needs a small, clean,
  correctly-labelled real dataset -- not a large one.

RAW-SIGNAL DISCIPLINE (Workstream AA/AY)
  Capture writes the ORIGINAL PCM. No RMS normalisation, no peak
  normalisation, no AGC, no EQ, no noise suppression, no resampling of the
  archived file. A 16 kHz derived stream is produced as a SEPARATE artefact
  next to the original, never in place of it. Absolute level is a physical cue
  under test; destroying it before archiving would silently invalidate the
  experiment.

PRIVACY (Workstream AA)
  Subjects get an ANONYMOUS wearer id (`W01`, `W02`, ...). No names, no emails,
  no device serials, no location. Nothing is uploaded anywhere: there is no
  network code in this module at all.

SOURCES
  This module is source-agnostic so it can be tested without hardware:
    ffmpeg-device  -- a real capture device via ffmpeg
    ffmpeg-test    -- a synthetic ffmpeg source (for pipeline smoke tests)
    ingest         -- adopt an existing WAV written by the browser/Mentra
                      WebSocket path (`server/audio/remote_receiver.py`)
  The production path for actual Mentra audio is `ingest`: the glasses stream
  over the existing transport and the receiver writes a WAV, which this module
  then labels and validates.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_ROOT = REPO_ROOT / "recordings/geowearnet_pilot"
TARGET_SR = 16000


# ---------------------------------------------------------------------------
# Workstream AB -- the pilot protocol
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Step:
    step_id: str
    seconds: float
    instruction: str
    condition: str
    source_label: str      # who should be speaking: wearer | environment | both | none
    notes: str = ""


PROTOCOL: List[Step] = [
    Step("quiet_room", 20, "Everyone silent. Do not speak. Stay still.",
         "quiet", "none"),
    Step("wearer_normal", 45, "WEARER: read the passage at a normal conversational volume.",
         "wearer_normal", "wearer"),
    Step("wearer_soft", 30, "WEARER: read the passage quietly, as if not to disturb anyone.",
         "wearer_soft", "wearer"),
    Step("wearer_loud", 30, "WEARER: read the passage loudly, as if across a room.",
         "wearer_loud", "wearer"),
    Step("wearer_whisper", 30, "WEARER: whisper the passage.",
         "wearer_whisper", "wearer"),
    Step("env_025m", 30, "PARTNER stands 0.25 m in front of the wearer and reads. Wearer silent.",
         "env_0.25m", "environment"),
    Step("env_050m", 30, "PARTNER at 0.5 m reads. Wearer silent.",
         "env_0.5m", "environment"),
    Step("env_100m", 30, "PARTNER at 1 m reads. Wearer silent.",
         "env_1m", "environment"),
    Step("env_200m", 30, "PARTNER at 2 m reads. Wearer silent.",
         "env_2m", "environment"),
    Step("env_loud", 30, "PARTNER at 1 m speaks loudly / shouts. Wearer silent.",
         "env_loud", "environment"),
    Step("env_behind", 30, "PARTNER at 1 m BEHIND the wearer reads. Wearer silent.",
         "env_behind", "environment"),
    Step("overlap", 45, "WEARER and PARTNER (1 m) both talk at once, interrupting each other.",
         "overlap", "both"),
    Step("turn_taking", 60, "WEARER and PARTNER (1 m) hold a normal back-and-forth conversation.",
         "turn_taking", "both"),
    Step("tv_phone", 45, "Play speech from a phone/TV speaker ~1 m away. Wearer silent.",
         "device_speech", "environment",
         "Tests a non-human environment source with a different spectral profile."),
    Step("wearer_moving", 45, "WEARER walks around and turns their head while reading.",
         "wearer_movement", "wearer"),
    Step("noisy_room", 45, "WEARER reads with background noise (fan/traffic/cafe) present.",
         "noisy", "wearer"),
]

PILOT_A_SUBJECTS = 5
PILOT_B_SUBJECTS = 10


def protocol_summary() -> Dict[str, object]:
    total_s = sum(s.seconds for s in PROTOCOL)
    by_source: Dict[str, float] = {}
    for s in PROTOCOL:
        by_source[s.source_label] = by_source.get(s.source_label, 0.0) + s.seconds
    return {
        "n_steps": len(PROTOCOL),
        "seconds_per_subject": total_s,
        "minutes_per_subject": total_s / 60.0,
        "seconds_by_source_label": by_source,
        "conditions": [s.condition for s in PROTOCOL],
    }


# ---------------------------------------------------------------------------
# Workstream AD -- pilot power plan
# ---------------------------------------------------------------------------
def pilot_plan() -> Dict[str, object]:
    """A concrete MINIMAL plan. The goal is not a production corpus; it is to
    answer one question: does E1-SIM transfer to real Mentra audio at all?"""
    ps = protocol_summary()
    per_subject_min = ps["minutes_per_subject"]

    def block(n_subjects: int, name: str) -> Dict[str, object]:
        total_min = per_subject_min * n_subjects
        wearer_min = ps["seconds_by_source_label"].get("wearer", 0) * n_subjects / 60.0
        env_min = ps["seconds_by_source_label"].get("environment", 0) * n_subjects / 60.0
        both_min = ps["seconds_by_source_label"].get("both", 0) * n_subjects / 60.0
        # 10 ms frames; the conversational blocks are where transitions live.
        conv_min = both_min
        return {
            "name": name,
            "n_subjects": n_subjects,
            "minutes_per_subject": round(per_subject_min, 1),
            "total_minutes": round(total_min, 1),
            "total_hours": round(total_min / 60.0, 2),
            "wearer_speech_minutes": round(wearer_min, 1),
            "environment_speech_minutes": round(env_min, 1),
            "overlap_and_conversation_minutes": round(both_min, 1),
            "labelled_frames_10ms": int(total_min * 60 * 100),
            "estimated_state_transitions": int(conv_min * 60 * 0.6),
            "examples_per_condition_per_subject": {s.condition: 1 for s in PROTOCOL},
        }

    return {
        "objective": (
            "Measure whether the best E1-SIM checkpoint transfers ZERO-SHOT to real "
            "Mentra audio. This is a go/no-go measurement, not a training set."
        ),
        "pilot_A": block(PILOT_A_SUBJECTS, "Pilot A -- feasibility"),
        "pilot_B": block(PILOT_B_SUBJECTS, "Pilot B -- confirmation"),
        "decision_gate": {
            "after": "Pilot A",
            "evaluate": "zero-shot E1-SIM, per-condition wearer AUROC on person-disjoint real audio",
            "PROCEED_if": (
                "wearer-vs-environment AUROC >= 0.70 on normal + soft wearer and on "
                "0.5-2 m bystander conditions, with the close-bystander false-wearer rate "
                "not catastrophic. That is enough signal to justify Pilot B and light fine-tuning."
            ),
            "STOP_AND_ANALYSE_if": (
                "AUROC <= 0.60 (near chance). Do NOT fine-tune on 5 subjects -- that memorises "
                "them. Instead diagnose the domain gap: compare real vs simulated feature "
                "statistics (level, spectral tilt, DRR proxy, device EQ) and fix the simulator."
            ),
            "note": (
                "With 5 subjects the person-level n is tiny, so report per-condition AUROC with "
                "leave-one-subject-out variation, never a single pooled number with a tight CI."
            ),
        },
        "labelling": (
            "Labels come from the PROTOCOL itself (scripted who-speaks-when), not from manual "
            "annotation. That is why the protocol enforces solo blocks: they are self-labelling. "
            "The conversation/overlap blocks need light manual segmentation, and are the only "
            "blocks that do."
        ),
        "protocol": ps,
        "person_disjointness": "Validation is always leave-subject-out. Never split within a subject.",
    }


# ---------------------------------------------------------------------------
# capture sources
# ---------------------------------------------------------------------------
def anonymous_wearer_id(index: int) -> str:
    return f"W{index:02d}"


def _ffmpeg(args: List[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args],
                          capture_output=True, timeout=timeout)


def capture_segment(dest: Path, seconds: float, source: str = "ffmpeg-test",
                    device: Optional[str] = None, sample_rate: int = 48000,
                    channels: int = 1) -> Dict[str, object]:
    """Write ONE segment at the device's native rate, unmodified.

    `-af` is never used: no filtering, no normalisation, no resampling of the
    archived file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if source == "ffmpeg-device":
        assert device, "ffmpeg-device requires --device (e.g. 'default' for ALSA)"
        args = ["-f", "alsa", "-i", device, "-t", str(seconds),
                "-ar", str(sample_rate), "-ac", str(channels), "-c:a", "pcm_s16le", "-y", str(dest)]
    elif source == "ffmpeg-test":
        # Synthetic source purely to prove the pipeline end to end on a host
        # with no microphone. Clearly marked as synthetic in the metadata.
        args = ["-f", "lavfi", "-i",
                f"sine=frequency=220:sample_rate={sample_rate}:duration={seconds}",
                "-ac", str(channels), "-c:a", "pcm_s16le", "-y", str(dest)]
    else:
        raise ValueError(f"unknown source {source!r}")
    p = _ffmpeg(args, timeout=seconds + 30)
    return {
        "source": source,
        "device": device,
        "synthetic": source == "ffmpeg-test",
        "requested_seconds": seconds,
        "requested_sample_rate": sample_rate,
        "requested_channels": channels,
        "wall_seconds": time.time() - t0,
        "ffmpeg_returncode": p.returncode,
        "ffmpeg_stderr": p.stderr.decode()[:500],
    }


def derive_16k(src: Path, dest: Path) -> Dict[str, object]:
    """Produce the 16 kHz mono stream the model consumes, ALONGSIDE (never
    instead of) the original. Resampling only -- no gain, no filtering."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    p = _ffmpeg(["-i", str(src), "-ar", str(TARGET_SR), "-ac", "1",
                 "-c:a", "pcm_s16le", "-y", str(dest)], timeout=120)
    return {"path": str(dest), "returncode": p.returncode,
            "note": "derived stream; the original-rate file is preserved untouched"}


# ---------------------------------------------------------------------------
# Workstream AC -- automatic capture validation
# ---------------------------------------------------------------------------
def validate_recording(path: Path, expected_seconds: Optional[float] = None,
                       meta: Optional[dict] = None) -> Dict[str, object]:
    """Every recording is checked immediately. The point is to catch a broken
    rig on subject 1, not after ten subjects have gone home."""
    rep: Dict[str, object] = {"path": str(path), "flags": []}
    if not path.exists():
        rep["flags"].append("MISSING_FILE")
        rep["ok"] = False
        return rep
    rep["size_bytes"] = path.stat().st_size
    try:
        with wave.open(str(path), "rb") as w:
            n = w.getnframes()
            sr = w.getframerate()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            raw = w.readframes(n)
    except Exception as ex:
        rep["flags"].append(f"UNREADABLE:{ex!r}")
        rep["ok"] = False
        return rep

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw)
    if dtype is None:
        rep["flags"].append(f"UNSUPPORTED_SAMPLE_WIDTH:{sw}")
        rep["ok"] = False
        return rep
    x = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    full_scale = float(np.iinfo(dtype).max)
    x = x / full_scale
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)

    dur = n / sr if sr else 0.0
    peak = float(np.abs(x).max()) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
    clip_frac = float((np.abs(x) >= 0.999).mean()) if len(x) else 1.0
    # non-silence on 10 ms frames, threshold 40 dB below this file's own peak
    hop = max(int(sr * 0.01), 1)
    nfr = max(len(x) // hop, 1)
    fr = x[: nfr * hop].reshape(nfr, hop)
    fr_rms = np.sqrt((fr**2).mean(axis=1) + 1e-12)
    thr = max(peak * 10 ** (-40 / 20), 1e-5)
    nonsilence = float((fr_rms > thr).mean())

    rep.update({
        "duration_s": dur, "sample_rate": sr, "channels": ch, "sample_width_bytes": sw,
        "peak": peak, "rms": rms, "rms_dbfs": 20 * np.log10(rms + 1e-12),
        "clipping_fraction": clip_frac, "non_silence_ratio": nonsilence,
        "n_samples": int(len(x)),
    })

    if dur < 0.5:
        rep["flags"].append("TOO_SHORT")
    if expected_seconds and abs(dur - expected_seconds) > max(1.0, 0.15 * expected_seconds):
        rep["flags"].append(f"DURATION_MISMATCH expected~{expected_seconds}s got {dur:.1f}s")
    if sr not in (16000, 22050, 32000, 44100, 48000):
        rep["flags"].append(f"UNEXPECTED_SAMPLE_RATE:{sr}")
    if clip_frac > 0.01:
        rep["flags"].append(f"CLIPPING {clip_frac*100:.1f}% -- lower the input gain")
    if peak < 0.005:
        rep["flags"].append("NEARLY_SILENT -- check the microphone is connected and unmuted")
    if rms > 0 and 20 * np.log10(rms) > -6:
        rep["flags"].append("VERY_HOT -- risk of clipping on louder steps")
    if nonsilence < 0.05:
        rep["flags"].append("ALMOST_NO_SPEECH_DETECTED")
    if meta is None:
        rep["flags"].append("NO_METADATA")
    else:
        for k in ("session_id", "wearer_id", "condition", "source_label", "capture_device", "utc"):
            if not meta.get(k):
                rep["flags"].append(f"METADATA_MISSING:{k}")

    rep["ok"] = len(rep["flags"]) == 0
    return rep


# ---------------------------------------------------------------------------
# session runner
# ---------------------------------------------------------------------------
def run_session(wearer_index: int, source: str = "ffmpeg-test", device: Optional[str] = None,
                sample_rate: int = 48000, root: Path = CAPTURE_ROOT,
                scale_seconds: float = 1.0, steps: Optional[List[str]] = None,
                environment_note: str = "", interactive: bool = False) -> Dict[str, object]:
    session_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{anonymous_wearer_id(wearer_index)}"
    sdir = root / session_id
    sdir.mkdir(parents=True, exist_ok=True)
    wid = anonymous_wearer_id(wearer_index)

    chosen = [s for s in PROTOCOL if steps is None or s.step_id in steps]
    records: List[Dict[str, object]] = []
    for s in chosen:
        secs = max(s.seconds * scale_seconds, 1.0)
        if interactive:
            input(f"\n[{s.step_id}] {s.instruction}\n  ({secs:.0f}s) press ENTER to start...")
        orig = sdir / f"{s.step_id}__orig.wav"
        cap = capture_segment(orig, secs, source, device, sample_rate, 1)
        meta = {
            "session_id": session_id,
            "wearer_id": wid,
            "step_id": s.step_id,
            "condition": s.condition,
            "source_label": s.source_label,
            "instruction": s.instruction,
            "capture_device": device or source,
            "capture_source_kind": source,
            "synthetic": cap["synthetic"],
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "requested_seconds": secs,
            "native_sample_rate": sample_rate,
            "environment_note": environment_note,
            "track_settings": {
                "echoCancellation": False, "noiseSuppression": False, "autoGainControl": False,
                "note": ("Requested constraints for the browser path. The ACTUAL applied values "
                         "must be read from track.getSettings() and recorded -- see "
                         "web/app/composables/useMicCapture.ts, which already does this."),
            },
            "processing": "NONE -- raw PCM at the native rate; no AGC/EQ/normalisation applied",
            "upload": "none (no network code in this module)",
        }
        derived = sdir / f"{s.step_id}__16k.wav"
        d16 = derive_16k(orig, derived) if cap["ffmpeg_returncode"] == 0 else {"returncode": -1}
        val = validate_recording(orig, expected_seconds=secs, meta=meta)
        rec = {"step": dataclasses.asdict(s), "capture": cap, "metadata": meta,
               "derived_16k": d16, "validation": val}
        (sdir / f"{s.step_id}__meta.json").write_text(json.dumps(rec, indent=2))
        records.append(rec)
        status = "OK" if val.get("ok") else "FLAGGED " + ";".join(val.get("flags", []))
        print(f"  [{s.step_id:16s}] {val.get('duration_s', 0):5.1f}s "
              f"rms {val.get('rms_dbfs', float('nan')):6.1f} dBFS clip {val.get('clipping_fraction', 0)*100:4.1f}%  {status}",
              flush=True)

    n_ok = sum(1 for r in records if r["validation"].get("ok"))
    summary = {
        "session_id": session_id,
        "wearer_id": wid,
        "directory": str(sdir),
        "n_steps": len(records),
        "n_valid": n_ok,
        "n_flagged": len(records) - n_ok,
        "synthetic_smoke_test": source == "ffmpeg-test",
        "total_seconds": sum(r["validation"].get("duration_s", 0) for r in records),
        "records": [{"step_id": r["step"]["step_id"], "ok": r["validation"].get("ok"),
                     "flags": r["validation"].get("flags")} for r in records],
    }
    (sdir / "session.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print the pilot power plan")
    ap.add_argument("--smoke-test", action="store_true",
                    help="run the whole pipeline with a synthetic ffmpeg source")
    ap.add_argument("--record", type=int, default=None, metavar="WEARER_INDEX")
    ap.add_argument("--source", default="ffmpeg-device", choices=["ffmpeg-device", "ffmpeg-test"])
    ap.add_argument("--device", default="default")
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--scale-seconds", type=float, default=1.0)
    ap.add_argument("--environment-note", default="")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--out", default="geowearnet_capture_status.json")
    a = ap.parse_args()

    out: Dict[str, object] = {}
    if a.plan or not (a.smoke_test or a.record is not None):
        out["pilot_plan"] = pilot_plan()
        print(json.dumps(out["pilot_plan"], indent=2))
    if a.smoke_test:
        print("\n== capture pipeline smoke test (synthetic source) ==", flush=True)
        out["smoke_test"] = run_session(99, source="ffmpeg-test", sample_rate=48000,
                                        scale_seconds=0.04,
                                        environment_note="synthetic smoke test, no microphone on this host")
        print(json.dumps({k: v for k, v in out["smoke_test"].items() if k != "records"}, indent=2))
    if a.record is not None:
        out["session"] = run_session(a.record, a.source, a.device, a.sample_rate,
                                     scale_seconds=a.scale_seconds,
                                     environment_note=a.environment_note,
                                     interactive=a.interactive)
        print(json.dumps(out["session"], indent=2))

    p = REPO_ROOT / "evaluation/geowearnet/results" / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
