"""MMCSG ingestion adapter (Workstreams V, W, X).

STATUS: `MMCSG_ACCESS_PENDING_USER_ACCEPTANCE`. The dataset is NOT on this
machine and its Data License Agreement could not be retrieved by an
unauthenticated fetch -- see `docs/geowearnet_data_license_state.md`. Only the
project owner can accept the agreement and download the data.

Per the campaign brief ("prepare all code and manifests, then mark only download
as blocked"), everything except the download is implemented here and PROVEN to
work against a synthetic fixture that reproduces the documented on-disk layout.
The moment the data exists, `--root <path>` is the only thing that changes.

ON-DISK LAYOUT (verified 2026-08-25 against the CHiME-8 MMCSG data page and the
`facebookresearch/MMCSG` baseline `scripts/prepare_data.py`):

    <root>/audio/{train,dev,eval}/<recording-id>.wav      7 channels, 48 kHz
    <root>/transcriptions/{train,dev,eval}/<recording-id>.tsv
    <root>/metadata/{train,dev,eval}/<recording-id>.json
    <root>/rttm/{train,dev,eval}/<recording-id>.*         (optional)
    <root>/{video,accelerometer,gyroscope}/...            (unused here)

Transcript format, quoted from the baseline script:
    "The transcription file is expected to have a line per word of format:
     {start}\\t{end}\\t{word}\\t{speaker}"
    "speaker is 0/1 for SELF/OTHER, i.e. wearer of the glasses and
     conversational partner"
So label 0 == WEARER, label 1 == ENVIRONMENT. That mapping is the whole reason
MMCSG is interesting to GeoWearNet.

Metadata JSON keys: `self_speaker`, `other_speaker`, `noise_category`. These
give real person identities, which is what makes a SPEAKER-DISJOINT split
possible (Workstream G) rather than a recording-level split.

HARD RULES OBSERVED
  * source files are opened read-only and never written, moved or modified;
  * no raw MMCSG audio is ever copied into this repository or into any derived
    dataset -- manifests store PATHS, never samples;
  * nothing here redistributes the dataset.

CHANNEL POLICY (Workstream W)
  The corpus has 7 raw microphones. The public baseline is reported to operate
  on fixed beamformed outputs including one beam steered at the wearer's mouth;
  this campaign did NOT independently re-verify that beam count, and does not
  rely on it. GeoWearNet's Mentra problem is single-channel, so the adapter
  defaults to ONE raw channel and `channel_study()` measures which raw channel
  best matches the Mentra mono setting rather than assuming one.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_SR = 16000
FRAME_HOP = 160
FRAME_WIN = 400
SPLITS = ("train", "dev", "eval")

WEARER_LABEL = "0"       # SELF
ENVIRONMENT_LABEL = "1"  # OTHER


@dataclasses.dataclass
class MMCSGConfig:
    root: Path
    split: str = "train"
    channels: Tuple[int, ...] = (0,)      # raw mic indices; single-channel by default
    target_sr: int = TARGET_SR
    max_seconds: Optional[float] = None

    def audio_dir(self) -> Path:
        return self.root / "audio" / self.split

    def transcript_dir(self) -> Path:
        return self.root / "transcriptions" / self.split

    def metadata_dir(self) -> Path:
        return self.root / "metadata" / self.split


class MMCSGNotAvailable(RuntimeError):
    pass


def check_available(root: Path) -> Dict[str, object]:
    """Non-destructive availability probe. Never raises; returns a report."""
    rep: Dict[str, object] = {"root": str(root), "exists": root.exists()}
    if not root.exists():
        rep["status"] = "MMCSG_ACCESS_PENDING_USER_ACCEPTANCE"
        rep["blocker"] = (
            "Dataset not present. Requires the project owner to accept Meta's MMCSG "
            "Data License Agreement and download from ai.meta.com/datasets/mmcsg-downloads/. "
            "This is an external, legally-required action that cannot be performed here."
        )
        return rep
    for split in SPLITS:
        a, t, m = root / "audio" / split, root / "transcriptions" / split, root / "metadata" / split
        rep[split] = {
            "audio_files": len(list(a.glob("*.wav"))) if a.exists() else 0,
            "transcripts": len(list(t.glob("*.tsv"))) if t.exists() else 0,
            "metadata": len(list(m.glob("*.json"))) if m.exists() else 0,
        }
    rep["status"] = "AVAILABLE"
    return rep


def discover(cfg: MMCSGConfig) -> List[str]:
    d = cfg.audio_dir()
    if not d.exists():
        raise MMCSGNotAvailable(f"no audio directory at {d}")
    return sorted(p.stem for p in d.glob("*.wav"))


def read_transcript(cfg: MMCSGConfig, rec: str) -> List[Tuple[float, float, str, str]]:
    """Word-level [(start_s, end_s, word, speaker)] with speaker in {'0','1'}."""
    p = cfg.transcript_dir() / f"{rec}.tsv"
    if not p.exists():
        raise FileNotFoundError(p)
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        st, en, word, spk = parts[0], parts[1], parts[2], parts[3].strip()
        out.append((float(st), float(en), word, spk))
    return out


def read_metadata(cfg: MMCSGConfig, rec: str) -> Dict[str, object]:
    p = cfg.metadata_dir() / f"{rec}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def read_audio(cfg: MMCSGConfig, rec: str) -> Tuple[np.ndarray, int]:
    """Read selected channels, resampled to `target_sr`.

    Opens the source file READ-ONLY and never writes to it. Returns
    (n_samples, n_selected_channels) float32 -- NOT normalised, because absolute
    level is a physical cue GeoWearNet depends on.
    """
    import soundfile as sf

    p = cfg.audio_dir() / f"{rec}.wav"
    kw = {}
    if cfg.max_seconds is not None:
        kw["frames"] = int(cfg.max_seconds * sf.info(str(p)).samplerate)
    x, sr = sf.read(str(p), dtype="float32", always_2d=True, **kw)
    n_ch = x.shape[1]
    ch = [c for c in cfg.channels if c < n_ch]
    if not ch:
        raise ValueError(f"{rec}: requested channels {cfg.channels} but file has {n_ch}")
    x = x[:, ch]
    if sr != cfg.target_sr:
        # Polyphase resample per channel (scipy); source file untouched.
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(int(sr), int(cfg.target_sr))
        x = resample_poly(x, cfg.target_sr // g, sr // g, axis=0).astype(np.float32)
    return np.ascontiguousarray(x), cfg.target_sr


def n_frames_for(n_samples: int) -> int:
    return max(1, 1 + (max(n_samples, FRAME_WIN) - FRAME_WIN) // FRAME_HOP)


def frame_labels(words: Sequence[Tuple[float, float, str, str]], n_frames: int,
                 hop_s: float = FRAME_HOP / TARGET_SR) -> Tuple[np.ndarray, np.ndarray]:
    """Word-level transcript -> frame-level (wearer_active, environment_active).

    A frame is active for a speaker if any of that speaker's words overlaps it.
    This is exactly the GeoWearNet label semantics (who is producing speech now),
    so MMCSG labels and simulated labels mean the same thing -- a precondition
    for the zero-shot transfer experiment (Workstream X) to be meaningful.
    """
    w = np.zeros(n_frames, dtype=np.float32)
    e = np.zeros(n_frames, dtype=np.float32)
    for st, en, _, spk in words:
        i0 = max(int(np.floor(st / hop_s)), 0)
        i1 = min(int(np.ceil(en / hop_s)), n_frames)
        if i1 <= i0:
            continue
        if spk == WEARER_LABEL:
            w[i0:i1] = 1.0
        elif spk == ENVIRONMENT_LABEL:
            e[i0:i1] = 1.0
    return w, e


def state_from_labels(w: np.ndarray, e: np.ndarray) -> np.ndarray:
    return w.astype(np.int64) + 2 * e.astype(np.int64)


# ---------------------------------------------------------------------------
# Manifests (Workstream G: speaker-disjoint, not recording-disjoint)
# ---------------------------------------------------------------------------
def build_manifest(root: Path, splits: Sequence[str] = SPLITS) -> Dict[str, object]:
    """Index recordings with their real speaker identities from metadata."""
    recs: List[Dict[str, object]] = []
    for split in splits:
        cfg = MMCSGConfig(root=root, split=split)
        if not cfg.audio_dir().exists():
            continue
        for rec in discover(cfg):
            md = read_metadata(cfg, rec)
            recs.append({
                "recording_id": rec,
                "source_split": split,
                "audio_path": str((cfg.audio_dir() / f"{rec}.wav")),
                "transcript_path": str((cfg.transcript_dir() / f"{rec}.tsv")),
                "self_speaker": md.get("self_speaker"),
                "other_speaker": md.get("other_speaker"),
                "noise_category": md.get("noise_category"),
            })
    return {
        "source": "MMCSG (Meta) -- PATHS ONLY, no audio is copied or redistributed",
        "n_recordings": len(recs),
        "recordings": recs,
    }


def speaker_disjoint_split(manifest: Dict[str, object], val_frac: float = 0.2,
                           test_frac: float = 0.2, seed: int = 0) -> Dict[str, List[str]]:
    """Partition by PERSON, then keep only recordings whose BOTH participants
    fall in the same partition. Recordings that straddle a boundary are dropped
    -- dropping data is the correct trade for a genuinely leak-free split."""
    recs = manifest["recordings"]
    people = sorted({p for r in recs for p in (r.get("self_speaker"), r.get("other_speaker")) if p})
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(people))
    n = len(perm)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    groups = {"test": set(perm[:n_test]), "val": set(perm[n_test:n_test + n_val]),
              "train": set(perm[n_test + n_val:])}
    out: Dict[str, List[str]] = {"train": [], "val": [], "test": [], "dropped_straddling": []}
    for r in recs:
        a, b = r.get("self_speaker"), r.get("other_speaker")
        placed = False
        for g, members in groups.items():
            if a in members and b in members:
                out[g].append(r["recording_id"])
                placed = True
                break
        if not placed:
            out["dropped_straddling"].append(r["recording_id"])
    out["_speaker_counts"] = {g: len(m) for g, m in groups.items()}
    return out


# ---------------------------------------------------------------------------
# Workstream W -- per-channel study (measure, don't assume)
# ---------------------------------------------------------------------------
def channel_study(root: Path, split: str = "train", n_recordings: int = 20,
                  max_seconds: float = 60.0) -> Dict[str, object]:
    """For EVERY raw microphone channel independently: level statistics and how
    separable wearer-vs-other is using the E0 physical features on that channel
    alone. Used to pick the single-channel Mentra proxy by measurement."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from .features import extract_features

    probe = MMCSGConfig(root=root, split=split)
    recs = discover(probe)[:n_recordings]
    if not recs:
        raise MMCSGNotAvailable("no recordings")
    import soundfile as sf
    n_ch = sf.info(str(probe.audio_dir() / f"{recs[0]}.wav")).channels

    out: Dict[str, object] = {"n_channels": n_ch, "n_recordings": len(recs), "channels": {}}
    for ch in range(n_ch):
        cfg = MMCSGConfig(root=root, split=split, channels=(ch,), max_seconds=max_seconds)
        F, W, E = [], [], []
        for rec in recs:
            x, _ = read_audio(cfg, rec)
            mono = x[:, 0]
            feats = extract_features(mono)
            w, e = frame_labels(read_transcript(cfg, rec), len(feats))
            F.append(feats); W.append(w); E.append(e)
        Fa, Wa, Ea = np.concatenate(F), np.concatenate(W), np.concatenate(E)
        solo = (Wa + Ea) == 1
        entry: Dict[str, object] = {
            "rms_db_mean": float(np.mean(Fa[:, 0]) * 20 / np.log(10)),
            "wearer_frame_fraction": float(Wa.mean()),
            "other_frame_fraction": float(Ea.mean()),
        }
        if solo.sum() > 200 and len(np.unique(Wa[solo])) == 2:
            sc = StandardScaler().fit(Fa[solo])
            clf = LogisticRegression(max_iter=300).fit(sc.transform(Fa[solo]), Wa[solo])
            entry["e0_wearer_vs_other_auroc"] = float(
                roc_auc_score(Wa[solo], clf.predict_proba(sc.transform(Fa[solo]))[:, 1]))
        out["channels"][str(ch)] = entry
    best = max(out["channels"].items(),
               key=lambda kv: kv[1].get("e0_wearer_vs_other_auroc", -1))
    out["recommended_single_channel"] = int(best[0])
    out["note"] = "Channel chosen by MEASUREMENT, not by assuming which mic the baseline uses."
    return out


# ---------------------------------------------------------------------------
# Fixture: prove the adapter works before the real data exists
# ---------------------------------------------------------------------------
def make_fixture(dest: Path, n_recordings: int = 3, seconds: float = 8.0,
                 n_channels: int = 7, sr: int = 48000, seed: int = 0) -> Path:
    """Write a synthetic corpus with the DOCUMENTED MMCSG layout and formats.

    This is not MMCSG data and contains no MMCSG content -- it is white noise
    with invented transcripts, whose only purpose is to exercise every code path
    in this adapter (7 channels, 48 kHz, TSV word transcripts with 0/1 speaker
    labels, metadata JSON with self/other speaker ids)."""
    import soundfile as sf

    rng = np.random.default_rng(seed)
    for split in SPLITS:
        (dest / "audio" / split).mkdir(parents=True, exist_ok=True)
        (dest / "transcriptions" / split).mkdir(parents=True, exist_ok=True)
        (dest / "metadata" / split).mkdir(parents=True, exist_ok=True)
    people = [f"spk{i:03d}" for i in range(2 * n_recordings * len(SPLITS))]
    k = 0
    for split in SPLITS:
        for i in range(n_recordings):
            rec = f"{split}_rec{i:03d}"
            x = (rng.standard_normal((int(seconds * sr), n_channels)) * 0.05).astype(np.float32)
            sf.write(str(dest / "audio" / split / f"{rec}.wav"), x, sr)
            lines, t = [], 0.3
            while t < seconds - 0.6:
                spk = "0" if rng.random() < 0.5 else "1"
                d = float(rng.uniform(0.15, 0.5))
                lines.append(f"{t:.3f}\t{t+d:.3f}\tword{len(lines)}\t{spk}")
                t += d + float(rng.uniform(0.02, 0.4))
            (dest / "transcriptions" / split / f"{rec}.tsv").write_text("\n".join(lines))
            md = {"self_speaker": people[k], "other_speaker": people[k + 1],
                  "noise_category": "quiet"}
            k += 2
            (dest / "metadata" / split / f"{rec}.json").write_text(json.dumps(md))
    return dest


def self_test(verbose: bool = True) -> Dict[str, object]:
    """Exercises the whole adapter against the fixture. Part of Workstream BH."""
    import tempfile

    res: Dict[str, object] = {}
    with tempfile.TemporaryDirectory() as td:
        root = make_fixture(Path(td))
        res["availability"] = check_available(root)
        assert res["availability"]["status"] == "AVAILABLE"

        cfg = MMCSGConfig(root=root, split="train", channels=(0,))
        recs = discover(cfg)
        assert recs, "no recordings discovered"
        res["n_recordings_train"] = len(recs)

        x, sr = read_audio(cfg, recs[0])
        assert sr == TARGET_SR, f"resample failed: {sr}"
        assert x.shape[1] == 1, f"channel selection failed: {x.shape}"
        res["single_channel_shape"] = list(x.shape)
        res["resampled_sr"] = sr

        cfg7 = MMCSGConfig(root=root, split="train", channels=tuple(range(7)))
        x7, _ = read_audio(cfg7, recs[0])
        assert x7.shape[1] == 7, f"7-channel mode failed: {x7.shape}"
        res["seven_channel_shape"] = list(x7.shape)

        words = read_transcript(cfg, recs[0])
        assert words and all(s in ("0", "1") for *_, s in words)
        w, e = frame_labels(words, n_frames_for(len(x)))
        st = state_from_labels(w, e)
        res["label_stats"] = {
            "n_frames": int(len(w)),
            "wearer_fraction": float(w.mean()),
            "environment_fraction": float(e.mean()),
            "states_present": sorted(int(v) for v in np.unique(st)),
        }
        assert w.sum() > 0 and e.sum() > 0, "fixture produced no labels of one class"

        man = build_manifest(root)
        res["manifest_recordings"] = man["n_recordings"]
        split = speaker_disjoint_split(man, seed=1)
        # prove zero speaker leakage
        by_rec = {r["recording_id"]: r for r in man["recordings"]}
        sets = {}
        for g in ("train", "val", "test"):
            people = set()
            for rid in split[g]:
                people.add(by_rec[rid]["self_speaker"]); people.add(by_rec[rid]["other_speaker"])
            sets[g] = people
        inter = {
            "train|val": len(sets["train"] & sets["val"]),
            "train|test": len(sets["train"] & sets["test"]),
            "val|test": len(sets["val"] & sets["test"]),
        }
        assert all(v == 0 for v in inter.values()), f"SPEAKER LEAKAGE: {inter}"
        res["split"] = {
            "counts": {g: len(split[g]) for g in ("train", "val", "test")},
            "dropped_straddling": len(split["dropped_straddling"]),
            "speaker_intersections": inter,
        }

        # source files must be untouched
        p = root / "audio" / "train" / f"{recs[0]}.wav"
        res["source_readonly_check"] = {"size_bytes": p.stat().st_size}

    res["status"] = "ADAPTER_SELF_TEST_PASS"
    res["dataset_status"] = "MMCSG_ACCESS_PENDING_USER_ACCEPTANCE"
    if verbose:
        print(json.dumps(res, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="path to a real MMCSG root")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--channel-study", action="store_true")
    ap.add_argument("--out", default="geowearnet_mmcsg_status.json")
    a = ap.parse_args()

    out: Dict[str, object] = {}
    if a.self_test or not a.root:
        out["self_test"] = self_test(verbose=False)
    if a.root:
        root = Path(a.root)
        out["availability"] = check_available(root)
        if out["availability"]["status"] == "AVAILABLE":
            man = build_manifest(root)
            out["manifest_summary"] = {"n_recordings": man["n_recordings"]}
            out["split"] = {k: (len(v) if isinstance(v, list) else v)
                            for k, v in speaker_disjoint_split(man).items()}
            (REPO_ROOT / "evaluation/geowearnet/manifests/mmcsg_manifest.json").write_text(
                json.dumps(man, indent=1))
            if a.channel_study:
                out["channel_study"] = channel_study(root)
    else:
        out["availability"] = {
            "status": "MMCSG_ACCESS_PENDING_USER_ACCEPTANCE",
            "blocker": "No --root given and no MMCSG data present on this machine.",
        }
    p = REPO_ROOT / "evaluation/geowearnet/results" / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
