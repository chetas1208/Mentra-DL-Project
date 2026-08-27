"""P1.7 -- SYNTHETIC PRODUCT STRESS BENCH.

*** THIS IS A SYNTHETIC PRODUCT STRESS BENCH. IT IS NOT REAL AUTOBODY    ***
*** VALIDATION. No audio here was recorded in a shop, on Mentra glasses, ***
*** or by a real technician. It is LibriSpeech speech digitally mixed    ***
*** with MUSAN noise under a hand-written room model. Real validation    ***
*** requires the capture protocol in docs/geowearnet_capture_protocol.md ***

Every artifact this module writes carries that label in a
`DISCLAIMER`/`is_synthetic` field, and every report section built from it
must repeat it. Treating these numbers as shop-floor evidence would be the
single easiest way to mislead this project.

WHY IT EXISTS ANYWAY
MMCSG is real wearable audio but it is a calm two-person conversation
corpus: no impact wrenches, no compressors, no shop radio. The customer
problem is defined by exactly those conditions. This bench lets us vary
noise type, SNR, bystander loudness, and overlap *deliberately and
repeatably*, and -- crucially -- it PRESERVES THE CLEAN SOURCES, so wearer
retention and bystander leakage can be measured exactly rather than
inferred.

CONSTRUCTION
  wearer      LibriSpeech utterance, near-field: high level, minimal reverb
  bystander   different LibriSpeech speaker, far-field: lower level, more
              reverb, band-limited slightly (distance/air absorption proxy)
  machinery   MUSAN `noise` (impact/continuous), MUSAN `music` for radio/TV
  reverb      synthetic exponentially-decaying RIR, seeded per scenario

Ground-truth SELF/OTHER frame activity comes from the CLEAN premixed
sources (energy-based VAD on a signal with no interference in it), so the
labels are exact by construction rather than estimated from the mixture.

DETERMINISM: every scenario is generated from an explicit integer seed. The
same seed reproduces byte-identical audio, so cached ASR transcripts stay
valid across runs and results are reproducible.

LICENSING: LibriSpeech is CC BY 4.0; MUSAN is CC BY 4.0. Both are already
in-repo under `evaluation/data/raw/` for prior workstreams. No new data is
downloaded and no licensed/restricted corpus is used.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRI_ROOT = REPO_ROOT / "evaluation/data/raw/LibriSpeech/test-clean"
MUSAN_ROOT = REPO_ROOT / "evaluation/data/raw/musan"

SR = 16000
HOP_S = 0.01

DISCLAIMER = ("SYNTHETIC PRODUCT STRESS BENCH -- digitally mixed LibriSpeech + MUSAN. "
              "NOT real autobody validation, NOT recorded on Mentra hardware.")


# ---------------------------------------------------------------------------
# scenario definitions
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Scenario:
    name: str
    wearer: bool = True
    bystander: bool = False
    overlap: bool = False              # bystander speaks OVER the wearer
    noise_kind: Optional[str] = None   # "noise" | "music" | None
    noise_snr_db: float = 15.0         # wearer-to-noise SNR
    bystander_rel_db: float = -8.0     # bystander level relative to wearer
    wearer_gain_db: float = 0.0        # quiet/loud wearer
    reverb_t60_s: float = 0.20
    description: str = ""


def scenario_suite() -> List[Scenario]:
    """The scenarios named in P1.7, plus the levels that make them distinct."""
    return [
        Scenario("wearer_only", description="baseline: technician alone, quiet bay",
                 reverb_t60_s=0.15),
        Scenario("coworker_only", wearer=False, bystander=True, bystander_rel_db=0.0,
                 description="only a coworker speaks -- nothing should reach the agent"),
        Scenario("wearer_plus_machinery", noise_kind="noise", noise_snr_db=5.0,
                 description="impact wrench / compressor over the technician"),
        Scenario("wearer_plus_loud_coworker", bystander=True, bystander_rel_db=+4.0,
                 description="coworker LOUDER than the wearer, sequential turns"),
        Scenario("wearer_coworker_overlap", bystander=True, overlap=True,
                 bystander_rel_db=-2.0,
                 description="simultaneous speech -- the state a router cannot solve"),
        Scenario("wearer_machinery_coworker", bystander=True, noise_kind="noise",
                 noise_snr_db=8.0, bystander_rel_db=-4.0,
                 description="full shop: technician + coworker + machinery"),
        Scenario("quiet_wearer", wearer_gain_db=-12.0, noise_kind="noise", noise_snr_db=10.0,
                 description="technician speaking softly under a mask/at distance"),
        Scenario("loud_bystander", bystander=True, bystander_rel_db=+10.0,
                 description="shouted bystander, worst case for a level-based detector"),
        Scenario("music_radio", noise_kind="music", noise_snr_db=6.0,
                 description="shop radio / TV-like spectral profile"),
        Scenario("high_reverberation", reverb_t60_s=0.85, bystander=True,
                 bystander_rel_db=-5.0,
                 description="bare-walled bay, long tail, smeared onsets"),
    ]


# ---------------------------------------------------------------------------
# source material
# ---------------------------------------------------------------------------
INDEX_CACHE = REPO_ROOT / "evaluation/agent_audio/cache/librispeech_index.json"


def _librispeech_index(max_speakers: int = 40) -> List[Dict[str, object]]:
    """[(flac path, transcript, speaker, duration_s)] from the local
    test-clean tree.

    Durations are indexed (and cached) because they are load-bearing, not
    cosmetic: an utterance longer than the mix window would be TRUNCATED in
    audio while keeping its FULL reference transcript, which silently
    inflates every wearer-deletion and WER number in the bench. Selection
    below only ever picks utterances that fit whole.
    """
    if INDEX_CACHE.exists():
        try:
            cached = json.loads(INDEX_CACHE.read_text())
            return [{**u, "path": Path(u["path"])} for u in cached]
        except Exception:
            pass
    out: List[Dict[str, object]] = []
    if not LIBRI_ROOT.is_dir():
        return out
    for spk_dir in sorted(LIBRI_ROOT.iterdir())[:max_speakers]:
        if not spk_dir.is_dir():
            continue
        for chap in sorted(spk_dir.iterdir()):
            trans = chap / f"{spk_dir.name}-{chap.name}.trans.txt"
            if not trans.exists():
                continue
            for line in trans.read_text().splitlines():
                uid, _, text = line.partition(" ")
                f = chap / f"{uid}.flac"
                if f.exists():
                    try:
                        dur = float(sf.info(str(f)).duration)
                    except Exception:
                        continue
                    out.append({"path": f, "text": text.strip(), "speaker": spk_dir.name,
                                "utt_id": uid, "duration_s": dur})
    try:
        INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        INDEX_CACHE.write_text(json.dumps(
            [{**u, "path": str(u["path"])} for u in out]))
    except Exception:
        pass
    return out


def _pick_fitting(libri: Sequence[Dict[str, object]], rng: np.random.Generator,
                  max_dur_s: float, exclude_speaker: Optional[str] = None,
                  tries: int = 200) -> Optional[Dict[str, object]]:
    """Pick a random utterance that fits WHOLE inside `max_dur_s`, optionally
    from a different speaker. Returns None rather than silently truncating."""
    pool = [u for u in libri
            if float(u.get("duration_s", 1e9)) <= max_dur_s
            and (exclude_speaker is None or u["speaker"] != exclude_speaker)]
    if not pool:
        return None
    return pool[int(rng.integers(0, len(pool)))]


def _musan_index(kind: str) -> List[Path]:
    d = MUSAN_ROOT / kind
    return sorted(d.rglob("*.wav")) if d.is_dir() else []


def _read(path: Path, want_s: float = None, rng: np.random.Generator = None) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    if want_s is not None:
        n = int(want_s * SR)
        if len(x) < n:
            reps = int(np.ceil(n / max(len(x), 1)))
            x = np.tile(x, reps)
        if len(x) > n:
            start = int(rng.integers(0, len(x) - n)) if rng is not None else 0
            x = x[start:start + n]
    return np.ascontiguousarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# room / level model
# ---------------------------------------------------------------------------
def synthetic_rir(t60_s: float, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Exponentially-decaying noise RIR with a direct-path impulse. Crude but
    deterministic, documented, and adequate for a relative-comparison bench --
    it is NOT a measured room response and is not presented as one."""
    n = max(int(t60_s * sr), 16)
    t = np.arange(n) / sr
    h = rng.normal(0, 1, n).astype(np.float32) * np.exp(-6.9078 * t / max(t60_s, 1e-3))
    h[0] += 3.0                       # direct path dominates
    return (h / np.sqrt(np.sum(h ** 2) + 1e-12)).astype(np.float32)


def _reverberate(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    y = np.convolve(x, rir, mode="full")[: len(x)]
    return y.astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-12))


def _scale_to_db(x: np.ndarray, ref: np.ndarray, db: float) -> np.ndarray:
    target = _rms(ref) * (10 ** (db / 20.0))
    return (x * (target / max(_rms(x), 1e-12))).astype(np.float32)


ABSOLUTE_SILENCE_RMS = 1e-5


def energy_vad(clean: np.ndarray, hop_s: float = HOP_S, sr: int = SR,
               rel_db: float = -32.0) -> np.ndarray:
    """Frame activity from a CLEAN premixed source. Because the source has no
    interference in it, a simple relative-energy threshold is exact enough to
    serve as ground truth -- this is the advantage of synthesising the mix.

    The absolute floor is not decoration: a purely relative threshold on an
    all-zero signal (a scenario where this speaker never talks) marks EVERY
    frame active, because max(e) collapses to the epsilon and every frame
    clears it. That silently inverts the labels for any single-speaker
    scenario, so silence is detected explicitly first.
    """
    hop = int(round(sr * hop_s))
    n = len(clean) // hop
    if n <= 0:
        return np.zeros(0, np.float32)
    fr = clean[: n * hop].reshape(n, hop).astype(np.float64)
    e = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    if float(e.max()) < ABSOLUTE_SILENCE_RMS:
        return np.zeros(n, np.float32)
    thr = max(e.max() * (10 ** (rel_db / 20.0)), ABSOLUTE_SILENCE_RMS)
    return (e > thr).astype(np.float32)


# ---------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class StressItem:
    item_id: str
    scenario: str
    audio: np.ndarray
    wearer_active: np.ndarray
    env_active: np.ndarray
    self_text: str
    other_text: str
    clean_wearer: np.ndarray
    clean_bystander: np.ndarray
    wearer_commands: Tuple[str, ...]
    bystander_commands: Tuple[str, ...]
    meta: Dict[str, object]


def _phrase_from(text: str, n_words: int = 5) -> str:
    """A 'command surrogate': a distinctive contiguous phrase the speaker
    ACTUALLY said in this utterance.

    Why surrogates and not the literal shop phrases in `commands.COMMAND_SET`:
    no local corpus contains a human saying "hey glasses what part is this",
    and no TTS model is present in this environment. Rather than fabricate
    audio for those exact words (or download a TTS model to do it), the bench
    measures the SAME quantities -- did a specific multi-word phrase spoken by
    the wearer survive, did a specific phrase spoken by the bystander get
    injected -- using phrases that are genuinely present in the audio. The
    metric code is phrase-agnostic, so real shop phrases drop straight in once
    the P1.17 capture produces them.
    """
    w = text.split()
    if len(w) < n_words:
        return " ".join(w)
    start = max(0, (len(w) - n_words) // 2)
    return " ".join(w[start:start + n_words])


def build_item(sc: Scenario, seed: int, duration_s: float = 8.0,
               libri: Optional[List[Dict[str, object]]] = None) -> Optional[StressItem]:
    rng = np.random.default_rng(seed)
    libri = libri if libri is not None else _librispeech_index()
    if len(libri) < 2:
        return None

    n = int(duration_s * SR)

    # --- pick two DIFFERENT speakers, both fitting WHOLE in the window ----
    # Budget: for sequential (turn-taking) scenarios the two utterances must
    # fit end-to-end; for overlap scenarios they may share the window.
    if sc.bystander and not sc.overlap:
        max_each = (duration_s - 1.0) / 2.0
    else:
        max_each = duration_s - 1.5
    wearer_utt = _pick_fitting(libri, rng, max_each)
    if wearer_utt is None:
        return None
    byst_utt = _pick_fitting(libri, rng, max_each, exclude_speaker=wearer_utt["speaker"])
    if byst_utt is None:
        return None

    rir_w = synthetic_rir(min(sc.reverb_t60_s, 0.25), rng)   # near-field: less reverb
    rir_b = synthetic_rir(sc.reverb_t60_s, rng)

    # --- wearer -----------------------------------------------------------
    clean_w = np.zeros(n, np.float32)
    self_text = ""
    if sc.wearer:
        w = _read(wearer_utt["path"])
        # place the wearer's speech starting near the beginning
        start = int(rng.integers(0, max(int(0.5 * SR), 1)))
        if start + len(w) > n:
            return None      # would truncate audio but not the transcript
        seg = w
        clean_w[start:start + len(seg)] = seg
        clean_w = _reverberate(clean_w, rir_w)
        clean_w = (clean_w * (10 ** (sc.wearer_gain_db / 20.0))).astype(np.float32)
        self_text = wearer_utt["text"]

    # --- bystander --------------------------------------------------------
    clean_b = np.zeros(n, np.float32)
    other_text = ""
    if sc.bystander:
        b = _read(byst_utt["path"])
        if sc.overlap:
            # start DURING the wearer's speech -> genuine state 11
            start = int(rng.integers(int(0.2 * SR), max(int(1.2 * SR), int(0.2 * SR) + 1)))
        else:
            # sequential turn-taking -> mostly state 01, minimal 11
            wa = energy_vad(clean_w)
            if wa.any():
                last = int(np.max(np.nonzero(wa)[0]))
                start = int((last + 30) * HOP_S * SR)   # 300 ms turn gap
            else:
                # nobody else is talking (e.g. coworker_only): start early
                start = int(rng.integers(0, max(int(0.5 * SR), 1)))
            start = max(0, min(start, max(n - len(b), 0)))
        if start + len(b) > n:
            return None      # would truncate audio but not the transcript
        seg = b
        clean_b[start:start + len(seg)] = seg
        clean_b = _reverberate(clean_b, rir_b)
        ref = clean_w if sc.wearer and _rms(clean_w) > 1e-6 else np.ones(n, np.float32) * 0.05
        clean_b = _scale_to_db(clean_b, ref, sc.bystander_rel_db)
        other_text = byst_utt["text"]

    # --- machinery / music -------------------------------------------------
    noise = np.zeros(n, np.float32)
    noise_file = None
    if sc.noise_kind:
        files = _musan_index(sc.noise_kind)
        if files:
            noise_file = files[int(rng.integers(0, len(files)))]
            noise = _read(noise_file, duration_s, rng)[:n]
            if len(noise) < n:
                noise = np.pad(noise, (0, n - len(noise)))
            ref = clean_w if sc.wearer and _rms(clean_w) > 1e-6 else clean_b
            if _rms(ref) > 1e-9:
                noise = _scale_to_db(noise, ref, -sc.noise_snr_db)

    mix = (clean_w + clean_b + noise).astype(np.float32)

    # headroom: scale the mix (and the clean references identically) if it
    # would clip, so level relationships are preserved exactly
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 0.99:
        k = 0.99 / peak
        mix, clean_w, clean_b = mix * k, clean_w * k, clean_b * k

    wa = energy_vad(clean_w) if sc.wearer else np.zeros(n // int(SR * HOP_S), np.float32)
    ea = energy_vad(clean_b) if sc.bystander else np.zeros(n // int(SR * HOP_S), np.float32)
    m = min(len(wa), len(ea))
    wa, ea = wa[:m], ea[:m]

    item_id = f"stress_{sc.name}_s{seed}"
    return StressItem(
        item_id=item_id, scenario=sc.name, audio=mix,
        wearer_active=wa, env_active=ea,
        self_text=self_text, other_text=other_text,
        clean_wearer=clean_w, clean_bystander=clean_b,
        wearer_commands=(_phrase_from(self_text),) if self_text else (),
        bystander_commands=(_phrase_from(other_text),) if other_text else (),
        meta={
            "DISCLAIMER": DISCLAIMER,
            "is_synthetic": True,
            "scenario_description": sc.description,
            "seed": seed,
            "duration_s": duration_s,
            "wearer_speaker": wearer_utt["speaker"] if sc.wearer else None,
            "bystander_speaker": byst_utt["speaker"] if sc.bystander else None,
            "wearer_utt": wearer_utt["utt_id"] if sc.wearer else None,
            "bystander_utt": byst_utt["utt_id"] if sc.bystander else None,
            "noise_kind": sc.noise_kind,
            "noise_file": str(noise_file) if noise_file else None,
            "noise_snr_db": sc.noise_snr_db if sc.noise_kind else None,
            "bystander_rel_db": sc.bystander_rel_db if sc.bystander else None,
            "wearer_gain_db": sc.wearer_gain_db,
            "reverb_t60_s": sc.reverb_t60_s,
            "overlap_by_design": sc.overlap,
            "measured_overlap_frac": float(((wa > 0.5) & (ea > 0.5)).mean()) if m else 0.0,
            "sources": "LibriSpeech test-clean (CC BY 4.0) + MUSAN (CC BY 4.0), local copies",
        },
    )


def build_suite(seeds_per_scenario: int = 3, duration_s: float = 12.0,
                base_seed: int = 20260827) -> List[StressItem]:
    """Deterministic: the same (base_seed, duration_s) always yields the same
    items. `build_item` returns None rather than emitting a window whose audio
    would be truncated relative to its transcript, so we advance the seed
    until each scenario has its full quota."""
    libri = _librispeech_index()
    items: List[StressItem] = []
    for si, sc in enumerate(scenario_suite()):
        made, attempt = 0, 0
        while made < seeds_per_scenario and attempt < 100:
            it = build_item(sc, base_seed + 1000 * si + attempt, duration_s, libri)
            attempt += 1
            if it is not None:
                items.append(it)
                made += 1
    return items


def availability_report() -> Dict[str, object]:
    libri = _librispeech_index()
    return {
        "DISCLAIMER": DISCLAIMER,
        "librispeech_root": str(LIBRI_ROOT),
        "librispeech_utterances_indexed": len(libri),
        "librispeech_speakers": len({u["speaker"] for u in libri}),
        "musan_noise_files": len(_musan_index("noise")),
        "musan_music_files": len(_musan_index("music")),
        "n_scenarios": len(scenario_suite()),
        "status": "AVAILABLE" if libri and _musan_index("noise") else "BLOCKED_NO_LOCAL_CORPORA",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(availability_report(), indent=2))
