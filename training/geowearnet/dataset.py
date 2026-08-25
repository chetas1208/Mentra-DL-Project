"""GeoWearNet simulated dataset builder (E1-SIM, Phase 19).

Loads the ALREADY-SPEAKER-DISJOINT LibriSpeech train/val/test manifests
this project already has (`evaluation/manifests/librispeech_train_clean_100_*`,
251 total speakers, 0 speaker overlap across splits -- verified in this
session), and generates SIMULATED wearer/environment examples via
`training/geowearnet/simulate.py`.

PERSON-DISJOINTNESS (Phase 16): because the underlying LibriSpeech manifest
splits are already speaker-disjoint, and a "person" in the GeoWearNet sense
IS a LibriSpeech speaker here (the entity being simulated as "the wearer"),
train/val/test are automatically person-disjoint -- verified by
`training_geowearnet_test.py::test_person_disjoint_split`.

ROLE RANDOMIZATION (Phase 18): within a split, `sample_pair` draws TWO
DISTINCT speakers uniformly at random and then independently coin-flips
which one is "wearer" and which is "environment" for that example -- so
across many examples the same speaker appears on both sides. This is the
control against the network solving WHO instead of WHERE.

NO IDENTITY LEAKAGE (Phase 17): the only signal carried from "speaker" to
"example" is the raw waveform audio itself run through a role-dependent
transfer function; no speaker ID, gender, book/chapter, or file-naming
feature is ever placed in the model input or label.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

from . import simulate

REPO_ROOT = Path(__file__).resolve().parents[2]

MANIFESTS = {
    "train": REPO_ROOT / "evaluation/manifests/librispeech_train_clean_100_train.json",
    "val": REPO_ROOT / "evaluation/manifests/librispeech_train_clean_100_val.json",
    "test": REPO_ROOT / "evaluation/manifests/librispeech_train_clean_100_test.json",
}


def load_speaker_index(split: str) -> Dict[str, List[str]]:
    """Returns {speaker_id: [utterance_paths]} for a split, pooling the
    manifest's 'enroll' and 'test' lists together -- GeoWearNet has no
    enrollment concept (per the brief's NO ENROLLMENT rule), so this
    module does not preserve that enroll/test distinction; it only reuses
    the manifest for its (already speaker-disjoint) speaker pool."""
    with open(MANIFESTS[split]) as f:
        m = json.load(f)
    out = {}
    for spk, d in m["speakers"].items():
        utts = list(d.get("enroll", [])) + list(d.get("test", []))
        out[spk] = [str(REPO_ROOT / u) if not Path(u).is_absolute() else u for u in utts]
    return out


def load_utterance(path: str, max_seconds: float = 4.0, rng: random.Random = None) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    assert sr == simulate.SAMPLE_RATE, f"expected {simulate.SAMPLE_RATE}, got {sr} for {path}"
    if x.ndim > 1:
        x = x.mean(axis=1)
    max_len = int(max_seconds * sr)
    if len(x) > max_len:
        rng = rng or random
        start = rng.randint(0, len(x) - max_len)
        x = x[start : start + max_len]
    return x


def sample_pair(speaker_index: Dict[str, List[str]], rng: random.Random) -> Tuple[str, str, str, str]:
    """Returns (wearer_speaker, wearer_utt_path, env_speaker, env_utt_path)
    with role assigned independently of which speaker was drawn first
    (Phase 18 role randomization)."""
    spk_a, spk_b = rng.sample(list(speaker_index.keys()), 2)
    utt_a = rng.choice(speaker_index[spk_a])
    utt_b = rng.choice(speaker_index[spk_b])
    if rng.random() < 0.5:
        return spk_a, utt_a, spk_b, utt_b
    return spk_b, utt_b, spk_a, utt_a


def build_e0_clip_examples(
    split: str,
    n_examples: int,
    condition: str = "normal",
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """E0 diagnostic dataset: single-source clips only (no overlap), label
    1=wearer 0=environment. Returns (audio_list padded not needed since we
    return features directly by caller) -- actually returns raw waveforms
    list + labels + speaker ids, feature extraction happens in the caller
    so ablation code can reuse the same waveforms across feature-group
    conditions without re-simulating audio each time."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    speaker_index = load_speaker_index(split)
    waveforms = []
    labels = []
    speakers = []
    for i in range(n_examples):
        w_spk, w_path, e_spk, e_path = sample_pair(speaker_index, rng)
        w_utt = load_utterance(w_path, rng=rng)
        e_utt = load_utterance(e_path, rng=rng)
        is_wearer = i % 2 == 0  # exactly balanced classes
        if is_wearer:
            clip = simulate.make_single_source_clip(w_utt, "wearer", nrng, condition=condition)
            speakers.append(w_spk)
        else:
            clip = simulate.make_single_source_clip(e_utt, "environment", nrng, condition=condition)
            speakers.append(e_spk)
        waveforms.append(clip)
        labels.append(1 if is_wearer else 0)
    return waveforms, np.array(labels, dtype=np.int64), speakers


def build_e0_random_gain_examples(split: str, n_examples: int, seed: int = 0):
    """Phase 9(B): independent random gain -18..+12dB applied to BOTH
    classes on top of their transfer function's own gain."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    speaker_index = load_speaker_index(split)
    waveforms, labels, speakers = [], [], []
    for i in range(n_examples):
        w_spk, w_path, e_spk, e_path = sample_pair(speaker_index, rng)
        w_utt = load_utterance(w_path, rng=rng)
        e_utt = load_utterance(e_path, rng=rng)
        is_wearer = i % 2 == 0
        extra_db = nrng.uniform(-18.0, 12.0)
        if is_wearer:
            clip = simulate.make_single_source_clip(w_utt, "wearer", nrng, "normal", extra_gain_db=extra_db)
            speakers.append(w_spk)
        else:
            clip = simulate.make_single_source_clip(e_utt, "environment", nrng, "normal", extra_gain_db=extra_db)
            speakers.append(e_spk)
        waveforms.append(clip)
        labels.append(1 if is_wearer else 0)
    return waveforms, np.array(labels, dtype=np.int64), speakers


def build_e1_mixture_examples(split: str, n_examples: int, condition: str = "normal", seed: int = 0):
    """E1 frame-level examples: overlap mixtures with independent wearer/
    environment placement, returns list of (mixture, wearer_activity,
    environment_activity) sample-resolution arrays."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    speaker_index = load_speaker_index(split)
    examples = []
    for _ in range(n_examples):
        w_spk, w_path, e_spk, e_path = sample_pair(speaker_index, rng)
        w_utt = load_utterance(w_path, max_seconds=2.0, rng=rng)
        e_utt = load_utterance(e_path, max_seconds=2.0, rng=rng)
        mix, w_act, e_act = simulate.make_overlap_mixture(w_utt, e_utt, nrng, condition=condition)
        examples.append((mix, w_act, e_act))
    return examples
