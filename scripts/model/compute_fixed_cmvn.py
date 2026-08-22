#!/usr/bin/env python3
"""Compute fixed global CMVN (mean/std per mel-feature channel) from real
audio, to replace SpeakerNet's default unbounded per_feature normalization
for the streaming (C0) variant. Uses the same day1_public LibriSpeech
speakers already in the repo -- not new data, no new download.
"""
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import nemo.collections.asr as nemo_asr


def main():
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name="speakerverification_speakernet")
    model = model.to("cpu")
    featurizer = model.preprocessor.featurizer

    manifest = json.loads(Path("evaluation/manifests/day1_public_speakers.json").read_text())
    all_files = []
    for entry in manifest["speakers"].values():
        all_files.extend(entry["enroll"] + entry["test"])

    print(f"computing CMVN stats from {len(all_files)} real files")

    sum_ = None
    sumsq = None
    count = 0
    for path in all_files:
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        audio = torch.from_numpy(data[:, 0]).unsqueeze(0)
        length = torch.tensor([audio.shape[1]])
        with torch.no_grad():
            # raw log-mel features BEFORE normalization -- featurizer.forward
            # applies STFT+mel+log but normalize_batch is called separately
            # inside AudioToMelSpectrogramPreprocessor.forward; call the
            # feature extractor directly to get pre-normalization features.
            feat, feat_len = featurizer(audio, length)
        valid = feat[0, :, :feat_len[0]]  # [n_mels, T]
        if sum_ is None:
            sum_ = valid.sum(dim=1)
            sumsq = (valid ** 2).sum(dim=1)
        else:
            sum_ += valid.sum(dim=1)
            sumsq += (valid ** 2).sum(dim=1)
        count += valid.shape[1]

    mean = sum_ / count
    var = sumsq / count - mean ** 2
    std = var.clamp(min=1e-8).sqrt()

    out = {
        "fixed_mean": mean.tolist(),
        "fixed_std": std.tolist(),
        "n_files": len(all_files),
        "n_frames": count,
        "source": "evaluation/manifests/day1_public_speakers.json (LibriSpeech test-clean subset)",
    }
    out_path = Path("training/models/speakernet_fixed_cmvn.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}: {len(mean)} channels, {count} total frames from {len(all_files)} files")


if __name__ == "__main__":
    main()
