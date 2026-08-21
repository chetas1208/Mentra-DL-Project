# models/

Model weights go here where license permits redistribution; otherwise a
`download.sh`/`download.py` script + checksum file, no binary in git.

For every model added, record in this file: name, source repo, commit/version,
license, model-license (may differ from repo license), commercial-use
restriction, redistribution restriction.

| Model | Source | Version | License | Commercial use | Redistribute weights |
|---|---|---|---|---|---|
| wespeaker_en_voxceleb_CAM++.onnx | k2-fsa/sherpa-onnx release `speaker-recongition-models`, orig. WeSpeaker | fetched 2026-08-21 | WeSpeaker repo is Apache-2.0; **model weights license NOT independently verified** | NOT VERIFIED | NOT VERIFIED |
| wespeaker_en_voxceleb_CAM++_LM.onnx | same | fetched 2026-08-21 | same as above | NOT VERIFIED | NOT VERIFIED |
| wespeaker_en_voxceleb_resnet34_LM.onnx | same | fetched 2026-08-21 | same as above | NOT VERIFIED | NOT VERIFIED |
| nemo_en_titanet_small.onnx | same release, orig. NVIDIA NeMo | fetched 2026-08-21 | NeMo pretrained checkpoints are typically distributed under NVIDIA's own terms (often CC-BY-4.0 on NGC/HF, but **not confirmed for this specific export**) | NOT VERIFIED | NOT VERIFIED |
| nemo_en_speakerverification_speakernet.onnx | same release, orig. NVIDIA NeMo | fetched 2026-08-21 | same caveat as TitaNet | NOT VERIFIED | NOT VERIFIED |

**Do not treat any row above as commercially cleared.** All five are real,
working, currently-downloaded files (`sha256sum` them before shipping
anything) — but "NOT VERIFIED" means exactly that: nobody has read the
actual license terms attached to these specific re-exported weights yet.
This blocks commercial deployment, not research/benchmarking.
