# Restore guide — after cloning from GitHub

Everything needed to rebuild the live stack from git alone (`.env` is never committed).

## 1. Prerequisites

- Python 3.11+
- Node 22+
- ~2 GB disk for models + venv (excluding MMCSG corpora)

## 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchaudio  # match your CUDA if training
pip install websockets numpy pytest soundfile sherpa-onnx
# Install remaining deps from project requirements / imports as needed
```

## 3. Download models (required for receiver)

```bash
python scripts/models/download_speakernet.py
```

Expected paths:

- `models/sherpa-speaker/nemo_en_speakerverification_speakernet.onnx`
- `models/sherpa-asr/` (optional — live transcript)

GeoWearNet G2 is **already in git**:

- `training/geowearnet/mmcsg/frozen/g2_selected_07c43c3d9e37.pt`

## 4. Environment

```bash
cp .env.example .env
# Minimum for local receiver:
#   MENTRA_SERVER_BIND_HOST=127.0.0.1
# For web frontend (Vercel or local):
#   NUXT_PUBLIC_BACKEND_WS_URL=ws://127.0.0.1:8765
#   (or wss://your-tunnel.trycloudflare.com)
```

## 5. Run backend

```bash
scripts/mentra/start_stack.sh
# Or manually:
#   .venv/bin/python scripts/mentra/run_receiver.py --listen 127.0.0.1 --port 8765
```

## 6. Run / deploy frontend

```bash
cd web && npm install
NUXT_PUBLIC_BACKEND_WS_URL=ws://127.0.0.1:8765 npm run dev
```

Production: push to `master` → Vercel auto-deploy. Set `NUXT_PUBLIC_BACKEND_WS_URL` in Vercel project env.

## 7. Verify

```bash
pytest tests/models/test_model_registry.py -q
cd web && npm test
```

Open https://mentra-dl-project.vercel.app — confirm `[ SpeakerNet | GeoWearNet G2 ]` tabs and BACKEND connects when tunnel is up.

## Local-only data (not in git)

| Path | Size (typical) | Purpose |
|------|----------------|---------|
| `evaluation/` caches | tens of GB | MMCSG eval artifacts |
| `training/geowearnet/mmcsg/` (except `frozen/`) | varies | training runs |
| `logs/` | small | receiver/tunnel logs |
| `recordings/` | varies | local Mentra captures |
| `.venv/`, `web/node_modules/` | ~500 MB | dev deps |

Re-download or re-generate as needed for research; not required for live demo.
