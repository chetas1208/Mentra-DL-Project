# MentraWearNet — Live Wearer-vs-Environment Detection

**Production web app:** https://mentra-dl-project.vercel.app  
**GitHub:** https://github.com/chetas1208/Mentra-DL-Project  
**Sprint:** 2026-08-21 → 2026-09-04

Detect whether speech in a 16 kHz mono PCM stream comes from the **wearer** or the **environment**. Four states: `SILENCE` · `WEARER` · `ENVIRONMENT` · `OVERLAP`.

---

## Live stack (as shipped)

```
Browser mic (getUserMedia)
  → AudioWorklet → resample 16 kHz → PCM16 frames
  → WebSocket (MTRA binary protocol)
  → Cloudflare tunnel → Python receiver (:8765)
  → per-session model (SpeakerNet OR GeoWearNet G2)
  → inference JSON → live dashboard
```

| Layer | Location | Notes |
|-------|----------|-------|
| **Frontend** | `web/` → Vercel | Nuxt 4 + Tailwind. **Git push `master` auto-deploys.** |
| **Backend** | `scripts/mentra/run_receiver.py` | Multi-model registry, port 8765 |
| **Tunnel** | `scripts/mentra/start_stack.sh` | Cloudflare quick tunnel + keepalive |
| **Models** | `server/models/` | SpeakerNet + frozen GeoWearNet G2 |

### Runtime models

| Model | ID | Enrollment | Status |
|-------|-----|------------|--------|
| **SpeakerNet** | `speakernet` | Required | Production baseline — WHO is speaking? |
| **GeoWearNet G2** | `geowearnet_g2` | None | WEARABLE-VALIDATED EXPERIMENTAL — wearer position detection |

Select via segmented tabs on the live console: `[ SpeakerNet | GeoWearNet G2 ]`.

GeoWearNet G2 checkpoint (in git):

```
training/geowearnet/mmcsg/frozen/g2_selected_07c43c3d9e37.pt
SHA-256: 07c43c3d9e37dbd490ae0477f46ff515e1e19e8fe0a20687ec9d13a8555c68ef
```

---

## Quick start (full stack)

### 1. Clone & Python env

```bash
git clone https://github.com/chetas1208/Mentra-DL-Project.git
cd Mentra-DL-Project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # if present; else install from pyproject/setup docs
```

### 2. Download runtime models (not in git — ~280 MB)

```bash
python scripts/models/download_speakernet.py
# SpeakerNet ONNX → models/sherpa-speaker/
# ASR models → models/sherpa-asr/ (optional, for transcript panel)
```

### 3. Config (never commit `.env`)

```bash
cp .env.example .env
# Edit bind host, ports, tunnel URL as needed
```

### 4. Start backend + tunnel

```bash
scripts/mentra/start_stack.sh
# Receiver on 127.0.0.1:8765, cloudflared tunnel, 60s keepalive
# Canonical WSS URL → logs/tunnel_url.txt
```

### 5. Frontend (local dev)

```bash
cd web
npm install
NUXT_PUBLIC_BACKEND_WS_URL=ws://127.0.0.1:8765 npm run dev
# Or use the tunnel WSS URL for remote testing
```

### 6. Production deploy

```bash
git push origin master
# Vercel auto-builds from web/ (rootDirectory=web, Nuxt preset)
# Set NUXT_PUBLIC_BACKEND_WS_URL in Vercel env to your tunnel WSS URL
```

**Important:** Vercel blocks deploys if git commit author email isn't on the team. Use an authorized email (e.g. `chetasparekh2003@gmail.com`) or add the author to Vercel.

---

## Architecture

### One audio pipeline, two selectable models

```
AUDIO INPUT (any mic — browser, Mentra, replay)
       │
       ▼
  AudioFrame (binary, model-agnostic)
       │
       ▼
  LiveSession → SessionRuntime → ModelRegistry
                    /              \
            speakernet          geowearnet_g2
                    \              /
                     ▼
              DETECTION JSON → UI
```

- Model selection = **per session** (`SESSION_CONFIG`), not process-global.
- Audio input selection = **INPUT dropdown** (separate from model tabs).
- No hot-swap mid-stream — stop session, switch model, start again.

### WebSocket control plane

| Message | Direction | Purpose |
|---------|-----------|---------|
| `STREAM_START` | client→server | Handshake + `features=model_selection` |
| `MODEL_CATALOG` | server→client | Available models + capabilities |
| `SESSION_CONFIG` | client→server | Bind model for this session |
| `SESSION_CONFIG_ACK` | server→client | Confirm + capabilities |
| `AUDIO_FRAME` | client→server | Binary PCM16 @ 16 kHz |
| `DETECTION` | server→client | Inference JSON |

See `mentra/audio/frame.py` and `docs/MENTRA_REMOTE_AUDIO.md`.

---

## Project layout

```
Mentra-DL-Project/
├── web/                    # Nuxt live console (Vercel)
├── server/                 # Receiver, model registry, GeoWearNet adapter
├── mentra/                 # Binary protocol, consumer, jitter buffer
├── scripts/mentra/         # run_receiver.py, start_stack.sh, keepalive
├── training/geowearnet/    # G2 training code + frozen checkpoint (in git)
├── evaluation/             # Benchmarks, manifests, results (mostly local)
├── models/                 # ONNX weights — DOWNLOAD, not in git (~280 MB)
├── research/               # sherpa-onnx detector, spikes
├── docs/                   # Architecture, reports, protocols
├── tests/                  # Python test suite
└── android/                # Kotlin app scaffold (hardware path)
```

---

## Testing

```bash
# Python (from repo root, .venv active)
pytest tests/ -q

# Web
cd web && npm test && npm run typecheck && npm run build

# E2E WebSocket (receiver must be running)
.venv/bin/python -c "..."  # see session smoke in CI docs
```

85 web tests + 334+ Python tests at last green run.

---

## What's in git vs local-only

| In git | Local-only (gitignored) |
|--------|-------------------------|
| All source code | `.env` (secrets) |
| GeoWearNet G2 frozen checkpoint (1.6 MB) | `models/` ONNX weights (~280 MB) — run download script |
| Evaluation JSON/CSV results | `evaluation/` bulk caches (~41 GB) |
| Web app + tests | `.venv/`, `logs/`, `web/node_modules/` |
| Docs + manifests | `training/runs/`, MMCSG raw corpora |

Restore after clone: download models, copy `.env.example` → `.env`, start stack.

---

## Audio input processing (microphones)

1. `enumerateDevices()` → INPUT dropdown lists every `audioinput`
2. User selects device (or default) — **while session stopped**
3. `getUserMedia({ deviceId: { exact } })` opens that mic
4. AudioWorklet → StreamingResampler → 16 kHz → 160-sample PCM16 frames
5. WebSocket → receiver — **model never knows which physical mic was used**

Debug page: `/audio-debug` — lists all inputs, shows per-device `track.getSettings()`.

---

## Key docs

| Doc | Content |
|-----|---------|
| `docs/MENTRA_REMOTE_AUDIO.md` | Binary protocol, tunnel, transport |
| `docs/MENTRAWEARNET_ARCHITECTURE.md` | Product architecture |
| `docs/MODEL_SELECTION.md` | SpeakerNet benchmark numbers |
| `docs/geowearnet_g2_mmcsg_report.md` | GeoWearNet G2 MMCSG validation |
| `MEMORY.md` | Session state / decisions (internal) |

---

## Status labels

Used everywhere: `VERIFIED` · `IMPLEMENTED` · `TESTED` · `PARTIAL` · `BLOCKED` · `RESEARCH ONLY` · `NOT STARTED`

GeoWearNet G2: **WEARABLE-VALIDATED EXPERIMENTAL** — not Mentra-hardware-validated.  
Source separation: **NOT SUPPORTED** (routing/gating only).

---

## License & models

SpeakerNet / sherpa-onnx weights: see `models/README.md` for license caveats.  
Do not treat model weights as commercially cleared without independent license review.
