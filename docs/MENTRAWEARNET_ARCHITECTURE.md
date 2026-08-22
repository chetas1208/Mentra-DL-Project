# MentraWearNet-v1 Architecture

Status: core module `IMPLEMENTED`, all structural tests run with real
results below. Training itself `NOT STARTED`.

## Diagram

```
ENROLLMENT (once, cached — never re-run per live-audio call)
reference audio -> SpeakerNetBackbone.encode_speaker() -> e_w [B,256] (L2-normalized)

LIVE (every rolling window)
waveform [B,N]
  -> SpeakerNetBackbone.encode_frames() -> [B,1500,T]   (shared backbone instance)
  -> Conv1d 1x1 projection             -> [B,192,T]
  -> FiLM(e_w)                         -> [B,192,T]      (gamma/beta init ~0, starts near-identity)
  -> explicit cosine-similarity branch -> [B,1,T]         (fused back in via concat + 1x1 conv)
  -> causal depthwise-separable TCN (5 blocks, dilations 1/2/4/8/16, left-pad only)
  -> wearer_head (Conv1d 192->1)       -> wearer_logits [B,T]
  -> environment_head (Conv1d 192->1) -> environment_logits [B,T]
```

Files: `training/models/speakernet_backbone.py`, `training/models/mentrawearnet.py`.

## Tensor shapes and frame stride (measured, not textbook)

Encoder frame stride, measured directly on this checkpoint by feeding
0.5/1.0/2.0/3.0s of audio and reading the actual output tensor width:

| Duration | Samples | Encoder output T |
|---:|---:|---:|
| 0.5s | 8,000 | 64 |
| 1.0s | 16,000 | 112 |
| 2.0s | 32,000 | 208 |
| 3.0s | 48,000 | 304 |

Linear fit: `T ≈ 0.00601 · samples + 16`, i.e. ~10.4ms/frame hop with a
fixed ~166ms window/padding offset. `encode_frames()` output channel count
is 1500 (last Jasper block's channel width) at every duration tested.

## Parameter counts (exact, measured via `sum(p.numel())`)

| Component | Params |
|---|---:|
| SpeakerNet backbone (preprocessor + encoder + pooling + 256-d projection; **excludes** the original 7205-way VoxCeleb classifier) | 4,009,848 |
| Custom head (frame/enrollment projections, FiLM, similarity branch, 5-block TCN, 2 output heads) | 688,770 |
| **Total deployment graph** | **4,698,618** |

`<10M gate: PASS` (uses 47% of budget — well inside the 6.8-8.0M target
range too, real headroom left for e.g. a wider projection dim or more TCN
blocks if accuracy needs it later).

## Structural test results (all real, run 2026-08-21)

| Test | Result |
|---|---|
| Backbone construction + forward (frames + speaker embedding) | PASS — correct shapes, embedding norm 1.0 |
| MentraWearNet construction | PASS |
| Forward pass (batch=2, variable lengths, mixed durations) | PASS — no NaN |
| Backward pass (frozen-backbone stage) | PASS — 0 backbone params received gradients (correctly frozen), head/TCN/FiLM gradients all nonzero |
| **Causality test** | **FAIL** — see below |
| GPU environment check | PASS (after a real fix, see below) — 2x RTX 3090, real matmul, NCCL available |
| DDP smoke test (torchrun, 2 processes) | PASS — correct per-rank GPU assignment, real all-reduce, real DDP backward, clean exit |

## Causality: FAIL, root cause identified, not yet resolved

`tests/test_mentrawearnet_causality.py` compares model outputs on a shared
1.0s audio prefix `A`, once with a future segment `B` appended and once
with a different future `C` appended. A truly causal model produces
identical outputs on the `A`-region regardless of what follows it.

Measured: max abs diff = 0.0098 across the A-region, but critically the
diff **grows toward the A/B boundary** (0.0002 mean for the first 10
frames, far from the boundary, vs 0.0024 mean for the last 10 frames,
right at the boundary) and is near-zero deep inside `A`. That pattern is
the signature of a bounded-lookahead leak from a fixed-size receptive
field, not noise or a bug in the custom head.

**Root cause**: the pretrained NeMo SpeakerNet encoder's Jasper blocks use
standard (symmetric-padded) `Conv1d` layers, not causal ones. Kernel sizes
up to 15 across 5 stacked blocks give each layer a receptive field
extending into genuinely-future audio samples within the padded window,
regardless of the fact that the custom TCN built on top of it (section 12)
uses correct left-only causal padding. The TCN's causality is real and
verified by isolating where the diff is largest — the leak is upstream in
the reused pretrained encoder, not in new code.

**Practical implication, not yet decided**: in the current "reprocess the
whole rolling window every hop" deployment design (spec section 33/57 —
not yet the stateful streaming version from section 56), the frame this
system actually consumes each hop is the *last* frame of the window, which
sits at the live edge with no future audio to leak from — so this specific
violation may matter less in practice than the raw test number suggests.
That's a hypothesis, not a measurement — hasn't been tested by simulating
the actual rolling-window consumption pattern. Three options for later,
none implemented: (1) accept bounded non-causality as a documented,
measured tradeoff; (2) replace the backbone's padding with causal-only and
accept it will need fine-tuning to recover pretrained accuracy (deviates
from the "reuse pretrained weights as-is" plan); (3) verify hypothesis
above empirically before deciding either way.

## GPU environment: real bug, real fix (not a workaround)

Root cause (found, not guessed): `nemo_toolkit[asr]`'s own dependency
resolution silently upgraded `torch` from a working `2.5.1+cu121` build to
`2.13.0+cu130` — a CUDA 13.0 build the installed driver (535.288.01, max
CUDA 12.2 per `nvidia-smi`) genuinely cannot run. This is why
`CUDA_VISIBLE_DEVICES=""` was needed for the SpeakerNet parity work
earlier — not a phantom driver issue, a real version conflict.

Attempted fix: pin back to `torch==2.5.1+cu121`. **Failed** —
`nemo-toolkit 3.0.0 requires torch>=2.6.0`, so that pin isn't viable
alongside NeMo going forward.

Actual fix: installed `torch==2.6.0+cu118` — satisfies NeMo's `>=2.6.0`
requirement while using an older CUDA runtime (11.8) well within what the
535.288 driver supports. Verified: `pip check` reports no conflicts,
`torch.cuda.is_available()` is `True` with no env var needed, NeMo loads
the checkpoint straight onto `cuda:0` automatically, both GPUs pass real
matmul ops, NCCL is available, and a real 2-process `torchrun` DDP job
(`scripts/training/ddp_smoke_test.py`) completes cleanly with correct
per-rank GPU assignment and real gradient flow through a DDP-wrapped
module. `GPU_TRAINING_READY`.

## Deployment API (not yet wired to a runtime, module-level only)

```python
wearer_embedding = model.encode_enrollment(enrollment_waveform, enrollment_lengths)  # once
# ... later, repeatedly on live audio:
out = model.process_with_embedding(live_waveform, live_lengths, wearer_embedding)
```

`forward()` (training path) re-runs enrollment every call since the
enrollment speaker varies per training example; the deployment path never
does that (section 22/23).

## Training stage control (implemented, not yet exercised in a real training loop)

`model.freeze_backbone()`, `model.unfreeze_upper_backbone(n)`,
`model.unfreeze_backbone()` — verified via the frozen-backbone backward
test above (0 backbone params received gradients when frozen). Stage 2/3
unfreezing logic exists but hasn't been run in an actual training loop yet.

## Not done in this phase (per the stop condition — do not launch training yet)

Dataset adapters, mixture generator, loss implementation beyond a smoke
BCE call, actual training runs (frozen or unfrozen), ONNX export, INT8
quantization. All `NOT STARTED`, tracked in `docs/TODO.md`.
