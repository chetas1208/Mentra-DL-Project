# SpeakerNet Future-Context Report

Status: `MEASURED` — computed programmatically from the real checkpoint's
module graph (`scripts/model/analyze_future_context.py`), not estimated.

## Preprocessor config (from the real checkpoint)

```
sample_rate: 16000
window_size: 0.02   (20ms)
window_stride: 0.01 (10ms hop)
window: hann
n_fft: 512
normalize: per_feature
stft_conv: false     (i.e. exact_pad=False -> STFT uses centered windows)
```

## STFT centering

`FilterbankFeatures.stft()` calls `torch.stft(..., center=False if
exact_pad else True, ...)`. `exact_pad` isn't set in this config, so it
defaults `False`, meaning **`center=True`** — every mel frame at time `t`
is computed from a window centered on `t`, so it needs samples up to
`t + window_size/2`.

**STFT lookahead: 10.0ms** (half of the 20ms window).

## Convolutional right-context (measured from the actual module tree)

All 5 Jasper blocks in `ConvASREncoder` use **stride 1** throughout — the
encoder never downsamples time, confirmed both by direct inspection (every
`Conv1d` printed `stride=(1,)`) and by the earlier empirical stride
measurement matching the mel-frame count almost exactly at every duration
tested. Right-context therefore accumulates **additively** across
sequential kernel>1 layers (no downsampling to shrink it):

| Block | Conv layers (kernel>1 only) | Right-context |
|---:|---|---:|
| 0 | k=3, depthwise | 1 frame |
| 1 | k=7 ×2 (depthwise, twice) | 6 frames |
| 2 | k=11 ×2 | 10 frames |
| 3 | k=15 ×2 | 14 frames |
| 4 | (only k=1 convs) | 0 frames |
| **Total** | | **31 frames** |

At the measured 10ms hop: **31 frames × 10ms = 310ms of bounded conv
right-context.**

## Total bounded future context

**320.0ms** (310ms conv + 10ms STFT centering).

## The dominant issue is NOT the 320ms — it's normalization

Read directly from `nemo.collections.asr.parts.preprocessing.features.normalize_batch`:
when `normalize_type == "per_feature"` (the config value here), the mean
and std used to normalize *every* frame are computed by summing over the
**entire valid-length time axis** (`x.sum(axis=2)` gated by `seq_len`, not
a bounded window). This means the normalized value at frame 0 depends on
every frame up to the end of whatever's currently buffered — i.e.
**unbounded** future dependency, not a fixed 320ms. For a 2-second rolling
window this is bounded by the window length (2000ms), which is much larger
than the 320ms conv/STFT budget and is almost certainly the dominant
contributor to the causality-test failure measured in
`docs/MENTRAWEARNET_ARCHITECTURE.md` (diff growing toward the boundary is
consistent with both explanations, so this alone doesn't distinguish them —
see `docs/STREAMING_CAUSALITY_AUDIT.md` for the isolated experiment).

## What this means for the streaming variants

- **C0 (strict causal, 0ms lookahead)** requires fixing BOTH: (1) replace
  `per_feature` normalization with something bounded/causal — a fixed
  global CMVN computed once from training data is the simplest fix NeMo
  already supports natively (`normalize_type` accepts a `fixed_mean`/
  `fixed_std` dict); (2) change every conv's padding from symmetric
  (`kernel//2` both sides) to causal (`kernel-1` on the left, `0` on the
  right) — reuses the exact same pretrained kernel weights, no new
  parameters, per the user's own instruction not to duplicate weights.
- **LowRC (bounded lookahead, e.g. ~100-250ms)** could keep some symmetric
  padding in early blocks (cheaper: less receptive-field cost per ms of
  lookahead spent) while still needing the normalization fix, since
  normalization's unboundedness swamps any conv-based lookahead budget
  otherwise.
