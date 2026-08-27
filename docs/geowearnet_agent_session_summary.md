# GeoWearNet Campaign G1 — Agent Session Summary

_Session date: 2026-08-25 → 2026-08-26 (UTC). Continuation of the autonomous
GeoWearNet convergence campaign after the initial 19-run training sweep
finished unattended overnight._

This document records what THIS agent session did, on top of the prior
session's work (which built the training infra, simulator, ablation matrix,
and launched the 19-run campaign queue). Full detail lives in
`docs/geowearnet_campaign_results.md` (results) and the transcript; this is
the task log.

## Starting state

- All 19 queued training runs (3 base models + 16 ablation/context/size/arch
  experiments) had finished on their own, ~9.5 hours before this session
  picked back up. Both GPUs were sitting idle.
- Every run had a `best.pt` checkpoint, but only mid-training smoke evals
  existed — **no run had been scored on its final checkpoint yet.**

## Tasks performed

1. **Audited campaign completion.** Read `campaign_status.json`,
   `campaign_followup.log`, and process/GPU state to confirm all 19 runs
   completed successfully (`returncode: 0`) and nothing was still running.

2. **Ran the full final-evaluation matrix (22 evaluations).** Built and
   dispatched, in parallel batches of 6 across the freed CPU cores:
   - 19 native-generation evaluations (each S1-trained ablation/context/size/
     arch run against the `S1_test` suite; `s2_base` against `S2_test`;
     `mixed_base` against both, since there is no dedicated "mixed" suite).
   - 3 additional cross-simulator evaluations (`s1_base`→S2, `s2_base`→S1) to
     fill out the Workstream T generalisation matrix.

3. **Found and fixed a real bug in `evaluate.py`.** The `cmvn` (causal
   running mean/variance) normalization path silently promoted features
   from float32 to float64 — a numpy type-promotion side effect of dividing
   a float32 array by an `int64` frame-count array — which then crashed
   inference with `RuntimeError: Input type (double) and bias type (float)
   should be the same`. Training was unaffected (its `data.py` version
   explicitly casts back to float32), but evaluation was not. Fixed by
   casting the normalized features to float32 before building tensors, and
   re-ran the previously-failed `norm_cmvn` evaluation to completion.

4. **Regenerated the campaign registry and report** (`registry_update.py`,
   `report.py`) against the newly-completed final evaluations. This produced
   the first true end-to-end ablation table, cross-simulator generalisation
   matrix, and model-selection ranking of the entire campaign.

5. **Read the results and surfaced the headline findings**:
   - Cross-simulator generalisation holds (S1↔S2, mixed→{S1,S2} all ~0.90–
     0.92 robust mean, close to native performance) — evidence against a
     simulator-specific shortcut.
   - Amplitude features are not load-bearing (`ablate_noamp`, `norm_cmvn`
     both cost little), consistent with the earlier
     `AMPLITUDE_INFORMATIVE_BUT_NOT_SUFFICIENT` verdict.
   - Context sweep: sharp regression below ~500ms, flattening by 680–1000ms.
   - **The CRNN architecture (28,950 params — the smallest model in the
     entire sweep) came out as the most robust model overall**, ahead of
     every larger TCN/ConvNeXt/DSConv variant, by the Workstream AS
     selection objective.

6. **Closed a shippability gap this discovery exposed.** The existing
   streaming-inference code (`deploy.py`) only supported the TCN
   architecture (`StreamingTCN`), which would have blocked shipping the
   actual best-scoring model. Implemented `StreamingCRNN`:
   - Reuses the TCN's conv-frontend ring-buffer trick for the finite-kernel
     conv layers.
   - Carries the GRU's hidden state frame-to-frame (no buffer needed there —
     GRUs are already stateful/causal by construction).
   - Verified exact numerical parity against the batched forward pass.
   - Refactored `verify_streaming_parity`, `cpu_benchmark`, and `soak` to
     dispatch generically via a new `make_streamer()` helper instead of
     hardcoding `StreamingTCN`, so future architectures plug in the same way.
   - Fixed a follow-on crash where `soak()` and the `main()` CLI assumed
     every model had TCN-shaped internal buffers (broke on ConvNeXt, which
     still has no streaming support and now fails/skips gracefully instead
     of crashing).
   - Added a new regression test, `test_streaming_crnn`, to
     `tests/geowearnet/test_geowearnet.py`.

7. **Ran CPU deployment benchmarks + 1-minute soak tests** for the four
   strongest candidates (CRNN, TCN base, size500k, ConvNeXt): parameter
   count, serialized size, chunked latency/RTF at 1/2/4 threads, streaming
   parity, TorchScript + ONNX export/parity, and soak-test health (NaN/Inf
   count, RSS growth, state-size stability, latency drift). All models are
   far under real-time on CPU at this scale (RTF ≈ 0.002–0.004); the winning
   CRNN's streaming state is bounded (512 elements) and showed 0 RSS growth
   and 0 NaNs over the soak minute.

8. **Extended `report.py`** to merge multiple per-checkpoint deploy JSON
   files (previously it only read one fixed `geowearnet_deploy_benchmark.json`),
   to render a streaming-parity column and a soak-test table, and to add a
   new **"Final verdict" section** to `docs/geowearnet_campaign_results.md`
   that states the selected model, the cross-sim evidence, the
   amplitude/context/architecture conclusions, and an explicit, honest
   overall status: `SIMULATION_CONVERGED, REAL_DOMAIN_TRANSFER_UNVERIFIED`.

9. **Ran the real-Mentra capture pipeline smoke test**
   (`capture.py --smoke-test`, synthetic ffmpeg source) to confirm it still
   works end-to-end: 16/16 steps valid, 0 flagged. This was a previously
   pending item.

10. **Verified everything with the full test suite** (`pytest
    tests/geowearnet/test_geowearnet.py`) — 17/17 passing, including the new
    CRNN streaming test — and re-ran the registry update as a final
    consistency check.

## Session 2 (2026-08-26 UTC) — MMCSG real-data ingestion + zero-shot transfer

The project owner independently registered with Meta, accepted the MMCSG Data
License Agreement (their own legal act; this session never did and could not
do this on their behalf), and provided signed CDN download links.

1. **Downloaded the full MMCSG corpus** (~47GB compressed: `MMCSG.zip` +
   3-part split archive + `MMCSG_eval.zip`) to `/usr/data/923873155/mmcsg`
   (outside the repo). Reassembled the split archive with `zip -F` (Meta's
   documented method) and MD5-verified against the provided checksums before
   extracting. `zip`/`unzip` were not installed system-wide and could not be
   installed with `sudo`; worked around this by `apt-get download`-ing the
   `.deb` packages and extracting them into a user-local `~/local_tools/`
   with `dpkg-deb -x`.
2. **Extracted and merged** train (172 recordings) + dev (169) + eval (189)
   into one unified `MMCSG/{audio,transcriptions,metadata,rttm,...}/{train,dev,eval}`
   tree matching exactly what `mmcsg_adapter.py` expects. Verified file counts
   against Meta's own README numbers (all exact matches). Deleted the ~65GB of
   intermediate zip/split archives once extraction was verified, reclaiming
   the space.
3. **Ran `mmcsg_adapter.py` against the real data for the first time.**
   Confirmed the documented format assumptions exactly (7-channel 48kHz PCM16
   audio, tab-separated word-level transcripts with 0/1 self/other labels,
   JSON metadata with real speaker IDs). Measured (not assumed) the
   best single raw microphone channel for the Mentra mono use case via
   `channel_study()`: channel 2, AUROC 0.879 vs. 0.734–0.873 for the others.
4. **Wrote a new script, `training/geowearnet/mmcsg_transfer.py`**, to close
   Workstream X/AZ (zero-shot sim-to-real transfer), which did not previously
   have a runnable implementation. It runs a checkpoint trained ENTIRELY on
   the simulator directly on real MMCSG audio with zero fine-tuning, reusing
   the exact training-time feature pipeline and normalization stats (never
   recomputed on real data, to avoid domain leakage), and scores with the
   same `evalsuite.core_metrics`/`bootstrap_ci` used throughout the campaign
   (recording-level cluster bootstrap, not frame-level).
5. **Scored 5 checkpoints on the full 189-recording MMCSG `eval` split**
   (mixed_base, s1_base, s2_base, s1_size500k, and the campaign's selected
   `s1_arch_crnn`). **Result: every checkpoint transfers zero-shot**, landing
   in a tight 0.87–0.89 (all-frames) / 0.95–0.97 (solo) AUROC band — the same
   range the simulated suites report — with **no fine-tuning or domain
   adaptation whatsoever**. The CRNN, selected purely on simulated
   robustness, is also the best real-data transferrer (solo AUROC 0.969,
   EER 0.083). This is the first non-simulated evidence produced anywhere in
   this campaign, and it is positive.
6. **Updated `docs/geowearnet_data_license_state.md`** with an honest
   addendum: obtaining data required accepting MMCSG's CC-BY-NC (non-
   commercial) license, which does NOT change the underlying license
   finding. MMCSG is used strictly as a research/evaluation signal here
   (exactly as the B.5 verdict recommended before data existed), never as a
   training dependency for a shippable checkpoint, and is not redistributed.
7. **Extended `report.py`** with a new "MMCSG zero-shot sim-to-real transfer"
   section (corpus stats, channel study, per-checkpoint AUROC/EER table with
   CIs) and updated the Final Verdict's honest status string from
   `SIMULATION_CONVERGED, REAL_DOMAIN_TRANSFER_UNVERIFIED` to
   `SIMULATION_CONVERGED, REAL_PROXY_TRANSFER_POSITIVE,
   MENTRA_HARDWARE_PILOT_PENDING` — explicitly still distinguishing "a real
   wearable proxy transferred" from "Mentra hardware was validated".
8. **Registry hygiene**: added `mmcsg_adapter.py`/`mmcsg_transfer.py` to the
   fingerprinted campaign artifacts list, and fixed `registry_update.py` to
   prune `result::*` fingerprints for files that no longer exist on disk
   (found this gap when cleaning up a superseded 5-recording smoke-test
   result).
9. **Full test suite still green**: 148/148 passing after all changes.

## What is still blocked

- **Real Mentra hardware pilot**: capture pipeline is built and smoke-tested
  (see item 9 in Session 1 above), but no actual glasses/subject recordings
  exist, so Mentra-specific transfer evidence (as opposed to the MMCSG-proxy
  evidence above) still does not exist. This remains blocked purely on
  hardware access, an external constraint this session cannot resolve.
- **Commercial licensing of MMCSG**: still CC-BY-NC on every fetchable
  primary source. Fine as an evaluation signal (used here only for that);
  training a shippable model on MMCSG would require a separate commercial
  license conversation with Meta that is out of scope for this session.

## Files touched this session

- `training/geowearnet/evaluate.py` — float32 cast fix for `cmvn` normalization.
- `training/geowearnet/deploy.py` — `StreamingCRNN`, `make_streamer()`
  dispatch, generalised `verify_streaming_parity`/`cpu_benchmark`/`soak`,
  guarded soak/streaming for architectures without streaming support.
- `training/geowearnet/report.py` — multi-file deploy loading, streaming
  parity + soak tables, new "Final verdict" section.
- `tests/geowearnet/test_geowearnet.py` — added `test_streaming_crnn`.
- `docs/geowearnet_campaign_results.md` — regenerated with final evaluation
  numbers, cross-sim matrix, deployment tables, and final verdict.
- `evaluation/geowearnet/experiment_registry.json`,
  `evaluation/geowearnet/results/geowearnet_campaign_report.json` —
  regenerated.
- `evaluation/geowearnet/results/eval_*_final.json` (22 new files) — final
  evaluation outputs for every checkpoint × generation pair.
- `evaluation/geowearnet/results/geowearnet_deploy_*.json` (4 files,
  regenerated) — CPU benchmark + streaming + export + soak results for the
  four strongest candidate models.
- `evaluation/geowearnet/results/geowearnet_capture_status.json` — capture
  pipeline smoke-test output.
- `recordings/geowearnet_pilot/20260826_023324_W99/` — synthetic capture
  smoke-test recording (not real data).

**Session 2 additions:**
- `training/geowearnet/mmcsg_transfer.py` — new: zero-shot sim-to-real
  transfer evaluation (Workstream X/AZ).
- `docs/geowearnet_data_license_state.md` — new addendum on real data access.
- `training/geowearnet/registry_update.py` — added MMCSG artifacts to the
  fingerprint list; added stale-`result::*`-artifact pruning.
- `training/geowearnet/report.py` — new MMCSG transfer section + updated
  final-verdict status string.
- `docs/geowearnet_campaign_results.md`,
  `evaluation/geowearnet/experiment_registry.json`,
  `evaluation/geowearnet/results/geowearnet_campaign_report.json` —
  regenerated.
- `evaluation/geowearnet/results/geowearnet_mmcsg_status.json`,
  `evaluation/geowearnet/results/geowearnet_mmcsg_transfer_*_oneval.json`
  (5 files), `evaluation/geowearnet/manifests/mmcsg_manifest.json` — new
  real-data evaluation artifacts.
- `/usr/data/923873155/mmcsg/MMCSG/` — real MMCSG corpus (outside repo, not
  tracked by git, ~70GB: 172 train + 169 dev + 189 eval recordings).
