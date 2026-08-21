# Benchmarks

Status: NOT STARTED. No model has been run against real or proxy audio yet.

Populate via `evaluation/scripts/` once a manifest exists in
`evaluation/manifests/`. Required metrics per model — see
`evaluation/metrics/` and sprint spec section 19:

- Classification: precision/recall/F1 (wearer, environment), balanced accuracy, ROC-AUC
- Verification: FAR, FRR, EER
- Systems: model size, peak/avg RAM, CPU util, RTF, median/p95 latency, dropped frames

No row goes in the results table without a value in `evaluation/results/`.
