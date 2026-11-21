# AnomalyScope AI — Methodology & Experiment Notes

This document describes how the six MVTec AD experiments were produced and how
to interpret the results. It is descriptive only and makes **no** state-of-the-art
or published-benchmark claims.

## 1. Dataset (MVTec AD)

| Category | Normal train | Test (good + defective) | Defect types |
|----------|-------------:|------------------------:|--------------|
| bottle   | 209 | 83  (20 good + 63 defective) | broken_large, broken_small, contamination |
| hazelnut | 391 | 110 (40 good + 70 defective) | crack, cut, hole, print |

Only defect-free (`good`) images are used for training.

## 2. Split & calibration (leakage prevention)

- The normal training set is split **80 / 20** into a **fit** subset and a
  **held-out calibration** subset. The split is deterministic (seeded) and the
  two subsets are **disjoint**.
- The model is fitted only on the fit subset. The decision threshold is
  calibrated on the held-out normal calibration scores at the **99.5th
  percentile**.
- The **test set is never used** for fitting or calibration. Threshold-based
  Precision/Recall/F1 on the test set are diagnostic only.

Observed counts (from the saved JSON files):

| Experiment | Fit | Calibration | Test |
|------------|----:|------------:|-----:|
| PaDiM · bottle | 167 | 42 | 83 |
| PatchCore · bottle | 167 | 42 | 83 |
| Autoencoder · bottle | 167 | 42 | 83 |
| PaDiM · hazelnut | 313 | 78 | 110 |
| PatchCore · hazelnut | 313 | 78 | 110 |
| Autoencoder · hazelnut | 313 | 78 | 110 |

## 3. Metrics

- **Image AUROC** — ROC-AUC of per-image anomaly scores vs. image labels.
- **Pixel AUROC** — ROC-AUC over all pixels (anomaly map vs. ground-truth mask).
- **PRO / AUPRO** — Per-Region Overlap integrated over false-positive rate
  (up to 0.3) and normalised to [0, 1]. Higher is better.
- **Precision / Recall / F1 / Accuracy** — computed on the test set at the
  calibrated threshold (diagnostic).
- **Threshold** — calibrated cut-off applied to anomaly scores.

`per_defect_results` are intentionally **excluded** from the dashboard: the
current per-defect AUROC implementation evaluates single-class subsets and is not
valid.

## 4. Model implementation notes

### PaDiM
- ResNet-18 features, random projection to 100 dims, per-position Gaussians.
- **Memory optimisation:** channels are projected **per batch before
  concatenation**, so the full 448-channel `(N, 448, 64, 64)` stack is never
  materialised (reduces peak memory ~4.5×). This is what allowed the larger
  hazelnut fit set (313 images) to run without OOM.

### PatchCore
- WideResNet-50-2 features, greedy minimax coreset (ratio 0.01).
- **Two-pass coreset construction:** Pass 1 stores only projected `(N, 128)`
  features for selection; Pass 2 re-extracts features deterministically and
  gathers only the selected full-dimensional vectors. The full `(N, D)` bank is
  never held in memory. Selection is deterministic (seeded) and matches the
  original single-pass algorithm exactly (verified by regression tests).

### Autoencoder
- Convolutional reconstruction model, Adam (lr 1e-4, wd 1e-5), StepLR
  (step 50, gamma 0.5), MSE loss, 200 epochs.
- **Periodic recovery checkpoints every 20 epochs** (weights + optimizer +
  scheduler + epoch + config) written to `<name>.recovery.pth`, separate from
  the final checkpoint. `--resume` restores from a valid recovery checkpoint at
  the next epoch and refuses incompatible checkpoints.

## 5. Methodological note — PatchCore implementation update

> PatchCore was implemented in two stages. The original Bottle experiment used
> shuffled data loading, while the later memory-efficient two-pass implementation
> introduced deterministic ordering to ensure consistent feature extraction
> across both passes. The core algorithm, backbone, image size, projection
> dimension, and coreset ratio were preserved. This implementation detail is
> documented for reproducibility when interpreting results.

- **PatchCore · Bottle** used the original (shuffled) data loading.
- **PatchCore · Hazelnut** used the later deterministic two-pass implementation.

## 6. Limitations

- Results are on a small benchmark subset and are **not** claimed to be
  state-of-the-art or to match published numbers.
- The percentile threshold is a heuristic and may not be optimal for every
  metric; calibration/test distribution shift can affect it.
- The Autoencoder is a weak reconstruction baseline (especially on bottle);
  PaDiM/PatchCore are density / nearest-neighbour methods that assume
  representative normal training data.
- All experiments used CPU (seed 42, image size 256).