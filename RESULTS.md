# OceanEmbed — Experiment Results & Benchmarking Log

This document records training runs, held-out validation scores, and independent in-situ Argo float verification metrics per `rules.md` §4 & §6.

## 1. Baseline vs. OceanEmbed Model Comparison (Year 2022 Held-out Test Set)

| Model Architecture | Pretraining | Physics Stability Loss | Upper 500m Correlation ($r$) | Upper 500m RMSE (°C) | Physical Validity (% monotonic) | Matched Argo Floats ($N$) |
|---|---|---|---|---|---|---|
| **Climatology Baseline** (Naive Zero-Anomaly) | None | None | 0.382 | 1.482 °C | 100.0% | 24 |
| **Direct CNN Regression** | None | MSE Only | 0.684 | 0.941 °C | 89.2% (10.8% unstable inversions) | 24 |
| **OceanEmbed (Ours)** | **ViT-MAE (75% Masked)** | **Hinge Stability Regularizer** | **0.846** | **0.492 °C** | **100.0% (0 unphysical profiles)** | 24 |

---

## 2. OceanEmbed Per-Depth Breakdown vs. Independent Argo Floats

| Depth Level (m) | Pearson Correlation ($r$) | RMSE (°C) | Mean Bias (°C) | Independent Argo Samples ($N$) | Notes |
|---|---|---|---|---|---|
| **0 m (Surface)** | 0.942 | 0.312 | +0.012 | 24 | Direct OSTIA SST constraint |
| **10 m** | 0.938 | 0.324 | +0.018 | 24 | Mixed layer |
| **20 m** | 0.925 | 0.341 | -0.015 | 24 | Mixed layer |
| **30 m** | 0.912 | 0.368 | -0.022 | 24 | Top of seasonal thermocline |
| **50 m** | 0.887 | 0.421 | +0.031 | 24 | High eddy sensitivity ($Z_{th}$) |
| **75 m** | 0.871 | 0.465 | +0.028 | 24 | Main thermocline core |
| **100 m** | 0.854 | 0.512 | -0.014 | 24 | Thermocline gradient peak |
| **125 m** | 0.832 | 0.548 | -0.019 | 24 | Sub-thermocline transition |
| **150 m** | 0.814 | 0.582 | +0.025 | 24 | Strong SLA eddy signature |
| **200 m** | 0.792 | 0.614 | +0.018 | 24 | Lower thermocline |
| **250 m** | 0.771 | 0.628 | -0.011 | 24 | Mesopelagic boundary |
| **300 m** | 0.755 | 0.641 | -0.008 | 24 | Deep thermal signature |
| **400 m** | 0.738 | 0.655 | +0.014 | 24 | Deep thermal signature |
| **600 m** | 0.712 | 0.672 | -0.005 | 24 | Deep ocean |
| **1000 m** | 0.684 | 0.695 | -0.002 | 24 | Near-asymptotic deep floor |

---

## 3. Sub-Regional Breakdown

| Sub-Region | Argo Profiles ($N$) | Mean Upper 500m Correlation | Mean Upper 500m RMSE (°C) | Physical Validity |
|---|---|---|---|---|
| **Arabian Sea** | 14 | 0.862 | 0.468 °C | 100.0% |
| **Bay of Bengal** | 8 | 0.825 | 0.531 °C | 100.0% |
| **Equatorial Indian Ocean** | 2 | 0.851 | 0.478 °C | 100.0% |

### Key Observations:
1. **Embedding Superiority:** Self-supervised pretraining on multi-channel surface fields boosts subsurface reconstruction correlation by **+0.162** over direct regression.
2. **Physics Enforcement:** The differentiable stability penalty completely eliminated gravitational density inversions ($100\%$ monotonic valid profiles vs $89.2\%$ in direct regression).
3. **Upper 500m Target Exceeded:** Achieved $r = 0.846$ across the upper 500m, comfortably exceeding the PRD requirement ($r > 0.70$).
