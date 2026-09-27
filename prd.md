# OceanEmbed — Product Requirements Document (PRD)

**Problem ID:** SIH26066
**Organization:** Ministry of Earth Sciences (MoES)
**Tagline:** See through the ocean using only satellite surface data

---

## 1. Background & Problem Statement

We can only directly measure ocean temperature **below** the surface using sparse
in-situ instruments (Argo floats, moored buoys). Satellites see the entire ocean
surface every day, but only the top layer (SST, salinity, sea level, currents, winds).

**The gap:** subsurface structure — where the heat and cold actually sit down to
1000 m — is invisible to satellites and only sparsely sampled in-situ.

**The opportunity:** surface fields carry statistical fingerprints of what is
happening underneath (fronts, eddies, upwelling signatures). A model trained on
years of paired surface/subsurface data (satellite in, reanalysis out) can learn
to infer the subsurface state from the surface alone, and generalize to real,
independently-measured Argo profiles it has never seen.

## 2. Goals

| # | Goal |
|---|------|
| G1 | Build a deep learning model that takes daily satellite surface fields as input and outputs subsurface temperature at 15 depth levels (0–1000 m) |
| G2 | Operate on a 0.25° daily grid over the North Indian Ocean (Bay of Bengal + Arabian Sea) |
| G3 | Validate against **independent** Argo float profiles (never seen in training), reporting correlation, RMSE, and bias per depth |
| G4 | Demonstrate the approach is physically sensible (stable density profiles, plausible thermocline) and not a black box |
| G5 | Deliver a working proof-of-concept demo, not just an offline notebook |

## 3. Non-Goals (Out of Scope)

- Global-ocean coverage (region restricted to North Indian Ocean for the PoC)
- Real-time operational deployment / integration into an operational forecast system
- Biogeochemical variables (oxygen, chlorophyll, pH, etc.)
- Sub-daily (hourly) temporal resolution
- Depths beyond 1000 m

## 4. Users & Stakeholders

| Stakeholder | Interest |
|---|---|
| MoES / INCOIS researchers | Cheaper proxy for subsurface state where Argo coverage is sparse |
| Marine heatwave monitoring teams | Early detection of subsurface warm anomalies |
| Fisheries departments | Thermocline depth affects fish aggregation zones |
| Ocean data assimilation systems | A candidate background/prior field where floats are absent |
| Hackathon judges (immediate) | Technical credibility, novelty, and a working demo |

## 5. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | Ingest daily satellite surface products: SST (OSTIA), SSS (SMAP/SMOS), SSH (DUACS), currents (OSCAR), winds (CCMP) |
| FR2 | Regrid all inputs to a common 0.25° daily grid over the study region |
| FR3 | Stack lagged inputs (past N days) to capture the deep ocean's delayed response |
| FR4 | Pretrain a self-supervised embedding (masked autoencoder / ViT) on surface fields alone, without temperature labels |
| FR5 | Decode the embedding into subsurface temperature **anomalies** at 15 depth levels using GLORYS reanalysis as the training target |
| FR6 | Predict thermocline depth as an auxiliary output |
| FR7 | Enforce a physical stability constraint so predicted profiles are not non-monotonic/unphysical |
| FR8 | Evaluate on held-out years and independent Argo profiles; report correlation, RMSE, bias per depth level |
| FR9 | Provide an interactive demo (dashboard) visualizing surface input → predicted subsurface profile → skill scores |

## 6. Non-Functional Requirements

- **Reproducibility:** training pipeline must run end-to-end from a documented environment (see `rules.md`)
- **Compute budget:** must be trainable on a single GPU (Colab Pro / Kaggle T4-class) within hackathon time constraints
- **Explainability:** per-depth error breakdown must be reportable, not just an aggregate score
- **Data provenance:** every dataset used must be traceable to its official source (Copernicus Marine, PODAAC, Argo GDAC)
- **Extensibility:** architecture should not hard-code the North Indian Ocean region; scaling to global coverage should require config changes, not a rewrite

## 7. Success Metrics

| Metric | Target (PoC) |
|---|---|
| Correlation vs. independent Argo (per depth) | > 0.7 in upper 500 m |
| RMSE vs. independent Argo (per depth) | Competitive with published baselines (e.g. Su et al. 2022 DORS) |
| Bias vs. independent Argo | Near-zero, no systematic depth-dependent drift |
| Physical validity | 0 unphysical (non-monotonic density) profiles in test set |
| Demo | Judges can select a date/location and see predicted vs. actual profile live |

## 8. Constraints & Assumptions

- Hackathon timeline is fixed; full historical reprocessing of all datasets is not feasible — assume a reduced training window (e.g. 5–10 years) for the PoC
- Team has access to at least one GPU-backed notebook environment
- GLORYS reanalysis is treated as ground truth for training (it is itself a model+assimilation product, not perfect truth — this is a known limitation, not a bug)
- Argo coverage in the Bay of Bengal is sparser than the Arabian Sea; validation sample sizes will differ by sub-region

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Looks like "a generic CNN/ViT on surface data" to reviewers | Emphasize embedding-first pretraining + physics-aware constraints + anomaly prediction as differentiators |
| Sparse Argo coverage limits validation confidence | Report sample counts per depth/region alongside metrics; be transparent about confidence intervals |
| Weak skill at depth (below thermocline) | Report per-depth metrics honestly rather than an aggregated single number; discuss as a known limitation |
| Compute/time overrun during hackathon | Scope PoC to Bay of Bengal *or* Arabian Sea, not both, if time is short |

## 10. Milestones

See `tasks.md` for the full phased breakdown. High-level:

1. Data pipeline + regridding
2. Self-supervised pretraining (embedding)
3. Depth decoder + physics constraints
4. Validation against Argo
5. Demo + documentation + pitch
