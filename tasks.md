# OceanEmbed — Tasks & Roadmap

This breaks the project into phases: a **pre-hackathon prep** phase (do this
before the clock starts) and a **36-hour build** phase (typical SIH grand
finale format), followed by a stretch-goal list if time remains.

---

## Phase 0 — Pre-Hackathon Prep (do in advance)

- [x] Set up shared repo with structure from `rules.md`
- [x] Create Copernicus Marine + NASA Earthdata accounts for all team members
- [x] Pre-download a small cached subset of GLORYS + OSTIA + Argo for the
      target region (1–2 years) so Day 1 doesn't stall on data access
- [x] Confirm GPU access (Colab Pro / Kaggle) for every team member
- [x] Read: Su et al. 2022 (DORS), MAESSTRO, and the Argo Program paper
      (see the reference list) — everyone should understand the closest
      prior work before building
- [x] Decide: Bay of Bengal **or** Arabian Sea as the primary demo region
      (both if time allows, one guaranteed)
- [x] Assign roles per `rules.md` §8

---

## Phase 1 — Data Pipeline (Hours 0–8)

- [x] `FR1` Ingest SST, SSS, SSH, currents, winds for the chosen region/period
- [x] `FR2` Build regridding pipeline to common 0.25° daily grid
- [x] Compute daily climatology per channel (training years only)
- [x] Build lagged-stack dataset class (`T_lag` window)
- [x] Ingest and grid GLORYS as training target (15 depth levels)
- [x] Ingest Argo profiles for the region via `argopy`; hold out entirely
      from anything used in Phase 2/3
- [x] Sanity-check plots: does regridded SST look like SST? Any NaN storms?
- [x] **Exit criteria:** a `Dataset`/`DataLoader` that yields
      `(input_tensor, target_tensor, aux_targets)` batches without crashing

## Phase 2 — Self-Supervised Pretraining (Hours 6–16, overlaps Phase 1 tail)

- [ ] `FR4` Implement patch embedding + ViT encoder
- [ ] Implement MAE-style masking (75% mask ratio) + lightweight reconstruction decoder
- [ ] Train on full available satellite record (no labels needed)
- [ ] Track reconstruction loss curve; confirm it's actually learning
      structure (visualize a few masked/reconstructed examples)
- [ ] Save encoder checkpoint
- **Exit criteria:** reconstruction loss plateaus and reconstructed patches
      visually resemble real surface fields, not blur/noise

## Phase 3 — Depth Decoder & Physics Constraints (Hours 14–24)

- [ ] `FR5` Implement depth-conditioned decoder head (per design.md §3.3)
- [ ] `FR6` Implement thermocline depth auxiliary head
- [ ] `FR7` Implement stability penalty (GSW-Python or simplified differentiable proxy)
- [ ] Wire up combined loss with initial weights from `design.md` §4
- [ ] Fine-tune on GLORYS targets with temporal (whole-year) train/val split
- [ ] Baseline comparison: train a "no-pretraining" direct-regression model
      for the same task (needed to prove the embedding-first approach's value)
- **Exit criteria:** validation loss (on held-out years) is stable and below
      the naive baseline

## Phase 4 — Validation Against Argo (Hours 22–28)

- [ ] `FR8` Match model predictions to independent Argo profiles in
      space/time
- [ ] Compute correlation, RMSE, bias **per depth level**
- [ ] Compute metrics separately for each sub-region if using both basins
- [ ] Compare against: (a) climatology-only baseline, (b) direct-regression
      baseline, (c) OceanEmbed
- [ ] Sanity-check physical validity: 0 unphysical (non-monotonic density)
      profiles in test outputs
- **Exit criteria:** a results table (correlation/RMSE/bias × 15 depths ×
      3 models) ready to drop into the pitch deck

## Phase 5 — Demo Dashboard (Hours 20–32, parallel track)

- [ ] `FR9` Build Streamlit app per the interface contract in `design.md` §7
- [ ] Wire up date/location picker → cached inference → plot
- [ ] Overlay predicted vs. GLORYS vs. Argo profile
- [ ] Display skill scores live for the selected profile
- [ ] Polish: labels, units, region map inset
- **Exit criteria:** a judge can pick a date/point and see a sensible profile
      appear within a few seconds, with no crashes

## Phase 6 — Docs & Pitch (Hours 28–36)

- [ ] Finalize `architecture.md` diagram to match what was actually built
      (update anything that changed from plan to reality)
- [ ] Write the PDF submission content per the "What to put in your PDF"
      checklist from the problem statement:
  - [ ] Architecture diagram (inputs → harmonization → encoder → decoder → validation)
  - [ ] Input dataset table with resolutions and regridding method
  - [ ] Train/validation plan (held-out years) and metrics reported per depth
  - [ ] Paragraph: why embeddings beat direct regression (cite Phase 3 baseline results)
  - [ ] Paragraph: physics-aware checks and what they caught
  - [ ] Users & impact: marine heatwave monitoring, fisheries, data assimilation
  - [ ] Risks & mitigations: sparse Argo coverage, weak skill at depth, year-to-year differences
- [ ] Rehearse the live demo (have a recorded backup video in case of Wi-Fi issues)
- [ ] Prepare answers for the anticipated reviewer risk: "how is this
      different from a generic CNN/ViT on surface data?"

---

## Stretch Goals (only if ahead of schedule)

- [ ] Extend coverage to both Bay of Bengal *and* Arabian Sea
- [ ] Add salinity as a second predicted variable (dual-output decoder)
- [ ] Add uncertainty estimates (e.g. ensemble or predictive variance) per profile
- [ ] Add a marine-heatwave flag: highlight subsurface anomalies exceeding a
      threshold on the dashboard map
- [ ] Add SHAP/attention-based explainability visualization (which surface
      region most influenced a given subsurface prediction)

---

## Task Ownership Template

| Task ID | Description | Owner | Status | Blocked by |
|---|---|---|---|---|
| P1-1 | Data ingestion | | Not started | — |
| P1-2 | Regridding pipeline | | Not started | P1-1 |
| P2-1 | ViT encoder + MAE pretraining | | Not started | P1-2 |
| P3-1 | Depth decoder | | Not started | P2-1 |
| P3-2 | Physics constraints | | Not started | P3-1 |
| P4-1 | Argo validation | | Not started | P3-2 |
| P5-1 | Streamlit demo | | Not started | P3-1 (can start early with dummy outputs) |
| P6-1 | PDF + pitch | | Not started | P4-1, P5-1 |

*(Duplicate this table into your team's task tracker — GitHub Projects,
Notion, or a shared sheet — and fill in owners/status as the build progresses.)*
