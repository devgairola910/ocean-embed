# OceanEmbed — Design Document

This document covers implementation-level design decisions that sit between
the high-level `architecture.md` and actual code: data schemas, model
internals, loss weighting, training configuration, and the demo's interface
contract.

## 1. Data Schema

### 1.1 Input tensor

```
shape: (B, T_lag, C, H, W)

B      = batch size
T_lag  = number of lagged days stacked (default: 5 — today + past 4 days)
C      = 7 channels:
           0: SST anomaly (°C)
           1: SSS anomaly (psu)
           2: SSH / SLA anomaly (m)
           3: U-current (m/s)
           4: V-current (m/s)
           5: Wind-u (m/s)
           6: Wind-v (m/s)
H, W   = spatial grid (0.25° resolution over North Indian Ocean,
           lat 0°–25°N, lon 40°–100°E → H≈100, W≈240, adjust to final bbox)
```

All channels are stored as **anomalies relative to a daily climatology**
(computed from the training period only, never including test years), not raw
values. This keeps the input distribution stationary and makes the decoder's
anomaly-prediction target consistent with the input representation.

### 1.2 Target tensor (training)

```
shape: (B, D, H, W)
D = 15 depth levels: [0, 10, 20, 30, 50, 75, 100, 125, 150, 200,
                       250, 300, 400, 600, 1000] m  (adjust to GLORYS's native levels)
value = temperature anomaly (°C) relative to depth-wise climatology
```

### 1.3 Auxiliary targets

```
thermocline_depth: (B, H, W)      — derived from GLORYS profile (max dT/dz)
land_sea_mask:     (H, W) static  — excluded from loss
```

## 2. Preprocessing Design

- **Climatology:** computed per calendar day-of-year (harmonic smoothing over
  a 31-day window) from GLORYS training years only
- **Normalization:** each channel is standardized (zero mean, unit variance)
  using statistics computed on the training split only
- **Masking:** land pixels and pixels with missing satellite retrievals
  (cloud-affected SST, etc.) are masked out of the loss, not imputed with
  fabricated values
- **Lag stacking:** implemented as a sliding window over the time axis at
  dataset-construction time, not inside the model — keeps the model input
  contract simple

## 3. Model Design

### 3.1 Patch embedding

- Patch size: 16×16 (tunable; smaller patches for higher spatial fidelity at
  higher compute cost)
- Each patch flattened across `T_lag × C` and linearly projected to embedding
  dimension `D_model` (default: 256 for hackathon-scale compute; 768 if
  compute allows)

### 3.2 Encoder (Stage B — self-supervised)

- Standard ViT encoder blocks (pre-norm, multi-head self-attention + MLP)
- Masking ratio: 75% (MAE default) — only visible patches pass through the
  full encoder; this is what makes MAE pretraining cheap relative to
  processing all patches
- Positional encoding: 2D sinusoidal (lat/lon grid position), since ocean
  fields are not translation-invariant the way natural images are (e.g.
  equatorial dynamics differ from higher latitudes)
- Pretraining decoder (separate, lightweight, discarded after Stage B):
  shallow transformer that reconstructs masked patches; not used at inference

### 3.3 Depth decoder (Stage C — supervised)

- Input: pooled or per-patch embedding from the (now frozen or fine-tuned)
  encoder
- Depth conditioning: each of the 15 depth levels has a **learned embedding
  token**; the decoder is queried once per depth level (cross-attention: depth
  token attends over spatial embedding tokens), producing a spatial anomaly
  map per depth
- Output: single linear projection to scalar temperature anomaly per
  (patch/pixel, depth)

### 3.4 Auxiliary heads

- **Thermocline depth head:** small MLP on the pooled embedding → scalar
  depth per spatial location
- **Stability constraint:** not a separate network — a differentiable penalty
  computed from the *output* profile. Using GSW-Python-derived potential
  density formulas (or a simplified linearized approximation for
  differentiability inside the training loop), penalize any depth transition
  where density does not increase downward

## 4. Loss Function Design

```python
L_total = (
    w_pretrain   * L_reconstruction        # Stage B only
    + w_temp     * L_temperature_anomaly   # per-depth MSE, masked over land/missing
    + w_thermo   * L_thermocline           # MSE on scalar thermocline depth
    + w_stability* L_stability_penalty     # hinge loss on non-monotonic density
)
```

Suggested starting weights (tune empirically):
`w_temp = 1.0, w_thermo = 0.2, w_stability = 0.1`

Per-depth loss should be **weighted equally across depths by default**, not
weighted by data density — otherwise the model over-fits to the well-sampled
upper ocean and ignores deeper levels. Report both weighted and unweighted
metrics to be transparent about this trade-off.

## 5. Training Configuration

| Setting | Stage B (pretrain) | Stage C (fine-tune) |
|---|---|---|
| Optimizer | AdamW | AdamW |
| LR schedule | cosine decay with warmup | cosine decay, lower peak LR |
| Batch size | as large as GPU allows | moderate (memory grows with 15-depth output) |
| Epochs | until reconstruction loss plateaus | early-stopped on held-out-year validation loss |
| Data split | full satellite record (no labels needed) | temporal split — hold out entire years |
| Augmentation | none (physical fields, not natural images — avoid rotations/flips that break lat/lon semantics) | none |

## 6. Validation Design

- Validation set = held-out years (never touched during Stage B or C training)
- Final reported metrics come from **independent Argo profiles**, matched in
  space/time to the nearest model grid cell/day
- Metrics computed **per depth level** and **per sub-region** (Bay of Bengal
  vs. Arabian Sea), not pooled into one number
- Report sample count (number of matched Argo profiles) alongside every metric
- Sanity check: compare model skill against a naive climatology-only baseline
  and against a direct-regression baseline (no self-supervised pretraining) to
  demonstrate the value of the embedding-first approach

## 7. Demo Interface Contract

### 7.1 Streamlit app inputs
- Date picker (within available cached range)
- Location picker (lat/lon within Bay of Bengal / Arabian Sea bounding box)

### 7.2 Displayed outputs
- Surface input maps for the selected date (small multiples: SST, SSS, SSH)
- Predicted subsurface temperature profile (0–1000 m) at the selected point
- Overlaid actual GLORYS profile and, where available, actual Argo profile
- Per-depth error (predicted − actual) as a secondary plot
- Skill score summary (correlation/RMSE/bias) for the currently displayed
  profile

### 7.3 Data contract (internal)

```json
{
  "date": "2019-08-15",
  "lat": 15.5,
  "lon": 88.25,
  "predicted_profile": [/* 15 floats, °C anomaly */],
  "glorys_profile": [/* 15 floats */],
  "argo_profile": [/* up to 15 floats, may have gaps */],
  "thermocline_depth_m": 62.3,
  "skill": {"correlation": 0.81, "rmse": 0.64, "bias": -0.05}
}
```

This JSON contract is what the model-serving function returns to the Streamlit
frontend — keeping it explicit means the demo and the model code can be built
in parallel by different team members.
