# OceanEmbed — Project Rules & Conventions

## 1. Repository Structure

```
oceanembed/
├── data/
│   ├── raw/              # never committed — .gitignore'd
│   ├── interim/          # regridded, harmonized intermediates
│   └── processed/        # final tensors ready for training
├── notebooks/            # exploration only, not the source of truth
├── src/
│   ├── data/             # ingestion, regridding, dataset classes
│   ├── models/           # encoder, decoder, physics losses
│   ├── training/         # training loops, configs
│   └── eval/             # validation against Argo, metrics
├── configs/              # YAML configs per experiment
├── demo/                 # Streamlit app
├── docs/                 # prd.md, architecture.md, design.md, tasks.md
├── tests/
├── requirements.txt / environment.yml
└── README.md
```

**Rule:** raw satellite/reanalysis data is never committed to git. Only scripts
to regenerate it are committed. Use `.gitignore` for `data/raw/` and
`data/interim/`.

## 2. Coding Standards

- Python throughout; target Python 3.10+
- Follow **PEP 8**; run `black` + `ruff` before every commit
- Type hints required on all function signatures in `src/`
- Every model class and loss function needs a docstring explaining shape
  contracts: `(B, ..., C, H, W)` conventions must be documented, not assumed
- No notebook code is considered "final" — anything that goes into the
  pipeline must be refactored into `src/` before being called production code

## 3. Data Handling Rules

- Every dataset used must be traceable to its official source (Copernicus
  Marine, NASA PODAAC, Argo GDAC) — record product name + version + access
  date in `data/SOURCES.md`
- Regridding must be deterministic and reproducible: pin `xESMF`/`CDO`
  versions in `environment.yml`
- Train/validation/test splits must be **by whole year**, never by random
  shuffling of daily samples — prevents temporal leakage
- Argo profiles used for final validation must **never** appear in any
  training or hyperparameter-tuning step, even indirectly

## 4. Experiment Tracking

- Every training run gets a config file in `configs/` (YAML), named
  `exp_<short-description>_<date>.yaml`
- Log: config used, git commit hash, metrics per depth level, and physical
  validity check results
- Model checkpoints named: `<stage>_<encoder-type>_<date>_<epoch>.pt`
  (e.g. `decoder_vit-mae_20260315_e40.pt`)
- Keep a `RESULTS.md` log table: run name, region, correlation/RMSE/bias per
  depth, notes

## 5. Git Workflow

- `main` branch is always demo-able
- Feature branches: `feature/<short-name>` (e.g. `feature/depth-decoder`)
- Commit messages: imperative mood, reference the PRD requirement ID where
  relevant (e.g. `FR5: add depth-conditioned decoder head`)
- No direct commits to `main` during the hackathon build window — even solo,
  use PRs for a clean history judges/reviewers can read

## 6. Physical & Scientific Validity Rules

- Predicted profiles must be checked for monotonic density with depth before
  being reported as valid outputs (via GSW-Python)
- Any reported skill metric (correlation, RMSE, bias) must state the sample
  size (number of Argo profiles) it is based on
- Do not report a single aggregated metric across all depths as the headline
  number — always show the per-depth breakdown alongside it
- Be explicit in all documentation that GLORYS is a **model+assimilation
  product**, not ground truth — it is the best available proxy, not perfect

## 7. Attribution & Licensing

- Copernicus Marine Service, NASA PODAAC, and Argo data are used under their
  respective open-data licenses — cite the specific product DOIs in
  `docs/architecture.md` and any submitted report
- Reference the Argo Program (Roemmich et al. 2009) and GLORYS12V1 (Lellouche
  et al. 2021) explicitly in the final PDF submission, not just in code
  comments

## 8. Team Roles (suggested)

| Role | Responsibility |
|---|---|
| Data engineer | Ingestion, regridding, dataset classes |
| ML engineer (encoder) | Self-supervised pretraining, embedding quality |
| ML engineer (decoder) | Depth decoder, physics losses, training loop |
| Validation lead | Argo comparison, metrics, per-depth reporting |
| Demo/frontend | Streamlit dashboard |
| Docs/PM | Keeps `prd.md`/`tasks.md` current, owns the pitch narrative |

## 9. Definition of Done (per component)

A component (e.g. "regridding pipeline", "depth decoder") is **done** when:
1. It runs end-to-end from the documented environment with no manual steps
2. It has at least one automated test (`tests/`) covering shape/sanity checks
3. It is referenced correctly in `architecture.md` (docs updated alongside code)
4. Its output has been visually or numerically sanity-checked, not just
   "ran without crashing"
