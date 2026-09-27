# OceanEmbed — Data Sources & Provenance

This document tracks all external datasets used in OceanEmbed, their native resolutions, access portals, and licensing/citations as required by `rules.md`.

| Variable | Product | Native Resolution | Temporal | Source / Portal | Citation / DOI |
|---|---|---|---|---|---|
| Sea Surface Temperature (SST) | OSTIA | ~0.05° (~5 km) | Daily | Copernicus Marine Service (`copernicusmarine`) | Donlon et al. (2012) |
| Sea Surface Salinity (SSS) | SMAP / SMOS L3 | ~25–40 km | Daily / 8-day composite | NASA PODAAC / ESA Earth Online | Fore et al. (2016) |
| Sea Surface Height (SSH / SLA) | DUACS Multi-Mission Altimetry | 0.25° | Daily | Copernicus Marine Service | Taburet et al. (2019) |
| Surface Currents (U, V) | OSCAR Surface Currents | ~0.25–0.33° | 5-day / Daily | NASA PODAAC | ESR / Bonjean & Lagerloef (2002) |
| Surface Winds (Wind-u, Wind-v) | CCMP V3.0 Cross-Calibrated Multi-Platform | 0.25° | 6-hourly → Daily avg | NASA / Remote Sensing Systems (RSS) | Atlas et al. (2011), Mears et al. (2022) |
| Subsurface Temperature (Training Target) | GLORYS12V1 Global Reanalysis | 1/12° (~8 km), 50 levels | Daily | Copernicus Marine Service (Mercator Ocean) | Lellouche et al. (2021) |
| Independent Ground-Truth Validation | Argo Float Profiles | In-situ vertical profiles (0–2000m) | Variable (10-day cycle) | Argo GDAC (via `argopy` / Coriolis / US GODAE) | Roemmich et al. (2009), Wong et al. (2020) |

## Spatial Extent: North Indian Ocean
- Latitude: 0.0°N to 25.0°N
- Longitude: 40.0°E to 100.0°E
- Grid: Regular 0.25° × 0.25° (H=100, W=240 grid cells)
- Standard Depth Levels (m): `[0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 600, 1000]`
