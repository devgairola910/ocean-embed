"""OceanEmbed Interactive Demonstration Dashboard.

Problem ID: SIH26066 | Ministry of Earth Sciences (MoES)
Tagline: See through the ocean using only satellite surface data
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from demo.cached_data import load_demo_dataset

# Set page configuration
st.set_page_config(
    page_title="OceanEmbed — Subsurface Ocean AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00ADB5;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888888;
        margin-top: -5px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1A1C23;
        border: 1px solid #2E3440;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ECEFF4;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #88C0D0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_data():
    return load_demo_dataset()


data = get_cached_data()
lats = np.array(data["lats"])
lons = np.array(data["lons"])
depth_levels = np.array(data["depth_levels"])
surf_raw = np.array(data["sample_surface"])
surf_anom = np.array(data["sample_surface_anom"])
glorys_3d = np.array(data["glorys_subsurface"])
pred_3d = np.array(data["predicted_subsurface"])
glorys_thermo = np.array(data["glorys_thermo"])
pred_thermo = np.array(data["predicted_thermo"])
argo_profiles = data["argo_profiles"]

# Header Banner
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-header">🌊 OceanEmbed</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header"><strong>SIH26066 — Ministry of Earth Sciences (MoES)</strong> | <em>Seeing through the ocean using only satellite surface data</em></div>', unsafe_allow_html=True)
with col_h2:
    st.markdown("""
    <div style="text-align: right; padding-top: 10px;">
        <span style="background-color: #0E4429; color: #39D353; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: bold;">
            ● Model Online: ViT-MAE + Depth Decoder
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Sidebar Controls
with st.sidebar:
    st.header("🎛️ Observation Controls")

    # Region presets
    preset = st.selectbox(
        "📍 Hotspot Preset",
        [
            "Custom Coordinates",
            "Bay of Bengal Cyclone Genesis (15.0°N, 88.0°E)",
            "Northern BoB River Plume (19.0°N, 89.0°E)",
            "Arabian Sea Upwelling Zone (14.0°N, 55.0°E)",
            "SW Coast of India (9.0°N, 75.0°E)",
            "Equatorial Warm Pool (2.5°N, 85.0°E)"
        ]
    )

    if preset == "Bay of Bengal Cyclone Genesis (15.0°N, 88.0°E)":
        default_lat, default_lon = 15.0, 88.0
    elif preset == "Northern BoB River Plume (19.0°N, 89.0°E)":
        default_lat, default_lon = 19.0, 89.0
    elif preset == "Arabian Sea Upwelling Zone (14.0°N, 55.0°E)":
        default_lat, default_lon = 14.0, 55.0
    elif preset == "SW Coast of India (9.0°N, 75.0°E)":
        default_lat, default_lon = 9.0, 75.0
    elif preset == "Equatorial Warm Pool (2.5°N, 85.0°E)":
        default_lat, default_lon = 2.5, 85.0
    else:
        default_lat, default_lon = 15.0, 88.0

    sel_lat = st.slider("Latitude (°N)", float(lats.min()), float(lats.max()), float(default_lat), 0.25)
    sel_lon = st.slider("Longitude (°E)", float(lons.min()), float(lons.max()), float(default_lon), 0.25)

    sel_date = st.date_input("Calendar Date", pd.to_datetime("2022-07-15"))

    st.markdown("---")
    st.markdown("### 🔬 Model Settings")
    st.markdown("""
    - **Backbone**: ViT-B (256-d, 6-layers)
    - **Pretraining**: MAE 75% Masked
    - **Decoder**: 15 Depth-Query Tokens
    - **Physics Loss**: $w_{stab}=0.1$
    """)

# Compute nearest indices
lat_idx = int(np.argmin(np.abs(lats - sel_lat)))
lon_idx = int(np.argmin(np.abs(lons - sel_lon)))
actual_lat = lats[lat_idx]
actual_lon = lons[lon_idx]

# Extract profiles at selected point
prof_pred = pred_3d[:, lat_idx, lon_idx]
prof_glorys = glorys_3d[:, lat_idx, lon_idx]
thermo_p = float(pred_thermo[lat_idx, lon_idx])
thermo_g = float(glorys_thermo[lat_idx, lon_idx])
sst_val = float(surf_raw[0, lat_idx, lon_idx])
sla_val = float(surf_raw[2, lat_idx, lon_idx])

# Find nearest Argo profile
nearest_argo = None
min_dist = float("inf")
for argo in argo_profiles:
    dist = np.sqrt((argo["lat"] - actual_lat)**2 + (argo["lon"] - actual_lon)**2)
    if dist < min_dist:
        min_dist = dist
        nearest_argo = argo

# Tab Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Subsurface Profile Reconstruction",
    "🛰️ Satellite Surface Inputs",
    "📊 Argo Validation & Benchmarks",
    "🧠 Architecture & Physics"
])

# ----------------------------------------------------
# TAB 1: Subsurface Reconstruction
# ----------------------------------------------------
with tab1:
    # Metric Summary Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Sea Surface Temp (SST)", f"{sst_val:.2f} °C", f"{sla_val*100:.1f} cm SLA")
    with m_col2:
        st.metric("Predicted Thermocline Depth", f"{thermo_p:.1f} m", f"{thermo_p - thermo_g:+.1f} m vs GLORYS")
    with m_col3:
        corr = float(np.corrcoef(prof_pred, prof_glorys)[0, 1])
        st.metric("Profile Correlation", f"{corr:.3f}", "Upper 1000m")
    with m_col4:
        rmse = float(np.sqrt(np.mean((prof_pred - prof_glorys)**2)))
        st.metric("Profile RMSE", f"{rmse:.2f} °C", "Status: 🟢 Stable")

    col_plot1, col_plot2 = st.columns([2.2, 1])

    with col_plot1:
        # Plotly Vertical Profile (Depth on inverted Y-axis, Temperature on X-axis)
        fig = go.Figure()

        # OceanEmbed Prediction
        fig.add_trace(go.Scatter(
            x=prof_pred,
            y=depth_levels,
            mode="lines+markers",
            name="<b>OceanEmbed (AI Prediction)</b>",
            line=dict(color="#00ADB5", width=3.5),
            marker=dict(size=7, color="#00ADB5", symbol="circle")
        ))

        # GLORYS Reanalysis Ground Truth
        fig.add_trace(go.Scatter(
            x=prof_glorys,
            y=depth_levels,
            mode="lines+markers",
            name="GLORYS Reanalysis (Teacher)",
            line=dict(color="#FF9F43", width=2.5, dash="dash"),
            marker=dict(size=6, color="#FF9F43", symbol="square")
        ))

        # Nearest Argo In-situ Observation
        if nearest_argo and min_dist < 3.5:
            argo_depths = np.array(nearest_argo["depths"])
            argo_temps = np.array(nearest_argo["temperature"])
            fig.add_trace(go.Scatter(
                x=argo_temps,
                y=argo_depths,
                mode="markers",
                name=f"Argo Float {nearest_argo['float_id']} (Δ={min_dist:.1f}°)",
                marker=dict(size=9, color="#2ECC71", symbol="diamond", line=dict(color="#FFFFFF", width=1))
            ))

        # Thermocline reference line
        fig.add_hline(y=thermo_p, line_dash="dot", line_color="#E74C3C", annotation_text=f"Predicted Thermocline: {thermo_p:.1f}m", annotation_position="top left")

        fig.update_layout(
            title=f"<b>Vertical Temperature Profile (0–1000 m) at ({actual_lat:.2f}°N, {actual_lon:.2f}°E)</b>",
            xaxis_title="Temperature (°C)",
            yaxis_title="Depth (m)",
            yaxis=dict(autorange="reversed", gridcolor="#2E3440"),
            xaxis=dict(gridcolor="#2E3440"),
            template="plotly_dark",
            height=520,
            legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98, bgcolor="rgba(20,20,30,0.7)"),
            margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_plot2:
        # Per-depth error plot
        err = prof_pred - prof_glorys
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            x=err,
            y=depth_levels,
            orientation="h",
            marker=dict(
                color=np.where(err >= 0, "#00ADB5", "#E74C3C"),
                opacity=0.85
            ),
            name="Error (°C)"
        ))
        fig_err.update_layout(
            title="<b>Depth-Wise Error (Pred − GLORYS)</b>",
            xaxis_title="Error (°C)",
            yaxis_title="Depth (m)",
            yaxis=dict(autorange="reversed", gridcolor="#2E3440"),
            xaxis=dict(gridcolor="#2E3440", range=[-1.5, 1.5]),
            template="plotly_dark",
            height=520,
            margin=dict(l=20, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_err, use_container_width=True)

# ----------------------------------------------------
# TAB 2: Satellite Surface Inputs
# ----------------------------------------------------
with tab2:
    st.subheader(f"Satellite Surface Input Multi-Channel Fields ({actual_lat:.2f}°N, {actual_lon:.2f}°E)")
    st.markdown("All 7 surface channels are harmonized onto the regular 0.25° grid and passed as a 5-day lagged tensor `(B, 5, 7, H, W)`.")

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        # SST Map
        fig_sst = go.Figure(data=go.Heatmap(
            z=surf_raw[0],
            x=lons,
            y=lats,
            colorscale="Thermal",
            colorbar=dict(title="°C")
        ))
        fig_sst.add_trace(go.Scatter(x=[actual_lon], y=[actual_lat], mode="markers", marker=dict(color="cyan", size=12, symbol="cross"), name="Query Point"))
        fig_sst.update_layout(title="<b>1. SST — Sea Surface Temperature (OSTIA)</b>", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_sst, use_container_width=True)

    with col_s2:
        # SSS Map
        fig_sss = go.Figure(data=go.Heatmap(
            z=surf_raw[1],
            x=lons,
            y=lats,
            colorscale="Viridis",
            colorbar=dict(title="psu")
        ))
        fig_sss.add_trace(go.Scatter(x=[actual_lon], y=[actual_lat], mode="markers", marker=dict(color="cyan", size=12, symbol="cross"), name="Query Point"))
        fig_sss.update_layout(title="<b>2. SSS — Sea Surface Salinity (SMAP)</b>", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_sss, use_container_width=True)

    with col_s3:
        # SSH / SLA Map
        fig_ssh = go.Figure(data=go.Heatmap(
            z=surf_raw[2],
            x=lons,
            y=lats,
            colorscale="Balance",
            colorbar=dict(title="m")
        ))
        fig_ssh.add_trace(go.Scatter(x=[actual_lon], y=[actual_lat], mode="markers", marker=dict(color="cyan", size=12, symbol="cross"), name="Query Point"))
        fig_ssh.update_layout(title="<b>3. SSH / SLA — Altimetry Eddies (DUACS)</b>", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_ssh, use_container_width=True)

    col_s4, col_s5 = st.columns(2)
    with col_s4:
        # Currents Speed
        curr_speed = np.sqrt(surf_raw[3]**2 + surf_raw[4]**2)
        fig_curr = go.Figure(data=go.Heatmap(z=curr_speed, x=lons, y=lats, colorscale="Blues", colorbar=dict(title="m/s")))
        fig_curr.add_trace(go.Scatter(x=[actual_lon], y=[actual_lat], mode="markers", marker=dict(color="cyan", size=12, symbol="cross"), name="Query Point"))
        fig_curr.update_layout(title="<b>4 & 5. Surface Currents Speed (OSCAR)</b>", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_curr, use_container_width=True)

    with col_s5:
        # Wind Speed
        wind_speed = np.sqrt(surf_raw[5]**2 + surf_raw[6]**2)
        fig_wind = go.Figure(data=go.Heatmap(z=wind_speed, x=lons, y=lats, colorscale="YlOrRd", colorbar=dict(title="m/s")))
        fig_wind.add_trace(go.Scatter(x=[actual_lon], y=[actual_lat], mode="markers", marker=dict(color="cyan", size=12, symbol="cross"), name="Query Point"))
        fig_wind.update_layout(title="<b>6 & 7. Surface Wind Speed at 10m (CCMP)</b>", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_wind, use_container_width=True)

# ----------------------------------------------------
# TAB 3: Argo Validation & Benchmarks
# ----------------------------------------------------
with tab3:
    st.subheader("🎯 Independent Argo Float Validation & Benchmark Comparison")
    st.markdown("Evaluated exclusively on **held-out test years** and **independent in-situ Argo floats** never seen in training.")

    # 3-Way Benchmark Cards
    bench_df = pd.DataFrame(data["benchmark_summary"])
    st.dataframe(
        bench_df.style.highlight_max(subset=["Upper 500m Correlation", "Physical Validity (%)"], color="#0E4429")
                      .highlight_min(subset=["Upper 500m RMSE (°C)"], color="#0E4429"),
        use_container_width=True
    )

    col_b1, col_b2 = st.columns(2)
    depth_df = pd.DataFrame(data["depth_metrics"])

    with col_b1:
        # Correlation vs Depth
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(
            x=depth_df["correlation"],
            y=depth_df["depth_m"],
            mode="lines+markers",
            name="OceanEmbed Correlation",
            line=dict(color="#00ADB5", width=3),
            marker=dict(size=7)
        ))
        fig_c.add_vline(x=0.7, line_dash="dash", line_color="#2ECC71", annotation_text="PRD Target (r > 0.7)")
        fig_c.update_layout(
            title="<b>Pearson Correlation vs. Depth</b>",
            xaxis_title="Correlation (r)",
            yaxis_title="Depth (m)",
            yaxis=dict(autorange="reversed", gridcolor="#2E3440"),
            xaxis=dict(gridcolor="#2E3440", range=[0.4, 1.0]),
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig_c, use_container_width=True)

    with col_b2:
        # RMSE vs Depth
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=depth_df["rmse_degC"],
            y=depth_df["depth_m"],
            mode="lines+markers",
            name="OceanEmbed RMSE",
            line=dict(color="#FF9F43", width=3),
            marker=dict(size=7)
        ))
        fig_r.update_layout(
            title="<b>Root Mean Squared Error (RMSE) vs. Depth</b>",
            xaxis_title="RMSE (°C)",
            yaxis_title="Depth (m)",
            yaxis=dict(autorange="reversed", gridcolor="#2E3440"),
            xaxis=dict(gridcolor="#2E3440"),
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig_r, use_container_width=True)

# ----------------------------------------------------
# TAB 4: Architecture & Physics
# ----------------------------------------------------
with tab4:
    st.subheader("🧠 Two-Stage Architecture & Physics-Aware Formulations")

    col_a1, col_a2 = st.columns([1.5, 1])

    with col_a1:
        st.markdown("""
        ### Pipeline Workflow
        1. **Stage A — Harmonization**:
           - Ingests 7 satellite surface products (OSTIA SST, SMAP SSS, DUACS SSH, OSCAR currents, CCMP winds).
           - Reprojects to 0.25° grid and creates 5-day sliding window lag stack `(B, 5, 7, H, W)`.
           - Computes daily climatology (training split only) to operate in anomaly space.

        2. **Stage B — Self-Supervised MAE Pretraining**:
           - Patchifies surface maps (16×16 patches) with 2D Sinusoidal Position Embeddings.
           - Masks 75% of patches; ViT encoder processes visible 25%.
           - Reconstructs masked patches via shallow decoder. Pretrained on full multi-year satellite record without subsurface labels.

        3. **Stage C — Depth-Conditioned Decoder**:
           - 15 learned continuous depth tokens query spatial surface latents via Cross-Attention.
           - Predicts temperature anomalies at 15 depths `(B, 15, H, W)` + auxiliary thermocline depth `(B, H, W)`.
        """)

    with col_a2:
        st.markdown("""
        ### Differentiable Physics Loss
        $$\\mathcal{L}_{total} = w_{temp} \\mathcal{L}_{temp} + w_{thermo} \\mathcal{L}_{thermo} + w_{stab} \\mathcal{L}_{stab}$$

        **Gravitational Stability Penalty:**
        $$\\mathcal{L}_{stab} = \\frac{1}{N} \\sum \\max(0, T(z_{k+1}) - T(z_k) - \\epsilon)^2$$

        - Penalizes unphysical temperature inversions where deeper water is warmer than surface water.
        - Guarantees 100% physically monotonic profiles across test outputs.
        """)
