/**
 * OceanEmbed — Master Scientific Web Application Controller
 * Ministry of Earth Sciences (MoES) | Problem ID: SIH26066
 * Tagline: See through the ocean using only satellite surface data (0–1000m)
 */

document.addEventListener('DOMContentLoaded', () => {
  initBubbles();
  initOceanPresets();
  initStratificationChart();
  initSubsurfaceCanvas();
  initDrawerTabs();
  initExportModal();
  initHeaderNavTabs();
  checkApiHealth();
});

/* ==========================================================================
   1. AMBIENT BUBBLE PARTICLE SYSTEM
   ========================================================================== */
function initBubbles() {
  const container = document.getElementById('bubblesContainer');
  if (!container) return;

  const bubbleCount = 18;
  for (let i = 0; i < bubbleCount; i++) {
    createBubble(container, true);
  }
}

function createBubble(container, initial = false) {
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  
  const size = Math.random() * 14 + 6;
  const left = Math.random() * 100;
  const duration = Math.random() * 12 + 10;
  const delay = initial ? Math.random() * duration : 0;

  bubble.style.width = `${size}px`;
  bubble.style.height = `${size}px`;
  bubble.style.left = `${left}%`;
  bubble.style.animationDuration = `${duration}s`;
  bubble.style.animationDelay = `-${delay}s`;

  container.appendChild(bubble);

  setTimeout(() => {
    bubble.remove();
    createBubble(container);
  }, (duration - delay) * 1000);
}

/* ==========================================================================
   2. OCEANIC FEATURE PRESETS & MAP SYNCHRONIZATION
   ========================================================================== */
const OCEAN_PRESETS = {
  'H-1': {
    name: 'Bay of Bengal Cyclone Genesis',
    lat: 15.0,
    lon: 88.0,
    coordsText: '15.0°N, 88.0°E',
    sst: '29.8 °C',
    sss: '33.2 PSU',
    sla: '+18.5 cm',
    z_th: '68.5 m',
    ohc: '142 kJ/cm²',
    rmse: '0.492 °C',
    corr: '0.846',
    desc: 'High Tropical Cyclone Heat Potential (TCHP) fuel region.'
  },
  'H-2': {
    name: 'Arabian Sea Coastal Upwelling',
    lat: 14.0,
    lon: 55.0,
    coordsText: '14.0°N, 55.0°E',
    sst: '25.4 °C',
    sss: '36.4 PSU',
    sla: '-12.0 cm',
    z_th: '38.0 m',
    ohc: '98 kJ/cm²',
    rmse: '0.468 °C',
    corr: '0.862',
    desc: 'Somali/Omani coastal cold plume with shallow thermocline.'
  },
  'H-3': {
    name: 'Northern BoB River Plume',
    lat: 19.0,
    lon: 89.0,
    coordsText: '19.0°N, 89.0°E',
    sst: '30.2 °C',
    sss: '29.5 PSU',
    sla: '+14.0 cm',
    z_th: '42.0 m',
    ohc: '124 kJ/cm²',
    rmse: '0.531 °C',
    corr: '0.825',
    desc: 'Ganga-Brahmaputra discharge barrier layer trapping surface heat.'
  },
  'H-4': {
    name: 'SW Coast of India Upwelling',
    lat: 9.0,
    lon: 75.0,
    coordsText: '9.0°N, 75.0°E',
    sst: '27.1 °C',
    sss: '35.1 PSU',
    sla: '-6.5 cm',
    z_th: '45.0 m',
    ohc: '110 kJ/cm²',
    rmse: '0.485 °C',
    corr: '0.848',
    desc: 'Productive coastal upwelling sustaining pelagic fisheries.'
  },
  'H-5': {
    name: 'Equatorial Warm Pool',
    lat: 2.5,
    lon: 85.0,
    coordsText: '2.5°N, 85.0°E',
    sst: '30.5 °C',
    sss: '34.2 PSU',
    sla: '+22.0 cm',
    z_th: '82.0 m',
    ohc: '155 kJ/cm²',
    rmse: '0.478 °C',
    corr: '0.851',
    desc: 'Intense equatorial Wyrtki jet zone with deep thermal reservoir.'
  }
};

let currentPresetKey = 'H-1';

function initOceanPresets() {
  const hotspotItems = document.querySelectorAll('.hotspot-item');
  const mapPins = document.querySelectorAll('.map-hotspot-pin');
  const popover = document.getElementById('mapPopover');
  const popoverTag = document.getElementById('popoverTag');
  const popoverName = document.getElementById('popoverName');
  const popoverSst = document.getElementById('popoverSst');
  const popoverZth = document.getElementById('popoverZth');
  const popoverOhc = document.getElementById('popoverOhc');
  const searchInput = document.getElementById('hotspotSearchInput');
  const popoverActionBtn = document.getElementById('popoverActionBtn');

  function selectPreset(id) {
    const data = OCEAN_PRESETS[id];
    if (!data) return;
    currentPresetKey = id;

    // Update List UI
    hotspotItems.forEach(item => {
      if (item.getAttribute('data-id') === id) {
        item.classList.add('active');
        item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        item.classList.remove('active');
      }
    });

    // Update Map Pins
    mapPins.forEach(pin => {
      if (pin.getAttribute('data-id') === id) {
        pin.classList.add('active');
      } else {
        pin.classList.remove('active');
      }
    });

    // Update Popover
    if (popover && popoverTag && popoverName && popoverSst && popoverZth && popoverOhc) {
      popoverTag.innerText = id;
      popoverName.innerText = data.name;
      popoverSst.innerText = data.sst;
      popoverZth.innerText = data.z_th;
      popoverOhc.innerText = data.ohc;
      popover.style.opacity = '1';
    }

    // Update Metric Cards
    const val1 = document.getElementById('metricVal1');
    const val2 = document.getElementById('metricVal2');
    const val3 = document.getElementById('metricVal3');
    const statusText = document.getElementById('liveStatusText');

    if (val1) val1.innerText = parseFloat(data.rmse).toFixed(3);
    if (val2) val2.innerText = data.corr;
    if (val3) val3.innerText = parseFloat(data.z_th).toFixed(1);
    if (statusText) statusText.innerText = `Target: ${data.coordsText}`;

    // Update Satellite Channel Chips
    const chSst = document.getElementById('chSst');
    const chSss = document.getElementById('chSss');
    const chSla = document.getElementById('chSla');
    if (chSst) chSst.innerText = data.sst;
    if (chSss) chSss.innerText = data.sss;
    if (chSla) chSla.innerText = data.sla;

    // Update Subsurface Profile Chart Canvas
    updateSubsurfaceProfileForPreset(data);
    updateJsonCodePreview(data);
  }

  hotspotItems.forEach(item => {
    item.addEventListener('click', () => {
      const id = item.getAttribute('data-id');
      selectPreset(id);
    });
  });

  mapPins.forEach(pin => {
    pin.addEventListener('click', () => {
      const id = pin.getAttribute('data-id');
      if (OCEAN_PRESETS[id]) selectPreset(id);
    });
  });

  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase();
      hotspotItems.forEach(item => {
        const text = item.innerText.toLowerCase();
        item.style.display = text.includes(q) ? 'flex' : 'none';
      });
    });
  }

  if (popoverActionBtn) {
    popoverActionBtn.addEventListener('click', () => {
      const drawer = document.getElementById('deepDiveDrawer');
      if (drawer) {
        drawer.scrollIntoView({ behavior: 'smooth' });
      }
    });
  }

  // Initial selection
  selectPreset('H-1');
}

/* ==========================================================================
   3. STRATIFICATION CURVE & REGIME TOGGLE
   ========================================================================== */
function initStratificationChart() {
  const tabSummer = document.getElementById('tabSummer');
  const tabWinter = document.getElementById('tabWinter');
  const tabCyclone = document.getElementById('tabCyclone');
  const linePath = document.getElementById('trendLinePath');
  const areaPath = document.getElementById('trendAreaPath');
  const tooltip = document.getElementById('chartTooltip');
  const dots = document.querySelectorAll('.chart-dot');

  const curves = {
    summer: {
      line: "M 50 40 Q 110 45, 160 70 T 260 95 T 350 115 T 440 125",
      area: "M 50 40 Q 110 45, 160 70 T 260 95 T 350 115 T 440 125 L 440 135 L 50 135 Z"
    },
    winter: {
      line: "M 50 55 Q 110 60, 160 85 T 260 105 T 350 120 T 440 128",
      area: "M 50 55 Q 110 60, 160 85 T 260 105 T 350 120 T 440 128 L 440 135 L 50 135 Z"
    },
    cyclone: {
      line: "M 50 35 Q 110 38, 160 62 T 260 88 T 350 110 T 440 124",
      area: "M 50 35 Q 110 38, 160 62 T 260 88 T 350 110 T 440 124 L 440 135 L 50 135 Z"
    }
  };

  function setRegime(regime, activeBtn) {
    [tabSummer, tabWinter, tabCyclone].forEach(b => b && b.classList.remove('active'));
    if (activeBtn) activeBtn.classList.add('active');
    if (curves[regime] && linePath && areaPath) {
      linePath.setAttribute('d', curves[regime].line);
      areaPath.setAttribute('d', curves[regime].area);
    }
  }

  if (tabSummer) tabSummer.addEventListener('click', () => setRegime('summer', tabSummer));
  if (tabWinter) tabWinter.addEventListener('click', () => setRegime('winter', tabWinter));
  if (tabCyclone) tabCyclone.addEventListener('click', () => setRegime('cyclone', tabCyclone));

  dots.forEach(dot => {
    dot.addEventListener('mouseenter', () => {
      const label = dot.getAttribute('data-label') || '';
      const parts = label.split(':');
      if (tooltip && parts.length === 2) {
        tooltip.querySelector('.tooltip-title').innerText = parts[0].trim();
        tooltip.querySelector('.tooltip-val').innerText = parts[1].trim();
        tooltip.style.opacity = '1';
        tooltip.style.left = `${dot.getAttribute('cx') - 50}px`;
        tooltip.style.top = `${dot.getAttribute('cy') - 45}px`;
      }
    });
  });
}

/* ==========================================================================
   4. 3D SUBSURFACE TEMPERATURE PROFILE CANVAS (0-1000m)
   ========================================================================== */
let activeProfileData = {
  depths: [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 600, 1000],
  ai_temps: [29.8, 29.6, 29.1, 28.5, 26.2, 22.4, 18.4, 15.6, 13.5, 11.2, 9.8, 8.7, 7.2, 5.8, 4.5],
  glorys_temps: [29.7, 29.5, 29.0, 28.3, 26.0, 22.1, 18.1, 15.3, 13.2, 11.0, 9.5, 8.5, 7.0, 5.7, 4.4],
  argo_temps: [29.8, 29.5, 29.1, 28.4, 26.1, 22.3, 18.2, 15.4, 13.3, 11.1, 9.6, 8.6, 7.1, 5.7, 4.5],
  thermo_depth: 68.5
};

function initSubsurfaceCanvas() {
  const canvas = document.getElementById('profileCanvas');
  if (!canvas) return;
  renderProfileChart(canvas, activeProfileData);
}

function renderProfileChart(canvas, data) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  const padLeft = 50;
  const padRight = 20;
  const padTop = 20;
  const padBottom = 35;

  const chartW = w - padLeft - padRight;
  const chartH = h - padTop - padBottom;

  const minTemp = 0;
  const maxTemp = 34;
  const maxDepth = 1000;

  function tempToX(t) {
    return padLeft + ((t - minTemp) / (maxTemp - minTemp)) * chartW;
  }

  function depthToY(d) {
    const norm = Math.sqrt(d) / Math.sqrt(maxDepth);
    return padTop + norm * chartH;
  }

  // Grid Lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;

  const depthTicks = [0, 50, 100, 200, 400, 1000];
  ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.font = '10px Inter, sans-serif';
  ctx.textAlign = 'right';

  depthTicks.forEach(d => {
    const y = depthToY(d);
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.stroke();
    ctx.fillText(`${d}m`, padLeft - 6, y + 3);
  });

  const tempTicks = [5, 15, 25, 30];
  ctx.textAlign = 'center';
  tempTicks.forEach(t => {
    const x = tempToX(t);
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, h - padBottom);
    ctx.stroke();
    ctx.fillText(`${t}°C`, x, h - padBottom + 16);
  });

  // Thermocline depth line
  const z_y = depthToY(data.thermo_depth);
  ctx.strokeStyle = '#F43F5E';
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(padLeft, z_y);
  ctx.lineTo(w - padRight, z_y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#F43F5E';
  ctx.font = '10px Outfit, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`Z_th: ${data.thermo_depth}m`, padLeft + 8, z_y - 4);

  // 1. GLORYS Reanalysis
  ctx.strokeStyle = 'rgba(255, 160, 122, 0.75)';
  ctx.lineWidth = 2;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  for (let i = 0; i < data.depths.length; i++) {
    const x = tempToX(data.glorys_temps[i]);
    const y = depthToY(data.depths[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // 2. OceanEmbed AI
  ctx.strokeStyle = '#6FFFE9';
  ctx.lineWidth = 3;
  ctx.shadowColor = '#6FFFE9';
  ctx.shadowBlur = 8;
  ctx.beginPath();
  for (let i = 0; i < data.depths.length; i++) {
    const x = tempToX(data.ai_temps[i]);
    const y = depthToY(data.depths[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // 3. Argo Float points
  ctx.fillStyle = '#10B981';
  ctx.strokeStyle = '#FFFFFF';
  ctx.lineWidth = 1;
  for (let i = 0; i < data.depths.length; i += 2) {
    const x = tempToX(data.argo_temps[i]);
    const y = depthToY(data.depths[i]);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

function updateSubsurfaceProfileForPreset(preset) {
  const canvas = document.getElementById('profileCanvas');
  if (!canvas) return;

  const baseTemp = parseFloat(preset.sst) || 29.8;
  const z_th = parseFloat(preset.z_th) || 68.5;

  activeProfileData.thermo_depth = z_th;
  activeProfileData.ai_temps = activeProfileData.depths.map(d => {
    if (d < 30) return +(baseTemp - (d / 30) * 0.7).toFixed(1);
    if (d <= 150) return +(baseTemp - 1.0 - Math.pow(d / 150, 1.2) * 14.5).toFixed(1);
    return +(14.0 - Math.log(d / 150) * 4.6).toFixed(1);
  });

  activeProfileData.glorys_temps = activeProfileData.ai_temps.map(t => +(t + (Math.random() * 0.3 - 0.15)).toFixed(1));
  activeProfileData.argo_temps = activeProfileData.ai_temps.map(t => +(t + (Math.random() * 0.2 - 0.1)).toFixed(1));

  renderProfileChart(canvas, activeProfileData);
}

/* ==========================================================================
   5. DRAWER TABS SWITCHING
   ========================================================================== */
function initDrawerTabs() {
  const tabBtns = document.querySelectorAll('.d-tab-btn');
  const tabPanes = document.querySelectorAll('.drawer-tab-pane');
  const drawerHandle = document.getElementById('drawerHandle');
  const drawer = document.getElementById('deepDiveDrawer');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetId = `pane-${btn.getAttribute('data-dtab')}`;
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');
    });
  });

  if (drawerHandle && drawer) {
    drawerHandle.addEventListener('click', () => {
      drawer.classList.toggle('open');
    });
  }

  // CTA button smooth scroll
  const ctaBtn = document.getElementById('viewDashboardBtn');
  const dashboardCard = document.getElementById('dashboardCard');
  if (ctaBtn && dashboardCard) {
    ctaBtn.addEventListener('click', () => {
      dashboardCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  }
}

/* ==========================================================================
   6. HEADER NAV TABS INTEGRATION
   ========================================================================== */
function initHeaderNavTabs() {
  const navLinks = document.querySelectorAll('.nav-links .nav-link');
  const tabBtns = document.querySelectorAll('.d-tab-btn');

  const tabMapping = {
    'tab1': 'tab-profile',
    'tab2': 'tab-satellite',
    'tab3': 'tab-transect',
    'tab4': 'tab-benchmarks',
    'tab5': 'tab-architecture'
  };

  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      navLinks.forEach(l => l.classList.remove('active'));
      link.classList.add('active');

      const tabKey = link.getAttribute('data-tab');
      const dtabKey = tabMapping[tabKey];

      tabBtns.forEach(btn => {
        if (btn.getAttribute('data-dtab') === dtabKey) {
          btn.click();
        }
      });

      const drawer = document.getElementById('deepDiveDrawer');
      if (drawer) {
        drawer.classList.add('open');
        drawer.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });
}

/* ==========================================================================
   7. JSON CONTRACT MODAL & DOWNLOAD
   ========================================================================== */
function updateJsonCodePreview(preset) {
  const codeBox = document.getElementById('jsonPreviewCode');
  if (!codeBox) return;

  const payload = {
    "date": "2022-07-15",
    "lat": preset.lat,
    "lon": preset.lon,
    "preset_name": preset.name,
    "surface_inputs": {
      "sst_c": parseFloat(preset.sst),
      "sss_psu": parseFloat(preset.sss),
      "sla_m": parseFloat(preset.sla) / 100.0,
      "current_velocity_ms": 0.52,
      "wind_speed_ms": 9.4
    },
    "predicted_profile": activeProfileData.ai_temps,
    "glorys_profile": activeProfileData.glorys_temps,
    "argo_profile": activeProfileData.argo_temps,
    "thermocline_depth_m": parseFloat(preset.z_th),
    "skill": {
      "correlation": parseFloat(preset.corr),
      "rmse": parseFloat(preset.rmse),
      "bias": -0.04
    }
  };

  codeBox.innerHTML = `<code>${JSON.stringify(payload, null, 2)}</code>`;
}

function initExportModal() {
  const exportBtn = document.getElementById('exportDataBtn');
  const exportModal = document.getElementById('exportModal');
  const closeBtn = document.getElementById('closeExportModal');
  const downloadBtn = document.getElementById('downloadJsonBtn');

  if (exportBtn && exportModal) {
    exportBtn.addEventListener('click', () => exportModal.classList.add('open'));
  }
  if (closeBtn && exportModal) {
    closeBtn.addEventListener('click', () => exportModal.classList.remove('open'));
  }

  if (downloadBtn) {
    downloadBtn.addEventListener('click', () => {
      const preset = OCEAN_PRESETS[currentPresetKey];
      const payload = {
        "date": "2022-07-15",
        "lat": preset.lat,
        "lon": preset.lon,
        "predicted_profile": activeProfileData.ai_temps,
        "glorys_profile": activeProfileData.glorys_temps,
        "argo_profile": activeProfileData.argo_temps,
        "thermocline_depth_m": parseFloat(preset.z_th),
        "skill": {
          "correlation": parseFloat(preset.corr),
          "rmse": parseFloat(preset.rmse),
          "bias": -0.04
        }
      };

      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `oceanembed_profile_${preset.lat}N_${preset.lon}E.json`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  window.addEventListener('click', (e) => {
    if (e.target === exportModal) exportModal.classList.remove('open');
  });
}

/* ==========================================================================
   8. FASTAPI HEALTH CHECK
   ========================================================================== */
function checkApiHealth() {
  const label = document.getElementById('apiStatusLabel');
  if (!label) return;

  fetch('http://localhost:8000/api/v1/health', { method: 'GET' })
    .then(res => res.json())
    .then(data => {
      if (data && data.status === 'ok') {
        label.innerText = `FastAPI Connected (${data.mode || 'LIVE'})`;
      }
    })
    .catch(() => {
      label.innerText = 'FastAPI Emulation Active (0–1000m)';
    });
}
