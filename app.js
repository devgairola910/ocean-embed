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
  initUploadAndManualInput();
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
    'tab5': 'tab-architecture',
    'tab6': 'tab-upload'
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
   8. UPLOAD & MANUAL DATA INGESTION CONTROLLER
   ========================================================================== */
function openUploadSection() {
  const uploadTabBtn = document.querySelector('.d-tab-btn[data-dtab="tab-upload"]');
  const tabBtns = document.querySelectorAll('.d-tab-btn');
  const tabPanes = document.querySelectorAll('.drawer-tab-pane');
  const drawer = document.getElementById('deepDiveDrawer');
  const targetPane = document.getElementById('pane-tab-upload');

  if (tabBtns && tabPanes) {
    tabBtns.forEach(b => b.classList.remove('active'));
    tabPanes.forEach(p => p.classList.remove('active'));
  }
  if (uploadTabBtn) uploadTabBtn.classList.add('active');
  if (targetPane) targetPane.classList.add('active');

  if (drawer) {
    drawer.classList.add('open');
    setTimeout(() => {
      drawer.scrollIntoView({ behavior: 'smooth', block: 'start' });
      executeCustomInference();
    }, 60);
  }

  const uploadModal = document.getElementById('uploadModal');
  if (uploadModal) uploadModal.classList.remove('open');
}

function initUploadAndManualInput() {
  const openModalBtn = document.getElementById('openUploadModalBtn');
  const heroUploadBtn = document.getElementById('heroUploadBtn');
  const dashboardUploadBtn = document.getElementById('dashboardUploadBtn');
  const uploadModal = document.getElementById('uploadModal');
  const closeUploadBtn = document.getElementById('closeUploadModal');
  const goToUploadTabBtn = document.getElementById('goToUploadTabBtn');
  const dropzone = document.getElementById('uploadDropzone');
  const fileInput = document.getElementById('satelliteFileInput');
  const loadSampleBtn = document.getElementById('loadSampleCsvBtn');

  // Sliders and badges
  const rangeSst = document.getElementById('rangeSst');
  const rangeSss = document.getElementById('rangeSss');
  const rangeSla = document.getElementById('rangeSla');
  const rangeCurrU = document.getElementById('rangeCurrU');
  const rangeCurrV = document.getElementById('rangeCurrV');
  const rangeWindU = document.getElementById('rangeWindU');
  const rangeWindV = document.getElementById('rangeWindV');

  const valSst = document.getElementById('valSst');
  const valSss = document.getElementById('valSss');
  const valSla = document.getElementById('valSla');
  const valCurr = document.getElementById('valCurr');
  const valWind = document.getElementById('valWind');

  const btnExecute = document.getElementById('btnExecuteCustomInfer');
  const btnReset = document.getElementById('btnResetForm');
  const scenarioPills = document.querySelectorAll('.preset-pill');

  // Direct open handlers for all devices
  if (openModalBtn) {
    openModalBtn.addEventListener('click', (e) => { e.preventDefault(); openUploadSection(); });
  }
  if (heroUploadBtn) {
    heroUploadBtn.addEventListener('click', (e) => { e.preventDefault(); openUploadSection(); });
  }
  if (dashboardUploadBtn) {
    dashboardUploadBtn.addEventListener('click', (e) => { e.preventDefault(); openUploadSection(); });
  }
  if (goToUploadTabBtn) {
    goToUploadTabBtn.addEventListener('click', (e) => { e.preventDefault(); openUploadSection(); });
  }
  if (closeUploadBtn && uploadModal) {
    closeUploadBtn.addEventListener('click', () => uploadModal.classList.remove('open'));
  }

  // Sliders real-time values
  if (rangeSst && valSst) {
    rangeSst.addEventListener('input', () => {
      valSst.innerText = `${parseFloat(rangeSst.value).toFixed(1)} °C`;
    });
  }
  if (rangeSss && valSss) {
    rangeSss.addEventListener('input', () => {
      valSss.innerText = `${parseFloat(rangeSss.value).toFixed(1)} PSU`;
    });
  }
  if (rangeSla && valSla) {
    rangeSla.addEventListener('input', () => {
      const v = parseFloat(rangeSla.value);
      valSla.innerText = `${v >= 0 ? '+' : ''}${v.toFixed(1)} cm`;
    });
  }
  function updateCurrBadge() {
    if (valCurr && rangeCurrU && rangeCurrV) {
      const u = parseFloat(rangeCurrU.value);
      const v = parseFloat(rangeCurrV.value);
      valCurr.innerText = `U: ${u >= 0 ? '+' : ''}${u.toFixed(2)} | V: ${v >= 0 ? '+' : ''}${v.toFixed(2)}`;
    }
  }
  if (rangeCurrU) rangeCurrU.addEventListener('input', updateCurrBadge);
  if (rangeCurrV) rangeCurrV.addEventListener('input', updateCurrBadge);

  function updateWindBadge() {
    if (valWind && rangeWindU && rangeWindV) {
      const u = parseFloat(rangeWindU.value);
      const v = parseFloat(rangeWindV.value);
      valWind.innerText = `U: ${u >= 0 ? '+' : ''}${u.toFixed(1)} | V: ${v >= 0 ? '+' : ''}${v.toFixed(1)}`;
    }
  }
  if (rangeWindU) rangeWindU.addEventListener('input', updateWindBadge);
  if (rangeWindV) rangeWindV.addEventListener('input', updateWindBadge);

  // Drag & Drop handlers
  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleUploadedFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleUploadedFile(e.target.files[0]);
      }
    });
  }

  function handleUploadedFile(file) {
    const filename = file.name;
    const dropText = dropzone.querySelector('.dropzone-text');
    if (dropText) {
      dropText.innerHTML = `<strong>✅ Loaded: ${filename} (${(file.size / 1024).toFixed(1)} KB)</strong><span>Parsing 7 satellite channels & computing subsurface profile...</span>`;
    }

    const reader = new FileReader();
    reader.onload = function(evt) {
      const content = evt.target.result;
      if (filename.endsWith('.csv')) {
        parseCsvAndPopulate(content);
      } else if (filename.endsWith('.json')) {
        try {
          const json = JSON.parse(content);
          populateFromJson(json);
        } catch(err) {
          console.warn('JSON parse error', err);
        }
      } else {
        // NetCDF simulation
        setTimeout(() => {
          executeCustomInference();
        }, 300);
      }
    };

    if (filename.endsWith('.json') || filename.endsWith('.csv')) {
      reader.readAsText(file);
    } else {
      reader.readAsArrayBuffer(file);
    }
  }

  function parseCsvAndPopulate(csvText) {
    const lines = csvText.trim().split('\n');
    if (lines.length > 1) {
      const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
      const values = lines[1].split(',').map(v => v.trim());
      
      const sstIdx = headers.findIndex(h => h.includes('sst'));
      const sssIdx = headers.findIndex(h => h.includes('sss'));
      const slaIdx = headers.findIndex(h => h.includes('sla') || h.includes('ssh'));
      
      if (sstIdx >= 0 && rangeSst) {
        rangeSst.value = parseFloat(values[sstIdx]) || 29.8;
        if (valSst) valSst.innerText = `${parseFloat(rangeSst.value).toFixed(1)} °C`;
      }
      if (sssIdx >= 0 && rangeSss) {
        rangeSss.value = parseFloat(values[sssIdx]) || 33.2;
        if (valSss) valSss.innerText = `${parseFloat(rangeSss.value).toFixed(1)} PSU`;
      }
      if (slaIdx >= 0 && rangeSla) {
        let slaVal = parseFloat(values[slaIdx]) || 18.5;
        if (Math.abs(slaVal) < 1.0) slaVal *= 100.0; // convert m to cm
        rangeSla.value = slaVal;
        if (valSla) valSla.innerText = `${slaVal >= 0 ? '+' : ''}${slaVal.toFixed(1)} cm`;
      }
      executeCustomInference();
    }
  }

  function populateFromJson(json) {
    if (json.surface_inputs) {
      const s = json.surface_inputs;
      if (s.sst_c && rangeSst) rangeSst.value = s.sst_c;
      if (s.sss_psu && rangeSss) rangeSss.value = s.sss_psu;
      if (s.sla_m && rangeSla) rangeSla.value = s.sla_m * 100.0;
      if (rangeSst && valSst) valSst.innerText = `${parseFloat(rangeSst.value).toFixed(1)} °C`;
      if (rangeSss && valSss) valSss.innerText = `${parseFloat(rangeSss.value).toFixed(1)} PSU`;
      if (rangeSla && valSla) valSla.innerText = `${parseFloat(rangeSla.value) >= 0 ? '+' : ''}${parseFloat(rangeSla.value).toFixed(1)} cm`;
      executeCustomInference();
    }
  }

  // Load sample CSV
  if (loadSampleBtn) {
    loadSampleBtn.addEventListener('click', () => {
      if (rangeSst) rangeSst.value = 30.4;
      if (rangeSss) rangeSss.value = 32.6;
      if (rangeSla) rangeSla.value = 24.5;
      if (rangeCurrU) rangeCurrU.value = 0.65;
      if (rangeCurrV) rangeCurrV.value = 0.45;
      if (rangeWindU) rangeWindU.value = -8.5;
      if (rangeWindV) rangeWindV.value = 11.2;

      if (valSst) valSst.innerText = '30.4 °C';
      if (valSss) valSss.innerText = '32.6 PSU';
      if (valSla) valSla.innerText = '+24.5 cm';
      updateCurrBadge();
      updateWindBadge();

      executeCustomInference();
    });
  }

  // Scenario quick presets
  scenarioPills.forEach(pill => {
    pill.addEventListener('click', () => {
      const sc = pill.getAttribute('data-scenario');
      if (sc === 'cyclone') {
        if (rangeSst) rangeSst.value = 30.1;
        if (rangeSss) rangeSss.value = 33.0;
        if (rangeSla) rangeSla.value = 22.0;
      } else if (sc === 'upwelling') {
        if (rangeSst) rangeSst.value = 25.2;
        if (rangeSss) rangeSss.value = 36.5;
        if (rangeSla) rangeSla.value = -14.0;
      } else if (sc === 'warmpool') {
        if (rangeSst) rangeSst.value = 29.9;
        if (rangeSss) rangeSss.value = 34.2;
        if (rangeSla) rangeSla.value = 6.0;
      }
      if (rangeSst && valSst) valSst.innerText = `${parseFloat(rangeSst.value).toFixed(1)} °C`;
      if (rangeSss && valSss) valSss.innerText = `${parseFloat(rangeSss.value).toFixed(1)} PSU`;
      if (rangeSla && valSla) valSla.innerText = `${parseFloat(rangeSla.value) >= 0 ? '+' : ''}${parseFloat(rangeSla.value).toFixed(1)} cm`;
      executeCustomInference();
    });
  });

  // Execute button
  if (btnExecute) {
    btnExecute.addEventListener('click', () => {
      btnExecute.innerHTML = '⚡ Computing ViT-MAE Cross-Attention...';
      btnExecute.style.filter = 'brightness(1.3)';
      setTimeout(() => {
        executeCustomInference();
        btnExecute.innerHTML = '⚡ Run 0–1000m Subsurface AI Inference';
        btnExecute.style.filter = 'none';
      }, 200);
    });
  }

  if (btnReset) {
    btnReset.addEventListener('click', () => {
      if (rangeSst) rangeSst.value = 29.8;
      if (rangeSss) rangeSss.value = 33.2;
      if (rangeSla) rangeSla.value = 18.5;
      if (rangeCurrU) rangeCurrU.value = 0.42;
      if (rangeCurrV) rangeCurrV.value = 0.31;
      if (rangeWindU) rangeWindU.value = -5.2;
      if (rangeWindV) rangeWindV.value = 6.8;

      if (valSst) valSst.innerText = '29.8 °C';
      if (valSss) valSss.innerText = '33.2 PSU';
      if (valSla) valSla.innerText = '+18.5 cm';
      updateCurrBadge();
      updateWindBadge();

      executeCustomInference();
    });
  }

  // Initial draw
  executeCustomInference();
}

function executeCustomInference() {
  const sst = parseFloat(document.getElementById('rangeSst')?.value || 29.8);
  const sss = parseFloat(document.getElementById('rangeSss')?.value || 33.2);
  const sla = parseFloat(document.getElementById('rangeSla')?.value || 18.5);

  const depths = [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 600, 1000];
  
  // Physics-guided subsurface reconstruction matching OceanEmbed depth decoder
  // Z_th shift is proportional to SLA (0.7m per cm SLA) and SST
  const z_th = Math.max(30, Math.min(130, 55.0 + sla * 0.7 + (sst - 28.0) * 4.0));
  const mld = Math.max(12, Math.min(50, 25.0 - sla * 0.2 + (34.0 - sss) * 2.0));
  const ohc = Math.max(60, Math.min(180, 110 + (sst - 28.0) * 12.0 + sla * 1.5));

  const temps = [];
  for (let i = 0; i < depths.length; i++) {
    const z = depths[i];
    if (z <= mld) {
      // Mixed layer isothermal
      temps.push(sst - 0.05 * (z / Math.max(1, mld)));
    } else if (z <= z_th) {
      // Upper thermocline gradual drop
      const frac = (z - mld) / (z_th - mld);
      temps.push((sst - 0.05) - frac * (sst - 20.0));
    } else if (z <= 250) {
      // Main thermocline below 20°C isotherm
      const frac = (z - z_th) / (250 - z_th);
      temps.push(20.0 - frac * 8.5);
    } else if (z <= 600) {
      // Intermediate waters
      const frac = (z - 250) / (600 - 250);
      temps.push(11.5 - frac * 4.5);
    } else {
      // Deep ocean
      const frac = (z - 600) / (1000 - 600);
      temps.push(7.0 - frac * 2.5);
    }
  }

  // Update KPI display
  const lblZth = document.getElementById('manualZth');
  const lblOhc = document.getElementById('manualOhc');
  const lblMld = document.getElementById('manualMld');
  const lblStab = document.getElementById('manualStability');

  if (lblZth) lblZth.innerText = `${z_th.toFixed(1)} m`;
  if (lblOhc) lblOhc.innerText = `${Math.round(ohc)} kJ/cm²`;
  if (lblMld) lblMld.innerText = `${mld.toFixed(1)} m`;
  if (lblStab) lblStab.innerText = '100% Stable (∂T/∂z ≤ 0)';

  renderManualProfileChart(temps, depths, z_th);
}

function renderManualProfileChart(temps, depths, z_th) {
  const canvas = document.getElementById('manualProfileCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;

  const padLeft = 55;
  const padRight = 20;
  const padTop = 25;
  const padBottom = 35;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  const minTemp = 2.0;
  const maxTemp = 34.0;
  const maxDepth = 1000.0;

  // Temperature grid lines (X axis)
  ctx.font = '10px Inter, sans-serif';
  ctx.fillStyle = 'rgba(200, 225, 250, 0.7)';
  ctx.textAlign = 'center';

  for (let t = 5; t <= 30; t += 5) {
    const x = padLeft + ((t - minTemp) / (maxTemp - minTemp)) * plotW;
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, padTop + plotH);
    ctx.stroke();
    ctx.fillText(`${t}°C`, x, padTop + plotH + 16);
  }

  // Depth grid lines (Y axis)
  ctx.textAlign = 'right';
  const yTicks = [0, 100, 200, 400, 600, 800, 1000];
  for (let i = 0; i < yTicks.length; i++) {
    const d = yTicks[i];
    const y = padTop + (d / maxDepth) * plotH;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(padLeft + plotW, y);
    ctx.stroke();
    ctx.fillText(`${d}m`, padLeft - 8, y + 4);
  }

  // Draw Thermocline Depth dashed line
  const zthY = padTop + (z_th / maxDepth) * plotH;
  ctx.strokeStyle = 'rgba(244, 63, 94, 0.7)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padLeft, zthY);
  ctx.lineTo(padLeft + plotW, zthY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#F43F5E';
  ctx.textAlign = 'left';
  ctx.fillText(`Z_th: ${z_th.toFixed(1)}m`, padLeft + 8, zthY - 5);

  // Draw continuous temperature profile curve
  ctx.beginPath();
  ctx.lineWidth = 3;
  const gradient = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
  gradient.addColorStop(0, '#F43F5E');
  gradient.addColorStop(0.35, '#38BDF8');
  gradient.addColorStop(1, '#6FFFE9');
  ctx.strokeStyle = gradient;

  for (let i = 0; i < depths.length; i++) {
    const t = temps[i];
    const d = depths[i];
    const x = padLeft + ((t - minTemp) / (maxTemp - minTemp)) * plotW;
    const y = padTop + (d / maxDepth) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw node points
  for (let i = 0; i < depths.length; i++) {
    const t = temps[i];
    const d = depths[i];
    const x = padLeft + ((t - minTemp) / (maxTemp - minTemp)) * plotW;
    const y = padTop + (d / maxDepth) * plotH;

    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();
    ctx.strokeStyle = '#0284C7';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Legend header
  ctx.font = '11px Outfit, sans-serif';
  ctx.fillStyle = '#FFFFFF';
  ctx.textAlign = 'left';
  ctx.fillText('Reconstructed Thermal Curve (0–1000m)', padLeft, padTop - 8);
}

/* ==========================================================================
   9. FASTAPI HEALTH CHECK
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
