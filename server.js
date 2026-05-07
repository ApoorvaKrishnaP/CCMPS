const express = require('express');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { spawn } = require('child_process');
const multer = require('multer');

const app = express();
const PORT = 3001;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ── UPLOADS DIRECTORY ─────────────────────────────────────────────────────────
const UPLOADS_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename: (req, file, cb) => cb(null, `zone_feed_${Date.now()}${path.extname(file.originalname)}`),
});
const upload = multer({ storage, limits: { fileSize: 500 * 1024 * 1024 } }); // 500 MB max

// ── PIPELINE PROCESS STATE ────────────────────────────────────────────────────
let pipelineProcess = null;   // child_process handle
let activeZoneName = null;   // e.g. "North Wing"
let pipelineMetrics = null;   // latest metrics from pipeline_api
let pipelineStatus = 'idle'; // idle | loading | running | error | stopped

const PIPELINE_PORT = 5002;
const PIPELINE_API = `http://localhost:${PIPELINE_PORT}`;

// ── VID-TWIN PROCESS STATE ───────────────────────────────────────────────────
let vidTwinProcess = null;   // child_process for vid_twin_api.py
let vidTwinStatus = 'idle'; // idle | loading | running | stopped | error
let vidTwinVideo = null;   // path to uploaded video

const VID_TWIN_PORT = 5004;
const VID_TWIN_API = `http://localhost:${VID_TWIN_PORT}`;

/** HTTP GET helper to reach pipeline_api.py */
function proxyGet(urlPath) {
  return new Promise((resolve, reject) => {
    const req = http.get(`${PIPELINE_API}${urlPath}`, (res) => {
      let raw = '';
      res.on('data', chunk => (raw += chunk));
      res.on('end', () => {
        try { resolve(JSON.parse(raw)); }
        catch (e) { reject(new Error('invalid JSON')); }
      });
    });
    req.setTimeout(2000, () => { req.destroy(); reject(new Error('timeout')); });
    req.on('error', reject);
  });
}

/** Kill any running pipeline process */
function killPipeline() {
  if (pipelineProcess) {
    try { pipelineProcess.kill('SIGTERM'); } catch (_) { }
    pipelineProcess = null;
  }
  pipelineStatus = 'idle';
}

/** Poll pipeline metrics every 2 s when a pipeline is active */
function startPollingPipeline() {
  const interval = setInterval(async () => {
    if (!pipelineProcess) { clearInterval(interval); return; }
    try {
      const data = await proxyGet('/metrics');
      pipelineMetrics = data;
      pipelineStatus = data.status || 'running';
    } catch (_) {
      // pipeline still loading — keep waiting
    }
  }, 2000);
}

// ── MOCK DATA GENERATOR ───────────────────────────────────────────────────────
const ZONE_META = [
  { name: 'Gate A', x: 10, y: 20 },
  { name: 'Gate B', x: 40, y: 20 },
  { name: 'Main Hall', x: 50, y: 50 },
  { name: 'North Wing', x: 20, y: 50 },
  { name: 'South Wing', x: 70, y: 50 },
  { name: 'Exit 1', x: 15, y: 80 },
  { name: 'Exit 2', x: 80, y: 80 },
  { name: 'Concourse', x: 50, y: 20 },
];

function computeRisk(score) {
  if (score < 30) return 'Safe';
  if (score < 55) return 'Moderate';
  if (score < 75) return 'High';
  return 'Critical';
}

function generateZoneData() {
  return ZONE_META.map((meta, i) => {
    // ── Check if this zone has a live pipeline feed ────────────────────────
    if (activeZoneName === meta.name && pipelineMetrics &&
      (pipelineStatus === 'running' || pipelineStatus === 'loading')) {

      const m = pipelineMetrics;
      const score = m.congestion_score || 0;
      return {
        zone_id: `Z${String(i + 1).padStart(2, '0')}`,
        name: meta.name,
        density: m.density || 0,
        avg_speed: m.avg_speed || 0,
        flow_direction: 'Live',
        inflow_rate: Math.round(m.inflow || 0),
        outflow_rate: Math.round(m.outflow || 0),
        congestion_score: score,
        risk: m.risk || computeRisk(score),
        people_count: m.people_count || 0,
        x: meta.x,
        y: meta.y,
        live: true,              // flag for dashboard to highlight
        pipeline_status: pipelineStatus,
      };
    }

    // ── Mock data for all other zones ─────────────────────────────────────
    const density = parseFloat((Math.random() * 8 + 0.5).toFixed(2));
    const avg_speed = parseFloat((Math.random() * 1.8 + 0.1).toFixed(2));
    const inflow = Math.floor(Math.random() * 120);
    const outflow = Math.floor(Math.random() * 100);
    const netFlow = inflow - outflow;
    const flowPres = Math.max(0, netFlow / 150) * 25;
    const spdPen = (1 - avg_speed / 2) * 20;
    const densComp = (density / 9) * 100;
    const score = Math.min(100, Math.floor(densComp + spdPen + flowPres + 10));
    const risk = computeRisk(score);

    return {
      zone_id: `Z${String(i + 1).padStart(2, '0')}`,
      name: meta.name,
      density,
      avg_speed,
      flow_direction: 'Mixed',
      inflow_rate: inflow,
      outflow_rate: outflow,
      congestion_score: score,
      risk,
      x: meta.x,
      y: meta.y,
      live: false,
    };
  });
}

function generatePredictions(zones) {
  return zones.map(z => {
    const delta = (Math.random() - 0.4) * 15;
    const future = Math.min(100, Math.max(0, z.congestion_score + delta));
    return {
      zone_id: z.zone_id,
      name: z.name,
      current_score: z.congestion_score,
      predicted_score: Math.floor(future),
      predicted_risk: computeRisk(Math.floor(future)),
      forecast_minutes: Math.floor(Math.random() * 8) + 2,
      live: z.live || false,
    };
  });
}

function generateAlerts(zones) {
  const actions = [
    'Redirect crowd flow', 'Open emergency exit',
    'Pause entry at gate', 'Deploy security personnel',
    'Announce crowd dispersal',
  ];
  const alerts = [];
  zones.forEach(z => {
    if (z.risk === 'Critical') {
      alerts.push({
        id: z.zone_id, zone: z.name, level: 'Critical',
        message: `Critical congestion in ${z.name}. Immediate action required.${z.live ? ' [LIVE DATA]' : ''}`,
        action: actions[Math.floor(Math.random() * actions.length)],
        time: new Date().toLocaleTimeString(),
      });
    } else if (z.risk === 'High') {
      alerts.push({
        id: z.zone_id, zone: z.name, level: 'High',
        message: `High risk detected in ${z.name}. Monitor closely.${z.live ? ' [LIVE DATA]' : ''}`,
        action: actions[Math.floor(Math.random() * actions.length)],
        time: new Date().toLocaleTimeString(),
      });
    }
  });
  return alerts;
}

// ── API ROUTES ────────────────────────────────────────────────────────────────

// Existing routes
app.get('/api/zones', (req, res) => {
  const zones = generateZoneData();
  res.json({ zones, timestamp: new Date().toISOString() });
});

app.get('/api/predictions', (req, res) => {
  const zones = generateZoneData();
  res.json({ predictions: generatePredictions(zones), timestamp: new Date().toISOString() });
});

app.get('/api/alerts', (req, res) => {
  const zones = generateZoneData();
  res.json({ alerts: generateAlerts(zones), timestamp: new Date().toISOString() });
});

app.get('/api/stats', (req, res) => {
  const zones = generateZoneData();
  const total = zones.length;
  const critical = zones.filter(z => z.risk === 'Critical').length;
  const high = zones.filter(z => z.risk === 'High').length;
  const totalFlow = zones.reduce((s, z) => s + z.inflow_rate, 0);
  const avgDensity = (zones.reduce((s, z) => s + z.density, 0) / total).toFixed(2);
  res.json({
    total_zones: total, Critical_zones: critical, high_risk_zones: high,
    total_people_flow: totalFlow, avg_density: avgDensity,
    system_status: 'Online', mode: pipelineStatus === 'running' ? 'Live+Sim' : 'Simulated',
    timestamp: new Date().toISOString(),
  });
});

app.post('/api/simulate', (req, res) => {
  const { scenario } = req.body;
  const zones = generateZoneData();
  let modified = zones.map(z => ({ ...z }));
  if (scenario === 'gate_closure') {
    modified = modified.map(z => z.name.includes('Gate')
      ? { ...z, density: Math.min(9, z.density * 1.8), congestion_score: Math.min(100, z.congestion_score + 30), risk: 'Critical' }
      : z);
  } else if (scenario === 'increased_inflow') {
    modified = modified.map(z => ({ ...z, inflow_rate: z.inflow_rate * 2, density: Math.min(9, z.density * 1.4), congestion_score: Math.min(100, z.congestion_score + 20) }));
  } else if (scenario === 'evacuation') {
    modified = modified.map(z => z.name.includes('Exit')
      ? { ...z, outflow_rate: z.outflow_rate * 3, density: Math.max(0.1, z.density * 0.5), congestion_score: Math.max(0, z.congestion_score - 20), risk: 'Safe' }
      : { ...z, density: Math.min(9, z.density * 1.6), congestion_score: Math.min(100, z.congestion_score + 25) });
  }
  res.json({ scenario, zones: modified, message: `Simulation for "${scenario}" complete.`, timestamp: new Date().toISOString() });
});

// ── NEW: Video Upload ─────────────────────────────────────────────────────────
app.post('/api/upload', upload.single('video'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No video file received.' });

  const { zone, width, height } = req.body;
  if (!zone) return res.status(400).json({ error: 'zone is required.' });

  const videoPath = req.file.path;
  const widthM = parseFloat(width) || 20.0;
  const heightM = parseFloat(height) || 12.0;

  // Kill any running pipeline
  killPipeline();
  activeZoneName = zone;
  pipelineMetrics = null;
  pipelineStatus = 'loading';

  // Spawn pipeline_api.py
  const py = process.platform === 'win32' ? 'python' : 'python3';
  pipelineProcess = spawn(py, [
    'crowd_analytics/pipeline_api.py',
    '--video', videoPath,
    '--zone', zone,
    '--width', String(widthM),
    '--height', String(heightM),
    '--port', String(PIPELINE_PORT),
  ], { cwd: __dirname });

  pipelineProcess.stdout.on('data', d => process.stdout.write(`[PY] ${d}`));
  pipelineProcess.stderr.on('data', d => process.stderr.write(`[PY-ERR] ${d}`));
  pipelineProcess.on('exit', code => {
    console.log(`[PY] pipeline_api exited with code ${code}`);
    pipelineProcess = null;
    pipelineStatus = 'stopped';
  });

  startPollingPipeline();

  res.json({
    message: `Pipeline started for zone "${zone}"`,
    zone,
    video: req.file.filename,
    width_m: widthM,
    height_m: heightM,
    status: 'loading',
  });
});

// ── NEW: Pipeline Status ───────────────────────────────────────────────────────
app.get('/api/pipeline-status', (req, res) => {
  if (!activeZoneName) {
    return res.json({ status: 'idle', zone: null, metrics: null });
  }
  res.json({
    status: pipelineStatus,
    zone: activeZoneName,
    metrics: pipelineMetrics,
  });
});

// ── NEW: Stop Pipeline ─────────────────────────────────────────────────────────
app.post('/api/pipeline-stop', (req, res) => {
  killPipeline();
  activeZoneName = null;
  pipelineMetrics = null;
  res.json({ message: 'Pipeline stopped.' });
});

// ── VID-TWIN: Kill helper ────────────────────────────────────────────────────
function killVidTwin() {
  if (vidTwinProcess) {
    try { vidTwinProcess.kill('SIGTERM'); } catch (_) { }
    vidTwinProcess = null;
  }
  vidTwinStatus = 'idle';
}

// ── VID-TWIN: Upload & Start ─────────────────────────────────────────────────
app.post('/api/vid-twin/upload', upload.single('video'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No video file received.' });

  const videoPath = req.file.path;
  killVidTwin();
  vidTwinVideo = videoPath;
  vidTwinStatus = 'loading';

  const py = process.platform === 'win32' ? 'python' : 'python3';
  vidTwinProcess = spawn(py, [
    'vid_twin_api.py',
    '--video', videoPath,
    '--port', String(VID_TWIN_PORT),
  ], { cwd: __dirname });

  vidTwinProcess.stdout.on('data', d => process.stdout.write(`[VID-TWIN] ${d}`));
  vidTwinProcess.stderr.on('data', d => process.stderr.write(`[VID-TWIN-ERR] ${d}`));
  vidTwinProcess.on('exit', code => {
    console.log(`[VID-TWIN] exited with code ${code}`);
    vidTwinProcess = null;
    vidTwinStatus = 'stopped';
  });

  // Poll status until running
  const poll = setInterval(async () => {
    try {
      const data = await proxyGet2('/status');
      vidTwinStatus = data.status || 'running';
      if (vidTwinStatus === 'running') clearInterval(poll);
    } catch (_) { }
  }, 2000);

  res.json({
    message: 'vid_twin_api started',
    video: req.file.filename,
    port: VID_TWIN_PORT,
    status: 'loading',
  });
});

/** HTTP GET helper for vid_twin_api */
function proxyGet2(urlPath) {
  return new Promise((resolve, reject) => {
    const req = http.get(`${VID_TWIN_API}${urlPath}`, (res) => {
      let raw = '';
      res.on('data', chunk => (raw += chunk));
      res.on('end', () => {
        try { resolve(JSON.parse(raw)); }
        catch (e) { reject(new Error('invalid JSON')); }
      });
    });
    req.setTimeout(2000, () => { req.destroy(); reject(new Error('timeout')); });
    req.on('error', reject);
  });
}

// ── VID-TWIN: Status ─────────────────────────────────────────────────────────
app.get('/api/vid-twin/status', async (req, res) => {
  if (!vidTwinProcess && vidTwinStatus === 'idle') {
    return res.json({ status: 'idle' });
  }
  try {
    const data = await proxyGet2('/status');
    vidTwinStatus = data.status;
    return res.json(data);
  } catch (_) {
    return res.json({ status: vidTwinStatus });
  }
});

// ── VID-TWIN: Stop ───────────────────────────────────────────────────────────
app.post('/api/vid-twin/stop', (req, res) => {
  // Send POST to vid_twin_api /stop then kill the process
  try {
    const stopReq = http.request({
      hostname: 'localhost', port: VID_TWIN_PORT,
      path: '/stop', method: 'POST',
    });
    stopReq.on('error', () => { });
    stopReq.end();
  } catch (_) { }
  setTimeout(() => killVidTwin(), 500);
  vidTwinVideo = null;
  res.json({ message: 'vid_twin stopped' });
});

// ── VID-TWIN: Stream proxy (so browser avoids CORS) ─────────────────────────
['video', 'zone', 'dot', 'graph'].forEach(kind => {
  app.get(`/api/vid-twin/stream/${kind}`, (req, res) => {
    if (!vidTwinProcess) {
      return res.status(503).send('vid_twin not running');
    }
    const upstream = http.get(`${VID_TWIN_API}/stream/${kind}`, upRes => {
      res.writeHead(upRes.statusCode, upRes.headers);
      upRes.pipe(res);
    });
    upstream.on('error', () => res.status(503).send('stream unavailable'));
    req.on('close', () => upstream.destroy());
  });
});

// ── SPA fallback ──────────────────────────────────────────────────────────────
app.get('/{*path}', (req, res) =>
  res.sendFile(path.join(__dirname, 'public', 'index.html'))
);

app.listen(PORT, () => console.log(`CCMPS running at http://localhost:${PORT}`));
