const express = require('express');
const path = require('path');
const app = express();
const PORT = 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Simulated zone data generator
function generateZoneData() {
  const zones = ['Gate A', 'Gate B', 'Main Hall', 'North Wing', 'South Wing', 'Exit 1', 'Exit 2', 'Concourse'];
  return zones.map((name, i) => {
    const density = parseFloat((Math.random() * 8 + 0.5).toFixed(2));
    const avg_speed = parseFloat((Math.random() * 1.8 + 0.1).toFixed(2));
    const inflow = Math.floor(Math.random() * 120);
    const outflow = Math.floor(Math.random() * 100);
    const directions ='Mixed'; // Simplified for demo, could be 'Inflow', 'Outflow', or 'Mixed'
    // Net flow rate: positive = people accumulating, negative = clearing out
const netFlow = inflow - outflow;  

// Flow pressure: if more in than out, congestion will worsen
const flowPressure = Math.max(0, netFlow / 150) * 25;  

// Speed penalty (same as before)
const speedPenalty = (1 - avg_speed / 2) * 20;

// Density component (same as before)
const densityComponent = (density / 9) * 100;

// Direction conflict penalty (simplified: if 'Mixed', add penalty)
const directionPenalty = directions === 'Mixed' ? 10 : 0;

// Combined congestion score
const congestion_score = Math.min(100, Math.floor(
  densityComponent + speedPenalty + flowPressure + directionPenalty
));
    let risk;
    if (congestion_score < 30) risk = 'Safe';
    else if (congestion_score < 55) risk = 'Moderate';
    else if (congestion_score < 75) risk = 'High';
    else risk = 'Critical';
    return {
      zone_id: `Z${String(i + 1).padStart(2, '0')}`,
      name,
      density,
      avg_speed,
      flow_direction: directions[Math.floor(Math.random() * directions.length)],
      inflow_rate: inflow,
      outflow_rate: outflow,
      congestion_score,
      risk,
      x: [10, 40, 50, 20, 70, 15, 80, 50][i],
      y: [20, 20, 50, 50, 50, 80, 80, 20][i]
    };
  });
}

function generatePredictions(zones) {
  return zones.map(z => {
    const delta = (Math.random() - 0.4) * 15;
    const future_score = Math.min(100, Math.max(0, z.congestion_score + delta));
    let future_risk;
    if (future_score < 30) future_risk = 'Safe';
    else if (future_score < 55) future_risk = 'Moderate';
    else if (future_score < 75) future_risk = 'High';
    else future_risk = 'Critical';
    return {
      zone_id: z.zone_id,
      name: z.name,
      current_score: z.congestion_score,
      predicted_score: Math.floor(future_score),
      predicted_risk: future_risk,
      forecast_minutes: Math.floor(Math.random() * 8) + 2
    };
  });
}

function generateAlerts(zones) {
  const alerts = [];
  const actions = ['Redirect crowd flow', 'Open emergency exit', 'Pause entry at gate', 'Deploy security personnel', 'Announce crowd dispersal'];
  zones.forEach(z => {
    if (z.risk === 'Critical') {
      alerts.push({ id: z.zone_id, zone: z.name, level: 'Critical', message: `Critical congestion in ${z.name}. Immediate action required.`, action: actions[Math.floor(Math.random() * actions.length)], time: new Date().toLocaleTimeString() });
    } else if (z.risk === 'High') {
      alerts.push({ id: z.zone_id, zone: z.name, level: 'High', message: `High risk detected in ${z.name}. Monitor closely.`, action: actions[Math.floor(Math.random() * actions.length)], time: new Date().toLocaleTimeString() });
    }
  });
  return alerts;
}

// API routes
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
  const Critical = zones.filter(z => z.risk === 'Critical').length;
  const high = zones.filter(z => z.risk === 'High').length;
  const totalPeople = zones.reduce((s, z) => s + z.inflow_rate, 0);
  const avgDensity = (zones.reduce((s, z) => s + z.density, 0) / total).toFixed(2);
  res.json({ total_zones: total, Critical_zones: Critical, high_risk_zones: high, total_people_flow: totalPeople, avg_density: avgDensity, system_status: 'Online', mode: 'Real-Time', timestamp: new Date().toISOString() });
});

app.post('/api/simulate', (req, res) => {
  const { scenario } = req.body;
  const zones = generateZoneData();
  let modified = zones.map(z => ({ ...z }));
  if (scenario === 'gate_closure') {
    modified = modified.map(z => z.name.includes('Gate') ? { ...z, density: Math.min(9, z.density * 1.8), congestion_score: Math.min(100, z.congestion_score + 30), risk: 'Critical' } : z);
  } else if (scenario === 'increased_inflow') {
    modified = modified.map(z => ({ ...z, inflow_rate: z.inflow_rate * 2, density: Math.min(9, z.density * 1.4), congestion_score: Math.min(100, z.congestion_score + 20) }));
  } else if (scenario === 'evacuation') {
    modified = modified.map(z => z.name.includes('Exit') ? { ...z, outflow_rate: z.outflow_rate * 3, density: Math.max(0.1, z.density * 0.5), congestion_score: Math.max(0, z.congestion_score - 20), risk: 'Safe' } : { ...z, density: Math.min(9, z.density * 1.6), congestion_score: Math.min(100, z.congestion_score + 25) });
  }
  res.json({ scenario, zones: modified, message: `Simulation for "${scenario}" complete.`, timestamp: new Date().toISOString() });
});

app.get('/{*path}', (req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));

app.listen(PORT, () => console.log(`CCMPS running at http://localhost:${PORT}`));
