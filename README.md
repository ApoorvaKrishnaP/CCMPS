# CCMPS — Crowd Congestion Management & Prediction System

**What is this?**  
A real-time dashboard that monitors crowd density across venue zones, predicts congestion 2–10 minutes ahead, simulates emergency scenarios, and sends alerts when crowds become unsafe.

**Why use it?**
- Prevent dangerous crowd crushes at events (concerts, stadiums, airports)
- Simulate "what-if" scenarios before events happen
- See exactly where crowds are building up on a live heatmap
- Get recommended actions when risk levels spike

---

## Features

- **Live Dashboard** — Real-time heatmap showing crowd density in each zone, color-coded by risk (Safe → Critical)
- **Zone Status Table** — Per-zone metrics: density, speed, flow direction, inflow/outflow, congestion score, risk level
- **Active Alerts Panel** — Immediate notifications for high-risk & critical zones with recommended actions
- **Prediction Engine** — 2–10 min risk forecasting; see which zones will become dangerous
- **Digital Twin Simulation** — Test 3 scenarios: gate closure, sudden inflow surge, emergency evacuation
- **Post-Event Analysis** — Historical charts showing peak congestion zones, crowd flow efficiency, event timeline
- **Decision Support** — Automated action recommendations (e.g., "Redirect crowd flow", "Open emergency exit")

---

## Quick Start (60 seconds)

1. **Open terminal** in this folder
2. **Run:**
   ```bash
   npm install
   npm start
   ```
3. **Open browser** → http://localhost:3000
4. **Watch the dashboard** update every 5 seconds

That's it! You're monitoring a simulated 8-zone venue with real-time crowd data.

---

## How the Congestion Score is Calculated

Each zone gets a score (0–100) based on **two metrics**:

| Factor | Weight | What it means |
|--------|--------|---------------|
| **Crowd Density** | 60% | How packed the zone is (0–9 people/m²) |
| **Speed Penalty** | 40% | How slowly people are moving (slower = more dangerous) |

**Formula:**  
```
Congestion = MIN(100, (density/9 × 100) + (1 − speed/2) × 20)
```

**Example:**
- Zone with 6 people/m² and average speed 1.0 m/s
- Density component: (6/9) × 100 = 67
- Speed penalty: (1 − 1/2) × 20 = 10
- **Total = 77** → "High" risk

---

## What Each Dashboard Page Does

| Page | Purpose | When to use |
|------|---------|-------------|
| **Dashboard** | See live zone status & active alerts | Normal operations, real-time monitoring |
| **Predictions** | View 2–10 min risk forecasts | Planning staff actions ahead of time |
| **Digital Twin** | Simulate emergency scenarios | Before events, to test response plans |
| **Alerts** | Detailed list of all active alerts with recommended actions | When incidents happen, quick reference |
| **Post-Event** | Charts & timeline of what happened during the event | After event, review performance |

### Prerequisites
- [Node.js](https://nodejs.org/) (v16 or higher)

### Installation & Run

```bash
# 1. Install dependencies (one time only)
npm install

# 2. Start the server (will run on port 3000)
npm start

# 3. Open browser
# Visit: http://localhost:3000
```

You should see the CCMPS dashboard load with:
- 5 stat cards at the top (zones, critical count, people flow, density)
- Live heatmap on the left
- Active alerts panel on the right
- Zone status table below

**Note:** All data is simulated. Zones generate random but realistic crowd metrics every time you refresh.

---

## Project Structure

```
CCMPS/
├── server.js                    # REST API backend (Node.js/Express)
│                                 # Generates simulated zone data & runs scenarios
├── package.json                 # Dependencies: express, path, body-parser
├── README.md                    # This file
├── crowd_digital_twin.py        # (Optional) Python-based crowd simulation
│                                 # Uses Social Force Model + HOG person detection
│                                 # Analyzes video or runs physics-based scenarios
└── public/
    ├── index.html               # Complete dashboard (HTML + CSS + JavaScript)
    │                            # 5 pages: Dashboard, Predictions, Digital Twin, Alerts, Post-Event
    └── index1.html              # Alternate version (backup)
```

---

## Files Explained

**server.js** — Backend REST API
- Simulates 8 venue zones with random density, speed, inflow, outflow
- Generates congestion scores using the formula above
- Provides `/api/zones`, `/api/predictions`, `/api/alerts`, `/api/stats`, `/api/simulate`
- Runs 3 "what-if" scenarios with modified zone parameters

**public/index.html** — Frontend Dashboard
- Single-page app: 5 tabs for different views
- Real-time heatmap canvas showing zone density
- Zone status table with all metrics
- Active alerts with color coding (Green=Safe, Yellow=Moderate, Orange=High, Red=Critical)
- Prediction charts
- Simulation result cards
- Post-event analysis

**crowd_digital_twin.py** — Advanced Python Simulator (Optional)
- Reads a crowd video (Crowd.mp4) and detects people using HOG (Histogram of Oriented Gradients)
- Displays split-zone view + heatmap + count graph
- Can run physics-based Social Force Model scenarios (gate closure, inflow surge, evacuation)
- More realistic than server.js; useful for research/validation

---

## Using the Digital Twin Simulator (Optional)

If you have a crowd video file (Crowd.mp4) or want to run physics-based simulations:

```bash
# Run the Python digital twin
python crowd_digital_twin.py

# Or specify a video file
python crowd_digital_twin.py --video path/to/your/video.mp4
```

**Controls in the simulator[via keyboard]:**
- `1` → Digital Twin mode (analyze video, detect people)
- `2` → Gate Closure scenario (simulated)
- `3` → Inflow Surge scenario (simulated)
- `4` → Emergency Evacuation scenario (simulated)
- `Q` → Quit



## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/zones` | GET | Live zone data with density, speed, risk |
| `/api/predictions` | GET | 2–10 min congestion forecasts |
| `/api/alerts` | GET | Active alerts and recommended actions |
| `/api/stats` | GET | Summary statistics |
| `/api/simulate` | POST | Run digital twin scenario |

### Simulation scenarios (POST /api/simulate)
```json
{ "scenario": "gate_closure" }
{ "scenario": "increased_inflow" }
{ "scenario": "evacuation" }
```
---

## Risk Levels Explained

Each zone gets a **Risk Level** based on its congestion score:

| Level | Score | Color | Meaning | Action |
|-------|-------|-------|---------|--------|
| **Safe** | 0–29 | 🟢 Green | All good | Continue normal operations |
| **Moderate** | 30–54 | 🟡 Yellow | Watch it | Brief alert to staff, standby |
| **High** | 55–74 | 🟠 Orange | Getting tight | Activate intervention (redirect, staff) |
| **Critical** | 75–100 | 🔴 Red | **DANGER** | Immediate action: pause entry, emergency exit |

**Example Alert:**
- Zone "Gate A" congestion score = 82 → **Critical**
- Alert says: "Critical congestion in Gate A. Pause entry at gate."
- Recommended action: "Redirect crowd flow" or "Open emergency exit"

---

## Data Flow

```
Real World (or Simulation)
        ↓
  server.js (generates zone data)
        ↓
  REST API endpoints (/api/zones, /api/predictions, etc.)
        ↓
 public/index.html (fetches every 5s)
        ↓
  Dashboard updates:
  • Heatmap redraws
  • Zone table updates
  • Alerts refreshed
  • Predictions recalculated
```

---


## Tech Stack

- **Backend:** Node.js (Express.js)
- **Frontend:** HTML5 Canvas + CSS3 + Vanilla JavaScript
- **Optional (Python):** OpenCV (HOG detection), NumPy (Social Force Model)
- **Data:** Simulated (can be replaced with real APIs)

---
