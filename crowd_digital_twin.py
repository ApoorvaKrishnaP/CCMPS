"""
Crowd Digital Twin + Social Force Model Simulator
===================================================
MODE 1 – Digital Twin  : Reads Crowd.mp4, detects people with HOG,
                         renders split-zone view + zone heatmap + dot grid + crowd graph.
MODE 2 – SFM Scenarios : Simulates Gate-Close, Inflow-Surge, Emergency-Evacuation
                         using the Social Force Model and renders them in the same layout.

Run:
    python crowd_digital_twin.py                     # picks up Crowd.mp4 from same folder
    python crowd_digital_twin.py --video path.mp4    # explicit path

Controls:
    [1] Digital Twin   [2] Gate Close   [3] Inflow Surge   [4] Emergency Evacuation
    [Q] Quit
"""

import sys, argparse, os, math, random, time
import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
DISPLAY_W, DISPLAY_H = 1280, 720
ZONE_ROWS, ZONE_COLS = 2, 3
LOW_TH, HIGH_TH = 8, 20
PROCESS_EVERY = 6          # HOG every N frames
GRAPH_LEN = 120

# SFM
SFM_DT      = 0.05         # seconds per step
SFM_MASS    = 80.0         # kg
SFM_TAU     = 0.5          # relaxation time
SFM_A       = 2000.0       # pedestrian repulsion magnitude
SFM_B       = 0.08         # repulsion range  (m)
SFM_AW      = 1500.0       # wall repulsion magnitude
SFM_BW      = 0.05         # wall repulsion range
SFM_DESIRE  = 1.4          # desired speed (m/s)
AGENT_RADIUS= 0.3          # metres
SCALE       = 30           # pixels per metre in SFM canvas

# ─────────────────────────────────────────────────────────────
#  HOG DETECTOR
# ─────────────────────────────────────────────────────────────
_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def hog_detect(frame):
    """Return list of (cx,cy,w,h) bounding boxes for detected people."""
    small = cv2.resize(frame, (640, 360))
    sx, sy = frame.shape[1] / 640, frame.shape[0] / 360
    boxes, _ = _hog.detectMultiScale(small, winStride=(8, 8), padding=(4, 4), scale=1.05)
    results = []
    if len(boxes):
        for (x, y, w, h) in boxes:
            results.append((int((x + w / 2) * sx),
                            int((y + h / 2) * sy),
                            int(w * sx), int(h * sy)))
    return results          # list of (cx, cy, w, h)

# ─────────────────────────────────────────────────────────────
#  DRAW HELPERS
# ─────────────────────────────────────────────────────────────
DENSITY_COLORS = {
    "LOW":    (0, 200, 0),
    "MEDIUM": (0, 165, 255),
    "HIGH":   (0, 0, 255),
}

def density_label(count):
    if count < LOW_TH:   return "LOW",    DENSITY_COLORS["LOW"]
    if count < HIGH_TH:  return "MEDIUM", DENSITY_COLORS["MEDIUM"]
    return "HIGH", DENSITY_COLORS["HIGH"]

def draw_split_view(frame, people):
    """
    Split frame into 2×3 zones, draw bounding circles/rectangles,
    colour zone borders by density, print count+label.
    Returns annotated frame.
    """
    H, W = frame.shape[:2]
    zh, zw = H // ZONE_ROWS, W // ZONE_COLS
    out = frame.copy()

    zone_id = 1
    for r in range(ZONE_ROWS):
        for c in range(ZONE_COLS):
            y1, y2 = r * zh, (r + 1) * zh
            x1, x2 = c * zw, (c + 1) * zw
            count = 0
            for (cx, cy, pw, ph) in people:
                if x1 <= cx < x2 and y1 <= cy < y2:
                    count += 1
                    # draw small circle on centroid
                    cv2.circle(out, (cx, cy), max(pw // 4, 8), (0, 255, 0), 2)
            lab, col = density_label(count)
            cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), col, 3)
            cv2.putText(out, f"Z{zone_id} {count} [{lab}]",
                        (x1 + 8, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
            zone_id += 1

    total = len(people)
    cv2.putText(out, f"TOTAL: {total}", (40, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
    return out

def draw_zone_twin(people, frame_shape):
    """200×600 coloured zone map."""
    H, W = frame_shape[:2]
    zh, zw = H // ZONE_ROWS, W // ZONE_COLS
    twin = np.zeros((300, 600, 3), dtype=np.uint8)
    for i in range(ZONE_ROWS * ZONE_COLS):
        r, c = i // ZONE_COLS, i % ZONE_COLS
        fy1, fy2 = r * zh, (r + 1) * zh
        fx1, fx2 = c * zw, (c + 1) * zw
        cnt = sum(1 for (cx, cy, _, __) in people if fx1 <= cx < fx2 and fy1 <= cy < fy2)
        _, col = density_label(cnt)
        tx1, ty1 = c * 200, r * 150
        tx2, ty2 = tx1 + 200, ty1 + 150
        cv2.rectangle(twin, (tx1, ty1), (tx2, ty2), col, -1)
        cv2.rectangle(twin, (tx1, ty1), (tx2, ty2), (255, 255, 255), 2)
        cv2.putText(twin, f"Z{i+1}: {cnt}", (tx1 + 10, ty1 + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return twin

def draw_dot_grid(people, frame_shape):
    """500×500 dot-density heatmap (20×20 grid)."""
    H, W = frame_shape[:2]
    grid = np.zeros((20, 20), dtype=np.float32)
    ch, cw = H / 20, W / 20
    for (cx, cy, _, __) in people:
        gr = min(int(cy / ch), 19)
        gc = min(int(cx / cw), 19)
        grid[gr, gc] += 1

    vis = np.zeros((500, 500, 3), dtype=np.uint8)
    for rr in range(20):
        for cc in range(20):
            v = grid[rr, cc]
            if v == 0:   col = (60, 60, 60)
            elif v < 2:  col = (0, 200, 0)
            elif v < 4:  col = (0, 165, 255)
            else:        col = (0, 0, 255)
            cv2.rectangle(vis, (cc * 25, rr * 25), (cc * 25 + 24, rr * 25 + 24), col, -1)
    return vis

_graph_history = []

def draw_graph(count):
    """600×200 crowd count time series."""
    _graph_history.append(count)
    if len(_graph_history) > GRAPH_LEN:
        _graph_history.pop(0)

    img = np.zeros((200, 600, 3), dtype=np.uint8)
    # grid lines
    for y in [50, 100, 150]:
        cv2.line(img, (0, y), (600, y), (40, 40, 40), 1)
    cv2.putText(img, "Crowd Count", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    if len(_graph_history) > 1:
        maxv = max(max(_graph_history), 1)
        n = len(_graph_history)
        for i in range(1, n):
            x1 = int((i - 1) / GRAPH_LEN * 600)
            x2 = int(i / GRAPH_LEN * 600)
            y1 = 195 - int(_graph_history[i - 1] / maxv * 185)
            y2 = 195 - int(_graph_history[i] / maxv * 185)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    if _graph_history:
        cv2.putText(img, str(_graph_history[-1]), (550, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img

def compose_dashboard(main_frame, zone_img, dot_img, graph_img, mode_label):
    """
    Layout:
      LEFT  column: main split-view (resized to fit)
      RIGHT column (stacked): zone twin | dot grid | graph
    """
    # Target layout: 1280 × 720
    left_w = 860
    right_w = 420

    # -- LEFT --
    left = cv2.resize(main_frame, (left_w, 720))

    # -- RIGHT panels --
    zone_r  = cv2.resize(zone_img,  (right_w, 200))
    dot_r   = cv2.resize(dot_img,   (right_w, 330))
    graph_r = cv2.resize(graph_img, (right_w, 190))

    right = np.vstack([zone_r, dot_r, graph_r])

    canvas = np.hstack([left, right])

    # Mode label top-right
    cv2.putText(canvas, mode_label, (870, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(canvas, "[1]Twin [2]Gate [3]Inflow [4]Evac [Q]Quit",
                (870, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return canvas

# ─────────────────────────────────────────────────────────────
#  SOCIAL FORCE MODEL
# ─────────────────────────────────────────────────────────────
class Agent:
    __slots__ = ["pos", "vel", "goal", "active", "radius", "mass"]
    def __init__(self, x, y, gx, gy):
        self.pos    = np.array([x, y], dtype=np.float64)
        self.vel    = np.zeros(2)
        self.goal   = np.array([gx, gy], dtype=np.float64)
        self.active = True
        self.radius = AGENT_RADIUS
        self.mass   = SFM_MASS

def sfm_step(agents, walls, desired_speed=SFM_DESIRE):
    """One SFM integration step for all active agents."""
    positions = np.array([a.pos for a in agents if a.active])
    active_agents = [a for a in agents if a.active]

    for i, agent in enumerate(active_agents):
        if not agent.active:
            continue

        # ── desired direction ──────────────────────────────
        to_goal = agent.goal - agent.pos
        dist_goal = np.linalg.norm(to_goal)
        if dist_goal < 0.5:
            agent.active = False
            agent.vel[:] = 0
            continue
        e_desired = to_goal / dist_goal
        f_self = agent.mass / SFM_TAU * (desired_speed * e_desired - agent.vel)

        # ── pedestrian repulsion ───────────────────────────
        f_ped = np.zeros(2)
        for j, other in enumerate(active_agents):
            if j == i or not other.active:
                continue
            diff = agent.pos - other.pos
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                dist = 1e-6
            n_ij = diff / dist
            r_sum = agent.radius + other.radius
            f_ped += SFM_A * math.exp((r_sum - dist) / SFM_B) * n_ij

        # ── wall repulsion ─────────────────────────────────
        f_wall = np.zeros(2)
        for (wx1, wy1, wx2, wy2) in walls:
            # closest point on wall segment
            wall_vec = np.array([wx2 - wx1, wy2 - wy1], dtype=np.float64)
            wl = np.linalg.norm(wall_vec)
            if wl < 1e-6:
                continue
            t = np.dot(agent.pos - np.array([wx1, wy1]), wall_vec) / (wl * wl)
            t = max(0.0, min(1.0, t))
            closest = np.array([wx1, wy1]) + t * wall_vec
            diff_w = agent.pos - closest
            dw = np.linalg.norm(diff_w)
            if dw < 1e-6:
                continue
            f_wall += SFM_AW * math.exp((agent.radius - dw) / SFM_BW) * (diff_w / dw)

        f_total = f_self + f_ped + f_wall

        # ── Euler integration ──────────────────────────────
        agent.vel += (f_total / agent.mass) * SFM_DT
        # clamp speed
        spd = np.linalg.norm(agent.vel)
        if spd > 3.5:
            agent.vel = agent.vel / spd * 3.5
        agent.pos += agent.vel * SFM_DT

def render_sfm(agents, walls, canvas_w_m, canvas_h_m, scenario_label, frame_count):
    """Render SFM agents on a canvas scaled by SCALE px/m."""
    cw = int(canvas_w_m * SCALE)
    ch = int(canvas_h_m * SCALE)
    img = np.zeros((ch, cw, 3), dtype=np.uint8)

    # floor grid
    for gx in range(0, cw, SCALE):
        cv2.line(img, (gx, 0), (gx, ch), (25, 25, 25), 1)
    for gy in range(0, ch, SCALE):
        cv2.line(img, (0, gy), (cw, gy), (25, 25, 25), 1)

    # walls
    for (wx1, wy1, wx2, wy2) in walls:
        p1 = (int(wx1 * SCALE), int(wy1 * SCALE))
        p2 = (int(wx2 * SCALE), int(wy2 * SCALE))
        cv2.line(img, p1, p2, (100, 100, 255), 3)

    # agents
    for a in agents:
        if not a.active:
            continue
        px = int(a.pos[0] * SCALE)
        py = int(a.pos[1] * SCALE)
        spd = np.linalg.norm(a.vel)
        if spd < 0.5:   col = (0, 200, 0)
        elif spd < 1.2: col = (0, 165, 255)
        else:           col = (0, 0, 255)
        cv2.circle(img, (px, py), max(int(AGENT_RADIUS * SCALE), 4), col, -1)
        # velocity arrow
        vend = (int(px + a.vel[0] * SCALE * 0.4),
                int(py + a.vel[1] * SCALE * 0.4))
        cv2.arrowedLine(img, (px, py), vend, (255, 255, 255), 1, tipLength=0.4)

    cv2.putText(img, scenario_label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    cv2.putText(img, f"Active: {sum(1 for a in agents if a.active)}",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
    cv2.putText(img, f"t = {frame_count * SFM_DT:.1f}s",
                (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    # colour legend
    for idx, (lbl, c) in enumerate([("Slow", (0, 200, 0)),
                                    ("Med",  (0, 165, 255)),
                                    ("Fast", (0, 0, 255))]):
        cv2.circle(img, (cw - 80, 20 + idx * 22), 7, c, -1)
        cv2.putText(img, lbl, (cw - 65, 26 + idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return img

# ─────────────────────────────────────────────────────────────
#  SCENARIO BUILDERS
# ─────────────────────────────────────────────────────────────
ARENA_W, ARENA_H = 40.0, 24.0   # metres

def _random_agents(n, spawn_x_range, spawn_y_range, goal_x_range, goal_y_range):
    agents = []
    for _ in range(n):
        x  = random.uniform(*spawn_x_range)
        y  = random.uniform(*spawn_y_range)
        gx = random.uniform(*goal_x_range)
        gy = random.uniform(*goal_y_range)
        agents.append(Agent(x, y, gx, gy))
    return agents

def scenario_gate_close():
    """
    60 people walk toward a gate (right wall).
    At t=5 s the gate narrows to half, forcing compression & slowdown.
    """
    n = 60
    agents = _random_agents(n,
                            spawn_x_range=(1, 15),
                            spawn_y_range=(2, ARENA_H - 2),
                            goal_x_range=(ARENA_W - 1, ARENA_W - 0.5),
                            goal_y_range=(ARENA_H / 2 - 2, ARENA_H / 2 + 2))
    # Boundary walls: top, bottom, left, right (with gate gap centre)
    gate_open  = 6.0  # metres wide
    gate_half  = 3.0
    gate_cy    = ARENA_H / 2
    walls_open = [
        (0, 0, ARENA_W, 0),                                  # top
        (0, ARENA_H, ARENA_W, ARENA_H),                      # bottom
        (0, 0, 0, ARENA_H),                                  # left
        (ARENA_W, 0, ARENA_W, gate_cy - gate_open / 2),      # right upper
        (ARENA_W, gate_cy + gate_open / 2, ARENA_W, ARENA_H),# right lower
    ]
    walls_closed = [
        (0, 0, ARENA_W, 0),
        (0, ARENA_H, ARENA_W, ARENA_H),
        (0, 0, 0, ARENA_H),
        (ARENA_W, 0, ARENA_W, gate_cy - gate_half / 2),
        (ARENA_W, gate_cy + gate_half / 2, ARENA_W, ARENA_H),
    ]
    return agents, walls_open, walls_closed, "Gate-Close Scenario"

def scenario_inflow_surge():
    """
    Starts with 30 agents; at t=4 s an extra 50 rush in from the left.
    """
    agents = _random_agents(30,
                            spawn_x_range=(1, 8),
                            spawn_y_range=(2, ARENA_H - 2),
                            goal_x_range=(ARENA_W - 2, ARENA_W - 1),
                            goal_y_range=(2, ARENA_H - 2))
    walls = [
        (0, 0, ARENA_W, 0),
        (0, ARENA_H, ARENA_W, ARENA_H),
        (0, 0, 0, ARENA_H),
        (ARENA_W, 0, ARENA_W, ARENA_H / 2 - 3),
        (ARENA_W, ARENA_H / 2 + 3, ARENA_W, ARENA_H),
    ]
    return agents, walls, "Inflow-Surge Scenario"

def scenario_emergency():
    """
    80 people scattered; emergency triggered → all sprint toward exit (bottom-right).
    """
    agents = _random_agents(80,
                            spawn_x_range=(5, ARENA_W - 5),
                            spawn_y_range=(5, ARENA_H - 5),
                            goal_x_range=(ARENA_W - 1, ARENA_W - 0.5),
                            goal_y_range=(ARENA_H - 3, ARENA_H - 1))
    walls = [
        (0, 0, ARENA_W, 0),
        (0, ARENA_H, ARENA_W, ARENA_H),
        (0, 0, 0, ARENA_H),
        (ARENA_W, 0, ARENA_W, ARENA_H - 4),     # right wall with exit at bottom
    ]
    return agents, walls, "Emergency Evacuation Scenario"

# ─────────────────────────────────────────────────────────────
#  EXPLANATION
# ─────────────────────────────────────────────────────────────
EXPLANATION = """
╔══════════════════════════════════════════════════════════════════════════════╗
║            WHY THIS DIGITAL TWIN + SOCIAL FORCE MODEL?                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DIGITAL TWIN (Mode 1)                                                       ║
║  ─────────────────────                                                       ║
║  • 2×3 zone split gives operators an instant spatial density view.           ║
║  • Dot-grid heatmap reveals fine-grained density "hot spots" not visible     ║
║    in per-zone counts.                                                        ║
║  • Time-series graph tracks trends (crowd building / dispersing) so staff    ║
║    can act before a zone exceeds safe capacity.                              ║
║  • HOG (Histogram of Oriented Gradients) person detector works entirely      ║
║    offline; no GPU/cloud required.                                           ║
║                                                                              ║
║  SOCIAL FORCE MODEL (Helbing & Molnár, 1995)                                 ║
║  ────────────────────────────────────────────                                ║
║  Each pedestrian has three forces:                                            ║
║    1. Self-drive  : pulls them toward their goal at their desired speed.     ║
║    2. Repulsion   : exponentially decays with distance to other agents       ║
║                     → naturally produces personal-space maintenance.         ║
║    3. Wall force  : identical decay from nearest wall segment                ║
║                     → prevents penetration and models bottleneck pressure.   ║
║                                                                              ║
║  SCENARIOS                                                                   ║
║  ─────────                                                                   ║
║  Gate-Close  : wall segment shrinks at t=5s → arching compression wave      ║
║               visible as agents slow and queue upstream of the gate.        ║
║  Inflow-Surge: 50 extra agents spawn at t=4s → density spike propagates     ║
║               inward, slowing existing pedestrians (red = fast, often        ║
║               pinned near walls).                                            ║
║  Emergency   : desired speed raised 2× → agents sprint, collision avoidance  ║
║               still active, showing realistic herding & bottleneck crush.    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Crowd Digital Twin + SFM Simulator")
    parser.add_argument("--video", default="", help="Path to crowd video file")
    args = parser.parse_args()

    # Locate video
    vid_path = args.video
    if not vid_path:
        # auto-search
        for candidate in ["Crowd.mp4", "crowd.mp4", "crowd_vid.mp4",
                          "/mnt/user-data/uploads/Crowd.mp4"]:
            if os.path.exists(candidate):
                vid_path = candidate
                break
    if not vid_path or not os.path.exists(vid_path):
        print("ERROR: video file not found. Pass --video <path>")
        sys.exit(1)

    print(EXPLANATION)

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {vid_path}")
        sys.exit(1)

    mode = 1          # 1=twin, 2=gate, 3=inflow, 4=evac
    frame_id = 0
    detected_people = []

    # SFM state (initialised when mode changes)
    sfm_agents = []
    sfm_walls  = []
    sfm_walls2 = None    # secondary walls for gate scenario
    sfm_label  = ""
    sfm_frame  = 0
    sfm_extra_spawned = False
    sfm_desired_speed = SFM_DESIRE

    def init_sfm(new_mode):
        nonlocal sfm_agents, sfm_walls, sfm_walls2, sfm_label
        nonlocal sfm_frame, sfm_extra_spawned, sfm_desired_speed
        _graph_history.clear()
        sfm_frame = 0
        sfm_extra_spawned = False
        sfm_desired_speed = SFM_DESIRE
        if new_mode == 2:
            sfm_agents, sfm_walls, sfm_walls2, sfm_label = scenario_gate_close()
        elif new_mode == 3:
            sfm_agents, sfm_walls, sfm_label = scenario_inflow_surge()
            sfm_walls2 = None
        elif new_mode == 4:
            sfm_agents, sfm_walls, sfm_label = scenario_emergency()
            sfm_walls2 = None

    # ── Create display windows ──────────────────────────────
    cv2.namedWindow("Crowd Digital Twin", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Crowd Digital Twin", DISPLAY_W, DISPLAY_H)

    print("Controls: [1] Digital Twin  [2] Gate Close  "
          "[3] Inflow Surge  [4] Emergency  [Q] Quit")

    prev_frame = None
    sfm_render_cache = None

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('1'):
            mode = 1
            _graph_history.clear()
        elif key == ord('2'):
            mode = 2; init_sfm(2)
        elif key == ord('3'):
            mode = 3; init_sfm(3)
        elif key == ord('4'):
            mode = 4; init_sfm(4)

        # ── MODE 1 : Digital Twin ────────────────────────────
        if mode == 1:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            frame_id += 1

            if frame_id % PROCESS_EVERY == 0:
                detected_people = hog_detect(frame)

            main_view  = draw_split_view(frame, detected_people)
            zone_img   = draw_zone_twin(detected_people, frame.shape)
            dot_img    = draw_dot_grid(detected_people, frame.shape)
            graph_img  = draw_graph(len(detected_people))

            canvas = compose_dashboard(main_view, zone_img, dot_img, graph_img,
                                       "MODE: Digital Twin")

        # ── MODE 2/3/4 : SFM Simulation ──────────────────────
        else:
            if not sfm_agents:
                init_sfm(mode)

            # Trigger scenario events
            t_sec = sfm_frame * SFM_DT

            if mode == 2 and t_sec >= 5.0 and sfm_walls2 is not None:
                sfm_walls = sfm_walls2    # gate narrows
                sfm_walls2 = None

            if mode == 3 and t_sec >= 4.0 and not sfm_extra_spawned:
                extra = _random_agents(
                    50,
                    spawn_x_range=(0.5, 3),
                    spawn_y_range=(2, ARENA_H - 2),
                    goal_x_range=(ARENA_W - 2, ARENA_W - 1),
                    goal_y_range=(2, ARENA_H - 2))
                sfm_agents.extend(extra)
                sfm_extra_spawned = True

            if mode == 4 and not sfm_extra_spawned:
                sfm_desired_speed = SFM_DESIRE * 2.0
                sfm_extra_spawned = True

            sfm_step(sfm_agents, sfm_walls, sfm_desired_speed)
            sfm_frame += 1

            # Re-seed once all agents have reached goals
            active_count = sum(1 for a in sfm_agents if a.active)
            if active_count == 0:
                init_sfm(mode)

            # Render SFM canvas
            sim_img = render_sfm(sfm_agents, sfm_walls,
                                 ARENA_W, ARENA_H, sfm_label, sfm_frame)

            # Build "people" list for twin panels from agent positions
            sim_people = []
            for a in sfm_agents:
                if a.active:
                    px = int(a.pos[0] / ARENA_W * DISPLAY_W)
                    py = int(a.pos[1] / ARENA_H * DISPLAY_H)
                    sim_people.append((px, py, 20, 40))

            dummy_frame_shape = (DISPLAY_H, DISPLAY_W, 3)
            zone_img  = draw_zone_twin(sim_people, dummy_frame_shape)
            dot_img   = draw_dot_grid(sim_people,  dummy_frame_shape)
            graph_img = draw_graph(len(sim_people))

            canvas = compose_dashboard(sim_img, zone_img, dot_img, graph_img,
                                       f"MODE: {sfm_label}")

        canvas_show = cv2.resize(canvas, (DISPLAY_W, DISPLAY_H))
        cv2.imshow("Crowd Digital Twin", canvas_show)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()