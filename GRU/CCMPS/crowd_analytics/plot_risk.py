"""
plot_risk.py  —  Visualise GRU crowd risk predictions from risk_log.csv
========================================================================
Run AFTER test_gru.py has finished:

    python plot_risk.py
    python plot_risk.py --csv risk_log.csv        # explicit path
    python plot_risk.py --csv risk_log.csv --save  # save PNG instead of showing

Requires:  pip install pandas matplotlib
"""

import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# ── ARGS ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",  default="risk_log.csv", help="Path to risk_log.csv")
# --save removed: window always pops up automatically
args = parser.parse_args()

# ── LOAD ──────────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(args.csv)
except FileNotFoundError:
    print(f"[ERROR] File not found: {args.csv}")
    print("        Run test_gru.py first to generate it.")
    sys.exit(1)

required = {"second", "raw_label", "confirmed_risk", "confidence",
            "people", "density", "speed", "stagnation"}
missing  = required - set(df.columns)
if missing:
    print(f"[ERROR] CSV is missing columns: {missing}")
    sys.exit(1)

# ── MAP LABELS TO NUMERIC LEVELS ──────────────────────────────────────────────
# SAFE=0  WARNING=1  HIGH=2  — used for the step-line plots
LEVEL = {"SAFE": 0, "WARNING": 1, "HIGH": 2, "WARMING UP": -1}

df["raw_level"]       = df["raw_label"].map(LEVEL).fillna(-1)
df["confirmed_level"] = df["confirmed_risk"].map(LEVEL).fillna(-1)

# Only plot rows where a real prediction exists (not warmup)
pred_df = df[df["raw_level"] >= 0].copy()

if pred_df.empty:
    print("[ERROR] No predictions found in CSV (all rows are warming-up).")
    sys.exit(1)

t = pred_df["second"].values

# ── COLOURS ───────────────────────────────────────────────────────────────────
C_SAFE    = "#00cc66"
C_WARNING = "#ffcc00"
C_HIGH    = "#ff3344"
C_CONF    = "#00d4ff"

BAND_COLOURS = {0: C_SAFE, 1: C_WARNING, 2: C_HIGH}

# ── FIGURE ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    4, 1,
    figsize=(14, 11),
    gridspec_kw={"height_ratios": [2.5, 2.5, 1.2, 1.8]},
    sharex=True,
)
fig.patch.set_facecolor("#070e18")
for ax in axes:
    ax.set_facecolor("#0a1520")
    ax.tick_params(colors="#7a9ab5", labelsize=8)
    ax.spines[:].set_color("#1a3050")
    ax.yaxis.label.set_color("#c8e6f5")
    ax.xaxis.label.set_color("#c8e6f5")
    ax.title.set_color("#e8f8ff")

fig.suptitle("GRU Crowd Risk Forecast Log", fontsize=13,
             color="#e8f8ff", fontweight="bold", y=0.98)

# ── PANEL 1: RAW PREDICTION LEVEL (coloured step line) ───────────────────────
ax1 = axes[0]
ax1.set_title("Raw GRU Prediction (per second)", fontsize=9, pad=4)

raw_levels = pred_df["raw_level"].values
# Draw coloured background bands per second
for i in range(len(t)):
    lv = int(raw_levels[i])
    ax1.axvspan(t[i] - 0.5, t[i] + 0.5, alpha=0.25,
                color=BAND_COLOURS.get(lv, "#333"), linewidth=0)

ax1.step(t, raw_levels, where="mid", color="#ffffff", linewidth=1.2, alpha=0.8)
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(["SAFE", "WARNING", "HIGH"], fontsize=8, color="#c8e6f5")
ax1.set_ylim(-0.4, 2.6)
ax1.grid(axis="x", color="#1a3050", linewidth=0.5)

# Confidence overlay on twin axis
ax1b = ax1.twinx()
ax1b.set_facecolor("#0a1520")
ax1b.fill_between(t, pred_df["confidence"].values * 100,
                  alpha=0.25, color=C_CONF, linewidth=0)
ax1b.plot(t, pred_df["confidence"].values * 100,
          color=C_CONF, linewidth=0.9, alpha=0.7, linestyle="--")
ax1b.set_ylabel("Confidence %", fontsize=7, color=C_CONF)
ax1b.tick_params(colors=C_CONF, labelsize=7)
ax1b.set_ylim(0, 115)
ax1b.spines[:].set_color("#1a3050")

# ── PANEL 2: CONFIRMED RISK (hysteresis applied) ──────────────────────────────
ax2 = axes[1]
ax2.set_title("Confirmed Risk (after hysteresis  ·  enter HIGH=3s, exit=5s SAFE)",
              fontsize=9, pad=4)

conf_levels = pred_df["confirmed_level"].values
for i in range(len(t)):
    lv = int(conf_levels[i])
    ax2.axvspan(t[i] - 0.5, t[i] + 0.5, alpha=0.35,
                color=BAND_COLOURS.get(lv, "#333"), linewidth=0)

ax2.step(t, conf_levels, where="mid", color="#ffffff", linewidth=1.4)
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(["SAFE", "WARNING", "HIGH"], fontsize=8, color="#c8e6f5")
ax2.set_ylim(-0.4, 2.6)
ax2.grid(axis="x", color="#1a3050", linewidth=0.5)

# ── PANEL 3: PEOPLE COUNT ─────────────────────────────────────────────────────
ax3 = axes[2]
ax3.set_title("People Count", fontsize=9, pad=4)
ax3.fill_between(t, pred_df["people"].values, alpha=0.4, color="#00d4ff")
ax3.plot(t,      pred_df["people"].values,    color="#00d4ff", linewidth=1.2)
ax3.set_ylabel("count", fontsize=7)
ax3.grid(axis="x", color="#1a3050", linewidth=0.5)

# ── PANEL 4: DENSITY + SPEED + STAGNATION ────────────────────────────────────
ax4 = axes[3]
ax4.set_title("Key Metrics  (density · speed · stagnation)", fontsize=9, pad=4)
ax4.plot(t, pred_df["density"].values,    color="#00ff88", linewidth=1.2, label="Density (p/m²)")
ax4.plot(t, pred_df["speed"].values,      color="#ffcc00", linewidth=1.2, label="Speed (m/s)")
ax4.plot(t, pred_df["stagnation"].values, color="#ff7700", linewidth=1.2, label="Stagnation")
ax4.set_xlabel("Time (seconds)", fontsize=8, color="#c8e6f5")
ax4.set_ylabel("value", fontsize=7)
ax4.grid(axis="x", color="#1a3050", linewidth=0.5)
ax4.legend(fontsize=7, facecolor="#0a1520", edgecolor="#1a3050",
           labelcolor="#c8e6f5", loc="upper right")

# ── SHARED X AXIS ─────────────────────────────────────────────────────────────
axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(5))
axes[-1].xaxis.set_minor_locator(ticker.MultipleLocator(1))

# ── LEGEND FOR RISK COLOURS ───────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=C_SAFE,    alpha=0.7, label="SAFE"),
    mpatches.Patch(color=C_WARNING, alpha=0.7, label="WARNING"),
    mpatches.Patch(color=C_HIGH,    alpha=0.7, label="HIGH"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3,
           fontsize=8, facecolor="#0a1520", edgecolor="#1a3050",
           labelcolor="#c8e6f5", framealpha=0.9,
           bbox_to_anchor=(0.5, 0.005))

plt.tight_layout(rect=[0, 0.04, 1, 0.97])

# ── OUTPUT ────────────────────────────────────────────────────────────────────
# Always save PNG alongside the CSV, then pop up the window
_png_out = args.csv.replace(".csv", "_plot.png")
plt.savefig(_png_out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"[INFO] Graph saved → {_png_out}")
print(f"[INFO] Opening window  (close it to exit)")
plt.show()