#!/usr/bin/env python3
import glob
import os
import re
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# === Matplotlib PGF/LaTeX config ===
mpl.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "xelatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join([
        r"\usepackage{fontspec}",
        r"\usepackage{unicode-math}",
        r"\usepackage{amsmath}",
        r"\setmainfont{DejaVu Serif}",
        r"\setmathfont{Latin Modern Math}",
        r"\providecommand{\mathdefault}[1]{#1}"
    ])
})

# === Config ===
DATA_DIR = "./"            # Directory with *_kalman.csv files
SAMPLE_RATE_HZ = 240       # Simulator sample rate
MAX_TIME_S = 60.0          # Limit to first 60 seconds
MAX_ROWS = int(SAMPLE_RATE_HZ * MAX_TIME_S)

# === Groups we care about (included heights in meters) ===
height_groups = {
    "low":    0.27,
    "medium": 1.50,
    "high":   8.50,
}

# === Allowed wave types ===
ALLOWED_WAVES = {"jonswap", "pmstokes"}

# === Regex to extract wave type and height from filename ===
pattern = re.compile(
    r".*?_(?P<wave>[a-zA-Z0-9]+)_H(?P<height>[-0-9\.]+).*?_kalman\.csv$"
)

# === Find all *_kalman.csv files ===
files = glob.glob(os.path.join(DATA_DIR, "*_kalman.csv"))
if not files:
    print("No *_kalman.csv files found in", DATA_DIR)
    exit()

generated = []  # store (wave_type, group_name, pgf_filename)

# === Process each file ===
for fname in files:
    basename = os.path.basename(fname)
    m = pattern.match(basename)
    if not m:
        print(f"Skipping {fname} (could not parse)")
        continue

    wave_type = m.group("wave").lower()
    if wave_type not in ALLOWED_WAVES:
        print(f"Skipping {fname} (wave={wave_type} not included)")
        continue

    try:
        height_val = float(m.group("height"))
    except (TypeError, ValueError):
        print(f"Skipping {fname} (invalid height)")
        continue

    # Map height to group
    group_name = None
    for name, h in height_groups.items():
        if abs(height_val - h) < 1e-6:
            group_name = name
            break
    if group_name is None:
        print(f"Skipping {fname} (height {height_val} m not in groups)")
        continue

    # Build output base name (safe for LaTeX)
    outbase = f"qmekf_{wave_type}_{group_name}"
    outbase = re.sub(r"[^A-Za-z0-9_\-]", "_", outbase)
    outbase = os.path.join(DATA_DIR, outbase)

    print(f"Plotting {fname} → {outbase} ...")

    # Load limited rows
    df = pd.read_csv(fname, nrows=MAX_ROWS)
    time = df["time"]

    # Reference vs estimated Euler angles
    angles = [
        ("roll_ref", "roll_est", "Roll (deg)"),
        ("pitch_ref", "pitch_est", "Pitch (deg)"),
        ("yaw_ref", "yaw_est", "Yaw (deg)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(basename)

    for ax, (ref_col, est_col, label) in zip(axes, angles):
        ax.plot(time, df[ref_col], label="Reference", linewidth=1.5)
        ax.plot(time, df[est_col], label="Estimated", linewidth=1.0, linestyle="--")
        ax.set_ylabel(label)
        ax.grid(True)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save to PGF and SVG
    pgf_out = f"{outbase}.pgf"
    svg_out = f"{outbase}.svg"
    plt.savefig(pgf_out, format="pgf", bbox_inches="tight")
    plt.savefig(svg_out, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {pgf_out} and {svg_out}")
    generated.append((wave_type, group_name, os.path.basename(pgf_out)))
