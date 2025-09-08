#!/usr/bin/env python3
import glob
import os
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

# === Find all *_kalman.csv files ===
files = glob.glob(os.path.join(DATA_DIR, "*_kalman.csv"))
if not files:
    print("No *_kalman.csv files found in", DATA_DIR)
    exit()

pgf_files = []

# === Process each file ===
for fname in files:
    print(f"Plotting {fname} ...")

    # Limit to first MAX_ROWS rows (~60 s of data)
    df = pd.read_csv(fname, nrows=MAX_ROWS)

    time = df["time"]

    # Reference vs estimated Euler angles
    angles = [
        ("roll_ref", "roll_est", "Roll (deg)"),
        ("pitch_ref", "pitch_est", "Pitch (deg)"),
        ("yaw_ref", "yaw_est", "Yaw (deg)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(os.path.basename(fname))

    for ax, (ref_col, est_col, label) in zip(axes, angles):
        ax.plot(time, df[ref_col], label="Reference", linewidth=1.5)
        ax.plot(time, df[est_col], label="Estimated", linewidth=1.0, linestyle="--")
        ax.set_ylabel(label)
        ax.grid(True)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # === Save to PGF and SVG ===
    base = os.path.splitext(os.path.basename(fname))[0]
    outbase = os.path.join(DATA_DIR, base)

    pgf_out = f"{outbase}.pgf"
    svg_out = f"{outbase}.svg"

    plt.savefig(pgf_out, format="pgf", bbox_inches="tight")
    plt.savefig(svg_out, format="svg", bbox_inches="tight")
    plt.close(fig)

    pgf_files.append((base, pgf_out))
    print(f"Saved {pgf_out} and {svg_out}")
