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
DATA_DIR = "./"             # Directory with *_w3d.csv files
SAMPLE_RATE_HZ = 240        # Simulator sample rate
SKIP_TIME_S = 60.0          # Skip first n seconds (warmup)
PLOT_TIME_S = 60.0          # Plot next m seconds
MAX_TIME_S  = SKIP_TIME_S + PLOT_TIME_S
MAX_ROWS    = int(SAMPLE_RATE_HZ * MAX_TIME_S)

# Toggle error plots
PLOT_ERRORS = False   # set True to enable

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
    r".*?_(?P<wave>[a-zA-Z0-9]+)_H(?P<height>[-0-9\.]+).*?_w3d\.csv$"
)

def latex_safe(s: str) -> str:
    return s.replace("_", r"\_")

# === Find all *_w3d.csv files ===
files = glob.glob(os.path.join(DATA_DIR, "*_w3d.csv"))
if not files:
    print("No *_w3d.csv files found in", DATA_DIR)
    exit()

for fname in files:
    basename = os.path.basename(fname)
    m = pattern.match(basename)
    if not m:
        continue
    wave_type = m.group("wave").lower()
    if wave_type not in ALLOWED_WAVES:
        continue
    try:
        height_val = float(m.group("height"))
    except (TypeError, ValueError):
        continue
    group_name = None
    for name, h in height_groups.items():
        if abs(height_val - h) < 1e-6:
            group_name = name
            break
    if group_name is None:
        continue

    outbase = f"w3d_{wave_type}_{group_name}"
    outbase = re.sub(r"[^A-Za-z0-9_\-]", "_", outbase)
    outbase = os.path.join(DATA_DIR, outbase)

    print(f"Plotting {fname} → {outbase} ...")
    df = pd.read_csv(fname, nrows=MAX_ROWS)
    df = df[(df["time"] >= SKIP_TIME_S) & (df["time"] <= MAX_TIME_S)].reset_index(drop=True)
    time = df["time"]

    # === Angles (Reference vs Estimated) ===
    angles = [
        ("roll_ref", "roll_est", "Roll (deg)"),
        ("pitch_ref", "pitch_est", "Pitch (deg)"),
        ("yaw_ref", "yaw_est", "Yaw (deg)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(latex_safe(basename))
    for ax, (ref_col, est_col, label) in zip(axes, angles):
        ax.plot(time, df[ref_col], label="Reference", linewidth=1.5)
        ax.plot(time, df[est_col], label="Estimated", linewidth=1.0, linestyle="--")
        ax.set_ylabel(latex_safe(label))
        ax.grid(True)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{outbase}.pgf", format="pgf", bbox_inches="tight")
    plt.savefig(f"{outbase}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    # === Angle errors (optional) ===
    if PLOT_ERRORS:
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(latex_safe(basename) + " (Angle Errors)")
        axes[0].plot(time, df["err_roll"], color="tab:red"); axes[0].set_ylabel("Roll err [deg]"); axes[0].grid(True)
        axes[1].plot(time, df["err_pitch"], color="tab:red"); axes[1].set_ylabel("Pitch err [deg]"); axes[1].grid(True)
        axes[2].plot(time, df["err_yaw"], color="tab:red"); axes[2].set_ylabel("Yaw err [deg]"); axes[2].grid(True)
        axes[3].plot(time, df["angle_err"], color="tab:purple"); axes[3].set_ylabel("Quat err [deg]"); axes[3].set_xlabel("Time (s)"); axes[3].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{outbase}_angle_errs.pgf", format="pgf", bbox_inches="tight")
        plt.savefig(f"{outbase}_angle_errs.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    # === Z-axis kinematics (always produce) ===
    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(latex_safe(basename) + " (Z-axis)")
    for i, prefix in enumerate(["disp", "vel", "acc"]):
        # Values
        axes[2*i].plot(time, df[f"{prefix}_ref_z"], label="Ref")
        axes[2*i].plot(time, df[f"{prefix}_est_z"], label="Est", linestyle="--")
        axes[2*i].set_ylabel(f"{prefix.capitalize()} Z")
        axes[2*i].legend(); axes[2*i].grid(True)

        # Errors
        if PLOT_ERRORS:
            axes[2*i+1].plot(time, df[f"{prefix}_err_z"], color="tab:red")
            axes[2*i+1].set_ylabel("Error"); axes[2*i+1].grid(True)
        else:
            axes[2*i+1].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{outbase}_zkin.pgf", format="pgf", bbox_inches="tight")
    plt.savefig(f"{outbase}_zkin.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    # === XY kinematics (always produce) ===
    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(latex_safe(basename) + " (X/Y axes)")
    for i, prefix in enumerate(["disp", "vel", "acc"]):
        # Values
        axes[2*i].plot(time, df[f"{prefix}_ref_x"], label="Ref X", color="tab:blue")
        axes[2*i].plot(time, df[f"{prefix}_est_x"], label="Est X", linestyle="--", color="tab:blue")
        axes[2*i].plot(time, df[f"{prefix}_ref_y"], label="Ref Y", color="tab:orange")
        axes[2*i].plot(time, df[f"{prefix}_est_y"], label="Est Y", linestyle="--", color="tab:orange")
        axes[2*i].set_ylabel(f"{prefix.capitalize()} XY")
        axes[2*i].legend(ncol=2, fontsize=8); axes[2*i].grid(True)

        # Errors
        if PLOT_ERRORS:
            axes[2*i+1].plot(time, df[f"{prefix}_err_x"], label="Err X", color="tab:blue")
            axes[2*i+1].plot(time, df[f"{prefix}_err_y"], label="Err Y", color="tab:orange")
            axes[2*i+1].set_ylabel("Error")
            axes[2*i+1].legend(ncol=2, fontsize=8); axes[2*i+1].grid(True)
        else:
            axes[2*i+1].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{outbase}_xykin.pgf", format="pgf", bbox_inches="tight")
    plt.savefig(f"{outbase}_xykin.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
