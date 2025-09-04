#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

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

# === Wave categories ===
height_groups = {
    "low":    [0.27],
    "medium": [4.0],
    "high":   [8.5],
}

# Color palettes for each height group (light → dark)
height_colors = {
    "low":    ['#e5f5f9', '#2ca25f'],
    "medium": ['#deebf7', '#3182bd'],
    "high":   ['#fee0d2', '#de2d26'],
}

# Wave types to include (must match filenames)
wave_types = ["gerstner", "jonswap", "fenton", "pmstokes", "cnoidal"]

# Components for world-frame
components = {
    'Displacement': ['disp_x', 'disp_y', 'disp_z'],
    'Velocity':     ['vel_x', 'vel_y', 'vel_z'],
    'Acceleration': ['acc_x', 'acc_y', 'acc_z'],
}

# Sampling cutoff
SAMPLE_RATE = 240
MAX_TIME = 300.0
MAX_RECORDS = int(SAMPLE_RATE * MAX_TIME)


def plot_wave_type(wave_type):
    """Generate 4 figures (PGF+SVG) for one wave type."""
    # --- Chart 1: world-frame disp/vel/acc ---
    fig, axes = plt.subplots(len(components), 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{wave_type.capitalize()} - World Frame")

    for group, heights in height_groups.items():
        for idx, h in enumerate(heights):
            fname = f"wave_data_{wave_type}_H{h:.3f}"  # prefix
            candidates = [f for f in os.listdir(".") if f.startswith(fname) and f.endswith(".csv")]
            if not candidates:
                continue
            data = pd.read_csv(candidates[0]).head(MAX_RECORDS)
            color = height_colors[group][idx % len(height_colors[group])]
            time = data["time"]

            for ax, (comp_label, cols) in zip(axes, components.items()):
                for col in cols:
                    ax.plot(time, data[col], label=f"H={h} {col}", color=color, alpha=0.8)
                ax.set_ylabel(comp_label)
                ax.grid(True)

    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(fontsize="small", ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{wave_type}_worldframe.pgf", bbox_inches="tight")
    fig.savefig(f"{wave_type}_worldframe.svg", bbox_inches="tight")
    plt.close(fig)

    # --- Chart 2: IMU acceleration ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{wave_type.capitalize()} - IMU Acceleration")

    for group, heights in height_groups.items():
        for idx, h in enumerate(heights):
            fname = f"wave_data_{wave_type}_H{h:.3f}"
            candidates = [f for f in os.listdir(".") if f.startswith(fname) and f.endswith(".csv")]
            if not candidates:
                continue
            data = pd.read_csv(candidates[0]).head(MAX_RECORDS)
            color = height_colors[group][idx % len(height_colors[group])]
            time = data["time"]

            for i, comp in enumerate(['acc_bx', 'acc_by', 'acc_bz']):
                axes[i].plot(time, data[comp], label=f"H={h} {comp}", color=color, alpha=0.8)
                axes[i].set_ylabel(comp)
                axes[i].grid(True)

    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(fontsize="small", ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{wave_type}_imu_acc.pgf", bbox_inches="tight")
    fig.savefig(f"{wave_type}_imu_acc.svg", bbox_inches="tight")
    plt.close(fig)

    # --- Chart 3: IMU gyro ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{wave_type.capitalize()} - IMU Gyro")

    for group, heights in height_groups.items():
        for idx, h in enumerate(heights):
            fname = f"wave_data_{wave_type}_H{h:.3f}"
            candidates = [f for f in os.listdir(".") if f.startswith(fname) and f.endswith(".csv")]
            if not candidates:
                continue
            data = pd.read_csv(candidates[0]).head(MAX_RECORDS)
            color = height_colors[group][idx % len(height_colors[group])]
            time = data["time"]

            for i, comp in enumerate(['gyro_x', 'gyro_y', 'gyro_z']):
                axes[i].plot(time, data[comp], label=f"H={h} {comp}", color=color, alpha=0.8)
                axes[i].set_ylabel(comp)
                axes[i].grid(True)

    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(fontsize="small", ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{wave_type}_imu_gyro.pgf", bbox_inches="tight")
    fig.savefig(f"{wave_type}_imu_gyro.svg", bbox_inches="tight")
    plt.close(fig)

    # --- Chart 4: Euler angles ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{wave_type.capitalize()} - Euler Angles")

    for group, heights in height_groups.items():
        for idx, h in enumerate(heights):
            fname = f"wave_data_{wave_type}_H{h:.3f}"
            candidates = [f for f in os.listdir(".") if f.startswith(fname) and f.endswith(".csv")]
            if not candidates:
                continue
            data = pd.read_csv(candidates[0]).head(MAX_RECORDS)
            color = height_colors[group][idx % len(height_colors[group])]
            time = data["time"]

            for i, comp in enumerate(['roll_deg', 'pitch_deg', 'yaw_deg']):
                axes[i].plot(time, data[comp], label=f"H={h} {comp}", color=color, alpha=0.8)
                axes[i].set_ylabel(comp)
                axes[i].grid(True)

    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(fontsize="small", ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{wave_type}_euler.pgf", bbox_inches="tight")
    fig.savefig(f"{wave_type}_euler.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for wt in wave_types:
        plot_wave_type(wt)
    print("All PGF and SVG plots saved.")
