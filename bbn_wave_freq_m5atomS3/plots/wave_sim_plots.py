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

# === Wave categories (heights to show explicitly) ===
height_groups = {
    "low":    [0.27],   # always green
    "medium": [1.5],    # always blue
    "high":   [8.5],    # always red
}

# Color palettes for each group (shades for x,y,z)
height_colors = {
    "low":    ['#a1d99b', '#41ab5d', '#005a32'],   # Greens
    "medium": ['#9ecae1', '#3182bd', '#08306b'],   # Blues
    "high":   ['#fcbba1', '#fb6a4a', '#a50f15'],   # Reds
}

# Wave types to include (must match filenames)
wave_types = ["gerstner", "jonswap", "fenton", "pmstokes", "cnoidal"]

# Components for world-frame plots
components = {
    'Displacement': ['disp_x', 'disp_y', 'disp_z'],
    'Velocity':     ['vel_x', 'vel_y', 'vel_z'],
    'Acceleration': ['acc_x', 'acc_y', 'acc_z'],
}

# Sampling cutoff
SAMPLE_RATE = 240
MAX_TIME = 120.0
MAX_RECORDS = int(SAMPLE_RATE * MAX_TIME)


def find_file(wave_type, height):
    """Find matching CSV file for wave_type and given height."""
    fname = f"wave_data_{wave_type}_H{height:.3f}"
    candidates = [f for f in os.listdir(".") if f.startswith(fname) and f.endswith(".csv")]
    return candidates[0] if candidates else None


def plot_wave_type(wave_type):
    """Generate plots (PGF+SVG) for one wave type."""

    # --- Chart 1: world-frame disp/vel/acc ---
    fig, axes = plt.subplots(len(components), 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{wave_type.capitalize()} - World Frame")

    for group, heights in height_groups.items():
        for h in heights:
            csv_file = find_file(wave_type, h)
            if not csv_file:
                continue
            data = pd.read_csv(csv_file).head(MAX_RECORDS)
            time = data["time"]

            for ax, (comp_label, cols) in zip(axes, components.items()):
                for j, col in enumerate(cols):
                    comp_color = height_colors[group][j % len(height_colors[group])]
                    ax.plot(time, data[col], label=f"H={h} {col}",
                            color=comp_color, alpha=1.0, linewidth=1.2)
                ax.set_ylabel(comp_label)
                ax.grid(True)

    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(fontsize="small", ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{wave_type}_worldframe.pgf", bbox_inches="tight")
    fig.savefig(f"{wave_type}_worldframe.svg", bbox_inches="tight")
    plt.close(fig)

    # Skip IMU/Euler plots for wave types without them
    if wave_type in ["fenton", "gerstner", "cnoidal"]:
        return

    # --- Chart 2: IMU acceleration ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{wave_type.capitalize()} - IMU Acceleration")

    for group, heights in height_groups.items():
        for h in heights:
            csv_file = find_file(wave_type, h)
            if not csv_file:
                continue
            data = pd.read_csv(csv_file).head(MAX_RECORDS)
            time = data["time"]

            for i, comp in enumerate(['acc_bx', 'acc_by', 'acc_bz']):
                comp_color = height_colors[group][i % len(height_colors[group])]
                axes[i].plot(time, data[comp], label=f"H={h} {comp}",
                             color=comp_color, alpha=1.0, linewidth=1.2)
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
        for h in heights:
            csv_file = find_file(wave_type, h)
            if not csv_file:
                continue
            data = pd.read_csv(csv_file).head(MAX_RECORDS)
            time = data["time"]

            for i, comp in enumerate(['gyro_x', 'gyro_y', 'gyro_z']):
                comp_color = height_colors[group][i % len(height_colors[group])]
                axes[i].plot(time, data[comp], label=f"H={h} {comp}",
                             color=comp_color, alpha=1.0, linewidth=1.2)
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
        for h in heights:
            csv_file = find_file(wave_type, h)
            if not csv_file:
                continue
            data = pd.read_csv(csv_file).head(MAX_RECORDS)
            time = data["time"]

            for i, comp in enumerate(['roll_deg', 'pitch_deg', 'yaw_deg']):
                comp_color = height_colors[group][i % len(height_colors[group])]
                axes[i].plot(time, data[comp], label=f"H={h} {comp}",
                             color=comp_color, alpha=1.0, linewidth=1.2)
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
