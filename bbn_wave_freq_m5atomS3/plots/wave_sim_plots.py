import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import os

# === Configure matplotlib to use PGF backend and LaTeX fonts ===
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

# === Explicit list of wave heights to plot ===
SELECTED_HEIGHTS = [0.27, 1.5, 8.5]   # adjust as needed

# === Sampling and cutoff ===
SAMPLE_RATE_HZ = 240
TIME_LIMIT_S   = 300
MAX_RECORDS    = SAMPLE_RATE_HZ * TIME_LIMIT_S   # 72,000 rows

# === Auto-discover CSV files ===
files = glob.glob("wave_data_*.csv")
if not files:
    raise FileNotFoundError("No wave_data_*.csv files found in current directory")

# === Extract info from filename ===
def extract_wave_info(filename: str):
    """Return (wave_type, height) from filename."""
    base = os.path.basename(filename)
    parts = base.split("_")
    wave_type = parts[2] if len(parts) > 2 else "Unknown"
    height = None
    for p in parts:
        if p.startswith("H"):
            try:
                height = float(p[1:])
            except ValueError:
                pass
    return wave_type, height

# === Categorize height ===
def height_category(height):
    if height < 2.0:
        return "Low"
    elif height < 7.0:
        return "Medium"
    else:
        return "High"

# === Color by height group, linestyle by axis ===
def get_color_and_style(height, axis):
    if height < 2.0:
        colors = ["#4daf4a", "#2b7a2b", "#a1d99b"]  # greens
    elif height < 7.0:
        colors = ["#377eb8", "#184f7d", "#9ecae1"]  # blues
    else:
        colors = ["#e41a1c", "#a50f15", "#fb6a4a"]  # reds

    axis_map = {"x": 0, "y": 1, "z": 2}
    color = colors[axis_map[axis]]

    styles = {"x": "-", "y": "--", "z": ":"}
    style = styles[axis]

    return color, style

# === Helper to add a single combined legend ===
def add_combined_legend(fig, axes):
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    seen = set()
    unique = []
    for h, l in zip(handles, labels):
        if l not in seen:
            unique.append((h, l))
            seen.add(l)
    if unique:
        handles, labels = zip(*unique)
        fig.legend(handles, labels, loc="upper center", fontsize="small", ncol=4)

# === Filter & group by wave type ===
groups = {}
for f in files:
    wave_type, height = extract_wave_info(f)
    if height in SELECTED_HEIGHTS:
        groups.setdefault(wave_type, []).append((height, f))

# === Load and trim CSV ===
def load_trimmed_csv(filename):
    data = pd.read_csv(filename)
    return data.iloc[:MAX_RECORDS]   # trim to 300s

# === Plotting function per wave type ===
def plot_group(wave_type, file_entries):
    # 1. World-frame Displacement/Velocity/Acceleration
    components = {
        'Displacement': ['disp_x', 'disp_y', 'disp_z'],
        'Velocity':     ['vel_x', 'vel_y', 'vel_z'],
        'Acceleration': ['acc_x', 'acc_y', 'acc_z'],
    }
    fig, axes = plt.subplots(len(components), 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{wave_type.capitalize()} — World Frame")
    for height, f in sorted(file_entries):
        data = load_trimmed_csv(f)
        time = data['time']
        cat = height_category(height)
        for ax, (label, cols) in zip(axes, components.items()):
            for col in cols:
                axis = col[-1]  # last char x,y,z
                color, style = get_color_and_style(height, axis)
                ax.plot(time, data[col], linestyle=style,
                        label=f"H={height} ({cat}) {col}", color=color, alpha=0.8)
            ax.set_ylabel(label)
            ax.grid(True)
    axes[-1].set_xlabel("Time [s]")
    add_combined_legend(fig, axes)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{wave_type}_world.svg", bbox_inches="tight")
    fig.savefig(f"{wave_type}_world.pgf", bbox_inches="tight")
    plt.close(fig)

    # 2. IMU Acceleration
    fig, axarr = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{wave_type.capitalize()} — IMU Acceleration")
    for height, f in sorted(file_entries):
        data = load_trimmed_csv(f)
        time = data['time']
        cat = height_category(height)
        for comp in ['acc_bx', 'acc_by', 'acc_bz']:
            axis = comp[-1]
            color, style = get_color_and_style(height, axis)
            idx = {"x": 0, "y": 1, "z": 2}[axis]
            axarr[idx].plot(time, data[comp], linestyle=style,
                            label=f"H={height} ({cat}) {comp}", color=color, alpha=0.8)
            axarr[idx].set_ylabel(comp)
            axarr[idx].grid(True)
    axarr[-1].set_xlabel("Time [s]")
    add_combined_legend(fig, axarr)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{wave_type}_imu_acc.svg", bbox_inches="tight")
    fig.savefig(f"{wave_type}_imu_acc.pgf", bbox_inches="tight")
    plt.close(fig)

    # 3. Gyro
    fig, axarr = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{wave_type.capitalize()} — IMU Gyro")
    for height, f in sorted(file_entries):
        data = load_trimmed_csv(f)
        time = data['time']
        cat = height_category(height)
        for comp in ['gyro_x', 'gyro_y', 'gyro_z']:
            axis = comp[-1]
            color, style = get_color_and_style(height, axis)
            idx = {"x": 0, "y": 1, "z": 2}[axis]
            axarr[idx].plot(time, data[comp], linestyle=style,
                            label=f"H={height} ({cat}) {comp}", color=color, alpha=0.8)
            axarr[idx].set_ylabel(comp)
            axarr[idx].grid(True)
    axarr[-1].set_xlabel("Time [s]")
    add_combined_legend(fig, axarr)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{wave_type}_gyro.svg", bbox_inches="tight")
    fig.savefig(f"{wave_type}_gyro.pgf", bbox_inches="tight")
    plt.close(fig)

    # 4. Euler Angles
    fig, axarr = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{wave_type.capitalize()} — Euler Angles")
    for height, f in sorted(file_entries):
        data = load_trimmed_csv(f)
        time = data['time']
        cat = height_category(height)
        for comp in ['roll_deg', 'pitch_deg', 'yaw_deg']:
            axis = comp.split("_")[0][0]  # r,p,y
            color, style = get_color_and_style(height, axis)
            idx = ['roll_deg', 'pitch_deg', 'yaw_deg'].index(comp)
            axarr[idx].plot(time, data[comp], linestyle=style,
                            label=f"H={height} ({cat}) {comp}", color=color, alpha=0.8)
            axarr[idx].set_ylabel(comp)
            axarr[idx].grid(True)
    axarr[-1].set_xlabel("Time [s]")
    add_combined_legend(fig, axarr)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{wave_type}_euler.svg", bbox_inches="tight")
    fig.savefig(f"{wave_type}_euler.pgf", bbox_inches="tight")
    plt.close(fig)

# === Run ===
for wave_type, entries in groups.items():
    plot_group(wave_type, entries)

print("All combined plots saved as .svg and .pgf")
