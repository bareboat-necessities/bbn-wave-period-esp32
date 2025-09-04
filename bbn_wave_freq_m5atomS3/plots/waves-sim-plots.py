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

# === Explicit list of wave heights to plot (reserve) ===
SELECTED_HEIGHTS = [0.27, 1.5, 4.0]   # change as needed

# === Auto-discover CSV files ===
files = glob.glob("wave_data_*.csv")

if not files:
    raise FileNotFoundError("No wave_data_*.csv files found in current directory")

# === Filter by height in filename ===
def extract_height(filename: str) -> float:
    base = os.path.basename(filename)
    parts = base.split("_")
    for p in parts:
        if p.startswith("H"):
            try:
                return float(p[1:])
            except ValueError:
                return None
    return None

files = [f for f in files if extract_height(f) in SELECTED_HEIGHTS]

# === Group files by wave type prefix ===
groups = {}
for f in files:
    name = os.path.basename(f)
    wave_type = name.split("_")[2] if "_" in name else "Unknown"
    groups.setdefault(wave_type, []).append(f)

# === Color palette for components ===
component_colors = {
    'disp': ['#e41a1c', '#a50f15', '#fb6a4a'],  # displacement
    'vel':  ['#4daf4a', '#2b7a2b', '#a1d99b'],  # velocity
    'acc':  ['#377eb8', '#184f7d', '#9ecae1'],  # acceleration
    'imu_acc': ['#ff7f00', '#e6550d', '#fdae6b'],
    'gyro': ['#984ea3', '#6a3d9a', '#bc80bd'],
    'euler': ['#999999', '#555555', '#000000'],
}

# === Plotting function per group ===
def plot_group(wave_type, files):
    for f in files:
        data = pd.read_csv(f)
        time = data['time']
        basename = os.path.splitext(os.path.basename(f))[0]

        # 1. World-frame Displacement/Velocity/Acceleration
        components = {
            'Displacement': (['disp_x', 'disp_y', 'disp_z'], component_colors['disp']),
            'Velocity':     (['vel_x', 'vel_y', 'vel_z'], component_colors['vel']),
            'Acceleration': (['acc_x', 'acc_y', 'acc_z'], component_colors['acc']),
        }
        fig, axes = plt.subplots(len(components), 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"{wave_type.capitalize()} — {basename} (World Frame)")
        for ax, (label, (cols, colors)) in zip(axes, components.items()):
            for col, color in zip(cols, colors):
                ax.plot(time, data[col], label=col, color=color, alpha=0.8)
            ax.set_ylabel(label)
            ax.grid(True)
        axes[-1].set_xlabel("Time [s]")
        axes[0].legend(loc="upper right", fontsize="small", ncol=3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{basename}_world.svg", bbox_inches="tight")
        fig.savefig(f"{basename}_world.pgf", bbox_inches="tight")
        plt.close(fig)

        # 2. IMU Acceleration
        fig, axarr = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"{wave_type.capitalize()} — {basename} (IMU Acceleration)")
        for i, comp in enumerate(['acc_bx', 'acc_by', 'acc_bz']):
            axarr[i].plot(time, data[comp], label=comp, color=component_colors['imu_acc'][i], alpha=0.8)
            axarr[i].set_ylabel(comp)
            axarr[i].grid(True)
        axarr[-1].set_xlabel("Time [s]")
        axarr[0].legend(loc="upper right", fontsize="small", ncol=3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{basename}_imu_acc.svg", bbox_inches="tight")
        fig.savefig(f"{basename}_imu_acc.pgf", bbox_inches="tight")
        plt.close(fig)

        # 3. Gyro
        fig, axarr = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"{wave_type.capitalize()} — {basename} (IMU Gyro)")
        for i, comp in enumerate(['gyro_x', 'gyro_y', 'gyro_z']):
            axarr[i].plot(time, data[comp], label=comp, color=component_colors['gyro'][i], alpha=0.8)
            axarr[i].set_ylabel(comp)
            axarr[i].grid(True)
        axarr[-1].set_xlabel("Time [s]")
        axarr[0].legend(loc="upper right", fontsize="small", ncol=3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{basename}_gyro.svg", bbox_inches="tight")
        fig.savefig(f"{basename}_gyro.pgf", bbox_inches="tight")
        plt.close(fig)

        # 4. Euler Angles
        fig, axarr = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"{wave_type.capitalize()} — {basename} (Euler Angles)")
        for i, comp in enumerate(['roll_deg', 'pitch_deg', 'yaw_deg']):
            axarr[i].plot(time, data[comp], label=comp, color=component_colors['euler'][i], alpha=0.8)
            axarr[i].set_ylabel(comp)
            axarr[i].grid(True)
        axarr[-1].set_xlabel("Time [s]")
        axarr[0].legend(loc="upper right", fontsize="small", ncol=3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{basename}_euler.svg", bbox_inches="tight")
        fig.savefig(f"{basename}_euler.pgf", bbox_inches="tight")
        plt.close(fig)

# === Run ===
for wave_type, flist in groups.items():
    plot_group(wave_type, flist)

print("All plots saved as .svg and .pgf")
