#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# === Matplotlib PGF/LaTeX config ===
from matplotlib.backends.backend_pgf import FigureCanvasPgf
mpl.backend_bases.register_backend('pgf', FigureCanvasPgf)

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

# Folder with CSV files
DATA_FOLDER = "./"

# Regex for new file naming convention
FNAME_RE = re.compile(
    r'^freq_track_(aranovskiy|kalmanf|zerocross)_'  # tracker
    r'(gerstner|jonswap|fenton|pmstokes|cnoidal)'   # wave type
    r'_H([0-9.]+)_L([0-9.]+)_A([0-9.]+)_P([0-9.]+)\.csv$'
)

# Load all CSV files
def load_all_data(folder):
    data = []
    for filepath in glob.glob(os.path.join(folder, "*.csv")):
        fname = os.path.basename(filepath)
        m = FNAME_RE.match(fname)
        if not m:
            continue
        tracker, wave, H, L, A, P = m.groups()
        df = pd.read_csv(filepath)
        df['tracker'] = tracker
        df['wave'] = wave
        df['height'] = float(H)
        df['length'] = float(L)
        df['azimuth'] = float(A)
        df['phase'] = float(P)
        data.append(df)
    if not data:
        raise RuntimeError("No matching CSV files found in folder")
    return pd.concat(data, ignore_index=True)

# Save helper: PGF + SVG
def save_all(fig, base):
    fig.savefig(f"{base}.pgf", bbox_inches="tight", backend="pgf")
    fig.savefig(f"{base}.svg", bbox_inches="tight", dpi=150)
    print(f"  saved {base}.pgf/.svg")

# Plot for each (wave, height, azimuth, phase) scenario
def plot_scenarios(df):
    scenarios = df.groupby(['wave', 'height', 'azimuth', 'phase'])
    for (wave, height, azimuth, phase), group in scenarios:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(
            f"Frequency Tracking Comparison\n"
            f"Wave: {wave}, H={height:.3f} m, A={azimuth:.1f}°, P={phase:.1f}°"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        for tracker in group['tracker'].unique():
            subset = group[group['tracker'] == tracker]
            ax.plot(subset['time'], subset['est_freq'], label=f"{tracker} raw")
            ax.plot(subset['time'], subset['smooth_freq'], linestyle='--', label=f"{tracker} smooth")

        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        base = f"freq_plot_{wave}_H{height:.3f}_A{azimuth:.1f}_P{phase:.1f}"
        save_all(fig, base)
        plt.close(fig)

# Optional: plot error for a specific wave/height
def plot_errors(df, wave, height, azimuth, phase):
    subset = df[(df['wave'] == wave) &
                (df['height'] == height) &
                (df['azimuth'] == azimuth) &
                (df['phase'] == phase)]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(
        f"Frequency Tracking Errors\n"
        f"Wave: {wave}, H={height:.3f} m, A={azimuth:.1f}°, P={phase:.1f}°"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency Error (Hz)")

    for tracker in subset['tracker'].unique():
        tr_data = subset[subset['tracker'] == tracker]
        ax.plot(tr_data['time'], tr_data['error'], label=f"{tracker} raw error")
        ax.plot(tr_data['time'], tr_data['smooth_error'], linestyle='--', label=f"{tracker} smooth error")

    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    base = f"freq_error_{wave}_H{height:.3f}_A{azimuth:.1f}_P{phase:.1f}"
    save_all(fig, base)
    plt.close(fig)

def main():
    df = load_all_data(DATA_FOLDER)
    print(f"Loaded {len(df)} rows from CSV files.")
    plot_scenarios(df)

    # Example: plot error for one scenario
    # plot_errors(df, 'pmstokes', 0.75, 30.0, 60.0)

if __name__ == "__main__":
    main()
