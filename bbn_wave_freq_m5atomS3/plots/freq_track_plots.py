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
})

# Folder with CSV files
DATA_FOLDER = "./"

# Updated regex for new naming
FNAME_RE = re.compile(
    r'^freq_track_(aranovskiy|kalmanf|zerocross)_(gerstner|jonswap|fenton|pmstokes|cnoidal)_H([0-9.]+)_L([0-9.]+)_A([0-9.]+)_P([0-9.]+)\.csv$'
)

def load_all_data(folder):
    data = []
    for filepath in glob.glob(os.path.join(folder, "*.csv")):
        fname = os.path.basename(filepath)
        m = FNAME_RE.match(fname)
        if not m:
            continue
        tracker, wave, H_str, L_str, A_str, P_str = m.groups()
        df = pd.read_csv(filepath)
        df['tracker'] = tracker
        df['wave'] = wave
        df['height'] = float(H_str)
        df['length'] = float(L_str)
        df['azimuth'] = float(A_str)
        df['phase'] = float(P_str)
        data.append(df)
    if not data:
        raise RuntimeError("No matching CSV files found in folder")
    return pd.concat(data, ignore_index=True)

def save_all(fig, base):
    fig.savefig(f"{base}.pgf", bbox_inches="tight", backend="pgf")
    fig.savefig(f"{base}.svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {base}.pgf/.svg")

def plot_scenarios(df):
    scenarios = df.groupby(['wave', 'height'])
    for (wave, height), group in scenarios:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(f"Frequency Tracking Comparison\nWave: {wave}, Height: {height:.3f} m")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        for tracker in group['tracker'].unique():
            subset = group[group['tracker'] == tracker]
            ax.plot(subset['time'], subset['est_freq'], label=f"{tracker} raw")
            ax.plot(subset['time'], subset['smooth_freq'], linestyle='--', label=f"{tracker} smooth")

        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        base = f"plot_freqtrack_{wave}_H{height:.3f}"
        save_all(fig, base)

def plot_errors(df, wave, height):
    subset = df[(df['wave'] == wave) & (df['height'] == height)]
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_title(f"Frequency Tracking Errors\nWave: {wave}, Height: {height:.3f} m")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency Error (Hz)")

    for tracker in subset['tracker'].unique():
        tr_data = subset[subset['tracker'] == tracker]
        ax.plot(tr_data['time'], tr_data['error'], label=f"{tracker} raw error")
        ax.plot(tr_data['time'], tr_data['smooth_error'], linestyle='--', label=f"{tracker} smooth error")

    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    base = f"plot_error_{wave}_H{height:.3f}"
    save_all(fig, base)

def main():
    df = load_all_data(DATA_FOLDER)
    print(f"Loaded {len(df)} rows from CSV files.")
    plot_scenarios(df)

    # Example: error plot for one case
    # plot_errors(df, 'pmstokes', 0.75)

if __name__ == "__main__":
    main()
