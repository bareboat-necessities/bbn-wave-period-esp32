#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

# === Folder with tracker CSV files ===
DATA_FOLDER = "./"

# Regex for naming convention with N and B in input CSV
FNAME_RE = re.compile(
    r'^freq_track_(?P<tracker>aranovskiy|kalmanf|zerocross)'
    r'_(?P<wave>gerstner|jonswap|fenton|pmstokes|cnoidal)'
    r'_H(?P<H>[0-9.]+)_L(?P<L>[0-9.]+)_A(?P<A>[-0-9.]+)_P(?P<P>[-0-9.]+)'
    r'_N(?P<N>[0-9.]+)_B(?P<B>[0-9.]+)\.csv$'
)

# Sampling config
SAMPLE_RATE_HZ = 240
MAX_TIME_S = 300.0
MAX_ROWS = int(SAMPLE_RATE_HZ * MAX_TIME_S)

# === Load all tracker CSV files ===
def load_all_data(folder):
    data = []
    for filepath in glob.glob(os.path.join(folder, "freq_track_*.csv")):
        fname = os.path.basename(filepath)
        m = FNAME_RE.match(fname)
        if not m:
            continue
        tracker = m.group("tracker")
        wave = m.group("wave")
        H = float(m.group("H"))
        L = float(m.group("L"))
        A = float(m.group("A"))
        P = float(m.group("P"))
        N = float(m.group("N"))
        B = float(m.group("B"))

        # Limit to first 300 seconds (≈72k samples)
        df = pd.read_csv(filepath, nrows=MAX_ROWS)
        df['tracker'] = tracker
        df['wave'] = wave
        df['H'] = H
        df['L'] = L
        df['A'] = A
        df['P'] = P
        df['N'] = N
        df['B'] = B
        data.append(df)

    if not data:
        raise RuntimeError("No matching tracker CSV files found in folder")
    return pd.concat(data, ignore_index=True)

# === Save function (PGF, SVG) ===
def save_all(fig, base):
    fig.savefig(f"{base}.pgf", bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight", dpi=150)
    print(f"  saved {base}.pgf/.svg")

# === Plot for each (wave, H) scenario ===
def plot_scenarios(df):
    scenarios = df.groupby(['wave', 'H', 'N', 'B'])
    for (wave, H, N, B), group in scenarios:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(
            f"Frequency Tracking Comparison\n"
            f"Wave: {wave}, H={H:.3f} m, Noise={N:.3f}, Bias={B:.3f}"
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

        base = f"freqtrack_{wave}_H{H:.3f}"
        save_all(fig, base)
        plt.close(fig)

# === Optional: plot errors for a specific scenario ===
def plot_errors(df, wave, H, N, B):
    subset = df[(df['wave'] == wave) & (df['H'] == H) & (df['N'] == N) & (df['B'] == B)]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(
        f"Frequency Tracking Errors\n"
        f"Wave: {wave}, H={H:.3f} m, Noise={N:.3f}, Bias={B:.3f}"
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

    base = f"freqtrack_{wave}_H{H:.3f}_errors"
    save_all(fig, base)
    plt.close(fig)

# === Main ===
def main():
    df = load_all_data(DATA_FOLDER)
    print(f"Loaded {len(df)} rows (first {MAX_TIME_S} seconds) from tracker CSV files.")
    plot_scenarios(df)
    # Example: plot_errors(df, 'pmstokes', 1.500, 0.080, 0.100)

if __name__ == "__main__":
    main()
