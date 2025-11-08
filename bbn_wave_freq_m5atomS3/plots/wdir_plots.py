
#!/usr/bin/env python3
"""
Plot wdir_*.csv outputs from the wave direction runner.

Generates one chart per metric (matplotlib defaults, no custom colors).
Compatible with Matplotlib versions where plt.figure(subplot_kw=...) is unsupported.
"""

import argparse
from pathlib import Path
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def rolling_fraction(x, window):
    # x is integer array of -1,0,+1; returns dict of fractions in rolling window
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    # indicators
    toward = (x > 0).astype(float)
    away   = (x < 0).astype(float)
    unc    = (x == 0).astype(float)
    def roll_mean(a, w):
        if w <= 1:
            return a
        c = np.cumsum(np.insert(a, 0, 0.0))
        out = (c[w:] - c[:-w]) / float(w)
        # pad to same length: prepend first value
        pad = np.full(w-1, out[0] if len(out)>0 else np.nan)
        return np.concatenate([pad, out])
    return roll_mean(toward, window), roll_mean(away, window), roll_mean(unc, window)

def save_line_plot(x, y, xlabel, ylabel, title, outpath):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def save_step_plot(x, y, xlabel, ylabel, title, outpath):
    fig, ax = plt.subplots()
    ax.step(x, y, where='post')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def circular_rose_180(df, col, outpath, bins=36):
    """Polar histogram on [0,180) mapped to [0, pi)."""
    vals = df[col].dropna().to_numpy()
    if vals.size == 0:
        return
    theta = np.deg2rad(vals % 180.0)  # [0, pi)
    # Use subplots() which accepts subplot_kw across Matplotlib versions
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    counts, edges = np.histogram(theta, bins=bins, range=(0, math.pi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    bars = ax.bar(centers, counts, width=(edges[1]-edges[0]), align='center')
    ax.set_title(f"Directional rose ({col})")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def process_csv(csv_path: Path, outdir: Path, window: int):
    df = pd.read_csv(csv_path)
    # Required columns check
    required = ["time","freq_hz","dir_deg","dir_uncert_deg","dir_conf","dir_amp"]
    for c in required:
        if c not in df.columns:
            print(f"[WARN] {csv_path.name}: missing column '{c}', skipping some plots.")
    # time-series plots
    if "time" in df.columns and "freq_hz" in df.columns:
        save_line_plot(df["time"], df["freq_hz"], "time (s)", "freq_hz", csv_path.name + " — freq_hz", outdir / "time_vs_freq.png")
    if "time" in df.columns and "dir_deg" in df.columns:
        save_line_plot(df["time"], df["dir_deg"], "time (s)", "dir_deg (0..180)", csv_path.name + " — dir_deg", outdir / "time_vs_dir_deg.png")
    if "time" in df.columns and "dir_uncert_deg" in df.columns:
        save_line_plot(df["time"], df["dir_uncert_deg"], "time (s)", "uncertainty (deg)", csv_path.name + " — dir_uncert_deg", outdir / "time_vs_dir_uncert_deg.png")
    if "time" in df.columns and "dir_conf" in df.columns:
        save_line_plot(df["time"], df["dir_conf"], "time (s)", "confidence", csv_path.name + " — dir_conf", outdir / "time_vs_dir_conf.png")
    if "time" in df.columns and "dir_amp" in df.columns:
        save_line_plot(df["time"], df["dir_amp"], "time (s)", "amplitude", csv_path.name + " — dir_amp", outdir / "time_vs_dir_amp.png")

    # sign_num step + rolling fractions
    if "time" in df.columns and "dir_sign_num" in df.columns:
        save_step_plot(df["time"], df["dir_sign_num"], "time (s)", "sign_num (-1/0/+1)", csv_path.name + " — dir_sign_num", outdir / "time_vs_dir_sign_num.png")
        # rolling fractions
        w = max(1, int(window))
        toward, away, unc = rolling_fraction(df["dir_sign_num"].to_numpy(), w)
        fig, ax = plt.subplots()
        ax.plot(df["time"], toward, label="toward")
        ax.plot(df["time"], away,   label="away")
        ax.plot(df["time"], unc,    label="uncertain")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"fraction (window={w})")
        ax.set_title(csv_path.name + " — dir sign fractions")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "sign_share_over_time.png", dpi=150)
        plt.close(fig)

    # Polar rose
    if "dir_deg" in df.columns:
        circular_rose_180(df, "dir_deg", outdir / "dir_rose.png", bins=36)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="wdir_*.csv", help="glob pattern for csv files")
    ap.add_argument("--outdir", default="wdir_plots", help="output directory (created per file)")
    ap.add_argument("--window", type=int, default=30, help="rolling window (samples) for sign fractions")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"No files match {args.glob}")
        return

    for f in files:
        csv_path = Path(f)
        outdir = Path(args.outdir) / csv_path.stem
        ensure_outdir(outdir)
        print(f"[plot_wdir] {csv_path} → {outdir}")
        process_csv(csv_path, outdir, args.window)

if __name__ == "__main__":
    main()
