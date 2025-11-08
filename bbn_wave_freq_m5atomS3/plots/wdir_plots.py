#!/usr/bin/env python3
# Rose-only plots for wdir_*.csv, with loud logging and safe backend (Agg)

import argparse
from pathlib import Path
import glob
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ensure non-GUI backend so saving works anywhere
import matplotlib.pyplot as plt
import os
from typing import Optional

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def rose_180(theta_deg: np.ndarray,
             outpath: Path,
             bins: int = 36,
             weights: Optional[np.ndarray] = None,
             title: str = "Directional rose"):
    if theta_deg.size == 0:
        print(f"  [skip] no angles → {outpath}")
        return
    theta = np.deg2rad(np.mod(theta_deg, 180.0))  # [0, pi)
    counts, edges = np.histogram(theta, bins=bins, range=(0.0, math.pi), weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = (edges[1] - edges[0]) * np.ones_like(centers)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(centers, counts, width=widths, align="center")
    ax.set_title(title)
    fig.tight_layout()
    ensure_dir(outpath.parent)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  [write] {outpath}")

def process_file(csv_path: Path, outroot: Path, bins: int):
    outdir = outroot / csv_path.stem
    ensure_dir(outdir)
    print(f"[rose] {csv_path} → {outdir}")

    df = pd.read_csv(csv_path)

    if "dir_deg" not in df.columns:
        print("  [skip] missing 'dir_deg'")
        return

    theta_all = df["dir_deg"].dropna().to_numpy()
    if theta_all.size == 0:
        print("  [skip] 'dir_deg' is empty")
        return

    # 1) plain counts
    rose_180(theta_all, outdir / "dir_rose.png", bins=bins,
             title=f"{csv_path.name} — dir_rose (counts)")

    # 2) confidence-weighted (if present)
    if "dir_conf" in df.columns:
        mask = df["dir_deg"].notna().to_numpy()
        w = df["dir_conf"].fillna(0.0).to_numpy()
        if w.shape[0] == df.shape[0]:
            rose_180(theta_all, outdir / "dir_rose_weighted_conf.png",
                     bins=bins, weights=w[mask],
                     title=f"{csv_path.name} — dir_rose (weighted by dir_conf)")

    # 3) split by sign if present
    if "dir_sign_num" in df.columns:
        sign = df["dir_sign_num"]
        toward = df.loc[(sign == 1) & df["dir_deg"].notna(), "dir_deg"].to_numpy()
        if toward.size:
            rose_180(toward, outdir / "dir_rose_toward.png", bins=bins,
                     title=f"{csv_path.name} — dir_rose (TOWARD)")
        away = df.loc[(sign == -1) & df["dir_deg"].notna(), "dir_deg"].to_numpy()
        if away.size:
            rose_180(away, outdir / "dir_rose_away.png", bins=bins,
                     title=f"{csv_path.name} — dir_rose (AWAY)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="wdir_*.csv",
                    help="glob for input CSV files (use ** for recursion)")
    ap.add_argument("--recursive", action="store_true",
                    help="enable recursive globbing")
    ap.add_argument("--outdir", default="wdir_plots",
                    help="root output directory")
    ap.add_argument("--bins", type=int, default=36,
                    help="number of bins on [0,180)")
    args = ap.parse_args()

    cwd = Path.cwd()
    print(f"[info] cwd = {cwd}")
    print(f"[info] searching: {args.glob}  (recursive={args.recursive})")

    files = sorted(glob.glob(args.glob, recursive=args.recursive))
    print(f"[info] matched {len(files)} file(s)")
    for f in files:
        print(f"  - {f}")

    if not files:
        print("No files matched. Tips:")
        print("  • Run from the folder that contains your wdir_*.csv")
        print(r"  • Or pass a pattern like --glob .\**\wdir_*.csv --recursive")
        print(r"  • Or give an absolute pattern, e.g. --glob D:\path\to\wdir_*.csv")
        return

    outroot = Path(args.outdir)
    for f in files:
        process_file(Path(f), outroot, bins=args.bins)

    print(f"[done] outputs are under: {outroot.resolve()}")

if __name__ == "__main__":
    main()
