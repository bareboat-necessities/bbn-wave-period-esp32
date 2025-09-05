#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import os
import re

# === Matplotlib PGF/LaTeX config ===
from matplotlib.backends.backend_pgf import FigureCanvasPgf

# Register PGF backend
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

# === Groups we care about ===
height_groups = {
    "low":    0.27,
    "medium": 1.50,
    "high":   8.50,
}

# Regex to parse filenames like:
#   wave_spectrum_jonswap_H0.270_L..._A..._P...csv
fname_re = re.compile(
    r"^wave_spectrum_(?P<wtype>[a-z]+)_H(?P<H>[0-9.]+)_L(?P<L>[0-9.]+)_A(?P<A>[-0-9.]+)_P(?P<P>[-0-9.]+)\.csv$"
)

def parse_filename(fname):
    stem = os.path.basename(fname)
    m = fname_re.match(stem)
    if not m:
        return None
    return {
        "wtype": m.group("wtype"),
        "H": float(m.group("H")),
        "L": float(m.group("L")),
        "A": float(m.group("A")),
        "P": float(m.group("P")),
    }

def save_all(fig, base):
    """Save PGF + SVG + PNG (PNG needed for LaTeX sidecar images)."""
    fig.savefig(f"{base}.pgf", bbox_inches="tight", backend="pgf")
    fig.savefig(f"{base}.svg", bbox_inches="tight", dpi=150)
    with mpl.rc_context({"text.usetex": False}):
        fig.savefig(f"{base}.png", bbox_inches="tight", dpi=300)  # critical for LaTeX
    print(f"  saved {base}.pgf/.svg/.png")

def make_plots(fname, group_label, meta):
    df = pd.read_csv(fname)
    freqs = np.sort(df["f_Hz"].unique())
    thetas_deg = np.sort(df["theta_deg"].unique())
    E = df.pivot(index="f_Hz", columns="theta_deg", values="E").values
    thetas_rad = np.deg2rad(thetas_deg)

    title = fr"{meta['wtype'].capitalize()} spectrum ($H_s={meta['H']:.2f}\,\mathrm{{m}}$)"
    base = f"spectrum_{meta['wtype']}_{group_label}"

    # === Polar plot ===
    R, T = np.meshgrid(freqs, thetas_rad)
    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    pcm = ax1.pcolormesh(T, R, E.T, shading='auto', cmap='viridis')
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    ax1.set_title(title)
    plt.colorbar(pcm, ax=ax1, orientation="vertical", label=r"$E(f,\theta)\,[m^2/Hz/deg]$")
    save_all(fig1, f"{base}_polar")
    plt.close(fig1)

    # === 3D surface plot ===
    T_grid, F_grid = np.meshgrid(thetas_deg, freqs)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    surf = ax2.plot_surface(F_grid, T_grid, E, cmap='viridis', linewidth=0, antialiased=True)
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Direction [deg]")
    ax2.set_zlabel(r"$E(f,\theta)$")
    ax2.set_title(title)
    fig2.colorbar(surf, shrink=0.5, aspect=10, label=r"$E(f,\theta)$")
    save_all(fig2, f"{base}_3d")
    plt.close(fig2)

if __name__ == "__main__":
    files = glob.glob("wave_spectrum_*.csv")
    if not files:
        print("No wave_spectrum_*.csv files found.")
        exit()

    for fname in files:
        meta = parse_filename(fname)
        if not meta:
            continue
        for group, target_H in height_groups.items():
            if abs(meta["H"] - target_H) < 1e-3:
                print(f"Processing {fname} as {group} ...")
                make_plots(fname, group, meta)

    print("All PGF, SVG, and PNG plots saved.")
