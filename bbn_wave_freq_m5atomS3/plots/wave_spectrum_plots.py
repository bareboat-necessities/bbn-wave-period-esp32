#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D projection
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib import rcParams

# ---------- Config ----------
G_STD = 9.81
USE_TEX = True  # set False if you don't have LaTeX installed
OUTPUT_DIR = "spectra_figs"
DPI_SVG = 150
POLAR_CMAP = None  # keep default; set e.g. "viridis" if you want
THETA_ZERO = "N"   # 0° at North
THETA_DIR  = -1    # clockwise

# Register PGF canvas so we can fig.savefig("*.pgf")
matplotlib.backend_bases.register_backend('pgf', FigureCanvasPgf)
if USE_TEX:
    rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}"
    })

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- File discovery ----------
files = sorted(glob.glob("wave_spectrum_*.csv"))
if not files:
    print("No 'wave_spectrum_*.csv' files found.")
    raise SystemExit

# pattern: wave_spectrum_<type>_H<height>_L<length>_A<azimuth>_P<phase>.csv
PAT = re.compile(
    r"^wave_spectrum_(?P<wtype>[a-z]+)"
    r"_H(?P<H>[-+0-9.eE]+)"
    r"_L(?P<L>[-+0-9.eE]+)"
    r"_A(?P<A>[-+0-9.eE]+)"
    r"_P(?P<P>[-+0-9.eE]+)\.csv$"
)

def parse_name(fname):
    stem = os.path.basename(fname)
    m = PAT.match(stem)
    if not m:
        return None
    wtype = m.group("wtype")
    H     = float(m.group("H"))
    L     = float(m.group("L"))
    A     = float(m.group("A"))
    P     = float(m.group("P"))
    # Deep-water approx: T = sqrt(L/g * 2π)
    T     = math.sqrt(L / G_STD * 2.0 * math.pi) if L > 0 else None
    return dict(wtype=wtype, H=H, L=L, A=A, P=P, T=T, stem=stem)

# Parse and group by wave type
meta = []
for f in files:
    info = parse_name(f)
    if info:
        info["path"] = f
        meta.append(info)

if not meta:
    print("No files matched the expected naming pattern.")
    raise SystemExit

# Group by wave type, pick low/med/high by HEIGHT
by_type = {}
for m in meta:
    by_type.setdefault(m["wtype"], []).append(m)

def pick_low_med_high(items):
    # sort by height
    items_sorted = sorted(items, key=lambda x: x["H"])
    Hs = [it["H"] for it in items_sorted]
    if len(items_sorted) <= 3:
        # return unique heights (up to 3)
        out = {lab: items_sorted[min(i, len(items_sorted)-1)]
               for lab, i in zip(["low","med","high"], range(len(items_sorted)))}
        return out
    # choose min, median (index floor), max
    low = items_sorted[0]
    med = items_sorted[len(items_sorted)//2]
    high= items_sorted[-1]
    return {"low": low, "med": med, "high": high}

# ---------- Plot helpers ----------
def prepare_grid(df):
    # pivot to E[f, theta]
    piv = df.pivot(index="f_Hz", columns="theta_deg", values="E")
    freqs = np.array(sorted(piv.index.values))
    thetas_deg = np.array(sorted(piv.columns.values))
    E = piv.loc[freqs, thetas_deg].values
    thetas_rad = np.deg2rad(thetas_deg)
    # mesh for polar (T=angle, R=freqs) and surface (deg)
    T_rad, R = np.meshgrid(thetas_rad, freqs)
    T_deg, F = np.meshgrid(thetas_deg, freqs)
    return freqs, thetas_deg, thetas_rad, E, T_rad, R, T_deg, F

def latex_wave_type(name):
    # pretty print
    mapping = {
        "jonswap": r"\textbf{JONSWAP}",
        "pmstokes": r"\textbf{Pierson--Moskowitz Stokes}",
        "gerstner": r"\textbf{Gerstner}",   # not expected here but ok
        "fenton": r"\textbf{Fenton}",
        "cnoidal": r"\textbf{Cnoidal}"
    }
    return mapping.get(name.lower(), name)

def fig_title(info):
    wt = latex_wave_type(info["wtype"])
    Hs = rf"$H_s={info['H']:.2f}\,\mathrm{{m}}$"
    Tt = rf"$T={info['T']:.2f}\,\mathrm{{s}}$" if info["T"] else r"$T\ \text{n/a}$"
    Az = rf"$\bar{{\alpha}}={info['A']:.0f}^\circ$"
    # Include phase if you like:
    # Ph = rf"$\phi_0={info['P']:.0f}^\circ$"
    return rf"{wt} directional spectrum\quad {Hs}\quad {Tt}\quad {Az}"

def save_both(fig, out_base):
    svg_path = os.path.join(OUTPUT_DIR, out_base + ".svg")
    pgf_path = os.path.join(OUTPUT_DIR, out_base + ".pgf")
    fig.savefig(svg_path, dpi=DPI_SVG, bbox_inches="tight")
    fig.savefig(pgf_path, bbox_inches="tight")
    print(f"  → saved: {svg_path}")
    print(f"  → saved: {pgf_path}")

def make_combined_figure(df, info, sel_label):
    freqs, thetas_deg, thetas_rad, E, T_rad, R, T_deg, F = prepare_grid(df)

    fig = plt.figure(figsize=(11, 4.5))

    # (1) Polar heatmap
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    pcm = ax1.pcolormesh(T_rad, R, E, shading="auto", cmap=POLAR_CMAP)
    ax1.set_theta_zero_location(THETA_ZERO)
    ax1.set_theta_direction(THETA_DIR)
    ax1.set_title(r"Polar heatmap: $E(f,\theta)$", pad=12.0)
    cbar1 = fig.colorbar(pcm, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(r"$E(f,\theta)\,[\mathrm{m}^2/(\mathrm{Hz}\cdot\mathrm{deg})]$")
    ax1.set_rlabel_position(135)

    # (2) 3D surface
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf = ax2.plot_surface(F, T_deg, E, linewidth=0, antialiased=True)
    ax2.set_xlabel(r"Frequency $f$ [Hz]")
    ax2.set_ylabel(r"Direction $\theta$ [deg]")
    ax2.set_zlabel(r"$E(f,\theta)$")
    ax2.set_title(r"3D surface: $E(f,\theta)$")
    cb2 = fig.colorbar(surf, ax=ax2, shrink=0.6, aspect=12, pad=0.10)
    cb2.set_label(r"$E(f,\theta)$")

    # Suptitle with decoded info
    fig.suptitle(fig_title(info), y=1.02)

    # Output basename: spectrum_<type>_H<...>_<low|med|high>
    out_base = f"spectrum_{info['wtype']}_H{info['H']:.3f}_{sel_label}"
    save_both(fig, out_base)
    plt.close(fig)

# ---------- Drive ----------
created = []
for wtype, items in by_type.items():
    picks = pick_low_med_high(items)
    print(f"Wave type '{wtype}': selecting {', '.join(picks.keys())}")
    for sel_label, info in picks.items():
        fpath = info["path"]
        print(f"Processing {sel_label}: {os.path.basename(fpath)}")
        df = pd.read_csv(fpath)
        make_combined_figure(df, info, sel_label)
        created.append((wtype, sel_label, fpath))

print("\nDone. Wrote figures to:", os.path.abspath(OUTPUT_DIR))
