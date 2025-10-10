#!/usr/bin/env python3
import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

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

# === Height classification ===
height_groups = {"low": 0.27, "medium": 1.50, "high": 8.50}
def classify_height(h):
    diffs = {lvl: abs(h - v) for lvl, v in height_groups.items()}
    return min(diffs, key=diffs.get)

# === Filename pattern ===
pattern = re.compile(
    r"reg_spectrum_"
    r"(?P<tracker>[^_]+)_"            # tracker (aranovskiy, kalmanf, zerocross)
    r"(?P<wave>[A-Za-z0-9]+)_"        # wave type (jonswap, pmstokes, etc.)
    r"H(?P<height>[-0-9\.]+)"         # H0.27
    r"(?:_L(?P<length>[-0-9\.]+))?"   # optional L14.047
    r"(?:_A(?P<azimuth>[-0-9\.]+))?"  # optional A25
    r"(?:_P(?P<phase>[-0-9\.]+))?"    # optional P60
    r"_N(?P<noise>[-0-9\.]+)"         # N0.080
    r"_B(?P<bias>[-0-9\.]+)"          # B0.100
    r"\.csv"
)

# === Load all files and group by (wave, height) ===
groups = {}
for f in sorted(glob.glob("reg_spectrum_*.csv")):
    m = pattern.search(os.path.basename(f))
    if not m:
        print(f"Skipping unrecognized: {f}")
        continue
    wave = m.group("wave")
    height = float(m.group("height"))
    tracker = m.group("tracker")
    key = (wave, height)
    groups.setdefault(key, []).append((tracker, f))

if not groups:
    print("No recognized reg_spectrum_*.csv files found.")
    exit(0)

# === Plot each wave × height group with all trackers ===
for (wave, height), tracker_files in groups.items():
    level = classify_height(height)
    height_str = f"H{height:.2f}".rstrip('0').rstrip('.')
    print(f"\nPlotting {wave} {level} sea ({height_str} m) with {len(tracker_files)} trackers")

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.5), sharex=True)
    axes = axes.flatten()

    for tracker, f in tracker_files:
        df = pd.read_csv(f, comment="#")
        if "freq_hz" not in df.columns:
            print(f"  skipping {f} (no freq_hz)")
            continue

        label = tracker.capitalize()
        lw = 1.8

        # 1. Amplitude spectra
        axes[0].semilogx(df["freq_hz"], df["A_eta_est"], label=f"{label} est", lw=lw)
        if "A_eta_ref" in df.columns and tracker_files.index((tracker, f)) == 0:
            axes[0].semilogx(df["freq_hz"], df["A_eta_ref"], "--", color="gray", label="Reference", lw=1.2)

        # 2. Energy density
        axes[1].semilogx(df["freq_hz"], df["E_eta_est"], label=f"{label} est", lw=lw)
        if "E_eta_ref" in df.columns and tracker_files.index((tracker, f)) == 0:
            axes[1].semilogx(df["freq_hz"], df["E_eta_ref"], "--", color="gray", label="Reference", lw=1.2)

        # 3. Cumulative variance
        if {"CumVar_est", "CumVar_ref"}.issubset(df.columns):
            est_norm = df["CumVar_est"] / df["CumVar_est"].iloc[-1]
            ref_norm = df["CumVar_ref"] / df["CumVar_ref"].iloc[-1]
            axes[2].semilogx(df["freq_hz"], est_norm, label=f"{label} est", lw=lw)
            if tracker_files.index((tracker, f)) == 0:
                axes[2].semilogx(df["freq_hz"], ref_norm, "--", color="gray", label="Reference", lw=1.2)

    # === Axis formatting ===
    axes[0].set_ylabel(r"$A_\eta(f)$ [m]")
    axes[1].set_ylabel(r"$E_\eta(f)=fS_\eta(f)$ [m$^2$]")
    axes[2].set_ylabel("Cumulative variance fraction")
    axes[2].set_xlabel("Frequency [Hz]")
    for ax in axes:
        ax.grid(True, which="both", lw=0.3, ls=":")
        ax.legend(fontsize=7)
        ax.set_xlim(left=0.02, right=2.0)

    fig.suptitle(f"{wave.upper()} — {level.capitalize()} sea ($H_s={height:.2f}$ m)", fontsize=11, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_pgf = f"reg_spectrum_plot_{wave}_{level}.pgf"
    plt.savefig(out_pgf, bbox_inches="tight")
    print(f"  Saved → {out_pgf}")

    plt.close(fig)

print("\nAll combined reg_spectrum plots generated successfully.")
