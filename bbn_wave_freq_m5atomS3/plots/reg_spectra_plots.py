#!/usr/bin/env python3
import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# === Matplotlib PGF/LaTeX config (optional, comment out for interactive use) ===
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

# === Locate all reg_spectrum CSVs ===
files = sorted(glob.glob("reg_spectrum_*.csv"))
if not files:
    print("No reg_spectrum_*.csv files found.")
    exit(0)

# === Extract wave identifier from filename ===
def extract_wave_id(fname):
    # Example: reg_spectrum_kalmanf_jonswap_H0.27_L14.0_N0.080_B0.100.csv
    m = re.search(r"reg_spectrum_[^_]+_(.+?)_N", os.path.basename(fname))
    if m:
        return m.group(1)
    return os.path.basename(fname)

# === Group by wave (so Aranovskiy, KalmanF, etc. go together) ===
waves = {}
for f in files:
    wid = extract_wave_id(f)
    waves.setdefault(wid, []).append(f)

# === Utility: read spectrum CSV ===
def load_spectrum_csv(fname):
    df = pd.read_csv(fname, comment="#")
    if "freq_hz" not in df.columns:
        raise ValueError(f"{fname}: missing freq_hz column")
    return df

# === Plot all wave groups ===
for wave_id, fnames in waves.items():
    print(f"\n=== Plotting {wave_id} ===")

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.5), sharex=True)
    axes = axes.flatten()

    for fname in fnames:
        tracker = re.search(r"reg_spectrum_([^_]+)_", os.path.basename(fname))
        tracker_name = tracker.group(1) if tracker else "unknown"
        df = load_spectrum_csv(fname)

        # --- 1. Amplitude spectra ---
        axes[0].semilogx(df["freq_hz"], df["A_eta_est"],
                         label=f"{tracker_name} est", lw=1.8)
        if "A_eta_ref" in df.columns:
            axes[0].semilogx(df["freq_hz"], df["A_eta_ref"],
                             "--", label=f"{tracker_name} ref", lw=1.2)

        # --- 2. Energy density spectra ---
        axes[1].semilogx(df["freq_hz"], df["E_eta_est"],
                         label=f"{tracker_name} est", lw=1.8)
        if "E_eta_ref" in df.columns:
            axes[1].semilogx(df["freq_hz"], df["E_eta_ref"],
                             "--", label=f"{tracker_name} ref", lw=1.2)

        # --- 3. Cumulative variance fraction ---
        est_norm = df["CumVar_est"] / df["CumVar_est"].iloc[-1]
        ref_norm = df["CumVar_ref"] / df["CumVar_ref"].iloc[-1]
        axes[2].semilogx(df["freq_hz"], est_norm,
                         label=f"{tracker_name} est", lw=1.8)
        axes[2].semilogx(df["freq_hz"], ref_norm,
                         "--", label=f"{tracker_name} ref", lw=1.2)

    # === Axis labels and limits ===
    axes[0].set_ylabel(r"$A_\eta(f)$ [m]")
    axes[1].set_ylabel(r"$E_\eta(f)=fS_\eta(f)$ [m$^2$]")
    axes[2].set_ylabel("Cumulative variance fraction")
    axes[2].set_xlabel("Frequency [Hz]")

    for ax in axes:
        ax.grid(True, which="both", lw=0.3, ls=":")
        ax.legend(fontsize=7)
        ax.set_xlim(left=0.02, right=2.0)

    fig.suptitle(f"Wave: {wave_id}", fontsize=11, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_pdf = f"reg_spectrum_plot_{wave_id}.pdf"
    plt.savefig(out_pdf)
    print(f"Saved → {out_pdf}")

    # Optional: uncomment to display interactively
    # plt.show()

print("\nAll reg_spectrum plots generated successfully.")
