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
height_groups = {
    "low":    0.27,
    "medium": 1.50,
    "high":   8.50,
}
def classify_height(h_val: float) -> str:
    """Return 'low', 'medium', or 'high' depending on closest height."""
    if not h_val or h_val <= 0:
        return "unknown"
    diffs = {lvl: abs(h_val - v) for lvl, v in height_groups.items()}
    return min(diffs, key=diffs.get)

# === Locate all reg_spectrum CSVs ===
files = sorted(glob.glob("reg_spectrum_*.csv"))
if not files:
    print("No reg_spectrum_*.csv files found.")
    exit(0)

# === Extract info from filename ===
pattern = re.compile(
    r"reg_spectrum_"
    r"(?P<tracker>[^_]+)_"
    r"(?P<wave>[A-Za-z0-9]+)"
    r"_H(?P<height>[-0-9\.]+)"
    r"(?:_L(?P<length>[-0-9\.]+))?"
    r"_N(?P<noise>[-0-9\.]+)_B(?P<bias>[-0-9\.]+)\.csv"
)

# === Load CSV ===
def load_spectrum_csv(fname):
    df = pd.read_csv(fname, comment="#")
    if "freq_hz" not in df.columns:
        raise ValueError(f"{fname}: missing freq_hz column")
    return df

# === Plot grouped by wave type and height ===
for f in files:
    m = pattern.search(os.path.basename(f))
    if not m:
        print(f"Skipping unrecognized: {f}")
        continue

    tracker = m.group("tracker")
    wave    = m.group("wave")
    height  = float(m.group("height"))
    length  = m.group("length") or ""
    noise   = m.group("noise")
    bias    = m.group("bias")

    level   = classify_height(height)
    height_str = f"H{height:.2f}".rstrip('0').rstrip('.')
    title_id = f"{wave} ({level} $H_s={height:.2f}$ m)"
    base_out = f"reg_spectrum_plot_{wave}_{level}"

    df = load_spectrum_csv(f)
    print(f"Plotting {wave} {level} → {base_out}.pgf")

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.5), sharex=True)
    axes = axes.flatten()

    # --- 1. Amplitude spectra ---
    axes[0].semilogx(df["freq_hz"], df["A_eta_est"], label=f"{tracker} est", lw=1.8)
    if "A_eta_ref" in df.columns:
        axes[0].semilogx(df["freq_hz"], df["A_eta_ref"], "--", label="ref", lw=1.2)

    # --- 2. Energy density spectra ---
    axes[1].semilogx(df["freq_hz"], df["E_eta_est"], label=f"{tracker} est", lw=1.8)
    if "E_eta_ref" in df.columns:
        axes[1].semilogx(df["freq_hz"], df["E_eta_ref"], "--", label="ref", lw=1.2)

    # --- 3. Cumulative variance fraction ---
    if "CumVar_est" in df.columns and "CumVar_ref" in df.columns:
        est_norm = df["CumVar_est"] / df["CumVar_est"].iloc[-1]
        ref_norm = df["CumVar_ref"] / df["CumVar_ref"].iloc[-1]
        axes[2].semilogx(df["freq_hz"], est_norm, label=f"{tracker} est", lw=1.8)
        axes[2].semilogx(df["freq_hz"], ref_norm, "--", label="ref", lw=1.2)

    # === Labels and layout ===
    axes[0].set_ylabel(r"$A_\eta(f)$ [m]")
    axes[1].set_ylabel(r"$E_\eta(f)=fS_\eta(f)$ [m$^2$]")
    axes[2].set_ylabel("Cumulative variance fraction")
    axes[2].set_xlabel("Frequency [Hz]")
    for ax in axes:
        ax.grid(True, which="both", lw=0.3, ls=":")
        ax.legend(fontsize=7)
        ax.set_xlim(left=0.02, right=2.0)

    fig.suptitle(f"{wave.upper()} — {level.capitalize()} sea ({height_str} m)", fontsize=11, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_pgf = f"{base_out}.pgf"
    plt.savefig(out_pgf, bbox_inches="tight")
    print(f"Saved → {out_pgf}")

    plt.close(fig)

print("\nAll reg_spectrum plots generated successfully.")
