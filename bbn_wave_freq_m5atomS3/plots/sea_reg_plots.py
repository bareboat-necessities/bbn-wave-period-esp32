#!/usr/bin/env python3
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import os

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

# === Directory where CSV files are saved ===
DATA_DIR = "./"

# === Sampling cutoff ===
SAMPLE_RATE = 240
MAX_TIME = 600.0
MAX_RECORDS = int(SAMPLE_RATE * MAX_TIME)

# === Groups we care about (included heights in meters) ===
height_groups = {
    "low":    0.27,
    "medium": 1.50,
    "high":   8.50,
}
INCLUDED_HEIGHTS = set(height_groups.values())

# === Match C++ output ===
files = glob.glob(os.path.join(DATA_DIR, "regularity_*.csv"))

pattern = re.compile(
    r"regularity_"
    r"(?P<tracker>[^_]+)_"
    r"(?P<wave>[^_]+)_"
    r"H(?P<height>[-0-9\.]+)"
    r"(?:_L(?P<length>[-0-9\.]+))?"
    r"(?:_A(?P<azimuth>[-0-9\.]+))?"
    r"(?:_P(?P<phase>[-0-9\.]+))?"
    r"(?:_N(?P<noise>[-0-9\.]+))?"
    r"(?:_B(?P<bias>[-0-9\.]+))?"
    r"\.csv"
)

# === Map wave type to base color ===
wave_colors = {
    "fenton": "Blues",
    "gerstner": "Purples",
    "jonswap": "Reds",
    "pmstokes": "Greens",
    "cnoidal": "Oranges",
}

# === Utility: escape dangerous LaTeX characters (keep $ for math) ===
def latex_safe(s: str) -> str:
    return (s.replace("&", "\\&")
             .replace("%", "\\%")
             .replace("#", "\\#")
             .replace("_", "\\_"))

# === Utility: save figure ===
def save_all(fig, base, title):
    fig.suptitle(latex_safe(title))
    fig.savefig(f"{base}.pgf", bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight", dpi=150)
    if not os.path.exists(f"{base}.pgf"):
        raise RuntimeError(f"PGF file not written: {base}.pgf")
    if not os.path.exists(f"{base}.svg"):
        raise RuntimeError(f"SVG file not written: {base}.svg")
    print(f"  saved {base}.pgf/.svg")

# === Group files by tracker ===
tracker_groups = {}
for f in files:
    m = pattern.search(os.path.basename(f))
    if not m:
        print(f"Skipping unrecognized filename: {f}")
        continue
    tracker = m.group("tracker")
    tracker_groups.setdefault(tracker, []).append(f)

# === Plot for each tracker ===
for tracker, tracker_files in tracker_groups.items():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Group files by wave type
    wave_grouped = {}
    for f in tracker_files:
        m = pattern.search(os.path.basename(f))
        wave = m.group("wave")
        if wave in {"gerstner", "cnoidal"}:  # skip wave types if desired
            continue
        wave_grouped.setdefault(wave, []).append(f)

    for wave, files_in_wave in wave_grouped.items():
        cmap = plt.get_cmap(wave_colors.get(wave, "gray"))
        n_files = len(files_in_wave)

        for idx, f in enumerate(sorted(files_in_wave)):
            m = pattern.search(os.path.basename(f))
            height_str = m.group("height").rstrip('0').rstrip('.')  # for label
            try:
                height_val = float(m.group("height"))
            except (TypeError, ValueError):
                print(f"Skipping {f} (invalid height value)")
                continue

            # Keep only included heights
            if all(abs(height_val - h) > 1e-6 for h in INCLUDED_HEIGHTS):
                print(f"Skipping {f} (height {height_val} m not in groups)")
                continue

            df = pd.read_csv(f).head(MAX_RECORDS)
            if not {"regularity", "significant_wave_height", "disp_freq_hz"}.issubset(df.columns):
                print(f"Skipping {f} (missing required columns)")
                continue

            color = cmap(0.3 + 0.7 * idx / max(1, n_files - 1))
            label = f"{wave}-H{height_str}"

            # Top: Regularity
            ax1.plot(df["time"], df["regularity"], label=label, alpha=0.8, color=color)

            # Middle: Wave Height Envelope
            ax2.plot(df["time"], df["significant_wave_height"], label=label, alpha=0.8, color=color)

            # Bottom: Displacement Frequency
            ax3.plot(df["time"], df["disp_freq_hz"], label=label, alpha=0.8, color=color)

    # === Formatting ===
    ax1.set_ylabel("Regularity score (R)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8, ncol=3, loc="lower left")  # single legend, solid frame

    ax2.set_ylabel("Wave Height Envelope [m]")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Displacement Frequency [Hz]")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    base = f"seareg_{tracker}"
    save_all(fig, base, f"Sea State Regularity, Height Envelope & Disp. Freq — {tracker} tracker")
    plt.close(fig)
