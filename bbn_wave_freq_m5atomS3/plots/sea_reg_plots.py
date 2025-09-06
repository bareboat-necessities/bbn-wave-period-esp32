#!/usr/bin/env python3
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# === Directory where CSV files from harness are saved ===
DATA_DIR = "./"

# === Sampling cutoff ===
SAMPLE_RATE = 240
MAX_TIME = 600.0
MAX_RECORDS = int(SAMPLE_RATE * MAX_TIME)

# === Match C++ output: regularity_<tracker>_<wave>_H..._L..._A..._P...csv ===
pattern = re.compile(
    r"regularity_(?P<tracker>[^_]+)_(?P<wave>[^_]+)"
    r"_H(?P<H>[0-9]+(?:\.[0-9]+)?)"
    r"_L(?P<L>[0-9]+(?:\.[0-9]+)?)"
    r"_A(?P<A>-?[0-9]+(?:\.[0-9]+)?)"
    r"_P(?P<P>-?[0-9]+(?:\.[0-9]+)?)"
    r"(?:_[^.]*)?\.csv"
)

files = glob.glob(os.path.join(DATA_DIR, "regularity_*.csv"))

# === Map wave type to base color ===
wave_colors = {
    "fenton": "Blues",
    "gerstner": "Purples",
    "jonswap": "Reds",
    "pmstokes": "Greens",
    "cnoidal": "Oranges"
}

# === Utility: escape dangerous LaTeX characters, but keep $ for math ===
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
        if not m:
            continue
        wave = m.group("wave")
        if wave == "gerstner":  # skip Gerstner if desired
            continue
        wave_grouped.setdefault(wave, []).append(f)

    for wave, files_in_wave in wave_grouped.items():
        cmap = plt.get_cmap(wave_colors.get(wave, "gray"))
        n_files = len(files_in_wave)

        for idx, f in enumerate(sorted(files_in_wave)):
            m = pattern.search(os.path.basename(f))
            if not m:
                continue

            H = m.group("H")
            L = float(m.group("L"))  # wavelength from filename
            target_freq = 1.0 / L    # compute from wavelength

            height_label = H.rstrip("0").rstrip(".")  # pretty
            label = f"{wave}-h{height_label}"

            df = pd.read_csv(f).head(MAX_RECORDS)
            if not {"regularity", "significant_wave_height", "disp_freq_hz"}.issubset(df.columns):
                print(f"Skipping {f} (missing required columns)")
                continue

            color = cmap(0.3 + 0.7 * idx / max(1, n_files - 1))

            # Top: Regularity
            ax1.plot(df["time"], df["regularity"], label=label, alpha=0.8, color=color)

            # Middle: Wave Height Envelope
            ax2.plot(df["time"], df["significant_wave_height"], label=label, alpha=0.8, color=color)

            # Bottom: Displacement Frequency
            ax3.plot(df["time"], df["disp_freq_hz"], label=label, alpha=0.8, color=color)

            # Add target frequency line
            ax3.hlines(target_freq,
                       xmin=df["time"].iloc[0], xmax=df["time"].iloc[-1],
                       colors=color, linestyles="dashed", alpha=0.5)

    # Formatting
    ax1.set_ylabel("Regularity score (R)")
    ax1.set_title(latex_safe(f"Sea State Regularity, Height Envelope & Disp. Freq — {tracker} tracker"))
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8, ncol=3)

    ax2.set_ylabel("Wave Height Envelope [m]")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Displacement Frequency [Hz]")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    base = f"seareg_{tracker}"
    save_all(fig, base, f"Sea State Regularity, Height Envelope & Disp. Freq — {tracker} tracker")
    plt.close(fig)
