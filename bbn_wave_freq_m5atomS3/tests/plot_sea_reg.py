import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import numpy as np

# Directory where CSV files from harness are saved
DATA_DIR = "./"   # change if needed

# Match C++ output: regularity_<tracker>_<wave>_h<height>.csv
files = glob.glob(os.path.join(DATA_DIR, "regularity_*.csv"))

pattern = re.compile(
    r"regularity_(?P<tracker>[^_]+)_(?P<wave>[^_]+)_h(?P<height>[0-9]+(?:\.[0-9]+)?)\.csv"
)

# Map wave type to base color
wave_colors = {
    "fenton": "Blues",
    "gerstner": "Greens",
    "jonswap": "Oranges"
}

# Group files by tracker
tracker_groups = {}
for f in files:
    m = pattern.search(os.path.basename(f))
    if not m:
        print(f"Skipping unrecognized filename: {f}")
        continue
    tracker = m.group("tracker")
    tracker_groups.setdefault(tracker, []).append(f)

# Plot
for tracker, tracker_files in tracker_groups.items():
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Secondary axis for SWH
    ax2 = ax1.twinx()
    ax2.set_ylabel("Significant Wave Height [m]")
    ax2.tick_params(axis='y')

    # Group files by wave type for consistent colormaps
    wave_grouped = {}
    for f in tracker_files:
        m = pattern.search(os.path.basename(f))
        wave   = m.group("wave")
        wave_grouped.setdefault(wave, []).append(f)

    for wave, files_in_wave in wave_grouped.items():
        cmap = plt.get_cmap(wave_colors.get(wave, "gray"))
        n_files = len(files_in_wave)

        for idx, f in enumerate(sorted(files_in_wave)):
            m = pattern.search(os.path.basename(f))
            height = m.group("height").rstrip('0').rstrip('.')  # normalize

            df = pd.read_csv(f)
            if "regularity" not in df.columns or "significant_wave_height" not in df.columns:
                print(f"Skipping {f} (missing required columns)")
                continue

            # Evenly space shades between light and dark
            color = cmap(0.3 + 0.7 * idx / max(1, n_files - 1))
            label = f"{wave}-h{height}"

            # Plot regularity (left axis)
            ax1.plot(df["time"], df["regularity"], label=label, alpha=0.8, color=color)

            # Plot SWH (right axis) with same color, dashed, lighter
            ax2.plot(df["time"], df["significant_wave_height"], 
                     alpha=0.6, color=color, linestyle="--")

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Regularity score (R)")
    ax1.set_title(f"Sea State Regularity & SWH — {tracker} tracker")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    plt.show()
