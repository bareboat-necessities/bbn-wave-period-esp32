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

# Map wave type & height to target frequency
# Must match your C++ waveParamsList
wave_target_freq = {
    ("gerstner", "0.135"): 1.0/3.0,
    ("gerstner", "0.75"):  1.0/5.7,
    ("gerstner", "2"):     1.0/8.5,
    ("gerstner", "4.25"):  1.0/11.4,
    ("gerstner", "7.4"):   1.0/14.3,
    ("jonswap", "0.135"):  1.0/3.0,
    ("jonswap", "0.75"):   1.0/5.7,
    ("jonswap", "2"):      1.0/8.5,
    ("jonswap", "4.25"):   1.0/11.4,
    ("jonswap", "7.4"):    1.0/14.3,
    ("fenton", "0.135"):   1.0/3.0,
    ("fenton", "0.75"):    1.0/5.7,
    ("fenton", "2"):       1.0/8.5,
    ("fenton", "4.25"):    1.0/11.4,
    ("fenton", "7.4"):     1.0/14.3,
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

# Plot for each tracker
for tracker, tracker_files in tracker_groups.items():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Group files by wave type
    wave_grouped = {}
    for f in tracker_files:
        m = pattern.search(os.path.basename(f))
        wave = m.group("wave")
        wave_grouped.setdefault(wave, []).append(f)

    for wave, files_in_wave in wave_grouped.items():
        cmap = plt.get_cmap(wave_colors.get(wave, "gray"))
        n_files = len(files_in_wave)

        for idx, f in enumerate(sorted(files_in_wave)):
            m = pattern.search(os.path.basename(f))
            height = m.group("height").rstrip('0').rstrip('.')  # normalize

            df = pd.read_csv(f)
            if "regularity" not in df.columns or \
                    "significant_wave_height" not in df.columns or \
                    "disp_freq_hz" not in df.columns:
                print(f"Skipping {f} (missing required columns)")
                continue

            color = cmap(0.3 + 0.7 * idx / max(1, n_files - 1))
            label = f"{wave}-h{height}"

            # Top: Regularity
            ax1.plot(df["time"], df["regularity"], label=label, alpha=0.8, color=color)

            # Middle: Wave Height Envelope
            ax2.plot(df["time"], df["significant_wave_height"], label=label, alpha=0.8, color=color)

            # Bottom: Displacement Frequency
            ax3.plot(df["time"], df["disp_freq_hz"], label=label, alpha=0.8, color=color)

            # Add target frequency line
            target_freq = wave_target_freq.get((wave, height))
            if target_freq:
                ax3.hlines(target_freq, xmin=df["time"].iloc[0], xmax=df["time"].iloc[-1],
                           colors=color, linestyles="dashed", alpha=0.5)

    # Formatting
    ax1.set_ylabel("Regularity score (R)")
    ax1.set_title(f"Sea State Regularity, Height Envelope & Disp. Freq — {tracker} tracker")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8, ncol=3)

    ax2.set_ylabel("Wave Height Envelope [m]")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Displacement Frequency [Hz]")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
