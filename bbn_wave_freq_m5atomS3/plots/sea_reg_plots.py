#!/usr/bin/env python3
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# Directory where CSV files from sea_reg are saved
DATA_DIR = "./"

# Match C++ output: regularity_<tracker>_<wave>_h<height>.csv
pattern = re.compile(
    r"regularity_(?P<tracker>[^_]+)_(?P<wave>[^_]+)_h(?P<height>[0-9]+(?:\.[0-9]+)?)\.csv"
)

# Map wave type to base colormap
wave_colors = {
    "fenton": "Blues",
    "gerstner": "Purples",
    "jonswap": "Reds",
    "pmstokes": "Greens",
    "cnoidal": "Oranges",
}

# Map wave type & height to target frequency (must match your C++ params)
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
    ("pmstokes", "0.135"): 1.0/3.0,
    ("pmstokes", "0.75"):  1.0/5.7,
    ("pmstokes", "2"):     1.0/8.5,
    ("pmstokes", "4.25"):  1.0/11.4,
    ("pmstokes", "7.4"):   1.0/14.3,
    ("cnoidal", "0.135"):  1.0/3.0,
    ("cnoidal", "0.75"):   1.0/5.7,
    ("cnoidal", "2"):      1.0/8.5,
    ("cnoidal", "4.25"):   1.0/11.4,
    ("cnoidal", "7.4"):    1.0/14.3,
}

# Save function
def save_all(fig, base):
    fig.savefig(f"{base}.pgf", bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight", dpi=150)
    print(f"  saved {base}.pgf/.svg")

# Main plotting
def main():
    files = glob.glob(os.path.join(DATA_DIR, "regularity_*.csv"))

    # Group files by tracker
    tracker_groups = {}
    for f in files:
        m = pattern.search(os.path.basename(f))
        if not m:
            continue
        tracker = m.group("tracker")
        tracker_groups.setdefault(tracker, []).append(f)

    if not tracker_groups:
        raise RuntimeError("No regularity_*.csv files found")

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
                df = df[df["time"] <= 300.0]  # limit to 300s
                if df.empty:
                    continue

                if not {"regularity", "significant_wave_height", "disp_freq_hz"} <= set(df.columns):
                    print(f"Skipping {f} (missing columns)")
                    continue

                color = cmap(0.3 + 0.7 * idx / max(1, n_files - 1))
                label = f"{wave}-h{height}"

                # Top: Regularity
                ax1.plot(df["time"], df["regularity"], label=label, alpha=0.85, color=color)

                # Middle: Wave Height Envelope
                ax2.plot(df["time"], df["significant_wave_height"], label=label, alpha=0.85, color=color)

                # Bottom: Displacement Frequency
                ax3.plot(df["time"], df["disp_freq_hz"], label=label, alpha=0.85, color=color)

                # Add target frequency line
                target_freq = wave_target_freq.get((wave, height))
                if target_freq:
                    ax3.hlines(target_freq, xmin=df["time"].iloc[0], xmax=df["time"].iloc[-1],
                               colors=color, linestyles="dashed", alpha=0.5)

        # Formatting
        ax1.set_ylabel("Regularity (R)")
        ax1.set_title(f"Sea State Regularity, Height Envelope & Disp. Freq — {tracker} tracker")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(fontsize=8, ncol=3)

        ax2.set_ylabel("Wave Height Envelope [m]")
        ax2.grid(True, linestyle="--", alpha=0.5)

        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Displacement Frequency [Hz]")
        ax3.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        base = f"sea_reg_{tracker}"
        save_all(fig, base)
        plt.close(fig)

if __name__ == "__main__":
    main()
