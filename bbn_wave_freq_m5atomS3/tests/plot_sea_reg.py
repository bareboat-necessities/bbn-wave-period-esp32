

import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# Directory where CSV files from harness are saved
DATA_DIR = "./"   # change if needed

# File pattern for SeaStateRegularity outputs
files = glob.glob(os.path.join(DATA_DIR, "reg_*.csv"))

# Regex to parse filenames like reg_<tracker>_<wave>_h<height>.csv
pattern = re.compile(r"reg_(?P<tracker>[^_]+)_(?P<wave>[^_]+)_h(?P<height>[0-9.]+)\.csv")

# Group files by tracker
tracker_groups = {}

for f in files:
    m = pattern.search(os.path.basename(f))
    if not m:
        print(f"Skipping unrecognized filename: {f}")
        continue

    tracker = m.group("tracker")
    tracker_groups.setdefault(tracker, []).append(f)

# Create one figure per tracker
for tracker, tracker_files in tracker_groups.items():
    plt.figure(figsize=(14, 8))

    for f in tracker_files:
        m = pattern.search(os.path.basename(f))
        wave   = m.group("wave")
        height = m.group("height")

        df = pd.read_csv(f)

        if "regularity" not in df.columns:
            print(f"Skipping {f} (missing 'regularity')")
            continue

        label = f"{wave}-h{height}"
        plt.plot(df["time"], df["regularity"], label=label, alpha=0.6)

    plt.xlabel("Time [s]")
    plt.ylabel("Regularity score (R)")
    plt.title(f"Sea State Regularity — {tracker} tracker")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.show()

