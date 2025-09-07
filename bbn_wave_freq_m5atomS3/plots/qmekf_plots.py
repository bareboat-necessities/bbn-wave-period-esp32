#!/usr/bin/env python3
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory to search (current dir by default)
DATA_DIR = "./"

# Find all *_kalman.csv files
files = glob.glob(os.path.join(DATA_DIR, "*_kalman.csv"))
if not files:
    print("No *_kalman.csv files found in", DATA_DIR)
    exit()

for fname in files:
    print(f"Plotting {fname} ...")
    df = pd.read_csv(fname)

    time = df["time"]

    # Reference vs estimated Euler angles
    angles = [
        ("roll_ref", "roll_est", "Roll (deg)"),
        ("pitch_ref", "pitch_est", "Pitch (deg)"),
        ("yaw_ref", "yaw_est", "Yaw (deg)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(os.path.basename(fname))

    for ax, (ref_col, est_col, label) in zip(axes, angles):
        ax.plot(time, df[ref_col], label="Reference", linewidth=1.5)
        ax.plot(time, df[est_col], label="Estimated", linewidth=1.0, linestyle="--")
        ax.set_ylabel(label)
        ax.grid(True)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
