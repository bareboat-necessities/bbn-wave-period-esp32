#!/usr/bin/env python3
import glob
import os
import re
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# === PGF / LaTeX friendly output (optional) ===
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

# === Configuration ===
DATA_DIR = "./"
SAMPLE_RATE_HZ = 240
SKIP_TIME_S = 1140.0     # skip initial transient
PLOT_TIME_S = 60.0       # plot next 60 seconds
MAX_TIME_S = SKIP_TIME_S + PLOT_TIME_S
MAX_ROWS = int(SAMPLE_RATE_HZ * MAX_TIME_S)

# === Regex to find valid files ===
pattern = re.compile(r".*_w3d.*\.csv$", re.IGNORECASE)

# === Utility ===
def latex_safe(s):
    return re.sub(r"([_#%$&{}])", r"\\\1", s)

def make_subplots(nrows, title, width=8, row_height=2.0):
    fig_height = row_height * nrows
    fig, axes = plt.subplots(nrows, 1, figsize=(width, fig_height), sharex=True)
    fig.suptitle(title, fontsize=11)
    if nrows == 1:
        axes = [axes]
    return fig, axes

def finalize_plot(fig, outbase, suffix=""):
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    for ext in ("pgf", "svg", "png"):
        fig.savefig(f"{outbase}{suffix}.{ext}", bbox_inches="tight")
    plt.close(fig)

# === Main ===
files = [f for f in glob.glob(os.path.join(DATA_DIR, "*_w3d*.csv")) if pattern.match(f)]
if not files:
    print("No *_w3d*.csv files found.")
    exit()

for fname in sorted(files):
    base = os.path.basename(fname)
    print(f"Plotting {base} ...")
    df = pd.read_csv(fname, nrows=MAX_ROWS)
    if "time" not in df.columns:
        print(f"Skipping {fname}: no 'time' column.")
        continue

    # Subset time range
    df = df[(df["time"] >= SKIP_TIME_S) & (df["time"] <= MAX_TIME_S)].reset_index(drop=True)
    if df.empty:
        print(f"Skipping {fname}: no data after {SKIP_TIME_S}s.")
        continue
    time = df["time"]

    # Required columns (applied values)
    required = ["freq_tracker_hz", "sigma_a_applied", "tau_applied", "R_S_applied"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns in {fname}: {missing}")
        continue

    # === Plot ===
    fig, axes = make_subplots(4, latex_safe(base) + " (Adaptive Parameters)")

    axes[0].plot(time, df["freq_tracker_hz"], color="tab:blue", linewidth=1.2)
    axes[0].set_ylabel(r"$f_\mathrm{tracker}$ [Hz]")
    axes[0].grid(True)

    axes[1].plot(time, df["sigma_a_applied"], color="tab:green", linewidth=1.2)
    axes[1].set_ylabel(r"$\sigma_a$ [m/s$^2$]")
    axes[1].grid(True)

    axes[2].plot(time, df["tau_applied"], color="tab:orange", linewidth=1.2)
    axes[2].set_ylabel(r"$\tau$ [s]")
    axes[2].grid(True)

    axes[3].plot(time, df["R_S_applied"], color="tab:red", linewidth=1.2)
    axes[3].set_ylabel(r"$R_S$")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True)

    outbase = os.path.splitext(fname)[0] + "_params"
    finalize_plot(fig, outbase)

print("Done.")
