#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl
mpl.use("pgf")
import matplotlib.pyplot as plt
import pandas as pd

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
        r"\providecommand{\mathdefault}[1]{#1}",
    ]),
})

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_CSV_CANDIDATES = (
    SCRIPT_DIR / "detrend.csv",
    SCRIPT_DIR.parent / "data-sim" / "detrend.csv",
)
GENERATED_CSV = SCRIPT_DIR.parent / "tests" / "adaptive_wave_detrender_test_output.csv"
OUT_BASE = SCRIPT_DIR / "adaptive_wave_detrender_performance"


def resolve_existing_file(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path

    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Required file not found. Checked: {joined}")


if __name__ == "__main__":
    reference_csv = resolve_existing_file(*REFERENCE_CSV_CANDIDATES)
    generated_csv = resolve_existing_file(GENERATED_CSV)

    ref = pd.read_csv(reference_csv)
    gen = pd.read_csv(generated_csv)

    time_s = gen["time_s"]

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(time_s, ref["original_from_screenshot_cm"], color="#444444", linewidth=1.4, label=r"Input wave")
    axes[0].plot(time_s, ref["baseline_slow"], color="#999999", linestyle="--", linewidth=1.2, label=r"Reference baseline")
    axes[0].plot(time_s, gen["actual_baseline_slow"], color="#1f77b4", linewidth=1.0, label=r"C++ baseline")
    axes[0].set_ylabel(r"cm")
    axes[0].set_title(r"AdaptiveWaveDetrender baseline tracking")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", ncol=3, fontsize="small")

    axes[1].plot(time_s, ref["wave_clean"], color="#2ca02c", linestyle="--", linewidth=1.2, label=r"Reference detrended wave")
    axes[1].plot(time_s, gen["actual_wave_clean"], color="#d62728", linewidth=1.0, label=r"C++ detrended wave")
    axes[1].set_ylabel(r"cm")
    axes[1].set_title(r"Detrended wave output")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", ncol=2, fontsize="small")

    axes[2].plot(time_s, gen["baseline_abs_error"], color="#1f77b4", linewidth=1.0, label=r"Baseline abs. error")
    axes[2].plot(time_s, gen["wave_clean_abs_error"], color="#d62728", linewidth=1.0, label=r"Wave abs. error")
    axes[2].plot(time_s, gen["wave_freq_abs_error"], color="#9467bd", linewidth=1.0, label=r"Frequency abs. error")
    axes[2].set_xlabel(r"Time [s]")
    axes[2].set_ylabel(r"Abs. error")
    axes[2].set_title(r"Regression error against stored reference")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", ncol=3, fontsize="small")

    plt.tight_layout()
    fig.savefig(OUT_BASE.with_suffix(".pgf"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    print(f"saved {OUT_BASE.with_suffix('.pgf')} / {OUT_BASE.with_suffix('.svg')}")
