#!/usr/bin/env python3
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import os
from collections import defaultdict

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

# === Trackers → style (linestyle + marker) ===
tracker_styles = {
    "goertzel":   {"linestyle": "-",  "marker": "o"},
    "hilbert":    {"linestyle": "--", "marker": "s"},
    "fft":        {"linestyle": "-.", "marker": "d"},
    "montecarlo": {"linestyle": ":",  "marker": "^"},
}
default_style = {"linestyle": "-", "marker": None}

# === Utility: escape dangerous LaTeX characters (keep $ for math) ===
def latex_safe(s: str) -> str:
    return (s.replace("&", "\\&")
             .replace("%", "\\%")
             .replace("#", "\\#")
             .replace("_", "\\_"))

# === Utility: save figure ===
def save_all(fig, base, title):
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(latex_safe(title))
    fig.savefig(f"{base}.pgf", bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight", dpi=150)
    if not os.path.exists(f"{base}.pgf"):
        raise RuntimeError(f"PGF file not written: {base}.pgf")
    if not os.path.exists(f"{base}.svg"):
        raise RuntimeError(f"SVG file not written: {base}.svg")
    print(f"  saved {base}.pgf/.svg")

# === Collect metadata for all files ===
records = []
for f in files:
    m = pattern.search(os.path.basename(f))
    if not m:
        continue
    try:
        height_val = float(m.group("height"))
    except (TypeError, ValueError):
        continue
    if all(abs(height_val - h) > 1e-6 for h in INCLUDED_HEIGHTS):
        continue
    rec = {
        "file": f,
        "tracker": m.group("tracker"),
        "wave": m.group("wave"),
        "height": height_val,
    }
    records.append(rec)

# === Group by (wave, height) ===
waveheight_groups = defaultdict(list)
for rec in records:
    waveheight_groups[(rec["wave"], rec["height"])].append(rec)

# === Plot each wave/height with all trackers ===
for (wave, height), recs in waveheight_groups.items():
    if wave in {"gerstner", "cnoidal"}:
        continue

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
    cmap = plt.get_cmap(wave_colors.get(wave, "gray"))
    n_files = len(recs)

    for idx, rec in enumerate(sorted(recs, key=lambda r: r["tracker"])):
        df = pd.read_csv(rec["file"]).head(MAX_RECORDS)
        if not {"regularity", "significant_wave_height", "disp_freq_hz"}.issubset(df.columns):
            continue

        color = cmap(0.3 + 0.7 * idx / max(1, n_files - 1))
        style = tracker_styles.get(rec["tracker"], default_style)

        # Legend label = tracker only
        label = f"{rec['tracker']}"

        ax1.plot(df["time"], df["regularity"], label=label,
                 alpha=0.8, color=color,
                 linestyle=style["linestyle"], marker=style["marker"], markersize=3)
        ax2.plot(df["time"], df["significant_wave_height"], label=label,
                 alpha=0.8, color=color,
                 linestyle=style["linestyle"], marker=style["marker"], markersize=3)
        ax3.plot(df["time"], df["disp_freq_hz"], label=label,
                 alpha=0.8, color=color,
                 linestyle=style["linestyle"], marker=style["marker"], markersize=3)

    # === Formatting ===
    ax1.set_ylabel("Regularity score (R)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8, ncol=3, loc="lower left")

    ax2.set_ylabel("Wave Height Envelope [m]")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Displacement Frequency [Hz]")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    h_str = str(height).rstrip("0").rstrip(".")
    base = f"seareg_{wave}_H{h_str}"
    title = f"Sea State Regularity — {wave}, H={h_str} m"
    
    save_all(fig, base, f"Sea State Regularity — {wave}, H={h_str} m (all trackers)")
    plt.close(fig)
