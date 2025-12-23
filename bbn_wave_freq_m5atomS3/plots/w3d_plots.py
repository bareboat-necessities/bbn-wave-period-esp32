#!/usr/bin/env python3
import glob
import os
import re
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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

# === Config ===
DATA_DIR = "./"             # Directory with *_w3d.csv files
SAMPLE_RATE_HZ = 240        # Simulator sample rate
SKIP_TIME_S = 1140.0        # Skip first n seconds (warmup)
PLOT_TIME_S = 60.0          # Plot next m seconds
MAX_TIME_S  = SKIP_TIME_S + PLOT_TIME_S
MAX_ROWS    = int(SAMPLE_RATE_HZ * MAX_TIME_S)

# Toggle error plots
PLOT_ERRORS = False   # set True to enable

# === Groups we care about (included heights in meters) ===
height_groups = {
    "low":    0.27,
    "medium": 1.50,
    "high":   8.50,
}

# === Allowed wave types ===
ALLOWED_WAVES = {"jonswap", "pmstokes"}

# === Regex to extract wave type and height from filename ===
pattern = re.compile(
    r".*?_(?P<wave>[a-zA-Z0-9]+)_H(?P<height>[-0-9\.]+).*?_fusion\.csv$"
)

def latex_safe(s: str) -> str:
    return s.replace("_", r"\_")

# === Helpers ===
def make_subplots(nrows: int, title: str, width: float = 10.0, row_height: float = 2.5, sharex: bool = True):
    """
    Create a figure with nrows stacked subplots, auto-scaling height.
    """
    fig_height = row_height * nrows
    fig, axes = plt.subplots(nrows, 1, figsize=(width, fig_height), sharex=sharex)
    fig.suptitle(title)
    if nrows == 1:
        axes = [axes]
    return fig, axes

def finalize_plot(fig, outbase: str, suffix: str = "", exts=("pgf", "svg")):
    """
    Finalize a plot: layout, save to multiple formats, and close.
    """
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    for ext in exts:
        fig.savefig(f"{outbase}{suffix}.{ext}", format=ext, bbox_inches="tight")
    plt.close(fig)

# === Find all *_w3d.csv files ===
files = glob.glob(os.path.join(DATA_DIR, "*_fusion.csv"))
if not files:
    print("No *_fusion.csv files found in", DATA_DIR)
    exit()

for fname in files:
    basename = os.path.basename(fname)
    m = pattern.match(basename)
    if not m:
        continue
    wave_type = m.group("wave").lower()
    if wave_type not in ALLOWED_WAVES:
        continue
    try:
        height_val = float(m.group("height"))
    except (TypeError, ValueError):
        continue
    group_name = None
    for name, h in height_groups.items():
        if abs(height_val - h) < 1e-6:
            group_name = name
            break
    if group_name is None:
        continue

    outbase = f"w3d_{wave_type}_{group_name}"
    outbase = re.sub(r"[^A-Za-z0-9_\-]", "_", outbase)
    outbase = os.path.join(DATA_DIR, outbase)

    print(f"Plotting {fname} → {outbase} ...")
    df = pd.read_csv(fname, nrows=MAX_ROWS)
    df = df[(df["time"] >= SKIP_TIME_S) & (df["time"] <= MAX_TIME_S)].reset_index(drop=True)
    time = df["time"]

    # === Angles (Reference vs Estimated, optional errors) ===
    nrows = 3 if not PLOT_ERRORS else 6
    fig, axes = make_subplots(nrows, latex_safe(basename))
    for i, (ref_col, est_col, label) in enumerate([
        ("roll_ref", "roll_est", "Roll (deg)"),
        ("pitch_ref", "pitch_est", "Pitch (deg)"),
        ("yaw_ref", "yaw_est", "Yaw (deg)")
    ]):
        if PLOT_ERRORS:
            ax_val = axes[2*i]
            ax_err = axes[2*i + 1]
        else:
            ax_val = axes[i]
            ax_err = None

        ax_val.plot(time, df[ref_col], label="Reference", linewidth=1.5)
        ax_val.plot(time, df[est_col], label="Estimated", linewidth=1.0, linestyle="--")
        ax_val.set_ylabel(latex_safe(label))
        ax_val.grid(True)
        ax_val.legend(loc="upper right")

        if PLOT_ERRORS:
            err_col = {
                "Roll (deg)": "err_roll",
                "Pitch (deg)": "err_pitch",
                "Yaw (deg)": "err_yaw"
            }[label]
            ax_err.plot(time, df[err_col], color="tab:red")
            ax_err.set_ylabel("Error [deg]")
            ax_err.grid(True)

    axes[-1].set_xlabel("Time (s)")
    finalize_plot(fig, outbase)

    # === Angle errors (summary, only if enabled) ===
    if PLOT_ERRORS:
        error_cols = [
            ("err_roll",  "Roll err [deg]",  "tab:red"),
            ("err_pitch", "Pitch err [deg]", "tab:red"),
            ("err_yaw",   "Yaw err [deg]",   "tab:red"),
            ("angle_err", "Quat err [deg]",  "tab:purple"),
        ]
        nrows = len(error_cols)
        fig, axes = make_subplots(nrows, latex_safe(basename) + " (Angle Errors)")
        for ax, (col, ylabel, color) in zip(axes, error_cols):
            ax.plot(time, df[col], color=color)
            ax.set_ylabel(ylabel)
            ax.grid(True)
        axes[-1].set_xlabel("Time (s)")
        finalize_plot(fig, outbase, "_angle_errs")

    # === Z-axis kinematics ===
    nrows = 6 if PLOT_ERRORS else 3
    fig, axes = make_subplots(nrows, latex_safe(basename) + " (Z-axis)")
    for i, prefix in enumerate(["disp", "vel", "acc"]):
        if PLOT_ERRORS:
            ax_val = axes[2*i]
            ax_err = axes[2*i + 1]
        else:
            ax_val = axes[i]
            ax_err = None

        ax_val.plot(time, df[f"{prefix}_ref_z"], label="Ref")
        ax_val.plot(time, df[f"{prefix}_est_z"], label="Est", linestyle="--")
        ax_val.set_ylabel(f"{prefix.capitalize()} Z")
        ax_val.legend()
        ax_val.grid(True)

        if PLOT_ERRORS:
            ax_err.plot(time, df[f"{prefix}_err_z"], color="tab:red")
            ax_err.set_ylabel("Error")
            ax_err.grid(True)

    axes[-1].set_xlabel("Time (s)")
    finalize_plot(fig, outbase, "_zkin")

    # === XY kinematics ===
    nrows = 6 if PLOT_ERRORS else 3
    fig, axes = make_subplots(nrows, latex_safe(basename) + " (X/Y axes)")
    for i, prefix in enumerate(["disp", "vel", "acc"]):
        if PLOT_ERRORS:
            ax_val = axes[2*i]
            ax_err = axes[2*i + 1]
        else:
            ax_val = axes[i]
            ax_err = None

        ax_val.plot(time, df[f"{prefix}_ref_x"], label="Ref X", color="tab:blue")
        ax_val.plot(time, df[f"{prefix}_est_x"], label="Est X", linestyle="--", color="tab:blue")
        ax_val.plot(time, df[f"{prefix}_ref_y"], label="Ref Y", color="tab:orange")
        ax_val.plot(time, df[f"{prefix}_est_y"], label="Est Y", linestyle="--", color="tab:orange")
        ax_val.set_ylabel(f"{prefix.capitalize()} XY")
        ax_val.legend(ncol=2, fontsize=8)
        ax_val.grid(True)

        if PLOT_ERRORS:
            ax_err.plot(time, df[f"{prefix}_err_x"], label="Err X", color="tab:blue")
            ax_err.plot(time, df[f"{prefix}_err_y"], label="Err Y", color="tab:orange")
            ax_err.set_ylabel("Error")
            ax_err.legend(ncol=2, fontsize=8)
            ax_err.grid(True)

    axes[-1].set_xlabel("Time (s)")
    finalize_plot(fig, outbase, "_xykin")

    # === Accelerometer bias estimates vs true ===
    acc_pairs = [
        ("acc_bias_x",  "acc_bias_est_x",  "Accel bias X"),
        ("acc_bias_y",  "acc_bias_est_y",  "Accel bias Y"),
        ("acc_bias_z",  "acc_bias_est_z",  "Accel bias Z"),
    ]
    fig, axes = make_subplots(len(acc_pairs), latex_safe(basename) + " (Accelerometer Biases)")
    for ax, (true_col, est_col, label) in zip(axes, acc_pairs):
        ax.plot(time, df[true_col], label="True", linewidth=1.5)
        ax.plot(time, df[est_col], label="Estimated", linestyle="--", linewidth=1.0)
        ax.set_ylabel(latex_safe(label))
        ax.grid(True)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    finalize_plot(fig, outbase, "_acc_bias")

    # === Gyroscope bias estimates vs true ===
    gyro_pairs = [
        ("gyro_bias_x", "gyro_bias_est_x", "Gyro bias X"),
        ("gyro_bias_y", "gyro_bias_est_y", "Gyro bias Y"),
        ("gyro_bias_z", "gyro_bias_est_z", "Gyro bias Z"),
    ]
    fig, axes = make_subplots(len(gyro_pairs), latex_safe(basename) + " (Gyroscope Biases)")
    for ax, (true_col, est_col, label) in zip(axes, gyro_pairs):
        ax.plot(time, df[true_col], label="True", linewidth=1.5)
        ax.plot(time, df[est_col], label="Estimated", linestyle="--", linewidth=1.0)
        ax.set_ylabel(latex_safe(label))
        ax.grid(True)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    finalize_plot(fig, outbase, "_gyro_bias")

    # === Magnetometer bias estimates vs true (+ errors) ===
    mag_pairs = [
        ("mag_bias_x", "mag_bias_est_x", "Mag bias X"),
        ("mag_bias_y", "mag_bias_est_y", "Mag bias Y"),
        ("mag_bias_z", "mag_bias_est_z", "Mag bias Z"),
    ]

    # Only plot if the required columns exist
    needed = [c for t,e,_ in mag_pairs for c in (t,e)]
    if all(c in df.columns for c in needed):
        nrows = 2 * len(mag_pairs) if PLOT_ERRORS else len(mag_pairs)
        fig, axes = make_subplots(nrows, latex_safe(basename) + " (Magnetometer Biases)")

        for i, (true_col, est_col, label) in enumerate(mag_pairs):
            if PLOT_ERRORS:
                ax_val = axes[2*i]
                ax_err = axes[2*i + 1]
            else:
                ax_val = axes[i]
                ax_err = None

            ax_val.plot(time, df[true_col], label="True", linewidth=1.5)
            ax_val.plot(time, df[est_col], label="Estimated", linestyle="--", linewidth=1.0)
            ax_val.set_ylabel(latex_safe(label) + r" [$\mu$T]")
            ax_val.grid(True)
            ax_val.legend(loc="upper right")

            if PLOT_ERRORS:
                err = df[est_col] - df[true_col]
                ax_err.plot(time, err, color="tab:red")
                ax_err.set_ylabel(r"Error [$\mu$T]")
                ax_err.grid(True)

        axes[-1].set_xlabel("Time (s)")
        finalize_plot(fig, outbase, "_mag_bias")
    else:
        missing = [c for c in needed if c not in df.columns]
        print(f"  (skip mag bias plots; missing columns: {missing})")

    # === Frequency / Tuner ===
    tuner_cols = [
        ("freq_tracker_hz", "Frequency (Hz)"),
        #("Tp_tuner_s",      "Period (s)"),
        ("accel_var_tuner", r"Accel variance ($m^2/s^4$)"),
        ("tau_applied",     r"$\tau$ applied (s)"),
        ("sigma_a_applied", r"$\sigma_a$ applied ($m/s^2$)"),
        ("R_S_applied",     r"$R_S$ applied (m$\cdot$s)"),
    ]

    fig, axes = make_subplots(len(tuner_cols), latex_safe(basename) + " (Frequency / Tuner)")
    for ax, (col, ylabel) in zip(axes, tuner_cols):
        if col not in df.columns:
            ax.text(0.01, 0.5, f"Missing: {col}", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ax.plot(time, df[col], linewidth=1.2)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    axes[-1].set_xlabel("Time (s)")
    finalize_plot(fig, outbase, "_tuner")

    # === Direction ===
    dir_cols = [
        ("dir_deg",        r"Dir (deg, axial)"),
        ("dir_uncert_deg", r"Uncert (deg, ~95%)"),
        ("dir_conf",       "Confidence"),
        #("dir_amp",        "Amplitude"),
        ("dir_sign_num",   "Sign (+1/-1/0)"),
        #("dir_vec_x",      r"Dir vec $x$"),
        #("dir_vec_y",      r"Dir vec $y$"),
    ]

    fig, axes = make_subplots(len(dir_cols), latex_safe(basename) + " (Direction)")
    for ax, (col, ylabel) in zip(axes, dir_cols):
        if col not in df.columns:
            ax.text(0.01, 0.5, f"Missing: {col}", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        if col == "dir_sign_num":
            ax.step(time, df[col], where="post", linewidth=1.2)
            ax.set_yticks([-1, 0, 1])
        else:
            ax.plot(time, df[col], linewidth=1.2)

        ax.set_ylabel(ylabel)
        ax.grid(True)

    axes[-1].set_xlabel("Time (s)")
    finalize_plot(fig, outbase, "_dir")
    
