#!/usr/bin/env python3
from pathlib import Path
import csv

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = SCRIPT_DIR.parent / "tests"
OUT_PATH = SCRIPT_DIR / "calibrate_imu_quality.pgf"
SAMPLES_CSV = TEST_DIR / "calibrate_imu_test_output.csv"
SUMMARY_CSV = TEST_DIR / "calibrate_imu_test_summary.csv"


def load_rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def coords(rows, x_key, y_key):
    return "\n".join(f"({float(r[x_key]):.3f},{float(r[y_key]):.6f})" for r in rows)


def gate_value(summary_rows, sensor, metric):
    for row in summary_rows:
        if row["sensor"] == sensor and row["metric"] == metric:
            return float(row["gate"]), float(row["value"])
    raise RuntimeError(f"missing gate for {sensor}/{metric}")


def main() -> None:
    sample_rows = load_rows(SAMPLES_CSV)
    summary_rows = load_rows(SUMMARY_CSV)

    accel_rows = [r for r in sample_rows if r["sensor"] == "accel"]
    mag_rows = [r for r in sample_rows if r["sensor"] == "mag"]
    gyro_rows = [r for r in sample_rows if r["sensor"] == "gyro"]

    accel_gate, accel_val = gate_value(summary_rows, "accel", "norm_rms_mps2")
    mag_gate, mag_val = gate_value(summary_rows, "mag", "norm_rms_uT")
    gyro_gate, _ = gate_value(summary_rows, "gyro", "bias_fit_rms_rads")

    tex = f"""\\begin{{tikzpicture}}
\\begin{{groupplot}}[
  group style={{group size=1 by 3, vertical sep=1.4cm}},
  width=14.5cm,
  height=3.9cm,
  grid=both,
  grid style={{line width=.1pt, draw=gray!25}},
  major grid style={{line width=.2pt, draw=gray!40}},
  xlabel={{Sample index}},
  ylabel near ticks,
]
\\nextgroupplot[
  title={{Accelerometer calibration error norm (RMS={accel_val:.4f}, gate={accel_gate:.4f})}},
  ylabel={{$|\\|a_{{cal}}\\|-g|$ [m/s$^2$]}},
]
\\addplot[line width=0.8pt, color=blue!70!black] coordinates {{
{coords(accel_rows, 'sample', 'error_norm')}
}};
\\addplot[dashed, line width=0.8pt, color=red!70!black] coordinates {{(0,{accel_gate:.6f}) ({len(accel_rows)-1},{accel_gate:.6f})}};
\\addplot[dashed, line width=0.8pt, color=red!70!black] coordinates {{(0,{-accel_gate:.6f}) ({len(accel_rows)-1},{-accel_gate:.6f})}};

\\nextgroupplot[
  title={{Magnetometer calibrated norm error (RMS={mag_val:.4f}, gate={mag_gate:.4f})}},
  ylabel={{$\\|m_{{cal}}\\|-B$ [$\\mu$T]}},
]
\\addplot[line width=0.8pt, color=teal!70!black] coordinates {{
{coords(mag_rows, 'sample', 'error_norm')}
}};
\\addplot[dashed, line width=0.8pt, color=red!70!black] coordinates {{(0,{mag_gate:.6f}) ({len(mag_rows)-1},{mag_gate:.6f})}};
\\addplot[dashed, line width=0.8pt, color=red!70!black] coordinates {{(0,{-mag_gate:.6f}) ({len(mag_rows)-1},{-mag_gate:.6f})}};

\\nextgroupplot[
  title={{Gyroscope residual norm after bias compensation}},
  ylabel={{$\\|\\omega_{{cal}}\\|$ [rad/s]}},
]
\\addplot[line width=0.8pt, color=purple!70!black] coordinates {{
{coords(gyro_rows, 'sample', 'error_norm')}
}};
\\addplot[dashed, line width=0.8pt, color=red!70!black] coordinates {{(0,{gyro_gate:.6f}) ({len(gyro_rows)-1},{gyro_gate:.6f})}};
\\end{{groupplot}}
\\end{{tikzpicture}}
"""
    OUT_PATH.write_text(tex)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
