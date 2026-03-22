#!/usr/bin/env python3
from pathlib import Path
import csv

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = SCRIPT_DIR.parent / "tests"
OUT_DIR = SCRIPT_DIR
SAMPLES_CSV = TEST_DIR / "adaptive_wave_detrender_sim_output.csv"
SUMMARY_CSV = TEST_DIR / "adaptive_wave_detrender_sim_summary.csv"
TIME_WINDOW_S = 80.0
SAMPLE_RATE_HZ = 200
DOWNSAMPLE = 10
HEIGHT_ORDER = [0.27, 1.5, 4.0, 8.5]
FILE_BY_HEIGHT = {
    0.27: "adaptive_wave_detrender_sim_h0_270.pgf",
    1.5: "adaptive_wave_detrender_sim_h1_500.pgf",
    4.0: "adaptive_wave_detrender_sim_h4_000.pgf",
    8.5: "adaptive_wave_detrender_sim_h8_500.pgf",
}


def load_rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def coordinate_block(rows, key):
    lines = []
    for row in rows:
        lines.append(f"({float(row['time_s']):.3f},{float(row[key]):.6f})")
    return "\n".join(lines)


def write_plot(path: Path, rows, case):
    title = (
        rf"$H_s={float(case['height_m']):.2f}\,\mathrm{{m}}$, "
        rf"$T_p={float(case['period_s']):.1f}\,\mathrm{{s}}$, "
        rf"$RMS={float(case['detrended_rms_m']):.3f}\,\mathrm{{m}}$ "
        rf"(gate {float(case['gate_rms_m']):.3f} m)"
    )
    content = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=14.5cm,
  height=5.2cm,
  grid=both,
  grid style={{line width=.1pt, draw=gray!25}},
  major grid style={{line width=.2pt, draw=gray!45}},
  xlabel={{Time [s]}},
  ylabel={{Displacement [m]}},
  title={{{title}}},
  legend style={{at={{(0.5,1.02)}}, anchor=south, legend columns=3, draw=none, /tikz/every even column/.append style={{column sep=0.4cm}}}},
  xmin={float(rows[0]['time_s']):.3f},
  xmax={float(rows[-1]['time_s']):.3f},
]
\\addplot[smooth, line width=0.8pt, color=black!60] coordinates {{
{coordinate_block(rows, 'drifted_z_m')}
}};
\\addlegendentry{{Drifted $z$}}

\\addplot[smooth, line width=0.9pt, color=red!75!black] coordinates {{
{coordinate_block(rows, 'detrended_z_m')}
}};
\\addlegendentry{{AdaptiveWaveDetrender}}

\\addplot[smooth, dashed, line width=0.9pt, color=blue!70!black] coordinates {{
{coordinate_block(rows, 'reference_z_m')}
}};
\\addlegendentry{{Reference $z$}}
\\end{{axis}}
\\end{{tikzpicture}}
"""
    path.write_text(content)


def main() -> None:
    samples = load_rows(SAMPLES_CSV)
    summary = load_rows(SUMMARY_CSV)

    for height_m in HEIGHT_ORDER:
        case = next(row for row in summary if abs(float(row['height_m']) - height_m) < 1.0e-9)
        scenario = case['scenario']
        case_rows = [row for row in samples if row['scenario'] == scenario]
        case_rows = case_rows[-int(TIME_WINDOW_S * SAMPLE_RATE_HZ):]
        case_rows = case_rows[::DOWNSAMPLE]
        out_path = OUT_DIR / FILE_BY_HEIGHT[height_m]
        write_plot(out_path, case_rows, case)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
