import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def load_spectrum_csv(filename):
    return pd.read_csv(filename)

def plot_last_block(files):
    for fname in files:
        df = pd.read_csv(fname)

        # get last block
        last_idx = len(df) - 1
        est_cols = [c for c in df.columns if c.startswith("S_eta_est_f")]
        ref_cols = [c for c in df.columns if c.startswith("S_eta_ref_f")]

        freqs = []
        for c in est_cols:
            if "Hz=" in c:
                hz = float(c.split("Hz=")[1])
                freqs.append(hz)
            else:
                freqs.append(len(freqs))

        est = df.loc[last_idx, est_cols].values
        ref = df.loc[last_idx, ref_cols].values

        Hs = df.loc[last_idx, "Hs"]
        Fp = df.loc[last_idx, "Fp"]

        plt.figure()
        plt.loglog(freqs, est, "o-", label=f"Estimated (last block)")
        plt.loglog(freqs, ref, "x--", label=f"Reference")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("S_eta [m^2/Hz]")
        plt.title(f"{os.path.basename(fname)} — Last Block (Hs={Hs:.2f} m, Fp={Fp:.3f} Hz)")
        plt.grid(True, which="both")
        plt.legend()

    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot last-block spectrum CSVs.")
    parser.add_argument("--file", type=str, default=None,
                        help="Single spectrum CSV file")
    args = parser.parse_args()

    files = []
    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob("spectrum_*.csv"))

    if not files:
        print("No spectrum_*.csv files found.")
    else:
        print("Plotting last block for files:", files)
        plot_last_block(files)
