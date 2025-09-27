import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def load_spectrum_csv(filename):
    return pd.read_csv(filename)

def plot_last_block(files):
    for fname in files:
        df = pd.read_csv(fname)

        # spectrum columns
        est_cols = [c for c in df.columns if c.startswith("S_eta_est_f")]
        ref_cols = [c for c in df.columns if c.startswith("S_eta_ref_f")]

        if not est_cols or not ref_cols:
            print(f"[warn] no spectrum columns in {fname}")
            continue

        # drop incomplete rows, then pick the row with the largest block
        df_clean = df.dropna(subset=est_cols)
        if df_clean.empty:
            print(f"[warn] no complete spectra in {fname}")
            continue

        if "block" in df_clean.columns:
            last_row = df_clean.loc[df_clean["block"].idxmax()]
            block_id = int(last_row["block"])
        else:
            last_row = df_clean.iloc[-1]
            block_id = len(df_clean) - 1

        # parse freqs from headers
        freqs = []
        for c in est_cols:
            if "Hz=" in c:
                hz = float(c.split("Hz=")[1])
                freqs.append(hz)
            else:
                freqs.append(len(freqs))

        est = last_row[est_cols].to_numpy(dtype=float)
        ref = last_row[ref_cols].to_numpy(dtype=float)
        Hs  = float(last_row["Hs"])
        Fp  = float(last_row["Fp"])

        plt.figure()
        plt.loglog(freqs, est, "o-", label="Estimated (last block)")
        plt.loglog(freqs, ref, "x--", label="Reference")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("S_eta [m^2/Hz]")
        plt.title(f"{os.path.basename(fname)} — Block {block_id} (Hs={Hs:.2f} m, Fp={Fp:.3f} Hz)")
        plt.grid(True, which="both")
        plt.legend()

    plt.show()

def discover_files():
    files = sorted(glob.glob("spectrum_*.csv"))
    # keep only JONSWAP or PM files
    return [f for f in files if ("JONSWAP" in f.upper() or "PM" in f.upper())]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot last-block spectrum CSVs (only JONSWAP/PM).")
    parser.add_argument("--file", type=str, default=None,
                        help="Single spectrum CSV file")
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = discover_files()

    if not files:
        print("No JONSWAP/PM spectrum_*.csv files found.")
    else:
        print("Plotting last block for files:", files)
        plot_last_block(files)
