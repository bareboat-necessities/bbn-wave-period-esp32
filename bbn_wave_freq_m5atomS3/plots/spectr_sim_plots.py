import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def load_spectrum_csv(filename):
    return pd.read_csv(filename)

def plot_block_spectrum(df, filename, block_idx=0):
    # find frequency columns
    est_cols = [c for c in df.columns if c.startswith("S_eta_est_f")]
    ref_cols = [c for c in df.columns if c.startswith("S_eta_ref_f")]

    # parse frequencies from header labels "S_eta_est_f{i}_Hz=val"
    freqs = []
    for c in est_cols:
        if "Hz=" in c:
            hz = float(c.split("Hz=")[1])
            freqs.append(hz)
        else:
            freqs.append(len(freqs))  # fallback index
    freqs = pd.Series(freqs)

    est = df.loc[block_idx, est_cols].values
    ref = df.loc[block_idx, ref_cols].values

    plt.figure()
    plt.loglog(freqs, est, "o-", label="Estimated")
    plt.loglog(freqs, ref, "x--", label="Reference")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("S_eta [m^2/Hz]")
    plt.title(f"{os.path.basename(filename)} — Block {block_idx}")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

def plot_summary(files):
    plt.figure()
    for fname in files:
        df = pd.read_csv(fname)
        plt.plot(df["time"], df["Hs"], label=os.path.basename(fname))
    plt.xlabel("Time [s]")
    plt.ylabel("Hs [m]")
    plt.title("Significant Wave Height over Time")
    plt.grid(True)
    plt.legend()

    plt.figure()
    for fname in files:
        df = pd.read_csv(fname)
        plt.plot(df["time"], df["Fp"], label=os.path.basename(fname))
    plt.xlabel("Time [s]")
    plt.ylabel("Fp [Hz]")
    plt.title("Peak Frequency over Time")
    plt.grid(True)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot spectrum CSVs.")
    parser.add_argument("--block", type=int, default=None,
                        help="Block index to plot (single file mode)")
    parser.add_argument("--file", type=str, default=None,
                        help="Single spectrum CSV file (for block spectrum)")
    args = parser.parse_args()

    # auto-discover spectrum CSVs
    files = sorted(glob.glob("spectrum_*.csv"))

    if args.file and args.block is not None:
        df = load_spectrum_csv(args.file)
        plot_block_spectrum(df, args.file, args.block)
    else:
        if not files:
            print("No spectrum_*.csv files found.")
        else:
            print("Plotting summary for files:", files)
            plot_summary(files)
