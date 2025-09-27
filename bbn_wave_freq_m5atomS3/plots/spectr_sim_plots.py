import pandas as pd
import matplotlib.pyplot as plt

def load_spectrum_csv(filename):
    df = pd.read_csv(filename)
    return df

def plot_block_spectrum(df, block_idx=0):
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
            freqs.append(len(freqs))  # fallback
    freqs = pd.Series(freqs)

    est = df.loc[block_idx, est_cols].values
    ref = df.loc[block_idx, ref_cols].values

    plt.figure()
    plt.loglog(freqs, est, "o-", label="Estimated")
    plt.loglog(freqs, ref, "x--", label="Reference")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("S_eta [m^2/Hz]")
    plt.title(f"Spectrum (Block {block_idx})")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

def plot_timeseries(df):
    plt.figure()
    plt.plot(df["time"], df["Hs"], label="Hs")
    plt.xlabel("Time [s]")
    plt.ylabel("Hs [m]")
    plt.title("Significant Wave Height over Time")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(df["time"], df["Fp"], label="Fp")
    plt.xlabel("Time [s]")
    plt.ylabel("Fp [Hz]")
    plt.title("Peak Frequency over Time")
    plt.grid(True)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot spectrum CSVs.")
    parser.add_argument("csvfile", help="spectrum_*.csv file")
    parser.add_argument("--block", type=int, default=0, help="Block index to plot")
    args = parser.parse_args()

    df = load_spectrum_csv(args.csvfile)

    plot_block_spectrum(df, args.block)
    plot_timeseries(df)
