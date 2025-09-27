import pandas as pd
import matplotlib.pyplot as plt
import glob, os

def plot_last_block(files):
    for fname in files:
        df = pd.read_csv(fname)

        # columns for spectra
        est_cols = [c for c in df.columns if c.startswith("S_eta_est_f")]
        ref_cols = [c for c in df.columns if c.startswith("S_eta_ref_f")]

        # robust: drop incomplete rows, then pick the row with the largest 'block'
        df_clean = df.dropna(subset=est_cols)
        if df_clean.empty:
            print(f"[warn] no complete spectra in {fname}")
            continue

        if "block" in df_clean.columns:
            last_row = df_clean.loc[df_clean["block"].idxmax()]
            block_id = int(last_row["block"])
        else:
            # fallback to last physical row
            last_row = df_clean.iloc[-1]
            block_id = len(df_clean) - 1

        # parse frequencies from header labels "S_eta_est_f{i}_Hz=val"
        freqs = []
        for c in est_cols:
            freqs.append(float(c.split("Hz=")[1]) if "Hz=" in c else len(freqs))

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

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Plot LAST block (one chart per wave).")
    p.add_argument("--file", type=str, default=None, help="Single spectrum CSV file")
    args = p.parse_args()

    files = [args.file] if args.file else sorted(glob.glob("spectrum_*.csv"))
    if not files:
        print("No spectrum_*.csv files found.")
    else:
        print("Plotting last block for files:", files)
        plot_last_block(files)
