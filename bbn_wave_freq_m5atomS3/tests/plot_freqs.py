import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Adjust this to your folder with CSV files
DATA_FOLDER = "./"

# Updated regex to include pmstokes (PN) waves
FNAME_RE = re.compile(
    r'^(aranovskiy|kalmANF|zerocrossing)_(gerstner|jonswap|fenton|pmstokes)_h([0-9.]+)\.csv$'
)

# Load all CSV files
def load_all_data(folder):
    data = []
    for filepath in glob.glob(os.path.join(folder, "*.csv")):
        fname = os.path.basename(filepath)
        m = FNAME_RE.match(fname)
        if not m:
            continue
        tracker, wave, height_str = m.groups()
        height = float(height_str)
        df = pd.read_csv(filepath)
        df['tracker'] = tracker
        df['wave'] = wave
        df['height'] = height
        data.append(df)
    if not data:
        raise RuntimeError("No matching CSV files found in folder")
    return pd.concat(data, ignore_index=True)

# Plot for each (wave, height) scenario all trackers on same figure
def plot_scenarios(df):
    scenarios = df.groupby(['wave', 'height'])
    for (wave, height), group in scenarios:
        plt.figure(figsize=(12, 6))
        plt.title(f"Frequency Tracking Comparison\nWave: {wave} Height: {height:.3f} m")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")

        for tracker in group['tracker'].unique():
            subset = group[group['tracker'] == tracker]

            # Plot raw estimate freq
            plt.plot(subset['time'], subset['est_freq'], label=f"{tracker} raw")

            # Plot smoothed freq
            plt.plot(subset['time'], subset['smooth_freq'], linestyle='--', label=f"{tracker} smooth")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Optional: plot error for a scenario (wave, height)
def plot_errors(df, wave, height):
    subset = df[(df['wave'] == wave) & (df['height'] == height)]
    plt.figure(figsize=(12,6))
    plt.title(f"Frequency Tracking Errors\nWave: {wave} Height: {height:.3f} m")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency Error (Hz)")

    for tracker in subset['tracker'].unique():
        tr_data = subset[subset['tracker'] == tracker]
        plt.plot(tr_data['time'], tr_data['error'], label=f"{tracker} raw error")
        plt.plot(tr_data['time'], tr_data['smooth_error'], linestyle='--', label=f"{tracker} smooth error")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    df = load_all_data(DATA_FOLDER)
    print(f"Loaded {len(df)} rows from CSV files.")
    plot_scenarios(df)

    # Uncomment to plot error for a specific wave/height
    # plot_errors(df, 'pmstokes', 0.75)

if __name__ == "__main__":
    main()
