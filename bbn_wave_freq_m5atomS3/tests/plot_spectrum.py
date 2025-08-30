import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os

# === Find all exported spectrum CSVs ===
files = glob.glob("*_jonswap_spectrum.csv")

if not files:
    print("No *_jonswap_spectrum.csv files found in this directory.")
    exit()

for fname in files:
    print(f"Processing {fname} ...")
    df = pd.read_csv(fname)

    # Extract unique freqs and directions
    freqs = np.sort(df["f_Hz"].unique())
    thetas_deg = np.sort(df["theta_deg"].unique())
    M, N = len(thetas_deg), len(freqs)

    # Pivot into 2D grid [freq × theta]
    E = df.pivot(index="f_Hz", columns="theta_deg", values="E").values

    # Convert degrees → radians for polar plot
    thetas_rad = np.deg2rad(thetas_deg)

    # === 1. Polar Heatmap ===
    R, T = np.meshgrid(freqs, thetas_rad)  # R = frequency, T = direction
    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    pcm = ax1.pcolormesh(T, R, E.T, shading='auto', cmap='viridis')
    ax1.set_theta_zero_location("N")  # 0° at North
    ax1.set_theta_direction(-1)       # clockwise
    ax1.set_title(f"Directional Spectrum\n{os.path.basename(fname)}")
    cbar = plt.colorbar(pcm, ax=ax1, orientation="vertical", label="E(f,θ) [m²/Hz/deg]")

    # === 2. 3D Surface Plot ===
    T_grid, F_grid = np.meshgrid(thetas_deg, freqs)  # degrees on axis
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    surf = ax2.plot_surface(F_grid, T_grid, E, cmap='viridis', linewidth=0, antialiased=True)
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Direction [deg]")
    ax2.set_zlabel("E(f,θ)")
    ax2.set_title(f"Directional Spectrum (3D)\n{os.path.basename(fname)}")
    fig2.colorbar(surf, shrink=0.5, aspect=10, label="E(f,θ)")

plt.show()
