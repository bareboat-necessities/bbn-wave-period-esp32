import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import numpy as np

# Configure matplotlib to use LaTeX fonts and export PGF
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

def plot_fenton_wave():
    data = pd.read_csv("wave_data.csv")

    plt.figure(figsize=(10, 4))
    plt.plot(data['x'], data['elevation'], label=r'Surface Elevation $\eta(x)$', color='blue')
    plt.xlabel("Horizontal Position $x$ (m)")
    plt.ylabel(r'Surface Elevation $\eta$ (m)')
    plt.title("Fenton Wave Profile")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("wave_profile.pgf", bbox_inches='tight')
    plt.close()

def plot_wave_kinematics():
    try:
        data = pd.read_csv("wave_tracker_data.csv")
    except FileNotFoundError:
        print("File 'wave_tracker_data.csv' not found.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax1.plot(data['Time(s)'][1:], data['Displacement(m)'][1:], 'b-', label='Surface Elevation')
    ax1.set_ylabel('Displacement (m)')
    ax1.set_title('Surface Float Kinematics')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(data['Time(s)'][1:], data['Velocity(m/s)'][1:], 'r-', label='Vertical Velocity')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.grid(True)
    ax2.legend()

    ax3.plot(data['Time(s)'], data['Acceleration(m/s²)'], 'g-', label='Vertical Acceleration')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.grid(True)
    ax3.legend()

    stats_text = (
        f"Max displacement: {data['Displacement(m)'].max():.2f} m\n"
        f"Max velocity: {data['Velocity(m/s)'].max():.2f} m/s\n"
        f"Max acceleration: {data['Acceleration(m/s²)'].max():.2f} m/s²"
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))

    plt.tight_layout()
    plt.savefig("wave_kinematics.pgf", bbox_inches='tight')
    plt.close()

def plot_wave_profile_path():
    try:
        data = pd.read_csv("wave_tracker_data.csv")
    except FileNotFoundError:
        print("File 'wave_tracker_data.csv' not found.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(data['X_Position(m)'], data['Displacement(m)'], 'b-')
    plt.xlabel('Horizontal Position (m)')
    plt.ylabel('Surface Elevation (m)')
    plt.title('Wave Profile Along Particle Path')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("wave_particle_path.pgf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_fenton_wave()
    plot_wave_kinematics()
    plot_wave_profile_path()
    print("All PGF plots saved successfully.")
