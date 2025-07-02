import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def visualize_wave_data(csv_file='wave_tracker_data.csv'):
    # Read the data from CSV file
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found. Please run the FentonWave_test first.")
        return

    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot displacement
    ax1.plot(data['Time(s)'][1:], data['Displacement(m)'][1:], 'b-', label='Surface Elevation')
    ax1.set_ylabel('Displacement (m)')
    ax1.set_title('Wave Surface Kinematics')
    ax1.grid(True)
    ax1.legend()

    # Plot velocity
    ax2.plot(data['Time(s)'][1:], data['Velocity(m/s)'][1:], 'r-', label='Vertical Velocity')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.grid(True)
    ax2.legend()

    # Plot acceleration
    ax3.plot(data['Time(s)'], data['Acceleration(m/s²)'], 'g-', label='Vertical Acceleration')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.grid(True)
    ax3.legend()

    # Add some statistics to the plots
    stats_text = (
        f"Max displacement: {data['Displacement(m)'].max():.2f} m\n"
        f"Max velocity: {data['Velocity(m/s)'].max():.2f} m/s\n"
        f"Max acceleration: {data['Acceleration(m/s²)'].max():.2f} m/s²"
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))

    plt.tight_layout()
    plt.show()

    # Optional: Plot the wave profile at a specific time
    plot_wave_profile(data)

def plot_wave_profile(data):
    # For the wave profile, we'll use the X_Position data
    # Since the particle is following the crest, we can plot displacement vs position
    plt.figure(figsize=(10, 5))
    plt.plot(data['X_Position(m)'], data['Displacement(m)'], 'b-')
    plt.xlabel('Horizontal Position (m)')
    plt.ylabel('Surface Elevation (m)')
    plt.title('Wave Profile Along Particle Path')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    visualize_wave_data()
