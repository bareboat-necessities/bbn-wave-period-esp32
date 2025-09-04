import pandas as pd
import matplotlib.pyplot as plt

# Filenames and wave type colors for PM Stokes waves
files = {
    'Short PM waves': 'short_pms_waves.csv',
    'Medium PM waves': 'medium_pms_waves.csv',
    'Long PM waves': 'long_pms_waves.csv',
}

# Colors for each wave type and component (x,y,z)
wave_colors = {
    'Short PM waves': ['#e41a1c', '#a50f15', '#fb6a4a'],   # Reds
    'Medium PM waves': ['#4daf4a', '#2b7a2b', '#a1d99b'],  # Greens
    'Long PM waves': ['#377eb8', '#184f7d', '#9ecae1'],    # Blues
}

# === Existing charts: Displacement, Velocity, Acceleration (world frame) ===
components = {
    'Displacement': ['disp_x', 'disp_y', 'disp_z'],
    'Velocity': ['vel_x', 'vel_y', 'vel_z'],
    'Acceleration': ['acc_x', 'acc_y', 'acc_z'],
}

fig, axes = plt.subplots(len(components), 1, figsize=(14, 10), sharex=True)
fig.suptitle('Wave Simulation Data from PMStokesN3dWaves')

for wave_label, filename in files.items():
    data = pd.read_csv(filename)
    time = data['time']
    colors = wave_colors[wave_label]

    for ax, (comp_label, cols) in zip(axes, components.items()):
        for col_name, color in zip(cols, colors):
            ax.plot(time, data[col_name], label=f'{wave_label} {col_name}', color=color, alpha=0.8)
        ax.set_ylabel(comp_label)
        ax.grid(True)

axes[-1].set_xlabel('Time [s]')

# Custom legend without duplicates
handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
seen = set()
unique = []
for h, l in zip(handles, labels):
    if l not in seen:
        unique.append((h, l))
        seen.add(l)
handles, labels = zip(*unique)
axes[0].legend(handles, labels, loc='upper right', fontsize='small', ncol=3)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# === New chart 1: IMU body-frame acceleration ===
fig1, ax1 = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
fig1.suptitle('IMU Body-frame Acceleration (PM Waves)')

for wave_label, filename in files.items():
    data = pd.read_csv(filename)
    time = data['time']
    colors = wave_colors[wave_label]
    for i, comp in enumerate(['accel_x', 'accel_y', 'accel_z']):
        ax1[i].plot(time, data[comp], label=f'{wave_label} {comp}', color=colors[i], alpha=0.8)
        ax1[i].set_ylabel(comp)
        ax1[i].grid(True)

ax1[-1].set_xlabel('Time [s]')
ax1[0].legend(loc='upper right', fontsize='small', ncol=3)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# === New chart 2: IMU gyro ===
fig2, ax2 = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
fig2.suptitle('IMU Gyro (Angular Velocity, PM Waves)')

for wave_label, filename in files.items():
    data = pd.read_csv(filename)
    time = data['time']
    colors = wave_colors[wave_label]
    for i, comp in enumerate(['gyro_x', 'gyro_y', 'gyro_z']):
        ax2[i].plot(time, data[comp], label=f'{wave_label} {comp}', color=colors[i], alpha=0.8)
        ax2[i].set_ylabel(comp)
        ax2[i].grid(True)

ax2[-1].set_xlabel('Time [s]')
ax2[0].legend(loc='upper right', fontsize='small', ncol=3)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# === New chart 3: Euler angles ===
fig3, ax3 = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
fig3.suptitle('IMU Euler Angles (degrees, constrained yaw, PM Waves)')

for wave_label, filename in files.items():
    data = pd.read_csv(filename)
    time = data['time']
    colors = wave_colors[wave_label]
    for i, comp in enumerate(['roll_deg', 'pitch_deg', 'yaw_deg']):
        ax3[i].plot(time, data[comp], label=f'{wave_label} {comp}', color=colors[i], alpha=0.8)
        ax3[i].set_ylabel(comp)
        ax3[i].grid(True)

ax3[-1].set_xlabel('Time [s]')
ax3[0].legend(loc='upper right', fontsize='small', ncol=3)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()
