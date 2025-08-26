import pandas as pd
import matplotlib.pyplot as plt

# Filenames and wave type colors
files = {
    'Short waves': 'short_waves_stokes.csv',
    'Medium waves': 'medium_waves_stokes.csv',
    'Long waves': 'long_waves_stokes.csv',
}

# Colors for each wave type and component (x,y,z)
wave_colors = {
    'Short waves': ['#e41a1c', '#a50f15', '#fb6a4a'],   # Reds
    'Medium waves': ['#4daf4a', '#2b7a2b', '#a1d99b'],  # Greens
    'Long waves': ['#377eb8', '#184f7d', '#9ecae1'],    # Blues
}

components = {
    'Displacement': ['disp_x', 'disp_y', 'disp_z'],
    'Velocity': ['vel_x', 'vel_y', 'vel_z'],
    'Acceleration': ['acc_x', 'acc_y', 'acc_z'],
}

fig, axes = plt.subplots(len(components), 1, figsize=(14, 10), sharex=True)
fig.suptitle('Wave Simulation Data from Jonswap3dStokesWaves')

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

# Create a custom legend with only one entry per wave type and component
# (optional: or show full legend)

handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

# Remove duplicates while preserving order
seen = set()
unique = []
for h, l in zip(handles, labels):
    if l not in seen:
        unique.append((h, l))
        seen.add(l)

handles, labels = zip(*unique)
axes[0].legend(handles, labels, loc='upper right', fontsize='small', ncol=3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
