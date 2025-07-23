import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

# Configure matplotlib to use LaTeX fonts and export PGF
mpl.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",  # or xelatex/lualatex if you use those
    "font.family": "serif",
    "text.usetex": True,          # Use LaTeX for text rendering
    "pgf.rcfonts": False,         # Don't setup fonts from rc parameters
})

# Load data
data = pd.read_csv("wave_data.csv")

# Create plot
plt.figure(figsize=(10, 4))
plt.plot(data['x'], data['elevation'], label='Surface Elevation $\eta$(x)', color='blue')
plt.xlabel("Horizontal Position x (m)")
plt.ylabel("Surface Elevation $\eta$ (m)")
plt.title("Fenton Wave Profile")
plt.grid(True)
plt.legend()
plt.gca().set_aspect('auto', adjustable='box')

# Save as PGF
plt.tight_layout()
plt.savefig("wave_profile.pgf", bbox_inches='tight')
