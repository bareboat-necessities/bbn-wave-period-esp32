import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV data
data = pd.read_csv("wave_data.csv")

# Plot surface elevation η(x)
plt.figure(figsize=(10, 4))
plt.plot(data['x'], data['elevation'], label='Surface Elevation η(x)', color='blue')

# Label axes
plt.xlabel("Horizontal Position x (m)")
plt.ylabel("Surface Elevation η (m)")
plt.title("Fenton Wave Profile")
plt.grid(True)
plt.legend()

# Optional: set aspect ratio
plt.gca().set_aspect('auto', adjustable='box')

# Show the plot
plt.tight_layout()
plt.show()
