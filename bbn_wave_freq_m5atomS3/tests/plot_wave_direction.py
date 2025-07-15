import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("wave_dir.csv")

t = df["t"]
ax = df["ax"]
ay = df["ay"]
fax = df["filtered_ax"]
fay = df["filtered_ay"]
freq = df["frequency"]
amp = df["amplitude"]
phase = df["phase"]
conf = df["confidence"]
deg = df["deg"]

# Plot raw and filtered signals
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(t, ax, label="Raw ax", alpha=0.5)
plt.plot(t, fax, label="Filtered ax")
plt.plot(t, ay, label="Raw ay", alpha=0.5)
plt.plot(t, fay, label="Filtered ay")
plt.legend()
plt.title("Raw vs Filtered Signals")
plt.ylabel("Acceleration")

plt.subplot(4, 1, 2)
plt.plot(t, deg)
plt.title("Estimated Angle")
plt.ylabel("Angle")

plt.subplot(4, 1, 3)
plt.scatter(ax, ay)
plt.scatter(fax, fay)
plt.title("A_est")
plt.xlabel("X")
plt.ylabel("Y")

plt.subplot(4, 1, 4)
plt.plot(t, phase)
plt.title("Estimated Phase")
plt.xlabel("Time (s)")
plt.ylabel("Phase (rad)")


plt.tight_layout()
plt.show()
