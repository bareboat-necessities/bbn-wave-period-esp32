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
plt.figure(figsize=(10, 14))

plt.subplot(4, 1, 1)
#plt.plot(t, ax, label="Raw $a_x$", alpha=0.5)
plt.plot(t, fax, label="Filtered $a_x$")
#plt.plot(t, ay, label="Raw $a_y$", alpha=0.5)
plt.plot(t, fay, label="Filtered $a_y$")
plt.plot(t, amp, label="Amplitude")
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
plt.xlabel("Time, sec")
plt.ylabel("Phase, rad")

plt.tight_layout()
plt.savefig("wave_dir.pgf", bbox_inches='tight')
plt.close()
