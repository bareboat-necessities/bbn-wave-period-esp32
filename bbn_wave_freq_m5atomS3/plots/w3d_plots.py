# === Z-axis kinematics (always produce) ===
fig, axes = plt.subplots(3 if not PLOT_ERRORS else 6, 1,
                         figsize=(10, 8 if not PLOT_ERRORS else 12), sharex=True)
fig.suptitle(latex_safe(basename) + " (Z-axis)")

for i, prefix in enumerate(["disp", "vel", "acc"]):
    axes[i].plot(time, df[f"{prefix}_ref_z"], label="Ref")
    axes[i].plot(time, df[f"{prefix}_est_z"], label="Est", linestyle="--")
    axes[i].set_ylabel(f"{prefix.capitalize()} Z")
    axes[i].legend(); axes[i].grid(True)

    if PLOT_ERRORS:
        axes[3+i].plot(time, df[f"{prefix}_err_z"], color="tab:red")
        axes[3+i].set_ylabel("Error")
        axes[3+i].grid(True)

axes[-1].set_xlabel("Time (s)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{outbase}_zkin.pgf", format="pgf", bbox_inches="tight")
plt.savefig(f"{outbase}_zkin.svg", format="svg", bbox_inches="tight")
plt.close(fig)

# === XY kinematics (always produce) ===
fig, axes = plt.subplots(3 if not PLOT_ERRORS else 6, 1,
                         figsize=(10, 8 if not PLOT_ERRORS else 12), sharex=True)
fig.suptitle(latex_safe(basename) + " (X/Y axes)")

for i, prefix in enumerate(["disp", "vel", "acc"]):
    axes[i].plot(time, df[f"{prefix}_ref_x"], label="Ref X", color="tab:blue")
    axes[i].plot(time, df[f"{prefix}_est_x"], label="Est X", linestyle="--", color="tab:blue")
    axes[i].plot(time, df[f"{prefix}_ref_y"], label="Ref Y", color="tab:orange")
    axes[i].plot(time, df[f"{prefix}_est_y"], label="Est Y", linestyle="--", color="tab:orange")
    axes[i].set_ylabel(f"{prefix.capitalize()} XY")
    axes[i].legend(ncol=2, fontsize=8); axes[i].grid(True)

    if PLOT_ERRORS:
        axes[3+i].plot(time, df[f"{prefix}_err_x"], label="Err X", color="tab:blue")
        axes[3+i].plot(time, df[f"{prefix}_err_y"], label="Err Y", color="tab:orange")
        axes[3+i].set_ylabel("Error")
        axes[3+i].legend(ncol=2, fontsize=8); axes[3+i].grid(True)

axes[-1].set_xlabel("Time (s)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{outbase}_xykin.pgf", format="pgf", bbox_inches="tight")
plt.savefig(f"{outbase}_xykin.svg", format="svg", bbox_inches="tight")
plt.close(fig)
