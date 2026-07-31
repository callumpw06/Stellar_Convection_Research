"""
This script loads the Dedalus HDF5 output files and produces a side-by-side 
polar plot of the final instantaneous temperature field and the time-averaged 
temperature field over the final 0.5 time units.

This can be run using the command:
    python plot_average_temperature.py <Ro>

"""

import sys
import pathlib
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 22,     # Plot title size
    'axes.labelsize': 22,     # X and Y label size
    'xtick.labelsize': 18,    # X-axis tick numbers
    'ytick.labelsize': 18,    # Y-axis tick numbers
    'legend.fontsize': 18,    # Legend font size
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["cmr10"],                   # Matplotlib's built-in Computer Modern
    "mathtext.fontset": "cm",                  # Use Computer Modern for math equations
    "axes.formatter.use_mathtext": True,       # Use math text for axis tick labels
})

# Configuration
Ro_param = sys.argv[1] if len(sys.argv) > 1 else "1.0"
data_folder = pathlib.Path(f'analysis_runs/Ro_{Ro_param}')
output_plot = f'temperature_comparison_Ro_{Ro_param}.png'

print(f"Searching for data in '{data_folder}'...")

# Sorting key for file chunks to ensure chronological order
def extract_numbers(path):
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', path.name)]

files = sorted(data_folder.glob("*.h5"), key=extract_numbers)

if not files:
    raise FileNotFoundError(f"No .h5 files found in {data_folder}.")

print(f"Found {len(files)} files. Loading data...")

all_data = []
all_times = []
phi = None
r = None

# Iterate through HDF5 time chunks
for file_path in files:
    with h5py.File(file_path, 'r') as f:
        task = f['tasks/temperature']
        
        # Extract coordinate axes from the first file using dimension scales
        if phi is None or r is None:
            try:
                phi = task.dims[1][0][:]
                r = task.dims[2][0][:]
            except (AttributeError, IndexError, KeyError):
                phi = f['scales/phi/1.0'][:]
                r = f['scales/r/1.0'][:]
        
        all_data.append(task[:])
        all_times.append(f['scales/sim_time'][:])

# Concatenate all chunks
full_data = np.concatenate(all_data, axis=0)
full_times = np.concatenate(all_times, axis=0)

# Build 2D meshgrid for polar coordinates
Phi, R = np.meshgrid(phi, r)
Ri = r.min()
Ro = r.max()

# --- Calculations ---
t_final = full_times[-1]
t_start_avg = max(0.0, t_final - 0.5)

# Create a boolean mask for the final 0.5 time units
time_mask = full_times >= t_start_avg
frames_in_avg = np.sum(time_mask)
print(f"Averaging over {frames_in_avg} frames (from t={t_start_avg:.2f} to t={t_final:.2f})...")

# Extract final state and compute time-averaged state
# Remember to transpose (.T) so the shape (r, phi) matches the meshgrid
T_final = full_data[-1, :, :].T
T_avg = np.mean(full_data[time_mask, :, :], axis=0).T

# Set consistent color limits for a fair visual comparison
vmin = min(T_final.min(), T_avg.min())
vmax = max(T_final.max(), T_avg.max())

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': 'polar'})
theta_circle = np.linspace(0, 2*np.pi, 300)

def format_polar_axis(ax, title):
    ax.set_rlim(Ri, Ro)
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.3)
    ax.set_title(title, pad=20, fontsize=14)
    # Draw inner and outer boundaries
    ax.plot(theta_circle, np.full_like(theta_circle, Ri), color='black', linewidth=2.5)
    ax.plot(theta_circle, np.full_like(theta_circle, Ro), color='black', linewidth=2.5)

# Plot 1: Final Instantaneous State
c1 = ax1.pcolormesh(Phi, R, T_final, cmap='RdBu_r', shading='nearest', vmin=vmin, vmax=vmax)
format_polar_axis(ax1, f"Final State\n(t = {t_final:.2f})")

# Plot 2: Time-Averaged State
c2 = ax2.pcolormesh(Phi, R, T_avg, cmap='RdBu_r', shading='nearest', vmin=vmin, vmax=vmax)
format_polar_axis(ax2, f"Time-Averaged State\n(t = {t_start_avg:.2f} to {t_final:.2f})")

# Add a single shared colorbar
cbar = fig.colorbar(c2, ax=[ax1, ax2], shrink=0.7, pad=0.05)
cbar.set_label('Temperature (T)', fontsize=12)

plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully as '{output_plot}'")
plt.show()