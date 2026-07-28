"""
This script will plot the Nusselt number as a function of time for a 2D Rayleigh-Benard 
convection simulation in the Dedalus framework.
"""

import h5py
import numpy as np
import pathlib
import re
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 14,     # Plot title size
    'axes.labelsize': 14,     # X and Y label size
    'xtick.labelsize': 12,    # X-axis tick numbers
    'ytick.labelsize': 12,    # Y-axis tick numbers
    'legend.fontsize': 14,    # Legend font size
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["cmr10"],                   # Matplotlib's built-in Computer Modern
    "mathtext.fontset": "cm",                  # Use Computer Modern for math equations
    "axes.formatter.use_mathtext": True,       # Use math text for axis tick labels
})

# Configuration
data_folder = 'analysis'

# Dimensions
extent = [0, 2, 0, 1]

print(f"Searching for data in '/{data_folder}'...")

# --- NEW SORTING LOGIC ---
def extract_numbers(path):
    # Extracts all numbers from the filename to use as a sorting key
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', path.name)]

# Sort the files using the custom key
files = sorted(pathlib.Path(data_folder).glob("*.h5"), key=extract_numbers)
# -------------------------
if not files:
    raise FileNotFoundError(f"No .h5 files found in {data_folder}.")

all_data = []
all_times = []

print(f"Found {len(files)} files. Loading data...")

for i, file_path in enumerate(files):
    with h5py.File(file_path, 'r') as f:
        chunk_data = f[f'tasks/Nu'][:,0]  # Load Nusselt number data
        #avg_flux_per_time = np.mean(chunk_data, axis=1)  # Average over x
        chunk_times = f['scales/sim_time'][:]

        #total_heat_flux = kappa + chunk_data  # Total heat flux
        #nu_t = total_heat_flux / kappa  # Nusselt number
        
        all_data.append(chunk_data)
        all_times.append(chunk_times)

# Concatenate all chunks
full_data = np.concatenate(all_data, axis=0)
full_times = np.concatenate(all_times, axis=0)



# 5. Plot the results
plt.plot(full_times, full_data, label='Nusselt Number')
plt.xlabel('Time')
plt.ylabel('Nusselt Number')
plt.title('Nusselt Number vs Time')
plt.grid(True)
plt.savefig('Nusselt_number_vs_time.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Calculate the time-averaged Nu (Steady State)
# Start averaging after the transient startup phase (e.g., after 5 units of time)
steady_state_mask = full_times > 5.0
nu_avg = np.mean(full_data[steady_state_mask])

print(f"Time-averaged Nusselt Number: {nu_avg:.4f}")