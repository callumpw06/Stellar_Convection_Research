"""
This script loads the Dedalus HDF5 output files and plots the 
Nusselt number (Nu) against simulation time.

This script can be run using the command:
    python visualise_Nusselt_number_polar.py <Ro>
    
"""

import sys
import pathlib
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Configuration
Ro_param = sys.argv[1] if len(sys.argv) > 1 else "1.0"
data_folder = pathlib.Path(f'analysis_runs/Ro_{Ro_param}')
output_plot = f'nusselt_vs_time_Ro_{Ro_param}.png'

print(f"Searching for data in '{data_folder}'...")

# Sorting key for file chunks to ensure chronological order
def extract_numbers(path):
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', path.name)]

files = sorted(data_folder.glob("*.h5"), key=extract_numbers)

if not files:
    raise FileNotFoundError(f"No .h5 files found in {data_folder}.")

all_times = []
all_nu = []

print(f"Found {len(files)} files. Loading data...")

# Iterate through HDF5 time chunks
for file_path in files:
    with h5py.File(file_path, 'r') as f:
        chunk_times = f['scales/sim_time'][:]
        
        # Extract Nu. We use .flatten() because Dedalus sometimes retains 
        # spatial dimensions of size 1 for fully integrated quantities 
        # (e.g., shape [Nt, 1, 1] becomes [Nt]).
        chunk_nu = f['tasks/Nu'][:].flatten()
        
        all_times.append(chunk_times)
        all_nu.append(chunk_nu)

# Concatenate all chunks into single 1D arrays
times = np.concatenate(all_times)
nu = np.concatenate(all_nu)

print(f"Loaded {len(times)} data points. Generating plot...")

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(times, nu, color='crimson', linewidth=2)

# Formatting
ax.set_xlabel('Simulation Time', fontsize=12)
ax.set_ylabel('Nusselt Number (Nu)', fontsize=12)
ax.set_title(f'Global Nusselt Number Evolution\n(Ro = {Ro_param})', fontsize=14, pad=15)
ax.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully as '{output_plot}'")

# Display the plot if running interactively
plt.show()