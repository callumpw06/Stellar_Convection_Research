"""
This script will plot the Nusselt number as a function of time for a 2D Rayleigh-Benard 
convection simulation in the Dedalus framework.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pathlib

# 1. Define your parameters (must match the simulation)
Rayleigh = 2e6
Prandtl = 1
kappa = (Rayleigh * Prandtl)**(-1/2)

# Configuration
data_folder = 'analysis'

# Dimensions
extent = [0, 4, 0, 1]

print(f"Searching for data in '/{data_folder}'...")

files = sorted(pathlib.Path(data_folder).glob("*.h5"))
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

        total_heat_flux = kappa + chunk_data  # Total heat flux
        nu_t = total_heat_flux / kappa  # Nusselt number
        
        all_data.append(nu_t)
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