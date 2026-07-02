"""
This is a python script to visualise the temperature field of a 2D Rayleigh-Benard convection 
simulation in the Dedalus framework. It loads the temperature field from the HDF5 files generated
by the simulation and creates a contour plot of the temperature field at a specified time step.
"""

import h5py
import numpy as np
import pathlib
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from Boussinesq_Convectiom import Lx, Lz # Import Lx and Lz from the simulation script

# Configuration
data_folder = 'analysis'
field_to_plot = 'temperature'
output_movie = 'temperature_fixed.mp4'

# Dimensions
extent = [0, 1, 0, 4]

print(f"Searching for data in '/{data_folder}'...")

files = sorted(pathlib.Path(data_folder).glob("*.h5"))
if not files:
    raise FileNotFoundError(f"No .h5 files found in {data_folder}.")

all_data = []
all_times = []

print(f"Found {len(files)} files. Loading data...")

for i, file_path in enumerate(files):
    with h5py.File(file_path, 'r') as f:
        chunk_data = f[f'tasks/{field_to_plot}'][:]
        chunk_times = f['scales/sim_time'][:]
        
        all_data.append(chunk_data)
        all_times.append(chunk_times)

# Concatenate all chunks
full_data = np.concatenate(all_data, axis=0)
full_times = np.concatenate(all_times, axis=0)

num_frames, nx, nz = full_data.shape
print(f"Data loaded. Frames: {num_frames}. Grid Size: {nx}x{nz}")

print("Setting up animation...")
fig, ax = plt.subplots(figsize=(10, 4))

initial_frame_data = full_data[0, :, :].T
im = ax.imshow(initial_frame_data, origin='lower', extent=extent, cmap='RdBu_r', aspect='auto')
colorbar = fig.colorbar(im, ax=ax, label=field_to_plot.capitalize())
title_text = ax.set_title(f"Time: {full_times[0]:.2f}")
ax.set_xlabel("x axis")
ax.set_ylabel("z axis")

def update(frame_idx):
    current_data = full_data[frame_idx, :, :].T
    im.set_data(current_data)

    clim_min, clim_max = np.min(current_data), np.max(current_data)
    im.set_clim(clim_min, clim_max)
    title_text.set_text(f"Time: {full_times[frame_idx]:.5f}")

    if frame_idx % 10 == 0:
         print(f"Rendering frame {frame_idx}/{num_frames}...")
    return [im, title_text]

# Create the animation
print("Creating animation...")
ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
ani.save(output_movie, writer='ffmpeg', fps=30)
plt.close(fig)
print(f"Animation saved as '{output_movie}'")