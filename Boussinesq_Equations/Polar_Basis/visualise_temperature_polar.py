"""
This script visualises the temporal progression of the temperature field 
for a 2D Boussinesq convection simulation in polar/annular coordinates.
It explicitly plots the inner and outer radial boundaries.
"""

import sys
import pathlib
import re
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for generating animation files
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configuration
Ro_param = sys.argv[1] if len(sys.argv) > 1 else "1.0"
data_folder = pathlib.Path(f'analysis_runs/Ro_{Ro_param}')
field_to_plot = 'temperature'
output_movie = f'temperature_polar_Ro_{Ro_param}.mp4'

print(f"Searching for data in '{data_folder}'...")

# Sorting key for file chunks
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
for i, file_path in enumerate(files):
    with h5py.File(file_path, 'r') as f:
        task = f[f'tasks/{field_to_plot}']
        chunk_data = task[:]
        chunk_times = f['scales/sim_time'][:]
        
        # Extract coordinate axes from the first file using dimension scales
        if phi is None or r is None:
            try:
                phi = task.dims[1][0][:]
                r = task.dims[2][0][:]
            except (AttributeError, IndexError, KeyError):
                phi = f['scales/phi/1.0'][:]
                r = f['scales/r/1.0'][:]
        
        all_data.append(chunk_data)
        all_times.append(chunk_times)

# Concatenate all chunks along the time axis (shape: [num_frames, Nphi, Nr])
full_data = np.concatenate(all_data, axis=0)
full_times = np.concatenate(all_times, axis=0)

num_frames = full_data.shape[0]
print(f"Data loaded successfully. Total frames: {num_frames}. Grid: phi={len(phi)}, r={len(r)}")

# Build 2D meshgrid for polar coordinates
Phi, R = np.meshgrid(phi, r)
Ri = r.min()
Ro = r.max()

print("Setting up polar animation...")
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

# Initial frame data (transposed to match the (r, phi) meshgrid ordering)
initial_frame = full_data[0, :, :].T

# Setup initial polar pcolormesh plot
im = ax.pcolormesh(Phi, R, initial_frame, cmap='RdBu_r', shading='nearest')
ax.set_rlim(Ri, Ro)                 # Crop the inner core hole
ax.set_yticklabels([])              # Hide radial numerical tick labels for a clean look
ax.grid(True, alpha=0.3)            # Subtle gridlines

# --- NEW: Plot the Inner and Outer Boundaries explicitly ---
# Create an array of angles from 0 to 2*pi for the circular lines
theta_circle = np.linspace(0, 2*np.pi, 300)
# Inner boundary line
ax.plot(theta_circle, np.full_like(theta_circle, Ri), color='black', linewidth=2.5)
# Outer boundary line
ax.plot(theta_circle, np.full_like(theta_circle, Ro), color='black', linewidth=2.5)
# ---------------------------------------------------------

# Set consistent color limits across the entire animation dataset
vmin, vmax = np.min(full_data), np.max(full_data)
im.set_clim(vmin, vmax)

colorbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.1, label='Temperature (T)')
title_text = ax.set_title(f"Temperature Field\nTime: {full_times[0]:.4f} (Ro = {Ro_param})", pad=20)

def update(frame_idx):
    current_data = full_data[frame_idx, :, :].T
    # Update pcolormesh color array with flattened frame data
    im.set_array(current_data.ravel())
    title_text.set_text(f"Temperature Field\nTime: {full_times[frame_idx]:.4f} (Ro = {Ro_param})")

    if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
        print(f"Rendering frame {frame_idx + 1}/{num_frames}...")
    return [im, title_text]

# Build and save animation using ffmpeg
print(f"Creating animation (saving to '{output_movie}')...")
ani = FuncAnimation(fig, update, frames=num_frames, blit=False)
ani.save(output_movie, writer='ffmpeg', fps=30)
plt.close(fig)
print(f"Animation saved successfully as '{output_movie}'")