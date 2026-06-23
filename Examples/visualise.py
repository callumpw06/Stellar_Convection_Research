import h5py
import numpy as np
import pathlib
import matplotlib
# Important for WSL: Tell matplotlib not to try opening a window screen
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === Configuration ===
data_folder = 'snapshots'
field_to_plot = 'buoyancy'
output_movie = 'convection_fixed.mp4'

# HARDCODED DIMENSIONS (From your simulation script)
# Lx = 4, Lz = 1
extent = [0, 4, 0, 1] 
# =====================

print(f"Searching for data in '/{data_folder}'...")

# 1. Find and stitch together all HDF5 files
files = sorted(pathlib.Path(data_folder).glob("*.h5"))
if not files:
    raise FileNotFoundError(f"No .h5 files found in {data_folder}.")

all_data = []
all_times = []

print(f"Found {len(files)} files. Loading data...")

for i, file_path in enumerate(files):
    with h5py.File(file_path, 'r') as f:
        # We SKIPPED the problematic 'scales' reading here.
        
        # Load the field data and time array
        # Data shape is (time, x, z)
        chunk_data = f[f'tasks/{field_to_plot}'][:]
        chunk_times = f['scales/sim_time'][:]
        
        all_data.append(chunk_data)
        all_times.append(chunk_times)

# Concatenate all chunks
full_data = np.concatenate(all_data, axis=0)
full_times = np.concatenate(all_times, axis=0)

num_frames, nx, nz = full_data.shape
print(f"Data loaded. Frames: {num_frames}. Grid Size: {nx}x{nz}")

# 2. Set up the plot
print("Setting up animation...")
fig, ax = plt.subplots(figsize=(10, 4)) 

# Prepare the initial frame (Transpose .T to swap x and z for plotting)
initial_frame_data = full_data[0, :, :].T

# Plot using the manual 'extent' we defined at the top
im = ax.imshow(initial_frame_data, origin='lower', extent=extent, cmap='RdBu_r', aspect='auto')
colorbar = fig.colorbar(im, ax=ax, label=field_to_plot.capitalize())
title_text = ax.set_title(f"Time: {full_times[0]:.2f}")
ax.set_xlabel("x axis")
ax.set_ylabel("z axis")

# 3. Define the update function
def update(frame_idx):
    current_data = full_data[frame_idx, :, :].T
    im.set_data(current_data)
    
    # Update color scale to avoid flickering
    clim_min = np.min(current_data)
    clim_max = np.max(current_data)
    im.set_clim(clim_min, clim_max)
    
    title_text.set_text(f"Time: {full_times[frame_idx]:.2f}")
    
    if frame_idx % 10 == 0:
         print(f"Rendering frame {frame_idx}/{num_frames}...")
    return im, title_text

# 4. Generate the movie
print("Rendering movie... (this takes 1-2 mins)")
anim = FuncAnimation(fig, update, frames=num_frames, blit=False)
anim.save(output_movie, writer='ffmpeg', fps=15, dpi=150)
plt.close(fig)

print(f"Done! Saved to: {output_movie}")