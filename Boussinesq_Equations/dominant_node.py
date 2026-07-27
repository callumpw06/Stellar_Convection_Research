import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Find and strictly sort all analysis files numerically
folder = Path('analysis')
files = list(folder.glob('analysis_s*.h5'))
files = sorted(files, key=lambda x: int(x.stem.split('_s')[-1]))

if not files:
    raise FileNotFoundError("No analysis files found in the 'analysis' directory.")

print(f"Found {len(files)} files. Loading data...")

t_list, T_list, w_list = [], [], []
z = None

# 2. Extract data from all files
for file in files:
    with h5py.File(file, 'r') as f:
        T_dataset = f['tasks']['temperature']
        w_dataset = f['tasks']['w_velocity']
        
        if z is None:
            z = T_dataset.dims[2][0][:]
            
        t_list.append(T_dataset.dims[0][0][:])
        T_list.append(T_dataset[:])
        w_list.append(w_dataset[:])

# 3. Concatenate all chunks into continuous arrays
time = np.concatenate(t_list)
T = np.concatenate(T_list, axis=0)
w = np.concatenate(w_list, axis=0)

# 4. Perform 1D Fourier Transform along the x-axis
T_hat = np.fft.rfft(T, axis=1)
w_hat = np.fft.rfft(w, axis=1)
Nx = T.shape[1]

# 5. Calculate the cospectrum for all timesteps
flux_spectrum_2d = np.real(w_hat * np.conj(T_hat)) / (Nx**2)

# 6. Integrate over the z-axis
# Shape is now (time, k_x)
flux_per_mode_time = np.trapz(flux_spectrum_2d, x=z, axis=2)

# 7. Time-average the heat flux
# We slice [len(time)//2:] to only average over the second half of the simulation,
# filtering out the initial chaotic transient phase before it settles.
mean_flux_per_mode = np.mean(flux_per_mode_time[len(time)//2:, :], axis=0)

# 8. Plot Heat Flux vs Log(k_x)
# Create arrays starting from index 1 (ignoring mode 0 mean background state)
k_x_indices = np.arange(1, len(mean_flux_per_mode))
flux_values = mean_flux_per_mode[1:]

plt.figure(figsize=(9, 5))
plt.plot(k_x_indices, flux_values, marker='o', linestyle='-', alpha=0.8)

# Set x-axis to log scale
plt.xscale('log')

plt.xlabel('Fourier Mode Index ($k_x$) [Log Scale]')
plt.ylabel('Time-Averaged Convective Heat Flux')
plt.title('Heat Transport Distribution Across Fourier Modes')

# Add minor gridlines for better log-scale readability
plt.grid(True, which="both", ls="--", alpha=0.6)

plt.savefig('dominant_mode_heat_flux.png', dpi=300, bbox_inches='tight')
plt.close()

dominant_mode = np.argmax(flux_values) + 1
print(f"The dominant heat-transporting mode in the steady-state is k_x = {dominant_mode}")