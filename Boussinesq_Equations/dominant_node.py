import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import sys

plt.rcParams.update({
    'font.size': 26,
    'axes.titlesize': 35,
    'axes.labelsize': 33,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 24, # Slightly smaller for multiple entries
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
})

# Define the L values you want to plot. 
# The script will look for folders named analysis_runs/Lx_{L}
Lx_values = [1.550, 1.583, 2.000, 2.375, 3.166, 4.749]
base_dir = Path('analysis_runs')

for Lx in Lx_values:
    print(f"--- Running simulation for Lx = {Lx} ---")
    # sys.executable ensures it uses the exact same python environment/conda env
    subprocess.run([sys.executable, "Boussinesq_Convection.py", str(Lx)], check=True)
    print(f"--- Finished simulation for Lx = {Lx} ---\n")

plt.figure(figsize=(15, 10))

# Loop over the different Lx values
for Lx in Lx_values:
    folder = base_dir / f'Lx_{Lx}'
    
    # Safely skip if a run failed or folder doesn't exist
    if not folder.exists():
        print(f"Warning: Folder {folder} not found. Skipping.")
        continue
        
    # Grab all .h5 files in the folder to bypass Dedalus stripping the decimal
    files = list(folder.glob('*.h5'))
    files = sorted(files, key=lambda x: int(x.stem.split('_s')[-1]))
    
    if not files:
        print(f"Warning: No .h5 files found in {folder}. Skipping.")
        continue
        
    print(f"Processing Lx = {Lx}...")

    t_list, T_list, w_list = [], [], []
    z = None

    for file in files:
        with h5py.File(file, 'r') as f:
            T_dataset = f['tasks']['temperature']
            w_dataset = f['tasks']['w_velocity']
            
            if z is None:
                z = T_dataset.dims[2][0][:]
                
            t_list.append(T_dataset.dims[0][0][:])
            T_list.append(T_dataset[:])
            w_list.append(w_dataset[:])

    time = np.concatenate(t_list)
    T = np.concatenate(T_list, axis=0)
    w = np.concatenate(w_list, axis=0)
    Nx = T.shape[1]

    T_hat = np.fft.rfft(T, axis=1)
    w_hat = np.fft.rfft(w, axis=1)

    flux_spectrum_2d = np.real(w_hat * np.conj(T_hat)) / (Nx**2)

    flux_spectrum_2d[:, 1:] *= 2

    flux_per_mode_time = np.trapz(flux_spectrum_2d, x=z, axis=2)

    # Average over the ENTIRE loaded dataset, since the simulation 
    # now only saves data during the final steady-state period.
    mean_flux_per_mode = np.mean(flux_per_mode_time[:, :], axis=0)

    k_x_indices = np.arange(1, len(mean_flux_per_mode))
    flux_values = mean_flux_per_mode[1:]

    # Plot this specific Lx as a line on the combined plot
    plt.plot(k_x_indices, flux_values, marker='o', linestyle='-', alpha=0.8, linewidth=4, markersize=10, label=f'$L = {Lx}$')

    dominant_mode = np.argmax(flux_values) + 1
    print(f"  Dominant mode for Lx={Lx} is k_x = {dominant_mode}")

# Finalize the plot formatting
plt.xscale('log')
plt.xlabel('Fourier Mode Index ($n$)')
plt.ylabel('Convective Heat Flux ($\\langle wT \\rangle$)')
plt.title('Heat Transport Distribution Across Fourier Modes')
plt.grid(True, which="both", ls="--", alpha=0.6, linewidth=2)
plt.legend()

plt.savefig('combined_dominant_modes.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nPlot saved as 'combined_dominant_modes.png'")