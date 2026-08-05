import matplotlib
matplotlib.use('Agg')  # Force headless rendering to silence Qt warnings
import matplotlib.pyplot as plt
import numpy as np
import os

if os.path.exists('J_history.npy'):
    J_history = np.load('J_history.npy')
else:
    J_history = [0.0] * 15 # Fallback if file isn't found

# --- Plotting Configuration ---
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Parameters matching your simulation
Nx, Nz = 128, 64
L_val = 2.0

# Reconstruct the physical grids for plotting
# x is a linear Fourier grid
x = np.linspace(0, L_val, Nx, endpoint=False)
# z is a Chebyshev grid mapped to [0, 1]
z = 0.5 - 0.5 * np.cos(np.pi * (2 * np.arange(Nz) + 1) / (2 * Nz))
X, Z = np.meshgrid(x, z, indexing='ij')

# Select which iterations to plot (e.g., First, a middle step, and the last)
iterations_to_plot = [0, 5, 10, 15]
num_plots = len(iterations_to_plot)

fig, axes = plt.subplots(2, num_plots, figsize=(4 * num_plots, 7), sharex=True, sharey=True)

for col, iter_num in enumerate(iterations_to_plot):
    file_path = os.path.join(BASE_DIR, f'ic_iter_{iter_num}.npz')
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping.")
        continue
        
    data = np.load(file_path)
    u_0 = data['u_0']
    T_0 = data['T_0']
    
    # Wrap the arrays visually to close the periodic boundary on the plot
    x_plot = np.append(x, L_val)
    X_plot, Z_plot = np.meshgrid(x_plot, z, indexing='ij')
    
    T_plot = np.vstack([T_0, T_0[0:1, :]])
    U_x_plot = np.vstack([u_0[0], u_0[0][0:1, :]])
    U_z_plot = np.vstack([u_0[1], u_0[1][0:1, :]])
    
    # Get the J value for this specific iteration
    if iter_num == 0:
        # Hide the J value for the trivial starting state (Iter 0)
        J_str = "" 
    elif (iter_num - 1) < len(J_history):
        # Shift the index so Iter 1 gets J_history[0], Iter 2 gets J_history[1], etc.
        current_J = J_history[iter_num - 1]
        J_str = f"\n(J = {current_J:.4f})"
    else:
        J_str = ""

    # --- Row 1: Temperature ---
    ax_T = axes[0, col]
    mesh = ax_T.pcolormesh(X_plot, Z_plot, T_plot, cmap='RdBu_r', shading='gouraud', vmin=0, vmax=1)
    ax_T.set_title(f'Iter {iter_num} - Temperature{J_str}')
    if col == 0:
        ax_T.set_ylabel('z')
    
    # Plot Velocity Magnitude (Keep background as raw magnitude for accurate colorbar)
    ax_U = axes[1, col]
    magnitude = np.sqrt(U_x_plot**2 + U_z_plot**2)
    mesh_U = ax_U.pcolormesh(X_plot, Z_plot, magnitude, cmap='viridis', shading='auto')
    
    # --- NEW: Log-scale the quiver arrows ---
    # 1. Prevent division by zero
    safe_magnitude = np.where(magnitude == 0, 1.0, magnitude)
    
    # 2. Calculate the logarithmic scaling factor (using log1p for stability near 0)
    log_magnitude = np.log1p(magnitude)
    
    # 3. Apply the scaling factor to the vector components
    U_x_log = U_x_plot * (log_magnitude / safe_magnitude)
    U_z_log = U_z_plot * (log_magnitude / safe_magnitude)
    
    # 4. Plot the log-scaled vectors
    if np.max(magnitude) > 1e-12:
        stride_x = max(1, Nx // 16)
        stride_z = max(1, Nz // 16)
        ax_U.quiver(X_plot[::stride_x, ::stride_z].T, Z_plot[::stride_x, ::stride_z].T, 
                    U_x_log[::stride_x, ::stride_z].T, U_z_log[::stride_x, ::stride_z].T, 
                    color='white', alpha=0.8)
    # ----------------------------------------
                
    ax_U.set_title(f'Iter {iter_num} - Velocity{J_str}')
    ax_U.set_xlabel('x')
    if col == 0: ax_U.set_ylabel('z')

# Add colorbars
cbar_ax_T = fig.add_axes([0.92, 0.55, 0.015, 0.35])
fig.colorbar(mesh, cax=cbar_ax_T, label='T')

cbar_ax_U = fig.add_axes([0.92, 0.1, 0.015, 0.35])
fig.colorbar(mesh_U, cax=cbar_ax_U, label='|u|')

plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.1, hspace=0.2, wspace=0.1)
plt.savefig('initial_conditions_evolution.png', dpi=300)
print("Saved visualization to 'initial_conditions_evolution.png'")