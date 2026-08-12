import matplotlib
matplotlib.use('Agg')  # Force headless rendering to silence Qt warnings
import matplotlib.pyplot as plt
import numpy as np
import os

if os.path.exists('J_history.npy'):
    J_history = np.load('J_history.npy')
else:
    J_history = [0.0] * 15 # Fallback if file isn't found

# --- SETTINGS ---
BOUNDARY_CONDITION = 'no-slip' 

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
Nx, Nz = 96, 48
L_val = 2.0

# Reconstruct the physical grids for plotting
x = np.linspace(0, L_val, Nx, endpoint=False)
z = 0.5 - 0.5 * np.cos(np.pi * (2 * np.arange(Nz) + 1) / (2 * Nz))
X, Z = np.meshgrid(x, z, indexing='ij')

iterations_to_plot = [0, 5, 10, 15]
num_plots = len(iterations_to_plot)

# --- FIX: Increased figure size for a 4x4 grid (16 tall instead of 12) ---
fig, axes = plt.subplots(4, num_plots, figsize=(4 * num_plots, 16))

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
    
    if iter_num == 0:
        J_str = "" 
    elif (iter_num - 1) < len(J_history):
        current_J = J_history[iter_num - 1]
        J_str = f"\n($J = {current_J:.4f}$)"
    else:
        J_str = ""

    # --- Row 1: Total Temperature ---
    ax_T = axes[0, col]
    mesh = ax_T.pcolormesh(X_plot, Z_plot, T_plot, cmap='RdBu_r', shading='gouraud', vmin=0.0, vmax=1.0)
    ax_T.set_title(f'Iter {iter_num} - Temperature{J_str}')
    if col == 0: ax_T.set_ylabel('$z$')
    
    # --- Row 2: Velocity ---
    ax_U = axes[1, col]
    magnitude = np.sqrt(U_x_plot**2 + U_z_plot**2)
    mesh_U = ax_U.pcolormesh(X_plot, Z_plot, magnitude, cmap='viridis', shading='auto')
    
    safe_magnitude = np.where(magnitude == 0, 1.0, magnitude)
    log_magnitude = np.log1p(magnitude)
    U_x_log = U_x_plot * (log_magnitude / safe_magnitude)
    U_z_log = U_z_plot * (log_magnitude / safe_magnitude)
    
    if np.max(magnitude) > 1e-12:
        stride_x = max(1, Nx // 16)
        stride_z = max(1, Nz // 16)
        ax_U.quiver(X_plot[::stride_x, ::stride_z].T, Z_plot[::stride_x, ::stride_z].T, 
                    U_x_log[::stride_x, ::stride_z].T, U_z_log[::stride_x, ::stride_z].T, 
                    color='white', alpha=0.8)
                
    ax_U.set_title(f'Iter {iter_num} - Velocity{J_str}')
    if col == 0: ax_U.set_ylabel('$z$')

    # --- Data Loading for Rows 3 & 4 ---
    try:
        nu_data = np.load(f'nu_history_iter_{iter_num}.npz')
        t_vals = nu_data['t_history']
        Nu_vals = nu_data['Nu_history']
        
        # --- Row 3: Nusselt Number vs Time ---
        ax_Nu = axes[2, col]
        ax_Nu.plot(t_vals, Nu_vals, color='#d9534f', linewidth=2)
        ax_Nu.set_title(f'Iter {iter_num} - Nu vs Time')
        ax_Nu.grid(True, linestyle=':', alpha=0.6)
        if col == 0: ax_Nu.set_ylabel('Nusselt Number ($Nu$)')
        ax_Nu.set_ylim(bottom=1.0) 
        ax_Nu.set_xlim(left=0.0, right=t_vals[-1])
        
        if BOUNDARY_CONDITION == 'no-slip':
            axins = ax_Nu.inset_axes([0.45, 0.45, 0.5, 0.45])
            axins.plot(t_vals, Nu_vals, color='#d9534f', linewidth=2)
            x1, x2 = 0.0, 0.15
            zoom_mask = (t_vals >= x1) & (t_vals <= x2)
            if np.any(zoom_mask):
                y1, y2 = np.min(Nu_vals[zoom_mask]), np.max(Nu_vals[zoom_mask])
                y_padding = (y2 - y1) * 0.1
                if y_padding == 0: y_padding = 0.1
                axins.set_ylim(y1 - y_padding, y2 + y_padding)
            axins.set_xlim(x1, x2)
            axins.tick_params(axis='both', labelsize=8)
            axins.grid(True, linestyle=':', alpha=0.6)

        # --- Row 4: Kinetic Energy vs Time ---
        ax_KE = axes[3, col]
        if 'KE_history' in nu_data:
            KE_vals = nu_data['KE_history']
            ax_KE.plot(t_vals, KE_vals, color='#337ab7', linewidth=2)
            ax_KE.set_title(f'Iter {iter_num} - KE vs Time')
            ax_KE.set_xlabel('Time (s)')
            ax_KE.grid(True, linestyle=':', alpha=0.6)
            if col == 0: ax_KE.set_ylabel('Kinetic Energy ($E_k$)')
            ax_KE.set_xlim(left=0.0, right=t_vals[-1])
        else:
            ax_KE.text(0.5, 0.5, 'KE Data Missing\n(Rerun Forward Solver)', ha='center', va='center')
            ax_KE.set_xlabel('Time (s)')

    except FileNotFoundError:
        axes[2, col].text(0.5, 0.5, 'Data missing', ha='center', va='center')
        axes[3, col].text(0.5, 0.5, 'Data missing', ha='center', va='center')

# Apply tight_layout at the end, reserving space for colorbars
plt.tight_layout(rect=[0, 0, 0.91, 1])

# --- FIX: Adjusted colorbar vertical positions for a 4-row layout ---
cbar_ax_T = fig.add_axes([0.93, 0.77, 0.015, 0.18])
fig.colorbar(mesh, cax=cbar_ax_T, label='$T$')

cbar_ax_U = fig.add_axes([0.93, 0.53, 0.015, 0.18])
fig.colorbar(mesh_U, cax=cbar_ax_U, label='$|v|$')

plt.savefig('initial_conditions_evolution.png', dpi=300)
print("Saved visualization to 'initial_conditions_evolution.png'")