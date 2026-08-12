import numpy as np
import matplotlib.pyplot as plt

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

# --- 1. System Parameters ---
Ra = 3500       # Weakly non-linear regime to keep bounds realistic
L = 2.0          # Domain width
H = 1.0          # Domain height
n_modes = 20      # Number of Fourier modes to sum

# --- 2. Spatial Grid Setup ---
nx, nz = 200, 100
x = np.linspace(0, L, nx)
z = np.linspace(0, H, nz)
X, Z = np.meshgrid(x, z)

pi = np.pi

# Initialize the base fields
T_total = 1.0 - Z
vx_total = np.zeros_like(X)
vz_total = np.zeros_like(X)

# --- 3. Mode Loop & Superposition ---
for m in range(1, n_modes + 1):
    k_m = 2 * pi * m / L
    Rac_m = ((k_m**2 + pi**2)**3) / (k_m**2)
    
    if Ra > Rac_m:
        A_m = np.sqrt(8 * (Ra - Rac_m)) / (k_m**2 + pi**2)
        B_m = ((k_m**2 + pi**2)**2 / (Ra * k_m)) * A_m
        C_m = -(Ra - Rac_m) / (pi * Ra)
        
        T_total += B_m * np.cos(k_m * X) * np.sin(pi * Z) + C_m * np.sin(2 * pi * Z)
        vx_total += -A_m * pi * np.sin(k_m * X) * np.cos(pi * Z)
        vz_total += A_m * k_m * np.cos(k_m * X) * np.sin(pi * Z)

# Calculate velocity magnitude |v|
speed = np.sqrt(vx_total**2 + vz_total**2)

# --- 4. Plotting ---
# Create 2 subplots vertically to match the layout in "initial_conditions_evolution.jpg"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# --- Plot 1: Temperature ---
# Using RdBu_r to match the Dark Blue -> White -> Dark Red gradient
# Locking vmin=0 and vmax=1 ensures the color mapping is physically consistent
contour1 = ax1.contourf(X, Z, T_total, levels=100, cmap='RdBu_r', vmin=0, vmax=1)
cbar1 = fig.colorbar(contour1, ax=ax1)
cbar1.set_label('T')
ax1.set_title('Temperature')
ax1.set_ylabel('z')
ax1.set_xlim(0, L)
ax1.set_ylim(0, H)

# --- Plot 2: Velocity Magnitude ---
# Using viridis to match the Purple -> Green -> Yellow gradient
contour2 = ax2.contourf(X, Z, speed, levels=100, cmap='viridis')
cbar2 = fig.colorbar(contour2, ax=ax2)
cbar2.set_label('|v|')

# Overlay white quiver arrows
# We slice the array [::10] to "skip" points so the arrows aren't too cluttered
skip = (slice(None, None, 10), slice(None, None, 10))
ax2.quiver(X[skip], Z[skip], vx_total[skip], vz_total[skip], 
           color='white', pivot='mid', scale=speed.max()*15, width=0.003)

ax2.set_title('Velocity')
ax2.set_xlabel('x')
ax2.set_ylabel('z')
ax2.set_xlim(0, L)
ax2.set_ylim(0, H)

plt.tight_layout()
plt.show()