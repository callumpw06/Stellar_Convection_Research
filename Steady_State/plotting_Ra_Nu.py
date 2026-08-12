import numpy as np
import matplotlib.pyplot as plt

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

# --- 1. Custom Data Points ---
# Replace these with your actual Dedalus simulation results!
# Example: Running simulations at Ra = 500, 800, 1500, 2500, 3500
custom_Ra = [1300, 1400, 1500, 1600, 1700, 1800, 1900,
            2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
            3000]
custom_Nu = [2.1157, 2.2177, 2.3107, 2.3962, 2.4754, 2.5493, 2.6187,
            2.6841, 2.7462, 2.8052,2.8615, 2.9155, 2.9673, 3.0172, 3.0654, 3.1120, 3.1572,
            3.2010]

# --- 2. Theoretical Mathematical Setup ---
pi = np.pi
Rac = (27 * pi**4) / 4  # Absolute minimum critical Ra (~657.511)

# Create a continuous array of Rayleigh numbers to plot the smooth curve
# We will plot from Ra=0 up to Ra=4000 (just past where the theory starts to break down)
Ra_continuous = np.linspace(500, 4000, 1000)

# Calculate the theoretical Nusselt number using our weakly non-linear formula
# np.where works like an if/else statement for arrays:
# If Ra < Rac, Nu = 1.0 (Conduction)
# If Ra >= Rac, Nu = 1 + 2 * (1 - Rac/Ra) (Convection)
Nu_theoretical = np.where(Ra_continuous < Rac, 1.0, 1.0 + 2.0 * (1.0 - Rac / Ra_continuous))

# --- 3. Plotting ---
plt.figure(figsize=(9, 6))

# Plot the theoretical curve
plt.plot(Ra_continuous, Nu_theoretical, color='blue', linewidth=2.5, 
         label='Theoretical $Nu(Ra)$ (3-Mode Truncation)')

# Plot a vertical dashed line to mark the critical Rayleigh number (Onset of convection)
plt.axvline(x=Rac, color='gray', linestyle='--', 
            label=f'Onset of Convection ($Ra_c \\approx {Rac:.1f}$)')

# Plot your custom changeable data points
plt.plot(custom_Ra, custom_Nu, color='red', marker='o', linestyle='-', 
         linewidth=2, markersize=8, label='Numerical Data (Dedalus Simulation)')

# Formatting
plt.title('Nusselt Number vs. Rayleigh Number', fontsize=14)
plt.xlabel('Rayleigh Number ($Ra$)', fontsize=12)
plt.ylabel('Nusselt Number ($Nu$)', fontsize=12)

plt.xscale('linear')  # Logarithmic scale for Ra
plt.yscale('linear')  # Linear scale for Nu

# Set axis limits
plt.xlim(500, 4000)
plt.ylim(0.8, 3.5)

plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()

# Save or display
plt.savefig('Nu_vs_Ra_Scaling.png', dpi=300)
plt.show()