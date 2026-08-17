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
custom_Ra = [657, 700, 800, 900,
            1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
            2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
            3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 
            4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900,
            5000]

Pr_0_5_Nu = [1.0000, 1.1139, 1.3599, 1.5609,
            1.7298, 1.8751, 2.0025, 2.1160, 2.2183, 2.3116, 2.3974, 2.4771, 2.5514, 2.6213,
            2.6873, 2.7498, 2.8094, 2.8663, 2.9209, 2.9734, 3.0329, 3.0727, 3.1200, 3.1658,
            3.2103, 3.2536, 3.2957, 3.3368, 3.3769, 3.4161, 3.4544, 3.4919, 3.5282, 3.5647,
            3.6001, 3.6348, 3.6689, 3.7025, 3.7354, 3.7679, 3.7998, 3.8313, 3.8622, 3.8928,
            3.9229]
Pr_1_Nu = [1.0000, 1.1139, 1.3599, 1.5610,
            1.7299, 1.8751, 2.0024, 2.1157, 2.2177, 2.3107, 2.3962, 2.4754, 2.5493, 2.6187,
            2.6841, 2.7462, 2.8052,2.8615, 2.9155, 2.9673, 3.0172, 3.0654, 3.1120, 3.1572,
            3.2010, 3.2437, 3.2852, 3.3256, 3.3651, 3.4037, 3.4414, 3.4783, 3.5145, 3.5499,
            3.5847, 3.6189, 3.6524, 3.68564, 3.7179, 3.7498, 3.7812, 3.8122, 3.8427, 3.8738,
            3.9024]

Pr_10_Nu = [1.0000, 1.1139, 1.3600, 1.5612,
            1.7302, 1.8756, 2.0030, 2.1163, 2.2184, 2.3113, 2.3967, 2.4757, 2.5493, 2.6183,
            2.6832, 2.7446, 2.8029, 2.8584, 2.9115, 2.9622, 3.0110, 3.0579, 3.1031, 3.1467,
            3.1889, 3.2298, 3.2695, 3.3080, 3.3454, 3.3818, 3.4173, 3.4519, 3.4857, 3.5187,
            3.5509, 3.6189, 3.6524, 3.6854, 3.7179, 3.7498, 3.7812, 3.8122, 3.8427, 3.8728, 
            3.9024]

Pr_100_Nu = [1.0000, 1.1139, 1.3600, 1.5612,
            1.7302, 1.8757, 2.0031, 2.1164, 2.2186, 2.3116, 2.3970, 2.4761, 2.5497, 2.6188,
            2.6838, 2.7453, 2.8037, 2.8593, 2.9124, 2.9632, 3.0120, 3.0590, 3.1043, 3.1480,
            3.1903, 3.2312, 3.2709, 3.3095, 3.3470, 3.3835, 3.4190, 3.4537, 3.4875, 3.5205,
            3.5527, 3.5843, 3.6151, 3.6454, 3.6750, 3.7040, 3.7325, 3.7604, 3.7878, 3.8148,
            3.8412]

# Map your parameters so we can loop through them easily
pr_datasets = {
    0.5: {'data': Pr_0_5_Nu, 'color': 'blue', 'marker': 'x'},
    1.0: {'data': Pr_1_Nu,   'color': 'red', 'marker': 'o'},
    10:  {'data': Pr_10_Nu,  'color': 'green', 'marker': 's'},
    100: {'data': Pr_100_Nu, 'color': 'purple', 'marker': '^'}
}

Pr_list = []
m_list = []
c_list = []

# --- 2. Theoretical Mathematical Setup ---
pi = np.pi
Rac = (27 * pi**4) / 4  # Absolute minimum critical Ra (~657.511)
Ra_continuous = np.linspace(500, 5000, 1000)

# RE-ADDED: Theoretical 3-Mode Truncation calculation
Nu_theoretical = np.where(Ra_continuous < Rac, 1.0, 1.0 + 2.0 * (1.0 - Rac / Ra_continuous))

# We define the array and mask it here just once to save time in the loop
Ra_arr = np.array(custom_Ra)
fit_mask = Ra_arr > Rac
Ra_fit = Ra_arr[fit_mask]

# Using np.log() for the natural log of Rayleigh numbers
ln_Ra_fit = np.log(Ra_fit)

# --- 3. Plotting & Automated Fitting ---
plt.figure(figsize=(10, 8)) 

# RE-ADDED: Plot the theoretical curve (3-Mode Truncation)
plt.plot(Ra_continuous, Nu_theoretical, color='black', linewidth=2.5, linestyle=':', zorder=5,
         label='Theoretical $Nu(Ra)$ (3-Mode Truncation)')

# Plot a vertical dashed line to mark the critical Rayleigh number
plt.axvline(x=Rac, color='gray', linestyle='--', 
            label=f'Onset of Convection ($Ra_c \\approx {Rac:.1f}$)')

print("--- Discovered Semi-Log Fits (Natural Log) ---")

# Iterate through every Prandtl number in our dictionary
for pr, props in pr_datasets.items():
    
    Nu_arr = np.array(props['data'])
    Nu_fit = Nu_arr[fit_mask]
    
    # Calculate the precise slope and intercept for THIS specific Pr using natural log
    slope, intercept = np.polyfit(ln_Ra_fit, Nu_fit, 1)
    Pr_list.append(pr)
    m_list.append(slope)
    c_list.append(intercept)
    sign = "+" if intercept >= 0 else "-"
    
    # Print the exact formulas to the console
    print(f"Pr = {pr:<4} | Nu = {slope:.4f} * ln(Ra) {sign} {abs(intercept):.4f}")
    
    # Generate the continuous plotting line for this specific Pr
    Nu_semilog_fit = np.where(
        Ra_continuous < Rac, 
        1.0, 
        slope * np.log(Ra_continuous) + intercept  # CHANGED to np.log()
    )
    
    # 1. Plot the numerical data points (Hollow markers)
    plt.plot(custom_Ra, props['data'], color=props['color'], marker=props['marker'], 
             linestyle='', markerfacecolor='none', linewidth=2, markersize=8, 
             label=f'Data ($Pr={pr}$)')
    
    # 2. Plot the fitted line
    # CHANGED: Legend updated to display \ln instead of \log_{10}
    plt.plot(Ra_continuous, Nu_semilog_fit, color=props['color'], linestyle='-', 
             linewidth=1.5, alpha=0.7, 
             label=f'Fit ($Pr={pr}$): $Nu = {slope:.2f}\\ln(Ra) {sign} {abs(intercept):.2f}$')


# Formatting
plt.title('Nusselt Number vs. Rayleigh Number by Prandtl Number', fontsize=14)
plt.xlabel('Rayleigh Number ($Ra$)', fontsize=12)
plt.ylabel('Nusselt Number ($Nu$)', fontsize=12)

# Semi-log axis (Renders visually in base 10 for clean axis ticks)
plt.xscale('log')  
plt.yscale('linear')  

# Set axis limits
plt.xlim(500, 5000)
plt.ylim(0.8, 4.0)

plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', fontsize=9, ncol=2) 
plt.tight_layout()

# Save or display
plt.savefig('Nu_vs_Ra_Multi_Pr_Fits_NaturalLog.png', dpi=300)
plt.show()

# --- 4. Secondary Regression: m and c vs. Pr (Natural Log) ---
# Convert lists to numpy arrays for math operations
Pr_arr = np.array(Pr_list)
m_arr = np.array(m_list)
c_arr = np.array(c_list)

# Transform Pr into natural logarithmic space (base e)
ln_Pr = np.log(Pr_arr)

# Fit a line to determine how m and c scale with ln(Pr)
# m(Pr) = m_a * ln(Pr) + m_b
m_a, m_b = np.polyfit(ln_Pr, m_arr, 1)

# c(Pr) = c_a * ln(Pr) + c_b
c_a, c_b = np.polyfit(ln_Pr, c_arr, 1)

print("\n--- Pr Dependency Formulas ---")
print(f"m(Pr) = {m_a:.4f} * ln(Pr) + {m_b:.4f}")
print(f"c(Pr) = {c_a:.4f} * ln(Pr) + {c_b:.4f}")

# Create a continuous array of Pr values for a smooth plotting line
# np.geomspace creates evenly spaced points in log space
Pr_continuous = np.geomspace(0.5, 100, 100)
ln_Pr_continuous = np.log(Pr_continuous)

m_fit = m_a * ln_Pr_continuous + m_b
c_fit = c_a * ln_Pr_continuous + c_b

# --- Plotting the Dependencies ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot m vs Pr
ax1.plot(Pr_arr, m_arr, 'ko', markersize=8, markerfacecolor='none', label='Calculated Slopes ($m$)')
ax1.plot(Pr_continuous, m_fit, 'r--', label=f'Fit: $m = {m_a:.3f}\\ln(Pr) + {m_b:.3f}$')
ax1.set_xscale('log') # Keeps the visual axis scaling proportional
ax1.set_xlabel('Prandtl Number ($Pr$)')
ax1.set_ylabel('Slope ($m$)')
ax1.set_title('Slope Dependency on Pr')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot c vs Pr
ax2.plot(Pr_arr, c_arr, 'ko', markersize=8, markerfacecolor='none', label='Calculated Intercepts ($c$)')
ax2.plot(Pr_continuous, c_fit, 'b--', label=f'Fit: $c = {c_a:.3f}\\ln(Pr) + {c_b:.3f}$')
ax2.set_xscale('log') 
ax2.set_xlabel('Prandtl Number ($Pr$)')
ax2.set_ylabel('Intercept ($c$)')
ax2.set_title('Intercept Dependency on Pr')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('m_and_c_vs_Pr.png', dpi=300)
plt.show()