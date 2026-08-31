import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Plotting Configuration ---
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
})

# --- 1. Custom Data Points [Ra, Nu] ---
Pr_1_data = [
    [657, 1.0000], [700, 1.1139], [800, 1.3599], [900, 1.568502], [1000, 1.738682], [1100, 1.884388], 
    [1200, 2.01218], [1300, 2.125991], [1400, 2.228654], [1500, 2.32225], [1600, 2.408351], [1700, 2.488205], 
    [1800, 2.565744], [1900, 2.632723], [2000, 2.698764], [2100, 2.761373], [2200, 2.820969], [2300, 2.8779], 
    [2400, 2.932459], [2500, 2.984894], [2600, 3.035415], [2700, 3.084203], [2800, 3.131417], [2900, 3.177191], 
    [3000, 3.221645], [3100, 3.264882], [3200, 3.306966], [3300, 3.348066], [3400, 3.388166], [3500, 3.427361], 
    [3600, 3.465707], [3700, 3.503258], [3800, 3.54006], [3900, 3.576155], [4000, 3.611583], [4100, 3.646378], 
    [4200, 3.680573], [4300, 3.714196], [4400, 3.747275], [4500, 3.779835], [4600, 3.811898], [4700, 3.843486], 
    [4800, 3.872618], [4900, 3.905312], [5000, 3.935353], [6000, 4.215292], [7000, 4.466907], [8000, 4.691637], [9000, 4.89974],
    [1e4, 5.097310], [2e4, 6.557562], [3e4, 7.668585], 
    [4e4, 8.509277], [5e4, 9.219622], [6e4, 9.841527], [7e4, 10.397941], [8e4, 10.903328], [9e4, 11.367127], 
    [1e5, 11.801203], [2e5, 15.061135], [3e5, 17.344679], [4e5, 19.175763], [5e5, 20.717278], [6e5, 22.061602], 
    [7e5, 23.267590], [8e5, 24.360903], [9e5, 25.353893], [1e6, 26.308577]
]

# Map your parameters
pr_datasets = {
    1.0: {'data': Pr_1_data, 'color': 'red', 'marker': 'o'}
}

# --- 2. Mathematical Setup & Pre-Calculating Fits ---
pi = np.pi
Rac = (27 * pi**4) / 4  
Ra_continuous = np.linspace(500, 1.5e6, 2000) 

def offset_power_law(eps, A, x):
    return 1.0 + A * eps**x

def pure_power_law(Ra, A, b):
    return 1.0 + A * Ra**b

Pr_list, A_list, x_list = [], [], []

print("--- Discovered Fit Parameters ---")
for pr, props in pr_datasets.items():
    # Dynamically extract Ra and Nu for this specific dataset
    data_matrix = np.array(props['data'])
    Ra_arr = data_matrix[:, 0]
    Nu_arr = data_matrix[:, 1]
    
    # Save back to props for easy plotting later
    props['Ra_arr'] = Ra_arr
    props['Nu_arr'] = Nu_arr
    
    # --- Mask for the epsilon fit (Ra > Rac) ---
    fit_mask = Ra_arr > Rac
    Ra_fit = Ra_arr[fit_mask]
    Nu_fit = Nu_arr[fit_mask]
    eps_fit = (Ra_fit - Rac) / Rac

    # 1. Existing epsilon fit (Low Ra)
    popt, _ = curve_fit(offset_power_law, eps_fit, Nu_fit, p0=[1.0, 0.3])
    A, x = popt
    
    Pr_list.append(pr)
    A_list.append(A)
    x_list.append(x)
    
    print(f"Pr = {pr:<4} | Epsilon Fit: Nu = 1 + {A:.4f} * epsilon^{x:.4f}")
    
    eps_continuous = np.maximum(0, Ra_continuous - Rac) / Rac
    props['fit_line'] = np.where(
        Ra_continuous < Rac, 
        1.0, 
        1.0 + A * eps_continuous**x 
    )
    props['legend_str'] = f'$Nu = 1 + {A:.3f}\\epsilon^{{{x:.3f}}}$'

    # 2. Exact Theoretical Fractional Fit (Replaces the curve_fit)
    A_pure = 1.0 / 7.0
    b_pure = 3.0 / 8.0
    
    # Store the pure coefficients to calculate the subplot ratio later
    props['A_pure'] = A_pure
    props['b_pure'] = b_pure
    
    print(f"Pr = {pr:<4} | Fractional Theoretical Fit: Nu = 1 + (1/7) * Ra^(3/8)")
    
    # Evaluate the exact fit across the ENTIRE continuous line
    props['fit_line_pure'] = 1.0 + A_pure * Ra_continuous**b_pure 
    props['legend_str_pure'] = r'$Nu = 1 + \frac{1}{7}Ra^{3/8}$'

# =============================================================================
# --- 3. Figure 1: Standard Linear Scale Plot ---
# =============================================================================
fig1 = plt.figure(figsize=(10, 5)) 

for pr, props in pr_datasets.items():
    plt.plot(props['Ra_arr'], props['Nu_arr'], color=props['color'], marker=props['marker'], 
             linestyle='', markerfacecolor='none', linewidth=2, markersize=8, label=f'Data ($Pr={pr}$)')
    
    plt.plot(Ra_continuous, props['fit_line'], color=props['color'], linestyle='-', 
             linewidth=1.5, alpha=0.7, label=f'Fit: {props["legend_str"]}')
    
    plt.plot(Ra_continuous, props['fit_line_pure'], color='black', linestyle='--', 
             linewidth=1.5, alpha=0.8, label=props['legend_str_pure'])

plt.title('Nusselt Number vs. Rayleigh Number (Linear Scale)', fontsize=14)
plt.xlabel('Rayleigh Number ($Ra$)', fontsize=12)
plt.ylabel('Nusselt Number ($Nu$)', fontsize=12)

plt.xscale('linear')  
plt.yscale('linear')  

plt.xlim(500, 5000)
plt.ylim(0.8, 4)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', fontsize=11, ncol=1) 
plt.tight_layout()
plt.savefig('Nu_vs_Ra_Linear.png', dpi=300)

# =============================================================================
# --- 4. Figure 2: Semi-Log Scale Plot with Inset Subplot ---
# =============================================================================
fig2, ax2_main = plt.subplots(figsize=(10, 6))

for pr, props in pr_datasets.items():
    # --- Main Plot: Standard Data & Fits ---
    ax2_main.plot(props['Ra_arr'], props['Nu_arr'], color=props['color'], marker=props['marker'], 
             linestyle='', markerfacecolor='none', linewidth=2, markersize=8, label=f'Data ($Pr={pr}$)')
    
    ax2_main.plot(Ra_continuous, props['fit_line_pure'], color='black', linestyle='--', 
             linewidth=1.5, alpha=0.8, label=props['legend_str_pure'])

# --- Format Main Plot ---
ax2_main.set_title('Nusselt Number vs. Rayleigh Number (Semi-Log Scale)', fontsize=18)
ax2_main.set_xlabel('Rayleigh Number ($Ra$)', fontsize=16)
ax2_main.set_ylabel('Nusselt Number ($Nu$)', fontsize=16)
ax2_main.set_xscale('log')
ax2_main.set_yscale('log')  
ax2_main.set_xlim(500, 1.5e6)
ax2_main.set_ylim(0.8, 32)
ax2_main.grid(True, alpha=0.3)
# Moved legend to top left so it does not collide with the new inset box
ax2_main.legend(loc='upper left', fontsize=14, ncol=1)

# --- Create & Format Inset Ratio Subplot (Bottom Right) ---
# [x, y, width, height] as fractions of the main plot space
ax2_ratio = ax2_main.inset_axes([0.52, 0.08, 0.45, 0.35])

for pr, props in pr_datasets.items():
    # Plot the ratio for ALL data points
    Ra_valid = props['Ra_arr']
    Nu_valid = props['Nu_arr']
    
    # Calculate what the pure power law predicts for ALL exact Ra points using the fractional coefficients
    Nu_predicted = pure_power_law(Ra_valid, props['A_pure'], props['b_pure'])
    Nu_ratio = Nu_valid / Nu_predicted
    
    ax2_ratio.plot(Ra_valid, Nu_ratio, color=props['color'], marker=props['marker'], 
                   linestyle='', markerfacecolor='none', markersize=5)

# Formatting the Inset
ax2_ratio.axhline(1.0, color='black', linestyle='--', linewidth=1.5)
ax2_ratio.set_ylabel(r'$Nu \,/\, (1 + \frac{1}{7} Ra^{3/8})$', fontsize=16)
ax2_ratio.set_xscale('log')  
ax2_ratio.set_yscale('log')
ax2_ratio.set_xlim(500, 1.5e6) 
ax2_ratio.set_ylim(0.4, 2)

# Lock y-axis to a single 10^0 tick and hide all other redundant labels
ax2_ratio.set_yticks([1.0])
ax2_ratio.set_yticklabels(['$10^0$'])
ax2_ratio.tick_params(axis='x', which='both', labelbottom=False) # Hides all x-axis labels
ax2_ratio.tick_params(axis='y', which='minor', labelleft=False)  # Hides minor y-axis labels
ax2_ratio.tick_params(axis='both', which='major', labelsize=12)

ax2_ratio.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Nu_vs_Ra_SemiLog.png', dpi=300)

# =============================================================================
# --- 5. Figure 3: Secondary Regression (Pr-Independence) ---
# =============================================================================
A_arr = np.array(A_list)
x_arr = np.array(x_list)
Pr_arr = np.array(Pr_list)

A_mean = np.mean(A_arr)
x_mean = np.mean(x_arr)

print("\n--- Final Master Equation (Pr-Independent) ---")
print(f"Nu(Ra) = 1 + {A_mean:.3f} * ((Ra - Rac)/Rac)^{x_mean:.3f}")

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot A vs Pr
ax1.plot(Pr_arr, A_arr, 'ko', markersize=8, markerfacecolor='none', label='Calculated Multipliers ($A$)')
ax1.axhline(A_mean, color='red', linestyle='--', label=f'Mean ($A \\approx {A_mean:.3f}$)')
ax1.axhspan(A_mean * 0.99, A_mean * 1.01, color='red', alpha=0.1, label=r'$\pm  0.01\bar{A}$')
ax1.set_xscale('log') 
ax1.set_xlabel('Prandtl Number ($Pr$)')
ax1.set_ylabel('Multiplier ($A$)')
ax1.set_title('Multiplier vs Pr (Pr-Independent)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot x vs Pr
ax2.plot(Pr_arr, x_arr, 'ko', markersize=8, markerfacecolor='none', label='Calculated Exponents ($x$)')
ax2.axhline(x_mean, color='blue', linestyle='--', label=f'Mean ($x \\approx {x_mean:.3f}$)')
ax2.axhspan(x_mean * 0.98, x_mean * 1.02, color='blue', alpha=0.1, label=r'$\pm  0.02\bar{x}$')
ax2.set_xscale('log') 
ax2.set_xlabel('Prandtl Number ($Pr$)')
ax2.set_ylabel('Exponent ($x$)')
ax2.set_title('Exponent vs Pr (Pr-Independent)')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('A_and_x_vs_Pr_Constant.png', dpi=300)