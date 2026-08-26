import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
custom_Ra = [657, 700, 800, 900,
            1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
            2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
            3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 
            4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900,
            5000,
            1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4,
            1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5,
            1e6]

#Pr_0_5_Nu = [1.0000, 1.1139, 1.3599, 1.5609,
            #1.7298, 1.8751, 2.0025, 2.1160, 2.2183, 2.3116, 2.3974, 2.4771, 2.5514, 2.6213,
            #2.6873, 2.7498, 2.8094, 2.8663, 2.9209, 2.9734, 3.0329, 3.0727, 3.1200, 3.1658,
            #3.2103, 3.2536, 3.2957, 3.3368, 3.3769, 3.4161, 3.4544, 3.4919, 3.5282, 3.5647,
            #3.6001, 3.6348, 3.6689, 3.7025, 3.7354, 3.7679, 3.7998, 3.8313, 3.8622, 3.8928,
            #3.9229]
Pr_1_data = [
    [657, 1.0000], [700, 1.1139], [800, 1.3599], [900, 1.5610], [1000, 1.7299], [1100, 1.8751], 
    [1200, 2.0024], [1300, 2.1157], [1400, 2.2177], [1500, 2.3107], [1600, 2.3962], [1700, 2.4754], 
    [1800, 2.5493], [1900, 2.6187], [2000, 2.6841], [2100, 2.7462], [2200, 2.8052], [2300, 2.8615], 
    [2400, 2.9155], [2500, 2.9673], [2600, 3.0172], [2700, 3.0654], [2800, 3.1120], [2900, 3.1572], 
    [3000, 3.2010], [3100, 3.2437], [3200, 3.2852], [3300, 3.3256], [3400, 3.3651], [3500, 3.4037], 
    [3600, 3.4414], [3700, 3.4783], [3800, 3.5145], [3900, 3.5499], [4000, 3.5847], [4100, 3.6189], 
    [4200, 3.6524], [4300, 3.68564], [4400, 3.7179], [4500, 3.7498], [4600, 3.7812], [4700, 3.8122], 
    [4800, 3.8427], [4900, 3.8738], [5000, 3.9024], [1e4, 4.453873], [2e4, 6.557562], [3e4, 7.668585], 
    [4e4, 8.509277], [5e4, 9.219622], [6e4, 9.841527], [7e4, 10.397941], [8e4, 10.903328], [9e4, 11.367127], 
    [1e5, 11.801203], [2e5, 15.061135], [3e5, 17.344679], [4e5, 19.175763], [5e5, 20.717278], [6e5, 22.061602], 
    [7e5, 23.267590], [8e5, 24.360903], [9e5, 25.353893], [1e6, 26.313938]
]
#Pr_10_Nu = [1.0000, 1.1139, 1.3600, 1.5612,
            #1.7302, 1.8756, 2.0030, 2.1163, 2.2184, 2.3113, 2.3967, 2.4757, 2.5493, 2.6183,
            #2.6832, 2.7446, 2.8029, 2.8584, 2.9115, 2.9622, 3.0110, 3.0579, 3.1031, 3.1467,
            #3.1889, 3.2298, 3.2695, 3.3080, 3.3454, 3.3818, 3.4173, 3.4519, 3.4857, 3.5187,
            #3.5509, 3.6189, 3.6524, 3.6854, 3.7179, 3.7498, 3.7812, 3.8122, 3.8427, 3.8728, 
            #3.9024]
#Pr_100_Nu = [1.0000, 1.1139, 1.3600, 1.5612,
            #1.7302, 1.8757, 2.0031, 2.1164, 2.2186, 2.3116, 2.3970, 2.4761, 2.5497, 2.6188,
            #2.6838, 2.7453, 2.8037, 2.8593, 2.9124, 2.9632, 3.0120, 3.0590, 3.1043, 3.1480,
            #3.1903, 3.2312, 3.2709, 3.3095, 3.3470, 3.3835, 3.4190, 3.4537, 3.4875, 3.5205,
            #3.5527, 3.5843, 3.6151, 3.6454, 3.6750, 3.7040, 3.7325, 3.7604, 3.7878, 3.8148,
            #3.8412]

# Map your parameters
pr_datasets = {
    #0.5: {'data': Pr_0_5_Nu, 'color': 'blue', 'marker': 'x'},
    1.0: {'data': Pr_1_data,   'color': 'red', 'marker': 'o'}#,
    #10:  {'data': Pr_10_Nu,  'color': 'green', 'marker': 's'},
    #100: {'data': Pr_100_Nu, 'color': 'purple', 'marker': '^'}
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

    # --- Strict mask for the pure Ra fit (Ra > 10,000) ---
    pure_fit_mask = Ra_arr > 1e4
    Ra_pure_fit = Ra_arr[pure_fit_mask]
    Nu_pure_fit = Nu_arr[pure_fit_mask]
    
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

    # 2. Pure Ra power law fit (High Ra only)
    popt_pure, _ = curve_fit(pure_power_law, Ra_pure_fit, Nu_pure_fit, p0=[0.1, 0.3])
    A_pure, b_pure = popt_pure
    
    print(f"Pr = {pr:<4} | Pure Ra Fit: Nu = 1 + {A_pure:.4f} * Ra^{b_pure:.4f}")
    
    # Use np.nan so the dashed line completely disappears below Ra=1e4
    props['fit_line_pure'] = np.where(
        Ra_continuous < 1e4, 
        np.nan, 
        1.0 + A_pure * Ra_continuous**b_pure 
    )
    props['legend_str_pure'] = f'$Nu = 1 + {A_pure:.3f}Ra^{{{b_pure:.3f}}}$ $(Ra > 10^4)$'

# =============================================================================
# --- 3. Figure 1: Standard Linear Scale Plot ---
# =============================================================================
fig1 = plt.figure(figsize=(10, 5)) 

for pr, props in pr_datasets.items():
    # Plot using the dynamically extracted Ra and Nu
    plt.plot(props['Ra_arr'], props['Nu_arr'], color=props['color'], marker=props['marker'], 
             linestyle='', markerfacecolor='none', linewidth=2, markersize=8, label=f'Data ($Pr={pr}$)')
    
    # Epsilon fit line
    plt.plot(Ra_continuous, props['fit_line'], color=props['color'], linestyle='-', 
             linewidth=1.5, alpha=0.7, label=f'Fit: {props["legend_str"]}')
    
    # Pure Ra fit line (dashed black)
    plt.plot(Ra_continuous, props['fit_line_pure'], color='black', linestyle='--', 
             linewidth=1.5, alpha=0.8, label=f'High-Ra Fit: {props["legend_str_pure"]}')

plt.title('Nusselt Number vs. Rayleigh Number (Linear Scale)', fontsize=14)
plt.xlabel('Rayleigh Number ($Ra$)', fontsize=12)
plt.ylabel('Nusselt Number ($Nu$)', fontsize=12)

plt.xscale('linear')  
plt.yscale('linear')  

plt.xlim(500, 1.5e6)
plt.ylim(0.8, 32)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', fontsize=11, ncol=1) 
plt.tight_layout()
plt.savefig('Nu_vs_Ra_Linear.png', dpi=300)

# =============================================================================
# --- 4. Figure 2: Semi-Log Scale Plot ---
# =============================================================================
fig2 = plt.figure(figsize=(10, 5)) 

for pr, props in pr_datasets.items():
    plt.plot(props['Ra_arr'], props['Nu_arr'], color=props['color'], marker=props['marker'], 
             linestyle='', markerfacecolor='none', linewidth=2, markersize=8, label=f'Data ($Pr={pr}$)')
    
    # Epsilon fit line
    plt.plot(Ra_continuous, props['fit_line'], color=props['color'], linestyle='-', 
             linewidth=1.5, alpha=0.7, label=f'Fit: {props["legend_str"]}')
    
    # Pure Ra fit line (dashed black)
    plt.plot(Ra_continuous, props['fit_line_pure'], color='black', linestyle='--', 
             linewidth=1.5, alpha=0.8, label=f'High-Ra Fit: {props["legend_str_pure"]}')

plt.title('Nusselt Number vs. Rayleigh Number (Semi-Log Scale)', fontsize=14)
plt.xlabel('Rayleigh Number ($Ra$)', fontsize=12)
plt.ylabel('Nusselt Number ($Nu$)', fontsize=12)

plt.xscale('log')  
plt.yscale('log')  

plt.xlim(500, 1.5e6)
plt.ylim(0.8, 32)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', fontsize=11, ncol=1)
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