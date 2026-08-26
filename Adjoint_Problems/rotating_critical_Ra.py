import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress standard font warnings that sometimes occur with cmr10
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

# Your working LaTeX and font options
plt.rcParams.update({
    'font.size': 12,
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

def exact_k(Ta):
    """Calculates the exact critical wavenumber for a given Taylor number."""
    T_star = Ta / (np.pi**4)
    inner_sqrt = 2 * np.sqrt(T_star * (1 + T_star))
    term1 = 1 + 2*T_star + inner_sqrt
    term2 = 1 + 2*T_star - inner_sqrt
    Y = 0.5 * (np.cbrt(term1) + np.cbrt(term2))
    X = Y - 0.5
    return np.pi * np.sqrt(X)

def exact_Ra(Ta, k):
    """Calculates the exact critical Rayleigh number."""
    return ((k**2 + np.pi**2)**3 + np.pi**2 * Ta) / (k**2)

def asymptotic_k(Ta):
    """Calculates the rapid rotation asymptotic limit for wavenumber."""
    return np.pi * (Ta / (2 * np.pi**4))**(1/6)

def asymptotic_Ra(Ta):
    """Calculates the rapid rotation asymptotic limit for Rayleigh number."""
    return 3 * (np.pi**2 / 2)**(2/3) * Ta**(2/3)

# Generate an array of Taylor numbers (e.g., from 10^0 to 10^8)
Ta_values = np.logspace(0, 8, 500)

# Compute wavenumbers and Rayleigh numbers
k_exact = exact_k(Ta_values)
k_asymp = asymptotic_k(Ta_values)

Ra_exact = exact_Ra(Ta_values, k_exact)
Ra_asymp = asymptotic_Ra(Ta_values)

# Create a figure with two side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Subplot 1: Critical Wavenumber ---
ax1.plot(Ta_values, k_exact, label=r"Exact Solution", color="navy", linewidth=2)
ax1.plot(Ta_values, k_asymp, label=r"Asymptotic Limit ($Ta \to \infty$)", 
         color="darkred", linestyle="--", linewidth=2)

# Add asymptotic scaling label for k
ax1.text(1e6, 8.5, r"$k_c \propto Ta^{1/6}$", fontsize=18, color="darkred",
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"Taylor Number ($Ta$)")
ax1.set_ylabel(r"Critical Wavenumber ($k_c$)")
ax1.set_title(r"Wavenumber Scaling")
ax1.grid(True, which="both", ls=":", alpha=0.6)
ax1.legend(loc="upper left")

# --- Subplot 2: Critical Rayleigh Number ---
ax2.plot(Ta_values, Ra_exact, label=r"Exact Solution", color="navy", linewidth=2)
ax2.plot(Ta_values, Ra_asymp, label=r"Asymptotic Limit ($Ta \to \infty$)", 
         color="darkred", linestyle="--", linewidth=2)

# Add asymptotic scaling label for Ra
ax2.text(1e6, 1.5e4, r"$Ra_c \propto Ta^{2/3}$", fontsize=18, color="darkred", 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"Taylor Number ($Ta$)")
ax2.set_ylabel(r"Critical Rayleigh Number ($Ra_c$)")
ax2.set_title(r"Rayleigh Number Scaling")
ax2.grid(True, which="both", ls=":", alpha=0.6)
ax2.legend(loc="upper left")

plt.tight_layout()

# Save headlessly
plt.savefig("rotating_critical_scalings.png", bbox_inches="tight")