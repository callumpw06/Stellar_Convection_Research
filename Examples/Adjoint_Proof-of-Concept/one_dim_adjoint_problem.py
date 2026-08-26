"""
This is a simple algorithm using adjoint-based optimisation to optimise an objective funtion, J,
subject to a PDE.

dy/dx + ay = u(x), y(0) = 0, a is a constant, u(x) is the control variable, and y(x) is the state variable.

J(u) = 0.5 * integral_0^1 (y(x) - y_target(x))^2 dx

We then attempt to solve the adjoint problem to find the optimal control u(x) that minimizes J(u).
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,          # Base font size
    'axes.titlesize': 18,     # Plot title size
    'axes.labelsize': 16,     # X and Y label size
    'xtick.labelsize': 14,    # X-axis tick numbers
    'ytick.labelsize': 14,    # Y-axis tick numbers
    'legend.fontsize': 12,    # Legend font size
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["cmr10"],                   # Matplotlib's built-in Computer Modern
    "mathtext.fontset": "cm",                  # Use Computer Modern for math equations
    "axes.formatter.use_mathtext": True,       # Use math text for axis tick labels
})

# Problem parameters
N = 100              # Number of grid points
L = 2.0              # Domain length
dx = L / (N - 1)
a = 1.0              # PDE coefficient
alpha = 0.1          # Learning rate for gradient descent
iterations = 100

# Discretised x-axis
x = np.linspace(0, L, N)

# Target state
y_target = np.cos(np.pi * x)*np.exp(-x)

# Initial guess for control u
u = np.zeros(N)

def forward_solve(u):
    """Solve dy/dx + a*y = u with y(0) = 0 using forward Euler."""
    y = np.zeros(N)
    for i in range(1, N):
        y[i] = y[i-1] + dx * (u[i-1] - a * y[i-1])
    return y

def adjoint_solve(y, y_target):
    """Solve -dλ/dx + a*λ = y - y_target with λ(L) = 0 (backward Euler)."""
    lam = np.zeros(N)
    for i in range(N-2, -1, -1):
        lam[i] = lam[i+1] + dx * ((y[i] - y_target[i]) + a * lam[i+1])
    return lam

def cost_function(y, y_target):
    """Compute cost functional J."""
    return 0.5 * np.sum((y - y_target)**2) * dx

# --- NEW: Setup storage for iterations to plot ---
plot_iters = [0, 50, 99]
saved_y = {}

# Gradient descent loop
for k in range(iterations):
    # Forward solve
    y = forward_solve(u)
    
    # --- NEW: Save specific iterations ---
    if k in plot_iters:
        saved_y[k] = y.copy()
    
    # Adjoint solve
    lam = adjoint_solve(y, y_target)
    
    # Gradient: dJ/du = λ
    grad = lam.copy()
    
    # Update control
    u -= alpha * grad
    
    # Monitor convergence
    if k % 10 == 0:
        J = cost_function(y, y_target)
        print(f"Iter {k:3d} | Cost: {J:.6f}")

# Final forward solve
y_opt = forward_solve(u)

# Plot results
plt.figure(figsize=(10, 4))
plt.plot(x, y_target, 'k--', linewidth=2, label='Target, $y_{target}$')

# --- NEW: Plot the saved iterations with transitioning colors ---
# Uses light grey to dark grey, finishing with a blue line for the final state
colors = ['#e0e0e0', '#a9a9a9', '#1f77b4'] 
for idx, iter_num in enumerate(plot_iters):
    label_str = f'Iter {iter_num}' if iter_num != plot_iters[-1] else 'Optimised, $y$ (Iter 99)'
    linewidth = 1.5 if iter_num != plot_iters[-1] else 2.0
    plt.plot(x, saved_y[iter_num], color=colors[idx], linewidth=linewidth, label=label_str)

plt.plot(x, u, 'r-', linewidth=1.5, label='Optimised control, $u$')

plt.xlabel('x')
plt.ylabel('State $y$ / Control $u$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Adjoint-Based Optimisation Example')
plt.tight_layout()
plt.savefig('one_dim_adjoint_optimisation_result.png', dpi=300)
plt.show()