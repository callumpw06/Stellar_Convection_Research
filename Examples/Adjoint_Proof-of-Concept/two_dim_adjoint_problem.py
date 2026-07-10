"""
This is a simple example of a two-dimensional adjoint problem. The forward problem is a one-dimensiona
heat equation defined by the PDE:
    ∂y/∂t - v ∂²y/∂x² = 0
    y(x, 0) = u(x)
    y(L, t) = 0, y(0, t) = 100

With objective functional:
    J(u) = ∫₀ᵀ ∫₀ᴸ -v ∂y/∂x - γ/2 (u(x))² dx dt

The adjoint problem is defined by the PDE:
    −∂λ/∂t + ∂2λ/∂x² = 0
    λ(x, T) = 0
    λ(L, t) = λ(0, t) = 0

The adjoint problem is solved backwards in time to find λ(x, 0) which is then used to compute the 
gradient of the objective functional with respect to the control variable u(x).

We then iterate through u_n+1 = (1 - alpha * gamma) * u_n + alpha * λ(x, 0) to find the optimal 
control variable u(x) that minimizes the objective functional J(u).
"""

import numpy as np
import matplotlib.pyplot as plt

# Problem parameters adjusted for CFL stability
Nx = 50                # Spatial resolution
Nt = 2500              # Time steps
L = 2.0                # Domain length
T = 10.0               # Total time
dx = L / (Nx - 1)
dt = T / (Nt - 1)
alpha = 0.05           # Learning rate
gamma = 1.0            # Regularization parameter
nu = 0.1               # Diffusion coefficient
iterations = 100

# Check CFL condition
cfl = nu * dt / (dx**2)
print(f"CFL Number: {cfl:.4f} (Must be <= 0.5)")

# Discretised x and t axes
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

# Initial guess for control u (Linear interpolation to satisfy boundaries)
u = np.linspace(100.0, 0.0, Nx)

def forward_solve(u):
    """Solve ∂y/∂t - v ∂²y/∂x² = 0 with y(x, 0) = u(x) using forward Euler."""
    y = np.zeros((Nt, Nx))
    y[0, :] = u
    for n in range(0, Nt-1):
        y[n+1, 1:-1] = y[n, 1:-1] + dt * nu * (y[n, 2:] - 2*y[n, 1:-1] + y[n, :-2]) / dx**2
        y[n+1, 0] = 100
        y[n+1, -1] = 0
    return y

def adjoint_solve(y):
    """Solve −∂λ/∂t + ∂²λ/∂x² = 0 with λ(x, T) = 0 solved backwards."""
    lam = np.zeros((Nt, Nx))
    for n in range(Nt-2, -1, -1):
        lam[n, 1:-1] = lam[n+1, 1:-1] + dt * nu * (lam[n+1, 2:] - 2*lam[n+1, 1:-1] + lam[n+1, :-2]) / dx**2
        lam[n, 0] = 0
        lam[n, -1] = 0
    return lam

def cost_function(y, u):
    """Compute cost functional J."""
    dydx = np.gradient(y, dx, axis=1)
    term1 = np.sum(-nu * dydx) * dt * dx
    term2 = np.sum(gamma/2 * u**2) * dx 
    return term1 + term2

# --- History Tracking ---
cost_history = []
u_history = []
q_history = []  # New list to track heat flux
capture_iters = [0, 10, 50, 99]  # Snapshots to plot later

# Gradient descent loop
for k in range(iterations):
    # Forward & Adjoint solves
    y = forward_solve(u)
    lam = adjoint_solve(y)
    
    # Calculate and store cost
    J = cost_function(y, u)
    cost_history.append(J)
    
    # Store u(x) and time-averaged heat flux q(x) at specific milestones
    if k in capture_iters:
        u_history.append((k, u.copy()))
        
        # Calculate time-averaged heat flux: q = -nu * dy/dx
        dydx = np.gradient(y, dx, axis=1)
        q_time_avg = np.mean(-nu * dydx, axis=0)
        q_history.append((k, q_time_avg))
        
    # Gradient calculation (with correct sign for regularization)
    grad = lam[0, :] + gamma * u
    
    # Update control (Only interior points to preserve boundary conditions)
    u[1:-1] -= alpha * grad[1:-1]

    # Terminal output
    if k % 10 == 0:
        print(f"Iteration {k:03d}, Cost: {J:.4f}")

# Final forward solve for plotting
y_final = forward_solve(u)

# --- Plotting Results (2x2 Grid) ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1 (Top Left): Final Forward Solution
im = axs[0, 0].imshow(y_final, extent=[0, L, 0, T], origin='lower', aspect='auto', cmap='hot')
fig.colorbar(im, ax=axs[0, 0], label='Temperature')
axs[0, 0].set_title('Final Forward Solution $y(x, t)$')
axs[0, 0].set_xlabel('Position (x)')
axs[0, 0].set_ylabel('Time (t)')

# Plot 2 (Top Right): Evolution of u(x)
colors = ['lightgray', 'lightblue', 'royalblue', 'darkblue']
for i, (k, u_snap) in enumerate(u_history):
    axs[0, 1].plot(x, u_snap, label=f'Iter {k}', color=colors[i], linewidth=2)
axs[0, 1].set_title('Evolution of Control $u(x)$')
axs[0, 1].set_xlabel('Position (x)')
axs[0, 1].set_ylabel('Temperature $u(x)$')
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=0.3)

# Plot 3 (Bottom Left): Cost Function History
axs[1, 0].plot(range(iterations), cost_history, color='crimson', linewidth=2)
axs[1, 0].set_title('Optimizer Progress')
axs[1, 0].set_xlabel('Iteration')
axs[1, 0].set_ylabel('Cost Function $J$')
axs[1, 0].grid(True, alpha=0.3)

# Plot 4 (Bottom Right): Evolution of Heat Flux
for i, (k, q_snap) in enumerate(q_history):
    axs[1, 1].plot(x, q_snap, label=f'Iter {k}', color=colors[i], linewidth=2)
axs[1, 1].set_title('Evolution of Time-Averaged Heat Flux')
axs[1, 1].set_xlabel('Position (x)')
axs[1, 1].set_ylabel('Heat Flux $q(x)$')
axs[1, 1].legend()
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.title('Two-Dimensional Adjoint Problem: Optimization Results')
plt.savefig('two_dim_adjoint_results.png', dpi=300)
plt.show()