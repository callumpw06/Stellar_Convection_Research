"""
This is a simple algorithm using adjoint-based optimisation to optimise an objective funtion, J,
subject to a PDE.

dy/dx + ay = u(x), y(0) = 0, a is a constant, u(x) is the control variable, and y(x) is the state variable.

J(u) = 0.5 * integral_0^1 (y(x) - y_target(x))^2 dx

We then attempt to solve the adjoint problem to find the optimal control u(x) that minimizes J(u).
"""

import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
N = 100               # Number of grid points
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

# Gradient descent loop
for k in range(iterations):
    # Forward solve
    y = forward_solve(u)
    
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
plt.figure(figsize=(8,5))
plt.plot(x, y_target, 'k--', label='Target $y_{target}$')
plt.plot(x, y_opt, 'b-', label='Optimised $y$')
plt.plot(x, u, 'r-', label='Optimised control $u$')
plt.xlabel('x')
plt.legend()
plt.title('Adjoint-Based Optimisation Example')
plt.savefig('one_dim_adjoint_optimisation_result.png', dpi=300)
plt.show()
