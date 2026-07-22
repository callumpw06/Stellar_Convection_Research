import numpy as np
import subprocess
import logging
import matplotlib.pyplot as plt
import os

# Set up logging for the wrapper
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- Optimization Hyperparameters ----------------
L_current = 2.0         # Starting domain width
max_iterations = 10     # Number of gradient ascent steps to take
alpha = 0.5             # Learning rate (step size)

L_history = [L_current]
J_history = []

logger.info("========== Starting Adjoint Shape Optimization Loop ==========")

for i in range(max_iterations):
    logger.info(f"\n--- Iteration {i+1}/{max_iterations} | Current L: {L_current:.4f} ---")
    
    # 1. Run the Forward Problem
    logger.info("Executing Forward Solver...")
    forward_result = subprocess.run(
        ["python", "forward_solver.py", str(L_current)], 
        capture_output=True, text=True
    )
    
    if forward_result.returncode != 0:
        logger.error(f"Forward solver failed! Log:\n{forward_result.stderr}")
        break
        
    # Read the Nusselt number (J) from the saved forward state
    data = np.load('steady_state_solution.npz')
    J_current = float(data['J_val'])
    J_history.append(J_current)
    logger.info(f"Forward pass complete. Objective (J) = {J_current:.4f}")

    # 2. Run the Adjoint Problem
    logger.info("Executing Adjoint Solver...")
    adjoint_result = subprocess.run(
        ["python", "adjoint_solver.py"], 
        capture_output=True, text=True
    )
    
    if adjoint_result.returncode != 0:
        logger.error(f"Adjoint solver failed! Log:\n{adjoint_result.stderr}")
        break

    # Read the computed gradient from the text file
    dJ_dL = np.loadtxt('gradient.txt')
    logger.info(f"Adjoint pass complete. Gradient (dJ/dL) = {dJ_dL:.6e}")
    
    # 3. Execute Gradient Ascent Step
    L_new = L_current + (alpha * dJ_dL)
    
    # Optional: Clip L to prevent it from collapsing to 0 or blowing up too large
    L_new = np.clip(L_new, 0.5, 10.0)
    
    logger.info(f"Update: L shifted from {L_current:.4f} to {L_new:.4f} (Step: {alpha * dJ_dL:.4e})")
    
    L_current = L_new
    L_history.append(L_current)

logger.info("\n========== Optimization Loop Complete ==========")

# ---------------- Plotting the Optimization History ----------------
logger.info("Generating optimization history plots...")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Safely align histories for plotting regardless of where the loop broke
# The number of evaluated L's is exactly the number of successful J evaluations
evaluated_L = L_history[:len(J_history)]

ax1.plot(range(len(evaluated_L)), evaluated_L, marker='o', color='b', linewidth=2)
ax1.set_title('Domain Width ($L$) over Iterations')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('$L$')
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.plot(range(len(J_history)), J_history, marker='s', color='r', linewidth=2)
ax2.set_title('Nusselt Number ($J$) over Iterations')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('$J$')
ax2.grid(True, linestyle='--', alpha=0.7)

ax3.plot(evaluated_L, J_history, marker='^', color='g', linewidth=2)
ax3.set_title('Objective vs Domain Width ($J$ vs $L$)')
ax3.set_xlabel('$L$')
ax3.set_ylabel('$J$')
ax3.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('optimization_results.png', dpi=300)
logger.info("Results saved to 'optimization_results.png'.")