import numpy as np
import subprocess
import logging
import matplotlib.pyplot as plt
import os
import sys
import shutil  # Added for backing up the fluid state

# Set up logging for the wrapper
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- File Path Safety ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORWARD_SCRIPT = os.path.join(BASE_DIR, "forward_solver.py")
ADJOINT_SCRIPT = os.path.join(BASE_DIR, "adjoint_solver.py")
DATA_FILE = os.path.join(BASE_DIR, 'steady_state_solution.npz')
BEST_DATA_FILE = os.path.join(BASE_DIR, 'best_state.npz')

# ---------------- Optimization Hyperparameters ----------------
L_current = 2.0         # Starting domain width
max_iterations = 15     # Increased slightly to allow for smaller backtracking steps
alpha = 0.1             # Learning rate (step size)

L_history = [L_current]
J_history = []

# Backtracking trackers
J_best = -np.inf
L_best = L_current
best_gradient = 0.0

# Safety clear to prevent loading stale data
if os.path.exists(DATA_FILE):
    os.remove(DATA_FILE)

logger.info("========== Starting Adjoint Shape Optimization Loop (Backtracking Mode) ==========")

for i in range(max_iterations):
    logger.info(f"\n--- Iteration {i+1}/{max_iterations} | Current L: {L_current:.4f} ---")
    
    restart_flag = "0" if i == 0 else "1"
    
    # 1. Run the Forward Problem
    logger.info("Executing Forward Solver...")
    forward_cmd = [sys.executable, FORWARD_SCRIPT, str(L_current), restart_flag]
    forward_result = subprocess.run(forward_cmd, capture_output=True, text=True, cwd=BASE_DIR)
    
    if forward_result.returncode != 0:
        logger.error(f"Forward solver failed! Log:\n{forward_result.stderr}")
        break
        
    data = np.load(DATA_FILE)
    J_current = float(data['J_val'])
    
    # --- BACKTRACKING LOGIC ---
    if J_current < J_best:
        logger.warning(f"Objective dropped to {J_current:.4f}! We fell off the cliff.")
        logger.info("Rejecting step, restoring previous fluid state, and halving the learning rate...")
        
        # Shrink the step size
        alpha *= 0.5 
        
        # Recalculate a smaller step from the LAST known good position
        raw_step = alpha * best_gradient
        safe_step = np.clip(raw_step, -0.15, 0.15)
        L_current = np.clip(L_best + safe_step, 0.5, 10.0)
        
        # Restore the good fluid state so continuation doesn't use the broken plumes
        shutil.copy(BEST_DATA_FILE, DATA_FILE)
        
        # Skip the adjoint solver and try the forward solver again with the new, smaller step
        continue 
    # --------------------------

    # If we made it here, it was a good step! Update our "best" trackers.
    J_best = J_current
    L_best = L_current
    shutil.copy(DATA_FILE, BEST_DATA_FILE) # Backup the good fluid state
    
    J_history.append(J_current)
    logger.info(f"Forward pass complete. New Best Objective (J) = {J_current:.4f}")

    # 2. Run the Adjoint Problem
    logger.info("Executing Adjoint Solver...")
    adjoint_cmd = [sys.executable, ADJOINT_SCRIPT]
    adjoint_result = subprocess.run(adjoint_cmd, capture_output=True, text=True, cwd=BASE_DIR)
    
    if adjoint_result.returncode != 0:
        logger.error(f"Adjoint solver failed! Log:\n{adjoint_result.stderr}")
        break

    dJ_dL = np.loadtxt(os.path.join(BASE_DIR, 'gradient.txt'))
    best_gradient = dJ_dL # Save the gradient in case we need to backtrack next round
    logger.info(f"Adjoint pass complete. Gradient (dJ/dL) = {dJ_dL:.6e}")
    
    # 3. Execute Gradient Ascent Step
    raw_step = alpha * dJ_dL
    safe_step = np.clip(raw_step, -0.15, 0.15)
    L_new = np.clip(L_current + safe_step, 0.5, 10.0)

    logger.info(f"Update: L shifted from {L_current:.4f} to {L_new:.4f} (Step: {safe_step:.4e})")
    
    L_current = L_new
    L_history.append(L_current)

logger.info("\n========== Optimization Loop Complete ==========")

# ---------------- Plotting the Optimization History ----------------
logger.info("Generating optimization history plots...")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Since backtracking skips appending to history on bad steps, 
# L_history will have more entries than J_history. We align them:
valid_L_history = []
current_L_idx = 0
for L in L_history:
    # Only plot the L values that resulted in a successful J increase (or the final step attempt)
    if current_L_idx < len(J_history):
        valid_L_history.append(L)
        current_L_idx += 1

ax1.plot(range(len(valid_L_history)), valid_L_history, marker='o', color='b', linewidth=2)
ax1.set_title('Successful Domain Widths ($L$)')
ax1.set_xlabel('Valid Iteration')
ax1.set_ylabel('$L$')
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.plot(range(len(J_history)), J_history, marker='s', color='r', linewidth=2)
ax2.set_title('Nusselt Number ($J$) over Iterations')
ax2.set_xlabel('Valid Iteration')
ax2.set_ylabel('$J$')
ax2.grid(True, linestyle='--', alpha=0.7)

ax3.plot(valid_L_history, J_history, marker='^', color='g', linewidth=2)
ax3.set_title('Objective vs Domain Width ($J$ vs $L$)')
ax3.set_xlabel('$L$')
ax3.set_ylabel('$J$')
ax3.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'optimization_results.png'), dpi=300)
logger.info("Results saved to 'optimization_results.png'.")