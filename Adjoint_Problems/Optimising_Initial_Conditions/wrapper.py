import numpy as np
import subprocess
import logging
import os
import sys
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORWARD_SCRIPT = os.path.join(BASE_DIR, "forward_solver.py")
ADJOINT_SCRIPT = os.path.join(BASE_DIR, "adjoint_solver.py")
IC_FILE = os.path.join(BASE_DIR, 'current_initial_conditions.npz')
GRADIENT_FILE = os.path.join(BASE_DIR, 'gradient.npz')

# --- Optimization Hyperparameters ---
max_iterations = 15
alpha = 0.5  # Learning rate for the initial conditions
L_val = 2.0   # Fixed domain width for this problem

J_history = []

logger.info("========== Starting Adjoint Initial Condition Optimization ==========")

# Initial setup: Clear old files
if os.path.exists(IC_FILE):
    os.remove(IC_FILE)

for i in range(max_iterations):
    logger.info(f"\n--- Iteration {i+1}/{max_iterations} ---")
    
    # 1. Run the Forward Problem
    logger.info("Executing Forward Solver...")
    forward_cmd = [sys.executable, FORWARD_SCRIPT, str(L_val)]
    forward_result = subprocess.run(forward_cmd, capture_output=True, text=True, cwd=BASE_DIR)

    if i == 0 and os.path.exists(IC_FILE):
        shutil.copy(IC_FILE, os.path.join(BASE_DIR, 'ic_iter_0.npz'))
    
    if forward_result.returncode != 0:
        logger.error(f"Forward solver failed!\n{forward_result.stderr}")
        break
        
    # Extract the objective function value from the forward solver's log
    # (Assuming forward solver saves J_val in a small tracking file or it's parsed from logs)
    data = np.load('steady_state_solution.npz')
    J_current = float(data['J_val'])
    J_history.append(J_current)
    np.save('J_history.npy', J_history)
    logger.info(f"Forward pass complete. Objective (J) = {J_current:.6f}")

    # 2. Run the Adjoint Problem
    logger.info("Executing Time-Dependent Adjoint Solver...")
    adjoint_cmd = [sys.executable, ADJOINT_SCRIPT, str(L_val)]
    adjoint_result = subprocess.run(adjoint_cmd, capture_output=True, text=True, cwd=BASE_DIR)
    
    if adjoint_result.returncode != 0:
        logger.error(f"Adjoint solver failed!\n{adjoint_result.stderr}")
        break

    # 3. Execute Gradient Ascent Step
    logger.info("Applying Gradient Ascent to Initial Conditions...")
    
    # Load initial conditions
    ic_data = np.load(IC_FILE)
    u_0 = ic_data['u_0']
    T_0 = ic_data['T_0']
    
    # Calculate the Kinetic Energy of the starting state (sum of squares)
    # We do this once at the start and save it as our constraint
    if i == 0:
        steady_state_data = np.load('steady_state_solution.npz')
        target_KE = steady_state_data['target_KE']
        logger.info(f"Kinetic Energy Budget locked at: {target_KE:.4e}")

    
    # Load raw gradients evaluated at t=0
    grad_data = np.load(GRADIENT_FILE)
    dJ_du = grad_data['lambda_0']
    dJ_dT = grad_data['theta_0']
    
    # 1. Find the maximum absolute value in each gradient array
    max_grad_u = np.max(np.abs(dJ_du))
    max_grad_T = np.max(np.abs(dJ_dT))
    
    logger.info(f"Raw Velocity Gradient Max: {max_grad_u:.2e}")
    logger.info(f"Raw Temperature Gradient Max: {max_grad_T:.2e}")
    
    # 2. Normalize Velocity Gradient (with a safety check to avoid division by zero)
    if max_grad_u > 1e-14:
        dJ_du = dJ_du / max_grad_u
    else:
        dJ_du = np.zeros_like(dJ_du)
        
    # 3. Normalize Temperature Gradient
    if max_grad_T > 1e-14:
        dJ_dT = dJ_dT / max_grad_T
    else:
        dJ_dT = np.zeros_like(dJ_dT)

    # --- Now apply the update using the decoupled learning rates ---
    alpha_T = 0.2
    alpha_u = 5 
    
    # Apply the temperature update (still using alpha_T = 0.05)
    T_0_new = T_0 + alpha_T * dJ_dT
    
    # Apply the velocity update (no gamma needed)
    u_0_new = u_0 + alpha_u * dJ_du
    
    # Enforce exact Kinetic Energy conservation
    current_KE = np.sum(u_0_new**2)
    if current_KE > 0:
        scaling_factor = np.sqrt(target_KE / current_KE)
        u_0_new = u_0_new * scaling_factor
        
    # Enforce boundary conditions and clamp temperature
    u_0_new[:, :, 0] = 0; u_0_new[:, :, -1] = 0
    T_0_new[:, 0] = 1.0; T_0_new[:, -1] = 0.0
    T_0_new = np.clip(T_0_new, 0.0, 1.0)

    # Save the updated conditions for the next forward pass
    np.savez(IC_FILE, u_0=u_0_new, T_0=T_0_new)
    logger.info("Initial conditions updated successfully.")
    
    # --- NEW: Archive the initial conditions for visualization ---
    archive_file = os.path.join(BASE_DIR, f'ic_iter_{i+1}.npz')
    shutil.copy(IC_FILE, archive_file)

logger.info("\n========== Optimization Loop Complete ==========")