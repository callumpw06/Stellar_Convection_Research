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

# --- Optimization Hyperparameters & Settings ---
max_iterations = 15
alpha = 0.5  # Learning rate for the initial conditions
L_val = 3.0  # Fixed domain width for this problem

# SET BOUNDARY CONDITION HERE: 'free-slip' or 'no-slip'
BOUNDARY_CONDITION = 'free-slip' 

J_history = []

logger.info(f"========== Starting Optimization ({BOUNDARY_CONDITION} boundaries) ==========")

# Initial setup: Clear old files
if os.path.exists(IC_FILE):
    os.remove(IC_FILE)

for i in range(max_iterations):
    logger.info(f"\n--- Iteration {i+1}/{max_iterations} ---")
    
    # 1. Run the Forward Problem
    logger.info("Executing Forward Solver...")
    # Pass both L_val and the boundary condition setting
    forward_cmd = [sys.executable, FORWARD_SCRIPT, str(L_val), BOUNDARY_CONDITION]
    forward_result = subprocess.run(forward_cmd, capture_output=True, text=True, cwd=BASE_DIR)

    # Save BOTH the ICs and the Nu History for Iter 0 here
    if i == 0 and os.path.exists(IC_FILE):
        shutil.copy(IC_FILE, os.path.join(BASE_DIR, 'ic_iter_0.npz'))
        shutil.copy('steady_state_solution.npz', os.path.join(BASE_DIR, 'nu_history_iter_0.npz'))
    
    if forward_result.returncode != 0:
        logger.error(f"Forward solver failed!\n{forward_result.stderr}")
        break
        
    data = np.load('steady_state_solution.npz')
    J_current = float(data['J_val'])
    J_history.append(J_current)
    np.save('J_history.npy', J_history)
    logger.info(f"Forward pass complete. Objective (J) = {J_current:.6f}")

    # 2. Run the Adjoint Problem
    logger.info("Executing Time-Dependent Adjoint Solver...")
    # Pass both L_val and the boundary condition setting
    adjoint_cmd = [sys.executable, ADJOINT_SCRIPT, str(L_val), BOUNDARY_CONDITION]
    adjoint_result = subprocess.run(adjoint_cmd, capture_output=True, text=True, cwd=BASE_DIR)
    
    if adjoint_result.returncode != 0:
        logger.error(f"Adjoint solver failed!\n{adjoint_result.stderr}")
        break

    # 3. Execute Gradient Ascent Step
    logger.info("Applying Gradient Ascent to Initial Conditions...")
    
    ic_data = np.load(IC_FILE)
    u_0 = ic_data['u_0']
    T_0 = ic_data['T_0']
    
    if i == 0:
        steady_state_data = np.load('steady_state_solution.npz')
        target_KE = steady_state_data['target_KE']
        logger.info(f"Kinetic Energy Budget locked at: {target_KE:.4e}")
    
    if BOUNDARY_CONDITION == 'no-slip':
        # Load raw gradients evaluated at t=0
        grad_data = np.load(GRADIENT_FILE)
        dJ_du = grad_data['lambda_0']
        dJ_dT = grad_data['theta_0']
        
        # 1. Find the 99th percentile for each to ignore boundary spikes
        p99_u = np.percentile(np.abs(dJ_du), 99)
        p99_T = np.percentile(np.abs(dJ_dT), 99)
        
        # --- FIX 1: Find the GLOBAL maximum to preserve the physical ratio ---
        global_max = max(p99_u, p99_T)
        
        logger.info(f"99th Percentile Velocity: {p99_u:.2e}, Temperature: {p99_T:.2e}")
        logger.info(f"Normalizing both fields by Global Max: {global_max:.2e}")
        
        # 2. Clip spikes and normalize TOGETHER
        if global_max > 1e-14:
            dJ_du = np.clip(dJ_du, -p99_u, p99_u) / global_max
            dJ_dT = np.clip(dJ_dT, -p99_T, p99_T) / global_max
        else:
            dJ_du = np.zeros_like(dJ_du)
            dJ_dT = np.zeros_like(dJ_dT)

        # --- FIX 2: Take a small, pure gradient step (No KE forcing!) ---
        # We use a single shared learning rate now.
        alpha_T = 1 
        alpha_u = 5
        
        T_0_new = T_0 + alpha_T * dJ_dT
        u_0_new = u_0 + alpha_u * dJ_du
    elif BOUNDARY_CONDITION == 'free-slip':
        # Load raw gradients evaluated at t=0
        grad_data = np.load(GRADIENT_FILE)
        dJ_du = grad_data['lambda_0']
        dJ_dT = grad_data['theta_0']
        
        # 1. Find the 99th percentile for each to ignore boundary spikes
        p99_u = np.percentile(np.abs(dJ_du), 99)
        p99_T = np.percentile(np.abs(dJ_dT), 99)
        
        global_max = max(p99_u, p99_T)
        
        logger.info(f"99th Percentile Velocity: {p99_u:.2e}, Temperature: {p99_T:.2e}")
        logger.info(f"Normalizing both fields by Global Max: {global_max:.2e}")
        
        # 2. Clip spikes and normalize TOGETHER
        if global_max > 1e-14:
            dJ_du = np.clip(dJ_du, -p99_u, p99_u) / global_max
            dJ_dT = np.clip(dJ_dT, -p99_T, p99_T) / global_max
        else:
            dJ_du = np.zeros_like(dJ_du)
            dJ_dT = np.zeros_like(dJ_dT)

        # We use a single shared learning rate now.
        alpha_T = 0.1
        alpha_u = 0.1
        
        T_0_new = T_0 + alpha_T * dJ_dT
        u_0_new = u_0 + alpha_u * dJ_du
    else:
        raise ValueError(f"Unknown BOUNDARY_CONDITION: {BOUNDARY_CONDITION}")
    

    # Enforce exact Kinetic Energy conservation
    current_KE = np.sum(u_0_new**2)
    if current_KE > 0:
        scaling_factor = np.sqrt(target_KE / current_KE)
        u_0_new = u_0_new * scaling_factor
        
    # --- DYNAMIC BOUNDARY CLAMPING ---
    if BOUNDARY_CONDITION == 'no-slip':
        # Zero out both velocity components at the walls
        u_0_new[:, :, 0] = 0; u_0_new[:, :, -1] = 0
    elif BOUNDARY_CONDITION == 'free-slip':
        # Zero out only the vertical velocity (u_z is index 1) at the walls
        u_0_new[1, :, 0] = 0; u_0_new[1, :, -1] = 0
    else:
        raise ValueError(f"Unknown BOUNDARY_CONDITION: {BOUNDARY_CONDITION}")
    
    T_0_new[:, 0] = 1.0; T_0_new[:, -1] = 0.0
    T_0_new = np.clip(T_0_new, 0.0, 1.0)

    # Save and Archive
    np.savez(IC_FILE, u_0=u_0_new, T_0=T_0_new)
    logger.info("Initial conditions updated successfully.")
    
    archive_file = os.path.join(BASE_DIR, f'ic_iter_{i+1}.npz')
    shutil.copy(IC_FILE, archive_file)
    shutil.copy('steady_state_solution.npz', os.path.join(BASE_DIR, f'nu_history_iter_{i+1}.npz'))

logger.info("\n========== Optimization Loop Complete ==========")