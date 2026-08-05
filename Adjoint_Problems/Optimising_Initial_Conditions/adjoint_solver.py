import numpy as np
import dedalus.public as d3
import h5py
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Parameters ---
Nx, Nz = 128, 64
L_val = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
Rayleigh = 1e5
Prandtl = 1.0
dtype = np.float64
timestepper = d3.RK222
stop_sim_time = 1.0
averaging_window = 1.0
start_avg_time = stop_sim_time - averaging_window

coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
ex, ez = coords.unit_vector_fields(dist)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, L_val), dealias=3/2)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, 1), dealias=3/2)

# Background Fields
u_bar = dist.VectorField(coords, name='u_bar', bases=(xbasis, zbasis))
T_bar = dist.Field(name='T_bar', bases=(xbasis, zbasis))

# --- NEW: Time-gating field for the objective function ---
avg_gate = dist.Field(name='avg_gate')

# Adjoint Fields
lambda_adj = dist.VectorField(coords, name='lambda_adj', bases=(xbasis, zbasis))
pi_adj = dist.Field(name='pi_adj', bases=(xbasis, zbasis))
theta_adj = dist.Field(name='theta_adj', bases=(xbasis, zbasis))

tau_pi = dist.Field(name='tau_pi')
tau_lambda1 = dist.VectorField(coords, name='tau_lambda1', bases=xbasis)
tau_lambda2 = dist.VectorField(coords, name='tau_lambda2', bases=xbasis)
tau_theta1 = dist.Field(name='tau_theta1', bases=xbasis)
tau_theta2 = dist.Field(name='tau_theta2', bases=xbasis)

lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_lambda = d3.grad(lambda_adj) + ez * lift(tau_lambda1)
grad_theta = d3.grad(theta_adj) + ez * lift(tau_theta1)

# --- Mathematical Forcing Terms ---
# Now multiplied by avg_gate so they vanish outside the end region
dF_dv = avg_gate * T_bar * ez
dF_dT = avg_gate * (u_bar @ ez)

transpose_term = (lambda_adj @ ex) * d3.grad(u_bar @ ex) + (lambda_adj @ ez) * d3.grad(u_bar @ ez)

# --- Time-Reversed Adjoint IVP[cite: 1] ---
# dt() here represents d/d(tau) where tau = T - t. 
# Therefore, standard diffusion terms maintain their signs on the LHS.
problem = d3.IVP([pi_adj, theta_adj, lambda_adj, tau_pi, tau_lambda1, tau_lambda2, tau_theta1, tau_theta2], namespace=locals())

problem.add_equation("trace(grad_lambda) + tau_pi = 0")
problem.add_equation("dt(lambda_adj) - Prandtl * div(grad_lambda) + d3.grad(pi_adj) + lift(tau_lambda2) = (u_bar @ grad_lambda) - transpose_term - theta_adj * d3.grad(T_bar) + dF_dv")
problem.add_equation("dt(theta_adj) - div(grad_theta) + lift(tau_theta2) = (u_bar @ grad_theta) - Prandtl * Rayleigh * (lambda_adj @ ez) + dF_dT")

problem.add_equation("lambda_adj(z=0) = 0")
problem.add_equation("lambda_adj(z=1) = 0")
problem.add_equation("theta_adj(z=0) = 0")
problem.add_equation("theta_adj(z=1) = 0")
problem.add_equation("integ(pi_adj) = 0")

solver = problem.build_solver(timestepper)

# --- Time-Stepping Backwards ---
# --- Time-Stepping Backwards (Exact Checkpointing) ---
import glob
snapshot_files = sorted(glob.glob('snapshots/*.h5'), reverse=True) 

logger.info("Integrating adjoint fields backward in time using exact checkpointed dt...")

lambda_adj['g'] = 0.0
theta_adj['g'] = 0.0

for file in snapshot_files:
    with h5py.File(file, mode='r') as f:
        # Extract the exact array of times recorded during the forward pass
        sim_times = f['scales']['sim_time'][:]
        num_writes = len(sim_times)
        
        # Loop backward from the last write down to the second write (index 1)
        for i in reversed(range(1, num_writes)):
            
            # 1. Calculate the exact dt taken by the forward solver for this specific step
            dt_exact = sim_times[i] - sim_times[i-1]
            
            u_bar.change_scales(1)
            T_bar.change_scales(1)
            
            # 2. Toggle the time-gate based on the exact simulation time
            current_t = sim_times[i]
            if current_t >= start_avg_time:
                avg_gate['g'] = 1.0
            else:
                avg_gate['g'] = 0.0
            
            # 3. Load the exact background fields
            u_bar['g'] = f['tasks']['u_bar'][i]
            T_bar['g'] = f['tasks']['T_bar'][i]
            
            # 4. Take the perfect backward step
            solver.step(dt_exact)

import scipy.ndimage as ndimage

# --- Output the Final Gradient at t=0 ---
if dist.comm.rank == 0:
    lambda_adj.change_scales(1)
    theta_adj.change_scales(1)
    
    # Extract raw arrays
    lam_raw = lambda_adj['g']
    theta_raw = theta_adj['g']
    
    # Apply Gaussian smoothing to remove spectral grid-scale noise
    lam_smoothed = np.zeros_like(lam_raw)
    lam_smoothed[0] = ndimage.gaussian_filter(lam_raw[0], sigma=2)
    lam_smoothed[1] = ndimage.gaussian_filter(lam_raw[1], sigma=2)
    theta_smoothed = ndimage.gaussian_filter(theta_raw, sigma=2)
    
    np.savez('gradient.npz', lambda_0=lam_smoothed, theta_0=theta_smoothed)
    logger.info("Adjoint pass complete. Smoothed initial condition gradients saved.")