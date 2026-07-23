import numpy as np
import dedalus.public as d3
import scipy.interpolate as interp
import logging
import os
import sys

# --- HPC PLOTTING FIX ---
import matplotlib
matplotlib.use('Agg')  # Force headless rendering to prevent MPI deadlocks
import matplotlib.pyplot as plt

from mpi4py import MPI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Global Parameters ----------------
Nx, Nz = 64, 32
L_val = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
restart = int(sys.argv[2]) if len(sys.argv) > 2 else 0
Rayleigh = 1e5
Prandtl = 1

stop_sim_time = 5.0  # Time allowed for the fluid to settle into steady state
max_timestep = 1e-3
dtype = np.float64
timestepper = d3.RK222

# --- Time-Averaging Parameters ---
averaging_window = 0.5  # Average over the last 0.5 time units
start_avg_time = stop_sim_time - averaging_window

coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
ex, ez = coords.unit_vector_fields(dist)

# ---------------- Setup Native Fields & Bases ----------------
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, L_val), dealias=3/2)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, 1), dealias=3/2)
x, z = dist.local_grids(xbasis, zbasis)

p = dist.Field(name='p', bases=(xbasis, zbasis))
T = dist.Field(name='T', bases=(xbasis, zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))

tau_p = dist.Field(name='tau_p')
tau_T1 = dist.Field(name='tau_T1', bases=xbasis)
tau_T2 = dist.Field(name='tau_T2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

grad_u = d3.grad(u) + ez*lift(tau_u1)
grad_T = d3.grad(T) + ez*lift(tau_T1)

# ---------------- IVP Problem Setup ----------------
namespace = {**globals(), **locals()}
problem = d3.IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2], namespace=namespace)

problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(T) - div(grad_T) + lift(tau_T2) = -(u@grad(T))")
problem.add_equation("dt(u) - Prandtl*div(grad_u) + grad(p) - Prandtl*Rayleigh*T*ez + lift(tau_u2) = -(u@grad(u))")

problem.add_equation("T(z=0) = 1")
problem.add_equation("T(z=1) = 0")
problem.add_equation("u(z=0) = 0")
problem.add_equation("u(z=1) = 0")
problem.add_equation("integ(p) = 0")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# ---------------- Initial Conditions (Continuation Logic) ----------------
T.change_scales(1)
u.change_scales(1)
p.change_scales(1)

if restart == 1:
    if dist.comm.rank == 0:
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            data = np.load(os.path.join(BASE_DIR, 'steady_state_solution.npz'))
            old_L = float(data['L_val'])
            T_g = data['T_global']
            u_g = data['u_global']
            p_g = data['p_global']
            success = True
            logger.info(f"Loaded previous state (L={old_L:.3f}). Interpolating to L={L_val:.3f}...")
        except Exception as e:
            logger.error(f"Restart failed: {e}. Falling back to analytical guess.")
            success = False
    else:
        success, old_L, T_g, u_g, p_g = None, None, None, None, None

    success = dist.comm.bcast(success, root=0)

    if success:
        old_L = dist.comm.bcast(old_L, root=0)
        T_g = dist.comm.bcast(T_g, root=0)
        u_g = dist.comm.bcast(u_g, root=0)
        p_g = dist.comm.bcast(p_g, root=0)

        # 1D Spline Interpolation: Map the local MPI grid coordinates to the old domain space
        x_local = x[:, 0]
        x_eval = x_local * (old_L / L_val)
        x_old_global = np.linspace(0, old_L, Nx, endpoint=False)

        # Interpolate and inject directly into the local Dedalus fields
        f_T = interp.interp1d(x_old_global, T_g, axis=0, kind='cubic', fill_value='extrapolate')
        T['g'] = f_T(x_eval)
        
        f_p = interp.interp1d(x_old_global, p_g, axis=0, kind='cubic', fill_value='extrapolate')
        p['g'] = f_p(x_eval)

        f_u = interp.interp1d(x_old_global, u_g, axis=1, kind='cubic', fill_value='extrapolate')
        u_g_new_local = f_u(x_eval)
        u['g'][0] = u_g_new_local[0]
        u['g'][1] = u_g_new_local[1]
    else:
        restart = 0

if restart == 0:
    if dist.comm.rank == 0:
        logger.info("Using analytical sine/cosine initial guess...")
    T['g'] = 1 - z
    amp = 1.0
    k_x = 2 * np.pi / L_val 
    u['g'][0] = 2 * amp * np.pi * np.sin(k_x * x) * np.sin(np.pi * z) * np.cos(np.pi * z)
    u['g'][1] = -amp * k_x * np.cos(k_x * x) * (np.sin(np.pi * z)**2)
    T['g'] += 0.2 * np.cos(k_x * x) * np.sin(np.pi * z)

# ---------------- CFL & Time-Stepping Loop ----------------
CFL = d3.CFL(solver, initial_dt=1e-6, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Initialize variables for running time-average
sample_count = 0
sum_u_c = np.zeros_like(u['c'])
sum_T_c = np.zeros_like(T['c'])
sum_p_c = np.zeros_like(p['c'])
sum_J = 0.0

logger.info("Starting IVP time-integration to reach convective steady state...")

while solver.proceed:
    timestep = CFL.compute_timestep()
    solver.step(timestep)
    
    # --- Time-Averaging Accumulator ---
    if solver.sim_time >= start_avg_time:
        # MPI-Safe extraction of current Nusselt number
        J_integrand = d3.integ(1 + (u@ez) * T) / L_val
        J_val_array = J_integrand.evaluate()['g']
        local_J = J_val_array.flatten()[0] if J_val_array.size > 0 else 0.0
        J_current = dist.comm.bcast(local_J, root=0)
        
        # Accumulate sums
        sum_J += J_current
        sum_u_c += u['c']
        sum_T_c += T['c']
        sum_p_c += p['c']
        sample_count += 1
    
    if solver.iteration % 100 == 0:
        logger.info(f"Iteration: {solver.iteration}, Sim Time: {solver.sim_time:.4f}, Timestep: {timestep:.2e}")

# ---------------- Apply Time-Average ----------------
if sample_count > 0:
    J_val = sum_J / sample_count
    u['c'] = sum_u_c / sample_count
    T['c'] = sum_T_c / sample_count
    p['c'] = sum_p_c / sample_count
    logger.info(f"Averaging Complete. Final Time-Averaged Nusselt (J): {J_val:.4f} (over {sample_count} samples)")
else:
    logger.warning("Simulation did not reach averaging window! Falling back to instantaneous final state.")
    J_integrand = d3.integ(1 + (u@ez) * T) / L_val
    J_val_array = J_integrand.evaluate()['g']
    local_J = J_val_array.flatten()[0] if J_val_array.size > 0 else 0.0
    J_val = dist.comm.bcast(local_J, root=0)


# ---------------- Save Steady State for Adjoint & Continuation ----------------
# Gather global data first so they can be written to the .npz file
T.change_scales(1)
u.change_scales(1)
p.change_scales(1)

x_global = xbasis.global_grid(dist, scale=1)
z_global = zbasis.global_grid(dist, scale=1)

T_global = T.allgather_data('g')
u_global = u.allgather_data('g')
p_global = p.allgather_data('g')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if dist.comm.rank == 0:
    # 1. Save data for optimization loop and next continuation run
    save_path = os.path.join(BASE_DIR, 'steady_state_solution.npz')
    np.savez(save_path, 
             u_coeffs=u['c'], T_coeffs=T['c'], p_coeffs=p['c'],
             T_global=T_global, u_global=u_global, p_global=p_global,
             L_val=L_val, Rayleigh=Rayleigh, Prandtl=Prandtl,
             Nx=Nx, Nz=Nz, J_val=J_val)
    logger.info(f"Steady-state solution saved successfully to {save_path}")

    # 2. Draw Plot
    logger.info("Generating steady-state plot...")
    
    X, Z = np.meshgrid(x_global[:,0], z_global[0,:], indexing='ij')
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(X, Z, T_global, cmap='RdBu_r', shading='gouraud')
    plt.colorbar(label='Temperature (T)')
    
    U_x = u_global[0]
    U_z = u_global[1]
    
    magnitude = np.sqrt(U_x**2 + U_z**2)
    safe_magnitude = np.where(magnitude == 0, 1.0, magnitude)
    
    log_magnitude = np.log1p(magnitude) 
    U_x_log = U_x * (log_magnitude / safe_magnitude)
    U_z_log = U_z * (log_magnitude / safe_magnitude)
    
    stride_x, stride_z = 4, 2 
    plt.quiver(X[::stride_x, ::stride_z].T, Z[::stride_x, ::stride_z].T, 
               U_x_log[::stride_x, ::stride_z].T, U_z_log[::stride_x, ::stride_z].T, 
               color='k', alpha=0.7, scale=100) 
    
    plt.title(f'Time-Averaged Convection (Log-scaled Arrows, L={L_val:.2f})')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.tight_layout()
    
    plot_filename = os.path.join(BASE_DIR, 'steady_state_boussinesq.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    logger.info(f"Plot saved successfully as '{plot_filename}'")

# Force all cores to safely exit together
dist.comm.Barrier()