import numpy as np
import dedalus.public as d3
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
stop_sim_time = 1.0
averaging_window = 1.0  # Time window over which to average the objective function
start_avg_time = stop_sim_time - averaging_window
max_timestep = 1e-3
dtype = np.float64
timestepper = d3.RK222

coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
ex, ez = coords.unit_vector_fields(dist)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, L_val), dealias=3/2)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, 1), dealias=3/2)
x, z = dist.local_grids(xbasis, zbasis)

# Fields
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

# --- IVP Setup ---
problem = d3.IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(T) - div(grad_T) + lift(tau_T2) = - u@grad(T)")
problem.add_equation("dt(u) - Prandtl*div(grad_u) + grad(p) - Prandtl*Rayleigh*T*ez + lift(tau_u2) = - u@grad(u)")
problem.add_equation("T(z=0) = 1")
problem.add_equation("T(z=1) = 0")
problem.add_equation("u(z=0) = 0")
problem.add_equation("u(z=1) = 0")
problem.add_equation("integ(p) = 0")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# --- Load Initial Conditions ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ic_file = os.path.join(BASE_DIR, 'current_initial_conditions.npz')

if os.path.exists(ic_file):
    logger.info("Loading optimized initial conditions...")
    ic_data = np.load(ic_file)
    u['g'] = ic_data['u_0']
    T['g'] = ic_data['T_0']
else:
    logger.info("Initializing baseline zero-velocity guess...")
    # 1. Base conductive state
    T['g'] = 1 - z
    
    # 2. Set velocity completely to zero
    u['g'][0] = 0.0
    u['g'][1] = 0.0
    
    # 3. Add a tiny temperature perturbation to break symmetry
    # This gives the adjoint solver a microscopic thread to pull on
    k_x = 2 * np.pi / L_val 
    T['g'] += 0.01 * np.cos(k_x * x) * np.sin(np.pi * z)
    
    # Save the baseline so the wrapper can update it later
    T.change_scales(1)
    u.change_scales(1)
    if dist.comm.rank == 0:
        np.savez(ic_file, u_0=u['g'], T_0=T['g'])

# --- Save Snapshots for Adjoint ---
# Triggering writes based on sim_dt prevents I/O throttling when CFL drops the timestep
# --- Save Snapshots for Adjoint (Exact Checkpointing) ---
# Saving every single iteration to ensure perfect time-symmetry in the adjoint pass
snapshots = solver.evaluator.add_file_handler('snapshots', iter=1, max_writes=5000)
snapshots.add_task(u, name='u_bar')
snapshots.add_task(T, name='T_bar')

# Time-stepping
CFL = d3.CFL(solver, initial_dt=1e-6, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

sum_J = 0.0; count = 0
while solver.proceed:
    timestep = CFL.compute_timestep()
    solver.step(timestep)
    
    if solver.sim_time >= start_avg_time:
        J_integrand = d3.integ(1 + (u@ez) * T) / L_val
        sum_J += J_integrand.evaluate()['g'].flatten()[0]
        count += 1

if count > 0:
    J_avg = sum_J / count
else:
    logger.warning("Missed the averaging window! Check your timestep/save frequency.")
    J_avg = 0.0

# --- Calculate Final Kinetic Energy for the Optimizer Budget ---
# We use np.sum(u['g']**2) to match the math we will use in the wrapper
final_KE = np.sum(u['g']**2)

if dist.comm.rank == 0:
    np.savez('steady_state_solution.npz', J_val=J_avg, target_KE=final_KE)
    logger.info(f"Forward pass complete. Time-averaged Nu = {J_avg:.4f}")
    logger.info(f"Final Kinetic Energy saved as budget: {final_KE:.4e}")