import numpy as np
import dedalus.public as d3
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Parameters ---
Nx, Nz = 96, 48
L_val = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
BC_TYPE = sys.argv[2] if len(sys.argv) > 2 else 'free-slip'
Rayleigh = float(sys.argv[3]) if len(sys.argv) > 3 else 1e5
Prandtl = 1
if BC_TYPE == 'no-slip':
    stop_sim_time = 1.0
    averaging_window = 1.0 
    max_timestep = 1e-3
    initial_dt = 1e-6
elif BC_TYPE == 'free-slip':
    stop_sim_time = 1.0
    averaging_window = 0.10
    max_timestep = 1e-4
    initial_dt = 1e-6
else:
    raise ValueError(f"Unknown boundary condition: {BC_TYPE}")

start_avg_time = stop_sim_time - averaging_window
dtype = np.float64
timestepper = d3.RK443

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
problem.add_equation("integ(p) = 0")

# --- DYNAMIC BOUNDARY CONDITIONS ---
if BC_TYPE == 'no-slip':
    logger.info("Applying NO-SLIP boundaries to forward solver.")
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("u(z=1) = 0")
elif BC_TYPE == 'free-slip':
    logger.info("Applying FREE-SLIP boundaries to forward solver.")
    problem.add_equation("u(z=0)@ez = 0")
    problem.add_equation("ex@(ez@grad_u)(z=0) = 0")
    problem.add_equation("u(z=1)@ez = 0")
    problem.add_equation("ex@(ez@grad_u)(z=1) = 0")
else:
    raise ValueError(f"Unknown boundary condition: {BC_TYPE}")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# --- Load Initial Conditions ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ic_file = os.path.join(BASE_DIR, 'current_initial_conditions.npz')

logger.info("Initializing baseline zero-velocity guess...")
T.fill_random('g', seed=42, distribution='normal', scale=1e-3)
T['g'] += 1 - z
u['g'][0] = 0.0
u['g'][1] = 0.0
    
T.change_scales(1)
u.change_scales(1)

# --- Save Snapshots for Adjoint ---
out_dir = Path("snapshots") / f"Lx_{L_val}"
snapshots = solver.evaluator.add_file_handler(out_dir, sim_dt=max_timestep, max_writes=50)
snapshots.add_task(u, name='u_bar')
snapshots.add_task(T, name='T_bar')

# --- Time-stepping ---
CFL = d3.CFL(solver, initial_dt=initial_dt, cadence=10, safety=0.3, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

sum_J = 0.0; count = 0
t_history = []  
Nu_history = [] 
KE_history = []  

while solver.proceed:
    timestep = CFL.compute_timestep()
    solver.step(timestep)
    
    # --- SAFE EVALUATION FOR NUSSELT NUMBER ---
    Nu_inst = d3.integ(1 + (u@ez) * T) / L_val
    Nu_array = Nu_inst.evaluate()['g']
    local_Nu = Nu_array.flatten()[0] if Nu_array.size > 0 else 0.0
    Nu_val = dist.comm.bcast(local_Nu, root=0)

    # --- SAFE EVALUATION FOR KINETIC ENERGY ---
    KE_inst = d3.integ(0.5 * (u@u))
    KE_array = KE_inst.evaluate()['g']
    local_KE = KE_array.flatten()[0] if KE_array.size > 0 else 0.0
    KE_val = dist.comm.bcast(local_KE, root=0)
    
    # --- UPDATED LOGGING: Now prints every 100 steps with KE included ---
    if solver.iteration % 100 == 0:
        logger.info(f"Iter: {solver.iteration}, Time: {solver.sim_time:.4f}, dt: {timestep:.2e}, KE: {KE_val:.4e}")
    
    t_history.append(solver.sim_time)
    Nu_history.append(Nu_val)
    KE_history.append(KE_val)

    if solver.sim_time >= start_avg_time:
        sum_J += Nu_val
        count += 1

if count > 0:
    J_avg = sum_J / count
else:
    logger.warning("Missed the averaging window! Check your timestep/save frequency.")
    J_avg = 0.0

final_KE = np.sum(u['g']**2)

if dist.comm.rank == 0:
    np.savez('steady_state_solution.npz', J_val=J_avg, target_KE=final_KE, 
             t_history=t_history, KE_history=KE_history, Nu_history=Nu_history)
    logger.info(f"Forward pass complete. Time-averaged Nu = {J_avg:.4f}")