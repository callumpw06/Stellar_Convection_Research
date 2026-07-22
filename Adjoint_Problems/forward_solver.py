import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Global Parameters ----------------
Nx, Nz = 128, 64
L_val = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
Rayleigh = 1e5
Prandtl = 1

stop_sim_time = 2.0  # Time allowed for the fluid to settle into steady state
max_timestep = 1e-3
dtype = np.float64
timestepper = d3.RK222

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

# ---------------- Boundary-Compliant Initial Guess ----------------
T['g'] = 1 - z
amp = 1.0
k_x = 2 * np.pi / L_val 

# Streamfunction ensuring zero velocity at z=0 and z=1 walls
u['g'][0] = 2 * amp * np.pi * np.sin(k_x * x) * np.sin(np.pi * z) * np.cos(np.pi * z)
u['g'][1] = -amp * k_x * np.cos(k_x * x) * (np.sin(np.pi * z)**2)

# Thermal perturbation to kickstart convection
T['g'] += 0.2 * np.cos(k_x * x) * np.sin(np.pi * z)

# ---------------- CFL & Time-Stepping Loop ----------------
CFL = d3.CFL(solver, initial_dt=1e-6, cadence=10, safety=0.5, threshold=0.05,
              max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

logger.info("Starting IVP time-integration to reach convective steady state...")

while solver.proceed:
    timestep = CFL.compute_timestep()
    solver.step(timestep)
    
    if solver.iteration % 100 == 0:
        logger.info(f"Iteration: {solver.iteration}, Sim Time: {solver.sim_time:.4f}, Timestep: {timestep:.2e}")

# ---------------- Evaluate Nusselt Number (J) ----------------
J_integrand = d3.integ(1 + (u@ez) * T) / L_val
J_val = J_integrand.evaluate()['g'].flatten()[0]
logger.info(f"Reached Steady State. Final Nusselt Number (J): {J_val:.4f}")

# ---------------- Plotting ----------------
if dist.comm.rank == 0:
    logger.info("Generating steady-state plot...")
    
    T.change_scales(1)
    u.change_scales(1)
    
    X, Z = np.meshgrid(x[:,0], z[0,:], indexing='ij')
    
    plt.figure(figsize=(10, 4))
    
    # Plot Temperature field
    plt.pcolormesh(X, Z, T['g'], cmap='RdBu_r', shading='gouraud')
    plt.colorbar(label='Temperature (T)')
    
    # Extract velocity components
    U_x = u['g'][0]
    U_z = u['g'][1]
    
    # --- Logarithmic Length Scaling ---
    magnitude = np.sqrt(U_x**2 + U_z**2)
    safe_magnitude = np.where(magnitude == 0, 1.0, magnitude)
    
    # Log transform the lengths while preserving direction units (U/magnitude)
    log_magnitude = np.log1p(magnitude) 
    U_x_log = U_x * (log_magnitude / safe_magnitude)
    U_z_log = U_z * (log_magnitude / safe_magnitude)
    
    # Subsample with a stride to prevent visual clutter
    stride_x, stride_z = 4, 2 
    plt.quiver(X[::stride_x, ::stride_z].T, Z[::stride_x, ::stride_z].T, 
               U_x_log[::stride_x, ::stride_z].T, U_z_log[::stride_x, ::stride_z].T, 
               color='k', alpha=0.7, scale=100) # Adjust scale as needed
    
    plt.title(f'Steady State Convection (Log-scaled Arrows, L={L_val:.2f})')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.tight_layout()
    
    plot_filename = 'steady_state_boussinesq.png'
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    logger.info(f"Plot saved successfully as '{plot_filename}'")

    # ---------------- Save Steady State for Adjoint ----------------
    np.savez('steady_state_solution.npz', 
         u_coeffs=u['c'], T_coeffs=T['c'], p_coeffs=p['c'],
         L_val=L_val, Rayleigh=Rayleigh, Prandtl=Prandtl,
         Nx=Nx, Nz=Nz, J_val=J_val)
    logger.info("Steady-state solution saved successfully to 'steady_state_solution.npz'")