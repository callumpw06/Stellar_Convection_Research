import numpy as np
import dedalus.public as d3
from scipy.optimize import minimize
import logging
import matplotlib.pyplot as plt
from mpi4py import MPI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Global Parameters ----------------
Nx, Nz = 128, 32
Lz = 1
Rayleigh = 2e4
Prandtl = 1

stop_sim_time = 1e-1
start_averaging_time = 5e-2  
max_timestep = 1e-5
dtype = np.float64
timestepper = d3.RK222

coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
ex, ez = coords.unit_vector_fields(dist)

def forward_problem(params):
    # Isolate L as the only optimization parameter
    L_val = params[0]
    
    if np.isnan(L_val) or np.isinf(L_val) or L_val <= 0:
        logger.warning(f"Optimizer attempted invalid L_val: {L_val}. Rejecting step.")
        return 1e9, np.zeros_like(params)

    logger.info(f"--- Starting Forward Pass for L = {L_val:.4f} ---")

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

    # ---------------- Forward Pass ----------------
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
    
    # Standard conductive profile initial conditions
    #local_shape = x.shape
    #local_noise = np.random.RandomState(seed=42 + dist.comm.rank).uniform(-0.01, 0.01, size=local_shape)
    perturbation = 0.01 * np.cos(2 * np.pi * x / L_val) * np.sin(np.pi * z)
    
    T['g'] = 1 - z + perturbation
    u['g'] = 0
    
    # Integrand for objective function J (divided by L_val to map dx to d(hat{x}))
    J_integrand = d3.integ(1 + (u@ez) * T) / L_val
    J_val = 0.0
    integrated_time = 0.0

    # CFL
    CFL = d3.CFL(solver, initial_dt=1e-7, cadence=10, safety=0.5, threshold=0.05,
                max_change=1.5, min_change=0.5, max_dt=max_timestep)
    CFL.add_velocity(u)

    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)

        # Fail fast if the simulation blows up
        if np.isnan(u['c']).any():
            logger.warning(f"Forward simulation blew up at t={solver.sim_time:.4f} for L={L_val:.4f}. Rejecting step.")
            return 1e9
        
        # Accumulate the time-averaged objective function safely ONLY after settling
        if solver.sim_time > start_averaging_time:
            local_J_array = J_integrand.evaluate()['g']
            local_J = local_J_array.flatten()[0] if local_J_array.size > 0 else 0.0
            
            J_val += local_J * timestep
            integrated_time += timestep

    if np.isnan(J_val) or np.isinf(J_val):
        logger.warning(f"Forward simulation diverged to NaN for L={L_val:.4f}. Rejecting step.")
        return 1e9

    # Safely compute the actual time average
    time_averaged_J = J_val / integrated_time if integrated_time > 0 else 0.0

    return time_averaged_J

if __name__ == "__main__":

    L_range = np.linspace(1.0, 4.0, 15)

    J_values = []
    for L_val in L_range:
        J_val = forward_problem([L_val])
        J_values.append(J_val)
    
    # --- Curve of Best Fit Calculation ---
    coefficients = np.polyfit(L_range, J_values, 10)
    polynomial = np.poly1d(coefficients)
    
    # Generate a high-resolution array of L values for a smooth curve
    L_smooth = np.linspace(min(L_range), max(L_range), 200)
    J_smooth = polynomial(L_smooth)
    # -------------------------------------

    # Plotting the objective function J against L
    plt.figure(figsize=(8, 5))
    
    # Plot the raw data points as scatter dots
    plt.plot(L_range, J_values, marker='o', linestyle='', color='b', label='Simulated Data')
    
    # Plot the smooth curve of best fit
    plt.plot(L_smooth, J_smooth, linestyle='-', color='r', label='Cubic Fit')
    
    plt.title('Objective Function (J) vs Domain Width (L)')
    plt.xlabel('Domain Width (L)')
    plt.ylabel('Time-Averaged Nusselt Number (J)')
    plt.legend()
    plt.grid(True)
    plt.savefig('objective_function_vs_width.png', dpi=300, bbox_inches='tight')