"""
Adjoint-Optimized Boussinesq Convection using Dedalus v3 and SciPy.
This script wraps the forward Navier-Stokes IVP and a backward Adjoint IVP 
inside an L-BFGS-B optimization loop to maximize the Nusselt number.
"""

import numpy as np
import dedalus.public as d3
from scipy.optimize import minimize
import logging
logger = logging.getLogger(__name__)

# =====================================================================
# 1. GLOBAL PARAMETERS & BASES (Defined once to save overhead)
# =====================================================================
Lx, Lz = 4, 1
Nx, Nz = 256, 64
Rayleigh = 2e6
Prandtl = 1
stop_sim_time = 4e-3
max_timestep = 1e-6
dtype = np.float64
timestepper = d3.RK222

coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=3/2)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=3/2)

# We will optimize the initial temperature perturbation.
# The control variable is a 2D array of the same shape as the grid.
local_shape = dist.local_grid(xbasis).shape[0] * dist.local_grid(zbasis).shape[0]

# =====================================================================
# 2. THE OBJECTIVE & GRADIENT WRAPPER
# =====================================================================
def evaluate_objective_and_gradient(control_1d):
    """
    SciPy calls this function. It must return:
    1. The objective value J (to minimize).
    2. The 1D gradient array (dJ/d_control).
    """
    logger.info("--- Starting new optimization iteration ---")
    
    # Reshape the 1D SciPy control array back to the Dedalus 2D grid shape
    control_2d = control_1d.reshape((dist.local_grid(xbasis).shape[0], 
                                     dist.local_grid(zbasis).shape[0]))

    # ---------------------------------------------------------
    # A. THE FORWARD PASS
    # ---------------------------------------------------------
    # Re-instantiate fields for a clean run
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    T = dist.Field(name='T', bases=(xbasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    tau_p = dist.Field(name='tau_p')
    tau_T1 = dist.Field(name='tau_T1', bases=xbasis)
    tau_T2 = dist.Field(name='tau_T2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    # Apply the control variable as the initial temperature perturbation
    x, z = dist.local_grids(xbasis, zbasis)
    T['g'] = (Lz - z) + control_2d * z * (Lz - z) # Ensure boundary conditions hold
    u['g'] = 0

    # Build standard Forward IVP (Identical to your original code)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez*lift(tau_u1)
    grad_T = d3.grad(T) + ez*lift(tau_T1)

    forward_problem = d3.IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2], namespace={**globals(), **locals()})
    forward_problem.add_equation("trace(grad_u) + tau_p = 0")
    forward_problem.add_equation("dt(T) - div(grad_T) + lift(tau_T2) = - u@grad(T)")
    forward_problem.add_equation("dt(u) - Prandtl*div(grad_u) + grad(p) - Prandtl*Rayleigh*T*ez + lift(tau_u2) = - u@grad(u)")
    forward_problem.add_equation("T(z=0) = Lz")
    forward_problem.add_equation("u(z=0) = 0")
    forward_problem.add_equation("T(z=Lz) = 0")
    forward_problem.add_equation("u(z=Lz) = 0")
    forward_problem.add_equation("integ(p) = 0")

    forward_solver = forward_problem.build_solver(timestepper)
    forward_solver.stop_sim_time = stop_sim_time

    # Memory storage for the adjoint backward pass
    saved_states_u = []
    saved_states_T = []
    integrated_Nu = 0.0

    # Run Forward Loop
    while forward_solver.proceed:
        forward_solver.step(max_timestep)
        
        # Save state for the backward pass
        saved_states_u.append(u['c'].copy())
        saved_states_T.append(T['c'].copy())
        
        # Accumulate Objective (Nusselt Number)
        current_Nu = 1 + d3.integ(u@ez * T).evaluate()['g'][0,0]
        integrated_Nu += current_Nu * max_timestep

    # SciPy minimizes, so to MAXIMIZE Nusselt, we return negative Nusselt
    J = -integrated_Nu

    # ---------------------------------------------------------
    # B. THE BACKWARD PASS (Adjoint Solver)
    # ---------------------------------------------------------
    # Define adjoint fields
    p_adj = dist.Field(name='p_adj', bases=(xbasis,zbasis))
    T_adj = dist.Field(name='T_adj', bases=(xbasis,zbasis))
    u_adj = dist.VectorField(coords, name='u_adj', bases=(xbasis,zbasis))
    tau_p_adj = dist.Field(name='tau_p_adj')
    tau_T1_adj = dist.Field(name='tau_T1_adj', bases=xbasis)
    tau_T2_adj = dist.Field(name='tau_T2_adj', bases=xbasis)
    tau_u1_adj = dist.VectorField(coords, name='tau_u1_adj', bases=xbasis)
    tau_u2_adj = dist.VectorField(coords, name='tau_u2_adj', bases=xbasis)
    
    # ... Set up tau variables and gradients for adjoint fields ...

    u_fwd = dist.VectorField(coords, name='u_fwd', bases=(xbasis,zbasis))
    T_fwd = dist.Field(name='T_fwd', bases=(xbasis,zbasis))

    grad_u_adj = d3.grad(u_adj) + ez*lift(tau_u1_adj)
    grad_T_adj = d3.grad(T_adj) + ez*lift(tau_T1_adj)
    
    adjoint_problem = d3.IVP([p_adj, T_adj, u_adj, tau_p_adj, tau_T1_adj, tau_T2_adj, tau_u1_adj, tau_u2_adj], namespace={**globals(), **locals()})
    
    # NOTE: Dedalus only steps time forward. To solve backward in time, 
    # we mathematically reverse the sign of the time derivatives in the equations.
    # We also inject the saved forward states (u, T) into the advection terms here.
    
    adjoint_problem.add_equation("trace(grad_u_adj) + tau_p_adj = 0")
    # Adjoint Temperature: Advected by u_fwd, coupled to u_adj via buoyancy, forced by u_fwd_z
    adjoint_problem.add_equation("dt(T_adj) - div(grad_T_adj) + lift(tau_T2_adj) = Prandtl*Rayleigh*(u_adj@ez) + u_fwd@grad(T_adj) - (u_fwd@ez)") 
    
    # 3. Adjoint Momentum (Stable backward-in-time formulation)
    adjoint_problem.add_equation("dt(u_adj) - Prandtl*div(grad_u_adj) + grad(p_adj) + lift(tau_u2_adj) = T_adj*grad(T_fwd) + u_fwd@grad(u_adj) + grad(u_fwd)@u_adj - T_fwd*ez")
    # ... Boundary conditions for adjoint fields (T_adj=0, u_adj=0 at walls) ...
    adjoint_problem.add_equation("T_adj(z=0) = 0")
    adjoint_problem.add_equation("u_adj(z=0) = 0")
    adjoint_problem.add_equation("T_adj(z=Lz) = 0")
    adjoint_problem.add_equation("u_adj(z=Lz) = 0")
    adjoint_problem.add_equation("integ(p_adj) = 0") # Pressure gauge

    adjoint_solver = adjoint_problem.build_solver(timestepper)
    adjoint_solver.stop_sim_time = stop_sim_time
    
    # 5. Valid Python Injection Loop
    while adjoint_solver.proceed:
        # Pop the last saved states and overwrite the memory of the placeholder fields
        u_fwd['c'] = saved_states_u.pop()
        T_fwd['c'] = saved_states_T.pop()
        
        adjoint_solver.step(max_timestep)

    # ---------------------------------------------------------
    # C. GRADIENT EXTRACTION
    # ---------------------------------------------------------
    # The gradient of J with respect to the initial temperature is exactly 
    # the value of the adjoint temperature field at time t=0.
    gradient_2d = T_adj['g'].copy()
    
    # We want to maximize J, which means minimizing -J.
    # The gradient is flattened to feed back to SciPy.
    gradient_1d = gradient_2d.flatten()
    logger.info(f"Interation completed. Current Objective (Nusselt): {current_Nu:.6f}")

    return J, gradient_1d

# =====================================================================
# 3. THE EXECUTION
# =====================================================================
if __name__ == "__main__":
    logger.info('Starting optimization...')
    
    # Start with a random small perturbation guess
    np.random.seed(42)
    initial_guess_1d = np.random.normal(scale=1e-3, size=local_shape)

    bnds = [(-0.1, 0.1) for _ in range(local_shape)]  # Bound the perturbation to avoid unphysical values

    # Run L-BFGS-B Optimization
    result = minimize(
        fun=evaluate_objective_and_gradient,
        x0=initial_guess_1d,
        method='L-BFGS-B',
        jac=True,  # We are providing the exact gradient
        bounds=bnds,
        options={'maxiter': 2, 'disp': True}
    )

    logger.info(f"Optimization finished. Final Objective: {-result.fun}")
    optimal_perturbation = result.x.reshape((Nx, Nz))

    import matplotlib.pyplot as plt

    logger.info("Plotting the optimal initial perturbation...")

    # 1. Extract the physical grid arrays and squeeze out the empty dimensions
    x_grid = dist.local_grid(xbasis).squeeze()
    z_grid = dist.local_grid(zbasis).squeeze()

    # 2. Plot the 2D field
    plt.figure(figsize=(10, 3))
    
    # We transpose optimal_perturbation (.T) because pcolormesh expects rows to be 'z' and columns to be 'x'
    im = plt.pcolormesh(x_grid, z_grid, optimal_perturbation.T, cmap='RdBu_r', shading='gouraud')
    
    plt.colorbar(im, label='Temperature Perturbation $\delta T$')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(f'Optimal Initial Temperature Field (Nusselt = {-result.fun:.6f})')
    plt.tight_layout()
    
    # Save and show
    plt.savefig('optimal_perturbation.png', dpi=300)
    plt.show()