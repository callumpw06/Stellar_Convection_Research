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
Nx, Nz = 256, 64
Lz = 1
Rayleigh = 2e6
Prandtl = 1
nu = (Rayleigh / Prandtl)**(-1/2)
stop_sim_time = 5e-2
max_timestep = 1e-5
dtype = np.float64
timestepper = d3.RK222

coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
ex, ez = coords.unit_vector_fields(dist)

def evaluate_objective_and_gradient(params):
    # Isolate L as the only optimization parameter
    L_val = params[0]
    
    if np.isnan(L_val) or np.isinf(L_val) or L_val <= 0:
        logger.warning(f"Optimizer attempted invalid L_val: {L_val}. Rejecting step.")
        return 1e9, np.zeros_like(params)

    logger.info(f"--- Starting Forward Pass for L = {L_val:.4f} ---")

    # ---------------- Setup Native Fields & Bases ----------------
    # Rebuilding bases dynamically allows native Dedalus operators to work on [0, L]
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
    # Generate reproducible local random noise for each MPI rank
    local_shape = x.shape
    local_noise = np.random.RandomState(seed=42 + dist.comm.rank).uniform(-0.01, 0.01, size=local_shape)
    
    # Apply noise in x, but restrict to boundaries in z
    perturbation = local_noise * np.sin(np.pi * z)
    
    T['g'] = 1 - z + perturbation
    u['g'] = 0

    if i == max_iterations - 1:
        # Analysis
        analysis = solver.evaluator.add_file_handler('analysis', sim_dt=10*max_timestep, max_writes=50)
        analysis.add_task(T, name='temperature')
        analysis.add_task(-d3.div(d3.skew(u)), name='vorticity')
        analysis.add_task(np.sqrt(u@u), name='velocity_magnitude')
        analysis.add_task(1 + d3.integ(u@ez * T)/(L_val*Lz), name='Nu') # Nusselt number
        analysis.add_task(p, name='pressure')
        analysis.add_task(-d3.integ(d3.grad(T)@ez, 'x')(z=Lz) / L_val, name='heat_flux_top')

        # CFL
        CFL = d3.CFL(solver, initial_dt=1e-7, cadence=10, safety=0.5, threshold=0.05,
                    max_change=1.5, min_change=0.5, max_dt=max_timestep)
        CFL.add_velocity(u)

        # Flow properties
        flow = d3.GlobalFlowProperty(solver, cadence=10)
        flow.add_property(np.sqrt(u@u)/nu, name='Re')

    saved_states_u, saved_states_T, saved_states_p = [], [], []
    saved_dts = []
    
    # Integrand for objective function J (divided by L_val to map dx to d(hat{x}))
    J_integrand = d3.integ(1 + (u@ez) * T) / L_val
    J_val = 0.0

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
            return 1e9, np.zeros_like(params)
        
        # Save states for the adjoint pass
        saved_states_u.append(u['c'].copy())
        saved_states_T.append(T['c'].copy())
        saved_states_p.append(p['c'].copy())
        saved_dts.append(timestep)
        
        # Accumulate the time-averaged objective function safely
        local_J_array = J_integrand.evaluate()['g']
        # .flatten() safely extracts the scalar value regardless of array shape
        local_J = local_J_array.flatten()[0] if local_J_array.size > 0 else 0.0
        
        J_val += local_J * timestep

    if np.isnan(J_val) or np.isinf(J_val):
        logger.warning(f"Forward simulation diverged to NaN for L={L_val:.4f}. Rejecting step.")
        return 1e9, np.zeros_like(params)

    # ---------------- Adjoint Pass ----------------
    logger.info("--- Starting Adjoint Pass ---")
    
    p_adj = dist.Field(name='p_adj', bases=(xbasis, zbasis))
    T_adj = dist.Field(name='T_adj', bases=(xbasis, zbasis))
    u_adj = dist.VectorField(coords, name='u_adj', bases=(xbasis, zbasis))
    
    tau_p_adj = dist.Field(name='tau_p_adj')
    tau_T1_adj = dist.Field(name='tau_T1_adj', bases=xbasis)
    tau_T2_adj = dist.Field(name='tau_T2_adj', bases=xbasis)
    tau_u1_adj = dist.VectorField(coords, name='tau_u1_adj', bases=xbasis)
    tau_u2_adj = dist.VectorField(coords, name='tau_u2_adj', bases=xbasis)

    grad_u_adj = d3.grad(u_adj) + ez*lift(tau_u1_adj)
    grad_T_adj = d3.grad(T_adj) + ez*lift(tau_T1_adj)

    u_fwd = dist.VectorField(coords, name='u_fwd', bases=(xbasis, zbasis))
    T_fwd = dist.Field(name='T_fwd', bases=(xbasis, zbasis))
    p_fwd = dist.Field(name='p_fwd', bases=(xbasis, zbasis))
    
    adj_namespace = {**globals(), **locals()}
    adjoint_problem = d3.IVP([p_adj, T_adj, u_adj, tau_p_adj, tau_T1_adj, tau_T2_adj, tau_u1_adj, tau_u2_adj], namespace=adj_namespace)

    adjoint_problem.add_equation("trace(grad_u_adj) + tau_p_adj = 0")
    adjoint_problem.add_equation("dt(T_adj) - div(grad_T_adj) + lift(tau_T2_adj) = u_fwd@grad(T_adj) + Prandtl*Rayleigh*(u_adj@ez) + (u_fwd@ez)")
    adjoint_problem.add_equation("dt(u_adj) - Prandtl*div(grad_u_adj) + grad(p_adj) + lift(tau_u2_adj) = u_fwd@grad(u_adj) - grad(u_fwd)@u_adj - T_adj*grad(T_fwd) + T_fwd*ez")
    
    adjoint_problem.add_equation("T_adj(z=0) = 0")
    adjoint_problem.add_equation("T_adj(z=1) = 0")
    adjoint_problem.add_equation("u_adj(z=0) = 0")
    adjoint_problem.add_equation("u_adj(z=1) = 0")
    adjoint_problem.add_equation("integ(p_adj) = 0")

    adjoint_solver = adjoint_problem.build_solver(timestepper)
    
    # ---------------- Gradient Calculation ----------------
    # Dedalus native derivatives (w.r.t x). 
    # Mapped directly to d(hat{x}) through analytical simplification.
    dx_u = d3.Differentiate(u_fwd, coords['x'])
    dxx_u = d3.Differentiate(dx_u, coords['x'])
    dx_p = d3.Differentiate(p_fwd, coords['x'])
    dx_T = d3.Differentiate(T_fwd, coords['x'])
    dxx_T = d3.Differentiate(dx_T, coords['x'])

    ux_fwd = u_fwd@ex

    term1 = p_adj * (dx_u @ ex)
    term2 = u_adj @ ( ux_fwd * dx_u + dx_p * ex - 2 * Prandtl * dxx_u )
    term3 = T_adj * ( ux_fwd * dx_T - 2 * dxx_T )

    # Divide by L_val to account for spatial dx vs d(hat{x})
    grad_L_integrand = d3.integ(term1 + term2 + term3) / L_val

    grad_L_val = 0.0
    while len(saved_states_u) > 0:
        u_fwd['c'] = saved_states_u.pop()
        T_fwd['c'] = saved_states_T.pop()
        p_fwd['c'] = saved_states_p.pop()
        backward_timestep = saved_dts.pop()
        
        adjoint_solver.step(backward_timestep)
        
        local_grad_array = grad_L_integrand.evaluate()['g']
        local_grad = local_grad_array.flatten()[0] if local_grad_array.size > 0 else 0.0
        
        grad_L_val += local_grad * backward_timestep

    # SciPy minimize looks for minimum, but we want to maximize Nusselt number. 
    # Therefore, return negative J and negative Gradient.
    objective_to_minimize = -J_val
    gradient_to_return = np.array([grad_L_val])
    
    gradient_to_return = np.nan_to_num(gradient_to_return, nan=0.0, posinf=1e5, neginf=-1e5)

    logger.info(f"Iteration finished! Current L: {L_val:.4f}, J: {J_val:.4f}, Grad: {gradient_to_return[0]:.4e}")
    return objective_to_minimize, gradient_to_return

# List to track L over iterations (initialized with the starting guess)
L_history = []

def tracking_callback(xk):
    """
    This function is called by the optimizer after every successful iteration.
    xk is the current parameter array.
    """
    current_L = xk[0]
    L_history.append(current_L)
    logger.info(f"--- Iteration Complete. Current L appended to history: {current_L:.4f} ---")

if __name__ == "__main__":
    # ---------------- Custom Gradient Ascent Setup ----------------
    L_current = 2.0        # Initial guess
    alpha = 0.01            # Learning rate (You may need to tune this higher or lower)
    max_iterations = 10    # Number of optimization steps
    
    L_history = [L_current]
    J_history = []
    
    logger.info("Starting explicit Gradient Ascent Loop...")
    
    for i in range(max_iterations):
        logger.info(f"========== Iteration {i+1}/{max_iterations} ==========")
        
        # 1. Evaluate the system
        J_val, grad_array = evaluate_objective_and_gradient([L_current])
        J_history.append(J_val)
        
        dJ_dL = grad_array[0]
        
        # 2. Apply the paper's update rule: L_{n+1} = L_n + alpha * dJ/dL
        L_new = L_current + (alpha * dJ_dL)
        
        # 3. Enforce physical boundaries so L doesn't go negative or infinitely huge
        L_new = np.clip(L_new, 0.5, 10.0)
        
        logger.info(f"Update: L changed from {L_current:.4f} to {L_new:.4f} (Step size: {L_new - L_current:.4e})")
        
        # 4. Step forward
        L_current = L_new
        L_history.append(L_current)
        
    logger.info("Optimization Loop Complete.")
    
    # ---------------- Plotting ----------------
    if dist.comm.rank == 0:  # Only the root process should handle plotting
        logger.info("Generating optimization history plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Convert the negative integrated values to positive time-averaged Nusselt numbers
        time_averaged_Nu_history = [-j / stop_sim_time for j in J_history]
        
        # Plot L history
        ax1.plot(range(len(L_history)), L_history, marker='o', color='b', linewidth=2)
        ax1.set_title('Domain Width (L) over Iterations')
        ax1.set_xlabel('Iteration Number')
        ax1.set_ylabel('L')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot J history
        ax2.plot(range(len(time_averaged_Nu_history)), time_averaged_Nu_history, marker='s', color='r', linewidth=2)
        ax2.set_title('Objective Function (J) over Iterations')
        ax2.set_xlabel('Iteration Number')
        ax2.set_ylabel('Nusselt Number (J)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plot_filename = 'gradient_ascent_progress.png'
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        
        logger.info(f"Plot saved successfully as '{plot_filename}'")