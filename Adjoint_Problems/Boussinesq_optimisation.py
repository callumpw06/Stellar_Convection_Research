import numpy as np
import dedalus.public as d3
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

# Parameters
Nx, Nz = 256, 64
Rayleigh = 2e6
Prandtl = 1
stop_sim_time = 4e-2
max_timestep = 1e-5
dtype = np.float64
timestepper = d3.RK222

# Setup domain globally
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)

# Grid parameters for reshaping safely (defaults to scale=1 natively in D3)
temp_xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, 1), dealias=3/2)
temp_zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, 1), dealias=3/2)
temp_x, temp_z = dist.local_grids(temp_xbasis, temp_zbasis)
N_grid_x, N_grid_z = temp_x.shape[0], temp_z.shape[1]
total_grid_points = N_grid_x * N_grid_z

def evaluate_objective_and_gradient(params):
    L_val = params[-1]
    
    if np.isnan(L_val) or np.isinf(L_val) or L_val <= 0:
        logger.warning(f"Optimizer attempted invalid L_val: {L_val}. Rejecting step.")
        return 1e9, np.zeros_like(params)

    # ---------------- Setup Native Fields & Bases ----------------
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, L_val), dealias=3/2)
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, 1), dealias=3/2)
    
    # Defaults to standard un-dealiased scale
    x, z = dist.local_grids(xbasis, zbasis)

    p = dist.Field(name='p', bases=(xbasis, zbasis))
    T = dist.Field(name='T', bases=(xbasis, zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
    
    tau_p = dist.Field(name='tau_p')
    tau_T1 = dist.Field(name='tau_T1', bases=xbasis)
    tau_T2 = dist.Field(name='tau_T2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    ex, ez = coords.unit_vector_fields(dist)
    
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
    
    # Force scale=1 before assignment to prevent broadcast errors
    T.change_scales(1)
    u.change_scales(1)
    
    control_data = params[:-1].reshape((N_grid_x, N_grid_z))
    T['g'] = (1 - z) + control_data * z * (1 - z)
    u['g'] = 0

    saved_states_u, saved_states_T, saved_states_p = [], [], []
    
    while solver.proceed:
        solver.step(max_timestep)
        saved_states_u.append(u['c'].copy())
        saved_states_T.append(T['c'].copy())
        saved_states_p.append(p['c'].copy())

    J = - (1 + d3.integ((u@ez) * T).evaluate()['g'][0,0]) * stop_sim_time
    
    if np.isnan(J) or np.isinf(J):
        logger.warning(f"Forward simulation diverged to NaN for L={L_val:.4f}. Rejecting step.")
        return 1e9, np.zeros_like(params)

    # ---------------- Adjoint Pass ----------------
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
    
    dx_u = d3.Differentiate(u_fwd, coords['x'])
    dxx_u = d3.Differentiate(dx_u, coords['x'])
    dx_p = d3.Differentiate(p_fwd, coords['x'])
    dx_T = d3.Differentiate(T_fwd, coords['x'])
    dxx_T = d3.Differentiate(dx_T, coords['x'])
    
    ux_fwd = u_fwd@ex
    term1 = (p_adj / L_val) * (dx_u@ex)
    term2 = u_adj @ ( (ux_fwd / L_val) * dx_u + (1 / L_val) * dx_p*ex - (2 * Prandtl / L_val) * dxx_u )
    term3 = T_adj * ( (ux_fwd / L_val) * dx_T - (2 / L_val) * dxx_T )
    
    grad_L_integrand = d3.integ(term1 + term2 + term3)

    grad_L_val = 0.0
    while len(saved_states_u) > 0:
        u_fwd['c'] = saved_states_u.pop()
        T_fwd['c'] = saved_states_T.pop()
        p_fwd['c'] = saved_states_p.pop()
        
        adjoint_solver.step(max_timestep)
        
        spatial_integral = grad_L_integrand.evaluate()['g'][0, 0]
        grad_L_val += spatial_integral * max_timestep

    # ---------------- CRITICAL FIX: Downscale Adjoint field before mapping ----------------
    T_adj.change_scales(1)
    
    spatial_chain_rule = z * (1 - z)
    control_gradient = (T_adj['g'] * spatial_chain_rule).flatten()
    
    final_grad = np.append(control_gradient, grad_L_val)
    final_grad = np.nan_to_num(final_grad, nan=0.0, posinf=1e5, neginf=-1e5)

    logger.info(f"Iteration successful! Current L: {L_val:.4f}, J: {J:.4f}, Max Grad Magnitude: {np.max(np.abs(final_grad)):.4e}")
    return J, final_grad

if __name__ == "__main__":
    initial_guess = np.concatenate([np.zeros(total_grid_points), [4.0]])
    bounds = [(None, None)] * total_grid_points + [(0.5, 10.0)]
    
    logger.info("Starting L-BFGS-B Optimization Loop...")
    result = minimize(evaluate_objective_and_gradient, x0=initial_guess, 
                      method='L-BFGS-B', jac=True, bounds=bounds,
                      options={'maxiter': 50, 'ftol': 1e-9})
                      
    logger.info(f"Optimization finished.")
    logger.info(f"Success: {result.success}")
    logger.info(f"Reason for stopping: {result.message}")
    logger.info(f"Final L: {result.x[-1]}")