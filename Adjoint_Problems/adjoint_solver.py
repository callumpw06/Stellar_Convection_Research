import numpy as np
import dedalus.public as d3
import logging
import sys
from mpi4py import MPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NEW: Read boundary condition from bash
BC_TYPE = str(sys.argv[1]) if len(sys.argv) > 1 else "no-slip"

logger.info("Loading steady-state solution...")
try:
    data = np.load('steady_state_solution.npz')
except FileNotFoundError:
    logger.error("Could not find 'steady_state_solution.npz'.")
    exit()

u_coeffs = data['u_coeffs']
T_coeffs = data['T_coeffs']
p_coeffs = data['p_coeffs']
L_val = float(data['L_val'])
J_val = float(data['J_val'])  # Added so we can save Nu to the CSV later
Rayleigh = float(data['Rayleigh'])
Prandtl = float(data['Prandtl'])
Nx = int(data['Nx'])
Nz = int(data['Nz'])

dtype = np.float64
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
ex, ez = coords.unit_vector_fields(dist)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, L_val), dealias=3/2)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, 1), dealias=3/2)

u_bar = dist.VectorField(coords, name='u_bar', bases=(xbasis, zbasis))
T_bar = dist.Field(name='T_bar', bases=(xbasis, zbasis))
p_bar = dist.Field(name='p_bar', bases=(xbasis, zbasis))

u_bar['c'] = u_coeffs
T_bar['c'] = T_coeffs
p_bar['c'] = p_coeffs

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

ux_bar = u_bar @ ex
uz_bar = u_bar @ ez
lx_adj = lambda_adj @ ex
lz_adj = lambda_adj @ ez
transpose_term = lx_adj * d3.grad(ux_bar) + lz_adj * d3.grad(uz_bar)

namespace = {**globals(), **locals()}
adjoint_problem = d3.LBVP([lambda_adj, pi_adj, theta_adj, tau_pi, tau_lambda1, tau_lambda2, tau_theta1, tau_theta2], namespace=namespace)

adjoint_problem.add_equation("trace(grad_lambda) + tau_pi = 0")
adjoint_problem.add_equation("- (u_bar @ grad_lambda) + transpose_term - d3.grad(pi_adj) - Prandtl * div(grad_lambda) + theta_adj * d3.grad(T_bar) + lift(tau_lambda2) = T_bar * ez")
adjoint_problem.add_equation("- (u_bar @ grad_theta) - div(grad_theta) - Prandtl * Rayleigh * lz_adj + lift(tau_theta2) = uz_bar")

adjoint_problem.add_equation("theta_adj(z=0) = 0")
adjoint_problem.add_equation("theta_adj(z=1) = 0")
adjoint_problem.add_equation("integ(pi_adj) = 0") 

# --- DYNAMIC ADJOINT BOUNDARY CONDITIONS ---
if BC_TYPE == "no-slip":
    adjoint_problem.add_equation("lambda_adj(z=0) = 0")
    adjoint_problem.add_equation("lambda_adj(z=1) = 0")
elif BC_TYPE == "free-slip":
    # Adjoint impermeability
    adjoint_problem.add_equation("lambda_adj@ez(z=0) = 0")
    adjoint_problem.add_equation("lambda_adj@ez(z=1) = 0")
    # Adjoint zero shear stress
    adjoint_problem.add_equation("ex @ (grad_lambda @ ez)(z=0) = 0")
    adjoint_problem.add_equation("ex @ (grad_lambda @ ez)(z=1) = 0")
else:
    raise ValueError(f"Unknown boundary condition specified: {BC_TYPE}")
# -------------------------------------------

logger.info("Solving steady adjoint LBVP matrices...")
solver = adjoint_problem.build_solver()
solver.solve()
logger.info("Adjoint problem solved successfully!")

logger.info("Evaluating objective gradient dJ/dL...")
dx = lambda A: d3.Differentiate(A, coords['x'])

dx_u = dx(u_bar)
dxx_u = dx(dx_u)
dx_p = dx(p_bar)
dx_T = dx(T_bar)
dxx_T = dx(dx_T)

term1 = pi_adj * (dx_u @ ex)
term2 = lambda_adj @ (ux_bar * dx_u + dx_p * ex - 2 * Prandtl * dxx_u)
term3 = theta_adj * (ux_bar * dx_T - 2 * dxx_T)

grad_integrand = (term1 + term2 + term3) / (L_val**2)
dJ_dL_array = d3.integ(grad_integrand).evaluate()['g']
dJ_dL = dJ_dL_array.flatten()[0] if dJ_dL_array.size > 0 else 0.0

logger.info(f"Computed Objective Gradient wrt Domain Width (dJ/dL): {dJ_dL:.6e}")

if dist.comm.rank == 0:
    # Save raw gradient for the bash script to compute L_next
    np.savetxt('gradient.txt', [dJ_dL])
    # Automatically log the iteration history so the bash script doesn't have to
    with open('optimization_history.csv', 'a') as f:
        f.write(f"{L_val},{J_val},{dJ_dL}\n")
    logger.info("Gradient and history saved.")