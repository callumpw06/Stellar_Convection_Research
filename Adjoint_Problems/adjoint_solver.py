import numpy as np
import dedalus.public as d3
import logging
from mpi4py import MPI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Load Saved Steady State ----------------
logger.info("Loading steady-state solution...")
try:
    data = np.load('steady_state_solution.npz')
except FileNotFoundError:
    logger.error("Could not find 'steady_state_solution.npz'. Please ensure the forward solver saved it.")
    exit()

u_coeffs = data['u_coeffs']
T_coeffs = data['T_coeffs']
p_coeffs = data['p_coeffs']
L_val = float(data['L_val'])
Rayleigh = float(data['Rayleigh'])
Prandtl = float(data['Prandtl'])
Nx = int(data['Nx'])
Nz = int(data['Nz'])

dtype = np.float64

coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
ex, ez = coords.unit_vector_fields(dist)

# ---------------- Setup Bases ----------------
# By setting bounds=(0, L_val), d3.grad() automatically applies the correct physical scaling
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, L_val), dealias=3/2)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, 1), dealias=3/2)

# ---------------- Background Steady Fields ----------------
u_bar = dist.VectorField(coords, name='u_bar', bases=(xbasis, zbasis))
T_bar = dist.Field(name='T_bar', bases=(xbasis, zbasis))
p_bar = dist.Field(name='p_bar', bases=(xbasis, zbasis))

# Load coefficients from the forward pass
u_bar['c'] = u_coeffs
T_bar['c'] = T_coeffs
p_bar['c'] = p_coeffs

# ---------------- Adjoint Fields & Taus ----------------
lambda_adj = dist.VectorField(coords, name='lambda_adj', bases=(xbasis, zbasis))
pi_adj = dist.Field(name='pi_adj', bases=(xbasis, zbasis))
theta_adj = dist.Field(name='theta_adj', bases=(xbasis, zbasis))

tau_pi = dist.Field(name='tau_pi')
tau_lambda1 = dist.VectorField(coords, name='tau_lambda1', bases=xbasis)
tau_lambda2 = dist.VectorField(coords, name='tau_lambda2', bases=xbasis)
tau_theta1 = dist.Field(name='tau_theta1', bases=xbasis)
tau_theta2 = dist.Field(name='tau_theta2', bases=xbasis)

# ---------------- First-Order Reductions ----------------
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

# Correctly binding Tau polynomials to the highest derivatives
grad_lambda = d3.grad(lambda_adj) + ez * lift(tau_lambda1)
grad_theta = d3.grad(theta_adj) + ez * lift(tau_theta1)

# ---------------- Mathematical Helper Terms ----------------
ux_bar = u_bar @ ex
uz_bar = u_bar @ ez
lx_adj = lambda_adj @ ex
lz_adj = lambda_adj @ ez

# Computing (grad_u_bar)^T * lambda_adj safely without breaking tensor ranks
transpose_term = lx_adj * d3.grad(ux_bar) + lz_adj * d3.grad(uz_bar)

# ---------------- Adjoint LBVP Problem Setup ----------------
namespace = {**globals(), **locals()}
adjoint_problem = d3.LBVP([lambda_adj, pi_adj, theta_adj, tau_pi, tau_lambda1, tau_lambda2, tau_theta1, tau_theta2], namespace=namespace)

# Adjoint Continuity[cite: 2]
adjoint_problem.add_equation("trace(grad_lambda) + tau_pi = 0")

# Adjoint Momentum[cite: 2]
adjoint_problem.add_equation("- (u_bar @ grad_lambda) + transpose_term - d3.grad(pi_adj) - Prandtl * div(grad_lambda) + theta_adj * d3.grad(T_bar) + lift(tau_lambda2) = T_bar * ez")

# Adjoint Thermal[cite: 2]
adjoint_problem.add_equation("- (u_bar @ grad_theta) - div(grad_theta) - Prandtl * Rayleigh * lz_adj + lift(tau_theta2) = uz_bar")

# Boundary Conditions: Homogeneous Dirichlet boundaries for adjoint states[cite: 2]
adjoint_problem.add_equation("lambda_adj(z=0) = 0")
adjoint_problem.add_equation("lambda_adj(z=1) = 0")
adjoint_problem.add_equation("theta_adj(z=0) = 0")
adjoint_problem.add_equation("theta_adj(z=1) = 0")
adjoint_problem.add_equation("integ(pi_adj) = 0")  # Pressure gauge

# Solve the Linear Adjoint System
logger.info("Solving steady adjoint LBVP matrices...")
solver = adjoint_problem.build_solver()
solver.solve()
logger.info("Adjoint problem solved successfully!")

# ---------------- Compute Domain Width Gradient (dJ/dL) ----------------
logger.info("Evaluating objective gradient dJ/dL...")

# Taking physical x-derivatives
dx = lambda A: d3.Differentiate(A, coords['x'])

dx_u = dx(u_bar)
dxx_u = dx(dx_u)
dx_p = dx(p_bar)
dx_T = dx(T_bar)
dxx_T = dx(dx_T)

# Assembling the gradient terms from the paper using physical coordinates and minus signs[cite: 2]
term1 = pi_adj * (dx_u @ ex)
term2 = lambda_adj @ (ux_bar * dx_u + dx_p * ex - 2 * Prandtl * dxx_u)
term3 = theta_adj * (ux_bar * dx_T - 2 * dxx_T)

# The physical substitution yields exactly 1/L^2 out front
grad_integrand = (term1 + term2 + term3) / (L_val**2)

# Evaluate the global integral
dJ_dL_array = d3.integ(grad_integrand).evaluate()['g']
dJ_dL = dJ_dL_array.flatten()[0] if dJ_dL_array.size > 0 else 0.0

logger.info(f"Computed Objective Gradient wrt Domain Width (dJ/dL): {dJ_dL:.6e}")

if dist.comm.rank == 0:
    np.savetxt('gradient.txt', [dJ_dL])
    logger.info("Gradient saved to 'gradient.txt'")