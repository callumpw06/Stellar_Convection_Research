import numpy as np
import dedalus.public as d3
import logging
import sys

# Set non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Plotting Configuration ---
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
})

# --- HPC Safe Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Problem Parameters ---
Lx = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
Lz = 1.0        
Nx = 128         
Nz = 64         
Pr = 1.0        
dealias = 3/2
dtype = np.float64

# --- Coordinates & Domain Bases ---
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
x, z = dist.local_grids(xbasis, zbasis)

# =====================================================================
# --- PHASE 1: FORWARD STEADY-STATE SOLVER (NLBVP) ---
# =====================================================================
logger.info("=== PHASE 1: Solving Forward Steady State ===")

p = dist.Field(name='p', bases=(xbasis, zbasis))
v = dist.VectorField(coords, name='v', bases=(xbasis, zbasis))
T = dist.Field(name='T', bases=(xbasis, zbasis))

grad_v = dist.TensorField(coords, name='grad_v', bases=(xbasis, zbasis))
grad_T = dist.VectorField(coords, name='grad_T', bases=(xbasis, zbasis))

tau_p = dist.Field(name='tau_p')
tau_v1 = dist.VectorField(coords, name='tau_v1', bases=xbasis)
tau_v2 = dist.VectorField(coords, name='tau_v2', bases=xbasis)
tau_T1 = dist.Field(name='tau_T1', bases=xbasis)
tau_T2 = dist.Field(name='tau_T2', bases=xbasis)

Ra = dist.Field(name='Ra')
epsilon = dist.Field(name='epsilon')

k_x = 2.0 * np.pi / Lx
z_norm = z / Lz
forcing_T = dist.Field(name='forcing_T', bases=(xbasis, zbasis))
forcing_T['g'] = np.cos(k_x * x) * np.sin(np.pi * z_norm) 

ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

namespace = locals()
problem = d3.NLBVP([p, v, T, grad_v, grad_T, tau_p, tau_v1, tau_v2, tau_T1, tau_T2], namespace=namespace)

problem.add_equation("trace(grad_v) + tau_p = 0")
problem.add_equation("grad_v - grad(v) - ez * lift(tau_v1) = 0")
problem.add_equation("v @ grad(v) + grad(p) - Pr * div(grad_v) - Pr * Ra * T * ez + lift(tau_v2) = 0")
problem.add_equation("grad_T - grad(T) - ez * lift(tau_T1) = 0")
problem.add_equation("v @ grad(T) - div(grad_T) - epsilon * forcing_T + lift(tau_T2) = 0")

u_z = ez @ (grad_v @ ex)
w = ez @ v

problem.add_equation("w(z=0) = 0")
problem.add_equation("u_z(z=0) = 0")
problem.add_equation("w(z=Lz) = 0")
problem.add_equation("u_z(z=Lz) = 0")
problem.add_equation("T(z=0) = 1")
problem.add_equation("T(z=Lz) = 0")
problem.add_equation("integ(p) = 0")

# Initial Seed
A_roll = 5.0  
T['g'] = 1.0 - z_norm + A_roll * np.cos(k_x * x) * np.sin(np.pi * z_norm)
v['g'][0] = -A_roll * np.sin(k_x * x) * (2 * z_norm * (1 - z_norm)**2 - 2 * (z_norm**2) * (1 - z_norm)) / Lz
v['g'][1] =  A_roll * k_x * np.cos(k_x * x) * (z_norm**2) * ((1 - z_norm)**2)
p['g'] = 0.0

solver = problem.build_solver(ncc_cutoff=1e-3)

# Target Ra = 1e5
schedule = [
    (10000.0, 0.5),   
    (10000.0, 0.0)#,   
    #(50000.0, 0.0),   
    #(100000.0, 0.0)
]

for target_Ra, target_eps in schedule:
    Ra['g'] = target_Ra
    epsilon['g'] = target_eps
    logger.info(f"-> NLBVP Continuation: Ra = {target_Ra:.1e} | Forcing = {target_eps:.1f}")
    
    for step in range(1, 15):
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        if pert_norm <= 1e-7:
            logger.info(f"   Converged perfectly in {step} steps.")
            break
    else:
        logger.error("   Solver failed to converge!")
        sys.exit(1)

# Final Nusselt Evaluation
Nu_integrand = d3.integ(1 + w * T) / (Lx * Lz)
Nu_val = Nu_integrand.evaluate()['g'].flatten()[0]
global_Nu = dist.comm.bcast(Nu_val, root=0)
if dist.comm.rank == 0:
    logger.info(f"-> FORWARD PASS COMPLETE. Current Nusselt Number: {global_Nu:.4f}")

# Extracting background fields for the adjoint solver
v_f = v.copy()
T_f = T.copy()
p_f = p.copy()
grad_v_f = grad_v.copy()
grad_T_f = grad_T.copy()
u_f = v_f @ ex
w_f = v_f @ ez


# =====================================================================
# --- PHASE 2: CONTINUOUS ADJOINT SOLVER (LBVP) ---
# =====================================================================
logger.info("=== PHASE 2: Solving Adjoint Linear System ===")

p_a = dist.Field(name='p_a', bases=(xbasis, zbasis))
v_a = dist.VectorField(coords, name='v_a', bases=(xbasis, zbasis))
T_a = dist.Field(name='T_a', bases=(xbasis, zbasis))

grad_v_a = dist.TensorField(coords, name='grad_v_a', bases=(xbasis, zbasis))
grad_T_a = dist.VectorField(coords, name='grad_T_a', bases=(xbasis, zbasis))

tau_p_a = dist.Field(name='tau_p_a')
tau_v1_a = dist.VectorField(coords, name='tau_v1_a', bases=xbasis)
tau_v2_a = dist.VectorField(coords, name='tau_v2_a', bases=xbasis)
tau_T1_a = dist.Field(name='tau_T1_a', bases=xbasis)
tau_T2_a = dist.Field(name='tau_T2_a', bases=xbasis)

u_a = v_a @ ex
w_a = v_a @ ez
u_z_a = ez @ (grad_v_a @ ex)

# Adjoint Momentum Transpose component math
transposed_advection = ex*(u_a*grad_v_f[0,0] + w_a*grad_v_f[0,1]) + ez*(u_a*grad_v_f[1,0] + w_a*grad_v_f[1,1])

namespace_adj = locals()
problem_adj = d3.LBVP([p_a, v_a, T_a, grad_v_a, grad_T_a, tau_p_a, tau_v1_a, tau_v2_a, tau_T1_a, tau_T2_a], namespace=namespace_adj)

# Adjoint Continuity and gradient definitions
problem_adj.add_equation("trace(grad_v_a) + tau_p_a = 0")
problem_adj.add_equation("grad_v_a - grad(v_a) - ez * lift(tau_v1_a) = 0")
problem_adj.add_equation("grad_T_a - grad(T_a) - ez * lift(tau_T1_a) = 0")

# Adjoint Momentum Equation (derived from Lagrangian Fréchet derivative)
problem_adj.add_equation("-Pr*div(grad_v_a) - grad(p_a) - u_f*dx(v_a) - w_f*dz(v_a) + transposed_advection + lift(tau_v2_a) = T_f * ez")

# Adjoint Energy Equation
problem_adj.add_equation("-div(grad_T_a) - u_f*dx(T_a) - w_f*dz(T_a) - Pr*Ra*w_a + lift(tau_T2_a) = w_f")

# Adjoint Boundary Conditions
problem_adj.add_equation("w_a(z=0) = 0")
problem_adj.add_equation("u_z_a(z=0) = 0")
problem_adj.add_equation("w_a(z=Lz) = 0")
problem_adj.add_equation("u_z_a(z=Lz) = 0")
problem_adj.add_equation("T_a(z=0) = 0")
problem_adj.add_equation("T_a(z=Lz) = 0")
problem_adj.add_equation("integ(p_a) = 0")

# Solve the Linear Adjoint Problem
solver_adj = problem_adj.build_solver(ncc_cutoff=1e-3)
solver_adj.solve()
logger.info("-> ADJOINT PASS COMPLETE.")


# =====================================================================
# --- PHASE 3: SHAPE DERIVATIVE & DOMAIN WIDTH PROPOSAL ---
# =====================================================================
logger.info("=== PHASE 3: Calculating Domain Gradient ===")

# Calculate required spatial derivatives of the forward fields
dx_uf = grad_v_f[0,0]
dx_wf = grad_v_f[0,1]
dx_Tf = grad_T_f[0]
dx_pf = d3.Differentiate(p_f, coords['x'])

dx2_uf = d3.Differentiate(dx_uf, coords['x'])
dx2_wf = d3.Differentiate(dx_wf, coords['x'])
dx2_Tf = d3.Differentiate(dx_Tf, coords['x'])

# Combine forward and adjoint fields into the shape derivative integrand
integrand = ( p_a * dx_uf + 
              u_a * (u_f * dx_uf + dx_pf - 2 * Pr * dx2_uf) + 
              w_a * (u_f * dx_wf - 2 * Pr * dx2_wf) + 
              T_a * (u_f * dx_Tf - 2 * dx2_Tf) )

# Integrate over the domain to compute d(Nu)/d(Lx)
dJ_dLx_expr = d3.integ(integrand) / (Lx**2)

grad_array = dJ_dLx_expr.evaluate()['g']
local_grad = grad_array.flatten()[0] if grad_array.size > 0 else 0.0
dJ_dLx = dist.comm.bcast(local_grad, root=0)

if dist.comm.rank == 0:
    logger.info(f"-> Objective Gradient (dNu/dLx) = {dJ_dLx:.5f}")
    
    # Simple Gradient Ascent Step
    learning_rate = 1.0  # Controls how aggressively to change the box size
    L_new = Lx + learning_rate * dJ_dLx
    
    print("\n" + "="*50)
    print(f"ADJOINT SHAPE OPTIMIZATION RESULT:")
    print(f"  Current Width: Lx = {Lx:.4f}  |  Nu = {global_Nu:.4f}")
    print(f"  Gradient:      dNu/dLx = {dJ_dLx:.4f}")
    print(f"  Proposed Next: Lx = {L_new:.4f}")
    print("="*50 + "\n")

    np.savetxt('next_L.txt', [L_new])
    
    # Optional: Log the history so you can plot J vs L later
    with open('optimization_history.csv', 'a') as f:
        f.write(f"{Lx},{global_Nu},{dJ_dLx}\n")