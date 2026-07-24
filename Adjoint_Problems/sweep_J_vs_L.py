import numpy as np
import dedalus.public as d3
import scipy.interpolate as interp
import logging
import os
import sys

# --- HPC PLOTTING FIX ---
import matplotlib
matplotlib.use('Agg')  # Force headless rendering
import matplotlib.pyplot as plt

from mpi4py import MPI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- Global Parameters ----------------
Nx, Nz = 64, 32  # Kept at 64x32 to prevent memory/CFL crashes
Rayleigh = 1e5
Prandtl = 1

stop_sim_time = 5.0
max_timestep = 1e-3
dtype = np.float64
timestepper = d3.RK222

# Average over the last 0.5 time units
averaging_window = 0.5  
start_avg_time = stop_sim_time - averaging_window

def run_convection(L_val, old_L=None, old_T_g=None, old_u_g=None, old_p_g=None):
    """
    Builds and runs the Boussinesq IVP for a specific domain width (L_val).
    If previous state data is provided, it interpolates it to use as the initial condition.
    """
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=dtype)
    ex, ez = coords.unit_vector_fields(dist)

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

    # ---------------- Initial Conditions (Continuation Logic) ----------------
    T.change_scales(1)
    u.change_scales(1)
    p.change_scales(1)

    if old_L is not None:
        old_L = dist.comm.bcast(old_L, root=0)
        old_T_g = dist.comm.bcast(old_T_g, root=0)
        old_u_g = dist.comm.bcast(old_u_g, root=0)
        old_p_g = dist.comm.bcast(old_p_g, root=0)

        # Map local X coordinates to old domain space
        x_old_global = np.linspace(0, old_L, Nx, endpoint=False)
        x_local = x[:, 0] 
        x_eval_local = x_local * (old_L / L_val)
        
        # Find local Z indices to prevent MPI broadcast errors
        z_global = zbasis.global_grid(dist, scale=1)[0, :]
        z_local = z[0, :]
        z_start = np.argmin(np.abs(z_global - z_local[0]))
        z_end = z_start + len(z_local)

        f_T = interp.interp1d(x_old_global, old_T_g, axis=0, kind='cubic', fill_value='extrapolate')
        T['g'] = f_T(x_eval_local)[:, z_start:z_end]
        
        f_p = interp.interp1d(x_old_global, old_p_g, axis=0, kind='cubic', fill_value='extrapolate')
        p['g'] = f_p(x_eval_local)[:, z_start:z_end]

        f_u = interp.interp1d(x_old_global, old_u_g, axis=1, kind='cubic', fill_value='extrapolate')
        u_local_all_z = f_u(x_eval_local)
        u['g'][0] = u_local_all_z[0][:, z_start:z_end]
        u['g'][1] = u_local_all_z[1][:, z_start:z_end]
    else:
        T['g'] = 1 - z
        amp = 1.0
        k_x = 2 * np.pi / L_val 
        u['g'][0] = 2 * amp * np.pi * np.sin(k_x * x) * np.sin(np.pi * z) * np.cos(np.pi * z)
        u['g'][1] = -amp * k_x * np.cos(k_x * x) * (np.sin(np.pi * z)**2)
        T['g'] += 0.2 * np.cos(k_x * x) * np.sin(np.pi * z)

    # ---------------- Time-Stepping & Averaging ----------------
    CFL = d3.CFL(solver, initial_dt=1e-6, cadence=10, safety=0.5, threshold=0.05,
                 max_change=1.5, min_change=0.5, max_dt=max_timestep)
    CFL.add_velocity(u)

    sample_count = 0
    sum_u_c = np.zeros_like(u['c'])
    sum_T_c = np.zeros_like(T['c'])
    sum_p_c = np.zeros_like(p['c'])
    sum_J = 0.0

    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        
        if solver.sim_time >= start_avg_time:
            J_integrand = d3.integ(1 + (u@ez) * T) / L_val
            J_val_array = J_integrand.evaluate()['g']
            local_J = J_val_array.flatten()[0] if J_val_array.size > 0 else 0.0
            J_current = dist.comm.bcast(local_J, root=0)
            
            sum_J += J_current
            sum_u_c += u['c']
            sum_T_c += T['c']
            sum_p_c += p['c']
            sample_count += 1
        
        if solver.iteration % 500 == 0 and dist.comm.rank == 0:
            logger.info(f"   ... Sim Time: {solver.sim_time:.3f} / {stop_sim_time}")

    if sample_count > 0:
        J_final = sum_J / sample_count
        u['c'] = sum_u_c / sample_count
        T['c'] = sum_T_c / sample_count
        p['c'] = sum_p_c / sample_count
    else:
        J_integrand = d3.integ(1 + (u@ez) * T) / L_val
        J_val_array = J_integrand.evaluate()['g']
        local_J = J_val_array.flatten()[0] if J_val_array.size > 0 else 0.0
        J_final = dist.comm.bcast(local_J, root=0)

    T.change_scales(1)
    u.change_scales(1)
    p.change_scales(1)

    T_global = T.allgather_data('g')
    u_global = u.allgather_data('g')
    p_global = p.allgather_data('g')
    
    return J_final, T_global, u_global, p_global

# ==============================================================================
# ----------------------------- MASTER SWEEP LOOP ------------------------------
# ==============================================================================

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # We sweep downwards from L=2.2 to L=1.2 to map the physical "cliff"
    L_values = np.linspace(2.2, 1.2, 20) 
    J_values = []

    old_L = None
    old_T_g, old_u_g, old_p_g = None, None, None

    if rank == 0:
        logger.info("========== Starting Parameter Sweep ==========")

    for i, L in enumerate(L_values):
        if rank == 0:
            logger.info(f"\n--- Run {i+1}/{len(L_values)} | Evaluating L = {L:.4f} ---")
        
        # Execute the forward solver for this L
        J, T_g_new, u_g_new, p_g_new = run_convection(L, old_L, old_T_g, old_u_g, old_p_g)
        
        J_values.append(J)
        old_L = L
        old_T_g, old_u_g, old_p_g = T_g_new, u_g_new, p_g_new
        
        if rank == 0:
            logger.info(f"-> Successfully evaluated: J = {J:.4f} at L = {L:.4f}")

    # ---------------- Plotting the J vs L Landscape ----------------
    if rank == 0:
        logger.info("\n========== Parameter Sweep Complete ==========")
        logger.info("Generating J vs L landscape plot...")

        # Set up a nicely proportioned figure for an A4 report
        plt.figure(figsize=(8, 5))
        
        # Plot the landscape
        plt.plot(L_values, J_values, marker='o', color='purple', linewidth=2.5, markersize=7, zorder=3)

        # Highlight the peak
        peak_idx = np.argmax(J_values)
        plt.plot(L_values[peak_idx], J_values[peak_idx], marker='*', color='gold', markersize=15, 
                 markeredgecolor='black', zorder=4, label=f'Local Peak (L={L_values[peak_idx]:.2f}, J={J_values[peak_idx]:.2f})')

        plt.title('Time-Averaged Nusselt Number vs. Domain Width', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Domain Width ($L$)', fontsize=12)
        plt.ylabel('Nusselt Number ($J$)', fontsize=12)

        plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
        plt.legend(loc='lower right', framealpha=0.9)

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(BASE_DIR, 'J_vs_L_landscape_clean.png')
        
        # tight_layout and bbox_inches='tight' ensure zero wasted margins
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Landscape plot saved successfully to {plot_path}")
        
    comm.Barrier()