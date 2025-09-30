"""
European Put Option - Explicit Finite Difference Method Simulation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from src.pde_solvers.black_scholes import BlackScholesPDE
from src.numerical_methods.explicit_fd import ExplicitFD
from src.visualization.surface_plots import plot_option_surface, plot_greeks_surface

S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

print("="*70)
print("EUROPEAN PUT OPTION - EXPLICIT FINITE DIFFERENCE METHOD")
print("="*70)
print(f"\nParameters: S₀=${S0}, K=${K}, T={T}yr, r={r*100}%, σ={sigma*100}%")

print("\nInitializing PDE solver...")
pde = BlackScholesPDE(S_max=300.0, T=T, r=r, sigma=sigma, N_S=150, N_t=3000)

payoff = pde.european_put_payoff(K)
boundary_func = lambda t_idx: pde.apply_boundary_conditions_put(K, t_idx)

print("Solving with Explicit FD method...")
solver = ExplicitFD(pde)
is_stable, alpha_max = solver.check_stability()
print(f"  Stability: {'STABLE' if is_stable else 'UNSTABLE'} (α_max={alpha_max:.4f})")

solver.solve_vectorized(payoff, boundary_func)

S_idx = np.argmin(np.abs(pde.S_grid - S0))
numerical_price = pde.V[S_idx, 0]
analytical_price = pde.get_analytical_bs_put(S0, K, 0)
error = abs(numerical_price - analytical_price)

print(f"\nResults:")
print(f"  Numerical:  ${numerical_price:.4f}")
print(f"  Analytical: ${analytical_price:.4f}")
print(f"  Error:      ${error:.6f} ({(error/analytical_price)*100:.4f}%)")

print("\nCalculating Greeks...")
delta_grid = np.zeros_like(pde.V)
gamma_grid = np.zeros_like(pde.V)
for t_idx in range(pde.N_t + 1):
    delta_grid[:, t_idx] = pde.calculate_delta(t_idx)
    gamma_grid[:, t_idx] = pde.calculate_gamma(t_idx)

print(f"  Delta: {delta_grid[S_idx, 0]:.4f}")
print(f"  Gamma: {gamma_grid[S_idx, 0]:.6f}")

print("\nGenerating visualizations...")
os.makedirs('simulations/european_put/plots', exist_ok=True)

plot_option_surface(
    pde.S_grid, pde.t_grid, pde.V,
    title="European Put Option (Explicit FD)",
    save_path="simulations/european_put/plots/put_surface_explicit.png"
)

plot_greeks_surface(
    pde.S_grid, pde.t_grid, delta_grid, gamma_grid,
    title_prefix="European Put (Explicit)",
    save_path="simulations/european_put/plots/put_greeks_explicit.png"
)

print("\n" + "="*70)
print("SIMULATION COMPLETE")
print("Plots: simulations/european_put/plots/")