"""
European Call Option - Explicit Finite Difference Method Simulation

Demonstrates the explicit FD method for European call pricing.
Shows stability considerations and fast computation for stable grids.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from src.pde_solvers.black_scholes import BlackScholesPDE
from src.numerical_methods.explicit_fd import ExplicitFD
from src.visualization.surface_plots import plot_option_surface, plot_greeks_surface

# Option parameters
S0 = 100
K = 100
T = 1.0
r = 0.05
sigma = 0.2

print("="*70)
print("EUROPEAN CALL OPTION - EXPLICIT FINITE DIFFERENCE METHOD")
print("="*70)
print(f"\nParameters:")
print(f"  S₀ = ${S0}, K = ${K}, T = {T}yr, r = {r*100}%, σ = {sigma*100}%")

# Create PDE instance with appropriate grid for stability
print("\nInitializing PDE solver...")
pde = BlackScholesPDE(
    S_max=300.0,
    T=T,
    r=r,
    sigma=sigma,
    N_S=150,
    N_t=3000  # Larger N_t for explicit stability
)

payoff = pde.european_call_payoff(K)
boundary_func = lambda t_idx: pde.apply_boundary_conditions_call(K, t_idx)

# Solve with explicit method
print("Solving with Explicit FD method...")
solver = ExplicitFD(pde)

# Check stability
is_stable, alpha_max = solver.check_stability()
print(f"  Stability check: {'STABLE' if is_stable else 'UNSTABLE'}")
print(f"  Alpha_max = {alpha_max:.4f} (must be ≤ 0.5)")

solver.solve_vectorized(payoff, boundary_func)

# Results
S_idx = np.argmin(np.abs(pde.S_grid - S0))
numerical_price = pde.V[S_idx, 0]
analytical_price = pde.get_analytical_bs_call(S0, K, 0)
error = abs(numerical_price - analytical_price)

print(f"\nResults:")
print(f"  Numerical:  ${numerical_price:.4f}")
print(f"  Analytical: ${analytical_price:.4f}")
print(f"  Error:      ${error:.6f} ({(error/analytical_price)*100:.4f}%)")

# Greeks
print("\nCalculating Greeks...")
delta_grid = np.zeros_like(pde.V)
gamma_grid = np.zeros_like(pde.V)

for t_idx in range(pde.N_t + 1):
    delta_grid[:, t_idx] = pde.calculate_delta(t_idx)
    gamma_grid[:, t_idx] = pde.calculate_gamma(t_idx)

print(f"  Delta: {delta_grid[S_idx, 0]:.4f}")
print(f"  Gamma: {gamma_grid[S_idx, 0]:.6f}")

# Generate plots
print("\nGenerating visualizations...")
os.makedirs('simulations/european_call/plots', exist_ok=True)

plot_option_surface(
    pde.S_grid,
    pde.t_grid,
    pde.V,
    title="European Call Option (Explicit FD)",
    save_path="simulations/european_call/plots/call_surface_explicit.png"
)

plot_greeks_surface(
    pde.S_grid,
    pde.t_grid,
    delta_grid,
    gamma_grid,
    title_prefix="European Call (Explicit)",
    save_path="simulations/european_call/plots/call_greeks_explicit.png"
)

print("\n" + "="*70)
print("SIMULATION COMPLETE")
print("="*70)
print("\nGenerated plots:")
print("  - simulations/european_call/plots/call_surface_explicit.png")
print("  - simulations/european_call/plots/call_greeks_explicit.png")