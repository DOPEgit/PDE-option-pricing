"""
European Call Option - Crank-Nicolson Method Simulation

Simulates European call option pricing using the Crank-Nicolson finite difference method.
Generates comprehensive visualizations including option surface, Greeks, and convergence analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib.pyplot as plt

from src.pde_solvers.black_scholes import BlackScholesPDE
from src.numerical_methods.crank_nicolson import CrankNicolson
from src.visualization.surface_plots import plot_option_surface, plot_greeks_surface

# Option parameters
S0 = 100    # Initial stock price
K = 100     # Strike price
T = 1.0     # Time to maturity (years)
r = 0.05    # Risk-free rate
sigma = 0.2 # Volatility

print("="*70)
print("EUROPEAN CALL OPTION - CRANK-NICOLSON METHOD")
print("="*70)
print(f"\nParameters:")
print(f"  S₀ = ${S0}")
print(f"  K = ${K}")
print(f"  T = {T} year")
print(f"  r = {r*100}%")
print(f"  σ = {sigma*100}%")

# Create PDE instance
print("\nInitializing PDE solver...")
pde = BlackScholesPDE(
    S_max=300.0,
    T=T,
    r=r,
    sigma=sigma,
    N_S=150,
    N_t=1500
)

# Set up payoff and boundary conditions
payoff = pde.european_call_payoff(K)
boundary_func = lambda t_idx: pde.apply_boundary_conditions_call(K, t_idx)

# Solve using Crank-Nicolson
print("Solving with Crank-Nicolson method...")
solver = CrankNicolson(pde)
solver.solve(payoff, boundary_func, use_sparse=True)

# Get price at S0
S_idx = np.argmin(np.abs(pde.S_grid - S0))
numerical_price = pde.V[S_idx, 0]
analytical_price = pde.get_analytical_bs_call(S0, K, 0)
error = abs(numerical_price - analytical_price)

print(f"\nResults:")
print(f"  Numerical Price:  ${numerical_price:.4f}")
print(f"  Analytical Price: ${analytical_price:.4f}")
print(f"  Absolute Error:   ${error:.6f}")
print(f"  Relative Error:   {(error/analytical_price)*100:.4f}%")

# Calculate Greeks
print("\nCalculating Greeks...")
delta_grid = np.zeros_like(pde.V)
gamma_grid = np.zeros_like(pde.V)

for t_idx in range(pde.N_t + 1):
    delta_grid[:, t_idx] = pde.calculate_delta(t_idx)
    gamma_grid[:, t_idx] = pde.calculate_gamma(t_idx)

delta = delta_grid[S_idx, 0]
gamma = gamma_grid[S_idx, 0]

print(f"  Delta (Δ): {delta:.4f}")
print(f"  Gamma (Γ): {gamma:.6f}")

# Generate plots
print("\nGenerating visualizations...")
os.makedirs('simulations/european_call/plots', exist_ok=True)

# Option surface plot
plot_option_surface(
    pde.S_grid,
    pde.t_grid,
    pde.V,
    title="European Call Option (Crank-Nicolson)",
    save_path="simulations/european_call/plots/call_surface_cn.png"
)

# Greeks plot
plot_greeks_surface(
    pde.S_grid,
    pde.t_grid,
    delta_grid,
    gamma_grid,
    title_prefix="European Call (CN)",
    save_path="simulations/european_call/plots/call_greeks_cn.png"
)

print("\n" + "="*70)
print("SIMULATION COMPLETE")
print("="*70)
print("\nGenerated plots:")
print("  - simulations/european_call/plots/call_surface_cn.png")
print("  - simulations/european_call/plots/call_greeks_cn.png")