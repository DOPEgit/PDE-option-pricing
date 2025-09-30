"""
Stability Analysis for Finite Difference Methods

Tests stability limits of explicit, implicit, and Crank-Nicolson methods
by varying grid parameters and checking for numerical blow-up.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib.pyplot as plt
from src.pde_solvers.black_scholes import BlackScholesPDE
from src.numerical_methods.explicit_fd import ExplicitFD
from src.numerical_methods.implicit_fd import ImplicitFD
from src.numerical_methods.crank_nicolson import CrankNicolson

print("="*70)
print("STABILITY ANALYSIS: FINITE DIFFERENCE METHODS")
print("="*70)

# Fixed parameters
S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
S_max = 300.0
N_S = 100

# Vary time steps to test stability
N_t_values = [100, 200, 500, 1000, 2000, 5000]

results = {
    'Explicit FD': {'stable': [], 'alpha_max': [], 'prices': []},
    'Implicit FD': {'stable': [], 'alpha_max': [], 'prices': []},
    'Crank-Nicolson': {'stable': [], 'alpha_max': [], 'prices': []}
}

print("\nTesting stability across different time step sizes...")
print(f"Grid: N_S={N_S}, varying N_t = {N_t_values}")

for N_t in N_t_values:
    print(f"\n--- N_t = {N_t} ---")

    # Create PDE instance
    pde = BlackScholesPDE(S_max=S_max, T=T, r=r, sigma=sigma, N_S=N_S, N_t=N_t)
    payoff = pde.european_call_payoff(K)
    boundary_func = lambda t_idx: pde.apply_boundary_conditions_call(K, t_idx)
    S_idx = np.argmin(np.abs(pde.S_grid - S0))

    # Test Explicit FD
    try:
        solver_exp = ExplicitFD(pde)
        is_stable, alpha_max = solver_exp.check_stability()
        solver_exp.solve_vectorized(payoff, boundary_func)
        price = pde.V[S_idx, 0]

        # Check for numerical blow-up
        if np.isnan(price) or np.isinf(price) or abs(price) > 1000:
            is_stable = False
            price = np.nan

        results['Explicit FD']['stable'].append(is_stable)
        results['Explicit FD']['alpha_max'].append(alpha_max)
        results['Explicit FD']['prices'].append(price)
        print(f"  Explicit:  α_max={alpha_max:.4f}, Stable={is_stable}, Price=${price:.4f if not np.isnan(price) else 'NaN'}")
    except Exception as e:
        results['Explicit FD']['stable'].append(False)
        results['Explicit FD']['alpha_max'].append(np.nan)
        results['Explicit FD']['prices'].append(np.nan)
        print(f"  Explicit:  FAILED - {str(e)[:50]}")

    # Test Implicit FD
    try:
        pde_imp = BlackScholesPDE(S_max=S_max, T=T, r=r, sigma=sigma, N_S=N_S, N_t=N_t)
        solver_imp = ImplicitFD(pde_imp)
        solver_imp.solve(payoff, boundary_func, use_sparse=True)
        price_imp = pde_imp.V[S_idx, 0]

        results['Implicit FD']['stable'].append(True)
        results['Implicit FD']['alpha_max'].append(0)  # Unconditionally stable
        results['Implicit FD']['prices'].append(price_imp)
        print(f"  Implicit:  Unconditionally stable, Price=${price_imp:.4f}")
    except Exception as e:
        results['Implicit FD']['stable'].append(False)
        results['Implicit FD']['alpha_max'].append(np.nan)
        results['Implicit FD']['prices'].append(np.nan)
        print(f"  Implicit:  FAILED - {str(e)[:50]}")

    # Test Crank-Nicolson
    try:
        pde_cn = BlackScholesPDE(S_max=S_max, T=T, r=r, sigma=sigma, N_S=N_S, N_t=N_t)
        solver_cn = CrankNicolson(pde_cn)
        solver_cn.solve(payoff, boundary_func, use_sparse=True)
        price_cn = pde_cn.V[S_idx, 0]

        results['Crank-Nicolson']['stable'].append(True)
        results['Crank-Nicolson']['alpha_max'].append(0)
        results['Crank-Nicolson']['prices'].append(price_cn)
        print(f"  Crank-Nicolson: Unconditionally stable, Price=${price_cn:.4f}")
    except Exception as e:
        results['Crank-Nicolson']['stable'].append(False)
        results['Crank-Nicolson']['alpha_max'].append(np.nan)
        results['Crank-Nicolson']['prices'].append(np.nan)
        print(f"  Crank-Nicolson: FAILED - {str(e)[:50]}")

# Analytical reference
pde_ref = BlackScholesPDE(S_max=S_max, T=T, r=r, sigma=sigma, N_S=100, N_t=1000)
analytical_price = pde_ref.get_analytical_bs_call(S0, K, 0)
print(f"\nAnalytical Black-Scholes Price: ${analytical_price:.4f}")

# Visualization
print("\nGenerating stability analysis plots...")
os.makedirs('simulations/convergence_analysis/plots', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Stability parameter (alpha_max) vs N_t
ax = axes[0, 0]
ax.plot(N_t_values, results['Explicit FD']['alpha_max'], 'o-', linewidth=2, markersize=8, label='Explicit FD')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Stability Limit (α=0.5)')
ax.set_xlabel('Number of Time Steps (N_t)', fontsize=11)
ax.set_ylabel('Stability Parameter (α_max)', fontsize=11)
ax.set_title('Explicit Method Stability Parameter', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 2: Option prices vs N_t
ax = axes[0, 1]
for method, data in results.items():
    prices = np.array(data['prices'])
    mask = ~np.isnan(prices)
    if mask.sum() > 0:
        ax.plot(np.array(N_t_values)[mask], prices[mask], 'o-', linewidth=2, markersize=6, label=method, alpha=0.8)

ax.axhline(y=analytical_price, color='black', linestyle='--', linewidth=2, label='Analytical')
ax.set_xlabel('Number of Time Steps (N_t)', fontsize=11)
ax.set_ylabel('Option Price ($)', fontsize=11)
ax.set_title('Price Convergence with Time Steps', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 3: Stability regions
ax = axes[1, 0]
stability_data = []
for method, data in results.items():
    stable_count = sum(data['stable'])
    stability_data.append((method, stable_count, len(data['stable']) - stable_count))

methods = [d[0] for d in stability_data]
stable = [d[1] for d in stability_data]
unstable = [d[2] for d in stability_data]

x = np.arange(len(methods))
width = 0.35
ax.bar(x, stable, width, label='Stable', color='green', alpha=0.7)
ax.bar(x, unstable, width, bottom=stable, label='Unstable', color='red', alpha=0.7)
ax.set_ylabel('Number of Test Cases', fontsize=11)
ax.set_title('Stability Test Results', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Errors vs N_t
ax = axes[1, 1]
for method, data in results.items():
    prices = np.array(data['prices'])
    errors = np.abs(prices - analytical_price)
    mask = ~np.isnan(errors)
    if mask.sum() > 0:
        ax.semilogy(np.array(N_t_values)[mask], errors[mask], 'o-', linewidth=2, markersize=6, label=method, alpha=0.8)

ax.set_xlabel('Number of Time Steps (N_t)', fontsize=11)
ax.set_ylabel('Absolute Error ($)', fontsize=11)
ax.set_title('Error vs Time Steps', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, which='both')
ax.set_xscale('log')

plt.suptitle('Stability Analysis: Finite Difference Methods', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('simulations/convergence_analysis/plots/stability_regions.png', dpi=300, bbox_inches='tight')
print("  Saved: simulations/convergence_analysis/plots/stability_regions.png")

print("\n" + "="*70)
print("STABILITY ANALYSIS COMPLETE")
print("="*70)
print("\nKey Findings:")
print("  - Explicit FD: Conditionally stable (requires small Δt)")
print("  - Implicit FD: Unconditionally stable (all N_t work)")
print("  - Crank-Nicolson: Unconditionally stable + best accuracy")