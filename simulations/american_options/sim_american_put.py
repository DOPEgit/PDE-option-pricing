"""
American Put Option Simulation

Demonstrates early exercise boundary detection for American options.
Compares American put with European put to show early exercise premium.

Note: This is a simplified implementation. For production, use proper
American option algorithms (e.g., Longstaff-Schwartz, binomial trees).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib.pyplot as plt
from src.pde_solvers.black_scholes import BlackScholesPDE
from src.numerical_methods.crank_nicolson import CrankNicolson
from src.visualization.surface_plots import plot_option_surface

S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

print("="*70)
print("AMERICAN PUT OPTION SIMULATION")
print("="*70)
print(f"\nParameters: S₀=${S0}, K=${K}, T={T}yr, r={r*100}%, σ={sigma*100}%")

# European put for comparison
print("\n1. Solving European Put (baseline)...")
pde_eur = BlackScholesPDE(S_max=300.0, T=T, r=r, sigma=sigma, N_S=150, N_t=1500)
payoff_eur = pde_eur.european_put_payoff(K)
boundary_eur = lambda t_idx: pde_eur.apply_boundary_conditions_put(K, t_idx)

solver_eur = CrankNicolson(pde_eur)
solver_eur.solve(payoff_eur, boundary_eur, use_sparse=True)

S_idx = np.argmin(np.abs(pde_eur.S_grid - S0))
european_price = pde_eur.V[S_idx, 0]
print(f"   European Put Price: ${european_price:.4f}")

# American put (approximate with early exercise check)
print("\n2. Solving American Put (with early exercise)...")
pde_am = BlackScholesPDE(S_max=300.0, T=T, r=r, sigma=sigma, N_S=150, N_t=1500)
payoff_am = pde_am.european_put_payoff(K)

# Solve with early exercise constraint
def american_boundary(t_idx):
    pde_am.apply_boundary_conditions_put(K, t_idx)
    # Apply early exercise constraint: V >= intrinsic value
    intrinsic = np.maximum(K - pde_am.S_grid, 0)
    pde_am.V[:, t_idx] = np.maximum(pde_am.V[:, t_idx], intrinsic)

solver_am = CrankNicolson(pde_am)
solver_am.solve(payoff_am, american_boundary, use_sparse=True)

american_price = pde_am.V[S_idx, 0]
early_exercise_premium = american_price - european_price

print(f"   American Put Price: ${american_price:.4f}")
print(f"   Early Exercise Premium: ${early_exercise_premium:.4f} ({(early_exercise_premium/european_price)*100:.2f}%)")

# Find early exercise boundary
print("\n3. Detecting Early Exercise Boundary...")
exercise_boundary = []
for t_idx in range(pde_am.N_t + 1):
    intrinsic = np.maximum(K - pde_am.S_grid, 0)
    # Find where American value equals intrinsic (exercise boundary)
    diff = np.abs(pde_am.V[:, t_idx] - intrinsic)
    # Find lowest S where difference is minimal (boundary point)
    boundary_idx = np.argmin(diff[1:pde_am.N_S//2]) + 1
    exercise_boundary.append(pde_am.S_grid[boundary_idx])

exercise_boundary = np.array(exercise_boundary)

# Visualizations
print("\n4. Generating visualizations...")
os.makedirs('simulations/american_options/plots', exist_ok=True)

# American put surface
plot_option_surface(
    pde_am.S_grid, pde_am.t_grid, pde_am.V,
    title="American Put Option with Early Exercise",
    save_path="simulations/american_options/plots/american_put_surface.png"
)

# Exercise boundary plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Exercise boundary over time
ax = axes[0]
ax.plot(pde_am.t_grid, exercise_boundary, 'r-', linewidth=2, label='Early Exercise Boundary')
ax.axhline(y=K, color='black', linestyle='--', alpha=0.5, label=f'Strike K=${K}')
ax.axhline(y=S0, color='blue', linestyle='--', alpha=0.5, label=f'Initial S₀=${S0}')
ax.set_xlabel('Time (years)', fontsize=11)
ax.set_ylabel('Stock Price ($)', fontsize=11)
ax.set_title('Early Exercise Boundary', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, T)

# Plot 2: American vs European comparison
ax = axes[1]
price_diff = pde_am.V[:, 0] - pde_eur.V[:, 0]
ax.plot(pde_am.S_grid, pde_am.V[:, 0], 'r-', linewidth=2, label='American Put')
ax.plot(pde_eur.S_grid, pde_eur.V[:, 0], 'b--', linewidth=2, label='European Put')
ax.axvline(x=K, color='black', linestyle='--', alpha=0.5, label=f'Strike K=${K}')
ax.set_xlabel('Stock Price ($)', fontsize=11)
ax.set_ylabel('Option Value ($)', fontsize=11)
ax.set_title('American vs European Put (t=0)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simulations/american_options/plots/exercise_boundary.png', dpi=300, bbox_inches='tight')
plt.close()

# Premium heatmap
fig, ax = plt.subplots(figsize=(10, 8))
premium = pde_am.V - pde_eur.V
T_mesh, S_mesh = np.meshgrid(pde_am.t_grid, pde_am.S_grid)
contour = ax.contourf(T_mesh, S_mesh, premium, levels=20, cmap='RdYlGn')
ax.set_xlabel('Time (years)', fontsize=11)
ax.set_ylabel('Stock Price ($)', fontsize=11)
ax.set_title('Early Exercise Premium (American - European)', fontsize=12, fontweight='bold')
plt.colorbar(contour, ax=ax, label='Premium ($)')
plt.savefig('simulations/american_options/plots/american_vs_european.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("SIMULATION COMPLETE")
print("="*70)
print("\nGenerated plots:")
print("  - simulations/american_options/plots/american_put_surface.png")
print("  - simulations/american_options/plots/exercise_boundary.png")
print("  - simulations/american_options/plots/american_vs_european.png")
print("\nKey Insight: American puts have early exercise premium because")
print("            holding them longer may not compensate for immediate exercise.")