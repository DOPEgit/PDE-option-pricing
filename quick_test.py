"""
Quick test to verify all components work before running full demo.
This should complete in ~30 seconds.
"""

import numpy as np
import sys
import os

print("="*70)
print("QUICK SYSTEM TEST")
print("="*70)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from src.pde_solvers.black_scholes import BlackScholesPDE
    from src.numerical_methods.crank_nicolson import CrankNicolson
    from src.visualization.surface_plots import plot_option_surface
    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Create PDE instance
print("\n2. Testing PDE solver initialization...")
try:
    pde = BlackScholesPDE(
        S_max=200.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        N_S=50,  # Small grid for quick test
        N_t=100
    )
    print(f"   ✅ PDE created: Grid {pde.N_S}x{pde.N_t}")
except Exception as e:
    print(f"   ❌ PDE creation failed: {e}")
    sys.exit(1)

# Test 3: Solve European call
print("\n3. Testing Crank-Nicolson solver...")
try:
    K = 100
    payoff = pde.european_call_payoff(K)
    boundary_func = lambda t_idx: pde.apply_boundary_conditions_call(K, t_idx)

    solver = CrankNicolson(pde)
    solver.solve(payoff, boundary_func, use_sparse=True)

    S_idx = np.argmin(np.abs(pde.S_grid - 100))
    price = pde.V[S_idx, 0]

    # Analytical comparison
    analytical = pde.get_analytical_bs_call(100, K, 0)
    error = abs(price - analytical)

    print(f"   ✅ Solver completed")
    print(f"      Numerical: ${price:.4f}")
    print(f"      Analytical: ${analytical:.4f}")
    print(f"      Error: ${error:.6f}")
except Exception as e:
    print(f"   ❌ Solver failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Calculate Greeks
print("\n4. Testing Greeks calculation...")
try:
    delta = pde.calculate_delta(0)
    gamma = pde.calculate_gamma(0)
    print(f"   ✅ Greeks calculated")
    print(f"      Delta at S=100: {delta[S_idx]:.4f}")
    print(f"      Gamma at S=100: {gamma[S_idx]:.6f}")
except Exception as e:
    print(f"   ❌ Greeks calculation failed: {e}")
    sys.exit(1)

# Test 5: Create a simple plot
print("\n5. Testing visualization...")
try:
    os.makedirs('plots/test', exist_ok=True)

    plot_option_surface(
        pde.S_grid,
        pde.t_grid,
        pde.V,
        title="Quick Test: European Call",
        save_path="plots/test/quick_test_surface.png"
    )
    print(f"   ✅ Plot saved: plots/test/quick_test_surface.png")
except Exception as e:
    print(f"   ❌ Visualization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nSystem is ready. You can now run:")
print("  python main_demo.py")
print("\nNote: Full demo takes 10-15 minutes and generates all visualizations.")