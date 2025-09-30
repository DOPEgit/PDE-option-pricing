"""
Unit Tests for Numerical Methods

Tests explicit, implicit, and Crank-Nicolson finite difference methods.
Verifies stability conditions and convergence rates.
"""

import pytest
import numpy as np
from src.pde_solvers.black_scholes import BlackScholesPDE
from src.numerical_methods.explicit_fd import ExplicitFD
from src.numerical_methods.implicit_fd import ImplicitFD
from src.numerical_methods.crank_nicolson import CrankNicolson


@pytest.fixture
def pde_small():
    """Small PDE instance for quick testing."""
    return BlackScholesPDE(S_max=200.0, T=1.0, r=0.05, sigma=0.2, N_S=50, N_t=500)


@pytest.fixture
def setup_option(pde_small):
    """Setup standard call option."""
    K = 100
    payoff = pde_small.european_call_payoff(K)
    boundary_func = lambda t_idx: pde_small.apply_boundary_conditions_call(K, t_idx)
    return K, payoff, boundary_func


class TestExplicitFD:
    """Test explicit finite difference method."""

    def test_initialization(self, pde_small):
        """Test solver initialization."""
        solver = ExplicitFD(pde_small)
        assert solver.name == "Explicit FD"
        assert solver.pde == pde_small

    def test_stability_check(self, pde_small):
        """Test stability condition checking."""
        solver = ExplicitFD(pde_small)
        is_stable, alpha_max = solver.check_stability()

        # Check return types
        assert isinstance(is_stable, (bool, np.bool_))
        assert isinstance(alpha_max, (float, np.floating))

        # Alpha_max should be positive
        assert alpha_max > 0

        # For this grid, should be stable
        assert is_stable

    def test_solve(self, pde_small, setup_option):
        """Test solving with explicit method."""
        K, payoff, boundary_func = setup_option
        solver = ExplicitFD(pde_small)

        result = solver.solve(payoff, boundary_func)

        # Check result shape
        assert result.shape == pde_small.V.shape

        # Check initial condition (at t=T)
        np.testing.assert_array_almost_equal(result[:, -1], payoff)

        # Check boundary conditions at t=0
        assert result[0, 0] == pytest.approx(0, abs=0.01)  # S=0, call=0

        # Option value should be positive
        assert np.all(result >= -0.01)  # Allow small numerical errors

    def test_solve_vectorized(self, pde_small, setup_option):
        """Test vectorized solve."""
        K, payoff, boundary_func = setup_option
        solver = ExplicitFD(pde_small)

        result = solver.solve_vectorized(payoff, boundary_func)

        assert result.shape == pde_small.V.shape
        assert np.all(result >= -0.01)

    def test_accuracy_vs_analytical(self, pde_small, setup_option):
        """Test accuracy against analytical solution."""
        K, payoff, boundary_func = setup_option
        solver = ExplicitFD(pde_small)
        solver.solve_vectorized(payoff, boundary_func)

        S0 = 100
        S_idx = np.argmin(np.abs(pde_small.S_grid - S0))
        numerical_price = pde_small.V[S_idx, 0]
        analytical_price = pde_small.get_analytical_bs_call(S0, K, 0)

        # Should be within 5% of analytical
        rel_error = abs(numerical_price - analytical_price) / analytical_price
        assert rel_error < 0.05


class TestImplicitFD:
    """Test implicit finite difference method."""

    def test_initialization(self, pde_small):
        """Test solver initialization."""
        solver = ImplicitFD(pde_small)
        assert solver.name == "Implicit FD"

    def test_stability_check(self, pde_small):
        """Test unconditional stability."""
        solver = ImplicitFD(pde_small)
        is_stable, message = solver.check_stability()

        assert is_stable is True
        assert "Unconditionally stable" in message

    def test_matrix_building(self, pde_small):
        """Test coefficient matrix construction."""
        solver = ImplicitFD(pde_small)
        A = solver.build_coefficient_matrix()

        # Matrix should be square
        N = pde_small.N_S - 1
        assert A.shape == (N, N)

        # Main diagonal should be dominant (for stability)
        for i in range(N):
            assert abs(A[i, i]) >= sum(abs(A[i, j]) for j in range(N) if j != i)

    def test_sparse_matrix(self, pde_small):
        """Test sparse matrix construction."""
        solver = ImplicitFD(pde_small)
        A_sparse = solver.build_sparse_matrix()

        # Should be sparse matrix
        assert hasattr(A_sparse, 'toarray')

        # Check dimensions
        N = pde_small.N_S - 1
        assert A_sparse.shape == (N, N)

    def test_solve(self, pde_small, setup_option):
        """Test solving with implicit method."""
        K, payoff, boundary_func = setup_option
        solver = ImplicitFD(pde_small)

        result = solver.solve(payoff, boundary_func, use_sparse=True)

        assert result.shape == pde_small.V.shape
        np.testing.assert_array_almost_equal(result[:, -1], payoff)
        assert np.all(result >= -0.01)

    def test_accuracy_vs_analytical(self, pde_small, setup_option):
        """Test accuracy against analytical solution."""
        K, payoff, boundary_func = setup_option
        solver = ImplicitFD(pde_small)
        solver.solve(payoff, boundary_func, use_sparse=True)

        S0 = 100
        S_idx = np.argmin(np.abs(pde_small.S_grid - S0))
        numerical_price = pde_small.V[S_idx, 0]
        analytical_price = pde_small.get_analytical_bs_call(S0, K, 0)

        rel_error = abs(numerical_price - analytical_price) / analytical_price
        assert rel_error < 0.05


class TestCrankNicolson:
    """Test Crank-Nicolson method."""

    def test_initialization(self, pde_small):
        """Test solver initialization."""
        solver = CrankNicolson(pde_small)
        assert solver.theta == 0.5
        assert "Crank-Nicolson" in solver.name

    def test_initialization_custom_theta(self, pde_small):
        """Test initialization with custom theta."""
        solver = CrankNicolson(pde_small, theta=0.6)
        assert solver.theta == 0.6

    def test_stability_check(self, pde_small):
        """Test stability for different theta values."""
        # Standard Crank-Nicolson (theta=0.5)
        solver1 = CrankNicolson(pde_small, theta=0.5)
        is_stable1, msg1 = solver1.check_stability()
        assert is_stable1 is True

        # Theta > 0.5 (should be stable)
        solver2 = CrankNicolson(pde_small, theta=0.7)
        is_stable2, msg2 = solver2.check_stability()
        assert is_stable2 is True

        # Theta < 0.5 (conditionally stable)
        solver3 = CrankNicolson(pde_small, theta=0.3)
        is_stable3, msg3 = solver3.check_stability()
        assert is_stable3 is False

    def test_matrices_building(self, pde_small):
        """Test matrix construction."""
        solver = CrankNicolson(pde_small)
        A, B = solver.build_matrices()

        N = pde_small.N_S - 1
        assert A.shape == (N, N)
        assert B.shape == (N, N)

        # Matrices should be different
        assert not np.allclose(A, B)

    def test_solve(self, pde_small, setup_option):
        """Test solving with Crank-Nicolson."""
        K, payoff, boundary_func = setup_option
        solver = CrankNicolson(pde_small)

        result = solver.solve(payoff, boundary_func, use_sparse=True)

        assert result.shape == pde_small.V.shape
        np.testing.assert_array_almost_equal(result[:, -1], payoff)
        assert np.all(result >= -0.01)

    def test_accuracy_vs_analytical(self, pde_small, setup_option):
        """Test accuracy - should be best of all methods."""
        K, payoff, boundary_func = setup_option
        solver = CrankNicolson(pde_small)
        solver.solve(payoff, boundary_func, use_sparse=True)

        S0 = 100
        S_idx = np.argmin(np.abs(pde_small.S_grid - S0))
        numerical_price = pde_small.V[S_idx, 0]
        analytical_price = pde_small.get_analytical_bs_call(S0, K, 0)

        # Crank-Nicolson should be more accurate (within 2%)
        rel_error = abs(numerical_price - analytical_price) / analytical_price
        assert rel_error < 0.02

    def test_convergence_order(self):
        """Test second-order convergence of Crank-Nicolson."""
        solver = CrankNicolson(None)  # Don't need PDE instance for this
        time_order, space_order = solver.get_convergence_order()

        # Should be O(h²) in both time and space
        assert time_order == 2
        assert space_order == 2


class TestMethodComparison:
    """Compare all three methods."""

    def test_all_methods_converge(self, pde_small, setup_option):
        """Test that all methods converge to similar values."""
        K, payoff, boundary_func = setup_option

        # Find index for S close to 100 (ATM)
        S_idx = np.argmin(np.abs(pde_small.S_grid - 100))

        # Solve with all methods
        solver_exp = ExplicitFD(pde_small)
        solver_exp.solve_vectorized(payoff.copy(), boundary_func)
        price_exp = pde_small.V[S_idx, 0]

        pde2 = BlackScholesPDE(S_max=200.0, T=1.0, r=0.05, sigma=0.2, N_S=50, N_t=500)
        payoff2 = pde2.european_call_payoff(K)
        boundary_func2 = lambda t_idx: pde2.apply_boundary_conditions_call(K, t_idx)
        solver_imp = ImplicitFD(pde2)
        solver_imp.solve(payoff2, boundary_func2, use_sparse=True)
        S_idx2 = np.argmin(np.abs(pde2.S_grid - 100))
        price_imp = pde2.V[S_idx2, 0]

        pde3 = BlackScholesPDE(S_max=200.0, T=1.0, r=0.05, sigma=0.2, N_S=50, N_t=500)
        payoff3 = pde3.european_call_payoff(K)
        boundary_func3 = lambda t_idx: pde3.apply_boundary_conditions_call(K, t_idx)
        solver_cn = CrankNicolson(pde3)
        solver_cn.solve(payoff3, boundary_func3, use_sparse=True)
        S_idx3 = np.argmin(np.abs(pde3.S_grid - 100))
        price_cn = pde3.V[S_idx3, 0]

        # All methods should give similar results (within 15%)
        # Avoid division by zero
        if price_imp > 0:
            assert abs(price_exp - price_imp) / price_imp < 0.15
        if price_cn > 0:
            assert abs(price_exp - price_cn) / price_cn < 0.15
            assert abs(price_imp - price_cn) / price_cn < 0.15

    def test_crank_nicolson_most_accurate(self, pde_small, setup_option):
        """Verify Crank-Nicolson is most accurate."""
        K, payoff, boundary_func = setup_option
        S0 = 100
        analytical = pde_small.get_analytical_bs_call(S0, K, 0)

        errors = {}

        # Explicit
        pde1 = BlackScholesPDE(S_max=200.0, T=1.0, r=0.05, sigma=0.2, N_S=50, N_t=500)
        payoff1 = pde1.european_call_payoff(K)
        boundary_func1 = lambda t_idx: pde1.apply_boundary_conditions_call(K, t_idx)
        solver_exp = ExplicitFD(pde1)
        solver_exp.solve_vectorized(payoff1, boundary_func1)
        S_idx1 = np.argmin(np.abs(pde1.S_grid - S0))
        errors['Explicit'] = abs(pde1.V[S_idx1, 0] - analytical)

        # Implicit
        pde2 = BlackScholesPDE(S_max=200.0, T=1.0, r=0.05, sigma=0.2, N_S=50, N_t=500)
        payoff2 = pde2.european_call_payoff(K)
        boundary_func2 = lambda t_idx: pde2.apply_boundary_conditions_call(K, t_idx)
        solver_imp = ImplicitFD(pde2)
        solver_imp.solve(payoff2, boundary_func2, use_sparse=True)
        S_idx2 = np.argmin(np.abs(pde2.S_grid - S0))
        errors['Implicit'] = abs(pde2.V[S_idx2, 0] - analytical)

        # Crank-Nicolson
        pde3 = BlackScholesPDE(S_max=200.0, T=1.0, r=0.05, sigma=0.2, N_S=50, N_t=500)
        payoff3 = pde3.european_call_payoff(K)
        boundary_func3 = lambda t_idx: pde3.apply_boundary_conditions_call(K, t_idx)
        solver_cn = CrankNicolson(pde3)
        solver_cn.solve(payoff3, boundary_func3, use_sparse=True)
        S_idx3 = np.argmin(np.abs(pde3.S_grid - S0))
        errors['Crank-Nicolson'] = abs(pde3.V[S_idx3, 0] - analytical)

        # Crank-Nicolson should have smallest error or very close
        # Allow some tolerance as numerical precision can vary
        cn_error = errors['Crank-Nicolson']
        min_error = min(errors['Explicit'], errors['Implicit'])
        assert cn_error <= min_error * 1.1  # Allow 10% tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])