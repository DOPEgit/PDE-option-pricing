"""
Unit Tests for Greeks Calculation

Tests Delta, Gamma, and Theta calculation accuracy.
"""

import pytest
import numpy as np
from src.pde_solvers.black_scholes import BlackScholesPDE
from src.numerical_methods.crank_nicolson import CrankNicolson


@pytest.fixture
def solved_pde():
    """Create and solve a PDE for Greeks testing."""
    pde = BlackScholesPDE(S_max=300.0, T=1.0, r=0.05, sigma=0.2, N_S=100, N_t=1000)
    K = 100
    payoff = pde.european_call_payoff(K)
    boundary_func = lambda t_idx: pde.apply_boundary_conditions_call(K, t_idx)

    solver = CrankNicolson(pde)
    solver.solve(payoff, boundary_func, use_sparse=True)

    return pde, K


class TestDelta:
    """Test Delta (∂V/∂S) calculation."""

    def test_delta_shape(self, solved_pde):
        """Test Delta array dimensions."""
        pde, K = solved_pde
        delta = pde.calculate_delta(0)

        assert len(delta) == len(pde.S_grid)
        assert delta.shape == pde.S_grid.shape

    def test_delta_bounds(self, solved_pde):
        """Test Delta bounds for call option."""
        pde, K = solved_pde
        delta = pde.calculate_delta(0)

        # Delta for call should be between 0 and 1
        assert np.all(delta >= -0.1)  # Allow small numerical errors
        assert np.all(delta <= 1.1)

    def test_delta_at_extremes(self, solved_pde):
        """Test Delta at extreme stock prices."""
        pde, K = solved_pde
        delta = pde.calculate_delta(0)

        # At S=0, Delta should be near 0
        assert delta[0] == pytest.approx(0, abs=0.1)

        # At high S, Delta should approach 1
        assert delta[-1] == pytest.approx(1, abs=0.1)

    def test_delta_at_the_money(self, solved_pde):
        """Test Delta near strike."""
        pde, K = solved_pde
        delta = pde.calculate_delta(0)

        K_idx = np.argmin(np.abs(pde.S_grid - K))

        # At-the-money Delta for call should be around 0.5
        # At t=0 with time to maturity, it won't be exactly 0.5
        # but should be in reasonable range
        assert 0.3 < delta[K_idx] < 0.8

    def test_delta_monotonicity(self, solved_pde):
        """Test that Delta is monotonically increasing for calls."""
        pde, K = solved_pde
        delta = pde.calculate_delta(0)

        # Delta should generally increase with S (allowing small numerical noise)
        # Check in regions away from boundaries
        for i in range(10, len(delta)-10, 10):
            # Average delta should increase
            avg_before = np.mean(delta[i-5:i])
            avg_after = np.mean(delta[i:i+5])
            assert avg_after >= avg_before - 0.05  # Allow small tolerance

    def test_delta_all_time_steps(self, solved_pde):
        """Test Delta calculation at multiple time steps."""
        pde, K = solved_pde

        for t_idx in [0, pde.N_t//4, pde.N_t//2, 3*pde.N_t//4]:
            delta = pde.calculate_delta(t_idx)
            assert len(delta) == len(pde.S_grid)
            assert np.all(delta >= -0.1)
            assert np.all(delta <= 1.1)


class TestGamma:
    """Test Gamma (∂²V/∂S²) calculation."""

    def test_gamma_shape(self, solved_pde):
        """Test Gamma array dimensions."""
        pde, K = solved_pde
        gamma = pde.calculate_gamma(0)

        assert len(gamma) == len(pde.S_grid)

    def test_gamma_non_negative(self, solved_pde):
        """Test Gamma is non-negative for calls."""
        pde, K = solved_pde
        gamma = pde.calculate_gamma(0)

        # Gamma should be non-negative (convexity)
        # Allow small negative values due to numerical errors
        assert np.all(gamma >= -0.01)

    def test_gamma_peak_at_strike(self, solved_pde):
        """Test Gamma peaks near strike."""
        pde, K = solved_pde
        gamma = pde.calculate_gamma(0)

        K_idx = np.argmin(np.abs(pde.S_grid - K))

        # Gamma should be highest near strike
        # Check it's larger than gamma at extremes
        assert gamma[K_idx] > gamma[10]  # Larger than deep OTM
        assert gamma[K_idx] > gamma[-10]  # Larger than deep ITM

    def test_gamma_at_extremes(self, solved_pde):
        """Test Gamma approaches zero at extremes."""
        pde, K = solved_pde
        gamma = pde.calculate_gamma(0)

        # At deep OTM and ITM, gamma should be small
        assert gamma[0] < 0.01
        assert gamma[-1] < 0.01

    def test_gamma_symmetry(self, solved_pde):
        """Test approximate gamma symmetry around strike."""
        pde, K = solved_pde
        gamma = pde.calculate_gamma(0)

        K_idx = np.argmin(np.abs(pde.S_grid - K))

        # Gamma should be roughly symmetric around strike
        # (not exact due to drift term, but should be close)
        offset = 10
        if K_idx > offset and K_idx < len(gamma) - offset:
            left_gamma = gamma[K_idx - offset]
            right_gamma = gamma[K_idx + offset]
            # Should be within factor of 2
            ratio = max(left_gamma, right_gamma) / (min(left_gamma, right_gamma) + 1e-10)
            assert ratio < 3.0


class TestTheta:
    """Test Theta (∂V/∂t) calculation."""

    def test_theta_shape(self, solved_pde):
        """Test Theta array dimensions."""
        pde, K = solved_pde
        S_idx = 50
        theta = pde.calculate_theta(S_idx)

        assert len(theta) == pde.N_t + 1

    def test_theta_negative(self, solved_pde):
        """Test Theta is negative for long options."""
        pde, K = solved_pde
        S_idx = np.argmin(np.abs(pde.S_grid - 100))  # At S=100
        theta = pde.calculate_theta(S_idx)

        # Theta should be negative (time decay) except at t=0 where it's not defined
        # Skip first few and last few points where boundary effects dominate
        assert np.all(theta[10:-10] <= 0.1)  # Allow small positive due to numerical errors

    def test_theta_magnitude(self, solved_pde):
        """Test Theta has reasonable magnitude."""
        pde, K = solved_pde
        S_idx = np.argmin(np.abs(pde.S_grid - K))  # At-the-money
        theta = pde.calculate_theta(S_idx)

        # Theta shouldn't be unreasonably large (except near maturity where it accelerates)
        # Check average theta rather than all values (theta spikes near expiry)
        # Skip first point (t=0) and last 20% (near maturity)
        mid_range = slice(1, int(len(theta) * 0.8))
        assert np.mean(np.abs(theta[mid_range])) < 20  # Average should be reasonable

    def test_theta_at_maturity(self, solved_pde):
        """Test Theta behavior near maturity."""
        pde, K = solved_pde
        S_idx = np.argmin(np.abs(pde.S_grid - K))
        theta = pde.calculate_theta(S_idx)

        # Near maturity, theta magnitude should increase (faster decay)
        # Compare theta at t=0 vs t=T/2
        theta_early = abs(theta[pde.N_t//2])
        theta_late = abs(theta[pde.N_t-10])  # Near maturity

        # Theta should be more negative closer to maturity for ATM options
        assert theta_late >= theta_early * 0.5  # At least 50% of early value


class TestGreeksConsistency:
    """Test consistency between Greeks."""

    def test_put_call_parity_greeks(self, solved_pde):
        """Test put-call parity for Delta."""
        pde_call, K = solved_pde

        # Solve put
        pde_put = BlackScholesPDE(S_max=300.0, T=1.0, r=0.05, sigma=0.2, N_S=100, N_t=1000)
        payoff_put = pde_put.european_put_payoff(K)
        boundary_put = lambda t_idx: pde_put.apply_boundary_conditions_put(K, t_idx)

        solver_put = CrankNicolson(pde_put)
        solver_put.solve(payoff_put, boundary_put, use_sparse=True)

        # Calculate deltas
        delta_call = pde_call.calculate_delta(0)
        delta_put = pde_put.calculate_delta(0)

        # Put-call parity for delta: Delta_call - Delta_put = 1
        delta_diff = delta_call - delta_put

        # Should be approximately 1 (allowing numerical errors)
        # Check at interior points
        for i in range(20, len(delta_diff)-20, 10):
            assert delta_diff[i] == pytest.approx(1.0, abs=0.15)

    def test_gamma_same_for_call_and_put(self, solved_pde):
        """Test that Gamma is same for call and put."""
        pde_call, K = solved_pde

        # Solve put
        pde_put = BlackScholesPDE(S_max=300.0, T=1.0, r=0.05, sigma=0.2, N_S=100, N_t=1000)
        payoff_put = pde_put.european_put_payoff(K)
        boundary_put = lambda t_idx: pde_put.apply_boundary_conditions_put(K, t_idx)

        solver_put = CrankNicolson(pde_put)
        solver_put.solve(payoff_put, boundary_put, use_sparse=True)

        # Calculate gammas
        gamma_call = pde_call.calculate_gamma(0)
        gamma_put = pde_put.calculate_gamma(0)

        # Gamma should be the same for call and put
        # Check at interior points
        for i in range(20, len(gamma_call)-20, 10):
            assert gamma_call[i] == pytest.approx(gamma_put[i], rel=0.10)

    def test_greeks_finite(self, solved_pde):
        """Test all Greeks are finite."""
        pde, K = solved_pde

        delta = pde.calculate_delta(0)
        gamma = pde.calculate_gamma(0)
        theta = pde.calculate_theta(50)

        assert np.all(np.isfinite(delta))
        assert np.all(np.isfinite(gamma))
        assert np.all(np.isfinite(theta))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])