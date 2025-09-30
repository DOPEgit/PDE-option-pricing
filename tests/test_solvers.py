"""
Unit Tests for PDE Solvers

Tests Black-Scholes PDE initialization, payoff functions, and
comparison with analytical solutions.
"""

import pytest
import numpy as np
from src.pde_solvers.black_scholes import BlackScholesPDE


class TestBlackScholesPDE:
    """Test suite for Black-Scholes PDE solver."""

    @pytest.fixture
    def pde(self):
        """Create standard PDE instance for testing."""
        return BlackScholesPDE(
            S_max=300.0,
            T=1.0,
            r=0.05,
            sigma=0.2,
            N_S=100,
            N_t=1000
        )

    def test_initialization(self, pde):
        """Test PDE instance initialization."""
        assert pde.S_max == 300.0
        assert pde.T == 1.0
        assert pde.r == 0.05
        assert pde.sigma == 0.2
        assert pde.N_S == 100
        assert pde.N_t == 1000

        # Check grid properties
        assert len(pde.S_grid) == 101  # N_S + 1
        assert len(pde.t_grid) == 1001  # N_t + 1
        assert pde.S_grid[0] == 0
        assert pde.S_grid[-1] == 300.0
        assert pde.t_grid[0] == 0
        assert pde.t_grid[-1] == 1.0

    def test_european_call_payoff(self, pde):
        """Test European call payoff function."""
        K = 100
        payoff = pde.european_call_payoff(K)

        assert len(payoff) == len(pde.S_grid)
        assert payoff[0] == 0  # At S=0, call worthless
        assert payoff[-1] == pde.S_max - K  # At S=S_max, call worth S-K

        # Test at-the-money
        K_idx = np.argmin(np.abs(pde.S_grid - K))
        assert payoff[K_idx] == pytest.approx(0, abs=pde.dS)

        # Test in-the-money
        S_itm = 150
        S_itm_idx = np.argmin(np.abs(pde.S_grid - S_itm))
        assert payoff[S_itm_idx] == pytest.approx(50, abs=pde.dS)

    def test_european_put_payoff(self, pde):
        """Test European put payoff function."""
        K = 100
        payoff = pde.european_put_payoff(K)

        assert len(payoff) == len(pde.S_grid)
        assert payoff[0] == K  # At S=0, put worth K
        assert payoff[-1] == 0  # At S=S_max, put worthless

        # Test at-the-money
        K_idx = np.argmin(np.abs(pde.S_grid - K))
        assert payoff[K_idx] == pytest.approx(0, abs=pde.dS)

        # Test in-the-money
        S_itm = 50
        S_itm_idx = np.argmin(np.abs(pde.S_grid - S_itm))
        assert payoff[S_itm_idx] == pytest.approx(50, abs=pde.dS)

    def test_barrier_call_payoff(self, pde):
        """Test barrier call payoff function."""
        K = 100
        H = 150  # Barrier level

        # Up-and-out
        payoff_uao = pde.barrier_call_payoff(K, H, "up_and_out")
        H_idx = np.argmin(np.abs(pde.S_grid - H))
        assert payoff_uao[H_idx] == 0  # At barrier, option knocked out
        assert payoff_uao[0] == 0  # At S=0, call worthless

        # Down-and-out
        H_low = 50
        payoff_dao = pde.barrier_call_payoff(K, H_low, "down_and_out")
        H_low_idx = np.argmin(np.abs(pde.S_grid - H_low))
        assert payoff_dao[H_low_idx] == 0  # At barrier, knocked out

    def test_analytical_bs_call(self, pde):
        """Test analytical Black-Scholes call formula."""
        S0 = 100
        K = 100
        t = 0

        analytical_price = pde.get_analytical_bs_call(S0, K, t)

        # At-the-money call with 1 year to maturity should be around $10
        assert 8 < analytical_price < 12

        # Deep in-the-money
        deep_itm_price = pde.get_analytical_bs_call(150, K, t)
        assert deep_itm_price > 45  # Should be close to 50

        # Deep out-of-the-money
        deep_otm_price = pde.get_analytical_bs_call(50, K, t)
        assert deep_otm_price < 1  # Should be close to 0

    def test_analytical_bs_put(self, pde):
        """Test analytical Black-Scholes put formula."""
        S0 = 100
        K = 100
        t = 0

        analytical_price = pde.get_analytical_bs_put(S0, K, t)

        # At-the-money put with 1 year to maturity
        assert 5 < analytical_price < 10

        # Deep in-the-money put
        deep_itm_price = pde.get_analytical_bs_put(50, K, t)
        assert deep_itm_price > 45  # Should be close to 50

        # Deep out-of-the-money put
        deep_otm_price = pde.get_analytical_bs_put(150, K, t)
        assert deep_otm_price < 1

    def test_put_call_parity(self, pde):
        """Test put-call parity relationship."""
        S0 = 100
        K = 100
        t = 0

        call_price = pde.get_analytical_bs_call(S0, K, t)
        put_price = pde.get_analytical_bs_put(S0, K, t)

        # Put-call parity: C - P = S0 - K*exp(-r*T)
        parity_lhs = call_price - put_price
        parity_rhs = S0 - K * np.exp(-pde.r * (pde.T - t))

        assert parity_lhs == pytest.approx(parity_rhs, rel=1e-10)

    def test_delta_calculation(self, pde):
        """Test Delta calculation."""
        # Set up a simple option value grid
        K = 100
        pde.V[:, 0] = pde.european_call_payoff(K)

        delta = pde.calculate_delta(0)

        assert len(delta) == len(pde.S_grid)
        # Delta should be 0 at S=0 and 1 at high S for calls
        assert delta[0] == pytest.approx(0, abs=0.1)
        assert delta[-1] == pytest.approx(1, abs=0.1)

        # Delta at-the-money should be around 0.5 for calls
        K_idx = np.argmin(np.abs(pde.S_grid - K))
        # This is for payoff at maturity, so delta is discontinuous
        # Just check it's between 0 and 1
        assert 0 <= delta[K_idx] <= 1

    def test_gamma_calculation(self, pde):
        """Test Gamma calculation."""
        K = 100
        pde.V[:, 0] = pde.european_call_payoff(K)

        gamma = pde.calculate_gamma(0)

        assert len(gamma) == len(pde.S_grid)
        # Gamma should be non-negative
        assert np.all(gamma >= -0.1)  # Allow small numerical errors

        # Gamma should be highest near strike
        K_idx = np.argmin(np.abs(pde.S_grid - K))
        # For discontinuous payoff, gamma will have a spike at strike
        assert gamma[K_idx] >= 0

    def test_theta_calculation(self, pde):
        """Test Theta calculation."""
        K = 100
        S_idx = 50

        # Set up a simple grid
        for t_idx in range(pde.N_t + 1):
            pde.V[:, t_idx] = pde.european_call_payoff(K) * (pde.N_t - t_idx) / pde.N_t

        theta = pde.calculate_theta(S_idx)

        assert len(theta) == pde.N_t + 1
        # Theta should generally be negative for long options
        # (except at t=0 where calculation is undefined)
        assert np.all(theta[1:] <= 0.1)  # Allow small positive due to numerical errors


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_volatility(self):
        """Test with zero volatility."""
        pde = BlackScholesPDE(S_max=200.0, T=1.0, r=0.05, sigma=0.0, N_S=50, N_t=100)
        # Should not raise an error
        assert pde.sigma == 0.0

    def test_zero_interest_rate(self):
        """Test with zero interest rate."""
        pde = BlackScholesPDE(S_max=200.0, T=1.0, r=0.0, sigma=0.2, N_S=50, N_t=100)
        assert pde.r == 0.0

        # Analytical formulas should still work
        price = pde.get_analytical_bs_call(100, 100, 0)
        assert price > 0

    def test_small_grid(self):
        """Test with very small grid."""
        pde = BlackScholesPDE(S_max=100.0, T=1.0, r=0.05, sigma=0.2, N_S=10, N_t=10)
        assert pde.N_S == 10
        assert pde.N_t == 10
        assert pde.V.shape == (11, 11)

    def test_large_strike(self):
        """Test with strike much larger than S_max."""
        pde = BlackScholesPDE(S_max=100.0, T=1.0, r=0.05, sigma=0.2, N_S=50, N_t=100)
        K = 500  # Strike >> S_max

        payoff = pde.european_call_payoff(K)
        # All payoffs should be 0 since S_max < K
        assert np.all(payoff <= 0.1)  # Allow small numerical error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])