"""
Black-Scholes PDE Implementation for Option Pricing

The Black-Scholes PDE:
∂V/∂t + rS∂V/∂S + (σ²/2)S²∂²V/∂S² - rV = 0

Where:
- V(S,t) = option value
- S = stock price
- t = time
- r = risk-free rate
- σ = volatility
"""

import numpy as np
from typing import Callable, Tuple, Optional


class BlackScholesPDE:
    """Black-Scholes PDE for European and Exotic options."""

    def __init__(
        self,
        S_max: float = 300.0,
        T: float = 1.0,
        r: float = 0.05,
        sigma: float = 0.2,
        N_S: int = 100,
        N_t: int = 1000
    ):
        """
        Initialize Black-Scholes PDE solver.

        Parameters:
        -----------
        S_max : float
            Maximum stock price in grid
        T : float
            Time to maturity (years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        N_S : int
            Number of stock price grid points
        N_t : int
            Number of time steps
        """
        self.S_max = S_max
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N_S = N_S
        self.N_t = N_t

        # Grid setup
        self.dS = S_max / N_S
        self.dt = T / N_t
        self.S_grid = np.linspace(0, S_max, N_S + 1)
        self.t_grid = np.linspace(0, T, N_t + 1)

        # Solution grid
        self.V = np.zeros((N_S + 1, N_t + 1))

    def european_call_payoff(self, K: float) -> np.ndarray:
        """
        Payoff for European call option at maturity.

        Parameters:
        -----------
        K : float
            Strike price

        Returns:
        --------
        payoff : np.ndarray
            Payoff values at each stock price
        """
        return np.maximum(self.S_grid - K, 0)

    def european_put_payoff(self, K: float) -> np.ndarray:
        """
        Payoff for European put option at maturity.

        Parameters:
        -----------
        K : float
            Strike price

        Returns:
        --------
        payoff : np.ndarray
            Payoff values at each stock price
        """
        return np.maximum(K - self.S_grid, 0)

    def barrier_call_payoff(self, K: float, H: float, barrier_type: str = "up_and_out") -> np.ndarray:
        """
        Payoff for barrier call option.

        Parameters:
        -----------
        K : float
            Strike price
        H : float
            Barrier level
        barrier_type : str
            Type of barrier: "up_and_out", "down_and_out", "up_and_in", "down_and_in"

        Returns:
        --------
        payoff : np.ndarray
            Payoff values at each stock price
        """
        payoff = np.maximum(self.S_grid - K, 0)

        if barrier_type == "up_and_out":
            payoff[self.S_grid >= H] = 0
        elif barrier_type == "down_and_out":
            payoff[self.S_grid <= H] = 0
        elif barrier_type == "up_and_in":
            payoff[self.S_grid < H] = 0
        elif barrier_type == "down_and_in":
            payoff[self.S_grid > H] = 0

        return payoff

    def apply_boundary_conditions_call(self, K: float, t_idx: int):
        """
        Apply boundary conditions for European call option.

        Parameters:
        -----------
        K : float
            Strike price
        t_idx : int
            Time index
        """
        # At S=0, call option is worthless
        self.V[0, t_idx] = 0

        # At S=S_max, call option behaves like stock - K*exp(-r*(T-t))
        t = self.t_grid[t_idx]
        self.V[-1, t_idx] = self.S_max - K * np.exp(-self.r * (self.T - t))

    def apply_boundary_conditions_put(self, K: float, t_idx: int):
        """
        Apply boundary conditions for European put option.

        Parameters:
        -----------
        K : float
            Strike price
        t_idx : int
            Time index
        """
        # At S=0, put option is worth K*exp(-r*(T-t))
        t = self.t_grid[t_idx]
        self.V[0, t_idx] = K * np.exp(-self.r * (self.T - t))

        # At S=S_max, put option is worthless
        self.V[-1, t_idx] = 0

    def apply_barrier_conditions(self, H: float, barrier_type: str, t_idx: int):
        """
        Apply barrier conditions during time evolution.

        Parameters:
        -----------
        H : float
            Barrier level
        barrier_type : str
            Type of barrier
        t_idx : int
            Time index
        """
        if "out" in barrier_type:
            # Knock-out: set value to 0 at barrier
            barrier_idx = np.argmin(np.abs(self.S_grid - H))
            if "up" in barrier_type:
                self.V[barrier_idx:, t_idx] = 0
            else:  # down
                self.V[:barrier_idx+1, t_idx] = 0

    def get_analytical_bs_call(self, S0: float, K: float, t: float) -> float:
        """
        Analytical Black-Scholes formula for European call.

        Parameters:
        -----------
        S0 : float
            Current stock price
        K : float
            Strike price
        t : float
            Current time

        Returns:
        --------
        call_price : float
            Theoretical call option price
        """
        from scipy.stats import norm

        tau = self.T - t
        if tau <= 0:
            return max(S0 - K, 0)

        d1 = (np.log(S0/K) + (self.r + 0.5*self.sigma**2)*tau) / (self.sigma*np.sqrt(tau))
        d2 = d1 - self.sigma*np.sqrt(tau)

        call_price = S0*norm.cdf(d1) - K*np.exp(-self.r*tau)*norm.cdf(d2)
        return call_price

    def get_analytical_bs_put(self, S0: float, K: float, t: float) -> float:
        """
        Analytical Black-Scholes formula for European put.

        Parameters:
        -----------
        S0 : float
            Current stock price
        K : float
            Strike price
        t : float
            Current time

        Returns:
        --------
        put_price : float
            Theoretical put option price
        """
        from scipy.stats import norm

        tau = self.T - t
        if tau <= 0:
            return max(K - S0, 0)

        d1 = (np.log(S0/K) + (self.r + 0.5*self.sigma**2)*tau) / (self.sigma*np.sqrt(tau))
        d2 = d1 - self.sigma*np.sqrt(tau)

        put_price = K*np.exp(-self.r*tau)*norm.cdf(-d2) - S0*norm.cdf(-d1)
        return put_price

    def get_option_value(self, S: float, t: float) -> float:
        """
        Interpolate option value at any (S, t).

        Parameters:
        -----------
        S : float
            Stock price
        t : float
            Time

        Returns:
        --------
        value : float
            Interpolated option value
        """
        # Find closest indices
        S_idx = np.argmin(np.abs(self.S_grid - S))
        t_idx = np.argmin(np.abs(self.t_grid - t))

        return self.V[S_idx, t_idx]

    def calculate_delta(self, t_idx: int) -> np.ndarray:
        """
        Calculate Delta (∂V/∂S) using central differences.

        Parameters:
        -----------
        t_idx : int
            Time index

        Returns:
        --------
        delta : np.ndarray
            Delta values at each stock price
        """
        delta = np.zeros_like(self.S_grid)

        # Central difference for interior points
        delta[1:-1] = (self.V[2:, t_idx] - self.V[:-2, t_idx]) / (2 * self.dS)

        # Forward/backward difference for boundaries
        delta[0] = (self.V[1, t_idx] - self.V[0, t_idx]) / self.dS
        delta[-1] = (self.V[-1, t_idx] - self.V[-2, t_idx]) / self.dS

        return delta

    def calculate_gamma(self, t_idx: int) -> np.ndarray:
        """
        Calculate Gamma (∂²V/∂S²) using central differences.

        Parameters:
        -----------
        t_idx : int
            Time index

        Returns:
        --------
        gamma : np.ndarray
            Gamma values at each stock price
        """
        gamma = np.zeros_like(self.S_grid)

        # Central difference for interior points
        gamma[1:-1] = (self.V[2:, t_idx] - 2*self.V[1:-1, t_idx] + self.V[:-2, t_idx]) / (self.dS**2)

        return gamma

    def calculate_theta(self, S_idx: int) -> np.ndarray:
        """
        Calculate Theta (∂V/∂t) using backward differences.

        Parameters:
        -----------
        S_idx : int
            Stock price index

        Returns:
        --------
        theta : np.ndarray
            Theta values at each time
        """
        theta = np.zeros(self.N_t + 1)

        # Backward difference (negative because option value decreases with time)
        theta[1:] = (self.V[S_idx, 1:] - self.V[S_idx, :-1]) / self.dt

        return theta