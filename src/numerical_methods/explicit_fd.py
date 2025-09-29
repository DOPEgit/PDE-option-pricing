"""
Explicit Finite Difference Method (Forward Time, Centered Space - FTCS)

Scheme: V(i, n+1) = V(i, n) + α[V(i+1,n) - 2V(i,n) + V(i-1,n)] + β[V(i+1,n) - V(i-1,n)]

where:
- α = σ²S²Δt/(2ΔS²)
- β = rSΔt/(2ΔS)

Stability condition: Δt ≤ ΔS²/(σ²S_max²)
"""

import numpy as np
from typing import Callable, Tuple
from .solver_base import PDESolverBase


class ExplicitFD(PDESolverBase):
    """Explicit Finite Difference solver for Black-Scholes PDE."""

    def __init__(self, pde_instance):
        """
        Initialize explicit FD solver.

        Parameters:
        -----------
        pde_instance : BlackScholesPDE
            Instance of BlackScholesPDE class
        """
        super().__init__(pde_instance)
        self.name = "Explicit FD"

    def check_stability(self) -> Tuple[bool, float]:
        """
        Check stability condition: Δt ≤ ΔS²/(σ²S_max²)

        Returns:
        --------
        is_stable : bool
            Whether the scheme is stable
        stability_param : float
            Actual value of stability parameter
        """
        S_max = self.pde.S_max
        dS = self.pde.dS
        dt = self.pde.dt
        sigma = self.pde.sigma

        # Stability parameter
        alpha_max = 0.5 * sigma**2 * S_max**2 * dt / dS**2

        # For stability, we need alpha_max ≤ 0.5
        is_stable = alpha_max <= 0.5

        return is_stable, alpha_max

    def solve(
        self,
        payoff: np.ndarray,
        boundary_func: Callable,
        barrier_func: Callable = None,
        **kwargs
    ):
        """
        Solve using explicit finite difference method.

        Parameters:
        -----------
        payoff : np.ndarray
            Terminal payoff at maturity
        boundary_func : callable
            Function to apply boundary conditions: boundary_func(t_idx)
        barrier_func : callable, optional
            Function to apply barrier conditions: barrier_func(t_idx)
        **kwargs : dict
            Additional parameters
        """
        # Check stability
        is_stable, alpha_max = self.check_stability()
        if not is_stable:
            print(f"Warning: Scheme may be unstable! alpha_max = {alpha_max:.4f} > 0.5")

        # Initialize with payoff at maturity
        self.pde.V[:, -1] = payoff

        # Time stepping (backward in time)
        for n in range(self.pde.N_t - 1, -1, -1):
            # Compute coefficients for each spatial point
            for i in range(1, self.pde.N_S):
                S = self.pde.S_grid[i]

                # Coefficients
                alpha = 0.5 * self.pde.sigma**2 * S**2 * self.pde.dt / self.pde.dS**2
                beta = 0.5 * self.pde.r * S * self.pde.dt / self.pde.dS
                discount = 1 - self.pde.r * self.pde.dt

                # Explicit update
                self.pde.V[i, n] = (
                    discount * self.pde.V[i, n+1]
                    + alpha * (self.pde.V[i+1, n+1] - 2*self.pde.V[i, n+1] + self.pde.V[i-1, n+1])
                    + beta * (self.pde.V[i+1, n+1] - self.pde.V[i-1, n+1])
                )

            # Apply boundary conditions
            boundary_func(n)

            # Apply barrier conditions if needed
            if barrier_func is not None:
                barrier_func(n)

        return self.pde.V

    def solve_vectorized(
        self,
        payoff: np.ndarray,
        boundary_func: Callable,
        barrier_func: Callable = None,
        **kwargs
    ):
        """
        Vectorized version for better performance.

        Parameters:
        -----------
        payoff : np.ndarray
            Terminal payoff at maturity
        boundary_func : callable
            Function to apply boundary conditions
        barrier_func : callable, optional
            Function to apply barrier conditions
        """
        # Initialize
        self.pde.V[:, -1] = payoff

        # Precompute coefficients
        S = self.pde.S_grid[1:-1]
        alpha = 0.5 * self.pde.sigma**2 * S**2 * self.pde.dt / self.pde.dS**2
        beta = 0.5 * self.pde.r * S * self.pde.dt / self.pde.dS
        discount = 1 - self.pde.r * self.pde.dt

        # Time stepping
        for n in range(self.pde.N_t - 1, -1, -1):
            # Vectorized update for interior points
            self.pde.V[1:-1, n] = (
                discount * self.pde.V[1:-1, n+1]
                + alpha * (self.pde.V[2:, n+1] - 2*self.pde.V[1:-1, n+1] + self.pde.V[:-2, n+1])
                + beta * (self.pde.V[2:, n+1] - self.pde.V[:-2, n+1])
            )

            boundary_func(n)

            if barrier_func is not None:
                barrier_func(n)

        return self.pde.V