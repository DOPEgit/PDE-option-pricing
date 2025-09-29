"""
Base class for PDE solvers using finite difference methods.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Tuple


class PDESolverBase(ABC):
    """Abstract base class for all PDE solvers."""

    def __init__(self, pde_instance):
        """
        Initialize solver with a PDE instance.

        Parameters:
        -----------
        pde_instance : BlackScholesPDE
            Instance of BlackScholesPDE class
        """
        self.pde = pde_instance

    @abstractmethod
    def solve(self, payoff: np.ndarray, boundary_func: Callable, **kwargs):
        """
        Solve the PDE with given payoff and boundary conditions.

        Parameters:
        -----------
        payoff : np.ndarray
            Terminal payoff at maturity
        boundary_func : callable
            Function to apply boundary conditions at each time step
        **kwargs : dict
            Additional solver-specific parameters
        """
        pass

    def check_stability(self) -> Tuple[bool, float]:
        """
        Check numerical stability condition.

        Returns:
        --------
        is_stable : bool
            Whether the scheme is stable
        stability_param : float
            Stability parameter value
        """
        pass

    def compute_error(self, analytical_func: Callable, S0: float, K: float) -> float:
        """
        Compute error against analytical solution.

        Parameters:
        -----------
        analytical_func : callable
            Analytical pricing function
        S0 : float
            Initial stock price
        K : float
            Strike price

        Returns:
        --------
        error : float
            Absolute error
        """
        numerical_price = self.pde.get_option_value(S0, 0)
        analytical_price = analytical_func(S0, K, 0)
        return abs(numerical_price - analytical_price)