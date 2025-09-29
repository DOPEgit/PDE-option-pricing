"""
Crank-Nicolson Method (Averaging Explicit and Implicit)

Scheme: Average of explicit and implicit schemes
- Second-order accurate in both time and space: O(Δt²) + O(ΔS²)
- Unconditionally stable
- Best accuracy-stability tradeoff

The scheme can be written as:
(I - θL)V^{n} = (I + (1-θ)L)V^{n+1}

where L is the spatial differential operator and θ = 0.5 for Crank-Nicolson.
"""

import numpy as np
from typing import Callable, Tuple
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from .solver_base import PDESolverBase


class CrankNicolson(PDESolverBase):
    """Crank-Nicolson solver for Black-Scholes PDE."""

    def __init__(self, pde_instance, theta: float = 0.5):
        """
        Initialize Crank-Nicolson solver.

        Parameters:
        -----------
        pde_instance : BlackScholesPDE
            Instance of BlackScholesPDE class
        theta : float
            Weighting parameter (0.5 for standard Crank-Nicolson)
            theta=0: explicit, theta=1: implicit, theta=0.5: Crank-Nicolson
        """
        super().__init__(pde_instance)
        self.theta = theta
        self.name = f"Crank-Nicolson (θ={theta})"

    def check_stability(self) -> Tuple[bool, str]:
        """
        Crank-Nicolson is unconditionally stable for θ >= 0.5.

        Returns:
        --------
        is_stable : bool
            True if theta >= 0.5
        message : str
            Stability message
        """
        if self.theta >= 0.5:
            return True, f"Unconditionally stable (θ={self.theta})"
        else:
            return False, f"Conditionally stable (θ={self.theta} < 0.5)"

    def build_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build coefficient matrices for Crank-Nicolson scheme.

        Returns:
        --------
        A : np.ndarray
            Left-hand side matrix (implicit part)
        B : np.ndarray
            Right-hand side matrix (explicit part)
        """
        N = self.pde.N_S - 1
        A = np.zeros((N, N))
        B = np.zeros((N, N))

        for i in range(1, N + 1):
            S = self.pde.S_grid[i]

            # Coefficients
            alpha = 0.5 * self.pde.sigma**2 * S**2 * self.pde.dt / self.pde.dS**2
            beta = 0.5 * self.pde.r * S * self.pde.dt / self.pde.dS

            idx = i - 1

            # Matrix A (implicit part - multiply by theta)
            if idx > 0:
                A[idx, idx-1] = -self.theta * (alpha - beta)
            A[idx, idx] = 1 + self.theta * (2*alpha + self.pde.r*self.pde.dt)
            if idx < N - 1:
                A[idx, idx+1] = -self.theta * (alpha + beta)

            # Matrix B (explicit part - multiply by (1-theta))
            if idx > 0:
                B[idx, idx-1] = (1-self.theta) * (alpha - beta)
            B[idx, idx] = 1 - (1-self.theta) * (2*alpha + self.pde.r*self.pde.dt)
            if idx < N - 1:
                B[idx, idx+1] = (1-self.theta) * (alpha + beta)

        return A, B

    def build_sparse_matrices(self):
        """
        Build sparse matrices for efficient computation.

        Returns:
        --------
        A : scipy.sparse matrix
            Sparse LHS matrix
        B : scipy.sparse matrix
            Sparse RHS matrix
        """
        N = self.pde.N_S - 1

        # Arrays for diagonals
        main_diag_A = np.zeros(N)
        upper_diag_A = np.zeros(N-1)
        lower_diag_A = np.zeros(N-1)

        main_diag_B = np.zeros(N)
        upper_diag_B = np.zeros(N-1)
        lower_diag_B = np.zeros(N-1)

        for i in range(1, N + 1):
            S = self.pde.S_grid[i]

            alpha = 0.5 * self.pde.sigma**2 * S**2 * self.pde.dt / self.pde.dS**2
            beta = 0.5 * self.pde.r * S * self.pde.dt / self.pde.dS

            idx = i - 1

            # Matrix A
            main_diag_A[idx] = 1 + self.theta * (2*alpha + self.pde.r*self.pde.dt)
            if idx > 0:
                lower_diag_A[idx-1] = -self.theta * (alpha - beta)
            if idx < N - 1:
                upper_diag_A[idx] = -self.theta * (alpha + beta)

            # Matrix B
            main_diag_B[idx] = 1 - (1-self.theta) * (2*alpha + self.pde.r*self.pde.dt)
            if idx > 0:
                lower_diag_B[idx-1] = (1-self.theta) * (alpha - beta)
            if idx < N - 1:
                upper_diag_B[idx] = (1-self.theta) * (alpha + beta)

        A = diags([lower_diag_A, main_diag_A, upper_diag_A], [-1, 0, 1], format='csr')
        B = diags([lower_diag_B, main_diag_B, upper_diag_B], [-1, 0, 1], format='csr')

        return A, B

    def solve(
        self,
        payoff: np.ndarray,
        boundary_func: Callable,
        barrier_func: Callable = None,
        use_sparse: bool = True,
        **kwargs
    ):
        """
        Solve using Crank-Nicolson method.

        Parameters:
        -----------
        payoff : np.ndarray
            Terminal payoff at maturity
        boundary_func : callable
            Function to apply boundary conditions
        barrier_func : callable, optional
            Function to apply barrier conditions
        use_sparse : bool
            Whether to use sparse matrix solver
        """
        # Initialize with payoff
        self.pde.V[:, -1] = payoff

        # Build matrices
        if use_sparse:
            A, B = self.build_sparse_matrices()
        else:
            A, B = self.build_matrices()

        # Time stepping (backward in time)
        for n in range(self.pde.N_t - 1, -1, -1):
            # Apply boundary conditions
            boundary_func(n)

            # Right-hand side
            if use_sparse:
                b = B.dot(self.pde.V[1:-1, n+1])
            else:
                b = B @ self.pde.V[1:-1, n+1]

            # Adjust for boundary conditions
            S = self.pde.S_grid[1]
            alpha_0 = 0.5 * self.pde.sigma**2 * S**2 * self.pde.dt / self.pde.dS**2
            beta_0 = 0.5 * self.pde.r * S * self.pde.dt / self.pde.dS
            b[0] += (alpha_0 - beta_0) * (self.theta * self.pde.V[0, n] +
                                          (1-self.theta) * self.pde.V[0, n+1])

            S = self.pde.S_grid[-2]
            alpha_N = 0.5 * self.pde.sigma**2 * S**2 * self.pde.dt / self.pde.dS**2
            beta_N = 0.5 * self.pde.r * S * self.pde.dt / self.pde.dS
            b[-1] += (alpha_N + beta_N) * (self.theta * self.pde.V[-1, n] +
                                           (1-self.theta) * self.pde.V[-1, n+1])

            # Solve linear system
            if use_sparse:
                self.pde.V[1:-1, n] = spsolve(A, b)
            else:
                self.pde.V[1:-1, n] = np.linalg.solve(A, b)

            # Reapply boundary conditions
            boundary_func(n)

            # Apply barrier conditions if needed
            if barrier_func is not None:
                barrier_func(n)

        return self.pde.V

    def get_convergence_order(self) -> Tuple[int, int]:
        """
        Return theoretical convergence order.

        Returns:
        --------
        time_order : int
            Order of convergence in time (2 for Crank-Nicolson)
        space_order : int
            Order of convergence in space (2)
        """
        if self.theta == 0.5:
            return 2, 2  # Second-order in both time and space
        else:
            return 1, 2  # First-order in time, second-order in space