"""
Implicit Finite Difference Method (Backward Time, Centered Space - BTCS)

Scheme: -αV(i-1,n) + (1+2α+rΔt)V(i,n) - αV(i+1,n) - βV(i+1,n) + βV(i-1,n) = V(i,n+1)

Solves tridiagonal system AV^n = V^{n+1} at each time step.

Advantages:
- Unconditionally stable (can use larger Δt)
- More accurate for large time steps

Disadvantages:
- Requires solving linear system (more computational cost per step)
"""

import numpy as np
from typing import Callable, Tuple
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from .solver_base import PDESolverBase


class ImplicitFD(PDESolverBase):
    """Implicit Finite Difference solver for Black-Scholes PDE."""

    def __init__(self, pde_instance):
        """
        Initialize implicit FD solver.

        Parameters:
        -----------
        pde_instance : BlackScholesPDE
            Instance of BlackScholesPDE class
        """
        super().__init__(pde_instance)
        self.name = "Implicit FD"

    def check_stability(self) -> Tuple[bool, str]:
        """
        Implicit method is unconditionally stable.

        Returns:
        --------
        is_stable : bool
            Always True for implicit method
        message : str
            Stability message
        """
        return True, "Unconditionally stable"

    def build_coefficient_matrix(self) -> np.ndarray:
        """
        Build tridiagonal coefficient matrix for implicit system.

        Returns:
        --------
        A : np.ndarray
            Tridiagonal coefficient matrix
        """
        N = self.pde.N_S - 1  # Interior points
        A = np.zeros((N, N))

        for i in range(1, N + 1):
            S = self.pde.S_grid[i]

            # Coefficients
            alpha = 0.5 * self.pde.sigma**2 * S**2 * self.pde.dt / self.pde.dS**2
            beta = 0.5 * self.pde.r * S * self.pde.dt / self.pde.dS

            # Matrix row index (0-indexed)
            idx = i - 1

            # Diagonal elements
            if idx > 0:
                A[idx, idx-1] = -alpha + beta  # Lower diagonal
            A[idx, idx] = 1 + 2*alpha + self.pde.r*self.pde.dt  # Main diagonal
            if idx < N - 1:
                A[idx, idx+1] = -alpha - beta  # Upper diagonal

        return A

    def solve(
        self,
        payoff: np.ndarray,
        boundary_func: Callable,
        barrier_func: Callable = None,
        use_sparse: bool = True,
        **kwargs
    ):
        """
        Solve using implicit finite difference method.

        Parameters:
        -----------
        payoff : np.ndarray
            Terminal payoff at maturity
        boundary_func : callable
            Function to apply boundary conditions
        barrier_func : callable, optional
            Function to apply barrier conditions
        use_sparse : bool
            Whether to use sparse matrix solver (faster for large grids)
        """
        # Initialize with payoff
        self.pde.V[:, -1] = payoff

        # Build coefficient matrix
        if use_sparse:
            A = self.build_sparse_matrix()
        else:
            A = self.build_coefficient_matrix()

        # Time stepping (backward in time)
        for n in range(self.pde.N_t - 1, -1, -1):
            # Right-hand side (known values from previous time step)
            b = self.pde.V[1:-1, n+1].copy()

            # Modify RHS for boundary conditions
            # Add boundary contributions
            S_0_contrib = self.pde.V[0, n]
            S_max_contrib = self.pde.V[-1, n]

            # Apply boundary conditions first
            boundary_func(n)

            # Adjust RHS for boundary values
            S = self.pde.S_grid[1]
            alpha_0 = 0.5 * self.pde.sigma**2 * S**2 * self.pde.dt / self.pde.dS**2
            beta_0 = 0.5 * self.pde.r * S * self.pde.dt / self.pde.dS
            b[0] += (alpha_0 - beta_0) * self.pde.V[0, n]

            S = self.pde.S_grid[-2]
            alpha_N = 0.5 * self.pde.sigma**2 * S**2 * self.pde.dt / self.pde.dS**2
            beta_N = 0.5 * self.pde.r * S * self.pde.dt / self.pde.dS
            b[-1] += (alpha_N + beta_N) * self.pde.V[-1, n]

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

    def build_sparse_matrix(self):
        """
        Build sparse tridiagonal matrix for efficient solving.

        Returns:
        --------
        A : scipy.sparse matrix
            Sparse tridiagonal coefficient matrix
        """
        N = self.pde.N_S - 1

        # Arrays for diagonals
        main_diag = np.zeros(N)
        upper_diag = np.zeros(N-1)
        lower_diag = np.zeros(N-1)

        for i in range(1, N + 1):
            S = self.pde.S_grid[i]

            alpha = 0.5 * self.pde.sigma**2 * S**2 * self.pde.dt / self.pde.dS**2
            beta = 0.5 * self.pde.r * S * self.pde.dt / self.pde.dS

            idx = i - 1

            main_diag[idx] = 1 + 2*alpha + self.pde.r*self.pde.dt

            if idx > 0:
                lower_diag[idx-1] = -alpha + beta
            if idx < N - 1:
                upper_diag[idx] = -alpha - beta

        # Create sparse matrix
        A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')

        return A

    def thomas_algorithm(self, a, b, c, d):
        """
        Thomas algorithm for solving tridiagonal systems.
        Faster than general LU decomposition.

        Parameters:
        -----------
        a : array
            Lower diagonal
        b : array
            Main diagonal
        c : array
            Upper diagonal
        d : array
            Right-hand side

        Returns:
        --------
        x : array
            Solution vector
        """
        n = len(d)
        c_star = np.zeros(n-1)
        d_star = np.zeros(n)
        x = np.zeros(n)

        # Forward sweep
        c_star[0] = c[0] / b[0]
        d_star[0] = d[0] / b[0]

        for i in range(1, n-1):
            denom = b[i] - a[i-1] * c_star[i-1]
            c_star[i] = c[i] / denom
            d_star[i] = (d[i] - a[i-1] * d_star[i-1]) / denom

        d_star[n-1] = (d[n-1] - a[n-2] * d_star[n-2]) / (b[n-1] - a[n-2] * c_star[n-2])

        # Backward substitution
        x[n-1] = d_star[n-1]
        for i in range(n-2, -1, -1):
            x[i] = d_star[i] - c_star[i] * x[i+1]

        return x