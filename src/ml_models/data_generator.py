"""
Generate training data from PDE solver for ML surrogate models.

This module creates large datasets of option prices and Greeks under varying
market conditions to train machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import sys
sys.path.append('..')

from src.pde_solvers.black_scholes import BlackScholesPDE
from src.numerical_methods.crank_nicolson import CrankNicolson


class OptionDataGenerator:
    """Generate training data for ML surrogate models."""

    def __init__(self, method='crank_nicolson'):
        """
        Initialize data generator.

        Parameters:
        -----------
        method : str
            Numerical method to use ('crank_nicolson', 'implicit', 'explicit')
        """
        self.method = method

    def generate_dataset(
        self,
        n_samples: int = 10000,
        option_type: str = 'call',
        parameter_ranges: Dict = None,
        grid_size: Tuple[int, int] = (100, 1000),
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate dataset of option prices and Greeks.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        option_type : str
            'call' or 'put'
        parameter_ranges : dict
            Ranges for parameters:
            - 'S0': (min, max) for stock price
            - 'K': (min, max) for strike price
            - 'T': (min, max) for time to maturity
            - 'r': (min, max) for risk-free rate
            - 'sigma': (min, max) for volatility
        grid_size : tuple
            (N_S, N_t) for PDE solver
        random_seed : int
            Random seed for reproducibility

        Returns:
        --------
        X : pd.DataFrame
            Features (market parameters)
        y : pd.DataFrame
            Target (option price and Greeks)
        """
        np.random.seed(random_seed)

        # Default parameter ranges
        if parameter_ranges is None:
            parameter_ranges = {
                'S0': (50, 150),
                'K': (70, 130),
                'T': (0.1, 2.0),
                'r': (0.01, 0.10),
                'sigma': (0.10, 0.50)
            }

        # Generate random parameters
        print(f"Generating {n_samples} samples...")

        S0_values = np.random.uniform(*parameter_ranges['S0'], n_samples)
        K_values = np.random.uniform(*parameter_ranges['K'], n_samples)
        T_values = np.random.uniform(*parameter_ranges['T'], n_samples)
        r_values = np.random.uniform(*parameter_ranges['r'], n_samples)
        sigma_values = np.random.uniform(*parameter_ranges['sigma'], n_samples)

        # Storage for results
        prices = []
        deltas = []
        gammas = []
        thetas = []

        # Generate each sample
        for i in range(n_samples):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{n_samples}")

            S0 = S0_values[i]
            K = K_values[i]
            T = T_values[i]
            r = r_values[i]
            sigma = sigma_values[i]

            # Solve PDE
            price, delta, gamma, theta = self._solve_option(
                S0, K, T, r, sigma, option_type, grid_size
            )

            prices.append(price)
            deltas.append(delta)
            gammas.append(gamma)
            thetas.append(theta)

        # Create DataFrames
        X = pd.DataFrame({
            'S0': S0_values,
            'K': K_values,
            'T': T_values,
            'r': r_values,
            'sigma': sigma_values
        })

        # Add derived features
        X['moneyness'] = X['S0'] / X['K']
        X['log_moneyness'] = np.log(X['S0'] / X['K'])
        X['sqrt_T'] = np.sqrt(X['T'])
        X['vol_sqrt_T'] = X['sigma'] * np.sqrt(X['T'])

        y = pd.DataFrame({
            'price': prices,
            'delta': deltas,
            'gamma': gammas,
            'theta': thetas
        })

        print(f"Dataset generated: {n_samples} samples")
        print(f"Features shape: {X.shape}")
        print(f"Targets shape: {y.shape}")

        return X, y

    def _solve_option(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        grid_size: Tuple[int, int]
    ) -> Tuple[float, float, float, float]:
        """
        Solve single option using PDE solver.

        Returns:
        --------
        price : float
            Option price at (S0, t=0)
        delta : float
            Delta at (S0, t=0)
        gamma : float
            Gamma at (S0, t=0)
        theta : float
            Theta at (S0, t=0)
        """
        N_S, N_t = grid_size

        # Create PDE instance
        pde = BlackScholesPDE(
            S_max=max(S0 * 3, K * 3),
            T=T,
            r=r,
            sigma=sigma,
            N_S=N_S,
            N_t=N_t
        )

        # Set up payoff
        if option_type == 'call':
            payoff = pde.european_call_payoff(K)
            boundary_func = lambda t_idx: pde.apply_boundary_conditions_call(K, t_idx)
        else:
            payoff = pde.european_put_payoff(K)
            boundary_func = lambda t_idx: pde.apply_boundary_conditions_put(K, t_idx)

        # Solve
        solver = CrankNicolson(pde)
        solver.solve(payoff, boundary_func, use_sparse=True)

        # Get price and Greeks at S0, t=0
        S_idx = np.argmin(np.abs(pde.S_grid - S0))

        price = pde.V[S_idx, 0]
        delta_grid = pde.calculate_delta(0)
        delta = delta_grid[S_idx]
        gamma_grid = pde.calculate_gamma(0)
        gamma = gamma_grid[S_idx]
        theta_grid = pde.calculate_theta(S_idx)
        theta = theta_grid[0]

        return price, delta, gamma, theta

    def save_dataset(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        filepath_X: str,
        filepath_y: str
    ):
        """
        Save dataset to CSV files.

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.DataFrame
            Targets
        filepath_X : str
            Path to save features
        filepath_y : str
            Path to save targets
        """
        X.to_csv(filepath_X, index=False)
        y.to_csv(filepath_y, index=False)
        print(f"Saved features to: {filepath_X}")
        print(f"Saved targets to: {filepath_y}")

    def load_dataset(
        self,
        filepath_X: str,
        filepath_y: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load dataset from CSV files.

        Parameters:
        -----------
        filepath_X : str
            Path to features file
        filepath_y : str
            Path to targets file

        Returns:
        --------
        X : pd.DataFrame
            Features
        y : pd.DataFrame
            Targets
        """
        X = pd.read_csv(filepath_X)
        y = pd.read_csv(filepath_y)
        print(f"Loaded dataset: {X.shape[0]} samples")
        return X, y