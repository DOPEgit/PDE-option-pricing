"""
Utility functions for the Option Pricing Dashboard app
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


def format_price(value: float) -> str:
    """Format price for display."""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format percentage for display."""
    return f"{value*100:.2f}%"


def format_time(seconds: float) -> str:
    """Format time duration for display."""
    if seconds < 0.001:
        return f"{seconds*1000000:.2f} μs"
    elif seconds < 1:
        return f"{seconds*1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def calculate_moneyness(S: float, K: float) -> float:
    """Calculate moneyness of an option."""
    return S / K


def calculate_intrinsic_value(S: float, K: float, option_type: str) -> float:
    """Calculate intrinsic value of an option."""
    if option_type.lower() == 'call':
        return max(S - K, 0)
    else:
        return max(K - S, 0)


def calculate_time_value(option_price: float, S: float, K: float, option_type: str) -> float:
    """Calculate time value of an option."""
    intrinsic = calculate_intrinsic_value(S, K, option_type)
    return option_price - intrinsic


def generate_strike_range(S0: float, num_strikes: int = 11, width: float = 0.2) -> np.ndarray:
    """
    Generate a range of strike prices around the spot price.

    Parameters:
    -----------
    S0 : float
        Current spot price
    num_strikes : int
        Number of strike prices to generate
    width : float
        Width of the range as fraction of spot price

    Returns:
    --------
    np.ndarray
        Array of strike prices
    """
    K_min = S0 * (1 - width)
    K_max = S0 * (1 + width)
    return np.linspace(K_min, K_max, num_strikes)


def generate_expiry_dates(start_date: datetime = None, num_expiries: int = 6) -> List[datetime]:
    """
    Generate a list of expiry dates.

    Parameters:
    -----------
    start_date : datetime
        Starting date (default: today)
    num_expiries : int
        Number of expiry dates to generate

    Returns:
    --------
    List[datetime]
        List of expiry dates
    """
    if start_date is None:
        start_date = datetime.now()

    expiries = []
    for i in range(1, num_expiries + 1):
        # Monthly expiries
        expiry = start_date + timedelta(days=30*i)
        expiries.append(expiry)

    return expiries


def calculate_days_to_expiry(expiry_date: datetime, current_date: datetime = None) -> int:
    """Calculate number of days to expiry."""
    if current_date is None:
        current_date = datetime.now()
    return (expiry_date - current_date).days


def calculate_years_to_expiry(expiry_date: datetime, current_date: datetime = None) -> float:
    """Calculate time to expiry in years."""
    days = calculate_days_to_expiry(expiry_date, current_date)
    return days / 365.25


def create_options_chain(S0: float, strikes: np.ndarray, T: float, r: float, sigma: float) -> pd.DataFrame:
    """
    Create a simulated options chain.

    Parameters:
    -----------
    S0 : float
        Current spot price
    strikes : np.ndarray
        Array of strike prices
    T : float
        Time to expiry in years
    r : float
        Risk-free rate
    sigma : float
        Volatility

    Returns:
    --------
    pd.DataFrame
        Options chain with calls and puts
    """
    chain = []

    for K in strikes:
        moneyness = calculate_moneyness(S0, K)

        chain.append({
            'Strike': K,
            'Moneyness': moneyness,
            'Type': 'Call' if moneyness > 1 else 'Put'
        })

    return pd.DataFrame(chain)


def calculate_portfolio_stats(positions: List[Dict]) -> Dict:
    """
    Calculate portfolio-level statistics.

    Parameters:
    -----------
    positions : List[Dict]
        List of position dictionaries

    Returns:
    --------
    Dict
        Portfolio statistics
    """
    total_value = sum(p['quantity'] * p['price'] for p in positions)
    total_delta = sum(p['quantity'] * p['delta'] for p in positions)
    total_gamma = sum(p['quantity'] * p['gamma'] for p in positions)
    total_theta = sum(p['quantity'] * p['theta'] for p in positions)

    # Calculate weighted average IV if available
    if all('iv' in p for p in positions):
        weights = [p['quantity'] * p['price'] for p in positions]
        total_weight = sum(weights)
        weighted_iv = sum(w * p['iv'] for w, p in zip(weights, positions)) / total_weight
    else:
        weighted_iv = None

    return {
        'total_value': total_value,
        'total_delta': total_delta,
        'total_gamma': total_gamma,
        'total_theta': total_theta,
        'weighted_iv': weighted_iv,
        'num_positions': len(positions)
    }


def generate_scenario_matrix(S0: float, sigma: float, T: float, num_scenarios: int = 100) -> np.ndarray:
    """
    Generate price scenarios using Monte Carlo simulation.

    Parameters:
    -----------
    S0 : float
        Current spot price
    sigma : float
        Volatility
    T : float
        Time horizon
    num_scenarios : int
        Number of scenarios to generate

    Returns:
    --------
    np.ndarray
        Array of simulated prices
    """
    np.random.seed(42)  # For reproducibility
    z = np.random.randn(num_scenarios)
    ST = S0 * np.exp((- 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    return ST


def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR).

    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    confidence : float
        Confidence level

    Returns:
    --------
    float
        VaR value
    """
    return np.percentile(returns, (1 - confidence) * 100)


def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR).

    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    confidence : float
        Confidence level

    Returns:
    --------
    float
        CVaR value
    """
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()


def format_large_number(num: float) -> str:
    """Format large numbers with K, M, B suffixes."""
    if abs(num) < 1000:
        return f"{num:.2f}"
    elif abs(num) < 1000000:
        return f"{num/1000:.1f}K"
    elif abs(num) < 1000000000:
        return f"{num/1000000:.1f}M"
    else:
        return f"{num/1000000000:.1f}B"


def create_performance_metrics(pde_results: Dict, ml_results: Dict) -> pd.DataFrame:
    """
    Create a comparison table of PDE vs ML performance.

    Parameters:
    -----------
    pde_results : Dict
        Results from PDE solver
    ml_results : Dict
        Results from ML model

    Returns:
    --------
    pd.DataFrame
        Comparison metrics
    """
    metrics = {
        'Method': ['PDE Solver', 'ML Model'],
        'Price': [pde_results['price'], ml_results['price']],
        'Computation Time': [
            format_time(pde_results['time']),
            format_time(ml_results['time'])
        ],
        'Speedup': ['1x', f"{pde_results['time']/ml_results['time']:.0f}x"]
    }

    if 'delta' in pde_results:
        metrics['Delta'] = [pde_results.get('delta'), ml_results.get('delta')]

    return pd.DataFrame(metrics)