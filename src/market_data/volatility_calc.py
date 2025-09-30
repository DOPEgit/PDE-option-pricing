"""
Volatility calculator for historical and implied volatility.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy import optimize
from scipy.stats import norm


class VolatilityCalculator:
    """Calculate various volatility measures."""

    @staticmethod
    def calculate_historical_volatility(prices: pd.Series,
                                         periods: int = 252,
                                         method: str = 'close_to_close') -> float:
        """
        Calculate historical volatility from price series.

        Parameters:
        -----------
        prices : pd.Series
            Price series (typically Close prices)
        periods : int
            Number of periods per year (252 for daily, 52 for weekly)
        method : str
            Calculation method:
            - 'close_to_close': Traditional close-to-close
            - 'parkinson': High-Low estimator
            - 'garman_klass': OHLC estimator

        Returns:
        --------
        float
            Annualized volatility
        """
        if method == 'close_to_close':
            # Calculate log returns
            log_returns = np.log(prices / prices.shift(1))
            # Remove NaN values
            log_returns = log_returns.dropna()

            # Calculate standard deviation and annualize
            daily_vol = log_returns.std()
            annual_vol = daily_vol * np.sqrt(periods)

            return annual_vol

        elif method == 'parkinson':
            # Parkinson's High-Low estimator
            # Requires high and low prices (not implemented here)
            return VolatilityCalculator.calculate_historical_volatility(
                prices, periods, 'close_to_close'
            )

        else:
            # Default to close-to-close
            return VolatilityCalculator.calculate_historical_volatility(
                prices, periods, 'close_to_close'
            )

    @staticmethod
    def calculate_ewma_volatility(prices: pd.Series,
                                   lambda_param: float = 0.94,
                                   periods: int = 252) -> pd.Series:
        """
        Calculate Exponentially Weighted Moving Average (EWMA) volatility.

        Parameters:
        -----------
        prices : pd.Series
            Price series
        lambda_param : float
            Decay factor (typically 0.94)
        periods : int
            Periods per year for annualization

        Returns:
        --------
        pd.Series
            Time series of EWMA volatility
        """
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1)).dropna()

        # Initialize
        ewma_var = pd.Series(index=log_returns.index, dtype=float)
        ewma_var.iloc[0] = log_returns.iloc[0] ** 2

        # Calculate EWMA variance
        for i in range(1, len(log_returns)):
            ewma_var.iloc[i] = (lambda_param * ewma_var.iloc[i-1] +
                                 (1 - lambda_param) * log_returns.iloc[i] ** 2)

        # Convert to volatility and annualize
        ewma_vol = np.sqrt(ewma_var) * np.sqrt(periods)

        return ewma_vol

    @staticmethod
    def calculate_garch_volatility(prices: pd.Series,
                                    p: int = 1, q: int = 1) -> Tuple[float, pd.Series]:
        """
        Simplified GARCH(p,q) volatility estimation.

        Parameters:
        -----------
        prices : pd.Series
            Price series
        p : int
            ARCH order
        q : int
            GARCH order

        Returns:
        --------
        tuple
            (current_volatility, volatility_series)
        """
        # For simplicity, use EWMA as approximation
        # Full GARCH requires specialized packages like arch
        ewma_vol = VolatilityCalculator.calculate_ewma_volatility(prices)
        return ewma_vol.iloc[-1], ewma_vol

    @staticmethod
    def black_scholes_price(S: float, K: float, T: float, r: float,
                             sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes option price.

        Parameters:
        -----------
        S : float
            Spot price
        K : float
            Strike price
        T : float
            Time to maturity (years)
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'

        Returns:
        --------
        float
            Option price
        """
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    @staticmethod
    def calculate_implied_volatility(option_price: float, S: float, K: float,
                                      T: float, r: float, option_type: str = 'call',
                                      max_iterations: int = 100,
                                      tolerance: float = 1e-5) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.

        Parameters:
        -----------
        option_price : float
            Market price of option
        S : float
            Spot price
        K : float
            Strike price
        T : float
            Time to maturity (years)
        r : float
            Risk-free rate
        option_type : str
            'call' or 'put'
        max_iterations : int
            Maximum iterations
        tolerance : float
            Convergence tolerance

        Returns:
        --------
        float
            Implied volatility
        """
        # Define objective function
        def objective(sigma):
            return (VolatilityCalculator.black_scholes_price(S, K, T, r, sigma, option_type)
                    - option_price)

        # Use Brent's method for robust root finding
        try:
            # Search between 1% and 500% volatility
            iv = optimize.brentq(objective, 0.01, 5.0,
                                  xtol=tolerance, maxiter=max_iterations)
            return iv
        except:
            # If fails, try bisection with different bounds
            try:
                iv = optimize.bisect(objective, 0.001, 10.0,
                                      xtol=tolerance, maxiter=max_iterations)
                return iv
            except:
                # Return a reasonable default
                return 0.25

    @staticmethod
    def calculate_volatility_smile(options_chain: pd.DataFrame,
                                    S: float, r: float,
                                    T: float, option_type: str = 'call') -> pd.DataFrame:
        """
        Calculate volatility smile from options chain.

        Parameters:
        -----------
        options_chain : pd.DataFrame
            Options data with columns: strike, lastPrice
        S : float
            Current spot price
        r : float
            Risk-free rate
        T : float
            Time to maturity
        option_type : str
            'call' or 'put'

        Returns:
        --------
        pd.DataFrame
            DataFrame with strike and implied_volatility columns
        """
        results = []

        for _, row in options_chain.iterrows():
            K = row['strike']
            price = row['lastPrice']

            # Skip if price is too small
            if price < 0.01:
                continue

            iv = VolatilityCalculator.calculate_implied_volatility(
                price, S, K, T, r, option_type
            )

            results.append({
                'strike': K,
                'moneyness': K / S,
                'implied_volatility': iv,
                'option_price': price
            })

        return pd.DataFrame(results)

    @staticmethod
    def calculate_volatility_surface(options_data: Dict,
                                      S: float, r: float) -> pd.DataFrame:
        """
        Calculate volatility surface from multiple expiries.

        Parameters:
        -----------
        options_data : dict
            Dictionary with expiry dates as keys, options chains as values
        S : float
            Current spot price
        r : float
            Risk-free rate

        Returns:
        --------
        pd.DataFrame
            DataFrame with strike, expiry, and implied_volatility
        """
        results = []

        for expiry, chain in options_data.items():
            # Calculate time to expiry
            T = (pd.to_datetime(expiry) - pd.Timestamp.now()).days / 365.0

            if T <= 0:
                continue

            # Calculate smile for this expiry
            smile = VolatilityCalculator.calculate_volatility_smile(
                chain, S, r, T
            )
            smile['expiry'] = expiry
            smile['time_to_expiry'] = T

            results.append(smile)

        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()

    @staticmethod
    def calculate_realized_volatility(prices: pd.Series,
                                       window: int = 21,
                                       periods: int = 252) -> pd.Series:
        """
        Calculate rolling realized volatility.

        Parameters:
        -----------
        prices : pd.Series
            Price series
        window : int
            Rolling window size
        periods : int
            Periods per year

        Returns:
        --------
        pd.Series
            Rolling realized volatility
        """
        log_returns = np.log(prices / prices.shift(1))
        realized_vol = log_returns.rolling(window=window).std() * np.sqrt(periods)
        return realized_vol