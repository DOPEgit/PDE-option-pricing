"""
Base class for market data fetchers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


class DataFetcherBase(ABC):
    """Abstract base class for all market data fetchers."""

    def __init__(self, cache_duration: int = 60):
        """
        Initialize data fetcher with caching.

        Parameters:
        -----------
        cache_duration : int
            Cache duration in seconds (default: 60)
        """
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_duration = cache_duration

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self.cache_timestamps:
            return False

        elapsed = (datetime.now() - self.cache_timestamps[key]).total_seconds()
        return elapsed < self.cache_duration

    def _get_from_cache(self, key: str) -> Optional[any]:
        """Get data from cache if valid."""
        if self._is_cache_valid(key):
            return self.cache.get(key)
        return None

    def _save_to_cache(self, key: str, data: any):
        """Save data to cache."""
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now()

    @abstractmethod
    def get_spot_price(self, ticker: str) -> float:
        """
        Get current spot price for a ticker.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol

        Returns:
        --------
        float
            Current spot price
        """
        pass

    @abstractmethod
    def get_historical_prices(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Get historical price data.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        period : str
            Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
        --------
        pd.DataFrame
            Historical price data with columns: Date, Open, High, Low, Close, Volume
        """
        pass

    @abstractmethod
    def get_options_chain(self, ticker: str, expiry: Optional[str] = None) -> Dict:
        """
        Get options chain data.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        expiry : str, optional
            Expiration date (YYYY-MM-DD format)

        Returns:
        --------
        dict
            Options chain with 'calls' and 'puts' DataFrames
        """
        pass

    def get_implied_volatility(self, ticker: str, strike: float,
                               expiry: str, option_type: str = 'call') -> float:
        """
        Get implied volatility from market prices.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        strike : float
            Strike price
        expiry : str
            Expiration date
        option_type : str
            'call' or 'put'

        Returns:
        --------
        float
            Implied volatility
        """
        # This would typically use option prices and back out IV
        # For now, return a placeholder
        return 0.25

    def get_dividend_yield(self, ticker: str) -> float:
        """
        Get dividend yield for a ticker.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol

        Returns:
        --------
        float
            Annual dividend yield as decimal
        """
        return 0.0  # Default to no dividends