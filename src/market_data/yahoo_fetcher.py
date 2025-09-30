"""
Yahoo Finance data fetcher for real-time market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .data_fetcher import DataFetcherBase


class YahooDataFetcher(DataFetcherBase):
    """Fetches market data from Yahoo Finance."""

    def __init__(self, cache_duration: int = 60):
        """
        Initialize Yahoo Finance data fetcher.

        Parameters:
        -----------
        cache_duration : int
            Cache duration in seconds
        """
        super().__init__(cache_duration)
        self.yf = None  # Will be imported when needed

    def _ensure_yfinance(self):
        """Ensure yfinance is imported (lazy loading)."""
        if self.yf is None:
            try:
                import yfinance as yf
                self.yf = yf
            except ImportError:
                raise ImportError(
                    "yfinance not installed. Install with: pip install yfinance"
                )

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
        cache_key = f"spot_{ticker}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        self._ensure_yfinance()
        try:
            stock = self.yf.Ticker(ticker)
            info = stock.info

            # Try different price fields in order of preference
            price = info.get('currentPrice') or \
                   info.get('regularMarketPrice') or \
                   info.get('previousClose', 0)

            if price == 0:
                # Fallback: get latest close from history
                hist = stock.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]

            self._save_to_cache(cache_key, float(price))
            return float(price)

        except Exception as e:
            print(f"Error fetching spot price for {ticker}: {e}")
            # Return a default value for demo purposes
            return 100.0

    def get_historical_prices(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Get historical price data from Yahoo Finance.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        period : str
            Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
        --------
        pd.DataFrame
            Historical price data
        """
        cache_key = f"hist_{ticker}_{period}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        self._ensure_yfinance()
        try:
            stock = self.yf.Ticker(ticker)
            hist = stock.history(period=period)

            # Reset index to have Date as a column
            hist = hist.reset_index()

            self._save_to_cache(cache_key, hist)
            return hist

        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
            # Return synthetic data for demo
            dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
            prices = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.01))
            return pd.DataFrame({
                'Date': dates,
                'Close': prices,
                'Open': prices * (1 + np.random.randn(252) * 0.001),
                'High': prices * (1 + np.abs(np.random.randn(252) * 0.005)),
                'Low': prices * (1 - np.abs(np.random.randn(252) * 0.005)),
                'Volume': np.random.randint(1000000, 10000000, 252)
            })

    def get_options_chain(self, ticker: str, expiry: Optional[str] = None) -> Dict:
        """
        Get options chain data from Yahoo Finance.

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
        self._ensure_yfinance()
        try:
            stock = self.yf.Ticker(ticker)

            # Get available expiration dates
            expirations = stock.options

            if not expirations:
                print(f"No options available for {ticker}")
                return self._generate_synthetic_options(ticker)

            # Use specified expiry or nearest one
            if expiry:
                # Find closest expiration date
                target_date = pd.to_datetime(expiry)
                exp_dates = pd.to_datetime(expirations)
                idx = (exp_dates - target_date).abs().argmin()
                expiry_to_use = expirations[idx]
            else:
                # Use nearest expiration (first in list)
                expiry_to_use = expirations[0]

            # Get options chain
            opt_chain = stock.option_chain(expiry_to_use)

            return {
                'calls': opt_chain.calls,
                'puts': opt_chain.puts,
                'expiry': expiry_to_use,
                'spot': self.get_spot_price(ticker)
            }

        except Exception as e:
            print(f"Error fetching options chain for {ticker}: {e}")
            return self._generate_synthetic_options(ticker)

    def _generate_synthetic_options(self, ticker: str) -> Dict:
        """Generate synthetic options data for testing."""
        spot = self.get_spot_price(ticker)

        # Generate strikes around spot price
        strikes = np.arange(spot * 0.8, spot * 1.2, spot * 0.05)

        # Generate synthetic implied volatilities (smile effect)
        moneyness = strikes / spot
        ivs = 0.2 + 0.1 * (moneyness - 1.0) ** 2

        # Generate synthetic prices using simplified Black-Scholes approximation
        T = 30/365  # 30 days to expiry
        r = 0.05

        calls_data = []
        puts_data = []

        for K, iv in zip(strikes, ivs):
            # Simplified option price (not exact BS, but good enough for demo)
            d1 = (np.log(spot/K) + (r + iv**2/2) * T) / (iv * np.sqrt(T))

            # Approximate prices
            call_price = max(0, spot - K) * 0.5 + np.random.rand() * 2
            put_price = max(0, K - spot) * 0.5 + np.random.rand() * 2

            calls_data.append({
                'strike': K,
                'lastPrice': call_price,
                'bid': call_price * 0.98,
                'ask': call_price * 1.02,
                'volume': np.random.randint(0, 1000),
                'openInterest': np.random.randint(0, 5000),
                'impliedVolatility': iv
            })

            puts_data.append({
                'strike': K,
                'lastPrice': put_price,
                'bid': put_price * 0.98,
                'ask': put_price * 1.02,
                'volume': np.random.randint(0, 1000),
                'openInterest': np.random.randint(0, 5000),
                'impliedVolatility': iv
            })

        return {
            'calls': pd.DataFrame(calls_data),
            'puts': pd.DataFrame(puts_data),
            'expiry': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'spot': spot
        }

    def get_implied_volatility(self, ticker: str, strike: float,
                               expiry: str, option_type: str = 'call') -> float:
        """
        Get implied volatility from options chain.

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
        options = self.get_options_chain(ticker, expiry)

        if option_type == 'call':
            df = options['calls']
        else:
            df = options['puts']

        # Find closest strike
        if 'strike' in df.columns:
            idx = (df['strike'] - strike).abs().argmin()

            if 'impliedVolatility' in df.columns:
                return float(df.iloc[idx]['impliedVolatility'])

        # Default fallback
        return 0.25

    def get_dividend_yield(self, ticker: str) -> float:
        """
        Get dividend yield from Yahoo Finance.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol

        Returns:
        --------
        float
            Annual dividend yield as decimal
        """
        self._ensure_yfinance()
        try:
            stock = self.yf.Ticker(ticker)
            info = stock.info

            # Get dividend yield (already as decimal)
            div_yield = info.get('dividendYield', 0.0)

            if div_yield is None:
                div_yield = 0.0

            return float(div_yield)

        except Exception as e:
            print(f"Error fetching dividend yield for {ticker}: {e}")
            return 0.0