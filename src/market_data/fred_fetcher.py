"""
Federal Reserve Economic Data (FRED) API fetcher for risk-free rates.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
import json
import urllib.request
import urllib.parse


class FREDRateFetcher:
    """Fetches risk-free rates from FRED API."""

    # Common Treasury series IDs
    TREASURY_SERIES = {
        '1M': 'DGS1MO',   # 1-Month Treasury
        '3M': 'DGS3MO',   # 3-Month Treasury
        '6M': 'DGS6MO',   # 6-Month Treasury
        '1Y': 'DGS1',     # 1-Year Treasury
        '2Y': 'DGS2',     # 2-Year Treasury
        '3Y': 'DGS3',     # 3-Year Treasury
        '5Y': 'DGS5',     # 5-Year Treasury
        '10Y': 'DGS10',   # 10-Year Treasury
        '30Y': 'DGS30'    # 30-Year Treasury
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED API fetcher.

        Parameters:
        -----------
        api_key : str, optional
            FRED API key. If not provided, uses fallback values.
        """
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_duration = 3600  # Cache for 1 hour

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

    def get_risk_free_rate(self, maturity: str = '3M') -> float:
        """
        Get current risk-free rate for given maturity.

        Parameters:
        -----------
        maturity : str
            Maturity period ('1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '10Y', '30Y')

        Returns:
        --------
        float
            Annual risk-free rate as decimal
        """
        cache_key = f"rate_{maturity}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        # If no API key, return reasonable defaults
        if not self.api_key:
            # Current approximate rates (as of 2024)
            default_rates = {
                '1M': 0.0525,
                '3M': 0.0535,
                '6M': 0.0530,
                '1Y': 0.0500,
                '2Y': 0.0450,
                '3Y': 0.0425,
                '5Y': 0.0415,
                '10Y': 0.0425,
                '30Y': 0.0455
            }
            rate = default_rates.get(maturity, 0.05)
            self._save_to_cache(cache_key, rate)
            return rate

        # Fetch from FRED API
        try:
            series_id = self.TREASURY_SERIES.get(maturity)
            if not series_id:
                print(f"Unknown maturity: {maturity}")
                return 0.05

            # Build API request
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': 1,
                'sort_order': 'desc',
                'observation_start': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            }

            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            response = urllib.request.urlopen(url)
            data = json.loads(response.read().decode())

            if data['observations']:
                # Convert percentage to decimal
                rate = float(data['observations'][0]['value']) / 100.0
                self._save_to_cache(cache_key, rate)
                return rate

        except Exception as e:
            print(f"Error fetching rate from FRED: {e}")

        # Fallback to default
        return 0.05

    def get_yield_curve(self) -> Dict[str, float]:
        """
        Get entire yield curve.

        Returns:
        --------
        dict
            Mapping of maturity to rate
        """
        yield_curve = {}
        for maturity in self.TREASURY_SERIES.keys():
            yield_curve[maturity] = self.get_risk_free_rate(maturity)

        return yield_curve

    def interpolate_rate(self, days_to_maturity: int) -> float:
        """
        Interpolate risk-free rate for specific maturity.

        Parameters:
        -----------
        days_to_maturity : int
            Days until maturity

        Returns:
        --------
        float
            Interpolated risk-free rate
        """
        # Define maturity buckets in days
        maturity_days = {
            30: '1M',
            90: '3M',
            180: '6M',
            365: '1Y',
            730: '2Y',
            1095: '3Y',
            1825: '5Y',
            3650: '10Y',
            10950: '30Y'
        }

        # Find bracketing maturities
        days_list = sorted(maturity_days.keys())

        if days_to_maturity <= days_list[0]:
            # Use shortest maturity
            return self.get_risk_free_rate(maturity_days[days_list[0]])

        if days_to_maturity >= days_list[-1]:
            # Use longest maturity
            return self.get_risk_free_rate(maturity_days[days_list[-1]])

        # Find bracketing points
        for i in range(len(days_list) - 1):
            if days_list[i] <= days_to_maturity <= days_list[i + 1]:
                # Linear interpolation
                d1 = days_list[i]
                d2 = days_list[i + 1]
                r1 = self.get_risk_free_rate(maturity_days[d1])
                r2 = self.get_risk_free_rate(maturity_days[d2])

                # Interpolate
                w = (days_to_maturity - d1) / (d2 - d1)
                return r1 * (1 - w) + r2 * w

        # Fallback
        return 0.05

    def get_real_rate(self, maturity: str = '10Y') -> float:
        """
        Get real interest rate (TIPS-based).

        Parameters:
        -----------
        maturity : str
            Maturity period

        Returns:
        --------
        float
            Real interest rate as decimal
        """
        # TIPS series for real rates
        tips_series = {
            '5Y': 'DFII5',
            '10Y': 'DFII10',
            '30Y': 'DFII30'
        }

        if maturity not in tips_series:
            # Estimate real rate as nominal - expected inflation (2%)
            nominal = self.get_risk_free_rate(maturity)
            return nominal - 0.02

        # Similar API call for TIPS rates
        # For demo, return approximation
        nominal = self.get_risk_free_rate(maturity)
        return nominal - 0.02  # Rough inflation expectation