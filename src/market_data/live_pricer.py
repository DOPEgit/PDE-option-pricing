"""
Live option pricing system using real market data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import joblib
import os

from .yahoo_fetcher import YahooDataFetcher
from .fred_fetcher import FREDRateFetcher
from .volatility_calc import VolatilityCalculator


class LiveOptionPricer:
    """Real-time option pricing using ML models and market data."""

    def __init__(self, ticker: str, model_type: str = 'xgboost',
                 fred_api_key: Optional[str] = None):
        """
        Initialize live option pricer.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        model_type : str
            ML model type ('xgboost', 'random_forest')
        fred_api_key : str, optional
            FRED API key for risk-free rates
        """
        self.ticker = ticker
        self.model_type = model_type

        # Initialize data fetchers
        self.yahoo_fetcher = YahooDataFetcher()
        self.fred_fetcher = FREDRateFetcher(fred_api_key)
        self.vol_calc = VolatilityCalculator()

        # Load ML models
        self.ml_models = self._load_models()

        # Cache for market data
        self.market_data_cache = {}

    def _load_models(self) -> Dict:
        """Load trained ML models."""
        models = {}
        model_dir = f'data/models/{self.model_type}'

        # Check if models exist
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory {model_dir} not found.")
            print("Using PDE solver fallback.")
            return None

        try:
            # Load all models
            for model_name in ['price', 'delta', 'gamma', 'theta']:
                model_path = os.path.join(model_dir, f'{model_name}_model.joblib')
                if os.path.exists(model_path):
                    models[model_name] = joblib.load(model_path)
                    print(f"Loaded {model_name} model")
        except Exception as e:
            print(f"Error loading models: {e}")
            return None

        return models

    def get_live_parameters(self) -> Dict:
        """
        Fetch current market parameters.

        Returns:
        --------
        dict
            Market parameters including spot, rate, volatility
        """
        # Get spot price
        S0 = self.yahoo_fetcher.get_spot_price(self.ticker)

        # Get historical prices for volatility calculation
        hist_prices = self.yahoo_fetcher.get_historical_prices(
            self.ticker, period='3mo'
        )

        # Calculate historical volatility
        if not hist_prices.empty:
            sigma = self.vol_calc.calculate_historical_volatility(
                hist_prices['Close']
            )
        else:
            sigma = 0.25  # Default

        # Get risk-free rate (3-month Treasury as default)
        r = self.fred_fetcher.get_risk_free_rate('3M')

        # Get dividend yield
        div_yield = self.yahoo_fetcher.get_dividend_yield(self.ticker)

        params = {
            'S0': S0,
            'sigma': sigma,
            'r': r,
            'div_yield': div_yield,
            'timestamp': datetime.now()
        }

        # Cache the parameters
        self.market_data_cache['params'] = params

        return params

    def price_option(self, K: float, T: float,
                     option_type: str = 'call',
                     use_ml: bool = True) -> Dict:
        """
        Price a single option.

        Parameters:
        -----------
        K : float
            Strike price
        T : float
            Time to maturity (years)
        option_type : str
            'call' or 'put'
        use_ml : bool
            Use ML model if available

        Returns:
        --------
        dict
            Pricing results including price and Greeks
        """
        # Get market parameters
        params = self.get_live_parameters()
        S0 = params['S0']
        r = params['r']
        sigma = params['sigma']

        # Prepare features for ML model
        features = pd.DataFrame([{
            'S0': S0,
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma,
            'moneyness': S0 / K,
            'log_moneyness': np.log(S0 / K),
            'sqrt_T': np.sqrt(T),
            'vol_sqrt_T': sigma * np.sqrt(T)
        }])

        results = {
            'spot': S0,
            'strike': K,
            'expiry': T,
            'risk_free_rate': r,
            'volatility': sigma,
            'option_type': option_type
        }

        # Use ML model if available
        if use_ml and self.ml_models:
            try:
                # Get predictions from ML models
                price = self.ml_models['price'].predict(features)[0]
                delta = self.ml_models['delta'].predict(features)[0] if 'delta' in self.ml_models else None
                gamma = self.ml_models['gamma'].predict(features)[0] if 'gamma' in self.ml_models else None
                theta = self.ml_models['theta'].predict(features)[0] if 'theta' in self.ml_models else None

                # Adjust for put options
                if option_type == 'put':
                    # Use put-call parity for price
                    call_price = price
                    price = call_price - S0 + K * np.exp(-r * T)
                    # Adjust Greeks
                    if delta is not None:
                        delta = delta - 1  # Put delta = Call delta - 1

                results.update({
                    'price': price,
                    'delta': delta,
                    'gamma': gamma,
                    'theta': theta,
                    'method': 'ML'
                })

            except Exception as e:
                print(f"ML prediction error: {e}")
                use_ml = False

        # Fallback to Black-Scholes
        if not use_ml or not self.ml_models:
            price = self.vol_calc.black_scholes_price(
                S0, K, T, r, sigma, option_type
            )
            results.update({
                'price': price,
                'method': 'Black-Scholes'
            })

        return results

    def price_option_chain(self, expiry_date: Optional[str] = None,
                           strikes: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Price entire option chain.

        Parameters:
        -----------
        expiry_date : str, optional
            Expiration date (YYYY-MM-DD)
        strikes : list, optional
            List of strike prices

        Returns:
        --------
        pd.DataFrame
            Option chain with model prices
        """
        # Get market data
        params = self.get_live_parameters()
        S0 = params['S0']

        # Get options chain from market
        market_chain = self.yahoo_fetcher.get_options_chain(
            self.ticker, expiry_date
        )

        # Calculate time to expiry
        if expiry_date:
            expiry = pd.to_datetime(expiry_date)
        else:
            expiry = pd.to_datetime(market_chain['expiry'])

        T = max((expiry - pd.Timestamp.now()).days / 365.0, 0.001)

        # If no strikes specified, use market strikes or generate
        if strikes is None:
            if 'calls' in market_chain and not market_chain['calls'].empty:
                strikes = market_chain['calls']['strike'].values
            else:
                # Generate strikes around spot
                strikes = np.arange(S0 * 0.7, S0 * 1.3, S0 * 0.025)

        results = []

        for K in strikes:
            # Price call
            call_result = self.price_option(K, T, 'call')
            # Price put
            put_result = self.price_option(K, T, 'put')

            # Get market prices if available
            market_call_price = None
            market_put_price = None

            if 'calls' in market_chain and not market_chain['calls'].empty:
                call_match = market_chain['calls'][
                    market_chain['calls']['strike'] == K
                ]
                if not call_match.empty:
                    market_call_price = call_match.iloc[0]['lastPrice']

            if 'puts' in market_chain and not market_chain['puts'].empty:
                put_match = market_chain['puts'][
                    market_chain['puts']['strike'] == K
                ]
                if not put_match.empty:
                    market_put_price = put_match.iloc[0]['lastPrice']

            results.append({
                'strike': K,
                'moneyness': K / S0,
                'call_price_model': call_result['price'],
                'call_price_market': market_call_price,
                'call_delta': call_result.get('delta'),
                'put_price_model': put_result['price'],
                'put_price_market': market_put_price,
                'put_delta': put_result.get('delta'),
                'gamma': call_result.get('gamma'),  # Same for call and put
                'expiry': expiry.strftime('%Y-%m-%d'),
                'time_to_expiry': T
            })

        return pd.DataFrame(results)

    def identify_arbitrage(self, threshold: float = 0.05) -> pd.DataFrame:
        """
        Identify arbitrage opportunities.

        Parameters:
        -----------
        threshold : float
            Minimum relative price difference to flag

        Returns:
        --------
        pd.DataFrame
            Arbitrage opportunities
        """
        # Get option chain with model and market prices
        chain = self.price_option_chain()

        opportunities = []

        for _, row in chain.iterrows():
            # Check call arbitrage
            if row['call_price_market'] is not None:
                call_diff = row['call_price_model'] - row['call_price_market']
                call_pct = abs(call_diff) / row['call_price_market'] if row['call_price_market'] > 0 else 0

                if call_pct > threshold:
                    opportunities.append({
                        'type': 'call',
                        'strike': row['strike'],
                        'model_price': row['call_price_model'],
                        'market_price': row['call_price_market'],
                        'difference': call_diff,
                        'pct_difference': call_pct * 100,
                        'signal': 'BUY' if call_diff < 0 else 'SELL'
                    })

            # Check put arbitrage
            if row['put_price_market'] is not None:
                put_diff = row['put_price_model'] - row['put_price_market']
                put_pct = abs(put_diff) / row['put_price_market'] if row['put_price_market'] > 0 else 0

                if put_pct > threshold:
                    opportunities.append({
                        'type': 'put',
                        'strike': row['strike'],
                        'model_price': row['put_price_model'],
                        'market_price': row['put_price_market'],
                        'difference': put_diff,
                        'pct_difference': put_pct * 100,
                        'signal': 'BUY' if put_diff < 0 else 'SELL'
                    })

        return pd.DataFrame(opportunities)

    def calculate_portfolio_risk(self, positions: List[Dict]) -> Dict:
        """
        Calculate portfolio-level Greeks.

        Parameters:
        -----------
        positions : list
            List of position dictionaries with keys:
            - 'strike', 'expiry', 'type', 'quantity'

        Returns:
        --------
        dict
            Portfolio Greeks
        """
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_value = 0

        for pos in positions:
            # Calculate time to expiry
            expiry = pd.to_datetime(pos['expiry'])
            T = max((expiry - pd.Timestamp.now()).days / 365.0, 0.001)

            # Price the option
            result = self.price_option(
                pos['strike'], T, pos['type']
            )

            # Aggregate Greeks
            quantity = pos['quantity']
            total_value += result['price'] * quantity
            if result.get('delta'):
                total_delta += result['delta'] * quantity
            if result.get('gamma'):
                total_gamma += result['gamma'] * quantity
            if result.get('theta'):
                total_theta += result['theta'] * quantity

        return {
            'total_value': total_value,
            'total_delta': total_delta,
            'total_gamma': total_gamma,
            'total_theta': total_theta,
            'spot': self.get_live_parameters()['S0']
        }

    def generate_scenario_analysis(self, strikes: List[float],
                                    expiry: float,
                                    spot_range: Tuple[float, float] = (0.8, 1.2),
                                    vol_range: Tuple[float, float] = (0.5, 1.5),
                                    n_scenarios: int = 20) -> pd.DataFrame:
        """
        Generate scenario analysis for options.

        Parameters:
        -----------
        strikes : list
            Strike prices to analyze
        expiry : float
            Time to expiry
        spot_range : tuple
            Range for spot price scenarios (as multiplier)
        vol_range : tuple
            Range for volatility scenarios (as multiplier)
        n_scenarios : int
            Number of scenarios per dimension

        Returns:
        --------
        pd.DataFrame
            Scenario analysis results
        """
        params = self.get_live_parameters()
        base_spot = params['S0']
        base_vol = params['sigma']

        # Generate scenarios
        spot_scenarios = np.linspace(
            base_spot * spot_range[0],
            base_spot * spot_range[1],
            n_scenarios
        )
        vol_scenarios = np.linspace(
            base_vol * vol_range[0],
            base_vol * vol_range[1],
            n_scenarios
        )

        results = []

        for spot in spot_scenarios:
            for vol in vol_scenarios:
                for strike in strikes:
                    # Temporarily override parameters
                    self.market_data_cache['params']['S0'] = spot
                    self.market_data_cache['params']['sigma'] = vol

                    # Price option
                    call_price = self.price_option(strike, expiry, 'call')['price']
                    put_price = self.price_option(strike, expiry, 'put')['price']

                    results.append({
                        'spot': spot,
                        'volatility': vol,
                        'strike': strike,
                        'call_price': call_price,
                        'put_price': put_price,
                        'spot_change_%': (spot / base_spot - 1) * 100,
                        'vol_change_%': (vol / base_vol - 1) * 100
                    })

        # Restore original parameters
        self.market_data_cache['params']['S0'] = base_spot
        self.market_data_cache['params']['sigma'] = base_vol

        return pd.DataFrame(results)