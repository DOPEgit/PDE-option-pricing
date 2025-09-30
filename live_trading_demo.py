#!/usr/bin/env python3
"""
Live Option Trading Demo using Real Market Data

This script demonstrates real-time option pricing with market data integration.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.market_data.live_pricer import LiveOptionPricer
from src.market_data.yahoo_fetcher import YahooDataFetcher
from src.market_data.volatility_calc import VolatilityCalculator


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)


def display_market_data(pricer: LiveOptionPricer):
    """Display current market data."""
    print_header("CURRENT MARKET DATA")

    params = pricer.get_live_parameters()

    print(f"\nTicker: {pricer.ticker}")
    print(f"Spot Price: ${params['S0']:.2f}")
    print(f"Historical Volatility: {params['sigma']*100:.1f}%")
    print(f"Risk-Free Rate: {params['r']*100:.2f}%")
    print(f"Dividend Yield: {params['div_yield']*100:.2f}%")
    print(f"Timestamp: {params['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")


def price_single_option(pricer: LiveOptionPricer):
    """Price a single option example."""
    print_header("SINGLE OPTION PRICING")

    # Example: Price an at-the-money call option
    params = pricer.get_live_parameters()
    S0 = params['S0']
    K = round(S0)  # ATM strike
    T = 30/365  # 30 days to expiry

    print(f"\nPricing {pricer.ticker} Call Option:")
    print(f"  Strike: ${K}")
    print(f"  Days to Expiry: 30")

    # Price with ML model
    result = pricer.price_option(K, T, 'call', use_ml=True)

    print(f"\nResults:")
    print(f"  Method: {result['method']}")
    print(f"  Price: ${result['price']:.2f}")
    if result.get('delta') is not None:
        print(f"  Delta: {result['delta']:.4f}")
    if result.get('gamma') is not None:
        print(f"  Gamma: {result['gamma']:.4f}")
    if result.get('theta') is not None:
        print(f"  Theta: ${result['theta']:.2f}/day")

    # Also price with Black-Scholes for comparison
    bs_result = pricer.price_option(K, T, 'call', use_ml=False)
    print(f"\nBlack-Scholes Price: ${bs_result['price']:.2f}")

    if result['method'] == 'ML':
        diff = abs(result['price'] - bs_result['price'])
        pct_diff = (diff / bs_result['price']) * 100 if bs_result['price'] > 0 else 0
        print(f"Difference: ${diff:.2f} ({pct_diff:.1f}%)")


def analyze_option_chain(pricer: LiveOptionPricer):
    """Analyze entire option chain."""
    print_header("OPTION CHAIN ANALYSIS")

    # Get next monthly expiry (approximately)
    expiry_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')

    print(f"\nAnalyzing option chain for expiry around {expiry_date}...")

    # Price the option chain
    chain = pricer.price_option_chain()

    if chain.empty:
        print("No option chain data available")
        return

    # Display summary statistics
    print(f"\nOption Chain Summary:")
    print(f"  Number of Strikes: {len(chain)}")
    print(f"  Strike Range: ${chain['strike'].min():.0f} - ${chain['strike'].max():.0f}")

    # Find at-the-money options
    params = pricer.get_live_parameters()
    S0 = params['S0']
    atm_idx = (chain['strike'] - S0).abs().argmin()
    atm_row = chain.iloc[atm_idx]

    print(f"\nAt-The-Money Options (Strike=${atm_row['strike']:.0f}):")
    print(f"  Call Price (Model): ${atm_row['call_price_model']:.2f}")
    if atm_row['call_price_market'] is not None:
        print(f"  Call Price (Market): ${atm_row['call_price_market']:.2f}")
    print(f"  Put Price (Model): ${atm_row['put_price_model']:.2f}")
    if atm_row['put_price_market'] is not None:
        print(f"  Put Price (Market): ${atm_row['put_price_market']:.2f}")

    # Show put-call parity
    if atm_row['call_price_model'] and atm_row['put_price_model']:
        r = params['r']
        T = atm_row['time_to_expiry']
        parity_lhs = atm_row['call_price_model'] - atm_row['put_price_model']
        parity_rhs = S0 - atm_row['strike'] * np.exp(-r * T)
        parity_diff = abs(parity_lhs - parity_rhs)

        print(f"\nPut-Call Parity Check:")
        print(f"  C - P = ${parity_lhs:.2f}")
        print(f"  S - Ke^(-rT) = ${parity_rhs:.2f}")
        print(f"  Difference: ${parity_diff:.2f}")


def find_arbitrage(pricer: LiveOptionPricer):
    """Find arbitrage opportunities."""
    print_header("ARBITRAGE SCANNER")

    print("\nScanning for pricing discrepancies...")

    opportunities = pricer.identify_arbitrage(threshold=0.10)  # 10% threshold

    if opportunities.empty:
        print("\nNo significant arbitrage opportunities found")
    else:
        print(f"\nFound {len(opportunities)} potential opportunities:\n")
        for _, opp in opportunities.head(5).iterrows():
            print(f"{opp['type'].upper()} Strike ${opp['strike']:.0f}:")
            print(f"  Model Price: ${opp['model_price']:.2f}")
            print(f"  Market Price: ${opp['market_price']:.2f}")
            print(f"  Difference: {opp['pct_difference']:.1f}%")
            print(f"  Signal: {opp['signal']}")
            print()


def portfolio_risk_demo(pricer: LiveOptionPricer):
    """Demonstrate portfolio risk calculations."""
    print_header("PORTFOLIO RISK ANALYSIS")

    params = pricer.get_live_parameters()
    S0 = params['S0']

    # Example portfolio
    positions = [
        {'strike': S0 * 0.95, 'expiry': '2025-01-31', 'type': 'call', 'quantity': 10},
        {'strike': S0 * 1.00, 'expiry': '2025-01-31', 'type': 'call', 'quantity': -20},
        {'strike': S0 * 1.05, 'expiry': '2025-01-31', 'type': 'call', 'quantity': 10},
        {'strike': S0 * 1.00, 'expiry': '2025-01-31', 'type': 'put', 'quantity': 5}
    ]

    print("\nExample Portfolio:")
    for pos in positions:
        sign = '+' if pos['quantity'] > 0 else ''
        print(f"  {sign}{pos['quantity']} {pos['type']}s @ ${pos['strike']:.0f}")

    # Calculate portfolio Greeks
    portfolio = pricer.calculate_portfolio_risk(positions)

    print(f"\nPortfolio Greeks:")
    print(f"  Total Value: ${portfolio['total_value']:.2f}")
    print(f"  Delta: {portfolio['total_delta']:.2f}")
    print(f"  Gamma: {portfolio['total_gamma']:.4f}")
    print(f"  Theta: ${portfolio['total_theta']:.2f}/day")

    # Risk metrics
    print(f"\nRisk Metrics:")
    print(f"  Delta-Neutral Hedge: {-portfolio['total_delta']:.1f} shares")
    print(f"  1% Move Impact: ${portfolio['total_delta'] * S0 * 0.01:.2f}")
    print(f"  Daily Decay: ${portfolio['total_theta']:.2f}")


def scenario_analysis(pricer: LiveOptionPricer):
    """Run scenario analysis."""
    print_header("SCENARIO ANALYSIS")

    params = pricer.get_live_parameters()
    S0 = params['S0']

    # Define strikes for analysis
    strikes = [S0 * 0.9, S0, S0 * 1.1]
    expiry = 30/365  # 30 days

    print(f"\nRunning scenarios for strikes: {strikes}")
    print("Spot range: -20% to +20%")
    print("Volatility range: -50% to +50%")

    scenarios = pricer.generate_scenario_analysis(
        strikes, expiry,
        spot_range=(0.8, 1.2),
        vol_range=(0.5, 1.5),
        n_scenarios=5
    )

    # Show extreme scenarios
    print("\n--- Best Case (Max Call Value) ---")
    best_call = scenarios.loc[scenarios['call_price'].idxmax()]
    print(f"Spot: ${best_call['spot']:.2f} ({best_call['spot_change_%']:+.1f}%)")
    print(f"Volatility: {best_call['volatility']*100:.1f}% ({best_call['vol_change_%']:+.1f}%)")
    print(f"Strike ${best_call['strike']:.0f} Call: ${best_call['call_price']:.2f}")

    print("\n--- Worst Case (Min Call Value) ---")
    worst_call = scenarios.loc[scenarios['call_price'].idxmin()]
    print(f"Spot: ${worst_call['spot']:.2f} ({worst_call['spot_change_%']:+.1f}%)")
    print(f"Volatility: {worst_call['volatility']*100:.1f}% ({worst_call['vol_change_%']:+.1f}%)")
    print(f"Strike ${worst_call['strike']:.0f} Call: ${worst_call['call_price']:.2f}")


def main():
    """Main demo function."""
    print_header("LIVE OPTION TRADING SYSTEM DEMO")

    # Get ticker from user or use default
    ticker = input("\nEnter ticker symbol (default: AAPL): ").strip().upper()
    if not ticker:
        ticker = 'AAPL'

    print(f"\nInitializing Live Option Pricer for {ticker}...")

    try:
        # Initialize pricer
        pricer = LiveOptionPricer(ticker, model_type='xgboost')

        # Run demos
        display_market_data(pricer)
        price_single_option(pricer)
        analyze_option_chain(pricer)
        find_arbitrage(pricer)
        portfolio_risk_demo(pricer)
        scenario_analysis(pricer)

        print_header("DEMO COMPLETE")
        print("\n✅ Successfully demonstrated live option pricing capabilities!")
        print("✅ System ready for real-time trading applications!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: Some features require:")
        print("  - Internet connection for market data")
        print("  - Trained ML models in data/models/")
        print("  - Valid ticker symbol")
        print("\nTo install market data dependencies:")
        print("  pip install -r requirements_realtime.txt")


if __name__ == "__main__":
    main()