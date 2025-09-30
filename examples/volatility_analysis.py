#!/usr/bin/env python3
"""
Volatility Analysis Example

Demonstrates calculating and comparing different volatility measures.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.market_data.yahoo_fetcher import YahooDataFetcher
from src.market_data.volatility_calc import VolatilityCalculator


def analyze_volatility(ticker: str = 'SPY'):
    """Analyze different volatility measures for a ticker."""
    print(f"\n📊 Volatility Analysis for {ticker}")
    print("=" * 50)

    # Initialize components
    fetcher = YahooDataFetcher()
    vol_calc = VolatilityCalculator()

    # Get historical data
    print("\nFetching historical data...")
    hist_data = fetcher.get_historical_prices(ticker, period='1y')

    if hist_data.empty:
        print("No data available")
        return

    prices = hist_data['Close']

    # 1. Historical Volatility (different windows)
    print("\n1. Historical Volatility (Annualized):")
    for window in [21, 63, 252]:  # 1 month, 3 months, 1 year
        vol = vol_calc.calculate_historical_volatility(prices.tail(window))
        print(f"   {window}-day: {vol*100:.1f}%")

    # 2. EWMA Volatility
    print("\n2. EWMA Volatility:")
    ewma_vol = vol_calc.calculate_ewma_volatility(prices, lambda_param=0.94)
    current_ewma = ewma_vol.iloc[-1]
    print(f"   Current: {current_ewma*100:.1f}%")
    print(f"   30-day avg: {ewma_vol.tail(30).mean()*100:.1f}%")

    # 3. Realized Volatility
    print("\n3. Rolling Realized Volatility:")
    realized_vol = vol_calc.calculate_realized_volatility(prices, window=21)
    current_realized = realized_vol.iloc[-1]
    print(f"   Current (21-day): {current_realized*100:.1f}%")

    # 4. Get implied volatility from options
    print("\n4. Implied Volatility (from options):")
    options_chain = fetcher.get_options_chain(ticker)

    if options_chain and 'calls' in options_chain:
        # Get ATM implied volatility
        spot = fetcher.get_spot_price(ticker)
        calls = options_chain['calls']

        if not calls.empty and 'strike' in calls.columns:
            # Find ATM option
            atm_idx = (calls['strike'] - spot).abs().argmin()
            if 'impliedVolatility' in calls.columns:
                atm_iv = calls.iloc[atm_idx]['impliedVolatility']
                print(f"   ATM Call IV: {atm_iv*100:.1f}%")

    # 5. Volatility Comparison Plot
    print("\n5. Creating volatility comparison chart...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot price
    ax1.plot(hist_data['Date'], prices, label='Close Price', color='blue', alpha=0.7)
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{ticker} Price and Volatility Analysis')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot volatilities
    dates = hist_data['Date'].iloc[20:]  # Skip first 20 for rolling calc
    ax2.plot(dates, realized_vol.dropna(), label='Realized (21-day)', alpha=0.7)
    ax2.plot(dates, ewma_vol.iloc[20:], label='EWMA', alpha=0.7)
    ax2.axhline(y=current_realized, color='red', linestyle='--',
                alpha=0.5, label=f'Current: {current_realized*100:.1f}%')

    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volatility')
    ax2.set_title('Volatility Measures Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'examples/volatility_{ticker}.png', dpi=100)
    print(f"   Saved: examples/volatility_{ticker}.png")

    # 6. Volatility Statistics
    print("\n6. Volatility Statistics (last 252 days):")
    vol_series = realized_vol.dropna().tail(252)
    print(f"   Mean: {vol_series.mean()*100:.1f}%")
    print(f"   Std Dev: {vol_series.std()*100:.1f}%")
    print(f"   Min: {vol_series.min()*100:.1f}%")
    print(f"   Max: {vol_series.max()*100:.1f}%")
    print(f"   Current Percentile: {(vol_series < current_realized).mean()*100:.0f}%")

    return {
        'historical': current_realized,
        'ewma': current_ewma,
        'prices': prices,
        'realized_series': realized_vol
    }


if __name__ == "__main__":
    # Analyze different tickers
    tickers = ['SPY', 'AAPL', 'TSLA']

    print("=" * 50)
    print("MULTI-ASSET VOLATILITY ANALYSIS")
    print("=" * 50)

    results = {}
    for ticker in tickers:
        try:
            results[ticker] = analyze_volatility(ticker)
        except Exception as e:
            print(f"\nError analyzing {ticker}: {e}")

    # Summary comparison
    if results:
        print("\n" + "=" * 50)
        print("VOLATILITY SUMMARY")
        print("=" * 50)
        print(f"{'Ticker':<10} {'Historical':<15} {'EWMA':<15}")
        print("-" * 40)
        for ticker, data in results.items():
            if data:
                hist_vol = data['historical'] * 100
                ewma_vol = data['ewma'] * 100
                print(f"{ticker:<10} {hist_vol:<15.1f}% {ewma_vol:<15.1f}%")

    print("\n✅ Volatility analysis complete!")