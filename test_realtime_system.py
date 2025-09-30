#!/usr/bin/env python3
"""
Test Real-Time System Components

Tests the real-time integration without requiring external packages.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_data_fetchers():
    """Test data fetcher components."""
    print("\n" + "="*60)
    print("TESTING DATA FETCHERS")
    print("="*60)

    # Test Yahoo fetcher (will use synthetic data if yfinance not installed)
    try:
        from src.market_data.yahoo_fetcher import YahooDataFetcher

        print("\n1. Testing Yahoo Data Fetcher...")
        fetcher = YahooDataFetcher()

        # Test spot price
        spot = fetcher.get_spot_price('AAPL')
        print(f"   ✓ Spot price: ${spot:.2f}")

        # Test historical data
        hist = fetcher.get_historical_prices('AAPL', period='1mo')
        if not hist.empty:
            print(f"   ✓ Historical data: {len(hist)} days")
        else:
            print("   ⚠ Using synthetic historical data")

        # Test options chain
        options = fetcher.get_options_chain('AAPL')
        if options and 'calls' in options:
            print(f"   ✓ Options chain: {len(options['calls'])} calls")
        else:
            print("   ⚠ Using synthetic options data")

        print("   ✅ Yahoo fetcher working!")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    # Test FRED fetcher
    try:
        from src.market_data.fred_fetcher import FREDRateFetcher

        print("\n2. Testing FRED Rate Fetcher...")
        fred = FREDRateFetcher()

        # Test risk-free rate
        rate = fred.get_risk_free_rate('3M')
        print(f"   ✓ 3-Month Treasury: {rate*100:.2f}%")

        # Test yield curve
        curve = fred.get_yield_curve()
        print(f"   ✓ Yield curve: {len(curve)} maturities")

        # Test rate interpolation
        rate_60d = fred.interpolate_rate(60)
        print(f"   ✓ 60-day rate: {rate_60d*100:.2f}%")

        print("   ✅ FRED fetcher working!")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    return True


def test_volatility_calculator():
    """Test volatility calculation."""
    print("\n" + "="*60)
    print("TESTING VOLATILITY CALCULATOR")
    print("="*60)

    try:
        from src.market_data.volatility_calc import VolatilityCalculator

        vol_calc = VolatilityCalculator()

        # Generate synthetic price data
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01
        prices = pd.Series(100 * np.exp(np.cumsum(returns)))

        print("\n1. Historical Volatility")
        hist_vol = vol_calc.calculate_historical_volatility(prices)
        print(f"   ✓ Annual volatility: {hist_vol*100:.1f}%")

        print("\n2. EWMA Volatility")
        ewma_vol = vol_calc.calculate_ewma_volatility(prices)
        print(f"   ✓ Current EWMA: {ewma_vol.iloc[-1]*100:.1f}%")

        print("\n3. Black-Scholes Pricing")
        bs_price = vol_calc.black_scholes_price(100, 100, 0.25, 0.05, 0.2, 'call')
        print(f"   ✓ Call price: ${bs_price:.2f}")

        print("\n4. Implied Volatility")
        iv = vol_calc.calculate_implied_volatility(
            bs_price, 100, 100, 0.25, 0.05, 'call'
        )
        print(f"   ✓ Implied vol: {iv*100:.1f}%")

        print("\n   ✅ Volatility calculator working!")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_live_pricer():
    """Test live option pricer."""
    print("\n" + "="*60)
    print("TESTING LIVE OPTION PRICER")
    print("="*60)

    try:
        from src.market_data.live_pricer import LiveOptionPricer

        print("\n1. Initializing pricer for AAPL...")
        pricer = LiveOptionPricer('AAPL', model_type='xgboost')

        print("\n2. Getting live parameters...")
        params = pricer.get_live_parameters()
        print(f"   ✓ Spot: ${params['S0']:.2f}")
        print(f"   ✓ Volatility: {params['sigma']*100:.1f}%")
        print(f"   ✓ Risk-free rate: {params['r']*100:.2f}%")

        print("\n3. Pricing single option...")
        K = round(params['S0'])  # ATM
        T = 30/365  # 30 days
        result = pricer.price_option(K, T, 'call', use_ml=False)  # Use BS
        print(f"   ✓ Call price: ${result['price']:.2f}")
        print(f"   ✓ Method: {result['method']}")

        print("\n4. Pricing option chain...")
        chain = pricer.price_option_chain()
        if not chain.empty:
            print(f"   ✓ Chain size: {len(chain)} strikes")
            atm_idx = (chain['strike'] - params['S0']).abs().argmin()
            atm = chain.iloc[atm_idx]
            print(f"   ✓ ATM call: ${atm['call_price_model']:.2f}")
            print(f"   ✓ ATM put: ${atm['put_price_model']:.2f}")

        print("\n5. Portfolio risk calculation...")
        positions = [
            {'strike': K*0.95, 'expiry': '2025-02-28', 'type': 'call', 'quantity': 10},
            {'strike': K*1.00, 'expiry': '2025-02-28', 'type': 'put', 'quantity': 5}
        ]
        portfolio = pricer.calculate_portfolio_risk(positions)
        print(f"   ✓ Portfolio value: ${portfolio['total_value']:.2f}")
        print(f"   ✓ Portfolio delta: {portfolio['total_delta']:.2f}")

        print("\n   ✅ Live pricer working!")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_integration():
    """Test full integration."""
    print("\n" + "="*60)
    print("INTEGRATION TEST")
    print("="*60)

    try:
        # Import all components
        from src.market_data.yahoo_fetcher import YahooDataFetcher
        from src.market_data.fred_fetcher import FREDRateFetcher
        from src.market_data.volatility_calc import VolatilityCalculator
        from src.market_data.live_pricer import LiveOptionPricer

        print("\n1. Creating integrated workflow...")

        # Get market data
        yahoo = YahooDataFetcher()
        spot = yahoo.get_spot_price('SPY')
        print(f"   ✓ SPY spot: ${spot:.2f}")

        # Get risk-free rate
        fred = FREDRateFetcher()
        rate = fred.get_risk_free_rate('3M')
        print(f"   ✓ Risk-free rate: {rate*100:.2f}%")

        # Calculate volatility
        vol_calc = VolatilityCalculator()
        hist_data = yahoo.get_historical_prices('SPY', period='3mo')
        if not hist_data.empty:
            vol = vol_calc.calculate_historical_volatility(hist_data['Close'])
            print(f"   ✓ SPY volatility: {vol*100:.1f}%")

        # Price option
        K = round(spot)
        T = 30/365
        price = vol_calc.black_scholes_price(spot, K, T, rate, vol, 'call')
        print(f"   ✓ 30-day ATM call: ${price:.2f}")

        print("\n   ✅ Full integration working!")
        return True

    except Exception as e:
        print(f"   ❌ Integration error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("REAL-TIME SYSTEM COMPONENT TESTS")
    print("="*60)
    print(f"Timestamp: {datetime.now()}")

    results = []

    # Run tests
    print("\nRunning tests...")

    results.append(("Data Fetchers", test_data_fetchers()))
    results.append(("Volatility Calculator", test_volatility_calculator()))
    results.append(("Live Pricer", test_live_pricer()))
    results.append(("Integration", test_integration()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:.<30} {status}")

    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! System ready for real-time trading!")
        print("\nNext steps:")
        print("  1. Install market data packages: pip install -r requirements_realtime.txt")
        print("  2. Run live demo: python live_trading_demo.py")
        print("  3. Try examples: python examples/volatility_analysis.py")
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")

    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)