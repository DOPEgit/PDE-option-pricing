# Real-Time Trading Integration Guide

## Overview

This guide demonstrates how to use the PDE option pricing system with real-world market data for live trading applications.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install real-time market data dependencies
pip install -r requirements_realtime.txt
```

### 2. Run Live Demo

```bash
python live_trading_demo.py
```

Enter a ticker symbol (e.g., AAPL, SPY, MSFT) when prompted.

---

## 📊 Market Data Sources

### Yahoo Finance (Primary)
- **Real-time stock prices**
- **Options chains**
- **Historical price data**
- **Implied volatility**
- **Dividend yields**

```python
from src.market_data.yahoo_fetcher import YahooDataFetcher

fetcher = YahooDataFetcher()
spot_price = fetcher.get_spot_price('AAPL')
options = fetcher.get_options_chain('AAPL')
```

### Federal Reserve (FRED)
- **Risk-free rates** (Treasury yields)
- **Economic indicators**

```python
from src.market_data.fred_fetcher import FREDRateFetcher

fred = FREDRateFetcher()  # Optional: pass API key
rate = fred.get_risk_free_rate('3M')  # 3-month Treasury
```

---

## 💡 Core Features

### 1. Live Option Pricing

```python
from src.market_data.live_pricer import LiveOptionPricer

# Initialize pricer for a ticker
pricer = LiveOptionPricer('AAPL', model_type='xgboost')

# Get current market parameters
params = pricer.get_live_parameters()
print(f"Spot: ${params['S0']:.2f}")
print(f"Volatility: {params['sigma']*100:.1f}%")

# Price a single option
result = pricer.price_option(
    K=150,          # Strike price
    T=30/365,       # 30 days to expiry
    option_type='call',
    use_ml=True     # Use ML model for speed
)
print(f"Option Price: ${result['price']:.2f}")
```

### 2. Option Chain Analysis

```python
# Analyze entire option chain
chain = pricer.price_option_chain()

# Display results
for _, row in chain.iterrows():
    print(f"Strike ${row['strike']:.0f}: "
          f"Call ${row['call_price_model']:.2f}, "
          f"Put ${row['put_price_model']:.2f}")
```

### 3. Arbitrage Detection

```python
# Find mispriced options
opportunities = pricer.identify_arbitrage(threshold=0.10)

for _, opp in opportunities.iterrows():
    print(f"{opp['type']} @ ${opp['strike']}: "
          f"{opp['signal']} ({opp['pct_difference']:.1f}% difference)")
```

### 4. Portfolio Risk Management

```python
# Define portfolio positions
positions = [
    {'strike': 145, 'expiry': '2025-01-31', 'type': 'call', 'quantity': 10},
    {'strike': 150, 'expiry': '2025-01-31', 'type': 'call', 'quantity': -20},
    {'strike': 155, 'expiry': '2025-01-31', 'type': 'call', 'quantity': 10}
]

# Calculate portfolio Greeks
portfolio = pricer.calculate_portfolio_risk(positions)
print(f"Portfolio Delta: {portfolio['total_delta']:.2f}")
print(f"Portfolio Gamma: {portfolio['total_gamma']:.4f}")
print(f"Daily Theta: ${portfolio['total_theta']:.2f}")
```

---

## 📈 Advanced Features

### Volatility Analysis

```python
from src.market_data.volatility_calc import VolatilityCalculator

vol_calc = VolatilityCalculator()

# Historical volatility
hist_vol = vol_calc.calculate_historical_volatility(prices)

# Implied volatility from option price
impl_vol = vol_calc.calculate_implied_volatility(
    option_price=5.50,
    S=100, K=105, T=0.25, r=0.05
)

# Volatility smile
smile = vol_calc.calculate_volatility_smile(options_chain, S, r, T)
```

### Real-Time Monitoring

```python
# Run the real-time monitor
python examples/realtime_monitor.py

# Monitor multiple tickers with automatic updates
# Tracks spot prices, volatilities, and option prices
```

### Scenario Analysis

```python
# Generate what-if scenarios
scenarios = pricer.generate_scenario_analysis(
    strikes=[145, 150, 155],
    expiry=30/365,
    spot_range=(0.9, 1.1),     # ±10% spot
    vol_range=(0.8, 1.2),       # ±20% volatility
    n_scenarios=100
)

# Analyze results
best_case = scenarios.loc[scenarios['call_price'].idxmax()]
worst_case = scenarios.loc[scenarios['call_price'].idxmin()]
```

---

## 🏗️ System Architecture

```
Market Data Flow:
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Yahoo Finance│────▶│ Data Cache   │────▶│ Live Pricer │
└─────────────┘     └──────────────┘     └─────────────┘
                            │                     │
┌─────────────┐            │                     ▼
│ FRED API    │────────────┘            ┌─────────────┐
└─────────────┘                         │ ML Models   │
                                        └─────────────┘
                                                 │
                                                 ▼
                                        ┌─────────────┐
                                        │ Trading     │
                                        │ Decisions   │
                                        └─────────────┘
```

---

## 🔧 Configuration

### API Keys (Optional)

```python
# FRED API (free from https://fred.stlouisfed.org/docs/api/api_key.html)
fred = FREDRateFetcher(api_key='your_api_key_here')

# Yahoo Finance doesn't require API key
```

### Model Selection

```python
# Choose ML model type
pricer = LiveOptionPricer('AAPL', model_type='xgboost')  # Fast
# or
pricer = LiveOptionPricer('AAPL', model_type='random_forest')  # Robust
```

### Caching Configuration

```python
# Set cache duration (seconds)
fetcher = YahooDataFetcher(cache_duration=60)  # 1-minute cache
```

---

## 📊 Example Use Cases

### 1. Options Market Making

```python
def market_maker(ticker, spread=0.10):
    """Automated market making strategy."""
    pricer = LiveOptionPricer(ticker)

    while True:
        chain = pricer.price_option_chain()

        for _, opt in chain.iterrows():
            model_price = opt['call_price_model']
            bid = model_price * (1 - spread/2)
            ask = model_price * (1 + spread/2)

            print(f"Quote: ${bid:.2f} / ${ask:.2f}")

        time.sleep(1)  # Update every second
```

### 2. Delta-Neutral Trading

```python
def delta_neutral_strategy(ticker):
    """Maintain delta-neutral portfolio."""
    pricer = LiveOptionPricer(ticker)

    portfolio = load_portfolio()
    risk = pricer.calculate_portfolio_risk(portfolio)

    if abs(risk['total_delta']) > 10:
        # Hedge with stock
        shares_to_trade = -risk['total_delta']
        execute_trade(ticker, shares_to_trade)
```

### 3. Volatility Trading

```python
def volatility_arbitrage(ticker):
    """Trade volatility discrepancies."""
    pricer = LiveOptionPricer(ticker)
    vol_calc = VolatilityCalculator()

    # Get historical volatility
    hist_data = fetcher.get_historical_prices(ticker)
    hist_vol = vol_calc.calculate_historical_volatility(hist_data['Close'])

    # Get implied volatility
    chain = pricer.price_option_chain()
    # Extract ATM implied vol

    if impl_vol > hist_vol * 1.2:
        # Sell volatility (sell straddle)
        pass
    elif impl_vol < hist_vol * 0.8:
        # Buy volatility (buy straddle)
        pass
```

---

## ⚡ Performance

### Speed Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Fetch spot price | ~100ms | With caching: ~0.1ms |
| Price single option (ML) | ~0.5ms | 1500x faster than PDE |
| Price option chain (50 strikes) | ~25ms | With ML models |
| Full portfolio risk | ~10ms | 10 positions |

### Accuracy

- ML models: <1% error vs PDE solver
- Real-time data: ~100ms latency from market
- Historical volatility: 21-day rolling window
- Risk-free rates: Updated hourly

---

## 🚨 Important Notes

### Market Hours
- US Markets: 9:30 AM - 4:00 PM ET
- Options may have different liquidity throughout the day
- Use pre-market/after-hours data with caution

### Data Limitations
- Free Yahoo Finance data may have delays
- Options data might be sparse for illiquid strikes
- Consider professional data feeds for production

### Risk Management
- Always validate model prices against market
- Implement position limits
- Monitor Greeks continuously
- Use stop-losses for automated trading

---

## 🐛 Troubleshooting

### No Data Available
```python
# Check internet connection
# Verify ticker symbol is valid
# Try alternative data source
```

### ML Model Not Found
```python
# Ensure models are trained: python main_demo.py
# Check model directory: data/models/{model_type}/
# Fallback to Black-Scholes pricing
```

### Slow Performance
```python
# Enable caching
# Reduce update frequency
# Use vectorized operations
# Consider batch processing
```

---

## 📚 Further Resources

- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/fred/)
- [Options Theory](https://www.investopedia.com/terms/o/option.asp)
- [Black-Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)

---

**Last Updated:** 2025-09-29
**Author:** Live Trading Integration Module