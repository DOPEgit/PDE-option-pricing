#!/usr/bin/env python3
"""
Real-Time Option Monitor

Monitors option prices and Greeks in real-time.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List

from src.market_data.live_pricer import LiveOptionPricer
from src.market_data.yahoo_fetcher import YahooDataFetcher


class RealTimeMonitor:
    """Real-time option monitoring system."""

    def __init__(self, tickers: List[str], strikes: Dict[str, List[float]] = None):
        """
        Initialize monitor.

        Parameters:
        -----------
        tickers : list
            List of tickers to monitor
        strikes : dict
            Dictionary of ticker -> list of strikes to monitor
        """
        self.tickers = tickers
        self.strikes = strikes or {}
        self.pricers = {}
        self.history = []

        # Initialize pricers for each ticker
        for ticker in tickers:
            self.pricers[ticker] = LiveOptionPricer(ticker)

    def get_snapshot(self) -> Dict:
        """Get current market snapshot."""
        snapshot = {
            'timestamp': datetime.now(),
            'data': {}
        }

        for ticker in self.tickers:
            pricer = self.pricers[ticker]

            # Get market data
            params = pricer.get_live_parameters()
            spot = params['S0']

            # Determine strikes to monitor
            if ticker in self.strikes:
                strikes_to_monitor = self.strikes[ticker]
            else:
                # Default: ATM, ±5%, ±10%
                strikes_to_monitor = [
                    round(spot * 0.90),
                    round(spot * 0.95),
                    round(spot),
                    round(spot * 1.05),
                    round(spot * 1.10)
                ]

            ticker_data = {
                'spot': spot,
                'volatility': params['sigma'],
                'options': []
            }

            # Price each strike
            for strike in strikes_to_monitor:
                T = 30/365  # 30 days

                # Price call
                call_result = pricer.price_option(strike, T, 'call')
                put_result = pricer.price_option(strike, T, 'put')

                ticker_data['options'].append({
                    'strike': strike,
                    'moneyness': strike / spot,
                    'call_price': call_result['price'],
                    'put_price': put_result['price'],
                    'call_delta': call_result.get('delta'),
                    'gamma': call_result.get('gamma')
                })

            snapshot['data'][ticker] = ticker_data

        return snapshot

    def display_snapshot(self, snapshot: Dict):
        """Display formatted snapshot."""
        timestamp = snapshot['timestamp']
        print(f"\n{'='*70}")
        print(f"Market Snapshot: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        for ticker, data in snapshot['data'].items():
            print(f"\n{ticker}")
            print(f"  Spot: ${data['spot']:.2f}")
            print(f"  Vol: {data['volatility']*100:.1f}%")
            print(f"\n  {'Strike':<10} {'Call':<10} {'Put':<10} {'C.Delta':<10} {'Gamma':<10}")
            print(f"  {'-'*50}")

            for opt in data['options']:
                strike = opt['strike']
                call_price = opt['call_price']
                put_price = opt['put_price']
                delta = opt['call_delta'] or 0
                gamma = opt['gamma'] or 0

                # Add indicators for moneyness
                if opt['moneyness'] < 0.98:
                    indicator = " (OTM)"
                elif opt['moneyness'] > 1.02:
                    indicator = " (ITM)"
                else:
                    indicator = " (ATM)"

                print(f"  {strike:<10.0f} ${call_price:<9.2f} ${put_price:<9.2f} "
                      f"{delta:<10.3f} {gamma:<10.4f}{indicator}")

    def calculate_changes(self, prev_snapshot: Dict, curr_snapshot: Dict) -> Dict:
        """Calculate changes between snapshots."""
        changes = {}

        for ticker in self.tickers:
            if ticker in prev_snapshot['data'] and ticker in curr_snapshot['data']:
                prev_data = prev_snapshot['data'][ticker]
                curr_data = curr_snapshot['data'][ticker]

                spot_change = curr_data['spot'] - prev_data['spot']
                spot_pct = (spot_change / prev_data['spot']) * 100

                vol_change = (curr_data['volatility'] - prev_data['volatility']) * 100

                changes[ticker] = {
                    'spot_change': spot_change,
                    'spot_pct': spot_pct,
                    'vol_change': vol_change,
                    'option_changes': []
                }

                # Calculate option price changes
                for i, curr_opt in enumerate(curr_data['options']):
                    if i < len(prev_data['options']):
                        prev_opt = prev_data['options'][i]
                        changes[ticker]['option_changes'].append({
                            'strike': curr_opt['strike'],
                            'call_change': curr_opt['call_price'] - prev_opt['call_price'],
                            'put_change': curr_opt['put_price'] - prev_opt['put_price']
                        })

        return changes

    def run(self, duration: int = 60, interval: int = 10):
        """
        Run monitor for specified duration.

        Parameters:
        -----------
        duration : int
            Total monitoring duration in seconds
        interval : int
            Update interval in seconds
        """
        print(f"\n📡 Starting Real-Time Monitor")
        print(f"   Tickers: {', '.join(self.tickers)}")
        print(f"   Duration: {duration}s")
        print(f"   Update interval: {interval}s")
        print(f"\nPress Ctrl+C to stop...\n")

        start_time = time.time()
        prev_snapshot = None

        try:
            while time.time() - start_time < duration:
                # Get current snapshot
                curr_snapshot = self.get_snapshot()
                self.history.append(curr_snapshot)

                # Display snapshot
                self.display_snapshot(curr_snapshot)

                # Show changes if we have previous snapshot
                if prev_snapshot:
                    changes = self.calculate_changes(prev_snapshot, curr_snapshot)
                    self._display_changes(changes)

                prev_snapshot = curr_snapshot

                # Wait for next update
                if time.time() - start_time < duration:
                    time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n⚠️  Monitor stopped by user")

        self._display_summary()

    def _display_changes(self, changes: Dict):
        """Display changes between snapshots."""
        if not changes:
            return

        print(f"\n📈 Changes since last update:")
        for ticker, change_data in changes.items():
            spot_change = change_data['spot_change']
            spot_pct = change_data['spot_pct']
            vol_change = change_data['vol_change']

            # Use arrows for direction
            spot_arrow = "↑" if spot_change > 0 else "↓" if spot_change < 0 else "→"
            vol_arrow = "↑" if vol_change > 0 else "↓" if vol_change < 0 else "→"

            print(f"  {ticker}: Spot {spot_arrow} ${abs(spot_change):.2f} ({spot_pct:+.2f}%), "
                  f"Vol {vol_arrow} {abs(vol_change):.1f}%")

    def _display_summary(self):
        """Display monitoring summary."""
        if len(self.history) < 2:
            return

        print(f"\n{'='*70}")
        print("MONITORING SUMMARY")
        print(f"{'='*70}")

        first_snapshot = self.history[0]
        last_snapshot = self.history[-1]

        for ticker in self.tickers:
            if ticker in first_snapshot['data'] and ticker in last_snapshot['data']:
                first_data = first_snapshot['data'][ticker]
                last_data = last_snapshot['data'][ticker]

                spot_change = last_data['spot'] - first_data['spot']
                spot_pct = (spot_change / first_data['spot']) * 100

                print(f"\n{ticker}:")
                print(f"  Starting Spot: ${first_data['spot']:.2f}")
                print(f"  Ending Spot: ${last_data['spot']:.2f}")
                print(f"  Total Change: ${spot_change:.2f} ({spot_pct:+.2f}%)")

                # Option price changes
                if first_data['options'] and last_data['options']:
                    atm_first = min(first_data['options'],
                                    key=lambda x: abs(x['moneyness'] - 1.0))
                    atm_last = min(last_data['options'],
                                   key=lambda x: abs(x['moneyness'] - 1.0))

                    call_change = atm_last['call_price'] - atm_first['call_price']
                    put_change = atm_last['put_price'] - atm_first['put_price']

                    print(f"  ATM Call Change: ${call_change:+.2f}")
                    print(f"  ATM Put Change: ${put_change:+.2f}")

        print(f"\nTotal snapshots collected: {len(self.history)}")


def main():
    """Main function."""
    print("=" * 70)
    print("REAL-TIME OPTION MONITOR")
    print("=" * 70)

    # Get tickers from user
    tickers_input = input("\nEnter tickers to monitor (comma-separated, default: AAPL,MSFT): ")
    if tickers_input.strip():
        tickers = [t.strip().upper() for t in tickers_input.split(',')]
    else:
        tickers = ['AAPL', 'MSFT']

    # Get monitoring duration
    duration_input = input("Monitoring duration in seconds (default: 60): ")
    duration = int(duration_input) if duration_input.strip() else 60

    # Get update interval
    interval_input = input("Update interval in seconds (default: 10): ")
    interval = int(interval_input) if interval_input.strip() else 10

    # Create and run monitor
    monitor = RealTimeMonitor(tickers)
    monitor.run(duration, interval)

    print("\n✅ Monitoring complete!")


if __name__ == "__main__":
    main()