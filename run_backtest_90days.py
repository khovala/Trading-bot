#!/usr/bin/env python3
"""
Run backtest on 90 days data and save results
"""

import sys
sys.path.insert(0, '/Users/sergeyeliseev/moex-sandbox-platform')

from pathlib import Path
from datetime import datetime, timezone
import json

workspace = Path('/Users/sergeyeliseev/moex-sandbox-platform')

print("=" * 60)
print("BACKTEST ON 90 DAYS DATA")
print("=" * 60)

# Load test data
print("\n[1/3] Loading test data...")
import pandas as pd

from src.data.feature_store.io import read_parquet
test_records = read_parquet(workspace / "data/processed/merged/test.parquet")
test_df = pd.DataFrame(test_records)
print(f"  Test samples: {len(test_df):,}")

# Load train data for reference
train_records = read_parquet(workspace / "data/processed/merged/train.parquet")
train_df = pd.DataFrame(train_records)
print(f"  Train samples: {len(train_df):,}")

# Combine for full backtest
full_df = test_df
print(f"  Full dataset: {len(full_df):,} rows")

# Run backtest using the strategy
print("\n[2/3] Running Mean Reversion + Market Timing strategy...")

# Import strategy
from src.strategies.final_strategy import MeanReversionMarketTimingStrategy, StrategyConfig

config = StrategyConfig(
    position_size_pct=0.3,
    market_threshold=-0.03,
    rsi_oversold=35,
    rsi_overbought=65,
    zscore_oversold=-1.5,
    zscore_overbought=1.5,
)

strategy = MeanReversionMarketTimingStrategy(config)
result = strategy.run_backtest(full_df)

print(f"\n  Strategy Results:")
print(f"    Total PnL: {result.total_pnl:,.2f} ₽")
print(f"    Total Trades: {result.trades}")
print(f"    Win Rate: {result.win_rate:.2%}")

print(f"\n  Results by Ticker:")
for ticker, data in sorted(result.tickers.items(), key=lambda x: x[1]['pnl'], reverse=True):
    print(f"    {ticker:6}: PnL={data['pnl']:+12,.2f} ₽  Trades={data['trades']:3}  W/L={data['wins']}/{data['losses']}")

# Save results
print("\n[3/3] Saving results...")

backtest_results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "strategy": "MeanReversionMarketTiming",
    "period": "90 days",
    "config": {
        "position_size_pct": config.position_size_pct,
        "market_threshold": config.market_threshold,
        "rsi_oversold": config.rsi_oversold,
        "rsi_overbought": config.rsi_overbought,
        "zscore_oversold": config.zscore_oversold,
        "zscore_overbought": config.zscore_overbought,
        "commission_bps": config.commission_bps,
        "slippage_bps": config.slippage_bps,
    },
    "results": {
        "total_pnl": result.total_pnl,
        "trades": result.trades,
        "win_rate": result.win_rate,
        "by_ticker": result.tickers,
    }
}

with open(workspace / "reports/backtest_90days.json", "w") as f:
    json.dump(backtest_results, f, indent=2)

print(f"\nResults saved to reports/backtest_90days.json")

# Print summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Period: 90 days ({len(full_df):,} samples)")
print(f"Total PnL: {result.total_pnl:+,.2f} ₽")
print(f"Total Trades: {result.trades}")
print(f"Win Rate: {result.win_rate:.2%}")
