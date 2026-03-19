#!/usr/bin/env python3
"""
Backtest on hourly data with directional classifier.
"""

import sys
sys.path.insert(0, '/Users/sergeyeliseev/moex-sandbox-platform')

from pathlib import Path
from datetime import datetime, timezone
import json
import pickle
import pandas as pd
import numpy as np

workspace = Path('/Users/sergeyeliseev/moex-sandbox-platform')

print("=" * 70)
print("BACKTEST: HOURLY DATA + DIRECTIONAL CLASSIFIER")
print("=" * 70)

# Load minute data
print("\n[1/6] Loading minute data...")
from src.data.feature_store.io import read_parquet

train_records = read_parquet(workspace / "data/processed/merged/train.parquet")
test_records = read_parquet(workspace / "data/processed/merged/test.parquet")

df_minute = pd.DataFrame(train_records + test_records)
df_minute['timestamp'] = pd.to_datetime(df_minute['timestamp'])
print(f"  Loaded {len(df_minute):,} minute bars")

# Aggregate to hourly
print("\n[2/6] Aggregating to hourly timeframe...")

df_minute['hour'] = df_minute['timestamp'].dt.floor('h')

agg_dict = {
    'close': 'last',
    'volume': 'sum',
    'return_1': 'sum',
    'log_return_1': 'sum',
    'rolling_volatility_20': 'last',
    'momentum_10': 'last',
    'rsi_14': 'last',
    'macd': 'last',
    'macd_signal': 'last',
    'atr_14': 'last',
    'zscore_20': 'last',
    'volume_ratio_20': 'last',
    'volume_zscore_20': 'last',
    'trend_regime': 'last',
    'volatility_regime': 'last',
}

df_hourly = df_minute.groupby(['ticker', 'hour']).agg(agg_dict).reset_index()
df_hourly.rename(columns={'hour': 'timestamp'}, inplace=True)
df_hourly = df_hourly.sort_values(['ticker', 'timestamp']).reset_index(drop=True)

print(f"  Aggregated to {len(df_hourly):,} hourly bars")

# Add lag features
print("\n[3/6] Adding lag features...")

df_hourly = df_hourly.sort_values(['ticker', 'timestamp']).reset_index(drop=True)

for col in ['return_1', 'rolling_volatility_20', 'rsi_14']:
    df_hourly[f'{col}_lag1'] = df_hourly.groupby('ticker')[col].shift(1).fillna(0)
    df_hourly[f'{col}_lag2'] = df_hourly.groupby('ticker')[col].shift(2).fillna(0)
    if col == 'return_1':
        df_hourly['return_lag5'] = df_hourly.groupby('ticker')[col].shift(5).fillna(0)

df_hourly['macd_momentum_interaction'] = df_hourly['macd'] * df_hourly['momentum_10']
df_hourly['volume_volatility_interaction'] = df_hourly['volume_ratio_20'] * df_hourly['rolling_volatility_20']

print(f"  Lag features added")

# Load classifier
print("\n[4/6] Loading directional classifier...")

from src.models.base.serialization import load_pickle
clf = load_pickle(workspace / 'models/base/directional_classifier.pkl')
print(f"  Model loaded: {type(clf).__name__}")

FEATURE_COLS = [
    'rolling_volatility_20', 'momentum_10', 'rsi_14', 'macd', 'macd_signal',
    'atr_14', 'zscore_20', 'volume_ratio_20', 'volume_zscore_20',
    'trend_regime', 'volatility_regime',
    'return_1_lag1', 'return_1_lag2', 'return_lag5',
    'rolling_volatility_20_lag1', 'rsi_14_lag1',
    'macd_momentum_interaction', 'volume_volatility_interaction',
]

# Generate predictions
print("\n[5/6] Generating predictions...")

X = df_hourly[FEATURE_COLS].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

proba = clf.predict_proba(X)
df_hourly['prob_up'] = proba[:, 1]
df_hourly['prob_down'] = proba[:, 0]
df_hourly['confidence'] = np.abs(df_hourly['prob_up'] - 0.5) * 2

# Signal: trade if probability > threshold (higher threshold = fewer trades)
prob_threshold = 0.55  # 55% probability

df_hourly['signal'] = 0
mask_up = df_hourly['prob_up'] > prob_threshold
mask_down = df_hourly['prob_down'] > prob_threshold
df_hourly.loc[mask_up, 'signal'] = 1
df_hourly.loc[mask_down, 'signal'] = -1
df_hourly['signal'] = df_hourly['signal'].astype(float)

n_long = (df_hourly['signal'] > 0).sum()
n_short = (df_hourly['signal'] < 0).sum()
n_neutral = (df_hourly['signal'] == 0).sum()
print(f"  Signals: Long={n_long:,}, Short={n_short:,}, Neutral={n_neutral:,}")
print(f"  Mean confidence: {df_hourly['confidence'].mean():.4f}")

# Backtest
print("\n[6/6] Running backtest...")

initial_cash = 1_000_000
position_size_pct = 0.3
commission_bps = 5.0
slippage_bps = 5.0

results_by_ticker = {}
total_pnl = 0
total_trades = 0
total_wins = 0
total_losses = 0

for ticker in sorted(df_hourly['ticker'].unique()):
    tdf = df_hourly[df_hourly['ticker'] == ticker].copy()
    tdf = tdf.sort_values('timestamp').reset_index(drop=True)
    
    if len(tdf) < 50:
        continue
    
    cash = initial_cash / 10
    position = 0
    pnl = 0
    trades = 0
    wins = 0
    losses = 0
    entry_price = 0
    
    prices = tdf['close'].values
    signals = tdf['signal'].values
    
    for i in range(1, len(tdf)):
        prev_signal = signals[i-1]
        curr_price = prices[i]
        prev_price = prices[i-1]
        
        if prev_signal != 0 and position == 0:
            target_value = cash * position_size_pct
            qty = int(target_value / curr_price)
            if qty > 0:
                cost = qty * curr_price * (1 + slippage_bps / 10000)
                commission = cost * commission_bps / 10000
                cash -= cost + commission
                position = qty * (1 if prev_signal > 0 else -1)
                entry_price = curr_price
                trades += 1
        
        if position > 0:
            pnl += (curr_price - prev_price) * position
        elif position < 0:
            pnl += (prev_price - curr_price) * abs(position)
        
        if position != 0 and prev_signal == 0:
            if position > 0:
                cash += position * curr_price * (1 - slippage_bps / 10000)
                if curr_price > entry_price:
                    wins += 1
                else:
                    losses += 1
            else:
                cash -= abs(position) * curr_price * (1 + slippage_bps / 10000)
                if curr_price < entry_price:
                    wins += 1
                else:
                    losses += 1
            position = 0
    
    if position != 0:
        final_price = prices[-1]
        if position > 0:
            cash += position * final_price * (1 - slippage_bps / 10000)
        else:
            cash -= abs(position) * final_price * (1 + slippage_bps / 10000)
    
    final_pnl = cash - (initial_cash / 10) + pnl
    
    results_by_ticker[ticker] = {
        'pnl': final_pnl,
        'trades': trades,
        'wins': wins,
        'losses': losses,
    }
    
    total_pnl += final_pnl
    total_trades += trades
    total_wins += wins
    total_losses += losses

# Print results
print("\n" + "=" * 70)
print("RESULTS: HOURLY + DIRECTIONAL CLASSIFIER")
print("=" * 70)

print(f"\n  Total PnL: {total_pnl:+,.2f} ₽")
print(f"  Total Trades: {total_trades}")
print(f"  Win Rate: {total_wins / max(1, total_trades):.2%}")

print(f"\n  Results by Ticker:")
for ticker, data in sorted(results_by_ticker.items(), key=lambda x: x[1]['pnl'], reverse=True):
    w = data['wins']
    l = data['losses']
    wl = f"{w}/{l}"
    print(f"    {ticker:6}: PnL={data['pnl']:+12,.2f} ₽  Trades={data['trades']:3}  W/L={wl}")

# Comparison
print("\n" + "-" * 70)
print("COMPARISON")
print("-" * 70)

minute_pnl = -11262578
hourly_clf_pnl = total_pnl
improvement = hourly_clf_pnl - minute_pnl

print(f"\n  Minute (RSI/Zscore):     PnL={minute_pnl:+,.0f} ₽")
print(f"  Hourly (Classifier):       PnL={hourly_clf_pnl:+,.2f} ₽")
print(f"\n  Improvement: {improvement:+,.2f} ₽")

# Save results
backtest_results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "strategy": "HourlyDirectionalClassifier",
    "period": f"{len(df_hourly['timestamp'].dt.date.unique())} days",
    "timeframe": "1H",
    "data_points": len(df_hourly),
    "model": {
        "name": "directional_classifier",
        "version": "v1",
        "training_accuracy": float(clf.score(X, (df_hourly['prob_up'] > 0.5).astype(int))),
    },
    "results": {
        "total_pnl": total_pnl,
        "trades": total_trades,
        "win_rate": total_wins / max(1, total_trades),
        "by_ticker": results_by_ticker,
    },
    "comparison": {
        "minute_pnl": minute_pnl,
        "hourly_clf_pnl": hourly_clf_pnl,
        "improvement": improvement,
    }
}

with open(workspace / "reports/backtest_hourly_classifier.json", "w") as f:
    json.dump(backtest_results, f, indent=2)

print(f"\nResults saved to reports/backtest_hourly_classifier.json")
