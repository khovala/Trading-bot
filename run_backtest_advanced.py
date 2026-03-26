#!/usr/bin/env python3
"""
Advanced backtest with ensemble signals, stop-loss, take-profit, and dynamic position sizing.
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
print("ADVANCED BACKTEST: ENSEMBLE + STOP-LOSS + POSITION SIZING")
print("=" * 70)

# Load minute data - TEST ONLY for speed
print("\n[1/7] Loading test data only...")
from src.data.feature_store.io import read_parquet

test_records = read_parquet(workspace / "data/processed/merged/test.parquet")

df_minute = pd.DataFrame(test_records)
df_minute['timestamp'] = pd.to_datetime(df_minute['timestamp'])
print(f"  Loaded {len(df_minute):,} minute bars")

# Aggregate to hourly
print("\n[2/7] Aggregating to hourly timeframe...")

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
print("\n[3/7] Adding lag features...")

for col in ['return_1', 'rolling_volatility_20', 'rsi_14']:
    df_hourly[f'{col}_lag1'] = df_hourly.groupby('ticker')[col].shift(1).fillna(0)
    df_hourly[f'{col}_lag2'] = df_hourly.groupby('ticker')[col].shift(2).fillna(0)
    if col == 'return_1':
        df_hourly['return_lag5'] = df_hourly.groupby('ticker')[col].shift(5).fillna(0)

df_hourly['macd_momentum_interaction'] = df_hourly['macd'] * df_hourly['momentum_10']
df_hourly['volume_volatility_interaction'] = df_hourly['volume_ratio_20'] * df_hourly['rolling_volatility_20']

print(f"  Lag features added")

# Load ensemble
print("\n[4/7] Loading ensemble models...")

with open(workspace / 'models/base/ensemble_models.pkl', 'rb') as f:
    ensemble = pickle.load(f)

models = ensemble['models']
feature_cols = ensemble['feature_cols']
print(f"  Loaded: {list(models.keys())}")

# Generate ensemble predictions
print("\n[5/7] Generating ensemble predictions...")

X = df_hourly[feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Get predictions from each model
pred_reg = models['reg_gb1'].predict(X)
proba_gb = models['cls_gb1'].predict_proba(X)[:, 1]
proba_et = models['cls_et'].predict_proba(X)[:, 1]

# Ensemble signal: average of classifiers + regression direction
df_hourly['pred_reg'] = pred_reg
df_hourly['proba_gb'] = proba_gb
df_hourly['proba_et'] = proba_et
df_hourly['proba_ensemble'] = (proba_gb + proba_et) / 2

# Signal generation - LONG ONLY with high confidence
prob_threshold = 0.60  # Increased from 0.55 for more confident trades
df_hourly['ensemble_signal'] = 0
mask_up = df_hourly['proba_ensemble'] > prob_threshold
df_hourly.loc[mask_up, 'ensemble_signal'] = 1
df_hourly['ensemble_signal'] = df_hourly['ensemble_signal'].astype(float)

n_long = (df_hourly['ensemble_signal'] > 0).sum()
n_short = (df_hourly['ensemble_signal'] < 0).sum()
n_neutral = (df_hourly['ensemble_signal'] == 0).sum()
print(f"  Signals: Long={n_long:,}, Short={n_short:,}, Neutral={n_neutral:,}")

# Advanced backtest with stop-loss and position sizing
print("\n[6/7] Running advanced backtest...")

initial_cash = 1_000_000
base_position_size_pct = 0.15  # Reduced from 0.30 to 0.15 for lower risk
max_position_size_pct = 0.25  # Reduced from 0.50
min_position_size_pct = 0.08  # Reduced from 0.10
stop_loss_pct = 0.02  # 2% stop-loss
take_profit_pct = 0.03  # 3% take-profit
trailing_stop_enabled = True
trailing_stop_pct = 0.015  # 1.5% trailing stop
commission_bps = 5.0
slippage_bps = 5.0

results_by_ticker = {}
total_pnl = 0
total_trades = 0
total_wins = 0
total_losses = 0
total_stop_losses = 0

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
    stop_losses = 0
    
    entry_price = 0
    position_size_pct = base_position_size_pct
    max_price = 0  # For trailing stop
    
    prices = tdf['close'].values
    signals = tdf['ensemble_signal'].values
    volatility = tdf['rolling_volatility_20'].values
    probs = tdf['proba_ensemble'].values
    
    for i in range(1, len(tdf)):
        prev_signal = signals[i-1]
        curr_price = prices[i]
        prev_price = prices[i-1]
        
        # Dynamic position sizing based on confidence
        confidence = abs(probs[i-1] - 0.5) * 2
        position_size_pct = base_position_size_pct * (0.5 + confidence)
        position_size_pct = max(min_position_size_pct, min(max_position_size_pct, position_size_pct))
        
        # Track max price for trailing stop
        if position > 0:
            if not hasattr(tdf, 'max_price'):
                max_price = entry_price
            max_price = max(max_price, curr_price)
            
        # Check stop-loss / take-profit / trailing stop
        if position != 0 and entry_price > 0:
            if position > 0:
                price_change = (curr_price - entry_price) / entry_price
                if price_change <= -stop_loss_pct:
                    # Stop-loss triggered
                    cash += position * curr_price * (1 - slippage_bps / 10000)
                    cash -= abs(position) * curr_price * commission_bps / 10000
                    stop_losses += 1
                    position = 0
                elif price_change >= take_profit_pct:
                    # Take-profit triggered
                    cash += position * curr_price * (1 - slippage_bps / 10000)
                    cash -= abs(position) * curr_price * commission_bps / 10000
                    wins += 1
                    position = 0
                elif trailing_stop_enabled:
                    # Trailing stop check
                    trailing_trigger = (max_price - curr_price) / max_price
                    if trailing_trigger >= trailing_stop_pct and max_price > entry_price * 1.02:
                        # Exit with profit using trailing stop
                        cash += position * curr_price * (1 - slippage_bps / 10000)
                        cash -= abs(position) * curr_price * commission_bps / 10000
                        wins += 1
                        position = 0
        
        # Entry logic - LONG ONLY with volume filter
        vol_ratio = volatility[i-1] if i > 0 else 1.0
        if prev_signal > 0 and position == 0 and vol_ratio > 0:
            target_value = cash * position_size_pct
            qty = int(target_value / curr_price)
            if qty > 0:
                cost = qty * curr_price * (1 + slippage_bps / 10000)
                commission = cost * commission_bps / 10000
                cash -= cost + commission
                position = qty
                entry_price = curr_price
                max_price = curr_price  # Initialize max_price on entry
                trades += 1
        
        # Track PnL - LONG ONLY
        if position > 0:
            pnl += (curr_price - prev_price) * position
        
        # Exit on signal change - LONG ONLY
        if position > 0 and prev_signal == 0:
            cash += position * curr_price * (1 - slippage_bps / 10000)
            cash -= position * curr_price * commission_bps / 10000
            if curr_price > entry_price:
                wins += 1
            else:
                losses += 1
            position = 0
    
    # Close final position - LONG ONLY
    if position > 0:
        final_price = prices[-1]
        cash += position * final_price * (1 - slippage_bps / 10000)
    
    final_pnl = cash - (initial_cash / 10) + pnl
    
    results_by_ticker[ticker] = {
        'pnl': final_pnl,
        'trades': trades,
        'wins': wins,
        'losses': losses,
        'stop_losses': stop_losses,
    }
    
    total_pnl += final_pnl
    total_trades += trades
    total_wins += wins
    total_losses += losses
    total_stop_losses += stop_losses

# Print results
print("\n" + "=" * 70)
print("RESULTS: ENSEMBLE + STOP-LOSS + POSITION SIZING")
print("=" * 70)

win_rate = total_wins / max(1, total_trades) if total_trades > 0 else 0

print(f"\n  Total PnL: {total_pnl:+,.2f} ₽")
print(f"  Total Trades: {total_trades}")
print(f"  Stop-Loss Triggers: {total_stop_losses}")
print(f"  Win Rate: {win_rate:.2%}")

print(f"\n  Results by Ticker:")
for ticker, data in sorted(results_by_ticker.items(), key=lambda x: x[1]['pnl'], reverse=True):
    w = data['wins']
    l = data['losses']
    sl = data['stop_losses']
    wl = f"{w}/{l}/{sl}"
    print(f"    {ticker:6}: PnL={data['pnl']:+12,.2f} ₽  Trades={data['trades']:3}  W/L/SL={wl}")

# Compare with previous results
print("\n" + "-" * 70)
print("COMPARISON")
print("-" * 70)

minute_pnl = -11262578
hourly_ml_pnl = -997167
ensemble_pnl = total_pnl

print(f"\n  1min (RSI/ZScore):    PnL={minute_pnl:+,.0f} ₽")
print(f"  1H (ML Regression):    PnL={hourly_ml_pnl:+,.0f} ₽")
print(f"  1H (Ensemble+SL):     PnL={ensemble_pnl:+,.2f} ₽")
print(f"\n  Improvement vs 1min: {ensemble_pnl - minute_pnl:+,.2f} ₽")

# Save results
backtest_results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "strategy": "EnsembleStopLoss",
    "period": f"{len(df_hourly['timestamp'].dt.date.unique())} days",
    "timeframe": "1H",
    "features": {
        "ensemble_models": list(models.keys()),
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "position_sizing": "dynamic",
        "prob_threshold": prob_threshold,
    },
    "results": {
        "total_pnl": total_pnl,
        "trades": total_trades,
        "stop_losses": total_stop_losses,
        "win_rate": win_rate,
        "by_ticker": results_by_ticker,
    },
    "comparison": {
        "minute_pnl": minute_pnl,
        "hourly_ml_pnl": hourly_ml_pnl,
        "ensemble_pnl": ensemble_pnl,
    }
}

with open(workspace / "reports/backtest_ensemble_stoploss.json", "w") as f:
    json.dump(backtest_results, f, indent=2)

print(f"\nResults saved to reports/backtest_ensemble_stoploss.json")
