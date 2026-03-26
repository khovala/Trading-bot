#!/usr/bin/env python3
"""
FINAL OPTIMIZED STRATEGY - Best Parameters from Sweep
======================================================
SL: 1.0%, TP: 5.0% (R:R = 1:5)
Win Rate: 37.5% but high PnL due to risk:reward
"""

import sys
sys.path.insert(0, '/Users/sergeyeliseev/moex-sandbox-platform')

from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import numpy as np

workspace = Path('/Users/sergeyeliseev/moex-sandbox-platform')

print("=" * 70)
print("FINAL OPTIMIZED STRATEGY (BEST PARAMETERS)")
print("=" * 70)

# ============================================================================
# OPTIMIZED CONFIGURATION
# ============================================================================

CONFIG = {
    # Entry (same as before)
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'zscore_oversold': -2.0,
    'zscore_overbought': 2.0,
    'volume_ratio_min': 1.2,
    
    # OPTIMIZED RISK MANAGEMENT
    'position_size_pct': 0.10,
    'stop_loss_pct': 0.01,      # 1.0% (tight stop)
    'take_profit_pct': 0.05,    # 5.0% (large target)
    'trailing_stop_pct': 0.015, # 1.5% trailing
    
    # Costs
    'commission_bps': 5.0,
    'slippage_bps': 5.0,
    
    # Filters
    'market_momentum_threshold': -0.02,
    'safe_hours': (10, 17),
}

print(f"\n📊 Configuration:")
print(f"   SL: {CONFIG['stop_loss_pct']*100:.1f}%")
print(f"   TP: {CONFIG['take_profit_pct']*100:.1f}%")
print(f"   R:R = 1:{CONFIG['take_profit_pct']/CONFIG['stop_loss_pct']:.1f}")

# ============================================================================
# DATA
# ============================================================================

print("\n[1/5] Loading data...")
from src.data.feature_store.io import read_parquet

test_records = read_parquet(workspace / "data/processed/merged/test.parquet")
df = pd.DataFrame(test_records)
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"  Loaded {len(df):,} minute bars")

# Aggregate
print("\n[2/5] Aggregating to hourly...")
df['hour_bucket'] = df['timestamp'].dt.floor('h')

agg_dict = {
    'close': 'last',
    'volume': 'sum',
    'return_1': 'sum',
    'rolling_volatility_20': 'last',
    'rsi_14': 'last',
    'zscore_20': 'last',
    'volume_ratio_20': 'last',
}

df_hourly = df.groupby(['ticker', 'hour_bucket']).agg(agg_dict).reset_index()
df_hourly.rename(columns={'hour_bucket': 'timestamp'}, inplace=True)
df_hourly = df_hourly.sort_values(['ticker', 'timestamp']).reset_index(drop=True)

# Add filters
df_hourly['volume_above_avg'] = df_hourly['volume_ratio_20'] >= CONFIG['volume_ratio_min']
df_hourly['hour'] = df_hourly['timestamp'].dt.hour
df_hourly['safe_hour'] = (df_hourly['hour'] >= CONFIG['safe_hours'][0]) & (df_hourly['hour'] <= CONFIG['safe_hours'][1])

# Market regime
market_momentum = df_hourly.groupby('timestamp')['return_1'].mean().to_dict()
df_hourly['market_momentum'] = df_hourly['timestamp'].map(market_momentum)
df_hourly['market_safe'] = df_hourly['market_momentum'] > CONFIG['market_momentum_threshold']

# Signals
def generate_signal(row):
    if not row.get('market_safe', True):
        return 0
    if not row.get('safe_hour', True):
        return 0
    if not row.get('volume_above_avg', True):
        return 0
    
    rsi = row.get('rsi_14', 50)
    zscore = row.get('zscore_20', 0)
    
    if rsi < CONFIG['rsi_oversold'] and zscore < CONFIG['zscore_oversold']:
        return 1
    return 0

df_hourly['signal'] = df_hourly.apply(generate_signal, axis=1)

print(f"  Signals: {df_hourly['signal'].sum()}")

# ============================================================================
# BACKTEST WITH OPTIMIZED PARAMS
# ============================================================================

print("\n[3/5] Running backtest...")

initial_capital = 1_000_000
sl = CONFIG['stop_loss_pct']
tp = CONFIG['take_profit_pct']
commission_bps = CONFIG['commission_bps']
slippage_bps = CONFIG['slippage_bps']

results_by_ticker = {}
total_pnl = 0
total_trades = 0
total_wins = 0
total_losses = 0
total_stop_losses = 0
total_take_profits = 0
total_trailing_exits = 0

for ticker in sorted(df_hourly['ticker'].unique()):
    tdf = df_hourly[df_hourly['ticker'] == ticker].copy()
    tdf = tdf.sort_values('timestamp').reset_index(drop=True)
    
    if len(tdf) < 20:
        continue
    
    capital_per_ticker = initial_capital / 10
    position = 0
    pnl = 0
    trades = 0
    wins = 0
    losses = 0
    stop_losses = 0
    take_profits = 0
    trailing_exits = 0
    
    entry_price = 0
    max_price = 0
    
    prices = tdf['close'].values
    signals = tdf['signal'].values
    
    for i in range(1, len(tdf)):
        curr_price = prices[i]
        prev_price = prices[i-1]
        signal = signals[i]
        
        if position > 0:
            price_change = (curr_price - entry_price) / entry_price
            
            # Stop loss
            if price_change <= -sl:
                gross_pnl = position * (entry_price * (1 - sl) - entry_price)
                pnl += gross_pnl - (position * entry_price * (commission_bps + slippage_bps) / 10000)
                stop_losses += 1
                losses += 1
                position = 0
            
            # Take profit
            elif price_change >= tp:
                gross_pnl = position * (entry_price * (1 + tp) - entry_price)
                pnl += gross_pnl - (position * entry_price * (commission_bps + slippage_bps) / 10000)
                take_profits += 1
                wins += 1
                position = 0
            
            # Trailing stop
            elif curr_price < max_price * (1 - CONFIG['trailing_stop_pct']) and max_price > entry_price * 1.02:
                gross_pnl = position * (max_price * (1 - CONFIG['trailing_stop_pct']) - entry_price)
                pnl += gross_pnl - (position * entry_price * (commission_bps + slippage_bps) / 10000)
                trailing_exits += 1
                wins += 1
                position = 0
            
            max_price = max(max_price, curr_price)
            pnl += (curr_price - prev_price) * position
        
        if signal == 1 and position == 0:
            position_size = capital_per_ticker * CONFIG['position_size_pct']
            qty = int(position_size / curr_price)
            
            if qty > 0:
                position = qty
                entry_price = curr_price * (1 + slippage_bps / 10000)
                max_price = entry_price
                trades += 1
    
    if position > 0:
        final_price = prices[-1]
        gross_pnl = position * (final_price - entry_price)
        pnl += gross_pnl - (position * entry_price * (commission_bps + slippage_bps) / 10000)
        if final_price > entry_price:
            wins += 1
        else:
            losses += 1
    
    final_pnl = pnl
    
    results_by_ticker[ticker] = {
        'pnl': final_pnl,
        'trades': trades,
        'wins': wins,
        'losses': losses,
        'stop_losses': stop_losses,
        'take_profits': take_profits,
        'trailing_exits': trailing_exits,
    }
    
    total_pnl += final_pnl
    total_trades += trades
    total_wins += wins
    total_losses += losses
    total_stop_losses += stop_losses
    total_take_profits += take_profits
    total_trailing_exits += trailing_exits

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("FINAL RESULTS (OPTIMIZED)")
print("=" * 70)

win_rate = total_wins / max(1, total_trades) if total_trades > 0 else 0

print(f"\n  📈 Configuration:")
print(f"     SL: {sl*100:.1f}% | TP: {tp*100:.1f}% | R:R = 1:{tp/sl:.1f}")
print(f"\n  💰 Results:")
print(f"     Total PnL: {total_pnl:+,.2f} ₽")
print(f"     Return: {total_pnl/initial_capital*100:+.2f}%")
print(f"     Trades: {total_trades}")
print(f"     Win Rate: {win_rate*100:.1f}%")
print(f"\n  📊 Exit Analysis:")
print(f"     Stop-Loss:      {total_stop_losses}")
print(f"     Take-Profit:    {total_take_profits}")
print(f"     Trailing Stop:  {total_trailing_exits}")

# Costs
total_costs = total_trades * initial_capital * CONFIG['position_size_pct'] * ((commission_bps + slippage_bps) / 10000)
print(f"\n  💸 Costs:")
print(f"     Estimated: {total_costs:,.0f} ₽")

print(f"\n  📋 By Ticker:")
for ticker, data in sorted(results_by_ticker.items(), key=lambda x: x[1]['pnl'], reverse=True):
    print(f"     {ticker:6}: PnL={data['pnl']:+12,.2f}₽  Trades={data['trades']:2}  W/L={data['wins']}/{data['losses']}  SL/TP/TS={data['stop_losses']}/{data['take_profits']}/{data['trailing_exits']}")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("COMPARISON WITH PREVIOUS STRATEGIES")
print("=" * 70)

comparisons = [
    ("1min RSI/Z-Score (90 days)", -11_262_578),
    ("1H ML Regression", -997_167),
    ("1H Ensemble+SL", -25_463),
    ("30 days Daily", 121_248),
    ("Intraday (original)", 1_968),
    ("FINAL OPTIMIZED", total_pnl),
]

for name, pnl in comparisons:
    print(f"  {name:35}: PnL={pnl:+12,.0f} ₽")

improvement = total_pnl - 1968
print(f"\n  Improvement vs Original: +{improvement:.0f}₽ (+{improvement/1968*100:.1f}%)")

# ============================================================================
# SAVE
# ============================================================================

backtest_results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "strategy": "FinalOptimized",
    "config": CONFIG,
    "results": {
        "total_pnl": total_pnl,
        "return_pct": total_pnl / initial_capital * 100,
        "trades": total_trades,
        "win_rate": win_rate,
        "sl_triggers": total_stop_losses,
        "tp_triggers": total_take_profits,
        "trailing_triggers": total_trailing_exits,
        "by_ticker": results_by_ticker,
    },
}

with open(workspace / "reports/backtest_final_optimized.json", "w") as f:
    json.dump(backtest_results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to reports/backtest_final_optimized.json")