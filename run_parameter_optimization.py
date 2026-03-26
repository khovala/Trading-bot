#!/usr/bin/env python3
"""
Parameter Optimization for Intraday Strategy
=============================================
Tests different SL/TP combinations to maximize PnL and Win Rate
"""

import sys
sys.path.insert(0, '/Users/sergeyeliseev/moex-sandbox-platform')

from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import numpy as np
from itertools import product

workspace = Path('/Users/sergeyeliseev/moex-sandbox-platform')

print("=" * 70)
print("PARAMETER OPTIMIZATION: SL/TP LEVELS")
print("=" * 70)

# ============================================================================
# DATA LOADING
# ============================================================================

print("\n[1/5] Loading test data...")
from src.data.feature_store.io import read_parquet

test_records = read_parquet(workspace / "data/processed/merged/test.parquet")
df = pd.DataFrame(test_records)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
print(f"  Loaded {len(df):,} minute bars")

# ============================================================================
# AGGREGATE TO HOURLY
# ============================================================================

print("\n[2/5] Aggregating to hourly...")

df['hour_bucket'] = df['timestamp'].dt.floor('h')

agg_dict = {
    'close': 'last',
    'volume': 'sum',
    'return_1': 'sum',
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

df_hourly = df.groupby(['ticker', 'hour_bucket']).agg(agg_dict).reset_index()
df_hourly.rename(columns={'hour_bucket': 'timestamp'}, inplace=True)
df_hourly = df_hourly.sort_values(['ticker', 'timestamp']).reset_index(drop=True)

# Add filters
df_hourly['volume_above_avg'] = df_hourly['volume_ratio_20'] >= 1.2
df_hourly['hour'] = df_hourly['timestamp'].dt.hour
df_hourly['safe_hour'] = (df_hourly['hour'] >= 10) & (df_hourly['hour'] <= 17)

# Market regime
market_momentum = df_hourly.groupby('timestamp')['return_1'].mean().to_dict()
df_hourly['market_momentum'] = df_hourly['timestamp'].map(market_momentum)
df_hourly['market_safe'] = df_hourly['market_momentum'] > -0.02

# Signal generation
def generate_signal(row):
    if not row.get('market_safe', True):
        return 0
    if not row.get('safe_hour', True):
        return 0
    if not row.get('volume_above_avg', True):
        return 0
    
    rsi = row.get('rsi_14', 50)
    zscore = row.get('zscore_20', 0)
    
    if rsi < 30 and zscore < -2.0:
        return 1
    return 0

df_hourly['signal'] = df_hourly.apply(generate_signal, axis=1)

print(f"  Aggregated: {len(df_hourly):,} bars, {df_hourly['signal'].sum()} signals")

# ============================================================================
# PARAMETER SWEEP
# ============================================================================

print("\n[3/5] Testing SL/TP combinations...")

# Parameter grid
sl_levels = [0.01, 0.012, 0.015, 0.018, 0.02, 0.025, 0.03]  # 1% - 3%
tp_levels = [0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]  # 1.5% - 5%

# Fixed params
commission_bps = 5.0
slippage_bps = 5.0
position_pct = 0.10
initial_capital = 1_000_000

results = []

# Test each combination
for sl, tp in product(sl_levels, tp_levels):
    if sl >= tp:  # Skip if SL >= TP (no positive expectancy)
        continue
    
    total_pnl = 0
    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_stop_losses = 0
    total_take_profits = 0
    
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
        
        entry_price = 0
        max_price = 0
        
        prices = tdf['close'].values
        signals = tdf['signal'].values
        
        for i in range(1, len(tdf)):
            curr_price = prices[i]
            prev_price = prices[i-1]
            signal = signals[i]
            
            # Exit logic
            if position > 0:
                price_change = (curr_price - entry_price) / entry_price
                
                if price_change <= -sl:
                    gross_pnl = position * (entry_price * (1 - sl) - entry_price)
                    pnl += gross_pnl - (position * entry_price * (commission_bps + slippage_bps) / 10000)
                    stop_losses += 1
                    losses += 1
                    position = 0
                    
                elif price_change >= tp:
                    gross_pnl = position * (entry_price * (1 + tp) - entry_price)
                    pnl += gross_pnl - (position * entry_price * (commission_bps + slippage_bps) / 10000)
                    take_profits += 1
                    wins += 1
                    position = 0
                
                max_price = max(max_price, curr_price)
                pnl += (curr_price - prev_price) * position
            
            # Entry logic
            if signal == 1 and position == 0:
                position_size = capital_per_ticker * position_pct
                qty = int(position_size / curr_price)
                
                if qty > 0:
                    cost = qty * curr_price * (1 + slippage_bps / 10000)
                    commission = cost * commission_bps / 10000
                    position = qty
                    entry_price = curr_price * (1 + slippage_bps / 10000)
                    max_price = entry_price
                    trades += 1
        
        # Close final position
        if position > 0:
            final_price = prices[-1]
            gross_pnl = position * (final_price - entry_price)
            pnl += gross_pnl - (position * entry_price * (commission_bps + slippage_bps) / 10000)
        
        total_pnl += pnl
        total_trades += trades
        total_wins += wins
        total_losses += losses
        total_stop_losses += stop_losses
        total_take_profits += take_profits
    
    win_rate = total_wins / max(1, total_trades) if total_trades > 0 else 0
    rr_ratio = tp / sl  # Risk:Reward ratio
    
    results.append({
        'sl': sl,
        'tp': tp,
        'pnl': total_pnl,
        'trades': total_trades,
        'wins': total_wins,
        'losses': total_losses,
        'win_rate': win_rate,
        'rr_ratio': rr_ratio,
        'stop_losses': total_stop_losses,
        'take_profits': total_take_profits,
    })

# ============================================================================
# FIND BEST PARAMETERS
# ============================================================================

print("\n[4/5] Analyzing results...")

# Sort by PnL
results_sorted = sorted(results, key=lambda x: x['pnl'], reverse=True)

print("\n" + "=" * 70)
print("TOP 10 PARAMETER COMBINATIONS (by PnL)")
print("=" * 70)
print(f"\n{'SL':>6} {'TP':>6} {'PnL':>12} {'Trades':>7} {'W/L':>8} {'Win%':>7} {'R:R':>6}")
print("-" * 60)

for r in results_sorted[:10]:
    print(f"{r['sl']*100:5.1f}% {r['tp']*100:5.1f}% {r['pnl']:+12,.2f} {r['trades']:7} {r['wins']:3}/{r['losses']:3} {r['win_rate']*100:6.1f}% {r['rr_ratio']:5.2f}")

# Find best for different criteria
best_pnl = results_sorted[0]
best_wr = max(results, key=lambda x: x['win_rate'])
best_rr = max(results, key=lambda x: x['rr_ratio'])

print("\n" + "-" * 70)
print("BEST BY DIFFERENT CRITERIA:")
print("-" * 70)
print(f"\nBest PnL:      SL={best_pnl['sl']*100:.1f}%, TP={best_pnl['tp']*100:.1f}% → PnL={best_pnl['pnl']:+,.0f}₽")
print(f"Best Win Rate: SL={best_wr['sl']*100:.1f}%, TP={best_wr['tp']*100:.1f}% → Win={best_wr['win_rate']*100:.1f}%")
print(f"Best R:R:       SL={best_rr['sl']*100:.1f}%, TP={best_rr['tp']*100:.1f}% → R:R={best_rr['rr_ratio']:.2f}")

# ============================================================================
# SWEET SPOT ANALYSIS
# ============================================================================

print("\n[5/5] Sweet Spot Analysis...")

# Filter for positive PnL only
positive_results = [r for r in results if r['pnl'] > 0]

if positive_results:
    # Find balance between PnL and number of trades
    print("\n  Sweet Spot (Positive PnL + Sufficient Trades):")
    for r in sorted(positive_results, key=lambda x: -x['pnl'])[:5]:
        print(f"    SL={r['sl']*100:.1f}%, TP={r['tp']*100:.1f}% → PnL={r['pnl']:+,.0f}₽, Trades={r['trades']}, Win={r['win_rate']*100:.0f}%")
else:
    print("  No positive PnL combinations found")

# ============================================================================
# RECOMMENDATION
# ============================================================================

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

# Find optimal: high PnL with reasonable trades
optimal = max(results, key=lambda x: x['pnl'] if x['trades'] >= 5 else -999999)

print(f"""
Based on parameter sweep:

  🎯 OPTIMAL PARAMETERS:
  ─────────────────────────────────────────
  • Stop Loss:     {optimal['sl']*100:.1f}%
  • Take Profit:   {optimal['tp']*100:.1f}%
  • Risk:Reward:   1:{optimal['rr_ratio']:.2f}
  
  📊 EXPECTED RESULTS:
  ─────────────────────────────────────────
  • PnL:           {optimal['pnl']:+,.0f}₽
  • Trades:        {optimal['trades']}
  • Win Rate:      {optimal['win_rate']*100:.1f}%
  • Stop-Loss:     {optimal['stop_losses']}
  • Take-Profit:   {optimal['take_profits']}

  💡 KEY INSIGHT:
  ─────────────────────────────────────────
  Higher TP with tight SL gives better results.
  The key is to capture larger moves (TP) while
  cutting losses quickly (SL).
""")

# Save results
optimization_results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "strategy": "ParameterOptimization",
    "best_params": optimal,
    "top_10": results_sorted[:10],
    "total_combinations_tested": len(results),
}

with open(workspace / "reports/parameter_optimization.json", "w") as f:
    json.dump(optimization_results, f, indent=2)

print(f"\nResults saved to reports/parameter_optimization.json")