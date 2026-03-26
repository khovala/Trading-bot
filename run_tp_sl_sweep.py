#!/usr/bin/env python3
"""
TP/SL PARAMETER SWEEP
=====================
Test multiple SL/TP combinations to find optimal win rate and PnL
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
print("TP/SL PARAMETER SWEEP")
print("=" * 70)

# ============================================================================
# DATA
# ============================================================================

print("\n[1/4] Loading data...")
from src.data.feature_store.io import read_parquet

test_records = read_parquet(workspace / "data/processed/merged/test.parquet")
df = pd.DataFrame(test_records)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Aggregate to hourly
df['hour_bucket'] = df['timestamp'].dt.floor('h')

agg_dict = {
    'close': 'last',
    'volume': 'sum',
    'return_1': 'sum',
    'rsi_14': 'last',
    'zscore_20': 'last',
    'volume_ratio_20': 'last',
}

df_hourly = df.groupby(['ticker', 'hour_bucket']).agg(agg_dict).reset_index()
df_hourly.rename(columns={'hour_bucket': 'timestamp'}, inplace=True)
df_hourly = df_hourly.sort_values(['ticker', 'timestamp']).reset_index(drop=True)

# Add filters
df_hourly['volume_above_avg'] = df_hourly['volume_ratio_20'] >= 1.1
df_hourly['hour'] = df_hourly['timestamp'].dt.hour
df_hourly['safe_hour'] = (df_hourly['hour'] >= 10) & (df_hourly['hour'] <= 17)

# Market regime
market_momentum = df_hourly.groupby('timestamp')['return_1'].mean().to_dict()
df_hourly['market_momentum'] = df_hourly['timestamp'].map(market_momentum)
df_hourly['market_safe'] = df_hourly['market_momentum'] > -0.02

# Signal
def generate_signal(row):
    if row['ticker'] not in ['NVTK', 'YNDX', 'TATN']:
        return 0
    if not row.get('market_safe', True):
        return 0
    if not row.get('safe_hour', True):
        return 0
    
    rsi = row.get('rsi_14', 50)
    zscore = row.get('zscore_20', 0)
    
    if rsi < 30 and zscore < -1.8:
        return 1
    return 0

df_hourly['signal'] = df_hourly.apply(generate_signal, axis=1)

print(f"  Signals: {df_hourly['signal'].sum()}")

# ============================================================================
# PARAMETER SWEEP
# ============================================================================

print("\n[2/4] Running parameter sweep...")

initial_capital = 1_000_000
position_size_pct = 0.35
commission_bps = 5.0
slippage_bps = 5.0

# Test combinations: SL from 0.5% to 2%, TP from 1.5% to 6%
sl_levels = [0.005, 0.008, 0.010, 0.012, 0.015, 0.020]
tp_levels = [0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060]

results = []

for sl in sl_levels:
    for tp in tp_levels:
        if tp <= sl:
            continue
        
        total_pnl = 0
        total_trades = 0
        total_wins = 0
        total_losses = 0
        
        for ticker in ['NVTK', 'YNDX', 'TATN']:
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
                    
                    if price_change <= -sl:
                        gross_pnl = position * (entry_price * (1 - sl) - entry_price)
                        pnl += gross_pnl - (position * entry_price * (commission_bps + slippage_bps) / 10000)
                        losses += 1
                        position = 0
                    
                    elif price_change >= tp:
                        gross_pnl = position * (entry_price * (1 + tp) - entry_price)
                        pnl += gross_pnl - (position * entry_price * (commission_bps + slippage_bps) / 10000)
                        wins += 1
                        position = 0
                    
                    max_price = max(max_price, curr_price)
                    pnl += (curr_price - prev_price) * position
                
                if signal == 1 and position == 0:
                    position_size = capital_per_ticker * position_size_pct
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
            
            total_pnl += pnl
            total_trades += trades
            total_wins += wins
            total_losses += losses
        
        win_rate = total_wins / max(1, total_trades) if total_trades > 0 else 0
        rr = tp / sl
        
        results.append({
            'sl': sl,
            'tp': tp,
            'pnl': total_pnl,
            'trades': total_trades,
            'win_rate': win_rate,
            'rr': rr,
            'avg_pnl': total_pnl / max(1, total_trades) if total_trades > 0 else 0,
        })

# ============================================================================
# RESULTS
# ============================================================================

print("\n[3/4] Sorting by PnL...")
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('pnl', ascending=False)

print("\n" + "=" * 70)
print("TOP 15 COMBINATIONS (by PnL)")
print("=" * 70)

print(f"\n{'SL':>6} {'TP':>6} {'PnL':>12} {'Trades':>8} {'Win%':>8} {'R:R':>6} {'Avg/Trade':>12}")
print("-" * 70)

for i, row in results_df.head(15).iterrows():
    print(f"{row['sl']*100:5.1f}% {row['tp']*100:5.1f}% {row['pnl']:+12,.0f}₽ {int(row['trades']):8} {row['win_rate']*100:7.1f}% {row['rr']:5.2f} {row['avg_pnl']:+11,.0f}₽")

print("\n" + "=" * 70)
print("TOP 10 BY WIN RATE (min 5 trades)")
print("=" * 70)

filtered = results_df[results_df['trades'] >= 5].sort_values('win_rate', ascending=False)

print(f"\n{'SL':>6} {'TP':>6} {'PnL':>12} {'Trades':>8} {'Win%':>8} {'R:R':>6} {'Avg/Trade':>12}")
print("-" * 70)

for i, row in filtered.head(10).iterrows():
    print(f"{row['sl']*100:5.1f}% {row['tp']*100:5.1f}% {row['pnl']:+12,.0f}₽ {int(row['trades']):8} {row['win_rate']*100:7.1f}% {row['rr']:5.2f} {row['avg_pnl']:+11,.0f}₽")

# Best combo
best = results_df.iloc[0]
best_wr = filtered.iloc[0] if len(filtered) > 0 else best

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print(f"\n🏆 Best PnL:")
print(f"   SL: {best['sl']*100:.1f}% | TP: {best['tp']*100:.1f}%")
print(f"   PnL: {best['pnl']:+,.0f}₽ | Win Rate: {best['win_rate']*100:.1f}%")

print(f"\n🎯 Best Win Rate (5+ trades):")
print(f"   SL: {best_wr['sl']*100:.1f}% | TP: {best_wr['tp']*100:.1f}%")
print(f"   PnL: {best_wr['pnl']:+,.0f}₽ | Win Rate: {best_wr['win_rate']*100:.1f}%")

# ============================================================================
# SAVE
# ============================================================================

print("\n[4/4] Saving results...")

output = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "sweep_config": {
        "sl_levels": [s*100 for s in sl_levels],
        "tp_levels": [t*100 for t in tp_levels],
        "position_size_pct": position_size_pct,
        "tickers": ['NVTK', 'YNDX', 'TATN'],
    },
    "results": results_df.to_dict('records'),
    "best_by_pnl": best.to_dict(),
    "best_by_winrate": best_wr.to_dict(),
}

with open(workspace / "reports/tp_sl_sweep.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Results saved to reports/tp_sl_sweep.json")
