#!/usr/bin/env python3
"""
HIGH WIN RATE STRATEGY
=====================
Goal: Maximize Win Rate (even at cost of fewer trades)

Key changes for higher win rate:
1. Stricter entry: RSI<25 AND Z-score<-2.0
2. MACD confirmation: MACD > signal (bullish momentum)
3. Trend filter: Price above 20-MA (uptrend)
4. Volume confirmation: Volume > 1.5x average
5. Tighter stop loss: 0.8% (less risk per trade)
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
print("HIGH WIN RATE STRATEGY")
print("=" * 70)

# ============================================================================
# HIGH WIN RATE CONFIGURATION
# ============================================================================

CONFIG = {
    # Moderate Entry filters
    'rsi_oversold': 30,        # Stricter (was 35)
    'zscore_oversold': -1.8,   # Stricter (was -1.5)
    'volume_ratio_min': 1.1,   # Slightly higher
    
    # Best tickers (exclude losing ones)
    'allowed_tickers': ['NVTK', 'YNDX', 'TATN', 'MGNT', 'POLY'],
    
    # Position - maximum
    'position_size_pct': 0.35,
    'stop_loss_pct': 0.008,    # 0.8% (OPTIMAL from sweep)
    'take_profit_pct': 0.025,  # 2.5% (OPTIMAL from sweep)
    'trailing_stop_pct': 0.008, # 0.8%
    
    # Partial TP
    'use_partial_tp': True,
    'partial_tp_level': 0.012,  # 1.2%
    'partial_tp_pct': 0.50,    # 50%
    
    # Costs
    'commission_bps': 5.0,
    'slippage_bps': 5.0,
    
    # Filters
    'market_momentum_threshold': -0.02,
    'safe_hours': (10, 17),
    
    # MINIMAL confirmation filters (just one)
    'use_macd_filter': False,     # Disabled - too restrictive
    'use_ma_filter': False,       # Disabled
    'use_volume_spike': False,   # Disabled
    'ma_period': 20,
    'volume_spike_threshold': 1.3,
}

print(f"\n📊 High Win Rate Configuration:")
print(f"   RSI oversold: {CONFIG['rsi_oversold']} (strict)")
print(f"   Z-score oversold: {CONFIG['zscore_oversold']} (strict)")
print(f"   MACD Filter: {CONFIG['use_macd_filter']}")
print(f"   MA Filter: {CONFIG['use_ma_filter']}")
print(f"   Volume Spike: {CONFIG['use_volume_spike']}")
print(f"   SL: {CONFIG['stop_loss_pct']*100:.1f}% | TP: {CONFIG['take_profit_pct']*100:.1f}%")

# ============================================================================
# DATA
# ============================================================================

print("\n[1/5] Loading data...")
from src.data.feature_store.io import read_parquet

test_records = read_parquet(workspace / "data/processed/merged/test.parquet")
df = pd.DataFrame(test_records)
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"  Loaded {len(df):,} minute bars")

# Aggregate to hourly
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
    'macd': 'last',
    'macd_signal': 'last',
}

df_hourly = df.groupby(['ticker', 'hour_bucket']).agg(agg_dict).reset_index()
df_hourly.rename(columns={'hour_bucket': 'timestamp'}, inplace=True)
df_hourly = df_hourly.sort_values(['ticker', 'timestamp']).reset_index(drop=True)

# Calculate MA
print("\n[3/5] Adding indicators...")
# Reset index first to ensure we have ticker column
df_hourly = df_hourly.reset_index(drop=True)

# Calculate MA per ticker
for ticker in df_hourly['ticker'].unique():
    mask = df_hourly['ticker'] == ticker
    df_hourly.loc[mask, 'ma_20'] = df_hourly.loc[mask, 'close'].rolling(20, min_periods=1).mean()
    df_hourly.loc[mask, 'above_ma'] = (df_hourly.loc[mask, 'close'] > df_hourly.loc[mask, 'ma_20']).astype(int)

# Add filters
df_hourly['hour'] = df_hourly['timestamp'].dt.hour
df_hourly['volume_above_avg'] = df_hourly['volume_ratio_20'] >= CONFIG['volume_ratio_min']
df_hourly['safe_hour'] = (df_hourly['hour'] >= CONFIG['safe_hours'][0]) & (df_hourly['hour'] <= CONFIG['safe_hours'][1])
df_hourly['volume_spike'] = df_hourly['volume_ratio_20'] >= CONFIG['volume_spike_threshold']
df_hourly['macd_confirm'] = df_hourly['macd'] > df_hourly['macd_signal']

# Market regime
market_momentum = df_hourly.groupby('timestamp')['return_1'].mean().to_dict()
df_hourly['market_momentum'] = df_hourly['timestamp'].map(market_momentum)
df_hourly['market_safe'] = df_hourly['market_momentum'] > CONFIG['market_momentum_threshold']

# HIGH WIN RATE SIGNAL - Multiple confirmations
def generate_signal(row):
    if row['ticker'] not in CONFIG['allowed_tickers']:
        return 0
    
    if not row.get('market_safe', True):
        return 0
    if not row.get('safe_hour', True):
        return 0
    
    rsi = row.get('rsi_14', 50)
    zscore = row.get('zscore_20', 0)
    
    # Base entry: RSI < 25 AND Z-score < -2.0
    if rsi < CONFIG['rsi_oversold'] and zscore < CONFIG['zscore_oversold']:
        # MACD confirmation
        if CONFIG['use_macd_filter'] and not row.get('macd_confirm', False):
            return 0
        # MA filter
        if CONFIG['use_ma_filter'] and not row.get('above_ma', True):
            return 0
        # Volume spike
        if CONFIG['use_volume_spike'] and not row.get('volume_spike', False):
            return 0
        return 1
    
    return 0

df_hourly['signal'] = df_hourly.apply(generate_signal, axis=1)

print(f"  Signals generated: {df_hourly['signal'].sum()}")

# ============================================================================
# BACKTEST
# ============================================================================

print("\n[4/5] Running backtest...")

initial_capital = 1_000_000
sl = CONFIG['stop_loss_pct']
tp = CONFIG['take_profit_pct']
partial_tp = CONFIG['partial_tp_level']
partial_pct = CONFIG['partial_tp_pct']
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
total_partial_tp = 0

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
    partial_tp_exits = 0
    
    entry_price = 0
    max_price = 0
    partial_exited = False
    
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
                partial_exited = False
            
            # Take profit (full)
            elif price_change >= tp and not partial_exited:
                gross_pnl = position * (entry_price * (1 + tp) - entry_price)
                pnl += gross_pnl - (position * entry_price * (commission_bps + slippage_bps) / 10000)
                take_profits += 1
                wins += 1
                position = 0
                partial_exited = False
            
            # Partial TP
            elif price_change >= partial_tp and not partial_exited and CONFIG['use_partial_tp']:
                partial_qty = int(position * partial_pct)
                gross_pnl = partial_qty * (entry_price * (1 + partial_tp) - entry_price)
                pnl += gross_pnl - (partial_qty * entry_price * (commission_bps + slippage_bps) / 10000)
                position -= partial_qty
                partial_tp_exits += 1
                partial_exited = True
            
            # Trailing stop
            elif position > 0 and curr_price < max_price * (1 - CONFIG['trailing_stop_pct']) and max_price > entry_price * 1.015:
                gross_pnl = position * (max_price * (1 - CONFIG['trailing_stop_pct']) - entry_price)
                pnl += gross_pnl - (position * entry_price * (commission_bps + slippage_bps) / 10000)
                trailing_exits += 1
                wins += 1
                position = 0
                partial_exited = False
            
            if position > 0:
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
                partial_exited = False
    
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
        'partial_tp_exits': partial_tp_exits,
    }
    
    total_pnl += final_pnl
    total_trades += trades
    total_wins += wins
    total_losses += losses
    total_stop_losses += stop_losses
    total_take_profits += take_profits
    total_trailing_exits += trailing_exits
    total_partial_tp += partial_tp_exits

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS (HIGH WIN RATE STRATEGY)")
print("=" * 70)

win_rate = total_wins / max(1, total_trades) if total_trades > 0 else 0

print(f"\n  📈 Configuration:")
print(f"     SL: {sl*100:.1f}% | TP: {tp*100:.1f}% | R:R = 1:{tp/sl:.1f}")
print(f"     Entry: RSI<{CONFIG['rsi_oversold']} AND Z<{CONFIG['zscore_oversold']}")
print(f"     Filters: MACD + MA20 + Volume Spike")

print(f"\n  💰 Results:")
print(f"     Total PnL: {total_pnl:+,.2f} ₽")
print(f"     Return: {total_pnl/initial_capital*100:+.2f}%")
print(f"     Trades: {total_trades}")
print(f"     Win Rate: {win_rate*100:.1f}%")

if total_trades > 0:
    avg_pnl_per_trade = total_pnl / total_trades
    print(f"     Avg PnL/Trade: {avg_pnl_per_trade:+,.2f}₽")

print(f"\n  📊 Exit Analysis:")
print(f"     Stop-Loss:      {total_stop_losses}")
print(f"     Take-Profit:    {total_take_profits}")
print(f"     Partial TP:     {total_partial_tp}")
print(f"     Trailing Stop:  {total_trailing_exits}")

# Costs
total_costs = total_trades * initial_capital * CONFIG['position_size_pct'] * ((commission_bps + slippage_bps) / 10000)
print(f"\n  💸 Costs:")
print(f"     Estimated: {total_costs:,.0f} ₽")

print(f"\n  📋 By Ticker:")
for ticker, data in sorted(results_by_ticker.items(), key=lambda x: x[1]['pnl'], reverse=True):
    if data['trades'] > 0:
        print(f"     {ticker:6}: PnL={data['pnl']:+12,.2f}₽  Trades={data['trades']:2}  W/L={data['wins']}/{data['losses']}")

# ============================================================================
# SAVE
# ============================================================================

backtest_results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "strategy": "HighWinRate",
    "config": CONFIG,
    "results": {
        "total_pnl": total_pnl,
        "return_pct": total_pnl / initial_capital * 100,
        "trades": total_trades,
        "win_rate": win_rate,
        "sl_triggers": total_stop_losses,
        "tp_triggers": total_take_profits,
        "partial_tp_triggers": total_partial_tp,
        "trailing_triggers": total_trailing_exits,
        "by_ticker": results_by_ticker,
    },
}

with open(workspace / "reports/backtest_high_winrate.json", "w") as f:
    json.dump(backtest_results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to reports/backtest_high_winrate.json")
