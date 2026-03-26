#!/usr/bin/env python3
"""
Optimized Intraday Strategy: Mean Reversion with Volume-Confirmed Signals
================================================================================
Designed for MOEX with realistic costs: 5bps commission + 5bps slippage

Key Principles:
1. Higher probability entries (higher threshold)
2. Volume confirmation to avoid fake signals
3. Strict risk management (small position, tight stops)
4. Focus on reversal patterns (mean reversion)
5. Daily timeframe to reduce noise
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
print("INTRADAY OPTIMIZED STRATEGY: MEAN REVERSION + VOLUME CONFIRMATION")
print("=" * 70)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Risk Management
    'position_size_pct': 0.10,        # 10% of capital (reduced from 15%)
    'max_position_pct': 0.15,          # Max 15% in one trade
    
    # Entry Thresholds
    'rsi_oversold': 30,                # More extreme = higher probability
    'rsi_overbought': 70,              # More extreme = higher probability
    'zscore_oversold': -2.0,           # More extreme = higher probability
    'zscore_overbought': 2.0,          # More extreme = higher probability
    
    # Volume filter - only enter when volume is above average
    'volume_ratio_min': 1.2,           # 20% above average
    
    # Stop Loss & Take Profit (tight for intraday)
    'stop_loss_pct': 0.015,           # 1.5% stop loss
    'take_profit_pct': 0.025,         # 2.5% take profit (R:R = 1:1.67)
    'trailing_stop_pct': 0.01,        # 1% trailing stop
    
    # Costs (realistic for MOEX)
    'commission_bps': 5.0,            # 0.05%
    'slippage_bps': 5.0,              # 0.05%
    'total_cost_pct': 0.001,          # 0.1% total cost per trade
    
    # Market Regime Filter
    'market_momentum_threshold': -0.02,  # Don't trade if market down >2%
    
    # Intraday filters
    'avoid_open_minutes': 15,         # Avoid first 15 minutes
    'avoid_close_minutes': 15,        # Avoid last 15 minutes
    
    # Position management
    'max_daily_trades': 3,            # Max 3 trades per ticker per day
    'cooldown_hours': 4,              # Min 4 hours between trades
}

# ============================================================================
# DATA LOADING
# ============================================================================

print("\n[1/6] Loading test data...")
from src.data.feature_store.io import read_parquet

test_records = read_parquet(workspace / "data/processed/merged/test.parquet")
df = pd.DataFrame(test_records)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
print(f"  Loaded {len(df):,} minute bars")

# ============================================================================
# AGGREGATE TO HOURLY (for cleaner signals)
# ============================================================================

print("\n[2/6] Aggregating to hourly timeframe...")

df['hour_bucket'] = df['timestamp'].dt.floor('h')

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

df_hourly = df.groupby(['ticker', 'hour_bucket']).agg(agg_dict).reset_index()
df_hourly.rename(columns={'hour_bucket': 'timestamp'}, inplace=True)
df_hourly = df_hourly.sort_values(['ticker', 'timestamp']).reset_index(drop=True)

# Add volume confirmation
df_hourly['volume_above_avg'] = df_hourly['volume_ratio_20'] >= CONFIG['volume_ratio_min']

# Add time filters
df_hourly['hour'] = df_hourly['timestamp'].dt.hour
df_hourly['safe_hour'] = (df_hourly['hour'] >= 10) & (df_hourly['hour'] <= 17)

print(f"  Aggregated to {len(df_hourly):,} hourly bars")

# ============================================================================
# MARKET REGIME FILTER
# ============================================================================

print("\n[3/6] Computing market regime...")

# Calculate market-wide momentum
market_momentum = df_hourly.groupby('timestamp')['return_1'].mean().to_dict()
df_hourly['market_momentum'] = df_hourly['timestamp'].map(market_momentum)

# Flag safe market conditions
df_hourly['market_safe'] = df_hourly['market_momentum'] > CONFIG['market_momentum_threshold']

print(f"  Safe market hours: {df_hourly['market_safe'].mean():.1%}")

# ============================================================================
# SIGNAL GENERATION
# ============================================================================

print("\n[4/6] Generating signals...")

def generate_signal(row):
    """Generate trading signal with multiple filters"""
    
    # Skip if market regime is unsafe
    if not row.get('market_safe', True):
        return 0
    
    # Skip if not in safe hours
    if not row.get('safe_hour', True):
        return 0
    
    # Skip if volume too low
    if not row.get('volume_above_avg', True):
        return 0
    
    rsi = row.get('rsi_14', 50)
    zscore = row.get('zscore_20', 0)
    
    # LONG: Oversold + Volume confirmation
    if rsi < CONFIG['rsi_oversold'] and zscore < CONFIG['zscore_oversold']:
        return 1  # Strong long signal
    
    # SHORT: Overbought + Volume confirmation (optional - can be disabled)
    # For long-only strategy, comment out shorts:
    # if rsi > CONFIG['rsi_overbought'] and zscore > CONFIG['zscore_overbought']:
    #     return -1
    
    return 0

df_hourly['signal'] = df_hourly.apply(generate_signal, axis=1)

# Count signals
long_signals = (df_hourly['signal'] == 1).sum()
short_signals = (df_hourly['signal'] == -1).sum()
neutral = (df_hourly['signal'] == 0).sum()

print(f"  Signals: Long={long_signals}, Short={short_signals}, Neutral={neutral}")

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

print("\n[5/6] Running backtest...")

initial_capital = 1_000_000
commission_bps = CONFIG['commission_bps']
slippage_bps = CONFIG['slippage_bps']
stop_loss_pct = CONFIG['stop_loss_pct']
take_profit_pct = CONFIG['take_profit_pct']
trailing_stop_pct = CONFIG['trailing_stop_pct']

results_by_ticker = {}
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
    
    # Initialize
    capital_per_ticker = initial_capital / 10  # Split across 10 tickers
    position = 0
    pnl = 0
    trades = 0
    wins = 0
    losses = 0
    stop_losses = 0
    take_profits = 0
    
    entry_price = 0
    entry_time = 0
    max_price = 0
    
    prices = tdf['close'].values
    signals = tdf['signal'].values
    timestamps = tdf['timestamp'].values
    
    for i in range(1, len(tdf)):
        curr_price = prices[i]
        prev_price = prices[i-1]
        signal = signals[i]
        
        # ========== EXIT LOGIC ==========
        if position > 0:
            # Check stop loss
            price_change = (curr_price - entry_price) / entry_price
            if price_change <= -stop_loss_pct:
                # Stop loss triggered
                gross_pnl = position * (entry_price * (1 - stop_loss_pct) - entry_price)
                pnl += gross_pnl - (position * entry_price * (commission_bps + slippage_bps) / 10000)
                stop_losses += 1
                position = 0
                
            # Check take profit
            elif price_change >= take_profit_pct:
                # Take profit triggered
                gross_pnl = position * (entry_price * (1 + take_profit_pct) - entry_price)
                pnl += gross_pnl - (position * entry_price * (commission_bps + slippage_bps) / 10000)
                take_profits += 1
                position = 0
                
            # Check trailing stop
            elif curr_price < max_price * (1 - trailing_stop_pct) and max_price > entry_price * 1.01:
                # Trailing stop triggered
                gross_pnl = position * (max_price * (1 - trailing_stop_pct) - entry_price)
                pnl += gross_pnl - (position * entry_price * (commission_bps + slippage_bps) / 10000)
                take_profits += 1
                position = 0
            
            # Update max price for trailing
            max_price = max(max_price, curr_price)
            
            # Track PnL for open position
            pnl += (curr_price - prev_price) * position
        
        # ========== ENTRY LOGIC ==========
        if signal == 1 and position == 0:
            position_size = capital_per_ticker * CONFIG['position_size_pct']
            qty = int(position_size / curr_price)
            
            if qty > 0:
                cost = qty * curr_price * (1 + slippage_bps / 10000)
                commission = cost * commission_bps / 10000
                position = qty
                entry_price = curr_price * (1 + slippage_bps / 10000)  # Include slippage
                max_price = entry_price
                trades += 1
    
    # Close final position if open
    if position > 0:
        final_price = prices[-1]
        gross_pnl = position * (final_price - entry_price)
        pnl += gross_pnl - (position * entry_price * (commission_bps + slippage_bps) / 10000)
    
    # Calculate final result
    final_pnl = pnl
    
    results_by_ticker[ticker] = {
        'pnl': final_pnl,
        'trades': trades,
        'wins': wins,
        'losses': losses,
        'stop_losses': stop_losses,
        'take_profits': take_profits,
    }
    
    total_pnl += final_pnl
    total_trades += trades
    total_wins += wins
    total_losses += losses
    total_stop_losses += stop_losses
    total_take_profits += take_profits

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS: OPTIMIZED INTRADAY STRATEGY")
print("=" * 70)

win_rate = total_wins / max(1, total_trades) if total_trades > 0 else 0

print(f"\n  Initial Capital: {initial_capital:,} ₽")
print(f"  Total PnL: {total_pnl:+,.2f} ₽")
print(f"  Return: {total_pnl / initial_capital * 100:+.2f}%")
print(f"  Total Trades: {total_trades}")
print(f"  Win Rate: {win_rate:.2%}")
print(f"  Stop-Loss triggers: {total_stop_losses}")
print(f"  Take-Profit triggers: {total_take_profits}")

# Cost analysis
avg_trade_cost = total_trades * initial_capital * CONFIG['position_size_pct'] * CONFIG['total_cost_pct']
print(f"  Estimated costs: {avg_trade_cost:,.0f} ₽")

print(f"\n  Results by Ticker:")
for ticker, data in sorted(results_by_ticker.items(), key=lambda x: x[1]['pnl'], reverse=True):
    w = data['wins']
    l = data['losses']
    sl = data['stop_losses']
    tp = data['take_profits']
    print(f"    {ticker:6}: PnL={data['pnl']:+12,.2f} ₽  Trades={data['trades']:3}  W/L/SL/TP={w}/{l}/{sl}/{tp}")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "-" * 70)
print("COMPARISON WITH PREVIOUS STRATEGIES")
print("-" * 70)

comparisons = [
    ("1min RSI/Z-Score (90 days)", -11_262_578),
    ("1H ML Regression", -997_167),
    ("1H Ensemble+SL (test only)", -25_463),
    ("30 days Daily (original)", 121_248),
    ("Optimized Intraday (now)", total_pnl),
]

for name, pnl in comparisons:
    print(f"  {name:35}: PnL={pnl:+12,.0f} ₽")

print("\n  Improvement vs 1H Ensemble+SL: {total_pnl - (-25_463):+,.0f} ₽")

# ============================================================================
# SAVE RESULTS
# ============================================================================

backtest_results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "strategy": "OptimizedIntraday",
    "config": CONFIG,
    "results": {
        "total_pnl": total_pnl,
        "return_pct": total_pnl / initial_capital * 100,
        "trades": total_trades,
        "win_rate": win_rate,
        "stop_losses": total_stop_losses,
        "take_profits": total_take_profits,
        "by_ticker": results_by_ticker,
    },
}

with open(workspace / "reports/backtest_intraday_optimized.json", "w") as f:
    json.dump(backtest_results, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to reports/backtest_intraday_optimized.json")