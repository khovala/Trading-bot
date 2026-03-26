#!/usr/bin/env python3
"""
Visualization: Ensemble Model Results with Entry/Exit Points
=============================================================
Shows trades with SL/TP levels on price charts
"""

import sys
sys.path.insert(0, '/Users/sergeyeliseev/moex-sandbox-platform')

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timezone

workspace = Path('/Users/sergeyeliseev/moex-sandbox-platform')

# Load data
print("[1/5] Loading data...")
from src.data.feature_store.io import read_parquet

test_records = read_parquet(workspace / "data/processed/merged/test.parquet")
df = pd.DataFrame(test_records)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Aggregate to hourly
print("[2/5] Aggregating to hourly...")
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
df_hourly['timestamp'] = pd.to_datetime(df_hourly['timestamp']).dt.tz_localize(None)
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

# Optimal parameters
sl = 0.01  # 1%
tp = 0.05  # 5%
trailing = 0.015  # 1.5%

# Trading costs
initial_capital = 1_000_000
commission_bps = 5.0
slippage_bps = 5.0

# Track trades for visualization
print("[3/5] Identifying trades...")

trades_data = []

for ticker in sorted(df_hourly['ticker'].unique()):
    tdf = df_hourly[df_hourly['ticker'] == ticker].copy()
    tdf = tdf.sort_values('timestamp').reset_index(drop=True)
    
    if len(tdf) < 20:
        continue
    
    position_size = (initial_capital / 10) * 0.10  # 10% of 1/10 capital
    position = 0
    entry_price = 0
    entry_time = None
    max_price = 0
    qty = 0
    
    prices = tdf['close'].values
    signals = tdf['signal'].values
    timestamps = tdf['timestamp'].values
    
    cumulative_pnl = 0  # Track unrealized PnL
    
    for i in range(1, len(tdf)):
        curr_price = prices[i]
        prev_price = prices[i-1]
        curr_time = timestamps[i]
        
        # Track cumulative PnL while in position
        if position > 0:
            cumulative_pnl += (curr_price - prev_price) * qty
        
        # Exit logic
        if position > 0:
            price_change = (curr_price - entry_price) / entry_price
            
            exit_type = None
            
            if price_change <= -sl:
                exit_type = 'STOP_LOSS'
            elif price_change >= tp:
                exit_type = 'TAKE_PROFIT'
            elif curr_price < max_price * (1 - trailing) and max_price > entry_price * 1.02:
                exit_type = 'TRAILING'
            
            if exit_type:
                # Calculate duration
                try:
                    duration = (pd.Timestamp(curr_time) - pd.Timestamp(entry_time)).total_seconds() / 3600
                except:
                    duration = 0
                
                qty = int(position_size / entry_price) if entry_price > 0 else 0
                
                # Calculate exit price based on exit type (matching original backtest logic)
                if exit_type == 'STOP_LOSS':
                    exit_price_calc = entry_price * (1 - sl)
                elif exit_type == 'TAKE_PROFIT':
                    exit_price_calc = entry_price * (1 + tp)
                else:  # TRAILING
                    exit_price_calc = max_price * (1 - trailing)
                
                # Add cumulative PnL while in position
                total_pnl = cumulative_pnl
                
                # Calculate realized PnL at exit
                entry_cost = qty * entry_price
                exit_revenue = qty * exit_price_calc
                costs = entry_cost * (commission_bps + slippage_bps) / 10000
                realized_pnl = exit_revenue - entry_cost - costs
                
                pnl_rub = total_pnl + realized_pnl
                
                trades_data.append({
                    'ticker': ticker,
                    'entry_time': entry_time,
                    'exit_time': curr_time,
                    'entry_price': entry_price,
                    'exit_price': curr_price,
                    'pnl': pnl_rub,
                    'pnl_pct': (curr_price - entry_price) / entry_price * 100,
                    'exit_type': exit_type,
                    'duration_hours': duration,
                    'qty': qty
                })
                position = 0
                entry_time = None
                cumulative_pnl = 0  # Reset for next trade
            else:
                max_price = max(max_price, curr_price)
        
        # Entry logic
        if signals[i] == 1 and position == 0:
            position = 1
            qty = int(position_size / curr_price)
            entry_price = curr_price * 1.0005  # Include slippage
            entry_time = curr_time
            max_price = entry_price

# Convert timestamps to datetime if needed
for trade in trades_data:
    try:
        duration = (pd.Timestamp(trade['exit_time']) - pd.Timestamp(trade['entry_time'])).total_seconds() / 3600
    except:
        duration = 0
    trade['duration_hours'] = duration

# Create trades DataFrame
trades_df = pd.DataFrame(trades_data)

# Convert timezone-aware to timezone-naive for plotting
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.tz_localize(None)
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.tz_localize(None)

print(f"\n[4/5] Total trades: {len(trades_df)}")
print(f"Exit types: {trades_df['exit_type'].value_counts().to_dict()}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("[5/5] Creating visualization...")

# Create figure with subplots for each ticker that had trades
tickers_with_trades = trades_df['ticker'].unique()
# Show ALL tickers (not just those with trades)
all_tickers = sorted(df_hourly['ticker'].unique())
n_tickers = len(all_tickers)

fig, axes = plt.subplots(n_tickers, 1, figsize=(16, 5 * n_tickers))
if n_tickers == 1:
    axes = [axes]

# Color scheme
colors = {
    'price': '#2C3E50',
    'entry': '#27AE60',
    'exit_stop': '#E74C3C',
    'exit_tp': '#3498DB',
    'exit_trailing': '#F39C12',
    'hold': '#95A5A6',
    'no_signal': '#BDC3C7',
}

for idx, ticker in enumerate(all_tickers):
    ax = axes[idx]
    
    # Get ticker data
    tdf = df_hourly[df_hourly['ticker'] == ticker].copy()
    tdf = tdf.sort_values('timestamp').reset_index(drop=True)
    
    # Plot price
    ax.plot(tdf['timestamp'], tdf['close'], color=colors['price'], linewidth=1.5, label='Price')
    
    # Get trades for this ticker
    ticker_trades = trades_df[trades_df['ticker'] == ticker]
    
    # Plot entry/exit points
    for _, trade in ticker_trades.iterrows():
        # Entry point
        ax.axvline(x=trade['entry_time'], color=colors['entry'], linestyle='--', alpha=0.7, linewidth=1)
        ax.scatter([trade['entry_time']], [trade['entry_price']], 
                  color=colors['entry'], s=100, marker='^', zorder=5, label='Entry' if idx == 0 else '')
        
        # Exit point
        exit_color = {
            'STOP_LOSS': colors['exit_stop'],
            'TAKE_PROFIT': colors['exit_tp'],
            'TRAILING': colors['exit_trailing']
        }[trade['exit_type']]
        
        ax.axvline(x=trade['exit_time'], color=exit_color, linestyle='--', alpha=0.7, linewidth=1)
        ax.scatter([trade['exit_time']], [trade['exit_price']], 
                  color=exit_color, s=100, marker='v', zorder=5, 
                  label=f"Exit ({trade['exit_type']})" if idx == 0 else '')
        
        # Draw PnL annotation
        pnl_color = '#27AE60' if trade['pnl'] > 0 else '#E74C3C'
        ax.annotate(f"{trade['pnl']:+.1f}%", 
                   xy=(trade['exit_time'], trade['exit_price']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, color=pnl_color, fontweight='bold')
        
        # Draw SL/TP levels
        sl_price = trade['entry_price'] * (1 - sl)
        tp_price = trade['entry_price'] * (1 + tp)
        
        ax.axhline(y=sl_price, color=colors['exit_stop'], linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=tp_price, color=colors['exit_tp'], linestyle=':', alpha=0.5, linewidth=1)
    
    # Fill hold periods (between entry and exit)
    for _, trade in ticker_trades.iterrows():
        ax.fill_between(tdf['timestamp'], tdf['close'].min(), tdf['close'].max(),
                       where=(tdf['timestamp'] >= trade['entry_time']) & (tdf['timestamp'] <= trade['exit_time']),
                       alpha=0.1, color=colors['hold'])
    
    # Add markers for all signals (not just executed trades)
    ticker_data = tdf[tdf['signal'] == 1]
    if len(ticker_data) > 0:
        ax.scatter(ticker_data['timestamp'], ticker_data['close'], 
                  color=colors['entry'], s=50, marker='^', alpha=0.5, label='Signal' if idx == 0 else '')
    
    # Calculate stats for this ticker
    ticker_signal_count = len(tdf[tdf['signal'] == 1])
    ticker_trade_count = len(ticker_trades)
    ticker_pnl_sum = ticker_trades['pnl'].sum() if len(ticker_trades) > 0 else 0
    
    # Status indicator
    if ticker_trade_count > 0:
        status = f"[TRADED] {ticker_trade_count} trades"
    elif ticker_signal_count > 0:
        status = f"[SIGNAL] {ticker_signal_count} signals (no trades)"
    else:
        status = f"[NO SIGNAL] filters not met"
    
    ax.set_title(f'{ticker} - {status} | PnL: {ticker_pnl_sum:+,.0f}₽', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Price (₽)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(workspace / 'backtest_plots/ensemble_trades_visualization.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: backtest_plots/ensemble_trades_visualization.png")

# ============================================================================
# SUMMARY PLOT
# ============================================================================

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# 1. PnL by Ticker (in rubles)
ax1 = axes2[0, 0]
ticker_pnl = trades_df.groupby('ticker')['pnl'].sum().sort_values(ascending=True)
colors_bar = ['#27AE60' if v > 0 else '#E74C3C' for v in ticker_pnl.values]
ax1.barh(ticker_pnl.index, ticker_pnl.values, color=colors_bar)
ax1.set_xlabel('PnL (₽)')
ax1.set_title('PnL by Ticker', fontweight='bold')
ax1.axvline(x=0, color='black', linewidth=0.5)

# 2. Exit Types Distribution
ax2 = axes2[0, 1]
exit_counts = trades_df['exit_type'].value_counts()
colors_pie = [colors['exit_stop'], colors['exit_tp'], colors['exit_trailing']]
ax2.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.0f%%', 
        colors=colors_pie, startangle=90)
ax2.set_title('Exit Types Distribution', fontweight='bold')

# 3. Win Rate by Exit Type
ax3 = axes2[1, 0]
exit_stats = trades_df.groupby('exit_type').agg({
    'pnl': ['count', 'sum', 'mean']
}).reset_index()
exit_stats.columns = ['exit_type', 'count', 'total_pnl', 'avg_pnl']

x = np.arange(len(exit_stats))
width = 0.35
bars1 = ax3.bar(x - width/2, exit_stats['count'], width, label='Trades', color='#3498DB')
ax3.set_ylabel('Number of Trades')
ax3.set_xticks(x)
ax3.set_xticklabels(exit_stats['exit_type'])
ax3.set_title('Trades by Exit Type', fontweight='bold')

ax3_twin = ax3.twinx()
ax3_twin.plot(x, exit_stats['avg_pnl'], 'o-', color='#27AE60', linewidth=2, markersize=8, label='Avg PnL %')
ax3_twin.set_ylabel('Avg PnL (%)')

# 4. Cumulative PnL over time
ax4 = axes2[1, 1]
trades_df_sorted = trades_df.sort_values('entry_time')
trades_df_sorted['cum_pnl'] = trades_df_sorted['pnl'].cumsum()
ax4.plot(range(len(trades_df_sorted)), trades_df_sorted['cum_pnl'], 'o-', 
        color='#2C3E50', linewidth=2, markersize=6)
ax4.fill_between(range(len(trades_df_sorted)), trades_df_sorted['cum_pnl'], 
                where=trades_df_sorted['cum_pnl'] > 0, alpha=0.3, color='#27AE60')
ax4.fill_between(range(len(trades_df_sorted)), trades_df_sorted['cum_pnl'], 
                where=trades_df_sorted['cum_pnl'] < 0, alpha=0.3, color='#E74C3C')
ax4.set_xlabel('Trade Number')
ax4.set_ylabel('Cumulative PnL (₽)')
ax4.set_title('Cumulative PnL', fontweight='bold')
ax4.axhline(y=0, color='black', linewidth=0.5)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(workspace / 'backtest_plots/ensemble_summary.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: backtest_plots/ensemble_summary.png")

# ============================================================================
# PRINT SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("TRADE SUMMARY")
print("=" * 70)

print(f"\n📊 Overall Statistics:")
print(f"   Total Trades: {len(trades_df)}")
print(f"   Winning Trades: {(trades_df['pnl'] > 0).sum()}")
print(f"   Losing Trades: {(trades_df['pnl'] < 0).sum()}")
print(f"   Win Rate: {(trades_df['pnl'] > 0).mean() * 100:.1f}%")
print(f"   Total PnL: {trades_df['pnl'].sum():+,.2f}₽")
print(f"   Avg PnL per Trade: {trades_df['pnl'].mean():+,.2f}₽")

print(f"\n📋 Exit Analysis:")
for exit_type, count in trades_df['exit_type'].value_counts().items():
    avg_pnl = trades_df[trades_df['exit_type'] == exit_type]['pnl'].mean()
    print(f"   {exit_type}: {count} trades ({count/len(trades_df)*100:.0f}%), Avg PnL: {avg_pnl:+,.0f}₽")

print(f"\n📈 By Ticker:")
for ticker, group in trades_df.groupby('ticker'):
    print(f"   {ticker}: {len(group)} trades, PnL: {group['pnl'].sum():+,.0f}₽")

# Save trade log
trades_df.to_csv(workspace / 'reports/trade_log.csv', index=False)
print(f"\n✅ Trade log saved to reports/trade_log.csv")

print("\n✅ Visualization complete!")