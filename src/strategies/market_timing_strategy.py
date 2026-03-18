"""
Production Trading Strategy with Market Timing
Uses daily aggregated data with market regime filtering.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import pickle
import json

from src.backtesting.engine import BacktestConfig, run_backtest


@dataclass
class MarketTimingConfig:
    position_size_pct: float = 0.2
    market_threshold: float = -0.01
    use_market_timing: bool = True
    rsi_oversold: float = 40
    rsi_overbought: float = 60
    momentum_lookback: int = 5
    signal_threshold: float = 0.01
    min_tickers: int = 3
    max_tickers: int = 5


@dataclass
class BacktestResult:
    total_pnl: float
    trades: int
    sharpe: float
    max_drawdown: float
    tickers_traded: list
    details: dict


class MarketTimingStrategy:
    def __init__(
        self,
        model_path: Path | None = None,
        config: MarketTimingConfig | None = None,
    ):
        self.model_path = model_path
        self.config = config or MarketTimingConfig()
        self.model = None
        self.feature_cols = None
        self.is_loaded = False
        
    def load(self) -> bool:
        if self.model_path and self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get('model')
                    self.feature_cols = data.get('features')
                self.is_loaded = True
                return True
            except Exception as e:
                print(f'Failed to load model: {e}')
        self.is_loaded = True
        return False
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df[mask].sort_values('date')
            
            df.loc[mask, 'return_lag_1'] = ticker_data['daily_return'].shift(1).fillna(0)
            df.loc[mask, 'return_lag_2'] = ticker_data['daily_return'].shift(2).fillna(0)
            df.loc[mask, 'return_lag_3'] = ticker_data['daily_return'].shift(3).fillna(0)
            df.loc[mask, 'vol_lag_1'] = ticker_data['rolling_volatility_20_last'].shift(1).fillna(0)
        
        df['macd_signal_interaction'] = df['macd_last'] * df['macd_signal_last']
        return df.fillna(0)
    
    def calculate_ticker_score(self, row: dict) -> float:
        score = 0.0
        
        # RSI component (inverse - lower is better for long)
        rsi = row.get('rsi_14_last', 50)
        if rsi < self.config.rsi_oversold:
            score += 1
        elif rsi > self.config.rsi_overbought:
            score -= 1
        
        # Momentum component
        mom = row.get('momentum_10d', 0) or row.get('momentum_10_last', 0)
        if mom > self.config.signal_threshold:
            score += 1
        elif mom < -self.config.signal_threshold:
            score -= 1
        
        # Z-score component
        zscore = row.get('zscore_20_last', 0)
        if zscore < -1.5:
            score += 0.5  # Oversold
        elif zscore > 1.5:
            score -= 0.5  # Overbought
        
        # MACD component
        macd = row.get('macd_last', 0)
        macd_sig = row.get('macd_signal_last', 0)
        if macd > macd_sig:
            score += 0.5
        elif macd < macd_sig:
            score -= 0.5
        
        return score
    
    def get_market_signal(self, df: pd.DataFrame, date: Any) -> int:
        if not self.config.use_market_timing:
            return 1  # Neutral - allow all trades
        
        market_data = df[df['date'] == date]
        if len(market_data) == 0:
            return 1
        
        market_mom = market_data['market_momentum_5d'].iloc[0]
        
        if market_mom < self.config.market_threshold:
            return -1  # Bearish market - reduce exposure
        elif market_mom > -self.config.market_threshold:
            return 1  # Bullish or neutral
        
        return 0  # No signal
    
    def select_tickers(self, df: pd.DataFrame, date: Any, direction: int) -> list:
        date_data = df[df['date'] == date]
        
        if len(date_data) == 0:
            return []
        
        scores = []
        for _, row in date_data.iterrows():
            score = self.calculate_ticker_score(row.to_dict())
            scores.append((row['ticker'], score, row['daily_return']))
        
        if direction > 0:
            scores.sort(key=lambda x: x[1], reverse=True)
        else:
            scores.sort(key=lambda x: x[1])
        
        tickers = [s[0] for s in scores[:self.config.max_tickers]]
        return tickers
    
    def run_backtest(self, df: pd.DataFrame) -> BacktestResult:
        df = self.add_features(df)
        dates = sorted(df['date'].unique())
        
        cfg = BacktestConfig(
            initial_cash=1_000_000,
            commission_bps=5,
            slippage_bps=5,
            lot_size=1,
            execution_delay_bars=1,
            position_size_pct=self.config.position_size_pct,
            target_position_column='policy_target_position',
        )
        
        results = []
        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker].sort_values('date').reset_index(drop=True)
            rows = ticker_df.to_dict('records')
            
            for row in rows:
                date = row['date']
                market_signal = self.get_market_signal(df, date)
                ticker_score = self.calculate_ticker_score(row)
                
                if market_signal > 0 and ticker_score > 0:
                    row['policy_target_position'] = 1.0
                elif market_signal < 0 and ticker_score < 0:
                    row['policy_target_position'] = -1.0
                else:
                    row['policy_target_position'] = 0.0
            
            result = run_backtest(rows, cfg)
            results.append({
                'ticker': ticker,
                'pnl': result['summary']['pnl'],
                'trades': len(result['trade_log']),
            })
        
        total_pnl = sum(r['pnl'] for r in results)
        
        return BacktestResult(
            total_pnl=total_pnl,
            trades=sum(r['trades'] for r in results),
            sharpe=0.0,
            max_drawdown=0.0,
            tickers_traded=[r['ticker'] for r in results if r['pnl'] != 0],
            details={r['ticker']: r for r in results},
        )


def main():
    from src.strategies.production_strategy import add_features
    
    df = pd.read_parquet('data/processed/merged/daily_aggregated.parquet')
    print(f'Daily data: {len(df)} rows, {df[\"date\"].nunique()} days')
    
    strategy = MarketTimingStrategy()
    
    print('\\n=== Market Timing Strategy ===')
    result = strategy.run_backtest(df)
    
    print(f'Total PnL: {result.total_pnl:.0f}')
    print(f'Total Trades: {result.trades}')
    print(f'Tickers traded: {result.tickers_traded}')
    
    for ticker, res in sorted(result.details.items(), key=lambda x: x[1]['pnl'], reverse=True):
        print(f'  {ticker}: PnL={res[\"pnl\"]:.0f}, Trades={res[\"trades\"]}')


if __name__ == '__main__':
    main()
