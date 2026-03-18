"""
Ensemble Strategy: Combines ML model predictions with simple rules.
Low-frequency trading to minimize transaction costs.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from dataclasses import dataclass
from typing import Any

from src.backtesting.engine import BacktestConfig, run_backtest


FEATURE_COLS = [
    'rolling_volatility_20', 'momentum_10', 'rsi_14', 'macd', 'macd_signal',
    'zscore_20', 'volume_ratio_20',
    'return_lag_1', 'return_lag_2', 'return_lag_5',
    'volatility_lag_1', 'rsi_lag_1',
    'macd_momentum_interaction', 'volume_volatility_interaction',
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in [1, 2, 5]:
        df[f'return_lag_{lag}'] = df.groupby('ticker')['return_1'].shift(lag).fillna(0)
        df[f'volatility_lag_{lag}'] = df.groupby('ticker')['rolling_volatility_20'].shift(lag).fillna(0)
        df[f'rsi_lag_{lag}'] = df.groupby('ticker')['rsi_14'].shift(lag).fillna(50)
    
    df['macd_momentum_interaction'] = df['macd'] * df['momentum_10']
    df['volume_volatility_interaction'] = df['volume_ratio_20'] * df['rolling_volatility_20']
    return df.fillna(0)


@dataclass
class TradingSignal:
    direction: int  # 1 = long, -1 = short, 0 = neutral
    confidence: float
    source: str  # 'model', 'rsi', 'macd', 'trend', 'combined'


class EnsembleStrategy:
    def __init__(
        self,
        model_path: Path | None = None,
        position_threshold: float = 0.0003,
        hold_bars: int = 30,
        confidence_threshold: float = 0.4,
        commission_bps: float = 5,
        slippage_bps: float = 5,
        position_size_pct: float = 0.3,
    ):
        self.model_path = model_path or Path('models/base/simple_model.pkl')
        self.position_threshold = position_threshold
        self.hold_bars = hold_bars
        self.confidence_threshold = confidence_threshold
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.position_size_pct = position_size_pct
        self.model = None
        self.feature_cols = FEATURE_COLS
        
    def load_model(self) -> None:
        if self.model_path.exists():
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data.get('model')
                if data.get('features'):
                    self.feature_cols = data.get('features')
    
    def _get_model_signal(self, row: dict) -> tuple[float, float]:
        if self.model is None:
            return 0.0, 0.0
        
        features = [row.get(c, 0) for c in self.feature_cols]
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        pred = float(self.model.predict([features])[0])
        confidence = min(1.0, abs(pred) * 50.0)
        return pred, confidence
    
    def _get_rsi_signal(self, row: dict) -> int:
        rsi = row.get('rsi_14', 50)
        if rsi < 30:
            return 1  # Oversold - long
        elif rsi > 70:
            return -1  # Overbought - short
        return 0
    
    def _get_macd_signal(self, row: dict) -> int:
        macd = row.get('macd', 0)
        macd_signal = row.get('macd_signal', 0)
        if macd > macd_signal:
            return 1
        elif macd < macd_signal:
            return -1
        return 0
    
    def _get_trend_signal(self, row: dict) -> int:
        momentum = row.get('momentum_10', 0)
        zscore = row.get('zscore_20', 0)
        if momentum > 0 and zscore > -1:
            return 1
        elif momentum < 0 and zscore < 1:
            return -1
        return 0
    
    def get_signal(self, row: dict) -> TradingSignal:
        pred, model_conf = self._get_model_signal(row)
        
        model_direction = 1 if pred > self.position_threshold else (-1 if pred < -self.position_threshold else 0)
        rsi_signal = self._get_rsi_signal(row)
        macd_signal = self._get_macd_signal(row)
        trend_signal = self._get_trend_signal(row)
        
        votes = []
        if abs(pred) > self.position_threshold:
            votes.append(model_direction)
        votes.append(rsi_signal)
        votes.append(macd_signal)
        votes.append(trend_signal)
        
        if votes:
            avg_vote = sum(votes) / len(votes)
            if avg_vote > 0.3:
                final_direction = 1
            elif avg_vote < -0.3:
                final_direction = -1
            else:
                final_direction = 0
        else:
            final_direction = 0
        
        confidence = model_conf
        
        return TradingSignal(
            direction=final_direction,
            confidence=confidence,
            source='combined'
        )
    
    def run_backtest(self, df: pd.DataFrame, ticker: str | None = None) -> dict:
        if ticker:
            df = df[df['ticker'] == ticker].copy()
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        rows = df.to_dict('records')
        
        signals = [self.get_signal(r) for r in rows]
        
        positions = []
        current_pos = 0
        counter = 0
        
        for i, sig in enumerate(signals):
            if counter == 0 and sig.direction != 0:
                current_pos = sig.direction
                counter = self.hold_bars
            positions.append(float(current_pos))
            if counter > 0:
                counter -= 1
        
        for r, pos in zip(rows, positions):
            r['policy_target_position'] = pos
        
        cfg = BacktestConfig(
            initial_cash=1_000_000,
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
            lot_size=1,
            execution_delay_bars=1,
            position_size_pct=self.position_size_pct,
            target_position_column='policy_target_position',
        )
        
        result = run_backtest(rows, cfg)
        return {
            'ticker': ticker or 'ALL',
            'pnl': result['summary']['pnl'],
            'trades': len(result['trade_log']),
            'turnover': result['summary']['turnover'],
        }
    
    def optimize(self, df: pd.DataFrame) -> dict:
        best_pnl = float('-inf')
        best_params = None
        results = []
        
        for hold in [20, 30, 60, 120]:
            for threshold in [0.0001, 0.0003, 0.0005]:
                self.hold_bars = hold
                self.position_threshold = threshold
                
                ticker_results = []
                for ticker in df['ticker'].unique():
                    res = self.run_backtest(df, ticker)
                    ticker_results.append(res)
                
                total_pnl = sum(r['pnl'] for r in ticker_results)
                results.append({
                    'hold': hold,
                    'threshold': threshold,
                    'total_pnl': total_pnl,
                    'details': ticker_results
                })
                
                if total_pnl > best_pnl:
                    best_pnl = total_pnl
                    best_params = {'hold_bars': hold, 'threshold': threshold}
        
        return {
            'best_params': best_params,
            'best_pnl': best_pnl,
            'all_results': results
        }


def main():
    import sys
    sys.path.insert(0, '.')
    
    df = pd.read_parquet('data/processed/merged/test_expanded.parquet')
    
    strategy = EnsembleStrategy()
    strategy.load_model()
    
    print('=== Testing Individual Tickers ===')
    for ticker in df['ticker'].unique()[:3]:
        res = strategy.run_backtest(df, ticker)
        print(ticker + ': PnL=' + str(round(res["pnl"])) + ', Trades=' + str(res["trades"]))
    
    print('\n=== Testing All Tickers ===')
    results = []
    for ticker in df['ticker'].unique():
        res = strategy.run_backtest(df, ticker)
        results.append(res)
        print(ticker + ': PnL=' + str(round(res['pnl'])) + ', Trades=' + str(res['trades']))
    
    total = sum(r['pnl'] for r in results)
    print('\nTotal PnL: ' + str(round(total)))


if __name__ == '__main__':
    main()
