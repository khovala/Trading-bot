#!/usr/bin/env python3
"""Generate market features"""
import sys
sys.path.insert(0, '/opt/airflow')

import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path('/opt/airflow/data')
models_dir = Path('/opt/airflow/models')
models_dir.mkdir(exist_ok=True)

print('Generating market features...')

try:
    candles_file = data_dir / 'candles.parquet'
    if candles_file.exists():
        df = pd.read_parquet(candles_file)
        print(f'Processing {len(df)} candles...')
        
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        df['ma20'] = df['close'].rolling(20).mean()
        df['std20'] = df['close'].rolling(20).std()
        df['zscore'] = (df['close'] - df['ma20']) / df['std20']
        
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['vol_ma20']
        
        features_file = data_dir / 'features.parquet'
        df.to_parquet(features_file)
        print(f'Features generated: {len(df)} rows, {len(df.columns)} columns')
        print(f'Saved to {features_file}')
    else:
        print('No candles data found, using sample features...')
        dates = pd.date_range('2024-01-01', periods=500, freq='1h')
        df = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(500) * 0.5),
            'volume': np.random.randint(1000, 10000, 500),
            'returns': np.random.randn(500) * 0.01,
            'log_returns': np.random.randn(500) * 0.01,
            'rsi_14': np.random.uniform(30, 70, 500),
            'ma20': 100 + np.cumsum(np.random.randn(500) * 0.1),
            'std20': np.random.uniform(1, 5, 500),
            'zscore': np.random.uniform(-2, 2, 500),
            'vol_ma20': np.random.uniform(5000, 6000, 500),
            'volume_ratio': np.random.uniform(0.5, 2, 500),
        }, index=dates)
        features_file = data_dir / 'features.parquet'
        df.to_parquet(features_file)
        print(f'Created sample features: {len(df)} rows')
    
    print('Market features generation complete')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
