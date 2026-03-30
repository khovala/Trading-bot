#!/usr/bin/env python3
"""Preprocess market data"""
import sys
sys.path.insert(0, '/opt/airflow')

import pandas as pd
from pathlib import Path

data_dir = Path('/opt/airflow/data')
data_dir.mkdir(exist_ok=True)

print('Preprocessing market data...')

try:
    candles_file = data_dir / 'candles.parquet'
    if candles_file.exists():
        df = pd.read_parquet(candles_file)
        print(f'Loaded {len(df)} candles')
    else:
        print('No candles.parquet found, checking CSV...')
        csv_file = data_dir / 'candles.csv'
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            print(f'Loaded {len(df)} candles from CSV')
        else:
            print('No market data found, creating sample data...')
            import numpy as np
            dates = pd.date_range('2024-01-01', periods=1000, freq='1h')
            df = pd.DataFrame({
                'close': 100 + np.cumsum(np.random.randn(1000) * 0.5),
                'volume': np.random.randint(1000, 10000, 1000),
            }, index=dates)
            df.to_parquet(candles_file)
            print(f'Created sample data: {len(df)} rows')
    
    print(f'Preprocessing complete: {len(df)} rows')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
