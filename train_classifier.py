#!/usr/bin/env python3
"""
Train classification model for directional prediction.
"""

import sys
sys.path.insert(0, '/Users/sergeyeliseev/moex-sandbox-platform')

from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

workspace = Path('/Users/sergeyeliseev/moex-sandbox-platform')

print("=" * 70)
print("TRAINING DIRECTIONAL CLASSIFICATION MODEL")
print("=" * 70)

# Load full training data
print("\n[1/4] Loading full training data...")
from src.data.feature_store.io import read_parquet

train_records = read_parquet(workspace / "data/processed/merged/train.parquet")
df = pd.DataFrame(train_records)
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"  Loaded {len(df):,} training samples")

# Add lag features
print("\n[2/4] Adding lag features...")

df = df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)

for col in ['return_1', 'rolling_volatility_20', 'rsi_14']:
    df[f'{col}_lag1'] = df.groupby('ticker')[col].shift(1).fillna(0)
    df[f'{col}_lag2'] = df.groupby('ticker')[col].shift(2).fillna(0)
    if col == 'return_1':
        df['return_lag5'] = df.groupby('ticker')[col].shift(5).fillna(0)

df['macd_momentum_interaction'] = df['macd'] * df['momentum_10']
df['volume_volatility_interaction'] = df['volume_ratio_20'] * df['rolling_volatility_20']

# Create directional target: 1 if return > 0, 0 otherwise
df['direction'] = (df['return_1'] > 0).astype(int)

print(f"  Target distribution: {df['direction'].value_counts().to_dict()}")

# Prepare features
print("\n[3/4] Preparing features...")

FEATURE_COLS = [
    'rolling_volatility_20', 'momentum_10', 'rsi_14', 'macd', 'macd_signal',
    'atr_14', 'zscore_20', 'volume_ratio_20', 'volume_zscore_20',
    'trend_regime', 'volatility_regime',
    'return_1_lag1', 'return_1_lag2', 'return_lag5',
    'rolling_volatility_20_lag1', 'rsi_14_lag1',
    'macd_momentum_interaction', 'volume_volatility_interaction',
]

X = df[FEATURE_COLS].values.astype(np.float32)
y = df['direction'].values.astype(int)

# Handle NaN/Inf
valid_mask = np.isfinite(X).all(axis=1)
X = X[valid_mask]
y = y[valid_mask]

print(f"  Training samples: {len(X):,}")
print(f"  Features: {X.shape[1]}")
print(f"  Class balance: {np.bincount(y)}")

# Train model
print("\n[4/4] Training classifier...")

clf = HistGradientBoostingClassifier(
    max_iter=500,
    max_depth=8,
    learning_rate=0.03,
    min_samples_leaf=20,
    random_state=42,
)

start_time = datetime.now()
clf.fit(X, y)
end_time = datetime.now()

training_time = (end_time - start_time).total_seconds()
print(f"  Training completed in {training_time:.1f} seconds")

# Evaluate
train_acc = clf.score(X, y)
print(f"\n  Training Accuracy: {train_acc:.4f}")

# Save model
from src.models.base.serialization import save_pickle

model_path = workspace / 'models/base/directional_classifier.pkl'
save_pickle(model_path, clf)
print(f"\n  Model saved to: {model_path}")

# Save metrics
metrics = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "model_name": "directional_classifier",
    "model_version": "v1",
    "training_samples": len(X),
    "features": FEATURE_COLS,
    "training_time_seconds": training_time,
    "metrics": {
        "train_accuracy": float(train_acc),
    }
}

with open(workspace / 'reports/model_directional_training.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
