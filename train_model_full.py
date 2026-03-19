#!/usr/bin/env python3
"""
Train ML model on FULL dataset (907K samples).
"""

import sys
sys.path.insert(0, '/Users/sergeyeliseev/moex-sandbox-platform')

from pathlib import Path
from datetime import datetime, timezone
import pickle
import numpy as np
import pandas as pd

workspace = Path('/Users/sergeyeliseev/moex-sandbox-platform')

print("=" * 70)
print("TRAINING ML MODEL ON FULL DATASET")
print("=" * 70)

# Load full training data
print("\n[1/4] Loading full training data...")
from src.data.feature_store.io import read_parquet

train_records = read_parquet(workspace / "data/processed/merged/train.parquet")
print(f"  Loaded {len(train_records):,} training samples")

# Add lag features
print("\n[2/4] Adding lag features...")

df = pd.DataFrame(train_records)
df['timestamp'] = pd.to_datetime(df['timestamp'])

for ticker in df['ticker'].unique():
    mask = df['ticker'] == ticker
    ticker_df = df[mask].sort_values('timestamp').reset_index(drop=True)
    
    ticker_df['return_lag_1'] = ticker_df['return_1'].shift(1).fillna(0)
    ticker_df['return_lag_2'] = ticker_df['return_1'].shift(2).fillna(0)
    ticker_df['return_lag_5'] = ticker_df['return_1'].shift(5).fillna(0)
    ticker_df['volatility_lag_1'] = ticker_df['rolling_volatility_20'].shift(1).fillna(0)
    ticker_df['rsi_lag_1'] = ticker_df['rsi_14'].shift(1).fillna(50)
    ticker_df['macd_momentum_interaction'] = ticker_df['macd'] * ticker_df['momentum_10']
    ticker_df['volume_volatility_interaction'] = ticker_df['volume_ratio_20'] * ticker_df['rolling_volatility_20']
    
    for col in ['return_lag_1', 'return_lag_2', 'return_lag_5', 'volatility_lag_1', 'rsi_lag_1', 'macd_momentum_interaction', 'volume_volatility_interaction']:
        df.loc[mask, col] = ticker_df[col].values

df = df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)

print(f"  Lag features added. DataFrame shape: {df.shape}")

# Extract features and target
print("\n[3/4] Preparing features...")

FEATURE_COLS = [
    'rolling_volatility_20', 'momentum_10', 'rsi_14', 'macd', 'macd_signal',
    'atr_14', 'zscore_20', 'volume_ratio_20', 'volume_zscore_20',
    'trend_regime', 'volatility_regime',
    'return_lag_1', 'return_lag_2', 'return_lag_5',
    'volatility_lag_1', 'rsi_lag_1',
    'macd_momentum_interaction', 'volume_volatility_interaction',
]

X = df[FEATURE_COLS].values.astype(np.float32)
y = df['return_1'].values.astype(np.float32)

# Handle NaN/Inf
valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X[valid_mask]
y = y[valid_mask]

print(f"  Training samples: {len(X):,}")
print(f"  Features: {X.shape[1]}")

# Train model
print("\n[4/4] Training model...")
from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(
    max_iter=500,  # Increased iterations
    max_depth=8,  # Deeper trees
    learning_rate=0.03,  # Lower learning rate
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=42,
    early_stopping=False,
)

print(f"  Training HistGradientBoostingRegressor...")
print(f"    max_iter={model.max_iter}, max_depth={model.max_depth}, learning_rate={model.learning_rate}")

start_time = datetime.now()
model.fit(X, y)
end_time = datetime.now()

training_time = (end_time - start_time).total_seconds()
print(f"  Training completed in {training_time:.1f} seconds")

# Evaluate
preds = model.predict(X)
mae = float(np.mean(np.abs(y - preds)))
rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
mean_ret = float(np.mean(y))
std_ret = float(np.std(y))

print(f"\n  Training Metrics:")
print(f"    MAE: {mae:.6f}")
print(f"    RMSE: {rmse:.6f}")
print(f"    Mean target: {mean_ret:.6f}")
print(f"    Std target: {std_ret:.6f}")

# Create model wrapper

# Save model
model_path = workspace / 'models/base/sklearn_gradient_boosting_full.pkl'

# Create proper model instance for saving
from src.models.regression.sklearn_gradient_boosting import SklearnGradientBoostingModel

final_model = SklearnGradientBoostingModel(
    model_name="sklearn_gradient_boosting_full",
    model_version="v1_full",
    prediction_horizon="60m",
    fitted_at=datetime.now(timezone.utc).isoformat(),
    metrics_={
        "train_mae": mae,
        "train_rmse": rmse,
        "train_samples": float(len(X)),
        "feature_count": float(len(FEATURE_COLS)),
        "training_time_seconds": training_time,
    }
)
final_model._model = model
final_model._feature_columns = tuple(FEATURE_COLS)
final_model.mean_return = mean_ret
final_model.std_return = std_ret

final_model.save(model_path)
print(f"\n  Model saved to: {model_path}")

# Save metrics
metrics = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "model_name": final_model.model_name,
    "model_version": final_model.model_version,
    "training_samples": len(X),
    "features": FEATURE_COLS,
    "training_time_seconds": training_time,
    "metrics": {
        "train_mae": mae,
        "train_rmse": rmse,
        "mean_target": mean_ret,
        "std_target": std_ret,
    }
}

import json
with open(workspace / 'reports/model_full_training.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n  Metrics saved to: reports/model_full_training.json")
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
