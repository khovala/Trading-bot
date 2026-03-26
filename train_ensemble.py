#!/usr/bin/env python3
"""
Train ensemble of ML models for improved predictions.
"""

import sys
sys.path.insert(0, '/Users/sergeyeliseev/moex-sandbox-platform')

from pathlib import Path
from datetime import datetime, timezone
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import mlflow
from mlflow.sklearn import log_model
from prometheus_client import start_http_server, Gauge, Counter, Histogram

workspace = Path('/Users/sergeyeliseev/moex-sandbox-platform')

MLFLOW_TRACKING_URI = "http://mlflow:5000"
MLFLOW_EXPERIMENT = "moex-ensemble-training"

TRAINER_METRICS_PORT = int(os.getenv("PROMETHEUS_PORT", "8002"))

TRAINING_DURATION = Histogram("trainer_duration_seconds", "Training duration", ["stage"])
TRAINING_SAMPLES = Gauge("trainer_training_samples", "Number of training samples")
MODEL_ACCURACY = Gauge("trainer_model_accuracy", "Model accuracy", ["model"])
MODEL_LOSS = Gauge("trainer_model_loss", "Model loss", ["model"])
TRAINING_RUNS = Counter("trainer_runs_total", "Total training runs", ["status"])


def setup_prometheus():
    try:
        start_http_server(TRAINER_METRICS_PORT)
        print(f"Prometheus metrics server started on port {TRAINER_METRICS_PORT}")
    except Exception as e:
        print(f"Failed to start Prometheus server: {e}")


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"MLflow tracking: {MLFLOW_TRACKING_URI}")
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT}")

print("=" * 70)
print("TRAINING ENSEMBLE ML MODELS")
print("=" * 70)

setup_prometheus()
setup_mlflow()
TRAINING_RUNS.labels(status="started").inc()

with mlflow.start_run(run_name=f"ensemble_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
    run_id = run.info.run_id
    print(f"\nMLflow Run ID: {run_id}")
    mlflow.log_param("experiment_name", MLFLOW_EXPERIMENT)
    mlflow.log_param("training_timestamp", datetime.now(timezone.utc).isoformat())

    # Load full training data
    print("\n[1/5] Loading full training data...")
    from src.data.feature_store.io import read_parquet

    train_records = read_parquet(workspace / "data/processed/merged/train.parquet")
    df = pd.DataFrame(train_records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Loaded {len(df):,} training samples")

    # Add lag features
    print("\n[2/5] Adding lag features...")

    df = df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)

    for col in ['return_1', 'rolling_volatility_20', 'rsi_14']:
        df[f'{col}_lag1'] = df.groupby('ticker')[col].shift(1).fillna(0)
        df[f'{col}_lag2'] = df.groupby('ticker')[col].shift(2).fillna(0)
        if col == 'return_1':
            df['return_lag5'] = df.groupby('ticker')[col].shift(5).fillna(0)

    df['macd_momentum_interaction'] = df['macd'] * df['momentum_10']
    df['volume_volatility_interaction'] = df['volume_ratio_20'] * df['rolling_volatility_20']

    # Create targets
    df['direction'] = (df['return_1'] > 0).astype(int)

    print(f"  Lag features added")

    # Prepare features
    print("\n[3/5] Preparing features...")

    FEATURE_COLS = [
        'rolling_volatility_20', 'momentum_10', 'rsi_14', 'macd', 'macd_signal',
        'atr_14', 'zscore_20', 'volume_ratio_20', 'volume_zscore_20',
        'trend_regime', 'volatility_regime',
        'return_1_lag1', 'return_1_lag2', 'return_lag5',
        'rolling_volatility_20_lag1', 'rsi_14_lag1',
        'macd_momentum_interaction', 'volume_volatility_interaction',
    ]

    X = df[FEATURE_COLS].values.astype(np.float32)
    y_reg = df['return_1'].values.astype(np.float32)
    y_cls = df['direction'].values.astype(int)

    # Handle NaN/Inf
    valid_mask = np.isfinite(X).all(axis=1)
    X = X[valid_mask]
    y_reg = y_reg[valid_mask]
    y_cls = y_cls[valid_mask]

    print(f"  Training samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class balance: {np.bincount(y_cls)}")

    mlflow.log_param("n_training_samples", len(X))
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("feature_columns", json.dumps(FEATURE_COLS))

    # Train ensemble models
    print("\n[4/5] Training ensemble models...")

    models = {}
    model_metrics = {}

    # Model 1: Gradient Boosting Regressor
    print("\n  Training Model 1: GradientBoostingRegressor...")
    start = datetime.now()
    reg1 = HistGradientBoostingRegressor(
        max_iter=500, max_depth=8, learning_rate=0.03,
        min_samples_leaf=20, l2_regularization=0.1, random_state=42
    )
    reg1.fit(X, y_reg)
    models['reg_gb1'] = reg1
    reg1_pred = reg1.predict(X)
    reg1_mae = float(np.mean(np.abs(y_reg - reg1_pred)))
    print(f"    Time: {(datetime.now() - start).total_seconds():.1f}s")
    print(f"    MAE: {reg1_mae:.6f}")
    mlflow.log_param("reg_gb1_max_iter", 500)
    mlflow.log_param("reg_gb1_max_depth", 8)
    mlflow.log_param("reg_gb1_learning_rate", 0.03)
    mlflow.log_metric("reg_gb1_train_mae", reg1_mae)
    model_metrics['reg_gb1'] = {'type': 'HistGradientBoostingRegressor', 'train_mae': reg1_mae}

    # Model 2: Gradient Boosting Classifier
    print("\n  Training Model 2: GradientBoostingClassifier...")
    start = datetime.now()
    cls1 = HistGradientBoostingClassifier(
        max_iter=500, max_depth=8, learning_rate=0.03,
        min_samples_leaf=20, random_state=42
    )
    cls1.fit(X, y_cls)
    models['cls_gb1'] = cls1
    cls1_acc = float(cls1.score(X, y_cls))
    print(f"    Time: {(datetime.now() - start).total_seconds():.1f}s")
    print(f"    Accuracy: {cls1_acc:.4f}")
    mlflow.log_param("cls_gb1_max_iter", 500)
    mlflow.log_param("cls_gb1_max_depth", 8)
    mlflow.log_param("cls_gb1_learning_rate", 0.03)
    mlflow.log_metric("cls_gb1_train_accuracy", cls1_acc)
    model_metrics['cls_gb1'] = {'type': 'HistGradientBoostingClassifier', 'train_accuracy': cls1_acc}

    # Model 3: Extra Trees Classifier
    print("\n  Training Model 3: ExtraTreesClassifier...")
    start = datetime.now()
    cls2 = ExtraTreesClassifier(
        n_estimators=100, max_depth=15, min_samples_leaf=20,
        random_state=42, n_jobs=-1
    )
    cls2.fit(X, y_cls)
    models['cls_et'] = cls2
    cls2_acc = float(cls2.score(X, y_cls))
    print(f"    Time: {(datetime.now() - start).total_seconds():.1f}s")
    print(f"    Accuracy: {cls2_acc:.4f}")
    mlflow.log_param("cls_et_n_estimators", 100)
    mlflow.log_param("cls_et_max_depth", 15)
    mlflow.log_metric("cls_et_train_accuracy", cls2_acc)
    model_metrics['cls_et'] = {'type': 'ExtraTreesClassifier', 'train_accuracy': cls2_acc}

    # Save ensemble
    print("\n[5/5] Saving ensemble...")

    ensemble_data = {
        'models': {
            'reg_gb1': reg1,
            'cls_gb1': cls1,
            'cls_et': cls2,
        },
        'feature_cols': FEATURE_COLS,
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'n_samples': len(X),
    }

    import pickle
    model_path = workspace / 'models/base/ensemble_models.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble_data, f)

    mlflow.log_artifact(str(model_path), artifact_path="models")

    # Save metrics
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": "ensemble",
        "mlflow_run_id": run_id,
        "models": model_metrics,
        "training_samples": len(X),
        "features": FEATURE_COLS,
    }

    metrics_path = workspace / 'reports/model_ensemble_training.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

    mlflow.log_param("ensemble_model_path", str(model_path))
    mlflow.log_param("n_models", 3)
    mlflow.log_param("training_completed_at", datetime.now(timezone.utc).isoformat())

    TRAINING_SAMPLES.set(len(X))
    MODEL_LOSS.labels(model="reg_gb1").set(reg1_mae)
    MODEL_ACCURACY.labels(model="cls_gb1").set(cls1_acc)
    MODEL_ACCURACY.labels(model="cls_et").set(cls2_acc)

    TRAINING_RUNS.labels(status="success").inc()

    print("\n" + "=" * 70)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModels saved:")
    print(f"  - reg_gb1: GradientBoostingRegressor")
    print(f"  - cls_gb1: GradientBoostingClassifier")
    print(f"  - cls_et: ExtraTreesClassifier")
    print(f"\nSaved to: models/base/ensemble_models.pkl")
    print(f"\nMLflow Run: {MLFLOW_TRACKING_URI}/#/runs/{run_id}")
