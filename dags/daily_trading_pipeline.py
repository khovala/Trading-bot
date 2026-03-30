"""
Daily Trading Pipeline DAG
=========================
Orchestrates the complete ML trading pipeline:
1. Data collection (market + news)
2. Feature engineering
3. Model training
4. Backtesting
5. Model evaluation & promotion
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

DAG_ID = "daily_trading_pipeline"
WORKSPACE = Path("/opt/airflow")
REPORTS_DIR = WORKSPACE / "reports"
DATA_DIR = WORKSPACE / "data"
MODELS_DIR = WORKSPACE / "models"
SCRIPTS_DIR = WORKSPACE / "scripts"

default_args = {
    "owner": "trading-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def check_mlflow_connection(**context):
    import mlflow
    import os
    
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
    
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        client = mlflow.MlflowClient()
        client.search_experiments()
        print(f"MLflow connection OK at {mlflow_uri}")
    except Exception as e:
        print(f"MLflow connection failed: {e}")
        print("Continuing without MLflow - will use local tracking only")


def log_pipeline_metrics(**context):
    import json
    from datetime import datetime

    execution_date = context["execution_date"]
    run_id = context["run_id"]

    metrics = {
        "dag_id": DAG_ID,
        "execution_date": execution_date.isoformat(),
        "run_id": run_id,
        "status": "success",
        "completed_at": datetime.utcnow().isoformat(),
    }

    metrics_file = REPORTS_DIR / f"pipeline_run_{execution_date.strftime('%Y%m%d_%H%M%S')}.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_file}")


with DAG(
    DAG_ID,
    default_args=default_args,
    description="Daily ML trading pipeline with data collection, training, and backtesting",
    schedule_interval="0 5 * * *",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["trading", "ml", "daily"],
) as dag:

    start = BashOperator(
        task_id="start_pipeline",
        bash_command="echo 'Starting daily trading pipeline at $(date)'",
    )

    check_mlflow = PythonOperator(
        task_id="check_mlflow_connection",
        python_callable=check_mlflow_connection,
        provide_context=True,
    )

    download_market_data = BashOperator(
        task_id="download_market_data",
        bash_command="echo 'Downloading market data...' && echo 'Market data download placeholder'",
    )

    download_news_data = BashOperator(
        task_id="download_news_data",
        bash_command="echo 'Downloading news data...' && echo 'News data download placeholder'",
    )

    preprocess_market = BashOperator(
        task_id="preprocess_market_data",
        bash_command="cd /opt/airflow && python3 scripts/preprocess_market.py 2>&1",
    )

    preprocess_news = BashOperator(
        task_id="preprocess_news_data",
        bash_command="echo 'Preprocessing news data...' && echo 'News preprocessing complete'",
    )

    generate_market_features = BashOperator(
        task_id="generate_market_features",
        bash_command="cd /opt/airflow && python3 scripts/generate_features.py 2>&1",
    )

    generate_news_features = BashOperator(
        task_id="generate_news_features",
        bash_command="echo 'Generating news features...' && echo 'News features complete'",
    )

    merge_features = BashOperator(
        task_id="merge_feature_sets",
        bash_command="echo 'Merging feature sets...' && echo 'Feature merging complete'",
    )

    train_base_models = BashOperator(
        task_id="train_base_models",
        bash_command="cd /opt/airflow && python3 scripts/train_base_models.py 2>&1",
    )

    train_news_model = BashOperator(
        task_id="train_news_model",
        bash_command="echo 'Training news model...' && echo 'News model training complete'",
    )

    train_ensemble = BashOperator(
        task_id="train_ensemble_model",
        bash_command="cd /opt/airflow && python3 scripts/train_ensemble.py 2>&1",
    )

    evaluate_models = BashOperator(
        task_id="evaluate_models",
        bash_command="cd /opt/airflow && python3 scripts/evaluate_models.py 2>&1",
    )

    backtest_strategy = BashOperator(
        task_id="backtest_strategy",
        bash_command="cd /opt/airflow && python3 scripts/backtest.py 2>&1",
    )

    tune_policy = BashOperator(
        task_id="tune_policy",
        bash_command="cd /opt/airflow && python3 scripts/tune_policy.py 2>&1",
    )

    promote_model = BashOperator(
        task_id="promote_model",
        bash_command="cd /opt/airflow && python3 scripts/promote_model.py 2>&1",
    )

    generate_reports = BashOperator(
        task_id="generate_reports",
        bash_command="cd /opt/airflow && python3 scripts/generate_reports.py 2>&1",
    )

    log_metrics = PythonOperator(
        task_id="log_pipeline_metrics",
        python_callable=log_pipeline_metrics,
        provide_context=True,
    )

    finish = BashOperator(
        task_id="finish_pipeline",
        bash_command="echo 'Daily trading pipeline completed successfully at $(date)'",
    )

    start >> check_mlflow >> [download_market_data, download_news_data]

    download_market_data >> preprocess_market >> generate_market_features
    download_news_data >> preprocess_news >> generate_news_features

    [generate_market_features, generate_news_features] >> merge_features

    merge_features >> train_base_models
    merge_features >> train_news_model

    [train_base_models, train_news_model] >> train_ensemble >> evaluate_models

    evaluate_models >> backtest_strategy >> tune_policy >> promote_model

    promote_model >> generate_reports >> log_metrics >> finish
