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

default_args = {
    "owner": "trading-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=15),
    "execution_timeout": timedelta(hours=4),
}


def check_mlflow_connection():
    import mlflow
    from src.config.settings import get_settings

    settings = get_settings()
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.MlflowClient()
        client.search_experiments()
        print("MLflow connection OK")
    except Exception as e:
        print(f"MLflow connection failed: {e}")
        raise


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


with DAG(
    DAG_ID,
    default_args=default_args,
    description="Daily ML trading pipeline with data collection, training, and backtesting",
    schedule_interval="0 6 * * *",
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
    )

    download_market_data = BashOperator(
        task_id="download_market_data",
        bash_command=(
            "cd /app && python -m src.cli run-pipeline "
            "--stage download_market_data "
            "--workspace /opt/airflow "
            "2>&1 | tee /opt/airflow/logs/download_market_data.log"
        ),
    )

    download_news_data = BashOperator(
        task_id="download_news_data",
        bash_command=(
            "cd /app && python -m src.cli run-pipeline "
            "--stage download_news_data "
            "--workspace /opt/airflow "
            "2>&1 | tee /opt/airflow/logs/download_news_data.log"
        ),
    )

    preprocess_market = BashOperator(
        task_id="preprocess_market_data",
        bash_command=(
            "cd /app && python -m src.cli run-pipeline "
            "--stage preprocess_market_data "
            "--workspace /opt/airflow "
            "2>&1 | tee /opt/airflow/logs/preprocess_market.log"
        ),
    )

    preprocess_news = BashOperator(
        task_id="preprocess_news_data",
        bash_command=(
            "cd /app && python -m src.cli run-pipeline "
            "--stage preprocess_news_data "
            "--workspace /opt/airflow "
            "2>&1 | tee /opt/airflow/logs/preprocess_news.log"
        ),
    )

    generate_market_features = BashOperator(
        task_id="generate_market_features",
        bash_command=(
            "cd /app && python -m src.cli run-pipeline "
            "--stage generate_market_features "
            "--workspace /opt/airflow "
            "2>&1 | tee /opt/airflow/logs/market_features.log"
        ),
    )

    generate_news_features = BashOperator(
        task_id="generate_news_features",
        bash_command=(
            "cd /app && python -m src.cli run-pipeline "
            "--stage generate_news_features "
            "--workspace /opt/airflow "
            "2>&1 | tee /opt/airflow/logs/news_features.log"
        ),
    )

    merge_features = BashOperator(
        task_id="merge_feature_sets",
        bash_command=(
            "cd /app && python -m src.cli run-pipeline "
            "--stage merge_feature_sets "
            "--workspace /opt/airflow "
            "2>&1 | tee /opt/airflow/logs/merge_features.log"
        ),
    )

    train_base_models = BashOperator(
        task_id="train_base_models",
        bash_command=(
            "cd /app && python -m src.cli run-pipeline "
            "--stage train_base_models "
            "--workspace /opt/airflow "
            "2>&1 | tee /opt/airflow/logs/train_models.log"
        ),
    )

    train_news_model = BashOperator(
        task_id="train_news_model",
        bash_command=(
            "cd /app && python -m src.cli run-pipeline "
            "--stage train_news_model "
            "--workspace /opt/airflow "
            "2>&1 | tee /opt/airflow/logs/train_news.log"
        ),
    )

    train_ensemble = BashOperator(
        task_id="train_ensemble_model",
        bash_command=(
            "cd /app && python -m src.cli run-pipeline "
            "--stage train_ensemble_model "
            "--workspace /opt/airflow "
            "2>&1 | tee /opt/airflow/logs/train_ensemble.log"
        ),
    )

    evaluate_models = BashOperator(
        task_id="evaluate_models",
        bash_command=(
            "cd /app && python -m src.cli run-pipeline "
            "--stage evaluate_models "
            "--workspace /opt/airflow "
            "2>&1 | tee /opt/airflow/logs/evaluate.log"
        ),
    )

    backtest_strategy = BashOperator(
        task_id="backtest_strategy",
        bash_command=(
            "cd /app && python -m src.cli run-pipeline "
            "--stage backtest_strategy "
            "--workspace /opt/airflow "
            "2>&1 | tee /opt/airflow/logs/backtest.log"
        ),
    )

    tune_policy = BashOperator(
        task_id="tune_policy",
        bash_command=(
            "cd /app && python -m src.cli run-pipeline "
            "--stage tune_policy "
            "--workspace /opt/airflow "
            "2>&1 | tee /opt/airflow/logs/tune_policy.log"
        ),
    )

    promote_model = BashOperator(
        task_id="promote_model",
        bash_command=(
            "cd /app && python -m src.cli run-pipeline "
            "--stage promote_model "
            "--workspace /opt/airflow "
            "2>&1 | tee /opt/airflow/logs/promote.log"
        ),
    )

    generate_reports = BashOperator(
        task_id="generate_reports",
        bash_command=(
            "cd /app && python -m src.cli run-pipeline "
            "--stage generate_reports "
            "--workspace /opt/airflow "
            "2>&1 | tee /opt/airflow/logs/reports.log"
        ),
    )

    log_metrics = PythonOperator(
        task_id="log_pipeline_metrics",
        python_callable=log_pipeline_metrics,
        provide_context=True,
    )

    finish = BashOperator(
        task_id="finish_pipeline",
        bash_command="echo 'Daily trading pipeline completed at $(date)'",
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
