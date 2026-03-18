from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.domain.enums import RunMode


class YandexCloudSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    yc_cloud_id: str | None = Field(default=None, alias="YC_CLOUD_ID")
    yc_folder_id: str | None = Field(default=None, alias="YC_FOLDER_ID")
    yc_service_account_key: Path | None = Field(default=None, alias="YC_SERVICE_ACCOUNT_KEY")
    yc_s3_bucket: str = Field(default="moex-trading-mlflow", alias="YC_S3_BUCKET")
    yc_s3_endpoint: str = Field(default="https://storage.yandexcloud.net", alias="YC_S3_ENDPOINT")
    yc_postgres_host: str = Field(default="localhost", alias="YC_POSTGRES_HOST")
    yc_postgres_port: int = Field(default=5432, alias="YC_POSTGRES_PORT")
    yc_postgres_db: str = Field(default="trading", alias="YC_POSTGRES_DB")
    yc_postgres_user: str = Field(default="trading_user", alias="YC_POSTGRES_USER")
    yc_postgres_password: str = Field(default="", alias="YC_POSTGRES_PASSWORD")


class TinkoffSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    tinkoff_token: str | None = Field(default=None, alias="TINKOFF_TOKEN")
    tinkoff_account_id: str | None = Field(default=None, alias="TINKOFF_ACCOUNT_ID")
    tinkoff_sandbox: bool = Field(default=True, alias="TINKOFF_SANDBOX")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="dev", alias="APP_ENV")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    trading_mode: RunMode = Field(default=RunMode.SANDBOX, alias="TRADING_MODE")
    dry_run: bool = Field(default=True, alias="DRY_RUN")
    real_trading_enabled: bool = Field(default=False, alias="REAL_TRADING_ENABLED")
    prometheus_enabled: bool = Field(default=True, alias="PROMETHEUS_ENABLED")

    mlflow_tracking_uri: str = Field(default="http://mlflow:5000", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment: str = Field(default="moex-sandbox", alias="MLFLOW_EXPERIMENT")
    mlflow_artifact_root: str = Field(default="s3://moex-trading-mlflow", alias="MLFLOW_ARTIFACT_ROOT")

    yc: YandexCloudSettings = Field(default_factory=YandexCloudSettings)
    tinkoff: TinkoffSettings = Field(default_factory=TinkoffSettings)

    alertmanager_url: str = Field(default="http://alertmanager:9093", alias="ALERTMANAGER_URL")
    telegram_bot_token: str | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str | None = Field(default=None, alias="TELEGRAM_CHAT_ID")

    airflow_webserver_url: str = Field(default="http://airflow:8080", alias="AIRFLOW_WEBSERVER_URL")
    airflow_dag_folder: str = Field(default="/opt/airflow/dags", alias="AIRFLOW_DAG_FOLDER")

    @model_validator(mode="after")
    def validate_trading_safety(self) -> "Settings":
        if self.trading_mode == RunMode.LIVE:
            if not self.real_trading_enabled:
                raise ValueError("LIVE mode requires REAL_TRADING_ENABLED=true.")
            if self.dry_run:
                raise ValueError("LIVE mode requires DRY_RUN=false.")
        elif self.real_trading_enabled:
            raise ValueError("REAL_TRADING_ENABLED=true is only allowed with TRADING_MODE=live.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
