from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.domain.enums import RunMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="dev", alias="APP_ENV")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    trading_mode: RunMode = Field(default=RunMode.SANDBOX, alias="TRADING_MODE")
    dry_run: bool = Field(default=True, alias="DRY_RUN")
    real_trading_enabled: bool = Field(default=False, alias="REAL_TRADING_ENABLED")
    prometheus_enabled: bool = Field(default=True, alias="PROMETHEUS_ENABLED")

    t_invest_token: str | None = Field(default=None, alias="T_INVEST_TOKEN")
    t_invest_account_id: str | None = Field(default=None, alias="T_INVEST_ACCOUNT_ID")

    mlflow_tracking_uri: str = Field(default="http://mlflow:5000", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment: str = Field(default="moex-sandbox", alias="MLFLOW_EXPERIMENT")

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
