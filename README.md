
# MOEX Sandbox Trading Platform (Phase 2 Upgrade)

Modular, sandbox-first architecture for MOEX equity trading with T-Invest API, staged retraining, MLflow tracking, and observability.

## Architecture Overview

- **Apps layer**: API, trader, trainer, backtester, notifier entrypoints.
- **Domain layer**: strict shared schemas and enums (normalized model output and auditable decisions).
- **Data/NLP/Features layer**: isolated ingestion and preprocessing for market/news pipelines.
- **Training layer**: stage abstraction + manifest-driven retraining (`pipeline.yaml`) + MLflow wrapper.
- **Execution boundary**: broker logic isolated from models/orchestrator.
- **Ops layer**: Docker Compose + Prometheus/Grafana/Alertmanager placeholders.

## Repository Tree (Target)

```text
apps/{api,trader,trainer,backtester,notifier}
src/
  config domain data features nlp models training orchestration risk execution backtesting monitoring utils
infra/{compose,prometheus,grafana,alertmanager,nginx}
data/{raw,interim,processed}
reports models artifacts tests scripts notebooks
params.yaml pipeline.yaml pyproject.toml .env.example README.md
```

## ADR-Style Decisions (Phase 1)

1. **Sandbox-first safety**: default is sandbox + dry-run; live trading requires explicit flag.
2. **Schema contract first**: all models must emit normalized prediction schema before orchestration.
3. **Manifest-driven pipeline**: all retraining/backtest steps declared in `pipeline.yaml` for reproducibility.
4. **MLflow mandatory**: every train/eval/backtest/promote stage logs params/metrics/artifacts.
5. **News as first-class subsystem**: dedicated ingestion adapters and dedicated news model path.
6. **Separation of concerns**: broker, orchestration, risk, and model code remain independent.
7. **Incremental delivery**: interfaces/stubs first, business logic in later phases.

## Pipeline and Params Design

- `pipeline.yaml` contains all 18 required stages with:
  - purpose
  - deps
  - outs
  - reports
  - mlflow logging behavior
- `params.yaml` centralizes:
  - environment/trading safety flags
  - market/news feature parameters
  - ensemble/risk settings
  - backtest assumptions
  - MLflow settings

## MLflow Integration Design

- `MLflowTracker` wrapper provides:
  - run context manager
  - `log_params`, `log_metrics`, `log_artifacts`
- Future phases add:
  - dataset + schema version tags
  - candidate/champion comparison tags
  - model registry integration

## MVP News Source Adapter Design

- Base interface: `NewsSourceAdapter.fetch(since_iso, limit) -> list[RawNewsItem]`.
- Source adapters are isolated per source (`src/nlp/sources/*` future files).
- MVP source priority:
  1. public RSS business/financial feeds
  2. issuer/corporate official feeds when available
  3. Telegram public channels in later phase

## Ticker/Entity Mapping Strategy (RU Issuers)

- Multi-step mapping pipeline (future implementation):
  1. alias dictionary (issuer legal names, short names, brand names)
  2. ticker lexicon (MOEX ticker, issuer, FIGI/ISIN where available)
  3. rule-based NER + fuzzy matching
  4. confidence score + fallback `unmapped`
- Base interface: `EntityMapper.map_news_item(item) -> TickerMappingResult`.

## First Commit File List (Phase 1)

- Root: `pyproject.toml`, `.env.example`, `params.yaml`, `pipeline.yaml`, `README.md`, `docker-compose.yml`
- Apps: `apps/api/main.py`, `apps/trainer/run_stage.py`
- Core src:
  - `src/config/settings.py`
  - `src/domain/enums.py`
  - `src/domain/schemas.py`
  - `src/training/pipeline/base.py`
  - `src/training/tracking/mlflow_utils.py`
  - `src/nlp/sources/base.py`
  - `src/nlp/entity_mapping/base.py`

## Quick Start (Local)

1. Copy env:
   - `cp .env.example .env`
2. Run API:
   - `docker compose up api`
3. Check:
   - `GET /health`
   - `GET /ready`
   - `GET /metrics`

## Phase 2 Additions

- Added config-driven Stage 2 modeling stack:
  - `train_news_encoder` for embedding/sentiment features
  - `train_foundation_models` for Chronos/TimesFM/Moirai/TimeXer/TFT/PatchTST wrappers
- Added `models_v2` section in `params.yaml`:
  - `news_encoder` hyperparameters
  - per-forecaster settings (`enabled`, `prediction_horizon`, `calibration_alpha`, covariates)
- Baseline stages (`train_base_models`, `train_news_model`, `train_ensemble_model`) remain intact for compatibility.

## Phase 3 Additions

- `train_ensemble_model` upgraded to adaptive weighting:
  - uses confidence-aware uncertainty penalty
  - uses expected-return-change turnover penalty
  - supports dynamic inclusion of foundation model outputs
- Added ablation diagnostics artifact:
  - `artifacts/evaluation/ensemble_ablation.json`
- New config block in `params.yaml`:
  - `models_v2.ensemble` (`uncertainty_penalty`, `turnover_penalty`, `min_weight`, `foundation_default_weight`)

## Phase 4 Additions

- `evaluate_models` now enriches report metrics with Stage 3 signals from:
  - `artifacts/evaluation/ensemble_ablation.json`
- `compare_with_production` now uses:
  - `evaluation.promotion_criteria` checks
  - Stage 3 signal thresholds (`min_ablation_positive_ratio`, `max_weight_concentration_hhi`)
- Decision payload now contains explicit gate breakdown:
  - `checks.promotion_criteria`
  - `checks.stage3_signals`
  - `details` with values used for promotion decision

## Promotion Registry & Bundle

- `promote_model` stores gate reasons in `models/registry/champion.json`
  - `promotion_checks`
  - `promotion_details`
- `publish_artifacts` stores `promotion_summary` in:
  - `artifacts/published/bundle_manifest.json`

## Phase 5 Additions

- New policy stage:
  - `train_policy_layer`
  - output model: `models/policy/offline_policy_layer.pkl`
  - output summary: `artifacts/evaluation/policy_layer_summary.json`
- New config section:
  - `models_v2.policy_layer`
- `evaluate_models` now reads policy summary metrics.
- `compare_with_production` now supports policy-related thresholds:
  - `min_policy_avg_utility`
  - `max_policy_turnover_proxy`

## Phase 6 Additions

- `backtest_strategy` now supports policy-driven execution:
  - automatically uses `models/policy/offline_policy_layer.pkl` when available
  - backtest rows are enriched with policy outputs (`policy_target_position`, `policy_signal`)
  - engine supports fractional target positions via `target_position_column`
- New stage parameter:
  - `stages.backtest_strategy.use_policy_layer` (default `true`)
- Backtest summary includes policy usage diagnostics:
  - `policy_backtest_enabled`
  - `policy_avg_abs_target_position`
  - `policy_avg_expected_utility`

## Phase 7 Additions

- `evaluate_models` now computes policy execution metrics on validation data:
  - `policy_hit_ratio`
  - `policy_utility_adjusted_score`
  - `policy_turnover_budget_violations`
  - `policy_turnover_budget_violation_ratio`
  - `policy_avg_decision_utility`
- New tuneable settings:
  - `stages.evaluate_models.turnover_budget_per_step`
  - `stages.evaluate_models.utility_scale`

## Phase 8 Additions

- Prometheus export now includes offline pipeline outputs automatically via `/metrics`:
  - full backtest quality set (`strategy_sharpe`, `strategy_sortino`, `strategy_calmar`, etc.)
  - evaluate/model metrics (`model_directional_accuracy`, `model_mae_proxy`)
  - policy execution metrics (`policy_hit_ratio`, `policy_utility_adjusted_score`, violation ratio)
  - ensemble diagnostics and promotion decision flag
  - pipeline freshness (`pipeline_last_update_unix`) and report count
- Grafana dashboard `MOEX Sandbox Operations` expanded with panels for these metrics.

## Phase 9 Additions

- Backtest plotting now produces real visual files in addition to JSON:
  - static PNG charts via matplotlib
  - interactive HTML charts via plotly
- Output location:
  - `artifacts/backtests/plots/`
- Generated charts:
  - `equity_vs_benchmark` (`.json`, `.png`, `.html`)
  - `drawdown_curve` (`.json`, `.png`, `.html`)
  - `rolling_performance` (`.json`, `.png`, `.html`)
  - `trade_distribution` (`.json`, `.png`, `.html`)

## Run Stage 2 Only

- `docker compose run --rm trainer python -m apps.trainer.run_stage --stage train_news_encoder`
- `docker compose run --rm trainer python -m apps.trainer.run_stage --stage train_foundation_models`

# Trading-bot
ML for finances (OTUS course)
88676d9b4169d2e96049d6d99c24ff8a96c63bd4
