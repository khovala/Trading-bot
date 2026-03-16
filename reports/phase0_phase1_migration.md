# Phase 0-10 Migration Notes

This document captures the first migration milestone toward a robust research pipeline.

## Phase 0: Baseline and Acceptance Gates

- Added baseline snapshot utility:
  - `src/research/baseline_snapshot.py`
  - CLI entrypoint: `python -m apps.trainer.snapshot_baseline`
- Added evaluation protocol and promotion gates:
  - `src/research/evaluation_protocol.py`
  - Config location: `params.yaml -> evaluation`

### Promotion criteria now configurable

- `walk_forward_sharpe_mean_min`
- `max_drawdown_max`
- `min_positive_folds_ratio`
- `min_turnover_reduction_vs_baseline`
- optional strict gates:
  - `require_cost_stress_pass`
  - `require_calibration_report`
  - `require_leakage_checks`

## Phase 1: Data Contract and Leakage Safety

- Added feature store contracts:
  - `src/data/feature_store/contracts.py`
- Added leakage guards:
  - `src/data/leakage/guards.py`
  - monotonic timestamp checks
  - split overlap + embargo checks
  - leakage-safe as-of join helper

## Added Tests

- `tests/test_evaluation_protocol.py`
- `tests/test_baseline_snapshot.py`
- `tests/test_feature_store_contracts.py`
- `tests/test_leakage_guards.py`

## Phase 2: Forecasting Stack Upgrade (Without RL)

- Added config-driven foundation wrappers:
  - `src/models/foundation/base_forecaster.py`
  - `src/models/foundation/chronos2_wrapper.py`
  - `src/models/foundation/timesfm_wrapper.py`
  - `src/models/foundation/moirai_wrapper.py`
  - `src/models/foundation/deep_sequence_wrappers.py`
  - `src/models/foundation/registry.py`
- Added modern news encoder pipeline:
  - `src/models/news/news_encoder_pipeline.py`
- Added new train stages:
  - `src/training/stages/foundation_models.py`
  - `train_news_encoder`
  - `train_foundation_models`
- Updated pipeline/config:
  - `pipeline.yaml` includes new Stage 2 steps
  - `params.yaml` includes `models_v2` (forecasters + news encoder config)

## Added Tests (Phase 2)

- `tests/test_foundation_registry.py`
- `tests/test_news_encoder_pipeline.py`
- `tests/test_foundation_stages.py`

## Phase 3: Adaptive Ensemble Integration

- Upgraded ensemble core:
  - `src/models/ensemble/weighted.py`
  - Added adaptive reweighting with:
    - uncertainty penalty
    - turnover penalty
    - minimum weight floor
  - Added diagnostics payload for per-model score components.
- Integrated Stage 3 logic into training:
  - `src/training/stages/model_training.py`
  - `train_ensemble_model` now:
    - loads base/news/foundation predictors
    - computes adaptive weights from prediction characteristics
    - exports `artifacts/evaluation/ensemble_ablation.json`
- Updated config/manifest:
  - `params.yaml -> models_v2.ensemble`
  - `pipeline.yaml -> train_ensemble_model` deps now include `models/foundation`

## Added Tests (Phase 3)

- `tests/test_models_baseline.py` (adaptive weighting diagnostics)
- `tests/test_model_training_stages.py` (ablation artifact existence)

## Phase 4: Promotion Logic with Stage 3 Signals

- Updated `evaluate_models` to ingest Stage 3 artifact:
  - source: `artifacts/evaluation/ensemble_ablation.json`
  - new metrics include:
    - `ablation_model_count`
    - `ablation_positive_ratio`
    - `ablation_mean_abs_delta`
    - `ablation_news_model_delta`
    - `ensemble_weight_max`
    - `ensemble_weight_concentration_hhi`
    - aggregated diagnostics confidence/turnover proxies
- Updated `compare_with_production` decision logic:
  - combines existing promotion criteria (`evaluation.promotion_criteria`)
  - adds Stage 3 signal checks from eval metrics:
    - minimum ablation-positive ratio
    - maximum weight concentration (HHI)
  - writes extended checks/details into:
    - `reports/compare_with_production.json`
    - `artifacts/comparison/decision.json`
- Extended Stage 3 artifact payload:
  - `ensemble_diagnostics` is now persisted in `ensemble_ablation.json`

## Added Tests (Phase 4)

- `tests/test_evaluation_reporting_stages.py`
  - validates Stage 3 signal ingestion in `evaluate_models`
  - validates Stage 3 signal checks appear in compare decision payload

## Promotion/Publishing Enrichment

- `promote_model` now persists full gate context in registry:
  - `promotion_checks`
  - `promotion_details`
  - decision snapshot reference
- `publish_artifacts` now includes promotion summary in bundle manifest:
  - decision, checks, details, champion model
  - explicit inclusion of compare/promote reports and decision artifact

## Phase 5: Policy Layer Foundation

- Added policy module:
  - `src/models/policy/offline_policy.py`
  - utility-aware policy with penalties for turnover, drawdown proxy, uncertainty
- Added training stage:
  - `src/training/stages/policy_layer.py`
  - stage name: `train_policy_layer`
  - outputs policy artifact + summary metrics
- Pipeline/config updates:
  - `pipeline.yaml` includes `train_policy_layer`
  - `params.yaml` includes `models_v2.policy_layer`
- Evaluation/decision integration:
  - `evaluate_models` now includes policy summary signals
  - `compare_with_production` now supports policy gate thresholds

## Added Tests (Phase 5)

- `tests/test_policy_layer_stage.py`
- `tests/test_promotion_publish_stages.py`

## Phase 6: Policy-Driven Backtesting

- Updated backtest engine:
  - `src/backtesting/engine.py`
  - supports optional `target_position_column` for fractional policy positions
  - execution sizing now scales by absolute policy target position when enabled
- Updated backtest stage:
  - `src/training/stages/evaluation_reporting.py`
  - `backtest_strategy` now auto-loads `models/policy/offline_policy_layer.pkl`
  - enriches rows with policy outputs (`policy_target_position`, `policy_signal`, utility/turnover proxies)
  - runs backtest using policy decisions (not only raw signal column)
  - emits policy-specific metrics in backtest summary
- Pipeline/config updates:
  - `pipeline.yaml` backtest deps include `models/policy`
  - `params.yaml -> stages.backtest_strategy.use_policy_layer: true`

## Added Tests (Phase 6)

- `tests/test_evaluation_reporting_stages.py`
  - verifies `backtest_strategy` runs in policy-backed mode and sets `policy_backtest_enabled`

## Phase 7: Policy Execution Metrics in Evaluate Stage

- `evaluate_models` now computes policy-execution quality metrics directly from policy decisions:
  - `policy_hit_ratio`
  - `policy_utility_adjusted_score`
  - `policy_turnover_budget_violations`
  - `policy_turnover_budget_violation_ratio`
  - `policy_avg_decision_utility`
  - `policy_active_decisions`
- New evaluate stage parameters in `params.yaml`:
  - `stages.evaluate_models.turnover_budget_per_step`
  - `stages.evaluate_models.utility_scale`
- Metrics are derived by loading `models/policy/offline_policy_layer.pkl` and replaying
  decisions over validation rows.

## Added Tests (Phase 7)

- `tests/test_evaluation_reporting_stages.py`
  - validates presence of policy execution metrics in evaluate output
  - validates budget violation ratio calculation path

## Phase 8: Backtest Metrics Export to Prometheus/Grafana

- Extended Prometheus exporter refresh in:
  - `src/monitoring/metrics.py`
- Added exported gauges from pipeline artifacts/reports:
  - backtest metrics: pnl/drawdown/sharpe/sortino/calmar/hit_ratio/turnover/exposure/walk-forward sharpe
  - evaluate metrics: directional_accuracy/mae_proxy/policy execution metrics
  - ensemble diagnostics: HHI + positive ablation ratio
  - promotion decision flag
  - pipeline freshness and report count
- Grafana dashboard expanded:
  - `infra/grafana/provisioning/dashboards/json/moex_operations.json`
  - includes dedicated panels for model quality, policy quality, ensemble diagnostics and promotion state
- Dashboard markdown explanation updated:
  - `infra/grafana/provisioning/dashboards/json/moex_operations.md`

## Added Tests (Phase 8)

- `tests/test_monitoring_metrics.py`
  - validates that Prometheus payload contains newly exported pipeline/backtest metrics

## Phase 9: Real Plot Rendering for Backtests

- Updated backtest reporting in:
  - `src/backtesting/reporting.py`
- The reporting stage now generates:
  - JSON datasets (existing behavior preserved)
  - static PNG charts (matplotlib)
  - interactive HTML charts (plotly)
- Rendered chart families:
  - equity vs benchmark
  - drawdown curve
  - rolling performance
  - trade size distribution
- Output files are saved under:
  - `artifacts/backtests/plots/*.png`
  - `artifacts/backtests/plots/*.html`
  - plus existing `*.json`

## Dependency Updates (Phase 9)

- Added visualization libraries:
  - `matplotlib`
  - `plotly`

## Phase 10: Per-Model Test Metrics and Model PNG Diagnostics

- `evaluate_models` now produces detailed per-model diagnostics:
  - `reports/evaluate_models_detailed.json`
  - sections for validation/test metrics for each model
  - separate ensemble section (`weighted_ensemble`)
- Added model-level metrics:
  - directional accuracy
  - MAE
  - avg confidence
  - pnl proxy
- Added dedicated PNG diagnostics directory:
  - `artifacts/evaluation/model_plots_png/`
  - `directional_accuracy_by_model.png`
  - `mae_by_model.png`
  - `pnl_proxy_by_model_test.png`

## Current Scope

This change set now includes governance + leakage safety + Stage 2 forecasting abstractions + Stage 3 adaptive ensemble logic + Stage 4 promotion gates + Phase 5 policy-layer foundation + Phase 6 policy-driven backtesting + Phase 7 policy execution metrics in evaluation + Phase 8 Prometheus/Grafana export + Phase 9 real plot rendering (PNG/HTML), while keeping baseline stages intact for backward compatibility.
