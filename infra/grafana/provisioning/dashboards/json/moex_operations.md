# Пояснение к файлу moex_operations.json

- Это JSON-описание дашборда Grafana для мониторинга операционных метрик платформы.
- Файл используется провиженингом Grafana и автоматически загружается как дашборд.

Ключевые разделы:
- `title` — название дашборда (`MOEX Sandbox Operations`).
- `schemaVersion` — версия схемы JSON дашборда Grafana.
- `version` — версия самого дашборда (внутренний счетчик изменений).
- `panels` — список виджетов/графиков на дашборде.

Расшифровка панелей:
- `PnL` (`strategy_pnl_rub`) — динамика прибыли/убытка стратегии в рублях.
- `Drawdown` (`strategy_drawdown`) — просадка стратегии во времени.
- `Sharpe / Sortino / Calmar` (`strategy_sharpe`, `strategy_sortino`, `strategy_calmar`) — риск-скорректированные метрики бэктеста.
- `Hit Ratio / Exposure` (`strategy_hit_ratio`, `strategy_exposure`) — доля прибыльных сигналов и средняя экспозиция.
- `Turnover` (`strategy_turnover`) — оборот стратегии.
- `Walk-forward Sharpe Mean` (`walk_forward_sharpe_mean`) — средний Sharpe по walk-forward фолдам.
- `Directional Accuracy` (`model_directional_accuracy`) — точность направления модели на validation.
- `MAE Proxy` (`model_mae_proxy`) — прокси ошибки модели.
- `Policy Hit Ratio` (`policy_hit_ratio`) — точность policy-решений.
- `Policy Utility Adjusted Score` (`policy_utility_adjusted_score`) — policy-score с учетом utility и штрафов.
- `Policy Turnover Violation Ratio` (`policy_turnover_budget_violation_ratio`) — доля нарушений лимита turnover policy-слоем.
- `Ensemble Diagnostics` (`ensemble_weight_concentration_hhi`, `ensemble_ablation_positive_ratio`) — концентрация весов ансамбля и доля положительных абляций.
- `Promotion Decision Flag` (`promotion_decision_flag`) — 1 если кандидат промоутнут, 0 если оставлен champion.
- `Pipeline Freshness (unix ts)` (`pipeline_last_update_unix`) — таймстемп последнего обновления пайплайна.
- `Signal Counts` (`signal_count_total`) — количество торговых сигналов.
- `Model Confidence` (`model_confidence`) — уровень уверенности моделей.
- `News Freshness` (`news_freshness_seconds`) — свежесть новостного потока.
- `Order Failures` (`order_failure_total`) — количество ошибок/отказов при выставлении ордеров.

Практический смысл:
- Этот файл не содержит сами метрики, он описывает, как Grafana должна их визуализировать.
- Источником значений выступает Prometheus, который собирает метрики с `/metrics` API-сервиса.
