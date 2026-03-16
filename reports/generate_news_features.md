# Пояснение к отчету generate_news_features

- Этап строит новостные фичи по временным бакетам.
- `run_id` — идентификатор запуска.
- `stage_name` — имя стадии.
- `success` — статус выполнения.
- `metrics.mapped_row_count` — сколько строк mapped-news использовано на входе.
- `metrics.feature_row_count` — сколько строк новостных фич построено.
- `metrics.bucket_minutes` — размер временного бакета агрегации в минутах.
- `artifacts` — файл с новостными фичами:
  - `data/processed/news/features/news_features.parquet`
