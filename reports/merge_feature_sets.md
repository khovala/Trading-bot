# Пояснение к отчету merge_feature_sets

- Этап объединяет рыночные и новостные признаки.
- После объединения данные делятся на train/validation/test.
- `run_id` — идентификатор запуска.
- `stage_name` — имя стадии.
- `success` — статус выполнения.
- `metrics.merged_row_count` — общее число строк после merge.
- `metrics.train_row_count` — размер train-выборки.
- `metrics.validation_row_count` — размер validation-выборки.
- `metrics.test_row_count` — размер test-выборки.
- `metrics.merge_bucket_minutes` — окно синхронизации по времени при merge.
- `metrics.news_lag_buckets` — количество лагов новостных бакетов.
- `artifacts` — итоговые датасеты:
  - `data/processed/merged/train.parquet`
  - `data/processed/merged/validation.parquet`
  - `data/processed/merged/test.parquet`
