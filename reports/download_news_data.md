# Пояснение к отчету download_news_data

- Этап скачивает новости из RSS-источников в raw-слой.
- `run_id` — идентификатор прогона.
- `stage_name` — имя стадии.
- `success` — успешно ли завершился этап.
- `metrics.raw_item_count` — число новостей до дедупликации.
- `metrics.deduped_item_count` — число новостей после удаления дублей.
- `metrics.dedup_removed` — сколько дублей удалено.
- `metrics.source_error_count` — число источников с ошибкой.
- `metrics.source_count` — общее число опрошенных источников.
- `artifacts` — путь к результату:
  - `data/raw/news/items.jsonl`
