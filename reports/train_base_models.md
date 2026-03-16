# Пояснение к отчету train_base_models

- Этап обучает базовые модели: tabular, gru-skeleton, binary, multiclass.
- `run_id` — идентификатор запуска.
- `stage_name` — имя стадии.
- `success` — статус выполнения.
- `tabular_regression_baseline_train_mae_proxy` — прокси-ошибка baseline-регрессии (меньше лучше).
- `tabular_regression_baseline_train_samples` — число train-строк для tabular-модели.
- `gru_regression_skeleton_train_state_value` — финальное сглаженное состояние GRU-skeleton.
- `gru_regression_skeleton_train_samples` — объем train-данных для GRU-skeleton.
- `binary_direction_classifier_binary_up_prob` — оцененная доля положительных движений.
- `binary_direction_classifier_binary_down_prob` — оцененная доля отрицательных движений.
- `binary_direction_classifier_train_samples` — объем train-данных для binary-модели.
- `multiclass_action_classifier_buy_prob/hold_prob/sell_prob` — вероятности классов BUY/HOLD/SELL.
- `multiclass_action_classifier_train_samples` — объем train-данных для multiclass-модели.
- `train_rows` — общий размер train-выборки на стадии.
- `artifacts` — каталог с сохраненными базовыми моделями:
  - `models/base`
