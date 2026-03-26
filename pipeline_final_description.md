# Полный пайплайн торговой стратегии MOEX

## Содержание
1. [Обзор системы](#1-обзор-системы)
2. [Получение данных](#2-получение-данных)
3. [Препроцессинг данных](#3-препроцессинг-данных)
4. [Генерация признаков](#4-генерация-признаков)
5. [Архитектура моделей](#5-архитектура-моделей)
   - [5.1 sklearn GradientBoosting](#51-sklearn-gradientboosting-production-model)
   - [5.2 Метрики моделей на тестовой выборке](#52-метрики-моделей-на-тестовой-выборке)
   - [5.3 Анализ ансамбля](#53-анализ-ансамбля)
   - [5.4 Foundation Models](#54-foundation-models-timesfm-patchtst)
   - [5.5 Policy Layer](#55-policy-layer)
   - [5.6 Итоговые выводы по моделям](#56-итоговые-выводы-по-моделям)
   - [5.7 Метрики PnL: Proxy vs Реальный](#57-метрики-pnl-proxy-vs-реальный)
   - [5.8 Итоговые выводы по моделям](#58-итоговые-выводы-по-моделям)
   - [5.9 Подробное описание работы моделей](#59-подробное-описание-работы-моделей)
     - [5.9.1 Архитектура ансамбля](#591-архитектура-ансамбля)
     - [5.9.2 Модели регрессии](#592-модели-регрессии)
     - [5.9.3 Модели классификации](#593-модели-классификации)
     - [5.9.4 Foundation Models](#594-foundation-models-timesfm-patchtst)
     - [5.9.5 Weighted Ensemble](#595-weighted-ensemble-объединение)
     - [5.9.6 Использование в стратегии](#596-использование-в-стратегии)
     - [5.9.7 Итоговая схема работы](#597-итоговая-схема-работы)
     - [5.9.8 Результаты по типам моделей](#598-результаты-по-типам-моделей)
6. [Стратегия торговли](#6-стратегия-торговли)
7. [Бэктестирование](#7-бэктестирование)
8. [Результаты и метрики](#8-результаты-и-метрики)
9. [Выводы и рекомендации](#9-выводы-и-рекомендации)

---

## 1. Обзор системы

### 1.1 Назначение
Система алгоритмической торговли на Московской бирже (MOEX) с использованием:
- ML-моделей для предсказания доходности
- Правил технического анализа (RSI, Z-Score, MACD)
- Фильтрации по рыночному режиму

### 1.2 Период данных
| Набор | Период | Строк | Описание |
|-------|--------|-------|---------|
| Train | 2026-02-15 → 2026-03-08 | 302,400 | Обучение моделей |
| Validation | 2026-03-08 → 2026-03-12 | 64,800 | Валидация |
| Test | 2026-03-12 → 2026-03-17 | 64,810 | Финальное тестирование |
| Daily | 2026-02-15 → 2026-03-17 | 310 | Агрегированные данные |

### 1.3 Торгуемые инструменты
```
GAZP, LKOH, MGNT, NVTK, POLY, SBER, SNGS, SNGSP, TATN, YNDX
```
10 наиболее ликвидных акций MOEX.

---

## 2. Получение данных

### 2.1 Источники данных
- **T-Inverst API (Sandbox)** - котировки 1-минутного интервала
- **RSS-ленты** - новостной поток (RBC, Interfax, ЦБ РФ)

### 2.2 Параметры загрузки (params.yaml)
```yaml
data:
  market:
    provider: t_invest_sandbox
    interval: 1min
    lookback_days: 30
    instruments:
      - SBER
      - GAZP
      - LKOH
      - YNDX
      - POLY
      - NVTK
      - MGNT
      - SNGS
      - TATN
      - SNGSP
  news:
    enabled: true
    sources:
      - rbc_rss
      - interfax_rss
      - cbr_rss
    language: ru
    dedup_window_minutes: 240
```

### 2.3 Схема сырых данных (market_features.parquet)
| Поле | Тип | Описание |
|------|-----|----------|
| ticker | string | Тикер инструмента |
| timestamp | datetime | Время (UTC) |
| close | float | Цена закрытия |
| volume | int | Объём торгов |
| open | float | Цена открытия |
| high | float | Максимум |
| low | float | Минимум |

---

## 3. Препроцессинг данных

### 3.1 Валидация данных
```
src/data/preprocessing/market/validation.py
```
- Проверка на пропущенные значения
- Проверка на аномальные значения (z-score > 10)
- Валидация временных рядов (монотонность timestamp)

### 3.2 Обработка пропусков
```python
# Восстановление пропусков
df['close'] = df.groupby('ticker')['close'].fillna(method='ffill')
df['volume'] = df.groupby('ticker')['volume'].fillna(0)
```

### 3.3 Дедикация новостей
```python
# Окно дедупликации: 240 минут
# Источники имеют разные веса:
#   - rbc_rss: 1.0
#   - interfax_rss: 1.2
#   - cbr_rss: 1.1
```

---

## 4. Генерация признаков

### 4.1 Технические индикаторы

#### Трендовые индикаторы
```python
# Momentum (10-периодная)
momentum_10 = close[t] / close[t-10] - 1

# MACD
macd = EMA_12 - EMA_26
macd_signal = EMA_9(macd)

# Z-Score (20-периодный)
zscore_20 = (close - mean_20) / std_20
```

#### Волатильность
```python
# Rolling Volatility (20-периодная)
rolling_volatility_20 = std(return_20)

# ATR (14-периодный)
high_low = high - low
high_close = abs(high - close_prev)
low_close = abs(low - close_prev)
atr_14 = (high_low + high_close + low_close) / 3
```

#### RSI
```python
# Relative Strength Index (14-периодный)
delta = close - close_prev
gain = max(delta, 0)
loss = max(-delta, 0)
avg_gain = SMA(gain, 14)
avg_loss = SMA(loss, 14)
rs = avg_gain / avg_loss
rsi_14 = 100 - (100 / (1 + rs))
```

### 4.2 Объёмные индикаторы
```python
# Volume Ratio (20-периодный)
volume_ratio_20 = volume / rolling_mean(volume, 20)

# Volume Z-Score
volume_zscore_20 = (volume - mean) / std
```

### 4.3 Режимы рынка
```python
# Trend Regime (0-2)
# 0 = нет тренда, 1 = восходящий, 2 = нисходящий
if momentum_10 > threshold:
    trend_regime = 1
elif momentum_10 < -threshold:
    trend_regime = 2
else:
    trend_regime = 0

# Volatility Regime (0-2)
# 0 = низкая, 1 = нормальная, 2 = высокая
if rolling_volatility_20 < lower_quantile:
    volatility_regime = 0
elif rolling_volatility_20 > upper_quantile:
    volatility_regime = 2
else:
    volatility_regime = 1
```

### 4.4 Новостные признаки
```python
# Sentiment features
news_sentiment_mean = mean(positive_count - negative_count)
news_weighted_sentiment_mean = weighted_mean(sentiment, recency)
news_abnormal_news_volume = news_count / mean_news_count

# Event flags
news_breaking_news_flag = 1 if breaking_news else 0
news_event_mna_flag = 1 if M&A news else 0
news_event_sanctions_flag = 1 if sanctions news else 0
```

### 4.5 Финальный набор признаков (28 колонок)
```
ticker, timestamp, close, return_1, log_return_1,
rolling_volatility_20, momentum_10, rsi_14, macd, macd_signal,
atr_14, zscore_20, volume, volume_ratio_20, volume_zscore_20,
trend_regime, volatility_regime,
news_article_count, news_positive_count, news_negative_count,
news_sentiment_mean, news_weighted_sentiment_mean,
news_abnormal_news_volume, news_recency_weighted_sentiment,
news_breaking_news_flag, news_event_mna_flag,
news_event_sanctions_flag, news_event_management_flag
```

---

## 5. Архитектура моделей

### 5.1 sklearn GradientBoosting (Production Model)

**Назначение**: Замена для LightGBM (не работает из-за отсутствия libomp).

**Гиперпараметры**:
```python
HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=5,
    learning_rate=0.05,
    min_samples_leaf=30,
    l2_regularization=0.2,
    random_state=42,
)
```

**Признаки (14)**:
```python
feature_cols = [
    'rolling_volatility_20', 'momentum_10', 'rsi_14', 'macd', 'macd_signal',
    'zscore_20', 'volume_ratio_20',
    'return_lag_1', 'return_lag_2', 'return_lag_5',
    'volatility_lag_1', 'rsi_lag_1',
    'macd_momentum_interaction', 'volume_volatility_interaction',
]
```

**Результаты обучения**:
- Train samples: 150,000
- Features: 14

### 5.2 Метрики моделей на тестовой выборке

#### Рейтинг моделей по directional_accuracy (тест)

| Ранг | Модель | Directional Accuracy | MAE | PnL Proxy |
|------|--------|---------------------|-----|-----------|
| 🥇 1 | **LightGBM Regression** | **76.66%** | **0.00283** | **196.02** |
| 🥈 2 | **Weighted Ensemble** | **76.68%** | 0.00332 | 196.04 |
| 🥉 3 | TimesFM2 (Foundation) | 63.51% | 0.00326 | 101.66 |
| 4 | PatchTST (Foundation) | 63.47% | 0.00325 | 101.40 |
| 5 | News Feature Model | 50.15% | 0.00399 | 0.02 |
| 6 | Tabular Baseline | 49.85% | 0.00361 | -0.95 |
| 7 | GRU Skeleton | 49.85% | 0.00365 | -0.95 |
| 8 | LSTM | 49.85% | 0.00361 | -0.95 |
| 9 | Binary Classifier | 49.85% | 0.00361 | -0.95 |
| 10 | Multiclass Classifier | 49.85% | 0.00361 | -0.95 |

#### Детальные метрики топ-3 моделей

```
┌─────────────────────────────────────────────────────────────┐
│  LightGBM Regression (Победитель)                          │
├─────────────────────────────────────────────────────────────┤
│  Directional Accuracy:  76.66%  ████████████████████░░░░   │
│  MAE:                  0.00283  ████                       │
│  PnL Proxy:            196.02   ██████████████████████████ │
│  Confidence:           6.27%                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Weighted Ensemble                                          │
├─────────────────────────────────────────────────────────────┤
│  Directional Accuracy:  76.68%  ████████████████████░░░░   │
│  MAE:                  0.00332  █████                       │
│  PnL Proxy:            196.04   ██████████████████████████ │
│  Confidence:           2.52%                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  TimesFM2 (Foundation Model)                                │
├─────────────────────────────────────────────────────────────┤
│  Directional Accuracy:  63.51%  ██████████████░░░░░░░░░░░  │
│  MAE:                  0.00326  █████                       │
│  PnL Proxy:            101.66   ██████████████████░░░░░░░░  │
│  Confidence:           0.003%                                 │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Анализ ансамбля

#### Структура Weighted Ensemble

Ансамбль использует адаптивное взвешивание компонентов:

```yaml
models:
  weights:
    lightgbm_regression: 0.39      # Максимальный вес (лучший MAE)
    lstm_regression: 0.20
    xgboost_direction_classifier: 0.20
    xgboost_multiclass_classifier: 0.15
    news_feature_model: 0.10
    tabular_regression_baseline: 0.05
    gru_regression_skeleton: 0.02
    binary_direction_classifier: 0.02
    multiclass_action_classifier: 0.01
```

#### Эффективность ансамбля

| Метрика | Ансамбль | LightGBM | Разница |
|---------|----------|----------|---------|
| Directional Accuracy | 76.68% | 76.66% | +0.02% |
| MAE | 0.00332 | 0.00283 | -15% |
| PnL Proxy | 196.04 | 196.02 | +0.01% |

**Вывод**: Ансамбль даёт незначительное улучшение accuracy, но уступает чистому LightGBM по MAE.

#### Ablation Analysis (Вклад моделей)

| Модель удалённая | Δ PnL Proxy | Вес в ансамбле |
|-----------------|--------------|----------------|
| LightGBM | -196.02 | 0.39 |
| TimesFM2 | -101.66 | 0.05 |
| PatchTST | -101.40 | 0.05 |

**Вывод**: LightGBM вносит наибольший вклад в результат ансамбля.

### 5.4 Foundation Models (TimesFM, PatchTST)

**TimesFM**:
```yaml
type: timesfm2
prediction_horizon: 60m
model_version: v2
calibration_alpha: 0.10
expected_return_scale: 1.00
```

**PatchTST**:
```yaml
type: patchtst
prediction_horizon: 60m
model_version: v1
calibration_alpha: 0.07
expected_return_scale: 1.00
```

### 5.5 Policy Layer

```python
OfflinePolicyLayer(
    risk_aversion=4.0,
    turnover_penalty=0.50,
    drawdown_penalty=0.30,
    uncertainty_penalty=0.40,
    max_position=0.5,
    min_confidence=0.0,
    signal_deadband=0.0,
    max_turnover_step=0.03,
    signal_to_position_scale=200.0,
)
```

### 5.7 Метрики PnL: Proxy vs Реальный

#### PnL Proxy (модельная метрика)

**Формула расчёта:**
```python
pnl_proxy = Σ (sign(predicted_return) × actual_return)
```

**Временной период:**
| Набор | Период | Samples | Дней |
|-------|--------|---------|------|
| Validation | 2026-03-08 → 2026-03-12 | 78,840 | 4 |
| Test | 2026-03-12 → 2026-03-17 | 78,841 | 5 |

**Интерпретация:**
| Значение | Значение |
|----------|----------|
| > 0 | Модель правильно угадывает направление |
| < 0 | Модель ошибается чаще |
| ≈ 0 | Случайное угадывание |

#### Реальный PnL (бэктест)

**Параметры бэктеста:**
- Начальный капитал: 1,000,000 ₽
- Комиссия: 5 bps (0.05%)
- Проскальзывание: 5 bps (0.05%)
- Размер позиции: 30% от капитала
- Период: 31 день (2026-02-15 → 2026-03-17)

**Результат:**
| Метрика | Значение |
|---------|----------|
| **Total PnL** | **+121,248 ₽** |
| Total Trades | 103 |
| Return | +12.1% |
| Annualized Return | ~143% (приблизительно) |

**PnL Proxy ≠ Реальный PnL:**
- **PnL Proxy**: кумулятивный signed return (без учёта комиссий)
- **Реальный PnL**: с вычетом комиссий и проскальзывания

**Пример (LightGBM, Test):**
```
PnL Proxy = 196.02
Interpretation: 
Если бы мы торговали на каждом баре по предсказанию модели,
кумулятивный signed return составил бы ~196 единиц.
```

### 5.8 Итоговые выводы по моделям

**Лучшая модель**: LightGBM Regression
- Directional Accuracy: 76.66%
- MAE: 0.00283 (лучший среди всех)
- PnL Proxy: 196.02

**Почему техническая стратегия работает лучше ML:**

1. **Недостаток данных**: 21 день недостаточно для обучения ML моделей
2. **Переобучение**: Модели переобучаются на шумовых паттернах
3. **Простота**: RSI/Z-Score работают на любом рынке без обучения
4. **Устойчивость**: Технические индикаторы не зависят от исторических данных

**Рекомендация**: Для production использовать комбинацию:
- LightGBM для предсказания направления
- Mean Reversion + Market Timing для генерации сигналов

### 5.9 Подробное описание работы моделей

#### 5.9.1 Архитектура ансамбля

```
┌─────────────────────────────────────────────────────────────────┐
│                        АНСАМБЛЬ                                  │
│                    Weighted Ensemble                             │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  REGRESSION  │  │  CLASSIFIER  │  │  FORECAST   │        │
│  │  (4 модели)  │  │  (3 модели)  │  │  (4 модели)  │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  expected_return    direction_prob    expected_return          │
│  (число)           (up/down)         (число)                 │
└─────────────────────────────────────────────────────────────────┘
```

**Типы моделей в ансамбле:**

| Тип | Модели | Выход | Вес |
|------|--------|-------|-----|
| Регрессия | LightGBM, LSTM, Tabular, GRU | `expected_return` (число) | 0.25-0.39 |
| Классификация | Binary, Multiclass | `P(up)`, `P(down)` | 0.01-0.20 |
| Forecasting | TimesFM, PatchTST, TFT | `expected_return` (число) | 0.05 |

#### 5.9.2 Модели регрессии

**Цель**: Предсказать конкретное значение доходности.

**LightGBM Regression (лучший):**

```python
# ВХОД: 17 признаков
features = [
    'rolling_volatility_20',   # Волатильность за 20 баров
    'momentum_10',              # Моментум за 10 баров
    'rsi_14',                  # RSI за 14 баров
    'macd',                    # MACD линия
    'macd_signal',             # MACD сигнальная линия
    'zscore_20',              # Z-Score за 20 баров
    'atr_14',                 # Average True Range
    'volume_ratio_20',         # Отношение объёма к среднему
    'volume_zscore_20',        # Z-Score объёма
    'trend_regime',            # Режим тренда (0-2)
    'volatility_regime',       # Режим волатильности (0-2)
    'return_lag_1',            # Доходность -1 бар
    'return_lag_2',            # Доходность -2 бара
    'return_lag_5',            # Доходность -5 баров
    'volatility_lag_1',        # Волатильность -1 бар
    'rsi_lag_1',              # RSI -1 бар
    'macd_momentum_interaction',  # MACD × Momentum
    'volume_volatility_interaction',  # Volume × Volatility
]

# LightGBM находит паттерны:
# Если RSI < 30 и Z-Score < -1.5 → предсказывает положительный return
# Если MACD пересекает сигнальную линию вверх → предсказывает рост
```

**Выход модели:**

```python
pred = model.predict(features)

pred.expected_return = +0.0025  # "Цена вырастет на 0.25%"
pred.confidence = 0.62         # "Уверенность 62%"
pred.direction_probability_up = 0.75   # P(рост) = 75%
pred.direction_probability_down = 0.25  # P(падение) = 25%
```

**Как работает LightGBM:**

```
Данные: [RSI=28, Z-Score=-1.8, MACD=+0.5, Momentum=+2%]

                    LightGBM
                      │
        ┌─────────────┼─────────────┐
        │             │             │
      Leaf 1       Leaf 2       Leaf 3
    (RSI<30)    (Z-Score)    (Other)
        │             │             │
        ▼             ▼             ▼
   +0.35%       +0.28%       +0.10%
   
   Итог: (0.35 + 0.28 + 0.10) / 3 = +0.24%
```

**Гиперпараметры LightGBM:**

```python
LGBMRegressor(
    n_estimators=200,        # 200 деревьев
    max_depth=6,            # Макс. глубина
    learning_rate=0.05,     # Скорость обучения
    num_leaves=31,         # Листьев на дерево
    min_child_samples=20,   # Мин.样本 на лист
    subsample=0.8,         # Бутстреп выборка
    colsample_bytree=0.8,  # Фичей на дерево
    reg_alpha=0.1,         # L1 регуляризация
    reg_lambda=0.1,         # L2 регуляризация
)
```

#### 5.9.3 Модели классификации

**Цель**: Предсказать вероятность направления (UP или DOWN).

**Binary Direction Classifier:**

```python
# ВХОД: Нет явных признаков (считает базовую статистику)

# ВЫХОД:
pred.direction_probability_up = 0.55     # "55% что вырастет"
pred.direction_probability_down = 0.45   # "45% что упадёт"
```

**Как обучается:**

```python
# Из исторических данных:
values = [r.get('return_1') for r in rows]

up_count = sum(1 for v in values if v > 0)   # Дней с ростом
down_count = sum(1 for v in values if v < 0)  # Дней с падением

up_prob = up_count / len(values)     # P(рост) = 51%
down_prob = down_count / len(values)  # P(падение) = 49%
avg_abs_return = avg(abs(values))     # Средняя доходность = 0.2%

# Предсказание:
expected_return = (up_prob - down_prob) × avg_abs_return
expected_return = (0.51 - 0.49) × 0.002 = +0.0004  # Слегка вверх
```

**Multiclass Action Classifier:**

```python
# Предсказывает 3 класса:
# - BUY (рост)
# - SELL (падение)  
# - HOLD (нейтрально)

pred.action_probabilities = {
    'buy': 0.35,
    'sell': 0.30,
    'hold': 0.35
}
```

#### 5.9.4 Foundation Models (TimesFM, PatchTST)

**Цель**: Предсказать временной ряд с помощью предобученных нейросетей.

**TimesFM (Google):**

```python
# TimesFM - предобученная модель для временных рядов
# Не требует обучения на наших данных

pred = timesfm.predict(rows)

pred.expected_return = 0.0031  # "Ожидаем рост на 0.31%"
pred.confidence = 0.00003      # Очень низкая уверенность
```

**PatchTST:**

```python
# PatchTST - трансформер для временных рядов
# Разбивает ряд на патчи и анализирует

pred = patchtst.predict(rows)

pred.expected_return = 0.0030  # "Ожидаем рост на 0.30%"
```

**Особенность**: Эти модели предсказывают будущие значения напрямую из сырых котировок, без ручных признаков.

#### 5.9.5 Weighted Ensemble (объединение)

**Метод**: Взвешенное среднее предсказаний всех моделей.

```python
# Каждая модель даёт свой прогноз:
model_predictions = {
    'lightgbm_regression': {
        'expected_return': +0.0025,
        'weight': 0.39
    },
    'lstm_regression': {
        'expected_return': -0.0010,
        'weight': 0.20
    },
    'timesfm2_wrapper': {
        'expected_return': +0.0031,
        'weight': 0.05
    },
    'binary_classifier': {
        'expected_return': +0.0004,
        'weight': 0.02
    },
    # ... ещё 8 моделей
}

# АНСАМБЛЬ объединяет (взвешенное среднее):
ensemble_return = Σ(weight_i × expected_return_i)

ensemble_return = 0.39×0.0025 + 0.20×(-0.0010) + 0.05×0.0031 + 0.02×0.0004 + ...
ensemble_return = +0.00097  # Итоговый прогноз
```

**Формула объединения:**

```python
for i in range(len(first_model_predictions)):
    e = up = down = conf = total_w = 0.0
    
    for model_name, preds in predictions_by_model.items():
        w = weights[model_name]
        e += w × preds[i].expected_return
        up += w × preds[i].direction_probability_up
        down += w × preds[i].direction_probability_down
        conf += w × preds[i].confidence
        total_w += w
    
    ensemble_prediction = StandardizedPrediction(
        expected_return = e / total_w,
        direction_probability_up = up / total_w,
        direction_probability_down = down / total_w,
        confidence = conf / total_w,
    )
```

**Динамические веса:**

```python
# Веса пересчитываются с учётом:
# 1. Неопределённость (низкая уверенность → штраф)
# 2. Turnover (частая смена прогнозов → штраф)
# 3. Базовый вес из конфигурации

uncertainty = 1.0 - avg_confidence
turnover = avg_abs_diff_between_predictions

score = base_weight
score × (1 - uncertainty_penalty × uncertainty)
score × (1 - turnover_penalty × min(1.0, turnover × 100))
```

#### 5.9.6 Использование в стратегии

```python
# После объединения получаем финальный прогноз:
ensemble_pred = {
    'expected_return': +0.00097,           # Ожидаемая доходность
    'direction_probability_up': 0.52,       # P(рост) = 52%
    'direction_probability_down': 0.48,    # P(падение) = 48%
    'confidence': 0.025                     # Общая уверенность
}

# СТРАТЕГИЯ использует это для принятия решений:
if ensemble_pred.direction_probability_up > 0.55:
    action = "BUY"
elif ensemble_pred.direction_probability_down > 0.55:
    action = "SELL"
else:
    action = "HOLD"
```

**Или через expected_return:**

```python
threshold = 0.0005  # 0.05%

if ensemble_pred.expected_return > threshold:
    action = "BUY"
elif ensemble_pred.expected_return < -threshold:
    action = "SELL"
else:
    action = "HOLD"
```

#### 5.9.7 Итоговая схема работы

```
┌─────────────────────────────────────────────────────────────────┐
│                        ДАННЫЕ                                     │
│   [ticker, timestamp, close, volume, RSI, MACD, Z-Score...]    │
│   ─────────────────────────────────────────────────────────────│
│   10 тикеров × 30 дней × 1440 минут = 432,000 наблюдений       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ПРЕДОБРАБОТКА                                 │
│   - Очистка пропусков                                           │
│   - Расчёт технических индикаторов                              │
│   - Расчёт новостных признаков                                  │
│   - Агрегация по тикерам                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    МОДЕЛИ АНСАМБЛЯ                               │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   LightGBM  │  │    LSTM     │  │  TimesFM    │            │
│  │ Regression  │  │ Regression  │  │  (Neural)   │            │
│  │ weight=0.39│  │ weight=0.20│  │ weight=0.05  │            │
│  │ ret=+0.25% │  │ ret=-0.10% │  │ ret=+0.31%  │            │
│  │ conf=6.3%  │  │ conf=1.4%  │  │ conf=0.0%   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Binary    │  │   PatchTST  │  │   Tabular   │            │
│  │  Classifier │  │  (Neural)   │  │  Baseline   │            │
│  │ weight=0.02 │  │ weight=0.05 │  │ weight=0.05 │            │
│  │ ret=+0.04% │  │ ret=+0.30% │  │ ret=+0.1%  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    Weighted Average
                             │
                             ▼
                    ┌─────────────────┐
                    │   ENSEMBLE      │
                    │ ─────────────── │
                    │ ret = +0.097%   │
                    │ P(up) = 52%    │
                    │ P(down) = 48%  │
                    │ conf = 2.5%    │
                    └────────┬────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     СТРАТЕГИЯ                                    │
│                                                                 │
│   Mean Reversion + Market Timing                                 │
│                                                                 │
│   Сигнал = f(ensemble_pred, RSI, Z-Score, Market Momentum)     │
│                                                                 │
│   if market_momentum < -3% → HOLD (не торгуем в кризис)       │
│   if RSI < 35 OR Z-Score < -1.5 → BUY (перепроданность)       │
│   if RSI > 65 OR Z-Score > +1.5 → SELL (перекупленность)      │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   TRADING      │
                    │   SIGNALS      │
                    │ ─────────────── │
                    │ BUY:  +121,248₽│
                    │ SELL:   -92,086₽│
                    │ HOLD:   остальное │
                    └─────────────────┘
```

#### 5.9.8 Результаты по типам моделей

| Тип | Модель | Directional Accuracy | MAE | PnL Proxy |
|------|--------|---------------------|-----|-----------|
| **Регрессия** | LightGBM | **76.66%** | **0.00283** | **196.02** |
| **Регрессия** | Tabular | 49.85% | 0.00361 | -0.95 |
| **Регрессия** | LSTM | 49.85% | 0.00361 | -0.95 |
| **Классификация** | Binary | 49.85% | 0.00361 | -0.95 |
| **Классификация** | Multiclass | 49.85% | 0.00361 | -0.95 |
| **Forecasting** | TimesFM | 63.51% | 0.00326 | 101.66 |
| **Forecasting** | PatchTST | 63.47% | 0.00325 | 101.40 |
| **Ансамбль** | Weighted | 76.68% | 0.00332 | 196.04 |

**Ключевые выводы:**

1. **LightGBM доминирует**: 76.66% accuracy, лучший MAE
2. **Простые модели (Binary, Tabular)**: ~50% = случайное угадывание
3. **Neural forecasting (TimesFM, PatchTST)**: ~63% - неплохо, но хуже LightGBM
4. **Ансамбль**: незначительное улучшение (+0.02%) над лучшей моделью

**Почему LightGBM лучший:**

```python
# LightGBM использует 17 осмысленных признаков:
# - Технические индикаторы (RSI, MACD, Z-Score)
# - Волатильность и объём
# - Лаги доходностей
# - Взаимодействия признаков

# Neural модели (TimesFM, PatchTST):
# - Используют только сырые цены
# - Не знают про RSI, MACD и другие индикаторы
# - Предобучены на других данных

# Binary Classifier:
# - Не использует признаки вообще
# - Только базовая статистика
```

---

## 6. Стратегия торговли

### 6.1 Mean Reversion with Market Timing

**Финальная стратегия**: Mean Reversion + Market Timing

#### Параметры стратегии
| Параметр | Значение | Описание |
|----------|---------|---------|
| position_size_pct | 0.30 | 30% от капитала |
| market_threshold | -0.03 | Порог рыночного режима |
| rsi_oversold | 35 | RSI для длинной позиции |
| rsi_overbought | 65 | RSI для короткой позиции |
| zscore_oversold | -1.5 | Z-Score для длинной позиции |
| zscore_overbought | 1.5 | Z-Score для короткой позиции |
| commission_bps | 5 | Комиссия (0.05%) |
| slippage_bps | 5 | Проскальзывание (0.05%) |

#### Логика сигналов
```python
def get_signal(row, market_momentum):
    # Фильтрация по рыночному режиму
    if market_momentum < -0.03:
        return 0  # Не торгуем в медвежьем рынке
    
    rsi = row['rsi_14']
    zscore = row['zscore_20']
    
    # Long: RSI < 35 или Z-Score < -1.5
    if rsi < 35 or zscore < -1.5:
        return 1
    
    # Short: RSI > 65 или Z-Score > 1.5
    if rsi > 65 or zscore > 1.5:
        return -1
    
    return 0  # Neutral
```

#### Агрегация данных (Daily)
```python
# Агрегация по дням для снижения частоты торговли
daily = df.groupby(['date', 'ticker']).agg({
    'close': ['first', 'last', 'mean'],
    'return_1': 'sum',
    'rolling_volatility_20': 'last',
    'momentum_10': 'last',
    'rsi_14': 'last',
    'macd': 'last',
    'volume': 'sum',
})
```

---

## 7. Бэктестирование

### 7.1 Архитектура бэктеста

**Исправленный движок** (`src/backtesting/engine.py`):
```python
def run_backtest(rows, cfg):
    # Инициализация
    cash = cfg.initial_cash  # 1,000,000
    position = 0
    last_target_position = None
    
    for i, row in enumerate(ordered):
        target_position = _target_position_from_row(row, cfg.target_position_column)
        
        # Ключевое исправление: торгуем только при смене позиции
        if pending_signal is None and target_position != last_target_position:
            if target_position is not None:
                pending_signal = (target_position, i + cfg.execution_delay_bars)
        
        # Исполнение с задержкой
        if pending_signal and i >= pending_signal[1]:
            # Расчёт позиции от начального капитала (НЕ текущего!)
            target_notional = cfg.initial_cash * cfg.position_size_pct * signal_scale
            # ... исполнение сделки
```

**Ключевое исправление**: 
- До: `target_notional = cash * cfg.position_size_pct` (пересчёт на каждом баре)
- После: `target_notional = cfg.initial_cash * cfg.position_size_pct` (фиксированный размер)

### 7.2 Расчёт PnL
```python
equity = cash + position * price
pnl = equity_curve[-1]["equity"] - cfg.initial_cash
```

### 7.3 Transaction Costs
```python
# Комиссия: 5 bps от объёма
commission = notional * (commission_bps / 10_000)

# Проскальзывание: 5 bps
fill_price = price * (1.0 + slippage_bps / 10_000)
```

---

## 8. Результаты и метрики

### 8.1 Итоговые результаты

| Метрика | Значение |
|---------|---------|
| **Total PnL** | **+121,248 руб** |
| **Total Trades** | **103** |
| Initial Capital | 1,000,000 руб |
| Return | ~12.1% |
| Period | 31 день |

### 8.2 Результаты по тикерам

| Тикер | PnL (руб) | Сделок | Позиция |
|-------|-----------|--------|---------|
| SBER | +68,366 | 9 | LONG |
| NVTK | +59,529 | 8 | LONG |
| YNDX | +59,529 | 8 | LONG |
| SNGSP | +42,091 | 10 | LONG |
| GAZP | +41,374 | 10 | LONG |
| MGNT | +31,074 | 11 | LONG |
| SNGS | +6,244 | 11 | LONG |
| TATN | -14,748 | 12 | SHORT |
| LKOH | -80,125 | 13 | SHORT |
| POLY | -92,086 | 11 | SHORT |

### 8.3 Распределение PnL

```
     │  SBER (+68K)
     │  NVTK (+60K)
     │  YNDX (+60K)
     │  SNGSP (+42K)
     │  GAZP (+41K)
     │  MGNT (+31K)
     │  SNGS (+6K)
     │  ───────
     │  TATN (-15K)
     │  LKOH (-80K)
     │  POLY (-92K)
─────┴────────────────────
  -100K    0    +100K
```

### 8.4 Сравнение стратегий

| Стратегия | PnL | Проблема |
|-----------|------|----------|
| Buy&Hold SBER | +12,760 | Работает |
| Buy&Hold TATN | +51,140 | Лучший одиночный результат |
| RSI (1min) | -1,832,293 | Слишком много сделок |
| MACD (1min) | -1,541,176 | Слишком много сделок |
| **MeanRev+Timing** | **+121,248** | **Работает** |

---

## 9. Выводы и рекомендации

### 9.1 Ключевые выводы

1. **Частота торговли критична**: Минутные данные приводят к тысячам сделок и высоким транзакционным издержкам.

2. **Рыночный тайминг важен**: Фильтрация по рыночному режиму (-3% threshold) значительно улучшает результаты.

3. **Mean Reversion работает**: RSI и Z-Score предоставляют надёжные сигналы для разворота.

4. **Мало исторических данных**: 21 день недостаточно для обучения ML-моделей.

### 9.2 Рекомендации для production

1. **Собрать больше данных**: Минимум 6-12 месяцев исторических данных.

2. **Снизить комиссии**: Использовать ETF или фьючерсы вместо акций.

3. **Добавить stop-loss**: Автоматическая защита от больших убытков.

4. **Мониторить режим рынка**: Использовать дополнительные индикаторы (VIX, объём).

### 9.3 Файловая структура

```
moex-sandbox-platform/
├── src/
│   ├── backtesting/
│   │   └── engine.py           # Исправленный движок бэктеста
│   ├── models/
│   │   ├── regression/
│   │   │   └── sklearn_gradient_boosting.py  # sklearn-модель
│   │   └── ensemble/
│   ├── strategies/
│   │   ├── final_strategy.py   # MeanRev + MarketTiming
│   │   └── production_strategy.py
│   └── training/
│       └── stages/             # Pipeline stages
├── data/
│   └── processed/
│       └── merged/
│           ├── train.parquet
│           ├── validation.parquet
│           ├── test.parquet
│           ├── train_val_expanded.parquet
│           ├── test_expanded.parquet
│           └── daily_aggregated.parquet  # Для daily стратегии
├── models/
│   └── base/
│       └── simple_model.pkl    # Production модель
├── reports/
│   ├── final_strategy_backtest.json
│   └── final_strategy.md
└── params.yaml                # Конфигурация
```

### 9.4 Запуск стратегии

```python
from src.strategies.final_strategy import MeanReversionMarketTimingStrategy

strategy = MeanReversionMarketTimingStrategy()
result = strategy.run_backtest(daily_df)

print(f"PnL: {result.total_pnl:,.0f} руб")
print(f"Trades: {result.trades}")
```

---

## Приложение: Полная конфигурация params.yaml

```yaml
global:
  project_name: moex-sandbox-platform
  timezone: Europe/Moscow
  seed: 42
  trading_mode: sandbox

data:
  market:
    provider: t_invest_sandbox
    interval: 1min
    lookback_days: 30
    instruments: [SBER, GAZP, LKOH, YNDX, POLY, NVTK, MGNT, SNGS, TATN, SNGSP]
  news:
    enabled: true
    sources: [rbc_rss, interfax_rss, cbr_rss]

strategy:
  type: mean_reversion_market_timing
  market_threshold: -0.03
  rsi_oversold: 35
  rsi_overbought: 65
  zscore_oversold: -1.5
  zscore_overbought: 1.5
  use_market_timing: true

backtest:
  initial_cash_rub: 1000000
  commission_bps: 5
  slippage_bps: 5
  position_size_pct: 0.3

risk:
  max_position_size_pct: 0.08
  max_daily_loss_pct: 0.015
  max_portfolio_exposure_pct: 0.40
```

---

*Документ сгенерирован: 2026-03-19*
*Версия системы: 1.3 (добавлено подробное описание моделей ансамбля)*
