# Полный пайплайн торговой стратегии MOEX

## Содержание
1. [Обзор системы](#1-обзор-системы)
2. [Получение данных](#2-получение-данных)
3. [Препроцессинг данных](#3-препроцессинг-данных)
4. [Генерация признаков](#4-генерация-признаков)
5. [Архитектура моделей](#5-архитектура-моделей)
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

### 5.2 Foundation Models (TimesFM, PatchTST)

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

### 5.3 Ensemble (Weighted Average)

**Метод**: Взвешенное среднее с адаптивными весами.

**Веса (params.yaml)**:
```yaml
models:
  weights:
    lightgbm_regression: 0.25
    lstm_regression: 0.20
    xgboost_direction_classifier: 0.20
    xgboost_multiclass_classifier: 0.15
    news_feature_model: 0.10
    tabular_regression_baseline: 0.05
    gru_regression_skeleton: 0.02
    binary_direction_classifier: 0.02
    multiclass_action_classifier: 0.01
```

### 5.4 Policy Layer

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

*Документ сгенерирован: 2026-03-18*
*Версия системы: 1.0*
