# Механизм отслеживания ботов по их таймфреймам

## Обзор

Каждый бот отслеживается и управляется по своему собственному таймфрейму (`entry_timeframe`), в котором была открыта позиция. Это позволяет системе работать с позициями из разных таймфреймов одновременно, независимо от текущего системного таймфрейма.

## Как это работает

### 1. Сохранение таймфрейма при открытии позиции

**Файл:** `bots_modules/bot_class.py`, метод `enter_position()`

```python
# ✅ КРИТИЧНО: Сохраняем таймфрейм при входе в позицию
# Это позволяет боту работать со своим таймфреймом независимо от системного
from bot_engine.bot_config import get_current_timeframe
self.entry_timeframe = get_current_timeframe()
```

**Что происходит:**
- При открытии позиции (LONG или SHORT) сохраняется текущий системный таймфрейм в `self.entry_timeframe`
- Это значение сохраняется в БД в поле `entry_timeframe` таблицы `bots`
- Для старых позиций (без `entry_timeframe`) используется `'6h'` по умолчанию

### 2. Определение таймфрейма для работы бота

**Логика выбора таймфрейма:**

```python
# ✅ КРИТИЧНО: Определяем таймфрейм для работы бота
# Если бот в позиции - используем его entry_timeframe, иначе системный
if self.entry_timeframe and self.status in [
    BOT_STATUS.get('IN_POSITION_LONG'),
    BOT_STATUS.get('IN_POSITION_SHORT')
]:
    # Бот в позиции - используем его таймфрейм
    timeframe_to_use = self.entry_timeframe
else:
    # Бот не в позиции - используем системный таймфрейм
    from bot_engine.bot_config import get_current_timeframe
    timeframe_to_use = get_current_timeframe()
```

**Правило:**
- **Бот в позиции** → использует свой `entry_timeframe` (например, `'6h'`)
- **Бот не в позиции** → использует системный таймфрейм (например, `'1h'`)

### 3. Получение RSI данных с учетом таймфрейма бота

**Файл:** `bots_modules/bot_class.py`, метод `update()`

```python
# ✅ Используем таймфрейм бота для получения RSI
current_rsi = get_rsi_from_coin_data(coin_data, timeframe=timeframe_to_use)
current_trend = get_trend_from_coin_data(coin_data, timeframe=timeframe_to_use)
```

**Что происходит:**
- Функция `get_rsi_from_coin_data()` получает RSI по ключу `rsi{timeframe}` (например, `rsi6h` или `rsi1h`)
- Функция `get_trend_from_coin_data()` получает тренд по ключу `trend{timeframe}` (например, `trend6h` или `trend1h`)
- Если бот в позиции с `entry_timeframe='6h'`, он всегда будет использовать `rsi6h` и `trend6h`, даже если системный таймфрейм изменен на `'1h'`

### 4. Получение свечей с учетом таймфрейма бота

**Файл:** `bots_modules/bot_class.py`, метод `update()`

```python
# ✅ Получаем свечи для анализа с учетом таймфрейма бота
chart_response = self.exchange.get_chart_data(self.symbol, timeframe_to_use, '30d')
```

**Что происходит:**
- Свечи загружаются с биржи для конкретного таймфрейма бота
- Если бот в позиции с `entry_timeframe='6h'`, загружаются 6-часовые свечи
- Если системный таймфрейм изменен на `'1h'`, бот продолжает использовать 6-часовые свечи

### 5. Проверка сигналов закрытия

**Файл:** `bots_modules/filters.py`, функция `process_trading_signals_for_all_bots()`

```python
# ✅ КРИТИЧНО: Определяем таймфрейм для проверки сигналов
bot_entry_timeframe = bot_data.get('entry_timeframe')
if bot_entry_timeframe and bot_data.get('status') in [
    BOT_STATUS.get('IN_POSITION_LONG'),
    BOT_STATUS.get('IN_POSITION_SHORT')
]:
    # Бот в позиции - используем его таймфрейм
    timeframe_to_use = bot_entry_timeframe
else:
    # Бот не в позиции - используем системный таймфрейм
    timeframe_to_use = get_current_timeframe()

# ✅ Используем таймфрейм бота для получения RSI и тренда
current_rsi = get_rsi_from_coin_data(rsi_data, timeframe=timeframe_to_use)
current_trend = get_trend_from_coin_data(rsi_data, timeframe=timeframe_to_use)
```

**Что происходит:**
- При проверке условий закрытия позиции используется RSI и тренд для таймфрейма бота
- Бот, открытый в `'6h'`, проверяет условия закрытия по `rsi6h`, а не по `rsi1h`

### 6. Мониторинг позиций

**Файл:** `bots_modules/workers.py`, функция `positions_monitor_worker()`

```python
# ✅ КРИТИЧНО: Используем таймфрейм бота для проверки сигналов закрытия
bot_entry_timeframe = bot_data.get('entry_timeframe')
if not bot_entry_timeframe:
    # Если entry_timeframe не сохранен - используем системный (для старых позиций)
    bot_entry_timeframe = get_current_timeframe()

# ✅ Используем таймфрейм бота для получения RSI
current_rsi = get_rsi_from_coin_data(rsi_data, timeframe=bot_entry_timeframe)
```

## Пример работы

1. **Системный таймфрейм: `'6h'`**
   - Бот открывает позицию LONG для монеты XVS
   - Сохраняется `entry_timeframe = '6h'`

2. **Системный таймфрейм изменен на `'1h'`**
   - Бот XVS продолжает использовать `entry_timeframe = '6h'`
   - Проверка сигналов закрытия использует `rsi6h` и `trend6h`
   - Свечи загружаются для таймфрейма `'6h'`

3. **Новый бот открывает позицию**
   - Новый бот получает `entry_timeframe = '1h'` (текущий системный)
   - Он будет использовать `rsi1h` и `trend1h`

4. **Результат:**
   - Бот XVS (открыт в `'6h'`) отслеживается по 6-часовому таймфрейму
   - Новый бот (открыт в `'1h'`) отслеживается по 1-часовому таймфрейму
   - Оба бота работают независимо друг от друга

## Места в коде, где используется entry_timeframe

1. **`bots_modules/bot_class.py`**:
   - `enter_position()` - сохранение таймфрейма при открытии
   - `update()` - определение таймфрейма для работы бота
   - `__init__()` - загрузка таймфрейма из конфига (по умолчанию `'6h'`)

2. **`bots_modules/filters.py`**:
   - `process_trading_signals_for_all_bots()` - проверка сигналов с учетом таймфрейма бота

3. **`bots_modules/workers.py`**:
   - `positions_monitor_worker()` - мониторинг позиций с учетом таймфрейма бота

4. **`bots_modules/init_functions.py`**:
   - `process_trading_signals_on_candle_close()` - обработка сигналов при закрытии свечи

5. **`bot_engine/bots_database.py`**:
   - `save_bots_state()` - сохранение `entry_timeframe` в БД
   - `load_bots_state()` - загрузка `entry_timeframe` из БД (по умолчанию `'6h'`)

## Обратная совместимость

Для старых позиций, у которых нет `entry_timeframe`:
- При загрузке из БД устанавливается `entry_timeframe = '6h'` (все старые позиции были открыты в 6ч)
- Это гарантирует, что старые позиции продолжают работать корректно

## Важные моменты

1. **Таймфрейм сохраняется только при открытии позиции** - после открытия он не изменяется
2. **Боты не в позиции используют системный таймфрейм** - для новых входов
3. **RSI данные должны быть рассчитаны для всех используемых таймфреймов** - система рассчитывает RSI для всех таймфреймов, которые используются активными ботами
4. **Свечи загружаются по требованию** - каждый бот запрашивает свечи для своего таймфрейма при необходимости
