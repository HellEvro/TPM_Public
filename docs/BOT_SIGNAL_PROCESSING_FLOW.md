# 🤖 КАК БОТ ОТРАБАТЫВАЕТ СИГНАЛЫ

## 📊 ОБЩАЯ СХЕМА РАБОТЫ

```
┌─────────────────────────────────────────────────────────────┐
│              CONTINUOUS DATA LOADER (Основной цикл)          │
│                    Обновление каждые ~23 сек                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ЭТАП 1: Загрузка свечей (load_all_coins_candles_fast)      │
│  ├─ Загружает свечи 6H для всех 573 монет                   │
│  └─ Кэширует в coins_rsi_data['candles_cache']              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ЭТАП 2: Расчет RSI (load_all_coins_rsi)                    │
│  ├─ Для каждой монеты:                                       │
│  │  ├─ get_coin_rsi_data(symbol)                            │
│  │  │  ├─ Рассчитывает RSI 6H                               │
│  │  │  ├─ Получает Optimal EMA периоды                      │
│  │  │  ├─ Определяет базовый сигнал (ENTER_LONG/SHORT/WAIT) │
│  │  │  │  └─ На основе Optimal EMA периодов!                │
│  │  │  ├─ Применяет Enhanced RSI анализ                     │
│  │  │  │  ├─ Volatility (волатильность)                     │
│  │  │  │  ├─ Divergence (дивергенции)                       │
│  │  │  │  ├─ Volume (объемы)                                │
│  │  │  │  └─ Stochastic RSI                                 │
│  │  │  ├─ Проверяет зрелость монеты                         │
│  │  │  ├─ Проверяет ExitScam фильтр                         │
│  │  │  └─ Проверяет RSI временной фильтр                    │
│  │  └─ Сохраняет в coins_rsi_data['coins'][symbol]          │
│  └─ Результат: 569 монет с полными данными                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ЭТАП 3: Расчет зрелости (calculate_all_coins_maturity)     │
│  ├─ СУПЕР-ОПТИМИЗАЦИЯ: Пропускает если не изменилось        │
│  └─ Проверяет только новые/измененные монеты                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ЭТАП 4: Анализ трендов (analyze_trends_for_signal_coins)   │
│  ├─ Анализирует ТОЛЬКО сигнальные монеты (RSI <=29 или >=71)│
│  └─ Определяет тренд 6H: UP/DOWN/NEUTRAL                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ЭТАП 5: Фильтрация (process_long_short_coins_with_filters) │
│  ├─ Берет монеты с ENTER_LONG/ENTER_SHORT                   │
│  └─ Дополнительные проверки фильтров                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ЭТАП 6: Автобот (_set_filtered_coins_for_autobot)          │
│  └─ Сохраняет отфильтрованные монеты в конфиг автобота      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              AUTO BOT WORKER (Каждые 60 секунд)              │
│  process_auto_bot_signals()                                  │
│  ├─ Берет отфильтрованные монеты                            │
│  ├─ Проверяет, нет ли уже ботов на этих монетах             │
│  ├─ Проверяет, нет ли ручных позиций на бирже               │
│  └─ Создает новых ботов: create_new_autobot(symbol)         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           ОБРАБОТКА СИГНАЛОВ АКТИВНЫХ БОТОВ                  │
│  process_trading_signals_for_all_bots()                      │
│  (Вызывается каждый раунд ContinuousDataLoader)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
```

## 🎯 ДЕТАЛЬНЫЙ ЦИКЛ ОБРАБОТКИ СИГНАЛОВ ДЛЯ ОДНОГО БОТА

### 1️⃣ **ИНИЦИАЛИЗАЦИЯ БОТА**
```python
bot = NewTradingBot(symbol, config, exchange)
```
**Параметры:**
- `symbol`: Название монеты (например, "BTC")
- `config`: Конфигурация бота
  - `volume_mode`: "usdt" или "percent"
  - `volume_value`: Размер позиции (например, 10 USDT)
  - `status`: Текущий статус (IDLE, IN_POSITION_LONG, IN_POSITION_SHORT)
  - `entry_price`: Цена входа
  - `position_side`: LONG или SHORT

---

### 2️⃣ **ОСНОВНОЙ ЦИКЛ: `bot.update()`**

Этот метод вызывается из `process_trading_signals_for_all_bots()`:

```python
# Для каждого активного бота:
signal_result = trading_bot.update(
    force_analysis=True, 
    external_signal=external_signal,  # Из coins_rsi_data
    external_trend=external_trend     # Из coins_rsi_data
)
```

**Что делает `update()`:**

1. **Получает данные:**
   ```python
   - current_price   # Текущая цена из coins_rsi_data
   - current_rsi     # RSI 6H из coins_rsi_data
   - current_trend   # UP/DOWN/NEUTRAL из coins_rsi_data
   - candles         # Свечи 6H за 30 дней с биржи
   ```

2. **Определяет действие по статусу:**
   ```python
   if status == IDLE:
       → _handle_idle_state()      # Ищет возможность входа
   elif status == IN_POSITION_LONG/SHORT:
       → _handle_position_state()  # Управляет позицией
   ```

---

### 3️⃣ **СОСТОЯНИЕ IDLE: ПОИСК ТОЧКИ ВХОДА**

**`_handle_idle_state(rsi, trend, candles, price)`**

#### A) Проверка LONG позиции (`should_open_long`)

```python
✅ ОТКРЫВАЕМ LONG если:
   1. RSI <= 29 (по умолчанию, настраивается)
   2. Тренд НЕ DOWN (если avoid_down_trend = True)
   3. RSI Time Filter разрешает:
      - Ищет в последних 8 свечах (настраивается)
      - RSI был >= 35 (настраивается)
      - Цель: избежать входа в "мертвую зону"
```

**Пример логики:**
```python
if rsi <= 29:                    # RSI в зоне перепроданности
    if trend != 'DOWN':          # Тренд не вниз
        if check_rsi_time_filter():  # Недавно RSI был выше 35
            → ОТКРЫВАЕМ LONG! 🚀
```

#### B) Проверка SHORT позиции (`should_open_short`)

```python
✅ ОТКРЫВАЕМ SHORT если:
   1. RSI >= 71 (по умолчанию, настраивается)
   2. Тренд НЕ UP (если avoid_up_trend = True)
   3. RSI Time Filter разрешает:
      - Ищет в последних 8 свечах
      - RSI был <= 65 (настраивается)
```

#### C) Открытие позиции на бирже

```python
def _open_position_on_exchange(side, price):
    # 1. Рассчитывает количество монет
    if volume_mode == 'usdt':
        qty = volume_value  # Например, 10 USDT
    
    # 2. Отправляет ордер на биржу
    result = exchange.place_order(
        symbol=symbol,
        side=side,           # 'LONG' или 'SHORT'
        order_type='Market',
        qty=qty,
        marketUnit='quoteCoin'  # ⚡ Указываем qty в USDT!
    )
    
    # 3. Сохраняет данные позиции
    if result['success']:
        self.order_id = result['order_id']
        self.entry_price = price
        self.entry_timestamp = result['timestamp']
        self.position_side = side
        
        # 4. Обновляет статус
        self.status = IN_POSITION_LONG или IN_POSITION_SHORT
        
        return True
```

---

### 4️⃣ **СОСТОЯНИЕ IN_POSITION: УПРАВЛЕНИЕ ПОЗИЦИЕЙ**

**`_handle_position_state(rsi, trend, candles, price)`**

#### A) Обновление защитных механизмов

```python
# 1. Обновляем UnrealizedPnL
current_pnl_percent = ((current_price - entry_price) / entry_price) * 100
if position_side == 'SHORT':
    current_pnl_percent *= -1

self.unrealized_pnl = current_pnl_percent

# 2. Отслеживаем максимальную прибыль
if current_pnl_percent > self.max_profit_achieved:
    self.max_profit_achieved = current_pnl_percent

# 3. Активируем Break-Even (если прибыль >= 2%)
if current_pnl_percent >= 2.0 and not self.break_even_activated:
    self.break_even_activated = True
    logger.info(f"✅ Break-Even активирован при {current_pnl_percent:.2f}%")

# 4. Обновляем Trailing Stop
if self.break_even_activated:
    # Trailing Stop = max_profit - 1.5%
    self.trailing_stop_price = self.max_profit_achieved - 1.5
```

#### B) Проверка условий выхода

```python
✅ ЗАКРЫВАЕМ LONG позицию если:
   1. RSI >= 71 (перекупленность)
   2. ИЛИ Тренд изменился на DOWN
   3. ИЛИ Trailing Stop сработал (прибыль упала на 1.5% от макс)
   4. ИЛИ Break-Even сработал (прибыль < 0% после активации)

✅ ЗАКРЫВАЕМ SHORT позицию если:
   1. RSI <= 29 (перепроданность)
   2. ИЛИ Тренд изменился на UP
   3. ИЛИ Trailing Stop сработал
   4. ИЛИ Break-Even сработал
```

**Пример:**
```python
if position_side == 'LONG':
    if rsi >= 71:
        → ЗАКРЫВАЕМ LONG (выход по RSI) 📊
    elif trend == 'DOWN':
        → ЗАКРЫВАЕМ LONG (смена тренда) 📉
    elif trailing_stop_triggered:
        → ЗАКРЫВАЕМ LONG (trailing stop) 🛑
    elif break_even_triggered:
        → ЗАКРЫВАЕМ LONG (break-even) 🔒
```

#### C) Закрытие позиции на бирже

```python
def _close_position_on_exchange():
    # 1. Отправляет ордер на закрытие
    result = exchange.close_position(
        symbol=symbol,
        side='LONG' или 'SHORT'
    )
    
    # 2. Сохраняет результат в историю
    if result['success']:
        realized_pnl = result['realized_pnl']
        
        # 3. Сбрасывает состояние бота
        self.status = BOT_STATUS['IDLE']
        self.position_side = None
        self.entry_price = None
        self.unrealized_pnl = 0
        self.max_profit_achieved = 0
        self.trailing_stop_price = None
        self.break_even_activated = False
        
        return True
```

---

## ⚙️ НАСТРОЙКИ ТОРГОВЫХ ПРАВИЛ

### 🎯 **RSI Thresholds (Пороги RSI)**
```python
rsi_long_threshold = 29   # Открываем LONG если RSI <= 29
rsi_short_threshold = 71  # Открываем SHORT если RSI >= 71
```

### 📈 **Trend Following (Следование тренду)**
```python
avoid_down_trend = True   # Не открываем LONG на DOWN тренде
avoid_up_trend = True     # Не открываем SHORT на UP тренде
```

### ⏱️ **RSI Time Filter (Временной фильтр RSI)**
```python
rsi_time_filter_enabled = True
rsi_time_filter_candles = 8      # Анализируем последние 8 свечей
rsi_time_filter_lower = 35       # Для LONG: RSI был >= 35
rsi_time_filter_upper = 65       # Для SHORT: RSI был <= 65
```

### 🛡️ **Enhanced RSI (Улучшенный RSI)**
```python
enhanced_rsi_enabled = True
- Volatility check (проверка волатильности)
- Divergence check (поиск дивергенций)
- Volume confirmation (подтверждение объемами)
- Stochastic RSI (дополнительный осциллятор)
```

### 🚨 **Exit Scam Filter (Фильтр памп/дамп)**
```python
exit_scam_enabled = True
exit_scam_candles = 10              # Анализ последних 10 свечей
single_candle_percent = 15.0        # Одна свеча > 15% = блок
multi_candle_count = 4              # 4 свечи подряд
multi_candle_percent = 50.0         # Суммарно > 50% = блок
```

### 🛡️ **Защитные механизмы в позиции**
```python
# Break-Even: фиксация безубытка
break_even_threshold = 2.0%         # Активируется при +2%
break_even_trigger = 0.0%           # Закрывает при 0%

# Trailing Stop: защита прибыли
trailing_stop_activation = 2.0%     # Активируется при +2%
trailing_stop_distance = 1.5%       # Дистанция от максимума
```

---

## 🔄 ЧАСТОТА ОБНОВЛЕНИЙ

```
Continuous Data Loader: ~23 секунды (один полный раунд)
├─ Свечи: обновляются каждый раунд
├─ RSI: пересчитывается каждый раунд
├─ Зрелость: пересчитывается только при изменениях
└─ Тренды: только для сигнальных монет

Auto Bot Worker: 60 секунд
└─ Создает новых ботов для отфильтрованных монет

Обработка сигналов: каждый раунд (~23 сек)
└─ process_trading_signals_for_all_bots()
```

---

## 📊 ПРИМЕР ПОЛНОГО ЦИКЛА

### Сценарий: Открытие и закрытие LONG позиции

```
Раунд #1 (00:00):
├─ RSI = 28 (< 29) ✅
├─ Trend = NEUTRAL ✅
├─ RSI Time Filter: OK ✅
└─ 🚀 ОТКРЫВАЕМ LONG @ 45,000 USDT

Раунд #2 (00:23):
├─ Price = 45,500 (+1.1%)
├─ RSI = 32
├─ UnrealizedPnL = +1.1%
└─ ⏳ ДЕРЖИМ позицию

Раунд #3 (00:46):
├─ Price = 46,000 (+2.2%)
├─ RSI = 38
├─ UnrealizedPnL = +2.2%
├─ ✅ Break-Even АКТИВИРОВАН
└─ ⏳ ДЕРЖИМ позицию

Раунд #4 (01:09):
├─ Price = 46,800 (+4.0%)
├─ RSI = 45
├─ UnrealizedPnL = +4.0%
├─ Max Profit = +4.0%
├─ Trailing Stop = +2.5% (4.0 - 1.5)
└─ ⏳ ДЕРЖИМ позицию

Раунд #5 (01:32):
├─ Price = 47,500 (+5.6%)
├─ RSI = 72 (> 71) ❌
└─ 📊 ЗАКРЫВАЕМ LONG @ 47,500 (причина: RSI перекуплен)
    Realized PnL: +5.6% 🎉
```

---

## 🎯 КЛЮЧЕВЫЕ МОМЕНТЫ

1. **Бот работает на свечах 6H** - это значит, что он реагирует на долгосрочные движения, а не на шум

2. **Множественные фильтры** - монета должна пройти 7+ проверок перед входом

3. **Защитные механизмы** - Break-Even и Trailing Stop защищают прибыль

4. **Автоматическое создание ботов** - Auto Bot Worker создает ботов для новых сигналов

5. **Непрерывное обновление** - данные обновляются каждые ~23 секунды

6. **Умная оптимизация** - кэширование и пропуск ненужных пересчетов

7. **Enhanced RSI** - дополнительная аналитика повышает точность сигналов

---

## 📝 ЛОГИРОВАНИЕ

Каждый этап логируется:
```
[NEW_BOT_BTC] 🤖 Инициализация нового торгового бота
[NEW_BOT_BTC] ✅ Все проверки пройдены - открываем LONG (RSI: 28.5, Trend: NEUTRAL)
[NEW_BOT_BTC] 🚀 Открываем LONG позицию @ 45000 USDT
[NEW_BOT_BTC] ✅ Break-Even активирован при +2.20%
[NEW_BOT_BTC] 📊 Закрываем LONG позицию (RSI: 72.1 >= 71)
[NEW_BOT_BTC] 🎉 Realized PnL: +5.60%
```

---

**Система полностью автоматизирована и работает 24/7!** 🚀

