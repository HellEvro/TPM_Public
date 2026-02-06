# Аудит: вход и выход по RSI (критическая логика)

## Цель

Одна точка истины: **где** и **как** проверяется RSI при входе и выходе, чтобы исключить вход/выход вне заданных порогов.

---

## 1. ВХОД В СДЕЛКУ

### 1.1 Где формируется сигнал ENTER_LONG / ENTER_SHORT

| Место | Файл:строка | Условие |
|-------|-------------|---------|
| Базовый сигнал по RSI | `bots_modules/filters.py` ~1379–1394 | `rsi <= rsi_long_threshold` → ENTER_LONG; `rsi >= rsi_short_threshold` → ENTER_SHORT. Пороги: individual_settings → auto_bot_config. |
| Enhanced RSI | `filters.py` ~1444–1449 | Может переопределить signal на ENTER_* или WAIT. |

RSI считается в `get_coin_rsi_data()` по **последней закрытой свече** (candles из кэша/биржи). Ключ RSI — по текущему таймфрейму (`get_rsi_key(current_timeframe)`).

### 1.2 Где сигнал превращается в «можно входить»

| Место | Файл:строка | Проверка RSI |
|-------|-------------|--------------|
| get_effective_signal | `filters.py` ~2523–2627 | **Обязательно:** `rsi = get_rsi_from_coin_data(coin, timeframe=current_timeframe)`. Если `signal == 'ENTER_LONG'` и `rsi > rsi_long_threshold` → WAIT. Если `signal == 'ENTER_SHORT'` и `rsi < rsi_short_threshold` → WAIT. Пороги из auto_config (глобальные). |
| process_auto_bot_signals, список кандидатов | `filters.py` ~2723–2739 | Для каждой монеты: `signal = get_effective_signal(coin_data)` — только ENTER_* при RSI в пороге. |
| Перед созданием бота и входом | `filters.py` ~2872–2895 | **Повторная проверка:** `rsi_now = get_rsi_from_coin_data(coin_data_now, timeframe=current_timeframe)`. Если нет данных или RSI вне порога (LONG: rsi_now > long_th, SHORT: rsi_now < short_th) — вход **пропускается**, в лог пишется причина. |

Итого: вход из автобота возможен только если RSI по текущему ТФ в пороге в **трёх** местах: базовый сигнал, get_effective_signal, проверка перед входом.

### 1.3 Единственный путь входа БЕЗ проверки RSI

| Место | Файл:строка | Условие |
|-------|-------------|---------|
| create_bot_endpoint → enter_position_async | `api_endpoints.py` ~1103–1105 | Если `force_manual_entry and manual_direction`: направление задаётся вручную, **RSI не проверяется**. |

Все остальные входы идут через `process_auto_bot_signals` и проходят проверки выше.

### 1.4 Рекомендации по входу

- Не использовать «Принудительный вход» без необходимости — это единственный обход RSI.
- В логах перед каждым входом есть строка вида: `вход LONG/SHORT — RSI=…, порог … (ТФ=…)`. По ней можно проверить, что система видела RSI в пороге.
- Для быстрой реакции на 1m уменьшить `check_interval` (например 30–60 сек).

---

## 2. ВЫХОД ИЗ СДЕЛКИ

### 2.1 Где принимается решение «закрыть по RSI»

| Место | Файл:строка | Логика |
|-------|-------------|--------|
| process_trading_signals_for_all_bots | `filters.py` ~2967–3024 | Для каждого бота в позиции: `timeframe_to_use = bot_entry_timeframe or get_current_timeframe()`. `current_rsi = get_rsi_from_coin_data(rsi_data, timeframe=timeframe_to_use)`. LONG: `external_signal = 'EXIT_LONG'` если `current_rsi >= thr` (thr = exit_long_with или exit_long_against по entry_trend). SHORT: `external_signal = 'EXIT_SHORT'` если `current_rsi <= thr`. |
| trading_bot.update(external_signal=…) | `bot_class.py` ~1290–1344 | Сначала проверяются защиты (protection_result). Затем, если разрешено по min_candles/min_move: `should_close, reason = self.should_close_position(rsi, price, self.position_side)`. При should_close вызывается `_close_position_on_exchange(reason)`. |
| check_should_close_by_rsi / should_close_position | `bot_class.py` ~1023–1087 | Пороги: individual → bot_data → auto_config. LONG: выход при RSI >= порог (with/against trend). SHORT: RSI <= порог. |

Важно: RSI для решения о выходе берётся по **таймфрейму бота** (`entry_timeframe`), не по глобальному.

### 2.2 Закрытие НЕ по RSI (вне порога)

Позиция может закрыться **без** участия нашей логики по RSI:

| Причина | Где |
|---------|-----|
| Take Profit / Stop Loss на бирже | Биржа закрывает ордер по TP/SL. Наш код не вызывается. |
| Ручное закрытие | Пользователь закрыл в UI или на бирже. |
| Защиты (trailing, break-even, ликвидация и т.д.) | `check_protection_mechanisms` в `bot_class.py` — может вернуть should_close до проверки RSI. |
| Делистинг, аварийное закрытие | Отдельные сценарии в sync_and_cache / workers. |

Поэтому в отчёте «выход не по порогу» часто означает: закрытие по TP/SL, вручную или по защите, а не по решению «RSI достиг порога».

### 2.3 Рекомендации по выходу

- В логах при решении «закрыть по RSI» должна быть строка с символом, RSI, порогом и ТФ (добавлено явным логированием).
- Сверять отчёт с логами: если в логе есть «Закрываем LONG по RSI» и указан RSI/порог — закрытие по нашей логике; если такой строки нет — закрытие по TP/SL/ручное/защите.
- Для анализа конкретной сделки: время входа/выхода → поиск по лог-файлу по символу и времени.

---

## 3. ОБЩИЕ ТОЧКИ ОТКАЗА

- **Неверный таймфрейм:** RSI должен браться по одному и тому же ТФ (вход: текущий системный; выход: entry_timeframe бота). Fallback на другой ТФ в config_loader убран.
- **Устаревшие данные:** coins_rsi_data обновляется загрузчиком. Если в момент решения данные старые — RSI может не совпадать с «реальным» на бирже. Уменьшение check_interval и интервала загрузки снижает риск.
- **Индивидуальные пороги:** Для входа get_effective_signal использует **глобальные** пороги; для выхода используются individual → auto_config. Если у монеты свой порог выхода — в отчёте нужно сравнивать с ним.

---

## 4. ДИАГНОСТИКА

- **Скрипт** `scripts/analyze_exchange_trades_full.py`:
  - По сделкам с биржи считает RSI на вход (по последней закрытой свече) и на выход, сверяет с эталонным конфигом.
  - Для каждой сделки при необходимости подгружает свечи по диапазону (`get_chart_data_range`), чтобы RSI был посчитан для **всех** сделок, а не только за последние 30 дней.
  - **--db** — сопоставление с `bot_trades_history`: в отчёт добавляются `close_reason`, `entry_rsi`, `exit_rsi` из БД.
  - **--log-file PATH** — для каждой сделки выводятся строки лога, содержащие символ и дату входа/выхода (для разбора по логам).
- **Лог:** при решении «вход» пишется строка с RSI и порогом; при решении «выход по RSI» — строка «РЕШЕНИЕ ВЫХОД LONG/SHORT — RSI=… порог … (ТФ=…)»; при фактическом закрытии — «Закрываем LONG/SHORT по RSI (RSI=…, reason=…)».
- **БД** (bot_trades_history): при наличии — сверка entry_rsi/exit_rsi и close_reason с отчётом и логами.
