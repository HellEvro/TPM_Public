# План реализации FullAI (Полный Режим ИИ)

**Версия:** 0.3  
**Ссылка на ТЗ:** [AI_FULL_CONTROL_PROPOSAL_AND_SOW.md](AI_FULL_CONTROL_PROPOSAL_AND_SOW.md)

Статус: `[ ]` — не сделано, `[x]` — сделано. **Все блоки реализованы.**

---

## Блок 1: Настройка и хранение (флаг, UI)

- [x] **1.1** Добавить ключ `full_ai_control: bool` (по умолчанию `False`) в `auto_bot_config`; сохранение через POST `/api/bots/auto-bot`, запись в `bot_config.py`/БД.
- [x] **1.2** UI: переключатель «Полный Режим ИИ» на вкладке **Управление** рядом с тумблером Auto Bot (уже добавлен в HTML/JS — проверить сохранение в basicSettings и отображение).
- [x] **1.3** Условие доступности переключателя: активен только при `ai_enabled === true` и валидной лицензии ИИ; иначе disabled + подсказка «Включите ИИ и проверьте лицензию».

---

## Блок 2: Отдельный конфиг FullAI и таблица в БД

- [x] **2.1** Реализовать хранение **отдельного конфига FullAI** (БД или файл, например `configs/full_ai_config.json` / ключ в bots_data). При первом включении FullAI инициализировать копией пользовательского конфига (без таймфрейма как изменяемого); пользовательский `auto_bot_config` не перезаписывать.
- [x] **2.2** Добавить **отдельную таблицу в БД** для FullAI: `full_ai_coin_params` (или `prii_coin_params`) — symbol, params (json или нормализованные поля), updated_at, created_at. Используется только при `full_ai_control === true`.
- [x] **2.3** Миграция в `bots_database.py`: создание таблицы при отсутствии. Методы: `save_full_ai_coin_params(symbol, params)`, `load_full_ai_coin_params(symbol)`, `load_all_full_ai_coin_params()`.
- [x] **2.4** API (по необходимости): GET/POST для одной монеты и списка параметров FullAI (по аналогии с individual-settings).
- [x] **2.5** При выключении FullAI: в рантайме перестать использовать конфиг FullAI и таблицу FullAI; использовать только пользовательский конфиг и `individual_coin_settings`. Конфиг FullAI и таблица FullAI остаются для следующего включения.

---

## Блок 3: Переключение источника настроек

- [x] **3.1** Ввести хелпер/геттер: при `full_ai_control === true` возвращать конфиг FullAI и данные из таблицы FullAI; при `false` — пользовательский конфиг и `individual_coin_settings`. Таймфрейм **всегда** из пользовательского конфига.
- [x] **3.2** Во всех местах, где при FullAI нужны параметры по монете или глобальные параметры, использовать этот геттер (конфиг FullAI + таблица FullAI при FullAI).

---

## Блок 4: Логика входа (FullAI)

- [x] **4.1** В `bot_class.py` в `should_open_long` / `should_open_short`: при `full_ai_control` и доступном ИИ **не** вызывать RSI/тренд/время; вызывать `get_ai_entry_decision(symbol, direction, candles, fullai_config, full_ai_coin_params, ...)` → `{ allowed, confidence?, reason? }`.
- [x] **4.2** Реализовать `get_ai_entry_decision` (в `ai_integration` или отдельном модуле): вход — symbol, direction, candles, current_price, fullai_config, per-coin params из таблицы FullAI; выход — allowed, confidence, reason. Пока можно заглушку (например консервативно `allowed: False` или вызов существующих LSTM/pattern/anomaly).
- [x] **4.3** При FullAI использовать только конфиг FullAI и таблицу FullAI для параметров входа; делистинг и нефункциональные проверки сохранить.

---

## Блок 5: Логика выхода (FullAI, вариант B)

- [x] **5.1** Найти в коде место цикла проверки выхода по позициям (RSI exit, trailing, break_even и т.д.).
- [x] **5.2** При `full_ai_control`: вызывать `get_ai_exit_decision(symbol, position, candles, pnl_percent, fullai_config, full_ai_coin_params)` → `{ close_now, reason?, confidence? }`. При `close_now === true` — закрывать позицию. Классическую логику RSI/trailing не использовать как обязательную.
- [x] **5.3** Реализовать `get_ai_exit_decision` (заглушка или интеграция с ИИ). Вход: symbol, position, candles, pnl_percent, fullai_config, params по монете из таблицы FullAI. Таймфрейм из пользовательского конфига.

---

## Блок 6: Таймфрейм только из конфига пользователя

- [x] **6.1** Убедиться, что ни конфиг FullAI, ни таблица FullAI не содержат и не переопределяют таймфрейм; везде читать таймфрейм из пользовательского конфига.
- [x] **6.2** В UI и в коде не давать ИИ/автоматике менять таймфрейм.

---

## Блок 7: Изучение совершённых сделок и доработка стратегии

- [x] **7.1** Модуль загрузки сделок: из `bot_trades_history` (bots_data.db), при необходимости из ai_data.db и биржи.
- [x] **7.2** Оценка правильности решений по каждой закрытой сделке (результат, PnL %, причина выхода; критерии успех/неудача).
- [x] **7.3** Выводы и обновление параметров: сохранять **только в конфиг FullAI** и **таблицу FullAI** (full_ai_coin_params). Пользовательский конфиг и `individual_coin_settings` не трогать.
- [x] **7.4** Запуск анализа: по расписанию (батчем) и/или после каждой закрытой сделки при включённом FullAI.

---

## Блок 8: Обучение и запись в конфиг FullAI и таблицу FullAI

- [x] **8.1** Оптимизатор (AIStrategyOptimizer или аналог) и модуль изучения сделок пишут параметры **только** в конфиг FullAI и таблицу FullAI.
- [x] **8.2** Использование истории сделок из `bot_trades_history` и при необходимости ai_data.db/биржи; результаты обучения не писать в пользовательский конфиг и не в `individual_coin_settings`.

---

## Блок 9: Безопасность и UI

- [x] **9.1** Явная индикация режима в UI: «Стандартный» / «Полный ИИ» (рядом с переключателем или в шапке раздела ботов).
- [x] **9.2** Логи: при каждом решении входа/выхода в FullAI логировать, что решение принято ИИ (reason/confidence при необходимости).
- [x] **9.3** При старте bots.py и при проверке: если `full_ai_control === true` и ИИ недоступен (лицензия не валидна) — сбросить `full_ai_control` в `False`, записать в конфиг, уведомить в логах и по возможности в UI.

---

## Блок 10: FullAI Data Context и мониторинг (ежесекундно)

- [x] **10.0** FullAI Data Context (`bot_engine/fullai_data_context.py`): доступ к БД (свечи candles_history), системным индикаторам (RSI, тренд, сигнал из coins_rsi_data), кастомным индикаторам.
- [x] **10.1** FullAI Monitor (`bot_engine/fullai_monitor.py`): мониторинг позиций каждую секунду при full_ai_control. Использует get_fullai_data_context и get_ai_exit_decision.
- [x] **10.2** Кастомные индикаторы FullAI: `register_fullai_indicator(name, compute_fn)`. Дефолтные: momentum_pct_5, volatility_atr_pct.
- [x] **10.3** get_ai_exit_decision принимает data_context (свечи из БД, system, custom). bot_class и fullai_monitor передают полный контекст.

---

## Блок 11: Тесты и проверки

- [x] **11.1** Проверка: включение FullAI не меняет пользовательский конфиг и individual_coin_settings.
- [x] **11.2** Проверка: выключение FullAI возвращает использование пользовательского конфига и individual_coin_settings без шага «восстановить».
- [x] **11.3** Проверка: таймфрейм нигде не берётся из конфига FullAI или таблицы FullAI.

---

## Порядок выполнения (рекомендуемый)

1. Блок 1 (флаг, UI) — уже частично есть; довести сохранение и доступность.  
2. Блок 2 (конфиг FullAI + таблица БД + миграция + API при необходимости).  
3. Блок 3 (переключение источника настроек).  
4. Блок 4 (логика входа + get_ai_entry_decision).  
5. Блок 5 (логика выхода + get_ai_exit_decision).  
6. Блок 6 (таймфрейм только из конфига).  
7. Блок 9 (безопасность и индикация).  
8. Блок 7–8 (изучение сделок и обучение в FullAI-хранилище).  
9. Блок 10 (Data Context, Monitor, кастомные индикаторы).  
10. Блок 11 (тесты и проверки).

---

## Итог реализации

- **API FullAI:** `GET/POST /api/bots/fullai-coin-params`, `GET/POST/DELETE /api/bots/fullai-coin-params/<symbol>`, `POST /api/bots/fullai-trades-analysis` (запуск анализа по расписанию).
- **Модуль изучения сделок:** `bots_modules/fullai_trades_learner.py` — загрузка из `bot_trades_history`, оценка, запись только в `full_ai_coin_params`; запуск после закрытия позиции в FullAI и по API.
- **Оптимизатор:** при `full_ai_control` сохраняет результаты только в `full_ai_coin_params` (ai_strategy_optimizer.py).
- **FullAI Data Context:** `bot_engine/fullai_data_context.py` — доступ к БД, свечам, системным и кастомным индикаторам.
- **FullAI Monitor:** `bot_engine/fullai_monitor.py` — ежесекундный мониторинг позиций при full_ai_control.
- **Проверки:** `tests/verify_fullai_plan.py` — проверки 11.1–11.3.

*По мере выполнения помечать пункты: `[ ]` → `[x]`.*
