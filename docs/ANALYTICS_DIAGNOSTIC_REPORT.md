# Отчёт диагностики аналитики и конфигов

## Дата проверки
Февраль 2026.

## Запущенные проверки

- **scripts/check_databases.py** — проверка bots_data.db, ai_data.db, app_data.db
- **scripts/analysis/trading_analytics_report.py** — полная аналитика сделок ботов
- **scripts/check_db_writing.py** — проверка записи в БД (после исправления кодировки)

---

## 1. Ошибки в логике аналитики (исправлено)

### Проблема: RSI_EXIT показывал 4713 «убытков» при PnL=0

В отчёте по причинам закрытия все сделки с **close_reason=RSI_EXIT** и **pnl=0** считались убытками (`losses`), так как в коде было:
- `if t.pnl > 0: wins += 1`
- `else: losses += 1`

У части записей в `bot_trades_history` при закрытии по RSI не сохранялся размер позиции (`position_size_usdt`/`position_size_coins`), поэтому пересчёт PnL в аналитике не срабатывал и pnl оставался 0. Такие сделки ошибочно учитывались как проигрыши.

### Исправление

В **bot_engine/trading_analytics.py** в `analyze_bot_trades()`:

- Сделки с **pnl=0** учитываются как **neutral** (ни win, ни loss).
- Во всех группировках (by_close_reason, by_symbol, by_decision_source, by_bot, by_symbol_rsi, by_symbol_trend) добавлено поле `"neutral"` и ветка `elif t.pnl < 0: losses += 1`; при `pnl == 0` увеличивается только `neutral`.

В результате по RSI_EXIT (и другим причинам) отображаются только реальные выигрыши и убытки; сделки с неизвестным PnL не завышают число убытков.

---

## 2. Конфиги: источник истины

- **Единый источник настроек бота:** `configs/bot_config.py`
  - Классы: `DefaultAutoBotConfig`, `AutoBotConfig`, `SystemConfig`, `RiskConfig`, `FilterConfig`, `ExchangeConfig`, `AIConfig`.
  - Загрузка: `bot_engine.config_loader` импортирует из `configs.bot_config`; `reload_config()` перезагружает этот модуль.
- **Загрузка в рантайме:** `bots_modules.imports_and_globals.load_auto_bot_config()` читает из файла через `config_writer.load_auto_bot_config_from_file()` (путь: `configs/bot_config.py`), мержит с дефолтами и кладёт результат в `bots_data['auto_bot_config']`.
- **bot_engine/bot_config.py** — устаревший/дублирующий; в коде используется только `configs/bot_config.py`. Не полагайтесь на `bot_engine/bot_config.py` для актуальных настроек.
- **Приложение:** `configs/app_config.py` (и при необходимости `configs/keys.py`). Заглушка `app/config.py` реэкспортирует из `configs.app_config`.

Итог: конфиги работают из `configs/`; двойного источника для бота в текущей логике нет, если везде импортировать через `config_loader` и `load_auto_bot_config`.

---

## 3. Диагностические скрипты (Windows)

На Windows при выводе в консоль с кодировкой cp1251 эмодзи (✅, ❌ и т.д.) вызывали `UnicodeEncodeError` в:

- **scripts/check_databases.py**
- **scripts/check_db_writing.py**

В начало обоих скриптов добавлена установка UTF-8 для stdout/stderr на Windows:

```python
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
```

После этого скрипты можно запускать без `PYTHONIOENCODING=utf-8`.

---

## 4. Рекомендации

1. **Аналитика в UI**  
   Страница «Аналитика» вызывает `GET /api/bots/analytics` и опционально `GET /api/bots/analytics/rsi-audit`. После правок в `trading_analytics.py` отчёт по причинам закрытия и по монетам/настройкам будет без ложных убытков по сделкам с PnL=0.

2. **Сохранение PnL при закрытии по RSI**  
   В `bots_modules/bot_class.py` при закрытии позиции в `trade_data` уже передаются `pnl`, `position_size_usdt`, `position_size_coins`. Старые записи с RSI_EXIT и пустым PnL могли появиться из миграций или старых версий. Для новых сделок данные должны сохраняться корректно.

3. **Проверка конфига**  
   При сомнениях, какой конфиг реально используется, смотрите:
   - импорты в `bot_engine/config_loader.py` (только `configs.bot_config`);
   - путь в `bots_modules/config_writer.load_auto_bot_config_from_file()` (по умолчанию `configs/bot_config.py`).

---

## 5. Отчёт о торговле за последние 3–4 дня

Чтобы получить **аналитику текущей ситуации** (как отторговали боты, какие ошибки и расхождения с конфигом за последние дни), а не анализ кода аналитики, используйте:

```bash
python scripts/report_trading_last_days.py --days 4 --output docs/TRADING_REPORT_LAST_DAYS.md
```

Отчёт включает: сводку по сделкам за период, разбивку по причинам закрытия и источникам, сравнение входов/выходов с **текущим** конфигом (пороги RSI), перечень входов/выходов, не соответствующих конфигу, и блок «Почему могли не сработать стратегии или настройки». Результат пишется в `docs/TRADING_REPORT_LAST_DAYS.md` (или в другой файл по `--output`).

---

## 6. Краткий итог

| Что проверено           | Результат |
|-------------------------|-----------|
| Логика wins/losses      | Исправлена: pnl=0 → neutral, не loss |
| Источник конфигов       | Один: `configs/bot_config.py` |
| Кодировка в scripts     | UTF-8 на Windows для check_databases, check_db_writing |
| Запись в bot_trades_history | При закрытии из bot_class передаются pnl и размер позиции |
| Отчёт за последние дни      | `scripts/report_trading_last_days.py --days 4 -o docs/TRADING_REPORT_LAST_DAYS.md` |
