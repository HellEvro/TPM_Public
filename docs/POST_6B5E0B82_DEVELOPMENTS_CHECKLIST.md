# Чеклист доработок после коммита 6b5e0b82 (KuCoin)

Документ фиксирует все разработки после коммита `6b5e0b82` для корректной переделки логики.

---

## 1. СДЕЛКИ НЕ ТЕРЯТЬСЯ

| # | Коммит | Что сделано | Логическая суть |
|---|--------|-------------|-----------------|
| 1.1 | c507c375 | FullAI Monitor: закрытие через close_position_for_bot при dict-боте | FullAI мониторит позиции, при close_now вызывает close_position_for_bot |
| 1.2 | 09605caa | Grace period 90 сек — не удалять бота, только что открывшего позицию | **ЛОГИКА:** race: API биржи не успевает показать позицию → sync удалял бота → сделка «ручная». Решение: 90 сек grace period. **ВАЖНО:** использовать только `position_start_time` или `entry_timestamp`, НЕ `last_update`! |
| 1.3 | 4b55cb09 | Fallback при определении размера позиции (bots_data + Bybit API) | close_position_for_bot: если биржа не вернула размер — брать из bots_data (position_size_coins, volume_value/entry_price), затем прямой Bybit API |
| 1.4 | — | positions-for-app + fallback в app.py | App и Bots в разных процессах. Когда app не видит позиции — брать с Bots `/api/bots/positions-for-app` |

---

## 2. AI PREMIUM + FullAI: ИЗОЛЯЦИЯ, НЕ БЛОКИРОВАТЬ

| # | Коммит | Что сделано | Логическая суть |
|---|--------|-------------|-----------------|
| 2.1 | e77a43b9 | Изоляция FullAI, отключаемое обучение, fallback leverage, безопасный get_ai_manager | AI Premium падает/недоступен → FullAI продолжает работать. Fallback: allowed=True, leverage из конфига. ParameterQuality/AIContinuousLearning — отключаемо |
| 2.2 | ce8c6467 | AI блокировка только при включённых AI модулях | При выкл AI Config — сброс full_ai_control, не блокировать сделки |
| 2.3 | 5b4f1cde | FullAI: protections и workers не закрывают — выход только через get_ai_exit_decision | Выход из FullAI-позиции только через AI решение, не через RSI/protections |
| 2.4 | b5744a37 | FullAI: sync не трогает protections при full_ai; блокировка при entry_ts=None | При full_ai — sync не меняет protections. При entry_ts=None не удалять |

---

## 3. СИГНАЛЫ И ОБУЧЕНИЕ

| # | Коммит | Что сделано | Логическая суть |
|---|--------|-------------|-----------------|
| 3.1 | 7edc10c3 | Sync/FullAI: причина из bots_db, самообучение для FullAI, transformer опционален | Sync берёт effective_reason из bots_db. Самообучение при закрытии FullAI |
| 3.2 | 6673d344 | _check_if_trade_already_closed: фильтр 10мин, decision_source в to_dict | Избежать дубликатов записей. Fallback: entry_price в пределах 1% |
| 3.3 | 10c73473 | Fallback в _check_if_trade_already_closed: дубликат при entry_price ±1% | Учёт проскальзывания |
| 3.4 | 920e5b61 | Убран спам логов AIContinuousLearning/ParameterQualityPredictor | При 0 результатах — ранний выход без INFO |
| 3.5 | 1216a8d3 | FullAI/Adaptive: аналитика не снижает confidence | ИИ учится, пробует каждый вход |

---

## 4. FullAI: ЗАЩИТЫ И ОТЧЁТНОСТЬ

| # | Коммит | Что сделано | Логическая суть |
|---|--------|-------------|-----------------|
| 4.1 | 91ed0679 | Защита: мин. 90 сек удержания, мин. SL 5% | От мгновенного выхода |
| 4.2 | fc1c3592 | Различать закрытие по лимитке бота (TP/SL) и вручную | Аналитика: BOT_LIMIT_TP vs MANUAL_OR_EXTERNAL |
| 4.3 | b23fdb74 | Не показывать 0.000000; подстановка entry/exit из get_closed_pnl | Отчётность |
| 4.4 | 43f2a5ae | Обогащение real_close PnL из closed_trades | Аналитика совпадает с закрытыми |

---

## 5. FullAI АНАЛИТИКА (UI, ЗАПИСЬ)

| # | Коммит | Что сделано |
|---|--------|-------------|
| 5.1 | ad938337 | Пояснение реал./вирт. входов, фильтр «Только входы» |
| 5.2 | 4826510f | record_real_open при входе из process_auto_bot_signals |
| 5.3 | 8c332502 | record_real_open при входе через API |
| 5.4 | 9f131e28 | record_real_close при ручном закрытии (sync + UI) |
| 5.5 | 6c1d214d | Всегда записывать real_close/real_open, полный extra |
| 5.6 | 8023b4ce | Тип ордера, проскальзывание, задержка, TP/SL |
| 5.7 | 7a74e9c7 | Столбец PnL USDT |
| 5.8 | a8eb5c77 | Показ убытков из fullai_analytics |
| 5.9 | 3f7918a7 | Отключение кэша браузера для кнопки Обновить |
| 5.10 | 991297ec | Аналитика в фоновых воркерах |
| 5.11 | 7d796281 | FullAI конфиги: все торговавшие монеты в селекторе |

---

## 6. Bybit, ФИЛЬТРЫ, ЗРЕЛОСТЬ

| # | Коммит | Что сделано |
|---|--------|-------------|
| 6.1 | 2bca3d2c | Bybit: minNotionalValue 2%→10% (ErrCode 110094) |
| 6.2 | 096cf667 | Bybit get_closed_pnl: ErrCode 10002, синхр. времени, retry |
| 6.3 | a5bc4f20 | loss_reentry: fallback time.time() при устаревших свечах |
| 6.4 | 76bd32ca | UI: замена {reason} в loss_reentry |
| 6.5 | 2ee2d49f | Зрелость: мягкий RSI, maturity_reason в API |
| 6.6 | 535d6a9e | Зрелость: min_candles + RSI ≤35/≥65 |
| 6.7 | 060c54c0 | Зрелость: только системный ТФ |
| 6.8 | 9c24afad | Зрелость: ТФ 4h, 1000 свечей |
| 6.9 | 81ba1eb9 | Зрелость: причина, RSI по истории |
| 6.10 | a2b2459c | Не перезаписывать is_mature из storage |
| 6.11 | 76e2c49e | Верификация перед входом: свежие свечи, RSI, фильтры |
| 6.12 | abd06940 | RSI Time: разрешить 1 свечу, NEUTRAL не penalize |
| 6.13 | 96f28c0d | RSI: таймаут 5 мин update_in_progress |
| 6.14 | 229bd253 | Загрузка только системного ТФ |
| 6.15 | 7360097b | Убрана оптимизация data_version — UI обновляется |
| 6.16 | 808c2f7c | get_rsi_from_coin_data из config_loader |
| 6.17 | 93269b7a | verify_coin_realtime: импорт get_rsi_from_coin_data |

---

## 7. ПРОКСИ API, ПРИИ, UI

| # | Коммит | Что сделано |
|---|--------|-------------|
| 7.1 | 05afae89 | Прокси: mature-coins-list, delisted-coins, history, statistics, refresh-rsi-all |
| 7.2 | 7043cce9 | Прокси: аналитика FullAI, смена таймфрейма |
| 7.3 | 5eccb232 | Таймаут RSI пакета 200→360 сек |
| 7.4 | 27a0f0fe | ПРИИ: rsi_limit_entry_enabled в TradingBot |
| 7.5 | bb7774c1 | Цвет PnL при минусе (красный) |
| 7.6 | d9d5f33a | ИИ проанализировать: обучение, insights |
| 7.7 | 6bee5bb7 | Бейдж лицензии: фактическая валидность |
| 7.8 | 11353e33 | Удален лог «Конфигурация перезагружена» |
| 7.9 | c1a13766 | ANALYTICS_REPORT актуализация |

---

## 8. КРИТИЧЕСКАЯ ЛОГИКА (отдельно)

**check_api_keys в app.py:** Сейчас проверяет `app/keys.py`. Ключи — в `configs/keys.py`. Если app/keys.py нет → DEMO режим → пустые позиции. **Нужно:** проверять EXCHANGES из конфига (configs), не требовать app/keys.py.

---

## ПРАВИЛЬНЫЙ ПОДХОД К ПЕРЕДЕЛКЕ

1. **Не конфликтовать:** AI Premium и FullAI — изолированы. Падение одного не ломает другого.
2. **Не терять сделки:** Grace period (только position_start_time/entry_timestamp), fallback close, positions-for-app.
3. **Единый источник правды:** Позиции — с биржи. App и Bots — разные процессы, fallback между ними.
4. **Сигналы:** AI проверка только когда AI включён. Fallback при ошибках.
5. **Обучение:** Опционально, не блокирует торговлю.

---

## СТАТУС ВНЕДРЕНИЯ (после reset 6b5e0b82)

| Задача | Статус | Файлы |
|--------|--------|-------|
| check_api_keys: configs/keys | ✅ | app.py |
| Grace period 90 сек | ✅ | sync_and_cache.py |
| close_position_for_bot fallback | ✅ | imports_and_globals.py |
| positions-for-app + fallback app | ✅ | api_endpoints.py, app.py |
| AI Premium/FullAI изоляция | ⏳ | — |
| FullAI Monitor close_position | ⏳ | — |
