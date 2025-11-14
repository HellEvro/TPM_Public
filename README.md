# 🤖 InfoBot — AI-Powered Trading System

**Версия:** 1.7 AI Edition  
**Дата:** 14 ноября 2025  
**Статус:** ✅ Production Ready + AI Launcher (LSTM + License)  
**AI прогресс:** 47% (3/4 модулей активны, 36/76 задач)  
**Автопроверка:** `python scripts/verify_ai_ready.py` → 10/10 ✅  
**Лицензия:** HWID `.lic` (генерация через `scripts/activate_premium.py`)  
**Репозиторий:** [github.com/HellEvro/TPM_Public](https://github.com/HellEvro/TPM_Public)

> Интеллектуальная торговая система с AI лаунчером, защищёнными премиум модулями, 6h LSTM-предсказаниями, адаптивным риск-менеджером и полной историей действий.

---

## 🚀 Что нового (ноябрь 2025)
- 🧠 **LSTM Predictor 1.0**: 6‑часовые прогнозы направления с автообучением (`docs/ai_development/PHASE_3_LSTM_COMPLETE.md`, `bot_engine/ai/lstm_predictor.py`) и валидацией сигналов прямо из `bots_modules/filters.py`.
- 🛰️ **AI Launcher & Scheduler**: `ai.py` проксирует защищённый `_ai_launcher.pyc`, запускает `data-service`, `train`, `scheduler` режимы (`python ai.py --mode all`) и следит за состоянием AI подсистем.
- 🔐 **Premium License & Protection**: единый HWID (`scripts/activate_premium.py`), проверка `.lic` через `bot_engine/ai/license_checker.pyc`, обновляемый билд `license_generator/build_ai_launcher.py`, загрузчик `_infobot_ai_protected.py`.
- ⚖️ **Smart Risk Manager**: премиум-модуль (`bot_engine/ai/smart_risk_manager.py`) анализирует стопы, делает быстрый бэктест перед входом и оптимизирует SL/TP на основе `ml_risk_predictor`.
- 📡 **AI Data Service & Automation**: постоянный сбор/хранение (`AIDataCollector`, `ai_data_storage.py`), backtest (`ai_backtester_new.py`), оптимизация (`ai_strategy_optimizer.py`), управление ботами (`ai_bot_manager.py`) и непрерывное обучение (`ai_continuous_learning.py`).
- 📦 **Release tooling**: `sync_to_public.py` формирует `InfoBot_Public` с обязательными стартовыми скриптами и скомпилированными AI файлами, `start_infobot_manager` получил пошаговый лицензионный этап.

---

## 🎯 Основные возможности

### Базовая система
- 📊 Реальный мониторинг позиций и сигналов + экспорт истории.
- 🤖 Auto Bot: RSI/EMA, фильтры зрелости, анти-скам, break-even, трейлинг.
- 👤 Manual Bots: ручные сценарии, индивидуальные конфиги, hot reload.
- 🔁 Optimal EMA и синхронизация (`bots_modules/sync_and_cache.py`) + фоновые воркеры.
- 🔔 Telegram-уведомления, логирование, JSON-хранилища + GUI `start_infobot_manager`.

### AI Модули (активно)
- 🛡️ **Anomaly Detection** (`bot_engine/ai/anomaly_detector.py`): IsolationForest 100 деревьев, >15 признаков для блокировки pump/dump.
- 🧠 **LSTM Predictor** (`bot_engine/ai/lstm_predictor.py`): подтверждение сигналов, вероятность движения, auto retrain.
- ⚖️ **Dynamic & Smart Risk** (`bot_engine/ai/risk_manager.py`, `smart_risk_manager.py`): адаптивные SL/TP, размер позиции, премиум backtest перед входом.
- 🔄 **Auto Training & Continuous Learning** (`auto_trainer.py`, `ai_continuous_learning.py`): автообновление данных, weekly retrain, база знаний по сделкам.
- 🛰️ **AI Launcher сервисы**: `AIDataCollector`, `AIBacktester`, `AIStrategyOptimizer`, `AIBotManager`, REST/UI управление (`static/js/managers/ai_config_manager.js`).

### Premium / Лицензирование
- 🎫 HWID → `.lic` проверка (`scripts/activate_premium.py`, `bot_engine/ai/license_checker.pyc`) с кэшированием статуса и мягким отключением.
- 🧩 SmartRisk, ML Risk Predictor и премиум SL/TP доступны только с валидной лицензией.
- 🛡️ `license_generator/source/@source/ai_launcher_source.py` + `build_ai_launcher.py` готовят защищённый `_ai_launcher.pyc` и обновляют обёртки `ai.py`, `_infobot_ai_protected.py`.

### В разработке (roadmap)
- 🧭 Pattern Recognition (`bot_engine/ai/pattern_detector.py`, CNN 0/7 задач).
- 🤖 Telegram/бот для продажи лицензий и мультипользовательские планы (см. `TODO.txt`).
- 🌍 Мульти-биржи и расширенный маркет-мейкинг (Binance, OKX интеграции в `exchanges/`).

---

## 🏗️ Архитектура
- **Сервисы:** `app.py` (порт 5000, Web UI) + `bots.py` (порт 5001, торговые/AI API + `/api/bots/*`).
- **AI Launcher:** `ai.py` → `_ai_launcher.pyc` orchestrator (data-service/train/scheduler) + процессы `AIDataCollector`, `AITrainer`, `AIBacktester`, `AIStrategyOptimizer`, `AIBotManager`, `AIContinuousLearning`.
- **AI ядро:** `bot_engine/ai/` (анализаторы, автообучение, premium, лицензирование, ml модели).
- **История и данные:** `bot_engine/bot_history.py`, REST `/api/bots/history|trades|statistics`, файлы `data/ai/*`, `data/bot_history.json`.
- **Фронтенд:** `templates/pages/bots.html`, `static/js/managers/*.js`, `static/css/`.
- **Лицензирование:** `.lic` в корне, `bot_engine/ai/license_checker.pyc`, `scripts/activate_premium.py`, генератор HWID в `license_generator/`.
- **Релизы:** `sync_to_public.py`, каталог `InfoBot_Public/`, авто-добавление `start_infobot_manager.{cmd|sh|vbs}`, `launcher/`.

---

## ⚡ Быстрый старт

### 1. Основной сценарий (рекомендуется всем)
1. Установите **Python 3.11+** (Windows/Mac/Linux).
2. Запустите соответствующий лаунчер:
   - Windows: двойной клик по `start_infobot_manager.vbs` (или `start_infobot_manager.cmd`).
   - Linux/macOS: `chmod +x start_infobot_manager.sh && ./start_infobot_manager.sh`.
3. Далее GUI сделает всё автоматически:
   - создаст/обновит виртуальное окружение и поставит зависимости (включая TensorFlow);
   - скопирует `config`/`keys` из примеров и проверит ключи;
   - выполнит Git sync или `git init` + добавление `origin`;
   - подскажет, как получить HWID и применить `.lic`, запуск `scripts/activate_premium.py` доступен из интерфейса;
   - запустит `app.py`, `bots.py`, `ai.py --mode all`, покажет статусы сервисов и логи.

Лаунчер отслеживает прогресс, блокирует кнопки на длительных шагах и хранит состояние, поэтому достаточно одного клика после установки Python.

### 2. Ручной запуск (опционально, если нужен контроль)
```bash
# Создайте окружение сами (пример для Windows PowerShell)
python -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt
pip install "tensorflow>=2.13.0" "scikit-learn>=1.3.0"

copy app\config.example.py app\config.py
copy app\keys.example.py app\keys.py
python scripts/activate_premium.py  # HWID -> положите .lic в корень

python app.py        # UI (порт 5000)
python bots.py       # Bot & AI API (порт 5001)
python ai.py --mode all  # либо data-service/train по отдельности

python scripts/verify_ai_ready.py   # итоговая проверка AI
```

Подробности и скриншоты: [docs/QUICKSTART.md](docs/QUICKSTART.md), [docs/AI_QUICK_START.md](docs/AI_QUICK_START.md), [docs/START_HERE.md](docs/START_HERE.md).

---

## 🔌 REST API
- `GET /api/status` — статус бота и подключений.
- `GET /api/bots/history|trades|statistics` — лог действий, сделки, аналитика.
- `POST /api/bots/history/demo` / `history/clear` — демо-данные и очистка UI.
- `GET /api/ai/status` — активность модулей, автообучение, лицензия.
- `GET|POST /api/ai/config` — чтение/сохранение AI параметров с логом изменений.
- `POST /api/ai/force-update` — немедленное обновление данных и переобучение.

См. `bot_engine/api/endpoints_ai.py`, `bot_engine/api/endpoints_history.py`, `bot_engine/api/endpoints_bots.py`.

---

## 🧪 Тестирование

```bash
python scripts/test_ai_initialization.py    # Проверка инициализации AI
python scripts/test_ai_detector_status.py  # Статус Anomaly Detector
python scripts/test_risk_manager.py        # Расчёты Risk Manager
python scripts/test_full_ai_system.py      # Комплексная проверка AI
python tests/test_hwid_check.py            # Сверка HWID и лицензий
python scripts/verify_ai_ready.py          # Финальная проверка (10/10)
```

---

## 📚 Документация
- 🚀 Быстрый старт: `docs/INSTALL.md`, `docs/QUICKSTART.md`, `docs/AI_QUICK_START.md`, `docs/START_HERE.md`.
- 📖 Обзор и архитектура: `docs/SYSTEM_OVERVIEW.md`, `docs/ARCHITECTURE.md`, `docs/MODULES.md`.
- 📊 История и сигналы: `docs/BOT_HISTORY.md`, `docs/BOT_SIGNAL_PROCESSING_FLOW.md`.
- 🤖 AI: `docs/AI_README.md`, `docs/ai_technical/*.md`, `docs/AI_RISK_MANAGER.md`, `docs/ai_development/PHASE_3_LSTM_COMPLETE.md`, `docs/AI_DOCS_STRUCTURE.md`.
- 🔐 Лицензии и защита: `docs/HWID_FIX_REPORT.md`, `docs/PREMIUM_STOP_ANALYSIS_ARCHITECTURE.md`, `docs/ML_MODELS_DISTRIBUTION.md`.
- 🛠️ Разработка и чеклисты: `docs/AI_IMPLEMENTATION_CHECKLIST.md`, `docs/READY_FOR_YOU.md`, `docs/DOCUMENTATION_COMPLETE.md`.

Полный список — каталог `docs/` (4000+ строк, 12+ актуальных AI-гидов и отчётов).

---

## 📊 Статус проекта
- ✅ Соответствие ТЗ: 100%.
- 🤖 AI: 3/4 модулей, 36/76 задач (47%) — осталось Pattern Recognition.
- ⚙️ Производительность: запуск <5 c, обработка сигналов <100 мс, синхронизация <500 мс.
- 📦 Компоненты: 190+ файлов (UI, API, AI, тесты, лицензирование).
- 🔐 Лицензирование: HWID `.lic`, SmartRisk и `_ai_launcher.pyc` активны при валидной лицензии.

---

## 🛣️ Следующие шаги
- Завершить Pattern Recognition (`bot_engine/ai/pattern_detector.py`, 0/7 задач).
- Вынести премиум лицензии в отдельный Telegram-бот/сервер (см. `TODO.txt`).
- Расширить торговлю на Binance/OKX с единым AI лаунчером.
- Добавить визуальные отчёты по AI backtest/optimizer в Web UI.

---

## 📞 Поддержка и мониторинг
- Документация: `docs/AI_README.md`, `docs/SYSTEM_OVERVIEW.md`.
- Логи: `logs/bots.log`, `logs/ai.log`.
- Web UI: http://localhost:5000 (вкладка «Боты» → «Конфигурация» → «AI»).
- API статус: http://localhost:5001/api/status, http://localhost:5001/api/ai/status.
- Telegram: [H3113vr0](https://t.me/H3113vr0)
- Email: gci.company.ou@gmail.com

---

**Успешных сделок с InfoBot!** 🚀🤖💰

