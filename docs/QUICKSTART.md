# 🚀 QUICKSTART (15 минут)

Актуальная инструкция для InfoBot 1.7 (ноябрь 2025). Два сценария: **лаунчер** (рекомендуется всем) и **ручной** (для dev/CI).

---

## 0. Требования

- Windows 10+, macOS 13+, Ubuntu 22.04+.
- Python 3.12+ установлен и добавлен в PATH (рекомендуется 3.14+).
- Git ≥ 2.40.
- Bybit API ключи (read/write, Unified Trading).
- 6 GB RAM / 15 GB свободно на диске (PyTorch + модели).
- **Для GPU (опционально):** NVIDIA GPU с драйверами, CUDA Toolkit 11.8 или 12.1.

---

## 1. Лаунчер (рекомендуется)

1. **Клонируйте репозиторий** (или обновитесь):
   ```powershell
   git clone https://github.com/HellEvro/InfoBot.git
   cd InfoBot
   git pull
   ```
2. **Запустите менеджер**:
   - Windows: двойной клик по `start_infobot_manager.vbs` (или `start_infobot_manager.bat`).
   - macOS/Linux: `chmod +x start_infobot_manager.sh && ./start_infobot_manager.sh`.
3. **Следуйте шагам GUI**:
   - установка/обновление `.venv` и зависимостей (PyTorch, scikit-learn, Flask, ccxt, ...);
   - копирование `app/config.example.py` → `app/config.py`, `app/keys.example.py` → `app/keys.py`;
   - проверка Git/branch, подтягивание `InfoBot_Public`;
   - получение HWID, применение `.lic` (кнопка запускает `scripts/activate_premium.py`);
   - автозапуск `app.py`, `bots.py`, `ai.py --mode all` с мониторингом логов.
4. **Проверка** — во вкладке `AI` жмите `Run verify_ai_ready`. Должно быть 10/10.

> Лаунчер запоминает прогресс, поэтому после первой настройки достаточно одного клика.

---

## 2. Ручной сценарий (dev / CI)

```powershell
# 1. Клонирование
git clone https://github.com/HellEvro/InfoBot.git
cd InfoBot

# 2. Виртуальное окружение
python -m venv .venv
.\.venv\Scripts\activate           # Linux/macOS: source .venv/bin/activate

# 3. Зависимости
pip install --upgrade pip
pip install -r requirements.txt

# 4. PyTorch с GPU (если GPU доступен)
python scripts/setup_python_gpu.py
# Или вручную:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Конфигурации
copy app\config.example.py app\config.py
copy app\keys.example.py app\keys.py
# заполните API ключи, ключи телеграма и т.д.

# 6. Лицензия (опционально, для премиум AI)
python scripts/activate_premium.py    # получите HWID, положите .lic в корень

# 7. Запуск сервисов
python app.py        # Web UI (порт 5000)
python bots.py       # Bot & AI API (порт 5001)
python ai.py         # AI Service (data-service + train + scheduler)

# 8. Проверка
python scripts/verify_ai_ready.py
curl http://localhost:5001/api/status
curl http://localhost:5001/api/ai/status
```

---

## 3. Что считать успешным запуском

- `http://localhost:5000` → интерфейс менеджера ботов.
- `http://localhost:5001/api/status` → `{"service":"bots","status":"online"}`.
- `python scripts/verify_ai_ready.py` → `10/10 checks passed`.
- `logs/bots.log` и `logs/ai.log` без критических ошибок.
- Во вкладке AI видно активные процессы (`AIDataCollector`, `AITrainer`, `AIBacktester`, `AIStrategyOptimizer`, `AIBotManager`).

---

## 4. Частые действия

| Задача | Команда |
| --- | --- |
| Обновить проект | `git pull && python scripts/sync_to_public.py` |
| Запустить AI сервис | `python ai.py` |
| Проверить статус AI | `python scripts/verify_ai_ready.py` |
| Проверить лицензию | `python scripts/test_hwid_check.py` |
| Сбросить состояние ботов | `python scripts/tools/reset_bot_state.py` |

---

## 5. Следующие шаги

1. Настройте `bot_engine/bot_config.py` (или используйте UI) для включения/отключения AI модулей.
2. Ознакомьтесь с `docs/AI_README.md` — там описаны активные модули, лицензирование и тесты.
3. Для детальной архитектуры прочитайте `docs/ARCHITECTURE.md` и `docs/BOT_SIGNAL_PROCESSING_FLOW.md`.
4. Перед релизом прогоняйте `scripts/verify_ai_ready.py` и smoke-тесты из раздела «🧪 Тесты и диагностика» в `docs/AI_README.md`.

---

**TL;DR:** Лаунчер + `.lic` + `verify_ai_ready` — всё остальное уже автоматизировано.

