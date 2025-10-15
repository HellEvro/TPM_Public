# 🚀 Быстрый старт за 5 минут

Самый быстрый способ запустить InfoBot Bots Service.

---

## ✅ Предварительные требования

- Python 3.8+
- API ключи Bybit
- 2GB+ свободной RAM

---

## 📝 Пошаговая инструкция

### 1️⃣ Клонируйте проект (если еще не сделали)
```bash
git clone <repository_url>
cd InfoBot
```

### 2️⃣ Установите зависимости
```bash
pip install flask flask-cors psutil requests ccxt
```

### 3️⃣ Настройте API ключи

Создайте/отредактируйте `app/config.py`:
```python
EXCHANGES = {
    'BYBIT': {
        'api_key': 'YOUR_API_KEY',
        'api_secret': 'YOUR_SECRET_KEY'
    }
}
```

### 4️⃣ Запустите сервис
```bash
python bots.py
```

### 5️⃣ Проверьте работу
В браузере откройте:
```
http://localhost:5001/api/status
```

Должны увидеть:
```json
{"service":"bots","status":"online"}
```

---

## ✅ Что дальше?

**Система запущена!** Теперь программа будет "висеть" на строке:
```
* Serving Flask app...
```

**Это нормально!** Веб-сервер работает и ждет запросов.

---

## 🧪 Проверка функциональности

### В другом терминале выполните:

```bash
# Получить список ботов
curl http://localhost:5001/api/bots

# Получить монеты с RSI
curl http://localhost:5001/api/coins-with-rsi

# Получить конфигурацию автобота
curl http://localhost:5001/api/auto-bot/config
```

---

## 🤖 Создание первого бота (вручную)

### Через API:
```bash
curl -X POST http://localhost:5001/api/bots/create \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC",
    "volume_mode": "usdt",
    "volume_value": 10
  }'
```

### Через UI:
1. Откройте http://localhost:5000 (основной app.py)
2. Перейдите в раздел "Боты"
3. Нажмите "Создать бота"
4. Выберите монету и параметры

---

## 🛑 Остановка сервиса

Нажмите **Ctrl+C** в терминале где запущен bots.py:
```
^C
🛑 Остановка сервиса...
✅ Сервис остановлен
```

Система остановится **мгновенно** и сохранит все данные.

---

## ❓ Частые вопросы

### Q: Программа зависла после "Serving Flask app"?
**A:** Это нормально! Веб-сервер запустился и работает. Откройте http://localhost:5001

### Q: Как узнать что система работает?
**A:** Откройте http://localhost:5001/api/status в браузере

### Q: Автобот запускается сам?
**A:** НЕТ! Автобот **выключен** по умолчанию. Включайте **только вручную** через UI.

### Q: Можно запустить в фоне?
**A:** Да! 
```bash
# Linux/Mac
nohup python bots.py > logs/bots.log 2>&1 &

# Windows
start /B python bots.py
```

### Q: Где смотреть логи?
**A:** `logs/bots.log` - все логи системы

---

## 🎯 Следующие шаги

Теперь когда система запущена:

1. **Изучите настройки** → [CONFIGURATION.md](CONFIGURATION.md)
2. **Настройте автобот** → [AUTO_BOT.md](AUTO_BOT.md)
3. **Изучите API** → [API_REFERENCE.md](API_REFERENCE.md)
4. **Создайте первого бота** → [MANUAL_BOTS.md](MANUAL_BOTS.md)

---

**Готово! Система работает!** 🎉
