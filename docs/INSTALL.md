# 📦 Установка InfoBot

## Шаг 1: Установка зависимостей

```bash
pip install -r requirements.txt
```

## Шаг 2: Настройка конфигурации

### 2.1. Скопируйте пример конфигурации:
```bash
cp app/config.example.py app/config.py
```

### 2.2. Отредактируйте app/config.py:
```python
EXCHANGES = {
    'BYBIT': {
        'api_key': 'ваш_api_key',
        'api_secret': 'ваш_secret'
    }
}

TELEGRAM_BOT_TOKEN = 'ваш_токен'
TELEGRAM_CHAT_ID = 'ваш_chat_id'
```

## Шаг 3: Создание директорий

```bash
# Создадутся автоматически при первом запуске:
# - data/
# - logs/
```

## Шаг 4: Запуск

⚠️ **ВАЖНО:** Нужно запустить ОБА сервиса в отдельных терминалах!

**Терминал 1 (Web UI):**
```bash
python app.py
```

**Терминал 2 (Bot Service):**
```bash
python bots.py
```

### Проверка что оба сервиса работают:
- **Web UI**: http://localhost:5000 → должен открыться интерфейс
- **Bot Service**: http://localhost:5001/api/status → должен вернуть `{"status": "online"}`

⚠️ **Ошибка "Сервис ботов недоступен"?**
- Убедитесь что `python bots.py` запущен на порту 5001
- Проверьте: http://localhost:5001/api/status

⚠️ **Зависает "Загрузка данных RSI..."?**
- Откройте порт 5001 в Windows Firewall (см. ниже)
- Убедитесь что оба сервиса (`app.py` и `bots.py`) запущены

## Шаг 5: Проверка

Откройте: http://localhost:5000

**Готово!** Система работает.

---

## 🔗 Доступ из локальной сети / интернета

### Настройка Windows Firewall

Для доступа к серверу из локальной сети или интернета откройте ОБА порта (5000 и 5001):

**PowerShell от имени администратора:**
```powershell
# Порт 5000 - Web UI (интерфейс)
New-NetFirewallRule -DisplayName "InfoBot Web UI" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow -Profile Any

# Порт 5001 - Bot Service (API ботов) - ОБЯЗАТЕЛЬНО для загрузки RSI!
New-NetFirewallRule -DisplayName "InfoBot Bot Service" -Direction Inbound -Protocol TCP -LocalPort 5001 -Action Allow -Profile Any
```

⚠️ **ВАЖНО**: Оба порта должны быть открыты! Иначе RSI данные не загрузятся и будет зависать "Загрузка данных RSI...".

### Настройка роутера (для доступа из интернета)

⚠️ **ВАЖНО**: Нужно пробросить ОБА порта!

1. Войдите в настройки роутера (обычно 192.168.1.1 или 192.168.0.1)
2. Найдите раздел "Port Forwarding" / "Виртуальные серверы"
3. Создайте ДВА правила:

   **Правило 1 - Web UI:**
   - **Внешний порт**: 5000
   - **Внутренний порт**: 5000
   - **Протокол**: TCP
   - **IP адрес**: [локальный IP этого ПК]

   **Правило 2 - Bot Service:**
   - **Внешний порт**: 5001
   - **Внутренний порт**: 5001
   - **Протокол**: TCP
   - **IP адрес**: [локальный IP этого ПК]

4. Узнайте внешний IP: https://whatismyipaddress.com

### Доступ

- **Локально**: http://localhost:5000
- **Локальная сеть**: http://192.168.x.x:5000
- **Интернет**: http://ВНЕШНИЙ_IP:5000

⚠️ **ВНИМАНИЕ**: Доступ из интернета открывает API ключи! Используйте VPN или HTTPS.

---

Подробнее: [docs/QUICKSTART.md](docs/QUICKSTART.md)

