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

**Терминал 1:**
```bash
python app.py
```

**Терминал 2:**
```bash
python bots.py
```

## Шаг 5: Проверка

Откройте: http://localhost:5000

**Готово!** Система работает.

---

## 🔗 Доступ из локальной сети / интернета

### Настройка Windows Firewall

Для доступа к серверу из локальной сети или интернета откройте порт 5000 в Windows Firewall:

**PowerShell от имени администратора:**
```powershell
New-NetFirewallRule -DisplayName "InfoBot Flask Server" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow -Profile Any
```

### Настройка роутера (для доступа из интернета)

1. Войдите в настройки роутера (обычно 192.168.1.1 или 192.168.0.1)
2. Найдите раздел "Port Forwarding" / "Виртуальные серверы"
3. Создайте правило:
   - **Внешний порт**: 5000
   - **Внутренний порт**: 5000
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

