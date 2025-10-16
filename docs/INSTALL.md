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

Подробнее: [docs/QUICKSTART.md](docs/QUICKSTART.md)

