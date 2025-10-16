# 📦 Развертывание InfoBot

**Версия:** 1.0 FINAL  
**Дата:** 15 октября 2025

---

## 📥 Что внутри архива

```
InfoBot_Clean_Deploy_v1.0_YYYYMMDD_HHMMSS.zip (~480 KB)
├── Весь код (без секретов)
├── Документация (docs/)
├── config.example.py (шаблон конфигурации)
├── INSTALL.md (инструкция установки)
└── Пустые директории (logs/, data/)
```

**Исключено из архива:**
- ❌ Логи (logs/)
- ❌ Личные данные (data/)
- ❌ API ключи (app/config.py, app/keys.py)
- ❌ Резервные копии (backups/)
- ❌ Python кэш (__pycache__)

---

## 🚀 Развертывание на новом сервере

### Шаг 1: Распакуйте архив
```bash
unzip InfoBot_Clean_Deploy_v1.0_*.zip
cd InfoBot
```

### Шаг 2: Установите зависимости
```bash
pip install -r requirements.txt
```

### Шаг 3: Настройте конфигурацию
```bash
# Скопируйте пример
cp app/config.example.py app/config.py

# Отредактируйте своими ключами
nano app/config.py  # или любой редактор
```

**Что нужно заполнить в config.py:**
```python
EXCHANGES = {
    'BYBIT': {
        'api_key': 'ваш_реальный_api_key',
        'api_secret': 'ваш_реальный_secret'
    }
}

TELEGRAM_BOT_TOKEN = 'ваш_telegram_token'
TELEGRAM_CHAT_ID = 'ваш_chat_id'
```

### Шаг 4: Запустите систему
```bash
# Терминал 1
python app.py

# Терминал 2  
python bots.py
```

### Шаг 5: Проверьте работу
Откройте: http://localhost:5000

**Готово!** Система развернута и работает.

---

## 🔒 Безопасность

### ⚠️ ВАЖНО:
1. **Никогда не коммитьте** `app/config.py` с реальными ключами в Git
2. **Добавьте в .gitignore:**
   ```
   app/config.py
   app/keys.py
   data/*.json
   logs/*.log
   ```
3. **API ключи** должны иметь минимальные права (только торговля)
4. **Не открывайте** порты 5000/5001 в интернет без защиты

---

## 📋 Контрольный список

- [ ] Архив распакован
- [ ] Зависимости установлены (`pip install -r requirements.txt`)
- [ ] config.py создан из config.example.py
- [ ] API ключи заполнены
- [ ] Telegram токен настроен (опционально)
- [ ] app.py запущен (порт 5000)
- [ ] bots.py запущен (порт 5001)
- [ ] UI открывается (http://localhost:5000)
- [ ] Боты видны в UI

---

## 🆘 Проблемы?

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "Port already in use"
```bash
# Windows
Get-Process -Id (Get-NetTCPConnection -LocalPort 5001).OwningProcess | Stop-Process
```

### "Биржа не инициализирована"
Проверьте API ключи в `app/config.py`

---

## 📚 Документация

После развертывания:
- **README.md** - Описание системы
- **INSTALL.md** - Детальная установка
- **docs/QUICKSTART.md** - Быстрый старт
- **docs/CONFIGURATION.md** - Настройка параметров

---

**Успешного развертывания!** 🚀

