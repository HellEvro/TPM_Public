# Скрипт автоматической настройки для первого запуска

## Описание

`first_run_setup.py` - автоматически настраивает проект при первом запуске на новом ПК.

## Что делает скрипт

1. ✅ Проверяет версию Python (требуется 3.8+)
2. ✅ Создает виртуальное окружение `.venv`
3. ✅ Устанавливает все зависимости из `requirements.txt` в `.venv`
4. ✅ Создает конфигурационные файлы из примеров:
   - `app/config.py` из `app/config.example.py`
   - `app/keys.py` из `app/keys.example.py`
5. ✅ Создает необходимые директории (`logs/`, `data/`)

## Автоматический запуск

Скрипт автоматически запускается при:
- Ошибке импорта модулей в `app.py`, `bots.py` или `ai.py`
- Отсутствии файла `app/config.py`
- Запуске `start_infobot_manager.bat` без установленных зависимостей

## Ручной запуск

```bash
# Windows
python scripts\first_run_setup.py

# Linux/macOS
python3 scripts/first_run_setup.py
```

## После настройки

После успешной настройки перезапустите приложение:
- `start_infobot_manager.bat` (Windows)
- `./start_infobot_manager.sh` (Linux/macOS)
- `python app.py`
- `python bots.py`

## Требования

- Python 3.8 или выше
- Доступ в интернет (для установки зависимостей)
- Права на создание директорий и файлов

