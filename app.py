import errno
import os
import sys
import warnings
# Подавление FutureWarning LeafSpec (PyTorch/зависимости) — до любых импортов, которые могут его вызвать
warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*TreeSpec.*is_leaf.*")
# Корень проекта в path до импорта utils — иначе sklearn_parallel_config не найдётся при запуске из другой директории
_root = os.path.dirname(os.path.abspath(__file__))
if _root and _root not in sys.path:
    sys.path.insert(0, _root)
# Подавление UserWarning sklearn и FutureWarning для дочерних процессов
_pw = os.environ.get("PYTHONWARNINGS", "").strip()
_add = "ignore::UserWarning:sklearn.utils.parallel,ignore::FutureWarning"
os.environ["PYTHONWARNINGS"] = f"{_pw},{_add}" if _pw else _add
import utils.sklearn_parallel_config  # noqa: F401 — первым до sklearn (вариант A: оба Parallel/delayed из sklearn)
import base64
from io import BytesIO
from flask import Flask, render_template, jsonify, request
import threading
import time
from datetime import datetime, timedelta
import subprocess
import webbrowser
from threading import Timer
from pathlib import Path

# Настройка кодировки для Windows консоли
if os.name == 'nt':
    try:
        # Пытаемся установить UTF-8 для консоли Windows
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        # Если не получилось, пробуем через os
        try:
            import subprocess
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
        except:
            pass

# Все конфиги в configs/
_PROJECT_ROOT = Path(__file__).resolve().parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_CONFIG_PATH = _PROJECT_ROOT / "app" / "config.py"  # заглушка, реэкспорт из configs
_APP_CONFIG_PATH = _CONFIGS_DIR / "app_config.py"
_APP_CONFIG_EXAMPLE_PATH = _CONFIGS_DIR / "app_config.example.py"
_KEYS_PATH = _CONFIGS_DIR / "keys.py"
_KEYS_EXAMPLE_PATH = _CONFIGS_DIR / "keys.example.py"

# Создать configs/app_config.py из примера, если нет
_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
if not _APP_CONFIG_PATH.exists() and _APP_CONFIG_EXAMPLE_PATH.exists():
    import shutil
    shutil.copyfile(_APP_CONFIG_EXAMPLE_PATH, _APP_CONFIG_PATH)
    sys.stderr.write("✅ Создан configs/app_config.py из примера\n")

# Миграция: app/keys.py → configs/keys.py (если configs/keys.py ещё нет)
try:
    from bots_modules.config_writer import migrate_old_keys_to_configs
    if migrate_old_keys_to_configs(str(_PROJECT_ROOT)):
        sys.stderr.write("✅ Ключи перенесены из app/ в configs/keys.py\n")
except Exception as e:
    sys.stderr.write(f"⚠️ Миграция ключей: {e}\n")
# Если configs/keys.py всё ещё нет — создать из примера (на случай сбоя миграции)
if not _KEYS_PATH.exists() and _KEYS_EXAMPLE_PATH.exists():
    import shutil
    shutil.copyfile(_KEYS_EXAMPLE_PATH, _KEYS_PATH)
    sys.stderr.write("✅ Создан configs/keys.py из примера (заполните ключи)\n")

# Миграция: создание configs/bot_config.py из примера при отсутствии (конфиги только в configs/)
try:
    from bots_modules.config_writer import migrate_old_bot_config_to_configs
    if migrate_old_bot_config_to_configs(str(_PROJECT_ROOT)):
        sys.stderr.write("✅ Создан configs/bot_config.py из примера\n")
except Exception as e:
    sys.stderr.write(f"⚠️ Миграция конфига ботов: {e}\n")

# RSI-фикс: автопатч configs/bot_config.py (устаревший fallback rsi6h → делегирование в config_loader)
try:
    from bot_engine import ensure_rsi_fix_applied
    ensure_rsi_fix_applied()
except Exception:
    pass

# Патчи обратной совместимости (например недостающие классы в configs/bot_config.py) — при запуске app.py и при лаунчере
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
try:
    from patches.runner import run_patches
    _applied = run_patches(_PROJECT_ROOT)
    if _applied:
        sys.stderr.write(f"[Patches] Применены патчи: {', '.join(_applied)}\n")
except Exception as _e:
    sys.stderr.write(f"[Patches] Ошибка применения патчей: {_e}\n")

if not _CONFIG_PATH.exists():
    # Используем stderr, так как logger еще не настроен
    sys.stderr.write("\n" + "=" * 80 + "\n")
    sys.stderr.write("⚠️  Файл app/config.py (заглушка) не найден.\n")
    sys.stderr.write("=" * 80 + "\n\n")

    # Создать заглушку app/config.py и убедиться, что configs/ заполнены
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not _APP_CONFIG_PATH.exists() and _APP_CONFIG_EXAMPLE_PATH.exists():
            import shutil
            shutil.copyfile(_APP_CONFIG_EXAMPLE_PATH, _APP_CONFIG_PATH)
        if not _KEYS_PATH.exists() and _KEYS_EXAMPLE_PATH.exists():
            import shutil
            shutil.copyfile(_KEYS_EXAMPLE_PATH, _KEYS_PATH)
        # Заглушка app/config.py
        _CONFIG_PATH.write_text('# Реальный конфиг в configs/app_config.py\nfrom configs.app_config import *  # noqa: F401, F403\n', encoding='utf-8')
        sys.stderr.write("✅ Создан app/config.py (заглушка) и configs/ при необходимости.\n")
        sys.stderr.write("   Отредактируйте configs/keys.py и configs/app_config.py под себя.\n\n")
    except Exception as e:
        sys.stderr.write("\n" + "=" * 80 + "\n")
        sys.stderr.write("❌ ОШИБКА: не удалось создать конфигурацию автоматически!\n")
        sys.stderr.write(f"Причина: {e}\n")
        sys.stderr.write("=" * 80 + "\n\n")
        sys.stderr.write("📝 Для первого запуска выполните:\n\n")
        if os.name == 'nt':
            sys.stderr.write("   copy configs\\app_config.example.py configs\\app_config.py\n")
            sys.stderr.write("   copy configs\\keys.example.py configs\\keys.py\n")
        else:
            sys.stderr.write("   cp configs/app_config.example.py configs/app_config.py\n")
            sys.stderr.write("   cp configs/keys.example.py configs/keys.py\n")
        sys.stderr.write("\n📖 Подробная инструкция: docs/INSTALL.md\n\n")
        sys.exit(1)

from app.config import *

# Импортируем БД для app.py
from bot_engine.app_database import get_app_database

# Конфигурация резервного копирования (значения по умолчанию)
_DATABASE_BACKUP_DEFAULTS = {
    'ENABLED': True,
    'INTERVAL_MINUTES': 180,
    'RUN_ON_START': True,
    'APP_ENABLED': True,   # app.py бэкапит только app_data.db
    'AI_ENABLED': False,
    'BOTS_ENABLED': False,
    'BACKUP_DIR': None,
    'MAX_RETRIES': 3,
    'KEEP_LAST_N': 5,
}

if 'DATABASE_BACKUP' not in globals() or not isinstance(globals().get('DATABASE_BACKUP'), dict):
    DATABASE_BACKUP = _DATABASE_BACKUP_DEFAULTS.copy()
else:
    _merged_backup_config = _DATABASE_BACKUP_DEFAULTS.copy()
    _merged_backup_config.update(DATABASE_BACKUP)
    DATABASE_BACKUP = _merged_backup_config

# Конфигурация синхронизации времени Windows (значения по умолчанию)
_TIME_SYNC_DEFAULTS = {
    'ENABLED': False,
    'INTERVAL_MINUTES': 30,  # синхронизация раз в полчаса
    'SERVER': 'time.windows.com',
    'RUN_ON_START': True,
    'REQUIRE_ADMIN': True,
}

if 'TIME_SYNC' not in globals() or not isinstance(globals().get('TIME_SYNC'), dict):
    TIME_SYNC = _TIME_SYNC_DEFAULTS.copy()
else:
    _merged_time_sync_config = _TIME_SYNC_DEFAULTS.copy()
    _merged_time_sync_config.update(TIME_SYNC)
    TIME_SYNC = _merged_time_sync_config

import sys
from app.telegram_notifier import TelegramNotifier
from exchanges.exchange_factory import ExchangeFactory
import json
import logging
from utils.color_logger import setup_color_logging
from bot_engine.backup_service import run_backup_scheduler_loop

# Проверка валидности API ключей
def check_api_keys():
    """Проверяет наличие настроенных API ключей"""
    try:
        # Проверяем наличие файла с ключами
        if not os.path.exists('app/keys.py'):
            return False
            
        active_exchange = EXCHANGES.get(ACTIVE_EXCHANGE, {})
        api_key = active_exchange.get('api_key', '')
        api_secret = active_exchange.get('api_secret', '')
        
        # Проверяем что ключи не пустые и не содержат "YOUR_" (из примера)
        if not api_key or not api_secret:
            return False
        if 'YOUR_' in api_key or 'YOUR_' in api_secret:
            return False
        if api_key == 'YOUR_API_KEY_HERE' or api_secret == 'YOUR_SECRET_KEY_HERE':
            return False
            
        return True
    except:
        return False

# DEMO режим: если ключи не настроены, приложение работает в UI-режиме без торговли
class DemoExchange:
    def get_positions(self):
        return [], []

    def get_wallet_balance(self):
        return {
            'total_balance': 0,
            'available_balance': 0,
            'realized_pnl': 0
        }

    def get_closed_pnl(self, *args, **kwargs):
        return []

    def get_ticker(self, symbol):
        return {
            'symbol': symbol,
            'price': None,
            'demo': True
        }

    def close_position(self, *args, **kwargs):
        return {'success': False, 'message': 'DEMO режим: торговля отключена'}

# Предупреждение если ключи не настроены
if not check_api_keys():
    # Логгер еще не настроен, используем stderr для критических предупреждений
    import sys
    sys.stderr.write("\n" + "="*80 + "\n")
    sys.stderr.write("⚠️  ВНИМАНИЕ: API ключи не настроены!\n")
    sys.stderr.write("="*80 + "\n")
    sys.stderr.write("\n")
    sys.stderr.write("📌 Текущий статус:\n")
    sys.stderr.write(f"   Биржа: {ACTIVE_EXCHANGE}\n")
    if not os.path.exists('app/keys.py'):
        sys.stderr.write("   Файл с ключами: app/keys.py НЕ НАЙДЕН\n")
    else:
        sys.stderr.write("   API ключи: НЕ НАСТРОЕНЫ или СОДЕРЖАТ ПРИМЕРЫ\n")
    sys.stderr.write("\n")
    sys.stderr.write("💡 Что нужно сделать:\n")
    sys.stderr.write("   1. Скопируйте configs/app_config.example.py -> configs/app_config.py (если еще не сделали)\n")
    sys.stderr.write("   2. Создайте app/keys.py с реальными ключами\n")
    sys.stderr.write("   3. Или добавьте ключи в configs/app_config.py или configs/keys.py\n")
    sys.stderr.write("   4. Перезапустите приложение\n")
    sys.stderr.write("\n")
    sys.stderr.write("⚠️  Приложение запущено в DEMO режиме (только UI, без торговли)\n")
    sys.stderr.write("\n")
    sys.stderr.write("="*80 + "\n")
    sys.stderr.write("\n")
import requests
from threading import Lock
from app.language import get_current_language, save_language
import concurrent.futures
from functools import partial
# from bot_engine.bot_manager import BotManager  # Убираем - теперь в отдельном сервисе

# Добавим константы
# BOTS_SERVICE_URL теперь определяется динамически на клиенте через JavaScript
class DEFAULTS:
    PNL_THRESHOLD = 10

# Инициализация БД для app.py
app_db = None

def get_app_db():
    """Получает экземпляр БД для app.py (ленивая инициализация)"""
    global app_db
    if app_db is None:
        app_db = get_app_database()
    return app_db

# Глобальные переменные для хранения данных (загружаются из БД)
def load_positions_data():
    """Загружает positions_data из БД"""
    db = get_app_db()
    return db.load_positions_data()

def save_positions_data(data):
    """Сохраняет positions_data в БД"""
    db = get_app_db()
    return db.save_positions_data(data)

def load_max_values():
    """Загружает max_values из БД"""
    db = get_app_db()
    return db.load_max_values()

def save_max_values(max_profit, max_loss):
    """Сохраняет max_values в БД"""
    db = get_app_db()
    return db.save_max_values(max_profit, max_loss)

# Инициализируем данные из БД при старте
positions_data = load_positions_data()
max_profit_values, max_loss_values = load_max_values()

app = Flask(__name__, static_folder='static')
app.config['DEBUG'] = APP_DEBUG
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# ✅ ОТКЛЮЧЕНИЕ КЭША ДЛЯ ВСЕХ СТАТИЧЕСКИХ ФАЙЛОВ (особенно JS)
# ✅ КРИТИЧНО: Flask по умолчанию может кэшировать статические файлы, поэтому принудительно отключаем
@app.after_request
def add_no_cache_headers(response):
    """Принудительно отключает кэширование для всех статических файлов, особенно JavaScript"""
    # ✅ Проверяем все возможные варианты статических файлов
    is_static = (
        request.endpoint == 'static' or 
        request.path.startswith('/static/') or
        '/static/' in request.path or
        request.path.endswith(('.js', '.css', '.ico', '.png', '.jpg', '.gif', '.svg'))
    )
    
    # ✅ Для JavaScript файлов - ОСОБЕННО СТРОГО отключаем кэш (проверяем первым!)
    if request.path.endswith('.js'):
        # Принудительно удаляем ВСЕ заголовки кэширования
        headers_to_remove = ['Last-Modified', 'ETag', 'Cache-Control', 'Expires', 'Pragma']
        for header in headers_to_remove:
            response.headers.pop(header, None)
        
        # Устанавливаем строгие заголовки против кэширования
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0, private'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        
        # ✅ Уникальный ETag с timestamp для принудительной перезагрузки
        response.headers['ETag'] = f'"nocache-{int(time.time() * 1000)}"'
        
        # ✅ Дополнительный заголовок для старых браузеров
        response.headers['Vary'] = '*'
        
        # DEBUG: Логируем в dev режиме
        if app.config.get('DEBUG'):
            cache_logger = logging.getLogger('app')
            pass
    
    # ✅ Для остальных статических файлов также отключаем кэш
    elif is_static:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['ETag'] = f'"nocache-{int(time.time() * 1000)}"'
    
    return response

telegram = TelegramNotifier()

# Создаем директорию для логов, если её нет
if not os.path.exists('logs'):
    os.makedirs('logs')

# Настройка цветного логирования с фильтром уровней из app/config
try:
    console_levels = CONSOLE_LOG_LEVELS if 'CONSOLE_LOG_LEVELS' in globals() else []
    setup_color_logging(console_log_levels=console_levels if console_levels else None, log_file='logs/app.log')
except Exception as e:
    setup_color_logging(log_file='logs/app.log')

# Отключаем DEBUG логи от внешних библиотек ДО их импорта
# flask-cors - логирует неформатированные сообщения типа "Settings CORS headers: %s"
flask_cors_logger = logging.getLogger('flask_cors')
flask_cors_logger.setLevel(logging.WARNING)
flask_cors_core_logger = logging.getLogger('flask_cors.core')
flask_cors_core_logger.setLevel(logging.WARNING)
for handler in flask_cors_logger.handlers[:]:
    flask_cors_logger.removeHandler(handler)
for handler in flask_cors_core_logger.handlers[:]:
    flask_cors_core_logger.removeHandler(handler)

# matplotlib - логирует неформатированные сообщения при импорте
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)
for handler in matplotlib_logger.handlers[:]:
    matplotlib_logger.removeHandler(handler)

matplotlib_font_manager_logger = logging.getLogger('matplotlib.font_manager')
matplotlib_font_manager_logger.setLevel(logging.WARNING)
for handler in matplotlib_font_manager_logger.handlers[:]:
    matplotlib_font_manager_logger.removeHandler(handler)

matplotlib_backends_logger = logging.getLogger('matplotlib.backends')
matplotlib_backends_logger.setLevel(logging.WARNING)
for handler in matplotlib_backends_logger.handlers[:]:
    matplotlib_backends_logger.removeHandler(handler)

# TensorFlow - логирует "Falling back to TensorFlow client..."
tensorflow_logger = logging.getLogger('tensorflow')
tensorflow_logger.setLevel(logging.WARNING)
tensorflow_python_logger = logging.getLogger('tensorflow.python')
tensorflow_python_logger.setLevel(logging.WARNING)
tensorflow_core_logger = logging.getLogger('tensorflow.core')
tensorflow_core_logger.setLevel(logging.WARNING)
for handler in tensorflow_logger.handlers[:]:
    tensorflow_logger.removeHandler(handler)
for handler in tensorflow_python_logger.handlers[:]:
    tensorflow_python_logger.removeHandler(handler)
for handler in tensorflow_core_logger.handlers[:]:
    tensorflow_core_logger.removeHandler(handler)

# Импортируем систему ротации логов
from utils.log_rotation import RotatingFileHandlerWithSizeLimit
import logging

# Словарь для кэширования логгеров
_log_file_handlers = {}

def log_to_file(filename, data):
    """
    Записывает данные в файл с временной меткой и автоматической ротацией при превышении 10MB
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_path = f'logs/{filename}'
    
    # Создаем или получаем существующий handler для файла
    if log_path not in _log_file_handlers:
        handler = RotatingFileHandlerWithSizeLimit(
            filename=log_path,
            max_bytes=10 * 1024 * 1024,  # 10MB
            backup_count=0,  # Перезаписываем файл
            encoding='utf-8'
        )
        logger = logging.getLogger(f'AppLog_{filename}')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False
        _log_file_handlers[log_path] = logger
    
    # Записываем в лог
    logger = _log_file_handlers[log_path]
    logger.info(f"\n=== {timestamp} ===\n{data}\n")

def format_positions(positions):
    """Форматирует позиции для записи в лог"""
    if not positions:
        return "No positions"
    
    result = []
    for pos in positions:
        result.append(
            f"Symbol: {pos['symbol']}\n"
            f"PnL: {pos['pnl']:.3f} USDT\n"
            f"ROI: {pos['roi']:.2f}%\n"
        )
    return "\n".join(result)

def format_stats(stats):
    """Форматирует стаику для записи в лог"""
    return (
        f"Total PnL: {stats['total_pnl']:.3f} USDT\n"
        f"Total profit: {stats['total_profit']:.3f} USDT\n"
        f"Total loss: {stats['total_loss']:.3f} USDT\n"
        f"Number of high-profitable positions: {stats['high_profitable_count']}\n"
        f"Number of profitable positions: {stats['profitable_count']}\n"
        f"Number of losing positions: {stats['losing_count']}\n"
        f"\nTOP-3 profitable:\n{format_positions(stats['top_profitable'])}\n"
        f"\nTOP-3 losing:\n{format_positions(stats['top_losing'])}"
    )

stats_lock = Lock()
# Блокировка для операций matplotlib (не потокобезопасен)
matplotlib_lock = Lock()
closed_pnl_loader_stop_event = threading.Event()
time_sync_stop_event = threading.Event()


def check_admin_rights():
    """Проверяет, запущен ли скрипт с правами администратора (только для Windows)"""
    if sys.platform != 'win32':
        return False
    try:
        result = subprocess.run(
            ['net', 'session'],
            capture_output=True,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def configure_time_service(server="time.windows.com", silent=False):
    """
    Настраивает службу времени для синхронизации с указанным сервером
    
    Args:
        server: Адрес сервера времени (по умолчанию time.windows.com)
        silent: Если True, не выводит сообщения в консоль
        
    Returns:
        Tuple[bool, str]: (успех, сообщение)
    """
    if sys.platform != 'win32':
        return False, "Синхронизация времени доступна только для Windows"
    
    if not check_admin_rights():
        return False, "Требуются права администратора для настройки службы времени"
    
    try:
        command = [
            'w32tm', '/config',
            f'/manualpeerlist:"{server}"',
            '/syncfromflags:manual',
            '/reliable:yes',
            '/update'
        ]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode != 0:
            return False, f"Ошибка настройки службы времени: {result.stderr}"
        
        # Перезапуск службы времени
        if not silent:
            app_logger = logging.getLogger('app')
            app_logger.info("[TimeSync] Перезапуск службы времени...")
        
        subprocess.run(['net', 'stop', 'w32time'], capture_output=True, check=False)
        subprocess.run(['net', 'start', 'w32time'], capture_output=True, check=False)
        
        return True, "Служба времени успешно настроена"
    except Exception as e:
        return False, f"Ошибка настройки службы времени: {str(e)}"


def sync_time(silent=False):
    """
    Выполняет принудительную синхронизацию времени
    
    Args:
        silent: Если True, не выводит сообщения в консоль
        
    Returns:
        Tuple[bool, str]: (успех, сообщение)
    """
    if sys.platform != 'win32':
        return False, "Синхронизация времени доступна только для Windows"
    
    time_sync_cfg = TIME_SYNC if 'TIME_SYNC' in globals() else {}
    require_admin = time_sync_cfg.get('REQUIRE_ADMIN', True)
    if require_admin and not check_admin_rights():
        return False, "Требуются права администратора для синхронизации времени"
    
    try:
        if not silent:
            app_logger = logging.getLogger('app')
            app_logger.info("[TimeSync] Выполняется синхронизация времени...")
        
        result = subprocess.run(
            ['w32tm', '/resync'],
            capture_output=True,
            text=True,
            check=False,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode == 0:
            return True, "Время успешно синхронизировано"
        else:
            # Если синхронизация не удалась, попробуем настроить службу
            if not silent:
                app_logger = logging.getLogger('app')
                app_logger.info("[TimeSync] Попытка настроить службу времени...")
            
            config_success, config_msg = configure_time_service(silent=True)
            if config_success:
                # Повторная попытка синхронизации
                result = subprocess.run(
                    ['w32tm', '/resync'],
                    capture_output=True,
                    text=True,
                    check=False,
                    encoding='utf-8',
                    errors='ignore'
                )
                if result.returncode == 0:
                    return True, "Время успешно синхронизировано после настройки службы"
            
            return False, f"Ошибка синхронизации времени: {result.stderr}"
    except Exception as e:
        return False, f"Ошибка синхронизации времени: {str(e)}"


def time_sync_loop():
    """Фоновый планировщик синхронизации времени Windows"""
    time_sync_logger = logging.getLogger('TimeSync')
    
    # Проверяем платформу
    if sys.platform != 'win32':
        time_sync_logger.info("[TimeSync] Синхронизация времени доступна только для Windows")
        return
    
    # Получаем настройки из конфига
    time_sync_config = TIME_SYNC if 'TIME_SYNC' in globals() else {}
    
    if not time_sync_config.get('ENABLED', False):
        time_sync_logger.info("[TimeSync] Автоматическая синхронизация времени выключена настройками")
        return
    
    # Проверка прав администратора (при REQUIRE_ADMIN=True без прав планировщик не запускаем)
    require_admin = time_sync_config.get('REQUIRE_ADMIN', True)
    if require_admin and not check_admin_rights():
        time_sync_logger.warning(
            "[TimeSync] ⚠️ Требуются права администратора для синхронизации времени. "
            "Запустите приложение от имени администратора или установите REQUIRE_ADMIN=False"
        )
        return
    if not require_admin and not check_admin_rights():
        time_sync_logger.info(
            "[TimeSync] Запуск без прав администратора (REQUIRE_ADMIN=False). "
            "Синхронизация может не выполняться; при запуске от администратора будет работать."
        )
    
    interval_minutes = time_sync_config.get('INTERVAL_MINUTES', 30)
    try:
        interval_minutes = float(interval_minutes)
    except (TypeError, ValueError):
        time_sync_logger.warning("[TimeSync] Некорректное значение INTERVAL_MINUTES, используется 30 минут")
        interval_minutes = 30
    
    interval_seconds = max(60, int(interval_minutes * 60))
    server = time_sync_config.get('SERVER', 'time.windows.com')
    
    time_sync_logger.info(
        "[TimeSync] Планировщик запущен: каждые %s минут (%.0f секунд). Сервер: %s",
        interval_minutes,
        interval_seconds,
        server
    )
    
    # Первоначальная настройка службы времени (только с правами админа)
    if check_admin_rights():
        time_sync_logger.info("[TimeSync] Первоначальная настройка службы времени...")
        configure_time_service(server=server, silent=True)
    
    # Синхронизация при запуске (если включено)
    if time_sync_config.get('RUN_ON_START', True):
        success, message = sync_time(silent=True)
        if success:
            time_sync_logger.info(f"[TimeSync] ✓ {message}")
        else:
            time_sync_logger.warning(f"[TimeSync] ✗ {message}")
    
    sync_count = 0
    error_count = 0
    
    try:
        while not time_sync_stop_event.wait(interval_seconds):
            sync_count += 1
            time_sync_logger.info(f"[TimeSync] --- Синхронизация #{sync_count} ---")
            
            success, message = sync_time(silent=True)
            
            if success:
                time_sync_logger.info(f"[TimeSync] ✓ {message}")
                error_count = 0  # Сбрасываем счетчик ошибок при успехе
            else:
                error_count += 1
                time_sync_logger.warning(f"[TimeSync] ✗ {message}")
                
                # Если много ошибок подряд, попробуем переконфигурировать службу
                if error_count >= 3:
                    time_sync_logger.info("[TimeSync] Много ошибок подряд. Попытка переконфигурации службы...")
                    configure_time_service(server=server, silent=True)
                    error_count = 0
            
            # Показываем время до следующей синхронизации
            next_sync = datetime.now() + timedelta(minutes=interval_minutes)
            pass
            
    except Exception as e:
        time_sync_logger.error(f"[TimeSync] Критическая ошибка в планировщике синхронизации времени: {e}")
        import traceback
        pass


def background_closed_pnl_loader():
    """Фоновый процесс для загрузки закрытых PnL из биржи в БД каждые 30 секунд"""
    app_logger = logging.getLogger('app')
    app_logger.info("[CLOSED_PNL_LOADER] Запуск фонового процесса загрузки closed_pnl...")
    
    while not closed_pnl_loader_stop_event.is_set():
        try:
            db = get_app_db()
            
            # Получаем timestamp последней загруженной позиции
            last_timestamp = db.get_latest_closed_pnl_timestamp(exchange=ACTIVE_EXCHANGE)
            
            pass
            
            # Загружаем новые закрытые позиции с биржи
            # Если есть last_timestamp, загружаем только новые (после этого timestamp)
            # Если нет, загружаем все данные (первая загрузка)
            try:
                if last_timestamp:
                    # Загружаем только новые позиции (после last_timestamp)
                    # Используем период 'custom' с start_date = last_timestamp
                    closed_pnl = current_exchange.get_closed_pnl(
                        sort_by='time',
                        period='custom',
                        start_date=last_timestamp + 1,  # +1 чтобы не дублировать последнюю (в миллисекундах)
                        end_date=None
                    )
                else:
                    # Первая загрузка - загружаем все данные
                    app_logger.info("[CLOSED_PNL_LOADER] Первая загрузка - загружаем все закрытые позиции...")
                    closed_pnl = current_exchange.get_closed_pnl(
                        sort_by='time',
                        period='all'
                    )
                
                if closed_pnl:
                    # Сохраняем в БД
                    saved = db.save_closed_pnl(closed_pnl, exchange=ACTIVE_EXCHANGE)
                    if saved:
                        app_logger.info(f"[CLOSED_PNL_LOADER] Загружено {len(closed_pnl)} новых закрытых позиций в БД")
                    else:
                        app_logger.warning(f"[CLOSED_PNL_LOADER] Не удалось сохранить {len(closed_pnl)} позиций в БД")
                else:
                    pass
                    
            except Exception as e:
                app_logger.error(f"[CLOSED_PNL_LOADER] Ошибка загрузки closed_pnl с биржи: {e}")
                import traceback
                pass
            
            # Ждем 30 секунд до следующей загрузки
            closed_pnl_loader_stop_event.wait(30)
            
        except Exception as e:
            app_logger.error(f"[CLOSED_PNL_LOADER] Критическая ошибка в фоновом процессе: {e}")
            import traceback
            pass
            # Ждем 30 секунд перед следующей попыткой
            closed_pnl_loader_stop_event.wait(30)
    
    app_logger.info("[CLOSED_PNL_LOADER] Фоновый процесс загрузки closed_pnl остановлен")

def background_update():
    global positions_data, max_profit_values, max_loss_values, last_stats_time
    last_log_minute = -1
    last_stats_time = None
    thread_id = threading.get_ident()
    
    while True:
        try:
            current_time = time.time()
            
            with stats_lock:
                time_since_last = current_time - last_stats_time if last_stats_time else None
                should_send_stats = (
                    TELEGRAM_NOTIFY.get('STATISTICS', False) and 
                    TELEGRAM_NOTIFY.get('STATISTICS_INTERVAL_ENABLED', True) and
                    (last_stats_time is None or 
                     current_time - last_stats_time >= TELEGRAM_NOTIFY['STATISTICS_INTERVAL'])
                )

            positions, rapid_growth = current_exchange.get_positions()
            if not positions:
                # Fallback: app и bots в разных процессах — пробуем взять позиции с Bots
                try:
                    resp = requests.get('http://127.0.0.1:5001/api/bots/positions-for-app', timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get('success') and data.get('total_trades', 0) > 0:
                            positions = (data.get('high_profitable', []) + data.get('profitable', []) +
                                        data.get('losing', []))
                            rapid_growth = data.get('rapid_growth', [])
                            if positions:
                                logging.getLogger('app').info(f"[POSITIONS] Fallback: {len(positions)} позиций с Bots")
                except Exception:
                    pass
            if not positions:
                # Закрытые позиции не возвращаются биржей — очищаем список
                positions_data.update({
                    'high_profitable': [], 'profitable': [], 'losing': [],
                    'total_trades': 0, 'rapid_growth': [],
                    'stats': {
                        'total_trades': 0, 'high_profitable_count': 0, 'profitable_count': 0, 'losing_count': 0,
                        'top_profitable': [], 'top_losing': [], 'total_pnl': 0, 'total_profit': 0, 'total_loss': 0
                    },
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                save_positions_data(positions_data)
                time.sleep(2)
                continue

            # Добавляем проверку каждой позиции для уведомлений
            for position in positions:
                telegram.check_position_notifications(position)

            # Проверяем быстрорастущие позиции
            if rapid_growth:
                telegram.check_rapid_growth(rapid_growth)

            high_profitable = []
            profitable = []
            losing = []
            
            total_profit = 0
            total_loss = 0
            
            # Обновляем общее количество сделок
            positions_data['total_trades'] = len(positions)
            positions_data['rapid_growth'] = rapid_growth
            
            # Распределяем позиции по категориям
            for position in positions:
                pnl = position['pnl']
                symbol = position.get('symbol', '')
                
                # Обновляем максимальные значения
                if pnl > 0:
                    if symbol not in max_profit_values or pnl > max_profit_values[symbol]:
                        max_profit_values[symbol] = pnl
                elif pnl < 0:
                    if symbol not in max_loss_values or abs(pnl) > abs(max_loss_values.get(symbol, 0)):
                        max_loss_values[symbol] = pnl
                
                if pnl > 0:
                    if pnl >= DEFAULTS.PNL_THRESHOLD:
                        high_profitable.append(position)
                    else:
                        profitable.append(position)
                    total_profit += pnl
                elif pnl < 0:
                    losing.append(position)
                    total_loss += pnl
            
            # Сортировка позиций
            high_profitable.sort(key=lambda x: x['pnl'], reverse=True)
            profitable.sort(key=lambda x: x['pnl'], reverse=True)
            losing.sort(key=lambda x: x['pnl'])
            
            # Получаем TOP-3
            all_profitable = high_profitable + profitable
            all_profitable.sort(key=lambda x: x['pnl'], reverse=True)
            top_profitable = all_profitable[:3] if all_profitable else []
            top_losing = losing[:3] if losing else []
            
            # Обновляем positions_data
            stats = {
                'total_pnl': total_profit + total_loss,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'high_profitable_count': len(high_profitable),
                'profitable_count': len(high_profitable) + len(profitable),
                'losing_count': len(losing),
                'top_profitable': top_profitable,
                'top_losing': top_losing,
                'total_trades': len(positions)
            }
            
            positions_data.update({
                'high_profitable': high_profitable,
                'profitable': profitable,
                'losing': losing,
                'stats': stats,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Сохраняем в БД
            save_positions_data(positions_data)
            save_max_values(max_profit_values, max_loss_values)

            # Отправка статистики в Telegram только если нужно
            if should_send_stats:
                try:
                    with stats_lock:
                        # Логирование для диагностики (только при реальной отправке)
                        thread_logger = logging.getLogger('app')
                        if last_stats_time is None:
                            thread_logger.info(f"[Thread {thread_id}] Первый запуск - отправляем статистику")
                        else:
                            minutes_passed = time_since_last / 60
                            thread_logger.info(f"[Thread {thread_id}] Прошло {minutes_passed:.1f} минут - отправляем статистику")
                        
                        thread_logger.info(f"[Thread {thread_id}] Acquired stats_lock for sending")
                        thread_logger.info(f"[Thread {thread_id}] Sending statistics...")
                        telegram.send_statistics(positions_data['stats'])
                        last_stats_time = current_time
                        thread_logger.info(f"[Thread {thread_id}] Stats sent at {datetime.fromtimestamp(current_time).strftime('%H:%M:%S.%f')}")
                        thread_logger.info(f"[Thread {thread_id}] Released stats_lock after sending")
                except Exception as e:
                    thread_logger = logging.getLogger('app')
                    thread_logger.error(f"[Thread {thread_id}] Error sending statistics: {e}")

            # Логируем только при изменении количества позиций или отправке статистики
            current_positions_count = positions_data['total_trades']
            if should_send_stats or current_positions_count != getattr(background_update, 'last_positions_count', -1):
                profitable_count = len([p for p in positions if p['pnl'] > 0])
                losing_count = len([p for p in positions if p['pnl'] < 0])
                thread_logger = logging.getLogger('app')
                thread_logger.info(f"[Thread {thread_id}] Updated positions: {current_positions_count} (прибыльные: {profitable_count}, убыточные: {losing_count})")
                background_update.last_positions_count = current_positions_count
            time.sleep(2)
            
        except Exception as e:
            thread_logger = logging.getLogger('app')
            thread_logger.error(f"Error in background_update: {str(e)}")
            telegram.send_error(str(e))
            time.sleep(5)

# Флаг для отслеживания первого апуска
FIRST_RUN = True

def open_browser():
    """Открывает браузер только при первом запуске"""
    global FIRST_RUN
    if FIRST_RUN and not os.environ.get('WERKZEUG_RUN_MAIN'):
        webbrowser.open(f'http://localhost:{APP_PORT}')
        FIRST_RUN = False

@app.route('/')
def index():
    # ✅ Передаем timestamp для cache-busting JavaScript файлов
    import time
    timestamp = int(time.time() * 1000)  # миллисекунды для уникальности
    return render_template('index.html', get_current_language=get_current_language, cache_version=timestamp)

@app.route('/bots')
def bots_page():
    """Страница управления ботами"""
    # ✅ Передаем timestamp для cache-busting JavaScript файлов
    import time
    timestamp = int(time.time() * 1000)  # миллисекунды для уникальности
    return render_template('index.html', get_current_language=get_current_language, cache_version=timestamp)

def analyze_symbol(symbol, force_update=False):
    """Анализирует отдельный символ"""
    clean_symbol = symbol.replace('USDT', '')
    analysis = determine_trend_and_position(clean_symbol, force_update)
    if analysis:
        return {
            'symbol': symbol,
            'trend_analysis': analysis
        }
    return None

def analyze_positions_parallel(positions, max_workers=10):
    """Параллельный анализ позиций"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        analyzed = list(filter(None, executor.map(
            lambda p: analyze_symbol(p['symbol']),
            positions
        )))
        
        for position, analysis in zip(positions, analyzed):
            if analysis and analysis['symbol'] == position['symbol']:
                position['trend_analysis'] = analysis['trend_analysis']
                
        return [p for p in positions if 'trend_analysis' in p]

def analyze_pairs_parallel(pairs, max_workers=10):
    """Параллельный анализ пар"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(filter(None, executor.map(analyze_symbol, pairs)))

@app.route('/get_positions')
def get_positions():
    pnl_threshold = float(request.args.get('pnl_threshold', DEFAULTS.PNL_THRESHOLD))
    
    all_available_pairs = []  # Больше не используется
    
    all_positions = (positions_data['high_profitable'] +
                    positions_data['profitable'] +
                    positions_data['losing'])

    # Виртуальные позиции ПРИИ — получаем до проверки «пусто», чтобы показывать только виртуальные
    virtual_positions = []
    try:
        from bots_modules.fullai_adaptive import get_virtual_positions_for_api, is_adaptive_enabled
        if is_adaptive_enabled() and current_exchange:
            vp_list = get_virtual_positions_for_api()
            for vp in vp_list:
                sym = (vp.get('symbol') or '').strip()
                if not sym:
                    continue
                sym_usdt = sym if sym.endswith('USDT') else (sym + 'USDT')
                entry = float(vp.get('entry_price') or 0)
                direction = (vp.get('direction') or 'LONG').upper()
                try:
                    ticker = current_exchange.get_ticker(sym_usdt)
                    current_price = float(ticker.get('last_price') or ticker.get('mark_price') or 0) if ticker else 0
                except Exception:
                    current_price = 0
                if entry and current_price:
                    if direction == 'LONG':
                        pnl_percent = (current_price - entry) / entry * 100
                    else:
                        pnl_percent = (entry - current_price) / entry * 100
                else:
                    pnl_percent = 0.0
                virtual_positions.append({
                    'symbol': sym_usdt,
                    'side': 'Long' if direction == 'LONG' else 'Short',
                    'pnl': 0,
                    'roi': round(pnl_percent, 2),
                    'unrealized_pnl_percent': round(pnl_percent, 2),
                    'max_profit': 0,
                    'max_loss': 0,
                    'size': 0,
                    'qty': 0,
                    'quantity': 0,
                    'entry_price': entry,
                    'is_virtual': True,
                })
    except Exception:
        pass

    # Fallback: если позиций нет — пробуем взять с Bots-сервиса (app и bots в разных процессах)
    if not all_positions and not virtual_positions:
        try:
            bots_url = getattr(request, 'headers', None) and request.headers.get('X-Bots-Service-URL') or 'http://127.0.0.1:5001'
            resp = requests.get(f'{bots_url}/api/bots/positions-for-app', timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('success') and data.get('total_trades', 0) > 0:
                    api_logger = logging.getLogger('app')
                    api_logger.info(f"[POSITIONS] Fallback: {data['total_trades']} позиций с Bots-сервиса")
                    hp, pf, ls = data.get('high_profitable', []), data.get('profitable', []), data.get('losing', [])
                    all_positions = hp + pf + ls
                    positions_data['high_profitable'] = hp
                    positions_data['profitable'] = pf
                    positions_data['losing'] = ls
                    positions_data['stats'] = data.get('stats', {})
                    positions_data['rapid_growth'] = data.get('rapid_growth', [])
                    positions_data['total_trades'] = data.get('total_trades', 0)
                    positions_data['last_update'] = time.strftime('%Y-%m-%d %H:%M:%S')
                    save_positions_data(positions_data)
        except Exception as fb_err:
            logging.getLogger('app').debug(f"[POSITIONS] Fallback Bots: {fb_err}")

    if not all_positions and not virtual_positions:
        try:
            wallet_data = current_exchange.get_wallet_balance()
        except Exception as e:
            api_logger = logging.getLogger('app')
            api_logger.error(f"[API] Error getting wallet data: {str(e)}")
            wallet_data = {
                'total_balance': 0,
                'available_balance': 0,
                'realized_pnl': 0
            }
        return jsonify({
            'high_profitable': [],
            'profitable': [],
            'losing': [],
            'stats': {
                'total_pnl': 0,
                'total_profit': 0,
                'total_loss': 0,
                'high_profitable_count': 0,
                'profitable_count': 0,
                'losing_count': 0,
                'top_profitable': [],
                'top_losing': [],
                'total_trades': 0
            },
            'rapid_growth': [],
            'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
            'growth_multiplier': GROWTH_MULTIPLIER,
            'all_pairs': [],
            'wallet_data': {
                'total_balance': wallet_data['total_balance'],
                'available_balance': wallet_data['available_balance'],
                'realized_pnl': wallet_data['realized_pnl']
            }
        })

    all_positions_with_virtual = all_positions + virtual_positions
    active_position_symbols = set(p['symbol'] for p in all_positions_with_virtual)
    
    # Фильтруем доступные пары
    available_pairs = [pair for pair in all_available_pairs if pair not in active_position_symbols]
    
    # Распределяем позиции по категориям (виртуальные не участвуют в total_profit/total_loss/total_trades)
    high_profitable = []
    profitable = []
    losing = []
    total_profit = 0
    total_loss = 0

    def _pnl_sort_key(pos):
        if pos.get('is_virtual'):
            return pos.get('unrealized_pnl_percent', 0)
        return pos.get('pnl', 0)

    for position in all_positions_with_virtual:
        is_virtual = position.get('is_virtual', False)
        pnl = position.get('pnl', 0)
        if is_virtual:
            pnl_for_cat = position.get('unrealized_pnl_percent', 0)
            if pnl_for_cat >= 0:
                if pnl_for_cat >= 5:
                    high_profitable.append(position)
                else:
                    profitable.append(position)
            else:
                losing.append(position)
            continue
        if pnl > 0:
            if pnl >= pnl_threshold:
                high_profitable.append(position)
            else:
                profitable.append(position)
            total_profit += pnl
        elif pnl < 0:
            losing.append(position)
            total_loss += pnl

    high_profitable.sort(key=_pnl_sort_key, reverse=True)
    profitable.sort(key=_pnl_sort_key, reverse=True)
    losing.sort(key=_pnl_sort_key)

    # TOP-3 и total_trades — только по реальным позициям
    real_profitable = [p for p in high_profitable + profitable if not p.get('is_virtual')]
    real_losing = [p for p in losing if not p.get('is_virtual')]
    real_profitable.sort(key=lambda x: x['pnl'], reverse=True)
    top_profitable = real_profitable[:3] if real_profitable else []
    top_losing = real_losing[:3] if real_losing else []

    stats = {
        'total_pnl': total_profit + total_loss,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'high_profitable_count': len(high_profitable),
        'profitable_count': len(profitable),
        'losing_count': len(losing),
        'top_profitable': top_profitable,
        'top_losing': top_losing,
        'total_trades': len(all_positions),
    }
    
    # Получаем данные аккаунта
    try:
        wallet_data = current_exchange.get_wallet_balance()
    except Exception as e:
        api_logger = logging.getLogger('app')
        api_logger.error(f"[API] Error getting wallet data: {str(e)}")
        wallet_data = {
            'total_balance': 0,
            'available_balance': 0,
            'realized_pnl': 0
        }
    
    return jsonify({
        'high_profitable': high_profitable,
        'profitable': profitable,
        'losing': losing,
        'stats': stats,
        'rapid_growth': positions_data['rapid_growth'],
        'last_update': positions_data['last_update'],
        'growth_multiplier': GROWTH_MULTIPLIER,
        'all_pairs': available_pairs,
        'wallet_data': {
            'total_balance': wallet_data['total_balance'],
            'available_balance': wallet_data['available_balance'],
            'realized_pnl': wallet_data['realized_pnl']
        }
    })

@app.route('/api/positions')
def api_positions():
    """API endpoint for positions - redirects to get_positions"""
    return get_positions()

@app.route('/api/balance')
def get_balance():
    """Получение баланса"""
    try:
        if not current_exchange:
            return jsonify({'error': 'Exchange not initialized'}), 500
        
        wallet_data = current_exchange.get_wallet_balance()
        return jsonify({
            'success': True,
            'balance': wallet_data['total_balance'],
            'available_balance': wallet_data['available_balance'],
            'realized_pnl': wallet_data['realized_pnl']
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/closed_pnl')
def get_closed_pnl():
    """Получение закрытых позиций из БД"""
    try:
        sort_by = request.args.get('sort', 'time')
        period = request.args.get('period', 'all')  # all, day, week, month, half_year, year, custom
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)
        
        api_logger = logging.getLogger('app')
        api_logger.info(f"[API] Getting closed PNL from DB, sort by: {sort_by}, period: {period}")
        
        # Получаем баланс и PNL
        wallet_data = current_exchange.get_wallet_balance()
        
        # Получаем закрытые позиции из БД
        db = get_app_db()
        closed_pnl = db.get_closed_pnl(
            sort_by=sort_by,
            period=period,
            start_date=start_date,
            end_date=end_date,
            exchange=ACTIVE_EXCHANGE
        )
        api_logger.info(f"[API] Found {len(closed_pnl)} closed positions in DB")
        
        return jsonify({
            'success': True,
            'closed_pnl': closed_pnl,
            'wallet_data': {
                'total_balance': wallet_data['total_balance'],
                'available_balance': wallet_data['available_balance'],
                'realized_pnl': wallet_data['realized_pnl']
            }
        })
    except Exception as e:
        api_logger = logging.getLogger('app')
        api_logger.error(f"[API] Error getting closed PNL from DB: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500





def calculate_statistics(positions):
    """Calculates statistics for positions"""
    total_profit = 0
    total_loss = 0
    high_profitable = []
    profitable = []
    losing = []

    for position in positions:
        pnl = position['pnl']
        if pnl > 0:
            if pnl >= DEFAULTS.PNL_THRESHOLD:
                high_profitable.append(position)
            else:
                profitable.append(position)
            total_profit += pnl
        else:
            losing.append(position)
            total_loss += pnl

    return {
        'total_pnl': total_profit + total_loss,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'total_trades': len(positions),
        'profitable_count': len(high_profitable) + len(profitable),
        'losing_count': len(losing),
        'top_profitable': sorted(high_profitable + profitable, key=lambda x: x['pnl'], reverse=True)[:3],
        'top_losing': sorted(losing, key=lambda x: x['pnl'])[:3]
    }

def send_daily_report():
    """Отправка ежедневного отчета"""
    while True:
        now = datetime.now()
        if now.strftime('%H:%M') == TELEGRAM_NOTIFY['DAILY_REPORT_TIME']:
            positions, _ = current_exchange.get_positions()
            if positions:
                stats = calculate_statistics(positions)
                telegram.send_daily_report(stats)
        time.sleep(60)  # Проверяем каждую минуту

# Глобальная переменная для хранения текущей биржи
current_exchange = None

def init_exchange():
    """Инициализация биржи"""
    app_logger = logging.getLogger('app')
    try:
        app_logger.info(f"[INIT] Получение конфигурации для {ACTIVE_EXCHANGE}...")
        exchange_config = dict(EXCHANGES[ACTIVE_EXCHANGE])
        # Bybit: режим маржи из SystemConfig (UI) или из keys
        if ACTIVE_EXCHANGE == 'BYBIT':
            try:
                from bot_engine.config_loader import SystemConfig
                exchange_config['margin_mode'] = getattr(SystemConfig, 'BYBIT_MARGIN_MODE', None) or exchange_config.get('margin_mode', 'auto')
            except Exception:
                exchange_config['margin_mode'] = exchange_config.get('margin_mode', 'auto')
        # БЕЗОПАСНОСТЬ: НЕ выводим конфигурацию с ключами!
        safe_config = {k: ('***HIDDEN***' if k in ['api_key', 'api_secret', 'passphrase'] else v) 
                       for k, v in exchange_config.items()}
        app_logger.info(f"[INIT] Конфигурация получена: {safe_config}")
        
        app_logger.info(f"[INIT] Создание экземпляра биржи {ACTIVE_EXCHANGE}...")
        exchange = ExchangeFactory.create_exchange(
            ACTIVE_EXCHANGE,
            exchange_config['api_key'],
            exchange_config['api_secret'],
            exchange_config.get('passphrase'),  # Добавляем passphrase для OKX
            exchange_config=exchange_config
        )
        
        app_logger.info(f"[INIT] ✅ Биржа {ACTIVE_EXCHANGE} успешно создана")
        return exchange
    except Exception as e:
        app_logger.error(f"[INIT] ❌ Ошибка создания биржи {ACTIVE_EXCHANGE}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/api/exchanges', methods=['GET'])
def get_exchanges():
    """Получение списка доступных бирж"""
    exchanges = [{
        'name': name,
        'enabled': config['enabled'],
        'active': name == ACTIVE_EXCHANGE
    } for name, config in EXCHANGES.items()]
    return jsonify({'exchanges': exchanges})

@app.route('/api/exchange', methods=['POST'])
def switch_exchange():
    """Переключение активной биржи"""
    global current_exchange
    try:
        data = request.get_json()
        exchange_name = data.get('exchange')
        
        if exchange_name not in EXCHANGES:
            return jsonify({'error': 'Exchange not found'}), 404
            
        if not EXCHANGES[exchange_name]['enabled']:
            return jsonify({'error': 'Exchange is disabled'}), 400
        
        try:
            # Создаем новый экземпляр биржи для проверки подключения
            exchange_config = dict(EXCHANGES[exchange_name])
            if exchange_name == 'BYBIT':
                try:
                    from bot_engine.config_loader import SystemConfig
                    exchange_config['margin_mode'] = getattr(SystemConfig, 'BYBIT_MARGIN_MODE', None) or exchange_config.get('margin_mode', 'auto')
                except Exception:
                    exchange_config['margin_mode'] = exchange_config.get('margin_mode', 'auto')
            new_exchange = ExchangeFactory.create_exchange(
                exchange_name,
                exchange_config['api_key'],
                exchange_config['api_secret'],
                exchange_config.get('passphrase'),  # Добавляем passphrase для OKX
                exchange_config=exchange_config
            )
            
            # Пробуем получить позиции для проверки работоспособности
            positions, _ = new_exchange.get_positions()
            
            # Если все хорошо, обновляем конфигурацию
            with open('configs/app_config.py', 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # Обновляем активную биржу в конфиге
            new_config = config_content.replace(
                f"ACTIVE_EXCHANGE = '{ACTIVE_EXCHANGE}'",
                f"ACTIVE_EXCHANGE = '{exchange_name}'"
            )
            
            with open('configs/app_config.py', 'w', encoding='utf-8') as f:
                f.write(new_config)
            
            # Обновляем текущую биржу
            current_exchange = new_exchange
            
            return jsonify({
                'success': True,
                'message': f'Switched to {exchange_name}'
            })
            
        except Exception as e:
            app_logger = logging.getLogger('app')
            app_logger.error(f"Error testing new exchange connection: {str(e)}")
            return jsonify({
                'error': f'Failed to connect to {exchange_name}: {str(e)}'
            }), 500
            
    except Exception as e:
        app_logger = logging.getLogger('app')
        app_logger.error(f"Error in switch_exchange: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Инициализируем биржу при запуске
# Используем logger вместо print для правильной фильтрации
app_logger = logging.getLogger('app')
app_logger.info(f"[INIT] Инициализация биржи {ACTIVE_EXCHANGE}...")
DEMO_MODE = not check_api_keys()
if DEMO_MODE:
    app_logger.warning("[INIT] ⚠️ DEMO режим: API ключи не настроены, торговля отключена")
    current_exchange = DemoExchange()
else:
    current_exchange = init_exchange()
    if not current_exchange:
        app_logger.error("[INIT] ❌ Не удалось инициализировать биржу")
        sys.exit(1)
    else:
        app_logger.info(f"[INIT] ✅ Биржа {ACTIVE_EXCHANGE} успешно инициализирована")

# Убираем инициализацию менеджера ботов - теперь он в отдельном сервисе
# bot_manager = BotManager(exchange)

# Добавляем функцию clean_symbol если она где-то используется
def clean_symbol(symbol):
    """Удаляет 'USDT' из названия символа"""
    return symbol.replace('USDT', '')

@app.route('/api/set_language', methods=['POST'])
def set_language():
    data = request.get_json()
    language = data.get('language', 'en')
    app_logger = logging.getLogger('app')
    app_logger.info(f"Setting language to: {language}")
    save_language(language)
    telegram.set_language(language)
    return jsonify({'status': 'success', 'language': language})

@app.route('/api/ticker/<symbol>')
def get_ticker(symbol):
    ticker_logger = logging.getLogger('app')
    try:
        ticker_logger.info(f"[TICKER] Getting ticker for {symbol}...")
        
        # Проверяем инициализацию биржи
        if not current_exchange:
            ticker_logger.error("[TICKER] Exchange not initialized")
            return jsonify({'error': 'Exchange not initialized'}), 500
            
        # Получаем данные тикера
        ticker_data = current_exchange.get_ticker(symbol)
        pass
        
        if ticker_data:
            ticker_logger.info(f"[TICKER] Successfully got ticker for {symbol}")
            return jsonify(ticker_data)
            
        ticker_logger.warning(f"[TICKER] No ticker data available for {symbol}")
        return jsonify({'error': 'No ticker data available'}), 404
        
    except Exception as e:
        ticker_logger.error(f"[TICKER] Error getting ticker for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/close_position', methods=['POST'])
def close_position():
    """Закрытие позиции"""
    try:
        data = request.json
        if not data or not all(k in data for k in ['symbol', 'size', 'side']):
            return jsonify({
                'success': False,
                'message': 'Не указаны обязательные параметры (symbol, size, side)'
            }), 400

        api_logger = logging.getLogger('app')
        api_logger.info(f"[API] Closing position: {data}")
        result = current_exchange.close_position(
            symbol=data['symbol'],
            size=float(data['size']),
            side=data['side'],
            order_type=data.get('order_type', 'Limit')  # По умолчанию используем Limit для обратной совместимости
        )
        
        api_logger.info(f"[API] Close position result: {result}")
        return jsonify(result)
        
    except Exception as e:
        api_logger = logging.getLogger('app')
        api_logger.error(f"[API] Error closing position: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Ошибка при закрытии позиции: {str(e)}'
        }), 500

@app.route('/api/get_language')
def get_language():
    """Получение текущего языка"""
    try:
        current_lang = get_current_language()
        return jsonify({
            'success': True,
            'language': current_lang
        })
    except Exception as e:
        app_logger = logging.getLogger('app')
        app_logger.error(f"Error getting language: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



@app.route('/api/blacklist', methods=['POST'])
def manage_blacklist():
    """Управление черным списком"""
    try:
        data = request.get_json()
        action = data.get('action')
        symbol = data.get('symbol')
        
        if not action or not symbol:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters'
            }), 400
            
        blacklist_file = 'data/blacklist.json'
        os.makedirs('data', exist_ok=True)
        
        try:
            with open(blacklist_file, 'r') as f:
                blacklist = json.load(f)
        except:
            blacklist = []
            
        if action == 'add':
            if symbol not in blacklist:
                blacklist.append(symbol)
        elif action == 'remove':
            if symbol in blacklist:
                blacklist.remove(symbol)
                
        with open(blacklist_file, 'w') as f:
            json.dump(blacklist, f)
            
        return jsonify({
            'success': True,
            'blacklist': blacklist
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Добавляем глобальные переменные для кэширования
ticker_analysis_cache = {}
CACHE_TIMEOUT = 300  # 5 минут

def determine_trend_and_position(symbol, force_update=False):
    """Определяет тренд и позицию цены на графике с кэшированием"""
    global ticker_analysis_cache
    current_time = time.time()
    
    # Проверяем кэш, если не требуется принудительное обновление
    if not force_update and symbol in ticker_analysis_cache:
        cached_data = ticker_analysis_cache[symbol]
        if current_time - cached_data['timestamp'] < CACHE_TIMEOUT:
            return cached_data['data']
    
    try:
        # ✅ ЧИТАЕМ НАПРЯМУЮ ИЗ ФАЙЛА (не требует запущенного bots.py)
        data = get_candles_from_file(symbol, timeframe='1d', period_days=30)
        if not data or not data.get('success'):
            return None
            
        candles = data.get('data', {}).get('candles', [])
        if not candles:
            return None
            
        # Находим минимальную и максимальную цену
        min_price = min(float(candle['low']) for candle in candles)
        max_price = max(float(candle['high']) for candle in candles)
        current_price = float(candles[-1]['close'])
        
        # Определяем позицию цены в процентах от диапазона
        price_range = max_price - min_price
        if price_range == 0:
            return None
            
        position_percent = ((current_price - min_price) / price_range) * 100
        
        # Определяем тренд
        period = 14  # период для определения тренда
        if len(candles) < period:
            return None
            
        recent_prices = [float(candle['close']) for candle in candles[-period:]]
        first_half = sum(recent_prices[:period//2]) / (period//2)
        second_half = sum(recent_prices[period//2:]) / (period//2)
        
        if second_half > first_half * 1.02:  # 2% разница для определения роста
            trend = 'рост'
        elif first_half > second_half * 1.02:  # 2% разница для определения падения
            trend = 'падение'
        else:
            trend = 'флэт'
            
        # Определяем состояние тикера
        state = None
        if 0 <= position_percent <= 10:
            if trend in ['флэт', 'рост']:
                state = 'дно рынка'
            else:
                state = 'падение'
        elif 10 < position_percent <= 40:
            state = trend
        elif 40 < position_percent <= 60:
            state = trend
        elif 60 < position_percent <= 90:
            if trend == 'флэт':
                state = 'диапазон распродажи'
            elif trend == 'падение':
                state = 'диапазон падения'
            else:
                state = 'рост'
        elif 90 < position_percent <= 100:
            if trend == 'флэт':
                state = 'хай рынка'
            elif trend == 'падение':
                state = 'падение'
            else:
                state = 'диапазон распродажи'
        
        result = {
            'trend': trend,
            'position_percent': position_percent,
            'state': state
        }
        
        # Сохраняем результат в кэш
        ticker_analysis_cache[symbol] = {
            'data': result,
            'timestamp': current_time
        }
        
        return result
        
    except Exception:
        return None

def clear_old_cache():
    """Очищает устаревшие данные из кэша"""
    global ticker_analysis_cache
    current_time = time.time()
    expired_symbols = [
        symbol for symbol, data in ticker_analysis_cache.items()
        if current_time - data['timestamp'] >= CACHE_TIMEOUT
    ]
    for symbol in expired_symbols:
        del ticker_analysis_cache[symbol]

@app.route('/api/ticker_analysis/<symbol>')
def get_ticker_analysis(symbol):
    """Получение анализа тикера (тренд и позиция на графике)"""
    try:
        force_update = request.args.get('force_update', '0') == '1'
        analysis = determine_trend_and_position(symbol, force_update)
        if analysis:
            return jsonify({
                'success': True,
                'data': analysis,
                'cached': not force_update and symbol in ticker_analysis_cache
            })
        return jsonify({
            'success': False,
            'error': 'Could not analyze ticker'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def background_cache_cleanup():
    """Фоновая очистка кэша и памяти (GC + PyTorch/CUDA при наличии)."""
    while True:
        try:
            clear_old_cache()
        except Exception as e:
            app_logger = logging.getLogger('app')
            app_logger.error(f"Error in cache cleanup: {str(e)}")
        try:
            from utils.memory_utils import force_collect_full
            force_collect_full()
        except Exception:
            pass
        time.sleep(60)  # Проверяем каждую минуту

# Кэш для хранения данных свечей
candles_cache = {}
CACHE_TIMEOUT = 300  # 5 минут

@app.route('/api/candles/<symbol>')
def get_candles(symbol):
    """Получение свечей для расчета тренда на клиенте"""
    current_time = time.time()
    
    # Проверяем кэш
    if symbol in candles_cache:
        cached_data = candles_cache[symbol]
        if current_time - cached_data['timestamp'] < CACHE_TIMEOUT:
            return jsonify(cached_data['data'])
    
    try:
        # ✅ ЧИТАЕМ НАПРЯМУЮ ИЗ ФАЙЛА (не требует запущенного bots.py)
        data = get_candles_from_file(symbol, timeframe='1d', period_days=30)
        if not data or not data.get('success'):
            return jsonify({'success': False, 'error': 'Не удалось получить данные из файла кэша'})
        
        # Сохраняем в кэш
        candles_cache[symbol] = {
            'data': data,
            'timestamp': current_time
        }
        
        return jsonify(data)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def clear_old_cache():
    """Очистка устаревших данных из кэша"""
    current_time = time.time()
    expired_symbols = [
        symbol for symbol, data in candles_cache.items()
        if current_time - data['timestamp'] >= CACHE_TIMEOUT
    ]
    for symbol in expired_symbols:
        del candles_cache[symbol]

def background_cache_cleanup():
    """Фоновая очистка кэша и памяти (GC + PyTorch/CUDA при наличии)."""
    while True:
        try:
            clear_old_cache()
        except Exception:
            pass
        try:
            from utils.memory_utils import force_collect_full
            force_collect_full()
        except Exception:
            pass
        time.sleep(60)  # Проверяем каждую минуту

# Прокси для API endpoints ботов (перенаправляем на внешний сервис)
def get_candles_from_file(symbol, timeframe=None, period_days=None):
    """
    Получает свечи из файла/кэша для символа.
    Если timeframe не указан, используется текущий таймфрейм из конфига.
    """
    from bot_engine.config_loader import get_current_timeframe
    if timeframe is None:
        timeframe = get_current_timeframe()
    """Читает свечи напрямую из БД (не требует запущенного bots.py)"""
    try:
        from bot_engine.storage import get_candles_for_symbol
        
        # Читаем из БД
        cached_data = get_candles_for_symbol(symbol)
        
        if not cached_data:
            return {'success': False, 'error': f'Свечи для {symbol} не найдены в БД кэша'}
        
        candles_6h = cached_data.get('candles', [])
        
        if not candles_6h:
            return {'success': False, 'error': f'Нет свечей в БД кэша для {symbol}'}
        
        # Конвертируем свечи в нужный таймфрейм
        if timeframe == '1d':
            # Конвертируем 6h свечи в дневные
            daily_candles = []
            current_day = None
            current_candle = None
            
            for candle in candles_6h:
                candle_time = datetime.fromtimestamp(int(candle['timestamp']) / 1000)
                day_key = candle_time.date()
                
                if current_day != day_key:
                    # Сохраняем предыдущую дневную свечу
                    if current_candle:
                        daily_candles.append(current_candle)
                    
                    # Начинаем новую дневную свечу
                    current_day = day_key
                    current_candle = {
                        'timestamp': candle['timestamp'],
                        'open': float(candle['open']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'close': float(candle['close']),
                        'volume': float(candle.get('volume', 0))
                    }
                else:
                    # Обновляем текущую дневную свечу
                    if current_candle:
                        current_candle['high'] = max(current_candle['high'], float(candle['high']))
                        current_candle['low'] = min(current_candle['low'], float(candle['low']))
                        current_candle['close'] = float(candle['close'])
                        current_candle['volume'] += float(candle.get('volume', 0))
            
            # Добавляем последнюю свечу
            if current_candle:
                daily_candles.append(current_candle)
            
            candles = daily_candles
        # Получаем текущий таймфрейм из конфига
        from bot_engine.config_loader import get_current_timeframe
        current_timeframe = get_current_timeframe()
        
        if timeframe == current_timeframe:
            # Используем свечи текущего таймфрейма как есть
            candles = candles_6h
        elif timeframe == '1d':
            # Конвертируем свечи текущего таймфрейма в дневные (если нужно)
            # Пока возвращаем свечи текущего таймфрейма
            candles = candles_6h
        else:
            # Для других таймфреймов возвращаем свечи текущего таймфрейма
            candles = candles_6h
        
        # Ограничиваем количество свечей по периоду (если указан)
        if period_days:
            try:
                period_days = int(period_days)
                # Для дневных свечей: period_days свечей
                # Для 6h свечей: period_days * 4 свечей (4 свечи в день)
                if timeframe == '1d':
                    candles = candles[-period_days:] if len(candles) > period_days else candles
                else:
                    candles = candles[-period_days * 4:] if len(candles) > period_days * 4 else candles
            except:
                pass
        
        # Форматируем ответ в том же формате, что и get_chart_data
        formatted_candles = []
        for candle in candles:
            formatted_candles.append({
                'timestamp': int(candle['timestamp']),
                'open': str(candle['open']),
                'high': str(candle['high']),
                'low': str(candle['low']),
                'close': str(candle['close']),
                'volume': str(candle.get('volume', 0))
            })
        
        return {
            'success': True,
            'data': {
                'candles': formatted_candles,
                'timeframe': timeframe,
                'count': len(formatted_candles)
            },
            'source': 'file'  # Указываем, что данные из файла
        }
        
    except Exception as e:
        import traceback
        logging.getLogger('app').error(f"❌ Ошибка чтения свечей из файла для {symbol}: {e}\n{traceback.format_exc()}")
        return {'success': False, 'error': str(e)}

def call_bots_service(endpoint, method='GET', data=None, timeout=10):
    """Универсальная функция для вызова API сервиса ботов"""
    # Определяем URL сервиса ботов динамически (доступен в обработчиках Flask)
    bots_service_url = request.headers.get('X-Bots-Service-URL', 'http://127.0.0.1:5001')
    
    try:
        url = f"{bots_service_url}{endpoint}"
        
        if method == 'GET':
            response = requests.get(url, timeout=timeout)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=timeout)
        else:
            return {'success': False, 'error': f'Unsupported method: {method}'}
        
        def _safe_json():
            try:
                return response.json()
            except ValueError:
                return None
        
        payload = _safe_json()
        if response.status_code == 200:
            if isinstance(payload, dict):
                payload.setdefault('success', True)
                payload['status_code'] = response.status_code
                return payload
            return {
                'success': True,
                'data': payload if payload is not None else response.text,
                'status_code': response.status_code
            }
        else:
            # Пытаемся извлечь человекочитаемое сообщение из ответа сервиса ботов
            if isinstance(payload, dict):
                error_message = payload.get('error') or payload.get('message')
                details = payload
            else:
                error_message = response.text.strip() or None
                details = response.text
            
            if not error_message:
                error_message = f'Bots service returned status {response.status_code}'
            
            return {
                'success': False,
                'error': error_message,
                'details': details,
                'status_code': response.status_code
            }
            
    except requests.exceptions.ConnectionError:
        return {
            'success': False, 
            'error': 'Сервис ботов недоступен. Запустите в отдельном терминале: python bots.py',
            'service_url': bots_service_url,
            'instructions': 'Откройте новый терминал и выполните: python bots.py',
            'status_code': 503
        }
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': f'Таймаут сервиса ботов ({timeout} сек). Сервис не успел ответить — возможно перегрузка или блокировка. Проверьте, что bots.py запущен и работает.',
            'status_code': 504
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error calling bots service: {str(e)}',
            'status_code': 500
        }



@app.route('/api/bots/list', methods=['GET'])
def get_bots_list():
    """Получение списка всех ботов (прокси к сервису ботов)"""
    result = call_bots_service('/api/bots/list')
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code

@app.route('/api/bots/control', methods=['POST'])
def control_bot():
    """Управление ботом (прокси к сервису ботов)"""
    data = request.get_json()
    result = call_bots_service('/api/bots/control', method='POST', data=data)
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code

@app.route('/api/bots/config', methods=['GET', 'POST'])
def bots_config():
    """Получение и обновление конфигурации Auto Bot (прокси к сервису ботов)"""
    if request.method == 'GET':
        result = call_bots_service('/api/bots/config')
    else:
        data = request.get_json()
        result = call_bots_service('/api/bots/config', method='POST', data=data)
    
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code

@app.route('/api/bots/status', methods=['GET'])
def get_bots_status():
    """Получение общего статуса ботов (прокси к сервису ботов)"""
    result = call_bots_service('/api/bots/status')
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code

@app.route('/api/bots/pairs', methods=['GET'])
def get_bots_pairs():
    """Получение списка доступных торговых пар (прокси к сервису ботов)"""
    result = call_bots_service('/api/bots/pairs')
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code

@app.route('/api/status', methods=['GET'])
def api_status_proxy():
    """Прокси /api/status для проверки сервиса ботов (фронт при порте 5000 дергает этот URL)."""
    result = call_bots_service('/api/status', timeout=5)
    status_code = result.get('status_code', 200 if result.get('status') == 'online' else 503)
    return jsonify(result), status_code


@app.route('/api/bots/health', methods=['GET'])
def get_bots_health():
    """Проверка состояния сервиса ботов"""
    result = call_bots_service('/health', timeout=5)
    status_code = result.get('status_code', 200 if result.get('status') == 'ok' else 503)
    return jsonify(result), status_code

@app.route('/api/bots/status/<symbol>', methods=['GET'])
def get_bot_status(symbol):
    """Получить статус конкретного бота (прокси к сервису ботов)"""
    result = call_bots_service(f'/api/bots/status/{symbol}')
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code

@app.route('/api/bots/create', methods=['POST'])
def create_bot():
    """Создать бота (прокси к сервису ботов)"""
    data = request.get_json()
    result = call_bots_service('/api/bots/create', method='POST', data=data)
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code

@app.route('/api/bots/auto-bot', methods=['GET', 'POST'])
def auto_bot_proxy():
    """Получить конфиг Auto Bot (GET) или обновить (POST) — прокси к сервису ботов."""
    cfg_timeout = 60  # Таймаут: сохранение конфига может быть долгим
    if request.method == 'GET':
        result = call_bots_service('/api/bots/auto-bot', method='GET', timeout=cfg_timeout)
    else:
        data = request.get_json()
        result = call_bots_service('/api/bots/auto-bot', method='POST', data=data, timeout=cfg_timeout)
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code

@app.route('/api/bots/account-info', methods=['GET'])
def get_account_info():
    """Получить информацию об аккаунте (прокси к сервису ботов)"""
    result = call_bots_service('/api/bots/account-info')
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code

@app.route('/api/bots/sync-positions', methods=['GET', 'POST'])
def sync_positions():
    """Синхронизировать позиции (прокси к сервису ботов, работает с GET и POST)"""
    method = request.method
    result = call_bots_service('/api/bots/sync-positions', method=method)
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code

@app.route('/api/bots/coins-with-rsi', methods=['GET'])
def get_coins_with_rsi():
    """Получить монеты с RSI данными (прокси к сервису ботов)"""
    result = call_bots_service('/api/bots/coins-with-rsi')
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code

@app.route('/api/bots/individual-settings/<symbol>', methods=['GET', 'POST'])
def individual_settings(symbol):
    """Индивидуальные настройки бота (прокси к сервису ботов)"""
    if request.method == 'GET':
        result = call_bots_service(f'/api/bots/individual-settings/{symbol}')
    else:
        data = request.get_json()
        result = call_bots_service(f'/api/bots/individual-settings/{symbol}', method='POST', data=data)
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code

@app.route('/api/bots/individual-settings/<symbol>/copy-to-all', methods=['POST'])
def copy_individual_settings(symbol):
    """Копировать индивидуальные настройки на все боты (прокси к сервису ботов)"""
    result = call_bots_service(f'/api/bots/individual-settings/{symbol}/copy-to-all', method='POST')
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code

@app.route('/api/bots/individual-settings/<symbol>/learn-exit-scam', methods=['POST'])
def learn_exit_scam(symbol):
    """Подбор ExitScam по истории свечей для монеты (прокси к сервису ботов)"""
    data = request.get_json(silent=True) or {}
    result = call_bots_service(
        f'/api/bots/individual-settings/{symbol}/learn-exit-scam',
        method='POST',
        data=data
    )
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code

@app.route('/api/bots/individual-settings/learn-exit-scam-all', methods=['POST'])
def learn_exit_scam_all():
    """Расчёт ExitScam по истории для всех монет (прокси к сервису ботов)"""
    data = request.get_json(silent=True) or {}
    result = call_bots_service(
        '/api/bots/individual-settings/learn-exit-scam-all',
        method='POST',
        data=data
    )
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code


@app.route('/api/bots/individual-settings/reset-exit-scam-all', methods=['POST'])
def reset_exit_scam_all():
    """Сброс ExitScam к общим настройкам для всех монет (прокси к сервису ботов)"""
    result = call_bots_service(
        '/api/bots/individual-settings/reset-exit-scam-all',
        method='POST',
        data={}
    )
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code


@app.route('/api/bots/start', methods=['POST'])
def start_bot():
    """Запустить бота (прокси к сервису ботов)"""
    data = request.get_json()
    result = call_bots_service('/api/bots/start', method='POST', data=data)
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code


@app.route('/api/bots/export-config', methods=['GET'])
def export_config():
    """Экспорт полного конфига InfoBot_Config_<TF>.json (прокси к сервису ботов)"""
    result = call_bots_service('/api/bots/export-config', method='GET', timeout=15)
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code


@app.route('/api/bots/system-config', methods=['GET', 'POST'])
def system_config():
    """Системные настройки (прокси к сервису ботов)"""
    cfg_timeout = 60
    if request.method == 'GET':
        result = call_bots_service('/api/bots/system-config', method='GET', timeout=cfg_timeout)
    else:
        data = request.get_json()
        result = call_bots_service('/api/bots/system-config', method='POST', data=data, timeout=cfg_timeout)
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code


@app.route('/api/ai/config', methods=['GET', 'POST'])
def ai_config():
    """Конфиг AI (прокси к сервису ботов)"""
    if request.method == 'GET':
        result = call_bots_service('/api/ai/config', method='GET')
    else:
        data = request.get_json()
        result = call_bots_service('/api/ai/config', method='POST', data=data)
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code


chart_render_lock = threading.Lock()

@app.route('/get_symbol_chart/<symbol>')
def get_symbol_chart(symbol):
    """Получение миниграфика RSI для символа (использует кэш из bots.py, таймфрейм из конфига)"""
    chart_logger = logging.getLogger('app')
    try:
        theme = request.args.get('theme', 'dark')
        # Получаем текущий таймфрейм для логирования
        from bot_engine.config_loader import get_current_timeframe
        current_timeframe = get_current_timeframe()
        chart_logger.info(f"[CHART] Getting RSI {current_timeframe} chart for {symbol} with theme {theme}")
        
        # ✅ Миниграфики используют ТЕ ЖЕ данные, что и боты: RSI и свечи из кэша bots.py
        # (обновляются continuous_data_loader раз в 1–3 мин; текущее RSI — из coins_rsi_data)
        rsi_response = call_bots_service(f'/api/bots/rsi-history/{symbol}')
        
        if not rsi_response or not rsi_response.get('success'):
            # Если нет в кэше, возвращаем ошибку (не запрашиваем с биржи)
            error_msg = rsi_response.get('error', 'RSI данные не найдены в кэше') if rsi_response else 'Сервис bots.py недоступен'
            chart_logger.warning(f"[CHART] {error_msg} for {symbol}")
            return jsonify({'error': error_msg}), 404
        
        # Получаем историю RSI из ответа
        rsi_values = rsi_response.get('rsi_history', [])
        if not rsi_values:
            chart_logger.warning(f"[CHART] Empty RSI history for {symbol}")
            return jsonify({'error': 'Empty RSI history'}), 404
        
        num_rsi_values = len(rsi_values)
        
        # ✅ КРИТИЧНО: Используем уже полученный current_timeframe для правильного расчета временных меток
        # Определяем интервал свечи в миллисекундах в зависимости от таймфрейма
        timeframe_ms = {
            '1m': 60 * 1000, '3m': 3 * 60 * 1000, '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000, '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000, '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000, '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000, '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000, '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000, '1M': 30 * 24 * 60 * 60 * 1000
        }
        candle_interval_ms = timeframe_ms.get(current_timeframe, 60 * 1000)  # По умолчанию 1m
        
        # Создаем временные метки на основе количества значений RSI
        # Каждое значение RSI соответствует одной свече текущего таймфрейма
        # Начинаем с текущего времени и идем назад
        current_timestamp = int(time.time() * 1000)
        times = []
        for i in range(num_rsi_values):
            # Каждое значение RSI отстоит на интервал свечи текущего таймфрейма от предыдущего
            # Последнее значение RSI - самое свежее (текущее время)
            ts = current_timestamp - (num_rsi_values - 1 - i) * candle_interval_ms
            times.append(datetime.fromtimestamp(ts / 1000))
            
        with chart_render_lock:
            # Создаем график RSI (с блокировкой для потокобезопасности)
            import matplotlib
            matplotlib.use('Agg')  # Используем неинтерактивный бэкенд
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Используем блокировку для всех операций matplotlib
            with matplotlib_lock:
                # Настраиваем стиль в зависимости от темы
                if theme == 'light':
                    plt.style.use('default')
                    bg_color = 'white'
                    rsi_color = '#1a1a1a'  # Темно-серая линия RSI на светлом фоне (более контрастная)
                    upper_color = '#e53935'  # Насыщенная красная граница 70
                    lower_color = '#43a047'  # Насыщенная зеленая граница 30
                    center_color = '#757575'  # Темно-серая линия 50 (хорошо видна на белом)
                else:
                    plt.style.use('dark_background')
                    bg_color = '#2d2d2d'
                    rsi_color = '#ffffff'  # Белая линия RSI на темном фоне
                    upper_color = '#ff9999'  # Светло-красная граница 70
                    lower_color = '#99ff99'  # Светло-зеленая граница 30
                    center_color = '#cccccc'  # Светло-серая линия 50
                
                # Создаем график с оптимальным размером для миниграфика
                fig, ax = plt.subplots(figsize=(4, 3), facecolor=bg_color)
                ax.set_facecolor(bg_color)
                
                # Рисуем линии границ (более заметные)
                ax.axhline(y=70, color=upper_color, linewidth=2, linestyle='-', alpha=0.8)
                ax.axhline(y=30, color=lower_color, linewidth=2, linestyle='-', alpha=0.8)
                ax.axhline(y=50, color=center_color, linewidth=2, linestyle='--', alpha=0.7, dashes=(5, 5))
                
                # Рисуем линию RSI
                ax.plot(times, rsi_values, color=rsi_color, linewidth=2.5, alpha=0.95)
                
                # Настраиваем ось Y для RSI (0-100)
                ax.set_ylim(0, 100)
                
                # Настраиваем внешний вид
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                # Конвертируем в base64
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                            facecolor=bg_color, edgecolor='none', pad_inches=0.1)
                buffer.seek(0)
                chart_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close(fig)
        
        # Получаем текущее значение RSI из ответа API (уже рассчитано в bots.py)
        current_rsi = rsi_response.get('current_rsi')
        
        chart_logger.info(f"[CHART] Successfully generated RSI {current_timeframe} chart for {symbol} (from cache)")
        return jsonify({
            'success': True,
            'chart': chart_data,
            'current_rsi': current_rsi
        })
        
    except Exception as e:
        chart_logger.error(f"[CHART] Error generating RSI chart for {symbol}: {str(e)}")
        import traceback
        chart_logger.error(f"[CHART] Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rsi_6h/<symbol>')
@app.route('/api/rsi/<symbol>')  # Новый универсальный endpoint
def get_rsi_6h(symbol):
    """Получение RSI данных для текущего таймфрейма за 56 свечей (неделя)"""
    rsi_logger = logging.getLogger('app')
    try:
        from bot_engine.config_loader import get_current_timeframe
        current_timeframe = get_current_timeframe()
        rsi_logger.info(f"[RSI {current_timeframe}] Getting RSI {current_timeframe} data for {symbol}")
        
        # Проверяем инициализацию биржи
        if not current_exchange:
            rsi_logger.error(f"[RSI {current_timeframe}] Exchange not initialized")
            return jsonify({'error': 'Exchange not initialized'}), 500
        
        # Импортируем функцию расчета RSI
        from bot_engine.utils.rsi_utils import calculate_rsi_history
        
        # ✅ ЧИТАЕМ НАПРЯМУЮ ИЗ ФАЙЛА (не требует запущенного bots.py)
        # Определяем период в днях в зависимости от таймфрейма
        # 56 свечей для разных таймфреймов = разное количество дней
        timeframe_hours = {'1m': 1/60, '3m': 3/60, '5m': 5/60, '15m': 15/60, '30m': 30/60, 
                          '1h': 1, '2h': 2, '4h': 4, '6h': 6, '8h': 8, '12h': 12, '1d': 24}
        hours_per_candle = timeframe_hours.get(current_timeframe, 6)
        period_days = max(1, int((56 * hours_per_candle) / 24))  # Минимум 1 день
        data = get_candles_from_file(symbol, timeframe=current_timeframe, period_days=period_days)
        if not data or not data.get('success'):
            rsi_logger.error(f"[RSI {current_timeframe}] Failed to get chart data from file for {symbol}")
            return jsonify({'error': 'Failed to get chart data from file'}), 500
            
        candles = data.get('data', {}).get('candles', [])
        if not candles:
            rsi_logger.warning(f"[RSI {current_timeframe}] No candles data for {symbol}")
            return jsonify({'error': 'No chart data available'}), 404
        
        # Берем последние 56 свечей
        candles = candles[-56:] if len(candles) >= 56 else candles
        
        if len(candles) < 15:  # Минимум для расчета RSI (период 14 + 1)
            rsi_logger.warning(f"[RSI {current_timeframe}] Not enough data for RSI calculation for {symbol}")
            return jsonify({'error': 'Not enough data for RSI calculation'}), 400
        
        # Извлекаем цены закрытия
        closes = [float(candle['close']) for candle in candles]
        
        # Рассчитываем историю RSI
        rsi_history = calculate_rsi_history(closes, period=14)
        
        if not rsi_history:
            rsi_logger.error(f"[RSI {current_timeframe}] Failed to calculate RSI for {symbol}")
            return jsonify({'error': 'Failed to calculate RSI'}), 500
        
        # Подготавливаем временные метки (берем только те, для которых есть RSI)
        # RSI начинается с индекса period (14), поэтому берем свечи с 14-го индекса
        timestamps = []
        for i in range(14, len(candles)):
            timestamp = candles[i].get('time') or candles[i].get('timestamp')
            if timestamp:
                # Преобразуем в миллисекунды если нужно
                if isinstance(timestamp, (int, float)):
                    if timestamp < 1e10:  # Если в секундах
                        timestamp = int(timestamp * 1000)
                    else:  # Уже в миллисекундах
                        timestamp = int(timestamp)
                    timestamps.append(timestamp)
        
        # Если временных меток меньше чем значений RSI, создаем их последовательно
        if len(timestamps) < len(rsi_history):
            last_timestamp = candles[-1].get('time') or candles[-1].get('timestamp')
            if isinstance(last_timestamp, (int, float)):
                if last_timestamp < 1e10:
                    last_timestamp = int(last_timestamp * 1000)
                else:
                    last_timestamp = int(last_timestamp)
            else:
                last_timestamp = int(time.time() * 1000)
            
            # ✅ КРИТИЧНО: Используем интервал свечи текущего таймфрейма вместо жестко зашитых 6 часов
            timeframe_ms = {
                '1m': 60 * 1000, '3m': 3 * 60 * 1000, '5m': 5 * 60 * 1000,
                '15m': 15 * 60 * 1000, '30m': 30 * 60 * 1000,
                '1h': 60 * 60 * 1000, '2h': 2 * 60 * 60 * 1000,
                '4h': 4 * 60 * 60 * 1000, '6h': 6 * 60 * 60 * 1000,
                '8h': 8 * 60 * 60 * 1000, '12h': 12 * 60 * 60 * 1000,
                '1d': 24 * 60 * 60 * 1000, '3d': 3 * 24 * 60 * 60 * 1000,
                '1w': 7 * 24 * 60 * 60 * 1000, '1M': 30 * 24 * 60 * 60 * 1000
            }
            candle_interval_ms = timeframe_ms.get(current_timeframe, 60 * 1000)  # По умолчанию 1m
            
            # Создаем временные метки с шагом интервала свечи текущего таймфрейма
            timestamps = []
            for i in range(len(rsi_history)):
                ts = last_timestamp - (len(rsi_history) - 1 - i) * candle_interval_ms
                timestamps.append(ts)
        
        # Берем только последние 56 значений
        if len(rsi_history) > 56:
            rsi_history = rsi_history[-56:]
            timestamps = timestamps[-56:]
        
        rsi_logger.info(f"[RSI {current_timeframe}] Successfully calculated RSI for {symbol}: {len(rsi_history)} values")
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'rsi_values': rsi_history,
            'timestamps': timestamps,
            'period': current_timeframe,
            'candles_count': len(rsi_history)
        })
        
    except Exception as e:
        import traceback
        try:
            from bot_engine.config_loader import get_current_timeframe
            _tf = get_current_timeframe()
        except Exception:
            _tf = '?'
        rsi_logger.error(f"[RSI {_tf}] Error calculating RSI for {symbol}: {str(e)}")
        rsi_logger.error(f"[RSI {_tf}] Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


def open_firewall_ports():
    """Открывает порты в брандмауэре при запуске (Windows/macOS/Linux)"""
    app_logger = logging.getLogger('app')
    try:
        import subprocess
        import platform
        
        app_logger.info("[APP] 🔥 Проверка открытия портов в брандмауэре...")
        
        system = platform.system()
        
        if system == 'Windows':
            # Windows Firewall
            for port in [5000, 5001]:
                service_name = "InfoBot Web UI" if port == 5000 else "InfoBot Bot Service"
                
                # Проверяем существует ли правило
                try:
                    check_result = subprocess.run(
                        ['netsh', 'advfirewall', 'firewall', 'show', 'rule', f'name={service_name}'],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace'
                    )
                    
                    # Если правило не найдено (код возврата != 0 или имя не в выводе)
                    rule_exists = (
                        check_result.returncode == 0 and 
                        check_result.stdout and 
                        service_name in check_result.stdout
                    )
                    
                    if not rule_exists:
                        app_logger.info(f"[APP] 🔥 Открываем порт {port}...")
                        add_result = subprocess.run([
                            'netsh', 'advfirewall', 'firewall', 'add', 'rule',
                            f'name={service_name}',
                            'dir=in',
                            'action=allow',
                            'protocol=TCP',
                            f'localport={port}'
                        ], capture_output=True, text=True, encoding='utf-8', errors='replace')
                        
                        if add_result.returncode == 0:
                            app_logger.info(f"[APP] ✅ Порт {port} открыт")
                        else:
                            # Возможно правило уже существует или нужны права администратора
                            if 'уже существует' in add_result.stderr or 'already exists' in add_result.stderr.lower():
                                app_logger.info(f"[APP] ✅ Порт {port} уже открыт")
                            else:
                                app_logger.warning(f"[APP] ⚠️ Не удалось открыть порт {port}: {add_result.stderr or add_result.stdout}")
                    else:
                        app_logger.info(f"[APP] ✅ Порт {port} уже открыт")
                except Exception as e:
                    app_logger.warning(f"[APP] ⚠️ Не удалось проверить/открыть порт {port}: {e}")
                    app_logger.info(f"[APP] ✅ Порт {port} уже открыт")
        
        elif system == 'Darwin':  # macOS
            # macOS Application Firewall через pfctl
            app_logger.info("[APP] 💡 На macOS откройте порты вручную через System Preferences → Security & Privacy → Firewall")
            app_logger.info("[APP] 💡 Или используйте: sudo pfctl -a pflog -f /etc/pf.conf")
        
        elif system == 'Linux':
            # Linux через iptables или ufw
            app_logger.info("[APP] 🔥 Открываем порты в Linux...")
            try:
                # Проверяем наличие ufw (Ubuntu/Debian)
                subprocess.run(['which', 'ufw'], check=True)
                app_logger.info("[APP] Найден ufw, открываем порты...")
                
                for port in [5000, 5001]:
                    # Проверяем, не открыт ли уже порт
                    result = subprocess.run(['ufw', 'status'], capture_output=True, text=True)
                    if f'{port}/tcp' not in result.stdout:
                        subprocess.run(['sudo', 'ufw', 'allow', str(port), '/tcp'], check=True)
                        app_logger.info(f"[APP] ✅ Порт {port} открыт")
                    else:
                        app_logger.info(f"[APP] ✅ Порт {port} уже открыт")
                        
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    # Пробуем iptables
                    for port in [5000, 5001]:
                        app_logger.warning(f"[APP] ⚠️ Настройте порт {port} вручную через iptables или ufw")
                except:
                    app_logger.info("[APP] 💡 Настройте порты вручную см. docs/INSTALL.md")
        
        else:
            app_logger.warning(f"[APP] ⚠️ Неизвестная система: {system}")
            app_logger.info("[APP] 💡 Настройте порты вручную см. docs/INSTALL.md")
            
    except Exception as e:
        app_logger.warning(f"[APP] ⚠️ Не удалось открыть порты автоматически: {e}")
        app_logger.warning("[APP] 💡 Откройте порты вручную см. docs/INSTALL.md")

if __name__ == '__main__':
    # Открываем порты в брандмауэре
    open_firewall_ports()
    
    # Создаем директорию для логов
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Очищаем старые логи при запуске
    app_logger = logging.getLogger('app')
    log_files = ['logs/bots.log', 'logs/app.log', 'logs/error.log']
    for log_file in log_files:
        if os.path.exists(log_file):
            file_size = os.path.getsize(log_file)
            if file_size > 2 * 1024 * 1024:  # 2MB
                app_logger.info(f"[APP] 🗑️ Очищаем большой лог файл: {log_file} ({file_size / 1024 / 1024:.1f}MB)")
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Лог файл очищен при запуске - {datetime.now().isoformat()}\n")
            else:
                app_logger.info(f"[APP] 📝 Лог файл в порядке: {log_file} ({file_size / 1024:.1f}KB)")
    
    # Открываем браузер с задержкой
    Timer(1.5, open_browser).start()

    # ✅ ПРИНУДИТЕЛЬНОЕ ОБНОВЛЕНИЕ positions_data в фоне — не блокирует старт при проблемах с биржей
    def _do_initial_positions_refresh():
        try:
            app_logger.info("[APP] 🔄 Принудительное обновление positions_data при запуске...")
            positions, rapid_growth = current_exchange.get_positions()
            if positions:
                positions_data['total_trades'] = len(positions)
                positions_data['rapid_growth'] = rapid_growth
                high_profitable = []
                profitable = []
                losing = []
                for position in positions:
                    pnl = position['pnl']
                    if pnl > 0:
                        if pnl >= DEFAULTS.PNL_THRESHOLD:
                            high_profitable.append(position)
                        else:
                            profitable.append(position)
                    elif pnl < 0:
                        losing.append(position)
                positions_data.update({
                    'high_profitable': high_profitable,
                    'profitable': profitable,
                    'losing': losing,
                    'stats': {
                        'total_trades': len(positions),
                        'high_profitable_count': len(high_profitable),
                        'profitable_count': len(profitable),
                        'losing_count': len(losing)
                    }
                })
                save_positions_data(positions_data)
                app_logger.info(f"[APP] ✅ positions_data обновлен и сохранен в БД: {len(positions)} позиций")
            else:
                positions_data.update({
                    'high_profitable': [], 'profitable': [], 'losing': [],
                    'total_trades': 0, 'rapid_growth': [],
                    'stats': {'total_trades': 0, 'high_profitable_count': 0, 'profitable_count': 0, 'losing_count': 0}
                })
                save_positions_data(positions_data)
                app_logger.info("[APP] ✅ positions_data очищен и сохранен в БД (нет позиций)")
        except Exception as e:
            app_logger.error(f"[APP] ❌ Ошибка принудительного обновления positions_data: {e}")

    threading.Thread(target=_do_initial_positions_refresh, daemon=True, name="InitialPositionsRefresh").start()

    # Запускаем фоновые процессы (теперь всегда, так как reloader отключен)
    update_thread = threading.Thread(target=background_update)
    update_thread.daemon = True
    update_thread.start()
    
    # Запускаем фоновый процесс загрузки closed_pnl каждые 30 секунд
    closed_pnl_loader_thread = threading.Thread(target=background_closed_pnl_loader, name='ClosedPnLLoader')
    closed_pnl_loader_thread.daemon = True
    closed_pnl_loader_thread.start()
    app_logger.info("[APP] ✅ Фоновый процесс загрузки closed_pnl запущен (обновление каждые 30 секунд)")
    
    # Выполняем первичную загрузку closed_pnl при старте (не ждем 30 секунд)
    app_logger.info("[APP] 🔄 Выполняем первичную загрузку closed_pnl...")
    try:
        db = get_app_db()
        last_timestamp = db.get_latest_closed_pnl_timestamp(exchange=ACTIVE_EXCHANGE)
        
        if not last_timestamp:
            # Первая загрузка - загружаем все данные
            app_logger.info("[APP] Первая загрузка closed_pnl - загружаем все данные...")
            closed_pnl = current_exchange.get_closed_pnl(sort_by='time', period='all')
            if closed_pnl:
                saved = db.save_closed_pnl(closed_pnl, exchange=ACTIVE_EXCHANGE)
                if saved:
                    app_logger.info(f"[APP] ✅ Первичная загрузка: сохранено {len(closed_pnl)} закрытых позиций в БД")
                else:
                    app_logger.warning(f"[APP] ⚠️ Не удалось сохранить {len(closed_pnl)} позиций при первичной загрузке")
            else:
                app_logger.info("[APP] ℹ️ Нет закрытых позиций для загрузки")
        else:
            app_logger.info(f"[APP] ✅ В БД уже есть данные closed_pnl (последний timestamp: {last_timestamp})")
    except Exception as e:
        app_logger.error(f"[APP] ❌ Ошибка первичной загрузки closed_pnl: {e}")
        import traceback
        pass
    try:
        from utils.memory_utils import force_collect_full
        force_collect_full()
    except Exception:
        pass

    # Запускаем поток для отправки дневного отчета
    if TELEGRAM_NOTIFY['DAILY_REPORT']:
        daily_report_thread = threading.Thread(target=send_daily_report)
        daily_report_thread.daemon = True
        daily_report_thread.start()
        
    # Запускаем поток очистки кэша
    cache_cleanup_thread = threading.Thread(target=background_cache_cleanup)
    cache_cleanup_thread.daemon = True
    cache_cleanup_thread.start()

    # Планировщик бэкапов только app_data.db (своя БД app.py)
    if DATABASE_BACKUP.get('ENABLED', True) and DATABASE_BACKUP.get('APP_ENABLED', True):
        _app_backup_cfg = {**DATABASE_BACKUP, 'APP_ENABLED': True, 'AI_ENABLED': False, 'BOTS_ENABLED': False}
        _backup_thread = threading.Thread(
            target=run_backup_scheduler_loop,
            args=(_app_backup_cfg,),
            name='DatabaseBackupScheduler',
            daemon=True
        )
        _backup_thread.start()
        app_logger.info("[APP] 💾 Планировщик бэкапов App БД (app_data.db) запущен")

    # Запускаем поток синхронизации времени (только для Windows)
    if TIME_SYNC.get('ENABLED', False) and sys.platform == 'win32':
        time_sync_thread = threading.Thread(target=time_sync_loop, name='TimeSyncScheduler')
        time_sync_thread.daemon = True
        time_sync_thread.start()
        app_logger.info("[APP] ✅ Фоновый процесс синхронизации времени запущен")
    
    # Отключаем логи werkzeug - они не нужны и засоряют консоль
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.WARNING)  # Показываем только ошибки
    werkzeug_logger.disabled = True  # Полностью отключаем
    
    # Отключаем DEBUG логи от внешних библиотек, которые шумят неформатированными сообщениями
    # urllib3 (используется requests) - логирует "%s://%s:%s "%s %s %s" %s %s"
    urllib3_logger = logging.getLogger('urllib3')
    urllib3_logger.setLevel(logging.WARNING)
    urllib3_connectionpool_logger = logging.getLogger('urllib3.connectionpool')
    urllib3_connectionpool_logger.setLevel(logging.WARNING)
    
    # flask-cors - логирует неформатированные сообщения типа "Settings CORS headers: %s"
    flask_cors_logger = logging.getLogger('flask_cors')
    flask_cors_logger.setLevel(logging.WARNING)
    flask_cors_core_logger = logging.getLogger('flask_cors.core')
    flask_cors_core_logger.setLevel(logging.WARNING)
    
    # matplotlib - логирует неформатированные сообщения типа "matplotlib data path: %s", "CONFIGDIR=%s" и т.д.
    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARNING)
    matplotlib_font_manager_logger = logging.getLogger('matplotlib.font_manager')
    matplotlib_font_manager_logger.setLevel(logging.WARNING)
    matplotlib_backends_logger = logging.getLogger('matplotlib.backends')
    matplotlib_backends_logger.setLevel(logging.WARNING)
    
    # Запускаем Flask-сервер (отключаем reloader для стабильности Telegram уведомлений)
    try:
        app.run(debug=False, host=APP_HOST, port=APP_PORT, use_reloader=False)
    except OSError as e:
        if getattr(e, "errno", None) in (errno.EADDRINUSE, 10048):
            logging.getLogger("app").error(
                f"Порт {APP_PORT} занят. Закройте другой процесс на этом порту или измените APP_PORT в configs/app_config.py."
            )
        raise 