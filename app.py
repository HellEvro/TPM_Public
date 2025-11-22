import base64
from io import BytesIO
from flask import Flask, render_template, jsonify, request
import threading
import time
from datetime import datetime
import os
import sys
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

# Проверка наличия конфигурации
if not os.path.exists('app/config.py'):
    # Используем stderr для критических ошибок, так как logger еще не настроен
    import sys
    sys.stderr.write("\n" + "="*80 + "\n")
    sys.stderr.write("❌ ОШИБКА: Файл конфигурации не найден!\n")
    sys.stderr.write("="*80 + "\n")
    sys.stderr.write("\n")
    sys.stderr.write("📝 Для первого запуска выполните:\n")
    sys.stderr.write("\n")
    sys.stderr.write("   1. Скопируйте файл конфигурации:\n")
    if os.name == 'nt':  # Windows
        sys.stderr.write("      copy app\\config.example.py app\\config.py\n")
    else:  # Linux/Mac
        sys.stderr.write("      cp app/config.example.py app/config.py\n")
    sys.stderr.write("\n")
    sys.stderr.write("   2. Отредактируйте app/config.py:\n")
    sys.stderr.write("      - Добавьте свои API ключи бирж\n")
    sys.stderr.write("      - Настройте Telegram (опционально)\n")
    sys.stderr.write("\n")
    sys.stderr.write("   3. Запустите снова:\n")
    sys.stderr.write("      python app.py\n")
    sys.stderr.write("\n")
    sys.stderr.write("   📖 Подробная инструкция: docs/INSTALL.md\n")
    sys.stderr.write("\n")
    sys.stderr.write("="*80 + "\n")
    sys.stderr.write("\n")
    sys.exit(1)

from app.config import *

# Импортируем БД для app.py
from bot_engine.app_database import get_app_database

# Конфигурация резервного копирования (значения по умолчанию)
_DATABASE_BACKUP_DEFAULTS = {
    'ENABLED': True,
    'INTERVAL_MINUTES': 60,
    'RUN_ON_START': True,
    'AI_ENABLED': True,
    'BOTS_ENABLED': True,
    'BACKUP_DIR': None,
    'MAX_RETRIES': 3,
}

if 'DATABASE_BACKUP' not in globals() or not isinstance(globals().get('DATABASE_BACKUP'), dict):
    DATABASE_BACKUP = _DATABASE_BACKUP_DEFAULTS.copy()
else:
    _merged_backup_config = _DATABASE_BACKUP_DEFAULTS.copy()
    _merged_backup_config.update(DATABASE_BACKUP)
    DATABASE_BACKUP = _merged_backup_config

import sys
from app.telegram_notifier import TelegramNotifier
from exchanges.exchange_factory import ExchangeFactory
import json
import logging
from utils.color_logger import setup_color_logging
from bot_engine.backup_service import get_backup_service

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
    sys.stderr.write("   1. Скопируйте app/config.example.py -> app/config.py (если еще не сделали)\n")
    sys.stderr.write("   2. Создайте app/keys.py с реальными ключами\n")
    sys.stderr.write("   3. Или добавьте ключи прямо в app/config.py\n")
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
    PNL_THRESHOLD = 100

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
            cache_logger.debug(f"[CACHE] ✅ JS файл: {request.path} - кэш отключен")
    
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

# Настройка цветного логирования с фильтром уровней из конфига
try:
    console_levels = CONSOLE_LOG_LEVELS if 'CONSOLE_LOG_LEVELS' in globals() else []
    setup_color_logging(console_log_levels=console_levels if console_levels else None, log_file='logs/app.log')
except Exception as e:
    # Если не удалось настроить, используем стандартное логирование
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
            f"PnL: {pos['pnl']:.2f} USDT\n"
            f"ROI: {pos['roi']:.2f}%\n"
        )
    return "\n".join(result)

def format_stats(stats):
    """Форматирует стаику для записи в лог"""
    return (
        f"Total PnL: {stats['total_pnl']:.2f} USDT\n"
        f"Total profit: {stats['total_profit']:.2f} USDT\n"
        f"Total loss: {stats['total_loss']:.2f} USDT\n"
        f"Number of high-profitable positions: {stats['high_profitable_count']}\n"
        f"Number of profitable positions: {stats['profitable_count']}\n"
        f"Number of losing positions: {stats['losing_count']}\n"
        f"\nTOP-3 profitable:\n{format_positions(stats['top_profitable'])}\n"
        f"\nTOP-3 losing:\n{format_positions(stats['top_losing'])}"
    )

stats_lock = Lock()
# Блокировка для операций matplotlib (не потокобезопасен)
matplotlib_lock = Lock()
backup_scheduler_stop_event = threading.Event()
closed_pnl_loader_stop_event = threading.Event()


def _run_backup_job(backup_service, backup_config):
    """Запускает единичный цикл резервного копирования."""
    backup_logger = logging.getLogger('BackupScheduler')
    include_ai = backup_config.get('AI_ENABLED', True)
    include_bots = backup_config.get('BOTS_ENABLED', True)

    if not include_ai and not include_bots:
        backup_logger.info("[Backup] Нет активных БД для резервного копирования, задание пропущено")
        return

    max_retries = backup_config.get('MAX_RETRIES', 3)

    try:
        result = backup_service.create_backup(
            include_ai=include_ai,
            include_bots=include_bots,
            max_retries=max_retries
        )
    except Exception as exc:
        backup_logger.exception(f"[Backup] Ошибка выполнения резервного копирования: {exc}")
        return

    timestamp = result.get('timestamp', 'unknown')
    if result.get('success'):
        backup_logger.info(f"[Backup] Резервное копирование завершено успешно (timestamp={timestamp})")
    else:
        backup_logger.warning(f"[Backup] Резервное копирование завершено с ошибками (timestamp={timestamp})")

    for db_key in ('ai', 'bots'):
        backup_info = result['backups'].get(db_key)
        if backup_info:
            backup_logger.info(
                "[Backup] %s: файл %s (%.2f MB, valid=%s)",
                db_key.upper(),
                backup_info['path'],
                backup_info['size_mb'],
                'yes' if backup_info.get('valid', True) else 'no'
            )

    for warning_msg in result.get('errors', []):
        backup_logger.warning(f"[Backup] {warning_msg}")


def backup_scheduler_loop():
    """Фоновый планировщик регулярных бэкапов."""
    backup_logger = logging.getLogger('BackupScheduler')
    backup_config = DATABASE_BACKUP or {}

    if not backup_config.get('ENABLED', True):
        backup_logger.info("[Backup] Автоматическое резервное копирование выключено настройками")
        return

    if not (backup_config.get('AI_ENABLED', True) or backup_config.get('BOTS_ENABLED', True)):
        backup_logger.info("[Backup] Ни одна база не выбрана для резервного копирования, поток остановлен")
        return

    backup_dir = backup_config.get('BACKUP_DIR')

    try:
        backup_service = get_backup_service(backup_dir)
    except Exception as exc:
        backup_logger.exception(f"[Backup] Не удалось инициализировать сервис бэкапов: {exc}")
        return

    interval_minutes = backup_config.get('INTERVAL_MINUTES', 60)
    try:
        interval_minutes = float(interval_minutes)
    except (TypeError, ValueError):
        backup_logger.warning("[Backup] Некорректное значение INTERVAL_MINUTES, используется 60 минут")
        interval_minutes = 60

    interval_seconds = max(60, int(interval_minutes * 60))
    backup_logger.info(
        "[Backup] Планировщик запущен: каждые %s минут (%.0f секунд). Директория: %s",
        interval_minutes,
        interval_seconds,
        backup_dir or 'data/backups'
    )

    if backup_config.get('RUN_ON_START', True):
        _run_backup_job(backup_service, backup_config)

    while not backup_scheduler_stop_event.wait(interval_seconds):
        _run_backup_job(backup_service, backup_config)

def background_closed_pnl_loader():
    """Фоновый процесс для загрузки закрытых PnL из биржи в БД каждые 30 секунд"""
    app_logger = logging.getLogger('app')
    app_logger.info("[CLOSED_PNL_LOADER] Запуск фонового процесса загрузки closed_pnl...")
    
    while not closed_pnl_loader_stop_event.is_set():
        try:
            db = get_app_db()
            
            # Получаем timestamp последней загруженной позиции
            last_timestamp = db.get_latest_closed_pnl_timestamp(exchange=ACTIVE_EXCHANGE)
            
            app_logger.debug(f"[CLOSED_PNL_LOADER] Последний timestamp в БД: {last_timestamp}")
            
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
                    app_logger.debug("[CLOSED_PNL_LOADER] Нет новых закрытых позиций")
                    
            except Exception as e:
                app_logger.error(f"[CLOSED_PNL_LOADER] Ошибка загрузки closed_pnl с биржи: {e}")
                import traceback
                app_logger.debug(traceback.format_exc())
            
            # Ждем 30 секунд до следующей загрузки
            closed_pnl_loader_stop_event.wait(30)
            
        except Exception as e:
            app_logger.error(f"[CLOSED_PNL_LOADER] Критическая ошибка в фоновом процессе: {e}")
            import traceback
            app_logger.debug(traceback.format_exc())
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
                    if pnl >= 100:
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
    pnl_threshold = float(request.args.get('pnl_threshold', 100))
    
    all_available_pairs = []  # Больше не используется
    
    if not positions_data['high_profitable'] and not positions_data['profitable'] and not positions_data['losing']:
        # Получаем данные аккаунта даже если нет позиций
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

    # Получаем все позиции
    all_positions = (positions_data['high_profitable'] + 
                    positions_data['profitable'] + 
                    positions_data['losing'])
    
    # Создаем множество символов из активных позиций
    active_position_symbols = set(position['symbol'] for position in all_positions)
    
    # Фильтруем доступные пары
    available_pairs = [pair for pair in all_available_pairs if pair not in active_position_symbols]
    
    # Распределяем позиции по категориям
    high_profitable = []
    profitable = []
    losing = []
    total_profit = 0
    total_loss = 0
    
    for position in all_positions:
        pnl = position['pnl']
        if pnl > 0:
            if pnl >= pnl_threshold:
                high_profitable.append(position)
            else:
                profitable.append(position)
            total_profit += pnl
        elif pnl < 0:
            losing.append(position)
            total_loss += pnl

    # Сортируем позиции
    high_profitable.sort(key=lambda x: x['pnl'], reverse=True)
    profitable.sort(key=lambda x: x['pnl'], reverse=True)
    losing.sort(key=lambda x: x['pnl'])
    
    # Получаем TOP-3
    all_profitable = high_profitable + profitable
    all_profitable.sort(key=lambda x: x['pnl'], reverse=True)
    top_profitable = all_profitable[:3] if all_profitable else []
    top_losing = losing[:3] if losing else []

    # Формируем статистику
    stats = {
        'total_pnl': total_profit + total_loss,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'high_profitable_count': len(high_profitable),
        'profitable_count': len(profitable),
        'losing_count': len(losing),
        'top_profitable': top_profitable,
        'top_losing': top_losing,
        'total_trades': len(all_positions)
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
            if pnl >= 100:
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
        exchange_config = EXCHANGES[ACTIVE_EXCHANGE]
        # БЕЗОПАСНОСТЬ: НЕ выводим конфигурацию с ключами!
        safe_config = {k: ('***HIDDEN***' if k in ['api_key', 'api_secret', 'passphrase'] else v) 
                       for k, v in exchange_config.items()}
        app_logger.info(f"[INIT] Конфигурация получена: {safe_config}")
        
        app_logger.info(f"[INIT] Создание экземпляра биржи {ACTIVE_EXCHANGE}...")
        exchange = ExchangeFactory.create_exchange(
            ACTIVE_EXCHANGE,
            exchange_config['api_key'],
            exchange_config['api_secret'],
            exchange_config.get('passphrase')  # Добавляем passphrase для OKX
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
            exchange_config = EXCHANGES[exchange_name]
            new_exchange = ExchangeFactory.create_exchange(
                exchange_name,
                exchange_config['api_key'],
                exchange_config['api_secret'],
                exchange_config.get('passphrase')  # Добавляем passphrase для OKX
            )
            
            # Пробуем получить позиции для проверки работоспособности
            positions, _ = new_exchange.get_positions()
            
            # Если все хорошо, обновляем конфигурацию
            with open('app/config.py', 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # Обновляем активную биржу в конфиге
            new_config = config_content.replace(
                f"ACTIVE_EXCHANGE = '{ACTIVE_EXCHANGE}'",
                f"ACTIVE_EXCHANGE = '{exchange_name}'"
            )
            
            with open('app/config.py', 'w', encoding='utf-8') as f:
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
        ticker_logger.debug(f"[TICKER] Raw ticker data: {ticker_data}")
        
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
    """Фоновая очистка кэша"""
    while True:
        try:
            clear_old_cache()
        except Exception as e:
            app_logger = logging.getLogger('app')
            app_logger.error(f"Error in cache cleanup: {str(e)}")
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
    """Фоновая очистка кэша"""
    while True:
        try:
            clear_old_cache()
        except Exception:
            pass
        time.sleep(60)  # Проверяем каждую минуту

# Прокси для API endpoints ботов (перенаправляем на внешний сервис)
def get_candles_from_file(symbol, timeframe='6h', period_days=None):
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
        elif timeframe == '6h':
            # Используем 6h свечи как есть
            candles = candles_6h
        else:
            # Для других таймфреймов возвращаем 6h
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
            'error': f'Timeout calling bots service ({timeout}s)',
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

@app.route('/api/bots/auto-bot', methods=['POST'])
def toggle_auto_bot():
    """Переключить Auto Bot (прокси к сервису ботов)"""
    data = request.get_json()
    result = call_bots_service('/api/bots/auto-bot', method='POST', data=data)
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

@app.route('/api/bots/start', methods=['POST'])
def start_bot():
    """Запустить бота (прокси к сервису ботов)"""
    data = request.get_json()
    result = call_bots_service('/api/bots/start', method='POST', data=data)
    status_code = result.get('status_code', 200 if result.get('success') else 500)
    return jsonify(result), status_code

chart_render_lock = threading.Lock()

@app.route('/get_symbol_chart/<symbol>')
def get_symbol_chart(symbol):
    """Получение миниграфика RSI 6ч для символа (использует кэш из bots.py)"""
    chart_logger = logging.getLogger('app')
    try:
        theme = request.args.get('theme', 'dark')
        chart_logger.info(f"[CHART] Getting RSI 6h chart for {symbol} with theme {theme}")
        
        # ✅ ИСПОЛЬЗУЕМ КЭШ ИЗ BOTS.PY вместо запроса к бирже
        # Запрашиваем историю RSI через API bots.py (использует кэш свечей)
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
        
        # Создаем временные метки на основе количества значений RSI
        # Каждое значение RSI соответствует одной 6-часовой свече
        # Начинаем с текущего времени и идем назад
        current_timestamp = int(time.time() * 1000)
        times = []
        for i in range(num_rsi_values):
            # Каждое значение RSI отстоит на 6 часов (21600000 мс) от предыдущего
            # Последнее значение RSI - самое свежее (текущее время)
            ts = current_timestamp - (num_rsi_values - 1 - i) * 21600000
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
        
        chart_logger.info(f"[CHART] Successfully generated RSI 6h chart for {symbol} (from cache)")
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
def get_rsi_6h(symbol):
    """Получение RSI 6ч данных за 56 свечей (неделя)"""
    rsi_logger = logging.getLogger('app')
    try:
        rsi_logger.info(f"[RSI 6h] Getting RSI 6h data for {symbol}")
        
        # Проверяем инициализацию биржи
        if not current_exchange:
            rsi_logger.error("[RSI 6h] Exchange not initialized")
            return jsonify({'error': 'Exchange not initialized'}), 500
        
        # Импортируем функцию расчета RSI
        from bot_engine.utils.rsi_utils import calculate_rsi_history
        
        # ✅ ЧИТАЕМ НАПРЯМУЮ ИЗ ФАЙЛА (не требует запущенного bots.py)
        # 56 свечей * 6 часов = 336 часов = 14 дней
        data = get_candles_from_file(symbol, timeframe='6h', period_days=14)
        if not data or not data.get('success'):
            rsi_logger.error(f"[RSI 6h] Failed to get chart data from file for {symbol}")
            return jsonify({'error': 'Failed to get chart data from file'}), 500
            
        candles = data.get('data', {}).get('candles', [])
        if not candles:
            rsi_logger.warning(f"[RSI 6h] No candles data for {symbol}")
            return jsonify({'error': 'No chart data available'}), 404
        
        # Берем последние 56 свечей
        candles = candles[-56:] if len(candles) >= 56 else candles
        
        if len(candles) < 15:  # Минимум для расчета RSI (период 14 + 1)
            rsi_logger.warning(f"[RSI 6h] Not enough data for RSI calculation for {symbol}")
            return jsonify({'error': 'Not enough data for RSI calculation'}), 400
        
        # Извлекаем цены закрытия
        closes = [float(candle['close']) for candle in candles]
        
        # Рассчитываем историю RSI
        rsi_history = calculate_rsi_history(closes, period=14)
        
        if not rsi_history:
            rsi_logger.error(f"[RSI 6h] Failed to calculate RSI for {symbol}")
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
            
            # Создаем временные метки с шагом 6 часов (21600000 мс)
            timestamps = []
            for i in range(len(rsi_history)):
                ts = last_timestamp - (len(rsi_history) - 1 - i) * 21600000
                timestamps.append(ts)
        
        # Берем только последние 56 значений
        if len(rsi_history) > 56:
            rsi_history = rsi_history[-56:]
            timestamps = timestamps[-56:]
        
        rsi_logger.info(f"[RSI 6h] Successfully calculated RSI for {symbol}: {len(rsi_history)} values")
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'rsi_values': rsi_history,
            'timestamps': timestamps,
            'period': '6h',
            'candles_count': len(rsi_history)
        })
        
    except Exception as e:
        rsi_logger.error(f"[RSI 6h] Error calculating RSI 6h for {symbol}: {str(e)}")
        import traceback
        rsi_logger.error(f"[RSI 6h] Traceback: {traceback.format_exc()}")
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
                result = subprocess.run(
                    ['netsh', 'advfirewall', 'firewall', 'show', 'rule', f'name={service_name}'],
                    capture_output=True,
                    text=True
                )
                
                if service_name not in result.stdout:
                    app_logger.info(f"[APP] 🔥 Открываем порт {port}...")
                    subprocess.run([
                        'netsh', 'advfirewall', 'firewall', 'add', 'rule',
                        f'name={service_name}',
                        'dir=in',
                        'action=allow',
                        'protocol=TCP',
                        f'localport={port}'
                    ], check=True)
                    app_logger.info(f"[APP] ✅ Порт {port} открыт")
                else:
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
    
    # ✅ ПРИНУДИТЕЛЬНОЕ ОБНОВЛЕНИЕ positions_data при запуске
    app_logger.info("[APP] 🔄 Принудительное обновление positions_data при запуске...")
    try:
        positions, rapid_growth = current_exchange.get_positions()
        if positions:
            # Обновляем positions_data с актуальными данными
            positions_data['total_trades'] = len(positions)
            positions_data['rapid_growth'] = rapid_growth
            
            high_profitable = []
            profitable = []
            losing = []
            
            for position in positions:
                pnl = position['pnl']
                if pnl > 0:
                    if pnl >= 100:  # Используем стандартный порог
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
            # Сохраняем в БД
            save_positions_data(positions_data)
            app_logger.info(f"[APP] ✅ positions_data обновлен и сохранен в БД: {len(positions)} позиций")
        else:
            # Очищаем positions_data если позиций нет
            positions_data.update({
                'high_profitable': [],
                'profitable': [],
                'losing': [],
                'total_trades': 0,
                'rapid_growth': [],
                'stats': {
                    'total_trades': 0,
                    'high_profitable_count': 0,
                    'profitable_count': 0,
                    'losing_count': 0
                }
            })
            # Сохраняем в БД
            save_positions_data(positions_data)
            app_logger.info("[APP] ✅ positions_data очищен и сохранен в БД (нет позиций)")
    except Exception as e:
        app_logger.error(f"[APP] ❌ Ошибка принудительного обновления positions_data: {e}")
    
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
        app_logger.debug(traceback.format_exc())
    
    # Запускаем поток для отправки дневного отчета
    if TELEGRAM_NOTIFY['DAILY_REPORT']:
        daily_report_thread = threading.Thread(target=send_daily_report)
        daily_report_thread.daemon = True
        daily_report_thread.start()
        
    # Запускаем поток очистки кэша
    cache_cleanup_thread = threading.Thread(target=background_cache_cleanup)
    cache_cleanup_thread.daemon = True
    cache_cleanup_thread.start()

    if DATABASE_BACKUP.get('ENABLED', True) and (
        DATABASE_BACKUP.get('AI_ENABLED', True) or DATABASE_BACKUP.get('BOTS_ENABLED', True)
    ):
        backup_thread = threading.Thread(target=backup_scheduler_loop, name='DatabaseBackupScheduler')
        backup_thread.daemon = True
        backup_thread.start()
    
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
    app.run(debug=False, host=APP_HOST, port=APP_PORT, use_reloader=False) 