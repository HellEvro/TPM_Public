"""Импорты, константы и глобальные переменные для bots.py"""

import os
import sys
import signal
import threading
import time
import logging
import json
import atexit
import asyncio
import requests
import socket
import psutil
from copy import deepcopy
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
import concurrent.futures

# Импортируем асинхронный процессор
try:
    from bot_engine.async_processor import AsyncMainProcessor
    ASYNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Асинхронный процессор недоступен: {e}")
    ASYNC_AVAILABLE = False

# Импортируем новые модули из bot_engine
try:
    from bot_engine.utils.rsi_utils import calculate_rsi, calculate_rsi_history
    from bot_engine.filters import check_rsi_time_filter, check_exit_scam_filter, check_no_existing_position
    # ✅ ИСПРАВЛЕНО: Используем модуль bots_modules.maturity вместо bot_engine.maturity_checker
    from bots_modules.maturity import (
        mature_coins_storage, mature_coins_lock,
        load_mature_coins_storage, save_mature_coins_storage,
        is_coin_mature_stored, add_mature_coin_to_storage,
        remove_mature_coin_from_storage
    )
    from bot_engine.storage import (
        save_rsi_cache as storage_save_rsi_cache,
        load_rsi_cache as storage_load_rsi_cache,
        clear_rsi_cache,
        save_bots_state as storage_save_bots_state,
        load_bots_state as storage_load_bots_state,
        save_individual_coin_settings as storage_save_individual_coin_settings,
        load_individual_coin_settings as storage_load_individual_coin_settings,
        save_mature_coins, load_mature_coins,
        save_process_state as storage_save_process_state,
        load_process_state as storage_load_process_state,
        save_system_config as storage_save_system_config,
        load_system_config as storage_load_system_config
    )
    from bot_engine.signal_processor import get_effective_signal, check_autobot_filters, process_auto_bot_signals
    # ❌ ОТКЛЮЧЕНО: optimal_ema_manager перемещен в backup (EMA фильтр убран)
    # from bot_engine.optimal_ema_manager import (
    #     load_optimal_ema_data, get_optimal_ema_periods,
    #     update_optimal_ema_data, save_optimal_ema_periods,
    #     optimal_ema_data
    # )
    MODULES_AVAILABLE = True
    # Используем логгер BOTS, чтобы не засорять логи AI модуля
    # Логируем только на уровне DEBUG и только если логгер BOTS уже настроен (когда запущен bots.py)
    bots_logger = logging.getLogger('BOTS')
    if bots_logger.handlers:  # Логируем только если есть обработчики
        pass
except ImportError as e:
    # Логируем только если логгер BOTS уже настроен
    bots_logger = logging.getLogger('BOTS')
    if bots_logger.handlers:
        bots_logger.warning(f"Failed to load new bot_engine modules: {e}")
        bots_logger.warning("Using legacy functions from bots.py")
    MODULES_AVAILABLE = False

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Создаем обертки для функций если модули не загружены
if not MODULES_AVAILABLE:
    # Определяем заглушки для функций которые будут определены позже в файле
    def calculate_rsi(prices, period=14):
        """Будет определена ниже в файле"""
        pass
    
    def calculate_rsi_history(prices, period=14):
        """Будет определена ниже в файле"""
        pass
    
    def calculate_ema(prices, period):
        """Будет определена ниже в файле"""
        pass

# ✅ Fallback версия calculate_ema (так как ema_utils.py перемещен в backup)
def calculate_ema(prices, period):
    """Рассчитывает EMA для массива цен"""
    if len(prices) < period:
        return None
    
    # Первое значение EMA = SMA
    sma = sum(prices[:period]) / period
    ema = sma
    multiplier = 2 / (period + 1)
    
    # Рассчитываем EMA для остальных значений
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema

def check_and_stop_existing_bots_processes():
    """
    Проверяет порт 5001 и останавливает процесс который его занимает.
    
    Returns:
        bool: True если можно продолжать запуск, False если нужно остановиться
    """
    try:
        print("=" * 80)
        print("🔍 ПРОВЕРКА ПОРТА 5001 (BOTS SERVICE)")
        print("=" * 80)
        
        current_pid = os.getpid()
        print(f"📍 Текущий PID: {current_pid}")
        
        # ГЛАВНАЯ ПРОВЕРКА: Проверяем порт 5001
        port_occupied = False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 5001))
            sock.close()
            
            if result == 0:
                port_occupied = True
                print("⚠️  Порт 5001 уже занят!")
            else:
                print("✅ Порт 5001 свободен")
        except Exception as e:
            print(f"⚠️  Ошибка проверки порта: {e}")
        
        # Если порт свободен - сразу выходим
        if not port_occupied:
            print("=" * 80)
            print()
            return True
        
        # Если порт занят - останавливаем процесс
        if port_occupied:
            print("\n⚠️  ПОРТ 5001 ЗАНЯТ - ищем процесс который его использует...")
            
            # Ищем процесс который слушает порт 5001
            process_to_stop = None
            
            try:
                # Ищем ВСЕ процессы python с bots.py в командной строке
                python_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['name'] and 'python' in proc.info['name'].lower():
                            cmdline = proc.info['cmdline']
                            if cmdline and any('bots.py' in arg for arg in cmdline):
                                if proc.info['pid'] != current_pid:
                                    python_processes.append(proc.info['pid'])
                                    print(f"🎯 Найден процесс bots.py: PID {proc.info['pid']}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                
                # Также проверяем порт 5001
                port_process = None
                for conn in psutil.net_connections(kind='inet'):
                    if conn.laddr.port == 5001 and conn.status == 'LISTEN':
                        port_process = conn.pid
                        if port_process != current_pid and port_process not in python_processes:
                            python_processes.append(port_process)
                            print(f"🎯 Найден процесс на порту 5001: PID {port_process}")
                        break
                
                if python_processes:
                    process_to_stop = python_processes[0]  # Останавливаем первый найденный
                
                if process_to_stop and process_to_stop != current_pid:
                    try:
                        proc = psutil.Process(process_to_stop)
                        proc_info = proc.as_dict(attrs=['pid', 'name', 'cmdline', 'create_time'])
                        
                        print(f"🎯 Найден процесс на порту 5001:")
                        print(f"   PID: {proc_info['pid']}")
                        print(f"   Команда: {' '.join(proc_info['cmdline'][:3]) if proc_info['cmdline'] else 'N/A'}...")
                        print()
                        
                        print(f"🔧 Останавливаем процесс {process_to_stop}...")
                        try:
                            proc.terminate()
                            # Ждем завершения с таймаутом
                            try:
                                proc.wait(timeout=3)  # Уменьшаем таймаут до 3 секунд
                                print(f"✅ Процесс {process_to_stop} остановлен")
                            except psutil.TimeoutExpired:
                                # Если не завершился за 3 секунды, принудительно убиваем
                                try:
                                    proc.kill()
                                    proc.wait(timeout=1)
                                    print(f"🔴 Процесс {process_to_stop} принудительно остановлен")
                                except:
                                    pass
                            except psutil.NoSuchProcess:
                                print(f"✅ Процесс {process_to_stop} уже завершен")
                        except Exception as term_error:
                            print(f"⚠️  Ошибка при остановке процесса: {term_error}")
                        
                        # Проверяем освобождение порта (до 10 секунд).
                        # ВАЖНО: если порт не освобожден — НЕ продолжаем запуск, иначе bots.py упадёт на bind,
                        # а UI будет выглядеть как "не коннектится по порту".
                        print("\n⏳ Ожидание освобождения порта 5001...")
                        port_freed = False
                        for i in range(10):
                            time.sleep(1)
                            try:
                                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                sock.settimeout(0.5)
                                result = sock.connect_ex(('127.0.0.1', 5001))
                                sock.close()

                                if result != 0:
                                    print("✅ Порт 5001 освобожден")
                                    port_freed = True
                                    break
                            except Exception:
                                pass

                        if not port_freed:
                            print("❌ Порт 5001 все еще занят!")
                            print("⚠️  Возможно нужно вручную остановить процесс, который слушает порт 5001")
                            print("=" * 80)
                            return False
                        
                    except Exception as e:
                        print(f"❌ Ошибка остановки процесса {process_to_stop}: {e}")
                        print("=" * 80)
                        return False
                
                elif not process_to_stop:
                    print("⚠️  Не удалось найти процесс на порту 5001")
                    print("=" * 80)
                    return False
                        
            except Exception as e:
                print(f"⚠️  Ошибка поиска процесса на порту: {e}")
                print("=" * 80)
                return False
            
            print("=" * 80)
            print("✅ ПРОВЕРКА ЗАВЕРШЕНА - ПРОДОЛЖАЕМ ЗАПУСК")
            print("=" * 80)
            print()
            return True
            
    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА ПРОВЕРКИ: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️  Продолжаем запуск без проверки...")
        print("=" * 80)
        print()
        return True

# Импорт цветного логирования
from utils.color_logger import setup_color_logging

# Импорт системы истории ботов - ПЕРЕНЕСЕН В ГЛАВНЫЙ bots.py
# Здесь только объявляем переменные как None, они будут установлены в main файле
bot_history_manager = None
BOT_HISTORY_AVAILABLE = False

# Импорты для бот-движка
from exchanges.exchange_factory import ExchangeFactory
from app.config import EXCHANGES, APP_DEBUG
from bot_engine.bot_config import (
    SystemConfig, RiskConfig, FilterConfig, ExchangeConfig,
    RSI_EXTREME_ZONE_TIMEOUT, RSI_EXTREME_OVERSOLD, RSI_EXTREME_OVERBOUGHT,
    RSI_VOLUME_CONFIRMATION_MULTIPLIER, RSI_DIVERGENCE_LOOKBACK,
    DEFAULT_AUTO_BOT_CONFIG as BOT_ENGINE_DEFAULT_CONFIG
)
from bot_engine.smart_rsi_manager import SmartRSIManager
from bot_engine.trading_bot import TradingBot as RealTradingBot

# Константы для файлов состояния
BOTS_STATE_FILE = 'data/bots_state.json'
BOTS_POSITIONS_REGISTRY_FILE = 'data/bot_positions_registry.json'  # Реестр позиций открытых ботами

# ✅ ВСЕ КОНСТАНТЫ НАСТРОЕК ПЕРЕНЕСЕНЫ В SystemConfig (bot_engine/bot_config.py)
# Используйте SystemConfig.КОНСТАНТА для доступа к настройкам

# Глобальные переменные для кэшированных данных (как в app.py)
bots_cache_data = {
    'bots': [],
    'account_info': {},
    'last_update': None
}
bots_cache_lock = threading.Lock()

# Кэш для подавления повторяющихся логов
log_suppression_cache = {
    'auto_bot_signals': {'count': 0, 'last_log': 0, 'message': ''},
    'position_sync': {'count': 0, 'last_log': 0, 'message': ''},
    'cache_update': {'count': 0, 'last_log': 0, 'message': ''},
    'exchange_positions': {'count': 0, 'last_log': 0, 'message': ''}
}
RSI_CACHE_FILE = 'data/rsi_cache.json'
DEFAULT_CONFIG_FILE = 'data/default_auto_bot_config.json'
PROCESS_STATE_FILE = 'data/process_state.json'
SYSTEM_CONFIG_FILE = 'data/system_config.json'
INDIVIDUAL_COIN_SETTINGS_FILE = 'data/individual_coin_settings.json'

# Создаем папку для данных если её нет
os.makedirs('data', exist_ok=True)

# Метаданные загрузки индивидуальных настроек
_individual_coin_settings_state = {
    'last_mtime': None
}

# Дефолтная конфигурация Auto Bot (для восстановления)
# ✅ ИСПОЛЬЗУЕМ КОНФИГ ИЗ bot_engine/bot_config.py
# Импортирован как BOT_ENGINE_DEFAULT_CONFIG
DEFAULT_AUTO_BOT_CONFIG = BOT_ENGINE_DEFAULT_CONFIG

# Состояние процессов системы
process_state = {
    'smart_rsi_manager': {
        'active': False,
        'last_update': None,
        'update_count': 0,
        'last_error': None
    },
    'auto_bot_worker': {
        'active': False,
        'last_check': None,
        'check_count': 0,
        'last_error': None
    },
    'auto_save_worker': {
        'active': False,
        'last_save': None,
        'save_count': 0,
        'last_error': None
    },
    'exchange_connection': {
        'initialized': False,
        'last_sync': None,
        'connection_count': 0,
        'last_error': None
    },
    'auto_bot_signals': {
        'last_check': None,
        'signals_processed': 0,
        'bots_created': 0,
        'last_error': None
    }
}

# Настройка цветного логирования
# ВАЖНО: Логирование уже настроено в bots.py, здесь только убеждаемся что оно работает
# Не создаем дублирующие handlers, чтобы избежать записи в неправильные файлы
setup_color_logging(enable_file_logging=False)  # Отключаем файловое логирование здесь, т.к. оно уже настроено в bots.py

# Настройка кодировки для stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def should_log_message(category, message, interval_seconds=60):
    """
    Определяет, нужно ли логировать сообщение или подавить его из-за частоты
    
    Args:
        category: Категория сообщения (auto_bot_signals, position_sync, etc.)
        message: Текст сообщения
        interval_seconds: Минимальный интервал между одинаковыми сообщениями
    
    Returns:
        tuple: (should_log: bool, summary_message: str or None)
    """
    import time
    
    current_time = time.time()
    
    if category not in log_suppression_cache:
        log_suppression_cache[category] = {'count': 0, 'last_log': 0, 'message': ''}
    
    cache_entry = log_suppression_cache[category]
    
    # Если это то же самое сообщение
    if cache_entry['message'] == message:
        cache_entry['count'] += 1
        
        # Если прошло достаточно времени, логируем с счетчиком
        if current_time - cache_entry['last_log'] >= interval_seconds:
            cache_entry['last_log'] = current_time
            
            if cache_entry['count'] > 1:
                summary_message = f"{message} (повторилось {cache_entry['count']} раз за {int(current_time - cache_entry['last_log'] + interval_seconds)}с)"
                cache_entry['count'] = 0
                return True, summary_message
            else:
                cache_entry['count'] = 0
                return True, message
        else:
            # Подавляем сообщение
            return False, None
    else:
        # Новое сообщение
        if cache_entry['count'] > 0:
            # Логируем сводку по предыдущему сообщению
            summary = f" Предыдущее сообщение повторилось {cache_entry['count']} раз"
            logger.info(f"[{category.upper()}] {summary}")
        
        cache_entry['message'] = message
        cache_entry['count'] = 1
        cache_entry['last_log'] = current_time
        return True, message

logger = logging.getLogger('BotsService')

# Отключаем HTTP логи Werkzeug для чистоты консоли
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.WARNING)  # Показывать только warnings и errors

# Отключаем DEBUG логи от внешних библиотек, которые шумят неформатированными сообщениями
# urllib3 (используется requests) - логирует "%s://%s:%s "%s %s %s" %s %s"
urllib3_logger = logging.getLogger('urllib3')
urllib3_logger.setLevel(logging.WARNING)
urllib3_connectionpool_logger = logging.getLogger('urllib3.connectionpool')
urllib3_connectionpool_logger.setLevel(logging.WARNING)

# flask-cors - логирует "Configuring CORS with resources: %s"
flask_cors_logger = logging.getLogger('flask_cors')
flask_cors_logger.setLevel(logging.WARNING)
flask_cors_core_logger = logging.getLogger('flask_cors.core')
flask_cors_core_logger.setLevel(logging.WARNING)

# matplotlib - логирует неформатированные сообщения типа "matplotlib data path: %s", "CONFIGDIR=%s" и т.д.
# КРИТИЧНО: Отключаем ДО импорта matplotlib, чтобы перехватить логи при инициализации
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)
matplotlib_logger.disabled = False  # Не отключаем полностью, только DEBUG
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

# Создаем Flask приложение для API ботов
# ВАЖНО: Создаем здесь чтобы было доступно при импорте api_endpoints
bots_app = Flask(__name__)
CORS(bots_app)

# API endpoint для проверки статуса сервиса ботов
@bots_app.route('/api/status', methods=['GET'])
def api_status():
    """API endpoint для проверки статуса сервиса ботов"""
    return jsonify({
        'status': 'online',
        'service': 'bots',
        'timestamp': datetime.now().isoformat(),
        'test': 'simple_endpoint'
    })

# Регистрируем AI endpoints
try:
    from bot_engine.api.endpoints_ai import register_ai_endpoints
    register_ai_endpoints(bots_app)
    logger.info("✅ AI endpoints зарегистрированы")
except ImportError as e:
    logger.warning(f"⚠️ AI endpoints недоступны: {e}")
except Exception as e:
    logger.error(f"❌ Ошибка регистрации AI endpoints: {e}")

# Добавляем обработчик ошибок JSON сериализации
@bots_app.errorhandler(TypeError)
def handle_json_error(e):
    """Обрабатывает ошибки JSON сериализации"""
    if "not JSON serializable" in str(e):
        logger.error(f" Ошибка JSON сериализации: {e}")
        return jsonify({'success': False, 'error': 'JSON serialization error'}), 500
    return jsonify({'success': False, 'error': str(e)}), 500

# Класс для хранения глобального состояния (чтобы изменения были видны во всех модулях)
class GlobalState:
    exchange = None
    smart_rsi_manager = None
    async_processor = None
    async_processor_task = None
    system_initialized = False

# Создаем единственный экземпляр
_state = GlobalState()

# Функции для работы с глобальным состоянием
def get_exchange():
    """Получить актуальную ссылку на биржу"""
    return _state.exchange

def set_exchange(exch):
    """Установить биржу во всех модулях"""
    _state.exchange = exch
    return exch

# Экспортируем как переменные для обратной совместимости
exchange = _state.exchange
shutdown_flag = threading.Event()
graceful_shutdown = False  # Флаг для graceful shutdown
system_initialized = _state.system_initialized
smart_rsi_manager = _state.smart_rsi_manager
async_processor = _state.async_processor
async_processor_task = _state.async_processor_task
service_start_time = time.time()  # Время запуска сервиса для расчета uptime

# БЛОКИРОВКИ для предотвращения race conditions
coin_processing_locks = {}  # Блокировки для обработки каждой монеты
coin_processing_lock = threading.Lock()  # Блокировка для управления coin_processing_locks

def get_coin_processing_lock(symbol):
    """Получает блокировку для обработки конкретной монеты"""
    with coin_processing_lock:
        if symbol not in coin_processing_locks:
            coin_processing_locks[symbol] = threading.Lock()
        return coin_processing_locks[symbol]

# Инициализируем биржу при импорте модуля
def init_exchange():
    """Инициализация биржи"""
    global exchange
    try:
        logger.info("[INIT] Инициализация биржи...")
        exchange = ExchangeFactory.create_exchange(
            'BYBIT', 
            EXCHANGES['BYBIT']['api_key'], 
            EXCHANGES['BYBIT']['api_secret']
        )
        logger.info("[INIT] ✅ Биржа инициализирована успешно")
    except Exception as e:
        logger.error(f"[INIT] ❌ Ошибка инициализации биржи: {e}")
        exchange = None

# Инициализация биржи будет выполнена в init_bot_service()

# ✅ ТОРГОВЫЕ ПАРАМЕТРЫ ПЕРЕНЕСЕНЫ В SystemConfig
# Используйте:
# - SystemConfig.RSI_OVERSOLD, SystemConfig.RSI_OVERBOUGHT
# - SystemConfig.RSI_EXIT_LONG, SystemConfig.RSI_EXIT_SHORT
# - SystemConfig.EMA_FAST, SystemConfig.EMA_SLOW
# - SystemConfig.TREND_CONFIRMATION_BARS

# Возможные статусы ботов
BOT_STATUS = {
    'IDLE': 'idle',
    'RUNNING': 'running',
    'IN_POSITION_LONG': 'in_position_long',
    'IN_POSITION_SHORT': 'in_position_short',
    'PAUSED': 'paused'
}

# Глобальная модель данных для всех монет с RSI 6H
coins_rsi_data = {
    'coins': {},  # Словарь всех монет с RSI данными
    'last_update': None,
    'update_in_progress': False,
    'total_coins': 0,
    'successful_coins': 0,
    'failed_coins': 0,
    'data_version': 0,  # ✅ Версия данных для предотвращения "гуляющих" данных
    'ui_update_paused': False,  # ✅ Флаг паузы UI обновлений
    'candles_cache': {},  # ✅ Кэш свечей для быстрого доступа при расчете RSI
    'last_candles_update': None  # ✅ Время последнего обновления свечей
}

# Модель данных для ботов
bots_data = {
    'bots': {},  # {symbol: bot_config}
    'auto_bot_config': DEFAULT_AUTO_BOT_CONFIG.copy(),  # Используем дефолтную конфигурацию
    'individual_coin_settings': {},  # {symbol: settings}
    'global_stats': {
        'active_bots': 0,
        'bots_in_position': 0,
        'total_pnl': 0.0
    }
}

# Блокировки для данных
rsi_data_lock = threading.Lock()
bots_data_lock = threading.Lock()

# Загружаем сохраненную конфигурацию Auto Bot
def load_auto_bot_config():
    """Загружает конфигурацию Auto Bot из bot_config.py
    
    ✅ ЕДИНСТВЕННЫЙ источник истины: bot_engine/bot_config.py
    - Все настройки в Python-файле с комментариями
    - Можно редактировать руками с подробными пояснениями
    - Система читает напрямую из DEFAULT_AUTO_BOT_CONFIG
    """
    try:
        # ✅ КРИТИЧЕСКИ ВАЖНО: Перезагружаем модуль перед чтением
        # Это гарантирует, что мы читаем актуальные данные из файла, а не из кэша Python
        import importlib
        import sys
        import os
        import re
        
        # Абсолютный путь к bot_config.py (не зависит от cwd — на лету подхватываем сохранённый конфиг)
        _bc_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_bc_dir)
        config_file_path = os.path.join(_project_root, 'bot_engine', 'bot_config.py')
        reloaded = False

        if os.path.exists(config_file_path):
            # Проверяем время модификации файла
            current_mtime = os.path.getmtime(config_file_path)
            
            # Кэшируем время последней модификации
            if not hasattr(load_auto_bot_config, '_last_mtime'):
                # При первом вызове принудительно перезагружаем модуль
                load_auto_bot_config._last_mtime = 0  # Устанавливаем 0, чтобы гарантировать перезагрузку
            
            # Перезагружаем только если файл изменился или это первый вызов или принудительная перезагрузка (_last_mtime == 0)
            # ✅ КРИТИЧНО: При _last_mtime == 0 ВСЕГДА перезагружаем модуль, даже если файл не изменился
            # Это нужно для принудительной перезагрузки из API endpoint
            is_forced_reload = load_auto_bot_config._last_mtime == 0
            
            if current_mtime > load_auto_bot_config._last_mtime or is_forced_reload:
                # ✅ КРИТИЧНО: Сохраняем таймфрейм из БД перед перезагрузкой
                saved_timeframe_from_db = None
                try:
                    from bot_engine.bots_database import get_bots_database
                    db = get_bots_database()
                    saved_timeframe_from_db = db.load_timeframe()
                except Exception as tf_save_err:
                    logger.warning(f"[CONFIG] ⚠️ Не удалось сохранить таймфрейм из БД: {tf_save_err}")
                
                # Импортируем модуль, если его еще нет
                if 'bot_engine.bot_config' not in sys.modules:
                    import bot_engine.bot_config
                else:
                    import bot_engine.bot_config
                    importlib.reload(bot_engine.bot_config)
                
                # ✅ КРИТИЧНО: Восстанавливаем таймфрейм из БД после перезагрузки
                if saved_timeframe_from_db:
                    try:
                        from bot_engine.bot_config import set_current_timeframe
                        set_current_timeframe(saved_timeframe_from_db)
                    except Exception as tf_restore_err:
                        logger.warning(f"[CONFIG] ⚠️ Не удалось восстановить таймфрейм: {tf_restore_err}")
                
                # ✅ ВАЖНО: ВСЕГДА обновляем _last_mtime после перезагрузки модуля
                # Это предотвращает бесконечную перезагрузку при принудительной перезагрузке
                load_auto_bot_config._last_mtime = current_mtime
                reloaded = True
        else:
            # ✅ КРИТИЧНО: Сохраняем таймфрейм из БД перед перезагрузкой
            saved_timeframe_from_db = None
            try:
                from bot_engine.bots_database import get_bots_database
                db = get_bots_database()
                saved_timeframe_from_db = db.load_timeframe()
            except:
                pass
            
            # Если файла нет, просто перезагружаем модуль
            if 'bot_engine.bot_config' in sys.modules:
                import bot_engine.bot_config
                importlib.reload(bot_engine.bot_config)
            else:
                import bot_engine.bot_config  # pragma: no cover
            reloaded = True
            
            # ✅ КРИТИЧНО: Восстанавливаем таймфрейм из БД после перезагрузки
            if saved_timeframe_from_db:
                try:
                    from bot_engine.bot_config import set_current_timeframe
                    set_current_timeframe(saved_timeframe_from_db)
                except:
                    pass
        
        # ✅ КРИТИЧНО: Принудительно перезагружаем модуль ПЕРЕД импортом DEFAULT_AUTO_BOT_CONFIG
        # Это гарантирует, что мы получим актуальное значение из файла, а не из кэша
        # ✅ КРИТИЧНО: Сохраняем таймфрейм из БД перед перезагрузкой
        saved_timeframe_from_db_final = None
        try:
            from bot_engine.bots_database import get_bots_database
            db = get_bots_database()
            saved_timeframe_from_db_final = db.load_timeframe()
        except Exception as tf_final_err:
            logger.warning(f"[CONFIG] ⚠️ Не удалось сохранить таймфрейм из БД (финальный): {tf_final_err}")
        
        if 'bot_engine.bot_config' in sys.modules:
            importlib.reload(sys.modules['bot_engine.bot_config'])
        
        # ✅ КРИТИЧНО: Восстанавливаем таймфрейм из БД после финальной перезагрузки
        if saved_timeframe_from_db_final:
            try:
                from bot_engine.bot_config import set_current_timeframe
                set_current_timeframe(saved_timeframe_from_db_final)
            except Exception as tf_final_restore_err:
                logger.warning(f"[CONFIG] ⚠️ Не удалось восстановить таймфрейм (финальный): {tf_final_restore_err}")
        
        from bot_engine.bot_config import DEFAULT_AUTO_BOT_CONFIG
        
        # ✅ ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: читаем значение напрямую из файла для сравнения
        try:
            config_file_path_check = config_file_path
            if config_file_path_check and os.path.exists(config_file_path_check):
                with open(config_file_path_check, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Ищем значение leverage в DEFAULT_AUTO_BOT_CONFIG
                    leverage_match = re.search(r"'leverage':\s*(\d+)", content)
                    if leverage_match:
                        leverage_from_file_direct = int(leverage_match.group(1))
                        leverage_from_module = DEFAULT_AUTO_BOT_CONFIG.get('leverage')
                        if leverage_from_file_direct != leverage_from_module:
                            logger.error(f"[CONFIG] ❌ РАСХОЖДЕНИЕ! leverage в файле: {leverage_from_file_direct}x, в модуле: {leverage_from_module}x")
                            # Принудительно обновляем значение из файла
                            DEFAULT_AUTO_BOT_CONFIG['leverage'] = leverage_from_file_direct
                            logger.info(f"[CONFIG] ✅ Исправлено: leverage обновлен до {leverage_from_file_direct}x из файла")
        except Exception as e:
            pass  # Игнорируем ошибки проверки leverage

        # ✅ ЕДИНСТВЕННЫЙ источник истины: bot_engine/bot_config.py
        # Все настройки загружаются ТОЛЬКО из файла, БД не используется для auto_bot_config
        merged_config = DEFAULT_AUTO_BOT_CONFIG.copy()
        
        # ✅ Логируем leverage только при первой загрузке или при изменении (не спамим)
        leverage_from_file = merged_config.get('leverage')
        # Логируем только если это первая загрузка ИЛИ значение действительно изменилось
        should_log_leverage = (
            not hasattr(load_auto_bot_config, '_leverage_logged') or 
            (hasattr(load_auto_bot_config, '_last_leverage') and 
             load_auto_bot_config._last_leverage != leverage_from_file)
        )
        if should_log_leverage:
            logger.info(f"[CONFIG] ⚡ Кредитное плечо загружено из bot_config.py: {leverage_from_file}x")
            load_auto_bot_config._leverage_logged = True
            load_auto_bot_config._last_leverage = leverage_from_file
        
        # ✅ Проверяем, что значение действительно есть в конфиге (только при ошибке)
        if leverage_from_file is None:
            logger.error(f"[CONFIG] ❌ КРИТИЧЕСКАЯ ОШИБКА: leverage отсутствует в DEFAULT_AUTO_BOT_CONFIG!")
        
        # ✅ Однократная миграция при следующем запуске bots.py: списки из БД → data/coin_filters.json
        try:
            from bot_engine.coin_filters_config import load_coin_filters, run_coin_filters_migration_once
            run_coin_filters_migration_once()
            filters_data = load_coin_filters()
            if 'whitelist' in filters_data:
                merged_config['whitelist'] = filters_data['whitelist']
            if 'blacklist' in filters_data:
                merged_config['blacklist'] = filters_data['blacklist']
            if 'scope' in filters_data:
                merged_config['scope'] = filters_data['scope']
            if 'scope' not in merged_config:
                merged_config['scope'] = 'all'
            if not hasattr(load_auto_bot_config, '_filters_logged_once'):
                load_auto_bot_config._filters_logged_once = True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки фильтров из файлов: {e}")
            if 'whitelist' not in merged_config:
                merged_config['whitelist'] = []
            if 'blacklist' not in merged_config:
                merged_config['blacklist'] = []
            if 'scope' not in merged_config:
                merged_config['scope'] = 'all'
        
        # ✅ Логируем подробности ТОЛЬКО при первом вызове или при реальном изменении файла
        # (не логируем при принудительной перезагрузке модуля из API, чтобы не спамить)
        should_log_verbose = (reloaded and load_auto_bot_config._last_mtime != 0) or not getattr(load_auto_bot_config, '_logged_once', False)
        if should_log_verbose:
            # Детальное логирование убрано для уменьшения спама (переведено в DEBUG если нужно)
            load_auto_bot_config._logged_once = True
        # ✅ УБРАНО: Логирование "без изменений" создавало спам в логах при частых вызовах из API
        
        # ✅ ВСЕГДА обновляем bots_data, даже если файл не изменился
        # Это гарантирует, что данные всегда актуальны, особенно после принудительной перезагрузки модуля в API
        with bots_data_lock:
            bots_data['auto_bot_config'] = merged_config
            # ✅ Логирование leverage убрано (было слишком много спама) - логируется только при загрузке из файла
        
        # ✅ RSI пороги — логируем не чаще раза в 5 минут (при каждом цикле конфиг перезагружается → без throttle спам)
        _now = time.time()
        _last = getattr(load_auto_bot_config, '_rsi_log_last_time', 0)
        if (reloaded or not getattr(load_auto_bot_config, '_rsi_logged_once', False)) and (_now - _last >= 300):
            load_auto_bot_config._rsi_logged_once = True
            load_auto_bot_config._rsi_log_last_time = _now
            rl = merged_config.get('rsi_long_threshold')
            rs = merged_config.get('rsi_short_threshold')
            elw = merged_config.get('rsi_exit_long_with_trend')
            ela = merged_config.get('rsi_exit_long_against_trend')
            esw = merged_config.get('rsi_exit_short_with_trend')
            esa = merged_config.get('rsi_exit_short_against_trend')
            logger.info(f"[CONFIG] 📊 RSI пороги из конфига (используются ботами): вход LONG≤{rl} SHORT≥{rs}, выход LONG(with={elw}, against={ela}) SHORT(with={esw}, against={esa})")
        
        # Конфигурация загружена и обновлена в bots_data
            
    except Exception as e:
        logger.error(f" ❌ Ошибка загрузки конфигурации: {e}")
        import traceback
        logger.error(f" ❌ Трассировка ошибки:\n{traceback.format_exc()}")

def get_auto_bot_config():
    """Получает текущую конфигурацию Auto Bot из bots_data"""
    try:
        with bots_data_lock:
            return bots_data.get('auto_bot_config', DEFAULT_AUTO_BOT_CONFIG.copy())
    except Exception as e:
        logger.error(f" ❌ Ошибка получения конфигурации: {e}")
        return DEFAULT_AUTO_BOT_CONFIG.copy()


def get_config_snapshot(symbol=None, force_reload=False):
    """
    Возвращает полный срез настроек: глобальный конфиг, индивидуальные настройки и итоговый merge.

    Args:
        symbol (str|None): символ монеты для получения overrides.
        force_reload (bool): перезагрузить конфиг из источника истины перед чтением.

    Returns:
        dict: {
            'global': dict,
            'individual': dict|None,
            'merged': dict,
            'symbol': str|None,
            'timestamp': iso8601 str
        }
    """
    normalized_symbol = _normalize_symbol(symbol) if symbol else None

    if force_reload:
        try:
            if hasattr(load_auto_bot_config, '_last_mtime'):
                load_auto_bot_config._last_mtime = 0
        except Exception:
            pass
        load_auto_bot_config()
    else:
        # Если конфиг еще не загружен, загружаем его один раз
        with bots_data_lock:
            has_config = bool(bots_data.get('auto_bot_config'))
        if not has_config:
            load_auto_bot_config()

    with bots_data_lock:
        global_config = deepcopy(bots_data.get('auto_bot_config', DEFAULT_AUTO_BOT_CONFIG.copy()))

    individual_settings = None
    if normalized_symbol:
        individual_settings = get_individual_coin_settings(normalized_symbol)

    merged_config = deepcopy(global_config)
    if individual_settings:
        merged_config.update(individual_settings)

    snapshot = {
        'global': global_config,
        'individual': deepcopy(individual_settings) if individual_settings else None,
        'merged': merged_config,
        'symbol': normalized_symbol,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    return snapshot


# ===== Индивидуальные настройки монет =====

def _normalize_symbol(symbol: str) -> str:
    return symbol.upper() if symbol else symbol


def load_individual_coin_settings():
    """Загружает индивидуальные настройки монет из файла"""
    try:
        loaded = storage_load_individual_coin_settings() or {}
        normalized = {
            _normalize_symbol(symbol): settings
            for symbol, settings in loaded.items()
            if isinstance(settings, dict)
        }
        
        # ✅ ОДНОРАЗОВАЯ МИГРАЦИЯ: Автоматически включаем временной фильтр для всех монет
        # и гарантируем минимум 2 свечи (выполняется только один раз)
        try:
            from bot_engine.storage import _get_bots_database
            db = _get_bots_database()
            migration_flag = 'rsi_time_filter_auto_enable_migration_completed'
            
            # Проверяем, выполнена ли уже миграция
            migration_completed = db._get_metadata_flag(migration_flag, '0') == '1'
            
            if not migration_completed:
                # Миграция еще не выполнена - выполняем её
                updated_count = 0
                for symbol, settings in normalized.items():
                    settings_updated = False
                    
                    # Включаем временной фильтр, если он отключен или отсутствует
                    rsi_time_filter_enabled = settings.get('rsi_time_filter_enabled')
                    if rsi_time_filter_enabled is None or rsi_time_filter_enabled is False or rsi_time_filter_enabled == 0:
                        settings['rsi_time_filter_enabled'] = True
                        settings_updated = True
                    
                    # Гарантируем минимум 2 свечи
                    rsi_time_filter_candles = settings.get('rsi_time_filter_candles')
                    if rsi_time_filter_candles is not None and rsi_time_filter_candles < 2:
                        settings['rsi_time_filter_candles'] = max(2, rsi_time_filter_candles)
                        settings_updated = True
                    
                    if settings_updated:
                        updated_count += 1
                        settings['updated_at'] = datetime.now().isoformat()
                
                # Сохраняем обновленные настройки обратно в БД, если были изменения
                if updated_count > 0:
                    with bots_data_lock:
                        bots_data['individual_coin_settings'] = normalized
                    try:
                        save_individual_coin_settings()
                        # Устанавливаем флаг миграции как выполненной
                        db._set_metadata_flag(migration_flag, '1')
                        logger.info(f" ✅ Одноразовая миграция: автоматически включен временной фильтр для {updated_count} монет(ы)")
                    except Exception as save_exc:
                        logger.warning(f" ⚠️ Не удалось сохранить обновленные настройки: {save_exc}")
                else:
                    # Если нет изменений, все равно помечаем миграцию как выполненную
                    db._set_metadata_flag(migration_flag, '1')
                    with bots_data_lock:
                        bots_data['individual_coin_settings'] = normalized
            else:
                # Миграция уже выполнена - просто загружаем настройки
                with bots_data_lock:
                    bots_data['individual_coin_settings'] = normalized
        except Exception as migration_exc:
            # Если не удалось выполнить миграцию (например, БД еще не инициализирована),
            # просто загружаем настройки без миграции
            pass
            with bots_data_lock:
                bots_data['individual_coin_settings'] = normalized
        
        try:
            _individual_coin_settings_state['last_mtime'] = os.path.getmtime(INDIVIDUAL_COIN_SETTINGS_FILE)
        except OSError:
            _individual_coin_settings_state['last_mtime'] = None
        logger.info(f" ✅ Загружено индивидуальных настроек: {len(normalized)}")
        return deepcopy(normalized)
    except Exception as exc:
        logger.error(f" ❌ Ошибка загрузки индивидуальных настроек: {exc}")
        return {}


def save_individual_coin_settings():
    """Сохраняет индивидуальные настройки монет в БД"""
    try:
        with bots_data_lock:
            settings = {
                _normalize_symbol(symbol): settings
                for symbol, settings in bots_data.get('individual_coin_settings', {}).items()
                if isinstance(settings, dict)
            }
        return storage_save_individual_coin_settings(settings)
    except Exception as exc:
        logger.error(f" ❌ Ошибка сохранения индивидуальных настроек: {exc}")
        return False


def get_individual_coin_settings(symbol):
    """Возвращает индивидуальные настройки монеты (копию)"""
    if not symbol:
        return None
    normalized = _normalize_symbol(symbol)
    with bots_data_lock:
        settings = bots_data.get('individual_coin_settings', {}).get(normalized)
    if not settings:
        try:
            current_mtime = os.path.getmtime(INDIVIDUAL_COIN_SETTINGS_FILE)
        except OSError:
            current_mtime = None
        last_mtime = _individual_coin_settings_state.get('last_mtime')
        if current_mtime and current_mtime != last_mtime:
            pass
            load_individual_coin_settings()
            with bots_data_lock:
                settings = bots_data.get('individual_coin_settings', {}).get(normalized)
    return deepcopy(settings) if settings else None


def set_individual_coin_settings(symbol, settings, persist=True):
    """Устанавливает индивидуальные настройки монеты"""
    if not symbol or not isinstance(settings, dict):
        raise ValueError("Symbol and settings dictionary are required")
    normalized = _normalize_symbol(symbol)
    with bots_data_lock:
        bots_data.setdefault('individual_coin_settings', {})[normalized] = deepcopy(settings)
    if persist:
        save_individual_coin_settings()
    logger.info(f" 💾 Настройки для {normalized} обновлены")
    return get_individual_coin_settings(normalized)


def remove_individual_coin_settings(symbol, persist=True):
    """Удаляет индивидуальные настройки монеты"""
    if not symbol:
        return False
    normalized = _normalize_symbol(symbol)
    removed = False
    with bots_data_lock:
        coin_settings = bots_data.get('individual_coin_settings', {})
        if normalized in coin_settings:
            del coin_settings[normalized]
            removed = True
    if removed and persist:
        save_individual_coin_settings()
    if removed:
        logger.info(f" 🗑️ Настройки для {normalized} удалены")
    else:
        logger.info(f" ℹ️ Настройки для {normalized} отсутствуют")
    return removed


def copy_individual_coin_settings_to_all(source_symbol, target_symbols=None, persist=True):
    """Копирует индивидуальные настройки монеты ко всем целевым монетам. Если у монеты нет настроек — возвращает 0 без ошибки."""
    if not source_symbol:
        raise ValueError("Source symbol is required")
    normalized_source = _normalize_symbol(source_symbol)
    template = get_individual_coin_settings(normalized_source)
    if not template:
        logger.info(f" ℹ️ У монеты {normalized_source} нет индивидуальных настроек — копировать нечего")
        return 0

    with bots_data_lock:
        destination = bots_data.setdefault('individual_coin_settings', {})
        if target_symbols is None:
            target_symbols = list(coins_rsi_data.get('coins', {}).keys())
        copied = 0
        for symbol in target_symbols:
            normalized = _normalize_symbol(symbol)
            if not normalized or normalized == normalized_source:
                continue
            destination[normalized] = deepcopy(template)
            copied += 1

    if persist:
        save_individual_coin_settings()

    logger.info(f" 📋 Настройки {normalized_source} скопированы к {copied} монетам")
    return copied


def remove_all_individual_coin_settings(persist=True):
    """Удаляет все индивидуальные настройки всех монет"""
    removed_count = 0
    with bots_data_lock:
        coin_settings = bots_data.get('individual_coin_settings', {})
        if coin_settings:
            removed_count = len(coin_settings)
            coin_settings.clear()
    
    if removed_count > 0 and persist:
        save_individual_coin_settings()
        logger.info(f" 🗑️ Удалены индивидуальные настройки для всех {removed_count} монет")
    else:
        logger.info(f" ℹ️ Индивидуальные настройки отсутствуют")
    
    return removed_count


# ВАЖНО: load_auto_bot_config() теперь вызывается в if __name__ == '__main__'
# чтобы check_and_stop_existing_bots_processes() мог вывести свои сообщения первым


# ===== РЕЕСТР ПОЗИЦИЙ БОТОВ =====

def load_bot_positions_registry():
    """Загружает реестр позиций, открытых ботами (из БД или JSON)"""
    try:
        from bot_engine.storage import load_bot_positions_registry as storage_load_positions
        registry = storage_load_positions()
        if registry:
            logger.info(f" ✅ Загружен реестр позиций: {len(registry)} записей")
            return registry
        else:
            logger.info(f" 📁 Реестр позиций не найден, создаём новый")
            return {}
    except Exception as e:
        logger.error(f" ❌ Ошибка загрузки реестра: {e}")
        return {}


def save_bot_positions_registry(registry):
    """Сохраняет реестр позиций ботов (в БД или JSON)"""
    try:
        from bot_engine.storage import save_bot_positions_registry as storage_save_positions
        success = storage_save_positions(registry)
        if success:
            pass
        return success
    except Exception as e:
        logger.error(f" ❌ Ошибка сохранения реестра: {e}")
        return False


def register_bot_position(symbol, order_id, side, entry_price, quantity):
    """
    Регистрирует позицию, открытую ботом
    
    Args:
        symbol: Символ монеты
        order_id: ID ордера на бирже
        side: Сторона (LONG/SHORT)
        entry_price: Цена входа
        quantity: Количество
    """
    try:
        registry = load_bot_positions_registry()
        
        # ВАЖНО:
        # Таблица bot_positions_registry в БД хранится как {bot_id -> position_data} и bot_id UNIQUE.
        # Ключ реестра = SYMBOL_SIDE (например BTCUSDT_LONG, BTCUSDT_SHORT), чтобы по одному символу
        # могли быть два бота одновременно — лонг и шорт.
        # order_id сохраняем как поле.
        side_norm = (side or 'LONG').upper()
        registry_key = f"{str(symbol).upper()}_{side_norm}"
        registry[registry_key] = {
            'symbol': str(symbol).upper(),
            'side': side_norm,
            'entry_price': entry_price,
            'quantity': quantity,
            'opened_at': datetime.now().isoformat(),
            'managed_by_bot': True,
            'order_id': order_id,
        }
        
        save_bot_positions_registry(registry)
        logger.info(f" ✅ Зарегистрирована позиция: {symbol} {side}, order_id={order_id}")
        return True
    except Exception as e:
        logger.error(f" ❌ Ошибка регистрации позиции: {e}")
        return False


def unregister_bot_position(order_id):
    """Удаляет позицию из реестра (когда позиция закрыта).

    Параметр исторически назывался order_id, но фактически может быть:
    - order_id (если известен) — ищем запись по полю order_id;
    - bot_id (symbol_side, например BTCUSDT_LONG) — удаляем одну позицию;
    - symbol (например BTCUSDT) — удаляем обе позиции по символу (LONG и SHORT).
    """
    try:
        registry = load_bot_positions_registry()

        key = str(order_id).upper() if order_id is not None else None

        # 1) Прямое удаление по ключу (bot_id = symbol_side, например BTCUSDT_LONG)
        if key and key in registry:
            position_info = registry.pop(key)
            save_bot_positions_registry(registry)
            logger.info(f" ✅ Удалена позиция из реестра: {position_info.get('symbol')} (key={key})")
            return True

        # 2) Удаление по symbol: убрать все записи по символу (BTCUSDT -> BTCUSDT_LONG, BTCUSDT_SHORT)
        if key and not (key.endswith('_LONG') or key.endswith('_SHORT')):
            to_remove = [k for k in registry if k == f"{key}_LONG" or k == f"{key}_SHORT"]
            if to_remove:
                for k in to_remove:
                    registry.pop(k, None)
                save_bot_positions_registry(registry)
                logger.info(f" ✅ Удалены позиции из реестра по символу: {base} (keys={to_remove})")
                return True

        # 3) Поиск по order_id внутри записей
        if order_id is not None:
            for bot_id, info in list(registry.items()):
                if isinstance(info, dict) and info.get('order_id') == order_id:
                    registry.pop(bot_id, None)
                    save_bot_positions_registry(registry)
                    logger.info(f" ✅ Удалена позиция из реестра: {info.get('symbol')} (order_id={order_id})")
                    return True

        pass
        return False
    except Exception as e:
        logger.error(f" ❌ Ошибка удаления позиции из реестра: {e}")
        return False


def is_bot_position(order_id):
    """Проверяет, является ли позиция с данным order_id позицией бота"""
    try:
        registry = load_bot_positions_registry()
        key = str(order_id).upper() if order_id is not None else None
        if key and key in registry:
            return True
        if order_id is None:
            return False
        return any(isinstance(v, dict) and v.get('order_id') == order_id for v in registry.values())
    except Exception as e:
        logger.error(f" ❌ Ошибка проверки позиции: {e}")
        return False


def get_bot_position_info(order_id):
    """Получает информацию о позиции бота из реестра"""
    try:
        registry = load_bot_positions_registry()
        key = str(order_id).upper() if order_id is not None else None
        if key and key in registry:
            return registry.get(key)
        if order_id is None:
            return None
        for v in registry.values():
            if isinstance(v, dict) and v.get('order_id') == order_id:
                return v
        return None
    except Exception as e:
        logger.error(f" ❌ Ошибка получения информации о позиции: {e}")
        return None


def restore_lost_bots():
    """Восстанавливает потерянных ботов на основе реестра позиций"""
    try:
        registry = load_bot_positions_registry()
        if not registry:
            logger.info(" ℹ️ Реестр позиций пуст - нет ботов для восстановления")
            return []
        
        # Получаем позиции с биржи
        exch = get_exchange()
        if not exch:
            logger.error(" ❌ Биржа не инициализирована")
            return []
        
        exchange_positions = exch.get_positions()
        if not exchange_positions:
            logger.warning(" ⚠️ Не удалось получить позиции с биржи")
            return []
        
        # Преобразуем в словарь для быстрого поиска: ключ = (symbol, side), чтобы по одному символу
        # могли быть лонг и шорт одновременно
        if isinstance(exchange_positions, tuple):
            positions_list = exchange_positions[0] if exchange_positions else []
        else:
            positions_list = exchange_positions if exchange_positions else []
        
        def _pos_key(pos):
            sym = (pos.get('symbol') or '').upper()
            if sym and 'USDT' not in sym:
                sym = sym + 'USDT'
            side_raw = pos.get('side', '') or pos.get('position_side', '')
            side_n = (side_raw.upper() if side_raw else 'LONG')
            if side_n not in ('LONG', 'SHORT'):
                side_n = 'LONG' if str(side_raw).lower() in ('buy', 'long') else 'SHORT'
            return (sym, side_n)
        
        exchange_positions_dict = {
            _pos_key(pos): pos for pos in positions_list
            if abs(float(pos.get('size', 0))) > 0
        }
        
        restored_bots = []
        
        with bots_data_lock:
            # Проверяем каждую позицию в реестре (bot_id = symbol_side, например BTCUSDT_LONG)
            for bot_id, position_info in registry.items():
                symbol = (position_info.get('symbol') if isinstance(position_info, dict) else None) or bot_id
                # Из bot_id вида BTCUSDT_LONG извлекаем чистый symbol
                if symbol and ('_LONG' in symbol or '_SHORT' in symbol):
                    symbol = symbol.rsplit('_', 1)[0]
                if not symbol:
                    continue
                
                registry_side = (position_info.get('side', 'UNKNOWN') if isinstance(position_info, dict) else 'UNKNOWN').upper()
                if registry_side not in ('LONG', 'SHORT'):
                    registry_side = 'LONG'
                
                # Проверяем, есть ли уже бот для этой пары (symbol + side)
                if bot_id in bots_data['bots']:
                    continue
                
                pos_lookup = (symbol, registry_side)
                if pos_lookup not in exchange_positions_dict:
                    logger.info(f" 🗑️ Позиция {symbol} {registry_side} не найдена на бирже - удаляем из реестра")
                    try:
                        unregister_bot_position(bot_id)
                        logger.info(f" ✅ Позиция {bot_id} удалена из реестра")
                    except Exception as unreg_error:
                        logger.error(f" ❌ Ошибка удаления позиции {bot_id} из реестра: {unreg_error}")
                    continue
                
                exchange_position = exchange_positions_dict[pos_lookup]
                exchange_side = (exchange_position.get('side') or 'UNKNOWN').upper()
                if exchange_side not in ('LONG', 'SHORT'):
                    exchange_side = 'LONG' if str(exchange_position.get('side', '')).lower() in ('buy', 'long') else 'SHORT'
                
                if exchange_side != registry_side:
                    logger.warning(f" ⚠️ Несовпадение стороны для {symbol}: реестр={registry_side}, биржа={exchange_side}")
                    continue
                
                restored_bot = {
                    'symbol': symbol,
                    'status': 'in_position_long' if registry_side == 'LONG' else 'in_position_short',
                    'position': {
                        'side': registry_side,
                        'quantity': float(exchange_position.get('size', 0)),
                        'entry_price': position_info.get('entry_price', 0),
                        'order_id': position_info.get('order_id')
                    },
                    'entry_price': position_info.get('entry_price', 0),
                    'entry_time': position_info.get('opened_at', datetime.now().isoformat()),
                    'created_time': datetime.now().isoformat(),
                    'restored_from_registry': True,
                    'restoration_order_id': position_info.get('order_id')
                }
                
                bots_data['bots'][bot_id] = restored_bot
                restored_bots.append(bot_id)
                
                logger.info(f" ✅ Восстановлен бот {symbol} {registry_side} из реестра (bot_id={bot_id})")
        
        if restored_bots:
            logger.info(f" 🎯 Восстановлено {len(restored_bots)} ботов: {restored_bots}")
            # Сохраняем состояние
            try:
                with bots_data_lock:
                    bots_snapshot = deepcopy(bots_data.get('bots', {}))
                    config_snapshot = deepcopy(bots_data.get('auto_bot_config', {}))
                storage_save_bots_state(bots_snapshot, config_snapshot)
            except Exception as save_error:
                logger.error(f" ❌ Не удалось сохранить состояние после восстановления: {save_error}")
        else:
            logger.info(" ℹ️ Ботов для восстановления не найдено")
        
        return restored_bots
        
    except Exception as e:
        logger.error(f" ❌ Ошибка восстановления ботов: {e}")
        return []

# ✅ ИСПРАВЛЕНИЕ: Загружаем зрелые монеты при импорте модуля
try:
    load_mature_coins_storage()
    logger.info(f" ✅ Загружено {len(mature_coins_storage)} зрелых монет при импорте")
except Exception as e:
    logger.error(f" ❌ Ошибка загрузки зрелых монет: {e}")


def open_position_for_bot(symbol, side, volume_value, current_price, take_profit_price=None):
    """
    Открывает позицию для бота с правильным расчетом количества в USDT
    
    Args:
        symbol (str): Символ монеты (например, 'OG')
        side (str): Сторона ('LONG' или 'SHORT')
        volume_value (float): Объем сделки в USDT
        current_price (float): Текущая цена
        take_profit_price (float, optional): Цена Take Profit
        
    Returns:
        dict: Результат открытия позиции с success, order_id, message
    """
    try:
        exch = get_exchange()
        if not exch:
            logger.error(f" {symbol}: ❌ Биржа не инициализирована")
            return {'success': False, 'error': 'Exchange not initialized'}
        
        logger.info(f" {symbol}: Открываем {side} позицию на {volume_value} USDT @ {current_price}")
        
        # Получаем leverage из конфига бота (индивидуальные настройки или глобальный конфиг)
        leverage = None
        try:
            with bots_data_lock:
                bot_data = bots_data.get('bots', {}).get(symbol, {})
                if bot_data and 'leverage' in bot_data:
                    leverage = bot_data.get('leverage')
                else:
                    # Пробуем получить из индивидуальных настроек или глобального конфига
                    individual_settings = get_individual_coin_settings(symbol)
                    if individual_settings and 'leverage' in individual_settings:
                        leverage = individual_settings.get('leverage')
                    else:
                        auto_bot_config = bots_data.get('auto_bot_config', {})
                        leverage = auto_bot_config.get('leverage')
        except Exception as e:
            pass
        
        logger.info(f" {symbol}: 📊 Используемое плечо из конфига: {leverage}x (для open_position_for_bot)")
        
        # Вызываем place_order с правильными параметрами
        # quantity передаем в USDT (не в монетах!)
        result = exch.place_order(
            symbol=symbol,
            side=side,
            quantity=volume_value,  # ⚡ Количество в USDT!
            order_type='market',
            take_profit=take_profit_price,
            leverage=leverage
        )
        
        if result and result.get('success'):
            order_id = result.get('order_id')
            logger.info(f" {symbol}: ✅ Позиция {side} открыта успешно, order_id={order_id}")
            
            # Регистрируем позицию в реестре
            register_bot_position(symbol, order_id, side, current_price, volume_value)
            
            return result
        else:
            error_msg = result.get('message', 'Unknown error') if result else 'No response'
            logger.error(f" {symbol}: ❌ Ошибка открытия позиции: {error_msg}")
            return {'success': False, 'error': error_msg}
            
    except Exception as e:
        logger.error(f" {symbol}: ❌ Ошибка открытия позиции: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def close_position_for_bot(symbol, position_side, reason='Manual close'):
    """
    Закрывает позицию для бота
    
    Args:
        symbol (str): Символ монеты (например, 'OG')
        position_side (str): Сторона позиции ('LONG' или 'SHORT')
        reason (str): Причина закрытия
        
    Returns:
        dict: Результат закрытия позиции с success, message
    """
    try:
        exch = get_exchange()
        if not exch:
            logger.error(f" {symbol}: ❌ Биржа не инициализирована")
            return {'success': False, 'error': 'Exchange not initialized'}
        
        logger.info(f" {symbol}: Закрываем {position_side} позицию (причина: {reason})")
        
        # Получаем размер позиции с биржи перед закрытием
        position_size = None
        try:
            positions = exch.get_positions()
            if isinstance(positions, tuple):
                positions_list = positions[0] if positions else []
            else:
                positions_list = positions if positions else []
            
            # Преобразуем position_side в формат биржи для сравнения
            side_for_exchange = 'Long' if position_side in ['LONG', 'Long'] else 'Short' if position_side in ['SHORT', 'Short'] else position_side
            
            for pos in positions_list:
                if pos.get('symbol', '').replace('USDT', '') == symbol:
                    pos_side = 'Long' if pos.get('side') == 'Buy' else 'Short'
                    if pos_side == side_for_exchange and abs(float(pos.get('size', 0))) > 0:
                        position_size = abs(float(pos.get('size', 0)))
                        logger.info(f" {symbol}: Найден размер позиции на бирже: {position_size}")
                        break
        except Exception as e:
            logger.error(f" {symbol}: ⚠️ Ошибка получения размера позиции с биржи: {e}")
        
        if not position_size:
            logger.error(f" {symbol}: ❌ Не удалось определить размер позиции")
            return {'success': False, 'error': 'Position size not found on exchange'}
        
        # Вызываем close_position с размером
        result = exch.close_position(
            symbol=symbol,
            size=position_size,
            side=side_for_exchange
        )
        
        if result and result.get('success'):
            logger.info(f" {symbol}: ✅ Позиция {position_side} закрыта успешно")
            return result
        else:
            error_msg = result.get('message', 'Unknown error') if result else 'No response'
            logger.error(f" {symbol}: ❌ Ошибка закрытия позиции: {error_msg}")
            return {'success': False, 'error': error_msg}
            
    except Exception as e:
        logger.error(f" {symbol}: ❌ Ошибка закрытия позиции: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

