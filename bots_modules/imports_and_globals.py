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
from datetime import datetime
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
    from bot_engine.utils.ema_utils import calculate_ema
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
        save_auto_bot_config as storage_save_auto_bot_config,
        load_auto_bot_config as storage_load_auto_bot_config,
        save_mature_coins, load_mature_coins,
        save_optimal_ema, load_optimal_ema,
        save_process_state as storage_save_process_state,
        load_process_state as storage_load_process_state,
        save_system_config as storage_save_system_config,
        load_system_config as storage_load_system_config
    )
    from bot_engine.signal_processor import get_effective_signal, check_autobot_filters, process_auto_bot_signals
    from bot_engine.optimal_ema_manager import (
        load_optimal_ema_data, get_optimal_ema_periods,
        update_optimal_ema_data, save_optimal_ema_periods,
        optimal_ema_data
    )
    MODULES_AVAILABLE = True
    print("[OK] New bot_engine modules loaded successfully")
except ImportError as e:
    print(f"[WARNING] Failed to load new bot_engine modules: {e}")
    print("[WARNING] Using legacy functions from bots.py")
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
                else:
                    process_to_stop = None
                
                if process_to_stop and process_to_stop != current_pid:
                    try:
                        proc = psutil.Process(process_to_stop)
                        proc_info = proc.as_dict(attrs=['pid', 'name', 'cmdline', 'create_time'])
                        
                        print(f"🎯 Найден процесс на порту 5001:")
                        print(f"   PID: {proc_info['pid']}")
                        print(f"   Команда: {' '.join(proc_info['cmdline'][:3]) if proc_info['cmdline'] else 'N/A'}...")
                        print()
                        
                        print(f"🔧 Останавливаем процесс {process_to_stop}...")
                        proc.terminate()
                        
                        try:
                            proc.wait(timeout=5)
                            print(f"✅ Процесс {process_to_stop} остановлен")
                        except psutil.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                            print(f"🔴 Процесс {process_to_stop} принудительно остановлен")
                        
                        print("\n⏳ Ожидание освобождения порта 5001...")
                        for i in range(10):
                            time.sleep(1)
                            try:
                                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                sock.settimeout(1)
                                result = sock.connect_ex(('127.0.0.1', 5001))
                                sock.close()
                                
                                if result != 0:
                                    print("✅ Порт 5001 освобожден")
                                    break
                            except:
                                pass
                            
                            if i == 9:
                                print("❌ Порт 5001 все еще занят!")
                                print("⚠️  Возможно нужно вручную остановить процесс")
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
AUTO_BOT_CONFIG_FILE = 'data/auto_bot_config.json'

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

# Создаем папку для данных если её нет
os.makedirs('data', exist_ok=True)

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
setup_color_logging()

# Добавляем файловый логгер с ротацией для сохранения в файл
from utils.log_rotation import RotatingFileHandlerWithSizeLimit

file_handler = RotatingFileHandlerWithSizeLimit(
    filename='logs/bots.log',
    max_bytes=10 * 1024 * 1024,  # 10MB
    backup_count=0,  # Перезаписываем файл
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('[BOTS] %(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Получаем корневой логгер и добавляем файловый обработчик
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)

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
            summary = f"[SUMMARY] Предыдущее сообщение повторилось {cache_entry['count']} раз"
            logger.info(f"[{category.upper()}] {summary}")
        
        cache_entry['message'] = message
        cache_entry['count'] = 1
        cache_entry['last_log'] = current_time
        return True, message

logger = logging.getLogger('BotsService')

# Отключаем HTTP логи Werkzeug для чистоты консоли
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.WARNING)  # Показывать только warnings и errors

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
        logger.error(f"[JSON_ERROR] Ошибка JSON сериализации: {e}")
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
    'failed_coins': 0
}

# Модель данных для ботов
bots_data = {
    'bots': {},  # {symbol: bot_config}
    'auto_bot_config': DEFAULT_AUTO_BOT_CONFIG.copy(),  # Используем дефолтную конфигурацию
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
def load_auto_bot_config(force_disable=False):
    """Загружает конфигурацию Auto Bot из bot_config.py
    
    ✅ ЕДИНСТВЕННЫЙ источник истины: bot_engine/bot_config.py
    - Все настройки в Python-файле с комментариями
    - Можно редактировать руками с подробными пояснениями
    - Система читает напрямую из DEFAULT_AUTO_BOT_CONFIG
    """
    try:
        from bot_engine.bot_config import DEFAULT_AUTO_BOT_CONFIG
        
        with bots_data_lock:
            # Загружаем конфигурацию напрямую из bot_config.py
            bots_data['auto_bot_config'] = DEFAULT_AUTO_BOT_CONFIG.copy()
            
            # Отключаем автобот только при принудительном вызове (при запуске сервера)
            if force_disable:
                bots_data['auto_bot_config']['enabled'] = False
                logger.info(f"[CONFIG] 🔒 Auto Bot принудительно выключен при запуске")
        
        logger.info(f"[CONFIG] ✅ Загружена конфигурация Auto Bot из bot_config.py")
            
    except Exception as e:
        logger.error(f"[CONFIG] ❌ Ошибка загрузки конфигурации: {e}")

# ВАЖНО: load_auto_bot_config() теперь вызывается в if __name__ == '__main__'
# чтобы check_and_stop_existing_bots_processes() мог вывести свои сообщения первым


# ===== РЕЕСТР ПОЗИЦИЙ БОТОВ =====

def load_bot_positions_registry():
    """Загружает реестр позиций, открытых ботами"""
    try:
        if os.path.exists(BOTS_POSITIONS_REGISTRY_FILE):
            with open(BOTS_POSITIONS_REGISTRY_FILE, 'r', encoding='utf-8') as f:
                registry = json.load(f)
                logger.info(f"[REGISTRY] ✅ Загружен реестр позиций: {len(registry)} записей")
                return registry
        else:
            logger.info(f"[REGISTRY] 📁 Реестр позиций не найден, создаём новый")
            return {}
    except Exception as e:
        logger.error(f"[REGISTRY] ❌ Ошибка загрузки реестра: {e}")
        return {}


def save_bot_positions_registry(registry):
    """Сохраняет реестр позиций ботов"""
    try:
        with open(BOTS_POSITIONS_REGISTRY_FILE, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        logger.debug(f"[REGISTRY] ✅ Реестр позиций сохранён: {len(registry)} записей")
        return True
    except Exception as e:
        logger.error(f"[REGISTRY] ❌ Ошибка сохранения реестра: {e}")
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
        
        # Ключ — order_id, значение — информация о позиции
        registry[order_id] = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'quantity': quantity,
            'opened_at': datetime.now().isoformat(),
            'managed_by_bot': True
        }
        
        save_bot_positions_registry(registry)
        logger.info(f"[REGISTRY] ✅ Зарегистрирована позиция: {symbol} {side}, order_id={order_id}")
        return True
    except Exception as e:
        logger.error(f"[REGISTRY] ❌ Ошибка регистрации позиции: {e}")
        return False


def unregister_bot_position(order_id):
    """Удаляет позицию из реестра (когда позиция закрыта)"""
    try:
        registry = load_bot_positions_registry()
        
        if order_id in registry:
            position_info = registry.pop(order_id)
            save_bot_positions_registry(registry)
            logger.info(f"[REGISTRY] ✅ Удалена позиция из реестра: {position_info.get('symbol')} (order_id={order_id})")
            return True
        else:
            logger.debug(f"[REGISTRY] ⚠️ Позиция с order_id={order_id} не найдена в реестре")
            return False
    except Exception as e:
        logger.error(f"[REGISTRY] ❌ Ошибка удаления позиции из реестра: {e}")
        return False


def is_bot_position(order_id):
    """Проверяет, является ли позиция с данным order_id позицией бота"""
    try:
        registry = load_bot_positions_registry()
        return order_id in registry
    except Exception as e:
        logger.error(f"[REGISTRY] ❌ Ошибка проверки позиции: {e}")
        return False


def get_bot_position_info(order_id):
    """Получает информацию о позиции бота из реестра"""
    try:
        registry = load_bot_positions_registry()
        return registry.get(order_id)
    except Exception as e:
        logger.error(f"[REGISTRY] ❌ Ошибка получения информации о позиции: {e}")
        return None


def restore_lost_bots():
    """Восстанавливает потерянных ботов на основе реестра позиций"""
    try:
        registry = load_bot_positions_registry()
        if not registry:
            logger.info("[REGISTRY] ℹ️ Реестр позиций пуст - нет ботов для восстановления")
            return []
        
        # Получаем позиции с биржи
        exch = get_exchange()
        if not exch:
            logger.error("[REGISTRY] ❌ Биржа не инициализирована")
            return []
        
        exchange_positions = exch.get_positions()
        if not exchange_positions:
            logger.warning("[REGISTRY] ⚠️ Не удалось получить позиции с биржи")
            return []
        
        # Преобразуем в словарь для быстрого поиска
        if isinstance(exchange_positions, tuple):
            positions_list = exchange_positions[0] if exchange_positions else []
        else:
            positions_list = exchange_positions if exchange_positions else []
        
        exchange_positions_dict = {pos.get('symbol'): pos for pos in positions_list if abs(float(pos.get('size', 0))) > 0}
        
        restored_bots = []
        
        with bots_data_lock:
            # Проверяем каждую позицию в реестре
            for order_id, position_info in registry.items():
                symbol = position_info.get('symbol')
                if not symbol:
                    continue
                
                # Проверяем, есть ли уже бот для этой монеты
                if symbol in bots_data['bots']:
                    continue
                
                # Проверяем, есть ли позиция на бирже
                if symbol not in exchange_positions_dict:
                    logger.debug(f"[REGISTRY] 🔍 Позиция {symbol} (order_id={order_id}) не найдена на бирже - возможно закрыта")
                    continue
                
                exchange_position = exchange_positions_dict[symbol]
                exchange_side = exchange_position.get('side', 'UNKNOWN')
                registry_side = position_info.get('side', 'UNKNOWN')
                
                # Проверяем совпадение стороны
                if exchange_side != registry_side:
                    logger.warning(f"[REGISTRY] ⚠️ Несовпадение стороны для {symbol}: реестр={registry_side}, биржа={exchange_side}")
                    continue
                
                # Создаём восстановленного бота
                restored_bot = {
                    'symbol': symbol,
                    'status': 'IN_POSITION_LONG' if registry_side == 'LONG' else 'IN_POSITION_SHORT',
                    'position': {
                        'side': registry_side,
                        'quantity': float(exchange_position.get('size', 0)),
                        'entry_price': position_info.get('entry_price', 0),
                        'order_id': order_id
                    },
                    'entry_price': position_info.get('entry_price', 0),
                    'entry_time': position_info.get('opened_at', datetime.now().isoformat()),
                    'created_time': datetime.now().isoformat(),
                    'restored_from_registry': True,
                    'restoration_order_id': order_id
                }
                
                bots_data['bots'][symbol] = restored_bot
                restored_bots.append(symbol)
                
                logger.info(f"[REGISTRY] ✅ Восстановлен бот {symbol} (order_id={order_id}) из реестра")
        
        if restored_bots:
            logger.info(f"[REGISTRY] 🎯 Восстановлено {len(restored_bots)} ботов: {restored_bots}")
            # Сохраняем состояние
            save_bots_state()
        else:
            logger.info("[REGISTRY] ℹ️ Ботов для восстановления не найдено")
        
        return restored_bots
        
    except Exception as e:
        logger.error(f"[REGISTRY] ❌ Ошибка восстановления ботов: {e}")
        return []

