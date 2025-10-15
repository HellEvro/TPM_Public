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
    from bot_engine.maturity_checker import (
        check_coin_maturity, check_coin_maturity_with_storage,
        check_coin_maturity_stored_or_verify, is_coin_mature_stored,
        add_mature_coin_to_storage, remove_mature_coin_from_storage,
        update_mature_coin_verification, get_mature_coins_storage,
        set_mature_coins_storage, clear_mature_coins_storage as clear_mature_storage,
        mature_coins_storage, mature_coins_lock
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
AUTO_BOT_CONFIG_FILE = 'data/auto_bot_config.json'

# Константы для обновления позиций
BOT_STATUS_UPDATE_INTERVAL = 30  # 30 секунд - интервал обновления детальной информации о состоянии ботов
STOP_LOSS_SETUP_INTERVAL = 300  # 5 минут - интервал установки недостающих стоп-лоссов
POSITION_SYNC_INTERVAL = 30  # 10 минут - интервал синхронизации позиций с биржей
INACTIVE_BOT_CLEANUP_INTERVAL = 600  # 10 минут - интервал проверки и удаления неактивных ботов
INACTIVE_BOT_TIMEOUT = 600  # 10 минут - время ожидания перед удалением бота без реальных позиций на бирже

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

# Константы для фильтрации зрелости монет
MIN_CANDLES_FOR_MATURITY = 200  # Минимум свечей для зрелой монеты (50 дней на 6H)
MIN_RSI_LOW = 35   # Минимальный достигнутый RSI
MAX_RSI_HIGH = 65  # Максимальный достигнутый RSI
MIN_VOLATILITY_THRESHOLD = 0.05  # Минимальная волатильность (5%)

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

# Добавляем файловый логгер для сохранения в файл
file_handler = logging.FileHandler('logs/bots.log', encoding='utf-8')
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

# Торговые параметры RSI согласно техзаданию (настраиваемые)
RSI_OVERSOLD = 29  # Зона покупки (LONG при RSI <= 29)
RSI_OVERBOUGHT = 71  # Зона продажи (SHORT при RSI >= 71)
RSI_EXIT_LONG = 65  # Выход из лонга (при RSI >= 65)
RSI_EXIT_SHORT = 35  # Выход из шорта (при RSI <= 35)

# EMA параметры для анализа тренда 6H
EMA_FAST = 50
EMA_SLOW = 200
TREND_CONFIRMATION_BARS = 3

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
def load_auto_bot_config():
    """Загружает конфигурацию Auto Bot из файла"""
    try:
        config_file = 'data/auto_bot_config.json'
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
                with bots_data_lock:
                    bots_data['auto_bot_config'].update(saved_config)
                    # ВАЖНО: Всегда отключаем автобот при запуске!
                    bots_data['auto_bot_config']['enabled'] = False
                logger.info(f"[CONFIG] ✅ Загружена конфигурация Auto Bot из {config_file}")
                logger.info(f"[CONFIG] 🔒 Auto Bot принудительно выключен при запуске")
        else:
            logger.info(f"[CONFIG] 📁 Файл конфигурации {config_file} не найден, используем дефолтные настройки")
            # Auto Bot уже выключен в дефолтной конфигурации
    except Exception as e:
        logger.error(f"[CONFIG] ❌ Ошибка загрузки конфигурации: {e}")

# ВАЖНО: load_auto_bot_config() теперь вызывается в if __name__ == '__main__'
# чтобы check_and_stop_existing_bots_processes() мог вывести свои сообщения первым

