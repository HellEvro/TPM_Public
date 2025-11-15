"""Функции кэширования, синхронизации и управления состоянием

Включает:
- Функции работы с RSI кэшом
- Сохранение/загрузка состояния ботов
- Синхронизация с биржей
- Обновление позиций
- Управление зрелыми монетами
"""

import os
import json
import time
import threading
import logging
import importlib
from datetime import datetime
from pathlib import Path
import copy
import math

logger = logging.getLogger('BotsService')

# Импорт SystemConfig
from bot_engine.bot_config import SystemConfig
from bot_engine.bot_history import log_position_closed as history_log_position_closed

# Константы теперь в SystemConfig

# Импортируем глобальные переменные из imports_and_globals
try:
    from bots_modules.imports_and_globals import (
        bots_data_lock, bots_data, rsi_data_lock, coins_rsi_data,
        bots_cache_data, bots_cache_lock, process_state, exchange,
        mature_coins_storage, mature_coins_lock, BOT_STATUS,
        DEFAULT_AUTO_BOT_CONFIG, RSI_CACHE_FILE, PROCESS_STATE_FILE,
        SYSTEM_CONFIG_FILE, BOTS_STATE_FILE, DEFAULT_CONFIG_FILE,
        should_log_message, get_coin_processing_lock, get_exchange,
        save_individual_coin_settings
    )
    # MATURE_COINS_FILE определен в maturity.py
    try:
        from bots_modules.maturity import MATURE_COINS_FILE, save_mature_coins_storage
    except:
        MATURE_COINS_FILE = 'data/mature_coins.json'
        def save_mature_coins_storage():
            pass  # Fallback function
    
    # Заглушка для ensure_exchange_initialized (избегаем циклического импорта)
    def ensure_exchange_initialized():
        """Заглушка, будет переопределена при первом использовании"""
        try:
            from bots_modules.init_functions import ensure_exchange_initialized as real_func
            # Заменяем глобальную функцию на настоящую
            globals()['ensure_exchange_initialized'] = real_func
            return real_func()
        except:
            return exchange is not None
except ImportError as e:
    print(f"Warning: Could not import globals in sync_and_cache: {e}")
    # Создаем заглушки
    bots_data_lock = threading.Lock()
    bots_data = {}
    rsi_data_lock = threading.Lock()
    coins_rsi_data = {}
    bots_cache_data = {}
    bots_cache_lock = threading.Lock()
    process_state = {}
    exchange = None
    mature_coins_storage = {}
    mature_coins_lock = threading.Lock()
    BOT_STATUS = {}
    DEFAULT_AUTO_BOT_CONFIG = {}
    RSI_CACHE_FILE = 'data/rsi_cache.json'
    PROCESS_STATE_FILE = 'data/process_state.json'
    SYSTEM_CONFIG_FILE = 'data/system_config.json'
    BOTS_STATE_FILE = 'data/bots_state.json'
    MATURE_COINS_FILE = 'data/mature_coins.json'
    DEFAULT_CONFIG_FILE = 'data/default_auto_bot_config.json'
    def should_log_message(cat, msg, interval=60):
        return (True, msg)

# Карта соответствия ключей UI и атрибутов SystemConfig
SYSTEM_CONFIG_FIELD_MAP = {
    'rsi_update_interval': 'RSI_UPDATE_INTERVAL',
    'auto_save_interval': 'AUTO_SAVE_INTERVAL',
    'debug_mode': 'DEBUG_MODE',
    'auto_refresh_ui': 'AUTO_REFRESH_UI',
    'refresh_interval': 'UI_REFRESH_INTERVAL',
    'position_sync_interval': 'POSITION_SYNC_INTERVAL',
    'inactive_bot_cleanup_interval': 'INACTIVE_BOT_CLEANUP_INTERVAL',
    'inactive_bot_timeout': 'INACTIVE_BOT_TIMEOUT',
    'stop_loss_setup_interval': 'STOP_LOSS_SETUP_INTERVAL',
    'enhanced_rsi_enabled': 'ENHANCED_RSI_ENABLED',
    'enhanced_rsi_require_volume_confirmation': 'ENHANCED_RSI_REQUIRE_VOLUME_CONFIRMATION',
    'enhanced_rsi_require_divergence_confirmation': 'ENHANCED_RSI_REQUIRE_DIVERGENCE_CONFIRMATION',
    'enhanced_rsi_use_stoch_rsi': 'ENHANCED_RSI_USE_STOCH_RSI',
    'rsi_extreme_zone_timeout': 'RSI_EXTREME_ZONE_TIMEOUT',
    'rsi_extreme_oversold': 'RSI_EXTREME_OVERSOLD',
    'rsi_extreme_overbought': 'RSI_EXTREME_OVERBOUGHT',
    'rsi_volume_confirmation_multiplier': 'RSI_VOLUME_CONFIRMATION_MULTIPLIER',
    'rsi_divergence_lookback': 'RSI_DIVERGENCE_LOOKBACK',
    'trend_confirmation_bars': 'TREND_CONFIRMATION_BARS',
    'trend_min_confirmations': 'TREND_MIN_CONFIRMATIONS',
    'trend_require_slope': 'TREND_REQUIRE_SLOPE',
    'trend_require_price': 'TREND_REQUIRE_PRICE',
    'trend_require_candles': 'TREND_REQUIRE_CANDLES'
}


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_timestamp(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        if value > 1e12:
            return value / 1000.0
        return value
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return None
        try:
            return datetime.fromisoformat(raw_value.replace('Z', '')).timestamp()
        except ValueError:
            try:
                return _normalize_timestamp(float(raw_value))
            except ValueError:
                return None
    return None


def _timestamp_to_iso(raw_value):
    ts = _normalize_timestamp(raw_value)
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).isoformat()


def _needs_price_update(position_side, desired_price, existing_price, tolerance=1e-6):
    if desired_price is None:
        return False
    if existing_price is None:
        return True
    if (position_side or '').upper() == 'LONG':
        return desired_price > existing_price + tolerance
    return desired_price < existing_price - tolerance


def _select_stop_loss_price(position_side, entry_price, current_price, config, break_even_price, trailing_price):
    entry_price = _safe_float(entry_price)
    current_price = _safe_float(current_price, entry_price)
    stops = []

    sl_percent = _safe_float(config.get('max_loss_percent', config.get('stop_loss_percent')), 0.0)
    if sl_percent and sl_percent > 0 and entry_price:
        if (position_side or '').upper() == 'LONG':
            stops.append(entry_price * (1 - sl_percent / 100.0))
        else:
            stops.append(entry_price * (1 + sl_percent / 100.0))

    if break_even_price is not None:
        stops.append(_safe_float(break_even_price))
    if trailing_price is not None:
        stops.append(_safe_float(trailing_price))

    stops = [price for price in stops if price is not None]
    if not stops:
        return None

    if (position_side or '').upper() == 'LONG':
        candidate = max(stops)
        if current_price is not None:
            candidate = min(candidate, current_price)
    else:
        candidate = min(stops)
        if current_price is not None:
            candidate = max(candidate, current_price)
    return candidate


def _select_take_profit_price(position_side, entry_price, config, trailing_take_price):
    entry_price = _safe_float(entry_price)
    trailing_take_price = _safe_float(trailing_take_price)
    if trailing_take_price:
        return trailing_take_price

    tp_percent = _safe_float(config.get('take_profit_percent'), 0.0)
    if not tp_percent or tp_percent <= 0 or not entry_price:
        return None

    if (position_side or '').upper() == 'LONG':
        return entry_price * (1 + tp_percent / 100.0)
    return entry_price * (1 - tp_percent / 100.0)


def _apply_protection_state_to_bot_data(bot_data, state):
    if not state or bot_data is None:
        return

    bot_data['max_profit_achieved'] = state.max_profit_percent
    bot_data['break_even_activated'] = state.break_even_activated
    bot_data['break_even_stop_price'] = state.break_even_stop_price
    bot_data['trailing_active'] = state.trailing_active
    bot_data['trailing_reference_price'] = state.trailing_reference_price
    bot_data['trailing_stop_price'] = state.trailing_stop_price
    bot_data['trailing_take_profit_price'] = state.trailing_take_profit_price
    bot_data['trailing_last_update_ts'] = state.trailing_last_update_ts


def get_system_config_snapshot():
    """Возвращает текущие значения SystemConfig в формате, ожидаемом UI."""
    snapshot = {}
    for key, attr in SYSTEM_CONFIG_FIELD_MAP.items():
        snapshot[key] = getattr(SystemConfig, attr, None)
    return snapshot


def _compute_margin_based_trailing(side: str,
                                   entry_price: float,
                                   current_price: float,
                                   position_qty: float,
                                   leverage: float,
                                   realized_pnl: float,
                                   profit_percent: float,
                                   max_profit_percent: float,
                                   trailing_activation_percent: float,
                                   trailing_distance_percent: float,
                                   trailing_profit_usdt_max: float = 0.0):
    """
    Рассчитывает параметры трейлинг-стопа на основе маржи сделки.

    Returns dict:
        {
            'active': bool,
            'stop_price': float | None,
            'locked_profit_usdt': float,
            'activation_threshold_usdt': float,
            'activation_profit_usdt': float,
            'profit_usdt': float,
            'margin_usdt': float
        }
    """
    try:
        normalized_side = (side or '').upper()
        entry_price = float(entry_price or 0.0)
        current_price = float(current_price or 0.0)
        position_qty = abs(float(position_qty or 0.0))
        leverage = float(leverage or 1.0)
        if leverage <= 0:
            leverage = 1.0
        realized_pnl = float(realized_pnl or 0.0)
        trailing_activation_percent = float(trailing_activation_percent or 0.0)
        trailing_distance_percent = float(trailing_distance_percent or 0.0)
        trailing_profit_usdt_max = float(trailing_profit_usdt_max or 0.0)
    except (ValueError, TypeError):
        return {
            'active': False,
            'stop_price': None,
            'locked_profit_usdt': 0.0,
            'activation_threshold_usdt': 0.0,
            'activation_profit_usdt': 0.0,
            'profit_usdt': 0.0,
            'profit_usdt_max': 0.0,
            'margin_usdt': 0.0,
            'trailing_step_usdt': 0.0,
            'trailing_step_price': 0.0,
            'steps': 0
        }

    if entry_price <= 0 or position_qty <= 0:
        return {
            'active': False,
            'stop_price': None,
            'locked_profit_usdt': 0.0,
            'activation_threshold_usdt': 0.0,
            'activation_profit_usdt': 0.0,
            'profit_usdt': 0.0,
            'profit_usdt_max': trailing_profit_usdt_max,
            'margin_usdt': 0.0,
            'trailing_step_usdt': 0.0,
            'trailing_step_price': 0.0,
            'steps': 0
        }

    position_value = entry_price * position_qty
    margin_usdt = position_value / leverage if leverage else position_value

    profit_usdt = 0.0
    if normalized_side == 'LONG':
        profit_usdt = position_qty * max(0.0, current_price - entry_price)
    elif normalized_side == 'SHORT':
        profit_usdt = position_qty * max(0.0, entry_price - current_price)
    profit_usdt = float(profit_usdt)

    realized_abs = abs(realized_pnl)
    activation_from_config = margin_usdt * (trailing_activation_percent / 100.0)
    realized_times_three = realized_abs * 3.0
    if activation_from_config >= realized_times_three:
        activation_threshold_usdt = activation_from_config
    else:
        activation_threshold_usdt = realized_abs * 4.0
    activation_threshold_usdt = float(activation_threshold_usdt)

    trailing_profit_usdt_max = max(trailing_profit_usdt_max, profit_usdt)

    trailing_step_usdt = margin_usdt * (trailing_distance_percent / 100.0)
    trailing_step_usdt = max(trailing_step_usdt, 0.0)
    trailing_step_price = trailing_step_usdt / position_qty if position_qty > 0 else 0.0

    trailing_active = False
    if margin_usdt > 0 and activation_threshold_usdt > 0:
        trailing_active = trailing_profit_usdt_max >= activation_threshold_usdt

    locked_profit_usdt = realized_abs * 3.0
    if locked_profit_usdt < 0:
        locked_profit_usdt = 0.0

    steps = 0
    stop_price = None

    if trailing_active:
        prirost_max = max(0.0, trailing_profit_usdt_max - activation_threshold_usdt)
        if trailing_step_usdt > 0:
            steps = int(math.floor(prirost_max / trailing_step_usdt))
        locked_profit_total = locked_profit_usdt + steps * trailing_step_usdt
        locked_profit_total = min(locked_profit_total, trailing_profit_usdt_max)

        profit_per_coin = locked_profit_total / position_qty if position_qty > 0 else 0.0

        if normalized_side == 'LONG':
            stop_price = entry_price + profit_per_coin
            if current_price > 0:
                stop_price = min(stop_price, current_price)
            stop_price = max(stop_price, entry_price)
        elif normalized_side == 'SHORT':
            stop_price = entry_price - profit_per_coin
            if current_price > 0:
                stop_price = max(stop_price, current_price)
            stop_price = min(stop_price, entry_price)

        locked_profit_usdt = locked_profit_total

    return {
        'active': trailing_active,
        'stop_price': stop_price,
        'locked_profit_usdt': locked_profit_usdt,
        'activation_threshold_usdt': activation_threshold_usdt,
        'activation_profit_usdt': activation_threshold_usdt,
        'profit_usdt': profit_usdt,
        'profit_usdt_max': trailing_profit_usdt_max,
        'margin_usdt': margin_usdt,
        'trailing_step_usdt': trailing_step_usdt,
        'trailing_step_price': trailing_step_price,
        'steps': steps
    }
    def get_coin_processing_lock(symbol):
        return threading.Lock()
    def ensure_exchange_initialized():
        return exchange is not None
    def get_exchange():
        return exchange

def get_rsi_cache():
    """Получить кэшированные RSI данные"""
    global coins_rsi_data
    with rsi_data_lock:
        return coins_rsi_data.get('coins', {})

def save_rsi_cache():
    """Сохранить кэш RSI данных в файл"""
    try:
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция в Python
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'coins': coins_rsi_data.get('coins', {}),
            'stats': {
                'total_coins': len(coins_rsi_data.get('coins', {})),
                'successful_coins': coins_rsi_data.get('successful_coins', 0),
                'failed_coins': coins_rsi_data.get('failed_coins', 0)
            }
        }
        
        with open(RSI_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f" RSI данные для {len(cache_data['coins'])} монет сохранены в кэш")
        
    except Exception as e:
        logger.error(f" Ошибка сохранения RSI кэша: {str(e)}")

def load_rsi_cache():
    """Загрузить кэш RSI данных из файла"""
    global coins_rsi_data
    
    try:
        if not os.path.exists(RSI_CACHE_FILE):
            logger.info(" Файл RSI кэша не найден, будет создан при первом обновлении")
            return False
            
        with open(RSI_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Проверяем возраст кэша (не старше 6 часов)
        cache_timestamp = datetime.fromisoformat(cache_data['timestamp'])
        age_hours = (datetime.now() - cache_timestamp).total_seconds() / 3600
        
        if age_hours > 6:
            logger.warning(f" RSI кэш устарел ({age_hours:.1f} часов), будет обновлен")
            return False
        
        # Загружаем данные из кэша
        cached_coins = cache_data.get('coins', {})
        
        # Проверяем формат кэша (старый массив или новый словарь)
        if isinstance(cached_coins, list):
            # Старый формат - преобразуем массив в словарь
            coins_dict = {}
            for coin in cached_coins:
                if 'symbol' in coin:
                    coins_dict[coin['symbol']] = coin
            cached_coins = coins_dict
            logger.info(" Преобразован старый формат кэша (массив -> словарь)")
        
        with rsi_data_lock:
            coins_rsi_data.update({
                'coins': cached_coins,
                'successful_coins': cache_data.get('stats', {}).get('successful_coins', len(cached_coins)),
                'failed_coins': cache_data.get('stats', {}).get('failed_coins', 0),
                'total_coins': len(cached_coins),
                'last_update': datetime.now().isoformat(),  # Всегда используем текущее время
                'update_in_progress': False
            })
        
        logger.info(f" Загружено {len(cached_coins)} монет из RSI кэша (возраст: {age_hours:.1f}ч)")
        return True
        
    except Exception as e:
        logger.error(f" Ошибка загрузки RSI кэша: {str(e)}")
        return False

def save_default_config():
    """Сохраняет дефолтную конфигурацию в файл для восстановления"""
    try:
        with open(DEFAULT_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_AUTO_BOT_CONFIG, f, indent=2, ensure_ascii=False)
        
        logger.info(f" ✅ Дефолтная конфигурация сохранена в {DEFAULT_CONFIG_FILE}")
        return True
        
    except Exception as e:
        logger.error(f" ❌ Ошибка сохранения дефолтной конфигурации: {e}")
        return False

def load_default_config():
    """Загружает дефолтную конфигурацию из файла"""
    try:
        if os.path.exists(DEFAULT_CONFIG_FILE):
            with open(DEFAULT_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Если файла нет, создаем его с текущими дефолтными значениями
            save_default_config()
            return DEFAULT_AUTO_BOT_CONFIG.copy()
            
    except Exception as e:
        logger.error(f" ❌ Ошибка загрузки дефолтной конфигурации: {e}")
        return DEFAULT_AUTO_BOT_CONFIG.copy()

def restore_default_config():
    """Восстанавливает дефолтную конфигурацию Auto Bot"""
    try:
        default_config = load_default_config()
        
        with bots_data_lock:
            # Сохраняем критически важные значения (не сбрасываем их при восстановлении)
            current_enabled = bots_data['auto_bot_config'].get('enabled', False)
            current_trading_enabled = bots_data['auto_bot_config'].get('trading_enabled', True)
            
            # Восстанавливаем дефолтные значения
            bots_data['auto_bot_config'] = default_config.copy()
            
            # Возвращаем текущие состояния важных настроек
            bots_data['auto_bot_config']['enabled'] = current_enabled
            bots_data['auto_bot_config']['trading_enabled'] = current_trading_enabled
        
        # Сохраняем состояние
        save_result = save_bots_state()
        
        logger.info(" ✅ Дефолтная конфигурация восстановлена")
        return save_result
        
    except Exception as e:
        logger.error(f" ❌ Ошибка восстановления дефолтной конфигурации: {e}")
        return False

def update_process_state(process_name, status_update):
    """Обновляет состояние процесса"""
    try:
        if process_name in process_state:
            process_state[process_name].update(status_update)
            
            # Автоматически сохраняем состояние процессов
            save_process_state()
            
    except Exception as e:
        logger.error(f" ❌ Ошибка обновления состояния {process_name}: {e}")

def save_process_state():
    """Сохраняет состояние всех процессов"""
    try:
        state_data = {
            'process_state': process_state.copy(),
            'last_saved': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(PROCESS_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        logger.error(f" ❌ Ошибка сохранения состояния процессов: {e}")
        return False

def load_process_state():
    """Загружает состояние процессов из файла"""
    try:
        if not os.path.exists(PROCESS_STATE_FILE):
            logger.info(f" 📁 Файл состояния процессов не найден, начинаем с дефолтного")
            save_process_state()  # Создаем файл
            return False
        
        with open(PROCESS_STATE_FILE, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        if 'process_state' in state_data:
            # Обновляем глобальное состояние
            for process_name, process_info in state_data['process_state'].items():
                if process_name in process_state:
                    process_state[process_name].update(process_info)
            
            last_saved = state_data.get('last_saved', 'неизвестно')
            logger.info(f" ✅ Состояние процессов восстановлено (сохранено: {last_saved})")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f" ❌ Ошибка загрузки состояния процессов: {e}")
        return False

def save_system_config(config_data):
    """Сохраняет системные настройки напрямую в bot_config.py."""
    try:
        from bots_modules.config_writer import save_system_config_to_py

        attrs_to_update = {}
        for key, attr in SYSTEM_CONFIG_FIELD_MAP.items():
            if key in config_data:
                attrs_to_update[attr] = config_data[key]

        if not attrs_to_update:
            logger.debug("[SYSTEM_CONFIG] ⚠️ Нет параметров для сохранения")
            return True

        success = save_system_config_to_py(attrs_to_update)
        if success:
            logger.info("[SYSTEM_CONFIG] ✅ Настройки сохранены в bot_engine/bot_config.py")
        return success

    except Exception as e:
        logger.error(f"[SYSTEM_CONFIG] ❌ Ошибка сохранения системных настроек: {e}")
        return False


def load_system_config():
    """Перезагружает SystemConfig из bot_config.py и применяет значения в память."""
    try:
        bot_config_module = importlib.import_module('bot_engine.bot_config')
        importlib.reload(bot_config_module)
        file_system_config = bot_config_module.SystemConfig

        for attr in SYSTEM_CONFIG_FIELD_MAP.values():
            if hasattr(file_system_config, attr):
                setattr(SystemConfig, attr, getattr(file_system_config, attr))

        logger.info("[SYSTEM_CONFIG] ✅ Конфигурация перезагружена из bot_engine/bot_config.py")
        return True

    except Exception as e:
        logger.error(f"[SYSTEM_CONFIG] ❌ Ошибка загрузки системных настроек: {e}")
        return False

def save_bots_state():
    """Сохраняет состояние всех ботов в файл"""
    try:
        state_data = {
            'bots': {},
            'auto_bot_config': {},
            'last_saved': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # ✅ ИСПРАВЛЕНИЕ: Используем таймаут для блокировки чтобы не висеть при остановке
        import threading
        
        # Пытаемся захватить блокировку с таймаутом
        acquired = bots_data_lock.acquire(timeout=2.0)
        if not acquired:
            logger.warning("[SAVE_STATE] ⚠️ Не удалось получить блокировку за 2 секунды - пропускаем сохранение")
            return False
        
        try:
            for symbol, bot_data in bots_data['bots'].items():
                state_data['bots'][symbol] = bot_data
            
            # Сохраняем конфигурацию Auto Bot
            state_data['auto_bot_config'] = bots_data['auto_bot_config'].copy()
        finally:
            bots_data_lock.release()
        
        # Записываем в файл
        with open(BOTS_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        total_bots = len(state_data['bots'])
        logger.debug(f"[SAVE_STATE] Состояние {total_bots} ботов сохранено")
        
        return True
        
    except Exception as e:
        logger.error(f"[SAVE_STATE] ❌ Ошибка сохранения состояния: {e}")
        return False

def save_auto_bot_config():
    """Сохраняет конфигурацию автобота в bot_config.py
    
    ✅ Теперь сохраняет напрямую в bot_engine/bot_config.py
    - Все изменения сохраняются в Python-файл
    - Комментарии в файле сохраняются
    - Автоматически перезагружает модуль после сохранения (НЕ требуется перезапуск!)
    """
    try:
        from bots_modules.config_writer import save_auto_bot_config_to_py
        import importlib
        import sys
        
        with bots_data_lock:
            config_data = bots_data['auto_bot_config'].copy()
        
        # Сохраняем в bot_config.py
        success = save_auto_bot_config_to_py(config_data)
        
        if success:
            logger.info(f"[SAVE_CONFIG] ✅ Конфигурация автобота сохранена в bot_engine/bot_config.py")
            # ✅ КРИТИЧНО: Обновляем конфигурацию в памяти из СОХРАНЕННЫХ данных (не из DEFAULT!)
            with bots_data_lock:
                # ✅ Используем новые RSI exit с учетом тренда
                old_rsi_long_with = bots_data['auto_bot_config'].get('rsi_exit_long_with_trend')
                old_rsi_long_against = bots_data['auto_bot_config'].get('rsi_exit_long_against_trend')
                old_rsi_short_with = bots_data['auto_bot_config'].get('rsi_exit_short_with_trend')
                old_rsi_short_against = bots_data['auto_bot_config'].get('rsi_exit_short_against_trend')
                
                # Используем ТОЛЬКО ЧТО СОХРАНЕННЫЕ значения, а не дефолтные!
                bots_data['auto_bot_config'].update(config_data)
                
                new_rsi_long_with = bots_data['auto_bot_config'].get('rsi_exit_long_with_trend')
                new_rsi_long_against = bots_data['auto_bot_config'].get('rsi_exit_long_against_trend')
                new_rsi_short_with = bots_data['auto_bot_config'].get('rsi_exit_short_with_trend')
                new_rsi_short_against = bots_data['auto_bot_config'].get('rsi_exit_short_against_trend')
            
            # Проверяем что значения действительно есть
            if new_rsi_long_with is None:
                logger.error(f"[SAVE_CONFIG] ❌ КРИТИЧЕСКАЯ ОШИБКА: rsi_exit_long_with_trend отсутствует в сохраненных данных!")
            if new_rsi_long_against is None:
                logger.error(f"[SAVE_CONFIG] ❌ КРИТИЧЕСКАЯ ОШИБКА: rsi_exit_long_against_trend отсутствует в сохраненных данных!")
            if new_rsi_short_with is None:
                logger.error(f"[SAVE_CONFIG] ❌ КРИТИЧЕСКАЯ ОШИБКА: rsi_exit_short_with_trend отсутствует в сохраненных данных!")
            if new_rsi_short_against is None:
                logger.error(f"[SAVE_CONFIG] ❌ КРИТИЧЕСКАЯ ОШИБКА: rsi_exit_short_against_trend отсутствует в сохраненных данных!")
            
            # Логируем изменения RSI exit порогов
            if old_rsi_long_with is not None and new_rsi_long_with is not None and old_rsi_long_with != new_rsi_long_with:
                logger.info(f"[SAVE_CONFIG] 🔄 RSI LONG exit (по тренду) изменен: {old_rsi_long_with} → {new_rsi_long_with}")
            if old_rsi_long_against is not None and new_rsi_long_against is not None and old_rsi_long_against != new_rsi_long_against:
                logger.info(f"[SAVE_CONFIG] 🔄 RSI LONG exit (против тренда) изменен: {old_rsi_long_against} → {new_rsi_long_against}")
            if old_rsi_short_with is not None and new_rsi_short_with is not None and old_rsi_short_with != new_rsi_short_with:
                logger.info(f"[SAVE_CONFIG] 🔄 RSI SHORT exit (по тренду) изменен: {old_rsi_short_with} → {new_rsi_short_with}")
            if old_rsi_short_against is not None and new_rsi_short_against is not None and old_rsi_short_against != new_rsi_short_against:
                logger.info(f"[SAVE_CONFIG] 🔄 RSI SHORT exit (против тренда) изменен: {old_rsi_short_against} → {new_rsi_short_against}")
            
            logger.info(f"[SAVE_CONFIG] ✅ Конфигурация обновлена в памяти из сохраненных данных!")
            if new_rsi_long_with is not None and new_rsi_short_with is not None:
                logger.info(f"[SAVE_CONFIG] 📊 Текущие RSI exit пороги: LONG(with)={new_rsi_long_with}, LONG(against)={new_rsi_long_against}, SHORT(with)={new_rsi_short_with}, SHORT(against)={new_rsi_short_against}")
            else:
                logger.error(f"[SAVE_CONFIG] ❌ НЕКОТОРЫЕ RSI exit пороги отсутствуют в конфигурации!")
            
            # ✅ Перезагружаем модуль bot_config и обновляем конфигурацию из него
            try:
                if 'bot_engine.bot_config' in sys.modules:
                    logger.debug(f"[SAVE_CONFIG] 🔄 Перезагружаем модуль bot_config...")
                    import bot_engine.bot_config
                    importlib.reload(bot_engine.bot_config)
                    logger.debug(f"[SAVE_CONFIG] ✅ Модуль перезагружен")
                    
                    # ✅ КРИТИЧЕСКИ ВАЖНО: Перезагружаем конфигурацию из обновленного bot_config.py
                    # Это нужно, чтобы значения сразу брались из файла, а не из старой памяти
                    from bots_modules.imports_and_globals import load_auto_bot_config
                    
                    # ✅ СБРАСЫВАЕМ кэш времени модификации файла, чтобы при следующем вызове модуль перезагрузился
                    if hasattr(load_auto_bot_config, '_last_mtime'):
                        load_auto_bot_config._last_mtime = 0
                        logger.debug(f"[SAVE_CONFIG] 🔄 Сброшен кэш времени модификации файла")
                    
                    load_auto_bot_config()
                    logger.info(f"[SAVE_CONFIG] ✅ Конфигурация перезагружена из bot_config.py после сохранения")
            except Exception as reload_error:
                logger.warning(f"[SAVE_CONFIG] ⚠️ Не удалось перезагрузить модуль (не критично): {reload_error}")
        
        return success
        
    except Exception as e:
        logger.error(f"[SAVE_CONFIG] ❌ Ошибка сохранения конфигурации автобота: {e}")
        return False

# ❌ ОТКЛЮЧЕНО: optimal_ema перемещен в backup (EMA фильтр убран)
# def save_optimal_ema_periods():
#     """Сохраняет оптимальные EMA периоды"""
#     return True  # Заглушка

def load_bots_state():
    """Загружает состояние ботов из файла"""
    try:
        if not os.path.exists(BOTS_STATE_FILE):
            logger.info(f" 📁 Файл состояния {BOTS_STATE_FILE} не найден, начинаем с пустого состояния")
            return False
        
        logger.info(f" 📂 Загрузка состояния ботов из {BOTS_STATE_FILE}...")
        
        with open(BOTS_STATE_FILE, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        version = state_data.get('version', '1.0')
        last_saved = state_data.get('last_saved', 'неизвестно')
        
        logger.info(f" 📊 Версия состояния: {version}, последнее сохранение: {last_saved}")
        
        # ✅ Конфигурация Auto Bot никогда не берётся из bots_state.json
        # Настройки загружаются только из bot_engine/bot_config.py
        # bots_state.json содержит только состояние ботов и глобальную статистику
        
        logger.info(f" ⚙️ Конфигурация Auto Bot НЕ загружается из bots_state.json")
        logger.info(f" 💡 Конфигурация загружается только из bot_engine/bot_config.py")
        
        # Восстанавливаем ботов
        restored_bots = 0
        failed_bots = 0
        
        if 'bots' in state_data:
            with bots_data_lock:
                for symbol, bot_data in state_data['bots'].items():
                    try:
                        # Проверяем валидность данных бота
                        if not isinstance(bot_data, dict) or 'status' not in bot_data:
                            logger.warning(f" ⚠️ Некорректные данные бота {symbol}, пропускаем")
                            failed_bots += 1
                            continue
                        
                        # ВАЖНО: НЕ проверяем зрелость при восстановлении!
                        # Причины:
                        # 1. Биржа еще не инициализирована (нет данных свечей)
                        # 2. Если бот был сохранен - он уже прошел проверку зрелости при создании
                        # 3. Проверка зрелости будет выполнена позже при обработке сигналов
                        
                        # Восстанавливаем бота
                        bots_data['bots'][symbol] = bot_data
                        restored_bots += 1
                        
                        logger.info(f" 🤖 Восстановлен бот {symbol}: статус={bot_data.get('status', 'UNKNOWN')}")
                        
                    except Exception as e:
                        logger.error(f" ❌ Ошибка восстановления бота {symbol}: {e}")
                        failed_bots += 1
        
        logger.info(f" ✅ Восстановлено ботов: {restored_bots}, ошибок: {failed_bots}")
        
        return restored_bots > 0
        
    except Exception as e:
        logger.error(f" ❌ Ошибка загрузки состояния: {e}")
        return False

def load_delisted_coins():
    """Загружает список делистинговых монет из файла"""
    delisted_file = Path("data/delisted.json")
    default_data = {"delisted_coins": {}, "last_scan": None, "scan_enabled": True}
    
    # Если файл не существует или пустой, создаем дефолтный
    if not delisted_file.exists() or delisted_file.stat().st_size == 0:
        logger.info("Создаем новый файл delisted.json с дефолтными данными")
        # Создаем папку data если её нет
        delisted_file.parent.mkdir(exist_ok=True)
        # Сохраняем дефолтные данные
        try:
            with open(delisted_file, 'w', encoding='utf-8') as f:
                json.dump(default_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Не удалось создать файл delisted.json: {e}")
        return default_data
    
    # Пытаемся загрузить существующий файл
    try:
        with open(delisted_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Если файл пустой после trim
            if not content:
                logger.info("Файл delisted.json пустой, используем дефолтные данные")
                # Записываем дефолтные данные
                with open(delisted_file, 'w', encoding='utf-8') as fw:
                    json.dump(default_data, fw, indent=2, ensure_ascii=False)
                return default_data
            # Парсим JSON
            data = json.loads(content)
            return data
    except json.JSONDecodeError as e:
        logger.warning(f"Невалидный JSON в delisted.json, восстанавливаем дефолтные данные: {e}")
        # Перезаписываем файл дефолтными данными
        try:
            with open(delisted_file, 'w', encoding='utf-8') as f:
                json.dump(default_data, f, indent=2, ensure_ascii=False)
        except Exception as write_error:
            logger.warning(f"Не удалось восстановить файл: {write_error}")
        return default_data
    except Exception as e:
        logger.warning(f"Ошибка загрузки delisted.json: {e}, используем дефолтные данные")
        return default_data

def save_delisted_coins(data):
    """Сохраняет список делистинговых монет в файл"""
    delisted_file = Path("data/delisted.json")
    
    try:
        # Создаем папку data если её нет
        delisted_file.parent.mkdir(exist_ok=True)
        
        with open(delisted_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Обновлен файл delisted.json")
        return True
    except Exception as e:
        logger.error(f"Ошибка сохранения delisted.json: {e}")
        return False

def scan_all_coins_for_delisting():
    """Сканирует все монеты на предмет делистинга и обновляет delisted.json"""
    try:
        logger.info("🔍 Сканирование всех монет на делистинг...")
        
        # Загружаем текущие данные
        delisted_data = load_delisted_coins()
        
        if not delisted_data.get('scan_enabled', True):
            logger.info("⏸️ Сканирование отключено в конфигурации")
            return
        
        exchange_obj = get_exchange()
        if not exchange_obj:
            logger.error("❌ Exchange не инициализирован")
            return
        
        # Получаем все пары
        all_pairs = exchange_obj.get_all_pairs()
        if not all_pairs:
            logger.warning("⚠️ Не удалось получить список пар")
            return
        
        # Фильтруем только USDT пары
        usdt_pairs = [pair for pair in all_pairs if pair.endswith('USDT')]
        
        logger.info(f"📊 Проверяем {len(usdt_pairs)} USDT пар")
        
        # Инициализируем структуру если её нет
        if 'delisted_coins' not in delisted_data:
            delisted_data['delisted_coins'] = {}
        
        new_delisted_count = 0
        checked_count = 0
        
        # Проверяем каждый символ
        for symbol in usdt_pairs:
            try:
                checked_count += 1
                coin_symbol = symbol.replace('USDT', '')
                
                # Пропускаем если уже в списке делистинговых
                if coin_symbol in delisted_data['delisted_coins']:
                    continue
                
                # Проверяем статус делистинга через API
                if hasattr(exchange_obj, 'get_instrument_status'):
                    status_info = exchange_obj.get_instrument_status(symbol)
                    
                    if status_info and status_info.get('is_delisting'):
                        delisted_data['delisted_coins'][coin_symbol] = {
                            'status': status_info.get('status'),
                            'reason': f"Delisting detected via API scan",
                            'delisting_date': datetime.now().strftime('%Y-%m-%d'),
                            'detected_at': datetime.now().isoformat(),
                            'source': 'api_scan'
                        }
                        
                        new_delisted_count += 1
                        logger.warning(f"🚨 НОВЫЙ ДЕЛИСТИНГ: {coin_symbol} - {status_info.get('status')}")
                
                # Небольшая задержка чтобы не перегружать API
                time.sleep(0.05)
                
            except Exception as e:
                logger.debug(f"Ошибка проверки {symbol}: {e}")
                continue
        
        # Обновляем время последнего сканирования
        delisted_data['last_scan'] = datetime.now().isoformat()
        
        # Сохраняем обновленные данные
        if save_delisted_coins(delisted_data):
            logger.info(f"✅ Сканирование завершено:")
            logger.info(f"   - Проверено символов: {checked_count}")
            logger.info(f"   - Новых делистинговых: {new_delisted_count}")
            logger.info(f"   - Всего делистинговых: {len(delisted_data['delisted_coins'])}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка сканирования делистинга: {e}")

def check_delisting_emergency_close():
    """Проверяет делистинг и выполняет экстренное закрытие позиций (раз в 10 минут)"""
    try:
        # Импорты для экстренного закрытия позиций
        from bots_modules.bot_class import NewTradingBot
        from bots_modules.imports_and_globals import get_exchange
        
        # ✅ СНАЧАЛА: Сканируем все монеты на делистинг
        scan_all_coins_for_delisting()
        
        logger.info(f"🔍 Проверка делистинга для активных ботов...")
        
        with bots_data_lock:
            bots_in_position = [
                (symbol, bot_data) for symbol, bot_data in bots_data['bots'].items()
                if bot_data.get('status') in ['in_position_long', 'in_position_short']
            ]
        
        if not bots_in_position:
            logger.debug(f"ℹ️ Нет активных ботов для проверки делистинга")
            return True
        
        logger.info(f"📊 Проверяем {len(bots_in_position)} активных ботов")
        
        delisting_closed_count = 0
        exchange_obj = get_exchange()
        
        if not exchange_obj:
            logger.error(f"❌ Exchange не инициализирован")
            return False
        
        for symbol, bot_data in bots_in_position:
            try:
                # Проверяем делистинг через RSI данные
                rsi_cache = get_rsi_cache()
                if symbol in rsi_cache:
                    rsi_data = rsi_cache[symbol]
                    is_delisting = rsi_data.get('is_delisting', False) or rsi_data.get('trading_status') in ['Closed', 'Delivering']
                    
                    if is_delisting:
                        logger.warning(f"🚨 ДЕЛИСТИНГ ОБНАРУЖЕН для {symbol}! Инициируем экстренное закрытие")
                        
                        bot_instance = NewTradingBot(symbol, bot_data, exchange_obj)
                        
                        # Выполняем экстренное закрытие
                        emergency_result = bot_instance.emergency_close_delisting()
                        
                        if emergency_result:
                            logger.warning(f"✅ ЭКСТРЕННОЕ ЗАКРЫТИЕ {symbol} УСПЕШНО")
                            # Обновляем статус бота
                            with bots_data_lock:
                                if symbol in bots_data['bots']:
                                    bots_data['bots'][symbol]['status'] = 'idle'
                                    bots_data['bots'][symbol]['position_side'] = None
                                    bots_data['bots'][symbol]['entry_price'] = None
                                    bots_data['bots'][symbol]['unrealized_pnl'] = 0
                                    bots_data['bots'][symbol]['last_update'] = datetime.now().isoformat()
                            
                            delisting_closed_count += 1
                        else:
                            logger.error(f"❌ ЭКСТРЕННОЕ ЗАКРЫТИЕ {symbol} НЕУДАЧНО")
                            
            except Exception as e:
                logger.error(f"❌ Ошибка проверки делистинга для {symbol}: {e}")
        
        if delisting_closed_count > 0:
            logger.warning(f"🚨 ЭКСТРЕННО ЗАКРЫТО {delisting_closed_count} позиций из-за делистинга!")
            # Сохраняем состояние после экстренного закрытия
            save_bots_state()
        
        logger.info(f"✅ Проверка делистинга завершена")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка проверки делистинга: {e}")
        return False

def update_bots_cache_data():
    """Обновляет кэшированные данные ботов (как background_update в app.py)"""
    global bots_cache_data
    
    try:
        if not ensure_exchange_initialized():
            return False
        
        # Подавляем частые сообщения об обновлении кэша
        should_log, log_message = should_log_message(
            'cache_update', 
            "🔄 Обновление кэшированных данных ботов...",
            interval_seconds=300  # Логируем раз в 5 минут
        )
        if should_log:
            logger.info(f" {log_message}")
        
        # Добавляем таймаут для предотвращения зависания (Windows-совместимый)
        import threading
        import time
        
        timeout_occurred = threading.Event()
        
        def timeout_worker():
            time.sleep(30)  # 30 секунд таймаут
            timeout_occurred.set()
        
        timeout_thread = threading.Thread(target=timeout_worker, daemon=True)
        timeout_thread.start()
        
        # ⚡ ОПТИМИЗАЦИЯ: Получаем данные ботов быстро без лишних операций
        bots_list = []
        for symbol, bot_data in bots_data['bots'].items():
            # Проверяем таймаут
            if timeout_occurred.is_set():
                logger.warning(" ⚠️ Таймаут достигнут, прерываем обновление")
                break
            
            # Добавляем RSI данные к боту (используем кэшированные данные)
            try:
                rsi_cache = get_rsi_cache()
                if symbol in rsi_cache:
                    rsi_data = rsi_cache[symbol]
                    bot_data['rsi_data'] = rsi_data
                else:
                    bot_data['rsi_data'] = {'rsi': 'N/A', 'signal': 'N/A'}
            except Exception as e:
                logger.error(f" Ошибка получения RSI для {symbol}: {e}")
                bot_data['rsi_data'] = {'rsi': 'N/A', 'signal': 'N/A'}
            
            # Добавляем бота в список
            bots_list.append(bot_data)
        
        # Получаем информацию о позициях с биржи один раз для всех ботов
        # ✅ КРИТИЧНО: Используем тот же способ что и positions_monitor_worker!
        try:
            # Получаем позиции тем же способом что и positions_monitor_worker
            logger.debug(f" Получаем позиции с биржи...")
            exchange_obj = get_exchange()
            if exchange_obj:
                exchange_positions = exchange_obj.get_positions()
                if isinstance(exchange_positions, tuple):
                    positions_list = exchange_positions[0] if exchange_positions else []
                else:
                    positions_list = exchange_positions if exchange_positions else []
                logger.debug(f" Получено {len(positions_list)} позиций с биржи")
            else:
                positions_list = []
                logger.warning(f" Exchange не инициализирован")
            
            if positions_list:
                # Создаем словарь позиций для быстрого поиска
                positions_dict = {pos.get('symbol'): pos for pos in positions_list}
                
                # Добавляем информацию о позициях к ботам (включая стоп-лоссы)
                for bot_data in bots_list:
                    symbol = bot_data.get('symbol')
                    if symbol in positions_dict and bot_data.get('status') in ['in_position_long', 'in_position_short']:
                        pos = positions_dict[symbol]
                        
                        bot_data['exchange_position'] = {
                            'size': pos.get('size', 0),
                            'side': pos.get('side', ''),
                            'unrealized_pnl': float(pos.get('pnl', 0)),  # ✅ Используем правильное поле 'pnl'
                            'mark_price': float(pos.get('mark_price', 0)),  # ✅ Используем правильное поле 'mark_price'
                            'entry_price': float(pos.get('avg_price', 0)),   # ✅ Используем правильное поле 'avg_price'
                            'leverage': float(pos.get('leverage', 1)),
                            'stop_loss': pos.get('stop_loss', ''),  # Стоп-лосс с биржи
                            'take_profit': pos.get('take_profit', ''),  # Тейк-профит с биржи
                            'roi': float(pos.get('roi', 0)),  # ✅ ROI есть в данных
                            'realized_pnl': float(pos.get('realized_pnl', 0)),
                            'margin_usdt': bot_data.get('margin_usdt')
                        }
                        
                        # ✅ КРИТИЧНО: Синхронизируем ВСЕ данные позиции с биржей
                        exchange_stop_loss = pos.get('stopLoss', '')
                        exchange_take_profit = pos.get('takeProfit', '')
                        exchange_entry_price = float(pos.get('avgPrice', 0))  # ❌ НЕТ в данных биржи
                        exchange_size = abs(float(pos.get('size', 0)))
                        exchange_unrealized_pnl = float(pos.get('pnl', 0))  # ✅ Используем правильное поле 'pnl'
                        exchange_mark_price = float(pos.get('markPrice', 0))  # ❌ НЕТ в данных биржи
                        exchange_roi = float(pos.get('roi', 0))  # ✅ ROI есть в данных
                        exchange_realized_pnl = float(pos.get('realized_pnl', 0))
                        exchange_leverage = float(pos.get('leverage', 1) or 1)
                        
                        # ✅ КРИТИЧНО: Обновляем данные бота актуальными данными с биржи
                        if exchange_entry_price > 0:
                            bot_data['entry_price'] = exchange_entry_price
                        
                        # ⚡ КРИТИЧНО: position_size должен быть в USDT, а не в монетах!
                        # Получаем volume_value из bot_data (это USDT)
                        if exchange_size > 0:
                            # Сохраняем volume_value как position_size (в USDT)
                            volume_value = bot_data.get('volume_value', 0)
                            if volume_value > 0:
                                bot_data['position_size'] = volume_value  # USDT
                                bot_data['position_size_coins'] = exchange_size  # Монеты для справки
                            else:
                                # Fallback: если volume_value нет, используем размер в монетах
                                bot_data['position_size'] = exchange_size
                        if exchange_mark_price > 0:
                            bot_data['current_price'] = exchange_mark_price
                            bot_data['mark_price'] = exchange_mark_price  # Дублируем для UI
                        else:
                            # ❌ НЕТ mark_price с биржи - получаем текущую цену напрямую с биржи
                            try:
                                exchange_obj = get_exchange()
                                if exchange_obj:
                                    ticker_data = exchange_obj.get_ticker(symbol)
                                    if ticker_data and ticker_data.get('last'):
                                        current_price = float(ticker_data.get('last'))
                                        bot_data['current_price'] = current_price
                                        bot_data['mark_price'] = current_price
                            except Exception as e:
                                logger.error(f" ❌ {symbol} - Ошибка получения цены с биржи: {e}")
                        
                        # ✅ КРИТИЧНО: Обновляем PnL ВСЕГДА, даже если он равен 0
                        bot_data['unrealized_pnl'] = exchange_unrealized_pnl
                        bot_data['unrealized_pnl_usdt'] = exchange_unrealized_pnl  # Точное значение в USDT
                        bot_data['realized_pnl'] = exchange_realized_pnl
                        bot_data['leverage'] = exchange_leverage
                        bot_data['position_size_coins'] = exchange_size
                        if exchange_entry_price > 0 and exchange_size > 0:
                            position_value = exchange_entry_price * exchange_size
                            bot_data['margin_usdt'] = position_value / exchange_leverage if exchange_leverage else position_value
                        
                        # Отладочный лог для проверки PnL
                        logger.debug(f"[POSITION_SYNC] {symbol}: PnL с биржи = {exchange_unrealized_pnl}, обновлен в bot_data")
                        
                        # ✅ Обновляем ROI
                        if exchange_roi != 0:
                            bot_data['roi'] = exchange_roi
                        
                        # Синхронизируем стоп-лосс
                        current_stop_loss = bot_data.get('trailing_stop_price')
                        if exchange_stop_loss:
                            # Есть стоп-лосс на бирже - обновляем данные бота
                            new_stop_loss = float(exchange_stop_loss)
                            if not current_stop_loss or abs(current_stop_loss - new_stop_loss) > 0.001:
                                bot_data['trailing_stop_price'] = new_stop_loss
                                logger.debug(f"[POSITION_SYNC] Обновлен стоп-лосс для {symbol}: {new_stop_loss}")
                        else:
                            # Нет стоп-лосса на бирже - очищаем данные бота
                            if current_stop_loss:
                                bot_data['trailing_stop_price'] = None
                                logger.info(f"[POSITION_SYNC] ⚠️ Стоп-лосс отменен на бирже для {symbol}")
                        
                        # Синхронизируем тейк-профит
                        if exchange_take_profit:
                            bot_data['take_profit_price'] = float(exchange_take_profit)
                        else:
                            bot_data['take_profit_price'] = None
                        
                        # Синхронизируем цену входа (может измениться при добавлении к позиции)
                        if exchange_entry_price and exchange_entry_price > 0:
                            current_entry_price = bot_data.get('entry_price')
                            if not current_entry_price or abs(current_entry_price - exchange_entry_price) > 0.001:
                                bot_data['entry_price'] = exchange_entry_price
                                logger.debug(f"[POSITION_SYNC] Обновлена цена входа для {symbol}: {exchange_entry_price}")
                        
                        # ⚡ Размер позиции уже синхронизирован выше (в USDT)
                        
                        # Обновляем время последнего обновления
                        bot_data['last_update'] = datetime.now().isoformat()
        except Exception as e:
            logger.error(f" Ошибка получения позиций с биржи: {e}")
        
        # Обновляем кэш (только данные ботов, account_info больше не кэшируется)
        current_time = datetime.now().isoformat()
        with bots_cache_lock:
            bots_cache_data.update({
                'bots': bots_list,
                'last_update': current_time
            })
        
        # ✅ СИНХРОНИЗАЦИЯ: Проверяем закрытые позиции на бирже
        try:
            logger.debug(f" Синхронизация...")
            sync_bots_with_exchange()
            logger.debug(f" Синхронизация завершена")
        except Exception as e:
            logger.error(f" ❌ Ошибка синхронизации с биржей: {e}")
        
        # ✅ КРИТИЧНО: Обновляем last_update в bots_data для UI
        # ⚡ БЕЗ БЛОКИРОВКИ: GIL делает запись атомарной
        bots_data['last_update'] = current_time
        
        # Отладочный лог для проверки частоты обновлений
        logger.debug(f" 🔄 Обновление завершено: {current_time}")
        
        logger.debug(f" Кэш обновлен: {len(bots_list)} ботов")
        return True
        
    except Exception as e:
        logger.error(f" ❌ Ошибка обновления кэша: {e}")
        return False

def update_bot_positions_status():
    """Обновляет статус позиций ботов (цена, PnL, ликвидация) каждые SystemConfig.BOT_STATUS_UPDATE_INTERVAL секунд"""
    try:
        if not ensure_exchange_initialized():
            return False
        
        with bots_data_lock:
            updated_count = 0
            
            for symbol, bot_data in bots_data['bots'].items():
                # Обновляем только ботов в позиции (НО НЕ остановленных!)
                bot_status = bot_data.get('status')
                if bot_status not in ['in_position_long', 'in_position_short']:
                    continue
                
                # ⚡ КРИТИЧНО: Не обновляем ботов на паузе!
                if bot_status == BOT_STATUS['PAUSED']:
                    logger.debug(f"[POSITION_UPDATE] ⏸️ {symbol}: Бот на паузе - пропускаем обновление")
                    continue
                
                try:
                    # Получаем текущую цену
                    current_exchange = get_exchange()
                    if not current_exchange:
                        continue
                    ticker_data = current_exchange.get_ticker(symbol)
                    if not ticker_data or 'last_price' not in ticker_data:
                        continue
                    current_price = float(ticker_data['last_price'])
                    
                    entry_price = bot_data.get('entry_price')
                    position_side = bot_data.get('position_side')
                    
                    if not entry_price or not position_side:
                        continue
                    
                    # Рассчитываем PnL
                    if position_side == 'LONG':
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    else:  # SHORT
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100
                    
                    # Обновляем данные бота
                    old_pnl = bot_data.get('unrealized_pnl', 0)
                    bot_data['unrealized_pnl'] = pnl_percent
                    bot_data['current_price'] = current_price
                    bot_data['last_update'] = datetime.now().isoformat()
                    
                    # Рассчитываем цену ликвидации (примерно)
                    volume_value = bot_data.get('volume_value', 10)
                    leverage = 10  # Предполагаем плечо 10x
                    
                    if position_side == 'LONG':
                        # Для LONG: ликвидация при падении цены
                        liquidation_price = entry_price * (1 - (100 / leverage) / 100)
                    else:  # SHORT
                        # Для SHORT: ликвидация при росте цены
                        liquidation_price = entry_price * (1 + (100 / leverage) / 100)
                    
                    bot_data['liquidation_price'] = liquidation_price
                    
                    # Расстояние до ликвидации
                    if position_side == 'LONG':
                        distance_to_liq = ((current_price - liquidation_price) / liquidation_price) * 100
                    else:  # SHORT
                        distance_to_liq = ((liquidation_price - current_price) / liquidation_price) * 100
                    
                    bot_data['distance_to_liquidation'] = distance_to_liq
                    
                    updated_count += 1
                    
                    # Логируем только если PnL изменился значительно
                    if abs(pnl_percent - old_pnl) > 0.1:
                        logger.info(f"[POSITION_UPDATE] 📊 {symbol} {position_side}: ${current_price:.6f} | PnL: {pnl_percent:+.2f}% | Ликвидация: ${liquidation_price:.6f} ({distance_to_liq:.1f}%)")
                
                except Exception as e:
                    logger.error(f"[POSITION_UPDATE] ❌ Ошибка обновления {symbol}: {e}")
                    continue
        
        if updated_count > 0:
            logger.debug(f"[POSITION_UPDATE] ✅ Обновлено {updated_count} позиций")
        
        return True
        
    except Exception as e:
        logger.error(f"[POSITION_UPDATE] ❌ Ошибка обновления позиций: {e}")
        return False

def get_exchange_positions():
    """Получает реальные позиции с биржи с retry логикой"""
    max_retries = 3
    retry_delay = 2  # секунды
    
    for attempt in range(max_retries):
        try:
            # Получаем актуальную ссылку на биржу
            current_exchange = get_exchange()
            
            if not current_exchange:
                logger.warning(f"[EXCHANGE_POSITIONS] Биржа не инициализирована (попытка {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None

            # Получаем СЫРЫЕ данные напрямую от API Bybit
            response = current_exchange.client.get_positions(
                category="linear",
                settleCoin="USDT",
                limit=100
            )

            if response['retCode'] != 0:
                error_msg = response['retMsg']
                logger.warning(f"[EXCHANGE_POSITIONS] ⚠️ Ошибка API (попытка {attempt + 1}/{max_retries}): {error_msg}")
                
                # Если это Rate Limit, увеличиваем задержку
                if "rate limit" in error_msg.lower() or "too many" in error_msg.lower():
                    retry_delay = min(retry_delay * 2, 10)  # Увеличиваем задержку до максимум 10 сек
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"[EXCHANGE_POSITIONS] ❌ Не удалось получить позиции после {max_retries} попыток")
                    return None
            
            raw_positions = response['result']['list']
            # ✅ Не логируем частые запросы позиций (только при изменениях)
            
            # Обрабатываем сырые позиции
            processed_positions = []
            for position in raw_positions:
                symbol = position.get('symbol', '').replace('USDT', '')  # Убираем USDT
                size = float(position.get('size', 0))
                side = position.get('side', '')  # 'Buy' или 'Sell'
                entry_price = float(position.get('avgPrice', 0))
                unrealized_pnl = float(position.get('unrealisedPnl', 0))
                mark_price = float(position.get('markPrice', 0))
                
                if abs(size) > 0:  # Только активные позиции
                    processed_positions.append({
                        'symbol': symbol,
                        'size': size,
                        'side': side,
                        'entry_price': entry_price,
                        'unrealized_pnl': unrealized_pnl,
                        'mark_price': mark_price,
                        'position_side': 'LONG' if side == 'Buy' else 'SHORT'
                    })
            
            # ✅ Не логируем частые запросы (только при изменениях)
            
            # Возвращаем ВСЕ позиции с биржи, не фильтруя по наличию ботов в системе
            # Это нужно для правильной работы синхронизации и очистки неактивных ботов
            filtered_positions = []
            ignored_positions = []
            
            for pos in processed_positions:
                symbol = pos['symbol']
                # Добавляем все позиции без фильтрации
                filtered_positions.append(pos)
            
            # ✅ Не логируем частые запросы (только при изменениях)
            return filtered_positions
            
        except Exception as api_error:
            logger.error(f"[EXCHANGE_POSITIONS] ❌ Ошибка прямого обращения к API: {api_error}")
            # Fallback к существующему методу
            current_exchange = get_exchange()
            if not current_exchange:
                logger.error("[EXCHANGE_POSITIONS] ❌ Биржа не инициализирована")
                return []
            positions, _ = current_exchange.get_positions()
            logger.info(f"[EXCHANGE_POSITIONS] Fallback: получено {len(positions) if positions else 0} позиций")
            
            if not positions:
                return []
            
            # Обрабатываем fallback позиции
            processed_positions = []
            for position in positions:
                # Позиции уже обработаны в exchange.get_positions()
                symbol = position.get('symbol', '')
                size = position.get('size', 0)
                side = position.get('side', '')  # 'Long' или 'Short'
                
                if abs(size) > 0:
                    processed_positions.append({
                        'symbol': symbol,
                        'size': size,
                        'side': side,
                        'entry_price': 0.0,  # Нет данных в обработанном формате
                        'unrealized_pnl': position.get('pnl', 0),
                        'mark_price': 0.0,
                        'position_side': side
                    })
            
            # КРИТИЧЕСКИ ВАЖНО: Фильтруем fallback позиции тоже
            with bots_data_lock:
                system_bot_symbols = set(bots_data['bots'].keys())
            
            filtered_positions = []
            ignored_positions = []
            
            for pos in processed_positions:
                symbol = pos['symbol']
                if symbol in system_bot_symbols:
                    filtered_positions.append(pos)
                else:
                    ignored_positions.append(pos)
            
            if ignored_positions:
                logger.info(f"[EXCHANGE_POSITIONS] 🚫 Fallback: Игнорируем {len(ignored_positions)} позиций без ботов в системе")
            
            logger.info(f"[EXCHANGE_POSITIONS] ✅ Fallback: Возвращаем {len(filtered_positions)} позиций с ботами в системе")
            return filtered_positions
            
        except Exception as e:
            logger.error(f"[EXCHANGE_POSITIONS] ❌ Ошибка в попытке {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                logger.error(f"[EXCHANGE_POSITIONS] ❌ Не удалось получить позиции после {max_retries} попыток")
                return None
    
    # Если мы дошли сюда, значит все попытки исчерпаны
    logger.error(f"[EXCHANGE_POSITIONS] ❌ Все попытки исчерпаны")
    return None

def compare_bot_and_exchange_positions():
    """Сравнивает позиции ботов в системе с реальными позициями на бирже"""
    try:
        # Получаем позиции с биржи
        exchange_positions = get_exchange_positions()
        
        # Получаем ботов в позиции из системы
        with bots_data_lock:
            bot_positions = []
            for symbol, bot_data in bots_data['bots'].items():
                if bot_data.get('status') in ['in_position_long', 'in_position_short']:
                    bot_positions.append({
                        'symbol': symbol,
                        'position_side': bot_data.get('position_side'),
                        'entry_price': bot_data.get('entry_price'),
                        'status': bot_data.get('status')
                    })
        
        # Создаем словари для удобного сравнения
        exchange_dict = {pos['symbol']: pos for pos in exchange_positions}
        bot_dict = {pos['symbol']: pos for pos in bot_positions}
        
        # Находим расхождения
        discrepancies = {
            'missing_in_bot': [],  # Есть на бирже, нет в боте (НЕ создаем ботов!)
            'missing_in_exchange': [],  # Есть в боте, нет на бирже (обновляем статус)
            'side_mismatch': []  # Есть в обоих, но стороны не совпадают (исправляем)
        }
        
        # Проверяем позиции на бирже
        for symbol, exchange_pos in exchange_dict.items():
            if symbol not in bot_dict:
                discrepancies['missing_in_bot'].append({
                    'symbol': symbol,
                    'exchange_side': exchange_pos['position_side'],
                    'exchange_entry_price': exchange_pos['entry_price'],
                    'exchange_pnl': exchange_pos['unrealized_pnl']
                })
            else:
                bot_pos = bot_dict[symbol]
                if bot_pos['position_side'] != exchange_pos['position_side']:
                    discrepancies['side_mismatch'].append({
                        'symbol': symbol,
                        'bot_side': bot_pos['position_side'],
                        'exchange_side': exchange_pos['position_side'],
                        'bot_entry_price': bot_pos['entry_price'],
                        'exchange_entry_price': exchange_pos['entry_price']
                    })
        
        # Проверяем позиции в боте
        for symbol, bot_pos in bot_dict.items():
            if symbol not in exchange_dict:
                discrepancies['missing_in_exchange'].append({
                    'symbol': symbol,
                    'bot_side': bot_pos['position_side'],
                    'bot_entry_price': bot_pos['entry_price'],
                    'bot_status': bot_pos['status']
                })
        
        # Логируем результаты
        total_discrepancies = (len(discrepancies['missing_in_bot']) + 
                             len(discrepancies['missing_in_exchange']) + 
                             len(discrepancies['side_mismatch']))
        
        if total_discrepancies > 0:
            logger.warning(f"[POSITION_SYNC] ⚠️ Обнаружено {total_discrepancies} расхождений между ботом и биржей")
            
            if discrepancies['missing_in_bot']:
                logger.info(f"[POSITION_SYNC] 📊 Позиции на бирже без бота в системе: {len(discrepancies['missing_in_bot'])} (игнорируем - не создаем ботов)")
                for pos in discrepancies['missing_in_bot']:
                    logger.info(f"[POSITION_SYNC]   - {pos['symbol']}: {pos['exchange_side']} ${pos['exchange_entry_price']:.6f} (PnL: {pos['exchange_pnl']:.2f}) - НЕ создаем бота")
            
            if discrepancies['missing_in_exchange']:
                logger.warning(f"[POSITION_SYNC] 🤖 Боты без позиций на бирже: {len(discrepancies['missing_in_exchange'])}")
                for pos in discrepancies['missing_in_exchange']:
                    logger.warning(f"[POSITION_SYNC]   - {pos['symbol']}: {pos['bot_side']} ${pos['bot_entry_price']:.6f} (статус: {pos['bot_status']})")
            
            if discrepancies['side_mismatch']:
                logger.warning(f"[POSITION_SYNC] 🔄 Несовпадение сторон: {len(discrepancies['side_mismatch'])}")
                for pos in discrepancies['side_mismatch']:
                    logger.warning(f"[POSITION_SYNC]   - {pos['symbol']}: бот={pos['bot_side']}, биржа={pos['exchange_side']}")
        else:
            logger.info(f"[POSITION_SYNC] ✅ Синхронизация позиций: все {len(bot_positions)} ботов соответствуют бирже")
        
        return discrepancies
        
    except Exception as e:
        logger.error(f"[POSITION_SYNC] ❌ Ошибка сравнения позиций: {e}")
        return None

def sync_positions_with_exchange():
    """Умная синхронизация позиций ботов с реальными позициями на бирже"""
    try:
        # ✅ Не логируем частые синхронизации (только результаты при изменениях)
        
        # Получаем позиции с биржи с retry логикой
        exchange_positions = get_exchange_positions()
        
        # Если не удалось получить позиции с биржи, НЕ сбрасываем ботов
        if exchange_positions is None:
            logger.warning("[POSITION_SYNC] ⚠️ Не удалось получить позиции с биржи - пропускаем синхронизацию")
            return False
        
        # Получаем ботов в позиции из системы
        with bots_data_lock:
            bot_positions = []
            # ✅ ИСПРАВЛЕНИЕ: Проверяем наличие ключа 'bots'
            if 'bots' not in bots_data:
                logger.warning("[POSITION_SYNC] ⚠️ bots_data не содержит ключ 'bots' - инициализируем")
                bots_data['bots'] = {}
                return False
            
            for symbol, bot_data in bots_data['bots'].items():
                if bot_data.get('status') in ['in_position_long', 'in_position_short']:
                    bot_positions.append({
                        'symbol': symbol,
                        'position_side': bot_data.get('position_side'),
                        'entry_price': bot_data.get('entry_price'),
                        'status': bot_data.get('status'),
                        'unrealized_pnl': bot_data.get('unrealized_pnl', 0)
                    })
        
        # ✅ Логируем только при изменениях или ошибках (убираем спам)
        # logger.info(f"[POSITION_SYNC] 📊 Биржа: {len(exchange_positions)}, Боты: {len(bot_positions)}")
        
        # Создаем словари для удобного сравнения
        exchange_dict = {pos['symbol']: pos for pos in exchange_positions}
        bot_dict = {pos['symbol']: pos for pos in bot_positions}
        
        synced_count = 0
        errors_count = 0
        
        # Обрабатываем ботов без позиций на бирже
        for symbol, bot_data in bot_dict.items():
            if symbol not in exchange_dict:
                logger.warning(f"[POSITION_SYNC] ⚠️ Бот {symbol} без позиции на бирже (статус: {bot_data['status']})")
                
                # ВАЖНО: Проверяем, действительно ли позиция закрылась
                # Не сбрасываем ботов сразу - даем им время на восстановление
                try:
                    # Проверяем, есть ли активные ордера для этого символа
                    has_active_orders = check_active_orders(symbol)
                    
                    if not has_active_orders:
                        # Только если нет активных ордеров, сбрасываем бота
                        with bots_data_lock:
                            if symbol in bots_data['bots']:
                                bots_data['bots'][symbol]['status'] = 'idle'
                                bots_data['bots'][symbol]['position_side'] = None
                                bots_data['bots'][symbol]['entry_price'] = None
                                bots_data['bots'][symbol]['unrealized_pnl'] = 0
                                bots_data['bots'][symbol]['last_update'] = datetime.now().isoformat()
                                synced_count += 1
                                logger.info(f"[POSITION_SYNC] ✅ Сброшен статус бота {symbol} на 'idle' (позиция закрыта)")
                    else:
                        logger.info(f"[POSITION_SYNC] ⏳ Бот {symbol} имеет активные ордера - оставляем в позиции")
                        
                except Exception as check_error:
                    logger.error(f"[POSITION_SYNC] ❌ Ошибка проверки ордеров для {symbol}: {check_error}")
                    errors_count += 1
        
        # Обрабатываем несовпадения сторон - исправляем данные бота в соответствии с биржей
        for symbol, exchange_pos in exchange_dict.items():
            if symbol in bot_dict:
                bot_data = bot_dict[symbol]
                exchange_side = exchange_pos['position_side']
                bot_side = bot_data['position_side']
                
                if exchange_side != bot_side:
                    logger.warning(f"[POSITION_SYNC] 🔄 Исправление стороны позиции: {symbol} {bot_side} -> {exchange_side}")
                    
                    try:
                        with bots_data_lock:
                            if symbol in bots_data['bots']:
                                bots_data['bots'][symbol]['position_side'] = exchange_side
                                bots_data['bots'][symbol]['entry_price'] = exchange_pos['entry_price']
                                bots_data['bots'][symbol]['status'] = f'in_position_{exchange_side.lower()}'
                                bots_data['bots'][symbol]['unrealized_pnl'] = exchange_pos['unrealized_pnl']
                                bots_data['bots'][symbol]['last_update'] = datetime.now().isoformat()
                                synced_count += 1
                                logger.info(f"[POSITION_SYNC] ✅ Исправлены данные бота {symbol} в соответствии с биржей")
                    except Exception as update_error:
                        logger.error(f"[POSITION_SYNC] ❌ Ошибка обновления бота {symbol}: {update_error}")
                        errors_count += 1
        
        # Логируем результаты
        if synced_count > 0:
            logger.info(f"[POSITION_SYNC] ✅ Синхронизировано {synced_count} ботов")
        if errors_count > 0:
            logger.warning(f"[POSITION_SYNC] ⚠️ Ошибок при синхронизации: {errors_count}")
        
        return synced_count > 0
        
    except Exception as e:
        logger.error(f"[POSITION_SYNC] ❌ Критическая ошибка синхронизации позиций: {e}")
        return False

def check_active_orders(symbol):
    """Проверяет, есть ли активные ордера для символа"""
    try:
        if not ensure_exchange_initialized():
            return False
        
        # Получаем активные ордера для символа
        current_exchange = get_exchange()
        if not current_exchange:
            return False
        orders = current_exchange.get_open_orders(symbol)
        return len(orders) > 0
        
    except Exception as e:
        logger.error(f"[ORDER_CHECK] ❌ Ошибка проверки ордеров для {symbol}: {e}")
        return False

def cleanup_inactive_bots():
    """Удаляет ботов, которые не имеют реальных позиций на бирже в течение SystemConfig.INACTIVE_BOT_TIMEOUT секунд"""
    try:
        current_time = time.time()
        removed_count = 0
        
        # Получаем реальные позиции с биржи
        exchange_positions = get_exchange_positions()
        
        # КРИТИЧЕСКИ ВАЖНО: Если не удалось получить позиции с биржи, НЕ УДАЛЯЕМ ботов!
        if exchange_positions is None:
            logger.warning(f" ⚠️ Не удалось получить позиции с биржи - пропускаем очистку для безопасности")
            return False
        
        # Нормализуем символы позиций (убираем USDT если есть)
        def normalize_symbol(symbol):
            """Нормализует символ, убирая USDT суффикс если есть"""
            if symbol.endswith('USDT'):
                return symbol[:-4]  # Убираем 'USDT'
            return symbol
        
        # Создаем множество нормализованных символов позиций на бирже
        exchange_symbols = {normalize_symbol(pos['symbol']) for pos in exchange_positions}
        
        logger.info(f" 🔍 Проверка {len(bots_data['bots'])} ботов на неактивность")
        logger.info(f" 📊 Найдено {len(exchange_symbols)} активных позиций на бирже: {sorted(exchange_symbols)}")
        
        with bots_data_lock:
            bots_to_remove = []
            
            for symbol, bot_data in bots_data['bots'].items():
                bot_status = bot_data.get('status', 'idle')
                last_update_str = bot_data.get('last_update')
                
                # КРИТИЧЕСКИ ВАЖНО: НЕ УДАЛЯЕМ ботов, которые находятся в позиции!
                if bot_status in ['in_position_long', 'in_position_short']:
                    logger.info(f" 🛡️ Бот {symbol} в позиции {bot_status} - НЕ УДАЛЯЕМ")
                    continue
                
                # Пропускаем ботов, которые имеют реальные позиции на бирже
                # Нормализуем символ бота для корректного сравнения
                normalized_bot_symbol = normalize_symbol(symbol)
                if normalized_bot_symbol in exchange_symbols:
                    continue
                
                # Убрали хардкод - теперь проверяем только реальные позиции на бирже
                
                # Пропускаем ботов в статусе 'idle' - они могут быть в ожидании
                if bot_status == 'idle':
                    continue
                
                # КРИТИЧЕСКИ ВАЖНО: Не удаляем ботов, которые только что загружены
                # Проверяем, что бот был создан недавно (в течение последних 5 минут)
                created_time_str = bot_data.get('created_time')
                if created_time_str:
                    try:
                        created_time = datetime.fromisoformat(created_time_str.replace('Z', '+00:00'))
                        time_since_creation = current_time - created_time.timestamp()
                        if time_since_creation < 300:  # 5 минут
                            logger.info(f" ⏳ Бот {symbol} создан {time_since_creation//60:.0f} мин назад, пропускаем удаление")
                            continue
                    except Exception as e:
                        logger.warning(f" ⚠️ Ошибка парсинга времени создания для {symbol}: {e}")
                
                # Проверяем время последнего обновления
                if last_update_str:
                    try:
                        last_update = datetime.fromisoformat(last_update_str.replace('Z', '+00:00'))
                        time_since_update = current_time - last_update.timestamp()
                        
                        if time_since_update >= SystemConfig.INACTIVE_BOT_TIMEOUT:
                            logger.warning(f" ⏰ Бот {symbol} неактивен {time_since_update//60:.0f} мин (статус: {bot_status})")
                            bots_to_remove.append(symbol)
                            
                            # Логируем удаление неактивного бота в историю
                            # log_bot_stop(symbol, f"Неактивен {time_since_update//60:.0f} мин (статус: {bot_status})")  # TODO: Функция не определена
                        else:
                            logger.info(f" ⏳ Бот {symbol} неактивен {time_since_update//60:.0f} мин, ждем до {SystemConfig.INACTIVE_BOT_TIMEOUT//60} мин")
                    except Exception as e:
                        logger.error(f" ❌ Ошибка парсинга времени для {symbol}: {e}")
                        # Если не можем распарсить время, считаем бота неактивным
                        bots_to_remove.append(symbol)
                else:
                    # ✅ КРИТИЧНО: Если нет last_update, проверяем created_at
                    # Свежесозданные боты не должны удаляться!
                    created_at_str = bot_data.get('created_at')
                    if created_at_str:
                        try:
                            created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                            time_since_creation = current_time - created_at.timestamp()
                            
                            if time_since_creation < 300:  # 5 минут
                                logger.info(f" ⏳ Бот {symbol} создан {time_since_creation//60:.0f} мин назад, нет last_update - пропускаем удаление")
                                continue
                            else:
                                logger.warning(f" ⏰ Бот {symbol} без last_update и создан {time_since_creation//60:.0f} мин назад - удаляем")
                                bots_to_remove.append(symbol)
                        except Exception as e:
                            logger.error(f" ❌ Ошибка парсинга created_at для {symbol}: {e}")
                            # Если не можем распарсить, НЕ УДАЛЯЕМ (безопаснее)
                            logger.warning(f" ⚠️ Бот {symbol} без времени - НЕ УДАЛЯЕМ для безопасности")
                    else:
                        # Нет ни last_update, ни created_at - очень странная ситуация
                        logger.warning(f" ⚠️ Бот {symbol} без времени обновления и создания - НЕ УДАЛЯЕМ для безопасности")
            
            # Удаляем неактивных ботов
            for symbol in bots_to_remove:
                bot_data = bots_data['bots'][symbol]
                logger.info(f" 🗑️ Удаление неактивного бота {symbol} (статус: {bot_data.get('status')})")
                
                # ✅ УДАЛЯЕМ ПОЗИЦИЮ ИЗ РЕЕСТРА ПРИ УДАЛЕНИИ НЕАКТИВНОГО БОТА
                try:
                    from bots_modules.imports_and_globals import unregister_bot_position
                    position = bot_data.get('position')
                    if position and position.get('order_id'):
                        order_id = position['order_id']
                        unregister_bot_position(order_id)
                        logger.info(f" ✅ Позиция удалена из реестра при удалении неактивного бота {symbol}: order_id={order_id}")
                    else:
                        logger.info(f" ℹ️ У неактивного бота {symbol} нет позиции в реестре")
                except Exception as registry_error:
                    logger.error(f" ❌ Ошибка удаления позиции из реестра для бота {symbol}: {registry_error}")
                    # Не блокируем удаление бота из-за ошибки реестра
                
                del bots_data['bots'][symbol]
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f" ✅ Удалено {removed_count} неактивных ботов")
            # Сохраняем состояние
            save_bots_state()
        else:
            logger.info(f" ✅ Неактивных ботов для удаления не найдено")
        
        return removed_count > 0
        
    except Exception as e:
        logger.error(f" ❌ Ошибка очистки неактивных ботов: {e}")
        return False

# УДАЛЕНО: cleanup_mature_coins_without_trades()
# Зрелость монеты необратима - если монета стала зрелой, она не может стать незрелой!
# Файл зрелых монет можно только дополнять новыми, но не очищать от старых

def remove_mature_coins(coins_to_remove):
    """
    Удаляет конкретные монеты из файла зрелых монет
    
    Args:
        coins_to_remove: список символов монет для удаления (например: ['ARIA', 'AVNT'])
    
    Returns:
        dict: результат операции с количеством удаленных монет
    """
    try:
        if not isinstance(coins_to_remove, list):
            coins_to_remove = [coins_to_remove]
        
        removed_count = 0
        not_found = []
        
        logger.info(f"[MATURE_REMOVE] 🗑️ Запрос на удаление монет: {coins_to_remove}")
        
        with mature_coins_lock:
            for symbol in coins_to_remove:
                if symbol in mature_coins_storage:
                    del mature_coins_storage[symbol]
                    removed_count += 1
                    logger.info(f"[MATURE_REMOVE] ✅ Удалена монета {symbol} из зрелых")
                else:
                    not_found.append(symbol)
                    logger.warning(f"[MATURE_REMOVE] ⚠️ Монета {symbol} не найдена в зрелых")
        
        # Сохраняем изменения
        if removed_count > 0:
            save_mature_coins_storage()
            logger.info(f"[MATURE_REMOVE] 💾 Сохранено состояние зрелых монет")
        
        return {
            'success': True,
            'removed_count': removed_count,
            'removed_coins': [coin for coin in coins_to_remove if coin not in not_found],
            'not_found': not_found,
            'message': f'Удалено {removed_count} монет из зрелых'
        }
        
    except Exception as e:
        logger.error(f"[MATURE_REMOVE] ❌ Ошибка удаления монет: {e}")
        return {
            'success': False,
            'error': str(e),
            'removed_count': 0
        }

def check_trading_rules_activation():
    """Проверяет и активирует правила торговли для зрелых монет"""
    try:
        # КРИТИЧЕСКАЯ ПРОВЕРКА: Auto Bot должен быть включен для автоматического создания ботов
        with bots_data_lock:
            auto_bot_enabled = bots_data.get('auto_bot_config', {}).get('enabled', False)
        
        if not auto_bot_enabled:
            logger.info(f" ⏹️ Auto Bot выключен - пропускаем активацию правил торговли")
            return False
        
        current_time = time.time()
        activated_count = 0
        
        logger.info(f" 🔍 Проверка активации правил торговли для зрелых монет")
        
        # ✅ ИСПРАВЛЕНИЕ: НЕ создаем ботов автоматически для всех зрелых монет!
        # Вместо этого просто обновляем время проверки в mature_coins_storage
        
        with mature_coins_lock:
            for symbol, coin_data in mature_coins_storage.items():
                last_verified = coin_data.get('last_verified', 0)
                time_since_verification = current_time - last_verified
                
                # Если монета зрелая и не проверялась более 5 минут, обновляем время проверки
                if time_since_verification > 300:  # 5 минут
                    # Обновляем время последней проверки
                    coin_data['last_verified'] = current_time
                    activated_count += 1
        
        if activated_count > 0:
            logger.info(f" ✅ Обновлено время проверки для {activated_count} зрелых монет")
            # Сохраняем обновленные данные зрелых монет
            save_mature_coins_storage()
        else:
            logger.info(f" ✅ Нет зрелых монет для обновления времени проверки")
        
        return activated_count > 0
        
    except Exception as e:
        logger.error(f" ❌ Ошибка активации правил торговли: {e}")
        return False

def check_missing_stop_losses():
    """Проверяет и устанавливает недостающие стоп-лоссы и трейлинг стопы для ботов
    
    КРИТИЧЕСКАЯ ФУНКЦИЯ: От работы этой функции зависит защита средств!
    Если exchange недоступен - это КРИТИЧЕСКАЯ ОШИБКА, а не предупреждение!
    """
    try:
        # Шаг 1: Проверка и инициализация exchange - АГРЕССИВНЫЙ ПОДХОД
        current_exchange = None
        max_retries = 3
        
        for attempt in range(max_retries):
            # Попытка 1: Получить через get_exchange()
            try:
                current_exchange = get_exchange()
                if current_exchange:
                    break
            except (NameError, AttributeError) as e:
                logger.debug(f" get_exchange() недоступен: {e}")
            
            # Попытка 2: Использовать глобальную переменную exchange
            if not current_exchange:
                try:
                    current_exchange = exchange
                    if current_exchange:
                        break
                except NameError:
                    pass
            
            # Попытка 3: Попытаться инициализировать через ensure_exchange_initialized
            if not current_exchange:
                logger.error(f" ❌ КРИТИЧЕСКАЯ ОШИБКА: Exchange недоступен (попытка {attempt + 1}/{max_retries})")
                if ensure_exchange_initialized():
                    # После инициализации ВСЕГДА используем get_exchange() (не локальную переменную exchange!)
                    # потому что set_exchange() обновляет _state.exchange, но не локальные переменные
                    try:
                        current_exchange = get_exchange()
                        if current_exchange:
                            logger.info(f" ✅ Exchange восстановлен после инициализации через get_exchange() (попытка {attempt + 1})")
                            break
                        else:
                            logger.error(f" ❌ ensure_exchange_initialized() вернул True, но get_exchange() всё ещё None!")
                    except Exception as e:
                        logger.error(f" ❌ Ошибка получения exchange после инициализации: {e}")
                else:
                    logger.error(f" ❌ КРИТИЧЕСКАЯ ОШИБКА: ensure_exchange_initialized() вернул False (попытка {attempt + 1}/{max_retries})")
            
            # Если не получилось, ждем немного перед следующей попыткой
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
        
        # ФИНАЛЬНАЯ ПРОВЕРКА: Если exchange всё ещё недоступен - это КРИТИЧЕСКАЯ ОШИБКА
        if not current_exchange:
            logger.error(f" ❌ КРИТИЧЕСКАЯ ОШИБКА: Exchange объект недоступен после {max_retries} попыток!")
            # Безопасная диагностика
            try:
                get_exchange_result = get_exchange()
                logger.error(f" ❌ get_exchange() = {get_exchange_result}")
            except Exception as e:
                logger.error(f" ❌ get_exchange() недоступен: {e}")
            try:
                logger.error(f" ❌ exchange = {exchange}")
            except Exception as e:
                logger.error(f" ❌ exchange недоступен: {e}")
            logger.error(f" ❌ СТОП-ЛОССЫ НЕ МОГУТ БЫТЬ УСТАНОВЛЕНЫ! Это критическая проблема безопасности!")
            logger.error(f" ❌ Проверьте: 1) Ключи API корректны 2) Сеть доступна 3) Биржа работает")
            return False
        
        logger.debug(f" ✅ Exchange получен успешно: {type(current_exchange)}")
        from bots_modules.bot_class import NewTradingBot
        
        with bots_data_lock:
            
            # Получаем все позиции с биржи
            try:
                
                positions_response = current_exchange.client.get_positions(
                    category="linear",
                    settleCoin="USDT"
                )
                
                if positions_response.get('retCode') != 0:
                    logger.error(f" ❌ КРИТИЧЕСКАЯ ОШИБКА получения позиций: {positions_response.get('retMsg')} (retCode={positions_response.get('retCode')})")
                    return False
                
                exchange_positions = positions_response.get('result', {}).get('list', [])
                
            except Exception as e:
                logger.error(f" ❌ Ошибка получения позиций с биржи: {e}")
                return False
            
            updated_count = 0
            failed_count = 0
            
            # Обрабатываем каждого бота в позиции
            for symbol, bot_data in bots_data['bots'].items():
                if bot_data.get('status') not in ['in_position_long', 'in_position_short']:
                    continue
                try:
                    # Ищем позицию на бирже для этого символа
                    pos = None
                    for position in exchange_positions:
                        pos_symbol = position.get('symbol', '').replace('USDT', '')
                        if pos_symbol == symbol:
                            pos = position
                            break
                    
                    if not pos:
                        logger.warning(f" ⚠️ Позиция {symbol} не найдена на бирже")
                        continue
                    
                    position_size = float(pos.get('size', 0))
                    if position_size <= 0:
                        logger.warning(f" ⚠️ Позиция {symbol} закрыта на бирже")
                        continue
                    
                    # Получаем данные позиции
                    entry_price = float(pos.get('avgPrice', 0))
                    current_price = float(pos.get('markPrice', 0))
                    unrealized_pnl = float(pos.get('unrealisedPnl', 0))
                    side = pos.get('side', '')
                    position_idx = pos.get('positionIdx', 0)
                    existing_stop_loss = pos.get('stopLoss', '')
                    existing_trailing_stop = pos.get('trailingStop', '')
                    existing_take_profit = pos.get('takeProfit', '')
                    
                    # Рассчитываем процент прибыли/убытка
                    if side == 'Buy':  # LONG позиция
                        profit_percent = ((current_price - entry_price) / entry_price) * 100
                    else:  # SHORT позиция
                        profit_percent = ((entry_price - current_price) / entry_price) * 100
                    
                    logger.info(f" 📊 {symbol}: PnL {profit_percent:.2f}%, текущая цена {current_price}, вход {entry_price}")
                    
                    # Импортируем состояние из Protection Engine для точного совпадения логики
                    position_side = 'LONG' if side == 'Buy' else 'SHORT'
                    position_qty = abs(_safe_float(pos.get('size'), 0.0) or 0.0)
                    if position_qty <= 0:
                        logger.warning(f" ⚠️ Позиция {symbol} имеет нулевой объём — пропуск")
                        continue

                    entry_timestamp = (
                        _normalize_timestamp(bot_data.get('entry_timestamp'))
                        or _normalize_timestamp(bot_data.get('position_start_time'))
                        or _normalize_timestamp(pos.get('createdTime') or pos.get('updatedTime'))
                    )
                    runtime_config = dict(bot_data or {})
                    runtime_config['entry_price'] = entry_price
                    runtime_config['position_side'] = position_side
                    runtime_config['position_size_coins'] = position_qty
                    runtime_config['volume_value'] = runtime_config.get('volume_value') or (
                        entry_price * position_qty if entry_price else None
                    )
                    if entry_timestamp:
                        runtime_config['entry_timestamp'] = entry_timestamp
                        runtime_config['position_start_time'] = _timestamp_to_iso(entry_timestamp)

                    bot_instance = NewTradingBot(symbol, config=runtime_config, exchange=current_exchange)
                    bot_instance.entry_price = entry_price
                    bot_instance.position_side = position_side
                    bot_instance.position_size_coins = position_qty
                    bot_instance.position_size = (entry_price * position_qty) if entry_price else bot_instance.position_size
                    bot_instance.realized_pnl = _safe_float(pos.get('cumRealisedPnl') or pos.get('realisedPnl') or pos.get('realizedPnl'), 0.0)
                    bot_instance.unrealized_pnl = unrealized_pnl
                    if entry_timestamp:
                        bot_instance.entry_timestamp = entry_timestamp
                        bot_instance.position_start_time = datetime.fromtimestamp(entry_timestamp)

                    decision = bot_instance._evaluate_protection_decision(current_price)
                    config = bot_instance._get_effective_protection_config()
                    _apply_protection_state_to_bot_data(bot_data, decision.state)

                    bot_data['entry_price'] = entry_price
                    bot_data['position_side'] = position_side
                    bot_data['position_size_coins'] = position_qty
                    bot_data['position_size'] = entry_price * position_qty if entry_price else bot_data.get('position_size')
                    bot_data['realized_pnl'] = bot_instance.realized_pnl
                    bot_data['unrealized_pnl'] = unrealized_pnl
                    bot_data['current_price'] = current_price
                    bot_data['leverage'] = _safe_float(pos.get('leverage'), bot_data.get('leverage', 1.0)) or 1.0
                    if entry_timestamp:
                        bot_data['entry_timestamp'] = entry_timestamp
                        bot_data['position_start_time'] = _timestamp_to_iso(entry_timestamp)

                    if decision.should_close:
                        logger.warning(
                            f" ⚠️ Protection Engine сигнализирует закрытие {symbol}: {decision.reason}"
                        )

                    # Синхронизируем существующие стопы/тейки из биржи
                    if existing_stop_loss:
                        bot_data['stop_loss_price'] = float(existing_stop_loss)
                    if existing_take_profit:
                        bot_data['take_profit_price'] = float(existing_take_profit)
                    if existing_trailing_stop:
                        bot_data['trailing_stop_price'] = float(existing_trailing_stop)

                    desired_stop = _select_stop_loss_price(
                        position_side,
                        entry_price,
                        current_price,
                        config,
                        bot_instance.break_even_stop_price,
                        bot_instance.trailing_stop_price,
                    )
                    existing_stop_value = _safe_float(existing_stop_loss)
                    if desired_stop and _needs_price_update(position_side, desired_stop, existing_stop_value):
                        try:
                            sl_response = current_exchange.update_stop_loss(
                                symbol=symbol,
                                stop_loss_price=desired_stop,
                                position_side=position_side,
                            )
                            if sl_response and sl_response.get('success'):
                                bot_data['stop_loss_price'] = desired_stop
                                updated_count += 1
                                logger.info(f" ✅ Стоп-лосс синхронизирован для {symbol}: {desired_stop:.6f}")
                            else:
                                failed_count += 1
                                logger.error(f" ❌ Ошибка установки стоп-лосса для {symbol}: {sl_response}")
                        except Exception as e:
                            failed_count += 1
                            logger.error(f" ❌ Ошибка установки стоп-лосса для {symbol}: {e}")

                    desired_take = _select_take_profit_price(
                        position_side,
                        entry_price,
                        config,
                        bot_instance.trailing_take_profit_price,
                    )
                    existing_take_value = _safe_float(existing_take_profit)
                    if desired_take and _needs_price_update(position_side, desired_take, existing_take_value):
                        try:
                            tp_response = current_exchange.update_take_profit(
                                symbol=symbol,
                                take_profit_price=desired_take,
                                position_side=position_side,
                            )
                            if tp_response and tp_response.get('success'):
                                bot_data['take_profit_price'] = desired_take
                                updated_count += 1
                                logger.info(f" ✅ Тейк-профит синхронизирован для {symbol}: {desired_take:.6f}")
                            else:
                                failed_count += 1
                                logger.error(f" ❌ Ошибка установки тейк-профита для {symbol}: {tp_response}")
                        except Exception as e:
                            failed_count += 1
                            logger.error(f" ❌ Ошибка установки тейк-профита для {symbol}: {e}")
                    
                    # Обновляем время последнего обновления
                    bot_data['last_update'] = datetime.now().isoformat()
                        
                except Exception as e:
                    logger.error(f" ❌ Ошибка обработки {symbol}: {e}")
                    failed_count += 1
                    continue
            
            if updated_count > 0 or failed_count > 0:
                logger.info(f" ✅ Установка завершена: установлено {updated_count}, ошибок {failed_count}")
                
                # Сохраняем обновленные данные ботов в файл
                if updated_count > 0:
                    try:
                        save_bots_state()
                        logger.info(f" 💾 Сохранено состояние ботов в файл")
                    except Exception as save_error:
                        logger.error(f" ❌ Ошибка сохранения состояния ботов: {save_error}")
            
            return True
            
    except Exception as e:
        logger.error(f" ❌ Ошибка установки стоп-лоссов: {e}")
        return False

def check_startup_position_conflicts():
    """Проверяет конфликты позиций при запуске системы и принудительно останавливает проблемные боты"""
    try:
        if not ensure_exchange_initialized():
            logger.warning(" ⚠️ Биржа не инициализирована, пропускаем проверку конфликтов")
            return False
        
        logger.info(" 🔍 Проверка конфликтов...")
        
        conflicts_found = 0
        bots_paused = 0
        
        with bots_data_lock:
            for symbol, bot_data in bots_data['bots'].items():
                try:
                    bot_status = bot_data.get('status')
                    
                    # Проверяем только активные боты (не idle/paused)
                    if bot_status in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']]:
                        continue
                    
                    # Проверяем позицию на бирже
                    from bots_modules.imports_and_globals import get_exchange
                    current_exchange = get_exchange() or exchange
                    positions_response = current_exchange.client.get_positions(
                        category="linear",
                        symbol=f"{symbol}USDT"
                    )
                    
                    if positions_response.get('retCode') == 0:
                        positions = positions_response['result']['list']
                        has_position = False
                        
                        # Фильтруем позиции только для нужного символа
                        target_symbol = f"{symbol}USDT"
                        for pos in positions:
                            pos_symbol = pos.get('symbol', '')
                            if pos_symbol == target_symbol:  # Проверяем только нужный символ
                                size = float(pos.get('size', 0))
                                if abs(size) > 0:  # Есть активная позиция
                                    has_position = True
                                    side = 'LONG' if pos.get('side') == 'Buy' else 'SHORT'
                                    break
                        
                        # Проверяем конфликт
                        if has_position:
                            # Есть позиция на бирже
                            if bot_status in [BOT_STATUS['RUNNING']]:
                                # КОНФЛИКТ: бот активен, но позиция уже есть на бирже
                                logger.warning(f" 🚨 {symbol}: КОНФЛИКТ! Бот {bot_status}, но позиция {side} уже есть на бирже!")
                                
                                # Принудительно останавливаем бота
                                bot_data['status'] = BOT_STATUS['PAUSED']
                                bot_data['last_update'] = datetime.now().isoformat()
                                
                                conflicts_found += 1
                                bots_paused += 1
                                
                                logger.warning(f" 🔴 {symbol}: Бот принудительно остановлен (PAUSED)")
                                
                            elif bot_status in [BOT_STATUS['IN_POSITION_LONG'], BOT_STATUS['IN_POSITION_SHORT']]:
                                # Корректное состояние - бот в позиции
                                logger.debug(f" ✅ {symbol}: Статус корректный - бот в позиции")
                        else:
                            # Нет позиции на бирже
                            if bot_status in [BOT_STATUS['IN_POSITION_LONG'], BOT_STATUS['IN_POSITION_SHORT']]:
                                # КОНФЛИКТ: бот думает что в позиции, но позиции нет на бирже
                                logger.warning(f" 🚨 {symbol}: КОНФЛИКТ! Бот показывает позицию, но на бирже её нет!")
                                
                                # Сбрасываем статус бота
                                bot_data['status'] = BOT_STATUS['IDLE']
                                bot_data['entry_price'] = None
                                bot_data['position_side'] = None
                                bot_data['unrealized_pnl'] = 0.0
                                bot_data['last_update'] = datetime.now().isoformat()
                                
                                conflicts_found += 1
                                
                                logger.warning(f" 🔄 {symbol}: Статус сброшен в IDLE")
                            else:
                                # Корректное состояние - нет позиций
                                logger.debug(f" ✅ {symbol}: Статус корректный - нет позиций")
                    else:
                        logger.warning(f" ❌ {symbol}: Ошибка получения позиций: {positions_response.get('retMsg', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f" ❌ Ошибка проверки {symbol}: {e}")
        
        if conflicts_found > 0:
            logger.warning(f" 🚨 Найдено {conflicts_found} конфликтов, остановлено {bots_paused} ботов")
            # Сохраняем обновленное состояние
            save_bots_state()
        else:
            logger.info(" ✅ Конфликтов позиций не найдено")
        
        return conflicts_found > 0
        
    except Exception as e:
        logger.error(f" ❌ Общая ошибка проверки конфликтов: {e}")
        return False

def sync_bots_with_exchange():
    """Синхронизирует состояние ботов с открытыми позициями на бирже"""
    import time
    start_time = time.time()
    
    try:
        # Убираем лишние логи - оставляем только итог
        if not ensure_exchange_initialized():
            logger.warning("[SYNC_EXCHANGE] ⚠️ Биржа не инициализирована, пропускаем синхронизацию")
            return False
        
        # Получаем ВСЕ открытые позиции с биржи (с пагинацией)
        try:
            exchange_positions = {}
            cursor = ""
            total_positions = 0
            iteration = 0
            
            while True:
                iteration += 1
                iter_start = time.time()
                
                # Запрашиваем позиции с cursor для получения всех страниц
                params = {
                    "category": "linear", 
                    "settleCoin": "USDT",
                    "limit": 200  # Максимум за запрос
                }
                if cursor:
                    params["cursor"] = cursor
                
                from bots_modules.imports_and_globals import get_exchange
                current_exchange = get_exchange() or exchange
                
                # Проверяем что биржа инициализирована
                if not current_exchange or not hasattr(current_exchange, 'client'):
                    logger.error(f"[SYNC_EXCHANGE] ❌ Биржа не инициализирована")
                    return False
                
                # 🔥 УПРОЩЕННЫЙ ПОДХОД: быстрый таймаут на уровне SDK
                positions_response = None
                timeout_seconds = 8  # Короткий таймаут
                max_retries = 2
                
                for retry in range(max_retries):
                    retry_start = time.time()
                    try:
                        # Устанавливаем короткий таймаут на уровне клиента
                        old_timeout = getattr(current_exchange.client, 'timeout', None)
                        current_exchange.client.timeout = timeout_seconds
                        
                        positions_response = current_exchange.client.get_positions(**params)
                        
                        # Восстанавливаем таймаут
                        if old_timeout is not None:
                            current_exchange.client.timeout = old_timeout
                        
                        break  # Успех!
                        
                    except Exception as e:
                        logger.debug(f"[SYNC_EXCHANGE] Повтор {retry + 1}/{max_retries}: {e}")
                        if retry < max_retries - 1:
                            time.sleep(2)
                        else:
                            logger.error(f"[SYNC_EXCHANGE] ❌ Все попытки провалились")
                            return False
                
                # Проверяем что получили ответ
                if positions_response is None:
                    logger.error(f"[SYNC_EXCHANGE] ❌ Пустой ответ")
                    return False
                
                if positions_response["retCode"] != 0:
                    logger.error(f"[SYNC_EXCHANGE] ❌ Ошибка: {positions_response['retMsg']}")
                    return False
                
                # Обрабатываем позиции на текущей странице
                positions_count = len(positions_response["result"]["list"])
                
                for idx, position in enumerate(positions_response["result"]["list"]):
                    symbol = position.get("symbol")
                    size = float(position.get("size", 0))
                    
                    if abs(size) > 0:  # Любые открытые позиции (LONG или SHORT)
                        # Убираем USDT из символа для сопоставления с ботами
                        clean_symbol = symbol.replace('USDT', '')
                        exchange_positions[clean_symbol] = {
                            'size': abs(size),
                            'side': position.get("side"),
                            'avg_price': float(position.get("avgPrice", 0)),
                            'unrealized_pnl': float(position.get("unrealisedPnl", 0)),
                            'position_value': float(position.get("positionValue", 0)),
                            'stop_loss': position.get("stopLoss", ''),
                            'take_profit': position.get("takeProfit", ''),
                            'mark_price': position.get("markPrice", 0)
                        }
                        total_positions += 1
                
                # Проверяем есть ли еще страницы
                next_page_cursor = positions_response["result"].get("nextPageCursor", "")
                if not next_page_cursor:
                    break
                cursor = next_page_cursor
            
            # ✅ Не логируем общее количество (избыточно)
            
            # Получаем символы ботов в системе для фильтрации
            with bots_data_lock:
                system_bot_symbols = set(bots_data['bots'].keys())
            
            # Разделяем позиции на бирже на "с ботом" и "без бота"
            positions_with_bots = {}
            positions_without_bots = {}
            
            for symbol, pos_data in exchange_positions.items():
                # Проверяем как символ без USDT, так и с USDT
                if symbol in system_bot_symbols or f"{symbol}USDT" in system_bot_symbols:
                    positions_with_bots[symbol] = pos_data
                else:
                    positions_without_bots[symbol] = pos_data
            
            # Логируем только итоговый результат
            
            # Синхронизируем только с позициями, для которых есть боты
            synchronized_bots = 0
            
            # ✅ ИСПРАВЛЕНИЕ: Используем list() для безопасной итерации (предотвращаем "dictionary changed size during iteration")
            with bots_data_lock:
                bot_items = list(bots_data['bots'].items())  # Создаем копию списка
            
            for symbol, bot_data in bot_items:
                    try:
                        if symbol in positions_with_bots:
                            # Есть позиция на бирже - обновляем данные бота
                            exchange_pos = positions_with_bots[symbol]
                            
                            # Получаем старые данные для логирования
                            with bots_data_lock:
                                if symbol not in bots_data['bots']:
                                    continue  # Бот был удалён в другом потоке
                                bot_data = bots_data['bots'][symbol]
                                old_status = bot_data.get('status', 'UNKNOWN')
                                old_pnl = bot_data.get('unrealized_pnl', 0)
                                
                                # ⚡ КРИТИЧНО: Не изменяем статус если бот был остановлен вручную!
                                is_paused = old_status == BOT_STATUS['PAUSED']
                                
                                bot_data['entry_price'] = exchange_pos['avg_price']
                                bot_data['unrealized_pnl'] = exchange_pos['unrealized_pnl']
                                bot_data['position_side'] = 'LONG' if exchange_pos['side'] == 'Buy' else 'SHORT'
                                
                                # Сохраняем стопы и тейки из биржи
                                if exchange_pos.get('stop_loss'):
                                    bot_data['stop_loss'] = exchange_pos['stop_loss']
                                if exchange_pos.get('take_profit'):
                                    bot_data['take_profit'] = exchange_pos['take_profit']
                                if exchange_pos.get('mark_price'):
                                    bot_data['current_price'] = exchange_pos['mark_price']
                                
                                # Определяем статус на основе наличия позиции (НЕ ИЗМЕНЯЕМ если бот на паузе!)
                                if not is_paused:
                                    if exchange_pos['side'] == 'Buy':
                                        bot_data['status'] = BOT_STATUS['IN_POSITION_LONG']
                                    else:
                                        bot_data['status'] = BOT_STATUS['IN_POSITION_SHORT']
                                else:
                                    logger.info(f"[SYNC_EXCHANGE] ⏸️ {symbol}: Бот на паузе - сохраняем статус PAUSED")
                            
                            synchronized_bots += 1
                            
                            # Добавляем детали позиции
                            entry_price = exchange_pos['avg_price']
                            current_price = exchange_pos.get('mark_price', entry_price)
                            position_size = exchange_pos.get('size', 0)
                            
                            # logger.info(f"[SYNC_EXCHANGE] 🔄 {symbol}: {old_status}→{bot_data['status']}, PnL: ${old_pnl:.2f}→${exchange_pos['unrealized_pnl']:.2f}")
                            # logger.info(f"[SYNC_EXCHANGE] 📊 {symbol}: Вход=${entry_price:.4f} | Текущая=${current_price:.4f} | Размер={position_size}")
                            
                        else:
                            # Нет позиции на бирже - проверяем статус инструмента
                            old_status = bot_data.get('status', 'UNKNOWN')
                            old_position_size = bot_data.get('position_size', 0)
                            manual_closed = old_status in [
                                BOT_STATUS.get('IN_POSITION_LONG'),
                                BOT_STATUS.get('IN_POSITION_SHORT')
                            ]

                            exit_price = None
                            entry_price = None
                            pnl_usdt = 0.0
                            roi_percent = 0.0
                            direction = bot_data.get('position_side')
                            position_size_coins = abs(float(bot_data.get('position_size_coins') or bot_data.get('position_size') or 0))

                            try:
                                entry_price = float(bot_data.get('entry_price') or 0.0)
                            except (TypeError, ValueError):
                                entry_price = 0.0

                            # Получаем рыночную цену для фиксации закрытия
                            if manual_closed:
                                try:
                                    exchange_obj = get_exchange()
                                    if exchange_obj and hasattr(exchange_obj, 'get_ticker'):
                                        ticker = exchange_obj.get_ticker(symbol)
                                        if ticker and ticker.get('last'):
                                            exit_price = float(ticker.get('last'))
                                except Exception as manual_price_error:
                                    logger.debug(f"[SYNC_EXCHANGE] ⚠️ Не удалось получить цену для ручного закрытия {symbol}: {manual_price_error}")

                            if not exit_price:
                                try:
                                    exit_price = float(bot_data.get('current_price') or 0.0)
                                except (TypeError, ValueError):
                                    exit_price = 0.0

                            if not direction:
                                if old_status == BOT_STATUS.get('IN_POSITION_LONG'):
                                    direction = 'LONG'
                                elif old_status == BOT_STATUS.get('IN_POSITION_SHORT'):
                                    direction = 'SHORT'

                            direction_upper = (direction or '').upper()
                            if manual_closed and entry_price and exit_price and position_size_coins and direction_upper in ('LONG', 'SHORT'):
                                price_diff = (exit_price - entry_price) if direction_upper == 'LONG' else (entry_price - exit_price)
                                pnl_usdt = price_diff * position_size_coins
                                margin_usdt = bot_data.get('margin_usdt')
                                try:
                                    margin_val = float(margin_usdt) if margin_usdt is not None else None
                                except (TypeError, ValueError):
                                    margin_val = None
                                if margin_val and margin_val != 0:
                                    roi_percent = (pnl_usdt / margin_val) * 100.0

                            if manual_closed:
                                entry_time_str = bot_data.get('position_start_time') or bot_data.get('entry_time')
                                duration_hours = 0.0
                                if entry_time_str:
                                    try:
                                        entry_time = datetime.fromisoformat(entry_time_str.replace('Z', ''))
                                        duration_hours = (datetime.utcnow() - entry_time).total_seconds() / 3600.0
                                    except Exception:
                                        duration_hours = 0.0

                                entry_data = {
                                    'entry_price': entry_price or None,
                                    'trend': bot_data.get('entry_trend'),
                                    'volatility': bot_data.get('entry_volatility'),
                                    'duration_hours': duration_hours,
                                    'max_profit_achieved': bot_data.get('max_profit_achieved')
                                }
                                market_data = {
                                    'exit_price': exit_price or entry_price or 0.0,
                                    'price_movement': ((exit_price - entry_price) / entry_price * 100.0) if entry_price else 0.0
                                }

                                bot_id = bot_data.get('id') or symbol
                                history_log_position_closed(
                                    bot_id=bot_id,
                                    symbol=symbol,
                                    direction=direction or 'UNKNOWN',
                                    exit_price=exit_price or entry_price or 0.0,
                                    pnl=pnl_usdt,
                                    roi=roi_percent,
                                    reason='MANUAL_CLOSE',
                                    entry_data=entry_data,
                                    market_data=market_data
                                )
                                logger.info(
                                    f"[SYNC_EXCHANGE] ✋ {symbol}: позиция закрыта вручную на бирже "
                                    f"(entry={entry_price:.6f}, exit={exit_price:.6f}, pnl={pnl_usdt:.2f} USDT)"
                                )
                            
                            # ✅ ПРОВЕРЯЕМ ДЕЛИСТИНГ: Получаем статус инструмента
                            try:
                                from bots_modules.imports_and_globals import get_exchange
                                exchange_obj = get_exchange()
                                if exchange_obj and hasattr(exchange_obj, 'get_instrument_status'):
                                    status_info = exchange_obj.get_instrument_status(f"{symbol}USDT")
                                    if status_info and status_info.get('is_delisting'):
                                        logger.warning(f"[SYNC_EXCHANGE] ⚠️ {symbol}: ДЕЛИСТИНГ обнаружен! Статус: {status_info.get('status')}")
                                        logger.info(f"[SYNC_EXCHANGE] 🗑️ {symbol}: Удаляем бота (делистинг: {status_info.get('status')})")
                                    else:
                                        logger.info(f"[SYNC_EXCHANGE] 🗑️ {symbol}: Удаляем бота (позиция закрыта на бирже, статус: {old_status})")
                                else:
                                    logger.info(f"[SYNC_EXCHANGE] 🗑️ {symbol}: Удаляем бота (позиция закрыта на бирже, статус: {old_status})")
                            except Exception as e:
                                logger.error(f"[SYNC_EXCHANGE] ❌ Ошибка проверки статуса {symbol}: {e}")
                                logger.info(f"[SYNC_EXCHANGE] 🗑️ {symbol}: Удаляем бота (позиция закрыта на бирже)")
                            
                            # Удаляем бота из системы (с блокировкой!)
                            with bots_data_lock:
                                if symbol in bots_data['bots']:
                                    del bots_data['bots'][symbol]
                            
                            # Сохраняем состояние после удаления
                            save_bots_state()
                            
                            synchronized_bots += 1
                        
                    except Exception as e:
                        logger.error(f"[SYNC_EXCHANGE] ❌ Ошибка синхронизации бота {symbol}: {e}")
            
            if synchronized_bots > 0:
                elapsed = time.time() - start_time
                logger.debug(f"[SYNC_EXCHANGE] Синхронизировано {synchronized_bots} ботов за {elapsed:.1f}с")
            
            # Сохраняем обновленное состояние
            save_bots_state()
            
            return True
            
        except Exception as e:
            logger.error(f"[SYNC_EXCHANGE] ❌ Ошибка получения позиций с биржи: {e}")
            return False
        
    except Exception as e:
        logger.error(f"[SYNC_EXCHANGE] ❌ Общая ошибка синхронизации: {e}")
        return False

