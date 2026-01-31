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
from datetime import datetime, timezone
from pathlib import Path
import copy
import math
import shutil

logger = logging.getLogger('BotsService')

# Импорт SystemConfig
from bot_engine.bot_config import SystemConfig
from bot_engine.bot_history import log_position_closed as history_log_position_closed
from bot_engine.storage import (
    save_bots_state as storage_save_bots_state,
    load_bots_state as storage_load_bots_state,
    save_rsi_cache as storage_save_rsi_cache,
    load_rsi_cache as storage_load_rsi_cache,
    save_process_state as storage_save_process_state,
    load_process_state as storage_load_process_state,
    save_delisted_coins as storage_save_delisted_coins,
    load_delisted_coins as storage_load_delisted_coins
)

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
    'mini_chart_update_interval': 'MINI_CHART_UPDATE_INTERVAL',
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
    'trend_require_candles': 'TREND_REQUIRE_CANDLES',
    'system_timeframe': 'SYSTEM_TIMEFRAME'  # Таймфрейм системы
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


def _check_if_trade_already_closed(bot_id, symbol, entry_price, entry_time_str):
    """✅ УПРОЩЕНО: Проверяет, была ли позиция уже закрыта ранее (предотвращает дубликаты)"""
    if not entry_price or entry_price <= 0:
        return False
    
    try:
        from bot_engine.bots_database import get_bots_database
        bots_db = get_bots_database()
        
        # Получаем entry_timestamp
        entry_timestamp = None
        if entry_time_str:
            try:
                entry_time = datetime.fromisoformat(entry_time_str.replace('Z', ''))
                entry_timestamp = entry_time.timestamp() * 1000
            except Exception:
                pass
        
        # Проверяем последние 10 закрытых сделок
        existing_trades = bots_db.get_bot_trades_history(
            bot_id=bot_id,
            symbol=symbol,
            status='CLOSED',
            limit=10
        )
        
        if not existing_trades:
            return False
        
        # Проверяем на дубликаты
        for existing_trade in existing_trades:
            existing_entry_price = existing_trade.get('entry_price')
            existing_entry_ts = existing_trade.get('entry_timestamp')
            existing_close_reason = existing_trade.get('close_reason')
            
            # Сравниваем цену входа (погрешность для float)
            price_match = existing_entry_price and abs(float(existing_entry_price) - float(entry_price)) < 0.0001
            
            # Сравниваем timestamp если есть (погрешность 1 минута)
            timestamp_match = True
            if entry_timestamp and existing_entry_ts:
                timestamp_match = abs(float(existing_entry_ts) - float(entry_timestamp)) < 60000
            
            # Если совпадает цена и timestamp, и это MANUAL_CLOSE - это дубликат
            if price_match and timestamp_match and existing_close_reason == 'MANUAL_CLOSE':
                return True
        
        return False
    except Exception as e:
        pass
        return False


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


def _snapshot_bots_for_protections():
    """Возвращает копию автоконфига и ботов в позициях для обработки вне блокировки."""
    with bots_data_lock:
        auto_config = copy.deepcopy(bots_data.get('auto_bot_config', DEFAULT_AUTO_BOT_CONFIG))
        bots_snapshot = {
            symbol: copy.deepcopy(bot_data)
            for symbol, bot_data in bots_data.get('bots', {}).items()
            if bot_data.get('status') in ['in_position_long', 'in_position_short']
        }
    return auto_config, bots_snapshot


def _update_bot_record(symbol, updates):
    """Безопасно применяет изменения к bot_data, минимизируя время блокировки."""
    if not updates:
        return False
    with bots_data_lock:
        bot_data = bots_data['bots'].get(symbol)
        if not bot_data:
            return False
        bot_data.update(updates)
    return True


def get_system_config_snapshot():
    """Возвращает текущие значения SystemConfig в формате, ожидаемом UI.
    Для system_timeframe берём фактический текущий таймфрейм (runtime/БД), а не только из конфига,
    иначе при сохранении других настроек в файл попадал бы старый SYSTEM_TIMEFRAME и сбрасывал таймфрейм на 6h.
    """
    snapshot = {}
    for key, attr in SYSTEM_CONFIG_FIELD_MAP.items():
        if key == 'system_timeframe':
            try:
                from bot_engine.bot_config import get_current_timeframe
                snapshot[key] = get_current_timeframe()
            except Exception:
                snapshot[key] = getattr(SystemConfig, attr, None)
        else:
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
    """Сохранить кэш RSI данных в БД"""
    try:
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция в Python
        coins_data = coins_rsi_data.get('coins', {})
        stats = {
            'total_coins': len(coins_data),
            'successful_coins': coins_rsi_data.get('successful_coins', 0),
            'failed_coins': coins_rsi_data.get('failed_coins', 0)
        }
        
        # ✅ Сохраняем в БД через storage.py
        if storage_save_rsi_cache(coins_data, stats):
            logger.info(f" RSI данные для {len(coins_data)} монет сохранены в БД")
            return True
        return False
        
    except Exception as e:
        logger.error(f" Ошибка сохранения RSI кэша в БД: {str(e)}")
        return False

def load_rsi_cache():
    """Загрузить кэш RSI данных из БД"""
    global coins_rsi_data
    
    try:
        # ✅ Загружаем из БД через storage.py
        cache_data = storage_load_rsi_cache()
        
        if not cache_data:
            logger.info(" RSI кэш в БД не найден, будет создан при первом обновлении")
            return False
        
        # Получаем данные из кэша
        cached_coins = cache_data.get('coins', {})
        stats = cache_data.get('stats', {})
        
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
                'successful_coins': stats.get('successful_coins', len(cached_coins)),
                'failed_coins': stats.get('failed_coins', 0),
                'total_coins': len(cached_coins),
                'last_update': datetime.now().isoformat(),  # Всегда используем текущее время
                'update_in_progress': False
            })
        
        logger.info(f" Загружено {len(cached_coins)} монет из RSI кэша (БД)")
        return True
        
    except Exception as e:
        logger.error(f" Ошибка загрузки RSI кэша из БД: {str(e)}")
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
    """Сохраняет состояние всех процессов в БД"""
    try:
        # ✅ Сохраняем в БД через storage.py
        if storage_save_process_state(process_state):
            # Убрано избыточное DEBUG логирование для уменьшения спама
            # logger.debug("💾 Состояние процессов сохранено в БД")
            return True
        return False
        
    except Exception as e:
        logger.error(f" ❌ Ошибка сохранения состояния процессов в БД: {e}")
        return False

def load_process_state():
    """Загружает состояние процессов из БД"""
    try:
        # ✅ Загружаем из БД через storage.py
        state_data = storage_load_process_state()
        
        if not state_data:
            logger.info(f" 📁 Состояние процессов в БД не найдено, начинаем с дефолтного")
            save_process_state()  # Создаем в БД
            return False
        
        if 'process_state' in state_data:
            # Обновляем глобальное состояние
            for process_name, process_info in state_data['process_state'].items():
                if process_name in process_state:
                    process_state[process_name].update(process_info)
            
            last_saved = state_data.get('last_saved', 'неизвестно')
            logger.info(f" ✅ Состояние процессов восстановлено из БД (сохранено: {last_saved})")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f" ❌ Ошибка загрузки состояния процессов из БД: {e}")
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
            pass
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
        # ✅ КРИТИЧНО: Сохраняем текущий таймфрейм из БД перед перезагрузкой модуля
        # чтобы не потерять его при reload (приоритет БД над конфигом)
        saved_timeframe_from_db = None
        try:
            from bot_engine.bots_database import get_bots_database
            db = get_bots_database()
            saved_timeframe_from_db = db.load_timeframe()
        except:
            pass
        
        bot_config_module = importlib.import_module('bot_engine.bot_config')
        importlib.reload(bot_config_module)
        file_system_config = bot_config_module.SystemConfig

        for attr in SYSTEM_CONFIG_FIELD_MAP.values():
            if hasattr(file_system_config, attr):
                setattr(SystemConfig, attr, getattr(file_system_config, attr))

        # ✅ КРИТИЧНО: Восстанавливаем таймфрейм после перезагрузки модуля
        # Приоритет: БД > SystemConfig.SYSTEM_TIMEFRAME из файла
        try:
            from bot_engine.bot_config import set_current_timeframe, get_current_timeframe
            if saved_timeframe_from_db:
                # Если есть таймфрейм в БД - используем его (пользователь переключал через UI)
                set_current_timeframe(saved_timeframe_from_db)
            else:
                # Если нет в БД - используем из конфига
                config_timeframe = getattr(file_system_config, 'SYSTEM_TIMEFRAME', None)
                if config_timeframe:
                    set_current_timeframe(config_timeframe)
        except Exception as tf_err:
            logger.warning(f"[SYSTEM_CONFIG] ⚠️ Ошибка восстановления таймфрейма: {tf_err}")

        logger.info("[SYSTEM_CONFIG] ✅ Конфигурация перезагружена из bot_engine/bot_config.py")
        return True

    except Exception as e:
        logger.error(f"[SYSTEM_CONFIG] ❌ Ошибка загрузки системных настроек: {e}")
        return False

def save_bots_state():
    """Сохраняет состояние всех ботов в БД. Без блокировки — быстрый снимок словаря (допустима минимальная рассинхронизация)."""
    try:
        # Снимок без lock: list(...) фиксирует ключи на момент вызова, копируем каждый bot dict
        bots_ref = bots_data.get('bots')
        if not bots_ref or not isinstance(bots_ref, dict):
            auto_bot_config_to_save = {}
            success = storage_save_bots_state({}, auto_bot_config_to_save)
        else:
            try:
                items_snapshot = list(bots_ref.items())
            except (RuntimeError, TypeError):
                items_snapshot = []
            bots_data_to_save = {}
            for symbol, bot_data in items_snapshot:
                try:
                    if isinstance(bot_data, dict):
                        bots_data_to_save[symbol] = dict(bot_data)
                except (TypeError, RuntimeError):
                    pass
            auto_bot_config_to_save = {}
            success = storage_save_bots_state(bots_data_to_save, auto_bot_config_to_save)
        if not success:
            logger.error("[SAVE_STATE] ❌ Ошибка сохранения состояния в БД")
            return False
        
        # Убрано избыточное DEBUG логирование для уменьшения спама
        # logger.debug("[SAVE_STATE] ✅ Состояние ботов сохранено в БД")
        return True
        
    except Exception as e:
        logger.error(f"[SAVE_STATE] ❌ Ошибка сохранения состояния: {e}")
        return False

def save_auto_bot_config(changed_data=None):
    """Сохраняет конфигурацию автобота в bot_config.py
    
    ✅ Теперь сохраняет напрямую в bot_engine/bot_config.py
    - Сохраняет ТОЛЬКО измененные значения (если передан changed_data)
    - Комментарии в файле сохраняются
    - Автоматически перезагружает модуль после сохранения (НЕ требуется перезапуск!)
    
    Args:
        changed_data: dict с только измененными значениями (опционально)
                      Если не передан, сохраняет весь config_data (для обратной совместимости)
    """
    try:
        from bots_modules.config_writer import save_auto_bot_config_to_py
        import importlib
        import sys
        
        # ✅ КРИТИЧЕСКИ ВАЖНО: Если передан changed_data, используем только его!
        # Иначе берем весь config_data (для обратной совместимости)
        if changed_data is not None:
            # Используем только измененные значения
            config_data = changed_data.copy()
            logger.info(f"[SAVE_CONFIG] 🔍 Сохраняем ТОЛЬКО измененные значения: {list(config_data.keys())}")
        else:
            # Обратная совместимость: берем весь config
            with bots_data_lock:
                config_data = bots_data['auto_bot_config'].copy()
            logger.info(f"[SAVE_CONFIG] 🔍 Сохраняем весь конфиг (changed_data не передан)")
        
        # ✅ КРИТИЧЕСКИ ВАЖНО: Логируем enabled перед сохранением
        logger.info(f"[SAVE_CONFIG] 🔍 enabled перед сохранением: {config_data.get('enabled')}")
        
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
            
            # ✅ КРИТИЧНО: Если сохранялся system_timeframe, сохраняем его в БД ПЕРЕД перезагрузкой модуля
            if 'system_timeframe' in config_data:
                try:
                    from bot_engine.bots_database import get_bots_database
                    from bot_engine.bot_config import set_current_timeframe
                    db = get_bots_database()
                    new_timeframe = config_data['system_timeframe']
                    db.save_timeframe(new_timeframe)
                    set_current_timeframe(new_timeframe)
                    logger.info(f"[SAVE_CONFIG] ✅ Таймфрейм сохранен в БД перед перезагрузкой модуля: {new_timeframe}")
                except Exception as tf_save_err:
                    logger.warning(f"[SAVE_CONFIG] ⚠️ Не удалось сохранить таймфрейм в БД: {tf_save_err}")
            
            # ✅ Перезагружаем модуль bot_config и обновляем конфигурацию из него
            try:
                if 'bot_engine.bot_config' in sys.modules:
                    pass
                    
                    # ✅ КРИТИЧНО: Сохраняем таймфрейм из БД перед перезагрузкой
                    saved_timeframe_from_db = None
                    try:
                        from bot_engine.bots_database import get_bots_database
                        db = get_bots_database()
                        saved_timeframe_from_db = db.load_timeframe()
                    except:
                        pass
                    
                    import bot_engine.bot_config
                    importlib.reload(bot_engine.bot_config)
                    pass
                    
                    # ✅ КРИТИЧНО: Восстанавливаем таймфрейм из БД после перезагрузки
                    if saved_timeframe_from_db:
                        try:
                            from bot_engine.bot_config import set_current_timeframe
                            set_current_timeframe(saved_timeframe_from_db)
                            logger.info(f"[SAVE_CONFIG] ✅ Таймфрейм восстановлен из БД после перезагрузки: {saved_timeframe_from_db}")
                        except Exception as tf_restore_err:
                            logger.warning(f"[SAVE_CONFIG] ⚠️ Не удалось восстановить таймфрейм: {tf_restore_err}")
                    
                    # ✅ КРИТИЧЕСКИ ВАЖНО: Перезагружаем конфигурацию из обновленного bot_config.py
                    # Это нужно, чтобы значения сразу брались из файла, а не из старой памяти
                    from bots_modules.imports_and_globals import load_auto_bot_config
                    
                    # ✅ СБРАСЫВАЕМ кэш времени модификации файла, чтобы при следующем вызове модуль перезагрузился
                    if hasattr(load_auto_bot_config, '_last_mtime'):
                        load_auto_bot_config._last_mtime = 0
                    
                    # ✅ НЕ сбрасываем флаг логирования leverage - иначе будет спам при каждой перезагрузке
                    # Флаг _leverage_logged остается, чтобы не логировать leverage при перезагрузке после сохранения
                    
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
    """Загружает состояние ботов из БД"""
    try:
        logger.info(f" 📂 Загрузка состояния ботов из БД...")
        
        # ✅ Загружаем из БД через storage.py
        state_data = storage_load_bots_state()
        
        if not state_data:
            logger.info(f" 📁 Состояние ботов в БД не найдено, начинаем с пустого состояния")
            return False
        
        version = state_data.get('version', '1.0')
        last_saved = state_data.get('last_saved', 'неизвестно')
        
        logger.info(f" 📊 Версия состояния: {version}, последнее сохранение: {last_saved}")
        
        # ✅ Конфигурация Auto Bot никогда не берётся из БД
        # Настройки загружаются только из bot_engine/bot_config.py
        
        logger.info(f" ⚙️ Конфигурация Auto Bot НЕ загружается из БД")
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
                        
                        bot_status = bot_data.get('status', 'UNKNOWN')
                        
                        # ВАЖНО: Раньше боты со статусом in_position_* пропускались на старте и ожидали
                        # восстановления через sync_bots_with_exchange(). На практике это приводило к тому,
                        # что "боты из ручных позиций" исчезали после перезапуска (позиция снова выглядела ручной),
                        # если синхронизация не успевала/не выполнялась сразу.
                        #
                        # Решение: загружаем таких ботов из БД, но помечаем их флагом needs_exchange_sync.
                        # Дальше sync_bots_with_exchange() всё равно приведёт состояние к реальным позициям на бирже
                        # (и удалит бота, если позиции уже нет).
                        if bot_status in ['in_position_long', 'in_position_short']:
                            bot_data['needs_exchange_sync'] = True
                            bots_data['bots'][symbol] = bot_data
                            restored_bots += 1
                            logger.info(f" 🤖 Восстановлен бот {symbol}: статус={bot_status} (ожидает sync)")
                            continue
                        
                        # ✅ КРИТИЧНО: НЕ загружаем ботов в статусе IDLE - они не имеют позиций!
                        # Боты в статусе IDLE должны удаляться при закрытии позиций, а не оставаться в БД.
                        if bot_status == 'idle':
                            pass  # бот без позиции должен быть удален
                            continue
                        
                        # ВАЖНО: НЕ проверяем зрелость при восстановлении!
                        # Причины:
                        # 1. Биржа еще не инициализирована (нет данных свечей)
                        # 2. Если бот был сохранен - он уже прошел проверку зрелости при создании
                        # 3. Проверка зрелости будет выполнена позже при обработке сигналов
                        
                        # Восстанавливаем бота
                        bots_data['bots'][symbol] = bot_data
                        restored_bots += 1
                        
                        logger.info(f" 🤖 Восстановлен бот {symbol}: статус={bot_status}")
                        
                    except Exception as e:
                        logger.error(f" ❌ Ошибка восстановления бота {symbol}: {e}")
                        failed_bots += 1
        
        logger.info(f" ✅ Восстановлено ботов: {restored_bots}, ошибок: {failed_bots}")
        
        return restored_bots > 0
        
    except Exception as e:
        logger.error(f" ❌ Ошибка загрузки состояния из БД: {e}")
        return False

def load_delisted_coins():
    """Загружает список делистинговых монет из БД"""
    try:
        # ✅ Загружаем из БД через storage.py
        delisted_list = storage_load_delisted_coins()
        
        # ✅ Загружаем last_scan из process_state
        last_scan = None
        try:
            from bots_modules.imports_and_globals import process_state
            if 'delisting_scan' in process_state:
                last_scan = process_state['delisting_scan'].get('last_scan')
        except Exception as state_error:
            pass
        
        # Преобразуем список в формат словаря для обратной совместимости
        if delisted_list:
            delisted_coins = {}
            for coin in delisted_list:
                if isinstance(coin, dict):
                    symbol = coin.get('symbol', '')
                    if symbol:
                        delisted_coins[symbol] = coin
                elif isinstance(coin, str):
                    delisted_coins[coin] = {}
            
            return {
                "delisted_coins": delisted_coins,
                "last_scan": last_scan,
                "scan_enabled": True
            }
        
        # Если данных нет, возвращаем дефолт
        return {"delisted_coins": {}, "last_scan": last_scan, "scan_enabled": True}
        
    except Exception as e:
        logger.warning(f"Ошибка загрузки делистированных монет из БД: {e}, используем дефолтные данные")
        return {"delisted_coins": {}, "last_scan": None, "scan_enabled": True}

def add_symbol_to_delisted(symbol: str, reason: str = "Delisting detected"):
    """Добавляет символ в список делистинговых (например, при ошибке 30228 при открытии позиции)."""
    try:
        if not symbol or not symbol.strip():
            return False
        sym = symbol.strip().upper()
        delisted_data = load_delisted_coins()
        if "delisted_coins" not in delisted_data:
            delisted_data["delisted_coins"] = {}
        if sym in delisted_data["delisted_coins"]:
            return True
        delisted_data["delisted_coins"][sym] = {
            "reason": reason,
            "delisting_date": datetime.now().strftime("%Y-%m-%d"),
            "detected_at": datetime.now().isoformat(),
            "source": "order_error_30228",
        }
        save_delisted_coins(delisted_data)
        logger.warning(f"🚨 Добавлен в список делистинга: {sym} — {reason}")
        return True
    except Exception as e:
        logger.error(f"Ошибка добавления {symbol} в список делистинга: {e}")
        return False


def save_delisted_coins(data):
    """Сохраняет список делистинговых монет в БД"""
    try:
        # Преобразуем словарь в список символов для БД
        # ✅ ИСПРАВЛЕНО: БД ожидает список строк (символов), а не список словарей
        delisted_coins_dict = data.get("delisted_coins", {}) if isinstance(data, dict) else {}
        delisted_list = []
        
        for symbol, coin_data in delisted_coins_dict.items():
            # Извлекаем только символ (строку) для сохранения в БД
            # Дополнительные данные (status, reason и т.д.) не сохраняются в таблицу delisted
            if isinstance(symbol, str):
                delisted_list.append(symbol)
            elif isinstance(coin_data, dict) and 'symbol' in coin_data:
                delisted_list.append(coin_data['symbol'])
            else:
                # Fallback: пытаемся извлечь символ из ключа
                delisted_list.append(str(symbol))
        
        # ✅ Сохраняем в БД через storage.py
        if storage_save_delisted_coins(delisted_list):
            logger.info(f"✅ Обновлены делистированные монеты в БД ({len(delisted_list)} монет)")
            return True
        return False
    except Exception as e:
        logger.error(f"Ошибка сохранения делистированных монет в БД: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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
        
        # Инициализируем структуру если её нет
        if 'delisted_coins' not in delisted_data:
            delisted_data['delisted_coins'] = {}
        
        new_delisted_count = 0
        
        # ✅ МЕТОД: Получаем ВСЕ инструменты сразу через API (без фильтра по статусу)
        # Это намного быстрее, чем проверять каждую монету отдельно!
        # ⚠️ ВАЖНО: Не указываем параметр status, чтобы получить ВСЕ инструменты, включая Closed/Delivering
        if hasattr(exchange_obj, 'client') and hasattr(exchange_obj.client, 'get_instruments_info'):
            try:
                logger.info("📊 Запрашиваем все инструменты с биржи (включая делистинговые)...")
                
                all_instruments = []
                cursor = None
                page = 0
                max_pages = 10  # Ограничение на количество страниц для безопасности
                
                # ✅ ОБРАБОТКА ПАГИНАЦИИ: Запрашиваем все страницы инструментов
                while page < max_pages:
                    page += 1
                    try:
                        # Запрашиваем ВСЕ инструменты без фильтра по статусу (не указываем status)
                        # Это соответствует API Bybit v5 - можно запросить все инструменты без symbol
                        params = {
                            'category': 'linear',
                            'limit': 1000  # Максимум инструментов за один запрос (Bybit API поддерживает до 1000)
                        }
                        
                        # Добавляем cursor для пагинации, если он есть
                        if cursor:
                            params['cursor'] = cursor
                        
                        response = exchange_obj.client.get_instruments_info(**params)
                        
                        if response and response.get('retCode') == 0:
                            result = response.get('result', {})
                            instruments_list = result.get('list', [])
                            
                            if not instruments_list:
                                pass
                                break
                            
                            all_instruments.extend(instruments_list)
                            logger.info(f"📊 Страница {page}: получено {len(instruments_list)} инструментов (всего: {len(all_instruments)})")
                            
                            # Проверяем, есть ли следующая страница
                            next_page_cursor = result.get('nextPageCursor')
                            if not next_page_cursor or next_page_cursor == '':
                                break
                            
                            cursor = next_page_cursor
                        else:
                            error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                            logger.warning(f"⚠️ Страница {page}: ошибка получения инструментов: {error_msg}")
                            break
                            
                    except Exception as page_error:
                        logger.error(f"❌ Ошибка при получении страницы {page}: {page_error}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        break
                
                logger.info(f"📊 Всего получено {len(all_instruments)} инструментов с биржи")
                
                # Фильтруем только USDT пары с статусом делистинга
                delisted_found = 0
                for instrument in all_instruments:
                    symbol = instrument.get('symbol', '')
                    if not symbol.endswith('USDT'):
                        continue
                    
                    coin_symbol = symbol.replace('USDT', '')
                    status = instrument.get('status', 'Unknown')
                    
                    # Пропускаем если уже в списке делистинговых
                    if coin_symbol in delisted_data['delisted_coins']:
                        continue
                    
                    # Проверяем статус делистинга (Closed или Delivering)
                    if status in ['Closed', 'Delivering']:
                        delisted_data['delisted_coins'][coin_symbol] = {
                            'status': status,
                            'reason': f"Delisting detected via API scan (status: {status})",
                            'delisting_date': datetime.now().strftime('%Y-%m-%d'),
                            'detected_at': datetime.now().isoformat(),
                            'source': 'api_bulk_scan'
                        }
                        
                        new_delisted_count += 1
                        delisted_found += 1
                        logger.warning(f"🚨 НОВЫЙ ДЕЛИСТИНГ: {coin_symbol} - {status}")
                
                if delisted_found == 0:
                    logger.info("✅ Делистинговых монет не обнаружено (или все уже в списке)")
                else:
                    logger.info(f"🚨 Обнаружено {delisted_found} новых делистинговых монет")
                    
            except Exception as bulk_scan_error:
                logger.error(f"❌ Ошибка массового сканирования делистинга: {bulk_scan_error}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.warning("⚠️ Массовое сканирование не удалось, делистинг будет обнаружен при попытке размещения ордеров")
        
        # Обновляем время последнего сканирования
        last_scan_time = datetime.now().isoformat()
        delisted_data['last_scan'] = last_scan_time
        
        # ✅ Сохраняем last_scan в process_state для персистентности
        try:
            update_process_state('delisting_scan', {
                'last_scan': last_scan_time,
                'total_delisted': len(delisted_data['delisted_coins']),
                'new_delisted': new_delisted_count
            })
        except Exception as state_error:
            pass
        
        # Сохраняем обновленные данные
        if save_delisted_coins(delisted_data):
            logger.info(f"✅ Сканирование завершено:")
            logger.info(f"   - Новых делистинговых: {new_delisted_count}")
            logger.info(f"   - Всего делистинговых: {len(delisted_data['delisted_coins'])}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка сканирования делистинга: {e}")

def check_delisting_emergency_close():
    """
    Проверяет делистинг и выполняет экстренное закрытие позиций (раз в 10 минут)
    ⚠️ ВАЖНО: scan_all_coins_for_delisting() вызывается не чаще раза в час,
    чтобы не перегружать API массовыми запросами
    """
    try:
        # Импорты для экстренного закрытия позиций
        from bots_modules.bot_class import NewTradingBot
        from bots_modules.imports_and_globals import get_exchange
        
        # ✅ СНАЧАЛА: Сканируем все монеты на делистинг (только если прошло достаточно времени)
        # Проверяем время последнего сканирования
        delisted_data = load_delisted_coins()
        last_scan_str = delisted_data.get('last_scan')
        
        should_scan = True
        if last_scan_str:
            try:
                from datetime import datetime
                last_scan_time = datetime.fromisoformat(last_scan_str)
                time_since_scan = (datetime.now() - last_scan_time).total_seconds()
                # Сканируем не чаще чем раз в час (3600 секунд), чтобы не перегружать API
                if time_since_scan < 3600:
                    should_scan = False
                    pass
            except Exception as time_check_error:
                pass
        
        if should_scan:
            scan_all_coins_for_delisting()
        else:
            pass
        
        logger.info(f"🔍 Проверка делистинга для активных ботов...")
        
        with bots_data_lock:
            bots_in_position = [
                (symbol, bot_data) for symbol, bot_data in bots_data['bots'].items()
                if bot_data.get('status') in ['in_position_long', 'in_position_short']
            ]
        
        if not bots_in_position:
            pass
            return True
        
        logger.info(f"📊 Проверяем {len(bots_in_position)} активных ботов")
        
        delisting_closed_count = 0
        exchange_obj = get_exchange()
        
        if not exchange_obj:
            logger.error(f"❌ Exchange не инициализирован")
            return False
        
        for symbol, bot_data in bots_in_position:
            try:
                # ✅ ПРОВЕРКА 1: Проверяем делистинг напрямую из delisted.json (самый быстрый способ)
                is_delisting = False
                delisting_reason = ""
                
                delisted_data = load_delisted_coins()
                delisted_coins = delisted_data.get('delisted_coins', {})
                if symbol in delisted_coins:
                    is_delisting = True
                    delisting_info = delisted_coins[symbol]
                    delisting_reason = delisting_info.get('reason', 'Delisting detected')
                    logger.warning(f"🚨 ДЕЛИСТИНГ ОБНАРУЖЕН для {symbol} в delisted.json: {delisting_reason}")
                
                # ✅ ПРОВЕРКА 2: Проверяем делистинг через RSI данные (fallback)
                if not is_delisting:
                    rsi_cache = get_rsi_cache()
                    if symbol in rsi_cache:
                        rsi_data = rsi_cache[symbol]
                        is_delisting = rsi_data.get('is_delisting', False) or rsi_data.get('trading_status') in ['Closed', 'Delivering']
                        if is_delisting:
                            delisting_reason = f"Delisting detected via RSI data (status: {rsi_data.get('trading_status', 'Unknown')})"
                            logger.warning(f"🚨 ДЕЛИСТИНГ ОБНАРУЖЕН для {symbol} через RSI данные")
                
                # Если делистинг обнаружен - закрываем позицию немедленно
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
            exchange_obj = get_exchange()
            if exchange_obj:
                exchange_positions = exchange_obj.get_positions()
                if isinstance(exchange_positions, tuple):
                    positions_list = exchange_positions[0] if exchange_positions else []
                else:
                    positions_list = exchange_positions if exchange_positions else []
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
                            volume_value_raw = bot_data.get('volume_value', 0)
                            try:
                                volume_value = float(volume_value_raw) if volume_value_raw is not None else 0.0
                            except (TypeError, ValueError):
                                volume_value = 0.0
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
                                pass
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
                                pass
                        
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
            sync_bots_with_exchange()
        except Exception as e:
            logger.error(f" ❌ Ошибка синхронизации с биржей: {e}")
        
        # ✅ КРИТИЧНО: Обновляем last_update в bots_data для UI
        # ⚡ БЕЗ БЛОКИРОВКИ: GIL делает запись атомарной
        bots_data['last_update'] = current_time
        
        # Отладочный лог для проверки частоты обновлений
        return True
        
    except Exception as e:
        logger.error(f" ❌ Ошибка обновления кэша: {e}")
        return False

def update_bot_positions_status():
    """Обновляет статус позиций ботов (цена, PnL, ликвидация) каждые SystemConfig.BOT_STATUS_UPDATE_INTERVAL секунд.
    ⚡ Блокировку держим только для копирования списка и записи результатов — запросы к бирже вне блокировки."""
    try:
        if not ensure_exchange_initialized():
            return False
        
        # 1) Под блокировкой только копируем список ботов в позиции (минимальное время)
        with bots_data_lock:
            to_update = [
                (symbol, bot_data.get('entry_price'), bot_data.get('position_side'), bot_data.get('volume_value', 10), bot_data.get('unrealized_pnl', 0))
                for symbol, bot_data in bots_data['bots'].items()
                if bot_data.get('status') in ['in_position_long', 'in_position_short']
                and bot_data.get('status') != BOT_STATUS.get('PAUSED')
                and bot_data.get('entry_price') and bot_data.get('position_side')
            ]
        
        if not to_update:
            return True
        
        current_exchange = get_exchange()
        if not current_exchange:
            return False
        
        # 2) Один запрос тикеров для всех символов (get_tickers_batch) — БЕЗ блокировки
        symbols_list = [t[0] for t in to_update]
        tickers_map = current_exchange.get_tickers_batch(symbols_list)
        
        results = []
        for (symbol, entry_price, position_side, volume_value, old_pnl) in to_update:
            try:
                ticker_data = tickers_map.get(symbol)
                if not ticker_data:
                    continue
                current_price = float(ticker_data.get('last_price') or ticker_data.get('last') or 0)
                if not current_price:
                    continue
                leverage = 10
                if position_side == 'LONG':
                    pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    liquidation_price = entry_price * (1 - (100 / leverage) / 100)
                    distance_to_liq = ((current_price - liquidation_price) / liquidation_price) * 100
                else:
                    pnl_percent = ((entry_price - current_price) / entry_price) * 100
                    liquidation_price = entry_price * (1 + (100 / leverage) / 100)
                    distance_to_liq = ((liquidation_price - current_price) / liquidation_price) * 100
                results.append((symbol, current_price, pnl_percent, liquidation_price, distance_to_liq, old_pnl))
            except Exception as e:
                logger.error(f"[POSITION_UPDATE] ❌ Ошибка обновления {symbol}: {e}")
        
        # 3) Под блокировкой только записываем результаты (короткий участок)
        if results:
            with bots_data_lock:
                now_iso = datetime.now().isoformat()
                for symbol, current_price, pnl_percent, liquidation_price, distance_to_liq, old_pnl in results:
                    bot_data = bots_data['bots'].get(symbol)
                    if not bot_data:
                        continue
                    bot_data['unrealized_pnl'] = pnl_percent
                    bot_data['current_price'] = current_price
                    bot_data['last_update'] = now_iso
                    bot_data['liquidation_price'] = liquidation_price
                    bot_data['distance_to_liquidation'] = distance_to_liq
                    if abs(pnl_percent - old_pnl) > 0.1:
                        logger.info(f"[POSITION_UPDATE] 📊 {symbol} {bot_data.get('position_side')}: ${current_price:.6f} | PnL: {pnl_percent:+.2f}% | Ликвидация: ${liquidation_price:.6f} ({distance_to_liq:.1f}%)")
        
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

            # ✅ ИСПРАВЛЕНИЕ: Используем exchange.get_positions() для получения ВСЕХ позиций с пагинацией
            # Это гарантирует, что мы получим все позиции, а не только первую страницу
            try:
                positions_result = current_exchange.get_positions()
                if isinstance(positions_result, tuple):
                    processed_positions_list, rapid_growth = positions_result
                else:
                    processed_positions_list = positions_result if positions_result else []
                
                # Конвертируем обработанные позиции в формат, ожидаемый функцией
                raw_positions = []
                for pos in processed_positions_list:
                    # Создаем формат сырых данных из обработанных
                    raw_pos = {
                        'symbol': pos.get('symbol', '') + 'USDT',
                        'size': pos.get('size', 0),
                        'side': 'Buy' if pos.get('side', '').upper() in ['LONG'] or pos.get('side', '') == 'Long' else 'Sell',
                        'avgPrice': pos.get('avg_price', 0) or pos.get('entry_price', 0),
                        'unrealisedPnl': pos.get('pnl', 0),
                        'markPrice': pos.get('mark_price', 0) or pos.get('current_price', 0)
                    }
                    raw_positions.append(raw_pos)
                
            except Exception as get_pos_error:
                # Fallback: используем прямой вызов API с пагинацией
                logger.warning(f"[EXCHANGE_POSITIONS] ⚠️ Ошибка получения через get_positions(), используем прямой API: {get_pos_error}")
                
                raw_positions = []
                cursor = None
                while True:
                    params = {
                        "category": "linear",
                        "settleCoin": "USDT",
                        "limit": 100
                    }
                    if cursor:
                        params["cursor"] = cursor
                    
                    response = current_exchange.client.get_positions(**params)
                    
                    if response.get('retCode') != 0:
                        error_msg = response.get('retMsg', 'Unknown error')
                        logger.warning(f"[EXCHANGE_POSITIONS] ⚠️ Ошибка API (попытка {attempt + 1}/{max_retries}): {error_msg}")
                        
                        # Если это Rate Limit, увеличиваем задержку
                        if "rate limit" in error_msg.lower() or "too many" in error_msg.lower():
                            retry_delay = min(retry_delay * 2, 10)
                        
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            break  # Выходим из цикла пагинации для retry
                        else:
                            logger.error(f"[EXCHANGE_POSITIONS] ❌ Не удалось получить позиции после {max_retries} попыток")
                            return None
                    
                    page_positions = response.get('result', {}).get('list', [])
                    raw_positions.extend(page_positions)
                    
                    cursor = response.get('result', {}).get('nextPageCursor')
                    if not cursor:
                        break
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
                # ✅ Нормализуем стороны для сравнения (LONG/Long -> LONG, SHORT/Short -> SHORT)
                bot_side_normalized = bot_pos['position_side'].upper() if bot_pos['position_side'] else None
                exchange_side_normalized = exchange_pos['position_side'].upper() if exchange_pos['position_side'] else None
                
                if bot_side_normalized != exchange_side_normalized:
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
                        # ✅ КРИТИЧНО: УДАЛЯЕМ бота, а не переводим в IDLE - позиции нет на бирже!
                        with bots_data_lock:
                            if symbol in bots_data['bots']:
                                del bots_data['bots'][symbol]
                                synced_count += 1
                                logger.info(f"[POSITION_SYNC] 🗑️ Удален бот {symbol} - позиция закрыта на бирже")
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
                
                # ✅ Нормализуем стороны для сравнения (LONG/Long -> LONG, SHORT/Short -> SHORT)
                exchange_side_normalized = exchange_side.upper() if exchange_side else None
                bot_side_normalized = bot_side.upper() if bot_side else None
                
                if exchange_side_normalized != bot_side_normalized:
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
                
                # ✅ КРИТИЧНО: УДАЛЯЕМ ботов в статусе 'idle' или 'running' БЕЗ позиции на бирже!
                # Если бот прошел все фильтры и должен был зайти в сделку, но не зашел - это ошибка системы
                # Такие боты не должны существовать и должны удаляться немедленно
                if bot_status in ['idle', 'running']:
                    # Если нет позиции на бирже - удаляем бота (ошибка системы)
                    if normalized_bot_symbol not in exchange_symbols:
                        # Проверяем время создания - не удаляем только что созданных ботов (в течение 5 минут)
                        created_time_str = bot_data.get('created_time') or bot_data.get('created_at')
                        if created_time_str:
                            try:
                                created_time = datetime.fromisoformat(created_time_str.replace('Z', '+00:00'))
                                time_since_creation = current_time - created_time.timestamp()
                                if time_since_creation < 300:  # 5 минут
                                    pass
                                    continue
                            except Exception:
                                pass  # Если ошибка парсинга - удаляем бота
                        
                        logger.warning(f" 🗑️ {symbol}: Удаляем бота в статусе {bot_status} без позиции на бирже (ошибка системы)")
                        bots_to_remove.append(symbol)
                        continue
                    else:
                        # Есть позиция на бирже - пропускаем (бот работает корректно)
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
                    # ✅ ИСПРАВЛЕНО: Обрабатываем некорректные значения типа 'Никогда'
                    if isinstance(last_update_str, str) and last_update_str.lower() in ['никогда', 'never', '']:
                        pass
                        # Пропускаем проверку last_update, проверяем created_at ниже
                        last_update_str = None
                    else:
                        try:
                            last_update = datetime.fromisoformat(last_update_str.replace('Z', '+00:00'))
                            time_since_update = current_time - last_update.timestamp()
                            
                            if time_since_update >= SystemConfig.INACTIVE_BOT_TIMEOUT:
                                logger.warning(f" ⏰ Бот {symbol} неактивен {time_since_update//60:.0f} мин (статус: {bot_status})")
                                bots_to_remove.append(symbol)
                                
                                # Логируем удаление неактивного бота в историю
                                # log_bot_stop(symbol, f"Неактивен {time_since_update//60:.0f} мин (статус: {bot_status})")  # TODO: Функция не определена
                            else:
                                pass
                                continue  # Бот активен - пропускаем удаление
                        except Exception as e:
                            logger.error(f" ❌ Ошибка парсинга времени для {symbol}: {e}, значение='{last_update_str}'")
                            # Если не можем распарсить время - проверяем created_at ниже
                            last_update_str = None
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
    """Проверяет и устанавливает недостающие стоп-лоссы и трейлинг стопы для ботов."""
    try:
        if not ensure_exchange_initialized():
            logger.error(" ❌ Биржа не инициализирована")
            return False

        current_exchange = get_exchange() or exchange
        if not current_exchange:
            logger.error(" ❌ Не удалось получить объект биржи")
            return False

        auto_config, bots_snapshot = _snapshot_bots_for_protections()
        if not bots_snapshot:
            pass
            return True

        # ✅ ИСПРАВЛЕНИЕ: Правильная нормализация символов и фильтрация только активных позиций
        def normalize_symbol(symbol):
            """Нормализует символ, убирая USDT суффикс если есть"""
            if symbol and symbol.endswith('USDT'):
                return symbol[:-4]  # Убираем 'USDT'
            return symbol
        
        # Инициализируем переменные для дополнительной проверки
        _raw_positions_for_check = []
        exchange_positions = {}
        
        try:
            # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем exchange.get_positions() вместо client.get_positions()
            # exchange.get_positions() обрабатывает пагинацию и возвращает ВСЕ позиции (как в app.py)
            positions_result = current_exchange.get_positions()
            if isinstance(positions_result, tuple):
                processed_positions, rapid_growth = positions_result
            else:
                processed_positions = positions_result if positions_result else []
            
            # ✅ Получаем СЫРЫЕ позиции напрямую через API для детальной проверки
            # Но используем обработанные позиции из exchange.get_positions() как основной источник
            try:
                positions_response = current_exchange.client.get_positions(
                    category="linear",
                    settleCoin="USDT",
                    limit=100
                )
                if positions_response.get('retCode') == 0:
                    # Получаем все страницы для сырых данных
                    raw_positions = []
                    cursor = None
                    while True:
                        params = {
                            "category": "linear",
                            "settleCoin": "USDT",
                            "limit": 100
                        }
                        if cursor:
                            params["cursor"] = cursor
                        
                        response = current_exchange.client.get_positions(**params)
                        if response.get('retCode') != 0:
                            break
                        
                        page_positions = response.get('result', {}).get('list', [])
                        raw_positions.extend(page_positions)
                        
                        cursor = response.get('result', {}).get('nextPageCursor')
                        if not cursor:
                            break
                else:
                    # Если не удалось получить сырые данные, используем обработанные
                    logger.warning(f" ⚠️ Не удалось получить сырые позиции, используем обработанные")
                    raw_positions = []
                    for pos in processed_positions:
                        # Создаем сырой формат из обработанного
                        raw_pos = {
                            'symbol': pos.get('symbol', '') + 'USDT' if not pos.get('symbol', '').endswith('USDT') else pos.get('symbol', ''),
                            'size': pos.get('size', 0),
                            'avgPrice': pos.get('avg_price', 0),
                            'markPrice': pos.get('mark_price', 0),
                            'unrealisedPnl': pos.get('pnl', 0),
                            'side': 'Buy' if pos.get('side') == 'LONG' else 'Sell'
                        }
                        raw_positions.append(raw_pos)
            except Exception as raw_error:
                logger.warning(f" ⚠️ Ошибка получения сырых позиций: {raw_error}, используем обработанные")
                raw_positions = []
            
            _raw_positions_for_check = raw_positions  # Сохраняем для дополнительной проверки
            
            # ✅ ИСПРАВЛЕНИЕ: Используем обработанные позиции из exchange.get_positions()
            # Они уже нормализованы и содержат все позиции (с пагинацией)
            exchange_positions = {}
            all_positions_dict = {}
            
            # Сначала заполняем из обработанных позиций (основной источник)
            # В processed_positions символы уже нормализованы (без USDT) через clean_symbol()
            for position in processed_positions:
                symbol = position.get('symbol', '')
                position_size = abs(float(position.get('size', 0) or 0))
                
                if symbol:
                    # Создаем формат для совместимости с сырыми данными
                    # В processed_positions side может быть 'Long'/'Short', конвертируем в 'Buy'/'Sell'
                    side_str = position.get('side', '')
                    if side_str.upper() == 'LONG':
                        side_api = 'Buy'
                    elif side_str.upper() == 'SHORT':
                        side_api = 'Sell'
                    else:
                        side_api = 'Buy'  # По умолчанию
                    
                    raw_format_position = {
                        'symbol': symbol + 'USDT',  # Добавляем USDT для совместимости
                        'size': position.get('size', 0),
                        'avgPrice': position.get('avg_price', 0) or position.get('entry_price', 0),
                        'markPrice': position.get('mark_price', 0) or position.get('current_price', 0),
                        'unrealisedPnl': position.get('pnl', 0),
                        'side': side_api,
                        'positionIdx': position.get('position_idx', 0),
                        'stopLoss': position.get('stop_loss', ''),
                        'takeProfit': position.get('take_profit', ''),
                        'trailingStop': position.get('trailing_stop', '')
                    }
                    
                    all_positions_dict[symbol] = raw_format_position
                    
                    # ✅ ТОЛЬКО активные позиции (size > 0)
                    if position_size > 0:
                        exchange_positions[symbol] = raw_format_position
            
            # Дополнительно добавляем из сырых позиций (на случай, если что-то пропущено)
            for position in raw_positions:
                raw_symbol = position.get('symbol', '')
                position_size = abs(float(position.get('size', 0) or 0))
                normalized_symbol = normalize_symbol(raw_symbol)
                
                if normalized_symbol and normalized_symbol not in exchange_positions:
                    # Добавляем только если еще нет в словаре
                    if position_size > 0:
                        exchange_positions[normalized_symbol] = position
                    all_positions_dict[normalized_symbol] = position
            
        except Exception as e:
            logger.error(f" ❌ Ошибка получения позиций с биржи: {e}")
            return False

        from bots_modules.bot_class import NewTradingBot

        updated_count = 0
        failed_count = 0

        for symbol, bot_snapshot in bots_snapshot.items():
            try:
                pos = exchange_positions.get(symbol)
                if not pos:
                    # ✅ КРИТИЧЕСКАЯ ПРОВЕРКА: Перед удалением проверяем позицию напрямую через API
                    logger.warning(f" ⚠️ Позиция {symbol} не найдена в словаре позиций. Проверяем напрямую через API...")
                    
                    # Дополнительная проверка - запрашиваем позицию напрямую
                    try:
                        direct_check = False
                        matching_raw_symbol = None
                        for raw_pos in _raw_positions_for_check:
                            raw_symbol = raw_pos.get('symbol', '')
                            position_size = abs(float(raw_pos.get('size', 0) or 0))
                            normalized = normalize_symbol(raw_symbol)
                            
                            if normalized == symbol and position_size > 0:
                                direct_check = True
                                matching_raw_symbol = raw_symbol
                                logger.info(f" ✅ Позиция {symbol} найдена при прямой проверке! raw='{raw_symbol}', normalized='{normalized}', размер: {position_size}")
                                # Обновляем словарь позиций
                                exchange_positions[symbol] = raw_pos
                                pos = raw_pos
                                break
                        
                        if not direct_check:
                            # Пробуем найти с учетом возможных вариантов символа
                            logger.error(f" ❌ Позиция {symbol} не найдена на бирже после прямой проверки")
                            
                            # Пробуем найти варианты: symbol, symbolUSDT, USDTsymbol
                            possible_symbols = [symbol, f"{symbol}USDT", f"USDT{symbol}"]
                            found_variants = []
                            for raw_pos in _raw_positions_for_check:
                                raw_symbol = raw_pos.get('symbol', '')
                                position_size = abs(float(raw_pos.get('size', 0) or 0))
                                if position_size > 0 and raw_symbol in possible_symbols:
                                    found_variants.append(f"raw='{raw_symbol}' (size={position_size})")
                            
                            if found_variants:
                                logger.warning(f" ⚠️ Найдены варианты символа {symbol} на бирже: {found_variants}")
                                logger.warning(f" ⚠️ Возможно, проблема в нормализации символов!")
                            
                            logger.error(f" ❌ Доступные позиции на бирже (normalized): {sorted([normalize_symbol(p.get('symbol', '')) for p in _raw_positions_for_check if abs(float(p.get('size', 0) or 0)) > 0])}")
                            logger.error(f" ❌ Доступные позиции на бирже (raw): {sorted([p.get('symbol', '') for p in _raw_positions_for_check if abs(float(p.get('size', 0) or 0)) > 0])}")
                            # НЕ УДАЛЯЕМ бота, если не уверены - просто пропускаем
                            logger.warning(f" ⚠️ Пропускаем бота {symbol} - позиция не найдена, но не удаляем для безопасности")
                            continue
                    except Exception as check_error:
                        logger.error(f" ❌ Ошибка прямой проверки позиции {symbol}: {check_error}")
                        # НЕ УДАЛЯЕМ бота при ошибке проверки
                        continue
                    
                    if not pos:
                        # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: НЕ УДАЛЯЕМ бота, если позиция не найдена!
                        # Позиция может быть на бирже, но не найдена из-за проблем с нормализацией символов
                        logger.warning(f" ⚠️ Позиция {symbol} не найдена на бирже - ПРОПУСКАЕМ (не удаляем для безопасности)")
                        continue

                # ✅ ИСПРАВЛЕНИЕ: Правильная проверка размера позиции (используем abs для учета LONG/SHORT)
                position_size = abs(_safe_float(pos.get('size'), 0.0) or 0.0)
                if position_size <= 0:
                    logger.warning(f" ⚠️ Позиция {symbol} закрыта на бирже - удаляем бота и позицию из реестра")
                    # ✅ УДАЛЯЕМ БОТА И ПОЗИЦИЮ ИЗ РЕЕСТРА, если позиция закрыта на бирже
                    try:
                        from bots_modules.imports_and_globals import unregister_bot_position
                        # Получаем order_id из бота
                        order_id = None
                        position = bot_snapshot.get('position')
                        if position and position.get('order_id'):
                            order_id = position['order_id']
                        elif bot_snapshot.get('restoration_order_id'):
                            order_id = bot_snapshot.get('restoration_order_id')
                        
                        # Удаляем позицию из реестра
                        if order_id:
                            unregister_bot_position(order_id)
                            logger.info(f" ✅ Позиция {symbol} (order_id={order_id}) удалена из реестра")
                        
                        # Удаляем бота из системы
                        bot_removed = False
                        with bots_data_lock:
                            if symbol in bots_data['bots']:
                                del bots_data['bots'][symbol]
                                logger.info(f" ✅ Бот {symbol} удален из системы")
                                bot_removed = True
                        # Сохраняем состояние после освобождения блокировки
                        if bot_removed:
                            save_bots_state()
                    except Exception as cleanup_error:
                        logger.error(f" ❌ Ошибка удаления бота {symbol}: {cleanup_error}")
                    continue

                entry_price = _safe_float(pos.get('avgPrice'), 0.0)
                current_price = _safe_float(pos.get('markPrice'), entry_price)
                unrealized_pnl = _safe_float(pos.get('unrealisedPnl'), 0.0) or 0.0
                side = pos.get('side', '')
                position_idx = pos.get('positionIdx', 0)
                existing_stop_loss = pos.get('stopLoss', '')
                existing_trailing_stop = pos.get('trailingStop', '')
                existing_take_profit = pos.get('takeProfit', '')

                position_side = 'LONG' if side == 'Buy' else 'SHORT'
                profit_percent = 0.0
                if entry_price:
                    if position_side == 'LONG':
                        profit_percent = ((current_price - entry_price) / entry_price) * 100
                    else:
                        profit_percent = ((entry_price - current_price) / entry_price) * 100

                logger.info(
                    f" 📊 {symbol}: PnL {profit_percent:.2f}%, текущая {current_price}, вход {entry_price}"
                )

                runtime_config = copy.deepcopy(bot_snapshot)
                runtime_config.setdefault('volume_value', runtime_config.get('position_size'))
                if entry_price and position_size:
                    runtime_config['position_size'] = entry_price * position_size
                    runtime_config['position_size_coins'] = position_size
                runtime_config['entry_price'] = runtime_config.get('entry_price') or entry_price
                runtime_config['position_side'] = runtime_config.get('position_side') or position_side

                entry_timestamp = (
                    _normalize_timestamp(bot_snapshot.get('entry_timestamp'))
                    or _normalize_timestamp(bot_snapshot.get('position_start_time'))
                    or _normalize_timestamp(pos.get('createdTime') or pos.get('updatedTime'))
                )
                if entry_timestamp:
                    runtime_config['entry_timestamp'] = entry_timestamp
                    runtime_config['position_start_time'] = _timestamp_to_iso(entry_timestamp)

                bot_instance = NewTradingBot(symbol, config=runtime_config, exchange=current_exchange)
                bot_instance.entry_price = entry_price
                bot_instance.position_side = position_side
                bot_instance.position_size_coins = position_size
                bot_instance.position_size = entry_price * position_size if entry_price else runtime_config.get('position_size')
                bot_instance.realized_pnl = _safe_float(
                    pos.get('cumRealisedPnl') or pos.get('realisedPnl') or pos.get('realizedPnl'), 0.0
                ) or 0.0
                bot_instance.unrealized_pnl = unrealized_pnl
                if entry_timestamp:
                    bot_instance.entry_timestamp = entry_timestamp
                    bot_instance.position_start_time = datetime.fromtimestamp(entry_timestamp)

                decision = bot_instance._evaluate_protection_decision(current_price)
                # ✅ ИСПРАВЛЕНО: Обновляем защитные механизмы (включая break-even стоп)
                # Это нужно для установки break-even стопа на бирже при изменении конфига
                bot_instance._update_protection_mechanisms(current_price)
                protection_config = bot_instance._get_effective_protection_config()

                updates = {
                    'entry_price': entry_price,
                    'position_side': position_side,
                    'position_size_coins': position_size,
                    'position_size': bot_instance.position_size,
                    'realized_pnl': bot_instance.realized_pnl,
                    'unrealized_pnl': unrealized_pnl,
                    'current_price': current_price,
                    'leverage': _safe_float(pos.get('leverage'), bot_snapshot.get('leverage', 1.0)) or 1.0,
                    'last_update': datetime.now().isoformat(),
                }
                if entry_timestamp:
                    updates['entry_timestamp'] = entry_timestamp
                    updates['position_start_time'] = _timestamp_to_iso(entry_timestamp)

                if existing_stop_loss:
                    updates['stop_loss_price'] = _safe_float(existing_stop_loss)
                if existing_take_profit:
                    updates['take_profit_price'] = _safe_float(existing_take_profit)
                if existing_trailing_stop:
                    updates['trailing_stop_price'] = _safe_float(existing_trailing_stop)

                _apply_protection_state_to_bot_data(updates, decision.state)

                if decision.should_close:
                    logger.warning(
                        f" ⚠️ Protection Engine сигнализирует закрытие {symbol}: {decision.reason}"
                    )

                desired_stop = _select_stop_loss_price(
                    position_side,
                    entry_price,
                    current_price,
                    protection_config,
                    bot_instance.break_even_stop_price,
                    bot_instance.trailing_stop_price,
                )
                existing_stop_value = _safe_float(existing_stop_loss)

                # ✅ ИСПРАВЛЕНО: Обновляем стоп-лосс, даже если он уже установлен, если нужен новый стоп
                # Проверяем, нужно ли обновить стоп-лосс на бирже
                if desired_stop and _needs_price_update(position_side, desired_stop, existing_stop_value):
                    try:
                        sl_response = current_exchange.update_stop_loss(
                            symbol=symbol,
                            stop_loss_price=desired_stop,
                            position_side=position_side,
                        )
                        if sl_response and sl_response.get('success'):
                            updates['stop_loss_price'] = desired_stop
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
                    protection_config,
                    bot_instance.trailing_take_profit_price,
                )
                existing_take_value = _safe_float(existing_take_profit)

                # Проверяем, есть ли уже тейк-профит на бирже
                if existing_take_profit and existing_take_profit.strip():
                    pass  # Тейк-профит уже установлен, пропускаем
                elif desired_take and _needs_price_update(position_side, desired_take, existing_take_value):
                    try:
                        tp_response = current_exchange.update_take_profit(
                            symbol=symbol,
                            take_profit_price=desired_take,
                            position_side=position_side,
                        )
                        if tp_response and tp_response.get('success'):
                            updates['take_profit_price'] = desired_take
                            updated_count += 1
                            logger.info(f" ✅ Тейк-профит синхронизирован для {symbol}: {desired_take:.6f}")
                        else:
                            failed_count += 1
                            logger.error(f" ❌ Ошибка установки тейк-профита для {symbol}: {tp_response}")
                    except Exception as e:
                        failed_count += 1
                        logger.error(f" ❌ Ошибка установки тейк-профита для {symbol}: {e}")

                if not _update_bot_record(symbol, updates):
                    pass

            except Exception as e:
                logger.error(f" ❌ Ошибка обработки {symbol}: {e}")
                failed_count += 1
                continue

        if updated_count > 0 or failed_count > 0:
            logger.info(f" ✅ Установка завершена: установлено {updated_count}, ошибок {failed_count}")
            if updated_count > 0:
                try:
                    save_bots_state()
                except Exception as save_error:
                    logger.error(f" ❌ Ошибка сохранения состояния ботов: {save_error}")

        # ✅ Синхронизируем ботов с биржей - удаляем ботов без позиций
        try:
            sync_bots_with_exchange()
        except Exception as sync_error:
            logger.error(f" ❌ Ошибка синхронизации ботов с биржей: {sync_error}")

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
            for bot_key, bot_data in bots_data['bots'].items():
                try:
                    bot_status = bot_data.get('status')
                    # Чистый символ для API (бот может быть ключом symbol или symbol_side, например BTCUSDT_LONG)
                    api_symbol = bot_data.get('symbol') or (bot_key.rsplit('_', 1)[0] if ('_LONG' in bot_key or '_SHORT' in bot_key) else bot_key)
                    symbol = api_symbol  # для логов и target_symbol ниже

                    # Проверяем только активные боты (не idle/paused)
                    if bot_status in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']]:
                        continue
                    # Символ для Bybit: если уже с USDT — как есть, иначе добавить USDT
                    symbol_for_api = api_symbol if (api_symbol and 'USDT' in api_symbol) else f"{api_symbol}USDT"

                    # Проверяем позицию на бирже
                    from bots_modules.imports_and_globals import get_exchange
                    current_exchange = get_exchange() or exchange
                    positions_response = current_exchange.client.get_positions(
                        category="linear",
                        symbol=symbol_for_api
                    )
                    
                    if positions_response.get('retCode') == 0:
                        positions = positions_response['result']['list']
                        has_position = False
                        
                        # Фильтруем позиции только для нужного символа
                        target_symbol = symbol_for_api
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
                                pass
                        else:
                            # Нет позиции на бирже
                            if bot_status in [BOT_STATUS['IN_POSITION_LONG'], BOT_STATUS['IN_POSITION_SHORT']]:
                                # КОНФЛИКТ: бот думает что в позиции, но позиции нет на бирже
                                logger.warning(f" 🚨 {symbol}: КОНФЛИКТ! Бот показывает позицию, но на бирже её нет!")
                                
                                # ✅ КРИТИЧНО: УДАЛЯЕМ бота, а не переводим в IDLE - позиции нет на бирже!
                                with bots_data_lock:
                                    if bot_key in bots_data['bots']:
                                        del bots_data['bots'][bot_key]
                                
                                conflicts_found += 1
                                
                                logger.warning(f" 🗑️ {symbol}: Бот удален - позиции нет на бирже")
                    else:
                        logger.warning(f" ❌ {symbol}: Ошибка получения позиций: {positions_response.get('retMsg', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f" ❌ Ошибка проверки {symbol}: {e}")
        
        if conflicts_found > 0:
            logger.warning(f" 🚨 Найдено {conflicts_found} конфликтов, остановлено {bots_paused} ботов")
            # Сохраняем обновленное состояние
            save_bots_state()

            # ✅ ДОПОЛНИТЕЛЬНО: синхронизируем реестр позиций ботов (bot_positions_registry)
            # Ключ реестра = SYMBOL_SIDE (например BTCUSDT_LONG), чтобы по одному символу могли быть лонг и шорт.
            try:
                from bots_modules.imports_and_globals import save_bot_positions_registry

                registry = {}
                positions_list = get_exchange_positions() or []
                # Словарь позиций с биржи: ключ (symbol, side) для поддержки лонг+шорт по одному символу
                exchange_by_symbol_side = {}
                for pos in (positions_list if isinstance(positions_list, list) else []):
                    sym = (pos.get('symbol') or '').upper()
                    if sym and 'USDT' not in sym:
                        sym = sym + 'USDT'
                    side_raw = pos.get('side', '') or pos.get('position_side', '')
                    side = 'LONG' if side_raw in ['Buy', 'LONG', 'Long'] else 'SHORT'
                    if sym:
                        exchange_by_symbol_side[(sym, side)] = pos

                for bot_key, bot_data in list(bots_data.get('bots', {}).items()):
                    try:
                        status = bot_data.get('status')
                        if status not in [BOT_STATUS.get('IN_POSITION_LONG'), BOT_STATUS.get('IN_POSITION_SHORT'), 'in_position_long', 'in_position_short']:
                            continue
                        sym_clean = bot_data.get('symbol') or (bot_key.rsplit('_', 1)[0] if ('_LONG' in bot_key or '_SHORT' in bot_key) else bot_key)
                        sym_clean = str(sym_clean).upper()
                        if sym_clean and 'USDT' not in sym_clean:
                            sym_clean = sym_clean + 'USDT'
                        side = (bot_data.get('position') or {}).get('side') or ('LONG' if 'long' in str(bot_data.get('status', '')) else 'SHORT')
                        side = str(side).upper() if side in ('LONG', 'SHORT') else 'LONG'
                        pos = exchange_by_symbol_side.get((sym_clean, side))
                        if not pos:
                            continue
                        entry_price = float(pos.get('avgPrice') or pos.get('entry_price') or bot_data.get('entry_price') or 0) or 0.0
                        quantity = float(pos.get('size') or bot_data.get('position_size_coins') or 0) or 0.0
                        if entry_price <= 0 or quantity <= 0:
                            continue
                        registry_key = f"{sym_clean}_{side}"
                        registry[registry_key] = {
                            'symbol': sym_clean,
                            'side': side,
                            'entry_price': entry_price,
                            'quantity': quantity,
                            'opened_at': bot_data.get('position_start_time') or bot_data.get('entry_time') or datetime.now().isoformat(),
                            'managed_by_bot': True,
                        }
                    except Exception as _e:
                        pass

                save_bot_positions_registry(registry)
            except Exception as reg_err:
                logger.warning(f"[SYNC_EXCHANGE] ⚠️ Не удалось синхронизировать реестр позиций ботов: {reg_err}")
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
                        pass
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
            
            # ✅ УПРОЩЕНО: Получаем список ботов один раз и проверяем напрямую
            with bots_data_lock:
                bot_items = list(bots_data['bots'].items())  # Копия для безопасной итерации
            
            synchronized_bots = 0
            
            for symbol, bot_data in bot_items:
                    try:
                        # Проверяем, есть ли позиция на бирже для этого бота
                        if symbol in exchange_positions:
                            # Есть позиция на бирже - обновляем данные бота
                            exchange_pos = exchange_positions[symbol]
                            
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
                            
                            # ✅ КРИТИЧНО: Обрабатываем ТОЛЬКО ботов, которые были в позиции!
                            # Пропускаем ботов со статусом idle, running, paused и т.д. - они не были в позиции
                            if old_status not in [
                                BOT_STATUS.get('IN_POSITION_LONG'),
                                BOT_STATUS.get('IN_POSITION_SHORT')
                            ]:
                                # Бот не был в позиции - просто пропускаем его
                                continue
                            
                            old_position_size = bot_data.get('position_size', 0)
                            manual_closed = True  # Если мы здесь, значит бот был в позиции

                            # ✅ УПРОЩЕНО: Получаем данные для проверки дубликатов
                            entry_time_str = bot_data.get('position_start_time') or bot_data.get('entry_time')
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
                            
                            # ✅ УПРОЩЕНО: Проверяем дубликаты сразу после получения entry_price
                            bot_id = bot_data.get('id') or symbol
                            already_closed_trade = _check_if_trade_already_closed(bot_id, symbol, entry_price, entry_time_str)

                            # Получаем рыночную цену для фиксации закрытия
                            if manual_closed:
                                try:
                                    exchange_obj = get_exchange()
                                    if exchange_obj and hasattr(exchange_obj, 'get_ticker'):
                                        ticker = exchange_obj.get_ticker(symbol)
                                        if ticker and ticker.get('last'):
                                            exit_price = float(ticker.get('last'))
                                except Exception as manual_price_error:
                                    pass

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
                                # entry_time_str уже получен выше
                                duration_hours = 0.0
                                if entry_time_str:
                                    try:
                                        entry_time = datetime.fromisoformat(entry_time_str.replace('Z', ''))
                                        now_utc = datetime.now(timezone.utc)
                                        entry_utc = entry_time.replace(tzinfo=timezone.utc) if entry_time.tzinfo is None else entry_time
                                        duration_hours = (now_utc - entry_utc).total_seconds() / 3600.0
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

                                if not already_closed_trade:
                                    # КРИТИЧНО: Логируем только если это была позиция бота (бот был в позиции)
                                    # Это НЕ ручные сделки трейдера, а закрытие позиций ботов вручную на бирже
                                    history_log_position_closed(
                                        bot_id=bot_id,
                                        symbol=symbol,
                                        direction=direction or 'UNKNOWN',
                                        exit_price=exit_price or entry_price or 0.0,
                                        pnl=pnl_usdt,
                                        roi=roi_percent,
                                        reason='MANUAL_CLOSE',
                                        entry_data=entry_data,
                                        market_data=market_data,
                                        is_simulated=False,  # КРИТИЧНО: это сделки ботов, закрытые вручную на бирже
                                    )

                                    # КРИТИЧНО: Также сохраняем в bots_data.db для истории торговли ботов
                                    try:
                                        from bot_engine.bots_database import get_bots_database
                                        bots_db = get_bots_database()
                                        # Аккуратно рассчитываем entry_timestamp, чтобы избежать NameError
                                        entry_timestamp = None
                                        if entry_time_str:
                                            try:
                                                entry_dt = datetime.fromisoformat(
                                                    entry_time_str.replace("Z", "")
                                                )
                                                entry_timestamp = entry_dt.timestamp() * 1000
                                            except Exception:
                                                entry_timestamp = datetime.now().timestamp() * 1000
                                        else:
                                            entry_time_str = datetime.now().isoformat()
                                            entry_timestamp = datetime.now().timestamp() * 1000

                                        trade_data = {
                                            "bot_id": bot_id,
                                            "symbol": symbol,
                                            "direction": direction or "UNKNOWN",
                                            "entry_price": entry_price or 0.0,
                                            "exit_price": exit_price or entry_price or 0.0,
                                            "entry_time": entry_time_str,
                                            "exit_time": datetime.now().isoformat(),
                                            "entry_timestamp": entry_timestamp,
                                            "exit_timestamp": datetime.now().timestamp() * 1000,
                                            "position_size_usdt": bot_data.get("volume_value"),
                                            "position_size_coins": position_size_coins,
                                            "pnl": pnl_usdt,
                                            "roi": roi_percent,
                                            "status": "CLOSED",
                                            "close_reason": "MANUAL_CLOSE",
                                            "decision_source": bot_data.get(
                                                "decision_source", "SCRIPT"
                                            ),
                                            "ai_decision_id": bot_data.get("ai_decision_id"),
                                            "ai_confidence": bot_data.get("ai_confidence"),
                                            "entry_rsi": None,  # TODO: получить из entry_data если есть
                                            "exit_rsi": None,
                                            "entry_trend": entry_data.get("trend"),
                                            "exit_trend": None,
                                            "entry_volatility": entry_data.get("volatility"),
                                            "entry_volume_ratio": None,
                                            "is_successful": pnl_usdt > 0 if pnl_usdt else False,
                                            "is_simulated": False,
                                            "source": "bot_manual_close",
                                            "order_id": None,
                                            "extra_data": {
                                                "entry_data": entry_data,
                                                "market_data": market_data,
                                            },
                                        }

                                        trade_id = bots_db.save_bot_trade_history(trade_data)
                                        if trade_id:
                                            logger.info(
                                                f"[SYNC_EXCHANGE] ✅ История сделки {symbol} сохранена в bots_data.db (ID: {trade_id})"
                                            )
                                    except Exception as bots_db_error:
                                        logger.warning(
                                            f"[SYNC_EXCHANGE] ⚠️ Ошибка сохранения истории в bots_data.db: {bots_db_error}"
                                        )
                                    logger.info(
                                        f"[SYNC_EXCHANGE] ✋ {symbol}: позиция закрыта вручную на бирже "
                                        f"(entry={entry_price:.6f}, exit={exit_price:.6f}, pnl={pnl_usdt:.2f} USDT)"
                                    )
                                else:
                                    # Позиция уже была обработана ранее - просто удаляем бота без повторного логирования
                                    pass
                            
                            # ✅ УПРОЩЕНО: Логируем удаление бота (делистинг проверяется в отдельной функции)
                            logger.info(f"[SYNC_EXCHANGE] 🗑️ {symbol}: Удаляем бота (позиция закрыта на бирже, статус: {old_status})")
                            
                            # ✅ КРИТИЧНО: Сохраняем timestamp последнего закрытия ДО удаления бота
                            try:
                                current_timestamp = datetime.now().timestamp()
                                with bots_data_lock:
                                    if 'last_close_timestamps' not in bots_data:
                                        bots_data['last_close_timestamps'] = {}
                                    bots_data['last_close_timestamps'][symbol] = current_timestamp
                                try:
                                    from bot_engine.bot_config import get_current_timeframe
                                    _tf = get_current_timeframe()
                                except Exception:
                                    _tf = '?'
                                logger.info(f"[SYNC_EXCHANGE] ⏰ Сохранен timestamp последнего закрытия для {symbol}: {current_timestamp} (через 1 свечу {_tf} разрешим новый вход)")
                            except Exception as timestamp_error:
                                logger.warning(f"[SYNC_EXCHANGE] ⚠️ Ошибка сохранения timestamp закрытия для {symbol}: {timestamp_error}")
                            
                            # Удаляем бота из системы (с блокировкой!)
                            with bots_data_lock:
                                if symbol in bots_data['bots']:
                                    del bots_data['bots'][symbol]
                            
                            # Сохраняем состояние после удаления
                            save_bots_state()
                            
                            synchronized_bots += 1
                        
                    except Exception as e:
                        logger.error(f"[SYNC_EXCHANGE] ❌ Ошибка синхронизации бота {symbol}: {e}")
            
            # Сохраняем обновленное состояние
            save_bots_state()
            
            return True
            
        except Exception as e:
            logger.error(f"[SYNC_EXCHANGE] ❌ Ошибка получения позиций с биржи: {e}")
            return False
        
    except Exception as e:
        logger.error(f"[SYNC_EXCHANGE] ❌ Общая ошибка синхронизации: {e}")
        return False

