"""╨д╤Г╨╜╨║╤Ж╨╕╨╕ ╨║╤Н╤И╨╕╤А╨╛╨▓╨░╨╜╨╕╤П, ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨░╤Ж╨╕╨╕ ╨╕ ╤Г╨┐╤А╨░╨▓╨╗╨╡╨╜╨╕╤П ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡╨╝

╨Т╨║╨╗╤О╤З╨░╨╡╤В:
- ╨д╤Г╨╜╨║╤Ж╨╕╨╕ ╤А╨░╨▒╨╛╤В╤Л ╤Б RSI ╨║╤Н╤И╨╛╨╝
- ╨б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╨╡/╨╖╨░╨│╤А╤Г╨╖╨║╨░ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╤П ╨▒╨╛╤В╨╛╨▓
- ╨б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨░╤Ж╨╕╤П ╤Б ╨▒╨╕╤А╨╢╨╡╨╣
- ╨Ю╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣
- ╨г╨┐╤А╨░╨▓╨╗╨╡╨╜╨╕╨╡ ╨╖╤А╨╡╨╗╤Л╨╝╨╕ ╨╝╨╛╨╜╨╡╤В╨░╨╝╨╕
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

# ╨Ш╨╝╨┐╨╛╤А╤В SystemConfig
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

# ╨Ъ╨╛╨╜╤Б╤В╨░╨╜╤В╤Л ╤В╨╡╨┐╨╡╤А╤М ╨▓ SystemConfig

# ╨Ш╨╝╨┐╨╛╤А╤В╨╕╤А╤Г╨╡╨╝ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Л╨╡ ╨┐╨╡╤А╨╡╨╝╨╡╨╜╨╜╤Л╨╡ ╨╕╨╖ imports_and_globals
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
    # MATURE_COINS_FILE ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜ ╨▓ maturity.py
    try:
        from bots_modules.maturity import MATURE_COINS_FILE, save_mature_coins_storage
    except:
        MATURE_COINS_FILE = 'data/mature_coins.json'
        def save_mature_coins_storage():
            pass  # Fallback function
    
    # ╨Ч╨░╨│╨╗╤Г╤И╨║╨░ ╨┤╨╗╤П ensure_exchange_initialized (╨╕╨╖╨▒╨╡╨│╨░╨╡╨╝ ╤Ж╨╕╨║╨╗╨╕╤З╨╡╤Б╨║╨╛╨│╨╛ ╨╕╨╝╨┐╨╛╤А╤В╨░)
    def ensure_exchange_initialized():
        """╨Ч╨░╨│╨╗╤Г╤И╨║╨░, ╨▒╤Г╨┤╨╡╤В ╨┐╨╡╤А╨╡╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜╨░ ╨┐╤А╨╕ ╨┐╨╡╤А╨▓╨╛╨╝ ╨╕╤Б╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╨╜╨╕╨╕"""
        try:
            from bots_modules.init_functions import ensure_exchange_initialized as real_func
            # ╨Ч╨░╨╝╨╡╨╜╤П╨╡╨╝ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Г╤О ╤Д╤Г╨╜╨║╤Ж╨╕╤О ╨╜╨░ ╨╜╨░╤Б╤В╨╛╤П╤Й╤Г╤О
            globals()['ensure_exchange_initialized'] = real_func
            return real_func()
        except:
            return exchange is not None
except ImportError as e:
    print(f"Warning: Could not import globals in sync_and_cache: {e}")
    # ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╨╖╨░╨│╨╗╤Г╤И╨║╨╕
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

# ╨Ъ╨░╤А╤В╨░ ╤Б╨╛╨╛╤В╨▓╨╡╤В╤Б╤В╨▓╨╕╤П ╨║╨╗╤О╤З╨╡╨╣ UI ╨╕ ╨░╤В╤А╨╕╨▒╤Г╤В╨╛╨▓ SystemConfig
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
    'system_timeframe': 'SYSTEM_TIMEFRAME'  # ╨в╨░╨╣╨╝╤Д╤А╨╡╨╣╨╝ ╤Б╨╕╤Б╤В╨╡╨╝╤Л
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
    """тЬЕ ╨г╨Я╨а╨Ю╨й╨Х╨Э╨Ю: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╤В, ╨▒╤Л╨╗╨░ ╨╗╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╤Г╨╢╨╡ ╨╖╨░╨║╤А╤Л╤В╨░ ╤А╨░╨╜╨╡╨╡ (╨┐╤А╨╡╨┤╨╛╤В╨▓╤А╨░╤Й╨░╨╡╤В ╨┤╤Г╨▒╨╗╨╕╨║╨░╤В╤Л)"""
    if not entry_price or entry_price <= 0:
        return False
    
    try:
        from bot_engine.bots_database import get_bots_database
        bots_db = get_bots_database()
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ entry_timestamp
        entry_timestamp = None
        if entry_time_str:
            try:
                entry_time = datetime.fromisoformat(entry_time_str.replace('Z', ''))
                entry_timestamp = entry_time.timestamp() * 1000
            except Exception:
                pass
        
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ 10 ╨╖╨░╨║╤А╤Л╤В╤Л╤Е ╤Б╨┤╨╡╨╗╨╛╨║
        existing_trades = bots_db.get_bot_trades_history(
            bot_id=bot_id,
            symbol=symbol,
            status='CLOSED',
            limit=10
        )
        
        if not existing_trades:
            return False
        
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╜╨░ ╨┤╤Г╨▒╨╗╨╕╨║╨░╤В╤Л
        for existing_trade in existing_trades:
            existing_entry_price = existing_trade.get('entry_price')
            existing_entry_ts = existing_trade.get('entry_timestamp')
            existing_close_reason = existing_trade.get('close_reason')
            
            # ╨б╤А╨░╨▓╨╜╨╕╨▓╨░╨╡╨╝ ╤Ж╨╡╨╜╤Г ╨▓╤Е╨╛╨┤╨░ (╨┐╨╛╨│╤А╨╡╤И╨╜╨╛╤Б╤В╤М ╨┤╨╗╤П float)
            price_match = existing_entry_price and abs(float(existing_entry_price) - float(entry_price)) < 0.0001
            
            # ╨б╤А╨░╨▓╨╜╨╕╨▓╨░╨╡╨╝ timestamp ╨╡╤Б╨╗╨╕ ╨╡╤Б╤В╤М (╨┐╨╛╨│╤А╨╡╤И╨╜╨╛╤Б╤В╤М 1 ╨╝╨╕╨╜╤Г╤В╨░)
            timestamp_match = True
            if entry_timestamp and existing_entry_ts:
                timestamp_match = abs(float(existing_entry_ts) - float(entry_timestamp)) < 60000
            
            # ╨Х╤Б╨╗╨╕ ╤Б╨╛╨▓╨┐╨░╨┤╨░╨╡╤В ╤Ж╨╡╨╜╨░ ╨╕ timestamp, ╨╕ ╤Н╤В╨╛ MANUAL_CLOSE - ╤Н╤В╨╛ ╨┤╤Г╨▒╨╗╨╕╨║╨░╤В
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
    """╨Т╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╤В ╨║╨╛╨┐╨╕╤О ╨░╨▓╤В╨╛╨║╨╛╨╜╤Д╨╕╨│╨░ ╨╕ ╨▒╨╛╤В╨╛╨▓ ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П╤Е ╨┤╨╗╤П ╨╛╨▒╤А╨░╨▒╨╛╤В╨║╨╕ ╨▓╨╜╨╡ ╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨╕."""
    with bots_data_lock:
        auto_config = copy.deepcopy(bots_data.get('auto_bot_config', DEFAULT_AUTO_BOT_CONFIG))
        bots_snapshot = {
            symbol: copy.deepcopy(bot_data)
            for symbol, bot_data in bots_data.get('bots', {}).items()
            if bot_data.get('status') in ['in_position_long', 'in_position_short']
        }
    return auto_config, bots_snapshot


def _update_bot_record(symbol, updates):
    """╨С╨╡╨╖╨╛╨┐╨░╤Б╨╜╨╛ ╨┐╤А╨╕╨╝╨╡╨╜╤П╨╡╤В ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П ╨║ bot_data, ╨╝╨╕╨╜╨╕╨╝╨╕╨╖╨╕╤А╤Г╤П ╨▓╤А╨╡╨╝╤П ╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨╕."""
    if not updates:
        return False
    with bots_data_lock:
        bot_data = bots_data['bots'].get(symbol)
        if not bot_data:
            return False
        bot_data.update(updates)
    return True


def get_system_config_snapshot():
    """╨Т╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╤В ╤В╨╡╨║╤Г╤Й╨╕╨╡ ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П SystemConfig ╨▓ ╤Д╨╛╤А╨╝╨░╤В╨╡, ╨╛╨╢╨╕╨┤╨░╨╡╨╝╨╛╨╝ UI.
    ╨Ф╨╗╤П system_timeframe ╨▒╨╡╤А╤С╨╝ ╤Д╨░╨║╤В╨╕╤З╨╡╤Б╨║╨╕╨╣ ╤В╨╡╨║╤Г╤Й╨╕╨╣ ╤В╨░╨╣╨╝╤Д╤А╨╡╨╣╨╝ (runtime/╨С╨Ф), ╨░ ╨╜╨╡ ╤В╨╛╨╗╤М╨║╨╛ ╨╕╨╖ ╨║╨╛╨╜╤Д╨╕╨│╨░,
    ╨╕╨╜╨░╤З╨╡ ╨┐╤А╨╕ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╨╕ ╨┤╤А╤Г╨│╨╕╤Е ╨╜╨░╤Б╤В╤А╨╛╨╡╨║ ╨▓ ╤Д╨░╨╣╨╗ ╨┐╨╛╨┐╨░╨┤╨░╨╗ ╨▒╤Л ╤Б╤В╨░╤А╤Л╨╣ SYSTEM_TIMEFRAME ╨╕ ╤Б╨▒╤А╨░╤Б╤Л╨▓╨░╨╗ ╤В╨░╨╣╨╝╤Д╤А╨╡╨╣╨╝ ╨╜╨░ 6h.
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
    ╨а╨░╤Б╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╤В ╨┐╨░╤А╨░╨╝╨╡╤В╤А╤Л ╤В╤А╨╡╨╣╨╗╨╕╨╜╨│-╤Б╤В╨╛╨┐╨░ ╨╜╨░ ╨╛╤Б╨╜╨╛╨▓╨╡ ╨╝╨░╤А╨╢╨╕ ╤Б╨┤╨╡╨╗╨║╨╕.

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
    """╨Я╨╛╨╗╤Г╤З╨╕╤В╤М ╨║╤Н╤И╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╨╡ RSI ╨┤╨░╨╜╨╜╤Л╨╡"""
    global coins_rsi_data
    with rsi_data_lock:
        return coins_rsi_data.get('coins', {})

def save_rsi_cache():
    """╨б╨╛╤Е╤А╨░╨╜╨╕╤В╤М ╨║╤Н╤И RSI ╨┤╨░╨╜╨╜╤Л╤Е ╨▓ ╨С╨Ф"""
    try:
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╤З╤В╨╡╨╜╨╕╨╡ ╤Б╨╗╨╛╨▓╨░╤А╤П - ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П ╨▓ Python
        coins_data = coins_rsi_data.get('coins', {})
        stats = {
            'total_coins': len(coins_data),
            'successful_coins': coins_rsi_data.get('successful_coins', 0),
            'failed_coins': coins_rsi_data.get('failed_coins', 0)
        }
        
        # тЬЕ ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨▓ ╨С╨Ф ╤З╨╡╤А╨╡╨╖ storage.py
        if storage_save_rsi_cache(coins_data, stats):
            logger.info(f" RSI ╨┤╨░╨╜╨╜╤Л╨╡ ╨┤╨╗╤П {len(coins_data)} ╨╝╨╛╨╜╨╡╤В ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╤Л ╨▓ ╨С╨Ф")
            return True
        return False
        
    except Exception as e:
        logger.error(f" ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П RSI ╨║╤Н╤И╨░ ╨▓ ╨С╨Ф: {str(e)}")
        return False

def load_rsi_cache():
    """╨Ч╨░╨│╤А╤Г╨╖╨╕╤В╤М ╨║╤Н╤И RSI ╨┤╨░╨╜╨╜╤Л╤Е ╨╕╨╖ ╨С╨Ф"""
    global coins_rsi_data
    
    try:
        # тЬЕ ╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╨╕╨╖ ╨С╨Ф ╤З╨╡╤А╨╡╨╖ storage.py
        cache_data = storage_load_rsi_cache()
        
        if not cache_data:
            logger.info(" RSI ╨║╤Н╤И ╨▓ ╨С╨Ф ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜, ╨▒╤Г╨┤╨╡╤В ╤Б╨╛╨╖╨┤╨░╨╜ ╨┐╤А╨╕ ╨┐╨╡╤А╨▓╨╛╨╝ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╕")
            return False
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨┤╨░╨╜╨╜╤Л╨╡ ╨╕╨╖ ╨║╤Н╤И╨░
        cached_coins = cache_data.get('coins', {})
        stats = cache_data.get('stats', {})
        
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Д╨╛╤А╨╝╨░╤В ╨║╤Н╤И╨░ (╤Б╤В╨░╤А╤Л╨╣ ╨╝╨░╤Б╤Б╨╕╨▓ ╨╕╨╗╨╕ ╨╜╨╛╨▓╤Л╨╣ ╤Б╨╗╨╛╨▓╨░╤А╤М)
        if isinstance(cached_coins, list):
            # ╨б╤В╨░╤А╤Л╨╣ ╤Д╨╛╤А╨╝╨░╤В - ╨┐╤А╨╡╨╛╨▒╤А╨░╨╖╤Г╨╡╨╝ ╨╝╨░╤Б╤Б╨╕╨▓ ╨▓ ╤Б╨╗╨╛╨▓╨░╤А╤М
            coins_dict = {}
            for coin in cached_coins:
                if 'symbol' in coin:
                    coins_dict[coin['symbol']] = coin
            cached_coins = coins_dict
            logger.info(" ╨Я╤А╨╡╨╛╨▒╤А╨░╨╖╨╛╨▓╨░╨╜ ╤Б╤В╨░╤А╤Л╨╣ ╤Д╨╛╤А╨╝╨░╤В ╨║╤Н╤И╨░ (╨╝╨░╤Б╤Б╨╕╨▓ -> ╤Б╨╗╨╛╨▓╨░╤А╤М)")
        
        with rsi_data_lock:
            coins_rsi_data.update({
                'coins': cached_coins,
                'successful_coins': stats.get('successful_coins', len(cached_coins)),
                'failed_coins': stats.get('failed_coins', 0),
                'total_coins': len(cached_coins),
                'last_update': datetime.now().isoformat(),  # ╨Т╤Б╨╡╨│╨┤╨░ ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╤В╨╡╨║╤Г╤Й╨╡╨╡ ╨▓╤А╨╡╨╝╤П
                'update_in_progress': False
            })
        
        logger.info(f" ╨Ч╨░╨│╤А╤Г╨╢╨╡╨╜╨╛ {len(cached_coins)} ╨╝╨╛╨╜╨╡╤В ╨╕╨╖ RSI ╨║╤Н╤И╨░ (╨С╨Ф)")
        return True
        
    except Exception as e:
        logger.error(f" ╨Ю╤И╨╕╨▒╨║╨░ ╨╖╨░╨│╤А╤Г╨╖╨║╨╕ RSI ╨║╤Н╤И╨░ ╨╕╨╖ ╨С╨Ф: {str(e)}")
        return False

def save_default_config():
    """╨б╨╛╤Е╤А╨░╨╜╤П╨╡╤В ╨┤╨╡╤Д╨╛╨╗╤В╨╜╤Г╤О ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤О ╨▓ ╤Д╨░╨╣╨╗ ╨┤╨╗╤П ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П"""
    try:
        with open(DEFAULT_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_AUTO_BOT_CONFIG, f, indent=2, ensure_ascii=False)
        
        logger.info(f" тЬЕ ╨Ф╨╡╤Д╨╛╨╗╤В╨╜╨░╤П ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤П ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨░ ╨▓ {DEFAULT_CONFIG_FILE}")
        return True
        
    except Exception as e:
        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П ╨┤╨╡╤Д╨╛╨╗╤В╨╜╨╛╨╣ ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╨╕: {e}")
        return False

def load_default_config():
    """╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╤В ╨┤╨╡╤Д╨╛╨╗╤В╨╜╤Г╤О ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤О ╨╕╨╖ ╤Д╨░╨╣╨╗╨░"""
    try:
        if os.path.exists(DEFAULT_CONFIG_FILE):
            with open(DEFAULT_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # ╨Х╤Б╨╗╨╕ ╤Д╨░╨╣╨╗╨░ ╨╜╨╡╤В, ╤Б╨╛╨╖╨┤╨░╨╡╨╝ ╨╡╨│╨╛ ╤Б ╤В╨╡╨║╤Г╤Й╨╕╨╝╨╕ ╨┤╨╡╤Д╨╛╨╗╤В╨╜╤Л╨╝╨╕ ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П╨╝╨╕
            save_default_config()
            return DEFAULT_AUTO_BOT_CONFIG.copy()
            
    except Exception as e:
        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╖╨░╨│╤А╤Г╨╖╨║╨╕ ╨┤╨╡╤Д╨╛╨╗╤В╨╜╨╛╨╣ ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╨╕: {e}")
        return DEFAULT_AUTO_BOT_CONFIG.copy()

def restore_default_config():
    """╨Т╨╛╤Б╤Б╤В╨░╨╜╨░╨▓╨╗╨╕╨▓╨░╨╡╤В ╨┤╨╡╤Д╨╛╨╗╤В╨╜╤Г╤О ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤О Auto Bot"""
    try:
        default_config = load_default_config()
        
        with bots_data_lock:
            # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨║╤А╨╕╤В╨╕╤З╨╡╤Б╨║╨╕ ╨▓╨░╨╢╨╜╤Л╨╡ ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П (╨╜╨╡ ╤Б╨▒╤А╨░╤Б╤Л╨▓╨░╨╡╨╝ ╨╕╤Е ╨┐╤А╨╕ ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╕)
            current_enabled = bots_data['auto_bot_config'].get('enabled', False)
            current_trading_enabled = bots_data['auto_bot_config'].get('trading_enabled', True)
            
            # ╨Т╨╛╤Б╤Б╤В╨░╨╜╨░╨▓╨╗╨╕╨▓╨░╨╡╨╝ ╨┤╨╡╤Д╨╛╨╗╤В╨╜╤Л╨╡ ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П
            bots_data['auto_bot_config'] = default_config.copy()
            
            # ╨Т╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╨╝ ╤В╨╡╨║╤Г╤Й╨╕╨╡ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╤П ╨▓╨░╨╢╨╜╤Л╤Е ╨╜╨░╤Б╤В╤А╨╛╨╡╨║
            bots_data['auto_bot_config']['enabled'] = current_enabled
            bots_data['auto_bot_config']['trading_enabled'] = current_trading_enabled
        
        # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡
        save_result = save_bots_state()
        
        logger.info(" тЬЕ ╨Ф╨╡╤Д╨╛╨╗╤В╨╜╨░╤П ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤П ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨░")
        return save_result
        
    except Exception as e:
        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П ╨┤╨╡╤Д╨╛╨╗╤В╨╜╨╛╨╣ ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╨╕: {e}")
        return False

def update_process_state(process_name, status_update):
    """╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╤В ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨┐╤А╨╛╤Ж╨╡╤Б╤Б╨░"""
    try:
        if process_name in process_state:
            process_state[process_name].update(status_update)
            
            # ╨Р╨▓╤В╨╛╨╝╨░╤В╨╕╤З╨╡╤Б╨║╨╕ ╤Б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨┐╤А╨╛╤Ж╨╡╤Б╤Б╨╛╨▓
            save_process_state()
            
    except Exception as e:
        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╤П {process_name}: {e}")

def save_process_state():
    """╨б╨╛╤Е╤А╨░╨╜╤П╨╡╤В ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨▓╤Б╨╡╤Е ╨┐╤А╨╛╤Ж╨╡╤Б╤Б╨╛╨▓ ╨▓ ╨С╨Ф"""
    try:
        # тЬЕ ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨▓ ╨С╨Ф ╤З╨╡╤А╨╡╨╖ storage.py
        if storage_save_process_state(process_state):
            # ╨г╨▒╤А╨░╨╜╨╛ ╨╕╨╖╨▒╤Л╤В╨╛╤З╨╜╨╛╨╡ DEBUG ╨╗╨╛╨│╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡ ╨┤╨╗╤П ╤Г╨╝╨╡╨╜╤М╤И╨╡╨╜╨╕╤П ╤Б╨┐╨░╨╝╨░
            # logger.debug("ЁЯТ╛ ╨б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨┐╤А╨╛╤Ж╨╡╤Б╤Б╨╛╨▓ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╛ ╨▓ ╨С╨Ф")
            return True
        return False
        
    except Exception as e:
        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╤П ╨┐╤А╨╛╤Ж╨╡╤Б╤Б╨╛╨▓ ╨▓ ╨С╨Ф: {e}")
        return False

def load_process_state():
    """╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╤В ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨┐╤А╨╛╤Ж╨╡╤Б╤Б╨╛╨▓ ╨╕╨╖ ╨С╨Ф"""
    try:
        # тЬЕ ╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╨╕╨╖ ╨С╨Ф ╤З╨╡╤А╨╡╨╖ storage.py
        state_data = storage_load_process_state()
        
        if not state_data:
            logger.info(f" ЁЯУБ ╨б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨┐╤А╨╛╤Ж╨╡╤Б╤Б╨╛╨▓ ╨▓ ╨С╨Ф ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜╨╛, ╨╜╨░╤З╨╕╨╜╨░╨╡╨╝ ╤Б ╨┤╨╡╤Д╨╛╨╗╤В╨╜╨╛╨│╨╛")
            save_process_state()  # ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╨▓ ╨С╨Ф
            return False
        
        if 'process_state' in state_data:
            # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╨╛╨╡ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡
            for process_name, process_info in state_data['process_state'].items():
                if process_name in process_state:
                    process_state[process_name].update(process_info)
            
            last_saved = state_data.get('last_saved', '╨╜╨╡╨╕╨╖╨▓╨╡╤Б╤В╨╜╨╛')
            logger.info(f" тЬЕ ╨б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨┐╤А╨╛╤Ж╨╡╤Б╤Б╨╛╨▓ ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨╛ ╨╕╨╖ ╨С╨Ф (╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╛: {last_saved})")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╖╨░╨│╤А╤Г╨╖╨║╨╕ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╤П ╨┐╤А╨╛╤Ж╨╡╤Б╤Б╨╛╨▓ ╨╕╨╖ ╨С╨Ф: {e}")
        return False

def save_system_config(config_data):
    """╨б╨╛╤Е╤А╨░╨╜╤П╨╡╤В ╤Б╨╕╤Б╤В╨╡╨╝╨╜╤Л╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╨╜╨░╨┐╤А╤П╨╝╤Г╤О ╨▓ bot_config.py."""
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
            logger.info("[SYSTEM_CONFIG] тЬЕ ╨Э╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╤Л ╨▓ bot_engine/bot_config.py")
        return success

    except Exception as e:
        logger.error(f"[SYSTEM_CONFIG] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П ╤Б╨╕╤Б╤В╨╡╨╝╨╜╤Л╤Е ╨╜╨░╤Б╤В╤А╨╛╨╡╨║: {e}")
        return False


def load_system_config():
    """╨Я╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╢╨░╨╡╤В SystemConfig ╨╕╨╖ bot_config.py ╨╕ ╨┐╤А╨╕╨╝╨╡╨╜╤П╨╡╤В ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П ╨▓ ╨┐╨░╨╝╤П╤В╤М."""
    try:
        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤В╨╡╨║╤Г╤Й╨╕╨╣ ╤В╨░╨╣╨╝╤Д╤А╨╡╨╣╨╝ ╨╕╨╖ ╨С╨Ф ╨┐╨╡╤А╨╡╨┤ ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╖╨║╨╛╨╣ ╨╝╨╛╨┤╤Г╨╗╤П
        # ╤З╤В╨╛╨▒╤Л ╨╜╨╡ ╨┐╨╛╤В╨╡╤А╤П╤В╤М ╨╡╨│╨╛ ╨┐╤А╨╕ reload (╨┐╤А╨╕╨╛╤А╨╕╤В╨╡╤В ╨С╨Ф ╨╜╨░╨┤ ╨║╨╛╨╜╤Д╨╕╨│╨╛╨╝)
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

        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Т╨╛╤Б╤Б╤В╨░╨╜╨░╨▓╨╗╨╕╨▓╨░╨╡╨╝ ╤В╨░╨╣╨╝╤Д╤А╨╡╨╣╨╝ ╨┐╨╛╤Б╨╗╨╡ ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╖╨║╨╕ ╨╝╨╛╨┤╤Г╨╗╤П
        # ╨Я╤А╨╕╨╛╤А╨╕╤В╨╡╤В: ╨С╨Ф > SystemConfig.SYSTEM_TIMEFRAME ╨╕╨╖ ╤Д╨░╨╣╨╗╨░
        try:
            from bot_engine.bot_config import set_current_timeframe, get_current_timeframe
            if saved_timeframe_from_db:
                # ╨Х╤Б╨╗╨╕ ╨╡╤Б╤В╤М ╤В╨░╨╣╨╝╤Д╤А╨╡╨╣╨╝ ╨▓ ╨С╨Ф - ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╡╨│╨╛ (╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╤В╨╡╨╗╤М ╨┐╨╡╤А╨╡╨║╨╗╤О╤З╨░╨╗ ╤З╨╡╤А╨╡╨╖ UI)
                set_current_timeframe(saved_timeframe_from_db)
            else:
                # ╨Х╤Б╨╗╨╕ ╨╜╨╡╤В ╨▓ ╨С╨Ф - ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╕╨╖ ╨║╨╛╨╜╤Д╨╕╨│╨░
                config_timeframe = getattr(file_system_config, 'SYSTEM_TIMEFRAME', None)
                if config_timeframe:
                    set_current_timeframe(config_timeframe)
        except Exception as tf_err:
            logger.warning(f"[SYSTEM_CONFIG] тЪая╕П ╨Ю╤И╨╕╨▒╨║╨░ ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П ╤В╨░╨╣╨╝╤Д╤А╨╡╨╣╨╝╨░: {tf_err}")

        logger.info("[SYSTEM_CONFIG] тЬЕ ╨Ъ╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤П ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╢╨╡╨╜╨░ ╨╕╨╖ bot_engine/bot_config.py")
        return True

    except Exception as e:
        logger.error(f"[SYSTEM_CONFIG] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╖╨░╨│╤А╤Г╨╖╨║╨╕ ╤Б╨╕╤Б╤В╨╡╨╝╨╜╤Л╤Е ╨╜╨░╤Б╤В╤А╨╛╨╡╨║: {e}")
        return False

def save_bots_state():
    """╨б╨╛╤Е╤А╨░╨╜╤П╨╡╤В ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨▓╤Б╨╡╤Е ╨▒╨╛╤В╨╛╨▓ ╨▓ ╨С╨Ф"""
    try:
        # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ш╨Х: ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╤В╨░╨╣╨╝╨░╤Г╤В ╨┤╨╗╤П ╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨╕ ╤З╤В╨╛╨▒╤Л ╨╜╨╡ ╨▓╨╕╤Б╨╡╤В╤М ╨┐╤А╨╕ ╨╛╤Б╤В╨░╨╜╨╛╨▓╨║╨╡
        import threading
        
        requester = threading.current_thread().name
        # ╨Я╤Л╤В╨░╨╡╨╝╤Б╤П ╨╖╨░╤Е╨▓╨░╤В╨╕╤В╤М ╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╤Г ╤Б ╤В╨░╨╣╨╝╨░╤Г╤В╨╛╨╝ (╤Г╨▓╨╡╨╗╨╕╤З╨╡╨╜╨╛ ╨┤╨╛ 5 ╤Б╨╡╨║╤Г╨╜╨┤)
        acquired = bots_data_lock.acquire(timeout=5.0)
        if not acquired:
            active_threads = [t.name for t in threading.enumerate()[:10]]
            logger.warning(
                "[SAVE_STATE] тЪая╕П ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╤Г ╨╖╨░ 5 ╤Б╨╡╨║╤Г╨╜╨┤ - ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╨╡ "
                f"(thread={requester}, active_threads={active_threads})"
            )
            return False
        
        try:
            # ╨б╨╛╨▒╨╕╤А╨░╨╡╨╝ ╨┤╨░╨╜╨╜╤Л╨╡ ╨▒╨╛╤В╨╛╨▓
            bots_data_to_save = {}
            for symbol, bot_data in bots_data['bots'].items():
                bots_data_to_save[symbol] = bot_data
            
            # тЬЕ ╨г╨С╨а╨Р╨Э╨Ю: auto_bot_config ╨▒╨╛╨╗╤М╤И╨╡ ╨Э╨Х ╤Б╨╛╤Е╤А╨░╨╜╤П╨╡╤В╤Б╤П ╨▓ ╨С╨Ф
            # ╨Э╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╤Е╤А╨░╨╜╤П╤В╤Б╤П ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╨▓ bot_engine/bot_config.py ╤З╨╡╤А╨╡╨╖ config_writer
            # ╨Я╨╡╤А╨╡╨┤╨░╨╡╨╝ ╨┐╤Г╤Б╤В╨╛╨╣ ╤Б╨╗╨╛╨▓╨░╤А╤М, ╤З╤В╨╛╨▒╤Л ╨╜╨╡ ╤Б╨╛╤Е╤А╨░╨╜╤П╤В╤М ╨▓ ╨С╨Ф
            auto_bot_config_to_save = {}
        finally:
            bots_data_lock.release()
        
        # тЬЕ ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨▓ ╨С╨Ф ╤З╨╡╤А╨╡╨╖ storage.py (╤В╨╛╨╗╤М╨║╨╛ ╨▒╨╛╤В╤Л, ╨▒╨╡╨╖ auto_bot_config)
        success = storage_save_bots_state(bots_data_to_save, auto_bot_config_to_save)
        if not success:
            logger.error("[SAVE_STATE] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╤П ╨▓ ╨С╨Ф")
            return False
        
        # ╨г╨▒╤А╨░╨╜╨╛ ╨╕╨╖╨▒╤Л╤В╨╛╤З╨╜╨╛╨╡ DEBUG ╨╗╨╛╨│╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡ ╨┤╨╗╤П ╤Г╨╝╨╡╨╜╤М╤И╨╡╨╜╨╕╤П ╤Б╨┐╨░╨╝╨░
        # logger.debug("[SAVE_STATE] тЬЕ ╨б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨▒╨╛╤В╨╛╨▓ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╛ ╨▓ ╨С╨Ф")
        return True
        
    except Exception as e:
        logger.error(f"[SAVE_STATE] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╤П: {e}")
        return False

def save_auto_bot_config(changed_data=None):
    """╨б╨╛╤Е╤А╨░╨╜╤П╨╡╤В ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤О ╨░╨▓╤В╨╛╨▒╨╛╤В╨░ ╨▓ bot_config.py
    
    тЬЕ ╨в╨╡╨┐╨╡╤А╤М ╤Б╨╛╤Е╤А╨░╨╜╤П╨╡╤В ╨╜╨░╨┐╤А╤П╨╝╤Г╤О ╨▓ bot_engine/bot_config.py
    - ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╤В ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╜╤Л╨╡ ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П (╨╡╤Б╨╗╨╕ ╨┐╨╡╤А╨╡╨┤╨░╨╜ changed_data)
    - ╨Ъ╨╛╨╝╨╝╨╡╨╜╤В╨░╤А╨╕╨╕ ╨▓ ╤Д╨░╨╣╨╗╨╡ ╤Б╨╛╤Е╤А╨░╨╜╤П╤О╤В╤Б╤П
    - ╨Р╨▓╤В╨╛╨╝╨░╤В╨╕╤З╨╡╤Б╨║╨╕ ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╢╨░╨╡╤В ╨╝╨╛╨┤╤Г╨╗╤М ╨┐╨╛╤Б╨╗╨╡ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П (╨Э╨Х ╤В╤А╨╡╨▒╤Г╨╡╤В╤Б╤П ╨┐╨╡╤А╨╡╨╖╨░╨┐╤Г╤Б╨║!)
    
    Args:
        changed_data: dict ╤Б ╤В╨╛╨╗╤М╨║╨╛ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╜╤Л╨╝╨╕ ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П╨╝╨╕ (╨╛╨┐╤Ж╨╕╨╛╨╜╨░╨╗╤М╨╜╨╛)
                      ╨Х╤Б╨╗╨╕ ╨╜╨╡ ╨┐╨╡╤А╨╡╨┤╨░╨╜, ╤Б╨╛╤Е╤А╨░╨╜╤П╨╡╤В ╨▓╨╡╤Б╤М config_data (╨┤╨╗╤П ╨╛╨▒╤А╨░╤В╨╜╨╛╨╣ ╤Б╨╛╨▓╨╝╨╡╤Б╤В╨╕╨╝╨╛╤Б╤В╨╕)
    """
    try:
        from bots_modules.config_writer import save_auto_bot_config_to_py
        import importlib
        import sys
        
        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Х╨б╨Ъ╨Ш ╨Т╨Р╨Ц╨Э╨Ю: ╨Х╤Б╨╗╨╕ ╨┐╨╡╤А╨╡╨┤╨░╨╜ changed_data, ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╨╡╨│╨╛!
        # ╨Ш╨╜╨░╤З╨╡ ╨▒╨╡╤А╨╡╨╝ ╨▓╨╡╤Б╤М config_data (╨┤╨╗╤П ╨╛╨▒╤А╨░╤В╨╜╨╛╨╣ ╤Б╨╛╨▓╨╝╨╡╤Б╤В╨╕╨╝╨╛╤Б╤В╨╕)
        if changed_data is not None:
            # ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╜╤Л╨╡ ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П
            config_data = changed_data.copy()
            logger.info(f"[SAVE_CONFIG] ЁЯФН ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╜╤Л╨╡ ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П: {list(config_data.keys())}")
        else:
            # ╨Ю╨▒╤А╨░╤В╨╜╨░╤П ╤Б╨╛╨▓╨╝╨╡╤Б╤В╨╕╨╝╨╛╤Б╤В╤М: ╨▒╨╡╤А╨╡╨╝ ╨▓╨╡╤Б╤М config
            with bots_data_lock:
                config_data = bots_data['auto_bot_config'].copy()
            logger.info(f"[SAVE_CONFIG] ЁЯФН ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨▓╨╡╤Б╤М ╨║╨╛╨╜╤Д╨╕╨│ (changed_data ╨╜╨╡ ╨┐╨╡╤А╨╡╨┤╨░╨╜)")
        
        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Х╨б╨Ъ╨Ш ╨Т╨Р╨Ц╨Э╨Ю: ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ enabled ╨┐╨╡╤А╨╡╨┤ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╨╡╨╝
        logger.info(f"[SAVE_CONFIG] ЁЯФН enabled ╨┐╨╡╤А╨╡╨┤ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╨╡╨╝: {config_data.get('enabled')}")
        
        # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨▓ bot_config.py
        success = save_auto_bot_config_to_py(config_data)
        
        if success:
            logger.info(f"[SAVE_CONFIG] тЬЕ ╨Ъ╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤П ╨░╨▓╤В╨╛╨▒╨╛╤В╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨░ ╨▓ bot_engine/bot_config.py")
            # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤О ╨▓ ╨┐╨░╨╝╤П╤В╨╕ ╨╕╨╖ ╨б╨Ю╨е╨а╨Р╨Э╨Х╨Э╨Э╨л╨е ╨┤╨░╨╜╨╜╤Л╤Е (╨╜╨╡ ╨╕╨╖ DEFAULT!)
            with bots_data_lock:
                # тЬЕ ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╜╨╛╨▓╤Л╨╡ RSI exit ╤Б ╤Г╤З╨╡╤В╨╛╨╝ ╤В╤А╨╡╨╜╨┤╨░
                old_rsi_long_with = bots_data['auto_bot_config'].get('rsi_exit_long_with_trend')
                old_rsi_long_against = bots_data['auto_bot_config'].get('rsi_exit_long_against_trend')
                old_rsi_short_with = bots_data['auto_bot_config'].get('rsi_exit_short_with_trend')
                old_rsi_short_against = bots_data['auto_bot_config'].get('rsi_exit_short_against_trend')
                
                # ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╨з╨в╨Ю ╨б╨Ю╨е╨а╨Р╨Э╨Х╨Э╨Э╨л╨Х ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П, ╨░ ╨╜╨╡ ╨┤╨╡╤Д╨╛╨╗╤В╨╜╤Л╨╡!
                bots_data['auto_bot_config'].update(config_data)
                
                new_rsi_long_with = bots_data['auto_bot_config'].get('rsi_exit_long_with_trend')
                new_rsi_long_against = bots_data['auto_bot_config'].get('rsi_exit_long_against_trend')
                new_rsi_short_with = bots_data['auto_bot_config'].get('rsi_exit_short_with_trend')
                new_rsi_short_against = bots_data['auto_bot_config'].get('rsi_exit_short_against_trend')
            
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤З╤В╨╛ ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П ╨┤╨╡╨╣╤Б╤В╨▓╨╕╤В╨╡╨╗╤М╨╜╨╛ ╨╡╤Б╤В╤М
            if new_rsi_long_with is None:
                logger.error(f"[SAVE_CONFIG] тЭМ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Х╨б╨Ъ╨Р╨п ╨Ю╨и╨Ш╨С╨Ъ╨Р: rsi_exit_long_with_trend ╨╛╤В╤Б╤Г╤В╤Б╤В╨▓╤Г╨╡╤В ╨▓ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╜╤Л╤Е ╨┤╨░╨╜╨╜╤Л╤Е!")
            if new_rsi_long_against is None:
                logger.error(f"[SAVE_CONFIG] тЭМ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Х╨б╨Ъ╨Р╨п ╨Ю╨и╨Ш╨С╨Ъ╨Р: rsi_exit_long_against_trend ╨╛╤В╤Б╤Г╤В╤Б╤В╨▓╤Г╨╡╤В ╨▓ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╜╤Л╤Е ╨┤╨░╨╜╨╜╤Л╤Е!")
            if new_rsi_short_with is None:
                logger.error(f"[SAVE_CONFIG] тЭМ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Х╨б╨Ъ╨Р╨п ╨Ю╨и╨Ш╨С╨Ъ╨Р: rsi_exit_short_with_trend ╨╛╤В╤Б╤Г╤В╤Б╤В╨▓╤Г╨╡╤В ╨▓ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╜╤Л╤Е ╨┤╨░╨╜╨╜╤Л╤Е!")
            if new_rsi_short_against is None:
                logger.error(f"[SAVE_CONFIG] тЭМ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Х╨б╨Ъ╨Р╨п ╨Ю╨и╨Ш╨С╨Ъ╨Р: rsi_exit_short_against_trend ╨╛╤В╤Б╤Г╤В╤Б╤В╨▓╤Г╨╡╤В ╨▓ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╜╤Л╤Е ╨┤╨░╨╜╨╜╤Л╤Е!")
            
            # ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П RSI exit ╨┐╨╛╤А╨╛╨│╨╛╨▓
            if old_rsi_long_with is not None and new_rsi_long_with is not None and old_rsi_long_with != new_rsi_long_with:
                logger.info(f"[SAVE_CONFIG] ЁЯФД RSI LONG exit (╨┐╨╛ ╤В╤А╨╡╨╜╨┤╤Г) ╨╕╨╖╨╝╨╡╨╜╨╡╨╜: {old_rsi_long_with} тЖТ {new_rsi_long_with}")
            if old_rsi_long_against is not None and new_rsi_long_against is not None and old_rsi_long_against != new_rsi_long_against:
                logger.info(f"[SAVE_CONFIG] ЁЯФД RSI LONG exit (╨┐╤А╨╛╤В╨╕╨▓ ╤В╤А╨╡╨╜╨┤╨░) ╨╕╨╖╨╝╨╡╨╜╨╡╨╜: {old_rsi_long_against} тЖТ {new_rsi_long_against}")
            if old_rsi_short_with is not None and new_rsi_short_with is not None and old_rsi_short_with != new_rsi_short_with:
                logger.info(f"[SAVE_CONFIG] ЁЯФД RSI SHORT exit (╨┐╨╛ ╤В╤А╨╡╨╜╨┤╤Г) ╨╕╨╖╨╝╨╡╨╜╨╡╨╜: {old_rsi_short_with} тЖТ {new_rsi_short_with}")
            if old_rsi_short_against is not None and new_rsi_short_against is not None and old_rsi_short_against != new_rsi_short_against:
                logger.info(f"[SAVE_CONFIG] ЁЯФД RSI SHORT exit (╨┐╤А╨╛╤В╨╕╨▓ ╤В╤А╨╡╨╜╨┤╨░) ╨╕╨╖╨╝╨╡╨╜╨╡╨╜: {old_rsi_short_against} тЖТ {new_rsi_short_against}")
            
            logger.info(f"[SAVE_CONFIG] тЬЕ ╨Ъ╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤П ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨░ ╨▓ ╨┐╨░╨╝╤П╤В╨╕ ╨╕╨╖ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╜╤Л╤Е ╨┤╨░╨╜╨╜╤Л╤Е!")
            if new_rsi_long_with is not None and new_rsi_short_with is not None:
                logger.info(f"[SAVE_CONFIG] ЁЯУК ╨в╨╡╨║╤Г╤Й╨╕╨╡ RSI exit ╨┐╨╛╤А╨╛╨│╨╕: LONG(with)={new_rsi_long_with}, LONG(against)={new_rsi_long_against}, SHORT(with)={new_rsi_short_with}, SHORT(against)={new_rsi_short_against}")
            else:
                logger.error(f"[SAVE_CONFIG] тЭМ ╨Э╨Х╨Ъ╨Ю╨в╨Ю╨а╨л╨Х RSI exit ╨┐╨╛╤А╨╛╨│╨╕ ╨╛╤В╤Б╤Г╤В╤Б╤В╨▓╤Г╤О╤В ╨▓ ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╨╕!")
            
            # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Х╤Б╨╗╨╕ ╤Б╨╛╤Е╤А╨░╨╜╤П╨╗╤Б╤П system_timeframe, ╤Б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨╡╨│╨╛ ╨▓ ╨С╨Ф ╨Я╨Х╨а╨Х╨Ф ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╖╨║╨╛╨╣ ╨╝╨╛╨┤╤Г╨╗╤П
            if 'system_timeframe' in config_data:
                try:
                    from bot_engine.bots_database import get_bots_database
                    from bot_engine.bot_config import set_current_timeframe
                    db = get_bots_database()
                    new_timeframe = config_data['system_timeframe']
                    db.save_timeframe(new_timeframe)
                    set_current_timeframe(new_timeframe)
                    logger.info(f"[SAVE_CONFIG] тЬЕ ╨в╨░╨╣╨╝╤Д╤А╨╡╨╣╨╝ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜ ╨▓ ╨С╨Ф ╨┐╨╡╤А╨╡╨┤ ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╖╨║╨╛╨╣ ╨╝╨╛╨┤╤Г╨╗╤П: {new_timeframe}")
                except Exception as tf_save_err:
                    logger.warning(f"[SAVE_CONFIG] тЪая╕П ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╤Б╨╛╤Е╤А╨░╨╜╨╕╤В╤М ╤В╨░╨╣╨╝╤Д╤А╨╡╨╣╨╝ ╨▓ ╨С╨Ф: {tf_save_err}")
            
            # тЬЕ ╨Я╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╨╝╨╛╨┤╤Г╨╗╤М bot_config ╨╕ ╨╛╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤О ╨╕╨╖ ╨╜╨╡╨│╨╛
            try:
                if 'bot_engine.bot_config' in sys.modules:
                    pass
                    
                    # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤В╨░╨╣╨╝╤Д╤А╨╡╨╣╨╝ ╨╕╨╖ ╨С╨Ф ╨┐╨╡╤А╨╡╨┤ ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╖╨║╨╛╨╣
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
                    
                    # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Т╨╛╤Б╤Б╤В╨░╨╜╨░╨▓╨╗╨╕╨▓╨░╨╡╨╝ ╤В╨░╨╣╨╝╤Д╤А╨╡╨╣╨╝ ╨╕╨╖ ╨С╨Ф ╨┐╨╛╤Б╨╗╨╡ ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╖╨║╨╕
                    if saved_timeframe_from_db:
                        try:
                            from bot_engine.bot_config import set_current_timeframe
                            set_current_timeframe(saved_timeframe_from_db)
                            logger.info(f"[SAVE_CONFIG] тЬЕ ╨в╨░╨╣╨╝╤Д╤А╨╡╨╣╨╝ ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜ ╨╕╨╖ ╨С╨Ф ╨┐╨╛╤Б╨╗╨╡ ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╖╨║╨╕: {saved_timeframe_from_db}")
                        except Exception as tf_restore_err:
                            logger.warning(f"[SAVE_CONFIG] тЪая╕П ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╕╤В╤М ╤В╨░╨╣╨╝╤Д╤А╨╡╨╣╨╝: {tf_restore_err}")
                    
                    # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Х╨б╨Ъ╨Ш ╨Т╨Р╨Ц╨Э╨Ю: ╨Я╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤О ╨╕╨╖ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╜╨╛╨│╨╛ bot_config.py
                    # ╨н╤В╨╛ ╨╜╤Г╨╢╨╜╨╛, ╤З╤В╨╛╨▒╤Л ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П ╤Б╤А╨░╨╖╤Г ╨▒╤А╨░╨╗╨╕╤Б╤М ╨╕╨╖ ╤Д╨░╨╣╨╗╨░, ╨░ ╨╜╨╡ ╨╕╨╖ ╤Б╤В╨░╤А╨╛╨╣ ╨┐╨░╨╝╤П╤В╨╕
                    from bots_modules.imports_and_globals import load_auto_bot_config
                    
                    # тЬЕ ╨б╨С╨а╨Р╨б╨л╨Т╨Р╨Х╨Ь ╨║╤Н╤И ╨▓╤А╨╡╨╝╨╡╨╜╨╕ ╨╝╨╛╨┤╨╕╤Д╨╕╨║╨░╤Ж╨╕╨╕ ╤Д╨░╨╣╨╗╨░, ╤З╤В╨╛╨▒╤Л ╨┐╤А╨╕ ╤Б╨╗╨╡╨┤╤Г╤О╤Й╨╡╨╝ ╨▓╤Л╨╖╨╛╨▓╨╡ ╨╝╨╛╨┤╤Г╨╗╤М ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╖╨╕╨╗╤Б╤П
                    if hasattr(load_auto_bot_config, '_last_mtime'):
                        load_auto_bot_config._last_mtime = 0
                    
                    # тЬЕ ╨Э╨Х ╤Б╨▒╤А╨░╤Б╤Л╨▓╨░╨╡╨╝ ╤Д╨╗╨░╨│ ╨╗╨╛╨│╨╕╤А╨╛╨▓╨░╨╜╨╕╤П leverage - ╨╕╨╜╨░╤З╨╡ ╨▒╤Г╨┤╨╡╤В ╤Б╨┐╨░╨╝ ╨┐╤А╨╕ ╨║╨░╨╢╨┤╨╛╨╣ ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╖╨║╨╡
                    # ╨д╨╗╨░╨│ _leverage_logged ╨╛╤Б╤В╨░╨╡╤В╤Б╤П, ╤З╤В╨╛╨▒╤Л ╨╜╨╡ ╨╗╨╛╨│╨╕╤А╨╛╨▓╨░╤В╤М leverage ╨┐╤А╨╕ ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╖╨║╨╡ ╨┐╨╛╤Б╨╗╨╡ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П
                    
                    load_auto_bot_config()
                    logger.info(f"[SAVE_CONFIG] тЬЕ ╨Ъ╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤П ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╢╨╡╨╜╨░ ╨╕╨╖ bot_config.py ╨┐╨╛╤Б╨╗╨╡ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П")
            except Exception as reload_error:
                logger.warning(f"[SAVE_CONFIG] тЪая╕П ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╖╨╕╤В╤М ╨╝╨╛╨┤╤Г╨╗╤М (╨╜╨╡ ╨║╤А╨╕╤В╨╕╤З╨╜╨╛): {reload_error}")
        
        return success
        
    except Exception as e:
        logger.error(f"[SAVE_CONFIG] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╨╕ ╨░╨▓╤В╨╛╨▒╨╛╤В╨░: {e}")
        return False

# тЭМ ╨Ю╨в╨Ъ╨Ы╨о╨з╨Х╨Э╨Ю: optimal_ema ╨┐╨╡╤А╨╡╨╝╨╡╤Й╨╡╨╜ ╨▓ backup (EMA ╤Д╨╕╨╗╤М╤В╤А ╤Г╨▒╤А╨░╨╜)
# def save_optimal_ema_periods():
#     """╨б╨╛╤Е╤А╨░╨╜╤П╨╡╤В ╨╛╨┐╤В╨╕╨╝╨░╨╗╤М╨╜╤Л╨╡ EMA ╨┐╨╡╤А╨╕╨╛╨┤╤Л"""
#     return True  # ╨Ч╨░╨│╨╗╤Г╤И╨║╨░

def load_bots_state():
    """╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╤В ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨▒╨╛╤В╨╛╨▓ ╨╕╨╖ ╨С╨Ф"""
    try:
        logger.info(f" ЁЯУВ ╨Ч╨░╨│╤А╤Г╨╖╨║╨░ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╤П ╨▒╨╛╤В╨╛╨▓ ╨╕╨╖ ╨С╨Ф...")
        
        # тЬЕ ╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╨╕╨╖ ╨С╨Ф ╤З╨╡╤А╨╡╨╖ storage.py
        state_data = storage_load_bots_state()
        
        if not state_data:
            logger.info(f" ЁЯУБ ╨б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨▒╨╛╤В╨╛╨▓ ╨▓ ╨С╨Ф ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜╨╛, ╨╜╨░╤З╨╕╨╜╨░╨╡╨╝ ╤Б ╨┐╤Г╤Б╤В╨╛╨│╨╛ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╤П")
            return False
        
        version = state_data.get('version', '1.0')
        last_saved = state_data.get('last_saved', '╨╜╨╡╨╕╨╖╨▓╨╡╤Б╤В╨╜╨╛')
        
        logger.info(f" ЁЯУК ╨Т╨╡╤А╤Б╨╕╤П ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╤П: {version}, ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╡╨╡ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╨╡: {last_saved}")
        
        # тЬЕ ╨Ъ╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤П Auto Bot ╨╜╨╕╨║╨╛╨│╨┤╨░ ╨╜╨╡ ╨▒╨╡╤А╤С╤В╤Б╤П ╨╕╨╖ ╨С╨Ф
        # ╨Э╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╨╖╨░╨│╤А╤Г╨╢╨░╤О╤В╤Б╤П ╤В╨╛╨╗╤М╨║╨╛ ╨╕╨╖ bot_engine/bot_config.py
        
        logger.info(f" тЪЩя╕П ╨Ъ╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤П Auto Bot ╨Э╨Х ╨╖╨░╨│╤А╤Г╨╢╨░╨╡╤В╤Б╤П ╨╕╨╖ ╨С╨Ф")
        logger.info(f" ЁЯТб ╨Ъ╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤П ╨╖╨░╨│╤А╤Г╨╢╨░╨╡╤В╤Б╤П ╤В╨╛╨╗╤М╨║╨╛ ╨╕╨╖ bot_engine/bot_config.py")
        
        # ╨Т╨╛╤Б╤Б╤В╨░╨╜╨░╨▓╨╗╨╕╨▓╨░╨╡╨╝ ╨▒╨╛╤В╨╛╨▓
        restored_bots = 0
        failed_bots = 0
        
        if 'bots' in state_data:
            with bots_data_lock:
                for symbol, bot_data in state_data['bots'].items():
                    try:
                        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨▓╨░╨╗╨╕╨┤╨╜╨╛╤Б╤В╤М ╨┤╨░╨╜╨╜╤Л╤Е ╨▒╨╛╤В╨░
                        if not isinstance(bot_data, dict) or 'status' not in bot_data:
                            logger.warning(f" тЪая╕П ╨Э╨╡╨║╨╛╤А╤А╨╡╨║╤В╨╜╤Л╨╡ ╨┤╨░╨╜╨╜╤Л╨╡ ╨▒╨╛╤В╨░ {symbol}, ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝")
                            failed_bots += 1
                            continue
                        
                        bot_status = bot_data.get('status', 'UNKNOWN')
                        
                        # ╨Т╨Р╨Ц╨Э╨Ю: ╨а╨░╨╜╤М╤И╨╡ ╨▒╨╛╤В╤Л ╤Б╨╛ ╤Б╤В╨░╤В╤Г╤Б╨╛╨╝ in_position_* ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╗╨╕╤Б╤М ╨╜╨░ ╤Б╤В╨░╤А╤В╨╡ ╨╕ ╨╛╨╢╨╕╨┤╨░╨╗╨╕
                        # ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П ╤З╨╡╤А╨╡╨╖ sync_bots_with_exchange(). ╨Э╨░ ╨┐╤А╨░╨║╤В╨╕╨║╨╡ ╤Н╤В╨╛ ╨┐╤А╨╕╨▓╨╛╨┤╨╕╨╗╨╛ ╨║ ╤В╨╛╨╝╤Г,
                        # ╤З╤В╨╛ "╨▒╨╛╤В╤Л ╨╕╨╖ ╤А╤Г╤З╨╜╤Л╤Е ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣" ╨╕╤Б╤З╨╡╨╖╨░╨╗╨╕ ╨┐╨╛╤Б╨╗╨╡ ╨┐╨╡╤А╨╡╨╖╨░╨┐╤Г╤Б╨║╨░ (╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╤Б╨╜╨╛╨▓╨░ ╨▓╤Л╨│╨╗╤П╨┤╨╡╨╗╨░ ╤А╤Г╤З╨╜╨╛╨╣),
                        # ╨╡╤Б╨╗╨╕ ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨░╤Ж╨╕╤П ╨╜╨╡ ╤Г╤Б╨┐╨╡╨▓╨░╨╗╨░/╨╜╨╡ ╨▓╤Л╨┐╨╛╨╗╨╜╤П╨╗╨░╤Б╤М ╤Б╤А╨░╨╖╤Г.
                        #
                        # ╨а╨╡╤И╨╡╨╜╨╕╨╡: ╨╖╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╤В╨░╨║╨╕╤Е ╨▒╨╛╤В╨╛╨▓ ╨╕╨╖ ╨С╨Ф, ╨╜╨╛ ╨┐╨╛╨╝╨╡╤З╨░╨╡╨╝ ╨╕╤Е ╤Д╨╗╨░╨│╨╛╨╝ needs_exchange_sync.
                        # ╨Ф╨░╨╗╤М╤И╨╡ sync_bots_with_exchange() ╨▓╤Б╤С ╤А╨░╨▓╨╜╨╛ ╨┐╤А╨╕╨▓╨╡╨┤╤С╤В ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨║ ╤А╨╡╨░╨╗╤М╨╜╤Л╨╝ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П╨╝ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
                        # (╨╕ ╤Г╨┤╨░╨╗╨╕╤В ╨▒╨╛╤В╨░, ╨╡╤Б╨╗╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Г╨╢╨╡ ╨╜╨╡╤В).
                        if bot_status in ['in_position_long', 'in_position_short']:
                            bot_data['needs_exchange_sync'] = True
                            bots_data['bots'][symbol] = bot_data
                            restored_bots += 1
                            logger.info(f" ЁЯдЦ ╨Т╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜ ╨▒╨╛╤В {symbol}: ╤Б╤В╨░╤В╤Г╤Б={bot_status} (╨╛╨╢╨╕╨┤╨░╨╡╤В sync)")
                            continue
                        
                        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Э╨Х ╨╖╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╨▒╨╛╤В╨╛╨▓ ╨▓ ╤Б╤В╨░╤В╤Г╤Б╨╡ IDLE - ╨╛╨╜╨╕ ╨╜╨╡ ╨╕╨╝╨╡╤О╤В ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣!
                        # ╨С╨╛╤В╤Л ╨▓ ╤Б╤В╨░╤В╤Г╤Б╨╡ IDLE ╨┤╨╛╨╗╨╢╨╜╤Л ╤Г╨┤╨░╨╗╤П╤В╤М╤Б╤П ╨┐╤А╨╕ ╨╖╨░╨║╤А╤Л╤В╨╕╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣, ╨░ ╨╜╨╡ ╨╛╤Б╤В╨░╨▓╨░╤В╤М╤Б╤П ╨▓ ╨С╨Ф.
                        if bot_status == 'idle':
                            pass  # ╨▒╨╛╤В ╨▒╨╡╨╖ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨┤╨╛╨╗╨╢╨╡╨╜ ╨▒╤Л╤В╤М ╤Г╨┤╨░╨╗╨╡╨╜
                            continue
                        
                        # ╨Т╨Р╨Ц╨Э╨Ю: ╨Э╨Х ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╖╤А╨╡╨╗╨╛╤Б╤В╤М ╨┐╤А╨╕ ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╕!
                        # ╨Я╤А╨╕╤З╨╕╨╜╤Л:
                        # 1. ╨С╨╕╤А╨╢╨░ ╨╡╤Й╨╡ ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░ (╨╜╨╡╤В ╨┤╨░╨╜╨╜╤Л╤Е ╤Б╨▓╨╡╤З╨╡╨╣)
                        # 2. ╨Х╤Б╨╗╨╕ ╨▒╨╛╤В ╨▒╤Л╨╗ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜ - ╨╛╨╜ ╤Г╨╢╨╡ ╨┐╤А╨╛╤И╨╡╨╗ ╨┐╤А╨╛╨▓╨╡╤А╨║╤Г ╨╖╤А╨╡╨╗╨╛╤Б╤В╨╕ ╨┐╤А╨╕ ╤Б╨╛╨╖╨┤╨░╨╜╨╕╨╕
                        # 3. ╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╨╖╤А╨╡╨╗╨╛╤Б╤В╨╕ ╨▒╤Г╨┤╨╡╤В ╨▓╤Л╨┐╨╛╨╗╨╜╨╡╨╜╨░ ╨┐╨╛╨╖╨╢╨╡ ╨┐╤А╨╕ ╨╛╨▒╤А╨░╨▒╨╛╤В╨║╨╡ ╤Б╨╕╨│╨╜╨░╨╗╨╛╨▓
                        
                        # ╨Т╨╛╤Б╤Б╤В╨░╨╜╨░╨▓╨╗╨╕╨▓╨░╨╡╨╝ ╨▒╨╛╤В╨░
                        bots_data['bots'][symbol] = bot_data
                        restored_bots += 1
                        
                        logger.info(f" ЁЯдЦ ╨Т╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜ ╨▒╨╛╤В {symbol}: ╤Б╤В╨░╤В╤Г╤Б={bot_status}")
                        
                    except Exception as e:
                        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П ╨▒╨╛╤В╨░ {symbol}: {e}")
                        failed_bots += 1
        
        logger.info(f" тЬЕ ╨Т╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨╛ ╨▒╨╛╤В╨╛╨▓: {restored_bots}, ╨╛╤И╨╕╨▒╨╛╨║: {failed_bots}")
        
        return restored_bots > 0
        
    except Exception as e:
        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╖╨░╨│╤А╤Г╨╖╨║╨╕ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╤П ╨╕╨╖ ╨С╨Ф: {e}")
        return False

def load_delisted_coins():
    """╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╤В ╤Б╨┐╨╕╤Б╨╛╨║ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╤Е ╨╝╨╛╨╜╨╡╤В ╨╕╨╖ ╨С╨Ф"""
    try:
        # тЬЕ ╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╨╕╨╖ ╨С╨Ф ╤З╨╡╤А╨╡╨╖ storage.py
        delisted_list = storage_load_delisted_coins()
        
        # тЬЕ ╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╨╝ last_scan ╨╕╨╖ process_state
        last_scan = None
        try:
            from bots_modules.imports_and_globals import process_state
            if 'delisting_scan' in process_state:
                last_scan = process_state['delisting_scan'].get('last_scan')
        except Exception as state_error:
            pass
        
        # ╨Я╤А╨╡╨╛╨▒╤А╨░╨╖╤Г╨╡╨╝ ╤Б╨┐╨╕╤Б╨╛╨║ ╨▓ ╤Д╨╛╤А╨╝╨░╤В ╤Б╨╗╨╛╨▓╨░╤А╤П ╨┤╨╗╤П ╨╛╨▒╤А╨░╤В╨╜╨╛╨╣ ╤Б╨╛╨▓╨╝╨╡╤Б╤В╨╕╨╝╨╛╤Б╤В╨╕
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
        
        # ╨Х╤Б╨╗╨╕ ╨┤╨░╨╜╨╜╤Л╤Е ╨╜╨╡╤В, ╨▓╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╨╝ ╨┤╨╡╤Д╨╛╨╗╤В
        return {"delisted_coins": {}, "last_scan": last_scan, "scan_enabled": True}
        
    except Exception as e:
        logger.warning(f"╨Ю╤И╨╕╨▒╨║╨░ ╨╖╨░╨│╤А╤Г╨╖╨║╨╕ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╤Е ╨╝╨╛╨╜╨╡╤В ╨╕╨╖ ╨С╨Ф: {e}, ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨┤╨╡╤Д╨╛╨╗╤В╨╜╤Л╨╡ ╨┤╨░╨╜╨╜╤Л╨╡")
        return {"delisted_coins": {}, "last_scan": None, "scan_enabled": True}

def add_symbol_to_delisted(symbol: str, reason: str = "Delisting detected"):
    """╨Ф╨╛╨▒╨░╨▓╨╗╤П╨╡╤В ╤Б╨╕╨╝╨▓╨╛╨╗ ╨▓ ╤Б╨┐╨╕╤Б╨╛╨║ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╤Е (╨╜╨░╨┐╤А╨╕╨╝╨╡╤А, ╨┐╤А╨╕ ╨╛╤И╨╕╨▒╨║╨╡ 30228 ╨┐╤А╨╕ ╨╛╤В╨║╤А╤Л╤В╨╕╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕)."""
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
        logger.warning(f"ЁЯЪи ╨Ф╨╛╨▒╨░╨▓╨╗╨╡╨╜ ╨▓ ╤Б╨┐╨╕╤Б╨╛╨║ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨░: {sym} тАФ {reason}")
        return True
    except Exception as e:
        logger.error(f"╨Ю╤И╨╕╨▒╨║╨░ ╨┤╨╛╨▒╨░╨▓╨╗╨╡╨╜╨╕╤П {symbol} ╨▓ ╤Б╨┐╨╕╤Б╨╛╨║ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨░: {e}")
        return False


def save_delisted_coins(data):
    """╨б╨╛╤Е╤А╨░╨╜╤П╨╡╤В ╤Б╨┐╨╕╤Б╨╛╨║ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╤Е ╨╝╨╛╨╜╨╡╤В ╨▓ ╨С╨Ф"""
    try:
        # ╨Я╤А╨╡╨╛╨▒╤А╨░╨╖╤Г╨╡╨╝ ╤Б╨╗╨╛╨▓╨░╤А╤М ╨▓ ╤Б╨┐╨╕╤Б╨╛╨║ ╤Б╨╕╨╝╨▓╨╛╨╗╨╛╨▓ ╨┤╨╗╤П ╨С╨Ф
        # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ю: ╨С╨Ф ╨╛╨╢╨╕╨┤╨░╨╡╤В ╤Б╨┐╨╕╤Б╨╛╨║ ╤Б╤В╤А╨╛╨║ (╤Б╨╕╨╝╨▓╨╛╨╗╨╛╨▓), ╨░ ╨╜╨╡ ╤Б╨┐╨╕╤Б╨╛╨║ ╤Б╨╗╨╛╨▓╨░╤А╨╡╨╣
        delisted_coins_dict = data.get("delisted_coins", {}) if isinstance(data, dict) else {}
        delisted_list = []
        
        for symbol, coin_data in delisted_coins_dict.items():
            # ╨Ш╨╖╨▓╨╗╨╡╨║╨░╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╤Б╨╕╨╝╨▓╨╛╨╗ (╤Б╤В╤А╨╛╨║╤Г) ╨┤╨╗╤П ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П ╨▓ ╨С╨Ф
            # ╨Ф╨╛╨┐╨╛╨╗╨╜╨╕╤В╨╡╨╗╤М╨╜╤Л╨╡ ╨┤╨░╨╜╨╜╤Л╨╡ (status, reason ╨╕ ╤В.╨┤.) ╨╜╨╡ ╤Б╨╛╤Е╤А╨░╨╜╤П╤О╤В╤Б╤П ╨▓ ╤В╨░╨▒╨╗╨╕╤Ж╤Г delisted
            if isinstance(symbol, str):
                delisted_list.append(symbol)
            elif isinstance(coin_data, dict) and 'symbol' in coin_data:
                delisted_list.append(coin_data['symbol'])
            else:
                # Fallback: ╨┐╤Л╤В╨░╨╡╨╝╤Б╤П ╨╕╨╖╨▓╨╗╨╡╤З╤М ╤Б╨╕╨╝╨▓╨╛╨╗ ╨╕╨╖ ╨║╨╗╤О╤З╨░
                delisted_list.append(str(symbol))
        
        # тЬЕ ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨▓ ╨С╨Ф ╤З╨╡╤А╨╡╨╖ storage.py
        if storage_save_delisted_coins(delisted_list):
            logger.info(f"тЬЕ ╨Ю╨▒╨╜╨╛╨▓╨╗╨╡╨╜╤Л ╨┤╨╡╨╗╨╕╤Б╤В╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л ╨▓ ╨С╨Ф ({len(delisted_list)} ╨╝╨╛╨╜╨╡╤В)")
            return True
        return False
    except Exception as e:
        logger.error(f"╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П ╨┤╨╡╨╗╨╕╤Б╤В╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╤Е ╨╝╨╛╨╜╨╡╤В ╨▓ ╨С╨Ф: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def scan_all_coins_for_delisting():
    """╨б╨║╨░╨╜╨╕╤А╤Г╨╡╤В ╨▓╤Б╨╡ ╨╝╨╛╨╜╨╡╤В╤Л ╨╜╨░ ╨┐╤А╨╡╨┤╨╝╨╡╤В ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨░ ╨╕ ╨╛╨▒╨╜╨╛╨▓╨╗╤П╨╡╤В delisted.json"""
    try:
        logger.info("ЁЯФН ╨б╨║╨░╨╜╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡ ╨▓╤Б╨╡╤Е ╨╝╨╛╨╜╨╡╤В ╨╜╨░ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│...")
        
        # ╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╤В╨╡╨║╤Г╤Й╨╕╨╡ ╨┤╨░╨╜╨╜╤Л╨╡
        delisted_data = load_delisted_coins()
        
        if not delisted_data.get('scan_enabled', True):
            logger.info("тП╕я╕П ╨б╨║╨░╨╜╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡ ╨╛╤В╨║╨╗╤О╤З╨╡╨╜╨╛ ╨▓ ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╨╕")
            return
        
        exchange_obj = get_exchange()
        if not exchange_obj:
            logger.error("тЭМ Exchange ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜")
            return
        
        # ╨Ш╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╤Г╨╡╨╝ ╤Б╤В╤А╤Г╨║╤В╤Г╤А╤Г ╨╡╤Б╨╗╨╕ ╨╡╤С ╨╜╨╡╤В
        if 'delisted_coins' not in delisted_data:
            delisted_data['delisted_coins'] = {}
        
        new_delisted_count = 0
        
        # тЬЕ ╨Ь╨Х╨в╨Ю╨Ф: ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨Т╨б╨Х ╨╕╨╜╤Б╤В╤А╤Г╨╝╨╡╨╜╤В╤Л ╤Б╤А╨░╨╖╤Г ╤З╨╡╤А╨╡╨╖ API (╨▒╨╡╨╖ ╤Д╨╕╨╗╤М╤В╤А╨░ ╨┐╨╛ ╤Б╤В╨░╤В╤Г╤Б╤Г)
        # ╨н╤В╨╛ ╨╜╨░╨╝╨╜╨╛╨│╨╛ ╨▒╤Л╤Б╤В╤А╨╡╨╡, ╤З╨╡╨╝ ╨┐╤А╨╛╨▓╨╡╤А╤П╤В╤М ╨║╨░╨╢╨┤╤Г╤О ╨╝╨╛╨╜╨╡╤В╤Г ╨╛╤В╨┤╨╡╨╗╤М╨╜╨╛!
        # тЪая╕П ╨Т╨Р╨Ц╨Э╨Ю: ╨Э╨╡ ╤Г╨║╨░╨╖╤Л╨▓╨░╨╡╨╝ ╨┐╨░╤А╨░╨╝╨╡╤В╤А status, ╤З╤В╨╛╨▒╤Л ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╨Т╨б╨Х ╨╕╨╜╤Б╤В╤А╤Г╨╝╨╡╨╜╤В╤Л, ╨▓╨║╨╗╤О╤З╨░╤П Closed/Delivering
        if hasattr(exchange_obj, 'client') and hasattr(exchange_obj.client, 'get_instruments_info'):
            try:
                logger.info("ЁЯУК ╨Ч╨░╨┐╤А╨░╤И╨╕╨▓╨░╨╡╨╝ ╨▓╤Б╨╡ ╨╕╨╜╤Б╤В╤А╤Г╨╝╨╡╨╜╤В╤Л ╤Б ╨▒╨╕╤А╨╢╨╕ (╨▓╨║╨╗╤О╤З╨░╤П ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╨╡)...")
                
                all_instruments = []
                cursor = None
                page = 0
                max_pages = 10  # ╨Ю╨│╤А╨░╨╜╨╕╤З╨╡╨╜╨╕╨╡ ╨╜╨░ ╨║╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╛ ╤Б╤В╤А╨░╨╜╨╕╤Ж ╨┤╨╗╤П ╨▒╨╡╨╖╨╛╨┐╨░╤Б╨╜╨╛╤Б╤В╨╕
                
                # тЬЕ ╨Ю╨С╨а╨Р╨С╨Ю╨в╨Ъ╨Р ╨Я╨Р╨У╨Ш╨Э╨Р╨ж╨Ш╨Ш: ╨Ч╨░╨┐╤А╨░╤И╨╕╨▓╨░╨╡╨╝ ╨▓╤Б╨╡ ╤Б╤В╤А╨░╨╜╨╕╤Ж╤Л ╨╕╨╜╤Б╤В╤А╤Г╨╝╨╡╨╜╤В╨╛╨▓
                while page < max_pages:
                    page += 1
                    try:
                        # ╨Ч╨░╨┐╤А╨░╤И╨╕╨▓╨░╨╡╨╝ ╨Т╨б╨Х ╨╕╨╜╤Б╤В╤А╤Г╨╝╨╡╨╜╤В╤Л ╨▒╨╡╨╖ ╤Д╨╕╨╗╤М╤В╤А╨░ ╨┐╨╛ ╤Б╤В╨░╤В╤Г╤Б╤Г (╨╜╨╡ ╤Г╨║╨░╨╖╤Л╨▓╨░╨╡╨╝ status)
                        # ╨н╤В╨╛ ╤Б╨╛╨╛╤В╨▓╨╡╤В╤Б╤В╨▓╤Г╨╡╤В API Bybit v5 - ╨╝╨╛╨╢╨╜╨╛ ╨╖╨░╨┐╤А╨╛╤Б╨╕╤В╤М ╨▓╤Б╨╡ ╨╕╨╜╤Б╤В╤А╤Г╨╝╨╡╨╜╤В╤Л ╨▒╨╡╨╖ symbol
                        params = {
                            'category': 'linear',
                            'limit': 1000  # ╨Ь╨░╨║╤Б╨╕╨╝╤Г╨╝ ╨╕╨╜╤Б╤В╤А╤Г╨╝╨╡╨╜╤В╨╛╨▓ ╨╖╨░ ╨╛╨┤╨╕╨╜ ╨╖╨░╨┐╤А╨╛╤Б (Bybit API ╨┐╨╛╨┤╨┤╨╡╤А╨╢╨╕╨▓╨░╨╡╤В ╨┤╨╛ 1000)
                        }
                        
                        # ╨Ф╨╛╨▒╨░╨▓╨╗╤П╨╡╨╝ cursor ╨┤╨╗╤П ╨┐╨░╨│╨╕╨╜╨░╤Ж╨╕╨╕, ╨╡╤Б╨╗╨╕ ╨╛╨╜ ╨╡╤Б╤В╤М
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
                            logger.info(f"ЁЯУК ╨б╤В╤А╨░╨╜╨╕╤Ж╨░ {page}: ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╛ {len(instruments_list)} ╨╕╨╜╤Б╤В╤А╤Г╨╝╨╡╨╜╤В╨╛╨▓ (╨▓╤Б╨╡╨│╨╛: {len(all_instruments)})")
                            
                            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨╡╤Б╤В╤М ╨╗╨╕ ╤Б╨╗╨╡╨┤╤Г╤О╤Й╨░╤П ╤Б╤В╤А╨░╨╜╨╕╤Ж╨░
                            next_page_cursor = result.get('nextPageCursor')
                            if not next_page_cursor or next_page_cursor == '':
                                break
                            
                            cursor = next_page_cursor
                        else:
                            error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                            logger.warning(f"тЪая╕П ╨б╤В╤А╨░╨╜╨╕╤Ж╨░ {page}: ╨╛╤И╨╕╨▒╨║╨░ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╨╕╨╜╤Б╤В╤А╤Г╨╝╨╡╨╜╤В╨╛╨▓: {error_msg}")
                            break
                            
                    except Exception as page_error:
                        logger.error(f"тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╕ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╨╕ ╤Б╤В╤А╨░╨╜╨╕╤Ж╤Л {page}: {page_error}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        break
                
                logger.info(f"ЁЯУК ╨Т╤Б╨╡╨│╨╛ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╛ {len(all_instruments)} ╨╕╨╜╤Б╤В╤А╤Г╨╝╨╡╨╜╤В╨╛╨▓ ╤Б ╨▒╨╕╤А╨╢╨╕")
                
                # ╨д╨╕╨╗╤М╤В╤А╤Г╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ USDT ╨┐╨░╤А╤Л ╤Б ╤Б╤В╨░╤В╤Г╤Б╨╛╨╝ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨░
                delisted_found = 0
                for instrument in all_instruments:
                    symbol = instrument.get('symbol', '')
                    if not symbol.endswith('USDT'):
                        continue
                    
                    coin_symbol = symbol.replace('USDT', '')
                    status = instrument.get('status', 'Unknown')
                    
                    # ╨Я╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╨╡╤Б╨╗╨╕ ╤Г╨╢╨╡ ╨▓ ╤Б╨┐╨╕╤Б╨║╨╡ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╤Е
                    if coin_symbol in delisted_data['delisted_coins']:
                        continue
                    
                    # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Б╤В╨░╤В╤Г╤Б ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨░ (Closed ╨╕╨╗╨╕ Delivering)
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
                        logger.warning(f"ЁЯЪи ╨Э╨Ю╨Т╨л╨Щ ╨Ф╨Х╨Ы╨Ш╨б╨в╨Ш╨Э╨У: {coin_symbol} - {status}")
                
                if delisted_found == 0:
                    logger.info("тЬЕ ╨Ф╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╤Е ╨╝╨╛╨╜╨╡╤В ╨╜╨╡ ╨╛╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜╨╛ (╨╕╨╗╨╕ ╨▓╤Б╨╡ ╤Г╨╢╨╡ ╨▓ ╤Б╨┐╨╕╤Б╨║╨╡)")
                else:
                    logger.info(f"ЁЯЪи ╨Ю╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜╨╛ {delisted_found} ╨╜╨╛╨▓╤Л╤Е ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╤Е ╨╝╨╛╨╜╨╡╤В")
                    
            except Exception as bulk_scan_error:
                logger.error(f"тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╝╨░╤Б╤Б╨╛╨▓╨╛╨│╨╛ ╤Б╨║╨░╨╜╨╕╤А╨╛╨▓╨░╨╜╨╕╤П ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨░: {bulk_scan_error}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.warning("тЪая╕П ╨Ь╨░╤Б╤Б╨╛╨▓╨╛╨╡ ╤Б╨║╨░╨╜╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡ ╨╜╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М, ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│ ╨▒╤Г╨┤╨╡╤В ╨╛╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜ ╨┐╤А╨╕ ╨┐╨╛╨┐╤Л╤В╨║╨╡ ╤А╨░╨╖╨╝╨╡╤Й╨╡╨╜╨╕╤П ╨╛╤А╨┤╨╡╤А╨╛╨▓")
        
        # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨▓╤А╨╡╨╝╤П ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╡╨│╨╛ ╤Б╨║╨░╨╜╨╕╤А╨╛╨▓╨░╨╜╨╕╤П
        last_scan_time = datetime.now().isoformat()
        delisted_data['last_scan'] = last_scan_time
        
        # тЬЕ ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ last_scan ╨▓ process_state ╨┤╨╗╤П ╨┐╨╡╤А╤Б╨╕╤Б╤В╨╡╨╜╤В╨╜╨╛╤Б╤В╨╕
        try:
            update_process_state('delisting_scan', {
                'last_scan': last_scan_time,
                'total_delisted': len(delisted_data['delisted_coins']),
                'new_delisted': new_delisted_count
            })
        except Exception as state_error:
            pass
        
        # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╜╤Л╨╡ ╨┤╨░╨╜╨╜╤Л╨╡
        if save_delisted_coins(delisted_data):
            logger.info(f"тЬЕ ╨б╨║╨░╨╜╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡ ╨╖╨░╨▓╨╡╤А╤И╨╡╨╜╨╛:")
            logger.info(f"   - ╨Э╨╛╨▓╤Л╤Е ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╤Е: {new_delisted_count}")
            logger.info(f"   - ╨Т╤Б╨╡╨│╨╛ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╤Е: {len(delisted_data['delisted_coins'])}")
        
    except Exception as e:
        logger.error(f"тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨║╨░╨╜╨╕╤А╨╛╨▓╨░╨╜╨╕╤П ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨░: {e}")

def check_delisting_emergency_close():
    """
    ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╤В ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│ ╨╕ ╨▓╤Л╨┐╨╛╨╗╨╜╤П╨╡╤В ╤Н╨║╤Б╤В╤А╨╡╨╜╨╜╨╛╨╡ ╨╖╨░╨║╤А╤Л╤В╨╕╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ (╤А╨░╨╖ ╨▓ 10 ╨╝╨╕╨╜╤Г╤В)
    тЪая╕П ╨Т╨Р╨Ц╨Э╨Ю: scan_all_coins_for_delisting() ╨▓╤Л╨╖╤Л╨▓╨░╨╡╤В╤Б╤П ╨╜╨╡ ╤З╨░╤Й╨╡ ╤А╨░╨╖╨░ ╨▓ ╤З╨░╤Б,
    ╤З╤В╨╛╨▒╤Л ╨╜╨╡ ╨┐╨╡╤А╨╡╨│╤А╤Г╨╢╨░╤В╤М API ╨╝╨░╤Б╤Б╨╛╨▓╤Л╨╝╨╕ ╨╖╨░╨┐╤А╨╛╤Б╨░╨╝╨╕
    """
    try:
        # ╨Ш╨╝╨┐╨╛╤А╤В╤Л ╨┤╨╗╤П ╤Н╨║╤Б╤В╤А╨╡╨╜╨╜╨╛╨│╨╛ ╨╖╨░╨║╤А╤Л╤В╨╕╤П ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣
        from bots_modules.bot_class import NewTradingBot
        from bots_modules.imports_and_globals import get_exchange
        
        # тЬЕ ╨б╨Э╨Р╨з╨Р╨Ы╨Р: ╨б╨║╨░╨╜╨╕╤А╤Г╨╡╨╝ ╨▓╤Б╨╡ ╨╝╨╛╨╜╨╡╤В╤Л ╨╜╨░ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│ (╤В╨╛╨╗╤М╨║╨╛ ╨╡╤Б╨╗╨╕ ╨┐╤А╨╛╤И╨╗╨╛ ╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ ╨▓╤А╨╡╨╝╨╡╨╜╨╕)
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨▓╤А╨╡╨╝╤П ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╡╨│╨╛ ╤Б╨║╨░╨╜╨╕╤А╨╛╨▓╨░╨╜╨╕╤П
        delisted_data = load_delisted_coins()
        last_scan_str = delisted_data.get('last_scan')
        
        should_scan = True
        if last_scan_str:
            try:
                from datetime import datetime
                last_scan_time = datetime.fromisoformat(last_scan_str)
                time_since_scan = (datetime.now() - last_scan_time).total_seconds()
                # ╨б╨║╨░╨╜╨╕╤А╤Г╨╡╨╝ ╨╜╨╡ ╤З╨░╤Й╨╡ ╤З╨╡╨╝ ╤А╨░╨╖ ╨▓ ╤З╨░╤Б (3600 ╤Б╨╡╨║╤Г╨╜╨┤), ╤З╤В╨╛╨▒╤Л ╨╜╨╡ ╨┐╨╡╤А╨╡╨│╤А╤Г╨╢╨░╤В╤М API
                if time_since_scan < 3600:
                    should_scan = False
                    pass
            except Exception as time_check_error:
                pass
        
        if should_scan:
            scan_all_coins_for_delisting()
        else:
            pass
        
        logger.info(f"ЁЯФН ╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨░ ╨┤╨╗╤П ╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓...")
        
        with bots_data_lock:
            bots_in_position = [
                (symbol, bot_data) for symbol, bot_data in bots_data['bots'].items()
                if bot_data.get('status') in ['in_position_long', 'in_position_short']
            ]
        
        if not bots_in_position:
            pass
            return True
        
        logger.info(f"ЁЯУК ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ {len(bots_in_position)} ╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓")
        
        delisting_closed_count = 0
        exchange_obj = get_exchange()
        
        if not exchange_obj:
            logger.error(f"тЭМ Exchange ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜")
            return False
        
        for symbol, bot_data in bots_in_position:
            try:
                # тЬЕ ╨Я╨а╨Ю╨Т╨Х╨а╨Ъ╨Р 1: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│ ╨╜╨░╨┐╤А╤П╨╝╤Г╤О ╨╕╨╖ delisted.json (╤Б╨░╨╝╤Л╨╣ ╨▒╤Л╤Б╤В╤А╤Л╨╣ ╤Б╨┐╨╛╤Б╨╛╨▒)
                is_delisting = False
                delisting_reason = ""
                
                delisted_data = load_delisted_coins()
                delisted_coins = delisted_data.get('delisted_coins', {})
                if symbol in delisted_coins:
                    is_delisting = True
                    delisting_info = delisted_coins[symbol]
                    delisting_reason = delisting_info.get('reason', 'Delisting detected')
                    logger.warning(f"ЁЯЪи ╨Ф╨Х╨Ы╨Ш╨б╨в╨Ш╨Э╨У ╨Ю╨С╨Э╨Р╨а╨г╨Ц╨Х╨Э ╨┤╨╗╤П {symbol} ╨▓ delisted.json: {delisting_reason}")
                
                # тЬЕ ╨Я╨а╨Ю╨Т╨Х╨а╨Ъ╨Р 2: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│ ╤З╨╡╤А╨╡╨╖ RSI ╨┤╨░╨╜╨╜╤Л╨╡ (fallback)
                if not is_delisting:
                    rsi_cache = get_rsi_cache()
                    if symbol in rsi_cache:
                        rsi_data = rsi_cache[symbol]
                        is_delisting = rsi_data.get('is_delisting', False) or rsi_data.get('trading_status') in ['Closed', 'Delivering']
                        if is_delisting:
                            delisting_reason = f"Delisting detected via RSI data (status: {rsi_data.get('trading_status', 'Unknown')})"
                            logger.warning(f"ЁЯЪи ╨Ф╨Х╨Ы╨Ш╨б╨в╨Ш╨Э╨У ╨Ю╨С╨Э╨Р╨а╨г╨Ц╨Х╨Э ╨┤╨╗╤П {symbol} ╤З╨╡╤А╨╡╨╖ RSI ╨┤╨░╨╜╨╜╤Л╨╡")
                
                # ╨Х╤Б╨╗╨╕ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│ ╨╛╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜ - ╨╖╨░╨║╤А╤Л╨▓╨░╨╡╨╝ ╨┐╨╛╨╖╨╕╤Ж╨╕╤О ╨╜╨╡╨╝╨╡╨┤╨╗╨╡╨╜╨╜╨╛
                if is_delisting:
                        logger.warning(f"ЁЯЪи ╨Ф╨Х╨Ы╨Ш╨б╨в╨Ш╨Э╨У ╨Ю╨С╨Э╨Р╨а╨г╨Ц╨Х╨Э ╨┤╨╗╤П {symbol}! ╨Ш╨╜╨╕╤Ж╨╕╨╕╤А╤Г╨╡╨╝ ╤Н╨║╤Б╤В╤А╨╡╨╜╨╜╨╛╨╡ ╨╖╨░╨║╤А╤Л╤В╨╕╨╡")
                        
                        bot_instance = NewTradingBot(symbol, bot_data, exchange_obj)
                        
                        # ╨Т╤Л╨┐╨╛╨╗╨╜╤П╨╡╨╝ ╤Н╨║╤Б╤В╤А╨╡╨╜╨╜╨╛╨╡ ╨╖╨░╨║╤А╤Л╤В╨╕╨╡
                        emergency_result = bot_instance.emergency_close_delisting()
                        
                        if emergency_result:
                            logger.warning(f"тЬЕ ╨н╨Ъ╨б╨в╨а╨Х╨Э╨Э╨Ю╨Х ╨Ч╨Р╨Ъ╨а╨л╨в╨Ш╨Х {symbol} ╨г╨б╨Я╨Х╨и╨Э╨Ю")
                            # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╤Б╤В╨░╤В╤Г╤Б ╨▒╨╛╤В╨░
                            with bots_data_lock:
                                if symbol in bots_data['bots']:
                                    bots_data['bots'][symbol]['status'] = 'idle'
                                    bots_data['bots'][symbol]['position_side'] = None
                                    bots_data['bots'][symbol]['entry_price'] = None
                                    bots_data['bots'][symbol]['unrealized_pnl'] = 0
                                    bots_data['bots'][symbol]['last_update'] = datetime.now().isoformat()
                            
                            delisting_closed_count += 1
                        else:
                            logger.error(f"тЭМ ╨н╨Ъ╨б╨в╨а╨Х╨Э╨Э╨Ю╨Х ╨Ч╨Р╨Ъ╨а╨л╨в╨Ш╨Х {symbol} ╨Э╨Х╨г╨Ф╨Р╨з╨Э╨Ю")
                            
            except Exception as e:
                logger.error(f"тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨░ ╨┤╨╗╤П {symbol}: {e}")
        
        if delisting_closed_count > 0:
            logger.warning(f"ЁЯЪи ╨н╨Ъ╨б╨в╨а╨Х╨Э╨Э╨Ю ╨Ч╨Р╨Ъ╨а╨л╨в╨Ю {delisting_closed_count} ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨╕╨╖-╨╖╨░ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨░!")
            # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨┐╨╛╤Б╨╗╨╡ ╤Н╨║╤Б╤В╤А╨╡╨╜╨╜╨╛╨│╨╛ ╨╖╨░╨║╤А╤Л╤В╨╕╤П
            save_bots_state()
        
        logger.info(f"тЬЕ ╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨░ ╨╖╨░╨▓╨╡╤А╤И╨╡╨╜╨░")
        return True
        
    except Exception as e:
        logger.error(f"тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨░: {e}")
        return False

def update_bots_cache_data():
    """╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╤В ╨║╤Н╤И╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╨╡ ╨┤╨░╨╜╨╜╤Л╨╡ ╨▒╨╛╤В╨╛╨▓ (╨║╨░╨║ background_update ╨▓ app.py)"""
    global bots_cache_data
    
    try:
        if not ensure_exchange_initialized():
            return False
        
        # ╨Я╨╛╨┤╨░╨▓╨╗╤П╨╡╨╝ ╤З╨░╤Б╤В╤Л╨╡ ╤Б╨╛╨╛╨▒╤Й╨╡╨╜╨╕╤П ╨╛╨▒ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╕ ╨║╤Н╤И╨░
        should_log, log_message = should_log_message(
            'cache_update', 
            "ЁЯФД ╨Ю╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╡ ╨║╤Н╤И╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╤Е ╨┤╨░╨╜╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓...",
            interval_seconds=300  # ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ ╤А╨░╨╖ ╨▓ 5 ╨╝╨╕╨╜╤Г╤В
        )
        if should_log:
            logger.info(f" {log_message}")
        
        # ╨Ф╨╛╨▒╨░╨▓╨╗╤П╨╡╨╝ ╤В╨░╨╣╨╝╨░╤Г╤В ╨┤╨╗╤П ╨┐╤А╨╡╨┤╨╛╤В╨▓╤А╨░╤Й╨╡╨╜╨╕╤П ╨╖╨░╨▓╨╕╤Б╨░╨╜╨╕╤П (Windows-╤Б╨╛╨▓╨╝╨╡╤Б╤В╨╕╨╝╤Л╨╣)
        import threading
        import time
        
        timeout_occurred = threading.Event()
        
        def timeout_worker():
            time.sleep(30)  # 30 ╤Б╨╡╨║╤Г╨╜╨┤ ╤В╨░╨╣╨╝╨░╤Г╤В
            timeout_occurred.set()
        
        timeout_thread = threading.Thread(target=timeout_worker, daemon=True)
        timeout_thread.start()
        
        # тЪб ╨Ю╨Я╨в╨Ш╨Ь╨Ш╨Ч╨Р╨ж╨Ш╨п: ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨┤╨░╨╜╨╜╤Л╨╡ ╨▒╨╛╤В╨╛╨▓ ╨▒╤Л╤Б╤В╤А╨╛ ╨▒╨╡╨╖ ╨╗╨╕╤И╨╜╨╕╤Е ╨╛╨┐╨╡╤А╨░╤Ж╨╕╨╣
        bots_list = []
        for symbol, bot_data in bots_data['bots'].items():
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤В╨░╨╣╨╝╨░╤Г╤В
            if timeout_occurred.is_set():
                logger.warning(" тЪая╕П ╨в╨░╨╣╨╝╨░╤Г╤В ╨┤╨╛╤Б╤В╨╕╨│╨╜╤Г╤В, ╨┐╤А╨╡╤А╤Л╨▓╨░╨╡╨╝ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╡")
                break
            
            # ╨Ф╨╛╨▒╨░╨▓╨╗╤П╨╡╨╝ RSI ╨┤╨░╨╜╨╜╤Л╨╡ ╨║ ╨▒╨╛╤В╤Г (╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨║╤Н╤И╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╨╡ ╨┤╨░╨╜╨╜╤Л╨╡)
            try:
                rsi_cache = get_rsi_cache()
                if symbol in rsi_cache:
                    rsi_data = rsi_cache[symbol]
                    bot_data['rsi_data'] = rsi_data
                else:
                    bot_data['rsi_data'] = {'rsi': 'N/A', 'signal': 'N/A'}
            except Exception as e:
                logger.error(f" ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П RSI ╨┤╨╗╤П {symbol}: {e}")
                bot_data['rsi_data'] = {'rsi': 'N/A', 'signal': 'N/A'}
            
            # ╨Ф╨╛╨▒╨░╨▓╨╗╤П╨╡╨╝ ╨▒╨╛╤В╨░ ╨▓ ╤Б╨┐╨╕╤Б╨╛╨║
            bots_list.append(bot_data)
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨╕╨╜╤Д╨╛╤А╨╝╨░╤Ж╨╕╤О ╨╛ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П╤Е ╤Б ╨▒╨╕╤А╨╢╨╕ ╨╛╨┤╨╕╨╜ ╤А╨░╨╖ ╨┤╨╗╤П ╨▓╤Б╨╡╤Е ╨▒╨╛╤В╨╛╨▓
        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╤В╨╛╤В ╨╢╨╡ ╤Б╨┐╨╛╤Б╨╛╨▒ ╤З╤В╨╛ ╨╕ positions_monitor_worker!
        try:
            # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤В╨╡╨╝ ╨╢╨╡ ╤Б╨┐╨╛╤Б╨╛╨▒╨╛╨╝ ╤З╤В╨╛ ╨╕ positions_monitor_worker
            exchange_obj = get_exchange()
            if exchange_obj:
                exchange_positions = exchange_obj.get_positions()
                if isinstance(exchange_positions, tuple):
                    positions_list = exchange_positions[0] if exchange_positions else []
                else:
                    positions_list = exchange_positions if exchange_positions else []
            else:
                positions_list = []
                logger.warning(f" Exchange ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜")
            
            if positions_list:
                # ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╤Б╨╗╨╛╨▓╨░╤А╤М ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨┤╨╗╤П ╨▒╤Л╤Б╤В╤А╨╛╨│╨╛ ╨┐╨╛╨╕╤Б╨║╨░
                positions_dict = {pos.get('symbol'): pos for pos in positions_list}
                
                # ╨Ф╨╛╨▒╨░╨▓╨╗╤П╨╡╨╝ ╨╕╨╜╤Д╨╛╤А╨╝╨░╤Ж╨╕╤О ╨╛ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П╤Е ╨║ ╨▒╨╛╤В╨░╨╝ (╨▓╨║╨╗╤О╤З╨░╤П ╤Б╤В╨╛╨┐-╨╗╨╛╤Б╤Б╤Л)
                for bot_data in bots_list:
                    symbol = bot_data.get('symbol')
                    if symbol in positions_dict and bot_data.get('status') in ['in_position_long', 'in_position_short']:
                        pos = positions_dict[symbol]
                        
                        bot_data['exchange_position'] = {
                            'size': pos.get('size', 0),
                            'side': pos.get('side', ''),
                            'unrealized_pnl': float(pos.get('pnl', 0)),  # тЬЕ ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨┐╤А╨░╨▓╨╕╨╗╤М╨╜╨╛╨╡ ╨┐╨╛╨╗╨╡ 'pnl'
                            'mark_price': float(pos.get('mark_price', 0)),  # тЬЕ ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨┐╤А╨░╨▓╨╕╨╗╤М╨╜╨╛╨╡ ╨┐╨╛╨╗╨╡ 'mark_price'
                            'entry_price': float(pos.get('avg_price', 0)),   # тЬЕ ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨┐╤А╨░╨▓╨╕╨╗╤М╨╜╨╛╨╡ ╨┐╨╛╨╗╨╡ 'avg_price'
                            'leverage': float(pos.get('leverage', 1)),
                            'stop_loss': pos.get('stop_loss', ''),  # ╨б╤В╨╛╨┐-╨╗╨╛╤Б╤Б ╤Б ╨▒╨╕╤А╨╢╨╕
                            'take_profit': pos.get('take_profit', ''),  # ╨в╨╡╨╣╨║-╨┐╤А╨╛╤Д╨╕╤В ╤Б ╨▒╨╕╤А╨╢╨╕
                            'roi': float(pos.get('roi', 0)),  # тЬЕ ROI ╨╡╤Б╤В╤М ╨▓ ╨┤╨░╨╜╨╜╤Л╤Е
                            'realized_pnl': float(pos.get('realized_pnl', 0)),
                            'margin_usdt': bot_data.get('margin_usdt')
                        }
                        
                        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨╕╤А╤Г╨╡╨╝ ╨Т╨б╨Х ╨┤╨░╨╜╨╜╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Б ╨▒╨╕╤А╨╢╨╡╨╣
                        exchange_stop_loss = pos.get('stopLoss', '')
                        exchange_take_profit = pos.get('takeProfit', '')
                        exchange_entry_price = float(pos.get('avgPrice', 0))  # тЭМ ╨Э╨Х╨в ╨▓ ╨┤╨░╨╜╨╜╤Л╤Е ╨▒╨╕╤А╨╢╨╕
                        exchange_size = abs(float(pos.get('size', 0)))
                        exchange_unrealized_pnl = float(pos.get('pnl', 0))  # тЬЕ ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨┐╤А╨░╨▓╨╕╨╗╤М╨╜╨╛╨╡ ╨┐╨╛╨╗╨╡ 'pnl'
                        exchange_mark_price = float(pos.get('markPrice', 0))  # тЭМ ╨Э╨Х╨в ╨▓ ╨┤╨░╨╜╨╜╤Л╤Е ╨▒╨╕╤А╨╢╨╕
                        exchange_roi = float(pos.get('roi', 0))  # тЬЕ ROI ╨╡╤Б╤В╤М ╨▓ ╨┤╨░╨╜╨╜╤Л╤Е
                        exchange_realized_pnl = float(pos.get('realized_pnl', 0))
                        exchange_leverage = float(pos.get('leverage', 1) or 1)
                        
                        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨┤╨░╨╜╨╜╤Л╨╡ ╨▒╨╛╤В╨░ ╨░╨║╤В╤Г╨░╨╗╤М╨╜╤Л╨╝╨╕ ╨┤╨░╨╜╨╜╤Л╨╝╨╕ ╤Б ╨▒╨╕╤А╨╢╨╕
                        if exchange_entry_price > 0:
                            bot_data['entry_price'] = exchange_entry_price
                        
                        # тЪб ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: position_size ╨┤╨╛╨╗╨╢╨╡╨╜ ╨▒╤Л╤В╤М ╨▓ USDT, ╨░ ╨╜╨╡ ╨▓ ╨╝╨╛╨╜╨╡╤В╨░╤Е!
                        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ volume_value ╨╕╨╖ bot_data (╤Н╤В╨╛ USDT)
                        if exchange_size > 0:
                            # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ volume_value ╨║╨░╨║ position_size (╨▓ USDT)
                            volume_value_raw = bot_data.get('volume_value', 0)
                            try:
                                volume_value = float(volume_value_raw) if volume_value_raw is not None else 0.0
                            except (TypeError, ValueError):
                                volume_value = 0.0
                            if volume_value > 0:
                                bot_data['position_size'] = volume_value  # USDT
                                bot_data['position_size_coins'] = exchange_size  # ╨Ь╨╛╨╜╨╡╤В╤Л ╨┤╨╗╤П ╤Б╨┐╤А╨░╨▓╨║╨╕
                            else:
                                # Fallback: ╨╡╤Б╨╗╨╕ volume_value ╨╜╨╡╤В, ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╤А╨░╨╖╨╝╨╡╤А ╨▓ ╨╝╨╛╨╜╨╡╤В╨░╤Е
                                bot_data['position_size'] = exchange_size
                        if exchange_mark_price > 0:
                            bot_data['current_price'] = exchange_mark_price
                            bot_data['mark_price'] = exchange_mark_price  # ╨Ф╤Г╨▒╨╗╨╕╤А╤Г╨╡╨╝ ╨┤╨╗╤П UI
                        else:
                            # тЭМ ╨Э╨Х╨в mark_price ╤Б ╨▒╨╕╤А╨╢╨╕ - ╨┐╨╛╨╗╤Г╤З╨░╨╡╨╝ ╤В╨╡╨║╤Г╤Й╤Г╤О ╤Ж╨╡╨╜╤Г ╨╜╨░╨┐╤А╤П╨╝╤Г╤О ╤Б ╨▒╨╕╤А╨╢╨╕
                            try:
                                exchange_obj = get_exchange()
                                if exchange_obj:
                                    ticker_data = exchange_obj.get_ticker(symbol)
                                    if ticker_data and ticker_data.get('last'):
                                        current_price = float(ticker_data.get('last'))
                                        bot_data['current_price'] = current_price
                                        bot_data['mark_price'] = current_price
                            except Exception as e:
                                logger.error(f" тЭМ {symbol} - ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╤Ж╨╡╨╜╤Л ╤Б ╨▒╨╕╤А╨╢╨╕: {e}")
                        
                        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ PnL ╨Т╨б╨Х╨У╨Ф╨Р, ╨┤╨░╨╢╨╡ ╨╡╤Б╨╗╨╕ ╨╛╨╜ ╤А╨░╨▓╨╡╨╜ 0
                        bot_data['unrealized_pnl'] = exchange_unrealized_pnl
                        bot_data['unrealized_pnl_usdt'] = exchange_unrealized_pnl  # ╨в╨╛╤З╨╜╨╛╨╡ ╨╖╨╜╨░╤З╨╡╨╜╨╕╨╡ ╨▓ USDT
                        bot_data['realized_pnl'] = exchange_realized_pnl
                        bot_data['leverage'] = exchange_leverage
                        bot_data['position_size_coins'] = exchange_size
                        if exchange_entry_price > 0 and exchange_size > 0:
                            position_value = exchange_entry_price * exchange_size
                            bot_data['margin_usdt'] = position_value / exchange_leverage if exchange_leverage else position_value
                        
                        # ╨Ю╤В╨╗╨░╨┤╨╛╤З╨╜╤Л╨╣ ╨╗╨╛╨│ ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ PnL
                        
                        # тЬЕ ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ROI
                        if exchange_roi != 0:
                            bot_data['roi'] = exchange_roi
                        
                        # ╨б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨╕╤А╤Г╨╡╨╝ ╤Б╤В╨╛╨┐-╨╗╨╛╤Б╤Б
                        current_stop_loss = bot_data.get('trailing_stop_price')
                        if exchange_stop_loss:
                            # ╨Х╤Б╤В╤М ╤Б╤В╨╛╨┐-╨╗╨╛╤Б╤Б ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ - ╨╛╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨┤╨░╨╜╨╜╤Л╨╡ ╨▒╨╛╤В╨░
                            new_stop_loss = float(exchange_stop_loss)
                            if not current_stop_loss or abs(current_stop_loss - new_stop_loss) > 0.001:
                                bot_data['trailing_stop_price'] = new_stop_loss
                                pass
                        else:
                            # ╨Э╨╡╤В ╤Б╤В╨╛╨┐-╨╗╨╛╤Б╤Б╨░ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ - ╨╛╤З╨╕╤Й╨░╨╡╨╝ ╨┤╨░╨╜╨╜╤Л╨╡ ╨▒╨╛╤В╨░
                            if current_stop_loss:
                                bot_data['trailing_stop_price'] = None
                                logger.info(f"[POSITION_SYNC] тЪая╕П ╨б╤В╨╛╨┐-╨╗╨╛╤Б╤Б ╨╛╤В╨╝╨╡╨╜╨╡╨╜ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ ╨┤╨╗╤П {symbol}")
                        
                        # ╨б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨╕╤А╤Г╨╡╨╝ ╤В╨╡╨╣╨║-╨┐╤А╨╛╤Д╨╕╤В
                        if exchange_take_profit:
                            bot_data['take_profit_price'] = float(exchange_take_profit)
                        else:
                            bot_data['take_profit_price'] = None
                        
                        # ╨б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨╕╤А╤Г╨╡╨╝ ╤Ж╨╡╨╜╤Г ╨▓╤Е╨╛╨┤╨░ (╨╝╨╛╨╢╨╡╤В ╨╕╨╖╨╝╨╡╨╜╨╕╤В╤М╤Б╤П ╨┐╤А╨╕ ╨┤╨╛╨▒╨░╨▓╨╗╨╡╨╜╨╕╨╕ ╨║ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕)
                        if exchange_entry_price and exchange_entry_price > 0:
                            current_entry_price = bot_data.get('entry_price')
                            if not current_entry_price or abs(current_entry_price - exchange_entry_price) > 0.001:
                                bot_data['entry_price'] = exchange_entry_price
                                pass
                        
                        # тЪб ╨а╨░╨╖╨╝╨╡╤А ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Г╨╢╨╡ ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜ ╨▓╤Л╤И╨╡ (╨▓ USDT)
                        
                        # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨▓╤А╨╡╨╝╤П ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╡╨│╨╛ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П
                        bot_data['last_update'] = datetime.now().isoformat()
        except Exception as e:
            logger.error(f" ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╤Б ╨▒╨╕╤А╨╢╨╕: {e}")
        
        # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨║╤Н╤И (╤В╨╛╨╗╤М╨║╨╛ ╨┤╨░╨╜╨╜╤Л╨╡ ╨▒╨╛╤В╨╛╨▓, account_info ╨▒╨╛╨╗╤М╤И╨╡ ╨╜╨╡ ╨║╤Н╤И╨╕╤А╤Г╨╡╤В╤Б╤П)
        current_time = datetime.now().isoformat()
        with bots_cache_lock:
            bots_cache_data.update({
                'bots': bots_list,
                'last_update': current_time
            })
        
        # тЬЕ ╨б╨Ш╨Э╨е╨а╨Ю╨Э╨Ш╨Ч╨Р╨ж╨Ш╨п: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╖╨░╨║╤А╤Л╤В╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
        try:
            sync_bots_with_exchange()
        except Exception as e:
            logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨░╤Ж╨╕╨╕ ╤Б ╨▒╨╕╤А╨╢╨╡╨╣: {e}")
        
        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ last_update ╨▓ bots_data ╨┤╨╗╤П UI
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: GIL ╨┤╨╡╨╗╨░╨╡╤В ╨╖╨░╨┐╨╕╤Б╤М ╨░╤В╨╛╨╝╨░╤А╨╜╨╛╨╣
        bots_data['last_update'] = current_time
        
        # ╨Ю╤В╨╗╨░╨┤╨╛╤З╨╜╤Л╨╣ ╨╗╨╛╨│ ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╤З╨░╤Б╤В╨╛╤В╤Л ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╣
        return True
        
    except Exception as e:
        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П ╨║╤Н╤И╨░: {e}")
        return False

def update_bot_positions_status():
    """╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╤В ╤Б╤В╨░╤В╤Г╤Б ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨▒╨╛╤В╨╛╨▓ (╤Ж╨╡╨╜╨░, PnL, ╨╗╨╕╨║╨▓╨╕╨┤╨░╤Ж╨╕╤П) ╨║╨░╨╢╨┤╤Л╨╡ SystemConfig.BOT_STATUS_UPDATE_INTERVAL ╤Б╨╡╨║╤Г╨╜╨┤"""
    try:
        if not ensure_exchange_initialized():
            return False
        
        with bots_data_lock:
            updated_count = 0
            
            for symbol, bot_data in bots_data['bots'].items():
                # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╨▒╨╛╤В╨╛╨▓ ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ (╨Э╨Ю ╨Э╨Х ╨╛╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨╜╤Л╤Е!)
                bot_status = bot_data.get('status')
                if bot_status not in ['in_position_long', 'in_position_short']:
                    continue
                
                # тЪб ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Э╨╡ ╨╛╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨▒╨╛╤В╨╛╨▓ ╨╜╨░ ╨┐╨░╤Г╨╖╨╡!
                if bot_status == BOT_STATUS['PAUSED']:
                    pass
                    continue
                
                try:
                    # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╤В╨╡╨║╤Г╤Й╤Г╤О ╤Ж╨╡╨╜╤Г
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
                    
                    # ╨а╨░╤Б╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╨╝ PnL
                    if position_side == 'LONG':
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    else:  # SHORT
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100
                    
                    # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨┤╨░╨╜╨╜╤Л╨╡ ╨▒╨╛╤В╨░
                    old_pnl = bot_data.get('unrealized_pnl', 0)
                    bot_data['unrealized_pnl'] = pnl_percent
                    bot_data['current_price'] = current_price
                    bot_data['last_update'] = datetime.now().isoformat()
                    
                    # ╨а╨░╤Б╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╨╝ ╤Ж╨╡╨╜╤Г ╨╗╨╕╨║╨▓╨╕╨┤╨░╤Ж╨╕╨╕ (╨┐╤А╨╕╨╝╨╡╤А╨╜╨╛)
                    volume_value = bot_data.get('volume_value', 10)
                    leverage = 10  # ╨Я╤А╨╡╨┤╨┐╨╛╨╗╨░╨│╨░╨╡╨╝ ╨┐╨╗╨╡╤З╨╛ 10x
                    
                    if position_side == 'LONG':
                        # ╨Ф╨╗╤П LONG: ╨╗╨╕╨║╨▓╨╕╨┤╨░╤Ж╨╕╤П ╨┐╤А╨╕ ╨┐╨░╨┤╨╡╨╜╨╕╨╕ ╤Ж╨╡╨╜╤Л
                        liquidation_price = entry_price * (1 - (100 / leverage) / 100)
                    else:  # SHORT
                        # ╨Ф╨╗╤П SHORT: ╨╗╨╕╨║╨▓╨╕╨┤╨░╤Ж╨╕╤П ╨┐╤А╨╕ ╤А╨╛╤Б╤В╨╡ ╤Ж╨╡╨╜╤Л
                        liquidation_price = entry_price * (1 + (100 / leverage) / 100)
                    
                    bot_data['liquidation_price'] = liquidation_price
                    
                    # ╨а╨░╤Б╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨┤╨╛ ╨╗╨╕╨║╨▓╨╕╨┤╨░╤Ж╨╕╨╕
                    if position_side == 'LONG':
                        distance_to_liq = ((current_price - liquidation_price) / liquidation_price) * 100
                    else:  # SHORT
                        distance_to_liq = ((liquidation_price - current_price) / liquidation_price) * 100
                    
                    bot_data['distance_to_liquidation'] = distance_to_liq
                    
                    updated_count += 1
                    
                    # ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╨╡╤Б╨╗╨╕ PnL ╨╕╨╖╨╝╨╡╨╜╨╕╨╗╤Б╤П ╨╖╨╜╨░╤З╨╕╤В╨╡╨╗╤М╨╜╨╛
                    if abs(pnl_percent - old_pnl) > 0.1:
                        logger.info(f"[POSITION_UPDATE] ЁЯУК {symbol} {position_side}: ${current_price:.6f} | PnL: {pnl_percent:+.2f}% | ╨Ы╨╕╨║╨▓╨╕╨┤╨░╤Ж╨╕╤П: ${liquidation_price:.6f} ({distance_to_liq:.1f}%)")
                
                except Exception as e:
                    logger.error(f"[POSITION_UPDATE] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П {symbol}: {e}")
                    continue
        
        if updated_count > 0:
            pass
        
        return True
        
    except Exception as e:
        logger.error(f"[POSITION_UPDATE] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣: {e}")
        return False

def get_exchange_positions():
    """╨Я╨╛╨╗╤Г╤З╨░╨╡╤В ╤А╨╡╨░╨╗╤М╨╜╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Б ╨▒╨╕╤А╨╢╨╕ ╤Б retry ╨╗╨╛╨│╨╕╨║╨╛╨╣"""
    max_retries = 3
    retry_delay = 2  # ╤Б╨╡╨║╤Г╨╜╨┤╤Л
    
    for attempt in range(max_retries):
        try:
            # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨░╨║╤В╤Г╨░╨╗╤М╨╜╤Г╤О ╤Б╤Б╤Л╨╗╨║╤Г ╨╜╨░ ╨▒╨╕╤А╨╢╤Г
            current_exchange = get_exchange()
            
            if not current_exchange:
                logger.warning(f"[EXCHANGE_POSITIONS] ╨С╨╕╤А╨╢╨░ ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░ (╨┐╨╛╨┐╤Л╤В╨║╨░ {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None

            # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ш╨Х: ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ exchange.get_positions() ╨┤╨╗╤П ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╨Т╨б╨Х╨е ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╤Б ╨┐╨░╨│╨╕╨╜╨░╤Ж╨╕╨╡╨╣
            # ╨н╤В╨╛ ╨│╨░╤А╨░╨╜╤В╨╕╤А╤Г╨╡╤В, ╤З╤В╨╛ ╨╝╤Л ╨┐╨╛╨╗╤Г╤З╨╕╨╝ ╨▓╤Б╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕, ╨░ ╨╜╨╡ ╤В╨╛╨╗╤М╨║╨╛ ╨┐╨╡╤А╨▓╤Г╤О ╤Б╤В╤А╨░╨╜╨╕╤Ж╤Г
            try:
                positions_result = current_exchange.get_positions()
                if isinstance(positions_result, tuple):
                    processed_positions_list, rapid_growth = positions_result
                else:
                    processed_positions_list = positions_result if positions_result else []
                
                # ╨Ъ╨╛╨╜╨▓╨╡╤А╤В╨╕╤А╤Г╨╡╨╝ ╨╛╨▒╤А╨░╨▒╨╛╤В╨░╨╜╨╜╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨▓ ╤Д╨╛╤А╨╝╨░╤В, ╨╛╨╢╨╕╨┤╨░╨╡╨╝╤Л╨╣ ╤Д╤Г╨╜╨║╤Ж╨╕╨╡╨╣
                raw_positions = []
                for pos in processed_positions_list:
                    # ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╤Д╨╛╤А╨╝╨░╤В ╤Б╤Л╤А╤Л╤Е ╨┤╨░╨╜╨╜╤Л╤Е ╨╕╨╖ ╨╛╨▒╤А╨░╨▒╨╛╤В╨░╨╜╨╜╤Л╤Е
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
                # Fallback: ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨┐╤А╤П╨╝╨╛╨╣ ╨▓╤Л╨╖╨╛╨▓ API ╤Б ╨┐╨░╨│╨╕╨╜╨░╤Ж╨╕╨╡╨╣
                logger.warning(f"[EXCHANGE_POSITIONS] тЪая╕П ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╤З╨╡╤А╨╡╨╖ get_positions(), ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨┐╤А╤П╨╝╨╛╨╣ API: {get_pos_error}")
                
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
                        logger.warning(f"[EXCHANGE_POSITIONS] тЪая╕П ╨Ю╤И╨╕╨▒╨║╨░ API (╨┐╨╛╨┐╤Л╤В╨║╨░ {attempt + 1}/{max_retries}): {error_msg}")
                        
                        # ╨Х╤Б╨╗╨╕ ╤Н╤В╨╛ Rate Limit, ╤Г╨▓╨╡╨╗╨╕╤З╨╕╨▓╨░╨╡╨╝ ╨╖╨░╨┤╨╡╤А╨╢╨║╤Г
                        if "rate limit" in error_msg.lower() or "too many" in error_msg.lower():
                            retry_delay = min(retry_delay * 2, 10)
                        
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            break  # ╨Т╤Л╤Е╨╛╨┤╨╕╨╝ ╨╕╨╖ ╤Ж╨╕╨║╨╗╨░ ╨┐╨░╨│╨╕╨╜╨░╤Ж╨╕╨╕ ╨┤╨╗╤П retry
                        else:
                            logger.error(f"[EXCHANGE_POSITIONS] тЭМ ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨┐╨╛╤Б╨╗╨╡ {max_retries} ╨┐╨╛╨┐╤Л╤В╨╛╨║")
                            return None
                    
                    page_positions = response.get('result', {}).get('list', [])
                    raw_positions.extend(page_positions)
                    
                    cursor = response.get('result', {}).get('nextPageCursor')
                    if not cursor:
                        break
            # тЬЕ ╨Э╨╡ ╨╗╨╛╨│╨╕╤А╤Г╨╡╨╝ ╤З╨░╤Б╤В╤Л╨╡ ╨╖╨░╨┐╤А╨╛╤Б╤Л ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ (╤В╨╛╨╗╤М╨║╨╛ ╨┐╤А╨╕ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П╤Е)
            
            # ╨Ю╨▒╤А╨░╨▒╨░╤В╤Л╨▓╨░╨╡╨╝ ╤Б╤Л╤А╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕
            processed_positions = []
            for position in raw_positions:
                symbol = position.get('symbol', '').replace('USDT', '')  # ╨г╨▒╨╕╤А╨░╨╡╨╝ USDT
                size = float(position.get('size', 0))
                side = position.get('side', '')  # 'Buy' ╨╕╨╗╨╕ 'Sell'
                entry_price = float(position.get('avgPrice', 0))
                unrealized_pnl = float(position.get('unrealisedPnl', 0))
                mark_price = float(position.get('markPrice', 0))
                
                if abs(size) > 0:  # ╨в╨╛╨╗╤М╨║╨╛ ╨░╨║╤В╨╕╨▓╨╜╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕
                    processed_positions.append({
                        'symbol': symbol,
                        'size': size,
                        'side': side,
                        'entry_price': entry_price,
                        'unrealized_pnl': unrealized_pnl,
                        'mark_price': mark_price,
                        'position_side': 'LONG' if side == 'Buy' else 'SHORT'
                    })
            
            # тЬЕ ╨Э╨╡ ╨╗╨╛╨│╨╕╤А╤Г╨╡╨╝ ╤З╨░╤Б╤В╤Л╨╡ ╨╖╨░╨┐╤А╨╛╤Б╤Л (╤В╨╛╨╗╤М╨║╨╛ ╨┐╤А╨╕ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П╤Е)
            
            # ╨Т╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╨╝ ╨Т╨б╨Х ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Б ╨▒╨╕╤А╨╢╨╕, ╨╜╨╡ ╤Д╨╕╨╗╤М╤В╤А╤Г╤П ╨┐╨╛ ╨╜╨░╨╗╨╕╤З╨╕╤О ╨▒╨╛╤В╨╛╨▓ ╨▓ ╤Б╨╕╤Б╤В╨╡╨╝╨╡
            # ╨н╤В╨╛ ╨╜╤Г╨╢╨╜╨╛ ╨┤╨╗╤П ╨┐╤А╨░╨▓╨╕╨╗╤М╨╜╨╛╨╣ ╤А╨░╨▒╨╛╤В╤Л ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨░╤Ж╨╕╨╕ ╨╕ ╨╛╤З╨╕╤Б╤В╨║╨╕ ╨╜╨╡╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓
            filtered_positions = []
            ignored_positions = []
            
            for pos in processed_positions:
                symbol = pos['symbol']
                # ╨Ф╨╛╨▒╨░╨▓╨╗╤П╨╡╨╝ ╨▓╤Б╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨▒╨╡╨╖ ╤Д╨╕╨╗╤М╤В╤А╨░╤Ж╨╕╨╕
                filtered_positions.append(pos)
            
            # тЬЕ ╨Э╨╡ ╨╗╨╛╨│╨╕╤А╤Г╨╡╨╝ ╤З╨░╤Б╤В╤Л╨╡ ╨╖╨░╨┐╤А╨╛╤Б╤Л (╤В╨╛╨╗╤М╨║╨╛ ╨┐╤А╨╕ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П╤Е)
            return filtered_positions
            
        except Exception as api_error:
            logger.error(f"[EXCHANGE_POSITIONS] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╤П╨╝╨╛╨│╨╛ ╨╛╨▒╤А╨░╤Й╨╡╨╜╨╕╤П ╨║ API: {api_error}")
            # Fallback ╨║ ╤Б╤Г╤Й╨╡╤Б╤В╨▓╤Г╤О╤Й╨╡╨╝╤Г ╨╝╨╡╤В╨╛╨┤╤Г
            current_exchange = get_exchange()
            if not current_exchange:
                logger.error("[EXCHANGE_POSITIONS] тЭМ ╨С╨╕╤А╨╢╨░ ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░")
                return []
            positions, _ = current_exchange.get_positions()
            logger.info(f"[EXCHANGE_POSITIONS] Fallback: ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╛ {len(positions) if positions else 0} ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣")
            
            if not positions:
                return []
            
            # ╨Ю╨▒╤А╨░╨▒╨░╤В╤Л╨▓╨░╨╡╨╝ fallback ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕
            processed_positions = []
            for position in positions:
                # ╨Я╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Г╨╢╨╡ ╨╛╨▒╤А╨░╨▒╨╛╤В╨░╨╜╤Л ╨▓ exchange.get_positions()
                symbol = position.get('symbol', '')
                size = position.get('size', 0)
                side = position.get('side', '')  # 'Long' ╨╕╨╗╨╕ 'Short'
                
                if abs(size) > 0:
                    processed_positions.append({
                        'symbol': symbol,
                        'size': size,
                        'side': side,
                        'entry_price': 0.0,  # ╨Э╨╡╤В ╨┤╨░╨╜╨╜╤Л╤Е ╨▓ ╨╛╨▒╤А╨░╨▒╨╛╤В╨░╨╜╨╜╨╛╨╝ ╤Д╨╛╤А╨╝╨░╤В╨╡
                        'unrealized_pnl': position.get('pnl', 0),
                        'mark_price': 0.0,
                        'position_side': side
                    })
            
            # ╨Ъ╨а╨Ш╨в╨Ш╨з╨Х╨б╨Ъ╨Ш ╨Т╨Р╨Ц╨Э╨Ю: ╨д╨╕╨╗╤М╤В╤А╤Г╨╡╨╝ fallback ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤В╨╛╨╢╨╡
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
                logger.info(f"[EXCHANGE_POSITIONS] ЁЯЪл Fallback: ╨Ш╨│╨╜╨╛╤А╨╕╤А╤Г╨╡╨╝ {len(ignored_positions)} ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨▒╨╡╨╖ ╨▒╨╛╤В╨╛╨▓ ╨▓ ╤Б╨╕╤Б╤В╨╡╨╝╨╡")
            
            logger.info(f"[EXCHANGE_POSITIONS] тЬЕ Fallback: ╨Т╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╨╝ {len(filtered_positions)} ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╤Б ╨▒╨╛╤В╨░╨╝╨╕ ╨▓ ╤Б╨╕╤Б╤В╨╡╨╝╨╡")
            return filtered_positions
            
        except Exception as e:
            logger.error(f"[EXCHANGE_POSITIONS] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨▓ ╨┐╨╛╨┐╤Л╤В╨║╨╡ {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                logger.error(f"[EXCHANGE_POSITIONS] тЭМ ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨┐╨╛╤Б╨╗╨╡ {max_retries} ╨┐╨╛╨┐╤Л╤В╨╛╨║")
                return None
    
    # ╨Х╤Б╨╗╨╕ ╨╝╤Л ╨┤╨╛╤И╨╗╨╕ ╤Б╤О╨┤╨░, ╨╖╨╜╨░╤З╨╕╤В ╨▓╤Б╨╡ ╨┐╨╛╨┐╤Л╤В╨║╨╕ ╨╕╤Б╤З╨╡╤А╨┐╨░╨╜╤Л
    logger.error(f"[EXCHANGE_POSITIONS] тЭМ ╨Т╤Б╨╡ ╨┐╨╛╨┐╤Л╤В╨║╨╕ ╨╕╤Б╤З╨╡╤А╨┐╨░╨╜╤Л")
    return None

def compare_bot_and_exchange_positions():
    """╨б╤А╨░╨▓╨╜╨╕╨▓╨░╨╡╤В ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨▒╨╛╤В╨╛╨▓ ╨▓ ╤Б╨╕╤Б╤В╨╡╨╝╨╡ ╤Б ╤А╨╡╨░╨╗╤М╨╜╤Л╨╝╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П╨╝╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡"""
    try:
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Б ╨▒╨╕╤А╨╢╨╕
        exchange_positions = get_exchange_positions()
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨▒╨╛╤В╨╛╨▓ ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╕╨╖ ╤Б╨╕╤Б╤В╨╡╨╝╤Л
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
        
        # ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╤Б╨╗╨╛╨▓╨░╤А╨╕ ╨┤╨╗╤П ╤Г╨┤╨╛╨▒╨╜╨╛╨│╨╛ ╤Б╤А╨░╨▓╨╜╨╡╨╜╨╕╤П
        exchange_dict = {pos['symbol']: pos for pos in exchange_positions}
        bot_dict = {pos['symbol']: pos for pos in bot_positions}
        
        # ╨Э╨░╤Е╨╛╨┤╨╕╨╝ ╤А╨░╤Б╤Е╨╛╨╢╨┤╨╡╨╜╨╕╤П
        discrepancies = {
            'missing_in_bot': [],  # ╨Х╤Б╤В╤М ╨╜╨░ ╨▒╨╕╤А╨╢╨╡, ╨╜╨╡╤В ╨▓ ╨▒╨╛╤В╨╡ (╨Э╨Х ╤Б╨╛╨╖╨┤╨░╨╡╨╝ ╨▒╨╛╤В╨╛╨▓!)
            'missing_in_exchange': [],  # ╨Х╤Б╤В╤М ╨▓ ╨▒╨╛╤В╨╡, ╨╜╨╡╤В ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ (╨╛╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╤Б╤В╨░╤В╤Г╤Б)
            'side_mismatch': []  # ╨Х╤Б╤В╤М ╨▓ ╨╛╨▒╨╛╨╕╤Е, ╨╜╨╛ ╤Б╤В╨╛╤А╨╛╨╜╤Л ╨╜╨╡ ╤Б╨╛╨▓╨┐╨░╨┤╨░╤О╤В (╨╕╤Б╨┐╤А╨░╨▓╨╗╤П╨╡╨╝)
        }
        
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
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
                # тЬЕ ╨Э╨╛╤А╨╝╨░╨╗╨╕╨╖╤Г╨╡╨╝ ╤Б╤В╨╛╤А╨╛╨╜╤Л ╨┤╨╗╤П ╤Б╤А╨░╨▓╨╜╨╡╨╜╨╕╤П (LONG/Long -> LONG, SHORT/Short -> SHORT)
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
        
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨▓ ╨▒╨╛╤В╨╡
        for symbol, bot_pos in bot_dict.items():
            if symbol not in exchange_dict:
                discrepancies['missing_in_exchange'].append({
                    'symbol': symbol,
                    'bot_side': bot_pos['position_side'],
                    'bot_entry_price': bot_pos['entry_price'],
                    'bot_status': bot_pos['status']
                })
        
        # ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╤Л
        total_discrepancies = (len(discrepancies['missing_in_bot']) + 
                             len(discrepancies['missing_in_exchange']) + 
                             len(discrepancies['side_mismatch']))
        
        if total_discrepancies > 0:
            logger.warning(f"[POSITION_SYNC] тЪая╕П ╨Ю╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜╨╛ {total_discrepancies} ╤А╨░╤Б╤Е╨╛╨╢╨┤╨╡╨╜╨╕╨╣ ╨╝╨╡╨╢╨┤╤Г ╨▒╨╛╤В╨╛╨╝ ╨╕ ╨▒╨╕╤А╨╢╨╡╨╣")
            
            if discrepancies['missing_in_bot']:
                logger.info(f"[POSITION_SYNC] ЁЯУК ╨Я╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ ╨▒╨╡╨╖ ╨▒╨╛╤В╨░ ╨▓ ╤Б╨╕╤Б╤В╨╡╨╝╨╡: {len(discrepancies['missing_in_bot'])} (╨╕╨│╨╜╨╛╤А╨╕╤А╤Г╨╡╨╝ - ╨╜╨╡ ╤Б╨╛╨╖╨┤╨░╨╡╨╝ ╨▒╨╛╤В╨╛╨▓)")
                for pos in discrepancies['missing_in_bot']:
                    logger.info(f"[POSITION_SYNC]   - {pos['symbol']}: {pos['exchange_side']} ${pos['exchange_entry_price']:.6f} (PnL: {pos['exchange_pnl']:.2f}) - ╨Э╨Х ╤Б╨╛╨╖╨┤╨░╨╡╨╝ ╨▒╨╛╤В╨░")
            
            if discrepancies['missing_in_exchange']:
                logger.warning(f"[POSITION_SYNC] ЁЯдЦ ╨С╨╛╤В╤Л ╨▒╨╡╨╖ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡: {len(discrepancies['missing_in_exchange'])}")
                for pos in discrepancies['missing_in_exchange']:
                    logger.warning(f"[POSITION_SYNC]   - {pos['symbol']}: {pos['bot_side']} ${pos['bot_entry_price']:.6f} (╤Б╤В╨░╤В╤Г╤Б: {pos['bot_status']})")
            
            if discrepancies['side_mismatch']:
                logger.warning(f"[POSITION_SYNC] ЁЯФД ╨Э╨╡╤Б╨╛╨▓╨┐╨░╨┤╨╡╨╜╨╕╨╡ ╤Б╤В╨╛╤А╨╛╨╜: {len(discrepancies['side_mismatch'])}")
                for pos in discrepancies['side_mismatch']:
                    logger.warning(f"[POSITION_SYNC]   - {pos['symbol']}: ╨▒╨╛╤В={pos['bot_side']}, ╨▒╨╕╤А╨╢╨░={pos['exchange_side']}")
        else:
            logger.info(f"[POSITION_SYNC] тЬЕ ╨б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨░╤Ж╨╕╤П ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣: ╨▓╤Б╨╡ {len(bot_positions)} ╨▒╨╛╤В╨╛╨▓ ╤Б╨╛╨╛╤В╨▓╨╡╤В╤Б╤В╨▓╤Г╤О╤В ╨▒╨╕╤А╨╢╨╡")
        
        return discrepancies
        
    except Exception as e:
        logger.error(f"[POSITION_SYNC] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╤А╨░╨▓╨╜╨╡╨╜╨╕╤П ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣: {e}")
        return None

def sync_positions_with_exchange():
    """╨г╨╝╨╜╨░╤П ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨░╤Ж╨╕╤П ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨▒╨╛╤В╨╛╨▓ ╤Б ╤А╨╡╨░╨╗╤М╨╜╤Л╨╝╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П╨╝╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡"""
    try:
        # тЬЕ ╨Э╨╡ ╨╗╨╛╨│╨╕╤А╤Г╨╡╨╝ ╤З╨░╤Б╤В╤Л╨╡ ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨░╤Ж╨╕╨╕ (╤В╨╛╨╗╤М╨║╨╛ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╤Л ╨┐╤А╨╕ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П╤Е)
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Б ╨▒╨╕╤А╨╢╨╕ ╤Б retry ╨╗╨╛╨│╨╕╨║╨╛╨╣
        exchange_positions = get_exchange_positions()
        
        # ╨Х╤Б╨╗╨╕ ╨╜╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Б ╨▒╨╕╤А╨╢╨╕, ╨Э╨Х ╤Б╨▒╤А╨░╤Б╤Л╨▓╨░╨╡╨╝ ╨▒╨╛╤В╨╛╨▓
        if exchange_positions is None:
            logger.warning("[POSITION_SYNC] тЪая╕П ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Б ╨▒╨╕╤А╨╢╨╕ - ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨░╤Ж╨╕╤О")
            return False
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨▒╨╛╤В╨╛╨▓ ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╕╨╖ ╤Б╨╕╤Б╤В╨╡╨╝╤Л
        with bots_data_lock:
            bot_positions = []
            # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ш╨Х: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╜╨░╨╗╨╕╤З╨╕╨╡ ╨║╨╗╤О╤З╨░ 'bots'
            if 'bots' not in bots_data:
                logger.warning("[POSITION_SYNC] тЪая╕П bots_data ╨╜╨╡ ╤Б╨╛╨┤╨╡╤А╨╢╨╕╤В ╨║╨╗╤О╤З 'bots' - ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╤Г╨╡╨╝")
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
        
        # тЬЕ ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╨┐╤А╨╕ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П╤Е ╨╕╨╗╨╕ ╨╛╤И╨╕╨▒╨║╨░╤Е (╤Г╨▒╨╕╤А╨░╨╡╨╝ ╤Б╨┐╨░╨╝)
        # logger.info(f"[POSITION_SYNC] ЁЯУК ╨С╨╕╤А╨╢╨░: {len(exchange_positions)}, ╨С╨╛╤В╤Л: {len(bot_positions)}")
        
        # ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╤Б╨╗╨╛╨▓╨░╤А╨╕ ╨┤╨╗╤П ╤Г╨┤╨╛╨▒╨╜╨╛╨│╨╛ ╤Б╤А╨░╨▓╨╜╨╡╨╜╨╕╤П
        exchange_dict = {pos['symbol']: pos for pos in exchange_positions}
        bot_dict = {pos['symbol']: pos for pos in bot_positions}
        
        synced_count = 0
        errors_count = 0
        
        # ╨Ю╨▒╤А╨░╨▒╨░╤В╤Л╨▓╨░╨╡╨╝ ╨▒╨╛╤В╨╛╨▓ ╨▒╨╡╨╖ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
        for symbol, bot_data in bot_dict.items():
            if symbol not in exchange_dict:
                logger.warning(f"[POSITION_SYNC] тЪая╕П ╨С╨╛╤В {symbol} ╨▒╨╡╨╖ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ (╤Б╤В╨░╤В╤Г╤Б: {bot_data['status']})")
                
                # ╨Т╨Р╨Ц╨Э╨Ю: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨┤╨╡╨╣╤Б╤В╨▓╨╕╤В╨╡╨╗╤М╨╜╨╛ ╨╗╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╨╖╨░╨║╤А╤Л╨╗╨░╤Б╤М
                # ╨Э╨╡ ╤Б╨▒╤А╨░╤Б╤Л╨▓╨░╨╡╨╝ ╨▒╨╛╤В╨╛╨▓ ╤Б╤А╨░╨╖╤Г - ╨┤╨░╨╡╨╝ ╨╕╨╝ ╨▓╤А╨╡╨╝╤П ╨╜╨░ ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╡
                try:
                    # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨╡╤Б╤В╤М ╨╗╨╕ ╨░╨║╤В╨╕╨▓╨╜╤Л╨╡ ╨╛╤А╨┤╨╡╤А╨░ ╨┤╨╗╤П ╤Н╤В╨╛╨│╨╛ ╤Б╨╕╨╝╨▓╨╛╨╗╨░
                    has_active_orders = check_active_orders(symbol)
                    
                    if not has_active_orders:
                        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨г╨Ф╨Р╨Ы╨п╨Х╨Ь ╨▒╨╛╤В╨░, ╨░ ╨╜╨╡ ╨┐╨╡╤А╨╡╨▓╨╛╨┤╨╕╨╝ ╨▓ IDLE - ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨╡╤В ╨╜╨░ ╨▒╨╕╤А╨╢╨╡!
                        with bots_data_lock:
                            if symbol in bots_data['bots']:
                                del bots_data['bots'][symbol]
                                synced_count += 1
                                logger.info(f"[POSITION_SYNC] ЁЯЧСя╕П ╨г╨┤╨░╨╗╨╡╨╜ ╨▒╨╛╤В {symbol} - ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╨╖╨░╨║╤А╤Л╤В╨░ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡")
                    else:
                        logger.info(f"[POSITION_SYNC] тП│ ╨С╨╛╤В {symbol} ╨╕╨╝╨╡╨╡╤В ╨░╨║╤В╨╕╨▓╨╜╤Л╨╡ ╨╛╤А╨┤╨╡╤А╨░ - ╨╛╤Б╤В╨░╨▓╨╗╤П╨╡╨╝ ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕")
                        
                except Exception as check_error:
                    logger.error(f"[POSITION_SYNC] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨╛╤А╨┤╨╡╤А╨╛╨▓ ╨┤╨╗╤П {symbol}: {check_error}")
                    errors_count += 1
        
        # ╨Ю╨▒╤А╨░╨▒╨░╤В╤Л╨▓╨░╨╡╨╝ ╨╜╨╡╤Б╨╛╨▓╨┐╨░╨┤╨╡╨╜╨╕╤П ╤Б╤В╨╛╤А╨╛╨╜ - ╨╕╤Б╨┐╤А╨░╨▓╨╗╤П╨╡╨╝ ╨┤╨░╨╜╨╜╤Л╨╡ ╨▒╨╛╤В╨░ ╨▓ ╤Б╨╛╨╛╤В╨▓╨╡╤В╤Б╤В╨▓╨╕╨╕ ╤Б ╨▒╨╕╤А╨╢╨╡╨╣
        for symbol, exchange_pos in exchange_dict.items():
            if symbol in bot_dict:
                bot_data = bot_dict[symbol]
                exchange_side = exchange_pos['position_side']
                bot_side = bot_data['position_side']
                
                # тЬЕ ╨Э╨╛╤А╨╝╨░╨╗╨╕╨╖╤Г╨╡╨╝ ╤Б╤В╨╛╤А╨╛╨╜╤Л ╨┤╨╗╤П ╤Б╤А╨░╨▓╨╜╨╡╨╜╨╕╤П (LONG/Long -> LONG, SHORT/Short -> SHORT)
                exchange_side_normalized = exchange_side.upper() if exchange_side else None
                bot_side_normalized = bot_side.upper() if bot_side else None
                
                if exchange_side_normalized != bot_side_normalized:
                    logger.warning(f"[POSITION_SYNC] ЁЯФД ╨Ш╤Б╨┐╤А╨░╨▓╨╗╨╡╨╜╨╕╨╡ ╤Б╤В╨╛╤А╨╛╨╜╤Л ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕: {symbol} {bot_side} -> {exchange_side}")
                    
                    try:
                        with bots_data_lock:
                            if symbol in bots_data['bots']:
                                bots_data['bots'][symbol]['position_side'] = exchange_side
                                bots_data['bots'][symbol]['entry_price'] = exchange_pos['entry_price']
                                bots_data['bots'][symbol]['status'] = f'in_position_{exchange_side.lower()}'
                                bots_data['bots'][symbol]['unrealized_pnl'] = exchange_pos['unrealized_pnl']
                                bots_data['bots'][symbol]['last_update'] = datetime.now().isoformat()
                                synced_count += 1
                                logger.info(f"[POSITION_SYNC] тЬЕ ╨Ш╤Б╨┐╤А╨░╨▓╨╗╨╡╨╜╤Л ╨┤╨░╨╜╨╜╤Л╨╡ ╨▒╨╛╤В╨░ {symbol} ╨▓ ╤Б╨╛╨╛╤В╨▓╨╡╤В╤Б╤В╨▓╨╕╨╕ ╤Б ╨▒╨╕╤А╨╢╨╡╨╣")
                    except Exception as update_error:
                        logger.error(f"[POSITION_SYNC] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П ╨▒╨╛╤В╨░ {symbol}: {update_error}")
                        errors_count += 1
        
        # ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╤Л
        if synced_count > 0:
            logger.info(f"[POSITION_SYNC] тЬЕ ╨б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨╛ {synced_count} ╨▒╨╛╤В╨╛╨▓")
        if errors_count > 0:
            logger.warning(f"[POSITION_SYNC] тЪая╕П ╨Ю╤И╨╕╨▒╨╛╨║ ╨┐╤А╨╕ ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨░╤Ж╨╕╨╕: {errors_count}")
        
        return synced_count > 0
        
    except Exception as e:
        logger.error(f"[POSITION_SYNC] тЭМ ╨Ъ╤А╨╕╤В╨╕╤З╨╡╤Б╨║╨░╤П ╨╛╤И╨╕╨▒╨║╨░ ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨░╤Ж╨╕╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣: {e}")
        return False

def check_active_orders(symbol):
    """╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╤В, ╨╡╤Б╤В╤М ╨╗╨╕ ╨░╨║╤В╨╕╨▓╨╜╤Л╨╡ ╨╛╤А╨┤╨╡╤А╨░ ╨┤╨╗╤П ╤Б╨╕╨╝╨▓╨╛╨╗╨░"""
    try:
        if not ensure_exchange_initialized():
            return False
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨░╨║╤В╨╕╨▓╨╜╤Л╨╡ ╨╛╤А╨┤╨╡╤А╨░ ╨┤╨╗╤П ╤Б╨╕╨╝╨▓╨╛╨╗╨░
        current_exchange = get_exchange()
        if not current_exchange:
            return False
        orders = current_exchange.get_open_orders(symbol)
        return len(orders) > 0
        
    except Exception as e:
        logger.error(f"[ORDER_CHECK] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨╛╤А╨┤╨╡╤А╨╛╨▓ ╨┤╨╗╤П {symbol}: {e}")
        return False

def cleanup_inactive_bots():
    """╨г╨┤╨░╨╗╤П╨╡╤В ╨▒╨╛╤В╨╛╨▓, ╨║╨╛╤В╨╛╤А╤Л╨╡ ╨╜╨╡ ╨╕╨╝╨╡╤О╤В ╤А╨╡╨░╨╗╤М╨╜╤Л╤Е ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ ╨▓ ╤В╨╡╤З╨╡╨╜╨╕╨╡ SystemConfig.INACTIVE_BOT_TIMEOUT ╤Б╨╡╨║╤Г╨╜╨┤"""
    try:
        current_time = time.time()
        removed_count = 0
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╤А╨╡╨░╨╗╤М╨╜╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Б ╨▒╨╕╤А╨╢╨╕
        exchange_positions = get_exchange_positions()
        
        # ╨Ъ╨а╨Ш╨в╨Ш╨з╨Х╨б╨Ъ╨Ш ╨Т╨Р╨Ц╨Э╨Ю: ╨Х╤Б╨╗╨╕ ╨╜╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Б ╨▒╨╕╤А╨╢╨╕, ╨Э╨Х ╨г╨Ф╨Р╨Ы╨п╨Х╨Ь ╨▒╨╛╤В╨╛╨▓!
        if exchange_positions is None:
            logger.warning(f" тЪая╕П ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Б ╨▒╨╕╤А╨╢╨╕ - ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╨╛╤З╨╕╤Б╤В╨║╤Г ╨┤╨╗╤П ╨▒╨╡╨╖╨╛╨┐╨░╤Б╨╜╨╛╤Б╤В╨╕")
            return False
        
        # ╨Э╨╛╤А╨╝╨░╨╗╨╕╨╖╤Г╨╡╨╝ ╤Б╨╕╨╝╨▓╨╛╨╗╤Л ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ (╤Г╨▒╨╕╤А╨░╨╡╨╝ USDT ╨╡╤Б╨╗╨╕ ╨╡╤Б╤В╤М)
        def normalize_symbol(symbol):
            """╨Э╨╛╤А╨╝╨░╨╗╨╕╨╖╤Г╨╡╤В ╤Б╨╕╨╝╨▓╨╛╨╗, ╤Г╨▒╨╕╤А╨░╤П USDT ╤Б╤Г╤Д╤Д╨╕╨║╤Б ╨╡╤Б╨╗╨╕ ╨╡╤Б╤В╤М"""
            if symbol.endswith('USDT'):
                return symbol[:-4]  # ╨г╨▒╨╕╤А╨░╨╡╨╝ 'USDT'
            return symbol
        
        # ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╨╝╨╜╨╛╨╢╨╡╤Б╤В╨▓╨╛ ╨╜╨╛╤А╨╝╨░╨╗╨╕╨╖╨╛╨▓╨░╨╜╨╜╤Л╤Е ╤Б╨╕╨╝╨▓╨╛╨╗╨╛╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
        exchange_symbols = {normalize_symbol(pos['symbol']) for pos in exchange_positions}
        
        logger.info(f" ЁЯФН ╨Я╤А╨╛╨▓╨╡╤А╨║╨░ {len(bots_data['bots'])} ╨▒╨╛╤В╨╛╨▓ ╨╜╨░ ╨╜╨╡╨░╨║╤В╨╕╨▓╨╜╨╛╤Б╤В╤М")
        logger.info(f" ЁЯУК ╨Э╨░╨╣╨┤╨╡╨╜╨╛ {len(exchange_symbols)} ╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡: {sorted(exchange_symbols)}")
        
        with bots_data_lock:
            bots_to_remove = []
            
            for symbol, bot_data in bots_data['bots'].items():
                bot_status = bot_data.get('status', 'idle')
                last_update_str = bot_data.get('last_update')
                
                # ╨Ъ╨а╨Ш╨в╨Ш╨з╨Х╨б╨Ъ╨Ш ╨Т╨Р╨Ц╨Э╨Ю: ╨Э╨Х ╨г╨Ф╨Р╨Ы╨п╨Х╨Ь ╨▒╨╛╤В╨╛╨▓, ╨║╨╛╤В╨╛╤А╤Л╨╡ ╨╜╨░╤Е╨╛╨┤╤П╤В╤Б╤П ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕!
                if bot_status in ['in_position_long', 'in_position_short']:
                    logger.info(f" ЁЯЫбя╕П ╨С╨╛╤В {symbol} ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ {bot_status} - ╨Э╨Х ╨г╨Ф╨Р╨Ы╨п╨Х╨Ь")
                    continue
                
                # ╨Я╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╨▒╨╛╤В╨╛╨▓, ╨║╨╛╤В╨╛╤А╤Л╨╡ ╨╕╨╝╨╡╤О╤В ╤А╨╡╨░╨╗╤М╨╜╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
                # ╨Э╨╛╤А╨╝╨░╨╗╨╕╨╖╤Г╨╡╨╝ ╤Б╨╕╨╝╨▓╨╛╨╗ ╨▒╨╛╤В╨░ ╨┤╨╗╤П ╨║╨╛╤А╤А╨╡╨║╤В╨╜╨╛╨│╨╛ ╤Б╤А╨░╨▓╨╜╨╡╨╜╨╕╤П
                normalized_bot_symbol = normalize_symbol(symbol)
                if normalized_bot_symbol in exchange_symbols:
                    continue
                
                # ╨г╨▒╤А╨░╨╗╨╕ ╤Е╨░╤А╨┤╨║╨╛╨┤ - ╤В╨╡╨┐╨╡╤А╤М ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╤А╨╡╨░╨╗╤М╨╜╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
                
                # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨г╨Ф╨Р╨Ы╨п╨Х╨Ь ╨▒╨╛╤В╨╛╨▓ ╨▓ ╤Б╤В╨░╤В╤Г╤Б╨╡ 'idle' ╨╕╨╗╨╕ 'running' ╨С╨Х╨Ч ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡!
                # ╨Х╤Б╨╗╨╕ ╨▒╨╛╤В ╨┐╤А╨╛╤И╨╡╨╗ ╨▓╤Б╨╡ ╤Д╨╕╨╗╤М╤В╤А╤Л ╨╕ ╨┤╨╛╨╗╨╢╨╡╨╜ ╨▒╤Л╨╗ ╨╖╨░╨╣╤В╨╕ ╨▓ ╤Б╨┤╨╡╨╗╨║╤Г, ╨╜╨╛ ╨╜╨╡ ╨╖╨░╤И╨╡╨╗ - ╤Н╤В╨╛ ╨╛╤И╨╕╨▒╨║╨░ ╤Б╨╕╤Б╤В╨╡╨╝╤Л
                # ╨в╨░╨║╨╕╨╡ ╨▒╨╛╤В╤Л ╨╜╨╡ ╨┤╨╛╨╗╨╢╨╜╤Л ╤Б╤Г╤Й╨╡╤Б╤В╨▓╨╛╨▓╨░╤В╤М ╨╕ ╨┤╨╛╨╗╨╢╨╜╤Л ╤Г╨┤╨░╨╗╤П╤В╤М╤Б╤П ╨╜╨╡╨╝╨╡╨┤╨╗╨╡╨╜╨╜╨╛
                if bot_status in ['idle', 'running']:
                    # ╨Х╤Б╨╗╨╕ ╨╜╨╡╤В ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ - ╤Г╨┤╨░╨╗╤П╨╡╨╝ ╨▒╨╛╤В╨░ (╨╛╤И╨╕╨▒╨║╨░ ╤Б╨╕╤Б╤В╨╡╨╝╤Л)
                    if normalized_bot_symbol not in exchange_symbols:
                        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨▓╤А╨╡╨╝╤П ╤Б╨╛╨╖╨┤╨░╨╜╨╕╤П - ╨╜╨╡ ╤Г╨┤╨░╨╗╤П╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╤З╤В╨╛ ╤Б╨╛╨╖╨┤╨░╨╜╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓ (╨▓ ╤В╨╡╤З╨╡╨╜╨╕╨╡ 5 ╨╝╨╕╨╜╤Г╤В)
                        created_time_str = bot_data.get('created_time') or bot_data.get('created_at')
                        if created_time_str:
                            try:
                                created_time = datetime.fromisoformat(created_time_str.replace('Z', '+00:00'))
                                time_since_creation = current_time - created_time.timestamp()
                                if time_since_creation < 300:  # 5 ╨╝╨╕╨╜╤Г╤В
                                    pass
                                    continue
                            except Exception:
                                pass  # ╨Х╤Б╨╗╨╕ ╨╛╤И╨╕╨▒╨║╨░ ╨┐╨░╤А╤Б╨╕╨╜╨│╨░ - ╤Г╨┤╨░╨╗╤П╨╡╨╝ ╨▒╨╛╤В╨░
                        
                        logger.warning(f" ЁЯЧСя╕П {symbol}: ╨г╨┤╨░╨╗╤П╨╡╨╝ ╨▒╨╛╤В╨░ ╨▓ ╤Б╤В╨░╤В╤Г╤Б╨╡ {bot_status} ╨▒╨╡╨╖ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ (╨╛╤И╨╕╨▒╨║╨░ ╤Б╨╕╤Б╤В╨╡╨╝╤Л)")
                        bots_to_remove.append(symbol)
                        continue
                    else:
                        # ╨Х╤Б╤В╤М ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ - ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ (╨▒╨╛╤В ╤А╨░╨▒╨╛╤В╨░╨╡╤В ╨║╨╛╤А╤А╨╡╨║╤В╨╜╨╛)
                        continue
                
                # ╨Ъ╨а╨Ш╨в╨Ш╨з╨Х╨б╨Ъ╨Ш ╨Т╨Р╨Ц╨Э╨Ю: ╨Э╨╡ ╤Г╨┤╨░╨╗╤П╨╡╨╝ ╨▒╨╛╤В╨╛╨▓, ╨║╨╛╤В╨╛╤А╤Л╨╡ ╤В╨╛╨╗╤М╨║╨╛ ╤З╤В╨╛ ╨╖╨░╨│╤А╤Г╨╢╨╡╨╜╤Л
                # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╤З╤В╨╛ ╨▒╨╛╤В ╨▒╤Л╨╗ ╤Б╨╛╨╖╨┤╨░╨╜ ╨╜╨╡╨┤╨░╨▓╨╜╨╛ (╨▓ ╤В╨╡╤З╨╡╨╜╨╕╨╡ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╤Е 5 ╨╝╨╕╨╜╤Г╤В)
                created_time_str = bot_data.get('created_time')
                if created_time_str:
                    try:
                        created_time = datetime.fromisoformat(created_time_str.replace('Z', '+00:00'))
                        time_since_creation = current_time - created_time.timestamp()
                        if time_since_creation < 300:  # 5 ╨╝╨╕╨╜╤Г╤В
                            logger.info(f" тП│ ╨С╨╛╤В {symbol} ╤Б╨╛╨╖╨┤╨░╨╜ {time_since_creation//60:.0f} ╨╝╨╕╨╜ ╨╜╨░╨╖╨░╨┤, ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╤Г╨┤╨░╨╗╨╡╨╜╨╕╨╡")
                            continue
                    except Exception as e:
                        logger.warning(f" тЪая╕П ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨░╤А╤Б╨╕╨╜╨│╨░ ╨▓╤А╨╡╨╝╨╡╨╜╨╕ ╤Б╨╛╨╖╨┤╨░╨╜╨╕╤П ╨┤╨╗╤П {symbol}: {e}")
                
                # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨▓╤А╨╡╨╝╤П ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╡╨│╨╛ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П
                if last_update_str:
                    # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ю: ╨Ю╨▒╤А╨░╨▒╨░╤В╤Л╨▓╨░╨╡╨╝ ╨╜╨╡╨║╨╛╤А╤А╨╡╨║╤В╨╜╤Л╨╡ ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П ╤В╨╕╨┐╨░ '╨Э╨╕╨║╨╛╨│╨┤╨░'
                    if isinstance(last_update_str, str) and last_update_str.lower() in ['╨╜╨╕╨║╨╛╨│╨┤╨░', 'never', '']:
                        pass
                        # ╨Я╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╨┐╤А╨╛╨▓╨╡╤А╨║╤Г last_update, ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ created_at ╨╜╨╕╨╢╨╡
                        last_update_str = None
                    else:
                        try:
                            last_update = datetime.fromisoformat(last_update_str.replace('Z', '+00:00'))
                            time_since_update = current_time - last_update.timestamp()
                            
                            if time_since_update >= SystemConfig.INACTIVE_BOT_TIMEOUT:
                                logger.warning(f" тП░ ╨С╨╛╤В {symbol} ╨╜╨╡╨░╨║╤В╨╕╨▓╨╡╨╜ {time_since_update//60:.0f} ╨╝╨╕╨╜ (╤Б╤В╨░╤В╤Г╤Б: {bot_status})")
                                bots_to_remove.append(symbol)
                                
                                # ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ ╤Г╨┤╨░╨╗╨╡╨╜╨╕╨╡ ╨╜╨╡╨░╨║╤В╨╕╨▓╨╜╨╛╨│╨╛ ╨▒╨╛╤В╨░ ╨▓ ╨╕╤Б╤В╨╛╤А╨╕╤О
                                # log_bot_stop(symbol, f"╨Э╨╡╨░╨║╤В╨╕╨▓╨╡╨╜ {time_since_update//60:.0f} ╨╝╨╕╨╜ (╤Б╤В╨░╤В╤Г╤Б: {bot_status})")  # TODO: ╨д╤Г╨╜╨║╤Ж╨╕╤П ╨╜╨╡ ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜╨░
                            else:
                                pass
                                continue  # ╨С╨╛╤В ╨░╨║╤В╨╕╨▓╨╡╨╜ - ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╤Г╨┤╨░╨╗╨╡╨╜╨╕╨╡
                        except Exception as e:
                            logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨░╤А╤Б╨╕╨╜╨│╨░ ╨▓╤А╨╡╨╝╨╡╨╜╨╕ ╨┤╨╗╤П {symbol}: {e}, ╨╖╨╜╨░╤З╨╡╨╜╨╕╨╡='{last_update_str}'")
                            # ╨Х╤Б╨╗╨╕ ╨╜╨╡ ╨╝╨╛╨╢╨╡╨╝ ╤А╨░╤Б╨┐╨░╤А╤Б╨╕╤В╤М ╨▓╤А╨╡╨╝╤П - ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ created_at ╨╜╨╕╨╢╨╡
                            last_update_str = None
                else:
                    # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Х╤Б╨╗╨╕ ╨╜╨╡╤В last_update, ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ created_at
                    # ╨б╨▓╨╡╨╢╨╡╤Б╨╛╨╖╨┤╨░╨╜╨╜╤Л╨╡ ╨▒╨╛╤В╤Л ╨╜╨╡ ╨┤╨╛╨╗╨╢╨╜╤Л ╤Г╨┤╨░╨╗╤П╤В╤М╤Б╤П!
                    created_at_str = bot_data.get('created_at')
                    if created_at_str:
                        try:
                            created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                            time_since_creation = current_time - created_at.timestamp()
                            
                            if time_since_creation < 300:  # 5 ╨╝╨╕╨╜╤Г╤В
                                logger.info(f" тП│ ╨С╨╛╤В {symbol} ╤Б╨╛╨╖╨┤╨░╨╜ {time_since_creation//60:.0f} ╨╝╨╕╨╜ ╨╜╨░╨╖╨░╨┤, ╨╜╨╡╤В last_update - ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╤Г╨┤╨░╨╗╨╡╨╜╨╕╨╡")
                                continue
                            else:
                                logger.warning(f" тП░ ╨С╨╛╤В {symbol} ╨▒╨╡╨╖ last_update ╨╕ ╤Б╨╛╨╖╨┤╨░╨╜ {time_since_creation//60:.0f} ╨╝╨╕╨╜ ╨╜╨░╨╖╨░╨┤ - ╤Г╨┤╨░╨╗╤П╨╡╨╝")
                                bots_to_remove.append(symbol)
                        except Exception as e:
                            logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨░╤А╤Б╨╕╨╜╨│╨░ created_at ╨┤╨╗╤П {symbol}: {e}")
                            # ╨Х╤Б╨╗╨╕ ╨╜╨╡ ╨╝╨╛╨╢╨╡╨╝ ╤А╨░╤Б╨┐╨░╤А╤Б╨╕╤В╤М, ╨Э╨Х ╨г╨Ф╨Р╨Ы╨п╨Х╨Ь (╨▒╨╡╨╖╨╛╨┐╨░╤Б╨╜╨╡╨╡)
                            logger.warning(f" тЪая╕П ╨С╨╛╤В {symbol} ╨▒╨╡╨╖ ╨▓╤А╨╡╨╝╨╡╨╜╨╕ - ╨Э╨Х ╨г╨Ф╨Р╨Ы╨п╨Х╨Ь ╨┤╨╗╤П ╨▒╨╡╨╖╨╛╨┐╨░╤Б╨╜╨╛╤Б╤В╨╕")
                    else:
                        # ╨Э╨╡╤В ╨╜╨╕ last_update, ╨╜╨╕ created_at - ╨╛╤З╨╡╨╜╤М ╤Б╤В╤А╨░╨╜╨╜╨░╤П ╤Б╨╕╤В╤Г╨░╤Ж╨╕╤П
                        logger.warning(f" тЪая╕П ╨С╨╛╤В {symbol} ╨▒╨╡╨╖ ╨▓╤А╨╡╨╝╨╡╨╜╨╕ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П ╨╕ ╤Б╨╛╨╖╨┤╨░╨╜╨╕╤П - ╨Э╨Х ╨г╨Ф╨Р╨Ы╨п╨Х╨Ь ╨┤╨╗╤П ╨▒╨╡╨╖╨╛╨┐╨░╤Б╨╜╨╛╤Б╤В╨╕")
            
            # ╨г╨┤╨░╨╗╤П╨╡╨╝ ╨╜╨╡╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓
            for symbol in bots_to_remove:
                bot_data = bots_data['bots'][symbol]
                logger.info(f" ЁЯЧСя╕П ╨г╨┤╨░╨╗╨╡╨╜╨╕╨╡ ╨╜╨╡╨░╨║╤В╨╕╨▓╨╜╨╛╨│╨╛ ╨▒╨╛╤В╨░ {symbol} (╤Б╤В╨░╤В╤Г╤Б: {bot_data.get('status')})")
                
                # тЬЕ ╨г╨Ф╨Р╨Ы╨п╨Х╨Ь ╨Я╨Ю╨Ч╨Ш╨ж╨Ш╨о ╨Ш╨Ч ╨а╨Х╨Х╨б╨в╨а╨Р ╨Я╨а╨Ш ╨г╨Ф╨Р╨Ы╨Х╨Э╨Ш╨Ш ╨Э╨Х╨Р╨Ъ╨в╨Ш╨Т╨Э╨Ю╨У╨Ю ╨С╨Ю╨в╨Р
                try:
                    from bots_modules.imports_and_globals import unregister_bot_position
                    position = bot_data.get('position')
                    if position and position.get('order_id'):
                        order_id = position['order_id']
                        unregister_bot_position(order_id)
                        logger.info(f" тЬЕ ╨Я╨╛╨╖╨╕╤Ж╨╕╤П ╤Г╨┤╨░╨╗╨╡╨╜╨░ ╨╕╨╖ ╤А╨╡╨╡╤Б╤В╤А╨░ ╨┐╤А╨╕ ╤Г╨┤╨░╨╗╨╡╨╜╨╕╨╕ ╨╜╨╡╨░╨║╤В╨╕╨▓╨╜╨╛╨│╨╛ ╨▒╨╛╤В╨░ {symbol}: order_id={order_id}")
                    else:
                        logger.info(f" тД╣я╕П ╨г ╨╜╨╡╨░╨║╤В╨╕╨▓╨╜╨╛╨│╨╛ ╨▒╨╛╤В╨░ {symbol} ╨╜╨╡╤В ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨▓ ╤А╨╡╨╡╤Б╤В╤А╨╡")
                except Exception as registry_error:
                    logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Г╨┤╨░╨╗╨╡╨╜╨╕╤П ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╕╨╖ ╤А╨╡╨╡╤Б╤В╤А╨░ ╨┤╨╗╤П ╨▒╨╛╤В╨░ {symbol}: {registry_error}")
                    # ╨Э╨╡ ╨▒╨╗╨╛╨║╨╕╤А╤Г╨╡╨╝ ╤Г╨┤╨░╨╗╨╡╨╜╨╕╨╡ ╨▒╨╛╤В╨░ ╨╕╨╖-╨╖╨░ ╨╛╤И╨╕╨▒╨║╨╕ ╤А╨╡╨╡╤Б╤В╤А╨░
                
                del bots_data['bots'][symbol]
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f" тЬЕ ╨г╨┤╨░╨╗╨╡╨╜╨╛ {removed_count} ╨╜╨╡╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓")
            # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡
            save_bots_state()
        else:
            logger.info(f" тЬЕ ╨Э╨╡╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓ ╨┤╨╗╤П ╤Г╨┤╨░╨╗╨╡╨╜╨╕╤П ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜╨╛")
        
        return removed_count > 0
        
    except Exception as e:
        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╛╤З╨╕╤Б╤В╨║╨╕ ╨╜╨╡╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓: {e}")
        return False

# ╨г╨Ф╨Р╨Ы╨Х╨Э╨Ю: cleanup_mature_coins_without_trades()
# ╨Ч╤А╨╡╨╗╨╛╤Б╤В╤М ╨╝╨╛╨╜╨╡╤В╤Л ╨╜╨╡╨╛╨▒╤А╨░╤В╨╕╨╝╨░ - ╨╡╤Б╨╗╨╕ ╨╝╨╛╨╜╨╡╤В╨░ ╤Б╤В╨░╨╗╨░ ╨╖╤А╨╡╨╗╨╛╨╣, ╨╛╨╜╨░ ╨╜╨╡ ╨╝╨╛╨╢╨╡╤В ╤Б╤В╨░╤В╤М ╨╜╨╡╨╖╤А╨╡╨╗╨╛╨╣!
# ╨д╨░╨╣╨╗ ╨╖╤А╨╡╨╗╤Л╤Е ╨╝╨╛╨╜╨╡╤В ╨╝╨╛╨╢╨╜╨╛ ╤В╨╛╨╗╤М╨║╨╛ ╨┤╨╛╨┐╨╛╨╗╨╜╤П╤В╤М ╨╜╨╛╨▓╤Л╨╝╨╕, ╨╜╨╛ ╨╜╨╡ ╨╛╤З╨╕╤Й╨░╤В╤М ╨╛╤В ╤Б╤В╨░╤А╤Л╤Е

def remove_mature_coins(coins_to_remove):
    """
    ╨г╨┤╨░╨╗╤П╨╡╤В ╨║╨╛╨╜╨║╤А╨╡╤В╨╜╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л ╨╕╨╖ ╤Д╨░╨╣╨╗╨░ ╨╖╤А╨╡╨╗╤Л╤Е ╨╝╨╛╨╜╨╡╤В
    
    Args:
        coins_to_remove: ╤Б╨┐╨╕╤Б╨╛╨║ ╤Б╨╕╨╝╨▓╨╛╨╗╨╛╨▓ ╨╝╨╛╨╜╨╡╤В ╨┤╨╗╤П ╤Г╨┤╨░╨╗╨╡╨╜╨╕╤П (╨╜╨░╨┐╤А╨╕╨╝╨╡╤А: ['ARIA', 'AVNT'])
    
    Returns:
        dict: ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В ╨╛╨┐╨╡╤А╨░╤Ж╨╕╨╕ ╤Б ╨║╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╛╨╝ ╤Г╨┤╨░╨╗╨╡╨╜╨╜╤Л╤Е ╨╝╨╛╨╜╨╡╤В
    """
    try:
        if not isinstance(coins_to_remove, list):
            coins_to_remove = [coins_to_remove]
        
        removed_count = 0
        not_found = []
        
        logger.info(f"[MATURE_REMOVE] ЁЯЧСя╕П ╨Ч╨░╨┐╤А╨╛╤Б ╨╜╨░ ╤Г╨┤╨░╨╗╨╡╨╜╨╕╨╡ ╨╝╨╛╨╜╨╡╤В: {coins_to_remove}")
        
        with mature_coins_lock:
            for symbol in coins_to_remove:
                if symbol in mature_coins_storage:
                    del mature_coins_storage[symbol]
                    removed_count += 1
                    logger.info(f"[MATURE_REMOVE] тЬЕ ╨г╨┤╨░╨╗╨╡╨╜╨░ ╨╝╨╛╨╜╨╡╤В╨░ {symbol} ╨╕╨╖ ╨╖╤А╨╡╨╗╤Л╤Е")
                else:
                    not_found.append(symbol)
                    logger.warning(f"[MATURE_REMOVE] тЪая╕П ╨Ь╨╛╨╜╨╡╤В╨░ {symbol} ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜╨░ ╨▓ ╨╖╤А╨╡╨╗╤Л╤Е")
        
        # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П
        if removed_count > 0:
            save_mature_coins_storage()
            logger.info(f"[MATURE_REMOVE] ЁЯТ╛ ╨б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╛ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨╖╤А╨╡╨╗╤Л╤Е ╨╝╨╛╨╜╨╡╤В")
        
        return {
            'success': True,
            'removed_count': removed_count,
            'removed_coins': [coin for coin in coins_to_remove if coin not in not_found],
            'not_found': not_found,
            'message': f'╨г╨┤╨░╨╗╨╡╨╜╨╛ {removed_count} ╨╝╨╛╨╜╨╡╤В ╨╕╨╖ ╨╖╤А╨╡╨╗╤Л╤Е'
        }
        
    except Exception as e:
        logger.error(f"[MATURE_REMOVE] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Г╨┤╨░╨╗╨╡╨╜╨╕╤П ╨╝╨╛╨╜╨╡╤В: {e}")
        return {
            'success': False,
            'error': str(e),
            'removed_count': 0
        }

def check_trading_rules_activation():
    """╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╤В ╨╕ ╨░╨║╤В╨╕╨▓╨╕╤А╤Г╨╡╤В ╨┐╤А╨░╨▓╨╕╨╗╨░ ╤В╨╛╤А╨│╨╛╨▓╨╗╨╕ ╨┤╨╗╤П ╨╖╤А╨╡╨╗╤Л╤Е ╨╝╨╛╨╜╨╡╤В"""
    try:
        # ╨Ъ╨а╨Ш╨в╨Ш╨з╨Х╨б╨Ъ╨Р╨п ╨Я╨а╨Ю╨Т╨Х╨а╨Ъ╨Р: Auto Bot ╨┤╨╛╨╗╨╢╨╡╨╜ ╨▒╤Л╤В╤М ╨▓╨║╨╗╤О╤З╨╡╨╜ ╨┤╨╗╤П ╨░╨▓╤В╨╛╨╝╨░╤В╨╕╤З╨╡╤Б╨║╨╛╨│╨╛ ╤Б╨╛╨╖╨┤╨░╨╜╨╕╤П ╨▒╨╛╤В╨╛╨▓
        with bots_data_lock:
            auto_bot_enabled = bots_data.get('auto_bot_config', {}).get('enabled', False)
        
        if not auto_bot_enabled:
            logger.info(f" тП╣я╕П Auto Bot ╨▓╤Л╨║╨╗╤О╤З╨╡╨╜ - ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╨░╨║╤В╨╕╨▓╨░╤Ж╨╕╤О ╨┐╤А╨░╨▓╨╕╨╗ ╤В╨╛╤А╨│╨╛╨▓╨╗╨╕")
            return False
        
        current_time = time.time()
        activated_count = 0
        
        logger.info(f" ЁЯФН ╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╨░╨║╤В╨╕╨▓╨░╤Ж╨╕╨╕ ╨┐╤А╨░╨▓╨╕╨╗ ╤В╨╛╤А╨│╨╛╨▓╨╗╨╕ ╨┤╨╗╤П ╨╖╤А╨╡╨╗╤Л╤Е ╨╝╨╛╨╜╨╡╤В")
        
        # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ш╨Х: ╨Э╨Х ╤Б╨╛╨╖╨┤╨░╨╡╨╝ ╨▒╨╛╤В╨╛╨▓ ╨░╨▓╤В╨╛╨╝╨░╤В╨╕╤З╨╡╤Б╨║╨╕ ╨┤╨╗╤П ╨▓╤Б╨╡╤Е ╨╖╤А╨╡╨╗╤Л╤Е ╨╝╨╛╨╜╨╡╤В!
        # ╨Т╨╝╨╡╤Б╤В╨╛ ╤Н╤В╨╛╨│╨╛ ╨┐╤А╨╛╤Б╤В╨╛ ╨╛╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨▓╤А╨╡╨╝╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨▓ mature_coins_storage
        
        with mature_coins_lock:
            for symbol, coin_data in mature_coins_storage.items():
                last_verified = coin_data.get('last_verified', 0)
                time_since_verification = current_time - last_verified
                
                # ╨Х╤Б╨╗╨╕ ╨╝╨╛╨╜╨╡╤В╨░ ╨╖╤А╨╡╨╗╨░╤П ╨╕ ╨╜╨╡ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╗╨░╤Б╤М ╨▒╨╛╨╗╨╡╨╡ 5 ╨╝╨╕╨╜╤Г╤В, ╨╛╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨▓╤А╨╡╨╝╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕
                if time_since_verification > 300:  # 5 ╨╝╨╕╨╜╤Г╤В
                    # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨▓╤А╨╡╨╝╤П ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╡╨╣ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕
                    coin_data['last_verified'] = current_time
                    activated_count += 1
        
        if activated_count > 0:
            logger.info(f" тЬЕ ╨Ю╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╛ ╨▓╤А╨╡╨╝╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨┤╨╗╤П {activated_count} ╨╖╤А╨╡╨╗╤Л╤Е ╨╝╨╛╨╜╨╡╤В")
            # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╜╤Л╨╡ ╨┤╨░╨╜╨╜╤Л╨╡ ╨╖╤А╨╡╨╗╤Л╤Е ╨╝╨╛╨╜╨╡╤В
            save_mature_coins_storage()
        else:
            logger.info(f" тЬЕ ╨Э╨╡╤В ╨╖╤А╨╡╨╗╤Л╤Е ╨╝╨╛╨╜╨╡╤В ╨┤╨╗╤П ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П ╨▓╤А╨╡╨╝╨╡╨╜╨╕ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕")
        
        return activated_count > 0
        
    except Exception as e:
        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨░╨║╤В╨╕╨▓╨░╤Ж╨╕╨╕ ╨┐╤А╨░╨▓╨╕╨╗ ╤В╨╛╤А╨│╨╛╨▓╨╗╨╕: {e}")
        return False

def check_missing_stop_losses():
    """╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╤В ╨╕ ╤Г╤Б╤В╨░╨╜╨░╨▓╨╗╨╕╨▓╨░╨╡╤В ╨╜╨╡╨┤╨╛╤Б╤В╨░╤О╤Й╨╕╨╡ ╤Б╤В╨╛╨┐-╨╗╨╛╤Б╤Б╤Л ╨╕ ╤В╤А╨╡╨╣╨╗╨╕╨╜╨│ ╤Б╤В╨╛╨┐╤Л ╨┤╨╗╤П ╨▒╨╛╤В╨╛╨▓."""
    try:
        if not ensure_exchange_initialized():
            logger.error(" тЭМ ╨С╨╕╤А╨╢╨░ ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░")
            return False

        current_exchange = get_exchange() or exchange
        if not current_exchange:
            logger.error(" тЭМ ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╨╛╨▒╤К╨╡╨║╤В ╨▒╨╕╤А╨╢╨╕")
            return False

        auto_config, bots_snapshot = _snapshot_bots_for_protections()
        if not bots_snapshot:
            pass
            return True

        # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ш╨Х: ╨Я╤А╨░╨▓╨╕╨╗╤М╨╜╨░╤П ╨╜╨╛╤А╨╝╨░╨╗╨╕╨╖╨░╤Ж╨╕╤П ╤Б╨╕╨╝╨▓╨╛╨╗╨╛╨▓ ╨╕ ╤Д╨╕╨╗╤М╤В╤А╨░╤Ж╨╕╤П ╤В╨╛╨╗╤М╨║╨╛ ╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣
        def normalize_symbol(symbol):
            """╨Э╨╛╤А╨╝╨░╨╗╨╕╨╖╤Г╨╡╤В ╤Б╨╕╨╝╨▓╨╛╨╗, ╤Г╨▒╨╕╤А╨░╤П USDT ╤Б╤Г╤Д╤Д╨╕╨║╤Б ╨╡╤Б╨╗╨╕ ╨╡╤Б╤В╤М"""
            if symbol and symbol.endswith('USDT'):
                return symbol[:-4]  # ╨г╨▒╨╕╤А╨░╨╡╨╝ 'USDT'
            return symbol
        
        # ╨Ш╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╤Г╨╡╨╝ ╨┐╨╡╤А╨╡╨╝╨╡╨╜╨╜╤Л╨╡ ╨┤╨╗╤П ╨┤╨╛╨┐╨╛╨╗╨╜╨╕╤В╨╡╨╗╤М╨╜╨╛╨╣ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕
        _raw_positions_for_check = []
        exchange_positions = {}
        
        try:
            # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Х╨б╨Ъ╨Ю╨Х ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ш╨Х: ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ exchange.get_positions() ╨▓╨╝╨╡╤Б╤В╨╛ client.get_positions()
            # exchange.get_positions() ╨╛╨▒╤А╨░╨▒╨░╤В╤Л╨▓╨░╨╡╤В ╨┐╨░╨│╨╕╨╜╨░╤Ж╨╕╤О ╨╕ ╨▓╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╤В ╨Т╨б╨Х ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ (╨║╨░╨║ ╨▓ app.py)
            positions_result = current_exchange.get_positions()
            if isinstance(positions_result, tuple):
                processed_positions, rapid_growth = positions_result
            else:
                processed_positions = positions_result if positions_result else []
            
            # тЬЕ ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨б╨л╨а╨л╨Х ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨░╨┐╤А╤П╨╝╤Г╤О ╤З╨╡╤А╨╡╨╖ API ╨┤╨╗╤П ╨┤╨╡╤В╨░╨╗╤М╨╜╨╛╨╣ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕
            # ╨Э╨╛ ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╛╨▒╤А╨░╨▒╨╛╤В╨░╨╜╨╜╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╕╨╖ exchange.get_positions() ╨║╨░╨║ ╨╛╤Б╨╜╨╛╨▓╨╜╨╛╨╣ ╨╕╤Б╤В╨╛╤З╨╜╨╕╨║
            try:
                positions_response = current_exchange.client.get_positions(
                    category="linear",
                    settleCoin="USDT",
                    limit=100
                )
                if positions_response.get('retCode') == 0:
                    # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨▓╤Б╨╡ ╤Б╤В╤А╨░╨╜╨╕╤Ж╤Л ╨┤╨╗╤П ╤Б╤Л╤А╤Л╤Е ╨┤╨░╨╜╨╜╤Л╤Е
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
                    # ╨Х╤Б╨╗╨╕ ╨╜╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╤Б╤Л╤А╤Л╨╡ ╨┤╨░╨╜╨╜╤Л╨╡, ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╛╨▒╤А╨░╨▒╨╛╤В╨░╨╜╨╜╤Л╨╡
                    logger.warning(f" тЪая╕П ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╤Б╤Л╤А╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕, ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╛╨▒╤А╨░╨▒╨╛╤В╨░╨╜╨╜╤Л╨╡")
                    raw_positions = []
                    for pos in processed_positions:
                        # ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╤Б╤Л╤А╨╛╨╣ ╤Д╨╛╤А╨╝╨░╤В ╨╕╨╖ ╨╛╨▒╤А╨░╨▒╨╛╤В╨░╨╜╨╜╨╛╨│╨╛
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
                logger.warning(f" тЪая╕П ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╤Б╤Л╤А╤Л╤Е ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣: {raw_error}, ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╛╨▒╤А╨░╨▒╨╛╤В╨░╨╜╨╜╤Л╨╡")
                raw_positions = []
            
            _raw_positions_for_check = raw_positions  # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨┤╨╗╤П ╨┤╨╛╨┐╨╛╨╗╨╜╨╕╤В╨╡╨╗╤М╨╜╨╛╨╣ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕
            
            # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ш╨Х: ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╛╨▒╤А╨░╨▒╨╛╤В╨░╨╜╨╜╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╕╨╖ exchange.get_positions()
            # ╨Ю╨╜╨╕ ╤Г╨╢╨╡ ╨╜╨╛╤А╨╝╨░╨╗╨╕╨╖╨╛╨▓╨░╨╜╤Л ╨╕ ╤Б╨╛╨┤╨╡╤А╨╢╨░╤В ╨▓╤Б╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ (╤Б ╨┐╨░╨│╨╕╨╜╨░╤Ж╨╕╨╡╨╣)
            exchange_positions = {}
            all_positions_dict = {}
            
            # ╨б╨╜╨░╤З╨░╨╗╨░ ╨╖╨░╨┐╨╛╨╗╨╜╤П╨╡╨╝ ╨╕╨╖ ╨╛╨▒╤А╨░╨▒╨╛╤В╨░╨╜╨╜╤Л╤Е ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ (╨╛╤Б╨╜╨╛╨▓╨╜╨╛╨╣ ╨╕╤Б╤В╨╛╤З╨╜╨╕╨║)
            # ╨Т processed_positions ╤Б╨╕╨╝╨▓╨╛╨╗╤Л ╤Г╨╢╨╡ ╨╜╨╛╤А╨╝╨░╨╗╨╕╨╖╨╛╨▓╨░╨╜╤Л (╨▒╨╡╨╖ USDT) ╤З╨╡╤А╨╡╨╖ clean_symbol()
            for position in processed_positions:
                symbol = position.get('symbol', '')
                position_size = abs(float(position.get('size', 0) or 0))
                
                if symbol:
                    # ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╤Д╨╛╤А╨╝╨░╤В ╨┤╨╗╤П ╤Б╨╛╨▓╨╝╨╡╤Б╤В╨╕╨╝╨╛╤Б╤В╨╕ ╤Б ╤Б╤Л╤А╤Л╨╝╨╕ ╨┤╨░╨╜╨╜╤Л╨╝╨╕
                    # ╨Т processed_positions side ╨╝╨╛╨╢╨╡╤В ╨▒╤Л╤В╤М 'Long'/'Short', ╨║╨╛╨╜╨▓╨╡╤А╤В╨╕╤А╤Г╨╡╨╝ ╨▓ 'Buy'/'Sell'
                    side_str = position.get('side', '')
                    if side_str.upper() == 'LONG':
                        side_api = 'Buy'
                    elif side_str.upper() == 'SHORT':
                        side_api = 'Sell'
                    else:
                        side_api = 'Buy'  # ╨Я╨╛ ╤Г╨╝╨╛╨╗╤З╨░╨╜╨╕╤О
                    
                    raw_format_position = {
                        'symbol': symbol + 'USDT',  # ╨Ф╨╛╨▒╨░╨▓╨╗╤П╨╡╨╝ USDT ╨┤╨╗╤П ╤Б╨╛╨▓╨╝╨╡╤Б╤В╨╕╨╝╨╛╤Б╤В╨╕
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
                    
                    # тЬЕ ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╨░╨║╤В╨╕╨▓╨╜╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ (size > 0)
                    if position_size > 0:
                        exchange_positions[symbol] = raw_format_position
            
            # ╨Ф╨╛╨┐╨╛╨╗╨╜╨╕╤В╨╡╨╗╤М╨╜╨╛ ╨┤╨╛╨▒╨░╨▓╨╗╤П╨╡╨╝ ╨╕╨╖ ╤Б╤Л╤А╤Л╤Е ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ (╨╜╨░ ╤Б╨╗╤Г╤З╨░╨╣, ╨╡╤Б╨╗╨╕ ╤З╤В╨╛-╤В╨╛ ╨┐╤А╨╛╨┐╤Г╤Й╨╡╨╜╨╛)
            for position in raw_positions:
                raw_symbol = position.get('symbol', '')
                position_size = abs(float(position.get('size', 0) or 0))
                normalized_symbol = normalize_symbol(raw_symbol)
                
                if normalized_symbol and normalized_symbol not in exchange_positions:
                    # ╨Ф╨╛╨▒╨░╨▓╨╗╤П╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╨╡╤Б╨╗╨╕ ╨╡╤Й╨╡ ╨╜╨╡╤В ╨▓ ╤Б╨╗╨╛╨▓╨░╤А╨╡
                    if position_size > 0:
                        exchange_positions[normalized_symbol] = position
                    all_positions_dict[normalized_symbol] = position
            
        except Exception as e:
            logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╤Б ╨▒╨╕╤А╨╢╨╕: {e}")
            return False

        from bots_modules.bot_class import NewTradingBot

        updated_count = 0
        failed_count = 0

        for symbol, bot_snapshot in bots_snapshot.items():
            try:
                pos = exchange_positions.get(symbol)
                if not pos:
                    # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Х╨б╨Ъ╨Р╨п ╨Я╨а╨Ю╨Т╨Х╨а╨Ъ╨Р: ╨Я╨╡╤А╨╡╨┤ ╤Г╨┤╨░╨╗╨╡╨╜╨╕╨╡╨╝ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┐╨╛╨╖╨╕╤Ж╨╕╤О ╨╜╨░╨┐╤А╤П╨╝╤Г╤О ╤З╨╡╤А╨╡╨╖ API
                    logger.warning(f" тЪая╕П ╨Я╨╛╨╖╨╕╤Ж╨╕╤П {symbol} ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜╨░ ╨▓ ╤Б╨╗╨╛╨▓╨░╤А╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣. ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╜╨░╨┐╤А╤П╨╝╤Г╤О ╤З╨╡╤А╨╡╨╖ API...")
                    
                    # ╨Ф╨╛╨┐╨╛╨╗╨╜╨╕╤В╨╡╨╗╤М╨╜╨░╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ - ╨╖╨░╨┐╤А╨░╤И╨╕╨▓╨░╨╡╨╝ ╨┐╨╛╨╖╨╕╤Ж╨╕╤О ╨╜╨░╨┐╤А╤П╨╝╤Г╤О
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
                                logger.info(f" тЬЕ ╨Я╨╛╨╖╨╕╤Ж╨╕╤П {symbol} ╨╜╨░╨╣╨┤╨╡╨╜╨░ ╨┐╤А╨╕ ╨┐╤А╤П╨╝╨╛╨╣ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╡! raw='{raw_symbol}', normalized='{normalized}', ╤А╨░╨╖╨╝╨╡╤А: {position_size}")
                                # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╤Б╨╗╨╛╨▓╨░╤А╤М ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣
                                exchange_positions[symbol] = raw_pos
                                pos = raw_pos
                                break
                        
                        if not direct_check:
                            # ╨Я╤А╨╛╨▒╤Г╨╡╨╝ ╨╜╨░╨╣╤В╨╕ ╤Б ╤Г╤З╨╡╤В╨╛╨╝ ╨▓╨╛╨╖╨╝╨╛╨╢╨╜╤Л╤Е ╨▓╨░╤А╨╕╨░╨╜╤В╨╛╨▓ ╤Б╨╕╨╝╨▓╨╛╨╗╨░
                            logger.error(f" тЭМ ╨Я╨╛╨╖╨╕╤Ж╨╕╤П {symbol} ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜╨░ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ ╨┐╨╛╤Б╨╗╨╡ ╨┐╤А╤П╨╝╨╛╨╣ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕")
                            
                            # ╨Я╤А╨╛╨▒╤Г╨╡╨╝ ╨╜╨░╨╣╤В╨╕ ╨▓╨░╤А╨╕╨░╨╜╤В╤Л: symbol, symbolUSDT, USDTsymbol
                            possible_symbols = [symbol, f"{symbol}USDT", f"USDT{symbol}"]
                            found_variants = []
                            for raw_pos in _raw_positions_for_check:
                                raw_symbol = raw_pos.get('symbol', '')
                                position_size = abs(float(raw_pos.get('size', 0) or 0))
                                if position_size > 0 and raw_symbol in possible_symbols:
                                    found_variants.append(f"raw='{raw_symbol}' (size={position_size})")
                            
                            if found_variants:
                                logger.warning(f" тЪая╕П ╨Э╨░╨╣╨┤╨╡╨╜╤Л ╨▓╨░╤А╨╕╨░╨╜╤В╤Л ╤Б╨╕╨╝╨▓╨╛╨╗╨░ {symbol} ╨╜╨░ ╨▒╨╕╤А╨╢╨╡: {found_variants}")
                                logger.warning(f" тЪая╕П ╨Т╨╛╨╖╨╝╨╛╨╢╨╜╨╛, ╨┐╤А╨╛╨▒╨╗╨╡╨╝╨░ ╨▓ ╨╜╨╛╤А╨╝╨░╨╗╨╕╨╖╨░╤Ж╨╕╨╕ ╤Б╨╕╨╝╨▓╨╛╨╗╨╛╨▓!")
                            
                            logger.error(f" тЭМ ╨Ф╨╛╤Б╤В╤Г╨┐╨╜╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ (normalized): {sorted([normalize_symbol(p.get('symbol', '')) for p in _raw_positions_for_check if abs(float(p.get('size', 0) or 0)) > 0])}")
                            logger.error(f" тЭМ ╨Ф╨╛╤Б╤В╤Г╨┐╨╜╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ (raw): {sorted([p.get('symbol', '') for p in _raw_positions_for_check if abs(float(p.get('size', 0) or 0)) > 0])}")
                            # ╨Э╨Х ╨г╨Ф╨Р╨Ы╨п╨Х╨Ь ╨▒╨╛╤В╨░, ╨╡╤Б╨╗╨╕ ╨╜╨╡ ╤Г╨▓╨╡╤А╨╡╨╜╤Л - ╨┐╤А╨╛╤Б╤В╨╛ ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝
                            logger.warning(f" тЪая╕П ╨Я╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╨▒╨╛╤В╨░ {symbol} - ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜╨░, ╨╜╨╛ ╨╜╨╡ ╤Г╨┤╨░╨╗╤П╨╡╨╝ ╨┤╨╗╤П ╨▒╨╡╨╖╨╛╨┐╨░╤Б╨╜╨╛╤Б╤В╨╕")
                            continue
                    except Exception as check_error:
                        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╤П╨╝╨╛╨╣ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ {symbol}: {check_error}")
                        # ╨Э╨Х ╨г╨Ф╨Р╨Ы╨п╨Х╨Ь ╨▒╨╛╤В╨░ ╨┐╤А╨╕ ╨╛╤И╨╕╨▒╨║╨╡ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕
                        continue
                    
                    if not pos:
                        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Х╨б╨Ъ╨Ю╨Х ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ш╨Х: ╨Э╨Х ╨г╨Ф╨Р╨Ы╨п╨Х╨Ь ╨▒╨╛╤В╨░, ╨╡╤Б╨╗╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜╨░!
                        # ╨Я╨╛╨╖╨╕╤Ж╨╕╤П ╨╝╨╛╨╢╨╡╤В ╨▒╤Л╤В╤М ╨╜╨░ ╨▒╨╕╤А╨╢╨╡, ╨╜╨╛ ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜╨░ ╨╕╨╖-╨╖╨░ ╨┐╤А╨╛╨▒╨╗╨╡╨╝ ╤Б ╨╜╨╛╤А╨╝╨░╨╗╨╕╨╖╨░╤Ж╨╕╨╡╨╣ ╤Б╨╕╨╝╨▓╨╛╨╗╨╛╨▓
                        logger.warning(f" тЪая╕П ╨Я╨╛╨╖╨╕╤Ж╨╕╤П {symbol} ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜╨░ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ - ╨Я╨а╨Ю╨Я╨г╨б╨Ъ╨Р╨Х╨Ь (╨╜╨╡ ╤Г╨┤╨░╨╗╤П╨╡╨╝ ╨┤╨╗╤П ╨▒╨╡╨╖╨╛╨┐╨░╤Б╨╜╨╛╤Б╤В╨╕)")
                        continue

                # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ш╨Х: ╨Я╤А╨░╨▓╨╕╨╗╤М╨╜╨░╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╤А╨░╨╖╨╝╨╡╤А╨░ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ (╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ abs ╨┤╨╗╤П ╤Г╤З╨╡╤В╨░ LONG/SHORT)
                position_size = abs(_safe_float(pos.get('size'), 0.0) or 0.0)
                if position_size <= 0:
                    logger.warning(f" тЪая╕П ╨Я╨╛╨╖╨╕╤Ж╨╕╤П {symbol} ╨╖╨░╨║╤А╤Л╤В╨░ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ - ╤Г╨┤╨░╨╗╤П╨╡╨╝ ╨▒╨╛╤В╨░ ╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╤О ╨╕╨╖ ╤А╨╡╨╡╤Б╤В╤А╨░")
                    # тЬЕ ╨г╨Ф╨Р╨Ы╨п╨Х╨Ь ╨С╨Ю╨в╨Р ╨Ш ╨Я╨Ю╨Ч╨Ш╨ж╨Ш╨о ╨Ш╨Ч ╨а╨Х╨Х╨б╨в╨а╨Р, ╨╡╤Б╨╗╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╨╖╨░╨║╤А╤Л╤В╨░ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
                    try:
                        from bots_modules.imports_and_globals import unregister_bot_position
                        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ order_id ╨╕╨╖ ╨▒╨╛╤В╨░
                        order_id = None
                        position = bot_snapshot.get('position')
                        if position and position.get('order_id'):
                            order_id = position['order_id']
                        elif bot_snapshot.get('restoration_order_id'):
                            order_id = bot_snapshot.get('restoration_order_id')
                        
                        # ╨г╨┤╨░╨╗╤П╨╡╨╝ ╨┐╨╛╨╖╨╕╤Ж╨╕╤О ╨╕╨╖ ╤А╨╡╨╡╤Б╤В╤А╨░
                        if order_id:
                            unregister_bot_position(order_id)
                            logger.info(f" тЬЕ ╨Я╨╛╨╖╨╕╤Ж╨╕╤П {symbol} (order_id={order_id}) ╤Г╨┤╨░╨╗╨╡╨╜╨░ ╨╕╨╖ ╤А╨╡╨╡╤Б╤В╤А╨░")
                        
                        # ╨г╨┤╨░╨╗╤П╨╡╨╝ ╨▒╨╛╤В╨░ ╨╕╨╖ ╤Б╨╕╤Б╤В╨╡╨╝╤Л
                        bot_removed = False
                        with bots_data_lock:
                            if symbol in bots_data['bots']:
                                del bots_data['bots'][symbol]
                                logger.info(f" тЬЕ ╨С╨╛╤В {symbol} ╤Г╨┤╨░╨╗╨╡╨╜ ╨╕╨╖ ╤Б╨╕╤Б╤В╨╡╨╝╤Л")
                                bot_removed = True
                        # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨┐╨╛╤Б╨╗╨╡ ╨╛╤Б╨▓╨╛╨▒╨╛╨╢╨┤╨╡╨╜╨╕╤П ╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨╕
                        if bot_removed:
                            save_bots_state()
                    except Exception as cleanup_error:
                        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Г╨┤╨░╨╗╨╡╨╜╨╕╤П ╨▒╨╛╤В╨░ {symbol}: {cleanup_error}")
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
                    f" ЁЯУК {symbol}: PnL {profit_percent:.2f}%, ╤В╨╡╨║╤Г╤Й╨░╤П {current_price}, ╨▓╤Е╨╛╨┤ {entry_price}"
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
                # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ю: ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨╖╨░╤Й╨╕╤В╨╜╤Л╨╡ ╨╝╨╡╤Е╨░╨╜╨╕╨╖╨╝╤Л (╨▓╨║╨╗╤О╤З╨░╤П break-even ╤Б╤В╨╛╨┐)
                # ╨н╤В╨╛ ╨╜╤Г╨╢╨╜╨╛ ╨┤╨╗╤П ╤Г╤Б╤В╨░╨╜╨╛╨▓╨║╨╕ break-even ╤Б╤В╨╛╨┐╨░ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ ╨┐╤А╨╕ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╨╕ ╨║╨╛╨╜╤Д╨╕╨│╨░
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
                        f" тЪая╕П Protection Engine ╤Б╨╕╨│╨╜╨░╨╗╨╕╨╖╨╕╤А╤Г╨╡╤В ╨╖╨░╨║╤А╤Л╤В╨╕╨╡ {symbol}: {decision.reason}"
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

                # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ю: ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╤Б╤В╨╛╨┐-╨╗╨╛╤Б╤Б, ╨┤╨░╨╢╨╡ ╨╡╤Б╨╗╨╕ ╨╛╨╜ ╤Г╨╢╨╡ ╤Г╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜, ╨╡╤Б╨╗╨╕ ╨╜╤Г╨╢╨╡╨╜ ╨╜╨╛╨▓╤Л╨╣ ╤Б╤В╨╛╨┐
                # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨╜╤Г╨╢╨╜╨╛ ╨╗╨╕ ╨╛╨▒╨╜╨╛╨▓╨╕╤В╤М ╤Б╤В╨╛╨┐-╨╗╨╛╤Б╤Б ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
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
                            logger.info(f" тЬЕ ╨б╤В╨╛╨┐-╨╗╨╛╤Б╤Б ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜ ╨┤╨╗╤П {symbol}: {desired_stop:.6f}")
                        else:
                            failed_count += 1
                            logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Г╤Б╤В╨░╨╜╨╛╨▓╨║╨╕ ╤Б╤В╨╛╨┐-╨╗╨╛╤Б╤Б╨░ ╨┤╨╗╤П {symbol}: {sl_response}")
                    except Exception as e:
                        failed_count += 1
                        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Г╤Б╤В╨░╨╜╨╛╨▓╨║╨╕ ╤Б╤В╨╛╨┐-╨╗╨╛╤Б╤Б╨░ ╨┤╨╗╤П {symbol}: {e}")

                desired_take = _select_take_profit_price(
                    position_side,
                    entry_price,
                    protection_config,
                    bot_instance.trailing_take_profit_price,
                )
                existing_take_value = _safe_float(existing_take_profit)

                # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨╡╤Б╤В╤М ╨╗╨╕ ╤Г╨╢╨╡ ╤В╨╡╨╣╨║-╨┐╤А╨╛╤Д╨╕╤В ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
                if existing_take_profit and existing_take_profit.strip():
                    pass  # ╨в╨╡╨╣╨║-╨┐╤А╨╛╤Д╨╕╤В ╤Г╨╢╨╡ ╤Г╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜, ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝
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
                            logger.info(f" тЬЕ ╨в╨╡╨╣╨║-╨┐╤А╨╛╤Д╨╕╤В ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜ ╨┤╨╗╤П {symbol}: {desired_take:.6f}")
                        else:
                            failed_count += 1
                            logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Г╤Б╤В╨░╨╜╨╛╨▓╨║╨╕ ╤В╨╡╨╣╨║-╨┐╤А╨╛╤Д╨╕╤В╨░ ╨┤╨╗╤П {symbol}: {tp_response}")
                    except Exception as e:
                        failed_count += 1
                        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Г╤Б╤В╨░╨╜╨╛╨▓╨║╨╕ ╤В╨╡╨╣╨║-╨┐╤А╨╛╤Д╨╕╤В╨░ ╨┤╨╗╤П {symbol}: {e}")

                if not _update_bot_record(symbol, updates):
                    pass

            except Exception as e:
                logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╛╨▒╤А╨░╨▒╨╛╤В╨║╨╕ {symbol}: {e}")
                failed_count += 1
                continue

        if updated_count > 0 or failed_count > 0:
            logger.info(f" тЬЕ ╨г╤Б╤В╨░╨╜╨╛╨▓╨║╨░ ╨╖╨░╨▓╨╡╤А╤И╨╡╨╜╨░: ╤Г╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨╛ {updated_count}, ╨╛╤И╨╕╨▒╨╛╨║ {failed_count}")
            if updated_count > 0:
                try:
                    save_bots_state()
                except Exception as save_error:
                    logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╤П ╨▒╨╛╤В╨╛╨▓: {save_error}")

        # тЬЕ ╨б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨╕╤А╤Г╨╡╨╝ ╨▒╨╛╤В╨╛╨▓ ╤Б ╨▒╨╕╤А╨╢╨╡╨╣ - ╤Г╨┤╨░╨╗╤П╨╡╨╝ ╨▒╨╛╤В╨╛╨▓ ╨▒╨╡╨╖ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣
        try:
            sync_bots_with_exchange()
        except Exception as sync_error:
            logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨░╤Ж╨╕╨╕ ╨▒╨╛╤В╨╛╨▓ ╤Б ╨▒╨╕╤А╨╢╨╡╨╣: {sync_error}")

        return True

    except Exception as e:
        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Г╤Б╤В╨░╨╜╨╛╨▓╨║╨╕ ╤Б╤В╨╛╨┐-╨╗╨╛╤Б╤Б╨╛╨▓: {e}")
        return False

def check_startup_position_conflicts():
    """╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╤В ╨║╨╛╨╜╤Д╨╗╨╕╨║╤В╤Л ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨┐╤А╨╕ ╨╖╨░╨┐╤Г╤Б╨║╨╡ ╤Б╨╕╤Б╤В╨╡╨╝╤Л ╨╕ ╨┐╤А╨╕╨╜╤Г╨┤╨╕╤В╨╡╨╗╤М╨╜╨╛ ╨╛╤Б╤В╨░╨╜╨░╨▓╨╗╨╕╨▓╨░╨╡╤В ╨┐╤А╨╛╨▒╨╗╨╡╨╝╨╜╤Л╨╡ ╨▒╨╛╤В╤Л"""
    try:
        if not ensure_exchange_initialized():
            logger.warning(" тЪая╕П ╨С╨╕╤А╨╢╨░ ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░, ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╨┐╤А╨╛╨▓╨╡╤А╨║╤Г ╨║╨╛╨╜╤Д╨╗╨╕╨║╤В╨╛╨▓")
            return False
        
        logger.info(" ЁЯФН ╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╨║╨╛╨╜╤Д╨╗╨╕╨║╤В╨╛╨▓...")
        
        conflicts_found = 0
        bots_paused = 0
        
        with bots_data_lock:
            for bot_key, bot_data in bots_data['bots'].items():
                try:
                    bot_status = bot_data.get('status')
                    # ╨з╨╕╤Б╤В╤Л╨╣ ╤Б╨╕╨╝╨▓╨╛╨╗ ╨┤╨╗╤П API (╨▒╨╛╤В ╨╝╨╛╨╢╨╡╤В ╨▒╤Л╤В╤М ╨║╨╗╤О╤З╨╛╨╝ symbol ╨╕╨╗╨╕ symbol_side, ╨╜╨░╨┐╤А╨╕╨╝╨╡╤А BTCUSDT_LONG)
                    api_symbol = bot_data.get('symbol') or (bot_key.rsplit('_', 1)[0] if ('_LONG' in bot_key or '_SHORT' in bot_key) else bot_key)
                    symbol = api_symbol  # ╨┤╨╗╤П ╨╗╨╛╨│╨╛╨▓ ╨╕ target_symbol ╨╜╨╕╨╢╨╡

                    # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╨░╨║╤В╨╕╨▓╨╜╤Л╨╡ ╨▒╨╛╤В╤Л (╨╜╨╡ idle/paused)
                    if bot_status in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']]:
                        continue
                    # ╨б╨╕╨╝╨▓╨╛╨╗ ╨┤╨╗╤П Bybit: ╨╡╤Б╨╗╨╕ ╤Г╨╢╨╡ ╤Б USDT тАФ ╨║╨░╨║ ╨╡╤Б╤В╤М, ╨╕╨╜╨░╤З╨╡ ╨┤╨╛╨▒╨░╨▓╨╕╤В╤М USDT
                    symbol_for_api = api_symbol if (api_symbol and 'USDT' in api_symbol) else f"{api_symbol}USDT"

                    # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┐╨╛╨╖╨╕╤Ж╨╕╤О ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
                    from bots_modules.imports_and_globals import get_exchange
                    current_exchange = get_exchange() or exchange
                    positions_response = current_exchange.client.get_positions(
                        category="linear",
                        symbol=symbol_for_api
                    )
                    
                    if positions_response.get('retCode') == 0:
                        positions = positions_response['result']['list']
                        has_position = False
                        
                        # ╨д╨╕╨╗╤М╤В╤А╤Г╨╡╨╝ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤В╨╛╨╗╤М╨║╨╛ ╨┤╨╗╤П ╨╜╤Г╨╢╨╜╨╛╨│╨╛ ╤Б╨╕╨╝╨▓╨╛╨╗╨░
                        target_symbol = symbol_for_api
                        for pos in positions:
                            pos_symbol = pos.get('symbol', '')
                            if pos_symbol == target_symbol:  # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╨╜╤Г╨╢╨╜╤Л╨╣ ╤Б╨╕╨╝╨▓╨╛╨╗
                                size = float(pos.get('size', 0))
                                if abs(size) > 0:  # ╨Х╤Б╤В╤М ╨░╨║╤В╨╕╨▓╨╜╨░╤П ╨┐╨╛╨╖╨╕╤Ж╨╕╤П
                                    has_position = True
                                    side = 'LONG' if pos.get('side') == 'Buy' else 'SHORT'
                                    break
                        
                        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨║╨╛╨╜╤Д╨╗╨╕╨║╤В
                        if has_position:
                            # ╨Х╤Б╤В╤М ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
                            if bot_status in [BOT_STATUS['RUNNING']]:
                                # ╨Ъ╨Ю╨Э╨д╨Ы╨Ш╨Ъ╨в: ╨▒╨╛╤В ╨░╨║╤В╨╕╨▓╨╡╨╜, ╨╜╨╛ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╤Г╨╢╨╡ ╨╡╤Б╤В╤М ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
                                logger.warning(f" ЁЯЪи {symbol}: ╨Ъ╨Ю╨Э╨д╨Ы╨Ш╨Ъ╨в! ╨С╨╛╤В {bot_status}, ╨╜╨╛ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П {side} ╤Г╨╢╨╡ ╨╡╤Б╤В╤М ╨╜╨░ ╨▒╨╕╤А╨╢╨╡!")
                                
                                # ╨Я╤А╨╕╨╜╤Г╨┤╨╕╤В╨╡╨╗╤М╨╜╨╛ ╨╛╤Б╤В╨░╨╜╨░╨▓╨╗╨╕╨▓╨░╨╡╨╝ ╨▒╨╛╤В╨░
                                bot_data['status'] = BOT_STATUS['PAUSED']
                                bot_data['last_update'] = datetime.now().isoformat()
                                
                                conflicts_found += 1
                                bots_paused += 1
                                
                                logger.warning(f" ЁЯФ┤ {symbol}: ╨С╨╛╤В ╨┐╤А╨╕╨╜╤Г╨┤╨╕╤В╨╡╨╗╤М╨╜╨╛ ╨╛╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜ (PAUSED)")
                                
                            elif bot_status in [BOT_STATUS['IN_POSITION_LONG'], BOT_STATUS['IN_POSITION_SHORT']]:
                                # ╨Ъ╨╛╤А╤А╨╡╨║╤В╨╜╨╛╨╡ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ - ╨▒╨╛╤В ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕
                                pass
                        else:
                            # ╨Э╨╡╤В ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
                            if bot_status in [BOT_STATUS['IN_POSITION_LONG'], BOT_STATUS['IN_POSITION_SHORT']]:
                                # ╨Ъ╨Ю╨Э╨д╨Ы╨Ш╨Ъ╨в: ╨▒╨╛╤В ╨┤╤Г╨╝╨░╨╡╤В ╤З╤В╨╛ ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕, ╨╜╨╛ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨╡╤В ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
                                logger.warning(f" ЁЯЪи {symbol}: ╨Ъ╨Ю╨Э╨д╨Ы╨Ш╨Ъ╨в! ╨С╨╛╤В ╨┐╨╛╨║╨░╨╖╤Л╨▓╨░╨╡╤В ╨┐╨╛╨╖╨╕╤Ж╨╕╤О, ╨╜╨╛ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ ╨╡╤С ╨╜╨╡╤В!")
                                
                                # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨г╨Ф╨Р╨Ы╨п╨Х╨Ь ╨▒╨╛╤В╨░, ╨░ ╨╜╨╡ ╨┐╨╡╤А╨╡╨▓╨╛╨┤╨╕╨╝ ╨▓ IDLE - ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨╡╤В ╨╜╨░ ╨▒╨╕╤А╨╢╨╡!
                                with bots_data_lock:
                                    if bot_key in bots_data['bots']:
                                        del bots_data['bots'][bot_key]
                                
                                conflicts_found += 1
                                
                                logger.warning(f" ЁЯЧСя╕П {symbol}: ╨С╨╛╤В ╤Г╨┤╨░╨╗╨╡╨╜ - ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨╡╤В ╨╜╨░ ╨▒╨╕╤А╨╢╨╡")
                    else:
                        logger.warning(f" тЭМ {symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣: {positions_response.get('retMsg', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ {symbol}: {e}")
        
        if conflicts_found > 0:
            logger.warning(f" ЁЯЪи ╨Э╨░╨╣╨┤╨╡╨╜╨╛ {conflicts_found} ╨║╨╛╨╜╤Д╨╗╨╕╨║╤В╨╛╨▓, ╨╛╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨╛ {bots_paused} ╨▒╨╛╤В╨╛╨▓")
            # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╜╨╛╨╡ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡
            save_bots_state()

            # тЬЕ ╨Ф╨Ю╨Я╨Ю╨Ы╨Э╨Ш╨в╨Х╨Ы╨м╨Э╨Ю: ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨╕╤А╤Г╨╡╨╝ ╤А╨╡╨╡╤Б╤В╤А ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨▒╨╛╤В╨╛╨▓ (bot_positions_registry)
            # ╨Ъ╨╗╤О╤З ╤А╨╡╨╡╤Б╤В╤А╨░ = SYMBOL_SIDE (╨╜╨░╨┐╤А╨╕╨╝╨╡╤А BTCUSDT_LONG), ╤З╤В╨╛╨▒╤Л ╨┐╨╛ ╨╛╨┤╨╜╨╛╨╝╤Г ╤Б╨╕╨╝╨▓╨╛╨╗╤Г ╨╝╨╛╨│╨╗╨╕ ╨▒╤Л╤В╤М ╨╗╨╛╨╜╨│ ╨╕ ╤И╨╛╤А╤В.
            try:
                from bots_modules.imports_and_globals import save_bot_positions_registry

                registry = {}
                positions_list = get_exchange_positions() or []
                # ╨б╨╗╨╛╨▓╨░╤А╤М ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╤Б ╨▒╨╕╤А╨╢╨╕: ╨║╨╗╤О╤З (symbol, side) ╨┤╨╗╤П ╨┐╨╛╨┤╨┤╨╡╤А╨╢╨║╨╕ ╨╗╨╛╨╜╨│+╤И╨╛╤А╤В ╨┐╨╛ ╨╛╨┤╨╜╨╛╨╝╤Г ╤Б╨╕╨╝╨▓╨╛╨╗╤Г
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
                logger.warning(f"[SYNC_EXCHANGE] тЪая╕П ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨╕╤А╨╛╨▓╨░╤В╤М ╤А╨╡╨╡╤Б╤В╤А ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨▒╨╛╤В╨╛╨▓: {reg_err}")
        else:
            logger.info(" тЬЕ ╨Ъ╨╛╨╜╤Д╨╗╨╕╨║╤В╨╛╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜╨╛")
        
        return conflicts_found > 0
        
    except Exception as e:
        logger.error(f" тЭМ ╨Ю╨▒╤Й╨░╤П ╨╛╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨║╨╛╨╜╤Д╨╗╨╕╨║╤В╨╛╨▓: {e}")
        return False

def sync_bots_with_exchange():
    """╨б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨╕╤А╤Г╨╡╤В ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨▒╨╛╤В╨╛╨▓ ╤Б ╨╛╤В╨║╤А╤Л╤В╤Л╨╝╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П╨╝╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡"""
    import time
    start_time = time.time()
    
    try:
        # ╨г╨▒╨╕╤А╨░╨╡╨╝ ╨╗╨╕╤И╨╜╨╕╨╡ ╨╗╨╛╨│╨╕ - ╨╛╤Б╤В╨░╨▓╨╗╤П╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╨╕╤В╨╛╨│
        if not ensure_exchange_initialized():
            logger.warning("[SYNC_EXCHANGE] тЪая╕П ╨С╨╕╤А╨╢╨░ ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░, ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨░╤Ж╨╕╤О")
            return False
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨Т╨б╨Х ╨╛╤В╨║╤А╤Л╤В╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Б ╨▒╨╕╤А╨╢╨╕ (╤Б ╨┐╨░╨│╨╕╨╜╨░╤Ж╨╕╨╡╨╣)
        try:
            exchange_positions = {}
            cursor = ""
            total_positions = 0
            iteration = 0
            
            while True:
                iteration += 1
                iter_start = time.time()
                
                # ╨Ч╨░╨┐╤А╨░╤И╨╕╨▓╨░╨╡╨╝ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Б cursor ╨┤╨╗╤П ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╨▓╤Б╨╡╤Е ╤Б╤В╤А╨░╨╜╨╕╤Ж
                params = {
                    "category": "linear", 
                    "settleCoin": "USDT",
                    "limit": 200  # ╨Ь╨░╨║╤Б╨╕╨╝╤Г╨╝ ╨╖╨░ ╨╖╨░╨┐╤А╨╛╤Б
                }
                if cursor:
                    params["cursor"] = cursor
                
                from bots_modules.imports_and_globals import get_exchange
                current_exchange = get_exchange() or exchange
                
                # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤З╤В╨╛ ╨▒╨╕╤А╨╢╨░ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░
                if not current_exchange or not hasattr(current_exchange, 'client'):
                    logger.error(f"[SYNC_EXCHANGE] тЭМ ╨С╨╕╤А╨╢╨░ ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░")
                    return False
                
                # ЁЯФе ╨г╨Я╨а╨Ю╨й╨Х╨Э╨Э╨л╨Щ ╨Я╨Ю╨Ф╨е╨Ю╨Ф: ╨▒╤Л╤Б╤В╤А╤Л╨╣ ╤В╨░╨╣╨╝╨░╤Г╤В ╨╜╨░ ╤Г╤А╨╛╨▓╨╜╨╡ SDK
                positions_response = None
                timeout_seconds = 8  # ╨Ъ╨╛╤А╨╛╤В╨║╨╕╨╣ ╤В╨░╨╣╨╝╨░╤Г╤В
                max_retries = 2
                
                for retry in range(max_retries):
                    retry_start = time.time()
                    try:
                        # ╨г╤Б╤В╨░╨╜╨░╨▓╨╗╨╕╨▓╨░╨╡╨╝ ╨║╨╛╤А╨╛╤В╨║╨╕╨╣ ╤В╨░╨╣╨╝╨░╤Г╤В ╨╜╨░ ╤Г╤А╨╛╨▓╨╜╨╡ ╨║╨╗╨╕╨╡╨╜╤В╨░
                        old_timeout = getattr(current_exchange.client, 'timeout', None)
                        current_exchange.client.timeout = timeout_seconds
                        
                        positions_response = current_exchange.client.get_positions(**params)
                        
                        # ╨Т╨╛╤Б╤Б╤В╨░╨╜╨░╨▓╨╗╨╕╨▓╨░╨╡╨╝ ╤В╨░╨╣╨╝╨░╤Г╤В
                        if old_timeout is not None:
                            current_exchange.client.timeout = old_timeout
                        
                        break  # ╨г╤Б╨┐╨╡╤Е!
                        
                    except Exception as e:
                        pass
                        if retry < max_retries - 1:
                            time.sleep(2)
                        else:
                            logger.error(f"[SYNC_EXCHANGE] тЭМ ╨Т╤Б╨╡ ╨┐╨╛╨┐╤Л╤В╨║╨╕ ╨┐╤А╨╛╨▓╨░╨╗╨╕╨╗╨╕╤Б╤М")
                            return False
                
                # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤З╤В╨╛ ╨┐╨╛╨╗╤Г╤З╨╕╨╗╨╕ ╨╛╤В╨▓╨╡╤В
                if positions_response is None:
                    logger.error(f"[SYNC_EXCHANGE] тЭМ ╨Я╤Г╤Б╤В╨╛╨╣ ╨╛╤В╨▓╨╡╤В")
                    return False
                
                if positions_response["retCode"] != 0:
                    logger.error(f"[SYNC_EXCHANGE] тЭМ ╨Ю╤И╨╕╨▒╨║╨░: {positions_response['retMsg']}")
                    return False
                
                # ╨Ю╨▒╤А╨░╨▒╨░╤В╤Л╨▓╨░╨╡╨╝ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨░ ╤В╨╡╨║╤Г╤Й╨╡╨╣ ╤Б╤В╤А╨░╨╜╨╕╤Ж╨╡
                positions_count = len(positions_response["result"]["list"])
                
                for idx, position in enumerate(positions_response["result"]["list"]):
                    symbol = position.get("symbol")
                    size = float(position.get("size", 0))
                    
                    if abs(size) > 0:  # ╨Ы╤О╨▒╤Л╨╡ ╨╛╤В╨║╤А╤Л╤В╤Л╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ (LONG ╨╕╨╗╨╕ SHORT)
                        # ╨г╨▒╨╕╤А╨░╨╡╨╝ USDT ╨╕╨╖ ╤Б╨╕╨╝╨▓╨╛╨╗╨░ ╨┤╨╗╤П ╤Б╨╛╨┐╨╛╤Б╤В╨░╨▓╨╗╨╡╨╜╨╕╤П ╤Б ╨▒╨╛╤В╨░╨╝╨╕
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
                
                # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╡╤Б╤В╤М ╨╗╨╕ ╨╡╤Й╨╡ ╤Б╤В╤А╨░╨╜╨╕╤Ж╤Л
                next_page_cursor = positions_response["result"].get("nextPageCursor", "")
                if not next_page_cursor:
                    break
                cursor = next_page_cursor
            
            # тЬЕ ╨Э╨╡ ╨╗╨╛╨│╨╕╤А╤Г╨╡╨╝ ╨╛╨▒╤Й╨╡╨╡ ╨║╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╛ (╨╕╨╖╨▒╤Л╤В╨╛╤З╨╜╨╛)
            
            # тЬЕ ╨г╨Я╨а╨Ю╨й╨Х╨Э╨Ю: ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╤Б╨┐╨╕╤Б╨╛╨║ ╨▒╨╛╤В╨╛╨▓ ╨╛╨┤╨╕╨╜ ╤А╨░╨╖ ╨╕ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╜╨░╨┐╤А╤П╨╝╤Г╤О
            with bots_data_lock:
                bot_items = list(bots_data['bots'].items())  # ╨Ъ╨╛╨┐╨╕╤П ╨┤╨╗╤П ╨▒╨╡╨╖╨╛╨┐╨░╤Б╨╜╨╛╨╣ ╨╕╤В╨╡╤А╨░╤Ж╨╕╨╕
            
            synchronized_bots = 0
            
            for symbol, bot_data in bot_items:
                    try:
                        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨╡╤Б╤В╤М ╨╗╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ ╨┤╨╗╤П ╤Н╤В╨╛╨│╨╛ ╨▒╨╛╤В╨░
                        if symbol in exchange_positions:
                            # ╨Х╤Б╤В╤М ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ - ╨╛╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨┤╨░╨╜╨╜╤Л╨╡ ╨▒╨╛╤В╨░
                            exchange_pos = exchange_positions[symbol]
                            
                            # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╤Б╤В╨░╤А╤Л╨╡ ╨┤╨░╨╜╨╜╤Л╨╡ ╨┤╨╗╤П ╨╗╨╛╨│╨╕╤А╨╛╨▓╨░╨╜╨╕╤П
                            with bots_data_lock:
                                if symbol not in bots_data['bots']:
                                    continue  # ╨С╨╛╤В ╨▒╤Л╨╗ ╤Г╨┤╨░╨╗╤С╨╜ ╨▓ ╨┤╤А╤Г╨│╨╛╨╝ ╨┐╨╛╤В╨╛╨║╨╡
                                bot_data = bots_data['bots'][symbol]
                                old_status = bot_data.get('status', 'UNKNOWN')
                                old_pnl = bot_data.get('unrealized_pnl', 0)
                                
                                # тЪб ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Э╨╡ ╨╕╨╖╨╝╨╡╨╜╤П╨╡╨╝ ╤Б╤В╨░╤В╤Г╤Б ╨╡╤Б╨╗╨╕ ╨▒╨╛╤В ╨▒╤Л╨╗ ╨╛╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜ ╨▓╤А╤Г╤З╨╜╤Г╤О!
                                is_paused = old_status == BOT_STATUS['PAUSED']
                                
                                bot_data['entry_price'] = exchange_pos['avg_price']
                                bot_data['unrealized_pnl'] = exchange_pos['unrealized_pnl']
                                bot_data['position_side'] = 'LONG' if exchange_pos['side'] == 'Buy' else 'SHORT'
                                
                                # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤Б╤В╨╛╨┐╤Л ╨╕ ╤В╨╡╨╣╨║╨╕ ╨╕╨╖ ╨▒╨╕╤А╨╢╨╕
                                if exchange_pos.get('stop_loss'):
                                    bot_data['stop_loss'] = exchange_pos['stop_loss']
                                if exchange_pos.get('take_profit'):
                                    bot_data['take_profit'] = exchange_pos['take_profit']
                                if exchange_pos.get('mark_price'):
                                    bot_data['current_price'] = exchange_pos['mark_price']
                                
                                # ╨Ю╨┐╤А╨╡╨┤╨╡╨╗╤П╨╡╨╝ ╤Б╤В╨░╤В╤Г╤Б ╨╜╨░ ╨╛╤Б╨╜╨╛╨▓╨╡ ╨╜╨░╨╗╨╕╤З╨╕╤П ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ (╨Э╨Х ╨Ш╨Ч╨Ь╨Х╨Э╨п╨Х╨Ь ╨╡╤Б╨╗╨╕ ╨▒╨╛╤В ╨╜╨░ ╨┐╨░╤Г╨╖╨╡!)
                                if not is_paused:
                                    if exchange_pos['side'] == 'Buy':
                                        bot_data['status'] = BOT_STATUS['IN_POSITION_LONG']
                                    else:
                                        bot_data['status'] = BOT_STATUS['IN_POSITION_SHORT']
                                else:
                                    logger.info(f"[SYNC_EXCHANGE] тП╕я╕П {symbol}: ╨С╨╛╤В ╨╜╨░ ╨┐╨░╤Г╨╖╨╡ - ╤Б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤Б╤В╨░╤В╤Г╤Б PAUSED")
                            
                            synchronized_bots += 1
                            
                            # ╨Ф╨╛╨▒╨░╨▓╨╗╤П╨╡╨╝ ╨┤╨╡╤В╨░╨╗╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕
                            entry_price = exchange_pos['avg_price']
                            current_price = exchange_pos.get('mark_price', entry_price)
                            position_size = exchange_pos.get('size', 0)
                            
                            # logger.info(f"[SYNC_EXCHANGE] ЁЯФД {symbol}: {old_status}тЖТ{bot_data['status']}, PnL: ${old_pnl:.2f}тЖТ${exchange_pos['unrealized_pnl']:.2f}")
                            # logger.info(f"[SYNC_EXCHANGE] ЁЯУК {symbol}: ╨Т╤Е╨╛╨┤=${entry_price:.4f} | ╨в╨╡╨║╤Г╤Й╨░╤П=${current_price:.4f} | ╨а╨░╨╖╨╝╨╡╤А={position_size}")
                            
                        else:
                            # ╨Э╨╡╤В ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ - ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Б╤В╨░╤В╤Г╤Б ╨╕╨╜╤Б╤В╤А╤Г╨╝╨╡╨╜╤В╨░
                            old_status = bot_data.get('status', 'UNKNOWN')
                            
                            # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Ю╨▒╤А╨░╨▒╨░╤В╤Л╨▓╨░╨╡╨╝ ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╨▒╨╛╤В╨╛╨▓, ╨║╨╛╤В╨╛╤А╤Л╨╡ ╨▒╤Л╨╗╨╕ ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕!
                            # ╨Я╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╨▒╨╛╤В╨╛╨▓ ╤Б╨╛ ╤Б╤В╨░╤В╤Г╤Б╨╛╨╝ idle, running, paused ╨╕ ╤В.╨┤. - ╨╛╨╜╨╕ ╨╜╨╡ ╨▒╤Л╨╗╨╕ ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕
                            if old_status not in [
                                BOT_STATUS.get('IN_POSITION_LONG'),
                                BOT_STATUS.get('IN_POSITION_SHORT')
                            ]:
                                # ╨С╨╛╤В ╨╜╨╡ ╨▒╤Л╨╗ ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ - ╨┐╤А╨╛╤Б╤В╨╛ ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╨╡╨│╨╛
                                continue
                            
                            old_position_size = bot_data.get('position_size', 0)
                            manual_closed = True  # ╨Х╤Б╨╗╨╕ ╨╝╤Л ╨╖╨┤╨╡╤Б╤М, ╨╖╨╜╨░╤З╨╕╤В ╨▒╨╛╤В ╨▒╤Л╨╗ ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕

                            # тЬЕ ╨г╨Я╨а╨Ю╨й╨Х╨Э╨Ю: ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨┤╨░╨╜╨╜╤Л╨╡ ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨┤╤Г╨▒╨╗╨╕╨║╨░╤В╨╛╨▓
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
                            
                            # тЬЕ ╨г╨Я╨а╨Ю╨й╨Х╨Э╨Ю: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┤╤Г╨▒╨╗╨╕╨║╨░╤В╤Л ╤Б╤А╨░╨╖╤Г ╨┐╨╛╤Б╨╗╨╡ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П entry_price
                            bot_id = bot_data.get('id') or symbol
                            already_closed_trade = _check_if_trade_already_closed(bot_id, symbol, entry_price, entry_time_str)

                            # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╤А╤Л╨╜╨╛╤З╨╜╤Г╤О ╤Ж╨╡╨╜╤Г ╨┤╨╗╤П ╤Д╨╕╨║╤Б╨░╤Ж╨╕╨╕ ╨╖╨░╨║╤А╤Л╤В╨╕╤П
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
                                # entry_time_str ╤Г╨╢╨╡ ╨┐╨╛╨╗╤Г╤З╨╡╨╜ ╨▓╤Л╤И╨╡
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
                                    # ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╨╡╤Б╨╗╨╕ ╤Н╤В╨╛ ╨▒╤Л╨╗╨░ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╨▒╨╛╤В╨░ (╨▒╨╛╤В ╨▒╤Л╨╗ ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕)
                                    # ╨н╤В╨╛ ╨Э╨Х ╤А╤Г╤З╨╜╤Л╨╡ ╤Б╨┤╨╡╨╗╨║╨╕ ╤В╤А╨╡╨╣╨┤╨╡╤А╨░, ╨░ ╨╖╨░╨║╤А╤Л╤В╨╕╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨▒╨╛╤В╨╛╨▓ ╨▓╤А╤Г╤З╨╜╤Г╤О ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
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
                                        is_simulated=False,  # ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╤Н╤В╨╛ ╤Б╨┤╨╡╨╗╨║╨╕ ╨▒╨╛╤В╨╛╨▓, ╨╖╨░╨║╤А╤Л╤В╤Л╨╡ ╨▓╤А╤Г╤З╨╜╤Г╤О ╨╜╨░ ╨▒╨╕╤А╨╢╨╡
                                    )

                                    # ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨в╨░╨║╨╢╨╡ ╤Б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨▓ bots_data.db ╨┤╨╗╤П ╨╕╤Б╤В╨╛╤А╨╕╨╕ ╤В╨╛╤А╨│╨╛╨▓╨╗╨╕ ╨▒╨╛╤В╨╛╨▓
                                    try:
                                        from bot_engine.bots_database import get_bots_database
                                        bots_db = get_bots_database()
                                        # ╨Р╨║╨║╤Г╤А╨░╤В╨╜╨╛ ╤А╨░╤Б╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╨╝ entry_timestamp, ╤З╤В╨╛╨▒╤Л ╨╕╨╖╨▒╨╡╨╢╨░╤В╤М NameError
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
                                            "entry_rsi": None,  # TODO: ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╨╕╨╖ entry_data ╨╡╤Б╨╗╨╕ ╨╡╤Б╤В╤М
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
                                                f"[SYNC_EXCHANGE] тЬЕ ╨Ш╤Б╤В╨╛╤А╨╕╤П ╤Б╨┤╨╡╨╗╨║╨╕ {symbol} ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨░ ╨▓ bots_data.db (ID: {trade_id})"
                                            )
                                    except Exception as bots_db_error:
                                        logger.warning(
                                            f"[SYNC_EXCHANGE] тЪая╕П ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П ╨╕╤Б╤В╨╛╤А╨╕╨╕ ╨▓ bots_data.db: {bots_db_error}"
                                        )
                                    logger.info(
                                        f"[SYNC_EXCHANGE] тЬЛ {symbol}: ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╨╖╨░╨║╤А╤Л╤В╨░ ╨▓╤А╤Г╤З╨╜╤Г╤О ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ "
                                        f"(entry={entry_price:.6f}, exit={exit_price:.6f}, pnl={pnl_usdt:.2f} USDT)"
                                    )
                                else:
                                    # ╨Я╨╛╨╖╨╕╤Ж╨╕╤П ╤Г╨╢╨╡ ╨▒╤Л╨╗╨░ ╨╛╨▒╤А╨░╨▒╨╛╤В╨░╨╜╨░ ╤А╨░╨╜╨╡╨╡ - ╨┐╤А╨╛╤Б╤В╨╛ ╤Г╨┤╨░╨╗╤П╨╡╨╝ ╨▒╨╛╤В╨░ ╨▒╨╡╨╖ ╨┐╨╛╨▓╤В╨╛╤А╨╜╨╛╨│╨╛ ╨╗╨╛╨│╨╕╤А╨╛╨▓╨░╨╜╨╕╤П
                                    pass
                            
                            # тЬЕ ╨г╨Я╨а╨Ю╨й╨Х╨Э╨Ю: ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ ╤Г╨┤╨░╨╗╨╡╨╜╨╕╨╡ ╨▒╨╛╤В╨░ (╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╤В╤Б╤П ╨▓ ╨╛╤В╨┤╨╡╨╗╤М╨╜╨╛╨╣ ╤Д╤Г╨╜╨║╤Ж╨╕╨╕)
                            logger.info(f"[SYNC_EXCHANGE] ЁЯЧСя╕П {symbol}: ╨г╨┤╨░╨╗╤П╨╡╨╝ ╨▒╨╛╤В╨░ (╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╨╖╨░╨║╤А╤Л╤В╨░ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡, ╤Б╤В╨░╤В╤Г╤Б: {old_status})")
                            
                            # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ timestamp ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╡╨│╨╛ ╨╖╨░╨║╤А╤Л╤В╨╕╤П ╨Ф╨Ю ╤Г╨┤╨░╨╗╨╡╨╜╨╕╤П ╨▒╨╛╤В╨░
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
                                logger.info(f"[SYNC_EXCHANGE] тП░ ╨б╨╛╤Е╤А╨░╨╜╨╡╨╜ timestamp ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╡╨│╨╛ ╨╖╨░╨║╤А╤Л╤В╨╕╤П ╨┤╨╗╤П {symbol}: {current_timestamp} (╤З╨╡╤А╨╡╨╖ 1 ╤Б╨▓╨╡╤З╤Г {_tf} ╤А╨░╨╖╤А╨╡╤И╨╕╨╝ ╨╜╨╛╨▓╤Л╨╣ ╨▓╤Е╨╛╨┤)")
                            except Exception as timestamp_error:
                                logger.warning(f"[SYNC_EXCHANGE] тЪая╕П ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П timestamp ╨╖╨░╨║╤А╤Л╤В╨╕╤П ╨┤╨╗╤П {symbol}: {timestamp_error}")
                            
                            # ╨г╨┤╨░╨╗╤П╨╡╨╝ ╨▒╨╛╤В╨░ ╨╕╨╖ ╤Б╨╕╤Б╤В╨╡╨╝╤Л (╤Б ╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨╛╨╣!)
                            with bots_data_lock:
                                if symbol in bots_data['bots']:
                                    del bots_data['bots'][symbol]
                            
                            # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡ ╨┐╨╛╤Б╨╗╨╡ ╤Г╨┤╨░╨╗╨╡╨╜╨╕╤П
                            save_bots_state()
                            
                            synchronized_bots += 1
                        
                    except Exception as e:
                        logger.error(f"[SYNC_EXCHANGE] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨░╤Ж╨╕╨╕ ╨▒╨╛╤В╨░ {symbol}: {e}")
            
            # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╜╨╛╨╡ ╤Б╨╛╤Б╤В╨╛╤П╨╜╨╕╨╡
            save_bots_state()
            
            return True
            
        except Exception as e:
            logger.error(f"[SYNC_EXCHANGE] тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╤Б ╨▒╨╕╤А╨╢╨╕: {e}")
            return False
        
    except Exception as e:
        logger.error(f"[SYNC_EXCHANGE] тЭМ ╨Ю╨▒╤Й╨░╤П ╨╛╤И╨╕╨▒╨║╨░ ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╕╨╖╨░╤Ж╨╕╨╕: {e}")
        return False

