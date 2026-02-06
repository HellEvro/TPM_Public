"""╨д╨╕╨╗╤М╤В╤А╤Л ╨┤╨╗╤П ╤В╨╛╤А╨│╨╛╨▓╤Л╤Е ╤Б╨╕╨│╨╜╨░╨╗╨╛╨▓

╨Т╨║╨╗╤О╤З╨░╨╡╤В:
- check_rsi_time_filter - ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╣ ╤Д╨╕╨╗╤М╤В╤А RSI
- check_exit_scam_filter - ╤Д╨╕╨╗╤М╤В╤А exit scam
- check_no_existing_position - ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╨╛╤В╤Б╤Г╤В╤Б╤В╨▓╨╕╤П ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕
- check_auto_bot_filters - ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╨▓╤Б╨╡╤Е ╤Д╨╕╨╗╤М╤В╤А╨╛╨▓ ╨░╨▓╤В╨╛╨▒╨╛╤В╨░
- test_exit_scam_filter - ╤В╨╡╤Б╤В╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡ exit scam ╤Д╨╕╨╗╤М╤В╤А╨░
- test_rsi_time_filter - ╤В╨╡╤Б╤В╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡ ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨│╨╛ ╤Д╨╕╨╗╤М╤В╤А╨░
"""

import logging
import time
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from bots_modules.imports_and_globals import shutdown_flag, should_log_message

try:
    from bot_engine.filters import (
        check_rsi_time_filter as engine_check_rsi_time_filter,
        check_exit_scam_filter as engine_check_exit_scam_filter,
    )
except ImportError:
    engine_check_rsi_time_filter = None
    engine_check_exit_scam_filter = None

logger = logging.getLogger('BotsService')

# ╨Ъ╤Н╤И ╨┤╨╗╤П ╨┐╤А╨╡╨┤╨╛╤В╨▓╤А╨░╤Й╨╡╨╜╨╕╤П ╤Б╨┐╨░╨╝╨░ ╨╗╨╛╨│╨╛╨▓ ╨╖╨░╤Й╨╕╤В╤Л ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓
_loss_reentry_log_cache = {}
_loss_reentry_log_lock = threading.Lock()
_loss_reentry_log_interval = 60  # ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ ╨╜╨╡ ╤З╨░╤Й╨╡ ╤А╨░╨╖╨░ ╨▓ ╨╝╨╕╨╜╤Г╤В╤Г ╨┤╨╗╤П ╨║╨░╨╢╨┤╨╛╨╣ ╨╝╨╛╨╜╨╡╤В╤Л

# тЬЕ ╨Ъ╨н╨и╨Ш╨а╨Ю╨Т╨Р╨Э╨Ш╨Х AI MANAGER ╨┤╨╗╤П ╨╕╨╖╨▒╨╡╨╢╨░╨╜╨╕╤П ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨░╤Ж╨╕╨╣
_ai_manager_cache = None
_ai_available_cache = None
_ai_cache_lock = threading.Lock()
_delisted_cache = {'ts': 0.0, 'coins': {}}

def get_cached_ai_manager():
    """
    ╨Я╨╛╨╗╤Г╤З╨░╨╡╤В ╨╖╨░╨║╤Н╤И╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╨╣ ╤Н╨║╨╖╨╡╨╝╨┐╨╗╤П╤А AI Manager.
    ╨Ш╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╤Г╨╡╤В╤Б╤П ╤В╨╛╨╗╤М╨║╨╛ ╨╛╨┤╨╕╨╜ ╤А╨░╨╖ ╨┤╨╗╤П ╨╕╨╖╨▒╨╡╨╢╨░╨╜╨╕╤П ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨╖╨░╨│╤А╤Г╨╖╨╛╨║ ╨╝╨╛╨┤╨╡╨╗╨╡╨╣.
    """
    global _ai_manager_cache, _ai_available_cache
    
    with _ai_cache_lock:
        # ╨Х╤Б╨╗╨╕ ╤Г╨╢╨╡ ╨╡╤Б╤В╤М ╨▓ ╨║╤Н╤И╨╡ - ╨▓╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╨╝
        if _ai_manager_cache is not None:
            return _ai_manager_cache, _ai_available_cache
        
        # ╨Ш╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╤Г╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╨╛╨┤╨╕╨╜ ╤А╨░╨╖
        try:
            from bot_engine.bot_config import AIConfig
            if AIConfig.AI_ENABLED:
                from bot_engine.ai.ai_manager import get_ai_manager
                _ai_manager_cache = get_ai_manager()
                _ai_available_cache = _ai_manager_cache.is_available() if _ai_manager_cache else False
            else:
                _ai_manager_cache = None
                _ai_available_cache = False
        except Exception as e:
            logger.debug(f" AI Manager ╨╜╨╡╨┤╨╛╤Б╤В╤Г╨┐╨╡╨╜: {e}")
            _ai_manager_cache = None
            _ai_available_cache = False
        
        return _ai_manager_cache, _ai_available_cache


def _get_cached_delisted_coins():
    """╨Т╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╤В ╨║╤Н╤И ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╤Е ╨╝╨╛╨╜╨╡╤В (╨╛╨▒╨╜╨╛╨▓╨╗╤П╨╡╤В╤Б╤П ╤А╨░╨╖ ╨▓ 60 ╤Б╨╡╨║╤Г╨╜╨┤)."""
    global _delisted_cache
    now_ts = time.time()
    if now_ts - _delisted_cache['ts'] >= 60:
        try:
            delisted_data = load_delisted_coins()
            coins = delisted_data.get('delisted_coins', {}) or {}
            _delisted_cache = {'ts': now_ts, 'coins': coins}
        except Exception as exc:  # pragma: no cover
            logger.warning(f"тЪая╕П ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨╛╨▒╨╜╨╛╨▓╨╕╤В╤М ╨║╤Н╤И ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨░: {exc}")
            # ╨╜╨╡ ╨╛╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ts, ╤З╤В╨╛╨▒╤Л ╨┐╨╛╨▓╤В╨╛╤А╨╕╤В╤М ╨┐╨╛╨┐╤Л╤В╨║╤Г ╨┐╤А╨╕ ╤Б╨╗╨╡╨┤╤Г╤О╤Й╨╡╨╝ ╨╖╨░╨┐╤А╨╛╤Б╨╡
    return _delisted_cache['coins']

# ╨Ш╨╝╨┐╨╛╤А╤В ╨║╨╗╨░╤Б╤Б╨░ ╨▒╨╛╤В╨░ - ╨Ю╨в╨Ъ╨Ы╨о╨з╨Х╨Э ╨╕╨╖-╨╖╨░ ╤Ж╨╕╨║╨╗╨╕╤З╨╡╤Б╨║╨╛╨│╨╛ ╨╕╨╝╨┐╨╛╤А╤В╨░
# NewTradingBot ╨▒╤Г╨┤╨╡╤В ╨╕╨╝╨┐╨╛╤А╤В╨╕╤А╨╛╨▓╨░╨╜ ╨╗╨╛╨║╨░╨╗╤М╨╜╨╛ ╨▓ ╤Д╤Г╨╜╨║╤Ж╨╕╤П╤Е

# ╨Ш╨╝╨┐╨╛╤А╤В ╤Д╤Г╨╜╨║╤Ж╨╕╨╣ ╤А╨░╤Б╤З╨╡╤В╨░ ╨╕╨╖ calculations
try:
    from bots_modules.calculations import (
        calculate_rsi, calculate_rsi_history, calculate_ema, 
        analyze_trend_6h, perform_enhanced_rsi_analysis
    )
except ImportError as e:
    print(f"Warning: Could not import calculation functions in filters: {e}")
    def calculate_rsi(prices, period=14):
        return None
    def calculate_rsi_history(prices, period=14):
        return None
    def calculate_ema(prices, period):
        return None
    def analyze_trend_6h(symbol, exchange_obj=None):
        return None
    def perform_enhanced_rsi_analysis(candles, rsi, symbol):
        return {'enabled': False, 'enhanced_signal': 'WAIT'}

def calculate_ema_list(prices, period):
    """
    ╨а╨░╤Б╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╤В ╤Б╨┐╨╕╤Б╨╛╨║ ╨╖╨╜╨░╤З╨╡╨╜╨╕╨╣ EMA ╨┤╨╗╤П ╨╝╨░╤Б╤Б╨╕╨▓╨░ ╤Ж╨╡╨╜.
    ╨Т╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╤В ╤Б╨┐╨╕╤Б╨╛╨║ ╨╖╨╜╨░╤З╨╡╨╜╨╕╨╣ EMA ╨╕╨╗╨╕ None, ╨╡╤Б╨╗╨╕ ╨╜╨╡╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ ╨┤╨░╨╜╨╜╤Л╤Е.
    """
    if len(prices) < period:
        return None
    
    ema_values = []
    # ╨Я╨╡╤А╨▓╨╛╨╡ ╨╖╨╜╨░╤З╨╡╨╜╨╕╨╡ EMA = SMA
    sma = sum(prices[:period]) / period
    ema = sma
    multiplier = 2 / (period + 1)
    
    # ╨Ф╨╛╨▒╨░╨▓╨╗╤П╨╡╨╝ None ╨┤╨╗╤П ╨┐╨╡╤А╨▓╤Л╤Е period-1 ╨╖╨╜╨░╤З╨╡╨╜╨╕╨╣ (╨│╨┤╨╡ EMA ╨╡╤Й╨╡ ╨╜╨╡ ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜)
    ema_values.extend([None] * (period - 1))
    ema_values.append(ema)
    
    # ╨а╨░╤Б╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╨╝ EMA ╨┤╨╗╤П ╨╛╤Б╤В╨░╨╗╤М╨╜╤Л╤Е ╨╖╨╜╨░╤З╨╡╨╜╨╕╨╣
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
        ema_values.append(ema)
    
    return ema_values

# ╨Ш╨╝╨┐╨╛╤А╤В ╤Д╤Г╨╜╨║╤Ж╨╕╨╣ ╨╖╤А╨╡╨╗╨╛╤Б╤В╨╕ ╨╕╨╖ maturity
try:
    from bots_modules.maturity import (
        check_coin_maturity, check_coin_maturity_with_storage,
        add_mature_coin_to_storage, is_coin_mature_stored
    )
except ImportError as e:
    print(f"Warning: Could not import maturity functions in filters: {e}")
    def check_coin_maturity(symbol, candles):
        return {'is_mature': True, 'reason': 'Not checked'}
    def check_coin_maturity_with_storage(symbol, candles):
        return {'is_mature': True, 'reason': 'Not checked'}
    def add_mature_coin_to_storage(symbol, data, auto_save=True):
        pass
    def is_coin_mature_stored(symbol):
        return True  # ╨Т╨а╨Х╨Ь╨Х╨Э╨Э╨Ю: ╤А╨░╨╖╤А╨╡╤И╨░╨╡╨╝ ╨▓╤Б╨╡ ╨╝╨╛╨╜╨╡╤В╤Л

# ╨Ш╨╝╨┐╨╛╤А╤В ╤Д╤Г╨╜╨║╤Ж╨╕╨╣ ╨┤╨╗╤П ╤А╨░╨▒╨╛╤В╤Л ╤Б ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╨╝╨╕ ╨╝╨╛╨╜╨╡╤В╨░╨╝╨╕
try:
    from bots_modules.sync_and_cache import load_delisted_coins, ensure_exchange_initialized
except ImportError as e:
    print(f"Warning: Could not import sync_and_cache helpers in filters: {e}")
    def load_delisted_coins():
        return {"delisted_coins": {}}
    def ensure_exchange_initialized():
        return False

# тЭМ ╨Ю╨в╨Ъ╨Ы╨о╨з╨Х╨Э╨Ю: optimal_ema ╨┐╨╡╤А╨╡╨╝╨╡╤Й╨╡╨╜ ╨▓ backup (╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╤В╤Б╤П ╨╖╨░╨│╨╗╤Г╤И╨║╨░ ╨╕╨╖ imports_and_globals)
# ╨Ш╨╝╨┐╨╛╤А╤В ╤Д╤Г╨╜╨║╤Ж╨╕╨╕ optimal_ema ╨╕╨╖ ╨╝╨╛╨┤╤Г╨╗╤П
# try:
#     from bots_modules.optimal_ema import get_optimal_ema_periods
# except ImportError as e:
#     print(f"Warning: Could not import optimal_ema functions in filters: {e}")
#     def get_optimal_ema_periods(symbol):
#         return {'ema_short': 50, 'ema_long': 200, 'accuracy': 0}

# ╨Ш╨╝╨┐╨╛╤А╤В ╤Д╤Г╨╜╨║╤Ж╨╕╨╣ ╨║╤Н╤И╨░ ╨╕╨╖ sync_and_cache
try:
    from bots_modules.sync_and_cache import save_rsi_cache
except ImportError as e:
    print(f"Warning: Could not import save_rsi_cache in filters: {e}")
    def save_rsi_cache():
        pass

# ╨Ш╨╝╨┐╨╛╤А╤В╨╕╤А╤Г╨╡╨╝ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Л╨╡ ╨┐╨╡╤А╨╡╨╝╨╡╨╜╨╜╤Л╨╡ ╨╕ ╤Д╤Г╨╜╨║╤Ж╨╕╨╕ ╨╕╨╖ imports_and_globals
try:
    from bots_modules.imports_and_globals import (
        bots_data_lock, bots_data, rsi_data_lock, coins_rsi_data,
        BOT_STATUS, system_initialized, get_exchange,
        get_individual_coin_settings
    )
    from bot_engine.bot_config import SystemConfig
except ImportError:
    bots_data_lock = threading.Lock()
    bots_data = {}
    rsi_data_lock = threading.Lock()
    coins_rsi_data = {}
    BOT_STATUS = {}
    system_initialized = False
    def get_exchange():
        return None
    def get_individual_coin_settings(symbol):
        return None
    # Fallback ╨┤╨╗╤П SystemConfig
    class SystemConfig:
        RSI_OVERSOLD = 29
        RSI_OVERBOUGHT = 71
        # тЬЕ ╨Э╨╛╨▓╤Л╨╡ ╨┐╨░╤А╨░╨╝╨╡╤В╤А╤Л ╤Б ╤Г╤З╨╡╤В╨╛╨╝ ╤В╤А╨╡╨╜╨┤╨░
        RSI_EXIT_LONG_WITH_TREND = 65
        RSI_EXIT_LONG_AGAINST_TREND = 60
        RSI_EXIT_SHORT_WITH_TREND = 35
        RSI_EXIT_SHORT_AGAINST_TREND = 40

def _legacy_check_rsi_time_filter(candles, rsi, signal, symbol=None, individual_settings=None):
    """
    ╨У╨Ш╨С╨а╨Ш╨Ф╨Э╨л╨Щ ╨Т╨а╨Х╨Ь╨Х╨Э╨Э╨Ю╨Щ ╨д╨Ш╨Ы╨м╨в╨а RSI
    
    ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╤В ╤З╤В╨╛:
    1. ╨Я╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╤Б╨▓╨╡╤З╨╡╨╣ (╨╕╨╖ ╨║╨╛╨╜╤Д╨╕╨│╨░, ╨┐╨╛ ╤Г╨╝╨╛╨╗╤З╨░╨╜╨╕╤О 8) ╨╜╨░╤Е╨╛╨┤╤П╤В╤Б╤П ╨▓ "╤Б╨┐╨╛╨║╨╛╨╣╨╜╨╛╨╣ ╨╖╨╛╨╜╨╡"
       - ╨Ф╨╗╤П SHORT: ╨▓╤Б╨╡ ╤Б╨▓╨╡╤З╨╕ ╨┤╨╛╨╗╨╢╨╜╤Л ╨▒╤Л╤В╤М >= 65
       - ╨Ф╨╗╤П LONG: ╨▓╤Б╨╡ ╤Б╨▓╨╡╤З╨╕ ╨┤╨╛╨╗╨╢╨╜╤Л ╨▒╤Л╤В╤М <= 35
    2. ╨Я╨╡╤А╨╡╨┤ ╤Н╤В╨╛╨╣ ╤Б╨┐╨╛╨║╨╛╨╣╨╜╨╛╨╣ ╨╖╨╛╨╜╨╛╨╣ ╨▒╤Л╨╗ ╤Н╨║╤Б╤В╤А╨╡╨╝╤Г╨╝
       - ╨Ф╨╗╤П SHORT: ╤Б╨▓╨╡╤З╨░ ╤Б RSI >= 71
       - ╨Ф╨╗╤П LONG: ╤Б╨▓╨╡╤З╨░ ╤Б RSI <= 29
    3. ╨б ╨╝╨╛╨╝╨╡╨╜╤В╨░ ╤Н╨║╤Б╤В╤А╨╡╨╝╤Г╨╝╨░ ╨┐╤А╨╛╤И╨╗╨╛ ╨╝╨╕╨╜╨╕╨╝╤Г╨╝ N ╤Б╨▓╨╡╤З╨╡╨╣
    
    Args:
        candles: ╨б╨┐╨╕╤Б╨╛╨║ ╤Б╨▓╨╡╤З╨╡╨╣
        rsi: ╨в╨╡╨║╤Г╤Й╨╡╨╡ ╨╖╨╜╨░╤З╨╡╨╜╨╕╨╡ RSI
        signal: ╨в╨╛╤А╨│╨╛╨▓╤Л╨╣ ╤Б╨╕╨│╨╜╨░╨╗ ('ENTER_LONG' ╨╕╨╗╨╕ 'ENTER_SHORT')
        symbol: ╨б╨╕╨╝╨▓╨╛╨╗ ╨╝╨╛╨╜╨╡╤В╤Л (╨╛╨┐╤Ж╨╕╨╛╨╜╨░╨╗╤М╨╜╨╛, ╨┤╨╗╤П ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╤Е ╨╜╨░╤Б╤В╤А╨╛╨╡╨║)
        individual_settings: ╨Ш╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╨╝╨╛╨╜╨╡╤В╤Л (╨╛╨┐╤Ж╨╕╨╛╨╜╨░╨╗╤М╨╜╨╛)
    
    Returns:
        dict: {'allowed': bool, 'reason': str, 'last_extreme_candles_ago': int, 'calm_candles': int}
    """
    try:
        # тЬЕ ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕: ╤Б╨╜╨░╤З╨░╨╗╨░ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡, ╨╖╨░╤В╨╡╨╝ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Л╨╡
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╨║╨╛╨╜╤Д╨╕╨│ ╨╜╨╡ ╨╝╨╡╨╜╤П╨╡╤В╤Б╤П, GIL ╨┤╨╡╨╗╨░╨╡╤В ╤З╤В╨╡╨╜╨╕╨╡ ╨░╤В╨╛╨╝╨░╤А╨╜╤Л╨╝
        if individual_settings is None and symbol:
            individual_settings = get_individual_coin_settings(symbol)
        
        auto_config = bots_data.get('auto_bot_config', {})
        
        # ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕, ╨╡╤Б╨╗╨╕ ╨╛╨╜╨╕ ╨╡╤Б╤В╤М, ╨╕╨╜╨░╤З╨╡ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Л╨╡
        rsi_time_filter_enabled = individual_settings.get('rsi_time_filter_enabled') if individual_settings else None
        if rsi_time_filter_enabled is None:
            rsi_time_filter_enabled = auto_config.get('rsi_time_filter_enabled', True)
        
        rsi_time_filter_candles = individual_settings.get('rsi_time_filter_candles') if individual_settings else None
        if rsi_time_filter_candles is None:
            rsi_time_filter_candles = auto_config.get('rsi_time_filter_candles', 8)
        rsi_time_filter_candles = max(2, rsi_time_filter_candles)  # ╨Ь╨╕╨╜╨╕╨╝╤Г╨╝ 2 ╤Б╨▓╨╡╤З╨╕ (╨╖╨░╤Й╨╕╤В╨░ ╨╛╤В ╨╜╨╡╨║╨╛╤А╤А╨╡╨║╤В╨╜╤Л╤Е ╨╖╨╜╨░╤З╨╡╨╜╨╕╨╣)
        
        rsi_time_filter_upper = individual_settings.get('rsi_time_filter_upper') if individual_settings else None
        if rsi_time_filter_upper is None:
            rsi_time_filter_upper = auto_config.get('rsi_time_filter_upper', 65)  # ╨б╨┐╨╛╨║╨╛╨╣╨╜╨░╤П ╨╖╨╛╨╜╨░ ╨┤╨╗╤П SHORT
        
        rsi_time_filter_lower = individual_settings.get('rsi_time_filter_lower') if individual_settings else None
        if rsi_time_filter_lower is None:
            rsi_time_filter_lower = auto_config.get('rsi_time_filter_lower', 35)  # ╨б╨┐╨╛╨║╨╛╨╣╨╜╨░╤П ╨╖╨╛╨╜╨░ ╨┤╨╗╤П LONG
        
        rsi_long_threshold = individual_settings.get('rsi_long_threshold') if individual_settings else None
        if rsi_long_threshold is None:
            rsi_long_threshold = auto_config.get('rsi_long_threshold', 29)  # ╨н╨║╤Б╤В╤А╨╡╨╝╤Г╨╝ ╨┤╨╗╤П LONG
        
        rsi_short_threshold = individual_settings.get('rsi_short_threshold') if individual_settings else None
        if rsi_short_threshold is None:
            rsi_short_threshold = auto_config.get('rsi_short_threshold', 71)  # ╨н╨║╤Б╤В╤А╨╡╨╝╤Г╨╝ ╨┤╨╗╤П SHORT
        
        # ╨Х╤Б╨╗╨╕ ╤Д╨╕╨╗╤М╤В╤А ╨╛╤В╨║╨╗╤О╤З╨╡╨╜ - ╤А╨░╨╖╤А╨╡╤И╨░╨╡╨╝ ╤Б╨┤╨╡╨╗╨║╤Г
        if not rsi_time_filter_enabled:
            return {'allowed': True, 'reason': 'RSI ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╣ ╤Д╨╕╨╗╤М╤В╤А ╨╛╤В╨║╨╗╤О╤З╨╡╨╜', 'last_extreme_candles_ago': None, 'calm_candles': None}
        
        if len(candles) < 50:
            return {'allowed': False, 'reason': '╨Э╨╡╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ ╤Б╨▓╨╡╤З╨╡╨╣ ╨┤╨╗╤П ╨░╨╜╨░╨╗╨╕╨╖╨░', 'last_extreme_candles_ago': None, 'calm_candles': 0}
        
        # ╨а╨░╤Б╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╨╝ ╨╕╤Б╤В╨╛╤А╨╕╤О RSI
        closes = [candle['close'] for candle in candles]
        rsi_history = calculate_rsi_history(closes, 14)
        
        min_rsi_history = max(rsi_time_filter_candles * 2 + 14, 30)
        if not rsi_history or len(rsi_history) < min_rsi_history:
            return {'allowed': False, 'reason': f'╨Э╨╡╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ RSI ╨╕╤Б╤В╨╛╤А╨╕╨╕ (╤В╤А╨╡╨▒╤Г╨╡╤В╤Б╤П {min_rsi_history})', 'last_extreme_candles_ago': None, 'calm_candles': 0}
        
        current_index = len(rsi_history) - 1
        
        # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╣ ╤Н╨╗╨╡╨╝╨╡╨╜╤В ╨╕╤Б╤В╨╛╤А╨╕╨╕ ╨┐╨╡╤А╨╡╨┤╨░╨╜╨╜╤Л╨╝ RSI, ╨╡╤Б╨╗╨╕ ╨╛╨╜ ╤Г╨║╨░╨╖╨░╨╜
        # ╨н╤В╨╛ ╨▓╨░╨╢╨╜╨╛ ╨┤╨╗╤П ╤Б╨╛╨│╨╗╨░╤Б╨╛╨▓╨░╨╜╨╜╨╛╤Б╤В╨╕ ╨┤╨░╨╜╨╜╤Л╤Е, ╤В╨░╨║ ╨║╨░╨║ ╨┐╨╡╤А╨╡╨┤╨░╨╜╨╜╤Л╨╣ RSI ╨╝╨╛╨╢╨╡╤В ╨▒╤Л╤В╤М ╨▒╨╛╨╗╨╡╨╡ ╨░╨║╤В╤Г╨░╨╗╤М╨╜╤Л╨╝
        if rsi is not None:
            rsi_history[current_index] = rsi
        
        if signal == 'ENTER_SHORT':
            # ╨Ы╨Ю╨У╨Ш╨Ъ╨Р ╨Ф╨Ы╨п SHORT (╨░╨╜╨░╨╗╨╛╨│╨╕╤З╨╜╨╛ LONG, ╤В╨╛╨╗╤М╨║╨╛ ╨╜╨░╨╛╨▒╨╛╤А╨╛╤В):
            # 1. ╨С╨╡╤А╨╡╨╝ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╤Б╨▓╨╡╤З╨╡╨╣ (rsi_time_filter_candles ╨╕╨╖ ╨║╨╛╨╜╤Д╨╕╨│╨░, ╨╜╨░╨┐╤А╨╕╨╝╨╡╤А 8)
            # 2. ╨Ш╤Й╨╡╨╝ ╤Б╤А╨╡╨┤╨╕ ╨╜╨╕╤Е ╨б╨Р╨Ь╨г╨о ╨а╨Р╨Э╨Э╨о╨о (╨╗╨╡╨▓╤Г╤О) ╤Б╨▓╨╡╤З╤Г ╤Б RSI >= 71 - ╤Н╤В╨╛ ╨╛╤В╨┐╤А╨░╨▓╨╜╨░╤П ╤В╨╛╤З╨║╨░
            # 3. ╨Ю╤В ╨╛╤В╨┐╤А╨░╨▓╨╜╨╛╨╣ ╤В╨╛╤З╨║╨╕ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨Т╨б╨Х ╨┐╨╛╤Б╨╗╨╡╨┤╤Г╤О╤Й╨╕╨╡ ╤Б╨▓╨╡╤З╨╕ (╨┤╨╛ ╤В╨╡╨║╤Г╤Й╨╡╨╣) - ╨┤╨╛╨╗╨╢╨╜╤Л ╨▒╤Л╤В╤М >= 65
            # 4. ╨Х╤Б╨╗╨╕ ╨▓╤Б╨╡ >= 65 ╨Ш ╨┐╤А╨╛╤И╨╗╨╛ ╨╝╨╕╨╜╨╕╨╝╤Г╨╝ N ╤Б╨▓╨╡╤З╨╡╨╣ - ╤А╨░╨╖╤А╨╡╤И╨░╨╡╨╝
            # 5. ╨Х╤Б╨╗╨╕ ╨║╨░╨║╨░╤П-╤В╨╛ ╤Б╨▓╨╡╤З╨░ < 65 - ╨▒╨╗╨╛╨║╨╕╤А╤Г╨╡╨╝ (╨▓╤Е╨╛╨┤ ╤Г╨┐╤Г╤Й╨╡╨╜)
            
            # ╨С╨╡╤А╨╡╨╝ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╤Б╨▓╨╡╤З╨╡╨╣ ╨╕╨╖ ╨║╨╛╨╜╤Д╨╕╨│╨░
            last_n_candles_start = max(0, current_index - rsi_time_filter_candles + 1)
            last_n_candles = rsi_history[last_n_candles_start:current_index + 1]
            
            # ╨Ш╤Й╨╡╨╝ ╨б╨Р╨Ь╨г╨о ╨а╨Р╨Э╨Э╨о╨о (╨╗╨╡╨▓╤Г╤О) ╤Б╨▓╨╡╤З╤Г ╤Б RSI >= 71 ╤Б╤А╨╡╨┤╨╕ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╤Е N ╤Б╨▓╨╡╤З╨╡╨╣
            peak_index = None
            for i in range(last_n_candles_start, current_index + 1):
                if rsi_history[i] >= rsi_short_threshold:
                    peak_index = i  # ╨Э╨░╤И╨╗╨╕ ╤Б╨░╨╝╤Г╤О ╤А╨░╨╜╨╜╤О╤О ╤Б╨▓╨╡╤З╤Г >= 71
                    break
            
            # ╨Х╤Б╨╗╨╕ ╨╜╨╡ ╨╜╨░╤И╨╗╨╕ ╨┐╨╕╨║ ╨▓ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╤Е N ╤Б╨▓╨╡╤З╨░╤Е - ╨▒╨╗╨╛╨║╨╕╤А╤Г╨╡╨╝ (╨╜╨╡╤В ╨╛╤В╨┐╤А╨░╨▓╨╜╨╛╨╣ ╤В╨╛╤З╨║╨╕)
            if peak_index is None:
                return {
                    'allowed': False,
                    'reason': f'╨С╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨░: ╨┐╨╕╨║ RSI >= {rsi_short_threshold} ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜ ╨▓ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╤Е {rsi_time_filter_candles} ╤Б╨▓╨╡╤З╨░╤Е',
                    'last_extreme_candles_ago': None,
                    'calm_candles': 0
                }
            
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨Т╨б╨Х ╤Б╨▓╨╡╤З╨╕ ╨Ю╨в ╨╛╤В╨┐╤А╨░╨▓╨╜╨╛╨╣ ╤В╨╛╤З╨║╨╕ (╨▓╨║╨╗╤О╤З╨░╤П ╨╡╤С) ╨┤╨╛ ╤В╨╡╨║╤Г╤Й╨╡╨╣ ╨▓╨║╨╗╤О╤З╨╕╤В╨╡╨╗╤М╨╜╨╛
            # ╨С╨╡╤А╨╡╨╝ ╨▓╤Б╨╡ ╤Б╨▓╨╡╤З╨╕ ╨Ю╨в peak_index (╨▓╨║╨╗╤О╤З╨░╤П ╤Б╨░╨╝ peak_index) ╨┤╨╛ current_index
            check_candles = rsi_history[peak_index:current_index + 1]
            
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤З╤В╨╛ ╨Т╨б╨Х ╤Б╨▓╨╡╤З╨╕ >= 65 (╨▓╨║╨╗╤О╤З╨░╤П ╤Б╨░╨╝╤Г ╨╛╤В╨┐╤А╨░╨▓╨╜╤Г╤О ╤В╨╛╤З╨║╤Г)
            invalid_candles = [rsi_val for rsi_val in check_candles if rsi_val < rsi_time_filter_upper]
            
            if len(invalid_candles) > 0:
                # ╨Х╤Б╤В╤М ╤Б╨▓╨╡╤З╨╕ < 65 - ╨▓╤Е╨╛╨┤ ╤Г╨┐╤Г╤Й╨╡╨╜
                candles_since_peak = current_index - peak_index + 1
                return {
                    'allowed': False,
                    'reason': f'╨С╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨░: {len(invalid_candles)} ╤Б╨▓╨╡╤З╨╡╨╣ ╨┐╨╛╤Б╨╗╨╡ ╨╛╤В╨┐╤А╨░╨▓╨╜╨╛╨╣ ╤В╨╛╤З╨║╨╕ ╨┐╤А╨╛╨▓╨░╨╗╨╕╨╗╨╕╤Б╤М < {rsi_time_filter_upper} (╨▓╤Е╨╛╨┤ ╤Г╨┐╤Г╤Й╨╡╨╜)',
                    'last_extreme_candles_ago': candles_since_peak - 1,
                    'calm_candles': len(check_candles) - len(invalid_candles)
                }
            
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤З╤В╨╛ ╨┐╤А╨╛╤И╨╗╨╛ ╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ ╤Б╨▓╨╡╤З╨╡╨╣ (╨╝╨╕╨╜╨╕╨╝╤Г╨╝ N ╨╕╨╖ ╨║╨╛╨╜╤Д╨╕╨│╨░)
            # candles_since_peak - ╤Н╤В╨╛ ╨║╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╛ ╤Б╨▓╨╡╤З╨╡╨╣ ╨Ю╨в ╨╛╤В╨┐╤А╨░╨▓╨╜╨╛╨╣ ╤В╨╛╤З╨║╨╕ (╨▓╨║╨╗╤О╤З╨░╤П ╨╡╤С) ╨┤╨╛ ╤В╨╡╨║╤Г╤Й╨╡╨╣
            candles_since_peak = current_index - peak_index + 1
            if candles_since_peak < rsi_time_filter_candles:
                return {
                    'allowed': False,
                    'reason': f'╨Ю╨╢╨╕╨┤╨░╨╜╨╕╨╡: ╤Б ╨╛╤В╨┐╤А╨░╨▓╨╜╨╛╨╣ ╤В╨╛╤З╨║╨╕ ╨┐╤А╨╛╤И╨╗╨╛ ╤В╨╛╨╗╤М╨║╨╛ {candles_since_peak} ╤Б╨▓╨╡╤З╨╡╨╣ (╤В╤А╨╡╨▒╤Г╨╡╤В╤Б╤П {rsi_time_filter_candles})',
                    'last_extreme_candles_ago': candles_since_peak - 1,
                    'calm_candles': candles_since_peak
                }
            
            # ╨Т╤Б╨╡ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨┐╤А╨╛╨╣╨┤╨╡╨╜╤Л!
            return {
                'allowed': True,
                'reason': f'╨а╨░╨╖╤А╨╡╤И╨╡╨╜╨╛: ╤Б ╨╛╤В╨┐╤А╨░╨▓╨╜╨╛╨╣ ╤В╨╛╤З╨║╨╕ (╤Б╨▓╨╡╤З╨░ -{candles_since_peak}) ╨┐╤А╨╛╤И╨╗╨╛ {candles_since_peak} ╤Б╨┐╨╛╨║╨╛╨╣╨╜╤Л╤Е ╤Б╨▓╨╡╤З╨╡╨╣ >= {rsi_time_filter_upper}',
                'last_extreme_candles_ago': candles_since_peak - 1,
                'calm_candles': candles_since_peak
            }
                
        elif signal == 'ENTER_LONG':
            # ╨Ы╨Ю╨У╨Ш╨Ъ╨Р ╨Ф╨Ы╨п LONG:
            # 1. ╨С╨╡╤А╨╡╨╝ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╤Б╨▓╨╡╤З╨╡╨╣ (rsi_time_filter_candles ╨╕╨╖ ╨║╨╛╨╜╤Д╨╕╨│╨░, ╨╜╨░╨┐╤А╨╕╨╝╨╡╤А 8)
            # 2. ╨Ш╤Й╨╡╨╝ ╤Б╤А╨╡╨┤╨╕ ╨╜╨╕╤Е ╨б╨Р╨Ь╨г╨о ╨а╨Р╨Э╨Э╨о╨о (╨╗╨╡╨▓╤Г╤О) ╤Б╨▓╨╡╤З╤Г ╤Б RSI <= 29 - ╤Н╤В╨╛ ╨╛╤В╨┐╤А╨░╨▓╨╜╨░╤П ╤В╨╛╤З╨║╨░
            # 3. ╨Ю╤В ╨╛╤В╨┐╤А╨░╨▓╨╜╨╛╨╣ ╤В╨╛╤З╨║╨╕ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨Т╨б╨Х ╨┐╨╛╤Б╨╗╨╡╨┤╤Г╤О╤Й╨╕╨╡ ╤Б╨▓╨╡╤З╨╕ (╨┤╨╛ ╤В╨╡╨║╤Г╤Й╨╡╨╣) - ╨┤╨╛╨╗╨╢╨╜╤Л ╨▒╤Л╤В╤М <= 35
            # 4. ╨Х╤Б╨╗╨╕ ╨▓╤Б╨╡ <= 35 ╨Ш ╨┐╤А╨╛╤И╨╗╨╛ ╨╝╨╕╨╜╨╕╨╝╤Г╨╝ N ╤Б╨▓╨╡╤З╨╡╨╣ - ╤А╨░╨╖╤А╨╡╤И╨░╨╡╨╝
            # 5. ╨Х╤Б╨╗╨╕ ╨║╨░╨║╨░╤П-╤В╨╛ ╤Б╨▓╨╡╤З╨░ > 35 - ╨▒╨╗╨╛╨║╨╕╤А╤Г╨╡╨╝ (╨▓╤Е╨╛╨┤ ╤Г╨┐╤Г╤Й╨╡╨╜)
            
            # ╨С╨╡╤А╨╡╨╝ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╤Б╨▓╨╡╤З╨╡╨╣ ╨╕╨╖ ╨║╨╛╨╜╤Д╨╕╨│╨░
            last_n_candles_start = max(0, current_index - rsi_time_filter_candles + 1)
            last_n_candles = rsi_history[last_n_candles_start:current_index + 1]
            
            # ╨Ш╤Й╨╡╨╝ ╨б╨Р╨Ь╨г╨о ╨а╨Р╨Э╨Э╨о╨о (╨╗╨╡╨▓╤Г╤О) ╤Б╨▓╨╡╤З╤Г ╤Б RSI <= 29 ╤Б╤А╨╡╨┤╨╕ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╤Е N ╤Б╨▓╨╡╤З╨╡╨╣
            low_index = None
            for i in range(last_n_candles_start, current_index + 1):
                if rsi_history[i] <= rsi_long_threshold:
                    low_index = i  # ╨Э╨░╤И╨╗╨╕ ╤Б╨░╨╝╤Г╤О ╤А╨░╨╜╨╜╤О╤О ╤Б╨▓╨╡╤З╤Г <= 29
                    break
            
            # ╨Х╤Б╨╗╨╕ ╨╜╨╡ ╨╜╨░╤И╨╗╨╕ ╨╗╨╛╨╣ ╨▓ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╤Е N ╤Б╨▓╨╡╤З╨░╤Е - ╨▒╨╗╨╛╨║╨╕╤А╤Г╨╡╨╝ (╨╜╨╡╤В ╨╛╤В╨┐╤А╨░╨▓╨╜╨╛╨╣ ╤В╨╛╤З╨║╨╕)
            if low_index is None:
                return {
                    'allowed': False,
                    'reason': f'╨С╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨░: ╨╗╨╛╨╣ RSI <= {rsi_long_threshold} ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜ ╨▓ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╤Е {rsi_time_filter_candles} ╤Б╨▓╨╡╤З╨░╤Е',
                    'last_extreme_candles_ago': None,
                    'calm_candles': 0
                }
            
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨Т╨б╨Х ╤Б╨▓╨╡╤З╨╕ ╨Ю╨в ╨╛╤В╨┐╤А╨░╨▓╨╜╨╛╨╣ ╤В╨╛╤З╨║╨╕ (╨▓╨║╨╗╤О╤З╨░╤П ╨╡╤С) ╨┤╨╛ ╤В╨╡╨║╤Г╤Й╨╡╨╣ ╨▓╨║╨╗╤О╤З╨╕╤В╨╡╨╗╤М╨╜╨╛
            # ╨С╨╡╤А╨╡╨╝ ╨▓╤Б╨╡ ╤Б╨▓╨╡╤З╨╕ ╨Ю╨в low_index (╨▓╨║╨╗╤О╤З╨░╤П ╤Б╨░╨╝ low_index) ╨┤╨╛ current_index
            check_candles = rsi_history[low_index:current_index + 1]
            
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤З╤В╨╛ ╨Т╨б╨Х ╤Б╨▓╨╡╤З╨╕ <= 35 (╨▓╨║╨╗╤О╤З╨░╤П ╤Б╨░╨╝╤Г ╨╛╤В╨┐╤А╨░╨▓╨╜╤Г╤О ╤В╨╛╤З╨║╤Г)
            invalid_candles = [rsi_val for rsi_val in check_candles if rsi_val > rsi_time_filter_lower]
            
            if len(invalid_candles) > 0:
                # ╨Х╤Б╤В╤М ╤Б╨▓╨╡╤З╨╕ > 35 - ╨▓╤Е╨╛╨┤ ╤Г╨┐╤Г╤Й╨╡╨╜
                candles_since_low = current_index - low_index + 1
                return {
                    'allowed': False,
                    'reason': f'╨С╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨░: {len(invalid_candles)} ╤Б╨▓╨╡╤З╨╡╨╣ ╨┐╨╛╤Б╨╗╨╡ ╨╛╤В╨┐╤А╨░╨▓╨╜╨╛╨╣ ╤В╨╛╤З╨║╨╕ ╨┐╨╛╨┤╨╜╤П╨╗╨╕╤Б╤М > {rsi_time_filter_lower} (╨▓╤Е╨╛╨┤ ╤Г╨┐╤Г╤Й╨╡╨╜)',
                    'last_extreme_candles_ago': candles_since_low - 1,
                    'calm_candles': len(check_candles) - len(invalid_candles)
                }
            
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤З╤В╨╛ ╨┐╤А╨╛╤И╨╗╨╛ ╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ ╤Б╨▓╨╡╤З╨╡╨╣ (╨╝╨╕╨╜╨╕╨╝╤Г╨╝ N ╨╕╨╖ ╨║╨╛╨╜╤Д╨╕╨│╨░)
            # candles_since_low - ╤Н╤В╨╛ ╨║╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╛ ╤Б╨▓╨╡╤З╨╡╨╣ ╨Ю╨в ╨╛╤В╨┐╤А╨░╨▓╨╜╨╛╨╣ ╤В╨╛╤З╨║╨╕ (╨▓╨║╨╗╤О╤З╨░╤П ╨╡╤С) ╨┤╨╛ ╤В╨╡╨║╤Г╤Й╨╡╨╣
            candles_since_low = current_index - low_index + 1
            if candles_since_low < rsi_time_filter_candles:
                return {
                    'allowed': False,
                    'reason': f'╨Ю╨╢╨╕╨┤╨░╨╜╨╕╨╡: ╤Б ╨╛╤В╨┐╤А╨░╨▓╨╜╨╛╨╣ ╤В╨╛╤З╨║╨╕ ╨┐╤А╨╛╤И╨╗╨╛ ╤В╨╛╨╗╤М╨║╨╛ {candles_since_low} ╤Б╨▓╨╡╤З╨╡╨╣ (╤В╤А╨╡╨▒╤Г╨╡╤В╤Б╤П {rsi_time_filter_candles})',
                    'last_extreme_candles_ago': candles_since_low - 1,
                    'calm_candles': candles_since_low
                }
            
            # ╨Т╤Б╨╡ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨┐╤А╨╛╨╣╨┤╨╡╨╜╤Л!
            return {
                'allowed': True,
                'reason': f'╨а╨░╨╖╤А╨╡╤И╨╡╨╜╨╛: ╤Б ╨╛╤В╨┐╤А╨░╨▓╨╜╨╛╨╣ ╤В╨╛╤З╨║╨╕ (╤Б╨▓╨╡╤З╨░ -{candles_since_low}) ╨┐╤А╨╛╤И╨╗╨╛ {candles_since_low} ╤Б╨┐╨╛╨║╨╛╨╣╨╜╤Л╤Е ╤Б╨▓╨╡╤З╨╡╨╣ <= {rsi_time_filter_lower}',
                'last_extreme_candles_ago': candles_since_low - 1,
                'calm_candles': candles_since_low
            }
        
        return {'allowed': True, 'reason': '╨Э╨╡╨╕╨╖╨▓╨╡╤Б╤В╨╜╤Л╨╣ ╤Б╨╕╨│╨╜╨░╨╗', 'last_extreme_candles_ago': None, 'calm_candles': 0}
    
    except Exception as e:
        logger.error(f" ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨│╨╛ ╤Д╨╕╨╗╤М╤В╤А╨░: {e}")
        return {'allowed': False, 'reason': f'╨Ю╤И╨╕╨▒╨║╨░ ╨░╨╜╨░╨╗╨╕╨╖╨░: {str(e)}', 'last_extreme_candles_ago': None, 'calm_candles': 0}

def get_coin_candles_only(symbol, exchange_obj=None):
    """тЪб ╨С╨л╨б╨в╨а╨Р╨п ╨╖╨░╨│╤А╤Г╨╖╨║╨░ ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╤Б╨▓╨╡╤З╨╡╨╣ ╨С╨Х╨Ч ╤А╨░╤Б╤З╨╡╤В╨╛╨▓"""
    try:
        if shutdown_flag.is_set():
            return None

        from bots_modules.imports_and_globals import get_exchange
        exchange_to_use = exchange_obj if exchange_obj is not None else get_exchange()
        
        if exchange_to_use is None:
            return None
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╤Б╨▓╨╡╤З╨╕
        chart_response = exchange_to_use.get_chart_data(symbol, '6h', '30d')
        
        if not chart_response or not chart_response.get('success'):
            return None
        
        candles = chart_response['data']['candles']
        if not candles or len(candles) < 15:
            return None
        
        # ╨Т╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╨╝ ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╤Б╨▓╨╡╤З╨╕ ╨╕ ╤Б╨╕╨╝╨▓╨╛╨╗
        return {
            'symbol': symbol,
            'candles': candles,
            'timeframe': '6h',  # тЬЕ ╨Ф╨╛╨▒╨░╨▓╨╗╤П╨╡╨╝ timeframe ╨┤╨╗╤П ╨┐╤А╨░╨▓╨╕╨╗╤М╨╜╨╛╨│╨╛ ╨╜╨░╨║╨╛╨┐╨╗╨╡╨╜╨╕╤П
            'last_update': datetime.now().isoformat()
        }
        
    except Exception as e:
        return None


def check_rsi_time_filter(candles, rsi, signal, symbol=None, individual_settings=None):
    """
    ╨Ю╨▒╤С╤А╤В╨║╨░ ╨╜╨░╨┤ bot_engine.filters.check_rsi_time_filter ╤Б fallback ╨╜╨░ ╨╗╨╡╨│╨░╤Б╨╕-╨╗╨╛╨│╨╕╨║╤Г.
    
    Args:
        candles: ╨б╨┐╨╕╤Б╨╛╨║ ╤Б╨▓╨╡╤З╨╡╨╣
        rsi: ╨в╨╡╨║╤Г╤Й╨╡╨╡ ╨╖╨╜╨░╤З╨╡╨╜╨╕╨╡ RSI
        signal: ╨в╨╛╤А╨│╨╛╨▓╤Л╨╣ ╤Б╨╕╨│╨╜╨░╨╗ ('ENTER_LONG' ╨╕╨╗╨╕ 'ENTER_SHORT')
        symbol: ╨б╨╕╨╝╨▓╨╛╨╗ ╨╝╨╛╨╜╨╡╤В╤Л (╨╛╨┐╤Ж╨╕╨╛╨╜╨░╨╗╤М╨╜╨╛, ╨┤╨╗╤П ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╤Е ╨╜╨░╤Б╤В╤А╨╛╨╡╨║)
        individual_settings: ╨Ш╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╨╝╨╛╨╜╨╡╤В╤Л (╨╛╨┐╤Ж╨╕╨╛╨╜╨░╨╗╤М╨╜╨╛)
    """
    try:
        if engine_check_rsi_time_filter is None:
            raise RuntimeError('engine filters unavailable')
        
        # тЬЕ ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨║╨╛╨╜╤Д╨╕╨│ ╤Б ╤Г╤З╨╡╤В╨╛╨╝ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╤Е ╨╜╨░╤Б╤В╤А╨╛╨╡╨║
        auto_config = bots_data.get('auto_bot_config', {}).copy()
        
        # ╨Х╤Б╨╗╨╕ ╨┐╨╡╤А╨╡╨┤╨░╨╜╤Л ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕ - ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╕╤Е
        if individual_settings is None and symbol:
            individual_settings = get_individual_coin_settings(symbol)
        
        if individual_settings:
            # ╨Ю╨▒╤К╨╡╨┤╨╕╨╜╤П╨╡╨╝ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Л╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╤Б ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╝╨╕ (╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡ ╨╕╨╝╨╡╤О╤В ╨┐╤А╨╕╨╛╤А╨╕╤В╨╡╤В)
            for key in ['rsi_time_filter_enabled', 'rsi_time_filter_candles', 
                       'rsi_time_filter_lower', 'rsi_time_filter_upper',
                       'rsi_long_threshold', 'rsi_short_threshold']:
                if key in individual_settings:
                    auto_config[key] = individual_settings[key]
        
        result = engine_check_rsi_time_filter(
            candles,
            rsi,
            signal,
            auto_config,
            calculate_rsi_history_func=calculate_rsi_history,
        )
        return {
            'allowed': bool(result.get('allowed')),
            'reason': result.get('reason'),
            'last_extreme_candles_ago': result.get('last_extreme_candles_ago'),
            'calm_candles': result.get('calm_candles'),
        }
    except Exception as exc:
        logger.error(f" ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨│╨╛ ╤Д╨╕╨╗╤М╤В╤А╨░: {exc}")
        return _legacy_check_rsi_time_filter(candles, rsi, signal, symbol=symbol, individual_settings=individual_settings)


def _run_exit_scam_ai_detection(symbol, candles):
    """AI-╨░╨╜╨░╨╗╨╕╨╖ ╤Б╨▓╨╡╤З╨╡╨╣ ╨╜╨░ ╨░╨╜╨╛╨╝╨░╨╗╨╕╨╕ (reuse ╨╕╨╖ ╨╗╨╡╨│╨░╤Б╨╕-╨╗╨╛╨│╨╕╨║╨╕)."""
    try:
        from bot_engine.bot_config import AIConfig
    except ImportError:
        return True

    if not (AIConfig.AI_ENABLED and AIConfig.AI_ANOMALY_DETECTION_ENABLED):
        return True

    try:
        ai_manager, ai_available = get_cached_ai_manager()
        if not ai_available or not ai_manager or not ai_manager.anomaly_detector:
            return True

        anomaly_result = ai_manager.anomaly_detector.detect(candles)
        if anomaly_result.get('is_anomaly'):
            severity = anomaly_result.get('severity', 0)
            anomaly_type = anomaly_result.get('anomaly_type', 'UNKNOWN')
            if severity > AIConfig.AI_ANOMALY_BLOCK_THRESHOLD:
                return False
            logger.warning(
                f"{symbol}: тЪая╕П ╨Я╨а╨Х╨Ф╨г╨Я╨а╨Х╨Ц╨Ф╨Х╨Э╨Ш╨Х (AI): "
                f"╨Р╨╜╨╛╨╝╨░╨╗╨╕╤П {anomaly_type} "
                f"(severity: {severity:.2%} - ╨╜╨╕╨╢╨╡ ╨┐╨╛╤А╨╛╨│╨░ {AIConfig.AI_ANOMALY_BLOCK_THRESHOLD:.2%})"
            )
    except ImportError as exc:
        logger.debug(f"{symbol}: AI ╨╝╨╛╨┤╤Г╨╗╤М ╨╜╨╡ ╨┤╨╛╤Б╤В╤Г╨┐╨╡╨╜: {exc}")
    except Exception as exc:
        logger.error(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ AI ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕: {exc}")
    return True


def _check_loss_reentry_protection_static(symbol, candles, loss_reentry_count, loss_reentry_candles, individual_settings=None):
    """
    ╨б╤В╨░╤В╨╕╤З╨╡╤Б╨║╨░╤П ╤Д╤Г╨╜╨║╤Ж╨╕╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨╖╨░╤Й╨╕╤В╤Л ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓ ╨┐╨╛╤Б╨╗╨╡ ╤Г╨▒╤Л╤В╨╛╤З╨╜╤Л╤Е ╨╖╨░╨║╤А╤Л╤В╨╕╨╣
    
    Args:
        symbol: ╨б╨╕╨╝╨▓╨╛╨╗ ╨╝╨╛╨╜╨╡╤В╤Л
        candles: ╨б╨┐╨╕╤Б╨╛╨║ ╤Б╨▓╨╡╤З╨╡╨╣ ╨┤╨╗╤П ╨┐╨╛╨┤╤Б╤З╨╡╤В╨░ ╨┐╤А╨╛╤И╨╡╨┤╤И╨╕╤Е ╤Б╨▓╨╡╤З╨╡╨╣
        loss_reentry_count: ╨Ъ╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╛ ╤Г╨▒╤Л╤В╨╛╤З╨╜╤Л╤Е ╤Б╨┤╨╡╨╗╨╛╨║ ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ (N)
        loss_reentry_candles: ╨Ъ╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╛ ╤Б╨▓╨╡╤З╨╡╨╣ ╨┤╨╗╤П ╨╛╨╢╨╕╨┤╨░╨╜╨╕╤П (X)
        individual_settings: ╨Ш╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╨╝╨╛╨╜╨╡╤В╤Л (╨╛╨┐╤Ж╨╕╨╛╨╜╨░╨╗╤М╨╜╨╛)
    
    Returns:
        dict: {'allowed': bool, 'reason': str, 'candles_passed': int}
    """
    try:
        # тЬЕ ╨г╨С╨а╨Р╨Э╨Ю: ╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╨╜╨░ ╨╛╤В╨║╤А╤Л╤В╤Г╤О ╨┐╨╛╨╖╨╕╤Ж╨╕╤О ╨┤╨╛╨╗╨╢╨╜╨░ ╨▒╤Л╤В╤М ╤В╨╛╨╗╤М╨║╨╛ ╨▓ should_open_long/short
        # ╨б╤В╨░╤В╨╕╤З╨╡╤Б╨║╨░╤П ╤Д╤Г╨╜╨║╤Ж╨╕╤П ╨▓╤Б╨╡╨│╨┤╨░ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╤В ╤Д╨╕╨╗╤М╤В╤А, ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨┤╨╡╨╗╨░╨╡╤В╤Б╤П ╨╜╨░ ╤Г╤А╨╛╨▓╨╜╨╡ ╨▒╨╛╤В╨░
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╨╖╨░╨║╤А╤Л╤В╤Л╤Е ╤Б╨┤╨╡╨╗╨╛╨║ ╨┤╨╗╤П ╤Н╤В╨╛╨│╨╛ ╤Б╨╕╨╝╨▓╨╛╨╗╨░
        from bot_engine.bots_database import get_bots_database
        bots_db = get_bots_database()
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╨╖╨░╨║╤А╤Л╤В╤Л╤Е ╤Б╨┤╨╡╨╗╨╛╨║ ╨┐╨╛ ╤Б╨╕╨╝╨▓╨╛╨╗╤Г, ╨╛╤В╤Б╨╛╤А╤В╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╨╡ ╨┐╨╛ ╨▓╤А╨╡╨╝╨╡╨╜╨╕ ╨╖╨░╨║╤А╤Л╤В╨╕╤П (╨╜╨╛╨▓╤Л╨╡ ╨┐╨╡╤А╨▓╤Л╨╝╨╕)
        closed_trades = bots_db.get_bot_trades_history(
            bot_id=None,
            symbol=symbol,
            status='CLOSED',
            decision_source=None,
            limit=loss_reentry_count,
            offset=0
        )
        
        # ╨Х╤Б╨╗╨╕ ╨╜╨╡╤В ╨╖╨░╨║╤А╤Л╤В╤Л╤Е ╤Б╨┤╨╡╨╗╨╛╨║ - ╤А╨░╨╖╤А╨╡╤И╨░╨╡╨╝ ╨▓╤Е╨╛╨┤
        if not closed_trades or len(closed_trades) < loss_reentry_count:
            return {
                'allowed': True,
                'reason': f'╨Э╨╡╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ ╨╖╨░╨║╤А╤Л╤В╤Л╤Е ╤Б╨┤╨╡╨╗╨╛╨║ ({len(closed_trades) if closed_trades else 0} < {loss_reentry_count})',
                'candles_passed': None
            }
        
        # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ю: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨▓╤Б╨╡ ╨╗╨╕ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╤Б╨┤╨╡╨╗╨╛╨║ ╨▒╤Л╨╗╨╕ ╨▓ ╨╝╨╕╨╜╤Г╤Б
        # ╨Т╨░╨╢╨╜╨╛: ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╕╨╝╨╡╨╜╨╜╨╛ ╨Я╨Ю╨б╨Ы╨Х╨Ф╨Э╨Ш╨Х N ╤Б╨┤╨╡╨╗╨╛╨║ ╨┐╨╛ ╨▓╤А╨╡╨╝╨╡╨╜╨╕ ╨╖╨░╨║╤А╤Л╤В╨╕╤П (╨╛╨╜╨╕ ╤Г╨╢╨╡ ╨╛╤В╤Б╨╛╤А╤В╨╕╤А╨╛╨▓╨░╨╜╤Л DESC)
        all_losses = True
        for trade in closed_trades:
            pnl = trade.get('pnl', 0)
            # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤З╤В╨╛ PnL ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜ ╨╕ ╨┤╨╡╨╣╤Б╤В╨▓╨╕╤В╨╡╨╗╤М╨╜╨╛ ╨╛╤В╤А╨╕╤Ж╨░╤В╨╡╨╗╤М╨╜╤Л╨╣ (╤Б╤В╤А╨╛╨│╨╛ < 0)
            try:
                pnl_float = float(pnl) if pnl is not None else 0.0
                # ╨Х╤Б╨╗╨╕ ╤Е╨╛╤В╤П ╨▒╤Л ╨╛╨┤╨╜╨░ ╤Б╨┤╨╡╨╗╨║╨░ >= 0 (╨┐╤А╨╕╨▒╤Л╨╗╤М╨╜╨░╤П ╨╕╨╗╨╕ ╨▒╨╡╨╖╤Г╨▒╤Л╤В╨╛╤З╨╜╨░╤П) - ╨╜╨╡ ╨▓╤Б╨╡ ╨▓ ╨╝╨╕╨╜╤Г╤Б
                if pnl_float >= 0:
                    all_losses = False
                    break
            except (ValueError, TypeError):
                # ╨Х╤Б╨╗╨╕ ╨╜╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╤А╨╡╨╛╨▒╤А╨░╨╖╨╛╨▓╨░╤В╤М PnL - ╤Б╤З╨╕╤В╨░╨╡╨╝ ╤З╤В╨╛ ╨╜╨╡ ╤Г╨▒╤Л╤В╨╛╤З╨╜╨░╤П
                all_losses = False
                break
        
        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Х╤Б╨╗╨╕ ╨Э╨Х ╨Т╨б╨Х ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╤Б╨┤╨╡╨╗╨╛╨║ ╨▓ ╨╝╨╕╨╜╤Г╤Б - ╨а╨Р╨Ч╨а╨Х╨и╨Р╨Х╨Ь ╨▓╤Е╨╛╨┤ (╤Д╨╕╨╗╤М╤В╤А ╨Э╨Х ╤А╨░╨▒╨╛╤В╨░╨╡╤В)
        if not all_losses:
            return {
                'allowed': True,
                'reason': f'╨Э╨╡ ╨▓╤Б╨╡ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ {loss_reentry_count} ╤Б╨┤╨╡╨╗╨╛╨║ ╨▓ ╨╝╨╕╨╜╤Г╤Б - ╨╡╤Б╤В╤М ╨┐╤А╨╕╨▒╤Л╨╗╤М╨╜╤Л╨╡ ╤Б╨┤╨╡╨╗╨║╨╕, ╤Д╨╕╨╗╤М╤В╤А ╨╜╨╡ ╨┐╤А╨╕╨╝╨╡╨╜╤П╨╡╤В╤Б╤П',
                'candles_passed': None
            }
        
        # ╨Т╤Б╨╡ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╤Б╨┤╨╡╨╗╨╛╨║ ╨▓ ╨╝╨╕╨╜╤Г╤Б - ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨║╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╛ ╨┐╤А╨╛╤И╨╡╨┤╤И╨╕╤Е ╤Б╨▓╨╡╤З╨╡╨╣
        last_trade = closed_trades[0]  # ╨б╨░╨╝╨░╤П ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╤П╤П ╨╖╨░╨║╤А╤Л╤В╨░╤П ╤Б╨┤╨╡╨╗╨║╨░
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ timestamp ╨╖╨░╨║╤А╤Л╤В╨╕╤П ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╡╨╣ ╤Б╨┤╨╡╨╗╨║╨╕
        exit_timestamp = last_trade.get('exit_timestamp')
        if not exit_timestamp:
            exit_time_str = last_trade.get('exit_time')
            if exit_time_str:
                try:
                    from datetime import datetime
                    if isinstance(exit_time_str, str):
                        exit_dt = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
                        exit_timestamp = int(exit_dt.timestamp())
                    else:
                        exit_timestamp = int(exit_time_str)
                except:
                    return {'allowed': True, 'reason': '╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╨▓╤А╨╡╨╝╤П ╨╖╨░╨║╤А╤Л╤В╨╕╤П', 'candles_passed': None}
            else:
                return {'allowed': True, 'reason': '╨Э╨╡╤В ╨┤╨░╨╜╨╜╤Л╤Е ╨╛ ╨▓╤А╨╡╨╝╨╡╨╜╨╕ ╨╖╨░╨║╤А╤Л╤В╨╕╤П', 'candles_passed': None}
        
        # ╨Х╤Б╨╗╨╕ exit_timestamp ╨▓ ╨╝╨╕╨╗╨╗╨╕╤Б╨╡╨║╤Г╨╜╨┤╨░╤Е, ╨║╨╛╨╜╨▓╨╡╤А╤В╨╕╤А╤Г╨╡╨╝ ╨▓ ╤Б╨╡╨║╤Г╨╜╨┤╤Л
        if exit_timestamp > 1e12:
            exit_timestamp = exit_timestamp / 1000
        
        # ╨Я╨╛╨┤╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╨╝ ╨║╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╛ ╤Б╨▓╨╡╤З╨╡╨╣, ╨┐╤А╨╛╤И╨╡╨┤╤И╨╕╤Е ╤Б ╨╝╨╛╨╝╨╡╨╜╤В╨░ ╨╖╨░╨║╤А╤Л╤В╨╕╤П
        CANDLE_INTERVAL_SECONDS = 6 * 3600  # 6 ╤З╨░╤Б╨╛╨▓
        
        if not candles or len(candles) == 0:
            return {'allowed': True, 'reason': '╨Э╨╡╤В ╤Б╨▓╨╡╤З╨╡╨╣ ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕', 'candles_passed': None}
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ timestamp ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╡╨╣ ╤Б╨▓╨╡╤З╨╕
        last_candle = candles[-1]
        last_candle_timestamp = last_candle.get('timestamp', 0)
        if last_candle_timestamp > 1e12:
            last_candle_timestamp = last_candle_timestamp / 1000
        
        # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ю: ╨Я╨╛╨┤╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╨╝ ╨║╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╛ ╤Б╨▓╨╡╤З╨╡╨╣ ╤Б ╨╝╨╛╨╝╨╡╨╜╤В╨░ ╨╖╨░╨║╤А╤Л╤В╨╕╤П
        # ╨б╨▓╨╡╤З╨╕ ╤Г╨╢╨╡ ╨╛╤В╤Б╨╛╤А╤В╨╕╤А╨╛╨▓╨░╨╜╤Л ╨┐╨╛ ╨▓╤А╨╡╨╝╨╡╨╜╨╕ (╤Б╤В╨░╤А╤Л╨╡ -> ╨╜╨╛╨▓╤Л╨╡)
        candles_passed = 0
        
        # ╨Ш╤Й╨╡╨╝ ╨┐╨╡╤А╨▓╤Г╤О ╤Б╨▓╨╡╤З╤Г, ╨║╨╛╤В╨╛╤А╨░╤П ╨Я╨Ю╨Ы╨Э╨Ю╨б╨в╨м╨о ╨┐╨╛╨╖╨╢╨╡ ╨▓╤А╨╡╨╝╨╡╨╜╨╕ ╨╖╨░╨║╤А╤Л╤В╨╕╤П
        # ╨б╨▓╨╡╤З╨░ ╤Б╤З╨╕╤В╨░╨╡╤В╤Б╤П ╨┐╤А╨╛╤И╨╡╨┤╤И╨╡╨╣, ╨╡╤Б╨╗╨╕ ╨╡╤С ╨╜╨░╤З╨░╨╗╨╛ >= ╨▓╤А╨╡╨╝╨╡╨╜╨╕ ╨╖╨░╨║╤А╤Л╤В╨╕╤П
        for i, candle in enumerate(candles):
            candle_timestamp = candle.get('timestamp', 0)
            if candle_timestamp > 1e12:
                candle_timestamp = candle_timestamp / 1000
            
            # ╨Х╤Б╨╗╨╕ ╨╜╨░╤З╨░╨╗╨╛ ╤Б╨▓╨╡╤З╨╕ >= ╨▓╤А╨╡╨╝╨╡╨╜╨╕ ╨╖╨░╨║╤А╤Л╤В╨╕╤П, ╤Б╤З╨╕╤В╨░╨╡╨╝ ╤Н╤В╤Г ╨╕ ╨▓╤Б╨╡ ╨┐╨╛╤Б╨╗╨╡╨┤╤Г╤О╤Й╨╕╨╡ ╤Б╨▓╨╡╤З╨╕
            if candle_timestamp >= exit_timestamp:
                candles_passed = len(candles) - i
                break
        
        # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ю: ╨Х╤Б╨╗╨╕ ╨╜╨╡ ╨╜╨░╤И╨╗╨╕ ╤Б╨▓╨╡╤З╨╡╨╣ ╤З╨╡╤А╨╡╨╖ ╨┐╨╡╤А╨╡╨▒╨╛╤А, ╤Б╤З╨╕╤В╨░╨╡╨╝ ╨┐╨╛ ╨▓╤А╨╡╨╝╨╡╨╜╨╕
        # ╨н╤В╨╛ ╨▒╨╛╨╗╨╡╨╡ ╨╜╨░╨┤╨╡╨╢╨╜╤Л╨╣ ╨╝╨╡╤В╨╛╨┤ ╨┤╨╗╤П 6h ╤Б╨▓╨╡╤З╨╡╨╣
        if candles_passed == 0:
            time_diff_seconds = last_candle_timestamp - exit_timestamp
            if time_diff_seconds > 0:
                # ╨б╤З╨╕╤В╨░╨╡╨╝ ╨║╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╛ ╨┐╨╛╨╗╨╜╤Л╤Е 6-╤З╨░╤Б╨╛╨▓╤Л╤Е ╨╕╨╜╤В╨╡╤А╨▓╨░╨╗╨╛╨▓
                candles_passed = max(1, int(time_diff_seconds / CANDLE_INTERVAL_SECONDS))
        
        # тЬЕ ╨Ф╨Ю╨Я╨Ю╨Ы╨Э╨Ш╨в╨Х╨Ы╨м╨Э╨Р╨п ╨Я╨а╨Ю╨Т╨Х╨а╨Ъ╨Р: ╨Х╤Б╨╗╨╕ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╤П╤П ╤Б╨▓╨╡╤З╨░ ╤П╨▓╨╜╨╛ ╨┐╨╛╤Б╨╗╨╡ ╨╖╨░╨║╤А╤Л╤В╨╕╤П
        if candles_passed == 0 and last_candle_timestamp > exit_timestamp:
            # ╨Ь╨╕╨╜╨╕╨╝╤Г╨╝ 1 ╤Б╨▓╨╡╤З╨░ ╨┐╤А╨╛╤И╨╗╨░, ╨╡╤Б╨╗╨╕ ╤В╨╡╨║╤Г╤Й╨░╤П ╤Б╨▓╨╡╤З╨░ ╨┐╨╛╤Б╨╗╨╡ ╨╖╨░╨║╤А╤Л╤В╨╕╤П
            candles_passed = 1
        
        # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ю: ╨Ъ╨╛╨╜╨▓╨╡╤А╤В╨╕╤А╤Г╨╡╨╝ loss_reentry_candles ╨▓ int ╨┤╨╗╤П ╨║╨╛╤А╤А╨╡╨║╤В╨╜╨╛╨│╨╛ ╤Б╤А╨░╨▓╨╜╨╡╨╜╨╕╤П
        try:
            loss_reentry_candles_int = int(loss_reentry_candles) if loss_reentry_candles is not None else 3
        except (ValueError, TypeError):
            loss_reentry_candles_int = 3
        
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨┐╤А╨╛╤И╨╗╨╛ ╨╗╨╕ ╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ ╤Б╨▓╨╡╤З╨╡╨╣
        if candles_passed < loss_reentry_candles_int:
            return {
                'allowed': False,
                'reason': f'╨Я╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ {loss_reentry_count} ╤Б╨┤╨╡╨╗╨╛╨║ ╨▓ ╨╝╨╕╨╜╤Г╤Б, ╨┐╤А╨╛╤И╨╗╨╛ ╤В╨╛╨╗╤М╨║╨╛ {candles_passed} ╤Б╨▓╨╡╤З╨╡╨╣ (╤В╤А╨╡╨▒╤Г╨╡╤В╤Б╤П {loss_reentry_candles_int})',
                'candles_passed': candles_passed
            }
        
        return {
            'allowed': True,
            'reason': f'╨Я╤А╨╛╤И╨╗╨╛ {candles_passed} ╤Б╨▓╨╡╤З╨╡╨╣ ╤Б ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╡╨│╨╛ ╤Г╨▒╤Л╤В╨║╨░ (╤В╤А╨╡╨▒╤Г╨╡╤В╤Б╤П {loss_reentry_candles})',
            'candles_passed': candles_passed
        }
        
    except Exception as e:
        # ╨Я╤А╨╕ ╨╛╤И╨╕╨▒╨║╨╡ ╤А╨░╨╖╤А╨╡╤И╨░╨╡╨╝ ╨▓╤Е╨╛╨┤ (╨▒╨╡╨╖╨╛╨┐╨░╤Б╨╜╨╡╨╡, ╨║╨░╨║ ╨▓ bot_class.py)
        logger.error(f"{symbol}: тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨╖╨░╤Й╨╕╤В╤Л ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓ (static): {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {
            'allowed': True,
            'reason': f'╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕: {str(e)}',
            'candles_passed': None
        }
        
    except Exception as e:
        logger.debug(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨╖╨░╤Й╨╕╤В╤Л ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓: {e}")
        return {'allowed': True, 'reason': f'╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕: {str(e)}', 'candles_passed': None}


def check_exit_scam_filter(symbol, coin_data):
    """╨г╨╜╨╕╤Д╨╕╤Ж╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╨╣ exit-scam ╤Д╨╕╨╗╤М╤В╤А ╤Б AI-╨░╨╜╨░╨╗╨╕╨╖╨╛╨╝ ╨╕ fallback."""
    try:
        if engine_check_exit_scam_filter is None:
            raise RuntimeError('engine filters unavailable')
        
        # тЬЕ ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨║╨╛╨╜╤Д╨╕╨│ ╤Б ╤Г╤З╨╡╤В╨╛╨╝ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╤Е ╨╜╨░╤Б╤В╤А╨╛╨╡╨║
        auto_config = bots_data.get('auto_bot_config', {}).copy()
        individual_settings = get_individual_coin_settings(symbol)
        
        if individual_settings:
            # ╨Ю╨▒╤К╨╡╨┤╨╕╨╜╤П╨╡╨╝ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Л╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╤Б ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╝╨╕ (╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡ ╨╕╨╝╨╡╤О╤В ╨┐╤А╨╕╨╛╤А╨╕╤В╨╡╤В)
            for key in ['exit_scam_enabled', 'exit_scam_candles', 
                       'exit_scam_single_candle_percent', 'exit_scam_multi_candle_count',
                       'exit_scam_multi_candle_percent']:
                if key in individual_settings:
                    auto_config[key] = individual_settings[key]
        
        exchange_obj = get_exchange()
        if not exchange_obj:
            return False

        base_allowed = engine_check_exit_scam_filter(
            symbol,
            coin_data,
            auto_config,
            exchange_obj,
            ensure_exchange_initialized,
        )

        if not base_allowed:
            return False

        chart_response = exchange_obj.get_chart_data(symbol, '6h', '30d')
        candles = chart_response.get('data', {}).get('candles', []) if chart_response and chart_response.get('success') else []
        if candles:
            return _run_exit_scam_ai_detection(symbol, candles)
        return True
    except Exception as exc:
        logger.error(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ exit-scam (core): {exc}")
        return _legacy_check_exit_scam_filter(symbol, coin_data, individual_settings=individual_settings)

def get_coin_rsi_data(symbol, exchange_obj=None):
    """╨Я╨╛╨╗╤Г╤З╨░╨╡╤В RSI ╨┤╨░╨╜╨╜╤Л╨╡ ╨┤╨╗╤П ╨╛╨┤╨╜╨╛╨╣ ╨╝╨╛╨╜╨╡╤В╤Л (6H ╤В╨░╨╣╨╝╤Д╤А╨╡╨╣╨╝)"""
    # тЪб ╨Т╨║╨╗╤О╤З╨░╨╡╨╝ ╤В╤А╨╡╨╣╤Б╨╕╨╜╨│ ╨┤╨╗╤П ╤Н╤В╨╛╨│╨╛ ╨┐╨╛╤В╨╛╨║╨░ (╨╡╤Б╨╗╨╕ ╨▓╨║╨╗╤О╤З╨╡╨╜ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╨╛)
    try:
        from bot_engine.bot_config import SystemConfig
        if SystemConfig.ENABLE_CODE_TRACING:
            from trace_debug import enable_trace
            enable_trace()
    except:
        pass

    if shutdown_flag.is_set():
        logger.debug(f"{symbol}: ╨Я╤А╨╛╨┐╤Г╤Б╨║ ╨░╨╜╨░╨╗╨╕╨╖╨░ RSI (shutdown requested)")
        return None
    
    # тЪб ╨б╨Х╨Ь╨Р╨д╨Ю╨а: ╨Ю╨│╤А╨░╨╜╨╕╤З╨╕╨▓╨░╨╡╨╝ ╨╛╨┤╨╜╨╛╨▓╤А╨╡╨╝╨╡╨╜╨╜╤Л╨╡ API ╨╖╨░╨┐╤А╨╛╤Б╤Л ╨║ ╨▒╨╕╤А╨╢╨╡ (╨╡╤Б╨╗╨╕ ╨╜╨╡╤В ╨▓ ╨║╤Н╤И╨╡)
    # ╨н╤В╨╛ ╨┐╤А╨╡╨┤╨╛╤В╨▓╤А╨░╤Й╨░╨╡╤В ╨┐╨╡╤А╨╡╨│╤А╤Г╨╖╨║╤Г API ╨▒╨╕╤А╨╢╨╕
    global _exchange_api_semaphore
    try:
        _exchange_api_semaphore
    except NameError:
        _exchange_api_semaphore = threading.Semaphore(5)  # тЪб ╨г╨╝╨╡╨╜╤М╤И╨╕╨╗╨╕ ╨┤╨╛ 5 ╨┤╨╗╤П ╤Б╤В╨░╨▒╨╕╨╗╤М╨╜╨╛╤Б╤В╨╕
    
    import time
    thread_start = time.time()
    data_source = 'cache'
    # print(f"[{time.strftime('%H:%M:%S')}] >>> ╨Э╨Р╨з╨Р╨Ы╨Ю get_coin_rsi_data({symbol})", flush=True)  # ╨Ю╤В╨║╨╗╤О╤З╨╡╨╜╨╛ ╨┤╨╗╤П ╤Б╨║╨╛╤А╨╛╤Б╤В╨╕
    
    try:
        # тЬЕ ╨д╨Ш╨Ы╨м╨в╨а 0: ╨Ф╨Х╨Ы╨Ш╨б╨в╨Ш╨Э╨У╨Ю╨Т╨л╨Х ╨Ь╨Ю╨Э╨Х╨в╨л - ╨б╨Р╨Ь╨л╨Щ ╨Я╨Х╨а╨Т╨л╨Щ!
        # ╨Ш╤Б╨║╨╗╤О╤З╨░╨╡╨╝ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л ╨Ф╨Ю ╨▓╤Б╨╡╤Е ╨╛╤Б╤В╨░╨╗╤М╨╜╤Л╤Е ╨┐╤А╨╛╨▓╨╡╤А╨╛╨║
        # ╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л ╨╕╨╖ ╤Д╨░╨╣╨╗╨░
        delisted_coins = _get_cached_delisted_coins()
        
        if symbol in delisted_coins:
            delisting_info = delisted_coins.get(symbol, {})
            logger.info(f"{symbol}: ╨Ш╤Б╨║╨╗╤О╤З╨░╨╡╨╝ ╨╕╨╖ ╨▓╤Б╨╡╤Е ╨┐╤А╨╛╨▓╨╡╤А╨╛╨║ - {delisting_info.get('reason', 'Delisting detected')}")
            # ╨Т╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╨╝ ╨╝╨╕╨╜╨╕╨╝╨░╨╗╤М╨╜╤Л╨╡ ╨┤╨░╨╜╨╜╤Л╨╡ ╨┤╨╗╤П ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╤Е ╨╝╨╛╨╜╨╡╤В
            return {
                'symbol': symbol,
                'rsi6h': 0,
                'trend6h': 'NEUTRAL',
                'rsi_zone': 'NEUTRAL',
                'signal': 'WAIT',
                'price': 0,
                'change24h': 0,
                'last_update': datetime.now().isoformat(),
                'trading_status': 'Closed',
                'is_delisting': True,
                'delisting_reason': delisting_info.get('reason', 'Delisting detected'),
                'blocked_by_delisting': True
            }
        
        # тЬЕ ╨д╨Ш╨Ы╨м╨в╨а 1: Whitelist/Blacklist/Scope - ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨Ф╨Ю ╨╖╨░╨│╤А╤Г╨╖╨║╨╕ ╨┤╨░╨╜╨╜╤Л╤Е ╤Б ╨▒╨╕╤А╨╢╨╕
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╨║╨╛╨╜╤Д╨╕╨│ ╨╜╨╡ ╨╝╨╡╨╜╤П╨╡╤В╤Б╤П ╨▓╨╛ ╨▓╤А╨╡╨╝╤П ╨▓╤Л╨┐╨╛╨╗╨╜╨╡╨╜╨╕╤П, ╨▒╨╡╨╖╨╛╨┐╨░╤Б╨╜╨╛ ╤З╨╕╤В╨░╤В╤М
        auto_config = bots_data.get('auto_bot_config', {})
        scope = auto_config.get('scope', 'all')
        whitelist = auto_config.get('whitelist', [])
        blacklist = auto_config.get('blacklist', [])
        
        is_blocked_by_scope = False
        
        if scope == 'whitelist':
            # ╨а╨╡╨╢╨╕╨╝ ╨в╨Ю╨Ы╨м╨Ъ╨Ю whitelist - ╤А╨░╨▒╨╛╤В╨░╨╡╨╝ ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╤Б ╨╝╨╛╨╜╨╡╤В╨░╨╝╨╕ ╨╕╨╖ ╨▒╨╡╨╗╨╛╨│╨╛ ╤Б╨┐╨╕╤Б╨║╨░
            if symbol not in whitelist:
                is_blocked_by_scope = True
                logger.debug(f"{symbol}: тЭМ ╨а╨╡╨╢╨╕╨╝ WHITELIST - ╨╝╨╛╨╜╨╡╤В╨░ ╨╜╨╡ ╨▓ ╨▒╨╡╨╗╨╛╨╝ ╤Б╨┐╨╕╤Б╨║╨╡")
        
        elif scope == 'blacklist':
            # ╨а╨╡╨╢╨╕╨╝ ╨в╨Ю╨Ы╨м╨Ъ╨Ю blacklist - ╤А╨░╨▒╨╛╤В╨░╨╡╨╝ ╤Б╨╛ ╨Т╨б╨Х╨Ь╨Ш ╨╝╨╛╨╜╨╡╤В╨░╨╝╨╕ ╨Ъ╨а╨Ю╨Ь╨Х ╤З╨╡╤А╨╜╨╛╨│╨╛ ╤Б╨┐╨╕╤Б╨║╨░
            if symbol in blacklist:
                is_blocked_by_scope = True
                logger.debug(f"{symbol}: тЭМ ╨а╨╡╨╢╨╕╨╝ BLACKLIST - ╨╝╨╛╨╜╨╡╤В╨░ ╨▓ ╤З╨╡╤А╨╜╨╛╨╝ ╤Б╨┐╨╕╤Б╨║╨╡")
        
        elif scope == 'all':
            # ╨а╨╡╨╢╨╕╨╝ ALL - ╤А╨░╨▒╨╛╤В╨░╨╡╨╝ ╤Б╨╛ ╨Т╨б╨Х╨Ь╨Ш ╨╝╨╛╨╜╨╡╤В╨░╨╝╨╕, ╨╜╨╛ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╛╨▒╨░ ╤Б╨┐╨╕╤Б╨║╨░
            if symbol in blacklist:
                is_blocked_by_scope = True
                logger.debug(f"{symbol}: тЭМ ╨Ь╨╛╨╜╨╡╤В╨░ ╨▓ ╤З╨╡╤А╨╜╨╛╨╝ ╤Б╨┐╨╕╤Б╨║╨╡")
            # ╨Х╤Б╨╗╨╕ ╨▓ whitelist - ╨┤╨░╨╡╨╝ ╨┐╤А╨╕╨╛╤А╨╕╤В╨╡╤В (╨╗╨╛╨│╨╕╤А╤Г╨╡╨╝, ╨╜╨╛ ╨╜╨╡ ╨▒╨╗╨╛╨║╨╕╤А╤Г╨╡╨╝)
            if whitelist and symbol in whitelist:
                logger.debug(f"{symbol}: тнР ╨Т ╨▒╨╡╨╗╨╛╨╝ ╤Б╨┐╨╕╤Б╨║╨╡ (╨┐╤А╨╕╨╛╤А╨╕╤В╨╡╤В)")
        
        # ╨С╨Х╨Ч ╨╖╨░╨┤╨╡╤А╨╢╨║╨╕ - ╤Б╨╡╨╝╨░╤Д╨╛╤А ╨╕ ThreadPool ╤Г╨╢╨╡ ╨║╨╛╨╜╤В╤А╨╛╨╗╨╕╤А╤Г╤О╤В rate limit
        
        # logger.debug(f"[DEBUG] ╨Ю╨▒╤А╨░╨▒╨╛╤В╨║╨░ {symbol}...")  # ╨Ю╤В╨║╨╗╤О╤З╨╡╨╜╨╛ ╨┤╨╗╤П ╤Г╤Б╨║╨╛╤А╨╡╨╜╨╕╤П
        
        # ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨┐╨╡╤А╨╡╨┤╨░╨╜╨╜╤Г╤О ╨▒╨╕╤А╨╢╤Г ╨╕╨╗╨╕ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Г╤О
        # print(f"[{time.strftime('%H:%M:%S')}] >>> {symbol}: ╨Я╨╛╨╗╤Г╤З╨╡╨╜╨╕╨╡ exchange...", flush=True)  # ╨Ю╤В╨║╨╗╤О╤З╨╡╨╜╨╛
        from bots_modules.imports_and_globals import get_exchange
        exchange_to_use = exchange_obj if exchange_obj is not None else get_exchange()
        
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╤З╤В╨╛ ╨▒╨╕╤А╨╢╨░ ╨┤╨╛╤Б╤В╤Г╨┐╨╜╨░
        if exchange_to_use is None:
            logger.error(f"╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╨┤╨░╨╜╨╜╤Л╤Е ╨┤╨╗╤П {symbol}: 'NoneType' object has no attribute 'get_chart_data'")
            return None
        
        # тЪб ╨Ю╨Я╨в╨Ш╨Ь╨Ш╨Ч╨Р╨ж╨Ш╨п: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨║╤Н╤И ╤Б╨▓╨╡╤З╨╡╨╣ ╨Я╨Х╨а╨Х╨Ф ╨╖╨░╨┐╤А╨╛╤Б╨╛╨╝ ╨║ ╨▒╨╕╤А╨╢╨╡!
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╤З╤В╨╡╨╜╨╕╨╡ ╨║╤Н╤И╨░ - ╨▒╨╡╨╖╨╛╨┐╨░╤Б╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
        candles = None
        candles_cache = coins_rsi_data.get('candles_cache', {})
        if symbol in candles_cache:
            cached_data = candles_cache[symbol]
            candles = cached_data.get('candles')
            # logger.debug(f"[CACHE] {symbol}: ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨║╤Н╤И ╤Б╨▓╨╡╤З╨╡╨╣")  # ╨Ю╤В╨║╨╗╤О╤З╨╡╨╜╨╛ ╨┤╨╗╤П ╤Б╨║╨╛╤А╨╛╤Б╤В╨╕
        
        # ╨Х╤Б╨╗╨╕ ╨╜╨╡╤В ╨▓ ╨║╤Н╤И╨╡ - ╨╖╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╤Б ╨▒╨╕╤А╨╢╨╕ (╤Б ╤Б╨╡╨╝╨░╤Д╨╛╤А╨╛╨╝!)
        if not candles:
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨╡╤Б╤В╤М ╨╗╨╕ ╨║╤Н╤И ╨▓ ╨┐╨░╨╝╤П╤В╨╕ ╨▓╨╛╨╛╨▒╤Й╨╡ (╨╝╨╛╨╢╨╡╤В ╨▒╤Л╤В╤М ╨╡╤Й╨╡ ╨╜╨╡ ╨╖╨░╨│╤А╤Г╨╢╨╡╨╜ ╨┐╤А╨╕ ╤Б╤В╨░╤А╤В╨╡)
            cache_loaded = bool(coins_rsi_data.get('candles_cache', {}))
            if not cache_loaded:
                logger.debug(f"тД╣я╕П {symbol}: ╨Ъ╤Н╤И ╤Б╨▓╨╡╤З╨╡╨╣ ╨╡╤Й╨╡ ╨╜╨╡ ╨╖╨░╨│╤А╤Г╨╢╨╡╨╜, ╨╖╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╤Б ╨▒╨╕╤А╨╢╨╕...")
            else:
                logger.info(f"тД╣я╕П {symbol}: ╨Э╨╡╤В ╨▓ ╨║╤Н╤И╨╡ ╤Б╨▓╨╡╤З╨╡╨╣, ╨╖╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╤Б ╨▒╨╕╤А╨╢╨╕...")
            # тЪб ╨б╨Х╨Ь╨Р╨д╨Ю╨а: ╨Ю╨│╤А╨░╨╜╨╕╤З╨╕╨▓╨░╨╡╨╝ ╨╛╨┤╨╜╨╛╨▓╤А╨╡╨╝╨╡╨╜╨╜╤Л╨╡ ╨╖╨░╨┐╤А╨╛╤Б╤Л ╨║ API ╨▒╨╕╤А╨╢╨╕
            with _exchange_api_semaphore:
                import time as time_module
                api_start = time_module.time()
                logger.info(f"ЁЯМР {symbol}: ╨Э╨░╤З╨░╨╗╨╛ ╨╖╨░╨┐╤А╨╛╤Б╨░ get_chart_data()...")
                
                chart_response = exchange_to_use.get_chart_data(symbol, '6h', '30d')
                
                api_duration = time_module.time() - api_start
                logger.info(f"ЁЯМР {symbol}: get_chart_data() ╨╖╨░╨▓╨╡╤А╤И╨╡╨╜ ╨╖╨░ {api_duration:.1f}╤Б")
                
                if not chart_response or not chart_response.get('success'):
                    logger.warning(f"тЭМ {symbol}: ╨Ю╤И╨╕╨▒╨║╨░: {chart_response.get('error', '╨Э╨╡╨╕╨╖╨▓╨╡╤Б╤В╨╜╨░╤П ╨╛╤И╨╕╨▒╨║╨░') if chart_response else '╨Э╨╡╤В ╨╛╤В╨▓╨╡╤В╨░'}")
                    return None
                
                candles = chart_response['data']['candles']
                logger.info(f"тЬЕ {symbol}: ╨б╨▓╨╡╤З╨╕ ╨╖╨░╨│╤А╤Г╨╢╨╡╨╜╤Л ╤Б ╨▒╨╕╤А╨╢╨╕ ({len(candles)} ╤Б╨▓╨╡╤З╨╡╨╣)")
                data_source = 'api'
                
                # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤Б╨▓╨╡╤З╨╕ ╨▓ ╨║╤Н╤И ╨┐╨╛╤Б╨╗╨╡ ╨╖╨░╨│╤А╤Г╨╖╨║╨╕ ╤Б ╨▒╨╕╤А╨╢╨╕!
                # ╨н╤В╨╛ ╨┐╤А╨╡╨┤╨╛╤В╨▓╤А╨░╤Й╨░╨╡╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╨╡ ╨╖╨░╨┐╤А╨╛╤Б╤Л ╨║ ╨▒╨╕╤А╨╢╨╡ ╨┤╨╗╤П ╤В╨╡╤Е ╨╢╨╡ ╨╝╨╛╨╜╨╡╤В
                try:
                    if candles and len(candles) >= 15:
                        # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨▓ ╤В╨╛╨╝ ╨╢╨╡ ╤Д╨╛╤А╨╝╨░╤В╨╡, ╤З╤В╨╛ ╨╕ get_coin_candles_only
                        candles_cache[symbol] = {
                            'symbol': symbol,
                            'candles': candles,
                            'timeframe': '6h',
                            'last_update': datetime.now().isoformat()
                        }
                        # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Л╨╣ ╨║╤Н╤И
                        coins_rsi_data['candles_cache'] = candles_cache
                        logger.debug(f"ЁЯТ╛ {symbol}: ╨б╨▓╨╡╤З╨╕ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╤Л ╨▓ ╨║╤Н╤И ({len(candles)} ╤Б╨▓╨╡╤З╨╡╨╣)")
                except Exception as cache_save_error:
                    logger.warning(f"тЪая╕П {symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П ╤Б╨▓╨╡╤З╨╡╨╣ ╨▓ ╨║╤Н╤И: {cache_save_error}")
        
        if not candles or len(candles) < 15:  # ╨С╨░╨╖╨╛╨▓╨░╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╨┤╨╗╤П RSI(14)
            return None
        
        # ╨а╨░╤Б╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╨╝ RSI ╨┤╨╗╤П 6H
        # Bybit ╨╛╤В╨┐╤А╨░╨▓╨╗╤П╨╡╤В ╤Б╨▓╨╡╤З╨╕ ╨▓ ╨┐╤А╨░╨▓╨╕╨╗╤М╨╜╨╛╨╝ ╨┐╨╛╤А╤П╨┤╨║╨╡ ╨┤╨╗╤П RSI (╨╛╤В ╤Б╤В╨░╤А╨╛╨╣ ╨║ ╨╜╨╛╨▓╨╛╨╣)
        closes = [candle['close'] for candle in candles]
        
        rsi = calculate_rsi(closes, 14)
        
        if rsi is None:
            logger.warning(f"╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╤А╨░╤Б╤Б╤З╨╕╤В╨░╤В╤М RSI ╨┤╨╗╤П {symbol}")
            return None
        
        # тЬЕ ╨а╨Р╨б╨з╨Ш╨в╨л╨Т╨Р╨Х╨Ь ╨в╨а╨Х╨Э╨Ф ╨б╨а╨Р╨Ч╨г ╨┤╨╗╤П ╨▓╤Б╨╡╤Е ╨╝╨╛╨╜╨╡╤В - ╨╕╨╖╨▒╨╡╨│╨░╨╡╨╝ "╨│╤Г╨╗╤П╨╜╨╕╤П" ╨┤╨░╨╜╨╜╤Л╤Е
        # ╨Э╨Х ╨г╨б╨в╨Р╨Э╨Р╨Т╨Ы╨Ш╨Т╨Р╨Х╨Ь ╨Ф╨Х╨д╨Ю╨Ы╨в╨Э╨л╨е ╨Ч╨Э╨Р╨з╨Х╨Э╨Ш╨Щ! ╨в╨╛╨╗╤М╨║╨╛ ╤А╨░╤Б╤Б╤З╨╕╤В╨░╨╜╨╜╤Л╨╡ ╨┤╨░╨╜╨╜╤Л╨╡!
        trend = None  # ╨Ш╨╖╨╜╨░╤З╨░╨╗╤М╨╜╨╛ None
        trend_analysis = None
        try:
            from bots_modules.calculations import analyze_trend_6h
            trend_analysis = analyze_trend_6h(symbol, exchange_obj=exchange_obj, candles_data=candles)
            if trend_analysis:
                trend = trend_analysis['trend']  # ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╤А╨░╤Б╤Б╤З╨╕╤В╨░╨╜╨╜╨╛╨╡ ╨╖╨╜╨░╤З╨╡╨╜╨╕╨╡!
            # ╨Э╨Х ╤Г╤Б╤В╨░╨╜╨░╨▓╨╗╨╕╨▓╨░╨╡╨╝ ╨┤╨╡╤Д╨╛╨╗╤В ╨╡╤Б╨╗╨╕ ╨░╨╜╨░╨╗╨╕╨╖ ╨╜╨╡ ╤Г╨┤╨░╨╗╤Б╤П - ╨╛╤Б╤В╨░╨▓╨╗╤П╨╡╨╝ None
        except Exception as e:
            logger.debug(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨░╨╜╨░╨╗╨╕╨╖╨░ ╤В╤А╨╡╨╜╨┤╨░: {e}")
            # ╨Э╨Х ╤Г╤Б╤В╨░╨╜╨░╨▓╨╗╨╕╨▓╨░╨╡╨╝ ╨┤╨╡╤Д╨╛╨╗╤В ╨┐╤А╨╕ ╨╛╤И╨╕╨▒╨║╨╡ - ╨╛╤Б╤В╨░╨▓╨╗╤П╨╡╨╝ None
        
        # ╨а╨░╤Б╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╨╝ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╨╡ ╨╖╨░ 24h (╨┐╤А╨╕╨╝╨╡╤А╨╜╨╛ 4 ╤Б╨▓╨╡╤З╨╕ 6H)
        change_24h = 0
        if len(closes) >= 5:
            change_24h = round(((closes[-1] - closes[-5]) / closes[-5]) * 100, 2)
        
        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨╛╨┐╤В╨╕╨╝╨░╨╗╤М╨╜╤Л╨╡ EMA ╨┐╨╡╤А╨╕╨╛╨┤╤Л ╨Ф╨Ю ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜╨╕╤П ╤Б╨╕╨│╨╜╨░╨╗╨░!
        # тЭМ ╨Ю╨в╨Ъ╨Ы╨о╨з╨Х╨Э╨Ю: EMA ╤Д╨╕╨╗╤М╤В╤А ╤Г╨┤╨░╨╗╨╡╨╜ ╨╕╨╖ ╤Б╨╕╤Б╤В╨╡╨╝╤Л
        # ema_periods = None
        # try:
        #     ema_periods = get_optimal_ema_periods(symbol)
        # except Exception as e:
        #     logger.debug(f"[EMA] ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╨╛╨┐╤В╨╕╨╝╨░╨╗╤М╨╜╤Л╤Е EMA ╨┤╨╗╤П {symbol}: {e}")
        #     ema_periods = {'ema_short': 50, 'ema_long': 200, 'accuracy': 0, 'analysis_method': 'default'}
        
        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╨╝╨╛╨╜╨╡╤В╤Л ╨Ф╨Ю ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜╨╕╤П ╤Б╨╕╨│╨╜╨░╨╗╨░!
        # ╨н╤В╨╛ ╨┐╨╛╨╖╨▓╨╛╨╗╤П╨╡╤В ╨╕╤Б╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╤В╤М ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡ ╨┐╨╛╤А╨╛╨│╨╕ RSI ╨┤╨╗╤П ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜╨╕╤П ╤Б╨╕╨│╨╜╨░╨╗╨░
        individual_settings = get_individual_coin_settings(symbol)
        
        # ╨Ю╨┐╤А╨╡╨┤╨╡╨╗╤П╨╡╨╝ ╨┐╨╛╤А╨╛╨│╨╕ RSI: ╤Б╨╜╨░╤З╨░╨╗╨░ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡, ╨╖╨░╤В╨╡╨╝ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Л╨╡
        rsi_long_threshold = individual_settings.get('rsi_long_threshold') if individual_settings else None
        if rsi_long_threshold is None:
            rsi_long_threshold = bots_data.get('auto_bot_config', {}).get('rsi_long_threshold', SystemConfig.RSI_OVERSOLD)
        
        rsi_short_threshold = individual_settings.get('rsi_short_threshold') if individual_settings else None
        if rsi_short_threshold is None:
            rsi_short_threshold = bots_data.get('auto_bot_config', {}).get('rsi_short_threshold', SystemConfig.RSI_OVERBOUGHT)
        
        # ╨Ю╨┐╤А╨╡╨┤╨╡╨╗╤П╨╡╨╝ RSI ╨╖╨╛╨╜╤Л ╤Б╨╛╨│╨╗╨░╤Б╨╜╨╛ ╤В╨╡╤Е╨╖╨░╨┤╨░╨╜╨╕╤О
        rsi_zone = 'NEUTRAL'
        signal = 'WAIT'
        
        # тЬЕ ╨д╨Ш╨Ы╨м╨в╨а 2: ╨С╨░╨╖╨╛╨▓╤Л╨╣ ╤Б╨╕╨│╨╜╨░╨╗ ╨Э╨Р ╨Ю╨б╨Э╨Ю╨Т╨Х OPTIMAL EMA ╨Я╨Х╨а╨Ш╨Ю╨Ф╨Ю╨Т!
        # тЬЕ ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╤Д╨╕╨╗╤М╤В╤А╨╛╨▓ ╨┐╨╛ ╤В╤А╨╡╨╜╨┤╤Г: ╤Б╨╜╨░╤З╨░╨╗╨░ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡, ╨╖╨░╤В╨╡╨╝ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Л╨╡
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╨║╨╛╨╜╤Д╨╕╨│ ╨╜╨╡ ╨╝╨╡╨╜╤П╨╡╤В╤Б╤П ╨▓╨╛ ╨▓╤А╨╡╨╝╤П ╨▓╤Л╨┐╨╛╨╗╨╜╨╡╨╜╨╕╤П, ╨▒╨╡╨╖╨╛╨┐╨░╤Б╨╜╨╛ ╤З╨╕╤В╨░╤В╤М
        avoid_down_trend = individual_settings.get('avoid_down_trend') if individual_settings else None
        if avoid_down_trend is None:
            avoid_down_trend = bots_data.get('auto_bot_config', {}).get('avoid_down_trend', False)
        
        avoid_up_trend = individual_settings.get('avoid_up_trend') if individual_settings else None
        if avoid_up_trend is None:
            avoid_up_trend = bots_data.get('auto_bot_config', {}).get('avoid_up_trend', False)
        
        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Ю╨┐╤А╨╡╨┤╨╡╨╗╤П╨╡╨╝ ╤Б╨╕╨│╨╜╨░╨╗ ╨╜╨░ ╨╛╤Б╨╜╨╛╨▓╨╡ Optimal EMA ╨┐╨╡╤А╨╕╨╛╨┤╨╛╨▓!
        # тЬЕ ╨г╨Я╨а╨Ю╨й╨Х╨Э╨Э╨Р╨п ╨Ы╨Ю╨У╨Ш╨Ъ╨Р: ╨г╨▒╤А╨░╨╗╨╕ ╤Д╨╕╨╗╤М╤В╤А ╨┐╨╛ EMA - ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ RSI
        # EMA ╤Б╨╗╨╕╤И╨║╨╛╨╝ ╨╖╨░╨┐╨░╨╖╨┤╤Л╨▓╨░╨╡╤В ╨╕ ╨▒╨╗╨╛╨║╨╕╤А╤Г╨╡╤В ╤Е╨╛╤А╨╛╤И╨╕╨╡ ╨▓╤Е╨╛╨┤╤Л ╨┐╨╛ RSI
        if True:  # ╨Ю╤Б╤В╨░╨▓╨╗╤П╨╡╨╝ ╤Б╤В╤А╤Г╨║╤В╤Г╤А╤Г ╨┤╨╗╤П ╨▓╨╛╨╖╨╝╨╛╨╢╨╜╨╛╨│╨╛ ╨▓╨╛╨╖╨▓╤А╨░╤В╨░ EMA ╨▓ ╨▒╤Г╨┤╤Г╤Й╨╡╨╝
            try:
                # тЬЕ ╨Ш╨б╨Я╨Ю╨Ы╨м╨Ч╨г╨Х╨Ь ╨Ш╨Э╨Ф╨Ш╨Т╨Ш╨Ф╨г╨Р╨Ы╨м╨Э╨л╨Х ╨Я╨Ю╨а╨Ю╨У╨Ш RSI ╨┤╨╗╤П ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜╨╕╤П ╤Б╨╕╨│╨╜╨░╨╗╨░!
                # ╨Ю╨┐╤А╨╡╨┤╨╡╨╗╤П╨╡╨╝ ╤Б╨╕╨│╨╜╨░╨╗ ╤В╨╛╨╗╤М╨║╨╛ ╨╜╨░ ╨╛╤Б╨╜╨╛╨▓╨╡ RSI ╤Б ╤Г╤З╨╡╤В╨╛╨╝ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╤Е ╨╜╨░╤Б╤В╤А╨╛╨╡╨║
                if rsi <= rsi_long_threshold:  # RSI тЙд ╨┐╨╛╤А╨╛╨│ LONG (╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╣ ╨╕╨╗╨╕ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Л╨╣)
                    # тЬЕ ╨з╨Ш╨б╨в╨л╨Щ ╨б╨Ш╨У╨Э╨Р╨Ы RSI: ╨Т╤Е╨╛╨┤╨╕╨╝ ╤Б╤А╨░╨╖╤Г, ╨▒╨╡╨╖ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╤В╤А╨╡╨╜╨┤╨░
                    # ╨Ч╨░╤Й╨╕╤В╨░ ╨╛╤В "╨┐╨░╨┤╨░╤О╤Й╨╡╨│╨╛ ╨╜╨╛╨╢╨░" ╤Г╨╢╨╡ ╨╡╤Б╤В╤М:
                    # - ╨Т╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╣ ╤Д╨╕╨╗╤М╤В╤А RSI (╨▒╨╗╨╛╨║╨╕╤А╤Г╨╡╤В ╨╡╤Б╨╗╨╕ oversold ╤Б╨╗╨╕╤И╨║╨╛╨╝ ╨┤╨╛╨╗╨│╨╛)
                    # - Pump-Dump ╤Д╨╕╨╗╤М╤В╤А (╨╛╨┐╤А╨╡╨┤╨╡╨╗╤П╨╡╤В ╨╕╤Б╨║╤Г╤Б╤Б╤В╨▓╨╡╨╜╨╜╤Л╨╡ ╨┤╨▓╨╕╨╢╨╡╨╜╨╕╤П)
                    # - ExitScam ╤Д╨╕╨╗╤М╤В╤А (╨╖╨░╤Й╨╕╤В╨░ ╨╛╤В ╤Б╨║╨░╨╝╨░)
                    # - AI ╤Д╨╕╨╗╤М╤В╤А (╨┤╨╛╨┐╨╛╨╗╨╜╨╕╤В╨╡╨╗╤М╨╜╤Л╨╣ ╨░╨╜╨░╨╗╨╕╨╖)
                    # - ╨б╤В╨╛╨┐-╨╗╨╛╤Б╤Б 15% (╨╛╨│╤А╨░╨╜╨╕╤З╨╕╨▓╨░╨╡╤В ╤Г╨▒╤Л╤В╨║╨╕)
                    rsi_zone = 'BUY_ZONE'
                    signal = 'ENTER_LONG'  # тЬЕ ╨Т╤Е╨╛╨┤╨╕╨╝ ╨▓ ╨╗╨╛╨╜╨│ ╨┐╨╛ ╤Б╨╕╨│╨╜╨░╨╗╤Г RSI
                
                elif rsi >= rsi_short_threshold:  # RSI тЙе ╨┐╨╛╤А╨╛╨│ SHORT (╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╣ ╨╕╨╗╨╕ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Л╨╣)
                    # тЬЕ ╨з╨Ш╨б╨в╨л╨Щ ╨б╨Ш╨У╨Э╨Р╨Ы RSI: ╨Т╤Е╨╛╨┤╨╕╨╝ ╤Б╤А╨░╨╖╤Г, ╨▒╨╡╨╖ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╤В╤А╨╡╨╜╨┤╨░
                    rsi_zone = 'SELL_ZONE'
                    signal = 'ENTER_SHORT'  # тЬЕ ╨Т╤Е╨╛╨┤╨╕╨╝ ╨▓ ╤И╨╛╤А╤В ╨┐╨╛ ╤Б╨╕╨│╨╜╨░╨╗╤Г RSI
                else:
                    # RSI ╨▓ ╨╜╨╡╨╣╤В╤А╨░╨╗╤М╨╜╨╛╨╣ ╨╖╨╛╨╜╨╡
                    pass
            except Exception as e:
                logger.debug(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜╨╕╤П RSI ╤Б╨╕╨│╨╜╨░╨╗╨░: {e}")
                # Fallback ╨║ ╨▒╨░╨╖╨╛╨▓╨╛╨╣ ╨╗╨╛╨│╨╕╨║╨╡ ╨┐╤А╨╕ ╨╛╤И╨╕╨▒╨║╨╡
                if rsi <= rsi_long_threshold:
                    rsi_zone = 'BUY_ZONE'
                    signal = 'ENTER_LONG'
                elif rsi >= rsi_short_threshold:
                    rsi_zone = 'SELL_ZONE'
                    signal = 'ENTER_SHORT'
        else:
            # Fallback ╨║ ╤Б╤В╨░╤А╨╛╨╣ ╨╗╨╛╨│╨╕╨║╨╡ ╨╡╤Б╨╗╨╕ EMA ╨┐╨╡╤А╨╕╨╛╨┤╤Л ╨╜╨╡╨┤╨╛╤Б╤В╤Г╨┐╨╜╤Л
            if rsi <= rsi_long_threshold:
                rsi_zone = 'BUY_ZONE'
                if avoid_down_trend and trend == 'DOWN':
                    signal = 'WAIT'
                else:
                    signal = 'ENTER_LONG'
            elif rsi >= rsi_short_threshold:
                rsi_zone = 'SELL_ZONE'
                if avoid_up_trend and trend == 'UP':
                    signal = 'WAIT'
                else:
                    signal = 'ENTER_SHORT'
        # RSI ╨╝╨╡╨╢╨┤╤Г ╨┐╨╛╤А╨╛╨│╨░╨╝╨╕ - ╨╜╨╡╨╣╤В╤А╨░╨╗╤М╨╜╨░╤П ╨╖╨╛╨╜╨░
        
        # тЬЕ ╨д╨Ш╨Ы╨м╨в╨а 3: ╨б╤Г╤Й╨╡╤Б╤В╨▓╤Г╤О╤Й╨╕╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ (╨Ю╨в╨Ъ╨Ы╨о╨з╨Х╨Э ╨┤╨╗╤П ╤Г╤Б╨║╨╛╤А╨╡╨╜╨╕╤П RSI ╤А╨░╤Б╤З╨╡╤В╨░)
        # тЪб ╨Ю╨Я╨в╨Ш╨Ь╨Ш╨Ч╨Р╨ж╨Ш╨п: ╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╤Б╨╗╨╕╤И╨║╨╛╨╝ ╨╝╨╡╨┤╨╗╨╡╨╜╨╜╨░╤П (API ╨╖╨░╨┐╤А╨╛╤Б ╨║ ╨▒╨╕╤А╨╢╨╡ ╨▓ ╨║╨░╨╢╨┤╨╛╨╝ ╨┐╨╛╤В╨╛╨║╨╡!)
        # ╨н╤В╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╨▒╤Г╨┤╨╡╤В ╨▓╤Л╨┐╨╛╨╗╨╜╨╡╨╜╨░ ╨┐╨╛╨╖╨╢╨╡ ╨▓ process_auto_bot_signals() ╨Я╨Х╨а╨Х╨Ф ╤Б╨╛╨╖╨┤╨░╨╜╨╕╨╡╨╝ ╨▒╨╛╤В╨░
        has_existing_position = False
        # ╨Я╨а╨Ю╨Я╨г╨б╨Ъ╨Р╨Х╨Ь ╨Я╨а╨Ю╨Т╨Х╨а╨Ъ╨г ╨Я╨Ю╨Ч╨Ш╨ж╨Ш╨Щ ╨Ч╨Ф╨Х╨б╨м - ╤Н╨║╨╛╨╜╨╛╨╝╨╕╨╝ ~50 API ╨╖╨░╨┐╤А╨╛╤Б╨╛╨▓ ╨║ ╨▒╨╕╤А╨╢╨╡!
        
        # тЬЕ ╨д╨Ш╨Ы╨м╨в╨а 4: Enhanced RSI тАФ ╤Б╤З╨╕╤В╨░╨╡╨╝ ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╨║╨╛╨│╨┤╨░ ╨╡╤Б╤В╤М ╨┐╨╛╤В╨╡╨╜╤Ж╨╕╨░╨╗╤М╨╜╤Л╨╣ ╤Б╨╕╨│╨╜╨░╨╗
        potential_signal = None
        enhanced_analysis = {
            'enabled': False,
            'warning_type': None,
            'warning_message': None,
            'extreme_duration': 0,
            'adaptive_levels': None,
            'confirmations': {},
            'enhanced_signal': None,
            'enhanced_reason': None,
        }

        if signal in ['ENTER_LONG', 'ENTER_SHORT'] or potential_signal in ['ENTER_LONG', 'ENTER_SHORT']:
            enhanced_analysis = perform_enhanced_rsi_analysis(candles, rsi, symbol) or enhanced_analysis

            # ╨Х╤Б╨╗╨╕ Enhanced RSI ╨▓╨║╨╗╤О╤З╨╡╨╜ ╨╕ ╨┤╨░╨╡╤В ╨┤╤А╤Г╨│╨╛╨╣ ╤Б╨╕╨│╨╜╨░╨╗ - ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╡╨│╨╛
            if enhanced_analysis.get('enabled') and enhanced_analysis.get('enhanced_signal'):
                original_signal = signal
                enhanced_signal = enhanced_analysis.get('enhanced_signal')
                if enhanced_signal != original_signal:
                    logger.info(f"{symbol}: ╨б╨╕╨│╨╜╨░╨╗ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜ {original_signal} тЖТ {enhanced_signal}")
                    signal = enhanced_signal
                    # ╨Х╤Б╨╗╨╕ Enhanced RSI ╨│╨╛╨▓╨╛╤А╨╕╤В WAIT - ╨▒╨╗╨╛╨║╨╕╤А╤Г╨╡╨╝
                    if signal == 'WAIT':
                        rsi_zone = 'NEUTRAL'
        
        # тЬЕ ╨д╨Ш╨Ы╨м╨в╨а 5: ╨Ч╤А╨╡╨╗╨╛╤Б╤В╤М ╨╝╨╛╨╜╨╡╤В╤Л (╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨Я╨Ю╨б╨Ы╨Х Enhanced RSI)
        # ЁЯФз ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ш╨Х: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╖╤А╨╡╨╗╨╛╤Б╤В╤М ╨┤╨╗╤П ╨Т╨б╨Х╨е ╨╝╨╛╨╜╨╡╤В (╨┤╨╗╤П UI ╤Д╨╕╨╗╤М╤В╤А╨░ "╨Ч╤А╨╡╨╗╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л")
        # тЬЕ ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕, ╨╡╤Б╨╗╨╕ ╨╛╨╜╨╕ ╨╡╤Б╤В╤М, ╨╕╨╜╨░╤З╨╡ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Л╨╡
        enable_maturity_check = individual_settings.get('enable_maturity_check') if individual_settings else None
        if enable_maturity_check is None:
            enable_maturity_check = bots_data.get('auto_bot_config', {}).get('enable_maturity_check', True)
        is_mature = True  # ╨Я╨╛ ╤Г╨╝╨╛╨╗╤З╨░╨╜╨╕╤О ╤Б╤З╨╕╤В╨░╨╡╨╝ ╨╖╤А╨╡╨╗╨╛╨╣ (╨╡╤Б╨╗╨╕ ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╨╛╤В╨║╨╗╤О╤З╨╡╨╜╨░)
        
        if enable_maturity_check:
            # тЬЕ ╨Ш╨б╨Я╨Ю╨Ы╨м╨Ч╨г╨Х╨Ь ╤Е╤А╨░╨╜╨╕╨╗╨╕╤Й╨╡ ╨╖╤А╨╡╨╗╤Л╤Е ╨╝╨╛╨╜╨╡╤В ╨┤╨╗╤П ╨▒╤Л╤Б╤В╤А╨╛╨╣ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕
            is_mature = check_coin_maturity_stored_or_verify(symbol)
            
            # ╨Х╤Б╨╗╨╕ ╨╡╤Б╤В╤М ╤Б╨╕╨│╨╜╨░╨╗ ╨▓╤Е╨╛╨┤╨░ ╨Ш ╨╝╨╛╨╜╨╡╤В╨░ ╨╜╨╡╨╖╤А╨╡╨╗╨░╤П - ╨▒╨╗╨╛╨║╨╕╤А╤Г╨╡╨╝ ╤Б╨╕╨│╨╜╨░╨╗
            if signal in ['ENTER_LONG', 'ENTER_SHORT'] and not is_mature:
                # ╨Ю╨│╤А╨░╨╜╨╕╤З╨╕╨▓╨░╨╡╨╝ ╤З╨░╤Б╤В╨╛╤В╤Г ╨╗╨╛╨│╨╕╤А╨╛╨▓╨░╨╜╨╕╤П - ╨╜╨╡ ╨▒╨╛╨╗╨╡╨╡ ╤А╨░╨╖╨░ ╨▓ 2 ╨╝╨╕╨╜╤Г╤В╤Л ╨┤╨╗╤П ╨║╨░╨╢╨┤╨╛╨╣ ╨╝╨╛╨╜╨╡╤В╤Л
                log_message = f"{symbol}: ╨Ь╨╛╨╜╨╡╤В╨░ ╨╜╨╡╨╖╤А╨╡╨╗╨░╤П - ╤Б╨╕╨│╨╜╨░╨╗ {signal} ╨╖╨░╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨░╨╜"
                category = f'maturity_check_{symbol}'
                should_log, message = should_log_message(category, log_message, interval_seconds=120)
                if should_log:
                    logger.debug(message)
                # ╨Ь╨╡╨╜╤П╨╡╨╝ ╤Б╨╕╨│╨╜╨░╨╗ ╨╜╨░ WAIT, ╨╜╨╛ ╨╜╨╡ ╨╕╤Б╨║╨╗╤О╤З╨░╨╡╨╝ ╨╝╨╛╨╜╨╡╤В╤Г ╨╕╨╖ ╤Б╨┐╨╕╤Б╨║╨░
                signal = 'WAIT'
                rsi_zone = 'NEUTRAL'
        
        # тЬЕ EMA ╨┐╨╡╤А╨╕╨╛╨┤╤Л ╤Г╨╢╨╡ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╤Л ╨▓╤Л╤И╨╡ - ╨Ф╨Ю ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜╨╕╤П ╤Б╨╕╨│╨╜╨░╨╗╨░!
        
        # closes[-1] - ╤Н╤В╨╛ ╤Б╨░╨╝╨░╤П ╨Э╨Ю╨Т╨Р╨п ╤Ж╨╡╨╜╨░ (╨┐╨╛╤Б╨╗╨╡╨┤╨╜╤П╤П ╤Б╨▓╨╡╤З╨░ ╨▓ ╨╝╨░╤Б╤Б╨╕╨▓╨╡)
        current_price = closes[-1]
        
        # тЬЕ ╨Я╨а╨Р╨Т╨Ш╨Ы╨м╨Э╨л╨Щ ╨Я╨Ю╨а╨п╨Ф╨Ю╨Ъ ╨д╨Ш╨Ы╨м╨в╨а╨Ю╨Т ╤Б╨╛╨│╨╗╨░╤Б╨╜╨╛ ╨╗╨╛╨│╨╕╨║╨╡:
        # 1. Whitelist/Blacklist/Scope тЖТ ╤Г╨╢╨╡ ╨┐╤А╨╛╨▓╨╡╤А╨╡╨╜╨╛ ╨▓ ╨╜╨░╤З╨░╨╗╨╡
        # 2. ╨С╨░╨╖╨╛╨▓╤Л╨╣ RSI + ╨в╤А╨╡╨╜╨┤ тЖТ ╤Г╨╢╨╡ ╨┐╤А╨╛╨▓╨╡╤А╨╡╨╜╨╛ ╨▓╤Л╤И╨╡
        # 3. ╨б╤Г╤Й╨╡╤Б╤В╨▓╤Г╤О╤Й╨╕╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ тЖТ ╤Г╨╢╨╡ ╨┐╤А╨╛╨▓╨╡╤А╨╡╨╜╨╛ ╨▓╤Л╤И╨╡ (╨а╨Р╨Э╨Э╨Ш╨Щ ╨▓╤Л╤Е╨╛╨┤!)
        # 4. Enhanced RSI тЖТ ╤Г╨╢╨╡ ╨┐╤А╨╛╨▓╨╡╤А╨╡╨╜╨╛ ╨▓╤Л╤И╨╡
        # 5. ╨Ч╤А╨╡╨╗╨╛╤Б╤В╤М ╨╝╨╛╨╜╨╡╤В╤Л тЖТ ╤Г╨╢╨╡ ╨┐╤А╨╛╨▓╨╡╤А╨╡╨╜╨╛ ╨▓╤Л╤И╨╡
        # 6. ExitScam ╤Д╨╕╨╗╤М╤В╤А тЖТ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╖╨┤╨╡╤Б╤М
        # 7. RSI ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╣ ╤Д╨╕╨╗╤М╤В╤А тЖТ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╖╨┤╨╡╤Б╤М
        
        exit_scam_info = None
        time_filter_info = None
        loss_reentry_info = None  # тЬЕ ╨Ш╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╤Г╨╡╨╝ ╨Ф╨Ю ╨╕╤Б╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╨╜╨╕╤П ╨▓ result
        
        # тЬЕ ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨┐╨╛╤А╨╛╨│╨╕ ╨┤╨╗╤П ╤Д╨╕╨╗╤М╤В╤А╨╛╨▓ ╤Б ╤Г╤З╨╡╤В╨╛╨╝ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╤Е ╨╜╨░╤Б╤В╤А╨╛╨╡╨║
        # ╨Я╨╛╤А╨╛╨│╨╕ RSI ╤Г╨╢╨╡ ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜╤Л ╨▓╤Л╤И╨╡ (╤Б ╤Г╤З╨╡╤В╨╛╨╝ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╤Е ╨╜╨░╤Б╤В╤А╨╛╨╡╨║)
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨┐╨╛╤А╨╛╨│╨╕ ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨│╨╛ ╤Д╨╕╨╗╤М╤В╤А╨░: ╤Б╨╜╨░╤З╨░╨╗╨░ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡, ╨╖╨░╤В╨╡╨╝ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Л╨╡
        rsi_time_filter_lower = individual_settings.get('rsi_time_filter_lower') if individual_settings else None
        if rsi_time_filter_lower is None:
            rsi_time_filter_lower = bots_data.get('auto_bot_config', {}).get('rsi_time_filter_lower', 35)  # ╨Э╨╕╨╢╨╜╤П╤П ╨│╤А╨░╨╜╨╕╤Ж╨░ ╨┤╨╗╤П LONG
        
        rsi_time_filter_upper = individual_settings.get('rsi_time_filter_upper') if individual_settings else None
        if rsi_time_filter_upper is None:
            rsi_time_filter_upper = bots_data.get('auto_bot_config', {}).get('rsi_time_filter_upper', 65)  # ╨Т╨╡╤А╤Е╨╜╤П╤П ╨│╤А╨░╨╜╨╕╤Ж╨░ ╨┤╨╗╤П SHORT
        
        # ╨Ю╨┐╤А╨╡╨┤╨╡╨╗╤П╨╡╨╝ ╨┐╨╛╤В╨╡╨╜╤Ж╨╕╨░╨╗╤М╨╜╤Л╨╣ ╤Б╨╕╨│╨╜╨░╨╗ ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╤Д╨╕╨╗╤М╤В╤А╨╛╨▓
        # ╨Т╨Р╨Ц╨Э╨Ю: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Д╨╕╨╗╤М╤В╤А╤Л ╨╡╤Б╨╗╨╕ RSI ╨▓ ╨╖╨╛╨╜╨╡ ╤Д╨╕╨╗╤М╤В╤А╨░:
        # - ╨Ф╨╗╤П LONG: RSI <= 35 (╨╜╨╕╨╢╨╜╤П╤П ╨│╤А╨░╨╜╨╕╤Ж╨░)
        # - ╨Ф╨╗╤П SHORT: RSI >= 65 (╨▓╨╡╤А╤Е╨╜╤П╤П ╨│╤А╨░╨╜╨╕╤Ж╨░)
        
        # ╨Ю╨┐╤А╨╡╨┤╨╡╨╗╤П╨╡╨╝ potential_signal ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╤Д╨╕╨╗╤М╤В╤А╨╛╨▓
        if rsi is not None:
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨▓ ╨║╨░╨║╨╛╨╣ ╨╖╨╛╨╜╨╡ ╨╜╨░╤Е╨╛╨┤╨╕╤В╤Б╤П RSI
            if rsi <= rsi_time_filter_lower:
                # RSI ╨▓ ╨╖╨╛╨╜╨╡ ╤Д╨╕╨╗╤М╤В╤А╨░ ╨┤╨╗╤П LONG - ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╤Б╨▓╨╡╤З╨╡╨╣ ╨╜╨░ ╨╜╨░╨╗╨╕╤З╨╕╨╡ ╨╗╨╛╤П
                potential_signal = 'ENTER_LONG'
            elif rsi >= rsi_time_filter_upper:
                # RSI ╨▓ ╨╖╨╛╨╜╨╡ ╤Д╨╕╨╗╤М╤В╤А╨░ ╨┤╨╗╤П SHORT - ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╤Б╨▓╨╡╤З╨╕ ╨╜╨░ ╨╜╨░╨╗╨╕╤З╨╕╨╡ ╨┐╨╕╨║╨░
                potential_signal = 'ENTER_SHORT'
            else:
                # RSI ╨▓╨╜╨╡ ╨╖╨╛╨╜╤Л ╤Д╨╕╨╗╤М╤В╤А╨░ - ╨┐╨╛╨║╨░╨╖╤Л╨▓╨░╨╡╨╝ ╤З╤В╨╛ ╤Д╨╕╨╗╤М╤В╤А ╨╜╨╡ ╨░╨║╤В╨╕╨▓╨╡╨╜
                potential_signal = None  # ╨Т╨╜╨╡ ╨╖╨╛╨╜╤Л ╨▓╤Е╨╛╨┤╨░
                time_filter_info = {
                    'blocked': False,
                    'reason': 'RSI ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╣ ╤Д╨╕╨╗╤М╤В╤А ╨▓╨╜╨╡ ╨╖╨╛╨╜╤Л ╨▓╤Е╨╛╨┤╨░ ╨▓ ╤Б╨┤╨╡╨╗╨║╤Г',
                    'filter_type': 'time_filter',
                    'last_extreme_candles_ago': None,
                    'calm_candles': None
                }
                # ╨Ф╨╗╤П ╨╝╨╛╨╜╨╡╤В ╨▓╨╜╨╡ ╨╖╨╛╨╜╤Л ╨▓╤Е╨╛╨┤╨░ ExitScam ╤Д╨╕╨╗╤М╤В╤А ╨╜╨╡ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╤В╤Б╤П (╨╛╨┐╤В╨╕╨╝╨╕╨╖╨░╤Ж╨╕╤П)
                exit_scam_info = {
                    'blocked': False,
                    'reason': 'ExitScam ╤Д╨╕╨╗╤М╤В╤А: RSI ╨▓╨╜╨╡ ╨╖╨╛╨╜╤Л ╨▓╤Е╨╛╨┤╨░ ╨▓ ╤Б╨┤╨╡╨╗╨║╤Г',
                    'filter_type': 'exit_scam'
                }
                # ╨Ф╨╗╤П ╨╝╨╛╨╜╨╡╤В ╨▓╨╜╨╡ ╨╖╨╛╨╜╤Л ╨▓╤Е╨╛╨┤╨░ ╨╖╨░╤Й╨╕╤В╨░ ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓ ╨╜╨╡ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╤В╤Б╤П
                loss_reentry_info = {
                    'blocked': False,
                    'reason': '╨Ч╨░╤Й╨╕╤В╨░ ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓: RSI ╨▓╨╜╨╡ ╨╖╨╛╨╜╤Л ╨▓╤Е╨╛╨┤╨░ ╨▓ ╤Б╨┤╨╡╨╗╨║╤Г',
                    'filter_type': 'loss_reentry_protection'
                }
        else:
            # RSI ╨╜╨╡ ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜ - ╨▓╤Б╨╡ ╤Д╨╕╨╗╤М╤В╤А╤Л ╨╜╨╡ ╨░╨║╤В╨╕╨▓╨╜╤Л
            potential_signal = None
            time_filter_info = {
                'blocked': False,
                'reason': 'RSI ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╣ ╤Д╨╕╨╗╤М╤В╤А: RSI ╨╜╨╡ ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜',
                'filter_type': 'time_filter',
                'last_extreme_candles_ago': None,
                'calm_candles': None
            }
            exit_scam_info = {
                'blocked': False,
                'reason': 'ExitScam ╤Д╨╕╨╗╤М╤В╤А: RSI ╨╜╨╡ ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜',
                'filter_type': 'exit_scam'
            }
            loss_reentry_info = {
                'blocked': False,
                'reason': '╨Ч╨░╤Й╨╕╤В╨░ ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓: RSI ╨╜╨╡ ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜',
                'filter_type': 'loss_reentry_protection'
            }
        
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Д╨╕╨╗╤М╤В╤А╤Л ╨╡╤Б╨╗╨╕ ╨╝╨╛╨╜╨╡╤В╨░ ╨▓ ╨╖╨╛╨╜╨╡ ╤Д╨╕╨╗╤М╤В╤А╨░ (LONG/SHORT)
        # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ш╨Х: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Д╨╕╨╗╤М╤В╤А╤Л ╨┤╨╗╤П UI, ╤З╤В╨╛╨▒╤Л ╨┐╨╛╨║╨░╨╖╤Л╨▓╨░╤В╤М ╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨╕
        # тЪб ╨Ю╨Я╨в╨Ш╨Ь╨Ш╨Ч╨Р╨ж╨Ш╨п: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Д╨╕╨╗╤М╤В╤А╤Л ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╨┤╨╗╤П ╨╝╨╛╨╜╨╡╤В ╨▓ ╨╖╨╛╨╜╨╡ ╨▓╤Е╨╛╨┤╨░ (RSI <= 35 ╨┤╨╗╤П LONG ╨╕╨╗╨╕ RSI >= 65 ╨┤╨╗╤П SHORT)
        # ╨н╤В╨╛ ╨╜╨╡ ╨▓╨╗╨╕╤П╨╡╤В ╨╜╨░ ╨┐╤А╨╛╨╕╨╖╨▓╨╛╨┤╨╕╤В╨╡╨╗╤М╨╜╨╛╤Б╤В╤М ╨▒╤Н╨║╨╡╨╜╨┤╨░, ╤В╨░╨║ ╨║╨░╨║ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨▓╤Л╨┐╨╛╨╗╨╜╤П╤О╤В╤Б╤П ╤В╨╛╨╗╤М╨║╨╛ ╨┤╨╗╤П ╨╝╨╛╨╜╨╡╤В, ╨║╨╛╤В╨╛╤А╤Л╨╡ ╤Г╨╢╨╡ ╨┤╨╛╨╗╨╢╨╜╤Л ╨┐╨╛╨╣╤В╨╕ ╨▓ ╨╗╨╛╨╜╨│/╤И╨╛╤А╤В
        # ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╤Г╨╢╨╡ ╨╖╨░╨│╤А╤Г╨╢╨╡╨╜╨╜╤Л╨╡ ╤Б╨▓╨╡╤З╨╕ ╨╕╨╖ ╨┐╨╡╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╣ candles, ╨╜╨╡ ╨┤╨╡╨╗╨░╨╡╨╝ ╨╜╨╛╨▓╤Л╤Е ╨╖╨░╨┐╤А╨╛╤Б╨╛╨▓ ╨║ ╨▒╨╕╤А╨╢╨╡!
        if potential_signal in ['ENTER_LONG', 'ENTER_SHORT']:
            # тЬЕ ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ RSI Time Filter ╨┤╨╗╤П UI
            if len(candles) >= 50:
                try:
                    time_filter_result = check_rsi_time_filter(
                        candles, 
                        rsi, 
                        potential_signal, 
                        symbol=symbol, 
                        individual_settings=individual_settings
                    )
                    if time_filter_result:
                        time_filter_info = {
                            'blocked': not time_filter_result.get('allowed', True),
                            'reason': time_filter_result.get('reason', ''),
                            'filter_type': 'time_filter',
                            'last_extreme_candles_ago': time_filter_result.get('last_extreme_candles_ago'),
                            'calm_candles': time_filter_result.get('calm_candles')
                        }
                    else:
                        time_filter_info = {
                            'blocked': False,
                            'reason': 'RSI ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╣ ╤Д╨╕╨╗╤М╤В╤А: ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╨╜╨╡ ╨▓╤Л╨┐╨╛╨╗╨╜╨╡╨╜╨░',
                            'filter_type': 'time_filter',
                            'last_extreme_candles_ago': None,
                            'calm_candles': None
                        }
                except Exception as e:
                    logger.debug(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ RSI Time Filter ╨┤╨╗╤П UI: {e}")
                    time_filter_info = {
                        'blocked': False,
                        'reason': f'╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕: {str(e)}',
                        'filter_type': 'time_filter',
                        'last_extreme_candles_ago': None,
                        'calm_candles': None
                    }
            else:
                time_filter_info = {
                    'blocked': False,
                    'reason': '╨Э╨╡╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ ╤Б╨▓╨╡╤З╨╡╨╣ ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ (╤В╤А╨╡╨▒╤Г╨╡╤В╤Б╤П ╨╝╨╕╨╜╨╕╨╝╤Г╨╝ 50)',
                    'filter_type': 'time_filter',
                    'last_extreme_candles_ago': None,
                    'calm_candles': None
                }
            
            # тЬЕ ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ExitScam Filter ╨┤╨╗╤П UI (╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╤Г╨╢╨╡ ╨╖╨░╨│╤А╤Г╨╢╨╡╨╜╨╜╤Л╨╡ ╤Б╨▓╨╡╤З╨╕ ╨╕╨╖ candles)
            # тЪб ╨Ю╨Я╨в╨Ш╨Ь╨Ш╨Ч╨Р╨ж╨Ш╨п: ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╤Б╨▓╨╡╤З╨╕, ╨║╨╛╤В╨╛╤А╤Л╨╡ ╤Г╨╢╨╡ ╨╖╨░╨│╤А╤Г╨╢╨╡╨╜╤Л ╨▓╤Л╤И╨╡, ╨╜╨╡ ╨┤╨╡╨╗╨░╨╡╨╝ ╨╜╨╛╨▓╤Л╨╣ ╨╖╨░╨┐╤А╨╛╤Б ╨║ ╨▒╨╕╤А╨╢╨╡!
            try:
                if len(candles) >= 10:  # ╨Ь╨╕╨╜╨╕╨╝╤Г╨╝ ╤Б╨▓╨╡╤З╨╡╨╣ ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ExitScam
                    # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨║╨╛╨╜╤Д╨╕╨│ ╤Б ╤Г╤З╨╡╤В╨╛╨╝ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╤Е ╨╜╨░╤Б╤В╤А╨╛╨╡╨║
                    auto_config = bots_data.get('auto_bot_config', {}).copy()
                    if individual_settings:
                        for key in ['exit_scam_enabled', 'exit_scam_candles', 
                                   'exit_scam_single_candle_percent', 'exit_scam_multi_candle_count',
                                   'exit_scam_multi_candle_percent']:
                            if key in individual_settings:
                                auto_config[key] = individual_settings[key]
                    
                    exit_scam_enabled = auto_config.get('exit_scam_enabled', True)
                    exit_scam_candles = auto_config.get('exit_scam_candles', 10)
                    single_candle_percent = auto_config.get('exit_scam_single_candle_percent', 15.0)
                    multi_candle_count = auto_config.get('exit_scam_multi_candle_count', 4)
                    multi_candle_percent = auto_config.get('exit_scam_multi_candle_percent', 50.0)
                    
                    exit_scam_allowed = True
                    exit_scam_reason = 'ExitScam ╤Д╨╕╨╗╤М╤В╤А ╨┐╤А╨╛╨╣╨┤╨╡╨╜'
                    
                    if exit_scam_enabled and len(candles) >= exit_scam_candles:
                        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╤Б╨▓╨╡╤З╨╡╨╣ (╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╤Г╨╢╨╡ ╨╖╨░╨│╤А╤Г╨╢╨╡╨╜╨╜╤Л╨╡ ╤Б╨▓╨╡╤З╨╕!)
                        recent_candles = candles[-exit_scam_candles:]
                        
                        # 1. ╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╨╛╤В╨┤╨╡╨╗╤М╨╜╤Л╤Е ╤Б╨▓╨╡╤З╨╡╨╣
                        for candle in recent_candles:
                            open_price = candle['open']
                            close_price = candle['close']
                            price_change = abs((close_price - open_price) / open_price) * 100
                            
                            if price_change > single_candle_percent:
                                exit_scam_allowed = False
                                exit_scam_reason = f'ExitScam ╤Д╨╕╨╗╤М╤В╤А: ╨╛╨┤╨╜╨░ ╤Б╨▓╨╡╤З╨░ ╨┐╤А╨╡╨▓╤Л╤Б╨╕╨╗╨░ ╨╗╨╕╨╝╨╕╤В {single_candle_percent}% (╨▒╤Л╨╗╨╛ {price_change:.1f}%)'
                                break
                        
                        # 2. ╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╤Б╤Г╨╝╨╝╨░╤А╨╜╨╛╨│╨╛ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П (╨╡╤Б╨╗╨╕ ╨┐╨╡╤А╨▓╨░╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╨┐╤А╨╛╤И╨╗╨░)
                        if exit_scam_allowed and len(recent_candles) >= multi_candle_count:
                            multi_candles = recent_candles[-multi_candle_count:]
                            first_open = multi_candles[0]['open']
                            last_close = multi_candles[-1]['close']
                            total_change = abs((last_close - first_open) / first_open) * 100
                            
                            if total_change > multi_candle_percent:
                                exit_scam_allowed = False
                                exit_scam_reason = f'ExitScam ╤Д╨╕╨╗╤М╤В╤А: {multi_candle_count} ╤Б╨▓╨╡╤З╨╡╨╣ ╨┐╤А╨╡╨▓╤Л╤Б╨╕╨╗╨╕ ╤Б╤Г╨╝╨╝╨░╤А╨╜╤Л╨╣ ╨╗╨╕╨╝╨╕╤В {multi_candle_percent}% (╨▒╤Л╨╗╨╛ {total_change:.1f}%)'
                        
                        # 3. AI ╨┤╨╡╤В╨╡╨║╤Ж╨╕╤П ╨░╨╜╨╛╨╝╨░╨╗╨╕╨╣ (╨╡╤Б╨╗╨╕ ╨▓╨║╨╗╤О╤З╨╡╨╜╨░ ╨╕ ╨▒╨░╨╖╨╛╨▓╤Л╨╡ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨┐╤А╨╛╤И╨╗╨╕)
                        if exit_scam_allowed:
                            try:
                                from bot_engine.bot_config import AIConfig
                                if AIConfig.AI_ENABLED and AIConfig.AI_ANOMALY_DETECTION_ENABLED:
                                    exit_scam_allowed = _run_exit_scam_ai_detection(symbol, candles)
                                    if not exit_scam_allowed:
                                        exit_scam_reason = 'ExitScam ╤Д╨╕╨╗╤М╤В╤А: AI ╨╛╨▒╨╜╨░╤А╤Г╨╢╨╕╨╗ ╨░╨╜╨╛╨╝╨░╨╗╨╕╤О'
                            except ImportError:
                                pass  # AI ╨╝╨╛╨┤╤Г╨╗╤М ╨╜╨╡ ╨┤╨╛╤Б╤В╤Г╨┐╨╡╨╜
                    
                    exit_scam_info = {
                        'blocked': not exit_scam_allowed,
                        'reason': exit_scam_reason,
                        'filter_type': 'exit_scam'
                    }
                else:
                    # ╨Э╨╡╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ ╤Б╨▓╨╡╤З╨╡╨╣ ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕
                    exit_scam_info = {
                        'blocked': False,
                        'reason': '╨Э╨╡╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ ╤Б╨▓╨╡╤З╨╡╨╣ ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ (╤В╤А╨╡╨▒╤Г╨╡╤В╤Б╤П ╨╝╨╕╨╜╨╕╨╝╤Г╨╝ 10)',
                        'filter_type': 'exit_scam'
                    }
            except Exception as e:
                logger.debug(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ExitScam Filter ╨┤╨╗╤П UI: {e}")
                exit_scam_info = {
                    'blocked': False,
                    'reason': f'╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕: {str(e)}',
                    'filter_type': 'exit_scam'
                }
            
            # тЬЕ ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╖╨░╤Й╨╕╤В╤Г ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓ ╨┐╨╛╤Б╨╗╨╡ ╤Г╨▒╤Л╤В╨╛╤З╨╜╤Л╤Е ╨╖╨░╨║╤А╤Л╤В╨╕╨╣ ╨┤╨╗╤П UI
            try:
                # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╜╨░╨╗╨╕╤З╨╕╨╡ ╨╛╤В╨║╤А╤Л╤В╨╛╨╣ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ - ╨╡╤Б╨╗╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╤Г╨╢╨╡ ╨╛╤В╨║╤А╤Л╤В╨░, ╤Д╨╕╨╗╤М╤В╤А ╨Э╨Х ╨┐╤А╨╕╨╝╨╡╨╜╤П╨╡╤В╤Б╤П
                has_existing_position_check = False
                from bots_modules.imports_and_globals import bots_data_lock
                with bots_data_lock:
                    bots = bots_data.get('bots', {})
                    bot = bots.get(symbol)
                    if bot:
                        bot_status = bot.get('status', '')
                        position_side = bot.get('position_side')
                        has_existing_position_check = (bot_status == BOT_STATUS['IN_POSITION_LONG'] or 
                                                      bot_status == BOT_STATUS['IN_POSITION_SHORT'] or 
                                                      position_side is not None)
                
                if len(candles) >= 10:  # ╨Ь╨╕╨╜╨╕╨╝╤Г╨╝ ╤Б╨▓╨╡╤З╨╡╨╣ ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕
                    # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨║╨╛╨╜╤Д╨╕╨│ ╤Б ╤Г╤З╨╡╤В╨╛╨╝ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╤Е ╨╜╨░╤Б╤В╤А╨╛╨╡╨║
                    auto_config = bots_data.get('auto_bot_config', {}).copy()
                    if individual_settings:
                        for key in ['loss_reentry_protection', 'loss_reentry_count', 'loss_reentry_candles']:
                            if key in individual_settings:
                                auto_config[key] = individual_settings[key]
                    
                    loss_reentry_protection_enabled = auto_config.get('loss_reentry_protection', True)
                    loss_reentry_count = auto_config.get('loss_reentry_count', 1)
                    loss_reentry_candles = auto_config.get('loss_reentry_candles', 3)
                    
                    # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ю: ╨Т╤Б╨╡╨│╨┤╨░ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Д╨╕╨╗╤М╤В╤А, ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨┤╨╡╨╗╨░╨╡╤В╤Б╤П ╨╜╨░ ╤Г╤А╨╛╨▓╨╜╨╡ ╨▒╨╛╤В╨░
                    if loss_reentry_protection_enabled:
                        # ╨Т╤Л╨╖╤Л╨▓╨░╨╡╨╝ ╨┐╤А╨╛╨▓╨╡╤А╨║╤Г ╨╖╨░╤Й╨╕╤В╤Л (╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╤Г╨▒╤А╨░╨╜╨░ - ╨╛╨╜╨░ ╨▓ should_open_long/short)
                        loss_reentry_result = _check_loss_reentry_protection_static(
                            symbol, candles, loss_reentry_count, loss_reentry_candles, individual_settings
                        )
                        if loss_reentry_result:
                            loss_reentry_info = {
                                'blocked': not loss_reentry_result.get('allowed', True),
                                'reason': loss_reentry_result.get('reason', ''),
                                'filter_type': 'loss_reentry_protection',
                                'candles_passed': loss_reentry_result.get('candles_passed'),
                                'required_candles': loss_reentry_candles,
                                'loss_count': loss_reentry_count
                            }
                        else:
                            loss_reentry_info = {
                                'blocked': False,
                                'reason': '╨Ч╨░╤Й╨╕╤В╨░ ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓: ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╨╜╨╡ ╨▓╤Л╨┐╨╛╨╗╨╜╨╡╨╜╨░',
                                'filter_type': 'loss_reentry_protection'
                            }
                    else:
                        loss_reentry_info = {
                            'blocked': False,
                            'reason': '╨Ч╨░╤Й╨╕╤В╨░ ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓: ╨▓╤Л╨║╨╗╤О╤З╨╡╨╜╨░',
                            'filter_type': 'loss_reentry_protection'
                        }
                else:
                    loss_reentry_info = {
                        'blocked': False,
                        'reason': '╨Э╨╡╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ ╤Б╨▓╨╡╤З╨╡╨╣ ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕',
                        'filter_type': 'loss_reentry_protection'
                    }
            except Exception as e:
                logger.debug(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨╖╨░╤Й╨╕╤В╤Л ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓ ╨┤╨╗╤П UI: {e}")
                loss_reentry_info = {
                    'blocked': False,
                    'reason': f'╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕: {str(e)}',
                    'filter_type': 'loss_reentry_protection'
                }
        
        # тЬЕ ╨Я╨а╨Ш╨Ь╨Х╨Э╨п╨Х╨Ь ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨г ╨Я╨Ю SCOPE
        # Scope ╤Д╨╕╨╗╤М╤В╤А (╨╡╤Б╨╗╨╕ ╨╝╨╛╨╜╨╡╤В╨░ ╨▓ ╤З╨╡╤А╨╜╨╛╨╝ ╤Б╨┐╨╕╤Б╨║╨╡ ╨╕╨╗╨╕ ╨╜╨╡ ╨▓ ╨▒╨╡╨╗╨╛╨╝)
        if is_blocked_by_scope:
            signal = 'WAIT'
            rsi_zone = 'NEUTRAL'
        
        # тЬЕ ╨Я╨а╨Ю╨Т╨Х╨а╨п╨Х╨Ь ╨б╨в╨Р╨в╨г╨б ╨в╨Ю╨а╨У╨Ю╨Т╨Ы╨Ш: ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨╕╨╜╤Д╨╛╤А╨╝╨░╤Ж╨╕╤О ╨╛ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╡/╨╜╨╛╨▓╤Л╤Е ╨╝╨╛╨╜╨╡╤В╨░╤Е
        # ╨Ю╨Я╨в╨Ш╨Ь╨Ш╨Ч╨Ш╨а╨Ю╨Т╨Р╨Э╨Э╨Р╨п ╨Т╨Х╨а╨б╨Ш╨п: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╨╕╨╖╨▓╨╡╤Б╤В╨╜╤Л╨╡ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л
        trading_status = 'Trading'  # ╨Я╨╛ ╤Г╨╝╨╛╨╗╤З╨░╨╜╨╕╤О
        is_delisting = False
        
        # тЬЕ ╨з╨Х╨а╨Э╨л╨Щ ╨б╨Я╨Ш╨б╨Ю╨Ъ ╨Ф╨Х╨Ы╨Ш╨б╨в╨Ш╨Э╨У╨Ю╨Т╨л╨е ╨Ь╨Ю╨Э╨Х╨в - ╨╕╤Б╨║╨╗╤О╤З╨░╨╡╨╝ ╨╕╨╖ ╨▓╤Б╨╡╤Е ╨┐╤А╨╛╨▓╨╡╤А╨╛╨║
        # ╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л ╨╕╨╖ ╤Д╨░╨╣╨╗╨░
        delisted_data = load_delisted_coins()
        delisted_coins = delisted_data.get('delisted_coins', {})
        
        known_delisting_coins = list(delisted_coins.keys())
        known_new_coins = []  # ╨Ь╨╛╨╢╨╜╨╛ ╨┤╨╛╨▒╨░╨▓╨╕╤В╤М ╨╜╨╛╨▓╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л
        
        if symbol in known_delisting_coins:
            trading_status = 'Closed'
            is_delisting = True
            logger.info(f"{symbol}: ╨Ш╨╖╨▓╨╡╤Б╤В╨╜╨░╤П ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╨░╤П ╨╝╨╛╨╜╨╡╤В╨░")
        elif symbol in known_new_coins:
            trading_status = 'Delivering'
            is_delisting = True
            logger.info(f"{symbol}: ╨Ш╨╖╨▓╨╡╤Б╤В╨╜╨░╤П ╨╜╨╛╨▓╨░╤П ╨╝╨╛╨╜╨╡╤В╨░")
        
        # TODO: ╨Т╨║╨╗╤О╤З╨╕╤В╤М ╨┐╨╛╨╗╨╜╤Г╤О ╨┐╤А╨╛╨▓╨╡╤А╨║╤Г ╤Б╤В╨░╤В╤Г╤Б╨░ ╤В╨╛╤А╨│╨╛╨▓╨╗╨╕ ╨┐╨╛╤Б╨╗╨╡ ╨╛╨┐╤В╨╕╨╝╨╕╨╖╨░╤Ж╨╕╨╕ API ╨╖╨░╨┐╤А╨╛╤Б╨╛╨▓
        # try:
        #     if exchange_obj and hasattr(exchange_obj, 'get_instrument_status'):
        #         status_info = exchange_obj.get_instrument_status(f"{symbol}USDT")
        #         if status_info:
        #             trading_status = status_info.get('status', 'Trading')
        #             is_delisting = status_info.get('is_delisting', False)
        #             
        #             # ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨╛╨▓╤Л╨╡ ╨╕ ╨╜╨╛╨▓╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л
        #             if trading_status != 'Trading':
        #                 logger.info(f"[TRADING_STATUS] {symbol}: ╨б╤В╨░╤В╤Г╤Б {trading_status} (╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│: {is_delisting})")
        # except Exception as e:
        #     # ╨Х╤Б╨╗╨╕ ╨╜╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╤Б╤В╨░╤В╤Г╤Б, ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П ╨┐╨╛ ╤Г╨╝╨╛╨╗╤З╨░╨╜╨╕╤О
        #     logger.debug(f"[TRADING_STATUS] {symbol}: ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╤Б╤В╨░╤В╤Г╤Б ╤В╨╛╤А╨│╨╛╨▓╨╗╨╕: {e}")
        
        result = {
            'symbol': symbol,
            'rsi6h': round(rsi, 1),
            'trend6h': trend,
            'rsi_zone': rsi_zone,
            'signal': signal,
            'price': current_price,
            'change24h': change_24h,
            'last_update': datetime.now().isoformat(),
            'trend_analysis': trend_analysis,
            # тЪб ╨Ю╨Я╨в╨Ш╨Ь╨Ш╨Ч╨Р╨ж╨Ш╨п: Enhanced RSI, ╤Д╨╕╨╗╤М╤В╤А╤Л ╨╕ ╤Д╨╗╨░╨│╨╕ ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╨╡╤Б╨╗╨╕ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╗╨╕╤Б╤М
            'enhanced_rsi': enhanced_analysis if enhanced_analysis else {'enabled': False},
            'time_filter_info': time_filter_info,
            'exit_scam_info': exit_scam_info,  # None - ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╤В╨╛╨╗╤М╨║╨╛ ╨┐╤А╨╕ ╨▓╤Е╨╛╨┤╨╡ ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╤О
            'blocked_by_scope': is_blocked_by_scope,
            'has_existing_position': has_existing_position,
            'is_mature': is_mature if enable_maturity_check else True,
            # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨д╨╗╨░╨│╨╕ ╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨╕ ╨┤╨╗╤П get_effective_signal ╨╕ UI
            # ╨г╤Б╤В╨░╨╜╨░╨▓╨╗╨╕╨▓╨░╨╡╨╝ ╤Д╨╗╨░╨│╨╕ ╨╜╨░ ╨╛╤Б╨╜╨╛╨▓╨╡ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╨╛╨▓ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╤Д╨╕╨╗╤М╤В╤А╨╛╨▓ ╨┤╨╗╤П UI
            'blocked_by_exit_scam': exit_scam_info.get('blocked', False) if exit_scam_info else False,
            'blocked_by_rsi_time': time_filter_info.get('blocked', False) if time_filter_info else False,
            'loss_reentry_info': loss_reentry_info,  # ╨Ш╨╜╤Д╨╛╤А╨╝╨░╤Ж╨╕╤П ╨╛ ╨╖╨░╤Й╨╕╤В╨╡ ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓
            'blocked_by_loss_reentry': loss_reentry_info.get('blocked', False) if loss_reentry_info else False,
            # тЬЕ ╨Ш╨Э╨д╨Ю╨а╨Ь╨Р╨ж╨Ш╨п ╨Ю ╨б╨в╨Р╨в╨г╨б╨Х ╨в╨Ю╨а╨У╨Ю╨Т╨Ы╨Ш: ╨Ф╨╗╤П ╨▓╨╕╨╖╤Г╨░╨╗╤М╨╜╤Л╤Е ╤Н╤Д╤Д╨╡╨║╤В╨╛╨▓ ╨┤╨╡╨╗╨╕╤Б╤В╨╕╨╜╨│╨░
            'trading_status': trading_status,
            'is_delisting': is_delisting
        }
        
        # ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ ╤В╨╛╤А╨│╨╛╨▓╤Л╨╡ ╤Б╨╕╨│╨╜╨░╨╗╤Л ╨╕ ╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨╕ ╤В╤А╨╡╨╜╨┤╨░
        # ╨Э╨Х ╨┐╨╛╨║╨░╨╖╤Л╨▓╨░╨╡╨╝ ╨┤╨╡╤Д╨╛╨╗╤В╨╜╤Л╨╡ ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П! ╨в╨╛╨╗╤М╨║╨╛ ╤А╨░╤Б╤Б╤З╨╕╤В╨░╨╜╨╜╤Л╨╡ ╨┤╨░╨╜╨╜╤Л╨╡!
        trend_display = trend if trend is not None else None
        # ╨Э╨Х ╨┐╨╛╨║╨░╨╖╤Л╨▓╨░╨╡╨╝ ╨┤╨╡╤Д╨╛╨╗╤В╨╜╤Л╨╡ emoji! ╨в╨╛╨╗╤М╨║╨╛ ╨┤╨╗╤П ╤А╨░╤Б╤Б╤З╨╕╤В╨░╨╜╨╜╤Л╤Е ╨┤╨░╨╜╨╜╤Л╤Е!
        if trend == 'UP':
            trend_emoji = 'ЁЯУИ'
        elif trend == 'DOWN':
            trend_emoji = 'ЁЯУЙ'
        elif trend == 'NEUTRAL':
            trend_emoji = 'тЮбя╕П'
        else:
            trend_emoji = None
        
        if signal in ['ENTER_LONG', 'ENTER_SHORT']:
            logger.info(f"ЁЯОп {symbol}: RSI={rsi:.1f} {trend_emoji}{trend_display} (${current_price:.4f}) тЖТ {signal}")
        elif signal == 'WAIT' and rsi <= SystemConfig.RSI_OVERSOLD and trend == 'DOWN' and avoid_down_trend:
            logger.debug(f"ЁЯЪл {symbol}: RSI={rsi:.1f} {trend_emoji}{trend_display} LONG ╨╖╨░╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨░╨╜ (╤Д╨╕╨╗╤М╤В╤А DOWN ╤В╤А╨╡╨╜╨┤╨░)")
        elif signal == 'WAIT' and rsi >= SystemConfig.RSI_OVERBOUGHT and trend == 'UP' and avoid_up_trend:
            logger.debug(f"ЁЯЪл {symbol}: RSI={rsi:.1f} {trend_emoji}{trend_display} SHORT ╨╖╨░╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨░╨╜ (╤Д╨╕╨╗╤М╤В╤А UP ╤В╤А╨╡╨╜╨┤╨░)")
        
        debug_payload = {
            'source': data_source,
            'duration': round(time.time() - thread_start, 3),
            'thread': threading.current_thread().name
        }
        result['debug_info'] = debug_payload
        return result
        
    except Exception as e:
        logger.error(f"╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╨┤╨░╨╜╨╜╤Л╤Е ╨┤╨╗╤П {symbol}: {e}")
        return None

def load_all_coins_candles_fast():
    """тЪб ╨С╨л╨б╨в╨а╨Р╨п ╨╖╨░╨│╤А╤Г╨╖╨║╨░ ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╤Б╨▓╨╡╤З╨╡╨╣ ╨┤╨╗╤П ╨▓╤Б╨╡╤Е ╨╝╨╛╨╜╨╡╤В ╨С╨Х╨Ч ╤А╨░╤Б╤З╨╡╤В╨╛╨▓"""
    try:
        from bots_modules.imports_and_globals import get_exchange
        current_exchange = get_exchange()
        
        if not current_exchange:
            logger.error("тЭМ ╨С╨╕╤А╨╢╨░ ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░")
            return False
        
        if shutdown_flag.is_set():
            logger.warning("тП╣я╕П ╨Ч╨░╨│╤А╤Г╨╖╨║╨░ ╤Б╨▓╨╡╤З╨╡╨╣ ╨╛╤В╨╝╨╡╨╜╨╡╨╜╨░: ╤Б╨╕╤Б╤В╨╡╨╝╨░ ╨╖╨░╨▓╨╡╤А╤И╨░╨╡╤В ╤А╨░╨▒╨╛╤В╤Г")
            return False

        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╤Б╨┐╨╕╤Б╨╛╨║ ╨▓╤Б╨╡╤Е ╨┐╨░╤А
        pairs = current_exchange.get_all_pairs()
        if not pairs:
            logger.error("тЭМ ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╤Б╨┐╨╕╤Б╨╛╨║ ╨┐╨░╤А")
            return False
        
        # ╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╤Б╨▓╨╡╤З╨╕ ╨┐╨░╨║╨╡╤В╨░╨╝╨╕ (╨г╨б╨Ъ╨Ю╨а╨Х╨Э╨Э╨Р╨п ╨Т╨Х╨а╨б╨Ш╨п)
        batch_size = 100  # ╨г╨▓╨╡╨╗╨╕╤З╨╕╨╗╨╕ ╤Б 50 ╨┤╨╛ 100
        candles_cache = {}
        
        import concurrent.futures
        # тЪб ╨Р╨Ф╨Р╨Я╨в╨Ш╨Т╨Э╨Ю╨Х ╨г╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ш╨Х ╨Т╨Ю╨а╨Ъ╨Х╨а╨Р╨Ь╨Ш: ╨╜╨░╤З╨╕╨╜╨░╨╡╨╝ ╤Б 20, ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛ ╤Г╨╝╨╡╨╜╤М╤И╨░╨╡╨╝ ╨┐╤А╨╕ rate limit
        current_max_workers = 20  # ╨С╨░╨╖╨╛╨▓╨╛╨╡ ╨║╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╛ ╨▓╨╛╤А╨║╨╡╤А╨╛╨▓
        rate_limit_detected = False  # ╨д╨╗╨░╨│ ╨╛╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜╨╕╤П rate limit ╨▓ ╨┐╤А╨╡╨┤╤Л╨┤╤Г╤Й╨╡╨╝ ╨▒╨░╤В╤З╨╡
        
        shutdown_requested = False

        for i in range(0, len(pairs), batch_size):
            if shutdown_flag.is_set():
                shutdown_requested = True
                break

            batch = pairs[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(pairs) + batch_size - 1)//batch_size
            
            # тЪб ╨Т╨а╨Х╨Ь╨Х╨Э╨Э╨Ю╨Х ╨г╨Ь╨Х╨Э╨м╨и╨Х╨Э╨Ш╨Х ╨Т╨Ю╨а╨Ъ╨Х╨а╨Ю╨Т: ╨╡╤Б╨╗╨╕ ╨▓ ╨┐╤А╨╡╨┤╤Л╨┤╤Г╤Й╨╡╨╝ ╨▒╨░╤В╤З╨╡ ╨▒╤Л╨╗ rate limit
            if rate_limit_detected:
                current_max_workers = max(17, current_max_workers - 3)  # ╨г╨╝╨╡╨╜╤М╤И╨░╨╡╨╝ ╨╜╨░ 3, ╨╜╨╛ ╨╜╨╡ ╨╝╨╡╨╜╤М╤И╨╡ 17
                logger.warning(f"тЪая╕П Rate limit ╨╛╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜ ╨▓ ╨┐╤А╨╡╨┤╤Л╨┤╤Г╤Й╨╡╨╝ ╨▒╨░╤В╤З╨╡. ╨Т╤А╨╡╨╝╨╡╨╜╨╜╨╛ ╤Г╨╝╨╡╨╜╤М╤И╨░╨╡╨╝ ╨▓╨╛╤А╨║╨╡╤А╤Л ╨┤╨╛ {current_max_workers}")
                rate_limit_detected = False  # ╨б╨▒╤А╨░╤Б╤Л╨▓╨░╨╡╨╝ ╤Д╨╗╨░╨│ ╨┤╨╗╤П ╤Б╨╗╨╡╨┤╤Г╤О╤Й╨╡╨│╨╛ ╨▒╨░╤В╤З╨░
            elif current_max_workers < 20:
                # ╨Т╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╨╝ ╨║ ╨▒╨░╨╖╨╛╨▓╨╛╨╝╤Г ╨╖╨╜╨░╤З╨╡╨╜╨╕╤О ╨┐╨╛╤Б╨╗╨╡ ╤Г╤Б╨┐╨╡╤И╨╜╨╛╨│╨╛ ╨▒╨░╤В╤З╨░
                logger.info(f"тЬЕ ╨Т╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╨╝ ╨▓╨╛╤А╨║╨╡╤А╤Л ╨║ ╨▒╨░╨╖╨╛╨▓╨╛╨╝╤Г ╨╖╨╜╨░╤З╨╡╨╜╨╕╤О: {current_max_workers} тЖТ 20")
                current_max_workers = 20
            
            # тЪб ╨Ю╨в╨б╨Ы╨Х╨Ц╨Ш╨Т╨Р╨Э╨Ш╨Х RATE LIMIT: ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╖╨░╨┤╨╡╤А╨╢╨║╤Г ╨┤╨╛ ╨╕ ╨┐╨╛╤Б╨╗╨╡ ╨▒╨░╤В╤З╨░
            delay_before_batch = current_exchange.current_request_delay if hasattr(current_exchange, 'current_request_delay') else None
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=current_max_workers) as executor:
                future_to_symbol = {
                    executor.submit(get_coin_candles_only, symbol, current_exchange): symbol
                    for symbol in batch
                }

                if shutdown_flag.is_set():
                    shutdown_requested = True
                    for future in future_to_symbol:
                        future.cancel()
                    break
                
                completed = 0
                done, not_done = concurrent.futures.wait(
                    future_to_symbol.keys(),
                    timeout=90,
                    return_when=concurrent.futures.ALL_COMPLETED
                )

                if shutdown_flag.is_set():
                    shutdown_requested = True
                    for future in future_to_symbol:
                        future.cancel()
                    break
                
                for future in done:
                    symbol = future_to_symbol.get(future)
                    try:
                        result = future.result()
                        if result:
                            candles_cache[result['symbol']] = result
                            completed += 1
                    except Exception:
                        pass
                
                if not_done:
                    unfinished_symbols = [future_to_symbol.get(future) for future in not_done if future in future_to_symbol]
                    logger.error(f"тЭМ Timeout: {len(unfinished_symbols)} (of {len(future_to_symbol)}) futures unfinished")
                    
                    # ╨Ю╤В╨╝╨╡╨╜╤П╨╡╨╝ ╨╜╨╡╨╖╨░╨▓╨╡╤А╤И╨╡╨╜╨╜╤Л╨╡ ╨╖╨░╨┤╨░╤З╨╕ ╨╕ ╤Д╨╕╨║╤Б╨╕╤А╤Г╨╡╨╝ ╨▓╨╛╨╖╨╝╨╛╨╢╨╜╤Л╨╣ rate limit
                    for future in not_done:
                        try:
                            future.cancel()
                        except Exception:
                            pass
                    rate_limit_detected = True
                
                # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╤Г╨▓╨╡╨╗╨╕╤З╨╕╨╗╨░╤Б╤М ╨╗╨╕ ╨╖╨░╨┤╨╡╤А╨╢╨║╨░ ╨┐╨╛╤Б╨╗╨╡ ╨▒╨░╤В╤З╨░ (╨┐╤А╨╕╨╖╨╜╨░╨║ rate limit)
                delay_after_batch = current_exchange.current_request_delay if hasattr(current_exchange, 'current_request_delay') else None
                if delay_before_batch is not None and delay_after_batch is not None:
                    if delay_after_batch > delay_before_batch:
                        # ╨Ч╨░╨┤╨╡╤А╨╢╨║╨░ ╤Г╨▓╨╡╨╗╨╕╤З╨╕╨╗╨░╤Б╤М - ╨▒╤Л╨╗ rate limit
                        rate_limit_detected = True
                        logger.warning(f"тЪая╕П Rate limit ╨╛╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜ ╨▓ ╨▒╨░╤В╤З╨╡ {batch_num}/{total_batches}: ╨╖╨░╨┤╨╡╤А╨╢╨║╨░ ╤Г╨▓╨╡╨╗╨╕╤З╨╕╨╗╨░╤Б╤М {delay_before_batch:.3f}╤Б тЖТ {delay_after_batch:.3f}╤Б")
                
                # ╨г╨╝╨╡╨╜╤М╤И╨╕╨╗╨╕ ╨┐╨░╤Г╨╖╤Г ╨╝╨╡╨╢╨┤╤Г ╨┐╨░╨║╨╡╤В╨░╨╝╨╕
                import time
                if shutdown_flag.wait(0.1):
                    shutdown_requested = True
                    break

            if shutdown_requested:
                break
        
        if shutdown_requested:
            logger.warning("тП╣я╕П ╨Ч╨░╨│╤А╤Г╨╖╨║╨░ ╤Б╨▓╨╡╤З╨╡╨╣ ╨┐╤А╨╡╤А╨▓╨░╨╜╨░ ╨╕╨╖-╨╖╨░ ╨╛╤Б╤В╨░╨╜╨╛╨▓╨║╨╕ ╤Б╨╕╤Б╤В╨╡╨╝╤Л")
            return False
        
        logger.info(f"тЬЕ ╨Ч╨░╨│╤А╤Г╨╖╨║╨░ ╨╖╨░╨▓╨╡╤А╤И╨╡╨╜╨░: {len(candles_cache)} ╨╝╨╛╨╜╨╡╤В")
        
        # тЪб ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ш╨Х DEADLOCK: ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨▓ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Л╨╣ ╨║╤Н╤И ╨С╨Х╨Ч ╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨╕
        # rsi_data_lock ╨╝╨╛╨╢╨╡╤В ╨▒╤Л╤В╤М ╨╖╨░╤Е╨▓╨░╤З╨╡╨╜ ContinuousDataLoader ╨▓ ╨┤╤А╤Г╨│╨╛╨╝ ╨┐╨╛╤В╨╛╨║╨╡
        try:
            logger.info(f"ЁЯТ╛ ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨║╤Н╤И ╨▓ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╨╛╨╡ ╤Е╤А╨░╨╜╨╕╨╗╨╕╤Й╨╡...")
            coins_rsi_data['candles_cache'] = candles_cache
            coins_rsi_data['last_candles_update'] = datetime.now().isoformat()
            logger.info(f"тЬЕ ╨Ъ╤Н╤И ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜: {len(candles_cache)} ╨╝╨╛╨╜╨╡╤В")
            logger.info(f"тЬЕ ╨Я╤А╨╛╨▓╨╡╤А╨║╨░: ╨▓ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╨╛╨╝ ╨║╤Н╤И╨╡ ╤Б╨╡╨╣╤З╨░╤Б {len(coins_rsi_data.get('candles_cache', {}))} ╨╝╨╛╨╜╨╡╤В")
        except Exception as cache_error:
            logger.warning(f"тЪая╕П ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П ╨║╤Н╤И╨░: {cache_error}")
        
        # тЬЕ ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤Б╨▓╨╡╤З╨╕ ╨▓ ╨С╨Ф ╨С╨Х╨Ч ╨╜╨░╨║╨╛╨┐╨╗╨╡╨╜╨╕╤П!
        # ╨Ч╨░╨┐╤А╨░╤И╨╕╨▓╨░╨╡╤В╤Б╤П ╤В╨╛╨╗╤М╨║╨╛ 30 ╨┤╨╜╨╡╨╣ (~120 ╤Б╨▓╨╡╤З╨╡╨╣), ╨┐╨╛╤Н╤В╨╛╨╝╤Г ╨Э╨Х ╨╜╤Г╨╢╨╜╨╛ ╨╜╨░╨║╨░╨┐╨╗╨╕╨▓╨░╤В╤М ╤Б╤В╨░╤А╤Л╨╡ ╨┤╨░╨╜╨╜╤Л╨╡
        # save_candles_cache() ╤Б╨░╨╝ ╤Г╨┤╨░╨╗╨╕╤В ╤Б╤В╨░╤А╤Л╨╡ ╤Б╨▓╨╡╤З╨╕ ╨╕ ╨▓╤Б╤В╨░╨▓╨╕╤В ╤В╨╛╨╗╤М╨║╨╛ ╨╜╨╛╨▓╤Л╨╡
        # тЪая╕П ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨╖╨░╨┐╤Г╤Й╨╡╨╜ ╨╗╨╕ ╨┐╤А╨╛╤Ж╨╡╤Б╤Б ╨║╨░╨║ ai.py - ╨╡╤Б╨╗╨╕ ╨┤╨░, ╨Э╨Х ╤Б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨▓ bots_data.db!
        try:
            import sys
            import os
            # ╨С╨╛╨╗╨╡╨╡ ╨╜╨░╨┤╨╡╨╢╨╜╨░╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨░: ╤Б╨╝╨╛╤В╤А╨╕╨╝ ╨╕╨╝╤П ╤Б╨║╤А╨╕╨┐╤В╨░, ╨╝╨╛╨┤╤Г╨╗╤М __main__ ╨╕ ╨┐╨╡╤А╨╡╨╝╨╡╨╜╨╜╤Л╨╡ ╨╛╨║╤А╤Г╨╢╨╡╨╜╨╕╤П
            script_name = os.path.basename(sys.argv[0]) if sys.argv else ''
            main_file = None
            try:
                if hasattr(sys.modules.get('__main__', None), '__file__') and sys.modules['__main__'].__file__:
                    main_file = str(sys.modules['__main__'].__file__).lower()
            except:
                pass
            
            # тЪая╕П ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨п╨▓╨╜╨╛ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╤Г╨╡╨╝ ╨┐╨╡╤А╨╡╨╝╨╡╨╜╨╜╤Л╨╡
            is_bots_process = False
            is_ai_process = False
            
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┐╨╛ ╨╕╨╝╨╡╨╜╨╕ ╤Б╨║╤А╨╕╨┐╤В╨░, ╨░╤А╨│╤Г╨╝╨╡╨╜╤В╨░╨╝, ╤Д╨░╨╣╨╗╤Г __main__ ╨╕ ╨┐╨╡╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╣ ╨╛╨║╤А╤Г╨╢╨╡╨╜╨╕╤П
            # тЪая╕П ╨Т╨Р╨Ц╨Э╨Ю: ╨б╨╜╨░╤З╨░╨╗╨░ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╤З╤В╨╛ ╤Н╤В╨╛ ╨Э╨Х bots.py, ╨┐╨╛╤В╨╛╨╝ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ai.py
            is_bots_process = (
                'bots.py' in script_name.lower() or 
                any('bots.py' in str(arg).lower() for arg in sys.argv) or
                (main_file and 'bots.py' in main_file)
            )
            
            # ╨Х╤Б╨╗╨╕ ╤Н╤В╨╛ ╤В╨╛╤З╨╜╨╛ bots.py - ╨Э╨Х ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┤╨░╨╗╤М╤И╨╡ ╨╕ ╨╕╨│╨╜╨╛╤А╨╕╤А╤Г╨╡╨╝ ╨┐╨╡╤А╨╡╨╝╨╡╨╜╨╜╤Г╤О ╨╛╨║╤А╤Г╨╢╨╡╨╜╨╕╤П
            if is_bots_process:
                is_ai_process = False
                logger.debug(f"ЁЯФН ╨Ю╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜ ╨┐╤А╨╛╤Ж╨╡╤Б╤Б bots.py - ╤Б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤Б╨▓╨╡╤З╨╕ ╨▓ bots_data.db (script_name={script_name}, main_file={main_file})")
            else:
                # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╤З╤В╨╛ ╤Н╤В╨╛ ai.py (╨┐╨╡╤А╨╡╨╝╨╡╨╜╨╜╨░╤П ╨╛╨║╤А╤Г╨╢╨╡╨╜╨╕╤П ╤Г╤З╨╕╤В╤Л╨▓╨░╨╡╤В╤Б╤П ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╨╡╤Б╨╗╨╕ ╤Н╤В╨╛ ╨╜╨╡ bots.py)
                env_flag = os.environ.get('INFOBOT_AI_PROCESS', '').lower() == 'true'
                is_ai_process = (
                    'ai.py' in script_name.lower() or 
                    any('ai.py' in str(arg).lower() for arg in sys.argv) or
                    (main_file and 'ai.py' in main_file) or
                    env_flag
                )
                if is_ai_process:
                    logger.info(f"ЁЯФН ╨Ю╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜ ╨┐╤А╨╛╤Ж╨╡╤Б╤Б ai.py - ╤Б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤Б╨▓╨╡╤З╨╕ ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╨▓ ai_data.db (script_name={script_name}, main_file={main_file}, env_flag={env_flag})")
            
            if is_ai_process:
                # ╨Х╤Б╨╗╨╕ ╤Н╤В╨╛ ╨┐╤А╨╛╤Ж╨╡╤Б╤Б ai.py - ╤Б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╨▓ ai_data.db, ╨Э╨Х ╨▓ bots_data.db!
                logger.info(f"ЁЯФН ╨Ю╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜ ╨┐╤А╨╛╤Ж╨╡╤Б╤Б ai.py - ╤Б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤Б╨▓╨╡╤З╨╕ ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╨▓ ai_data.db (script_name={script_name}, main_file={main_file}, env={os.environ.get('INFOBOT_AI_PROCESS', '')})")
                try:
                    from bot_engine.ai.ai_database import get_ai_database
                    ai_db = get_ai_database()
                    if ai_db:
                        # ╨Я╤А╨╡╨╛╨▒╤А╨░╨╖╤Г╨╡╨╝ ╤Д╨╛╤А╨╝╨░╤В ╨┤╨╗╤П ai_database
                        saved_count = 0
                        for symbol, candle_data in candles_cache.items():
                            if isinstance(candle_data, dict):
                                candles = candle_data.get('candles', [])
                                if candles:
                                    ai_db.save_candles(symbol, candles, timeframe='6h')
                                    saved_count += 1
                        logger.info(f"тЬЕ ╨б╨▓╨╡╤З╨╕ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╤Л ╨▓ ai_data.db: {saved_count} ╨╝╨╛╨╜╨╡╤В (╨┐╤А╨╛╤Ж╨╡╤Б╤Б ai.py)")
                    else:
                        logger.error("тЭМ AI Database ╨╜╨╡╨┤╨╛╤Б╤В╤Г╨┐╨╜╨░, ╤Б╨▓╨╡╤З╨╕ ╨Э╨Х ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╤Л!")
                except Exception as ai_db_error:
                    logger.error(f"тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П ╨▓ ai_data.db: {ai_db_error}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                # ╨н╤В╨╛ ╨┐╤А╨╛╤Ж╨╡╤Б╤Б bots.py ╨╕╨╗╨╕ ╨╜╨╡╨╕╨╖╨▓╨╡╤Б╤В╨╜╤Л╨╣ ╨┐╤А╨╛╤Ж╨╡╤Б╤Б - ╤Б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨▓ bots_data.db
                # тЪая╕П ╨Т╨Р╨Ц╨Э╨Ю: ╨Х╤Б╨╗╨╕ ╤Н╤В╨╛ ╨Э╨Х bots.py ╨╕ ╨Э╨Х ai.py - ╤Н╤В╨╛ ╨╝╨╛╨╢╨╡╤В ╨▒╤Л╤В╤М ╨╛╤И╨╕╨▒╨║╨░!
                if not is_bots_process:
                    logger.warning(f"тЪая╕П ╨Э╨╡╨╕╨╖╨▓╨╡╤Б╤В╨╜╤Л╨╣ ╨┐╤А╨╛╤Ж╨╡╤Б╤Б ╨▓╤Л╨╖╤Л╨▓╨░╨╡╤В load_all_coins_candles_fast()! script_name={script_name}, main_file={main_file}")
                    logger.warning(f"тЪая╕П ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨▓ bots_data.db (╨┐╨╛ ╤Г╨╝╨╛╨╗╤З╨░╨╜╨╕╤О)")
                
                from bot_engine.storage import save_candles_cache
                
                # ╨Я╤А╨╛╤Б╤В╨╛ ╤Б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╤В╨╡╨║╤Г╤Й╨╕╨╡ ╤Б╨▓╨╡╤З╨╕ - save_candles_cache() ╤Б╨░╨╝ ╨╛╨│╤А╨░╨╜╨╕╤З╨╕╤В ╨┤╨╛ 1000 ╨╕ ╤Г╨┤╨░╨╗╨╕╤В ╤Б╤В╨░╤А╤Л╨╡
                if save_candles_cache(candles_cache):
                    logger.info(f"ЁЯТ╛ ╨Ъ╤Н╤И ╤Б╨▓╨╡╤З╨╡╨╣ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜ ╨▓ bots_data.db: {len(candles_cache)} ╨╝╨╛╨╜╨╡╤В (╨┐╤А╨╛╤Ж╨╡╤Б╤Б bots.py)")
                else:
                    logger.error(f"тЭМ ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╤Б╨╛╤Е╤А╨░╨╜╨╕╤В╤М ╤Б╨▓╨╡╤З╨╕ ╨▓ bots_data.db!")
            
        except Exception as db_error:
            logger.warning(f"тЪая╕П ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╤П ╨▓ ╨С╨Ф ╨║╤Н╤И╨░: {db_error}")
        
        # ЁЯФД ╨б╨▒╤А╨░╤Б╤Л╨▓╨░╨╡╨╝ ╨╖╨░╨┤╨╡╤А╨╢╨║╤Г ╨╖╨░╨┐╤А╨╛╤Б╨╛╨▓ ╨┐╨╛╤Б╨╗╨╡ ╤Г╤Б╨┐╨╡╤И╨╜╨╛╨╣ ╨╖╨░╨│╤А╤Г╨╖╨║╨╕ ╤А╨░╤Г╨╜╨┤╨░
        try:
            if current_exchange and hasattr(current_exchange, 'reset_request_delay'):
                current_exchange.reset_request_delay()
                logger.info(f"ЁЯФД ╨Ч╨░╨┤╨╡╤А╨╢╨║╨░ ╨╖╨░╨┐╤А╨╛╤Б╨╛╨▓ ╤Б╨▒╤А╨╛╤И╨╡╨╜╨░ ╨║ ╨▒╨░╨╖╨╛╨▓╨╛╨╝╤Г ╨╖╨╜╨░╤З╨╡╨╜╨╕╤О")
        except Exception as reset_error:
            logger.warning(f"тЪая╕П ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨▒╤А╨╛╤Б╨░ ╨╖╨░╨┤╨╡╤А╨╢╨║╨╕: {reset_error}")
        
        return True
        
    except Exception as e:
        logger.error(f"тЭМ ╨Ю╤И╨╕╨▒╨║╨░: {e}")
        return False

def load_all_coins_rsi():
    """╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╤В RSI 6H ╨┤╨╗╤П ╨▓╤Б╨╡╤Е ╨┤╨╛╤Б╤В╤Г╨┐╨╜╤Л╤Е ╨╝╨╛╨╜╨╡╤В"""
    global coins_rsi_data
    
    try:
        operation_start = time.time()
        logger.info("ЁЯУК RSI: ╨╖╨░╨┐╤Г╤Б╨║╨░╨╡╨╝ ╨┐╨╛╨╗╨╜╨╛╨╡ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╡")
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Д╨╗╨░╨│ ╨▒╨╡╨╖ ╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨╕
        if coins_rsi_data['update_in_progress']:
            logger.info("╨Ю╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╡ RSI ╤Г╨╢╨╡ ╨▓╤Л╨┐╨╛╨╗╨╜╤П╨╡╤В╤Б╤П...")
            return False
        
        # тЪб ╨г╨б╨в╨Р╨Э╨Р╨Т╨Ы╨Ш╨Т╨Р╨Х╨Ь ╤Д╨╗╨░╨│╨╕ ╨С╨Х╨Ч ╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨╕
        coins_rsi_data['update_in_progress'] = True
        # тЬЕ UI ╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨░ ╤Г╨╢╨╡ ╤Г╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨░ ╨▓ continuous_data_loader
        
        if shutdown_flag.is_set():
            logger.warning("тП╣я╕П ╨Ю╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╡ RSI ╨╛╤В╨╝╨╡╨╜╨╡╨╜╨╛: ╤Б╨╕╤Б╤В╨╡╨╝╨░ ╨╖╨░╨▓╨╡╤А╤И╨░╨╡╤В ╤А╨░╨▒╨╛╤В╤Г")
            coins_rsi_data['update_in_progress'] = False
            return False

        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╨Т╨а╨Х╨Ь╨Х╨Э╨Э╨Ю╨Х ╤Е╤А╨░╨╜╨╕╨╗╨╕╤Й╨╡ ╨┤╨╗╤П ╨▓╤Б╨╡╤Е ╨╝╨╛╨╜╨╡╤В
        # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ coins_rsi_data ╨в╨Ю╨Ы╨м╨Ъ╨Ю ╨┐╨╛╤Б╨╗╨╡ ╨╖╨░╨▓╨╡╤А╤И╨╡╨╜╨╕╤П ╨▓╤Б╨╡╤Е ╨┐╤А╨╛╨▓╨╡╤А╨╛╨║!
        temp_coins_data = {}
        
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨║╤Н╤И ╤Б╨▓╨╡╤З╨╡╨╣ ╨┐╨╡╤А╨╡╨┤ ╨╜╨░╤З╨░╨╗╨╛╨╝
        candles_cache_size = len(coins_rsi_data.get('candles_cache', {}))
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨░╨║╤В╤Г╨░╨╗╤М╨╜╤Г╤О ╤Б╤Б╤Л╨╗╨║╤Г ╨╜╨░ ╨▒╨╕╤А╨╢╤Г
        try:
            from bots_modules.imports_and_globals import get_exchange
            current_exchange = get_exchange()
        except Exception as e:
            logger.error(f"тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╤П ╨▒╨╕╤А╨╢╨╕: {e}")
            current_exchange = None
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╤Б╨┐╨╕╤Б╨╛╨║ ╨▓╤Б╨╡╤Е ╨┐╨░╤А
        if not current_exchange:
            logger.error("тЭМ ╨С╨╕╤А╨╢╨░ ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░")
            coins_rsi_data['update_in_progress'] = False
            return False
            
        pairs = current_exchange.get_all_pairs()
        
        if not pairs or not isinstance(pairs, list):
            logger.error("тЭМ ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╤Б╨┐╨╕╤Б╨╛╨║ ╨┐╨░╤А ╤Б ╨▒╨╕╤А╨╢╨╕")
            return False
        
        logger.info(f"ЁЯУК RSI: ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╛ {len(pairs)} ╨┐╨░╤А, ╨│╨╛╤В╨╛╨▓╨╕╨╝ ╨▒╨░╤В╤З╨╕ ╨┐╨╛ 100 ╨╝╨╛╨╜╨╡╤В")
        
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╨╛╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╤Б╤З╨╡╤В╤З╨╕╨║╨╕ ╨╜╨░╨┐╤А╤П╨╝╤Г╤О
        coins_rsi_data['total_coins'] = len(pairs)
        coins_rsi_data['successful_coins'] = 0
        coins_rsi_data['failed_coins'] = 0
        
        # тЬЕ ╨Я╨Р╨а╨Р╨Ы╨Ы╨Х╨Ы╨м╨Э╨Р╨п ╨╖╨░╨│╤А╤Г╨╖╨║╨░ ╤Б ╤В╨╡╨║╤Б╤В╨╛╨▓╤Л╨╝ ╨┐╤А╨╛╨│╤А╨╡╤Б╤Б╨╛╨╝ (╤А╨░╨▒╨╛╤В╨░╨╡╤В ╨▓ ╨╗╨╛╨│-╤Д╨░╨╣╨╗╨╡)
        batch_size = 100
        total_batches = (len(pairs) + batch_size - 1) // batch_size
        
        shutdown_requested = False

        for i in range(0, len(pairs), batch_size):
            if shutdown_flag.is_set():
                shutdown_requested = True
                break
            batch = pairs[i:i + batch_size]
            batch_num = i // batch_size + 1
            batch_start = time.time()
            request_delay = getattr(current_exchange, 'current_request_delay', 0) or 0
            logger.info(
                f"ЁЯУж RSI Batch {batch_num}/{total_batches}: size={len(batch)}, "
                f"workers=50, delay={request_delay:.2f}s"
            )
            batch_success = 0
            batch_fail = 0
            
            # ╨Я╨░╤А╨░╨╗╨╗╨╡╨╗╤М╨╜╨░╤П ╨╛╨▒╤А╨░╨▒╨╛╤В╨║╨░ ╨┐╨░╨║╨╡╤В╨░
            with ThreadPoolExecutor(max_workers=50) as executor:
                future_to_symbol = {
                    executor.submit(get_coin_rsi_data, symbol, current_exchange): symbol 
                    for symbol in batch
                }

                if shutdown_flag.is_set():
                    shutdown_requested = True
                    for future in future_to_symbol:
                        future.cancel()
                    break
                
                try:
                    for future in concurrent.futures.as_completed(future_to_symbol, timeout=60):
                        if shutdown_flag.is_set():
                            shutdown_requested = True
                            break

                        symbol = future_to_symbol[future]
                        try:
                            result = future.result(timeout=20)
                            if result:
                                temp_coins_data[result['symbol']] = result
                                coins_rsi_data['successful_coins'] += 1
                                batch_success += 1
                            else:
                                coins_rsi_data['failed_coins'] += 1
                                batch_fail += 1
                        except Exception as e:
                            logger.error(f"тЭМ {symbol}: {e}")
                            coins_rsi_data['failed_coins'] += 1
                            batch_fail += 1
                except concurrent.futures.TimeoutError:
                    pending = list(future_to_symbol.values())
                    logger.error(
                        f"тЪая╕П Timeout ╨┐╤А╨╕ ╨╖╨░╨│╤А╤Г╨╖╨║╨╡ RSI ╨┤╨╗╤П ╨┐╨░╨║╨╡╤В╨░ {batch_num} "
                        f"(╨╛╨╢╨╕╨┤╨░╨╗╨╕ {len(pending)} ╤Б╨╕╨╝╨▓╨╛╨╗╨╛╨▓, ╨┐╤А╨╕╨╝╨╡╤А╤Л: {pending[:5]})"
                    )
                    coins_rsi_data['failed_coins'] += len(batch)
                    batch_fail += len(batch)

                if shutdown_flag.is_set():
                    shutdown_requested = True
                    for future in future_to_symbol:
                        future.cancel()
                    break
            
            logger.info(
                f"ЁЯУж RSI Batch {batch_num}/{total_batches} ╨╖╨░╨▓╨╡╤А╤И╨╡╨╜: "
                f"{batch_success} ╤Г╤Б╨┐╨╡╤Е╨╛╨▓ / {batch_fail} ╨╛╤И╨╕╨▒╨╛╨║ ╨╖╨░ "
                f"{time.time() - batch_start:.1f}s"
            )
            
            # тЬЕ ╨Т╤Л╨▓╨╛╨┤╨╕╨╝ ╨┐╤А╨╛╨│╤А╨╡╤Б╤Б ╨▓ ╨╗╨╛╨│
            processed = coins_rsi_data['successful_coins'] + coins_rsi_data['failed_coins']
            if batch_num <= total_batches:
                logger.info(f"ЁЯУК ╨Я╤А╨╛╨│╤А╨╡╤Б╤Б: {processed}/{len(pairs)} ({processed*100//len(pairs)}%)")

            if shutdown_requested:
                break

        if shutdown_requested:
            logger.warning("тП╣я╕П ╨Ю╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╡ RSI ╨┐╤А╨╡╤А╨▓╨░╨╜╨╛ ╨╕╨╖-╨╖╨░ ╨╛╤Б╤В╨░╨╜╨╛╨▓╨║╨╕ ╤Б╨╕╤Б╤В╨╡╨╝╤Л")
            coins_rsi_data['update_in_progress'] = False
            return False
        
        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Р╨в╨Ю╨Ь╨Р╨а╨Э╨Ю╨Х ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╡ ╨▓╤Б╨╡╤Е ╨┤╨░╨╜╨╜╤Л╤Е ╨Ю╨Ф╨Э╨Ш╨Ь ╨Ь╨Р╨е╨Ю╨Ь!
        coins_rsi_data['coins'] = temp_coins_data
        coins_rsi_data['last_update'] = datetime.now().isoformat()
        coins_rsi_data['update_in_progress'] = False
        
        # ╨д╨╕╨╜╨░╨╗╤М╨╜╤Л╨╣ ╨╛╤В╤З╨╡╤В
        success_count = coins_rsi_data['successful_coins']
        failed_count = coins_rsi_data['failed_coins']
            
        # ╨Я╨╛╨┤╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╨╝ ╤Б╨╕╨│╨╜╨░╨╗╤Л
        enter_long_count = sum(1 for coin in coins_rsi_data['coins'].values() if coin.get('signal') == 'ENTER_LONG')
        enter_short_count = sum(1 for coin in coins_rsi_data['coins'].values() if coin.get('signal') == 'ENTER_SHORT')
        
        logger.info(f"тЬЕ {success_count} ╨╝╨╛╨╜╨╡╤В | ╨б╨╕╨│╨╜╨░╨╗╤Л: {enter_long_count} LONG + {enter_short_count} SHORT")
        
        if failed_count > 0:
            logger.warning(f"тЪая╕П ╨Ю╤И╨╕╨▒╨╛╨║: {failed_count} ╨╝╨╛╨╜╨╡╤В")
        
        # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╤Д╨╗╨░╨│╨╕ is_mature
        try:
            update_is_mature_flags_in_rsi_data()
        except Exception as update_error:
            logger.warning(f"тЪая╕П ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨╛╨▒╨╜╨╛╨▓╨╕╤В╤М is_mature: {update_error}")
        
        # ЁЯФД ╨б╨▒╤А╨░╤Б╤Л╨▓╨░╨╡╨╝ ╨╖╨░╨┤╨╡╤А╨╢╨║╤Г ╨╖╨░╨┐╤А╨╛╤Б╨╛╨▓ ╨┐╨╛╤Б╨╗╨╡ ╤Г╤Б╨┐╨╡╤И╨╜╨╛╨╣ ╨╖╨░╨│╤А╤Г╨╖╨║╨╕ ╤А╨░╤Г╨╜╨┤╨░
        try:
            if current_exchange and hasattr(current_exchange, 'reset_request_delay'):
                current_exchange.reset_request_delay()
                logger.info(f"ЁЯФД ╨Ч╨░╨┤╨╡╤А╨╢╨║╨░ ╨╖╨░╨┐╤А╨╛╤Б╨╛╨▓ ╤Б╨▒╤А╨╛╤И╨╡╨╜╨░ ╨║ ╨▒╨░╨╖╨╛╨▓╨╛╨╝╤Г ╨╖╨╜╨░╤З╨╡╨╜╨╕╤О")
        except Exception as reset_error:
            logger.warning(f"тЪая╕П ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨▒╤А╨╛╤Б╨░ ╨╖╨░╨┤╨╡╤А╨╢╨║╨╕: {reset_error}")
        
        return True
        
    except Exception as e:
        logger.error(f"╨Ю╤И╨╕╨▒╨║╨░ ╨╖╨░╨│╤А╤Г╨╖╨║╨╕ RSI ╨┤╨░╨╜╨╜╤Л╤Е: {str(e)}")
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
        coins_rsi_data['update_in_progress'] = False
        return False
    finally:
        elapsed = time.time() - operation_start
        logger.info(f"ЁЯУК RSI: ╨┐╨╛╨╗╨╜╨╛╨╡ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╡ ╨╖╨░╨▓╨╡╤А╤И╨╡╨╜╨╛ ╨╖╨░ {elapsed:.1f}s")
        # ╨У╨░╤А╨░╨╜╤В╨╕╤А╨╛╨▓╨░╨╜╨╜╨╛ ╤Б╨▒╤А╨░╤Б╤Л╨▓╨░╨╡╨╝ ╤Д╨╗╨░╨│ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
        if coins_rsi_data['update_in_progress']:
            logger.warning(f"тЪая╕П ╨Я╤А╨╕╨╜╤Г╨┤╨╕╤В╨╡╨╗╤М╨╜╤Л╨╣ ╤Б╨▒╤А╨╛╤Б ╤Д╨╗╨░╨│╨░ update_in_progress")
            coins_rsi_data['update_in_progress'] = False

def _recalculate_signal_with_trend(rsi, trend, symbol):
    """╨Я╨╡╤А╨╡╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╤В ╤Б╨╕╨│╨╜╨░╨╗ ╤Б ╤Г╤З╨╡╤В╨╛╨╝ ╨╜╨╛╨▓╨╛╨│╨╛ ╤В╤А╨╡╨╜╨┤╨░"""
    try:
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╨░╨▓╤В╨╛╨▒╨╛╤В╨░
        auto_config = bots_data.get('auto_bot_config', {})
        # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ю: ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ False ╨┐╨╛ ╤Г╨╝╨╛╨╗╤З╨░╨╜╨╕╤О (╨║╨░╨║ ╨▓ bot_config.py), ╨░ ╨╜╨╡ True
        avoid_down_trend = auto_config.get('avoid_down_trend', False)
        avoid_up_trend = auto_config.get('avoid_up_trend', False)
        
        # ╨Ю╨┐╤А╨╡╨┤╨╡╨╗╤П╨╡╨╝ ╨▒╨░╨╖╨╛╨▓╤Л╨╣ ╤Б╨╕╨│╨╜╨░╨╗ ╨┐╨╛ RSI
        if rsi <= SystemConfig.RSI_OVERSOLD:  # RSI тЙд 29 
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╜╤Г╨╢╨╜╨╛ ╨╗╨╕ ╨╕╨╖╨▒╨╡╨│╨░╤В╤М DOWN ╤В╤А╨╡╨╜╨┤╨░ ╨┤╨╗╤П LONG
            if avoid_down_trend and trend == 'DOWN':
                return 'WAIT'  # ╨Ц╨┤╨╡╨╝ ╤Г╨╗╤Г╤З╤И╨╡╨╜╨╕╤П ╤В╤А╨╡╨╜╨┤╨░
            else:
                return 'ENTER_LONG'  # ╨Т╤Е╨╛╨┤╨╕╨╝ ╨╜╨╡╨╖╨░╨▓╨╕╤Б╨╕╨╝╨╛ ╨╛╤В ╤В╤А╨╡╨╜╨┤╨░ ╨╕╨╗╨╕ ╨┐╤А╨╕ ╤Е╨╛╤А╨╛╤И╨╡╨╝ ╤В╤А╨╡╨╜╨┤╨╡
        elif rsi >= SystemConfig.RSI_OVERBOUGHT:  # RSI тЙе 71
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╜╤Г╨╢╨╜╨╛ ╨╗╨╕ ╨╕╨╖╨▒╨╡╨│╨░╤В╤М UP ╤В╤А╨╡╨╜╨┤╨░ ╨┤╨╗╤П SHORT
            if avoid_up_trend and trend == 'UP':
                return 'WAIT'  # ╨Ц╨┤╨╡╨╝ ╨╛╤Б╨╗╨░╨▒╨╗╨╡╨╜╨╕╤П ╤В╤А╨╡╨╜╨┤╨░
            else:
                return 'ENTER_SHORT'  # ╨Т╤Е╨╛╨┤╨╕╨╝ ╨╜╨╡╨╖╨░╨▓╨╕╤Б╨╕╨╝╨╛ ╨╛╤В ╤В╤А╨╡╨╜╨┤╨░ ╨╕╨╗╨╕ ╨┐╤А╨╕ ╤Е╨╛╤А╨╛╤И╨╡╨╝ ╤В╤А╨╡╨╜╨┤╨╡
        else:
            # RSI ╨╝╨╡╨╢╨┤╤Г 30-70 - ╨╜╨╡╨╣╤В╤А╨░╨╗╤М╨╜╨░╤П ╨╖╨╛╨╜╨░
            logger.debug(f"ЁЯФН {symbol}: RSI {rsi:.1f} ╨╝╨╡╨╢╨┤╤Г 30-70 тЖТ WAIT")
            return 'WAIT'
            
    except Exception as e:
        logger.error(f"тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨╡╤А╨╡╤Б╤З╨╡╤В╨░ ╤Б╨╕╨│╨╜╨░╨╗╨░ ╨┤╨╗╤П {symbol}: {e}")
        return 'WAIT'

def get_effective_signal(coin):
    """
    ╨г╨╜╨╕╨▓╨╡╤А╤Б╨░╨╗╤М╨╜╨░╤П ╤Д╤Г╨╜╨║╤Ж╨╕╤П ╨┤╨╗╤П ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜╨╕╤П ╤Н╤Д╤Д╨╡╨║╤В╨╕╨▓╨╜╨╛╨│╨╛ ╤Б╨╕╨│╨╜╨░╨╗╨░ ╨╝╨╛╨╜╨╡╤В╤Л
    
    ╨Ы╨Ю╨У╨Ш╨Ъ╨Р ╨Я╨а╨Ю╨Т╨Х╨а╨Ъ╨Ш ╨в╨а╨Х╨Э╨Ф╨Ю╨Т (╤Г╨┐╤А╨╛╤Й╨╡╨╜╨╜╨░╤П):
    - ╨Э╨Х ╨╛╤В╨║╤А╤Л╨▓╨░╨╡╨╝ SHORT ╨╡╤Б╨╗╨╕ RSI > 71 ╨Ш ╤В╤А╨╡╨╜╨┤ = UP
    - ╨Э╨Х ╨╛╤В╨║╤А╤Л╨▓╨░╨╡╨╝ LONG ╨╡╤Б╨╗╨╕ RSI < 29 ╨Ш ╤В╤А╨╡╨╜╨┤ = DOWN
    - NEUTRAL ╤В╤А╨╡╨╜╨┤ ╤А╨░╨╖╤А╨╡╤И╨░╨╡╤В ╨╗╤О╨▒╤Л╨╡ ╤Б╨┤╨╡╨╗╨║╨╕
    - ╨в╤А╨╡╨╜╨┤ ╤В╨╛╨╗╤М╨║╨╛ ╤Г╤Б╨╕╨╗╨╕╨▓╨░╨╡╤В ╨▓╨╛╨╖╨╝╨╛╨╢╨╜╨╛╤Б╤В╤М, ╨╜╨╛ ╨╜╨╡ ╨▒╨╗╨╛╨║╨╕╤А╤Г╨╡╤В ╨┐╨╛╨╗╨╜╨╛╤Б╤В╤М╤О
    
    Args:
        coin (dict): ╨Ф╨░╨╜╨╜╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л
        
    Returns:
        str: ╨н╤Д╤Д╨╡╨║╤В╨╕╨▓╨╜╤Л╨╣ ╤Б╨╕╨│╨╜╨░╨╗ (ENTER_LONG, ENTER_SHORT, WAIT)
    """
    symbol = coin.get('symbol', 'UNKNOWN')
    
    # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╨░╨▓╤В╨╛╨▒╨╛╤В╨░
    # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╨║╨╛╨╜╤Д╨╕╨│ ╨╜╨╡ ╨╝╨╡╨╜╤П╨╡╤В╤Б╤П, GIL ╨┤╨╡╨╗╨░╨╡╤В ╤З╤В╨╡╨╜╨╕╨╡ ╨░╤В╨╛╨╝╨░╤А╨╜╤Л╨╝
    auto_config = bots_data.get('auto_bot_config', {})
    # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ю: ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ False ╨┐╨╛ ╤Г╨╝╨╛╨╗╤З╨░╨╜╨╕╤О (╨║╨░╨║ ╨▓ bot_config.py), ╨░ ╨╜╨╡ True
    avoid_down_trend = auto_config.get('avoid_down_trend', False)
    avoid_up_trend = auto_config.get('avoid_up_trend', False)
    rsi_long_threshold = auto_config.get('rsi_long_threshold', 29)
    rsi_short_threshold = auto_config.get('rsi_short_threshold', 71)
        
    # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨┤╨░╨╜╨╜╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л
    rsi = coin.get('rsi6h', 50)
    trend = coin.get('trend', coin.get('trend6h', 'NEUTRAL'))
    
    # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╖╤А╨╡╨╗╨╛╤Б╤В╤М ╨╝╨╛╨╜╨╡╤В╤Л ╨Я╨Х╨а╨Т╨л╨Ь ╨Ф╨Х╨Ы╨Ю╨Ь
    # ╨Э╨╡╨╖╤А╨╡╨╗╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л ╨Э╨Х ╨Ь╨Ю╨У╨г╨в ╨╕╨╝╨╡╤В╤М ╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓ ╨╕ ╨Э╨Х ╨Ф╨Ю╨Ы╨Ц╨Э╨л ╨┐╨╛╨║╨░╨╖╤Л╨▓╨░╤В╤М╤Б╤П ╨▓ LONG/SHORT ╤Д╨╕╨╗╤М╤В╤А╨░╤Е!
    base_signal = coin.get('signal', 'WAIT')
    if base_signal == 'WAIT':
        # ╨Ь╨╛╨╜╨╡╤В╨░ ╨╜╨╡╨╖╤А╨╡╨╗╨░╤П - ╨╜╨╡ ╨┐╨╛╨║╨░╨╖╤Л╨▓╨░╨╡╨╝ ╨╡╤С ╨▓ ╤Д╨╕╨╗╤М╤В╤А╨░╤Е
        return 'WAIT'
    
    # тЬЕ ╨Ь╨╛╨╜╨╡╤В╨░ ╨╖╤А╨╡╨╗╨░╤П - ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ Enhanced RSI ╤Б╨╕╨│╨╜╨░╨╗
    enhanced_rsi = coin.get('enhanced_rsi', {})
    if enhanced_rsi.get('enabled') and enhanced_rsi.get('enhanced_signal'):
        signal = enhanced_rsi.get('enhanced_signal')
    else:
        # ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨▒╨░╨╖╨╛╨▓╤Л╨╣ ╤Б╨╕╨│╨╜╨░╨╗
        signal = base_signal
    
    # ╨Х╤Б╨╗╨╕ ╤Б╨╕╨│╨╜╨░╨╗ WAIT - ╨▓╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╨╝ ╤Б╤А╨░╨╖╤Г
    if signal == 'WAIT':
        return signal
    
    # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╤Л ╨Т╨б╨Х╨е ╤Д╨╕╨╗╤М╤В╤А╨╛╨▓!
    # ╨Х╤Б╨╗╨╕ ╨╗╤О╨▒╨╛╨╣ ╤Д╨╕╨╗╤М╤В╤А ╨╖╨░╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨░╨╗ ╤Б╨╕╨│╨╜╨░╨╗ - ╨▓╨╛╨╖╨▓╤А╨░╤Й╨░╨╡╨╝ WAIT
    
    # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ExitScam ╤Д╨╕╨╗╤М╤В╤А
    if coin.get('blocked_by_exit_scam', False):
        logger.debug(f"{symbol}: тЭМ {signal} ╨╖╨░╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨░╨╜ ExitScam ╤Д╨╕╨╗╤М╤В╤А╨╛╨╝")
        return 'WAIT'
    
    # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ RSI Time ╤Д╨╕╨╗╤М╤В╤А
    if coin.get('blocked_by_rsi_time', False):
        logger.debug(f"{symbol}: тЭМ {signal} ╨╖╨░╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨░╨╜ RSI Time ╤Д╨╕╨╗╤М╤В╤А╨╛╨╝")
        return 'WAIT'
    
    # тЬЕ ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╖╨░╤Й╨╕╤В╤Г ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓ ╨┐╨╛╤Б╨╗╨╡ ╤Г╨▒╤Л╤В╨╛╤З╨╜╤Л╤Е ╨╖╨░╨║╤А╤Л╤В╨╕╨╣
    if coin.get('blocked_by_loss_reentry', False):
        loss_reentry_info = coin.get('loss_reentry_info', {})
        reason = loss_reentry_info.get('reason', '╨Ч╨░╤Й╨╕╤В╨░ ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓') if loss_reentry_info else '╨Ч╨░╤Й╨╕╤В╨░ ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓'
        
        # тЬЕ ╨Р╨Э╨в╨Ш╨б╨Я╨Р╨Ь: ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ ╨╜╨╡ ╤З╨░╤Й╨╡ ╤А╨░╨╖╨░ ╨▓ ╨╝╨╕╨╜╤Г╤В╤Г ╨┤╨╗╤П ╨║╨░╨╢╨┤╨╛╨╣ ╨╝╨╛╨╜╨╡╤В╤Л
        current_time = time.time()
        with _loss_reentry_log_lock:
            last_log_time = _loss_reentry_log_cache.get(symbol, 0)
            if current_time - last_log_time >= _loss_reentry_log_interval:
                logger.debug(f"{symbol}: тЭМ {signal} ╨╖╨░╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨░╨╜ ╨╖╨░╤Й╨╕╤В╨╛╨╣ ╨╛╤В ╨┐╨╛╨▓╤В╨╛╤А╨╜╤Л╤Е ╨▓╤Е╨╛╨┤╨╛╨▓ ╨┐╨╛╤Б╨╗╨╡ ╤Г╨▒╤Л╤В╨║╨░: {reason}")
                _loss_reentry_log_cache[symbol] = current_time
        
        return 'WAIT'
    
    # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╖╤А╨╡╨╗╨╛╤Б╤В╤М ╨╝╨╛╨╜╨╡╤В╤Л
    if not coin.get('is_mature', True):
        # ╨Ю╨│╤А╨░╨╜╨╕╤З╨╕╨▓╨░╨╡╨╝ ╤З╨░╤Б╤В╨╛╤В╤Г ╨╗╨╛╨│╨╕╤А╨╛╨▓╨░╨╜╨╕╤П - ╨╜╨╡ ╨▒╨╛╨╗╨╡╨╡ ╤А╨░╨╖╨░ ╨▓ 2 ╨╝╨╕╨╜╤Г╤В╤Л ╨┤╨╗╤П ╨║╨░╨╢╨┤╨╛╨╣ ╨╝╨╛╨╜╨╡╤В╤Л
        log_message = f"{symbol}: тЭМ {signal} ╨╖╨░╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨░╨╜ - ╨╝╨╛╨╜╨╡╤В╨░ ╨╜╨╡╨╖╤А╨╡╨╗╨░╤П"
        category = f'maturity_check_{symbol}'
        should_log, message = should_log_message(category, log_message, interval_seconds=120)
        if should_log:
            logger.debug(message)
        return 'WAIT'
    
    # ╨г╨Я╨а╨Ю╨й╨Х╨Э╨Э╨Р╨п ╨Я╨а╨Ю╨Т╨Х╨а╨Ъ╨Р ╨в╨а╨Х╨Э╨Ф╨Ю╨Т - ╤В╨╛╨╗╤М╨║╨╛ ╤Н╨║╤Б╤В╤А╨╡╨╝╨░╨╗╤М╨╜╤Л╨╡ ╤Б╨╗╤Г╤З╨░╨╕
    if signal == 'ENTER_SHORT' and avoid_up_trend and rsi >= rsi_short_threshold and trend == 'UP':
        logger.debug(f"{symbol}: тЭМ SHORT ╨╖╨░╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨░╨╜ (RSI={rsi:.1f} >= {rsi_short_threshold} + UP ╤В╤А╨╡╨╜╨┤)")
        return 'WAIT'
    
    if signal == 'ENTER_LONG' and avoid_down_trend and rsi <= rsi_long_threshold and trend == 'DOWN':
        logger.debug(f"{symbol}: тЭМ LONG ╨╖╨░╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨░╨╜ (RSI={rsi:.1f} <= {rsi_long_threshold} + DOWN ╤В╤А╨╡╨╜╨┤)")
        return 'WAIT'
    
    # ╨Т╤Б╨╡ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨┐╤А╨╛╨╣╨┤╨╡╨╜╤Л
    return signal

def process_auto_bot_signals(exchange_obj=None):
    """╨Э╨╛╨▓╨░╤П ╨╗╨╛╨│╨╕╨║╨░ ╨░╨▓╤В╨╛╨▒╨╛╤В╨░ ╤Б╨╛╨│╨╗╨░╤Б╨╜╨╛ ╤В╤А╨╡╨▒╨╛╨▓╨░╨╜╨╕╤П╨╝"""
    try:
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨▓╨║╨╗╤О╤З╨╡╨╜ ╨╗╨╕ ╨░╨▓╤В╨╛╨▒╨╛╤В
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╨║╨╛╨╜╤Д╨╕╨│ ╨╜╨╡ ╨╝╨╡╨╜╤П╨╡╤В╤Б╤П, ╤З╤В╨╡╨╜╨╕╨╡ ╨▒╨╡╨╖╨╛╨┐╨░╤Б╨╜╨╛
        auto_bot_enabled = bots_data['auto_bot_config']['enabled']
        
        if not auto_bot_enabled:
            logger.info(" тП╣я╕П ╨Р╨▓╤В╨╛╨▒╨╛╤В ╨▓╤Л╨║╨╗╤О╤З╨╡╨╜")  # ╨Ш╨╖╨╝╨╡╨╜╨╡╨╜╨╛ ╨╜╨░ INFO
            return
        
        logger.info(" тЬЕ ╨Р╨▓╤В╨╛╨▒╨╛╤В ╨▓╨║╨╗╤О╤З╨╡╨╜, ╨╜╨░╤З╨╕╨╜╨░╨╡╨╝ ╨┐╤А╨╛╨▓╨╡╤А╨║╤Г ╤Б╨╕╨│╨╜╨░╨╗╨╛╨▓")
        
        max_concurrent = bots_data['auto_bot_config']['max_concurrent']
        current_active = sum(1 for bot in bots_data['bots'].values() 
                           if bot['status'] not in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']])
        
        if current_active >= max_concurrent:
            logger.debug(f" ЁЯЪл ╨Ф╨╛╤Б╤В╨╕╨│╨╜╤Г╤В ╨╗╨╕╨╝╨╕╤В ╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓ ({current_active}/{max_concurrent})")
            return
        
        logger.info(" ЁЯФН ╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╤Б╨╕╨│╨╜╨░╨╗╨╛╨▓ ╨┤╨╗╤П ╤Б╨╛╨╖╨┤╨░╨╜╨╕╤П ╨╜╨╛╨▓╤Л╤Е ╨▒╨╛╤В╨╛╨▓...")
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨╝╨╛╨╜╨╡╤В╤Л ╤Б ╤Б╨╕╨│╨╜╨░╨╗╨░╨╝╨╕
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╤З╤В╨╡╨╜╨╕╨╡ ╤Б╨╗╨╛╨▓╨░╤А╤П - ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
        potential_coins = []
        for symbol, coin_data in coins_rsi_data['coins'].items():
            rsi = coin_data.get('rsi6h')
            trend = coin_data.get('trend6h', 'NEUTRAL')
            
            if rsi is None:
                continue
            
            # тЬЕ ╨Ш╨б╨Я╨Ю╨Ы╨м╨Ч╨г╨Х╨Ь get_effective_signal() ╨║╨╛╤В╨╛╤А╤Л╨╣ ╤Г╤З╨╕╤В╤Л╨▓╨░╨╡╤В ╨Т╨б╨Х ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕:
            # - RSI ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╣ ╤Д╨╕╨╗╤М╤В╤А
            # - Enhanced RSI
            # - ╨Ч╤А╨╡╨╗╨╛╤Б╤В╤М ╨╝╨╛╨╜╨╡╤В╤Л (base_signal)
            # - ╨в╤А╨╡╨╜╨┤╤Л
            signal = get_effective_signal(coin_data)
            
            # ╨Х╤Б╨╗╨╕ ╤Б╨╕╨│╨╜╨░╨╗ ENTER_LONG ╨╕╨╗╨╕ ENTER_SHORT - ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╛╤Б╤В╨░╨╗╤М╨╜╤Л╨╡ ╤Д╨╕╨╗╤М╤В╤А╤Л
            if signal in ['ENTER_LONG', 'ENTER_SHORT']:
                # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┤╨╛╨┐╨╛╨╗╨╜╨╕╤В╨╡╨╗╤М╨╜╤Л╨╡ ╤Г╤Б╨╗╨╛╨▓╨╕╤П (whitelist/blacklist, ExitScam, ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕)
                if check_new_autobot_filters(symbol, signal, coin_data):
                    potential_coins.append({
                        'symbol': symbol,
                        'rsi': rsi,
                        'trend': trend,
                        'signal': signal,
                        'coin_data': coin_data
                    })
        
        logger.info(f" ЁЯОп ╨Э╨░╨╣╨┤╨╡╨╜╨╛ {len(potential_coins)} ╨┐╨╛╤В╨╡╨╜╤Ж╨╕╨░╨╗╤М╨╜╤Л╤Е ╤Б╨╕╨│╨╜╨░╨╗╨╛╨▓")
        
        # тЬЕ ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ ╨╜╨░╨╣╨┤╨╡╨╜╨╜╤Л╨╡ ╤Б╨╕╨│╨╜╨░╨╗╤Л ╨┤╨╗╤П ╨┤╨╕╨░╨│╨╜╨╛╤Б╤В╨╕╨║╨╕
        if potential_coins:
            logger.info(f" ЁЯУЛ ╨Я╨╛╤В╨╡╨╜╤Ж╨╕╨░╨╗╤М╨╜╤Л╨╡ ╤Б╨╕╨│╨╜╨░╨╗╤Л: {[(c['symbol'], c['signal'], f'RSI={c['rsi']:.1f}') for c in potential_coins[:10]]}")
        
        # ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╨▒╨╛╤В╨╛╨▓ ╨┤╨╗╤П ╨╜╨░╨╣╨┤╨╡╨╜╨╜╤Л╤Е ╤Б╨╕╨│╨╜╨░╨╗╨╛╨▓
        created_bots = 0
        for coin in potential_coins[:max_concurrent - current_active]:
            symbol = coin['symbol']
            
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨╜╨╡╤В ╨╗╨╕ ╤Г╨╢╨╡ ╨▒╨╛╤В╨░ ╨┤╨╗╤П ╤Н╤В╨╛╨│╨╛ ╤Б╨╕╨╝╨▓╨╛╨╗╨░
            # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╤З╤В╨╡╨╜╨╕╨╡ ╨▒╨╡╨╖╨╛╨┐╨░╤Б╨╜╨╛
            if symbol in bots_data['bots']:
                logger.info(f" тЪая╕П {symbol}: ╨С╨╛╤В ╤Г╨╢╨╡ ╤Б╤Г╤Й╨╡╤Б╤В╨▓╤Г╨╡╤В (╤Б╤В╨░╤В╤Г╤Б: {bots_data['bots'][symbol].get('status')})")
                continue
            
            # тЬЕ ╨Я╨а╨Ю╨Т╨Х╨а╨Ъ╨Р ╨Я╨Ю╨Ч╨Ш╨ж╨Ш╨Щ: ╨Х╤Б╤В╤М ╨╗╨╕ ╤А╤Г╤З╨╜╨░╤П ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╨╜╨░ ╨▒╨╕╤А╨╢╨╡?
            try:
                from bots_modules.workers import positions_cache
                
                # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨╡╤Б╤В╤М ╨╗╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╨┤╨╗╤П ╤Н╤В╨╛╨╣ ╨╝╨╛╨╜╨╡╤В╤Л
                if symbol in positions_cache['symbols_with_positions']:
                    # ╨Я╨╛╨╖╨╕╤Ж╨╕╤П ╨╡╤Б╤В╤М! ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨╡╤Б╤В╤М ╨╗╨╕ ╨░╨║╤В╨╕╨▓╨╜╤Л╨╣ ╨▒╨╛╤В ╨┤╨╗╤П ╨╜╨╡╤С
                    has_active_bot = False
                    if symbol in bots_data['bots']:
                        bot_status = bots_data['bots'][symbol].get('status')
                        if bot_status not in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']]:
                            has_active_bot = True
                    
                    if not has_active_bot:
                        # ╨Я╨╛╨╖╨╕╤Ж╨╕╤П ╨╡╤Б╤В╤М, ╨╜╨╛ ╨░╨║╤В╨╕╨▓╨╜╨╛╨│╨╛ ╨▒╨╛╤В╨░ ╨╜╨╡╤В - ╤Н╤В╨╛ ╨а╨г╨з╨Э╨Р╨п ╨┐╨╛╨╖╨╕╤Ж╨╕╤П!
                        logger.warning(f" ЁЯЪл {symbol}: ╨Ю╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜╨░ ╨а╨г╨з╨Э╨Р╨п ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╨╜╨░ ╨▒╨╕╤А╨╢╨╡ - ╨▒╨╗╨╛╨║╨╕╤А╤Г╨╡╨╝ ╤Б╨╛╨╖╨┤╨░╨╜╨╕╨╡ ╨▒╨╛╤В╨░!")
                        continue
                        
            except Exception as pos_error:
                logger.warning(f" тЪая╕П {symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣: {pos_error}")
                # ╨Я╤А╨╛╨┤╨╛╨╗╨╢╨░╨╡╨╝ ╤Б╨╛╨╖╨┤╨░╨╜╨╕╨╡ ╨▒╨╛╤В╨░ ╨╡╤Б╨╗╨╕ ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╨╜╨╡ ╤Г╨┤╨░╨╗╨░╤Б╤М
            
            # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨Т╨б╨Х ╤Д╨╕╨╗╤М╤В╤А╤Л ╨Я╨Х╨а╨Х╨Ф ╤Б╨╛╨╖╨┤╨░╨╜╨╕╨╡╨╝ ╨▒╨╛╤В╨░!
            try:
                from bot_engine.ai.filter_utils import apply_entry_filters
                from bots_modules.imports_and_globals import get_config_snapshot
                
                # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨║╨╛╨╜╤Д╨╕╨│
                config_snapshot = get_config_snapshot(symbol)
                filter_config = config_snapshot.get('merged', {})
                
                # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╤Б╨▓╨╡╤З╨╕ ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╤Д╨╕╨╗╤М╤В╤А╨╛╨▓
                candles = None
                # ╨б╨╜╨░╤З╨░╨╗╨░ ╨┐╤А╨╛╨▒╤Г╨╡╨╝ ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╨╕╨╖ ╨║╤Н╤И╨░
                candles_cache = coins_rsi_data.get('candles_cache', {})
                if symbol in candles_cache:
                    cached_data = candles_cache[symbol]
                    candles = cached_data.get('candles')
                
                # ╨Х╤Б╨╗╨╕ ╨╜╨╡╤В ╨▓ ╨║╤Н╤И╨╡, ╨┐╤А╨╛╨▒╤Г╨╡╨╝ ╨╖╨░╨│╤А╤Г╨╖╨╕╤В╤М
                if not candles:
                    try:
                        candles_data = get_coin_candles_only(symbol, exchange_obj=exchange_obj)
                        if candles_data:
                            candles = candles_data.get('candles')
                    except Exception as candles_error:
                        logger.debug(f" {symbol}: ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨╖╨░╨│╤А╤Г╨╖╨╕╤В╤М ╤Б╨▓╨╡╤З╨╕ ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╤Д╨╕╨╗╤М╤В╤А╨╛╨▓: {candles_error}")
                
                # ╨Х╤Б╨╗╨╕ ╤Б╨▓╨╡╤З╨╕ ╨▓╤Б╨╡ ╨╡╤Й╨╡ ╨╜╨╡╤В, ╨┐╤А╨╛╨▒╤Г╨╡╨╝ ╨╕╨╖ ╨С╨Ф
                if not candles:
                    try:
                        from bot_engine.storage import get_candles_for_symbol
                        db_cached_data = get_candles_for_symbol(symbol)
                        if db_cached_data:
                            candles = db_cached_data.get('candles', [])
                    except Exception as db_error:
                        logger.debug(f" {symbol}: ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨╖╨░╨│╤А╤Г╨╖╨╕╤В╤М ╤Б╨▓╨╡╤З╨╕ ╨╕╨╖ ╨С╨Ф: {db_error}")
                
                # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Д╨╕╨╗╤М╤В╤А╤Л
                if candles and len(candles) >= 10:
                    current_rsi = coin.get('rsi') or coin_data.get('rsi6h')
                    current_trend = coin.get('trend') or coin_data.get('trend6h', 'NEUTRAL')
                    signal = coin['signal']
                    
                    filters_allowed, filters_reason = apply_entry_filters(
                        symbol,
                        candles,
                        current_rsi if current_rsi is not None else 50.0,
                        signal,
                        filter_config,
                        trend=current_trend
                    )
                    
                    if not filters_allowed:
                        logger.warning(f" ЁЯЪл {symbol}: ╨д╨╕╨╗╤М╤В╤А╤Л ╨╖╨░╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨░╨╗╨╕ ╤Б╨╛╨╖╨┤╨░╨╜╨╕╨╡ ╨▒╨╛╤В╨░: {filters_reason}")
                        continue  # ╨Я╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╤Б╨╛╨╖╨┤╨░╨╜╨╕╨╡ ╨▒╨╛╤В╨░
                else:
                    logger.warning(f" тЪая╕П {symbol}: ╨Э╨╡╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ ╤Б╨▓╨╡╤З╨╡╨╣ ╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╤Д╨╕╨╗╤М╤В╤А╨╛╨▓ ({len(candles) if candles else 0}), ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝")
                    continue  # ╨Я╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╤Б╨╛╨╖╨┤╨░╨╜╨╕╨╡ ╨▒╨╛╤В╨░ ╨╡╤Б╨╗╨╕ ╨╜╨╡╤В ╤Б╨▓╨╡╤З╨╡╨╣
                    
            except Exception as filter_check_error:
                logger.error(f" тЭМ {symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╤Д╨╕╨╗╤М╤В╤А╨╛╨▓ ╨┐╨╡╤А╨╡╨┤ ╤Б╨╛╨╖╨┤╨░╨╜╨╕╨╡╨╝ ╨▒╨╛╤В╨░: {filter_check_error}")
                import traceback
                logger.error(traceback.format_exc())
                # тЪая╕П ╨Т╨Р╨Ц╨Э╨Ю: ╨Х╤Б╨╗╨╕ ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╤Д╨╕╨╗╤М╤В╤А╨╛╨▓ ╨╜╨╡ ╤А╨░╨▒╨╛╤В╨░╨╡╤В, ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨г╨Х╨Ь ╤Б╨╛╨╖╨┤╨░╨╜╨╕╨╡ ╨▒╨╛╤В╨░ ╨┤╨╗╤П ╨▒╨╡╨╖╨╛╨┐╨░╤Б╨╜╨╛╤Б╤В╨╕!
                logger.warning(f" ЁЯЪл {symbol}: ╨С╨╗╨╛╨║╨╕╤А╤Г╨╡╨╝ ╤Б╨╛╨╖╨┤╨░╨╜╨╕╨╡ ╨▒╨╛╤В╨░ ╨╕╨╖-╨╖╨░ ╨╛╤И╨╕╨▒╨║╨╕ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╤Д╨╕╨╗╤М╤В╤А╨╛╨▓!")
                continue
            
            # ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╨╜╨╛╨▓╨╛╨│╨╛ ╨▒╨╛╤В╨░ (╤Д╨╕╨╗╤М╤В╤А╤Л ╤Г╨╢╨╡ ╨┐╤А╨╛╨▓╨╡╤А╨╡╨╜╤Л!)
            try:
                logger.info(f" ЁЯЪА ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╨▒╨╛╤В╨░ ╨┤╨╗╤П {symbol} ({coin['signal']}, RSI: {coin['rsi']:.1f})")
                new_bot = create_new_bot(symbol, exchange_obj=exchange_obj)
                
                # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨б╤А╨░╨╖╤Г ╨▓╤Е╨╛╨┤╨╕╨╝ ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╤О!
                signal = coin['signal']
                direction = 'LONG' if signal == 'ENTER_LONG' else 'SHORT'
                logger.info(f" ЁЯУИ ╨Т╤Е╨╛╨┤╨╕╨╝ ╨▓ ╨┐╨╛╨╖╨╕╤Ж╨╕╤О {direction} ╨┤╨╗╤П {symbol}")
                new_bot.enter_position(direction)
                
                created_bots += 1
                
            except Exception as e:
                # ╨С╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨░ ╤Д╨╕╨╗╤М╤В╤А╨░╨╝╨╕ - ╤Н╤В╨╛ ╨╜╨╛╤А╨╝╨░╨╗╤М╨╜╨░╤П ╤А╨░╨▒╨╛╤В╨░ ╤Б╨╕╤Б╤В╨╡╨╝╤Л, ╨╗╨╛╨│╨╕╤А╤Г╨╡╨╝ ╨║╨░╨║ WARNING
                error_str = str(e)
                if '╨╖╨░╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨░╨╜ ╤Д╨╕╨╗╤М╤В╤А╨░╨╝╨╕' in error_str or 'filters_blocked' in error_str:
                    logger.warning(f" тЪая╕П ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╨╖╨┤╨░╨╜╨╕╤П ╨▒╨╛╤В╨░ ╨┤╨╗╤П {symbol}: {e}")
                else:
                    logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╨╖╨┤╨░╨╜╨╕╤П ╨▒╨╛╤В╨░ ╨┤╨╗╤П {symbol}: {e}")
        
        if created_bots > 0:
            logger.info(f" тЬЕ ╨б╨╛╨╖╨┤╨░╨╜╨╛ {created_bots} ╨╜╨╛╨▓╤Л╤Е ╨▒╨╛╤В╨╛╨▓")
        
    except Exception as e:
        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╛╨▒╤А╨░╨▒╨╛╤В╨║╨╕ ╤Б╨╕╨│╨╜╨░╨╗╨╛╨▓: {e}")

def process_trading_signals_for_all_bots(exchange_obj=None):
    """╨Ю╨▒╤А╨░╨▒╨░╤В╤Л╨▓╨░╨╡╤В ╤В╨╛╤А╨│╨╛╨▓╤Л╨╡ ╤Б╨╕╨│╨╜╨░╨╗╤Л ╨┤╨╗╤П ╨▓╤Б╨╡╤Е ╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓ ╤Б ╨╜╨╛╨▓╤Л╨╝ ╨║╨╗╨░╤Б╤Б╨╛╨╝"""
    try:
        logger.info("ЁЯФД ╨Э╨░╤З╨╕╨╜╨░╨╡╨╝ ╨╛╨▒╤А╨░╨▒╨╛╤В╨║╤Г ╤В╨╛╤А╨│╨╛╨▓╤Л╤Е ╤Б╨╕╨│╨╜╨░╨╗╨╛╨▓ ╨┤╨╗╤П ╨▓╤Б╨╡╤Е ╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓...")
        
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░ ╨╗╨╕ ╤Б╨╕╤Б╤В╨╡╨╝╨░
        if not system_initialized:
            logger.warning("тП│ ╨б╨╕╤Б╤В╨╡╨╝╨░ ╨╡╤Й╨╡ ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░ - ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╨╛╨▒╤А╨░╨▒╨╛╤В╨║╤Г")
            return
        
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╤З╤В╨╡╨╜╨╕╨╡ ╤Б╨╗╨╛╨▓╨░╤А╤П - ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
        # ╨д╨╕╨╗╤М╤В╤А╤Г╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓ (╨╕╤Б╨║╨╗╤О╤З╨░╨╡╨╝ IDLE ╨╕ PAUSED)
        active_bots = {symbol: bot for symbol, bot in bots_data['bots'].items() 
                      if bot['status'] not in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']]}
        
        if not active_bots:
            logger.info("тП│ ╨Э╨╡╤В ╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓ ╨┤╨╗╤П ╨╛╨▒╤А╨░╨▒╨╛╤В╨║╨╕")
            return
        
        logger.debug(f"ЁЯФН ╨Ю╨▒╤А╨░╨▒╨░╤В╤Л╨▓╨░╨╡╨╝ {len(active_bots)} ╨░╨║╤В╨╕╨▓╨╜╤Л╤Е ╨▒╨╛╤В╨╛╨▓: {list(active_bots.keys())}")
        
        for symbol, bot_data in active_bots.items():
            try:
                logger.info(f"ЁЯФН ╨Ю╨▒╤А╨░╨▒╨░╤В╤Л╨▓╨░╨╡╨╝ ╨▒╨╛╤В╨░ {symbol} (╤Б╤В╨░╤В╤Г╤Б: {bot_data.get('status')}, ╨┐╨╛╨╖╨╕╤Ж╨╕╤П: {bot_data.get('position_side')})...")
                
                # ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨┐╨╡╤А╨╡╨┤╨░╨╜╨╜╤Г╤О ╨▒╨╕╤А╨╢╤Г ╨╕╨╗╨╕ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Г╤О ╨┐╨╡╤А╨╡╨╝╨╡╨╜╨╜╤Г╤О
                from bots_modules.imports_and_globals import get_exchange
                exchange_to_use = exchange_obj if exchange_obj else get_exchange()
                
                # ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╤Н╨║╨╖╨╡╨╝╨┐╨╗╤П╤А ╨╜╨╛╨▓╨╛╨│╨╛ ╨▒╨╛╤В╨░ ╨╕╨╖ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╜╤Л╤Е ╨┤╨░╨╜╨╜╤Л╤Е
                from bots_modules.bot_class import NewTradingBot
                trading_bot = NewTradingBot(symbol, bot_data, exchange_to_use)
                
                # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ RSI ╨┤╨░╨╜╨╜╤Л╨╡ ╨┤╨╗╤П ╨╝╨╛╨╜╨╡╤В╤Л
                # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╤З╤В╨╡╨╜╨╕╨╡ ╤Б╨╗╨╛╨▓╨░╤А╤П - ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
                rsi_data = coins_rsi_data['coins'].get(symbol)
                
                if not rsi_data:
                    logger.warning(f"тЭМ {symbol}: RSI ╨┤╨░╨╜╨╜╤Л╨╡ ╨╜╨╡ ╨╜╨░╨╣╨┤╨╡╨╜╤Л - ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╨┐╤А╨╛╨▓╨╡╤А╨║╤Г")
                    continue
                
                current_rsi = rsi_data.get('rsi6h')
                current_trend = rsi_data.get('trend6h')
                logger.info(f"тЬЕ {symbol}: RSI={current_rsi}, Trend={current_trend}, ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Г╤Б╨╗╨╛╨▓╨╕╤П ╨╖╨░╨║╤А╤Л╤В╨╕╤П...")
                
                # ╨Ю╨▒╤А╨░╨▒╨░╤В╤Л╨▓╨░╨╡╨╝ ╤В╨╛╤А╨│╨╛╨▓╤Л╨╡ ╤Б╨╕╨│╨╜╨░╨╗╤Л ╤З╨╡╤А╨╡╨╖ ╨╝╨╡╤В╨╛╨┤ update
                external_signal = rsi_data.get('signal')
                external_trend = rsi_data.get('trend6h')
                
                signal_result = trading_bot.update(
                    force_analysis=True, 
                    external_signal=external_signal, 
                    external_trend=external_trend
                )
                
                logger.debug(f"ЁЯФД {symbol}: ╨а╨╡╨╖╤Г╨╗╤М╤В╨░╤В update: {signal_result}")
                
                # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╨┤╨░╨╜╨╜╤Л╨╡ ╨▒╨╛╤В╨░ ╨▓ ╤Е╤А╨░╨╜╨╕╨╗╨╕╤Й╨╡ ╨╡╤Б╨╗╨╕ ╨╡╤Б╤В╤М ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П
                if signal_result and signal_result.get('success', False):
                    # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╨┐╤А╨╕╤Б╨▓╨░╨╕╨▓╨░╨╜╨╕╨╡ - ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
                    bots_data['bots'][symbol] = trading_bot.to_dict()
                    
                    # ╨Ы╨╛╨│╨╕╤А╤Г╨╡╨╝ ╤В╨╛╤А╨│╨╛╨▓╤Л╨╡ ╨┤╨╡╨╣╤Б╤В╨▓╨╕╤П
                    action = signal_result.get('action')
                    if action in ['OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT']:
                        logger.info(f"ЁЯОп {symbol}: {action} ╨▓╤Л╨┐╨╛╨╗╨╜╨╡╨╜╨╛")
                else:
                    logger.debug(f"тП│ {symbol}: ╨Э╨╡╤В ╤В╨╛╤А╨│╨╛╨▓╤Л╤Е ╤Б╨╕╨│╨╜╨░╨╗╨╛╨▓")
        
            except Exception as e:
                logger.error(f"тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╛╨▒╤А╨░╨▒╨╛╤В╨║╨╕ ╤Б╨╕╨│╨╜╨░╨╗╨╛╨▓ ╨┤╨╗╤П {symbol}: {e}")
        
    except Exception as e:
        logger.error(f"тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╛╨▒╤А╨░╨▒╨╛╤В╨║╨╕ ╤В╨╛╤А╨│╨╛╨▓╤Л╤Е ╤Б╨╕╨│╨╜╨░╨╗╨╛╨▓: {str(e)}")

def check_new_autobot_filters(symbol, signal, coin_data):
    """╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╤В ╤Д╨╕╨╗╤М╤В╤А╤Л ╨┤╨╗╤П ╨╜╨╛╨▓╨╛╨│╨╛ ╨░╨▓╤В╨╛╨▒╨╛╤В╨░"""
    try:
        # тЬЕ ╨Т╨б╨Х ╨д╨Ш╨Ы╨м╨в╨а╨л ╨г╨Ц╨Х ╨Я╨а╨Ю╨Т╨Х╨а╨Х╨Э╨л ╨▓ get_coin_rsi_data():
        # 1. Whitelist/blacklist/scope
        # 2. ╨С╨░╨╖╨╛╨▓╤Л╨╣ RSI + ╨в╤А╨╡╨╜╨┤
        # 3. ╨б╤Г╤Й╨╡╤Б╤В╨▓╤Г╤О╤Й╨╕╨╡ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ (╨а╨Р╨Э╨Э╨Ш╨Щ ╨▓╤Л╤Е╨╛╨┤!)
        # 4. Enhanced RSI
        # 5. ╨Ч╤А╨╡╨╗╨╛╤Б╤В╤М ╨╝╨╛╨╜╨╡╤В╤Л
        # 6. ExitScam ╤Д╨╕╨╗╤М╤В╤А
        # 7. RSI ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╣ ╤Д╨╕╨╗╤М╤В╤А
        
        # ╨Ч╨┤╨╡╤Б╤М ╨┤╨╡╨╗╨░╨╡╨╝ ╤В╨╛╨╗╤М╨║╨╛ ╨┤╤Г╨▒╨╗╤М-╨┐╤А╨╛╨▓╨╡╤А╨║╤Г ╨╖╤А╨╡╨╗╨╛╤Б╤В╨╕ ╨╕ ExitScam ╨╜╨░ ╨▓╤Б╤П╨║╨╕╨╣ ╤Б╨╗╤Г╤З╨░╨╣
        
        # ╨Ф╤Г╨▒╨╗╤М-╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╨╖╤А╨╡╨╗╨╛╤Б╤В╨╕ ╨╝╨╛╨╜╨╡╤В╤Л
        if not check_coin_maturity_stored_or_verify(symbol):
            return False
        
        # ╨Ф╤Г╨▒╨╗╤М-╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ExitScam
        if not check_exit_scam_filter(symbol, coin_data):
            logger.warning(f" {symbol}: тЭМ ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Р: ╨Ю╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜╤Л ╤А╨╡╨╖╨║╨╕╨╡ ╨┤╨▓╨╕╨╢╨╡╨╜╨╕╤П ╤Ж╨╡╨╜╤Л (ExitScam)")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f" {symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╤Д╨╕╨╗╤М╤В╤А╨╛╨▓: {e}")
        return False

def analyze_trends_for_signal_coins():
    """ЁЯОп ╨Ю╨┐╤А╨╡╨┤╨╡╨╗╤П╨╡╤В ╤В╤А╨╡╨╜╨┤ ╨┤╨╗╤П ╨╝╨╛╨╜╨╡╤В ╤Б ╤Б╨╕╨│╨╜╨░╨╗╨░╨╝╨╕ (RSI тЙд29 ╨╕╨╗╨╕ тЙе71)"""
    try:
        from bots_modules.imports_and_globals import rsi_data_lock, coins_rsi_data, get_exchange, get_auto_bot_config
        
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Д╨╗╨░╨│ trend_detection_enabled
        config = get_auto_bot_config()
        trend_detection_enabled = config.get('trend_detection_enabled', True)
        
        if not trend_detection_enabled:
            logger.info(" тП╕я╕П ╨Р╨╜╨░╨╗╨╕╨╖ ╤В╤А╨╡╨╜╨┤╨╛╨▓ ╨╛╤В╨║╨╗╤О╤З╨╡╨╜ (trend_detection_enabled=False)")
            return False
        
        logger.info(" ЁЯОп ╨Э╨░╤З╨╕╨╜╨░╨╡╨╝ ╨░╨╜╨░╨╗╨╕╨╖ ╤В╤А╨╡╨╜╨┤╨╛╨▓ ╨┤╨╗╤П ╤Б╨╕╨│╨╜╨░╨╗╤М╨╜╤Л╤Е ╨╝╨╛╨╜╨╡╤В...")
        from bots_modules.calculations import analyze_trend_6h
        
        exchange = get_exchange()
        if not exchange:
            logger.error(" тЭМ ╨С╨╕╤А╨╢╨░ ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░")
            return False
        
        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╨Т╨а╨Х╨Ь╨Х╨Э╨Э╨Ю╨Х ╤Е╤А╨░╨╜╨╕╨╗╨╕╤Й╨╡ ╨┤╨╗╤П ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╣
        # ╨Э╨╡ ╨╕╨╖╨╝╨╡╨╜╤П╨╡╨╝ coins_rsi_data ╨┤╨╛ ╨╖╨░╨▓╨╡╤А╤И╨╡╨╜╨╕╤П ╨▓╤Б╨╡╤Е ╤А╨░╤Б╤З╨╡╤В╨╛╨▓!
        temp_updates = {}
        
        # ╨Э╨░╤Е╨╛╨┤╨╕╨╝ ╨╝╨╛╨╜╨╡╤В╤Л ╤Б ╤Б╨╕╨│╨╜╨░╨╗╨░╨╝╨╕ ╨┤╨╗╤П ╨░╨╜╨░╨╗╨╕╨╖╨░ ╤В╤А╨╡╨╜╨┤╨░
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╤З╤В╨╡╨╜╨╕╨╡ ╤Б╨╗╨╛╨▓╨░╤А╤П - ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
        signal_coins = []
        for symbol, coin_data in coins_rsi_data['coins'].items():
            rsi = coin_data.get('rsi6h')
            if rsi is not None and (rsi <= 29 or rsi >= 71):
                signal_coins.append(symbol)
        
        logger.info(f" ЁЯУК ╨Э╨░╨╣╨┤╨╡╨╜╨╛ {len(signal_coins)} ╤Б╨╕╨│╨╜╨░╨╗╤М╨╜╤Л╤Е ╨╝╨╛╨╜╨╡╤В ╨┤╨╗╤П ╨░╨╜╨░╨╗╨╕╨╖╨░ ╤В╤А╨╡╨╜╨┤╨░")
        
        if not signal_coins:
            logger.warning(" тЪая╕П ╨Э╨╡╤В ╤Б╨╕╨│╨╜╨░╨╗╤М╨╜╤Л╤Е ╨╝╨╛╨╜╨╡╤В ╨┤╨╗╤П ╨░╨╜╨░╨╗╨╕╨╖╨░ ╤В╤А╨╡╨╜╨┤╨░")
            return False
        
        # ╨Р╨╜╨░╨╗╨╕╨╖╨╕╤А╤Г╨╡╨╝ ╤В╤А╨╡╨╜╨┤ ╨┤╨╗╤П ╨║╨░╨╢╨┤╨╛╨╣ ╤Б╨╕╨│╨╜╨░╨╗╤М╨╜╨╛╨╣ ╨╝╨╛╨╜╨╡╤В╤Л
        analyzed_count = 0
        failed_count = 0
        
        for i, symbol in enumerate(signal_coins, 1):
            try:
                # ╨Р╨╜╨░╨╗╨╕╨╖╨╕╤А╤Г╨╡╨╝ ╤В╤А╨╡╨╜╨┤
                trend_analysis = analyze_trend_6h(symbol, exchange_obj=exchange)
                
                if trend_analysis:
                    # тЬЕ ╨б╨Ю╨С╨Ш╨а╨Р╨Х╨Ь ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П ╨▓╨╛ ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╝ ╤Е╤А╨░╨╜╨╕╨╗╨╕╤Й╨╡
                    if symbol in coins_rsi_data['coins']:
                        coin_data = coins_rsi_data['coins'][symbol]
                        rsi = coin_data.get('rsi6h')
                        new_trend = trend_analysis['trend']
                        
                        # ╨Я╨╡╤А╨╡╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╨╝ ╤Б╨╕╨│╨╜╨░╨╗ ╤Б ╤Г╤З╨╡╤В╨╛╨╝ ╨╜╨╛╨▓╨╛╨│╨╛ ╤В╤А╨╡╨╜╨┤╨░
                        old_signal = coin_data.get('signal')
                        
                        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨Э╨Х ╨┐╨╡╤А╨╡╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╨╝ ╤Б╨╕╨│╨╜╨░╨╗ ╨╡╤Б╨╗╨╕ ╨╛╨╜ WAIT ╨╕╨╖-╨╖╨░ ╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨║╨╕ ╤Д╨╕╨╗╤М╤В╤А╨╛╨▓!
                        blocked_by_exit_scam = coin_data.get('blocked_by_exit_scam', False)
                        blocked_by_rsi_time = coin_data.get('blocked_by_rsi_time', False)
                        
                        if blocked_by_exit_scam or blocked_by_rsi_time:
                            new_signal = 'WAIT'  # ╨Ю╤Б╤В╨░╨▓╨╗╤П╨╡╨╝ WAIT
                        else:
                            new_signal = _recalculate_signal_with_trend(rsi, new_trend, symbol)
                        
                        # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П ╨▓╨╛ ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╝ ╤Е╤А╨░╨╜╨╕╨╗╨╕╤Й╨╡
                        temp_updates[symbol] = {
                            'trend6h': new_trend,
                            'trend_analysis': trend_analysis,
                            'signal': new_signal,
                            'old_signal': old_signal
                        }
                    
                    analyzed_count += 1
                else:
                    failed_count += 1
                
                # ╨Т╤Л╨▓╨╛╨┤╨╕╨╝ ╨┐╤А╨╛╨│╤А╨╡╤Б╤Б ╨║╨░╨╢╨┤╤Л╨╡ 5 ╨╝╨╛╨╜╨╡╤В
                if i % 5 == 0 or i == len(signal_coins):
                    logger.info(f" ЁЯУК ╨Я╤А╨╛╨│╤А╨╡╤Б╤Б: {i}/{len(signal_coins)} ({i*100//len(signal_coins)}%)")
                
                # ╨Э╨╡╨▒╨╛╨╗╤М╤И╨░╤П ╨┐╨░╤Г╨╖╨░ ╨╝╨╡╨╢╨┤╤Г ╨╖╨░╨┐╤А╨╛╤Б╨░╨╝╨╕
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f" тЭМ {symbol}: {e}")
                failed_count += 1
        
        # тЬЕ ╨Р╨в╨Ю╨Ь╨Р╨а╨Э╨Ю ╨┐╤А╨╕╨╝╨╡╨╜╤П╨╡╨╝ ╨Т╨б╨Х ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П ╨╛╨┤╨╜╨╕╨╝ ╨╝╨░╤Е╨╛╨╝!
        for symbol, updates in temp_updates.items():
            coins_rsi_data['coins'][symbol]['trend6h'] = updates['trend6h']
            coins_rsi_data['coins'][symbol]['trend_analysis'] = updates['trend_analysis']
            coins_rsi_data['coins'][symbol]['signal'] = updates['signal']
        
        logger.info(f" тЬЕ {analyzed_count} ╨┐╤А╨╛╨░╨╜╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨╛ | {len(temp_updates)} ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╨╣")
        
        return True
        
    except Exception as e:
        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨░╨╜╨░╨╗╨╕╨╖╨░ ╤В╤А╨╡╨╜╨┤╨╛╨▓: {e}")
        return False

def process_long_short_coins_with_filters():
    """ЁЯФН ╨Ю╨▒╤А╨░╨▒╨░╤В╤Л╨▓╨░╨╡╤В ╨╗╨╛╨╜╨│/╤И╨╛╤А╤В ╨╝╨╛╨╜╨╡╤В╤Л ╨▓╤Б╨╡╨╝╨╕ ╤Д╨╕╨╗╤М╤В╤А╨░╨╝╨╕"""
    try:
        logger.info(" ЁЯФН ╨Э╨░╤З╨╕╨╜╨░╨╡╨╝ ╨╛╨▒╤А╨░╨▒╨╛╤В╨║╤Г ╨╗╨╛╨╜╨│/╤И╨╛╤А╤В ╨╝╨╛╨╜╨╡╤В ╤Д╨╕╨╗╤М╤В╤А╨░╨╝╨╕...")
        
        from bots_modules.imports_and_globals import rsi_data_lock, coins_rsi_data
        
        # ╨Э╨░╤Е╨╛╨┤╨╕╨╝ ╨╝╨╛╨╜╨╡╤В╤Л ╤Б ╤Б╨╕╨│╨╜╨░╨╗╨░╨╝╨╕ ╨╗╨╛╨╜╨│/╤И╨╛╤А╤В
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╤З╤В╨╡╨╜╨╕╨╡ ╤Б╨╗╨╛╨▓╨░╤А╤П - ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
        long_short_coins = []
        for symbol, coin_data in coins_rsi_data['coins'].items():
            signal = coin_data.get('signal', 'WAIT')
            if signal in ['ENTER_LONG', 'ENTER_SHORT']:
                long_short_coins.append(symbol)
        
        logger.info(f" ЁЯУК ╨Э╨░╨╣╨┤╨╡╨╜╨╛ {len(long_short_coins)} ╨╗╨╛╨╜╨│/╤И╨╛╤А╤В ╨╝╨╛╨╜╨╡╤В ╨┤╨╗╤П ╨╛╨▒╤А╨░╨▒╨╛╤В╨║╨╕")
        
        if not long_short_coins:
            logger.warning(" тЪая╕П ╨Э╨╡╤В ╨╗╨╛╨╜╨│/╤И╨╛╤А╤В ╨╝╨╛╨╜╨╡╤В ╨┤╨╗╤П ╨╛╨▒╤А╨░╨▒╨╛╤В╨║╨╕")
            return []
        
        # ╨Ю╨▒╤А╨░╨▒╨░╤В╤Л╨▓╨░╨╡╨╝ ╨║╨░╨╢╨┤╤Г╤О ╨╝╨╛╨╜╨╡╤В╤Г ╨▓╤Б╨╡╨╝╨╕ ╤Д╨╕╨╗╤М╤В╤А╨░╨╝╨╕
        filtered_coins = []
        blocked_count = 0
        
        for i, symbol in enumerate(long_short_coins, 1):
            try:
                # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨┤╨░╨╜╨╜╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л
                # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╤З╤В╨╡╨╜╨╕╨╡ ╤Б╨╗╨╛╨▓╨░╤А╤П - ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
                coin_data = coins_rsi_data['coins'].get(symbol, {})
                
                if not coin_data:
                    logger.warning(f" тЪая╕П {symbol}: ╨Э╨╡╤В ╨┤╨░╨╜╨╜╤Л╤Е")
                    blocked_count += 1
                    continue
                
                # ╨Я╤А╨╕╨╝╨╡╨╜╤П╨╡╨╝ ╨▓╤Б╨╡ ╤Д╨╕╨╗╤М╤В╤А╤Л
                signal = coin_data.get('signal', 'WAIT')
                passes_filters = check_new_autobot_filters(symbol, signal, coin_data)
                
                if passes_filters:
                    filtered_coins.append(symbol)
                else:
                    blocked_count += 1
                
            except Exception as e:
                logger.error(f" тЭМ {symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨╛╨▒╤А╨░╨▒╨╛╤В╨║╨╕ ╤Д╨╕╨╗╤М╤В╤А╨░╨╝╨╕: {e}")
                blocked_count += 1
        
        logger.info(f" тЬЕ ╨Ю╨▒╤А╨░╨▒╨╛╤В╨║╨░ ╤Д╨╕╨╗╤М╤В╤А╨░╨╝╨╕ ╨╖╨░╨▓╨╡╤А╤И╨╡╨╜╨░:")
        logger.info(f" ЁЯУК ╨Я╤А╨╛╤И╨╗╨╕ ╤Д╨╕╨╗╤М╤В╤А╤Л: {len(filtered_coins)}")
        logger.info(f" ЁЯУК ╨Ч╨░╨▒╨╗╨╛╨║╨╕╤А╨╛╨▓╨░╨╜╤Л: {blocked_count}")
        logger.info(f" ЁЯУК ╨Т╤Б╨╡╨│╨╛ ╨╛╨▒╤А╨░╨▒╨╛╤В╨░╨╜╨╛: {len(filtered_coins) + blocked_count}")
        
        return filtered_coins
        
    except Exception as e:
        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╛╨▒╤А╨░╨▒╨╛╤В╨║╨╕ ╤Д╨╕╨╗╤М╤В╤А╨░╨╝╨╕: {e}")
        return []

def set_filtered_coins_for_autobot(filtered_coins):
    """тЬЕ ╨Я╨╡╤А╨╡╨┤╨░╨╡╤В ╨╛╤В╤Д╨╕╨╗╤М╤В╤А╨╛╨▓╨░╨╜╨╜╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л ╨░╨▓╤В╨╛╨▒╨╛╤В╤Г ╨╕ ╨б╨а╨Р╨Ч╨г ╨╖╨░╨┐╤Г╤Б╨║╨░╨╡╤В ╨┐╤А╨╛╨▓╨╡╤А╨║╤Г ╤Б╨╕╨│╨╜╨░╨╗╨╛╨▓"""
    try:
        logger.info(f" тЬЕ ╨Я╨╡╤А╨╡╨┤╨░╨╡╨╝ {len(filtered_coins)} ╨╛╤В╤Д╨╕╨╗╤М╤В╤А╨╛╨▓╨░╨╜╨╜╤Л╤Е ╨╝╨╛╨╜╨╡╤В ╨░╨▓╤В╨╛╨▒╨╛╤В╤Г...")
        
        from bots_modules.imports_and_globals import bots_data_lock, bots_data
        
        # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨╛╤В╤Д╨╕╨╗╤М╤В╤А╨╛╨▓╨░╨╜╨╜╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л ╨▓ ╨║╨╛╨╜╤Д╨╕╨│ ╨░╨▓╤В╨╛╨▒╨╛╤В╨░
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╨┐╤А╨╕╤Б╨▓╨░╨╕╨▓╨░╨╜╨╕╨╡ - ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
        if 'auto_bot_config' not in bots_data:
            bots_data['auto_bot_config'] = {}
        
        bots_data['auto_bot_config']['filtered_coins'] = filtered_coins
        bots_data['auto_bot_config']['last_filter_update'] = datetime.now().isoformat()
        
        logger.info(f" тЬЕ ╨Ю╤В╤Д╨╕╨╗╤М╤В╤А╨╛╨▓╨░╨╜╨╜╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╤Л ╨▓ ╨║╨╛╨╜╤Д╨╕╨│ ╨░╨▓╤В╨╛╨▒╨╛╤В╨░")
        logger.info(f" ЁЯУК ╨Ь╨╛╨╜╨╡╤В╤Л ╨┤╨╗╤П ╨░╨▓╤В╨╛╨▒╨╛╤В╨░: {', '.join(filtered_coins[:10])}{'...' if len(filtered_coins) > 10 else ''}")
        
        # тЬЕ ╨Ъ╨а╨Ш╨в╨Ш╨з╨Э╨Ю: ╨б╨а╨Р╨Ч╨г ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Б╨╕╨│╨╜╨░╨╗╤Л ╨╕ ╤Б╨╛╨╖╨┤╨░╨╡╨╝ ╨▒╨╛╤В╨╛╨▓ ╨▒╨╡╨╖ ╨╖╨░╨┤╨╡╤А╨╢╨║╨╕!
        # ╨Э╨╡ ╨╢╨┤╨╡╨╝ ╤Б╨╗╨╡╨┤╤Г╤О╤Й╨╡╨│╨╛ ╤Ж╨╕╨║╨╗╨░ ╨▓╨╛╤А╨║╨╡╤А╨░ (180 ╤Б╨╡╨║╤Г╨╜╨┤) - ╨╛╨▒╤А╨░╨▒╨░╤В╤Л╨▓╨░╨╡╨╝ ╨╜╨╡╨╝╨╡╨┤╨╗╨╡╨╜╨╜╨╛!
        if filtered_coins and bots_data.get('auto_bot_config', {}).get('enabled', False):
            logger.info(f" ЁЯЪА ╨Э╨╡╨╝╨╡╨┤╨╗╨╡╨╜╨╜╨╛ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Б╨╕╨│╨╜╨░╨╗╤Л ╨┤╨╗╤П {len(filtered_coins)} ╨╝╨╛╨╜╨╡╤В...")
            try:
                from bots_modules.imports_and_globals import get_exchange
                exchange_obj = get_exchange()
                process_auto_bot_signals(exchange_obj=exchange_obj)
            except Exception as e:
                logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╜╨╡╨╝╨╡╨┤╨╗╨╡╨╜╨╜╨╛╨╣ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╤Б╨╕╨│╨╜╨░╨╗╨╛╨▓: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f" тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╨╡╤А╨╡╨┤╨░╤З╨╕ ╨╝╨╛╨╜╨╡╤В ╨░╨▓╤В╨╛╨▒╨╛╤В╤Г: {e}")
        return False

def check_coin_maturity_stored_or_verify(symbol):
    """╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╤В ╨╖╤А╨╡╨╗╨╛╤Б╤В╤М ╨╝╨╛╨╜╨╡╤В╤Л ╨╕╨╖ ╤Е╤А╨░╨╜╨╕╨╗╨╕╤Й╨░ ╨╕╨╗╨╕ ╨▓╤Л╨┐╨╛╨╗╨╜╤П╨╡╤В ╨┐╤А╨╛╨▓╨╡╤А╨║╤Г"""
    try:
        # ╨б╨╜╨░╤З╨░╨╗╨░ ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Е╤А╨░╨╜╨╕╨╗╨╕╤Й╨╡
        if is_coin_mature_stored(symbol):
            return True
        
        # ╨Х╤Б╨╗╨╕ ╨╜╨╡╤В ╨▓ ╤Е╤А╨░╨╜╨╕╨╗╨╕╤Й╨╡, ╨▓╤Л╨┐╨╛╨╗╨╜╤П╨╡╨╝ ╨┐╤А╨╛╨▓╨╡╤А╨║╤Г
        exch = get_exchange()
        if not exch:
            logger.warning(f"{symbol}: ╨С╨╕╤А╨╢╨░ ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░")
            return False
        
        chart_response = exch.get_chart_data(symbol, '6h', '30d')
        if not chart_response or not chart_response.get('success'):
            logger.warning(f"{symbol}: ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╤Б╨▓╨╡╤З╨╕")
            return False
        
        candles = chart_response.get('data', {}).get('candles', [])
        if not candles:
            logger.warning(f"{symbol}: ╨Э╨╡╤В ╤Б╨▓╨╡╤З╨╡╨╣")
            return False
        
        maturity_result = check_coin_maturity_with_storage(symbol, candles)
        return maturity_result['is_mature']
        
    except Exception as e:
        logger.error(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨╖╤А╨╡╨╗╨╛╤Б╤В╨╕: {e}")
        return False

def update_is_mature_flags_in_rsi_data():
    """╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╤В ╤Д╨╗╨░╨│╨╕ is_mature ╨▓ ╨║╤Н╤И╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╤Е ╨┤╨░╨╜╨╜╤Л╤Е RSI ╨╜╨░ ╨╛╤Б╨╜╨╛╨▓╨╡ ╤Е╤А╨░╨╜╨╕╨╗╨╕╤Й╨░ ╨╖╤А╨╡╨╗╤Л╤Е ╨╝╨╛╨╜╨╡╤В"""
    try:
        from bots_modules.imports_and_globals import is_coin_mature_stored
        
        updated_count = 0
        total_count = len(coins_rsi_data['coins'])
        
        # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╤Д╨╗╨░╨│╨╕ is_mature ╨┤╨╗╤П ╨▓╤Б╨╡╤Е ╨╝╨╛╨╜╨╡╤В ╨▓ RSI ╨┤╨░╨╜╨╜╤Л╤Е
        for symbol, coin_data in coins_rsi_data['coins'].items():
            # ╨Ю╨▒╨╜╨╛╨▓╨╗╤П╨╡╨╝ ╤Д╨╗╨░╨│ is_mature ╨╜╨░ ╨╛╤Б╨╜╨╛╨▓╨╡ ╤Е╤А╨░╨╜╨╕╨╗╨╕╤Й╨░
            old_status = coin_data.get('is_mature', False)
            coin_data['is_mature'] = is_coin_mature_stored(symbol)
            
            # ╨Я╨╛╨┤╤Б╤З╨╕╤В╤Л╨▓╨░╨╡╨╝ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╜╤Л╨╡
            if coin_data['is_mature']:
                updated_count += 1
        
        logger.info(f"тЬЕ ╨Ю╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╛ ╤Д╨╗╨░╨│╨╛╨▓: {updated_count} ╨╖╤А╨╡╨╗╤Л╤Е ╨╕╨╖ {total_count} ╨╝╨╛╨╜╨╡╤В")
        
    except Exception as e:
        logger.error(f"тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╨╛╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П ╤Д╨╗╨░╨│╨╛╨▓: {e}")

def _legacy_check_exit_scam_filter(symbol, coin_data, individual_settings=None):
    """
    EXIT SCAM ╨д╨Ш╨Ы╨м╨в╨а + AI ANOMALY DETECTION
    
    ╨Ч╨░╤Й╨╕╤В╨░ ╨╛╤В ╤А╨╡╨╖╨║╨╕╤Е ╨┤╨▓╨╕╨╢╨╡╨╜╨╕╨╣ ╤Ж╨╡╨╜╤Л (╨┐╨░╨╝╨┐/╨┤╨░╨╝╨┐/╤Б╨║╨░╨╝):
    1. ╨Ю╨┤╨╜╨░ ╤Б╨▓╨╡╤З╨░ ╨┐╤А╨╡╨▓╤Л╤Б╨╕╨╗╨░ ╨╝╨░╨║╤Б╨╕╨╝╨░╨╗╤М╨╜╤Л╨╣ % ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П
    2. N ╤Б╨▓╨╡╤З╨╡╨╣ ╤Б╤Г╨╝╨╝╨░╤А╨╜╨╛ ╨┐╤А╨╡╨▓╤Л╤Б╨╕╨╗╨╕ ╨╝╨░╨║╤Б╨╕╨╝╨░╨╗╤М╨╜╤Л╨╣ % ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П
    3. ╨Ш╨Ш ╨╛╨▒╨╜╨░╤А╤Г╨╢╨╕╨╗ ╨░╨╜╨╛╨╝╨░╨╗╨╕╤О (╨╡╤Б╨╗╨╕ ╨▓╨║╨╗╤О╤З╨╡╨╜)
    
    Args:
        symbol: ╨б╨╕╨╝╨▓╨╛╨╗ ╨╝╨╛╨╜╨╡╤В╤Л
        coin_data: ╨Ф╨░╨╜╨╜╤Л╨╡ ╨╝╨╛╨╜╨╡╤В╤Л
        individual_settings: ╨Ш╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╨╝╨╛╨╜╨╡╤В╤Л (╨╛╨┐╤Ж╨╕╨╛╨╜╨░╨╗╤М╨╜╨╛)
    """
    try:
        # тЬЕ ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕: ╤Б╨╜╨░╤З╨░╨╗╨░ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡, ╨╖╨░╤В╨╡╨╝ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╤Л╨╡
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╨║╨╛╨╜╤Д╨╕╨│ ╨╜╨╡ ╨╝╨╡╨╜╤П╨╡╤В╤Б╤П, GIL ╨┤╨╡╨╗╨░╨╡╤В ╤З╤В╨╡╨╜╨╕╨╡ ╨░╤В╨╛╨╝╨░╤А╨╜╤Л╨╝
        if individual_settings is None:
            individual_settings = get_individual_coin_settings(symbol)
        
        auto_config = bots_data.get('auto_bot_config', {})
        
        exit_scam_enabled = individual_settings.get('exit_scam_enabled') if individual_settings else None
        if exit_scam_enabled is None:
            exit_scam_enabled = auto_config.get('exit_scam_enabled', True)
        
        exit_scam_candles = individual_settings.get('exit_scam_candles') if individual_settings else None
        if exit_scam_candles is None:
            exit_scam_candles = auto_config.get('exit_scam_candles', 10)
        
        single_candle_percent = individual_settings.get('exit_scam_single_candle_percent') if individual_settings else None
        if single_candle_percent is None:
            single_candle_percent = auto_config.get('exit_scam_single_candle_percent', 15.0)
        
        multi_candle_count = individual_settings.get('exit_scam_multi_candle_count') if individual_settings else None
        if multi_candle_count is None:
            multi_candle_count = auto_config.get('exit_scam_multi_candle_count', 4)
        
        multi_candle_percent = individual_settings.get('exit_scam_multi_candle_percent') if individual_settings else None
        if multi_candle_percent is None:
            multi_candle_percent = auto_config.get('exit_scam_multi_candle_percent', 50.0)
        
        # ╨Х╤Б╨╗╨╕ ╤Д╨╕╨╗╤М╤В╤А ╨╛╤В╨║╨╗╤О╤З╨╡╨╜ - ╤А╨░╨╖╤А╨╡╤И╨░╨╡╨╝
        if not exit_scam_enabled:
            logger.debug(f"{symbol}: ╨д╨╕╨╗╤М╤В╤А ╨╛╤В╨║╨╗╤О╤З╨╡╨╜")
            return True
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╤Б╨▓╨╡╤З╨╕
        exch = get_exchange()
        if not exch:
            return False
        
        chart_response = exch.get_chart_data(symbol, '6h', '30d')
        if not chart_response or not chart_response.get('success'):
            return False
        
        candles = chart_response.get('data', {}).get('candles', [])
        if len(candles) < exit_scam_candles:
            return False
        
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╤Б╨▓╨╡╤З╨╡╨╣ (╨╕╨╖ ╨║╨╛╨╜╤Д╨╕╨│╨░)
        recent_candles = candles[-exit_scam_candles:]
        
        logger.debug(f"{symbol}: ╨Р╨╜╨░╨╗╨╕╨╖ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╤Е {exit_scam_candles} ╤Б╨▓╨╡╤З╨╡╨╣")
        
        # 1. ╨Я╨а╨Ю╨Т╨Х╨а╨Ъ╨Р: ╨Ю╨┤╨╜╨░ ╤Б╨▓╨╡╤З╨░ ╨┐╤А╨╡╨▓╤Л╤Б╨╕╨╗╨░ ╨╝╨░╨║╤Б╨╕╨╝╨░╨╗╤М╨╜╤Л╨╣ % ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П
        for i, candle in enumerate(recent_candles):
            open_price = candle['open']
            close_price = candle['close']
            
            # ╨Я╤А╨╛╤Ж╨╡╨╜╤В ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П ╤Б╨▓╨╡╤З╨╕ (╨╛╤В ╨╛╤В╨║╤А╤Л╤В╨╕╤П ╨┤╨╛ ╨╖╨░╨║╤А╤Л╤В╨╕╤П)
            price_change = abs((close_price - open_price) / open_price) * 100
            
            if price_change > single_candle_percent:
                logger.debug(f"{symbol}: ╨б╨▓╨╡╤З╨░: O={open_price:.4f} C={close_price:.4f} H={candle['high']:.4f} L={candle['low']:.4f}")
                return False
        
        # 2. ╨Я╨а╨Ю╨Т╨Х╨а╨Ъ╨Р: N ╤Б╨▓╨╡╤З╨╡╨╣ ╤Б╤Г╨╝╨╝╨░╤А╨╜╨╛ ╨┐╤А╨╡╨▓╤Л╤Б╨╕╨╗╨╕ ╨╝╨░╨║╤Б╨╕╨╝╨░╨╗╤М╨╜╤Л╨╣ % ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П
        if len(recent_candles) >= multi_candle_count:
            # ╨С╨╡╤А╨╡╨╝ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╤Б╨▓╨╡╤З╨╡╨╣ ╨┤╨╗╤П ╤Б╤Г╨╝╨╝╨░╤А╨╜╨╛╨│╨╛ ╨░╨╜╨░╨╗╨╕╨╖╨░
            multi_candles = recent_candles[-multi_candle_count:]
            
            first_open = multi_candles[0]['open']
            last_close = multi_candles[-1]['close']
            
            # ╨б╤Г╨╝╨╝╨░╤А╨╜╨╛╨╡ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╨╡ ╨╛╤В ╨┐╨╡╤А╨▓╨╛╨╣ ╤Б╨▓╨╡╤З╨╕ ╨┤╨╛ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╡╨╣
            total_change = abs((last_close - first_open) / first_open) * 100
            
            if total_change > multi_candle_percent:
                logger.warning(f"{symbol}: тЭМ ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Р: {multi_candle_count} ╤Б╨▓╨╡╤З╨╡╨╣ ╨┐╤А╨╡╨▓╤Л╤Б╨╕╨╗╨╕ ╤Б╤Г╨╝╨╝╨░╤А╨╜╤Л╨╣ ╨╗╨╕╨╝╨╕╤В {multi_candle_percent}% (╨▒╤Л╨╗╨╛ {total_change:.1f}%)")
                logger.debug(f"{symbol}: ╨Я╨╡╤А╨▓╨░╤П ╤Б╨▓╨╡╤З╨░: {first_open:.4f}, ╨Я╨╛╤Б╨╗╨╡╨┤╨╜╤П╤П ╤Б╨▓╨╡╤З╨░: {last_close:.4f}")
                return False
        
        logger.debug(f"{symbol}: тЬЕ ╨С╨░╨╖╨╛╨▓╤Л╨╡ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨┐╤А╨╛╨╣╨┤╨╡╨╜╤Л")
        
        # 3. ╨Я╨а╨Ю╨Т╨Х╨а╨Ъ╨Р: AI Anomaly Detection (╨╡╤Б╨╗╨╕ ╨▓╨║╨╗╤О╤З╨╡╨╜)
        ai_check_enabled = True  # ╨Т╨║╨╗╤О╤З╨░╨╡╨╝ ╨╛╨▒╤А╨░╤В╨╜╨╛ - ╨┐╤А╨╛╨▒╨╗╨╡╨╝╨░ ╨▒╤Л╨╗╨░ ╨╜╨╡ ╨▓ AI!
        
        if ai_check_enabled:
            try:
                from bot_engine.bot_config import AIConfig
                
                # ╨С╤Л╤Б╤В╤А╨░╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨░: AI ╨▓╨║╨╗╤О╤З╨╡╨╜ ╨╕ Anomaly Detection ╨▓╨║╨╗╤О╤З╨╡╨╜
                if AIConfig.AI_ENABLED and AIConfig.AI_ANOMALY_DETECTION_ENABLED:
                    try:
                        # тЬЕ ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╖╨░╨║╤Н╤И╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╨╣ AI Manager
                        ai_manager, ai_available = get_cached_ai_manager()
                        
                        # ╨С╤Л╤Б╤В╤А╨░╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╨┤╨╛╤Б╤В╤Г╨┐╨╜╨╛╤Б╤В╨╕: ╨╡╤Б╨╗╨╕ AI ╨╜╨╡╨┤╨╛╤Б╤В╤Г╨┐╨╡╨╜, ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝
                        if not ai_available or not ai_manager:
                            # AI ╨╝╨╛╨┤╤Г╨╗╨╕ ╨╜╨╡ ╨╖╨░╨│╤А╤Г╨╢╨╡╨╜╤Л (╨╜╨╡╤В ╨╗╨╕╤Ж╨╡╨╜╨╖╨╕╨╕ ╨╕╨╗╨╕ ╨╜╨╡ ╤Г╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╤Л)
                            # ╨Э╨╡ ╨╗╨╛╨│╨╕╤А╤Г╨╡╨╝ ╨║╨░╨╢╨┤╤Л╨╣ ╤А╨░╨╖, ╤З╤В╨╛╨▒╤Л ╨╜╨╡ ╤Б╨┐╨░╨╝╨╕╤В╤М
                            pass
                        elif ai_manager.anomaly_detector:
                            # ╨Р╨╜╨░╨╗╨╕╨╖╨╕╤А╤Г╨╡╨╝ ╤Б╨▓╨╡╤З╨╕ ╤Б ╨┐╨╛╨╝╨╛╤Й╤М╤О ╨Ш╨Ш
                            anomaly_result = ai_manager.anomaly_detector.detect(candles)
                        
                            if anomaly_result.get('is_anomaly'):
                                severity = anomaly_result.get('severity', 0)
                                anomaly_type = anomaly_result.get('anomaly_type', 'UNKNOWN')
                                
                                # ╨С╨╗╨╛╨║╨╕╤А╤Г╨╡╨╝ ╨╡╤Б╨╗╨╕ severity > threshold
                                if severity > AIConfig.AI_ANOMALY_BLOCK_THRESHOLD:
                                    return False
                                else:
                                    logger.warning(
                                        f"{symbol}: тЪая╕П ╨Я╨а╨Х╨Ф╨г╨Я╨а╨Х╨Ц╨Ф╨Х╨Э╨Ш╨Х (AI): "
                                        f"╨Р╨╜╨╛╨╝╨░╨╗╨╕╤П {anomaly_type} "
                                        f"(severity: {severity:.2%} - ╨╜╨╕╨╢╨╡ ╨┐╨╛╤А╨╛╨│╨░ {AIConfig.AI_ANOMALY_BLOCK_THRESHOLD:.2%})"
                                    )
                            else:
                                logger.debug(f"{symbol}: тЬЕ AI: ╨Р╨╜╨╛╨╝╨░╨╗╨╕╨╣ ╨╜╨╡ ╨╛╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜╨╛")
                    
                    except ImportError as e:
                        logger.debug(f"{symbol}: AI ╨╝╨╛╨┤╤Г╨╗╤М ╨╜╨╡ ╨┤╨╛╤Б╤В╤Г╨┐╨╡╨╜: {e}")
                    except Exception as e:
                        logger.error(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ AI ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕: {e}")
        
            except ImportError:
                pass  # AIConfig ╨╜╨╡ ╨┤╨╛╤Б╤В╤Г╨┐╨╡╨╜ - ╨┐╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ AI ╨┐╤А╨╛╨▓╨╡╤А╨║╤Г
        
                logger.debug(f"{symbol}: тЬЕ ╨Я╨а╨Ю╨Щ╨Ф╨Х╨Э")
        return True
        
    except Exception as e:
        logger.error(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕: {e}")
        return False

# ╨Р╨╗╨╕╨░╤Б ╨┤╨╗╤П ╨╛╨▒╤А╨░╤В╨╜╨╛╨╣ ╤Б╨╛╨▓╨╝╨╡╤Б╤В╨╕╨╝╨╛╤Б╤В╨╕
check_anti_dump_pump = check_exit_scam_filter


def get_lstm_prediction(symbol, signal, current_price):
    """
    ╨Я╨╛╨╗╤Г╤З╨░╨╡╤В ╨┐╤А╨╡╨┤╤Б╨║╨░╨╖╨░╨╜╨╕╨╡ LSTM ╨┤╨╗╤П ╨╝╨╛╨╜╨╡╤В╤Л
    
    Args:
        symbol: ╨б╨╕╨╝╨▓╨╛╨╗ ╨╝╨╛╨╜╨╡╤В╤Л
        signal: ╨б╨╕╨│╨╜╨░╨╗ ('LONG' ╨╕╨╗╨╕ 'SHORT')
        current_price: ╨в╨╡╨║╤Г╤Й╨░╤П ╤Ж╨╡╨╜╨░
    
    Returns:
        Dict ╤Б ╨┐╤А╨╡╨┤╤Б╨║╨░╨╖╨░╨╜╨╕╨╡╨╝ ╨╕╨╗╨╕ None
    """
    try:
        from bot_engine.bot_config import AIConfig
        
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨▓╨║╨╗╤О╤З╨╡╨╜ ╨╗╨╕ LSTM
        if not (AIConfig.AI_ENABLED and AIConfig.AI_LSTM_ENABLED):
            return None
        
        try:
            # тЬЕ ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╖╨░╨║╤Н╤И╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╨╣ AI Manager
            ai_manager, ai_available = get_cached_ai_manager()
            
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┤╨╛╤Б╤В╤Г╨┐╨╜╨╛╤Б╤В╤М LSTM
            if not ai_available or not ai_manager or not ai_manager.lstm_predictor:
                return None
            
            # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╤Б╨▓╨╡╤З╨╕ ╨┤╨╗╤П ╨░╨╜╨░╨╗╨╕╨╖╨░
            exch = get_exchange()
            if not exch:
                return None
            
            chart_response = exch.get_chart_data(symbol, '6h', '30d')
            if not chart_response or not chart_response.get('success'):
                return None
            
            candles = chart_response.get('data', {}).get('candles', [])
            if len(candles) < 60:  # LSTM ╤В╤А╨╡╨▒╤Г╨╡╤В ╨╝╨╕╨╜╨╕╨╝╤Г╨╝ 60 ╤Б╨▓╨╡╤З╨╡╨╣
                return None
            
            # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨┐╤А╨╡╨┤╤Б╨║╨░╨╖╨░╨╜╨╕╨╡ ╤Б ╨в╨Р╨Щ╨Ь╨Р╨г╨в╨Ю╨Ь
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(ai_manager.lstm_predictor.predict, candles, current_price)
                try:
                    prediction = future.result(timeout=5)  # 5 ╤Б╨╡╨║╤Г╨╜╨┤ ╤В╨░╨╣╨╝╨░╤Г╤В ╨┤╨╗╤П LSTM
                except concurrent.futures.TimeoutError:
                    logger.warning(f"{symbol}: тП▒я╕П LSTM prediction ╤В╨░╨╣╨╝╨░╤Г╤В (5╤Б)")
                    prediction = None  # ╨Я╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ AI ╨┐╤А╨╛╨▓╨╡╤А╨║╤Г ╨┐╤А╨╕ ╤В╨░╨╣╨╝╨░╤Г╤В╨╡
            
            if prediction and prediction.get('confidence', 0) >= AIConfig.AI_LSTM_MIN_CONFIDENCE:
                # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╤Б╨╛╨▓╨┐╨░╨┤╨╡╨╜╨╕╨╡ ╨╜╨░╨┐╤А╨░╨▓╨╗╨╡╨╜╨╕╨╣
                lstm_direction = "LONG" if prediction['direction'] > 0 else "SHORT"
                confidence = prediction['confidence']
                
                if lstm_direction == signal:
                    logger.info(
                        f"{symbol}: тЬЕ ╨Я╨Ю╨Ф╨в╨Т╨Х╨а╨Ц╨Ф╨Х╨Э╨Ш╨Х: "
                        f"LSTM ╨┐╤А╨╡╨┤╤Б╨║╨░╨╖╤Л╨▓╨░╨╡╤В {lstm_direction} "
                        f"(╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╨╡: {prediction['change_percent']:+.2f}%, "
                        f"╤Г╨▓╨╡╤А╨╡╨╜╨╜╨╛╤Б╤В╤М: {confidence:.1f}%)"
                    )
                else:
                    logger.warning(
                        f"{symbol}: тЪая╕П ╨Я╨а╨Ю╨в╨Ш╨Т╨Ю╨а╨Х╨з╨Ш╨Х: "
                        f"╨б╨╕╨│╨╜╨░╨╗ {signal}, ╨╜╨╛ LSTM ╨┐╤А╨╡╨┤╤Б╨║╨░╨╖╤Л╨▓╨░╨╡╤В {lstm_direction} "
                        f"(╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╨╡: {prediction['change_percent']:+.2f}%, "
                        f"╤Г╨▓╨╡╤А╨╡╨╜╨╜╨╛╤Б╤В╤М: {confidence:.1f}%)"
                    )
                
                return {
                    **prediction,
                    'lstm_direction': lstm_direction,
                    'matches_signal': lstm_direction == signal
                }
            
            return None
            
        except ImportError as e:
            logger.debug(f"{symbol}: AI ╨╝╨╛╨┤╤Г╨╗╤М ╨╜╨╡ ╨┤╨╛╤Б╤В╤Г╨┐╨╡╨╜: {e}")
            return None
        except Exception as e:
            logger.error(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ LSTM ╨┐╤А╨╡╨┤╤Б╨║╨░╨╖╨░╨╜╨╕╤П: {e}")
            return None
    
    except ImportError:
        return None


def get_pattern_analysis(symbol, signal, current_price):
    """
    ╨Я╨╛╨╗╤Г╤З╨░╨╡╤В ╨░╨╜╨░╨╗╨╕╨╖ ╨┐╨░╤В╤В╨╡╤А╨╜╨╛╨▓ ╨┤╨╗╤П ╨╝╨╛╨╜╨╡╤В╤Л
    
    Args:
        symbol: ╨б╨╕╨╝╨▓╨╛╨╗ ╨╝╨╛╨╜╨╡╤В╤Л
        signal: ╨б╨╕╨│╨╜╨░╨╗ ('LONG' ╨╕╨╗╨╕ 'SHORT')
        current_price: ╨в╨╡╨║╤Г╤Й╨░╤П ╤Ж╨╡╨╜╨░
    
    Returns:
        Dict ╤Б ╨░╨╜╨░╨╗╨╕╨╖╨╛╨╝ ╨┐╨░╤В╤В╨╡╤А╨╜╨╛╨▓ ╨╕╨╗╨╕ None
    """
    try:
        from bot_engine.bot_config import AIConfig
        
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨▓╨║╨╗╤О╤З╨╡╨╜ ╨╗╨╕ Pattern Recognition
        if not (AIConfig.AI_ENABLED and AIConfig.AI_PATTERN_ENABLED):
            return None
        
        try:
            # тЬЕ ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╖╨░╨║╤Н╤И╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╨╣ AI Manager
            ai_manager, ai_available = get_cached_ai_manager()
            
            # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┤╨╛╤Б╤В╤Г╨┐╨╜╨╛╤Б╤В╤М Pattern Detector
            if not ai_available or not ai_manager or not ai_manager.pattern_detector:
                return None
            
            # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╤Б╨▓╨╡╤З╨╕ ╨┤╨╗╤П ╨░╨╜╨░╨╗╨╕╨╖╨░
            exch = get_exchange()
            if not exch:
                return None
            
            chart_response = exch.get_chart_data(symbol, '6h', '30d')
            if not chart_response or not chart_response.get('success'):
                return None
            
            candles = chart_response.get('data', {}).get('candles', [])
            if len(candles) < 100:  # Pattern ╤В╤А╨╡╨▒╤Г╨╡╤В ╨╝╨╕╨╜╨╕╨╝╤Г╨╝ 100 ╤Б╨▓╨╡╤З╨╡╨╣
                return None
            
            # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨░╨╜╨░╨╗╨╕╨╖ ╨┐╨░╤В╤В╨╡╤А╨╜╨╛╨▓ ╤Б ╨в╨Р╨Щ╨Ь╨Р╨г╨в╨Ю╨Ь
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    ai_manager.pattern_detector.get_pattern_signal,
                    candles, 
                    current_price, 
                    signal
                )
                try:
                    pattern_signal = future.result(timeout=5)  # 5 ╤Б╨╡╨║╤Г╨╜╨┤ ╤В╨░╨╣╨╝╨░╤Г╤В
                except concurrent.futures.TimeoutError:
                    logger.warning(f"{symbol}: тП▒я╕П Pattern detection ╤В╨░╨╣╨╝╨░╤Г╤В (5╤Б)")
                    pattern_signal = {'patterns_found': 0, 'confirmation': False}  # ╨Я╤А╨╛╨┐╤Г╤Б╨║╨░╨╡╨╝ ╨┐╤А╨╕ ╤В╨░╨╣╨╝╨░╤Г╤В╨╡
            
            if pattern_signal['patterns_found'] > 0:
                # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ ╨┐╨╛╨┤╤В╨▓╨╡╤А╨╢╨┤╨╡╨╜╨╕╨╡
                if pattern_signal['confirmation']:
                    logger.info(
                        f"{symbol}: тЬЕ ╨Я╨Ю╨Ф╨в╨Т╨Х╨а╨Ц╨Ф╨Х╨Э╨Ш╨Х: "
                        f"╨Я╨░╤В╤В╨╡╤А╨╜╤Л ╨┐╨╛╨┤╤В╨▓╨╡╤А╨╢╨┤╨░╤О╤В {signal} "
                        f"(╨╜╨░╨╣╨┤╨╡╨╜╨╛: {pattern_signal['patterns_found']}, "
                        f"╤Г╨▓╨╡╤А╨╡╨╜╨╜╨╛╤Б╤В╤М: {pattern_signal['confidence']:.1f}%)"
                    )
                    
                    if pattern_signal['strongest_pattern']:
                        strongest = pattern_signal['strongest_pattern']
                        logger.info(
                            f"{symbol}:    тФФтФА {strongest['name']}: "
                            f"{strongest['description']}"
                        )
                else:
                    logger.warning(
                        f"{symbol}: тЪая╕П ╨Я╨а╨Ю╨в╨Ш╨Т╨Ю╨а╨Х╨з╨Ш╨Х: "
                        f"╨б╨╕╨│╨╜╨░╨╗ {signal}, ╨╜╨╛ ╨┐╨░╤В╤В╨╡╤А╨╜╤Л ╤Г╨║╨░╨╖╤Л╨▓╨░╤О╤В ╨╜╨░ {pattern_signal['signal']} "
                        f"(╤Г╨▓╨╡╤А╨╡╨╜╨╜╨╛╤Б╤В╤М: {pattern_signal['confidence']:.1f}%)"
                    )
                
                return pattern_signal
            
            return None
            
        except ImportError as e:
            logger.debug(f"{symbol}: AI ╨╝╨╛╨┤╤Г╨╗╤М ╨╜╨╡ ╨┤╨╛╤Б╤В╤Г╨┐╨╡╨╜: {e}")
            return None
        except Exception as e:
            logger.error(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨░╨╜╨░╨╗╨╕╨╖╨░ ╨┐╨░╤В╤В╨╡╤А╨╜╨╛╨▓: {e}")
            return None
    
    except ImportError:
        return None  # AIConfig ╨╜╨╡ ╨┤╨╛╤Б╤В╤Г╨┐╨╡╨╜

def check_no_existing_position(symbol, signal):
    """╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╤В, ╤З╤В╨╛ ╨╜╨╡╤В ╤Б╤Г╤Й╨╡╤Б╤В╨▓╤Г╤О╤Й╨╕╤Е ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣ ╨╜╨░ ╨▒╨╕╤А╨╢╨╡"""
    try:
        exch = get_exchange()
        if not exch:
            return False
        
        exchange_positions = exch.get_positions()
        if isinstance(exchange_positions, tuple):
            positions_list = exchange_positions[0] if exchange_positions else []
        else:
            positions_list = exchange_positions if exchange_positions else []
        
        expected_side = 'LONG' if signal == 'ENTER_LONG' else 'SHORT'
        
        # ╨Я╤А╨╛╨▓╨╡╤А╤П╨╡╨╝, ╨╡╤Б╤В╤М ╨╗╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╤П ╤В╨╛╨╣ ╨╢╨╡ ╤Б╤В╨╛╤А╨╛╨╜╤Л
        for pos in positions_list:
            if pos.get('symbol') == symbol and abs(float(pos.get('size', 0))) > 0:
                existing_side = pos.get('side', 'UNKNOWN')
                if existing_side == expected_side:
                    logger.debug(f"{symbol}: ╨г╨╢╨╡ ╨╡╤Б╤В╤М ╨┐╨╛╨╖╨╕╤Ж╨╕╤П {existing_side}")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╣: {e}")
        return False

def create_new_bot(symbol, config=None, exchange_obj=None):
    """╨б╨╛╨╖╨┤╨░╨╡╤В ╨╜╨╛╨▓╨╛╨│╨╛ ╨▒╨╛╤В╨░"""
    try:
        # ╨Ы╨╛╨║╨░╨╗╤М╨╜╤Л╨╣ ╨╕╨╝╨┐╨╛╤А╤В ╨┤╨╗╤П ╨╕╨╖╨▒╨╡╨╢╨░╨╜╨╕╤П ╤Ж╨╕╨║╨╗╨╕╤З╨╡╤Б╨║╨╛╨│╨╛ ╨╕╨╝╨┐╨╛╤А╤В╨░
        from bots_modules.bot_class import NewTradingBot
        from bots_modules.imports_and_globals import get_exchange
        exchange_to_use = exchange_obj if exchange_obj else get_exchange()
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╤А╨░╨╖╨╝╨╡╤А╨░ ╨┐╨╛╨╖╨╕╤Ж╨╕╨╕ ╨╕╨╖ ╨║╨╛╨╜╤Д╨╕╨│╨░
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╤З╤В╨╡╨╜╨╕╨╡ ╤Б╨╗╨╛╨▓╨░╤А╤П - ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
        auto_bot_config = bots_data['auto_bot_config']
        default_volume = auto_bot_config.get('default_position_size')
        default_volume_mode = auto_bot_config.get('default_position_mode', 'usdt')
        
        # ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤О ╨▒╨╛╤В╨░
        bot_config = {
            'symbol': symbol,
            'status': BOT_STATUS['RUNNING'],  # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ю: ╨▒╨╛╤В ╨┤╨╛╨╗╨╢╨╡╨╜ ╨▒╤Л╤В╤М ╨░╨║╤В╨╕╨▓╨╜╤Л╨╝
            'created_at': datetime.now().isoformat(),
            'opened_by_autobot': True,
            'volume_mode': default_volume_mode,
            'volume_value': default_volume,  # тЬЕ ╨Ш╨б╨Я╨а╨Р╨Т╨Ы╨Х╨Э╨Ю: ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝ ╨╖╨╜╨░╤З╨╡╨╜╨╕╨╡ ╨╕╨╖ ╨║╨╛╨╜╤Д╨╕╨│╨░
            'leverage': auto_bot_config.get('leverage', 1)  # тЬЕ ╨Ф╨╛╨▒╨░╨▓╨╗╤П╨╡╨╝ leverage ╨╕╨╖ ╨│╨╗╨╛╨▒╨░╨╗╤М╨╜╨╛╨│╨╛ ╨║╨╛╨╜╤Д╨╕╨│╨░
        }

        individual_settings = get_individual_coin_settings(symbol)
        if individual_settings:
            bot_config.update(individual_settings)

        # ╨У╨░╤А╨░╨╜╤В╨╕╤А╤Г╨╡╨╝ ╨╛╨▒╤П╨╖╨░╤В╨╡╨╗╤М╨╜╤Л╨╡ ╨┐╨╛╨╗╤П
        bot_config['symbol'] = symbol
        bot_config['status'] = BOT_STATUS['RUNNING']
        bot_config.setdefault('volume_mode', default_volume_mode)
        if bot_config.get('volume_value') is None:
            bot_config['volume_value'] = default_volume
        if bot_config.get('leverage') is None:
            bot_config['leverage'] = auto_bot_config.get('leverage', 1)  # тЬЕ Fallback ╨┤╨╗╤П leverage
        
        # ╨б╨╛╨╖╨┤╨░╨╡╨╝ ╨▒╨╛╤В╨░
        new_bot = NewTradingBot(symbol, bot_config, exchange_to_use)
        
        # ╨б╨╛╤Е╤А╨░╨╜╤П╨╡╨╝ ╨▓ bots_data
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╨┐╤А╨╕╤Б╨▓╨░╨╕╨▓╨░╨╜╨╕╨╡ - ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
        bots_data['bots'][symbol] = new_bot.to_dict()
        
        logger.info(f"тЬЕ ╨С╨╛╤В ╨┤╨╗╤П {symbol} ╤Б╨╛╨╖╨┤╨░╨╜ ╤Г╤Б╨┐╨╡╤И╨╜╨╛")
        return new_bot
        
    except Exception as e:
        logger.error(f"тЭМ ╨Ю╤И╨╕╨▒╨║╨░ ╤Б╨╛╨╖╨┤╨░╨╜╨╕╤П ╨▒╨╛╤В╨░ ╨┤╨╗╤П {symbol}: {e}")
        raise

def check_auto_bot_filters(symbol):
    """╨б╤В╨░╤А╨░╤П ╤Д╤Г╨╜╨║╤Ж╨╕╤П - ╨╛╤Б╤В╨░╨▓╨╗╨╡╨╜╨░ ╨┤╨╗╤П ╤Б╨╛╨▓╨╝╨╡╤Б╤В╨╕╨╝╨╛╤Б╤В╨╕"""
    return False  # ╨С╨╗╨╛╨║╨╕╤А╤Г╨╡╨╝ ╨▓╤Б╨╡

def test_exit_scam_filter(symbol):
    """╨в╨╡╤Б╤В╨╕╤А╤Г╨╡╤В ExitScam ╤Д╨╕╨╗╤М╤В╤А ╨┤╨╗╤П ╨║╨╛╨╜╨║╤А╨╡╤В╨╜╨╛╨╣ ╨╝╨╛╨╜╨╡╤В╤Л"""
    try:
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕ ╨╕╨╖ ╨║╨╛╨╜╤Д╨╕╨│╨░
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╤З╤В╨╡╨╜╨╕╨╡ ╤Б╨╗╨╛╨▓╨░╤А╤П - ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
        exit_scam_enabled = bots_data.get('auto_bot_config', {}).get('exit_scam_enabled', True)
        exit_scam_candles = bots_data.get('auto_bot_config', {}).get('exit_scam_candles', 10)
        single_candle_percent = bots_data.get('auto_bot_config', {}).get('exit_scam_single_candle_percent', 15.0)
        multi_candle_count = bots_data.get('auto_bot_config', {}).get('exit_scam_multi_candle_count', 4)
        multi_candle_percent = bots_data.get('auto_bot_config', {}).get('exit_scam_multi_candle_percent', 50.0)
        
        logger.info(f"ЁЯФН ╨в╨╡╤Б╤В╨╕╤А╤Г╨╡╨╝ ExitScam ╤Д╨╕╨╗╤М╤В╤А ╨┤╨╗╤П {symbol}")
        logger.info(f"тЪЩя╕П ╨Э╨░╤Б╤В╤А╨╛╨╣╨║╨╕:")
        logger.info(f"тЪЩя╕П - ╨Т╨║╨╗╤О╤З╨╡╨╜: {exit_scam_enabled}")
        logger.info(f"тЪЩя╕П - ╨Р╨╜╨░╨╗╨╕╨╖ ╤Б╨▓╨╡╤З╨╡╨╣: {exit_scam_candles}")
        logger.info(f"тЪЩя╕П - ╨Ы╨╕╨╝╨╕╤В ╨╛╨┤╨╜╨╛╨╣ ╤Б╨▓╨╡╤З╨╕: {single_candle_percent}%")
        logger.info(f"тЪЩя╕П - ╨Ы╨╕╨╝╨╕╤В {multi_candle_count} ╤Б╨▓╨╡╤З╨╡╨╣: {multi_candle_percent}%")
        
        if not exit_scam_enabled:
            logger.info(f"{symbol}: тЪая╕П ╨д╨╕╨╗╤М╤В╤А ╨Ю╨в╨Ъ╨Ы╨о╨з╨Х╨Э ╨▓ ╨║╨╛╨╜╤Д╨╕╨│╨╡")
            return
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╤Б╨▓╨╡╤З╨╕
        exch = get_exchange()
        if not exch:
            logger.error(f"{symbol}: ╨С╨╕╤А╨╢╨░ ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░")
            return
        
        chart_response = exch.get_chart_data(symbol, '6h', '30d')
        if not chart_response or not chart_response.get('success'):
            logger.error(f"{symbol}: ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╤Б╨▓╨╡╤З╨╕")
            return
        
        candles = chart_response.get('data', {}).get('candles', [])
        if len(candles) < exit_scam_candles:
            logger.error(f"{symbol}: ╨Э╨╡╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ ╤Б╨▓╨╡╤З╨╡╨╣ ({len(candles)})")
            return
        
        # ╨Р╨╜╨░╨╗╨╕╨╖╨╕╤А╤Г╨╡╨╝ ╨┐╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ N ╤Б╨▓╨╡╤З╨╡╨╣ (╨╕╨╖ ╨║╨╛╨╜╤Д╨╕╨│╨░)
        recent_candles = candles[-exit_scam_candles:]
        
        # ╨Я╨╛╨║╨░╨╖╤Л╨▓╨░╨╡╨╝ ╨┤╨╡╤В╨░╨╗╨╕ ╨║╨░╨╢╨┤╨╛╨╣ ╤Б╨▓╨╡╤З╨╕
        for i, candle in enumerate(recent_candles):
            open_price = candle['open']
            close_price = candle['close']
            high_price = candle['high']
            low_price = candle['low']
            
            price_change = ((close_price - open_price) / open_price) * 100
            candle_range = ((high_price - low_price) / open_price) * 100
            
            logger.info(f"{symbol}: ╨б╨▓╨╡╤З╨░ {i+1}: O={open_price:.4f} C={close_price:.4f} H={high_price:.4f} L={low_price:.4f} | ╨Ш╨╖╨╝╨╡╨╜╨╡╨╜╨╕╨╡: {price_change:+.1f}% | ╨Ф╨╕╨░╨┐╨░╨╖╨╛╨╜: {candle_range:.1f}%")
        
        # ╨в╨╡╤Б╤В╨╕╤А╤Г╨╡╨╝ ╤Д╨╕╨╗╤М╤В╤А ╤Б ╨┤╨╡╤В╨░╨╗╤М╨╜╤Л╨╝ ╨╗╨╛╨│╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡╨╝
        logger.info(f"{symbol}: ЁЯФН ╨Ч╨░╨┐╤Г╤Б╨║╨░╨╡╨╝ ╨┐╤А╨╛╨▓╨╡╤А╨║╤Г ExitScam ╤Д╨╕╨╗╤М╤В╤А╨░...")
        result = check_exit_scam_filter(symbol, {})
        
        if result:
            logger.info(f"{symbol}: тЬЕ ╨а╨Х╨Ч╨г╨Ы╨м╨в╨Р╨в: ╨Я╨а╨Ю╨Щ╨Ф╨Х╨Э")
        else:
            logger.warning(f"{symbol}: тЭМ ╨а╨Х╨Ч╨г╨Ы╨м╨в╨Р╨в: ╨Ч╨Р╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Р╨Э")
        
        # ╨Ф╨╛╨┐╨╛╨╗╨╜╨╕╤В╨╡╨╗╤М╨╜╤Л╨╣ ╨░╨╜╨░╨╗╨╕╨╖
        logger.info(f"{symbol}: ЁЯУК ╨Ф╨╛╨┐╨╛╨╗╨╜╨╕╤В╨╡╨╗╤М╨╜╤Л╨╣ ╨░╨╜╨░╨╗╨╕╨╖:")
        
        # 1. ╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╨╛╤В╨┤╨╡╨╗╤М╨╜╤Л╤Е ╤Б╨▓╨╡╤З╨╡╨╣
        extreme_single_count = 0
        for i, candle in enumerate(recent_candles):
            open_price = candle['open']
            close_price = candle['close']
            
            price_change = abs((close_price - open_price) / open_price) * 100
            
            if price_change > single_candle_percent:
                extreme_single_count += 1
                logger.warning(f"{symbol}: тЭМ ╨Я╤А╨╡╨▓╤Л╤И╨╡╨╜╨╕╨╡ ╨╗╨╕╨╝╨╕╤В╨░ ╨╛╨┤╨╜╨╛╨╣ ╤Б╨▓╨╡╤З╨╕ #{i+1}: {price_change:.1f}% > {single_candle_percent}%")
        
        # 2. ╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╤Б╤Г╨╝╨╝╨░╤А╨╜╨╛╨│╨╛ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П ╨╖╨░ N ╤Б╨▓╨╡╤З╨╡╨╣
        if len(recent_candles) >= multi_candle_count:
            multi_candles = recent_candles[-multi_candle_count:]
            first_open = multi_candles[0]['open']
            last_close = multi_candles[-1]['close']
            
            total_change = abs((last_close - first_open) / first_open) * 100
            
            logger.info(f"{symbol}: ЁЯУИ {multi_candle_count}-╤Б╨▓╨╡╤З╨╡╤З╨╜╤Л╨╣ ╨░╨╜╨░╨╗╨╕╨╖: {total_change:.1f}% (╨┐╨╛╤А╨╛╨│: {multi_candle_percent}%)")
            
            if total_change > multi_candle_percent:
                logger.warning(f"{symbol}: тЭМ ╨Я╤А╨╡╨▓╤Л╤И╨╡╨╜╨╕╨╡ ╤Б╤Г╨╝╨╝╨░╤А╨╜╨╛╨│╨╛ ╨╗╨╕╨╝╨╕╤В╨░: {total_change:.1f}% > {multi_candle_percent}%")
        
    except Exception as e:
        logger.error(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╤В╨╡╤Б╤В╨╕╤А╨╛╨▓╨░╨╜╨╕╤П: {e}")

# ╨Р╨╗╨╕╨░╤Б ╨┤╨╗╤П ╨╛╨▒╤А╨░╤В╨╜╨╛╨╣ ╤Б╨╛╨▓╨╝╨╡╤Б╤В╨╕╨╝╨╛╤Б╤В╨╕
test_anti_pump_filter = test_exit_scam_filter

def test_rsi_time_filter(symbol):
    """╨в╨╡╤Б╤В╨╕╤А╤Г╨╡╤В RSI ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╣ ╤Д╨╕╨╗╤М╤В╤А ╨┤╨╗╤П ╨║╨╛╨╜╨║╤А╨╡╤В╨╜╨╛╨╣ ╨╝╨╛╨╜╨╡╤В╤Л"""
    try:
        logger.info(f"ЁЯФН ╨в╨╡╤Б╤В╨╕╤А╤Г╨╡╨╝ RSI ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╣ ╤Д╨╕╨╗╤М╤В╤А ╨┤╨╗╤П {symbol}")
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╤Б╨▓╨╡╤З╨╕
        exch = get_exchange()
        if not exch:
            logger.error(f"{symbol}: ╨С╨╕╤А╨╢╨░ ╨╜╨╡ ╨╕╨╜╨╕╤Ж╨╕╨░╨╗╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨░")
            return
                
        chart_response = exch.get_chart_data(symbol, '6h', '30d')
        if not chart_response or not chart_response.get('success'):
            logger.error(f"{symbol}: ╨Э╨╡ ╤Г╨┤╨░╨╗╨╛╤Б╤М ╨┐╨╛╨╗╤Г╤З╨╕╤В╤М ╤Б╨▓╨╡╤З╨╕")
            return
        
        candles = chart_response.get('data', {}).get('candles', [])
        if len(candles) < 50:
            logger.error(f"{symbol}: ╨Э╨╡╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ ╤Б╨▓╨╡╤З╨╡╨╣ ({len(candles)})")
            return
        
        # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╤В╨╡╨║╤Г╤Й╨╕╨╣ RSI
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╤З╤В╨╡╨╜╨╕╨╡ ╤Б╨╗╨╛╨▓╨░╤А╤П - ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
        coin_data = coins_rsi_data['coins'].get(symbol)
        if not coin_data:
            logger.error(f"{symbol}: ╨Э╨╡╤В RSI ╨┤╨░╨╜╨╜╤Л╤Е")
            return
        
        current_rsi = coin_data.get('rsi6h', 0)
        signal = coin_data.get('signal', 'WAIT')
        
        # тЬЕ ╨Ю╨┐╤А╨╡╨┤╨╡╨╗╤П╨╡╨╝ ╨Ю╨а╨Ш╨У╨Ш╨Э╨Р╨Ы╨м╨Э╨л╨Щ ╤Б╨╕╨│╨╜╨░╨╗ ╨╜╨░ ╨╛╤Б╨╜╨╛╨▓╨╡ ╤В╨╛╨╗╤М╨║╨╛ RSI ╤Б ╤Г╤З╨╡╤В╨╛╨╝ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╤Е ╨╜╨░╤Б╤В╤А╨╛╨╡╨║
        # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╤З╤В╨╡╨╜╨╕╨╡ ╤Б╨╗╨╛╨▓╨░╤А╤П - ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
        individual_settings = get_individual_coin_settings(symbol)
        
        rsi_long_threshold = individual_settings.get('rsi_long_threshold') if individual_settings else None
        if rsi_long_threshold is None:
            rsi_long_threshold = bots_data.get('auto_bot_config', {}).get('rsi_long_threshold', 29)
        
        rsi_short_threshold = individual_settings.get('rsi_short_threshold') if individual_settings else None
        if rsi_short_threshold is None:
            rsi_short_threshold = bots_data.get('auto_bot_config', {}).get('rsi_short_threshold', 71)
        
        original_signal = 'WAIT'
        if current_rsi <= rsi_long_threshold:
            original_signal = 'ENTER_LONG'
        elif current_rsi >= rsi_short_threshold:
            original_signal = 'ENTER_SHORT'
        
        logger.info(f"{symbol}: ╨в╨╡╨║╤Г╤Й╨╕╨╣ RSI={current_rsi:.1f}, ╨Ю╤А╨╕╨│╨╕╨╜╨░╨╗╤М╨╜╤Л╨╣ ╤Б╨╕╨│╨╜╨░╨╗={original_signal}, ╨д╨╕╨╜╨░╨╗╤М╨╜╤Л╨╣ ╤Б╨╕╨│╨╜╨░╨╗={signal}")
        if individual_settings:
            logger.info(f"{symbol}: ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╤О╤В╤Б╤П ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕: rsi_long={rsi_long_threshold}, rsi_short={rsi_short_threshold}")
        
        # ╨в╨╡╤Б╤В╨╕╤А╤Г╨╡╨╝ ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨╣ ╤Д╨╕╨╗╤М╤В╤А ╤Б ╨Ю╨а╨Ш╨У╨Ш╨Э╨Р╨Ы╨м╨Э╨л╨Ь ╤Б╨╕╨│╨╜╨░╨╗╨╛╨╝ ╨╕ ╨╕╨╜╨┤╨╕╨▓╨╕╨┤╤Г╨░╨╗╤М╨╜╤Л╨╝╨╕ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨░╨╝╨╕
        time_filter_result = check_rsi_time_filter(candles, current_rsi, original_signal, symbol=symbol, individual_settings=individual_settings)
        
        logger.info(f"{symbol}: ╨а╨╡╨╖╤Г╨╗╤М╤В╨░╤В ╨▓╤А╨╡╨╝╨╡╨╜╨╜╨╛╨│╨╛ ╤Д╨╕╨╗╤М╤В╤А╨░:")
        logger.info(f"{symbol}: ╨а╨░╨╖╤А╨╡╤И╨╡╨╜╨╛: {time_filter_result['allowed']}")
        logger.info(f"{symbol}: ╨Я╤А╨╕╤З╨╕╨╜╨░: {time_filter_result['reason']}")
        if 'calm_candles' in time_filter_result and time_filter_result['calm_candles'] is not None:
            logger.info(f"{symbol}: ╨б╨┐╨╛╨║╨╛╨╣╨╜╤Л╤Е ╤Б╨▓╨╡╤З╨╡╨╣: {time_filter_result['calm_candles']}")
        if 'last_extreme_candles_ago' in time_filter_result and time_filter_result['last_extreme_candles_ago'] is not None:
            logger.info(f"{symbol}: ╨Я╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╣ ╤Н╨║╤Б╤В╤А╨╡╨╝╤Г╨╝: {time_filter_result['last_extreme_candles_ago']} ╤Б╨▓╨╡╤З╨╡╨╣ ╨╜╨░╨╖╨░╨┤")
        
        # ╨Я╨╛╨║╨░╨╖╤Л╨▓╨░╨╡╨╝ ╨╕╤Б╤В╨╛╤А╨╕╤О RSI ╨┤╨╗╤П ╨░╨╜╨░╨╗╨╕╨╖╨░
        closes = [candle['close'] for candle in candles]
        rsi_history = calculate_rsi_history(closes, 14)
        
        if rsi_history:
            logger.info(f"{symbol}: ╨Я╨╛╤Б╨╗╨╡╨┤╨╜╨╕╨╡ 20 ╨╖╨╜╨░╤З╨╡╨╜╨╕╨╣ RSI:")
            last_20_rsi = rsi_history[-20:] if len(rsi_history) >= 20 else rsi_history
            
            # ╨Я╨╛╨╗╤Г╤З╨░╨╡╨╝ ╨┐╨╛╤А╨╛╨│╨╕ ╨┤╨╗╤П ╨┐╨╛╨┤╤Б╨▓╨╡╤В╨║╨╕
            # тЪб ╨С╨Х╨Ч ╨С╨Ы╨Ю╨Ъ╨Ш╨а╨Ю╨Т╨Ъ╨Ш: ╤З╤В╨╡╨╜╨╕╨╡ ╤Б╨╗╨╛╨▓╨░╤А╤П - ╨░╤В╨╛╨╝╨░╤А╨╜╨░╤П ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤П
            rsi_long_threshold = bots_data.get('auto_bot_config', {}).get('rsi_long_threshold', 29)
            rsi_short_threshold = bots_data.get('auto_bot_config', {}).get('rsi_short_threshold', 71)
            rsi_time_filter_upper = bots_data.get('auto_bot_config', {}).get('rsi_time_filter_upper', 65)
            rsi_time_filter_lower = bots_data.get('auto_bot_config', {}).get('rsi_time_filter_lower', 35)
            
            for i, rsi_val in enumerate(last_20_rsi):
                # ╨Ш╨╜╨┤╨╡╨║╤Б ╨╛╤В ╨║╨╛╨╜╤Ж╨░ ╨╕╤Б╤В╨╛╤А╨╕╨╕
                index_from_end = len(last_20_rsi) - i - 1
                
                # ╨Ю╨┐╤А╨╡╨┤╨╡╨╗╤П╨╡╨╝ ╨╝╨░╤А╨║╨╡╤А╤Л ╨┤╨╗╤П ╨╜╨░╨│╨╗╤П╨┤╨╜╨╛╤Б╤В╨╕
                markers = []
                if rsi_val >= rsi_short_threshold:
                    markers.append(f"ЁЯФ┤╨Я╨Ш╨Ъ>={rsi_short_threshold}")
                elif rsi_val <= rsi_long_threshold:
                    markers.append(f"ЁЯЯв╨Ы╨Ю╨Щ<={rsi_long_threshold}")
                
                if rsi_val >= rsi_time_filter_upper:
                    markers.append(f"тЬЕ>={rsi_time_filter_upper}")
                elif rsi_val <= rsi_time_filter_lower:
                    markers.append(f"тЬЕ<={rsi_time_filter_lower}")
                
                marker_str = " ".join(markers) if markers else ""
                logger.info(f"{symbol}: ╨б╨▓╨╡╤З╨░ -{index_from_end}: RSI={rsi_val:.1f} {marker_str}")
        
    except Exception as e:
        logger.error(f"{symbol}: ╨Ю╤И╨╕╨▒╨║╨░ ╤В╨╡╤Б╤В╨╕╤А╨╛╨▓╨░╨╜╨╕╤П: {e}")

