"""
Единая точка входа для конфига. Данные — configs/bot_config.py.
Импорт: from bot_engine.config_loader import ...
"""
from __future__ import annotations

import importlib

from configs.bot_config import (
    DefaultAutoBotConfig,
    AutoBotConfig,
    DefaultBotConfig,
    SystemConfig,
    RiskConfig,
    FilterConfig,
    ExchangeConfig,
    AIConfig,
)


def config_class_to_dict(cls):
    """Класс конфига → словарь (ключи lower) для config.get('enabled') и т.д."""
    return {
        k.lower(): getattr(cls, k)
        for k in dir(cls)
        if not k.startswith('_') and not callable(getattr(cls, k))
    }


def _get_default_timeframe():
    """Единый источник: configs/bot_config.py. Сначала AutoBotConfig (рабочий), затем Default."""
    try:
        auto = config_class_to_dict(AutoBotConfig).get('system_timeframe')
        if auto is not None:
            return auto
        return config_class_to_dict(DefaultAutoBotConfig).get('system_timeframe')
    except Exception:
        return None


# Константы для indicators/AI
RSI_PERIOD = SystemConfig.RSI_PERIOD
RSI_OVERSOLD = SystemConfig.RSI_OVERSOLD
RSI_OVERBOUGHT = SystemConfig.RSI_OVERBOUGHT
RSI_EXIT_LONG_WITH_TREND = SystemConfig.RSI_EXIT_LONG_WITH_TREND
RSI_EXIT_SHORT_WITH_TREND = SystemConfig.RSI_EXIT_SHORT_WITH_TREND
RSI_EXIT_LONG_AGAINST_TREND = SystemConfig.RSI_EXIT_LONG_AGAINST_TREND
RSI_EXIT_SHORT_AGAINST_TREND = SystemConfig.RSI_EXIT_SHORT_AGAINST_TREND
RSI_EXTREME_OVERSOLD = SystemConfig.RSI_EXTREME_OVERSOLD
RSI_EXTREME_OVERBOUGHT = SystemConfig.RSI_EXTREME_OVERBOUGHT
RSI_VOLATILITY_THRESHOLD_HIGH = SystemConfig.RSI_VOLATILITY_THRESHOLD_HIGH
RSI_VOLATILITY_THRESHOLD_LOW = SystemConfig.RSI_VOLATILITY_THRESHOLD_LOW
RSI_DIVERGENCE_LOOKBACK = SystemConfig.RSI_DIVERGENCE_LOOKBACK
RSI_VOLUME_CONFIRMATION_MULTIPLIER = SystemConfig.RSI_VOLUME_CONFIRMATION_MULTIPLIER
RSI_STOCH_PERIOD = SystemConfig.RSI_STOCH_PERIOD
RSI_EXTREME_ZONE_TIMEOUT = SystemConfig.RSI_EXTREME_ZONE_TIMEOUT
EMA_FAST = SystemConfig.EMA_FAST
EMA_SLOW = SystemConfig.EMA_SLOW
TREND_CONFIRMATION_BARS = SystemConfig.TREND_CONFIRMATION_BARS
TREND_MIN_CONFIRMATIONS = SystemConfig.TREND_MIN_CONFIRMATIONS
TREND_REQUIRE_SLOPE = SystemConfig.TREND_REQUIRE_SLOPE
TREND_REQUIRE_PRICE = SystemConfig.TREND_REQUIRE_PRICE
TREND_REQUIRE_CANDLES = SystemConfig.TREND_REQUIRE_CANDLES
# Единый источник таймфрейма — AutoBotConfig (см. _get_default_timeframe); SystemConfig только fallback для старых путей
TIMEFRAME = _get_default_timeframe() or getattr(SystemConfig, 'SYSTEM_TIMEFRAME', None)

_current_timeframe = None


def get_current_timeframe():
    if _current_timeframe is not None:
        return _current_timeframe
    return _get_default_timeframe()


def set_current_timeframe(timeframe: str) -> bool:
    supported = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    if timeframe not in supported:
        return False
    global _current_timeframe
    _current_timeframe = timeframe
    return True


def reset_timeframe_to_config():
    global _current_timeframe
    _current_timeframe = None


def get_rsi_key(timeframe=None):
    return f'rsi{timeframe or get_current_timeframe()}'


def get_trend_key(timeframe=None):
    return f'trend{timeframe or get_current_timeframe()}'


def get_timeframe_suffix(timeframe=None):
    return timeframe or get_current_timeframe()


def get_rsi_from_coin_data(coin_data, timeframe=None):
    """
    Возвращает RSI ТОЛЬКО по запрошенному таймфрейму (единый источник истины).
    НЕ ДОБАВЛЯТЬ fallback на rsi6h/rsi: при работе на 1m подставлялся бы 6h RSI,
    бот открывал SHORT при 6h RSI 85, хотя на 1m RSI 13 → минусы по балансу.
    Если нужен RSI по другому ТФ — вызывать с явным timeframe=.
    """
    tf = timeframe or get_current_timeframe()
    rsi = coin_data.get(get_rsi_key(tf))
    return rsi


def get_trend_from_coin_data(coin_data, timeframe=None):
    """Тренд только по запрошенному ТФ (без подстановки trend6h)."""
    tf = timeframe or get_current_timeframe()
    trend = coin_data.get(get_trend_key(tf))
    return trend if trend is not None else 'NEUTRAL'


class BotStatus:
    IDLE = 'IDLE'
    IN_POSITION_LONG = 'IN_POSITION_LONG'
    IN_POSITION_SHORT = 'IN_POSITION_SHORT'
    PAUSED = 'PAUSED'


class TrendDirection:
    UP = 'UP'
    DOWN = 'DOWN'
    NEUTRAL = 'NEUTRAL'


class VolumeMode:
    FIXED_QTY = 'fixed_qty'
    FIXED_USDT = 'fixed_usdt'
    PERCENT_BALANCE = 'percent_balance'


DEFAULT_AUTO_BOT_CONFIG = config_class_to_dict(DefaultAutoBotConfig)
AUTO_BOT_CONFIG = config_class_to_dict(AutoBotConfig)
DEFAULT_BOT_CONFIG = config_class_to_dict(DefaultBotConfig)


def get_config_value(config_dict, key):
    """Значение только из конфига: config_dict[key] или DEFAULT_AUTO_BOT_CONFIG[key]. Без хардкодов."""
    if not config_dict:
        return DEFAULT_AUTO_BOT_CONFIG.get(key)
    val = config_dict.get(key)
    if val is not None:
        return val
    return DEFAULT_AUTO_BOT_CONFIG.get(key)


def reload_config():
    """Перезагрузить configs.bot_config и config_loader. Вызывать после изменения конфига на диске.
    После reload восстанавливает таймфрейм из файла (или сохраняет текущий, если файл старый)."""
    import configs.bot_config as _cfg
    import bot_engine.config_loader as _loader
    # Сохраняем текущий таймфрейм до reload (на случай race с load_system_config)
    saved_tf = _loader._current_timeframe if hasattr(_loader, '_current_timeframe') else None
    importlib.reload(_cfg)
    importlib.reload(_loader)
    # После reload _current_timeframe = None. Берём из файла, иначе восстанавливаем
    new_tf = _loader._get_default_timeframe()
    if new_tf:
        _loader._current_timeframe = new_tf
    elif saved_tf:
        _loader._current_timeframe = saved_tf
    return _loader
