# -*- coding: utf-8 -*-
"""
FullAI Data Context — единый контекст данных для FullAI.

Даёт FullAI доступ к:
- БД (свечи candles_history, bot_trades_history)
- Свечам из БД (ai_data.db, bots_data.db)
- Системным индикаторам (RSI, тренд, сигнал из coins_rsi_data)
- Возможность формировать свои индикаторы (custom indicators)
"""

import logging
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger('FullAI')


# Реестр кастомных индикаторов FullAI: name -> compute_fn(symbol, candles, base_context) -> value
_fullai_custom_indicators: Dict[str, Callable] = {}


def register_fullai_indicator(name: str, compute_fn: Callable[[str, List[Dict], Dict], Any]) -> None:
    """Регистрирует кастомный индикатор FullAI. compute_fn(symbol, candles, base_context) -> value."""
    _fullai_custom_indicators[name] = compute_fn


def get_fullai_data_context(
    symbol: str,
    timeframe: Optional[str] = None,
    candles_limit: int = 200,
    include_system_indicators: bool = True,
    include_custom_indicators: bool = True,
) -> Dict[str, Any]:
    """
    Собирает полный контекст данных для FullAI по символу.

    Returns:
        {
            'candles': [...],           # свечи из БД (ai_data.db candles_history)
            'current_price': float,
            'system': {                 # системные индикаторы (RSI, тренд, сигнал)
                'rsi': float,
                'trend': str,
                'signal': str,
            },
            'custom': {                 # кастомные индикаторы FullAI
                'indicator_name': value,
            },
            'position': {...} | None,   # данные позиции если есть (из bots_data)
        }
    """
    ctx = {
        'symbol': symbol,
        'candles': [],
        'current_price': None,
        'system': {},
        'custom': {},
        'position': None,
    }
    try:
        tf = timeframe
        if not tf:
            try:
                from bot_engine.config_loader import get_current_timeframe
                tf = get_current_timeframe()
            except Exception:
                tf = '6h'

        # 1. Свечи из БД (ai_data.db)
        try:
            from bot_engine.ai.ai_database import get_ai_database
            ai_db = get_ai_database()
            if ai_db:
                candles = ai_db.get_candles(symbol, timeframe=tf, limit=candles_limit)
                ctx['candles'] = candles or []
                if candles and len(candles) > 0:
                    ctx['current_price'] = float(candles[-1].get('close', 0))
        except Exception as e:
            logger.debug("fullai_data_context candles from ai_db: %s", e)

        # Fallback: свечи из bots_database.candles_cache
        if not ctx['candles']:
            try:
                from bot_engine.storage import get_candles_for_symbol
                candles_data = get_candles_for_symbol(symbol)
                if candles_data and isinstance(candles_data, dict):
                    candles = candles_data.get('candles') or candles_data.get('data') or []
                    if isinstance(candles, list) and candles:
                        ctx['candles'] = candles[-candles_limit:]
                        if ctx['candles']:
                            ctx['current_price'] = float(ctx['candles'][-1].get('close', 0))
            except Exception as e:
                logger.debug("fullai_data_context candles from bots_db: %s", e)

        # 2. Системные индикаторы (RSI, тренд, сигнал)
        if include_system_indicators:
            try:
                from bots_modules.imports_and_globals import coins_rsi_data, rsi_data_lock
                from bot_engine.config_loader import get_rsi_from_coin_data, get_trend_from_coin_data, get_current_timeframe
                with rsi_data_lock:
                    coin_data = (coins_rsi_data.get('coins') or {}).get(symbol, {})
                if coin_data:
                    tf = get_current_timeframe()
                    ctx['system'] = {
                        'rsi': get_rsi_from_coin_data(coin_data, timeframe=tf),
                        'trend': get_trend_from_coin_data(coin_data, timeframe=tf),
                        'signal': coin_data.get('signal', 'WAIT'),
                    }
            except Exception as e:
                logger.debug("fullai_data_context system indicators: %s", e)

        # 3. Позиция (из bots_data)
        try:
            from bots_modules.imports_and_globals import bots_data, bots_data_lock
            with bots_data_lock:
                bot_data = (bots_data.get('bots') or {}).get(symbol, {})
            if bot_data and bot_data.get('position'):
                ctx['position'] = {
                    'entry_price': bot_data.get('entry_price'),
                    'position_side': bot_data.get('position_side') or bot_data.get('position', {}).get('side'),
                    'position_size': bot_data.get('position_size_coins') or bot_data.get('position', {}).get('size'),
                }
        except Exception as e:
            logger.debug("fullai_data_context position: %s", e)

        # 4. Кастомные индикаторы FullAI
        if include_custom_indicators and ctx['candles'] and _fullai_custom_indicators:
            for name, fn in _fullai_custom_indicators.items():
                try:
                    ctx['custom'][name] = fn(symbol, ctx['candles'], ctx)
                except Exception as e:
                    logger.debug("fullai custom indicator %s: %s", name, e)

        return ctx
    except Exception as e:
        logger.warning("get_fullai_data_context %s: %s", symbol, e)
        return ctx


def _register_default_indicators() -> None:
    """Регистрирует примеры кастомных индикаторов FullAI (можно расширять)."""
    def momentum_pct(symbol: str, candles: list, ctx: dict) -> float:
        """Изменение цены за последние 5 свечей в %."""
        if not candles or len(candles) < 5:
            return 0.0
        old = float(candles[-5].get('close', 0))
        new = float(candles[-1].get('close', 0))
        if not old:
            return 0.0
        return (new - old) / old * 100.0

    def volatility_atr(symbol: str, candles: list, ctx: dict) -> float:
        """Упрощённая волатильность: средний диапазон (high-low) за 10 свечей."""
        if not candles or len(candles) < 5:
            return 0.0
        recent = candles[-10:] if len(candles) >= 10 else candles
        ranges = [float(c.get('high', 0)) - float(c.get('low', 0)) for c in recent if c.get('high') and c.get('low')]
        if not ranges:
            return 0.0
        avg_range = sum(ranges) / len(ranges)
        close = float(candles[-1].get('close', 1))
        return (avg_range / close * 100.0) if close else 0.0

    register_fullai_indicator('momentum_pct_5', momentum_pct)
    register_fullai_indicator('volatility_atr_pct', volatility_atr)


# Регистрируем дефолтные индикаторы при импорте
try:
    _register_default_indicators()
except Exception:
    pass
