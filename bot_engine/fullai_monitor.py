# -*- coding: utf-8 -*-
"""
FullAI Monitor — мониторинг позиций каждую секунду при full_ai_control.

FullAI ведёт все сделки ежесекундно: проверяет цену и позицию, решает закрыть или держать.
Использует get_fullai_data_context (БД, свечи, индикаторы) и get_ai_exit_decision.
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger('FullAI')

_monitor_thread: Optional[threading.Thread] = None
_monitor_stop = threading.Event()
_monitor_interval_sec = 1.0


def start_fullai_monitor() -> bool:
    """Запускает мониторинг FullAI (ежесекундная проверка позиций)."""
    global _monitor_thread
    if _monitor_thread and _monitor_thread.is_alive():
        return True
    _monitor_stop.clear()
    _monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
    _monitor_thread.start()
    logger.info("[FullAI Monitor] Запущен — проверка позиций каждую секунду")
    return True


def stop_fullai_monitor() -> None:
    """Останавливает мониторинг FullAI."""
    _monitor_stop.set()
    if _monitor_thread:
        _monitor_thread.join(timeout=5)
    logger.info("[FullAI Monitor] Остановлен")


def _is_fullai_enabled() -> bool:
    try:
        from bots_modules.imports_and_globals import bots_data, bots_data_lock
        with bots_data_lock:
            return bool((bots_data.get('auto_bot_config') or {}).get('full_ai_control', False))
    except Exception:
        return False


def _get_symbols_in_position() -> list:
    """Возвращает список символов с открытыми позициями."""
    try:
        from bots_modules.imports_and_globals import bots_data, bots_data_lock
        with bots_data_lock:
            bots = bots_data.get('bots') or {}
        return [
            sym for sym, b in bots.items()
            if b and (b.get('position') or b.get('entry_price'))
        ]
    except Exception:
        return []


def _monitor_loop() -> None:
    while not _monitor_stop.wait(_monitor_interval_sec):
        if not _is_fullai_enabled():
            continue
        symbols = _get_symbols_in_position()
        if not symbols:
            continue
        for symbol in symbols:
            try:
                _check_position_and_decide(symbol)
            except Exception as e:
                logger.debug("[FullAI Monitor] %s: %s", symbol, e)


def _check_position_and_decide(symbol: str) -> None:
    """Проверяет позицию по символу и принимает решение о закрытии."""
    try:
        from bot_engine.fullai_data_context import get_fullai_data_context
        from bots_modules.imports_and_globals import bots_data, bots_data_lock, get_effective_auto_bot_config, get_effective_coin_settings
        from bot_engine.ai.ai_integration import get_ai_exit_decision

        with bots_data_lock:
            bot = (bots_data.get('bots') or {}).get(symbol)
        if not bot or not getattr(bot, 'entry_price', None):
            return

        ctx = get_fullai_data_context(symbol)
        candles = ctx.get('candles') or []
        current_price = ctx.get('current_price')
        if not current_price and candles:
            current_price = float(candles[-1].get('close', 0))
        if not current_price:
            return

        entry_price = float(bot.entry_price or 0)
        if not entry_price:
            return
        position_side = getattr(bot, 'position_side', 'LONG') or 'LONG'
        if position_side == 'LONG':
            profit_percent = (current_price - entry_price) / entry_price * 100.0
        else:
            profit_percent = (entry_price - current_price) / entry_price * 100.0

        position_info = {
            'entry_price': entry_price,
            'position_side': position_side,
            'position_size_coins': getattr(bot, 'position_size_coins', None),
        }
        fullai_config = get_effective_auto_bot_config()
        coin_params = get_effective_coin_settings(symbol)

        decision = get_ai_exit_decision(
            symbol,
            position_info,
            candles,
            profit_percent,
            fullai_config,
            coin_params,
            data_context=ctx,
        )
        if not decision.get('close_now'):
            return

        reason = decision.get('reason') or 'FullAI_EXIT'
        logger.info("[FullAI Monitor] %s: закрытие — %s", symbol, reason)
        if hasattr(bot, '_close_position_on_exchange'):
            bot._close_position_on_exchange(reason)
        try:
            from bots_modules.fullai_scoring import record_trade_result
            record_trade_result(symbol, success=(profit_percent >= 0))
        except Exception:
            pass
        try:
            from bots_modules.fullai_trades_learner import run_fullai_trades_analysis_after_close
            run_fullai_trades_analysis_after_close(symbol)
        except Exception:
            pass
    except Exception as e:
        logger.exception("[FullAI Monitor] _check_position_and_decide %s: %s", symbol, e)
