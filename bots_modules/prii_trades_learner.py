# -*- coding: utf-8 -*-
"""
ПРИИ: изучение совершённых сделок и доработка параметров (блок 7).
Загружает сделки из bot_trades_history, оценивает успех/неудача, обновляет
**только** конфиг ПРИИ и таблицу full_ai_coin_params. Пользовательский конфиг
и individual_coin_settings не трогаются.
"""
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger('BOTS')


def _is_prii_enabled() -> bool:
    """ПРИИ включён только если full_ai_control в пользовательском конфиге."""
    try:
        from bots_modules.imports_and_globals import bots_data, bots_data_lock
        with bots_data_lock:
            return (bots_data.get('auto_bot_config') or {}).get('full_ai_control', False)
    except Exception:
        return False


def _evaluate_trade(trade: Dict[str, Any]) -> Dict[str, Any]:
    """
    Оценка одной сделки: успех/неудача по roi, is_successful, close_reason.
    Возвращает dict с ключами: success (bool), roi (float), reason (str).
    """
    roi = trade.get('roi')
    if roi is None and trade.get('pnl') is not None and trade.get('position_size_usdt'):
        try:
            roi = float(trade['pnl']) / float(trade['position_size_usdt']) * 100.0
        except (TypeError, ZeroDivisionError):
            roi = 0.0
    roi = float(roi) if roi is not None else 0.0
    is_ok = trade.get('is_successful', False)
    if isinstance(is_ok, (int, float)):
        is_ok = bool(is_ok)
    if not is_ok and roi == 0.0:
        is_ok = roi > 0
    reason = trade.get('close_reason') or ''
    return {'success': is_ok, 'roi': roi, 'reason': reason, 'symbol': trade.get('symbol', '')}


def run_prii_trades_analysis(
    days_back: int = 7,
    min_trades_per_symbol: int = 2,
    adjust_params: bool = True,
) -> Dict[str, Any]:
    """
    Анализ закрытых сделок и обновление параметров ПРИИ по монетам.
    Вызывать при включённом ПРИИ: по расписанию и/или после закрытия сделки.
    Пишет только в full_ai_config (БД) и full_ai_coin_params. Не трогает
    пользовательский конфиг и individual_coin_settings.
    """
    if not _is_prii_enabled():
        logger.debug("[ПРИИ learner] Пропуск: full_ai_control выключен")
        return {'success': True, 'skipped': True, 'reason': 'PRII disabled'}
    try:
        from bot_engine.bots_database import get_bots_database
        from bots_modules.imports_and_globals import (
            get_effective_auto_bot_config,
            load_full_ai_config_from_db,
            save_full_ai_config_to_db,
        )
        db = get_bots_database()
        trades = db.get_bot_trades_history(
            status='CLOSED',
            days_back=days_back,
            limit=500,
        )
        if not trades:
            return {'success': True, 'analyzed': 0, 'updated_symbols': []}
        by_symbol: Dict[str, List[Dict]] = {}
        for t in trades:
            sym = (t.get('symbol') or '').upper()
            if not sym:
                continue
            if sym not in by_symbol:
                by_symbol[sym] = []
            by_symbol[sym].append(_evaluate_trade(t))
        updated = []
        for symbol, evals in by_symbol.items():
            if len(evals) < min_trades_per_symbol:
                continue
            wins = sum(1 for e in evals if e.get('success'))
            total = len(evals)
            win_rate = wins / total if total else 0
            avg_roi = sum(e.get('roi', 0) for e in evals) / total if total else 0
            current = db.load_full_ai_coin_params(symbol) or {}
            if not adjust_params:
                continue
            changed = False
            tp = current.get('take_profit_percent')
            sl = current.get('max_loss_percent')
            prii_global = get_effective_auto_bot_config()
            if tp is None:
                tp = prii_global.get('take_profit_percent') or 15
            if sl is None:
                sl = prii_global.get('max_loss_percent') or 10
            tp_f = float(tp)
            sl_f = float(sl)
            if win_rate < 0.4 and total >= 3:
                tp_f = max(5, tp_f - 2)
                sl_f = min(20, sl_f + 2)
                changed = True
            elif win_rate >= 0.6 and total >= 3:
                tp_f = min(50, tp_f + 2)
                sl_f = max(5, sl_f - 1)
                changed = True
            if changed:
                new_params = {**current, 'take_profit_percent': round(tp_f, 1), 'max_loss_percent': round(sl_f, 1)}
                if db.save_full_ai_coin_params(symbol, new_params):
                    updated.append(symbol)
                    logger.info(
                        f"[ПРИИ learner] {symbol}: win_rate={win_rate:.2f}, n={total} -> TP={tp_f:.1f}%, SL={sl_f:.1f}%"
                    )
        return {
            'success': True,
            'analyzed': len(trades),
            'symbols_evaluated': len(by_symbol),
            'updated_symbols': updated,
        }
    except Exception as e:
        logger.exception(f"[ПРИИ learner] Ошибка: {e}")
        return {'success': False, 'error': str(e)}


def run_prii_trades_analysis_after_close(symbol: Optional[str] = None):
    """
    Короткий запуск анализа после закрытия сделки (например по одной монете или все за 1 день).
    Вызывать из bot_class после успешного закрытия позиции в режиме ПРИИ.
    """
    run_prii_trades_analysis(days_back=1, min_trades_per_symbol=1, adjust_params=True)
