# -*- coding: utf-8 -*-
"""
FullAI Scoring: очки и рейтинг комбинаций параметров.

- Успешная сделка: +1 очко, серия побед +1.
- Убыточная сделка: -1 очко, серия обнуляется.
- Комбинации ранжируются по (current_streak, best_streak, score). Топовая используется первой.
- Если топовая перестаёт работать — по иерархии пробуются следующие по очкам.
- Если все комбинации провальные (все в минусе / серии нулевые) — переобучение с нуля: очистка рейтинга и новая комбинация.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger('BOTS')

# Параметры, с которыми открыта текущая позиция по символу (для записи очков при закрытии)
_position_params: Dict[str, Dict[str, Any]] = {}
_scoring_lock = __import__('threading').Lock()


def _is_fullai_and_scoring_enabled() -> bool:
    try:
        from bots_modules.imports_and_globals import bots_data, bots_data_lock, get_effective_auto_bot_config
        with bots_data_lock:
            if not (bots_data.get('auto_bot_config') or {}).get('full_ai_control', False):
                return False
        ac = get_effective_auto_bot_config() or {}
        return ac.get('fullai_scoring_enabled', True) is not False
    except Exception:
        return False


def get_effective_params_for_symbol(symbol: str) -> Dict[str, Any]:
    """
    Возвращает параметры для символа: топовая комбинация из рейтинга по (current_streak, best_streak, score).
    Если рейтинг пуст — берёт из full_ai_coin_params и добавляет комбинацию в рейтинг.
    Сохраняет возвращённые params в _position_params[symbol], чтобы при закрытии сделки записать очки.
    """
    norm = (symbol or '').upper()
    if not norm:
        return {}
    if not _is_fullai_and_scoring_enabled():
        try:
            from bot_engine.storage import _get_bots_database
            db = _get_bots_database()
            params = db.load_full_ai_coin_params(norm)
            return dict(params) if params else {}
        except Exception:
            return {}
    try:
        from bot_engine.storage import _get_bots_database
        from bots_modules.imports_and_globals import get_effective_auto_bot_config
        db = _get_bots_database()
        top = db.fullai_leaderboard_top_params(norm)
        if top:
            with _scoring_lock:
                _position_params[norm] = dict(top)
            return dict(top)
        base = db.load_full_ai_coin_params(norm)
        if not base:
            base = get_effective_auto_bot_config() or {}
        base = {k: v for k, v in (base or {}).items() if v is not None}
        if not base:
            return {}
        db.fullai_leaderboard_add(norm, base)
        with _scoring_lock:
            _position_params[norm] = dict(base)
        return dict(base)
    except Exception as e:
        logger.debug("[FullAI Scoring] get_effective_params %s: %s", symbol, e)
        try:
            from bot_engine.storage import _get_bots_database
            params = _get_bots_database().load_full_ai_coin_params(norm)
            return dict(params) if params else {}
        except Exception:
            return {}


def record_trade_result(symbol: str, success: bool) -> None:
    """
    Вызывать при закрытии сделки по символу. success = (pnl >= 0).
    Начисляет +1 очко и увеличивает серию при успехе; -1 очко и обнуляет серию при минусе.
    Если по символу не было сохранённых params (не через get_effective_params) — пытается взять из full_ai_coin_params.
    Если все комбинации в рейтинге с нулевой серией и отрицательным счётом — очищает рейтинг и добавляет одну новую (переобучение).
    """
    norm = (symbol or '').upper()
    if not norm or not _is_fullai_and_scoring_enabled():
        return
    try:
        from bot_engine.storage import _get_bots_database
        from bots_modules.fullai_adaptive import mutate_params
        db = _get_bots_database()
        with _scoring_lock:
            params = _position_params.pop(norm, None)
        if not params:
            params = db.load_full_ai_coin_params(norm)
        if not params:
            return
        db.fullai_leaderboard_upsert(norm, params, success)
        entries = db.fullai_leaderboard_get(norm, limit=50)
        if not entries:
            return
        all_failing = all(
            (e.get('current_streak') or 0) == 0 and (e.get('score') or 0) < 0
            for e in entries
        )
        if all_failing and len(entries) >= 2:
            logger.info("[FullAI Scoring] %s: все комбинации в минусе — переобучение с нуля", norm)
            db.fullai_leaderboard_clear(norm)
            mutate_params(norm)
            new_params = db.load_full_ai_coin_params(norm)
            if new_params:
                db.fullai_leaderboard_add(norm, new_params)
    except Exception as e:
        logger.exception("[FullAI Scoring] record_trade_result %s: %s", symbol, e)


def get_leaderboard_for_api(symbol: str, limit: int = 20) -> list:
    """Для API/UI: список комбинаций с очками и сериями (без params_json в ответе при желании можно обрезать)."""
    norm = (symbol or '').upper()
    if not norm:
        return []
    try:
        from bot_engine.storage import _get_bots_database
        return _get_bots_database().fullai_leaderboard_get(norm, limit=limit)
    except Exception:
        return []
