# -*- coding: utf-8 -*-
"""
FullAI Adaptive: принудительная смена параметров при «мёртвом» периоде и виртуальная обкатка.

- Таймаут в свечах: если за N свечей нет ни одной сделки (ни виртуальной, ни реальной) — меняем
  параметры (начиная с RSI) и запускаем виртуальную серию.
- Виртуальные сделки: новые параметры тестируются виртуально на реальных данных; реальный ордер
  не выставляется. После N успешных виртуальных подряд разрешаем 1 реальную сделку.
- Если 1 из N виртуальных неудачная — подбираем новые параметры и снова N виртуальных.
- Если реальная сделка в минусе — снова меняем параметры и N виртуальных.
- Настройки: dead_candles, virtual_success_count, real_loss_to_retry, virtual_round_size, virtual_max_failures.
"""
import logging
import random
from typing import Dict, Any, Optional, List, Tuple
from copy import deepcopy

logger = logging.getLogger('BOTS')

# Состояние по символу (в памяти)
_state: Dict[str, Dict[str, Any]] = {}
_state_lock = __import__('threading').Lock()

# Виртуальные открытые позиции: symbol -> list of {direction, entry_price, entry_time, entry_candle_id}
_virtual_positions: Dict[str, List[Dict[str, Any]]] = {}

# Дефолты адаптивного блока (в свечах / штуках)
DEFAULT_ADAPTIVE = {
    'fullai_adaptive_enabled': False,
    'fullai_adaptive_dead_candles': 100,
    'fullai_adaptive_virtual_success_count': 3,
    'fullai_adaptive_real_loss_to_retry': 1,
    'fullai_adaptive_virtual_round_size': 3,
    'fullai_adaptive_virtual_max_failures': 0,
}


def _get_adaptive_config() -> Dict[str, Any]:
    """Читает конфиг FullAI и возвращает блок fullai_adaptive_* с дефолтами."""
    try:
        from bots_modules.imports_and_globals import get_effective_auto_bot_config
        cfg = get_effective_auto_bot_config() or {}
    except Exception:
        cfg = {}
    out = dict(DEFAULT_ADAPTIVE)
    for k in DEFAULT_ADAPTIVE:
        if k in cfg and cfg[k] is not None:
            out[k] = cfg[k]
    return out


def is_adaptive_enabled() -> bool:
    """FullAI включён и включён адаптивный режим (таймаут + виртуальная обкатка).
    Обкатка считается включённой, если явно fullai_adaptive_enabled=True ИЛИ
    задано «удачных виртуальных подряд» > 0 (тогда виртуальная обкатка нужна)."""
    try:
        from bots_modules.imports_and_globals import bots_data, bots_data_lock
        with bots_data_lock:
            ac = bots_data.get('auto_bot_config') or {}
        if not ac.get('full_ai_control', False):
            return False
    except Exception:
        return False
    ad = _get_adaptive_config()
    if ad.get('fullai_adaptive_enabled', False):
        return True
    # При Full AI: если «удачных виртуальных подряд» > 0 — обкатка по смыслу включена
    n = ad.get('fullai_adaptive_virtual_success_count') or 0
    try:
        if int(n) > 0:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _get_symbol_state(symbol: str, lock: bool = True) -> Dict[str, Any]:
    def _get():
        s = _state.get(symbol)
        if s is None:
            s = {
                'phase': 'virtual',
                'virtual_success_streak': 0,
                'virtual_failures_in_round': 0,
                'virtual_done_in_round': 0,
                'real_slots_left': 0,
                'candles_since_last_trade': 0,
                'last_candle_id': None,
                'real_losses_count': 0,
            }
            _state[symbol] = s
        return s
    if lock:
        with _state_lock:
            return _get()
    return _get()


def _reset_to_virtual(symbol: str, reason: str = ''):
    """Сброс в режим виртуальной серии (после мёртвого периода или после убыточной реальной)."""
    with _state_lock:
        s = _get_symbol_state(symbol, lock=False)
        s['phase'] = 'virtual'
        s['virtual_success_streak'] = 0
        s['virtual_failures_in_round'] = 0
        s['virtual_done_in_round'] = 0
        s['real_slots_left'] = 0
        s['real_losses_count'] = 0
    if reason:
        logger.info("[FullAI Adaptive] %s: сброс в виртуальный режим. %s", symbol, reason)


def mutate_params(symbol: str) -> bool:
    """
    Подбор новых параметров для монеты (сначала RSI, затем TP/SL и др.) и запись в full_ai_coin_params.
    Возвращает True если параметры изменены и сохранены.
    """
    norm = (symbol or '').upper()
    if not norm:
        return False
    try:
        from bot_engine.bots_database import get_bots_database
        from bots_modules.imports_and_globals import get_effective_auto_bot_config
        db = get_bots_database()
        current = db.load_full_ai_coin_params(norm) or {}
        global_cfg = get_effective_auto_bot_config() or {}
        # Базовые значения из глобального или дефолты
        base = {
            'rsi_long_threshold': int(current.get('rsi_long_threshold') or global_cfg.get('rsi_long_threshold') or 29),
            'rsi_short_threshold': int(current.get('rsi_short_threshold') or global_cfg.get('rsi_short_threshold') or 71),
            'take_profit_percent': float(current.get('take_profit_percent') or global_cfg.get('take_profit_percent') or 15),
            'max_loss_percent': float(current.get('max_loss_percent') or global_cfg.get('max_loss_percent') or 10),
        }
        # Мутация: небольшой случайный шаг в допустимых пределах (сначала RSI)
        new_params = dict(current)
        rsi_long = base['rsi_long_threshold']
        rsi_short = base['rsi_short_threshold']
        step = random.randint(-3, 3)
        new_params['rsi_long_threshold'] = max(20, min(35, rsi_long + step))
        step = random.randint(-3, 3)
        new_params['rsi_short_threshold'] = max(65, min(80, rsi_short + step))
        step_tp = random.uniform(-2, 2)
        new_params['take_profit_percent'] = round(max(8, min(40, base['take_profit_percent'] + step_tp)), 1)
        step_sl = random.uniform(-1.5, 1.5)
        new_params['max_loss_percent'] = round(max(5, min(25, base['max_loss_percent'] + step_sl)), 1)
        if db.save_full_ai_coin_params(norm, new_params):
            logger.info(
                "[FullAI Adaptive] %s: новые параметры (мутация) RSI long=%s short=%s, TP=%s%%, SL=%s%%",
                norm,
                new_params['rsi_long_threshold'],
                new_params['rsi_short_threshold'],
                new_params['take_profit_percent'],
                new_params['max_loss_percent'],
            )
            return True
    except Exception as e:
        logger.exception("[FullAI Adaptive] mutate_params %s: %s", symbol, e)
    return False


def on_candle_tick(symbol: str, candle_id: Any = None) -> None:
    """
    Вызывать при каждой новой свече по символу. Увеличивает счётчик свечей без сделок.
    Если достигнут fullai_adaptive_dead_candles — сбрасывает в виртуальный режим и мутирует параметры.
    """
    if not is_adaptive_enabled():
        return
    norm = (symbol or '').upper()
    if not norm:
        return
    ad = _get_adaptive_config()
    dead = int(ad.get('fullai_adaptive_dead_candles') or 100)
    with _state_lock:
        s = _get_symbol_state(norm, lock=False)
        if candle_id is not None and s.get('last_candle_id') == candle_id:
            return
        s['last_candle_id'] = candle_id
        s['candles_since_last_trade'] = s.get('candles_since_last_trade', 0) + 1
        candles_since = s['candles_since_last_trade']
    if candles_since >= dead:
        _reset_to_virtual(norm, "Мёртвый период: %s свечей без сделок (лимит %s)" % (candles_since, dead))
        mutate_params(norm)
        with _state_lock:
            _get_symbol_state(norm)['candles_since_last_trade'] = 0


def on_trade_open(symbol: str) -> None:
    """Вызывать при открытии любой сделки (виртуальной или реальной) — сбрасывает счётчик свечей."""
    norm = (symbol or '').upper()
    if not norm:
        return
    with _state_lock:
        s = _get_symbol_state(norm, lock=False)
        s['candles_since_last_trade'] = 0


def get_next_action(symbol: str, decision_allowed: bool) -> str:
    """
    Определяет, разрешать ли реальный вход или только виртуальный.
    Возвращает: "real_open" | "virtual_open" | "none".
    - real_open: открыть реальную сделку (после N успешных виртуальных).
    - virtual_open: решение «вход» есть, но открываем только виртуально.
    - none: не открывать (или решение «не входить»).
    """
    if not decision_allowed:
        return "none"
    if not is_adaptive_enabled():
        return "real_open"
    norm = (symbol or '').upper()
    if not norm:
        return "none"
    ad = _get_adaptive_config()
    need_success = int(ad.get('fullai_adaptive_virtual_success_count') or 3)
    if need_success <= 0:
        return "real_open"  # виртуальная обкатка выключена (0) — сразу реальные сделки
    with _state_lock:
        s = _get_symbol_state(norm, lock=False)
        phase = s.get('phase', 'virtual')
        streak = s.get('virtual_success_streak', 0)
        real_slots = s.get('real_slots_left', 0)
    if phase == 'real_allowed' and real_slots > 0:
        with _state_lock:
            s = _get_symbol_state(norm, lock=False)
            s['real_slots_left'] = 0
        return "real_open"
    if phase == 'virtual':
        return "virtual_open"
    return "virtual_open"


def record_virtual_open(symbol: str, direction: str, entry_price: float, entry_time: float = None) -> None:
    """Записать открытие виртуальной позиции."""
    import time
    norm = (symbol or '').upper()
    if not norm:
        return
    on_trade_open(norm)
    with _state_lock:
        if norm not in _virtual_positions:
            _virtual_positions[norm] = []
        _virtual_positions[norm].append({
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': entry_time or time.time(),
        })


def _close_virtual_position(symbol: str, success: bool) -> None:
    """Учёт закрытия одной виртуальной позиции: обновляет streak и счётчики раунда."""
    ad = _get_adaptive_config()
    round_size = int(ad.get('fullai_adaptive_virtual_round_size') or 3)
    max_failures = int(ad.get('fullai_adaptive_virtual_max_failures') or 0)
    need_success = int(ad.get('fullai_adaptive_virtual_success_count') or 3)
    do_mutate = False
    with _state_lock:
        s = _get_symbol_state(symbol, lock=False)
        s['virtual_done_in_round'] = s.get('virtual_done_in_round', 0) + 1
        done = s['virtual_done_in_round']
        if success:
            s['virtual_success_streak'] = s.get('virtual_success_streak', 0) + 1
            streak = s['virtual_success_streak']
            if streak >= need_success:
                s['phase'] = 'real_allowed'
                s['real_slots_left'] = 1
                s['virtual_success_streak'] = 0
                s['virtual_done_in_round'] = 0
                s['virtual_failures_in_round'] = 0
                logger.info("[FullAI Adaptive] %s: %s виртуальных успешны → разрешена 1 реальная сделка", symbol, need_success)
        else:
            s['virtual_failures_in_round'] = s.get('virtual_failures_in_round', 0) + 1
            fails = s['virtual_failures_in_round']
            s['virtual_success_streak'] = 0
            if fails > max_failures or done >= round_size:
                s['virtual_done_in_round'] = 0
                s['virtual_failures_in_round'] = 0
                do_mutate = True
            else:
                do_mutate = False
    if do_mutate:
        logger.info("[FullAI Adaptive] %s: виртуальная неудача в серии → подбор новых параметров", symbol)
        mutate_params(symbol)


def record_virtual_close(symbol: str, success: bool) -> None:
    """Вызывать после закрытия виртуальной позиции (по TP/SL или решению ИИ)."""
    norm = (symbol or '').upper()
    if not norm:
        return
    _close_virtual_position(norm, success)


def process_virtual_positions(
    symbol: str,
    candles: List[Dict],
    current_price: float,
    fullai_config: Dict[str, Any],
    coin_params: Dict[str, Any],
) -> None:
    """
    Проверить виртуальные позиции по символу: вычислить PnL, вызвать get_ai_exit_decision;
    при close_now — закрыть виртуальную позицию и записать результат (record_virtual_close).
    Вызывать из цикла бота при каждой итерации по символу.
    """
    norm = (symbol or '').upper()
    if not norm or not is_adaptive_enabled():
        return
    with _state_lock:
        positions = list(_virtual_positions.get(norm, []))
    if not positions or not candles:
        return
    try:
        from bot_engine.ai.ai_integration import get_ai_exit_decision
    except ImportError:
        return
    tp = float(fullai_config.get('take_profit_percent') or coin_params.get('take_profit_percent') or 15)
    sl = float(fullai_config.get('max_loss_percent') or coin_params.get('max_loss_percent') or 10)
    to_remove = []
    for i, pos in enumerate(positions):
        entry = pos.get('entry_price') or 0
        direction = (pos.get('direction') or 'LONG').upper()
        if entry <= 0:
            to_remove.append(i)
            continue
        if direction == 'LONG':
            pnl_percent = (current_price - entry) / entry * 100.0
        else:
            pnl_percent = (entry - current_price) / entry * 100.0
        fake_position = {'side': direction, 'entry_price': entry}
        decision = get_ai_exit_decision(
            symbol=norm,
            position=fake_position,
            candles=candles,
            pnl_percent=pnl_percent,
            prii_config=fullai_config,
            coin_params=coin_params,
        )
        close_now = decision.get('close_now', False) or pnl_percent >= tp or pnl_percent <= -sl
        if close_now:
            success = pnl_percent >= 0
            record_virtual_close(norm, success)
            to_remove.append(i)
    with _state_lock:
        if norm in _virtual_positions:
            for idx in reversed(to_remove):
                if 0 <= idx < len(_virtual_positions[norm]):
                    _virtual_positions[norm].pop(idx)
            if not _virtual_positions[norm]:
                del _virtual_positions[norm]


def record_real_close(symbol: str, pnl_percent: float) -> None:
    """
    Вызывать при закрытии реальной сделки. Если pnl < 0 — сброс в виртуальный режим и мутация параметров.
    """
    norm = (symbol or '').upper()
    if not norm:
        return
    if not is_adaptive_enabled():
        return
    ad = _get_adaptive_config()
    loss_to_retry = int(ad.get('fullai_adaptive_real_loss_to_retry') or 1)
    with _state_lock:
        s = _get_symbol_state(norm, lock=False)
        s['real_slots_left'] = 0
        s['phase'] = 'virtual'
        s['virtual_success_streak'] = 0
        s['virtual_done_in_round'] = 0
        s['virtual_failures_in_round'] = 0
        if pnl_percent < 0:
            s['real_losses_count'] = s.get('real_losses_count', 0) + 1
            if s['real_losses_count'] >= loss_to_retry:
                need_reset = True
            else:
                need_reset = False
        else:
            s['real_losses_count'] = 0
            need_reset = False
    if pnl_percent < 0 and need_reset:
        _reset_to_virtual(norm, "Реальная сделка в минусе (%.2f%%), подбор новых параметров" % pnl_percent)
        mutate_params(norm)


def get_adaptive_settings_for_api() -> Dict[str, Any]:
    """Текущие настройки адаптивного блока для API/UI."""
    return _get_adaptive_config()
