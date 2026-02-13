"""
Унифицированные функции защитных механизмов (stop-loss, break-even, trailing, max_position_hours)

Модуль извлечён из логики NewTradingBot, чтобы AI-симуляции и оптимизаторы
использовали те же правила, что и реальный торговый бот.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple
import math
import time


BREAK_EVEN_FEE_MULTIPLIER = 2.5


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class ProtectionState:
    """
    Хранит состояние защитных механизмов для позиции.

    entry_time — UNIX timestamp (секунды), quantity — количество монет.
    notional_usdt позволяет вычислить количество при отсутствии quantity.
    """

    position_side: str
    entry_price: float
    entry_time: Optional[float] = None
    quantity: Optional[float] = None
    notional_usdt: Optional[float] = None
    max_profit_percent: float = 0.0
    break_even_activated: bool = False
    break_even_stop_set: bool = False  # Флаг, что break-even стоп уже установлен на бирже (устанавливается один раз)
    break_even_stop_price: Optional[float] = None
    trailing_active: bool = False
    trailing_reference_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    trailing_take_profit_price: Optional[float] = None
    trailing_last_update_ts: float = 0.0


@dataclass
class ProtectionDecision:
    should_close: bool
    reason: Optional[str]
    state: ProtectionState
    profit_percent: float


def _get_quantity(state: ProtectionState) -> Optional[float]:
    quantity = _safe_float(state.quantity)
    if quantity and quantity > 0:
        return quantity

    entry_price = _safe_float(state.entry_price)
    notional = _safe_float(state.notional_usdt)
    if entry_price and entry_price > 0 and notional and notional > 0:
        return notional / entry_price
    return None


def _calculate_break_even_stop(
    state: ProtectionState,
    current_price: Optional[float],
    realized_pnl_usdt: float = 0.0,
) -> Optional[float]:
    entry_price = _safe_float(state.entry_price)
    position_side = (state.position_side or '').upper()
    if entry_price is None or entry_price == 0 or position_side not in ('LONG', 'SHORT'):
        return None

    quantity = _get_quantity(state)
    if not quantity or quantity <= 0:
        return entry_price

    fee_usdt = abs(_safe_float(realized_pnl_usdt, 0.0) or 0.0)
    if fee_usdt <= 0:
        return entry_price

    protected_profit_usdt = fee_usdt * BREAK_EVEN_FEE_MULTIPLIER
    protected_profit_per_coin = protected_profit_usdt / quantity if quantity else 0.0
    if protected_profit_per_coin <= 0:
        return entry_price

    stop_price = entry_price
    price = _safe_float(current_price)

    if position_side == 'LONG':
        stop_price = entry_price + protected_profit_per_coin
        if price is not None:
            stop_price = min(stop_price, price)
        stop_price = max(stop_price, entry_price)
    else:
        stop_price = entry_price - protected_profit_per_coin
        if price is not None:
            stop_price = max(stop_price, price)
        stop_price = min(stop_price, entry_price)

    return stop_price


def _update_trailing(
    state: ProtectionState,
    config: Dict[str, Any],
    current_price: float,
    profit_percent: float,
    now_ts: float,
) -> Tuple[ProtectionState, Optional[str]]:
    """
    Обновляет trailing-stop/take состояние и определяет, нужно ли закрыть позицию.
    Возвращает (updated_state, closing_reason).
    """

    activation = _safe_float(config.get('trailing_stop_activation'), 0.0) or 0.0
    stop_distance = max(0.0, _safe_float(config.get('trailing_stop_distance'), 0.0) or 0.0)
    take_distance = max(0.0, _safe_float(config.get('trailing_take_distance'), 0.0) or 0.0)
    update_interval = max(0.0, _safe_float(config.get('trailing_update_interval'), 0.0) or 0.0)
    position_side = (state.position_side or '').upper()

    if stop_distance <= 0 or position_side not in ('LONG', 'SHORT'):
        state.trailing_active = False
        state.trailing_reference_price = None
        state.trailing_stop_price = None
        return state, None

    entry_price = _safe_float(state.entry_price, current_price)
    current_price = _safe_float(current_price)
    if current_price is None or entry_price is None or entry_price <= 0:
        return state, None

    if activation > 0 and profit_percent < activation and not state.trailing_active:
        state.trailing_reference_price = _safe_float(state.trailing_reference_price, entry_price)
        return state, None

    if not state.trailing_active:
        state.trailing_active = True
        state.trailing_reference_price = current_price
    else:
        reference = _safe_float(state.trailing_reference_price, entry_price)
        if position_side == 'LONG':
            reference = max(reference or entry_price, current_price)
        else:
            reference = min(reference or entry_price, current_price)
        state.trailing_reference_price = reference

    reference_price = _safe_float(state.trailing_reference_price, entry_price)
    stop_price = None
    if position_side == 'LONG':
        stop_price = reference_price * (1 - stop_distance / 100.0)
        stop_price = max(stop_price, entry_price)
        if state.break_even_stop_price is not None:
            stop_price = max(stop_price, state.break_even_stop_price)
    else:
        stop_price = reference_price * (1 + stop_distance / 100.0)
        stop_price = min(stop_price, entry_price)
        if state.break_even_stop_price is not None:
            stop_price = min(stop_price, state.break_even_stop_price)

    stop_price = _safe_float(stop_price)
    previous_stop = _safe_float(state.trailing_stop_price)
    tolerance = 1e-8

    should_update_stop = False
    if position_side == 'LONG':
        if stop_price is not None and (previous_stop is None or stop_price > previous_stop + tolerance):
            should_update_stop = True
    else:
        if stop_price is not None and (previous_stop is None or stop_price < previous_stop - tolerance):
            should_update_stop = True

    can_update_now = update_interval <= 0 or (now_ts - (state.trailing_last_update_ts or 0.0)) >= update_interval

    if should_update_stop and can_update_now:
        state.trailing_stop_price = stop_price
        state.trailing_last_update_ts = now_ts
    elif state.trailing_stop_price is None:
        state.trailing_stop_price = stop_price

    tp_price = None
    if take_distance > 0 and reference_price:
        if position_side == 'LONG':
            tp_price = reference_price * (1 - take_distance / 100.0)
            tp_price = max(tp_price, entry_price)
            if stop_price is not None:
                tp_price = max(tp_price, stop_price + tolerance)
        else:
            tp_price = reference_price * (1 + take_distance / 100.0)
            tp_price = min(tp_price, entry_price)
            if stop_price is not None:
                tp_price = min(tp_price, stop_price - tolerance)
        if tp_price is not None and (state.trailing_take_profit_price is None):
            state.trailing_take_profit_price = tp_price
        elif tp_price is not None:
            if position_side == 'LONG' and tp_price > state.trailing_take_profit_price + tolerance:
                state.trailing_take_profit_price = tp_price
            elif position_side == 'SHORT' and tp_price < state.trailing_take_profit_price - tolerance:
                state.trailing_take_profit_price = tp_price

    effective_stop = state.trailing_stop_price if state.trailing_stop_price is not None else previous_stop
    if effective_stop is None:
        return state, None

    if position_side == 'LONG' and current_price <= effective_stop:
        return state, f'TRAILING_STOP_{profit_percent:.2f}%'
    if position_side == 'SHORT' and current_price >= effective_stop:
        return state, f'TRAILING_STOP_{profit_percent:.2f}%'

    return state, None


def evaluate_protections(
    current_price: float,
    config: Dict[str, Any],
    state: ProtectionState,
    realized_pnl: float = 0.0,
    now_ts: Optional[float] = None,
) -> ProtectionDecision:
    """
    Основная функция проверки защитных механизмов.

    Args:
        current_price: Текущая цена.
        config: Конфигурация (глобальная или merged).
        state: Текущее состояние защит.
        realized_pnl: Реализованный PnL (используется для рассчёта break-even стопа).
        now_ts: UNIX timestamp; если None, используется time.time().
    """

    state = replace(state)
    now_ts = now_ts or time.time()

    entry_price = _safe_float(state.entry_price)
    current_price = _safe_float(current_price)
    position_side = (state.position_side or '').upper()

    if entry_price is None or entry_price == 0 or current_price is None or position_side not in ('LONG', 'SHORT'):
        return ProtectionDecision(False, None, state, 0.0)

    if position_side == 'LONG':
        profit_percent = ((current_price - entry_price) / entry_price) * 100
    else:
        profit_percent = ((entry_price - current_price) / entry_price) * 100

    max_loss_percent = _safe_float(
        config.get('max_loss_percent', config.get('stop_loss_percent', 15.0)),
        15.0
    ) or 0.0

    if max_loss_percent > 0 and profit_percent <= -max_loss_percent:
        return ProtectionDecision(True, f'STOP_LOSS_{profit_percent:.2f}%', state, profit_percent)

    state.max_profit_percent = max(state.max_profit_percent or 0.0, profit_percent)

    # Выход по достижении заданного процента прибыли (вкл. close_at_profit_enabled, порог take_profit_percent)
    # Минимум 1% — чтобы не закрывать при первом же выходе в плюс (шум/ошибка конфига)
    close_at_profit_enabled = config.get('close_at_profit_enabled', True)
    take_profit_percent = _safe_float(config.get('take_profit_percent'), 0.0) or 0.0
    if take_profit_percent > 0 and take_profit_percent < 1.0:
        take_profit_percent = 1.0
    if close_at_profit_enabled and take_profit_percent >= 1.0 and profit_percent >= take_profit_percent:
        return ProtectionDecision(True, f'TAKE_PROFIT_{profit_percent:.2f}%', state, profit_percent)

    max_position_hours = _safe_float(config.get('max_position_hours'), 0.0) or 0.0
    if max_position_hours > 0 and state.entry_time:
        held_hours = (now_ts - state.entry_time) / 3600.0
        if held_hours >= max_position_hours:
            return ProtectionDecision(True, f'MAX_POSITION_HOURS_{held_hours:.1f}h', state, profit_percent)

    break_even_enabled = bool(config.get('break_even_protection', True))
    break_even_trigger = _safe_float(
        config.get('break_even_trigger_percent', config.get('break_even_trigger')),
        0.0
    ) or 0.0
    if break_even_trigger < 0:
        break_even_trigger = 0.0
    # Не активируем break-even при триггере < 1% — иначе стоп у входа срабатывает на первом откате
    if 0 < break_even_trigger < 1.0:
        break_even_trigger = 1.0

    if break_even_enabled and break_even_trigger >= 1.0:
        if not state.break_even_activated and profit_percent >= break_even_trigger:
            state.break_even_activated = True

        if state.break_even_activated:
            state.break_even_stop_price = _calculate_break_even_stop(
                state,
                current_price=current_price,
                realized_pnl_usdt=realized_pnl,
            )
            # Закрывать только когда мы у входа или в мелком минусе (<= 0.05%). При реальном минусе (>0.05%)
            # не считать это «безубытком» — пусть exit_wait_breakeven или стоп-лосс обработают.
            if profit_percent <= 0.05 and profit_percent >= -0.05:
                return ProtectionDecision(True, f'BREAK_EVEN_MAX_{state.max_profit_percent:.2f}%', state, profit_percent)
    else:
        state.break_even_activated = False
        state.break_even_stop_price = None

    state, trailing_reason = _update_trailing(state, config, current_price, profit_percent, now_ts)
    if trailing_reason:
        return ProtectionDecision(True, trailing_reason, state, profit_percent)

    return ProtectionDecision(False, None, state, profit_percent)

