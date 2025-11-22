#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parity-тест: проверяет, что NewTradingBot и AI-бэктестер строят одинаковый ProtectionState.

Это гарантирует, что evaluate_protections() получает идентичные входные данные
как в боевом контуре, так и в AI-симуляциях.
"""

import math
from datetime import datetime

from bots_modules.bot_class import NewTradingBot
from bot_engine.ai.ai_backtester_new import _create_protection_state
from bot_engine.protections import evaluate_protections


TEST_CONFIG = {
    'max_loss_percent': 7.0,
    'take_profit_percent': 18.0,
    'trailing_stop_activation': 4.0,
    'trailing_stop_distance': 1.5,
    'trailing_take_distance': 2.0,
    'trailing_update_interval': 0.0,
    'break_even_protection': True,
    'break_even_trigger_percent': 3.5,
    'max_position_hours': 48,
}

SIMULATED_REALIZED_PNL = 0.75
TIME_STEP_SECONDS = 1_800


def _build_bot_for_test(entry_price: float, notional_usdt: float, direction: str, entry_ts: float) -> NewTradingBot:
    position_coins = notional_usdt / entry_price
    config = {
        'entry_price': entry_price,
        'position_side': direction,
        'position_size_coins': position_coins,
        'position_start_time': datetime.fromtimestamp(entry_ts).isoformat(),
        'volume_value': notional_usdt,
    }
    return NewTradingBot(symbol='TESTCOIN', config=config, exchange=None)


def _simulate_ai_sequence(state, prices, config, start_ts):
    results = []
    current_state = state
    for idx, price in enumerate(prices):
        now_ts = start_ts + idx * TIME_STEP_SECONDS
        decision = evaluate_protections(
            current_price=price,
            config=config,
            state=current_state,
            realized_pnl=SIMULATED_REALIZED_PNL,
            now_ts=now_ts,
        )
        results.append((decision.should_close, decision.reason))
        current_state = decision.state
    return results


def _simulate_bot_sequence(bot, prices, config, start_ts):
    results = []
    state = bot._build_protection_state()
    for idx, price in enumerate(prices):
        now_ts = start_ts + idx * TIME_STEP_SECONDS
        decision = evaluate_protections(
            current_price=price,
            config=config,
            state=state,
            realized_pnl=SIMULATED_REALIZED_PNL,
            now_ts=now_ts,
        )
        bot._apply_protection_state(decision.state)
        state = bot._build_protection_state()
        results.append((decision.should_close, decision.reason))
    return results


def test_protection_state_parity_between_bot_and_ai_backtester():
    entry_price = 123.45
    notional_usdt = 246.9
    entry_ts = 1_717_171_700  # seconds

    bot = _build_bot_for_test(entry_price, notional_usdt, 'LONG', entry_ts)
    bot_state = bot._build_protection_state()

    ai_state = _create_protection_state('LONG', entry_price, notional_usdt, entry_ts * 1000)  # ms timestamp

    assert bot_state.position_side == ai_state.position_side == 'LONG'
    assert math.isclose(bot_state.entry_price, ai_state.entry_price, rel_tol=1e-9)
    assert math.isclose(bot_state.entry_time, ai_state.entry_time, rel_tol=1e-9)
    assert math.isclose(bot_state.quantity, ai_state.quantity, rel_tol=1e-9)
    assert math.isclose(bot_state.notional_usdt, ai_state.notional_usdt, rel_tol=1e-9)


def _assert_sequence_parity(direction: str, prices):
    entry_price = prices[0]
    notional_usdt = 500.0
    entry_ts = 1_717_200_000

    bot = _build_bot_for_test(entry_price, notional_usdt, direction, entry_ts)
    ai_state = _create_protection_state(direction, entry_price, notional_usdt, entry_ts * 1000)

    ai_results = _simulate_ai_sequence(ai_state, prices, TEST_CONFIG, entry_ts)
    bot_results = _simulate_bot_sequence(bot, prices, TEST_CONFIG, entry_ts)

    assert ai_results == bot_results
    assert any(flag for flag, _ in ai_results), "ожидаем хотя бы одно срабатывание защиты"


def test_full_sequence_parity_long_and_short():
    long_prices = [100.0, 104.0, 109.0, 108.5, 106.0, 101.0, 95.0, 90.0]
    short_prices = [150.0, 143.0, 138.0, 132.0, 136.0, 145.0, 155.0]

    _assert_sequence_parity('LONG', long_prices)
    _assert_sequence_parity('SHORT', short_prices)

