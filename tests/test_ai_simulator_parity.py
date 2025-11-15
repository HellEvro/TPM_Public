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

