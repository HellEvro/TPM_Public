#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke-тест AITrainer: прогоняет обучение на 2 монетах и проверяет,
что индивидуальные настройки полностью записываются.

Запускать из корня проекта:
    python scripts/run_ai_trainer_smoke.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bot_engine.ai.ai_trainer import AITrainer

EXPECTED_KEYS = [
    'rsi_long_threshold',
    'rsi_short_threshold',
    'rsi_exit_long_with_trend',
    'rsi_exit_long_against_trend',
    'rsi_exit_short_with_trend',
    'rsi_exit_short_against_trend',
    'max_loss_percent',
    'take_profit_percent',
    'trailing_stop_activation',
    'trailing_stop_distance',
    'trailing_take_distance',
    'trailing_update_interval',
    'break_even_trigger',
    'break_even_protection',
    'max_position_hours',
    'rsi_time_filter_enabled',
    'rsi_time_filter_candles',
    'rsi_time_filter_upper',
    'rsi_time_filter_lower',
    'exit_scam_enabled',
    'exit_scam_candles',
    'exit_scam_single_candle_percent',
    'exit_scam_multi_candle_count',
    'exit_scam_multi_candle_percent',
    'trend_detection_enabled',
    'avoid_down_trend',
    'avoid_up_trend',
    'trend_analysis_period',
    'trend_price_change_threshold',
    'trend_candles_threshold',
    'enable_maturity_check',
    'min_candles_for_maturity',
    'min_rsi_low',
    'max_rsi_high',
    'ai_trained',
    'ai_win_rate',
    'ai_rating',
    'ai_trained_at',
    'ai_trades_count',
    'ai_total_pnl',
]


def _build_params(
    oversold: float,
    overbought: float,
    win_rate: float,
    trades_count: int,
    rating: float,
    pnl: float,
):
    coin_rsi_params = {
        'oversold': oversold,
        'overbought': overbought,
        'exit_long_with_trend': 66,
        'exit_long_against_trend': 60,
        'exit_short_with_trend': 34,
        'exit_short_against_trend': 38,
    }
    risk_params = {
        'max_loss_percent': 12.5,
        'take_profit_percent': 25.0,
        'trailing_stop_activation': 18.0,
        'trailing_stop_distance': 10.0,
        'trailing_take_distance': 0.6,
        'trailing_update_interval': 3.0,
        'break_even_trigger': 80.0,
        'break_even_protection': True,
        'max_position_hours': 48,
    }
    filter_params = {
        'rsi_time_filter': {
            'enabled': True,
            'candles': 6,
            'upper': 65,
            'lower': 35,
        },
        'exit_scam': {
            'enabled': True,
            'candles': 8,
            'single_candle_percent': 12.0,
            'multi_candle_count': 4,
            'multi_candle_percent': 40.0,
        },
    }
    trend_params = {
        'trend_detection_enabled': True,
        'avoid_down_trend': True,
        'avoid_up_trend': False,
        'trend_analysis_period': 30,
        'trend_price_change_threshold': 7.5,
        'trend_candles_threshold': 70,
    }
    maturity_params = {
        'enable_maturity_check': True,
        'min_candles_for_maturity': 400,
        'min_rsi_low': 35,
        'max_rsi_high': 65,
    }
    ai_meta = {
        'win_rate': win_rate,
        'rating': rating,
        'total_pnl': pnl,
        'trades_count': trades_count,
    }
    return {
        'coin_rsi_params': coin_rsi_params,
        'risk_params': risk_params,
        'filter_params': filter_params,
        'trend_params': trend_params,
        'maturity_params': maturity_params,
        'ai_meta': ai_meta,
    }


def main() -> None:
    trainer = AITrainer()
    symbols = {
        'SMOKECOIN1': _build_params(oversold=28, overbought=72, win_rate=82.5, trades_count=12, rating=73.4, pnl=123.45),
        'SMOKECOIN2': _build_params(oversold=30, overbought=68, win_rate=70.0, trades_count=9, rating=60.0, pnl=42.0),
    }

    for symbol, params in symbols.items():
        settings = trainer._build_individual_settings(**params)
        missing = EXPECTED_KEYS and [key for key in EXPECTED_KEYS if key not in settings]
        status = "✅" if not missing else f"⚠️ отсутствуют поля: {missing}"
        trained_at = settings.get('ai_trained_at')
        try:
            if trained_at:
                datetime.fromisoformat(trained_at)
        except Exception:
            status = f"⚠️ неверная дата ai_trained_at ({trained_at})"
        print(f"[SMOKE] {symbol}: {len(settings)} полей, ai_trained={settings.get('ai_trained')}, статус: {status}")

    print("[SMOKE] Готово.")


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUTF8', '1')
    main()

