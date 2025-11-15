#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверяет, что AITrainer формирует полный набор индивидуальных настроек.
"""

from bot_engine.ai.ai_trainer import AITrainer


def test_ai_trainer_builds_full_individual_settings():
    trainer = AITrainer()

    coin_rsi_params = {
        'oversold': 28,
        'overbought': 72,
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
        'win_rate': 82.5,
        'rating': 73.4,
        'total_pnl': 123.45,
        'trades_count': 12,
    }

    settings = trainer._build_individual_settings(
        coin_rsi_params=coin_rsi_params,
        risk_params=risk_params,
        filter_params=filter_params,
        trend_params=trend_params,
        maturity_params=maturity_params,
        ai_meta=ai_meta,
    )

    required_keys = [
        'rsi_long_threshold',
        'rsi_short_threshold',
        'rsi_time_filter_enabled',
        'exit_scam_enabled',
        'trend_detection_enabled',
        'enable_maturity_check',
        'ai_trained',
        'ai_win_rate',
        'ai_rating',
        'ai_trades_count',
        'ai_total_pnl',
    ]
    for key in required_keys:
        assert key in settings, f"missing key {key}"

    assert settings['ai_trained'] is True
    assert settings['ai_win_rate'] == 82.5
    assert settings['ai_rating'] == 73.4
    assert settings['ai_trades_count'] == 12
    assert settings['ai_total_pnl'] == 123.45
    assert settings['rsi_time_filter_candles'] == 6
    assert settings['exit_scam_candles'] == 8

