#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke-тесты для AITrainer: проверяем, что индивидуальные настройки
формируются полностью и независимы для нескольких монет.
"""

from datetime import datetime

from bot_engine.ai.ai_trainer import AITrainer

EXPECTED_KEYS = {
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
}


def _build_params(
    *,
    oversold,
    overbought,
    win_rate,
    trades_count,
    rating,
    pnl,
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


def test_ai_trainer_builds_full_individual_settings():
    trainer = AITrainer()
    params = _build_params(
        oversold=28,
        overbought=72,
        win_rate=82.5,
        trades_count=12,
        rating=73.4,
        pnl=123.45,
    )
    settings = trainer._build_individual_settings(**params)

    assert set(settings.keys()) == EXPECTED_KEYS
    assert settings['ai_trained'] is True
    assert settings['ai_win_rate'] == 82.5
    assert settings['ai_rating'] == 73.4
    assert settings['ai_trades_count'] == 12
    assert settings['ai_total_pnl'] == 123.45
    assert settings['rsi_time_filter_candles'] == 6
    assert settings['exit_scam_candles'] == 8
    # ai_trained_at должен быть валидным ISO 8601
    datetime.fromisoformat(settings['ai_trained_at'])


def test_ai_trainer_smoke_two_coins_settings_integrity():
    trainer = AITrainer()
    settings_a = trainer._build_individual_settings(
        **_build_params(
            oversold=30,
            overbought=70,
            win_rate=75.0,
            trades_count=20,
            rating=55.0,
            pnl=45.6,
        )
    )
    settings_b = trainer._build_individual_settings(
        **_build_params(
            oversold=25,
            overbought=78,
            win_rate=88.8,
            trades_count=9,
            rating=81.5,
            pnl=210.0,
        )
    )

    assert set(settings_a.keys()) == EXPECTED_KEYS
    assert set(settings_b.keys()) == EXPECTED_KEYS
    assert settings_a != settings_b  # значения независимы для монет
    datetime.fromisoformat(settings_a['ai_trained_at'])
    datetime.fromisoformat(settings_b['ai_trained_at'])

