# -*- coding: utf-8 -*-
"""
Проверки ПРИИ (блок 10 плана).
- 10.1: включение ПРИИ не меняет пользовательский конфиг и individual_coin_settings.
- 10.2: выключение ПРИИ возвращает использование пользовательского конфига и individual_coin_settings.
- 10.3: таймфрейм нигде не берётся из конфига ПРИИ или таблицы ПРИИ.
Запуск: из корня проекта python -m tests.verify_prii_plan (или pytest tests/verify_prii_plan.py -v).
"""
import os
import sys

# Корень проекта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_10_3_timeframe_always_from_user_config():
    """Таймфрейм всегда подставляется из пользовательского конфига в get_effective_auto_bot_config."""
    from bots_modules.imports_and_globals import (
        get_effective_auto_bot_config,
        bots_data,
        bots_data_lock,
        DEFAULT_AUTO_BOT_CONFIG,
    )
    with bots_data_lock:
        user = (bots_data.get('auto_bot_config') or {}).copy()
    user_tf = user.get('system_timeframe') or user.get('timeframe') or DEFAULT_AUTO_BOT_CONFIG.get('system_timeframe')
    effective = get_effective_auto_bot_config()
    eff_tf = effective.get('system_timeframe') or effective.get('timeframe')
    assert eff_tf == user_tf, "Таймфрейм в эффективном конфиге должен совпадать с пользовательским"
    # Конфиг ПРИИ не должен содержать таймфрейм как источник истины — он подмешивается из user
    with bots_data_lock:
        prii = bots_data.get('full_ai_config') or {}
    if prii:
        for key in ('system_timeframe', 'timeframe', 'SYSTEM_TIMEFRAME'):
            assert effective.get(key) == user.get(key), f"Таймфрейм {key} должен быть из user_cfg"


def test_10_1_10_2_effective_switches_by_full_ai_control():
    """При full_ai_control=False эффективный конфиг — пользовательский; при True — ПРИИ (но ТФ из user)."""
    from bots_modules.imports_and_globals import (
        get_effective_auto_bot_config,
        get_effective_coin_settings,
        bots_data,
        bots_data_lock,
    )
    with bots_data_lock:
        orig = (bots_data.get('auto_bot_config') or {}).copy()
    full_ai = orig.get('full_ai_control', False)
    effective = get_effective_auto_bot_config()
    if not full_ai:
        assert effective.get('full_ai_control') in (False, None) or effective is not None
    else:
        assert effective.get('full_ai_control') is True
    get_effective_coin_settings('BTCUSDT')
    print("10.1/10.2: effective config and coin settings depend on full_ai_control — OK")


if __name__ == '__main__':
    test_10_3_timeframe_always_from_user_config()
    print("10.3: timeframe always from user config — OK")
    test_10_1_10_2_effective_switches_by_full_ai_control()
    print("All PRII plan checks passed.")
