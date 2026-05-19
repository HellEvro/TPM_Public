"""
Проверка единой логики входа: пороги RSI + apply_entry_filters через check_entry_allowed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_candles(n=50, base=100.0):
    candles = []
    for i in range(n):
        candles.append({
            "open": base + i * 0.01,
            "close": base + i * 0.02,
            "high": base + i * 0.03,
            "low": base - 0.5,
            "time": i,
        })
    return candles


def test_rsi_threshold_blocks_long_when_high():
    from bot_engine.ai.filter_utils import check_rsi_entry_thresholds

    config = {"rsi_long_threshold": 29, "rsi_short_threshold": 71}
    ok, reason = check_rsi_entry_thresholds("ENTER_LONG", 45.0, config)
    assert not ok, reason
    ok, _ = check_rsi_entry_thresholds("ENTER_LONG", 25.0, config)
    assert ok


def test_rsi_threshold_blocks_short_when_low():
    from bot_engine.ai.filter_utils import check_rsi_entry_thresholds

    config = {"rsi_long_threshold": 29, "rsi_short_threshold": 71}
    ok, _ = check_rsi_entry_thresholds("ENTER_SHORT", 80.0, config)
    assert ok
    ok, reason = check_rsi_entry_thresholds("ENTER_SHORT", 50.0, config)
    assert not ok, reason


def test_check_entry_allowed_blocks_without_rsi():
    from bot_engine.ai.filter_utils import check_entry_allowed

    config = {
        "rsi_long_threshold": 29,
        "rsi_short_threshold": 71,
        "trading_enabled": True,
        "enable_maturity_check": False,
        "exit_scam_enabled": False,
        "rsi_time_filter_enabled": False,
    }
    candles = _make_candles()
    ok, reason = check_entry_allowed(
        "BTCUSDT",
        candles,
        None,
        "ENTER_LONG",
        config,
        trend="NEUTRAL",
        source="test",
    )
    assert not ok
    assert "RSI" in reason


def test_check_entry_allowed_same_for_force_flag():
    from bot_engine.ai.filter_utils import check_entry_allowed

    config = {
        "rsi_long_threshold": 29,
        "rsi_short_threshold": 71,
        "trading_enabled": True,
        "enable_maturity_check": False,
        "exit_scam_enabled": False,
        "rsi_time_filter_enabled": False,
        "scope": "all",
    }
    candles = _make_candles()
    ok_manual, r_manual = check_entry_allowed(
        "BTCUSDT", candles, 80.0, "ENTER_SHORT", config,
        source="manual", force_market_entry=False,
    )
    ok_auto, r_auto = check_entry_allowed(
        "BTCUSDT", candles, 80.0, "ENTER_SHORT", config,
        source="autobot", force_market_entry=True,
    )
    assert ok_manual == ok_auto
    assert ok_manual is True


if __name__ == "__main__":
    test_rsi_threshold_blocks_long_when_high()
    test_rsi_threshold_blocks_short_when_low()
    test_check_entry_allowed_blocks_without_rsi()
    test_check_entry_allowed_same_for_force_flag()
    print("test_entry_filters_parity passed.")
