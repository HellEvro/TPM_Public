"""
Регрессионный тест: RSI только по запрошенному таймфрейму, БЕЗ fallback на rsi6h.
Если вернуть fallback — бот открывает SHORT при 1m RSI 13 (подставляется 6h RSI 85).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_get_rsi_from_coin_data_no_fallback():
    from bot_engine.config_loader import get_rsi_from_coin_data, set_current_timeframe
    set_current_timeframe('1m')
    # Данные только по 6h — по 1m ничего нет
    coin_data = {'rsi6h': 85.0, 'trend6h': 'UP'}
    rsi = get_rsi_from_coin_data(coin_data, timeframe='1m')
    assert rsi is None, (
        "get_rsi_from_coin_data НЕ ДОЛЖЕН подставлять rsi6h при timeframe='1m'. "
        "Иначе бот открывает SHORT при 1m RSI 13. Получено: rsi=%s" % rsi
    )
    # По 6h — должен взять rsi6h
    rsi6h = get_rsi_from_coin_data(coin_data, timeframe='6h')
    assert rsi6h == 85.0, "При timeframe='6h' должен возвращаться rsi6h: %s" % rsi6h
    # По 1m с заполненным rsi1m — должен вернуть его
    coin_data['rsi1m'] = 13.0
    rsi1m = get_rsi_from_coin_data(coin_data, timeframe='1m')
    assert rsi1m == 13.0, "При наличии rsi1m должен возвращаться он: %s" % rsi1m
    print("OK: RSI без fallback по другому ТФ")


if __name__ == '__main__':
    test_get_rsi_from_coin_data_no_fallback()
    print("test_rsi_no_fallback passed.")
