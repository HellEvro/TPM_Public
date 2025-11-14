"""
Управление оптимальными EMA периодами для монет
"""

import logging
from .storage import load_optimal_ema, save_optimal_ema

logger = logging.getLogger('OptimalEMA')

# Глобальное хранилище оптимальных EMA
optimal_ema_data = {}


def load_optimal_ema_data():
    """Загружает данные об оптимальных EMA"""
    global optimal_ema_data
    optimal_ema_data = load_optimal_ema()
    return optimal_ema_data


def get_optimal_ema_periods(symbol):
    """Получает оптимальные EMA периоды и параметры подтверждения для монеты"""
    global optimal_ema_data
    
    if symbol in optimal_ema_data:
        data = optimal_ema_data[symbol]
        
        # Поддержка нового формата
        if 'ema_short_period' in data and 'ema_long_period' in data:
            return {
                'ema_short': data['ema_short_period'],
                'ema_long': data['ema_long_period'],
                'accuracy': data.get('accuracy', 0),
                'long_signals': data.get('long_signals', 0),
                'short_signals': data.get('short_signals', 0),
                'analysis_method': data.get('analysis_method', 'unknown'),
                # Параметры подтверждения тренда (индивидуальные для монеты)
                'trend_confirmation_bars': data.get('trend_confirmation_bars', None),
                'trend_min_confirmations': data.get('trend_min_confirmations', None),
                'trend_require_slope': data.get('trend_require_slope', None),
                'trend_require_price': data.get('trend_require_price', None),
                'trend_require_candles': data.get('trend_require_candles', None)
            }
        # Поддержка старого формата
        elif 'ema_short' in data and 'ema_long' in data:
            return {
                'ema_short': data['ema_short'],
                'ema_long': data['ema_long'],
                'accuracy': data.get('accuracy', 0),
                'long_signals': 0,
                'short_signals': 0,
                'analysis_method': 'legacy',
                # Дефолтные параметры
                'trend_confirmation_bars': None,
                'trend_min_confirmations': None,
                'trend_require_slope': None,
                'trend_require_price': None,
                'trend_require_candles': None
            }
        else:
            logger.warning(f"[OPTIMAL_EMA] Неизвестный формат данных для {symbol}")
            return get_default_ema_periods()
    else:
        return get_default_ema_periods()


def get_default_ema_periods():
    """Возвращает дефолтные EMA периоды и параметры подтверждения"""
    return {
        'ema_short': 50,
        'ema_long': 200,
        'accuracy': 0,
        'long_signals': 0,
        'short_signals': 0,
        'analysis_method': 'default',
        # Дефолтные параметры подтверждения (используются глобальные из SystemConfig)
        'trend_confirmation_bars': None,
        'trend_min_confirmations': None,
        'trend_require_slope': None,
        'trend_require_price': None,
        'trend_require_candles': None
    }


def update_optimal_ema_data(new_data):
    """Обновляет данные об оптимальных EMA"""
    global optimal_ema_data
    try:
        if isinstance(new_data, dict):
            optimal_ema_data.update(new_data)
            logger.info(f"[OPTIMAL_EMA] Обновлено {len(new_data)} записей")
            return True
        else:
            logger.error("[OPTIMAL_EMA] Неверный формат данных")
            return False
    except Exception as e:
        logger.error(f"[OPTIMAL_EMA] Ошибка обновления: {e}")
        return False


def save_optimal_ema_periods():
    """Сохраняет оптимальные EMA периоды"""
    global optimal_ema_data
    if not optimal_ema_data:
        logger.warning("[OPTIMAL_EMA] Нет данных для сохранения")
        return False
    return save_optimal_ema(optimal_ema_data)

