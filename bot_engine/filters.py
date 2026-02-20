"""
Торговые фильтры для защиты от плохих входов
- RSI временной фильтр
- ExitScam фильтр (защита от памп/дамп)
"""

import logging

from .utils.rsi_utils import calculate_rsi_history

logger = logging.getLogger('Filters')

def check_rsi_time_filter(candles, rsi, signal, config, calculate_rsi_history_func=None):
    """
    ГИБРИДНЫЙ ВРЕМЕННОЙ ФИЛЬТР RSI

    Проверяет что:
    1. Последние N свечей находятся в "спокойной зоне"
    2. Перед этой зоной был экстремум
    3. С момента экстремума прошло минимум N свечей

    Args:
        candles: Список свечей
        rsi: Текущее значение RSI
        signal: Торговый сигнал ('ENTER_LONG' или 'ENTER_SHORT')
        config: Конфигурация фильтра
        calculate_rsi_history_func: Функция для расчета RSI истории (опционально)

    Returns:
        dict: {'allowed': bool, 'reason': str, 'last_extreme_candles_ago': int, 'calm_candles': int}
    """
    try:
        # Используем переданную функцию или импортированную
        calc_rsi_hist = calculate_rsi_history_func or calculate_rsi_history

        # Получаем настройки из конфига
        rsi_time_filter_enabled = config.get('rsi_time_filter_enabled', True)
        rsi_time_filter_candles = max(1, config.get('rsi_time_filter_candles', 8))  # Минимум 1 свеча (конфиг соблюдается)
        rsi_time_filter_upper = config.get('rsi_time_filter_upper', 65)
        rsi_time_filter_lower = config.get('rsi_time_filter_lower', 35)
        rsi_long_threshold = config.get('rsi_long_threshold', 29)
        rsi_short_threshold = config.get('rsi_short_threshold', 71)

        # Если фильтр отключен
        if not rsi_time_filter_enabled:
            return {'allowed': True, 'reason': 'RSI временной фильтр отключен', 'last_extreme_candles_ago': None, 'calm_candles': None}

        if len(candles) < 50:
            return {'allowed': False, 'reason': 'Недостаточно свечей для анализа', 'last_extreme_candles_ago': None, 'calm_candles': 0}

        # Рассчитываем историю RSI
        closes = [candle['close'] for candle in candles]
        rsi_history = calc_rsi_hist(closes, 14)

        min_rsi_history = max(rsi_time_filter_candles * 2 + 14, 30)
        if not rsi_history or len(rsi_history) < min_rsi_history:
            return {'allowed': False, 'reason': f'Недостаточно RSI истории (требуется {min_rsi_history})', 'last_extreme_candles_ago': None, 'calm_candles': 0}

        current_index = len(rsi_history) - 1

        # Обновляем последний элемент истории переданным RSI, если он указан
        # Это важно для согласованности данных, так как переданный RSI может быть более актуальным
        if rsi is not None:
            rsi_history[current_index] = rsi

        if signal == 'ENTER_SHORT':
            # ЛОГИКА ДЛЯ SHORT:
            # 1. Берем последние N свечей (rsi_time_filter_candles из конфига, например 8)
            # 2. Ищем среди них САМУЮ РАННЮЮ (левую) свечу с RSI >= 71 - это отправная точка
            # 3. От отправной точки проверяем ВСЕ последующие свечи (до текущей) - должны быть >= 65
            # 4. Если все >= 65 И прошло минимум N свечей - разрешаем
            # 5. Если какая-то свеча < 65 - блокируем (вход упущен)

            # Берем последние N свечей из конфига
            last_n_candles_start = max(0, current_index - rsi_time_filter_candles + 1)
            last_n_candles = rsi_history[last_n_candles_start:current_index + 1]

            # Ищем САМУЮ РАННЮЮ (левую) свечу с RSI >= 71 среди последних N свечей
            peak_index = None
            for i in range(last_n_candles_start, current_index + 1):
                if rsi_history[i] >= rsi_short_threshold:
                    peak_index = i  # Нашли самую раннюю свечу >= 71
                    break

            # Если не нашли пик в последних N свечах - блокируем (нет отправной точки)
            if peak_index is None:
                return {
                    'allowed': False,
                    'reason': f'Блокировка: пик RSI >= {rsi_short_threshold} не найден в последних {rsi_time_filter_candles} свечах',
                    'last_extreme_candles_ago': None,
                    'calm_candles': 0
                }

            # Проверяем ВСЕ свечи ОТ отправной точки (включая её) до текущей включительно
            # Берем все свечи ОТ peak_index (включая сам peak_index) до current_index
            check_candles = rsi_history[peak_index:current_index + 1]

            # Проверяем что ВСЕ свечи >= 65 (включая саму отправную точку)
            invalid_candles = [rsi_val for rsi_val in check_candles if rsi_val < rsi_time_filter_upper]

            if len(invalid_candles) > 0:
                # Есть свечи < 65 - вход упущен
                candles_since_peak = current_index - peak_index + 1
                return {
                    'allowed': False,
                    'reason': f'Блокировка: {len(invalid_candles)} свечей после отправной точки провалились < {rsi_time_filter_upper} (вход упущен)',
                    'last_extreme_candles_ago': candles_since_peak - 1,
                    'calm_candles': len(check_candles) - len(invalid_candles)
                }

            # Проверяем что прошло достаточно свечей (минимум N из конфига)
            # candles_since_peak - это количество свечей ОТ отправной точки (включая её) до текущей
            candles_since_peak = current_index - peak_index + 1
            if candles_since_peak < rsi_time_filter_candles:
                return {
                    'allowed': False,
                    'reason': f'Ожидание: с отправной точки прошло только {candles_since_peak} свечей (требуется {rsi_time_filter_candles})',
                    'last_extreme_candles_ago': candles_since_peak - 1,
                    'calm_candles': candles_since_peak
                }

            # Все проверки пройдены!
            return {
                'allowed': True,
                'reason': f'Разрешено: с отправной точки (свеча -{candles_since_peak}) прошло {candles_since_peak} спокойных свечей >= {rsi_time_filter_upper}',
                'last_extreme_candles_ago': candles_since_peak - 1,
                'calm_candles': candles_since_peak
            }

        elif signal == 'ENTER_LONG':
            # ЛОГИКА ДЛЯ LONG:
            # 1. Берем последние N свечей (rsi_time_filter_candles из конфига, например 8)
            # 2. Ищем среди них САМУЮ РАННЮЮ (левую) свечу с RSI <= 29 - это отправная точка
            # 3. От отправной точки проверяем ВСЕ последующие свечи (до текущей) - должны быть <= 35
            # 4. Если все <= 35 И прошло минимум N свечей - разрешаем
            # 5. Если какая-то свеча > 35 - блокируем (вход упущен)

            # Берем последние N свечей из конфига
            last_n_candles_start = max(0, current_index - rsi_time_filter_candles + 1)
            last_n_candles = rsi_history[last_n_candles_start:current_index + 1]

            # Ищем САМУЮ РАННЮЮ (левую) свечу с RSI <= 29 среди последних N свечей
            low_index = None
            for i in range(last_n_candles_start, current_index + 1):
                if rsi_history[i] <= rsi_long_threshold:
                    low_index = i  # Нашли самую раннюю свечу <= 29
                    break

            # Если не нашли лой в последних N свечах - блокируем (нет отправной точки)
            if low_index is None:
                return {
                    'allowed': False,
                    'reason': f'Блокировка: лой RSI <= {rsi_long_threshold} не найден в последних {rsi_time_filter_candles} свечах',
                    'last_extreme_candles_ago': None,
                    'calm_candles': 0
                }

            # Проверяем ВСЕ свечи ОТ отправной точки (включая её) до текущей включительно
            # Берем все свечи ОТ low_index (включая сам low_index) до current_index
            check_candles = rsi_history[low_index:current_index + 1]

            # Проверяем что ВСЕ свечи <= 35 (включая саму отправную точку)
            invalid_candles = [rsi_val for rsi_val in check_candles if rsi_val > rsi_time_filter_lower]

            if len(invalid_candles) > 0:
                # Есть свечи > 35 - вход упущен
                candles_since_low = current_index - low_index + 1
                return {
                    'allowed': False,
                    'reason': f'Блокировка: {len(invalid_candles)} свечей после отправной точки поднялись > {rsi_time_filter_lower} (вход упущен)',
                    'last_extreme_candles_ago': candles_since_low - 1,
                    'calm_candles': len(check_candles) - len(invalid_candles)
                }

            # Проверяем что прошло достаточно свечей (минимум N из конфига)
            # candles_since_low - это количество свечей ОТ отправной точки (включая её) до текущей
            candles_since_low = current_index - low_index + 1
            if candles_since_low < rsi_time_filter_candles:
                return {
                    'allowed': False,
                    'reason': f'Ожидание: с отправной точки прошло только {candles_since_low} свечей (требуется {rsi_time_filter_candles})',
                    'last_extreme_candles_ago': candles_since_low - 1,
                    'calm_candles': candles_since_low
                }

            # Все проверки пройдены!
            return {
                'allowed': True,
                'reason': f'Разрешено: с отправной точки (свеча -{candles_since_low}) прошло {candles_since_low} спокойных свечей <= {rsi_time_filter_lower}',
                'last_extreme_candles_ago': candles_since_low - 1,
                'calm_candles': candles_since_low
            }

        return {'allowed': True, 'reason': 'Неизвестный сигнал', 'last_extreme_candles_ago': None, 'calm_candles': 0}

    except Exception as e:
        logger.error(f"[RSI_TIME_FILTER] Ошибка проверки временного фильтра: {e}")
        return {'allowed': False, 'reason': f'Ошибка анализа: {str(e)}', 'last_extreme_candles_ago': None, 'calm_candles': 0}

def check_exit_scam_filter(symbol, coin_data, config, exchange_obj, ensure_exchange_func):
    """
    EXIT SCAM ФИЛЬТР

    Защита от резких движений цены (памп/дамп/скам):
    1. Одна свеча превысила максимальный % (тело свечи: |C-O|/O×100). Лимит = значение из конфига как есть (0.5 = 0.5%).
    2. N свечей суммарно превысили максимальный % (суммарное изменение).

    Args:
        symbol: Символ монеты
        coin_data: Данные монеты (не используется пока)
        config: Конфигурация фильтра
        exchange_obj: Объект биржи
        ensure_exchange_func: Функция проверки инициализации биржи
    """
    try:
        from .config_loader import get_config_value, get_current_timeframe
        exit_scam_enabled = get_config_value(config, 'exit_scam_enabled')
        exit_scam_candles = get_config_value(config, 'exit_scam_candles')
        single_candle_percent = get_config_value(config, 'exit_scam_single_candle_percent')
        multi_candle_count = get_config_value(config, 'exit_scam_multi_candle_count')
        multi_candle_percent = get_config_value(config, 'exit_scam_multi_candle_percent')
        single_candle_percent = float(single_candle_percent) if single_candle_percent is not None else None
        multi_candle_percent = float(multi_candle_percent) if multi_candle_percent is not None else None
        if single_candle_percent is None or multi_candle_percent is None:
            return False
        exit_scam_candles = int(exit_scam_candles) if exit_scam_candles is not None else None
        multi_candle_count = int(multi_candle_count) if multi_candle_count is not None else None
        if exit_scam_candles is None or multi_candle_count is None:
            return False
        if not exit_scam_enabled:
            return True

        timeframe = get_current_timeframe()
        if not timeframe:
            return False

        # Свечи из кэша (уже загружены для RSI) — без повторного API-запроса
        candles = (coin_data or {}).get('_candles')
        if not candles or len(candles) < exit_scam_candles:
            if not ensure_exchange_func():
                return False
            chart_response = exchange_obj.get_chart_data(symbol, timeframe, '30d')
            if not chart_response or not chart_response.get('success'):
                return False
            candles = chart_response.get('data', {}).get('candles', [])
        if exit_scam_candles is None or len(candles) < exit_scam_candles:
            return False

        recent_candles = candles[-exit_scam_candles:]

        for i, candle in enumerate(recent_candles):
            open_price = float(candle.get('open', 0) or 0)
            close_price = float(candle.get('close', 0) or 0)
            if open_price <= 0:
                continue
            # % тела свечи = (close-open)/open×100. 100% = цена удвоилась (close=2×open), 0.5% = close=1.005×open. open/close из API биржи.
            price_change = abs((close_price - open_price) / open_price) * 100
            if price_change > single_candle_percent:
                num_from_end = len(recent_candles) - i
                candle_label = "последняя" if num_from_end == 1 else f"#{num_from_end} с конца"
                logger.warning(
                    f"{symbol}: ❌ ExitScam: {candle_label} свеча — тело {price_change:.2f}% > лимит {single_candle_percent}% (как в конфиге). "
                    f"O={open_price:.6g} C={close_price:.6g}"
                )
                return False

        if len(recent_candles) >= multi_candle_count:
            multi_candles = recent_candles[-multi_candle_count:]
            first_open = float(multi_candles[0].get('open', 0) or 0)
            last_close = float(multi_candles[-1].get('close', 0) or 0)
            if first_open > 0:
                total_change = abs((last_close - first_open) / first_open) * 100
                if total_change > multi_candle_percent:
                    logger.warning(f"{symbol}: ❌ БЛОКИРОВКА: {multi_candle_count} свечей превысили суммарный лимит {multi_candle_percent}% (было {total_change:.1f}%)")
                    logger.info(f"{symbol}: Первая свеча: {first_open:.4f}, Последняя свеча: {last_close:.4f}")
                    return False

        return True

    except Exception as e:
        logger.error(f"{symbol}: ❌ Ошибка проверки ExitScam фильтра: {e}")
        import traceback

        return False

def check_no_existing_position(symbol, signal, exchange_obj, ensure_exchange_func):
    """Проверяет, что нет существующих позиций на бирже"""
    try:
        if not ensure_exchange_func():
            return False

        exchange_positions = exchange_obj.get_positions()
        if isinstance(exchange_positions, tuple):
            positions_list = exchange_positions[0] if exchange_positions else []
        else:
            positions_list = exchange_positions if exchange_positions else []

        expected_side = 'LONG' if signal == 'ENTER_LONG' else 'SHORT'

        # Проверяем, есть ли позиция той же стороны
        for pos in positions_list:
            if pos.get('symbol') == symbol and abs(float(pos.get('size', 0))) > 0:
                existing_side = pos.get('side', 'UNKNOWN')
                if existing_side == expected_side:

                    return False

        return True

    except Exception as e:
        logger.error(f"[POSITION_CHECK] {symbol}: Ошибка проверки позиций: {e}")
        return False
