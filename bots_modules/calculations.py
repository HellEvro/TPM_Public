"""Функции расчета RSI, EMA и анализа тренда

Включает:
- calculate_rsi - расчет RSI
- calculate_rsi_history - история RSI
- calculate_ema - расчет EMA
- analyze_trend_6h - анализ тренда
- perform_enhanced_rsi_analysis - расширенный анализ RSI
"""

import logging
from datetime import datetime
import time
import threading

# Импорты из bot_engine
try:
    from bot_engine.indicators import SignalGenerator, TechnicalIndicators
    from bot_engine.config_loader import SystemConfig
except ImportError:
    pass

# Импорт констант из imports_and_globals
try:
    from bot_engine.config_loader import SystemConfig
    TREND_CONFIRMATION_BARS = SystemConfig.TREND_CONFIRMATION_BARS
except ImportError:
    TREND_CONFIRMATION_BARS = 3  # Значение по умолчанию

# ❌ ОТКЛЮЧЕНО: optimal_ema перемещен в backup (используется заглушка из imports_and_globals)
# # Импорт функции optimal_ema из модуля
# try:
#     from bots_modules.optimal_ema import get_optimal_ema_periods
# except ImportError:
#     def get_optimal_ema_periods(symbol):
#         return {'ema_short': 50, 'ema_long': 200, 'accuracy': 0}

logger = logging.getLogger('BotsService')

def calculate_rsi(prices, period=14):
    """Рассчитывает RSI на основе массива цен (Wilder's RSI алгоритм)"""
    if len(prices) < period + 1:
        return None

    # Рассчитываем изменения цен
    changes = []
    for i in range(1, len(prices)):
        changes.append(prices[i] - prices[i-1])

    if len(changes) < period:
        return None

    # Разделяем на прибыли и убытки
    gains = []
    losses = []

    for change in changes:
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0) 
            losses.append(-change)

    # Первоначальные средние значения (простое среднее для первого периода)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Рассчитываем RSI используя сглаживание Wilder's
    # (это тип экспоненциального сглаживания)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    # Избегаем деления на ноль
    if avg_loss == 0:
        return 100.0

    # Рассчитываем RS и RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return round(rsi, 2)

def calculate_rsi_history(prices, period=14):
    """Рассчитывает полную историю RSI для анализа зрелости монеты"""
    if len(prices) < period + 1:
        return None

    # Рассчитываем изменения цен
    changes = []
    for i in range(1, len(prices)):
        changes.append(prices[i] - prices[i-1])

    if len(changes) < period:
        return None

    # Разделяем на прибыли и убытки
    gains = []
    losses = []

    for change in changes:
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0) 
            losses.append(-change)

    # Первоначальные средние значения
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Рассчитываем полную историю RSI
    rsi_history = []

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        rsi_history.append(round(rsi, 2))

    return rsi_history

# Глобальные переменные (импортируются из главного файла)
# Эти переменные будут доступны после импорта из bots_modules.imports_and_globals
try:
    from bots_modules.imports_and_globals import (
        mature_coins_storage, mature_coins_lock, MATURE_COINS_FILE,
        bots_data_lock, bots_data
    )
except:
    # Заглушки если импорт не удался
    mature_coins_storage = {}
    MATURE_COINS_FILE = 'data/mature_coins.json'
    mature_coins_lock = threading.Lock()
    bots_data_lock = threading.Lock()
    bots_data = {}

# ❌ ОТКЛЮЧЕНО: Все функции optimal_ema удалены (EMA фильтр убран из системы)
# optimal_ema_data = {}
# OPTIMAL_EMA_FILE = 'data/optimal_ema.json'

def check_coin_maturity_with_storage(symbol, candles):
    """Проверяет зрелость монеты с использованием постоянного хранилища"""
    # Сначала проверяем постоянное хранилище
    if is_coin_mature_stored(symbol):

        # Обновляем время последней проверки
        update_mature_coin_verification(symbol)
        return {
            'is_mature': True,
            'reason': 'Монета зрелая (из постоянного хранилища)',
            'details': {'stored': True, 'from_storage': True}
        }

    # Если не в хранилище, выполняем полную проверку
    maturity_result = check_coin_maturity(symbol, candles)

    # Если монета зрелая, добавляем в постоянное хранилище (без автосохранения)
    if maturity_result['is_mature']:
        add_mature_coin_to_storage(symbol, maturity_result, auto_save=False)

    return maturity_result

def check_coin_maturity(symbol, candles):
    """Проверяет зрелость монеты для торговли"""
    try:
        # Получаем настройки зрелости из конфигурации
        with bots_data_lock:
            config = bots_data.get('auto_bot_config', {})

        min_candles = config.get('min_candles_for_maturity', MIN_CANDLES_FOR_MATURITY)
        min_rsi_low = config.get('min_rsi_low', MIN_RSI_LOW)
        max_rsi_high = config.get('max_rsi_high', MAX_RSI_HIGH)
        # Убрали min_volatility - больше не проверяем волатильность

        if not candles or len(candles) < min_candles:
            return {
                'is_mature': False,
                'reason': f'Недостаточно свечей: {len(candles) if candles else 0}/{min_candles}',
                'details': {
                    'candles_count': len(candles) if candles else 0,
                    'min_required': min_candles
                }
            }

        # ✅ ИСПРАВЛЕНИЕ: Берем только последние N свечей для анализа зрелости
        # Это означает что монета должна иметь достаточно истории в РЕЦЕНТНОЕ время
        recent_candles = candles[-min_candles:] if len(candles) >= min_candles else candles

        # Извлекаем цены закрытия из последних свечей
        closes = [candle['close'] for candle in recent_candles]

        # Рассчитываем историю RSI
        rsi_history = calculate_rsi_history(closes, 14)
        if not rsi_history:
            return {
                'is_mature': False,
                'reason': 'Не удалось рассчитать историю RSI',
                'details': {}
            }

        # Анализируем диапазон RSI
        rsi_min = min(rsi_history)
        rsi_max = max(rsi_history)
        rsi_range = rsi_max - rsi_min

        # Проверяем критерии зрелости (убрали проверку волатильности)
        maturity_checks = {
            'sufficient_candles': len(candles) >= min_candles,
            'rsi_reached_low': rsi_min <= min_rsi_low,
            'rsi_reached_high': rsi_max >= max_rsi_high
        }

        # Убрали проверку волатильности - она была слишком строгой
        volatility = 0  # Для совместимости с детальной информацией

        # Определяем общую зрелость
        # Монета зрелая, если достаточно свечей И RSI достигал низких И высоких значений (полный цикл)
        is_mature = maturity_checks['sufficient_candles'] and maturity_checks['rsi_reached_low'] and maturity_checks['rsi_reached_high']

        # Детальное логирование для отладки (отключено для уменьшения спама)
        # logger.info(f"[MATURITY_DEBUG] {symbol}: свечи={maturity_checks['sufficient_candles']} ({len(candles)}/{min_candles}), RSI_low={maturity_checks['rsi_reached_low']} (min={rsi_min:.1f}<=>{min_rsi_low}), RSI_high={maturity_checks['rsi_reached_high']} (max={rsi_max:.1f}>={max_rsi_high}), зрелая={is_mature}")

        # Формируем детальную информацию
        details = {
            'candles_count': len(candles),
            'min_required': min_candles,
            'rsi_min': rsi_min,
            'rsi_max': rsi_max,
            'rsi_range': rsi_range,
            'checks': maturity_checks
        }

        # Определяем причину незрелости
        if not is_mature:
            failed_checks = [check for check, passed in maturity_checks.items() if not passed]
            reason = f'Не пройдены проверки: {", ".join(failed_checks)}'
        else:
            reason = 'Монета зрелая для торговли'

        return {
            'is_mature': is_mature,
            'reason': reason,
            'details': details
        }

    except Exception as e:
        logger.error(f"[MATURITY] Ошибка проверки зрелости {symbol}: {e}")
        return {
            'is_mature': False,
            'reason': f'Ошибка анализа: {str(e)}',
            'details': {}
        }

def calculate_ema(prices, period):
    """Рассчитывает EMA для массива цен"""
    if len(prices) < period:
        return None

    # Первое значение EMA = SMA
    sma = sum(prices[:period]) / period
    ema = sma
    multiplier = 2 / (period + 1)

    # Рассчитываем EMA для остальных значений
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))

    return ema

def analyze_trend(symbol, exchange_obj=None, candles_data=None, timeframe=None):
    """
    Анализирует тренд на основе ПРОСТОГО АНАЛИЗА ЦЕНЫ (без EMA)

    Логика:
    - Берет последние N свечей (из конфига: trend_analysis_period)
    - Сравниваем цену начала и конца периода
    - Считаем % изменения и количество растущих/падающих свечей
    - Определяет тренд: UP / DOWN / NEUTRAL (по порогам из конфига)

    Args:
        symbol: Символ монеты
        exchange_obj: Объект биржи (опционально)
        candles_data: Данные свечей (опционально, если None - загружаются с биржи)
        timeframe: Таймфрейм для анализа (если None - используется текущий таймфрейм)
    """
    try:
        # Получаем текущий таймфрейм
        if timeframe is None:
            from bot_engine.config_loader import get_current_timeframe
            timeframe = get_current_timeframe()

        # Получаем свечи для анализа тренда
        from bots_modules.imports_and_globals import get_exchange, get_auto_bot_config
        exchange_to_use = exchange_obj if exchange_obj else get_exchange()
        if not exchange_to_use:
            logger.error(f"[TREND] ❌ Биржа не доступна для анализа тренда {symbol}")
            return None

        # ✅ Получаем параметры анализа тренда из конфига
        config = get_auto_bot_config()
        period = config.get('trend_analysis_period', 30)  # Количество свечей (20-50)
        price_threshold = config.get('trend_price_change_threshold', 7)  # Порог изменения цены (3-15%)
        candles_threshold = config.get('trend_candles_threshold', 70)  # Порог процента свечей (50-80%)

        if candles_data is None:
            # Определяем период для загрузки свечей (примерно 30 дней для большинства таймфреймов)
            chart_response = exchange_to_use.get_chart_data(symbol, timeframe, '30d')
            if not chart_response or not chart_response.get('success'):
                return None
            candles = chart_response['data']['candles']
        else:
            candles = candles_data

        if not candles or len(candles) < period:
            return None

        # Извлекаем цены закрытия
        closes = [candle['close'] for candle in candles]
        current_close = closes[-1]

        # ✅ АНАЛИЗ: Берем последние N свечей (из конфига)
        recent_closes = closes[-period:]
        start_price = recent_closes[0]
        end_price = recent_closes[-1]

        # Считаем % изменения
        price_change_pct = ((end_price - start_price) / start_price) * 100

        # Считаем растущие/падающие свечи
        rising_candles = sum(1 for i in range(1, len(recent_closes)) if recent_closes[i] > recent_closes[i-1])
        falling_candles = sum(1 for i in range(1, len(recent_closes)) if recent_closes[i] < recent_closes[i-1])

        # ✅ ОПРЕДЕЛЕНИЕ ТРЕНДА (используем пороги из конфига):
        # UP: если цена выросла > price_threshold% ИЛИ больше candles_threshold% свечей растут
        # DOWN: если цена упала > price_threshold% ИЛИ больше candles_threshold% свечей падают
        # NEUTRAL: иначе
        trend = 'NEUTRAL'

        candles_threshold_pct = candles_threshold / 100.0  # Конвертируем в десятичную дробь

        if price_change_pct > price_threshold or rising_candles > (period * candles_threshold_pct):
            trend = 'UP'
        elif price_change_pct < -price_threshold or falling_candles > (period * candles_threshold_pct):
            trend = 'DOWN'

        return {
            'trend': trend,
            'price_change_pct': price_change_pct,
            'rising_candles': rising_candles,
            'falling_candles': falling_candles,
            'current_close': current_close,
            'start_price': start_price,
            'period': period,
            'timeframe': timeframe,
            'method': 'simple_price_analysis'  # Метод анализа
        }

    except Exception as e:
        logger.error(f"Ошибка анализа тренда для {symbol}: {e}")
        return None

# Обратная совместимость: оставляем старую функцию для существующего кода
def analyze_trend_6h(symbol, exchange_obj=None, candles_data=None):
    """
    Устаревшая функция для обратной совместимости.
    Использует analyze_trend с таймфреймом '6h'.
    """
    try:
        from bot_engine.config_loader import get_current_timeframe, TIMEFRAME
        current_timeframe = get_current_timeframe()
    except Exception:
        current_timeframe = TIMEFRAME
    return analyze_trend(symbol, exchange_obj, candles_data, timeframe=current_timeframe)

def perform_enhanced_rsi_analysis(candles, current_rsi, symbol):
    """Выполняет улучшенный анализ RSI для монеты"""
    try:
        # ✅ Проверяем индивидуальные настройки монеты (имеют приоритет над глобальными)
        enhanced_rsi_enabled = SystemConfig.ENHANCED_RSI_ENABLED

        # Получаем индивидуальные настройки монеты, если есть
        try:
            from bots_modules.imports_and_globals import get_individual_coin_settings
            individual_settings = get_individual_coin_settings(symbol)
            if individual_settings is not None and 'enhanced_rsi_enabled' in individual_settings:
                # Индивидуальная настройка переопределяет глобальную
                enhanced_rsi_enabled = bool(individual_settings['enhanced_rsi_enabled'])
        except Exception:
            pass  # Если не удалось получить настройки, используем глобальные

        # Проверяем, включена ли улучшенная система
        if not enhanced_rsi_enabled:
            return {
                'enabled': False,
                'warning_type': None,
                'warning_message': None,
                'extreme_duration': 0,
                'adaptive_levels': None,
                'confirmations': {},
                'enhanced_signal': None
            }

        # Импортируем SignalGenerator для использования улучшенной логики
        from bot_engine.indicators import SignalGenerator, TechnicalIndicators

        # Создаем объект для анализа
        signal_generator = SignalGenerator()

        # Форматируем данные свечей для анализа
        # Bybit отправляет свечи в правильном порядке для анализа
        formatted_candles = []
        for candle in candles:  # Используем оригинальный порядок
            formatted_candles.append({
                'timestamp': candle.get('time', 0),
                'open': float(candle.get('open', 0)),
                'high': float(candle.get('high', 0)),
                'low': float(candle.get('low', 0)),
                'close': float(candle.get('close', 0)),
                'volume': float(candle.get('volume', 0))
            })

        # Получаем полный анализ
        if len(formatted_candles) >= 50:
            try:
                analysis_result = signal_generator.generate_signals(formatted_candles)

                # Получаем базовые данные для анализа
                closes = [candle['close'] for candle in formatted_candles]
                volumes = [candle['volume'] for candle in formatted_candles]

                # Рассчитываем дополнительные индикаторы
                rsi_history = TechnicalIndicators.calculate_rsi_history(formatted_candles)
                adaptive_levels = TechnicalIndicators.calculate_adaptive_rsi_levels(formatted_candles)
                divergence = TechnicalIndicators.detect_rsi_divergence(closes, rsi_history)
                volume_confirmation = TechnicalIndicators.confirm_with_volume(volumes)

                # Для Stochastic RSI используем ВСЮ историю RSI
                # Параметры Bybit: stoch_period=14, k_smooth=3, d_smooth=3
                stoch_rsi_result = TechnicalIndicators.calculate_stoch_rsi(
                    rsi_history, 
                    stoch_period=14, 
                    k_smooth=3,
                    d_smooth=3
                )
                stoch_rsi = stoch_rsi_result['k'] if stoch_rsi_result else None
                stoch_rsi_d = stoch_rsi_result['d'] if stoch_rsi_result else None

                # Определяем продолжительность в экстремальной зоне
                extreme_duration = 0
                if rsi_history:
                    for rsi_val in reversed(rsi_history):
                        if rsi_val <= SystemConfig.RSI_EXTREME_OVERSOLD or rsi_val >= SystemConfig.RSI_EXTREME_OVERBOUGHT:
                            extreme_duration += 1
                        else:
                            break

                # Определяем тип предупреждения
                warning_type = None
                warning_message = None

                # Проверяем экстремальные условия
                if current_rsi <= SystemConfig.RSI_EXTREME_OVERSOLD:
                    if extreme_duration > SystemConfig.RSI_EXTREME_ZONE_TIMEOUT:
                        warning_type = 'EXTREME_OVERSOLD_LONG'
                        warning_message = f'RSI в экстремальной зоне {extreme_duration} свечей'
                    else:
                        warning_type = 'OVERSOLD'
                        warning_message = 'Возможная зона для LONG'

                elif current_rsi >= SystemConfig.RSI_EXTREME_OVERBOUGHT:
                    if extreme_duration > SystemConfig.RSI_EXTREME_ZONE_TIMEOUT:
                        warning_type = 'EXTREME_OVERBOUGHT_LONG'
                        warning_message = f'RSI в экстремальной зоне {extreme_duration} свечей'
                    else:
                        warning_type = 'OVERBOUGHT'
                        warning_message = 'Возможная зона для SHORT'

                # Анализ подтверждений (явно преобразуем в стандартные Python типы)
                confirmations = {
                    'volume': bool(volume_confirmation) if volume_confirmation is not None else False,
                    'divergence': bool(divergence) if divergence is not None else False,
                    'stoch_rsi_k': float(stoch_rsi) if stoch_rsi is not None else None,
                    'stoch_rsi_d': float(stoch_rsi_d) if stoch_rsi_d is not None else None
                }

                return {
                    'enabled': True,
                    'warning_type': warning_type,
                    'warning_message': warning_message,
                    'extreme_duration': int(extreme_duration),
                    'adaptive_levels': adaptive_levels,
                    'confirmations': confirmations,
                    'enhanced_signal': analysis_result.get('signal', 'WAIT'),
                    'enhanced_reason': analysis_result.get('reason', 'enhanced_analysis')
                }

            except Exception as e:
                logger.error(f"Ошибка анализа для {symbol}: {e}")
                return {
                    'enabled': True,
                    'warning_type': 'ERROR',
                    'warning_message': f'Ошибка анализа: {str(e)}',
                    'extreme_duration': 0,
                    'adaptive_levels': [29, 71],
                    'confirmations': {
                        'volume': False,
                        'divergence': False,
                        'stoch_rsi_k': None,
                        'stoch_rsi_d': None
                    },
                    'enhanced_signal': 'WAIT'
                }
        else:
            # Недостаточно данных для полного анализа
            return {
                'enabled': True,
                'warning_type': None,
                'warning_message': 'Недостаточно данных для анализа',
                'extreme_duration': 0,
                'adaptive_levels': [29, 71],
                'confirmations': {
                    'volume': False,
                    'divergence': False,
                    'stoch_rsi_k': None,
                    'stoch_rsi_d': None
                },
                'enhanced_signal': 'WAIT'
            }

    except Exception as e:
        logger.error(f"Ошибка анализа для {symbol}: {e}")
        return {
            'enabled': False,
            'warning_type': 'ERROR',
            'warning_message': f'Ошибка анализа: {str(e)}',
            'extreme_duration': 0,
            'adaptive_levels': [29, 71],
            'confirmations': {},
            'enhanced_signal': 'WAIT'
        }
