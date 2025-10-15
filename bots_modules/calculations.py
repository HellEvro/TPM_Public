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
    from bot_engine.bot_config import (
        RSI_EXTREME_ZONE_TIMEOUT, RSI_EXTREME_OVERSOLD, RSI_EXTREME_OVERBOUGHT,
        SystemConfig
    )
except ImportError:
    pass

# Импорт констант из imports_and_globals
try:
    from bots_modules.imports_and_globals import TREND_CONFIRMATION_BARS
except ImportError:
    TREND_CONFIRMATION_BARS = 3  # Значение по умолчанию

# Импорт функции optimal_ema из модуля
try:
    from bots_modules.optimal_ema import get_optimal_ema_periods
except ImportError:
    def get_optimal_ema_periods(symbol):
        return {'ema_short': 50, 'ema_long': 200, 'accuracy': 0}

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

# Оптимальные EMA для определения тренда
optimal_ema_data = {}
OPTIMAL_EMA_FILE = 'data/optimal_ema.json'

def load_mature_coins_storage():
    """Загружает постоянное хранилище зрелых монет из файла"""
    global mature_coins_storage
    try:
        if os.path.exists(MATURE_COINS_FILE):
            with open(MATURE_COINS_FILE, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем блокировку при обновлении глобального хранилища
            with mature_coins_lock:
                mature_coins_storage = loaded_data
            
            logger.info(f"[MATURITY_STORAGE] ✅ Загружено {len(mature_coins_storage)} зрелых монет из файла")
        else:
            with mature_coins_lock:
                mature_coins_storage = {}
            logger.info("[MATURITY_STORAGE] Файл хранилища не найден, создаем новый")
    except Exception as e:
        logger.error(f"[MATURITY_STORAGE] Ошибка загрузки хранилища: {e}")
        with mature_coins_lock:
            mature_coins_storage = {}

def save_mature_coins_storage():
    """Сохраняет постоянное хранилище зрелых монет в файл"""
    try:
        with mature_coins_lock:
            # Создаем копию для безопасной сериализации
            storage_copy = mature_coins_storage.copy()
        
        os.makedirs(os.path.dirname(MATURE_COINS_FILE), exist_ok=True)
        
        # Создаем временный файл для атомарной записи
        temp_file = MATURE_COINS_FILE + '.tmp'
        max_retries = 3
        retry_delay = 0.1  # 100ms
        
        for attempt in range(max_retries):
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(storage_copy, f, ensure_ascii=False, indent=2)
                
                # Атомарно заменяем оригинальный файл
                if os.name == 'nt':  # Windows
                    if os.path.exists(MATURE_COINS_FILE):
                        os.remove(MATURE_COINS_FILE)
                    os.rename(temp_file, MATURE_COINS_FILE)
                else:  # Unix/Linux
                    os.rename(temp_file, MATURE_COINS_FILE)
                    
                logger.debug(f"[MATURITY_STORAGE] Хранилище сохранено: {len(storage_copy)} монет")
                break  # Успешно сохранили, выходим из цикла
                
            except (OSError, IOError) as temp_error:
                if attempt < max_retries - 1:
                    logger.warning(f"[MATURITY_STORAGE] Попытка {attempt + 1} неудачна, повторяем через {retry_delay}с: {temp_error}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Увеличиваем задержку
                    continue
                else:
                    # Удаляем временный файл в случае ошибки
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    raise temp_error
            except Exception as temp_error:
                # Удаляем временный файл в случае ошибки
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                raise temp_error
            
    except Exception as e:
        logger.error(f"[MATURITY_STORAGE] Ошибка сохранения хранилища: {e}")
        # Попробуем создать резервную копию
        try:
            backup_file = MATURE_COINS_FILE + '.backup'
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(storage_copy, f, ensure_ascii=False, indent=2)
            logger.info(f"[MATURITY_STORAGE] Создана резервная копия: {backup_file}")
        except Exception as backup_error:
            logger.error(f"[MATURITY_STORAGE] Не удалось создать резервную копию: {backup_error}")

def is_coin_mature_stored(symbol):
    """Проверяет, есть ли монета в постоянном хранилище зрелых монет"""
    with mature_coins_lock:
        return symbol in mature_coins_storage

def add_mature_coin_to_storage(symbol, maturity_data, auto_save=True):
    """Добавляет монету в постоянное хранилище зрелых монет (только если её там еще нет)"""
    global mature_coins_storage
    
    with mature_coins_lock:
        # Проверяем, есть ли уже монета в хранилище
        if symbol in mature_coins_storage:
            # Обновляем только время последней проверки
            mature_coins_storage[symbol]['last_verified'] = time.time()
            logger.debug(f"[MATURITY_STORAGE] {symbol}: обновлено время последней проверки")
            return
        
        # Добавляем новую монету в хранилище
        mature_coins_storage[symbol] = {
            'timestamp': time.time(),
            'maturity_data': maturity_data,
            'last_verified': time.time()
        }
    
    if auto_save:
        save_mature_coins_storage()
        logger.info(f"[MATURITY_STORAGE] Монета {symbol} добавлена в постоянное хранилище зрелых монет")
    else:
        logger.debug(f"[MATURITY_STORAGE] Монета {symbol} добавлена в хранилище (без автосохранения)")

def remove_mature_coin_from_storage(symbol):
    """Удаляет монету из постоянного хранилища зрелых монет"""
    global mature_coins_storage
    if symbol in mature_coins_storage:
        del mature_coins_storage[symbol]
        # Отключаем автоматическое сохранение - будет сохранено пакетно
        logger.debug(f"[MATURITY_STORAGE] Монета {symbol} удалена из хранилища (без автосохранения)")

def update_mature_coin_verification(symbol):
    """Обновляет время последней проверки зрелости монеты"""
    global mature_coins_storage
    if symbol in mature_coins_storage:
        mature_coins_storage[symbol]['last_verified'] = time.time()
        # Отключаем автоматическое сохранение - будет сохранено пакетно
        logger.debug(f"[MATURITY_STORAGE] Обновлено время проверки для {symbol} (без автосохранения)")

def load_optimal_ema_data():
    """Загружает данные об оптимальных EMA из файла"""
    global optimal_ema_data
    try:
        if os.path.exists(OPTIMAL_EMA_FILE):
            with open(OPTIMAL_EMA_FILE, 'r', encoding='utf-8') as f:
                optimal_ema_data = json.load(f)
                logger.info(f"[OPTIMAL_EMA] Загружено {len(optimal_ema_data)} записей об оптимальных EMA")
        else:
            optimal_ema_data = {}
            logger.info("[OPTIMAL_EMA] Файл с оптимальными EMA не найден")
    except Exception as e:
        logger.error(f"[OPTIMAL_EMA] Ошибка загрузки данных об оптимальных EMA: {e}")
        optimal_ema_data = {}

def get_optimal_ema_periods(symbol):
    """Получает оптимальные EMA периоды для монеты"""
    global optimal_ema_data
    if symbol in optimal_ema_data:
        data = optimal_ema_data[symbol]
        
        # Поддержка нового формата (ema_short_period, ema_long_period)
        if 'ema_short_period' in data and 'ema_long_period' in data:
            return {
                'ema_short': data['ema_short_period'],
                'ema_long': data['ema_long_period'],
                'accuracy': data.get('accuracy', 0),
                'long_signals': data.get('long_signals', 0),
                'short_signals': data.get('short_signals', 0),
                'analysis_method': data.get('analysis_method', 'unknown')
            }
        # Поддержка старого формата (ema_short, ema_long)
        elif 'ema_short' in data and 'ema_long' in data:
            return {
                'ema_short': data['ema_short'],
                'ema_long': data['ema_long'],
                'accuracy': data.get('accuracy', 0),
                'long_signals': 0,
                'short_signals': 0,
                'analysis_method': 'legacy'
            }
        else:
            # Неизвестный формат данных
            logger.warning(f"[OPTIMAL_EMA] Неизвестный формат данных для {symbol}")
            return {
                'ema_short': 50,
                'ema_long': 200,
                'accuracy': 0,
                'long_signals': 0,
                'short_signals': 0,
                'analysis_method': 'default'
            }
    else:
        # Возвращаем дефолтные значения
        return {
            'ema_short': 50,
            'ema_long': 200,
            'accuracy': 0,
            'long_signals': 0,
            'short_signals': 0,
            'analysis_method': 'default'
        }

def update_optimal_ema_data(new_data):
    """Обновляет данные об оптимальных EMA из внешнего источника"""
    global optimal_ema_data
    try:
        if isinstance(new_data, dict):
            optimal_ema_data.update(new_data)
            logger.info(f"[OPTIMAL_EMA] Обновлено {len(new_data)} записей об оптимальных EMA")
            return True
        else:
            logger.error("[OPTIMAL_EMA] Неверный формат данных для обновления")
            return False
    except Exception as e:
        logger.error(f"[OPTIMAL_EMA] Ошибка обновления данных: {e}")
        return False

def check_coin_maturity_with_storage(symbol, candles):
    """Проверяет зрелость монеты с использованием постоянного хранилища"""
    # Сначала проверяем постоянное хранилище
    if is_coin_mature_stored(symbol):
        logger.debug(f"[MATURITY_STORAGE] {symbol}: найдена в постоянном хранилище зрелых монет")
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
        
        logger.debug(f"[MATURITY] {symbol}: {reason}")
        logger.debug(f"[MATURITY] {symbol}: Свечи={len(candles)}, RSI={rsi_min:.1f}-{rsi_max:.1f}")
        
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

def analyze_trend_6h(symbol, exchange_obj=None):
    """Анализирует тренд 6H с использованием оптимальных EMA периодов"""
    try:
        # Получаем оптимальные EMA периоды для монеты
        ema_periods = get_optimal_ema_periods(symbol)
        ema_short_period = ema_periods['ema_short']
        ema_long_period = ema_periods['ema_long']
        
        # Получаем свечи 6H для анализа тренда (нужно больше данных для длинной EMA)
        # Используем переданную биржу или глобальную переменную
        exchange_to_use = exchange_obj if exchange_obj else exchange
        if not exchange_to_use:
            logger.error(f"[TREND] ❌ Биржа не доступна для анализа тренда {symbol}")
            return None
            
        chart_response = exchange_to_use.get_chart_data(symbol, '6h', '60d')
        
        if not chart_response or not chart_response.get('success'):
            return None
        
        candles = chart_response['data']['candles']
        min_candles = max(ema_long_period + 50, 210)  # Минимум для длинной EMA + запас
        if not candles or len(candles) < min_candles:
            return None
        
        # Извлекаем цены закрытия
        closes = [candle['close'] for candle in candles]
        
        # Рассчитываем оптимальные EMA
        ema_short = calculate_ema(closes, ema_short_period)
        ema_long = calculate_ema(closes, ema_long_period)
        
        if ema_short is None or ema_long is None:
            return None
        
        current_close = closes[-1]
        
        # Проверяем наклон длинной EMA (сравниваем с предыдущим значением)
        if len(closes) >= ema_long_period + 1:
            prev_ema_long = calculate_ema(closes[:-1], ema_long_period)
            ema_long_slope = ema_long - prev_ema_long if prev_ema_long else 0
        else:
            ema_long_slope = 0
        
        # Проверяем минимум 3 закрытия подряд относительно длинной EMA
        recent_closes = closes[-TREND_CONFIRMATION_BARS:]
        all_above_ema_long = all(close > ema_long for close in recent_closes)
        all_below_ema_long = all(close < ema_long for close in recent_closes)
        
        # Определяем тренд согласно техзаданию
        trend = 'NEUTRAL'
        
        # UP: Close > EMA_long, EMA_short > EMA_long, наклон EMA_long > 0, минимум 3 закрытия > EMA_long
        if (current_close > ema_long and 
            ema_short > ema_long and 
            ema_long_slope > 0 and 
            all_above_ema_long):
            trend = 'UP'
        
        # DOWN: Close < EMA_long, EMA_short < EMA_long, наклон EMA_long < 0, минимум 3 закрытия < EMA_long
        elif (current_close < ema_long and 
              ema_short < ema_long and 
              ema_long_slope < 0 and 
              all_below_ema_long):
            trend = 'DOWN'
        
        return {
            'trend': trend,
            'ema_short': ema_short,
            'ema_long': ema_long,
            'ema_short_period': ema_short_period,
            'ema_long_period': ema_long_period,
            'ema_long_slope': ema_long_slope,
            'current_close': current_close,
            'confirmations': TREND_CONFIRMATION_BARS,
            'accuracy': ema_periods['accuracy']
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка анализа тренда для {symbol}: {e}")
        return None

def perform_enhanced_rsi_analysis(candles, current_rsi, symbol):
    """Выполняет улучшенный анализ RSI для монеты"""
    try:
        # Проверяем, включена ли улучшенная система
        if not SystemConfig.ENHANCED_RSI_ENABLED:
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
                        if rsi_val <= RSI_EXTREME_OVERSOLD or rsi_val >= RSI_EXTREME_OVERBOUGHT:
                            extreme_duration += 1
                        else:
                            break
                
                # Определяем тип предупреждения
                warning_type = None
                warning_message = None
            
                # Проверяем экстремальные условия
                if current_rsi <= RSI_EXTREME_OVERSOLD:
                    if extreme_duration > RSI_EXTREME_ZONE_TIMEOUT:
                        warning_type = 'EXTREME_OVERSOLD_LONG'
                        warning_message = f'RSI в экстремальной зоне {extreme_duration} свечей'
                    else:
                        warning_type = 'OVERSOLD'
                        warning_message = 'Возможная зона для LONG'
                        
                elif current_rsi >= RSI_EXTREME_OVERBOUGHT:
                    if extreme_duration > RSI_EXTREME_ZONE_TIMEOUT:
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
                logger.error(f"[ENHANCED_RSI] Ошибка анализа для {symbol}: {e}")
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
        logger.error(f"[ENHANCED_RSI] Ошибка анализа для {symbol}: {e}")
        return {
            'enabled': False,
            'warning_type': 'ERROR',
            'warning_message': f'Ошибка анализа: {str(e)}',
            'extreme_duration': 0,
            'adaptive_levels': [29, 71],
            'confirmations': {},
            'enhanced_signal': 'WAIT'
        }

