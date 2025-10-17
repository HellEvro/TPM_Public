"""Фильтры для торговых сигналов

Включает:
- check_rsi_time_filter - временной фильтр RSI
- check_exit_scam_filter - фильтр exit scam
- check_no_existing_position - проверка отсутствия позиции
- check_auto_bot_filters - проверка всех фильтров автобота
- test_exit_scam_filter - тестирование exit scam фильтра
- test_rsi_time_filter - тестирование временного фильтра
"""

import logging
import time
import threading
import concurrent.futures
from datetime import datetime

logger = logging.getLogger('BotsService')

# Импорт класса бота
try:
    from bots_modules.bot_class import NewTradingBot
except ImportError as e:
    print(f"Warning: Could not import NewTradingBot in filters: {e}")
    NewTradingBot = None

# Импорт функций расчета из calculations
try:
    from bots_modules.calculations import (
        calculate_rsi, calculate_rsi_history, calculate_ema, 
        analyze_trend_6h, perform_enhanced_rsi_analysis
    )
except ImportError as e:
    print(f"Warning: Could not import calculation functions in filters: {e}")
    def calculate_rsi(prices, period=14):
        return None
    def calculate_rsi_history(prices, period=14):
        return None
    def calculate_ema(prices, period):
        return None
    def analyze_trend_6h(symbol, exchange_obj=None):
        return None
    def perform_enhanced_rsi_analysis(candles, rsi, symbol):
        return {'enabled': False, 'enhanced_signal': 'WAIT'}

# Импорт функций зрелости из maturity
try:
    from bots_modules.maturity import (
        check_coin_maturity, check_coin_maturity_with_storage,
        add_mature_coin_to_storage, is_coin_mature_stored
    )
except ImportError as e:
    print(f"Warning: Could not import maturity functions in filters: {e}")
    def check_coin_maturity(symbol, candles):
        return {'is_mature': True, 'reason': 'Not checked'}
    def check_coin_maturity_with_storage(symbol, candles):
        return {'is_mature': True, 'reason': 'Not checked'}
    def add_mature_coin_to_storage(symbol, data, auto_save=True):
        pass
    def is_coin_mature_stored(symbol):
        return False

# Импорт функции optimal_ema из модуля
try:
    from bots_modules.optimal_ema import get_optimal_ema_periods
except ImportError as e:
    print(f"Warning: Could not import optimal_ema functions in filters: {e}")
    def get_optimal_ema_periods(symbol):
        return {'ema_short': 50, 'ema_long': 200, 'accuracy': 0}

# Импорт функций кэша из sync_and_cache
try:
    from bots_modules.sync_and_cache import save_rsi_cache
except ImportError as e:
    print(f"Warning: Could not import save_rsi_cache in filters: {e}")
    def save_rsi_cache():
        pass

# Импортируем глобальные переменные и функции из imports_and_globals
try:
    from bots_modules.imports_and_globals import (
        bots_data_lock, bots_data, rsi_data_lock, coins_rsi_data,
        BOT_STATUS, system_initialized, get_exchange
    )
    from bot_engine.bot_config import SystemConfig
except ImportError:
    bots_data_lock = threading.Lock()
    bots_data = {}
    rsi_data_lock = threading.Lock()
    coins_rsi_data = {}
    BOT_STATUS = {}
    system_initialized = False
    def get_exchange():
        return None
    # Fallback для SystemConfig
    class SystemConfig:
        RSI_OVERSOLD = 29
        RSI_OVERBOUGHT = 71
        RSI_EXIT_LONG = 65
        RSI_EXIT_SHORT = 35

def check_rsi_time_filter(candles, rsi, signal):
    """
    ГИБРИДНЫЙ ВРЕМЕННОЙ ФИЛЬТР RSI
    
    Проверяет что:
    1. Последние N свечей (из конфига, по умолчанию 8) находятся в "спокойной зоне"
       - Для SHORT: все свечи должны быть >= 65
       - Для LONG: все свечи должны быть <= 35
    2. Перед этой спокойной зоной был экстремум
       - Для SHORT: свеча с RSI >= 71
       - Для LONG: свеча с RSI <= 29
    3. С момента экстремума прошло минимум N свечей
    
    Args:
        candles: Список свечей
        rsi: Текущее значение RSI
        signal: Торговый сигнал ('ENTER_LONG' или 'ENTER_SHORT')
    
    Returns:
        dict: {'allowed': bool, 'reason': str, 'last_extreme_candles_ago': int, 'calm_candles': int}
    """
    try:
        # Получаем настройки из конфига
        with bots_data_lock:
            rsi_time_filter_enabled = bots_data.get('auto_bot_config', {}).get('rsi_time_filter_enabled', True)
            rsi_time_filter_candles = bots_data.get('auto_bot_config', {}).get('rsi_time_filter_candles', 8)
            rsi_time_filter_upper = bots_data.get('auto_bot_config', {}).get('rsi_time_filter_upper', 65)  # Спокойная зона для SHORT
            rsi_time_filter_lower = bots_data.get('auto_bot_config', {}).get('rsi_time_filter_lower', 35)  # Спокойная зона для LONG
            rsi_long_threshold = bots_data.get('auto_bot_config', {}).get('rsi_long_threshold', 29)  # Экстремум для LONG
            rsi_short_threshold = bots_data.get('auto_bot_config', {}).get('rsi_short_threshold', 71)  # Экстремум для SHORT
        
        # Если фильтр отключен - разрешаем сделку
        if not rsi_time_filter_enabled:
            return {'allowed': True, 'reason': 'RSI временной фильтр отключен', 'last_extreme_candles_ago': None, 'calm_candles': None}
        
        if len(candles) < 50:
            return {'allowed': False, 'reason': 'Недостаточно свечей для анализа', 'last_extreme_candles_ago': None, 'calm_candles': 0}
        
        # Рассчитываем историю RSI
        closes = [candle['close'] for candle in candles]
        rsi_history = calculate_rsi_history(closes, 14)
        
        min_rsi_history = max(rsi_time_filter_candles * 2 + 14, 30)
        if not rsi_history or len(rsi_history) < min_rsi_history:
            return {'allowed': False, 'reason': f'Недостаточно RSI истории (требуется {min_rsi_history})', 'last_extreme_candles_ago': None, 'calm_candles': 0}
        
        current_index = len(rsi_history) - 1
        
        if signal == 'ENTER_SHORT':
            # ПРАВИЛЬНАЯ ЛОГИКА ДЛЯ SHORT:
            # 1. Берем последние N свечей (8)
            # 2. Ищем среди них пик >= 71
            #    - Если несколько пиков - берем САМЫЙ РАННИЙ (8-ую свечу)
            #    - Если нет пиков - идем дальше в историю до 50 свечей
            # 3. От найденного пика проверяем ВСЕ свечи до текущей
            # 4. Все должны быть >= 65 (иначе был провал - вход упущен)
            
            # Шаг 1: Проверяем последние N свечей
            last_n_candles_start = max(0, current_index - rsi_time_filter_candles + 1)
            last_n_candles = rsi_history[last_n_candles_start:current_index + 1]
            
            # Ищем пики (>= 71) в последних N свечах
            peak_index = None
            for i in range(last_n_candles_start, current_index + 1):
                if rsi_history[i] >= rsi_short_threshold:
                    peak_index = i
                    break  # Берем САМЫЙ РАННИЙ пик
            
            # Шаг 2: Если не нашли пик в последних N - ищем дальше в ВСЕЙ истории
            if peak_index is None:
                # Ищем по всей доступной истории (без ограничений)
                for i in range(last_n_candles_start - 1, -1, -1):
                    if rsi_history[i] >= rsi_short_threshold:
                        peak_index = i
                        break
            
            if peak_index is None:
                # Пик не найден вообще - разрешаем (никогда не было экстремума)
                return {
                    'allowed': True,
                    'reason': f'Разрешено: пик RSI >= {rsi_short_threshold} не найден во всей истории',
                    'last_extreme_candles_ago': None,
                    'calm_candles': len(last_n_candles)
                }
            
            # Шаг 3: Проверяем ВСЕ свечи от пика до текущей
            # candles_since_peak = количество свечей С МОМЕНТА пика (включая сам пик)
            candles_since_peak = current_index - peak_index + 1
            
            # Берем все свечи ПОСЛЕ пика (не включая сам пик)
            start_check = peak_index + 1
            check_candles = rsi_history[start_check:current_index + 1]
            
            # Проверяем что ВСЕ свечи >= 65
            invalid_candles = [rsi_val for rsi_val in check_candles if rsi_val < rsi_time_filter_upper]
            
            if len(invalid_candles) > 0:
                # Есть провалы < 65 - вход упущен
                return {
                    'allowed': False,
                    'reason': f'Блокировка: {len(invalid_candles)} свечей после пика провалились < {rsi_time_filter_upper} (вход упущен)',
                    'last_extreme_candles_ago': candles_since_peak,
                    'calm_candles': len(check_candles) - len(invalid_candles)
                }
            
            # Проверяем что прошло достаточно свечей
            if len(check_candles) < rsi_time_filter_candles:
                return {
                    'allowed': False,
                    'reason': f'Блокировка: с пика прошло только {len(check_candles)} свечей (требуется {rsi_time_filter_candles})',
                    'last_extreme_candles_ago': candles_since_peak,
                    'calm_candles': len(check_candles)
                }
            
            # Все проверки пройдены!
            return {
                'allowed': True,
                'reason': f'Разрешено: с пика (свеча -{candles_since_peak}) прошло {len(check_candles)} спокойных свечей >= {rsi_time_filter_upper}',
                'last_extreme_candles_ago': candles_since_peak - 1,  # Для соответствия с вашим пониманием
                'calm_candles': len(check_candles)
            }
                
        elif signal == 'ENTER_LONG':
            # ЗЕРКАЛЬНАЯ ЛОГИКА ДЛЯ LONG (как для SHORT, только наоборот):
            # 1. Берем последние N свечей (8)
            # 2. Ищем среди них лой <= 29
            #    - Если несколько лоев - берем САМЫЙ РАННИЙ (8-ую свечу)
            #    - Если нет лоев - идем дальше в историю (БЕЗ ОГРАНИЧЕНИЙ)
            # 3. От найденного лоя проверяем ВСЕ свечи до текущей
            # 4. Все должны быть <= 35 (иначе был прорыв вверх - вход упущен)
            
            # Шаг 1: Проверяем последние N свечей
            last_n_candles_start = max(0, current_index - rsi_time_filter_candles + 1)
            last_n_candles = rsi_history[last_n_candles_start:current_index + 1]
            
            # Ищем лои (<= 29) в последних N свечах
            low_index = None
            for i in range(last_n_candles_start, current_index + 1):
                if rsi_history[i] <= rsi_long_threshold:
                    low_index = i
                    break  # Берем САМЫЙ РАННИЙ лой
            
            # Шаг 2: Если не нашли лой в последних N - ищем дальше в ВСЕЙ истории
            if low_index is None:
                # Ищем по всей доступной истории (без ограничений)
                for i in range(last_n_candles_start - 1, -1, -1):
                    if rsi_history[i] <= rsi_long_threshold:
                        low_index = i
                        break
            
            if low_index is None:
                # Лой не найден вообще - разрешаем (никогда не было экстремума)
                return {
                    'allowed': True,
                    'reason': f'Разрешено: лой RSI <= {rsi_long_threshold} не найден во всей истории',
                    'last_extreme_candles_ago': None,
                    'calm_candles': len(last_n_candles)
                }
            
            # Шаг 3: Проверяем ВСЕ свечи от лоя до текущей
            # candles_since_low = количество свечей С МОМЕНТА лоя (включая сам лой)
            candles_since_low = current_index - low_index + 1
            
            # Берем все свечи ПОСЛЕ лоя (не включая сам лой)
            start_check = low_index + 1
            check_candles = rsi_history[start_check:current_index + 1]
            
            # Проверяем что ВСЕ свечи <= 35
            invalid_candles = [rsi_val for rsi_val in check_candles if rsi_val > rsi_time_filter_lower]
            
            if len(invalid_candles) > 0:
                # Есть прорывы > 35 - вход упущен
                return {
                    'allowed': False,
                    'reason': f'Блокировка: {len(invalid_candles)} свечей после лоя поднялись > {rsi_time_filter_lower} (вход упущен)',
                    'last_extreme_candles_ago': candles_since_low,
                    'calm_candles': len(check_candles) - len(invalid_candles)
                }
            
            # Проверяем что прошло достаточно свечей
            if len(check_candles) < rsi_time_filter_candles:
                return {
                    'allowed': False,
                    'reason': f'Блокировка: с лоя прошло только {len(check_candles)} свечей (требуется {rsi_time_filter_candles})',
                    'last_extreme_candles_ago': candles_since_low,
                    'calm_candles': len(check_candles)
                }
            
            # Все проверки пройдены!
            return {
                'allowed': True,
                'reason': f'Разрешено: с лоя (свеча -{candles_since_low}) прошло {len(check_candles)} спокойных свечей <= {rsi_time_filter_lower}',
                'last_extreme_candles_ago': candles_since_low - 1,  # Для соответствия с вашим пониманием
                'calm_candles': len(check_candles)
            }
        
        return {'allowed': True, 'reason': 'Неизвестный сигнал', 'last_extreme_candles_ago': None, 'calm_candles': 0}
    
    except Exception as e:
        logger.error(f"[RSI_TIME_FILTER] Ошибка проверки временного фильтра: {e}")
        return {'allowed': False, 'reason': f'Ошибка анализа: {str(e)}', 'last_extreme_candles_ago': None, 'calm_candles': 0}

def get_coin_rsi_data(symbol, exchange_obj=None):
    """Получает RSI данные для одной монеты (6H таймфрейм)"""
    try:
        # ✅ ФИЛЬТР 1: Whitelist/Blacklist/Scope - САМЫЙ ПЕРВЫЙ!
        # Проверяем ДО загрузки данных с биржи (экономим API запросы)
        with bots_data_lock:
            auto_config = bots_data.get('auto_bot_config', {})
            scope = auto_config.get('scope', 'all')
            whitelist = auto_config.get('whitelist', [])
            blacklist = auto_config.get('blacklist', [])
        
        is_blocked_by_scope = False
        
        if scope == 'whitelist':
            # Режим ТОЛЬКО whitelist - работаем ТОЛЬКО с монетами из белого списка
            if symbol not in whitelist:
                is_blocked_by_scope = True
                logger.debug(f"[SCOPE_FILTER] {symbol}: ❌ Режим WHITELIST - монета не в белом списке")
        
        elif scope == 'blacklist':
            # Режим ТОЛЬКО blacklist - работаем со ВСЕМИ монетами КРОМЕ черного списка
            if symbol in blacklist:
                is_blocked_by_scope = True
                logger.debug(f"[SCOPE_FILTER] {symbol}: ❌ Режим BLACKLIST - монета в черном списке")
        
        elif scope == 'all':
            # Режим ALL - работаем со ВСЕМИ монетами, но проверяем оба списка
            if symbol in blacklist:
                is_blocked_by_scope = True
                logger.debug(f"[SCOPE_FILTER] {symbol}: ❌ Монета в черном списке")
            # Если в whitelist - даем приоритет (логируем, но не блокируем)
            if whitelist and symbol in whitelist:
                logger.debug(f"[SCOPE_FILTER] {symbol}: ⭐ В белом списке (приоритет)")
        
        # Минимальная задержка для избежания API Rate Limit
        time.sleep(0.1)  # Было 0.5 сек, стало 0.1 сек
        
        # logger.debug(f"[DEBUG] Обработка {symbol}...")  # Отключено для ускорения
        
        # Используем переданную биржу или глобальную
        from bots_modules.imports_and_globals import get_exchange
        exchange_to_use = exchange_obj if exchange_obj is not None else get_exchange()
        
        # Проверяем, что биржа доступна
        if exchange_to_use is None:
            logger.error(f"[ERROR] Ошибка получения данных для {symbol}: 'NoneType' object has no attribute 'get_chart_data'")
            return None
        
        # Получаем свечи 6H для расчета RSI
        chart_response = exchange_to_use.get_chart_data(symbol, '6h', '30d')
        
        if not chart_response or not chart_response.get('success'):
            logger.debug(f"[WARNING] Не удалось получить данные для {symbol}: {chart_response.get('error', 'Неизвестная ошибка') if chart_response else 'Нет ответа'}")
            return None
        
        candles = chart_response['data']['candles']
        if not candles or len(candles) < 15:  # Базовая проверка для RSI(14)
            logger.debug(f"[WARNING] Недостаточно свечей для {symbol}: {len(candles) if candles else 0}/15")
            return None
        
        # Рассчитываем RSI для 6H
        # Bybit отправляет свечи в правильном порядке для RSI (от старой к новой)
        closes = [candle['close'] for candle in candles]
        
        rsi = calculate_rsi(closes, 14)
        
        if rsi is None:
            logger.warning(f"[WARNING] Не удалось рассчитать RSI для {symbol}")
            return None
        
        # Получаем полный анализ тренда 6H
        trend_analysis = analyze_trend_6h(symbol, exchange_obj=exchange_obj)
        trend = trend_analysis['trend'] if trend_analysis else 'NEUTRAL'
        
        # Рассчитываем изменение за 24h (примерно 4 свечи 6H)
        change_24h = 0
        if len(closes) >= 5:
            change_24h = round(((closes[-1] - closes[-5]) / closes[-5]) * 100, 2)
        
        # Определяем RSI зоны согласно техзаданию
        rsi_zone = 'NEUTRAL'
        signal = 'WAIT'
        
        # ✅ ФИЛЬТР 2: Базовый RSI + Тренд
        # Получаем настройки фильтров по тренду (по умолчанию включены)
        with bots_data_lock:
            avoid_down_trend = bots_data.get('auto_bot_config', {}).get('avoid_down_trend', True)
            avoid_up_trend = bots_data.get('auto_bot_config', {}).get('avoid_up_trend', True)
        
        if rsi <= SystemConfig.RSI_OVERSOLD:  # RSI ≤ 29 
            rsi_zone = 'BUY_ZONE'
            # Проверяем нужно ли избегать DOWN тренда для LONG
            if avoid_down_trend and trend == 'DOWN':
                signal = 'WAIT'  # Ждем улучшения тренда
            else:
                signal = 'ENTER_LONG'  # Входим независимо от тренда или при хорошем тренде
        elif rsi >= SystemConfig.RSI_OVERBOUGHT:  # RSI ≥ 71
            rsi_zone = 'SELL_ZONE'
            # Проверяем нужно ли избегать UP тренда для SHORT
            if avoid_up_trend and trend == 'UP':
                signal = 'WAIT'  # Ждем ослабления тренда
            else:
                signal = 'ENTER_SHORT'  # Входим независимо от тренда или при хорошем тренде
        # RSI между 30 and 70 - нейтральная зона
        
        # ✅ ФИЛЬТР 3: Существующие позиции (СРАЗУ после базового RSI, экономим время!)
        # Проверяем: есть ли позиция на бирже БЕЗ активного бота в системе
        has_existing_position = False
        if signal in ['ENTER_LONG', 'ENTER_SHORT']:
            try:
                exch = get_exchange()
                if exch:
                    exchange_positions = exch.get_positions()
                    if isinstance(exchange_positions, tuple):
                        positions_list = exchange_positions[0] if exchange_positions else []
                    else:
                        positions_list = exchange_positions if exchange_positions else []
                    
                    # Проверяем, есть ли позиция для этой монеты
                    for pos in positions_list:
                        pos_symbol = pos.get('symbol', '').replace('USDT', '')
                        if pos_symbol == symbol and abs(float(pos.get('size', 0))) > 0:
                            # Нашли позицию! Теперь проверяем, есть ли бот для неё
                            has_bot_for_position = False
                            
                            with bots_data_lock:
                                if symbol in bots_data.get('bots', {}):
                                    bot_data = bots_data['bots'][symbol]
                                    bot_status = bot_data.get('status')
                                    # Бот считается активным если он не IDLE и не PAUSED
                                    if bot_status not in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']]:
                                        has_bot_for_position = True
                                        logger.debug(f"[POSITION_FILTER] {symbol}: ✅ Есть позиция на бирже, но есть и активный бот (статус: {bot_status})")
                            
                            if not has_bot_for_position:
                                # Позиция есть, но бота нет - это ручная позиция или бот был удалён
                                has_existing_position = True
                                signal = 'WAIT'  # Блокируем
                                rsi_zone = 'NEUTRAL'
                                logger.debug(f"[POSITION_FILTER] {symbol}: ❌ Есть позиция БЕЗ активного бота - блокируем создание нового бота")
                            break
            except Exception as e:
                logger.warning(f"[POSITION_FILTER] {symbol}: Ошибка проверки позиций: {e}")
        
        # ✅ ФИЛЬТР 4: Enhanced RSI (после проверки позиций)
        # Проверяем волатильность, дивергенции, объемы
        enhanced_analysis = perform_enhanced_rsi_analysis(candles, rsi, symbol)
        
        # Если Enhanced RSI включен и дает другой сигнал - используем его
        if signal in ['ENTER_LONG', 'ENTER_SHORT']:
            if enhanced_analysis.get('enabled') and enhanced_analysis.get('enhanced_signal'):
                original_signal = signal
                enhanced_signal = enhanced_analysis.get('enhanced_signal')
                if enhanced_signal != original_signal:
                    logger.info(f"[ENHANCED_RSI] {symbol}: Сигнал изменен {original_signal} → {enhanced_signal}")
                    signal = enhanced_signal
                    # Если Enhanced RSI говорит WAIT - блокируем
                    if signal == 'WAIT':
                        rsi_zone = 'NEUTRAL'
        
        # ✅ ФИЛЬТР 5: Зрелость монеты (проверяем ПОСЛЕ Enhanced RSI)
        # Проверяем зрелость монеты ДЛЯ ВСЕХ МОНЕТ при каждой загрузке
        with bots_data_lock:
            enable_maturity_check = bots_data.get('auto_bot_config', {}).get('enable_maturity_check', True)
        
        # Проверяем зрелость монеты из хранилища или выполняем проверку
        if enable_maturity_check:
            # ✅ ИСПОЛЬЗУЕМ хранилище зрелых монет для быстрой проверки
            is_mature = check_coin_maturity_stored_or_verify(symbol)
            
            if not is_mature and signal in ['ENTER_LONG', 'ENTER_SHORT']:
                logger.debug(f"[MATURITY] {symbol}: Монета незрелая - сигнал {signal} заблокирован")
                # Меняем сигнал на WAIT, но не исключаем монету из списка
                signal = 'WAIT'
                rsi_zone = 'NEUTRAL'
        
        # Получаем оптимальные EMA периоды для монеты
        ema_periods = get_optimal_ema_periods(symbol)
        
        # closes[-1] - это самая НОВАЯ цена (последняя свеча в массиве)
        current_price = closes[-1]
        
        # ✅ ПРАВИЛЬНЫЙ ПОРЯДОК ФИЛЬТРОВ согласно логике:
        # 1. Whitelist/Blacklist/Scope → уже проверено в начале
        # 2. Базовый RSI + Тренд → уже проверено выше
        # 3. Существующие позиции → уже проверено выше (РАННИЙ выход!)
        # 4. Enhanced RSI → уже проверено выше
        # 5. Зрелость монеты → уже проверено выше
        # 6. ExitScam фильтр → проверяем здесь
        # 7. RSI временной фильтр → проверяем здесь
        
        exit_scam_info = None
        time_filter_info = None
        
        # Проверяем фильтры только если монета в зоне входа (LONG/SHORT)
        if signal in ['ENTER_LONG', 'ENTER_SHORT']:
            # 6. Проверка ExitScam фильтра
            exit_scam_passed = check_exit_scam_filter(symbol, {})
            if not exit_scam_passed:
                exit_scam_info = {
                    'blocked': True,
                    'reason': 'Обнаружены резкие движения цены (ExitScam фильтр)',
                    'filter_type': 'exit_scam'
                }
                signal = 'WAIT'
                rsi_zone = 'NEUTRAL'
            else:
                exit_scam_info = {
                    'blocked': False,
                    'reason': 'ExitScam фильтр пройден',
                    'filter_type': 'exit_scam'
                }
            
            # 7. Проверка RSI временного фильтра (только если ExitScam пройден)
            if signal in ['ENTER_LONG', 'ENTER_SHORT']:
                time_filter_result = check_rsi_time_filter(candles, rsi, signal)
                time_filter_info = {
                    'blocked': not time_filter_result['allowed'],
                    'reason': time_filter_result['reason'],
                    'filter_type': 'time_filter',
                    'last_extreme_candles_ago': time_filter_result.get('last_extreme_candles_ago'),
                    'calm_candles': time_filter_result.get('calm_candles')
                }
                
                # Если временной фильтр блокирует - меняем сигнал на WAIT
                if not time_filter_result['allowed']:
                    signal = 'WAIT'
                    rsi_zone = 'NEUTRAL'
        
        # ✅ ПРИМЕНЯЕМ БЛОКИРОВКУ ПО SCOPE
        # Scope фильтр (если монета в черном списке или не в белом)
        if is_blocked_by_scope:
            signal = 'WAIT'
            rsi_zone = 'NEUTRAL'
        
        result = {
            'symbol': symbol,
            'rsi6h': round(rsi, 1),
            'trend6h': trend,
            'rsi_zone': rsi_zone,
            'signal': signal,
            'price': current_price,
            'change24h': change_24h,
            'last_update': datetime.now().isoformat(),
            'trend_analysis': trend_analysis,
            'ema_periods': {
                'ema_short': ema_periods['ema_short'],
                'ema_long': ema_periods['ema_long'],
                'accuracy': ema_periods['accuracy'],
                'analysis_method': ema_periods['analysis_method']
            },
            # Добавляем результаты улучшенного анализа RSI
            'enhanced_rsi': enhanced_analysis,
            # Добавляем информацию о временном фильтре
            'time_filter_info': time_filter_info,
            # Добавляем информацию об ExitScam фильтре
            'exit_scam_info': exit_scam_info,
            # ✅ ДОБАВЛЯЕМ флаги блокировки (для UI)
            'blocked_by_scope': is_blocked_by_scope,
            'has_existing_position': has_existing_position,
            # ✅ ДОБАВЛЯЕМ флаг зрелости монеты
            'is_mature': is_mature if enable_maturity_check else True
        }
        
        # Логируем торговые сигналы и блокировки тренда
        trend_emoji = '📈' if trend == 'UP' else '📉' if trend == 'DOWN' else '➡️'
        
        if signal in ['ENTER_LONG', 'ENTER_SHORT']:
            logger.info(f"[SIGNAL] 🎯 {symbol}: RSI={rsi:.1f} {trend_emoji}{trend} (${current_price:.4f}) → {signal}")
        elif signal == 'WAIT' and rsi <= SystemConfig.RSI_OVERSOLD and trend == 'DOWN' and avoid_down_trend:
            logger.debug(f"[FILTER] 🚫 {symbol}: RSI={rsi:.1f} {trend_emoji}{trend} LONG заблокирован (фильтр DOWN тренда)")
        elif signal == 'WAIT' and rsi >= SystemConfig.RSI_OVERBOUGHT and trend == 'UP' and avoid_up_trend:
            logger.debug(f"[FILTER] 🚫 {symbol}: RSI={rsi:.1f} {trend_emoji}{trend} SHORT заблокирован (фильтр UP тренда)")
        
        return result
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка получения данных для {symbol}: {e}")
        return None

def load_all_coins_rsi():
    """Загружает RSI 6H для всех доступных монет"""
    global coins_rsi_data
    
    try:
        with rsi_data_lock:
            if coins_rsi_data['update_in_progress']:
                logger.info("Обновление RSI уже выполняется...")
                return False
            coins_rsi_data['update_in_progress'] = True
        
        logger.info("[RSI] 🔄 Начинаем загрузку RSI 6H для всех монет...")
        
        # Получаем актуальную ссылку на биржу
        try:
            from bots_modules.imports_and_globals import get_exchange
            current_exchange = get_exchange()
        except Exception as e:
            logger.error(f"[RSI] ❌ Ошибка получения биржи: {e}")
            current_exchange = None
        
        # Получаем список всех пар
        if not current_exchange:
            logger.error("[RSI] ❌ Биржа не инициализирована")
            with rsi_data_lock:
                coins_rsi_data['update_in_progress'] = False
            return False
            
        pairs = current_exchange.get_all_pairs()
        logger.info(f"[RSI] 🔍 Получено пар с биржи: {len(pairs) if pairs else 0}")
        
        if not pairs or not isinstance(pairs, list):
            logger.error("[RSI] ❌ Не удалось получить список пар с биржи")
            return False
        
        logger.info(f"[RSI] 📊 Найдено {len(pairs)} торговых пар для анализа")
        
        # Обновляем счетчики
        with rsi_data_lock:
            coins_rsi_data['total_coins'] = len(pairs)
            coins_rsi_data['successful_coins'] = 0
            coins_rsi_data['failed_coins'] = 0
        
        # Получаем RSI данные для всех пар пакетно с инкрементальным обновлением
        batch_size = 50  # Увеличиваем размер пакета для ускорения загрузки
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(pairs) + batch_size - 1)//batch_size
            
            logger.info(f"[BATCH] Обработка пакета {batch_num}/{total_batches} ({len(batch)} монет)")
            
            # Параллельная загрузка RSI для пакета (3 воркера для ускорения)
            batch_coins_data = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_symbol = {executor.submit(get_coin_rsi_data, symbol, current_exchange): symbol for symbol in batch}
                
                # Уменьшаем таймауты для ускорения (2 минуты для пакета, 15 секунд на монету)
                try:
                    for future in concurrent.futures.as_completed(future_to_symbol, timeout=120):
                        try:
                            result = future.result(timeout=15)  # Уменьшаем до 15 секунд
                            if result:
                                batch_coins_data[result['symbol']] = result
                                
                                # Зрелость монеты проверяется в check_coin_maturity_stored_or_verify
                                # при попытке создать бота, а не здесь
                                symbol = result['symbol']
                                
                                with rsi_data_lock:
                                    coins_rsi_data['successful_coins'] += 1
                            else:
                                with rsi_data_lock:
                                    coins_rsi_data['failed_coins'] += 1
                        except concurrent.futures.TimeoutError:
                            symbol = future_to_symbol[future]
                            # logger.warning(f"⏰ Таймаут для {symbol} (пропускаем)")  # Отключено для чистоты логов
                            with rsi_data_lock:
                                coins_rsi_data['failed_coins'] += 1
                        except Exception as e:
                            symbol = future_to_symbol[future]
                            # logger.warning(f"[WARNING] Ошибка обработки {symbol}: {e}")  # Отключено для чистоты логов
                            with rsi_data_lock:
                                coins_rsi_data['failed_coins'] += 1
                except concurrent.futures.TimeoutError:
                    # Обработка таймаута всего пакета
                    unfinished = len([f for f in future_to_symbol.keys() if not f.done()])
                    logger.warning(f"⏰ Таймаут пакета! Не завершено: {unfinished} из {len(batch)} монет")
                    with rsi_data_lock:
                        coins_rsi_data['failed_coins'] += unfinished
            
            # ИНКРЕМЕНТАЛЬНОЕ ОБНОВЛЕНИЕ: Обновляем данные после каждого пакета
            with rsi_data_lock:
                coins_rsi_data['coins'].update(batch_coins_data)
                coins_rsi_data['last_update'] = datetime.now().isoformat()
                logger.info(f"[INCREMENTAL] Обновлено {len(batch_coins_data)} монет из пакета {batch_num}")
            
            # Пауза между пакетами для предотвращения rate limiting
            time.sleep(2.0)  # 2 секунды между пакетами (было 10 сек)
            
            # Логируем прогресс каждые 5 пакетов (чаще для инкрементального обновления)
            if batch_num % 5 == 0:
                with rsi_data_lock:
                    success_count = coins_rsi_data['successful_coins']
                    failed_count = coins_rsi_data['failed_coins']
                    total_processed = success_count + failed_count
                    progress_percent = round((total_processed / len(pairs)) * 100, 1)
                    coins_count = len(coins_rsi_data['coins'])
                    logger.info(f"[RSI] ⏳ Прогресс: {progress_percent}% ({total_processed}/{len(pairs)}) - В UI доступно {coins_count} монет")
        
        # Финальное обновление флага
        with rsi_data_lock:
            coins_rsi_data['update_in_progress'] = False
        
        logger.info(f"[RSI] ✅ Обновление завершено, флаг update_in_progress сброшен")
        
        # Финальный отчет
        with rsi_data_lock:
            success_count = coins_rsi_data['successful_coins']
            failed_count = coins_rsi_data['failed_coins']
            
        # Подсчитываем сигналы
        with rsi_data_lock:
            enter_long_count = sum(1 for coin in coins_rsi_data['coins'].values() if coin.get('signal') == 'ENTER_LONG')
            enter_short_count = sum(1 for coin in coins_rsi_data['coins'].values() if coin.get('signal') == 'ENTER_SHORT')
        
        logger.info(f"[RSI] ✅ Загрузка завершена: {success_count}/{len(pairs)} монет | Сигналы: {enter_long_count} LONG + {enter_short_count} SHORT")
        
        if failed_count > 0:
            logger.warning(f"[RSI] ⚠️ Ошибок: {failed_count} монет")
        
        # Сохраняем RSI данные в кэш
        save_rsi_cache()
        
        # Обрабатываем торговые сигналы для существующих ботов
        process_trading_signals_for_all_bots(exchange_obj=current_exchange)
        
        # Проверяем автобот сигналы для создания новых ботов
        process_auto_bot_signals(exchange_obj=current_exchange)  # ВКЛЮЧЕНО!
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка загрузки RSI данных: {str(e)}")
        with rsi_data_lock:
            coins_rsi_data['update_in_progress'] = False
    finally:
        # Гарантированно сбрасываем флаг обновления
        with rsi_data_lock:
            if coins_rsi_data['update_in_progress']:
                logger.warning(f"[RSI] ⚠️ Принудительный сброс флага update_in_progress")
                coins_rsi_data['update_in_progress'] = False
        return False

def get_effective_signal(coin):
    """
    Универсальная функция для определения эффективного сигнала монеты
    
    ЛОГИКА ПРОВЕРКИ ТРЕНДОВ (упрощенная):
    - НЕ открываем SHORT если RSI > 71 И тренд = UP
    - НЕ открываем LONG если RSI < 29 И тренд = DOWN
    - NEUTRAL тренд разрешает любые сделки
    - Тренд только усиливает возможность, но не блокирует полностью
    
    Args:
        coin (dict): Данные монеты
        
    Returns:
        str: Эффективный сигнал (ENTER_LONG, ENTER_SHORT, WAIT)
    """
    symbol = coin.get('symbol', 'UNKNOWN')
    
    # Получаем настройки автобота
    with bots_data_lock:
        auto_config = bots_data.get('auto_bot_config', {})
        avoid_down_trend = auto_config.get('avoid_down_trend', True)
        avoid_up_trend = auto_config.get('avoid_up_trend', True)
        rsi_long_threshold = auto_config.get('rsi_long_threshold', 29)
        rsi_short_threshold = auto_config.get('rsi_short_threshold', 71)
        
    # Получаем данные монеты
    rsi = coin.get('rsi6h', 50)
    trend = coin.get('trend', coin.get('trend6h', 'NEUTRAL'))
    
    # ✅ КРИТИЧНО: Проверяем зрелость монеты ПЕРВЫМ ДЕЛОМ
    # Незрелые монеты НЕ МОГУТ иметь активных ботов и НЕ ДОЛЖНЫ показываться в LONG/SHORT фильтрах!
    base_signal = coin.get('signal', 'WAIT')
    if base_signal == 'WAIT':
        # Монета незрелая - не показываем её в фильтрах
        return 'WAIT'
    
    # ✅ Монета зрелая - проверяем Enhanced RSI сигнал
    enhanced_rsi = coin.get('enhanced_rsi', {})
    if enhanced_rsi.get('enabled') and enhanced_rsi.get('enhanced_signal'):
        signal = enhanced_rsi.get('enhanced_signal')
    else:
        # Используем базовый сигнал
        signal = base_signal
    
    # Если сигнал WAIT - возвращаем сразу
    if signal == 'WAIT':
        return signal
    
    # УПРОЩЕННАЯ ПРОВЕРКА ТРЕНДОВ - только экстремальные случаи
    if signal == 'ENTER_SHORT' and avoid_up_trend and rsi >= rsi_short_threshold and trend == 'UP':
        logger.debug(f"[SIGNAL] {symbol}: ❌ SHORT заблокирован (RSI={rsi:.1f} >= {rsi_short_threshold} + UP тренд)")
        return 'WAIT'
    
    if signal == 'ENTER_LONG' and avoid_down_trend and rsi <= rsi_long_threshold and trend == 'DOWN':
        logger.debug(f"[SIGNAL] {symbol}: ❌ LONG заблокирован (RSI={rsi:.1f} <= {rsi_long_threshold} + DOWN тренд)")
        return 'WAIT'
    
    # Все проверки пройдены
    logger.debug(f"[SIGNAL] {symbol}: ✅ {signal} разрешен (RSI={rsi:.1f}, Trend={trend})")
    return signal

def process_auto_bot_signals(exchange_obj=None):
    """Новая логика автобота согласно требованиям"""
    try:
        # Проверяем, включен ли автобот
        with bots_data_lock:
            auto_bot_enabled = bots_data['auto_bot_config']['enabled']
            
            if not auto_bot_enabled:
                logger.debug("[NEW_AUTO] ⏹️ Автобот выключен")
                return
            
            max_concurrent = bots_data['auto_bot_config']['max_concurrent']
            current_active = sum(1 for bot in bots_data['bots'].values() 
                               if bot['status'] not in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']])
            
            if current_active >= max_concurrent:
                logger.debug(f"[NEW_AUTO] 🚫 Достигнут лимит активных ботов ({current_active}/{max_concurrent})")
                return
        
        logger.info("[NEW_AUTO] 🔍 Проверка сигналов для создания новых ботов...")
        
        # Получаем монеты с сигналами
        potential_coins = []
        with rsi_data_lock:
            for symbol, coin_data in coins_rsi_data['coins'].items():
                rsi = coin_data.get('rsi6h')
                trend = coin_data.get('trend6h', 'NEUTRAL')
                
                if rsi is None:
                    continue
                
                # ✅ ИСПОЛЬЗУЕМ get_effective_signal() который учитывает ВСЕ проверки:
                # - RSI временной фильтр
                # - Enhanced RSI
                # - Зрелость монеты (base_signal)
                # - Тренды
                signal = get_effective_signal(coin_data)
                
                # Если сигнал ENTER_LONG или ENTER_SHORT - проверяем остальные фильтры
                if signal in ['ENTER_LONG', 'ENTER_SHORT']:
                    # Проверяем дополнительные условия (whitelist/blacklist, ExitScam, позиции)
                    if check_new_autobot_filters(symbol, signal, coin_data):
                        potential_coins.append({
                            'symbol': symbol,
                            'rsi': rsi,
                            'trend': trend,
                            'signal': signal,
                            'coin_data': coin_data
                        })
        
        logger.info(f"[NEW_AUTO] 🎯 Найдено {len(potential_coins)} потенциальных сигналов")
        
        # Создаем ботов для найденных сигналов
        created_bots = 0
        for coin in potential_coins[:max_concurrent - current_active]:
            symbol = coin['symbol']
            
            # Проверяем, нет ли уже бота для этого символа
            with bots_data_lock:
                if symbol in bots_data['bots']:
                    logger.debug(f"[NEW_AUTO] ⚠️ Бот для {symbol} уже существует")
                    continue
            
            # Создаем нового бота
            try:
                logger.info(f"[NEW_AUTO] 🚀 Создаем бота для {symbol} ({coin['signal']}, RSI: {coin['rsi']:.1f})")
                create_new_bot(symbol, exchange_obj=exchange_obj)
                created_bots += 1
                
            except Exception as e:
                logger.error(f"[NEW_AUTO] ❌ Ошибка создания бота для {symbol}: {e}")
        
        if created_bots > 0:
            logger.info(f"[NEW_AUTO] ✅ Создано {created_bots} новых ботов")
        
    except Exception as e:
        logger.error(f"[NEW_AUTO] ❌ Ошибка обработки сигналов: {e}")

def process_trading_signals_for_all_bots(exchange_obj=None):
    """Обрабатывает торговые сигналы для всех активных ботов с новым классом"""
    try:
        # Проверяем, инициализирована ли система
        if not system_initialized:
            logger.warning("[NEW_BOT_SIGNALS] ⏳ Система еще не инициализирована - пропускаем обработку")
            return
        
        with bots_data_lock:
            # Фильтруем только активных ботов (исключаем IDLE и PAUSED)
            active_bots = {symbol: bot for symbol, bot in bots_data['bots'].items() 
                          if bot['status'] not in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']]}
        
        if not active_bots:
            logger.debug("[NEW_BOT_SIGNALS] ⏳ Нет активных ботов для обработки")
            return
        
        logger.info(f"[NEW_BOT_SIGNALS] 🔍 Обрабатываем {len(active_bots)} активных ботов: {list(active_bots.keys())}")
        
        for symbol, bot_data in active_bots.items():
            try:
                logger.debug(f"[NEW_BOT_SIGNALS] 🔍 Обрабатываем бота {symbol}...")
                
                # Используем переданную биржу или глобальную переменную
                from bots_modules.imports_and_globals import get_exchange
                exchange_to_use = exchange_obj if exchange_obj else get_exchange()
                
                # Создаем экземпляр нового бота из сохраненных данных
                trading_bot = NewTradingBot(symbol, bot_data, exchange_to_use)
                
                # Получаем RSI данные для монеты
                rsi_data = None
                with rsi_data_lock:
                    rsi_data = coins_rsi_data['coins'].get(symbol)
                
                if not rsi_data:
                    logger.debug(f"[NEW_BOT_SIGNALS] ❌ {symbol}: RSI данные не найдены")
                    continue
                
                logger.debug(f"[NEW_BOT_SIGNALS] ✅ {symbol}: RSI={rsi_data.get('rsi6h')}, Trend={rsi_data.get('trend6h')}")
                
                # Обрабатываем торговые сигналы через метод update
                external_signal = rsi_data.get('signal')
                external_trend = rsi_data.get('trend6h')
                
                signal_result = trading_bot.update(
                    force_analysis=True, 
                    external_signal=external_signal, 
                    external_trend=external_trend
                )
                
                logger.debug(f"[NEW_BOT_SIGNALS] 🔄 {symbol}: Результат update: {signal_result}")
                
                # Обновляем данные бота в хранилище если есть изменения
                if signal_result and signal_result.get('success', False):
                    with bots_data_lock:
                        bots_data['bots'][symbol] = trading_bot.to_dict()
                    
                    # Логируем торговые действия
                    action = signal_result.get('action')
                    if action in ['OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT']:
                        logger.info(f"[NEW_BOT_SIGNALS] 🎯 {symbol}: {action} выполнено")
                else:
                    logger.debug(f"[NEW_BOT_SIGNALS] ⏳ {symbol}: Нет торговых сигналов")
        
            except Exception as e:
                logger.error(f"[NEW_BOT_SIGNALS] ❌ Ошибка обработки сигналов для {symbol}: {e}")
        
    except Exception as e:
        logger.error(f"[NEW_BOT_SIGNALS] ❌ Ошибка обработки торговых сигналов: {str(e)}")

def check_new_autobot_filters(symbol, signal, coin_data):
    """Проверяет фильтры для нового автобота"""
    try:
        # ✅ ВСЕ ФИЛЬТРЫ УЖЕ ПРОВЕРЕНЫ в get_coin_rsi_data():
        # 1. Whitelist/blacklist/scope
        # 2. Базовый RSI + Тренд
        # 3. Существующие позиции (РАННИЙ выход!)
        # 4. Enhanced RSI
        # 5. Зрелость монеты
        # 6. ExitScam фильтр
        # 7. RSI временной фильтр
        
        # Здесь делаем только дубль-проверку зрелости и ExitScam на всякий случай
        
        # Дубль-проверка зрелости монеты
        if not check_coin_maturity_stored_or_verify(symbol):
            logger.debug(f"[NEW_AUTO_FILTER] {symbol}: Монета незрелая")
            return False
        
        # Дубль-проверка ExitScam
        if not check_exit_scam_filter(symbol, coin_data):
            logger.warning(f"[NEW_AUTO_FILTER] {symbol}: ❌ БЛОКИРОВКА: Обнаружены резкие движения цены (ExitScam)")
            return False
        else:
            logger.info(f"[NEW_AUTO_FILTER] {symbol}: ✅ ExitScam фильтр пройден")
        
        logger.debug(f"[NEW_AUTO_FILTER] {symbol}: ✅ Все дубль-проверки пройдены")
        return True
        
    except Exception as e:
        logger.error(f"[NEW_AUTO_FILTER] {symbol}: Ошибка проверки фильтров: {e}")
        return False

def check_coin_maturity_stored_or_verify(symbol):
    """Проверяет зрелость монеты из хранилища или выполняет проверку"""
    try:
        # Сначала проверяем хранилище
        if is_coin_mature_stored(symbol):
            return True
        
        # Если нет в хранилище, выполняем проверку
        exch = get_exchange()
        if not exch:
            logger.warning(f"[MATURITY_CHECK] {symbol}: Биржа не инициализирована")
            return False
        
        chart_response = exch.get_chart_data(symbol, '6h', '30d')
        if not chart_response or not chart_response.get('success'):
            logger.warning(f"[MATURITY_CHECK] {symbol}: Не удалось получить свечи")
            return False
        
        candles = chart_response.get('data', {}).get('candles', [])
        if not candles:
            logger.warning(f"[MATURITY_CHECK] {symbol}: Нет свечей")
            return False
        
        maturity_result = check_coin_maturity_with_storage(symbol, candles)
        return maturity_result['is_mature']
        
    except Exception as e:
        logger.error(f"[MATURITY_CHECK] {symbol}: Ошибка проверки зрелости: {e}")
        return False

def check_exit_scam_filter(symbol, coin_data):
    """
    EXIT SCAM ФИЛЬТР + AI ANOMALY DETECTION
    
    Защита от резких движений цены (памп/дамп/скам):
    1. Одна свеча превысила максимальный % изменения
    2. N свечей суммарно превысили максимальный % изменения
    3. ИИ обнаружил аномалию (если включен)
    """
    try:
        # Получаем настройки из конфига
        with bots_data_lock:
            exit_scam_enabled = bots_data.get('auto_bot_config', {}).get('exit_scam_enabled', True)
            exit_scam_candles = bots_data.get('auto_bot_config', {}).get('exit_scam_candles', 10)
            single_candle_percent = bots_data.get('auto_bot_config', {}).get('exit_scam_single_candle_percent', 15.0)
            multi_candle_count = bots_data.get('auto_bot_config', {}).get('exit_scam_multi_candle_count', 4)
            multi_candle_percent = bots_data.get('auto_bot_config', {}).get('exit_scam_multi_candle_percent', 50.0)
        
        # Если фильтр отключен - разрешаем
        if not exit_scam_enabled:
            logger.debug(f"[EXIT_SCAM] {symbol}: Фильтр отключен")
            return True
        
        # Получаем свечи
        exch = get_exchange()
        if not exch:
            return False
        
        chart_response = exch.get_chart_data(symbol, '6h', '30d')
        if not chart_response or not chart_response.get('success'):
            return False
        
        candles = chart_response.get('data', {}).get('candles', [])
        if len(candles) < exit_scam_candles:
            return False
        
        # Проверяем последние N свечей (из конфига)
        recent_candles = candles[-exit_scam_candles:]
        
        logger.info(f"[EXIT_SCAM] {symbol}: Анализ последних {exit_scam_candles} свечей")
        logger.info(f"[EXIT_SCAM] {symbol}: Настройки - одна свеча: {single_candle_percent}%, {multi_candle_count} свечей: {multi_candle_percent}%")
        
        # 1. ПРОВЕРКА: Одна свеча превысила максимальный % изменения
        for i, candle in enumerate(recent_candles):
            open_price = candle['open']
            close_price = candle['close']
            
            # Процент изменения свечи (от открытия до закрытия)
            price_change = abs((close_price - open_price) / open_price) * 100
            
            if price_change > single_candle_percent:
                logger.warning(f"[EXIT_SCAM] {symbol}: ❌ БЛОКИРОВКА: Свеча #{i+1} превысила лимит {single_candle_percent}% (было {price_change:.1f}%)")
                logger.info(f"[EXIT_SCAM] {symbol}: Свеча: O={open_price:.4f} C={close_price:.4f} H={candle['high']:.4f} L={candle['low']:.4f}")
                return False
        
        # 2. ПРОВЕРКА: N свечей суммарно превысили максимальный % изменения
        if len(recent_candles) >= multi_candle_count:
            # Берем последние N свечей для суммарного анализа
            multi_candles = recent_candles[-multi_candle_count:]
            
            first_open = multi_candles[0]['open']
            last_close = multi_candles[-1]['close']
            
            # Суммарное изменение от первой свечи до последней
            total_change = abs((last_close - first_open) / first_open) * 100
            
            if total_change > multi_candle_percent:
                logger.warning(f"[EXIT_SCAM] {symbol}: ❌ БЛОКИРОВКА: {multi_candle_count} свечей превысили суммарный лимит {multi_candle_percent}% (было {total_change:.1f}%)")
                logger.info(f"[EXIT_SCAM] {symbol}: Первая свеча: {first_open:.4f}, Последняя свеча: {last_close:.4f}")
                return False
        
        logger.info(f"[EXIT_SCAM] {symbol}: ✅ Базовые проверки пройдены")
        
        # 3. ПРОВЕРКА: AI Anomaly Detection (если включен)
        try:
            from bot_engine.bot_config import AIConfig
            
            if AIConfig.AI_ENABLED and AIConfig.AI_ANOMALY_DETECTION_ENABLED:
                try:
                    from bot_engine.ai.ai_manager import get_ai_manager
                    
                    ai_manager = get_ai_manager()
                    
                    if ai_manager and ai_manager.anomaly_detector:
                        # Анализируем свечи с помощью ИИ
                        anomaly_result = ai_manager.anomaly_detector.detect(candles)
                        
                        if anomaly_result.get('is_anomaly'):
                            severity = anomaly_result.get('severity', 0)
                            anomaly_type = anomaly_result.get('anomaly_type', 'UNKNOWN')
                            
                            # Блокируем если severity > threshold
                            if severity > AIConfig.AI_ANOMALY_BLOCK_THRESHOLD:
                                logger.warning(
                                    f"[EXIT_SCAM] {symbol}: ❌ БЛОКИРОВКА (AI): "
                                    f"Обнаружена аномалия {anomaly_type} "
                                    f"(severity: {severity:.2%})"
                                )
                                return False
                            else:
                                logger.warning(
                                    f"[EXIT_SCAM] {symbol}: ⚠️ ПРЕДУПРЕЖДЕНИЕ (AI): "
                                    f"Аномалия {anomaly_type} "
                                    f"(severity: {severity:.2%} - ниже порога {AIConfig.AI_ANOMALY_BLOCK_THRESHOLD:.2%})"
                                )
                        else:
                            logger.debug(f"[EXIT_SCAM] {symbol}: ✅ AI: Аномалий не обнаружено")
                    
                except ImportError as e:
                    logger.debug(f"[EXIT_SCAM] {symbol}: AI модуль не доступен: {e}")
                except Exception as e:
                    logger.error(f"[EXIT_SCAM] {symbol}: Ошибка AI проверки: {e}")
        
        except ImportError:
            pass  # AIConfig не доступен - пропускаем AI проверку
        
        logger.info(f"[EXIT_SCAM] {symbol}: ✅ РЕЗУЛЬТАТ: ПРОЙДЕН (включая AI)")
        return True
        
    except Exception as e:
        logger.error(f"[EXIT_SCAM] {symbol}: Ошибка проверки: {e}")
        return False

# Алиас для обратной совместимости
check_anti_dump_pump = check_exit_scam_filter

def check_no_existing_position(symbol, signal):
    """Проверяет, что нет существующих позиций на бирже"""
    try:
        exch = get_exchange()
        if not exch:
            return False
        
        exchange_positions = exch.get_positions()
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
                    logger.debug(f"[POSITION_CHECK] {symbol}: Уже есть позиция {existing_side}")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"[POSITION_CHECK] {symbol}: Ошибка проверки позиций: {e}")
        return False

def create_new_bot(symbol, config=None, exchange_obj=None):
    """Создает нового бота"""
    try:
        from bots_modules.imports_and_globals import get_exchange
        exchange_to_use = exchange_obj if exchange_obj else get_exchange()
        
        # Получаем размер позиции из конфига
        with bots_data_lock:
            default_volume = bots_data['auto_bot_config']['default_position_size']
        
        # Создаем конфигурацию бота
        bot_config = {
            'symbol': symbol,
            'status': BOT_STATUS['RUNNING'],  # ✅ ИСПРАВЛЕНО: бот должен быть активным
            'created_at': datetime.now().isoformat(),
            'opened_by_autobot': True,
            'volume_mode': 'usdt',
            'volume_value': default_volume  # ✅ ИСПРАВЛЕНО: используем значение из конфига
        }
        
        # Создаем бота
        new_bot = NewTradingBot(symbol, bot_config, exchange_to_use)
        
        # Сохраняем в bots_data
        with bots_data_lock:
            bots_data['bots'][symbol] = new_bot.to_dict()
        
        logger.info(f"[CREATE_BOT] ✅ Бот для {symbol} создан успешно")
        return new_bot
        
    except Exception as e:
        logger.error(f"[CREATE_BOT] ❌ Ошибка создания бота для {symbol}: {e}")
        raise

def check_auto_bot_filters(symbol):
    """Старая функция - оставлена для совместимости"""
    return False  # Блокируем все

def test_exit_scam_filter(symbol):
    """Тестирует ExitScam фильтр для конкретной монеты"""
    try:
        # Получаем настройки из конфига
        with bots_data_lock:
            exit_scam_enabled = bots_data.get('auto_bot_config', {}).get('exit_scam_enabled', True)
            exit_scam_candles = bots_data.get('auto_bot_config', {}).get('exit_scam_candles', 10)
            single_candle_percent = bots_data.get('auto_bot_config', {}).get('exit_scam_single_candle_percent', 15.0)
            multi_candle_count = bots_data.get('auto_bot_config', {}).get('exit_scam_multi_candle_count', 4)
            multi_candle_percent = bots_data.get('auto_bot_config', {}).get('exit_scam_multi_candle_percent', 50.0)
        
        logger.info(f"[TEST_EXIT_SCAM] 🔍 Тестируем ExitScam фильтр для {symbol}")
        logger.info(f"[TEST_EXIT_SCAM] ⚙️ Настройки:")
        logger.info(f"[TEST_EXIT_SCAM] ⚙️ - Включен: {exit_scam_enabled}")
        logger.info(f"[TEST_EXIT_SCAM] ⚙️ - Анализ свечей: {exit_scam_candles}")
        logger.info(f"[TEST_EXIT_SCAM] ⚙️ - Лимит одной свечи: {single_candle_percent}%")
        logger.info(f"[TEST_EXIT_SCAM] ⚙️ - Лимит {multi_candle_count} свечей: {multi_candle_percent}%")
        
        if not exit_scam_enabled:
            logger.info(f"[TEST_EXIT_SCAM] {symbol}: ⚠️ Фильтр ОТКЛЮЧЕН в конфиге")
            return
        
        # Получаем свечи
        exch = get_exchange()
        if not exch:
            logger.error(f"[TEST_EXIT_SCAM] {symbol}: Биржа не инициализирована")
            return
        
        chart_response = exch.get_chart_data(symbol, '6h', '30d')
        if not chart_response or not chart_response.get('success'):
            logger.error(f"[TEST_EXIT_SCAM] {symbol}: Не удалось получить свечи")
            return
        
        candles = chart_response.get('data', {}).get('candles', [])
        if len(candles) < exit_scam_candles:
            logger.error(f"[TEST_EXIT_SCAM] {symbol}: Недостаточно свечей ({len(candles)})")
            return
        
        # Анализируем последние N свечей (из конфига)
        recent_candles = candles[-exit_scam_candles:]
        
        logger.info(f"[TEST_EXIT_SCAM] {symbol}: Анализ последних {exit_scam_candles} свечей (6H каждая)")
        
        # Показываем детали каждой свечи
        for i, candle in enumerate(recent_candles):
            open_price = candle['open']
            close_price = candle['close']
            high_price = candle['high']
            low_price = candle['low']
            
            price_change = ((close_price - open_price) / open_price) * 100
            candle_range = ((high_price - low_price) / open_price) * 100
            
            logger.info(f"[TEST_EXIT_SCAM] {symbol}: Свеча {i+1}: O={open_price:.4f} C={close_price:.4f} H={high_price:.4f} L={low_price:.4f} | Изменение: {price_change:+.1f}% | Диапазон: {candle_range:.1f}%")
        
        # Тестируем фильтр с детальным логированием
        logger.info(f"[TEST_EXIT_SCAM] {symbol}: 🔍 Запускаем проверку ExitScam фильтра...")
        result = check_exit_scam_filter(symbol, {})
        
        if result:
            logger.info(f"[TEST_EXIT_SCAM] {symbol}: ✅ РЕЗУЛЬТАТ: ПРОЙДЕН")
        else:
            logger.warning(f"[TEST_EXIT_SCAM] {symbol}: ❌ РЕЗУЛЬТАТ: ЗАБЛОКИРОВАН")
        
        # Дополнительный анализ
        logger.info(f"[TEST_EXIT_SCAM] {symbol}: 📊 Дополнительный анализ:")
        
        # 1. Проверка отдельных свечей
        extreme_single_count = 0
        for i, candle in enumerate(recent_candles):
            open_price = candle['open']
            close_price = candle['close']
            
            price_change = abs((close_price - open_price) / open_price) * 100
            
            if price_change > single_candle_percent:
                extreme_single_count += 1
                logger.warning(f"[TEST_EXIT_SCAM] {symbol}: ❌ Превышение лимита одной свечи #{i+1}: {price_change:.1f}% > {single_candle_percent}%")
        
        # 2. Проверка суммарного изменения за N свечей
        if len(recent_candles) >= multi_candle_count:
            multi_candles = recent_candles[-multi_candle_count:]
            first_open = multi_candles[0]['open']
            last_close = multi_candles[-1]['close']
            
            total_change = abs((last_close - first_open) / first_open) * 100
            
            logger.info(f"[TEST_EXIT_SCAM] {symbol}: 📈 {multi_candle_count}-свечечный анализ: {total_change:.1f}% (порог: {multi_candle_percent}%)")
            
            if total_change > multi_candle_percent:
                logger.warning(f"[TEST_EXIT_SCAM] {symbol}: ❌ Превышение суммарного лимита: {total_change:.1f}% > {multi_candle_percent}%")
        
    except Exception as e:
        logger.error(f"[TEST_EXIT_SCAM] {symbol}: Ошибка тестирования: {e}")

# Алиас для обратной совместимости
test_anti_pump_filter = test_exit_scam_filter

def test_rsi_time_filter(symbol):
    """Тестирует RSI временной фильтр для конкретной монеты"""
    try:
        logger.info(f"[TEST_RSI_TIME] 🔍 Тестируем RSI временной фильтр для {symbol}")
        
        # Получаем свечи
        exch = get_exchange()
        if not exch:
            logger.error(f"[TEST_RSI_TIME] {symbol}: Биржа не инициализирована")
            return
                
        chart_response = exch.get_chart_data(symbol, '6h', '30d')
        if not chart_response or not chart_response.get('success'):
            logger.error(f"[TEST_RSI_TIME] {symbol}: Не удалось получить свечи")
            return
        
        candles = chart_response.get('data', {}).get('candles', [])
        if len(candles) < 50:
            logger.error(f"[TEST_RSI_TIME] {symbol}: Недостаточно свечей ({len(candles)})")
            return
        
        # Получаем текущий RSI
        with rsi_data_lock:
            coin_data = coins_rsi_data['coins'].get(symbol)
            if not coin_data:
                logger.error(f"[TEST_RSI_TIME] {symbol}: Нет RSI данных")
                return
            
            current_rsi = coin_data.get('rsi6h', 0)
            signal = coin_data.get('signal', 'WAIT')
        
        # Определяем ОРИГИНАЛЬНЫЙ сигнал на основе только RSI (игнорируя другие фильтры)
        with bots_data_lock:
            rsi_long_threshold = bots_data.get('auto_bot_config', {}).get('rsi_long_threshold', 29)
            rsi_short_threshold = bots_data.get('auto_bot_config', {}).get('rsi_short_threshold', 71)
        
        original_signal = 'WAIT'
        if current_rsi <= rsi_long_threshold:
            original_signal = 'ENTER_LONG'
        elif current_rsi >= rsi_short_threshold:
            original_signal = 'ENTER_SHORT'
        
        logger.info(f"[TEST_RSI_TIME] {symbol}: Текущий RSI={current_rsi:.1f}, Оригинальный сигнал={original_signal}, Финальный сигнал={signal}")
        
        # Тестируем временной фильтр с ОРИГИНАЛЬНЫМ сигналом
        time_filter_result = check_rsi_time_filter(candles, current_rsi, original_signal)
        
        logger.info(f"[TEST_RSI_TIME] {symbol}: Результат временного фильтра:")
        logger.info(f"[TEST_RSI_TIME] {symbol}: Разрешено: {time_filter_result['allowed']}")
        logger.info(f"[TEST_RSI_TIME] {symbol}: Причина: {time_filter_result['reason']}")
        if 'calm_candles' in time_filter_result and time_filter_result['calm_candles'] is not None:
            logger.info(f"[TEST_RSI_TIME] {symbol}: Спокойных свечей: {time_filter_result['calm_candles']}")
        if 'last_extreme_candles_ago' in time_filter_result and time_filter_result['last_extreme_candles_ago'] is not None:
            logger.info(f"[TEST_RSI_TIME] {symbol}: Последний экстремум: {time_filter_result['last_extreme_candles_ago']} свечей назад")
        
        # Показываем историю RSI для анализа
        closes = [candle['close'] for candle in candles]
        rsi_history = calculate_rsi_history(closes, 14)
        
        if rsi_history:
            logger.info(f"[TEST_RSI_TIME] {symbol}: Последние 20 значений RSI:")
            last_20_rsi = rsi_history[-20:] if len(rsi_history) >= 20 else rsi_history
            
            # Получаем пороги для подсветки
            with bots_data_lock:
                rsi_long_threshold = bots_data.get('auto_bot_config', {}).get('rsi_long_threshold', 29)
                rsi_short_threshold = bots_data.get('auto_bot_config', {}).get('rsi_short_threshold', 71)
                rsi_time_filter_upper = bots_data.get('auto_bot_config', {}).get('rsi_time_filter_upper', 65)
                rsi_time_filter_lower = bots_data.get('auto_bot_config', {}).get('rsi_time_filter_lower', 35)
            
            for i, rsi_val in enumerate(last_20_rsi):
                # Индекс от конца истории
                index_from_end = len(last_20_rsi) - i - 1
                
                # Определяем маркеры для наглядности
                markers = []
                if rsi_val >= rsi_short_threshold:
                    markers.append(f"🔴ПИК>={rsi_short_threshold}")
                elif rsi_val <= rsi_long_threshold:
                    markers.append(f"🟢ЛОЙ<={rsi_long_threshold}")
                
                if rsi_val >= rsi_time_filter_upper:
                    markers.append(f"✅>={rsi_time_filter_upper}")
                elif rsi_val <= rsi_time_filter_lower:
                    markers.append(f"✅<={rsi_time_filter_lower}")
                
                marker_str = " ".join(markers) if markers else ""
                logger.info(f"[TEST_RSI_TIME] {symbol}: Свеча -{index_from_end}: RSI={rsi_val:.1f} {marker_str}")
        
    except Exception as e:
        logger.error(f"[TEST_RSI_TIME] {symbol}: Ошибка тестирования: {e}")

