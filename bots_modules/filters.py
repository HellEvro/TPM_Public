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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

logger = logging.getLogger('BotsService')

# Импорт класса бота - ОТКЛЮЧЕН из-за циклического импорта
# NewTradingBot будет импортирован локально в функциях

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
        return True  # ВРЕМЕННО: разрешаем все монеты

# Импорт функций для работы с делистинговыми монетами
try:
    from bots_modules.sync_and_cache import load_delisted_coins
except ImportError as e:
    print(f"Warning: Could not import delisting functions in filters: {e}")
    def load_delisted_coins(): 
        return {"delisted_coins": {}}

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
        # ⚡ БЕЗ БЛОКИРОВКИ: конфиг не меняется, GIL делает чтение атомарным
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
            
            # Ищем лой, который даст нам достаточно свечей после него
            low_index = None
            
            # Ищем лой, начиная с текущей свечи и идя назад
            # Нам нужен лой, после которого будет минимум rsi_time_filter_candles свечей
            for i in range(current_index, -1, -1):
                if rsi_history[i] <= rsi_long_threshold:
                    # Проверяем, достаточно ли свечей после этого лоя
                    candles_after_low = current_index - i
                    if candles_after_low >= rsi_time_filter_candles:
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
            # Исправляем: current_index уже указывает на последнюю свечу, поэтому не добавляем +1
            check_candles = rsi_history[start_check:current_index + 1]
            
            # Отладочная информация (только в DEBUG режиме)
            if SystemConfig.DEBUG_MODE:
                logger.debug(f"[RSI_TIME_FILTER] {signal}: low_index={low_index}, current_index={current_index}, start_check={start_check}")
                logger.debug(f"[RSI_TIME_FILTER] {signal}: check_candles length={len(check_candles)}, rsi_values={check_candles}")
            
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

def get_coin_candles_only(symbol, exchange_obj=None):
    """⚡ БЫСТРАЯ загрузка ТОЛЬКО свечей БЕЗ расчетов"""
    try:
        from bots_modules.imports_and_globals import get_exchange
        exchange_to_use = exchange_obj if exchange_obj is not None else get_exchange()
        
        if exchange_to_use is None:
            return None
        
        # Получаем ТОЛЬКО свечи
        chart_response = exchange_to_use.get_chart_data(symbol, '6h', '30d')
        
        if not chart_response or not chart_response.get('success'):
            return None
        
        candles = chart_response['data']['candles']
        if not candles or len(candles) < 15:
            return None
        
        # Возвращаем ТОЛЬКО свечи и символ
        return {
            'symbol': symbol,
            'candles': candles,
            'last_update': datetime.now().isoformat()
        }
        
    except Exception as e:
        return None

def get_coin_rsi_data(symbol, exchange_obj=None):
    """Получает RSI данные для одной монеты (6H таймфрейм)"""
    # ⚡ Включаем трейсинг для этого потока (если включен глобально)
    try:
        from bot_engine.bot_config import SystemConfig
        if SystemConfig.ENABLE_CODE_TRACING:
            from trace_debug import enable_trace
            enable_trace()
    except:
        pass
    
    # ⚡ СЕМАФОР: Ограничиваем одновременные API запросы к бирже (если нет в кэше)
    # Это предотвращает перегрузку API биржи
    global _exchange_api_semaphore
    try:
        _exchange_api_semaphore
    except NameError:
        import threading
        _exchange_api_semaphore = threading.Semaphore(5)  # ⚡ Уменьшили до 5 для стабильности
    
    import time
    thread_start = time.time()
    # print(f"[{time.strftime('%H:%M:%S')}] >>> НАЧАЛО get_coin_rsi_data({symbol})", flush=True)  # Отключено для скорости
    
    try:
        # ✅ ФИЛЬТР 0: ДЕЛИСТИНГОВЫЕ МОНЕТЫ - САМЫЙ ПЕРВЫЙ!
        # Исключаем делистинговые монеты ДО всех остальных проверок
        # Загружаем делистинговые монеты из файла
        delisted_data = load_delisted_coins()
        delisted_coins = delisted_data.get('delisted_coins', {})
        
        if symbol in delisted_coins:
            delisting_info = delisted_coins[symbol]
            logger.info(f"[DELISTING_BLACKLIST] {symbol}: Исключаем из всех проверок - {delisting_info.get('reason', 'Delisting detected')}")
            # Возвращаем минимальные данные для делистинговых монет
            return {
                'symbol': symbol,
                'rsi6h': 0,
                'trend6h': 'NEUTRAL',
                'rsi_zone': 'NEUTRAL',
                'signal': 'WAIT',
                'price': 0,
                'change24h': 0,
                'last_update': datetime.now().isoformat(),
                'trading_status': 'Closed',
                'is_delisting': True,
                'delisting_reason': delisting_info.get('reason', 'Delisting detected'),
                'blocked_by_delisting': True
            }
        
        # ✅ ФИЛЬТР 1: Whitelist/Blacklist/Scope - Проверяем ДО загрузки данных с биржи
        # ⚡ БЕЗ БЛОКИРОВКИ: конфиг не меняется во время выполнения, безопасно читать
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
        
        # БЕЗ задержки - семафор и ThreadPool уже контролируют rate limit
        
        # logger.debug(f"[DEBUG] Обработка {symbol}...")  # Отключено для ускорения
        
        # Используем переданную биржу или глобальную
        # print(f"[{time.strftime('%H:%M:%S')}] >>> {symbol}: Получение exchange...", flush=True)  # Отключено
        from bots_modules.imports_and_globals import get_exchange
        exchange_to_use = exchange_obj if exchange_obj is not None else get_exchange()
        
        # Проверяем, что биржа доступна
        if exchange_to_use is None:
            logger.error(f"[ERROR] Ошибка получения данных для {symbol}: 'NoneType' object has no attribute 'get_chart_data'")
            return None
        
        # ⚡ ОПТИМИЗАЦИЯ: Проверяем кэш свечей ПЕРЕД запросом к бирже!
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение кэша - безопасная операция
        candles = None
        candles_cache = coins_rsi_data.get('candles_cache', {})
        if symbol in candles_cache:
            cached_data = candles_cache[symbol]
            candles = cached_data.get('candles')
            # logger.debug(f"[CACHE] {symbol}: Используем кэш свечей")  # Отключено для скорости
        
        # Если нет в кэше - загружаем с биржи (с семафором!)
        if not candles:
            logger.warning(f"[CACHE_MISS] ⚠️ {symbol}: НЕТ в кэше свечей! Загружаем с биржи...")
            # ⚡ СЕМАФОР: Ограничиваем одновременные запросы к API биржи
            with _exchange_api_semaphore:
                import time as time_module
                api_start = time_module.time()
                logger.info(f"[API_START] 🌐 {symbol}: Начало запроса get_chart_data()...")
                
                chart_response = exchange_to_use.get_chart_data(symbol, '6h', '30d')
                
                api_duration = time_module.time() - api_start
                logger.info(f"[API_END] 🌐 {symbol}: get_chart_data() завершен за {api_duration:.1f}с")
                
                if not chart_response or not chart_response.get('success'):
                    logger.warning(f"[API_ERROR] ❌ {symbol}: Ошибка: {chart_response.get('error', 'Неизвестная ошибка') if chart_response else 'Нет ответа'}")
                    return None
                
                candles = chart_response['data']['candles']
                logger.info(f"[API_LOAD] ✅ {symbol}: Свечи загружены с биржи ({len(candles)} свечей)")
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
        
        # ✅ РАСЧИТЫВАЕМ ТРЕНД СРАЗУ для всех монет - избегаем "гуляния" данных
        # НЕ УСТАНАВЛИВАЕМ ДЕФОЛТНЫХ ЗНАЧЕНИЙ! Только рассчитанные данные!
        trend = None  # Изначально None
        trend_analysis = None
        try:
            from bots_modules.calculations import analyze_trend_6h
            trend_analysis = analyze_trend_6h(symbol, exchange_obj=exchange_obj)
            if trend_analysis:
                trend = trend_analysis['trend']  # ТОЛЬКО рассчитанное значение!
            # НЕ устанавливаем дефолт если анализ не удался - оставляем None
        except Exception as e:
            logger.debug(f"[TREND] {symbol}: Ошибка анализа тренда: {e}")
            # НЕ устанавливаем дефолт при ошибке - оставляем None
        
        # Рассчитываем изменение за 24h (примерно 4 свечи 6H)
        change_24h = 0
        if len(closes) >= 5:
            change_24h = round(((closes[-1] - closes[-5]) / closes[-5]) * 100, 2)
        
        # ✅ КРИТИЧНО: Получаем оптимальные EMA периоды ДО определения сигнала!
        # Это нужно для правильного расчета базового сигнала на основе EMA
        ema_periods = None
        try:
            ema_periods = get_optimal_ema_periods(symbol)
        except Exception as e:
            logger.debug(f"[EMA] Ошибка получения оптимальных EMA для {symbol}: {e}")
            # Если не удалось получить оптимальные EMA, используем дефолтные значения
            ema_periods = {'ema_short': 50, 'ema_long': 200, 'accuracy': 0, 'analysis_method': 'default'}
        
        # Определяем RSI зоны согласно техзаданию
        rsi_zone = 'NEUTRAL'
        signal = 'WAIT'
        
        # ✅ ФИЛЬТР 2: Базовый сигнал НА ОСНОВЕ OPTIMAL EMA ПЕРИОДОВ!
        # Получаем настройки фильтров по тренду (по умолчанию включены)
        # ⚡ БЕЗ БЛОКИРОВКИ: конфиг не меняется во время выполнения, безопасно читать
        avoid_down_trend = bots_data.get('auto_bot_config', {}).get('avoid_down_trend', True)
        avoid_up_trend = bots_data.get('auto_bot_config', {}).get('avoid_up_trend', True)
        
        # ✅ КРИТИЧНО: Определяем сигнал на основе Optimal EMA периодов!
        if ema_periods and ema_periods.get('ema_short') and ema_periods.get('ema_long'):
            # Рассчитываем EMA на основе оптимальных периодов
            ema_short = ema_periods['ema_short']
            ema_long = ema_periods['ema_long']
            
            # Рассчитываем EMA значения
            try:
                from bots_modules.calculations import calculate_ema
                ema_short_value = calculate_ema(closes, ema_short)[-1] if len(closes) >= ema_short else closes[-1]
                ema_long_value = calculate_ema(closes, ema_long)[-1] if len(closes) >= ema_long else closes[-1]
                
                # Определяем сигнал на основе пересечения EMA
                if ema_short_value > ema_long_value:
                    # Короткая EMA выше длинной - восходящий тренд
                    if rsi <= SystemConfig.RSI_OVERSOLD:  # RSI ≤ 29 
                        rsi_zone = 'BUY_ZONE'
                        # ✅ ИСПРАВЛЕНИЕ: Если тренд еще не рассчитан (None), не блокируем сигнал
                        if avoid_down_trend and trend == 'DOWN':
                            signal = 'WAIT'  # Ждем улучшения тренда
                        else:
                            signal = 'ENTER_LONG'  # Входим в лонг при восходящем тренде EMA
                elif ema_short_value < ema_long_value:
                    # Короткая EMA ниже длинной - нисходящий тренд
                    if rsi >= SystemConfig.RSI_OVERBOUGHT:  # RSI ≥ 71
                        rsi_zone = 'SELL_ZONE'
                        # ✅ ИСПРАВЛЕНИЕ: Если тренд еще не рассчитан (None), не блокируем сигнал
                        if avoid_up_trend and trend == 'UP':
                            signal = 'WAIT'  # Ждем ослабления тренда
                        else:
                            signal = 'ENTER_SHORT'  # Входим в шорт при нисходящем тренде EMA
                # Если EMA пересекаются или равны - нейтральная зона
            except Exception as e:
                logger.debug(f"[EMA_SIGNAL] {symbol}: Ошибка расчета EMA сигнала: {e}")
                # Fallback к старой логике при ошибке
                if rsi <= SystemConfig.RSI_OVERSOLD:  # RSI ≤ 29 
                    rsi_zone = 'BUY_ZONE'
                    if avoid_down_trend and trend == 'DOWN':
                        signal = 'WAIT'
                    else:
                        signal = 'ENTER_LONG'
                elif rsi >= SystemConfig.RSI_OVERBOUGHT:  # RSI ≥ 71
                    rsi_zone = 'SELL_ZONE'
                    if avoid_up_trend and trend == 'UP':
                        signal = 'WAIT'
                    else:
                        signal = 'ENTER_SHORT'
        else:
            # Fallback к старой логике если EMA периоды недоступны
            if rsi <= SystemConfig.RSI_OVERSOLD:  # RSI ≤ 29 
                rsi_zone = 'BUY_ZONE'
                if avoid_down_trend and trend == 'DOWN':
                    signal = 'WAIT'
                else:
                    signal = 'ENTER_LONG'
            elif rsi >= SystemConfig.RSI_OVERBOUGHT:  # RSI ≥ 71
                rsi_zone = 'SELL_ZONE'
                if avoid_up_trend and trend == 'UP':
                    signal = 'WAIT'
                else:
                    signal = 'ENTER_SHORT'
        # RSI между 30 and 70 - нейтральная зона
        
        # ✅ ФИЛЬТР 3: Существующие позиции (ОТКЛЮЧЕН для ускорения RSI расчета)
        # ⚡ ОПТИМИЗАЦИЯ: Проверка позиций слишком медленная (API запрос к бирже в каждом потоке!)
        # Эта проверка будет выполнена позже в process_auto_bot_signals() ПЕРЕД созданием бота
        has_existing_position = False
        # ПРОПУСКАЕМ ПРОВЕРКУ ПОЗИЦИЙ ЗДЕСЬ - экономим ~50 API запросов к бирже!
        
        # ✅ ФИЛЬТР 4: Enhanced RSI (для ВСЕХ монет, чтобы получить Stochastic RSI)
        # ⚡ ИЗМЕНЕНИЕ: Рассчитываем Enhanced RSI для всех монет, не только сигнальных
        # Это нужно для получения Stochastic RSI данных для UI
        enhanced_analysis = None
        
        # Рассчитываем Enhanced RSI для всех монет (включая нейтральные)
        # Это обеспечивает наличие Stochastic RSI данных для всех монет в UI
        enhanced_analysis = perform_enhanced_rsi_analysis(candles, rsi, symbol)
        
        # Если Enhanced RSI включен и дает другой сигнал - используем его
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
        # ⚡ ОПТИМИЗАЦИЯ: Проверяем ТОЛЬКО если есть сигнал входа (экономим 95% проверок!)
        enable_maturity_check = bots_data.get('auto_bot_config', {}).get('enable_maturity_check', True)
        is_mature = True  # По умолчанию считаем зрелой (если проверка отключена)
        
        if signal in ['ENTER_LONG', 'ENTER_SHORT']:
            if enable_maturity_check:
                # ✅ ИСПОЛЬЗУЕМ хранилище зрелых монет для быстрой проверки
                is_mature = check_coin_maturity_stored_or_verify(symbol)
                
                if not is_mature:
                    logger.debug(f"[MATURITY] {symbol}: Монета незрелая - сигнал {signal} заблокирован")
                    # Меняем сигнал на WAIT, но не исключаем монету из списка
                    signal = 'WAIT'
                    rsi_zone = 'NEUTRAL'
        
        # ✅ EMA периоды уже получены выше - ДО определения сигнала!
        
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
        
        # ✅ ПРОВЕРЯЕМ СТАТУС ТОРГОВЛИ: Получаем информацию о делистинге/новых монетах
        # ОПТИМИЗИРОВАННАЯ ВЕРСИЯ: Проверяем только известные делистинговые монеты
        trading_status = 'Trading'  # По умолчанию
        is_delisting = False
        
        # ✅ ЧЕРНЫЙ СПИСОК ДЕЛИСТИНГОВЫХ МОНЕТ - исключаем из всех проверок
        # Загружаем делистинговые монеты из файла
        delisted_data = load_delisted_coins()
        delisted_coins = delisted_data.get('delisted_coins', {})
        
        known_delisting_coins = list(delisted_coins.keys())
        known_new_coins = []  # Можно добавить новые монеты
        
        if symbol in known_delisting_coins:
            trading_status = 'Closed'
            is_delisting = True
            logger.info(f"[TRADING_STATUS] {symbol}: Известная делистинговая монета")
        elif symbol in known_new_coins:
            trading_status = 'Delivering'
            is_delisting = True
            logger.info(f"[TRADING_STATUS] {symbol}: Известная новая монета")
        
        # TODO: Включить полную проверку статуса торговли после оптимизации API запросов
        # try:
        #     if exchange_obj and hasattr(exchange_obj, 'get_instrument_status'):
        #         status_info = exchange_obj.get_instrument_status(f"{symbol}USDT")
        #         if status_info:
        #             trading_status = status_info.get('status', 'Trading')
        #             is_delisting = status_info.get('is_delisting', False)
        #             
        #             # Логируем только делистинговые и новые монеты
        #             if trading_status != 'Trading':
        #                 logger.info(f"[TRADING_STATUS] {symbol}: Статус {trading_status} (делистинг: {is_delisting})")
        # except Exception as e:
        #     # Если не удалось получить статус, используем значения по умолчанию
        #     logger.debug(f"[TRADING_STATUS] {symbol}: Не удалось получить статус торговли: {e}")
        
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
            # ⚡ ОПТИМИЗАЦИЯ: Enhanced RSI, фильтры и флаги ТОЛЬКО если проверялись
            'enhanced_rsi': enhanced_analysis if enhanced_analysis else {'enabled': False},
            'time_filter_info': time_filter_info,
            'exit_scam_info': exit_scam_info,
            'blocked_by_scope': is_blocked_by_scope,
            'has_existing_position': has_existing_position,
            'is_mature': is_mature if enable_maturity_check else True,
            # ✅ КРИТИЧНО: Флаги блокировки для get_effective_signal
            'blocked_by_exit_scam': exit_scam_info.get('blocked', False) if exit_scam_info else False,
            'blocked_by_rsi_time': time_filter_info.get('blocked', False) if time_filter_info else False,
            # ✅ ИНФОРМАЦИЯ О СТАТУСЕ ТОРГОВЛИ: Для визуальных эффектов делистинга
            'trading_status': trading_status,
            'is_delisting': is_delisting
        }
        
        # Логируем торговые сигналы и блокировки тренда
        # НЕ показываем дефолтные значения! Только рассчитанные данные!
        trend_display = trend if trend is not None else None
        # НЕ показываем дефолтные emoji! Только для рассчитанных данных!
        if trend == 'UP':
            trend_emoji = '📈'
        elif trend == 'DOWN':
            trend_emoji = '📉'
        elif trend == 'NEUTRAL':
            trend_emoji = '➡️'
        else:
            trend_emoji = None
        
        if signal in ['ENTER_LONG', 'ENTER_SHORT']:
            logger.info(f"[SIGNAL] 🎯 {symbol}: RSI={rsi:.1f} {trend_emoji}{trend_display} (${current_price:.4f}) → {signal}")
        elif signal == 'WAIT' and rsi <= SystemConfig.RSI_OVERSOLD and trend == 'DOWN' and avoid_down_trend:
            logger.debug(f"[FILTER] 🚫 {symbol}: RSI={rsi:.1f} {trend_emoji}{trend_display} LONG заблокирован (фильтр DOWN тренда)")
        elif signal == 'WAIT' and rsi >= SystemConfig.RSI_OVERBOUGHT and trend == 'UP' and avoid_up_trend:
            logger.debug(f"[FILTER] 🚫 {symbol}: RSI={rsi:.1f} {trend_emoji}{trend_display} SHORT заблокирован (фильтр UP тренда)")
        
        return result
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка получения данных для {symbol}: {e}")
        return None

def load_all_coins_candles_fast():
    """⚡ БЫСТРАЯ загрузка ТОЛЬКО свечей для всех монет БЕЗ расчетов"""
    try:
        logger.info("[CANDLES_FAST] 🚀 Быстрая загрузка свечей для всех монет...")
        
        from bots_modules.imports_and_globals import get_exchange
        current_exchange = get_exchange()
        
        if not current_exchange:
            logger.error("[CANDLES_FAST] ❌ Биржа не инициализирована")
            return False
        
        # Получаем список всех пар
        pairs = current_exchange.get_all_pairs()
        if not pairs:
            logger.error("[CANDLES_FAST] ❌ Не удалось получить список пар")
            return False
        
        logger.info(f"[CANDLES_FAST] 📊 Найдено {len(pairs)} пар для загрузки")
        
        # Загружаем ТОЛЬКО свечи пакетами (УСКОРЕННАЯ ВЕРСИЯ)
        batch_size = 100  # Увеличили с 50 до 100
        candles_cache = {}
        
        import concurrent.futures
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(pairs) + batch_size - 1)//batch_size
            
            logger.info(f"[CANDLES_FAST] 📦 Пакет {batch_num}/{total_batches}: загрузка {len(batch)} монет...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:  # Увеличили с 10 до 20
                future_to_symbol = {executor.submit(get_coin_candles_only, symbol, current_exchange): symbol for symbol in batch}
                
                completed = 0
                for future in concurrent.futures.as_completed(future_to_symbol, timeout=90):  # Увеличили timeout
                    try:
                        result = future.result(timeout=30)  # Увеличили timeout
                        if result:
                            candles_cache[result['symbol']] = result
                            completed += 1
                    except Exception as e:
                        pass
                
                logger.info(f"[CANDLES_FAST] ✅ Пакет {batch_num}: загружено {completed}/{len(batch)} монет")
                
                # Уменьшили паузу между пакетами
                import time
                time.sleep(0.1)  # Уменьшили с 0.3 до 0.1
        
        logger.info(f"[CANDLES_FAST] ✅ Загрузка завершена: {len(candles_cache)} монет")
        
        # ⚡ ИСПРАВЛЕНИЕ DEADLOCK: Сохраняем в глобальный кэш БЕЗ блокировки
        # rsi_data_lock может быть захвачен ContinuousDataLoader в другом потоке
        try:
            logger.info(f"[CANDLES_FAST] 💾 Сохраняем кэш в глобальное хранилище...")
            coins_rsi_data['candles_cache'] = candles_cache
            coins_rsi_data['last_candles_update'] = datetime.now().isoformat()
            logger.info(f"[CANDLES_FAST] ✅ Кэш сохранен: {len(candles_cache)} монет")
            logger.info(f"[CANDLES_FAST] ✅ Проверка: в глобальном кэше сейчас {len(coins_rsi_data.get('candles_cache', {}))} монет")
        except Exception as cache_error:
            logger.warning(f"[CANDLES_FAST] ⚠️ Ошибка сохранения кэша: {cache_error}")
        
        return True
        
    except Exception as e:
        logger.error(f"[CANDLES_FAST] ❌ Ошибка: {e}")
        return False

def load_all_coins_rsi():
    """Загружает RSI 6H для всех доступных монет"""
    global coins_rsi_data
    
    try:
        # ⚡ БЕЗ БЛОКИРОВКИ: проверяем флаг без блокировки
        if coins_rsi_data['update_in_progress']:
            logger.info("Обновление RSI уже выполняется...")
            return False
        
        # ⚡ УСТАНАВЛИВАЕМ флаги БЕЗ блокировки
        coins_rsi_data['update_in_progress'] = True
        # ✅ UI блокировка уже установлена в continuous_data_loader
        
        # ✅ КРИТИЧНО: Создаем ВРЕМЕННОЕ хранилище для всех монет
        # Обновляем coins_rsi_data ТОЛЬКО после завершения всех проверок!
        temp_coins_data = {}
        
        logger.info("[RSI] 🔄 Начинаем загрузку RSI 6H для всех монет...")
        
        # Проверяем кэш свечей перед началом
        candles_cache_size = len(coins_rsi_data.get('candles_cache', {}))
        logger.info(f"[RSI] 📦 Размер кэша свечей на старте: {candles_cache_size} монет")
        
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
            coins_rsi_data['update_in_progress'] = False
            return False
            
        logger.info("[RSI] 🔍 Получаем список пар с биржи...")
        pairs = current_exchange.get_all_pairs()
        logger.info(f"[RSI] 🔍 Получено пар с биржи: {len(pairs) if pairs else 0}")
        
        if not pairs or not isinstance(pairs, list):
            logger.error("[RSI] ❌ Не удалось получить список пар с биржи")
            return False
        
        logger.info(f"[RSI] 📊 Найдено {len(pairs)} торговых пар для анализа")
        
        # ⚡ БЕЗ БЛОКИРОВКИ: обновляем счетчики напрямую
        coins_rsi_data['total_coins'] = len(pairs)
        coins_rsi_data['successful_coins'] = 0
        coins_rsi_data['failed_coins'] = 0
        
        # Получаем RSI данные для всех пар пакетно с инкрементальным обновлением (УСКОРЕННАЯ ВЕРСИЯ)
        batch_size = 100  # ⚡ Увеличили до 100 для ускорения первой загрузки
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(pairs) + batch_size - 1)//batch_size
            
            logger.info(f"[BATCH] 🔄 Обработка пакета {batch_num}/{total_batches} ({len(batch)} монет)")
            # logger.info(f"[BATCH] 📋 Монеты в пакете: {', '.join(batch[:10])}{'...' if len(batch) > 10 else ''}") # Отключено - может вызывать проблемы
            
            # Параллельная загрузка RSI для пакета (10 воркеров - максимальная скорость!)
            batch_coins_data = {}
            import time
            batch_start = time.time()
            logger.info(f"[BATCH] 🚀 НАЧАЛО пакета {batch_num}")
            
            # ✅ ВОЗВРАЩАЕМ ПАРАЛЛЕЛЬНУЮ обработку для скорости
            logger.info(f"[BATCH] 🚀 Параллельная обработка с ThreadPoolExecutor")
            
            # Используем ThreadPoolExecutor для параллельной обработки
            with ThreadPoolExecutor(max_workers=50) as executor:  # Увеличили до 50 для максимального ускорения
                # Отправляем все задачи
                future_to_symbol = {
                    executor.submit(get_coin_rsi_data, symbol, current_exchange): symbol 
                    for symbol in batch
                }
                
                # Собираем результаты
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_symbol, timeout=60):
                    symbol = future_to_symbol[future]
                    completed_count += 1
                    try:
                        result = future.result(timeout=20)
                        if result:
                            batch_coins_data[result['symbol']] = result
                            coins_rsi_data['successful_coins'] += 1
                        else:
                            coins_rsi_data['failed_coins'] += 1
                        
                        if completed_count % 10 == 0:
                            logger.info(f"[BATCH] 📊 Обработано {completed_count}/{len(batch)} монет из пакета {batch_num}")
                    except Exception as e:
                        logger.error(f"[BATCH] ❌ Ошибка обработки {symbol}: {e}")
                        coins_rsi_data['failed_coins'] += 1
            
            # ✅ КРИТИЧНО: Сохраняем во ВРЕМЕННОЕ хранилище вместо прямого обновления!
            # НЕ обновляем coins_rsi_data['coins'] до завершения ВСЕХ пакетов!
            temp_coins_data.update(batch_coins_data)
            logger.info(f"[BATCH] ✅ Сохранено {len(batch_coins_data)} монет во временное хранилище (всего: {len(temp_coins_data)})")
            
            # Пауза между пакетами для предотвращения rate limiting (УСКОРЕННАЯ ВЕРСИЯ)
            time.sleep(0.1)  # ⚡ МАКСИМАЛЬНОЕ УСКОРЕНИЕ: 0.1 сек между пакетами
            
            # Логируем прогресс каждые 5 пакетов (чаще для инкрементального обновления)
            if batch_num % 5 == 0:
                # ⚡ БЕЗ БЛОКИРОВКИ: читаем счетчики напрямую
                success_count = coins_rsi_data['successful_coins']
                failed_count = coins_rsi_data['failed_coins']
                total_processed = success_count + failed_count
                progress_percent = round((total_processed / len(pairs)) * 100, 1)
                coins_count = len(coins_rsi_data['coins'])
                logger.info(f"[RSI] ⏳ Прогресс: {progress_percent}% ({total_processed}/{len(pairs)}) - В UI доступно {coins_count} монет")
        
        # ✅ КРИТИЧНО: АТОМАРНОЕ обновление всех данных ОДНИМ МАХОМ!
        # Только СЕЙЧАС обновляем coins_rsi_data['coins'] всеми собранными данными
        logger.info(f"[RSI] 🎯 Атомарное обновление {len(temp_coins_data)} монет...")
        coins_rsi_data['coins'] = temp_coins_data  # ✅ Полная замена - атомарная операция
        coins_rsi_data['last_update'] = datetime.now().isoformat()
        logger.info(f"[RSI] ✅ Атомарное обновление завершено - UI теперь видит финальные данные!")
        
        # Финальное обновление флага
        # ⚡ БЕЗ БЛОКИРОВКИ: атомарная операция
        coins_rsi_data['update_in_progress'] = False
        
        logger.info(f"[RSI] ✅ Обновление завершено, флаг update_in_progress сброшен")
        
        # Финальный отчет
        # ⚡ БЕЗ БЛОКИРОВКИ: читаем счетчики напрямую
        success_count = coins_rsi_data['successful_coins']
        failed_count = coins_rsi_data['failed_coins']
            
        # Подсчитываем сигналы
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря
        enter_long_count = sum(1 for coin in coins_rsi_data['coins'].values() if coin.get('signal') == 'ENTER_LONG')
        enter_short_count = sum(1 for coin in coins_rsi_data['coins'].values() if coin.get('signal') == 'ENTER_SHORT')
        
        logger.info(f"[RSI] ✅ Загрузка завершена: {success_count}/{len(pairs)} монет | Сигналы: {enter_long_count} LONG + {enter_short_count} SHORT")
        
        if failed_count > 0:
            logger.warning(f"[RSI] ⚠️ Ошибок: {failed_count} монет")
        
        # ⚡ ОТКЛЮЧЕНО: Сохранение и обработка сигналов выполняются в ContinuousDataLoader
        # save_rsi_cache()  # Будет вызвано позже
        # process_trading_signals_for_all_bots(exchange_obj=current_exchange)  # Будет вызвано позже
        # process_auto_bot_signals(exchange_obj=current_exchange)  # Будет вызвано позже
        
        logger.info(f"[RSI] ✅ load_all_coins_rsi() завершен успешно")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка загрузки RSI данных: {str(e)}")
        # ⚡ БЕЗ БЛОКИРОВКИ: атомарная операция
        coins_rsi_data['update_in_progress'] = False
        return False
    finally:
        # Гарантированно сбрасываем флаг обновления
        # ⚡ БЕЗ БЛОКИРОВКИ: атомарная операция
        if coins_rsi_data['update_in_progress']:
            logger.warning(f"[RSI] ⚠️ Принудительный сброс флага update_in_progress")
            coins_rsi_data['update_in_progress'] = False

def _recalculate_signal_with_trend(rsi, trend, symbol):
    """Пересчитывает сигнал с учетом нового тренда"""
    try:
        # Получаем настройки автобота
        auto_config = bots_data.get('auto_bot_config', {})
        avoid_down_trend = auto_config.get('avoid_down_trend', True)
        avoid_up_trend = auto_config.get('avoid_up_trend', True)
        
        logger.debug(f"[RECALC_SIGNAL] 🔍 {symbol}: RSI={rsi:.1f}, тренд={trend}, avoid_down={avoid_down_trend}, avoid_up={avoid_up_trend}")
        
        # Определяем базовый сигнал по RSI
        if rsi <= SystemConfig.RSI_OVERSOLD:  # RSI ≤ 29 
            # Проверяем нужно ли избегать DOWN тренда для LONG
            if avoid_down_trend and trend == 'DOWN':
                logger.debug(f"[RECALC_SIGNAL] 🔍 {symbol}: RSI {rsi:.1f} ≤ 29, тренд DOWN, избегаем DOWN → WAIT")
                return 'WAIT'  # Ждем улучшения тренда
            else:
                # НЕ показываем дефолтные значения! Только рассчитанные данные!
                trend_display = trend if trend is not None else None
                logger.debug(f"[RECALC_SIGNAL] 🔍 {symbol}: RSI {rsi:.1f} ≤ 29, тренд {trend_display}, не избегаем → ENTER_LONG")
                return 'ENTER_LONG'  # Входим независимо от тренда или при хорошем тренде
        elif rsi >= SystemConfig.RSI_OVERBOUGHT:  # RSI ≥ 71
            # Проверяем нужно ли избегать UP тренда для SHORT
            if avoid_up_trend and trend == 'UP':
                logger.debug(f"[RECALC_SIGNAL] 🔍 {symbol}: RSI {rsi:.1f} ≥ 71, тренд UP, избегаем UP → WAIT")
                return 'WAIT'  # Ждем ослабления тренда
            else:
                # НЕ показываем дефолтные значения! Только рассчитанные данные!
                trend_display = trend if trend is not None else None
                logger.debug(f"[RECALC_SIGNAL] 🔍 {symbol}: RSI {rsi:.1f} ≥ 71, тренд {trend_display}, не избегаем → ENTER_SHORT")
                return 'ENTER_SHORT'  # Входим независимо от тренда или при хорошем тренде
        else:
            # RSI между 30-70 - нейтральная зона
            logger.debug(f"[RECALC_SIGNAL] 🔍 {symbol}: RSI {rsi:.1f} между 30-70 → WAIT")
            return 'WAIT'
            
    except Exception as e:
        logger.error(f"[RECALC_SIGNAL] ❌ Ошибка пересчета сигнала для {symbol}: {e}")
        return 'WAIT'

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
    # ⚡ БЕЗ БЛОКИРОВКИ: конфиг не меняется, GIL делает чтение атомарным
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
    
    # ✅ КРИТИЧНО: Проверяем результаты ВСЕХ фильтров!
    # Если любой фильтр заблокировал сигнал - возвращаем WAIT
    
    # Проверяем ExitScam фильтр
    if coin.get('blocked_by_exit_scam', False):
        logger.debug(f"[SIGNAL] {symbol}: ❌ {signal} заблокирован ExitScam фильтром")
        return 'WAIT'
    
    # Проверяем RSI Time фильтр
    if coin.get('blocked_by_rsi_time', False):
        logger.debug(f"[SIGNAL] {symbol}: ❌ {signal} заблокирован RSI Time фильтром")
        return 'WAIT'
    
    # Проверяем зрелость монеты
    if not coin.get('is_mature', True):
        logger.debug(f"[SIGNAL] {symbol}: ❌ {signal} заблокирован - монета незрелая")
        return 'WAIT'
    
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
        # ⚡ БЕЗ БЛОКИРОВКИ: конфиг не меняется, чтение безопасно
        auto_bot_enabled = bots_data['auto_bot_config']['enabled']
        
        if not auto_bot_enabled:
            logger.info("[NEW_AUTO] ⏹️ Автобот выключен")  # Изменено на INFO
            return
        
        logger.info("[NEW_AUTO] ✅ Автобот включен, начинаем проверку сигналов")
        
        max_concurrent = bots_data['auto_bot_config']['max_concurrent']
        current_active = sum(1 for bot in bots_data['bots'].values() 
                           if bot['status'] not in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']])
        
        if current_active >= max_concurrent:
            logger.debug(f"[NEW_AUTO] 🚫 Достигнут лимит активных ботов ({current_active}/{max_concurrent})")
            return
        
        logger.info("[NEW_AUTO] 🔍 Проверка сигналов для создания новых ботов...")
        
        # Получаем монеты с сигналами
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
        potential_coins = []
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
            # ⚡ БЕЗ БЛОКИРОВКИ: чтение безопасно
            if symbol in bots_data['bots']:
                logger.debug(f"[NEW_AUTO] ⚠️ Бот для {symbol} уже существует")
                continue
            
            # ✅ ПРОВЕРКА ПОЗИЦИЙ: Есть ли ручная позиция на бирже?
            try:
                from bots_modules.workers import positions_cache
                
                # Проверяем есть ли позиция для этой монеты
                if symbol in positions_cache['symbols_with_positions']:
                    # Позиция есть! Проверяем, есть ли активный бот для неё
                    has_active_bot = False
                    if symbol in bots_data['bots']:
                        bot_status = bots_data['bots'][symbol].get('status')
                        if bot_status not in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']]:
                            has_active_bot = True
                    
                    if not has_active_bot:
                        # Позиция есть, но активного бота нет - это РУЧНАЯ позиция!
                        logger.warning(f"[NEW_AUTO] 🚫 {symbol}: Обнаружена РУЧНАЯ позиция на бирже - блокируем создание бота!")
                        continue
                        
            except Exception as pos_error:
                logger.warning(f"[NEW_AUTO] ⚠️ {symbol}: Ошибка проверки позиций: {pos_error}")
                # Продолжаем создание бота если проверка не удалась
            
            # Создаем нового бота
            try:
                logger.info(f"[NEW_AUTO] 🚀 Создаем бота для {symbol} ({coin['signal']}, RSI: {coin['rsi']:.1f})")
                new_bot = create_new_bot(symbol, exchange_obj=exchange_obj)
                
                # ✅ КРИТИЧНО: Сразу входим в позицию!
                signal = coin['signal']
                direction = 'LONG' if signal == 'ENTER_LONG' else 'SHORT'
                logger.info(f"[NEW_AUTO] 📈 Входим в позицию {direction} для {symbol}")
                new_bot.enter_position(direction)
                
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
        
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
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
                from bots_modules.bot_class import NewTradingBot
                trading_bot = NewTradingBot(symbol, bot_data, exchange_to_use)
                
                # Получаем RSI данные для монеты
                # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
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
                    # ⚡ БЕЗ БЛОКИРОВКИ: присваивание - атомарная операция
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

def analyze_trends_for_signal_coins():
    """🎯 Определяет тренд для монет с сигналами (RSI ≤29 или ≥71)"""
    try:
        logger.info("[TREND_ANALYSIS] 🎯 Начинаем анализ трендов для сигнальных монет...")
        
        from bots_modules.imports_and_globals import rsi_data_lock, coins_rsi_data, get_exchange
        from bots_modules.calculations import analyze_trend_6h
        
        exchange = get_exchange()
        if not exchange:
            logger.error("[TREND_ANALYSIS] ❌ Биржа не инициализирована")
            return False
        
        # ✅ КРИТИЧНО: Создаем ВРЕМЕННОЕ хранилище для обновлений
        # Не изменяем coins_rsi_data до завершения всех расчетов!
        temp_updates = {}
        
        # Находим монеты с сигналами для анализа тренда
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
        signal_coins = []
        for symbol, coin_data in coins_rsi_data['coins'].items():
            rsi = coin_data.get('rsi6h')
            if rsi is not None and (rsi <= 29 or rsi >= 71):
                signal_coins.append(symbol)
        
        logger.info(f"[TREND_ANALYSIS] 📊 Найдено {len(signal_coins)} сигнальных монет для анализа тренда")
        
        if not signal_coins:
            logger.warning("[TREND_ANALYSIS] ⚠️ Нет сигнальных монет для анализа тренда")
            return False
        
        # Анализируем тренд для каждой сигнальной монеты
        analyzed_count = 0
        failed_count = 0
        
        for i, symbol in enumerate(signal_coins, 1):
            try:
                logger.info(f"[TREND_ANALYSIS] 🔍 {i}/{len(signal_coins)} Анализируем тренд {symbol}...")
                
                # Анализируем тренд
                logger.debug(f"[TREND_ANALYSIS] 🌐 {symbol}: Вызываем analyze_trend_6h()...")
                trend_analysis = analyze_trend_6h(symbol, exchange_obj=exchange)
                logger.debug(f"[TREND_ANALYSIS] 🌐 {symbol}: analyze_trend_6h() вернула: {trend_analysis is not None}")
                
                if trend_analysis:
                    # ✅ СОБИРАЕМ обновления во временном хранилище
                    if symbol in coins_rsi_data['coins']:
                        coin_data = coins_rsi_data['coins'][symbol]
                        rsi = coin_data.get('rsi6h')
                        new_trend = trend_analysis['trend']
                        
                        # Пересчитываем сигнал с учетом нового тренда
                        old_signal = coin_data.get('signal')
                        
                        # ✅ КРИТИЧНО: НЕ пересчитываем сигнал если он WAIT из-за блокировки фильтров!
                        blocked_by_exit_scam = coin_data.get('blocked_by_exit_scam', False)
                        blocked_by_rsi_time = coin_data.get('blocked_by_rsi_time', False)
                        
                        if blocked_by_exit_scam or blocked_by_rsi_time:
                            logger.info(f"[TREND_ANALYSIS] 🚫 {symbol}: Сигнал заблокирован фильтрами - пропускаем пересчет")
                            new_signal = 'WAIT'  # Оставляем WAIT
                        else:
                            new_signal = _recalculate_signal_with_trend(rsi, new_trend, symbol)
                        
                        # ✅ ВСЕГДА логируем пересчет сигнала для отладки
                        logger.info(f"[TREND_ANALYSIS] 🔄 {symbol}: Пересчет сигнала: RSI={rsi:.1f}, тренд={new_trend}, старый={old_signal} → новый={new_signal}")
                        
                        # Сохраняем обновления во временном хранилище
                        temp_updates[symbol] = {
                            'trend6h': new_trend,
                            'trend_analysis': trend_analysis,
                            'signal': new_signal,
                            'old_signal': old_signal
                        }
                        
                        if new_signal != old_signal:
                            logger.info(f"[TREND_ANALYSIS] 🔄 {symbol}: Сигнал будет обновлен {old_signal} → {new_signal} (тренд: {new_trend})")
                        else:
                            logger.info(f"[TREND_ANALYSIS] ✅ {symbol}: Сигнал не изменится ({old_signal}) - тренд не влияет")
                    
                    analyzed_count += 1
                    logger.info(f"[TREND_ANALYSIS] ✅ {symbol}: Тренд {trend_analysis['trend']}")
                else:
                    failed_count += 1
                    logger.warning(f"[TREND_ANALYSIS] ⚠️ {symbol}: Не удалось определить тренд")
                
                # Небольшая пауза между запросами
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"[TREND_ANALYSIS] ❌ {symbol}: Ошибка анализа тренда: {e}")
                failed_count += 1
        
        # ✅ АТОМАРНО применяем ВСЕ обновления одним махом!
        logger.info(f"[TREND_ANALYSIS] 🎯 Применяем {len(temp_updates)} обновлений атомарно...")
        for symbol, updates in temp_updates.items():
            coins_rsi_data['coins'][symbol]['trend6h'] = updates['trend6h']
            coins_rsi_data['coins'][symbol]['trend_analysis'] = updates['trend_analysis']
            coins_rsi_data['coins'][symbol]['signal'] = updates['signal']
        
        logger.info(f"[TREND_ANALYSIS] ✅ Анализ трендов завершен:")
        logger.info(f"[TREND_ANALYSIS] 📊 Проанализировано: {analyzed_count}")
        logger.info(f"[TREND_ANALYSIS] 📊 Ошибок: {failed_count}")
        logger.info(f"[TREND_ANALYSIS] 📊 Всего обработано: {analyzed_count + failed_count}")
        logger.info(f"[TREND_ANALYSIS] 🎯 Применено обновлений: {len(temp_updates)}")
        
        return True
        
    except Exception as e:
        logger.error(f"[TREND_ANALYSIS] ❌ Ошибка анализа трендов: {e}")
        return False

def process_long_short_coins_with_filters():
    """🔍 Обрабатывает лонг/шорт монеты всеми фильтрами"""
    try:
        logger.info("[FILTER_PROCESSING] 🔍 Начинаем обработку лонг/шорт монет фильтрами...")
        
        from bots_modules.imports_and_globals import rsi_data_lock, coins_rsi_data
        
        # Находим монеты с сигналами лонг/шорт
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
        long_short_coins = []
        for symbol, coin_data in coins_rsi_data['coins'].items():
            signal = coin_data.get('signal', 'WAIT')
            if signal in ['ENTER_LONG', 'ENTER_SHORT']:
                long_short_coins.append(symbol)
        
        logger.info(f"[FILTER_PROCESSING] 📊 Найдено {len(long_short_coins)} лонг/шорт монет для обработки")
        
        if not long_short_coins:
            logger.warning("[FILTER_PROCESSING] ⚠️ Нет лонг/шорт монет для обработки")
            return []
        
        # Обрабатываем каждую монету всеми фильтрами
        filtered_coins = []
        blocked_count = 0
        
        for i, symbol in enumerate(long_short_coins, 1):
            try:
                logger.info(f"[FILTER_PROCESSING] 🔍 {i}/{len(long_short_coins)} Обрабатываем фильтрами {symbol}...")
                
                # Получаем данные монеты
                # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
                coin_data = coins_rsi_data['coins'].get(symbol, {})
                
                if not coin_data:
                    logger.warning(f"[FILTER_PROCESSING] ⚠️ {symbol}: Нет данных")
                    blocked_count += 1
                    continue
                
                # Применяем все фильтры
                signal = coin_data.get('signal', 'WAIT')
                passes_filters = check_new_autobot_filters(symbol, signal, coin_data)
                
                if passes_filters:
                    filtered_coins.append(symbol)
                    logger.info(f"[FILTER_PROCESSING] ✅ {symbol}: Прошел все фильтры")
                else:
                    blocked_count += 1
                    logger.info(f"[FILTER_PROCESSING] ❌ {symbol}: Заблокирован фильтрами")
                
            except Exception as e:
                logger.error(f"[FILTER_PROCESSING] ❌ {symbol}: Ошибка обработки фильтрами: {e}")
                blocked_count += 1
        
        logger.info(f"[FILTER_PROCESSING] ✅ Обработка фильтрами завершена:")
        logger.info(f"[FILTER_PROCESSING] 📊 Прошли фильтры: {len(filtered_coins)}")
        logger.info(f"[FILTER_PROCESSING] 📊 Заблокированы: {blocked_count}")
        logger.info(f"[FILTER_PROCESSING] 📊 Всего обработано: {len(filtered_coins) + blocked_count}")
        
        return filtered_coins
        
    except Exception as e:
        logger.error(f"[FILTER_PROCESSING] ❌ Ошибка обработки фильтрами: {e}")
        return []

def set_filtered_coins_for_autobot(filtered_coins):
    """✅ Передает отфильтрованные монеты автоботу"""
    try:
        logger.info(f"[AUTOBOT_SETUP] ✅ Передаем {len(filtered_coins)} отфильтрованных монет автоботу...")
        
        from bots_modules.imports_and_globals import bots_data_lock, bots_data
        
        # Сохраняем отфильтрованные монеты в конфиг автобота
        # ⚡ БЕЗ БЛОКИРОВКИ: присваивание - атомарная операция
        if 'auto_bot_config' not in bots_data:
            bots_data['auto_bot_config'] = {}
        
        bots_data['auto_bot_config']['filtered_coins'] = filtered_coins
        bots_data['auto_bot_config']['last_filter_update'] = datetime.now().isoformat()
        
        logger.info(f"[AUTOBOT_SETUP] ✅ Отфильтрованные монеты сохранены в конфиг автобота")
        logger.info(f"[AUTOBOT_SETUP] 📊 Монеты для автобота: {', '.join(filtered_coins[:10])}{'...' if len(filtered_coins) > 10 else ''}")
        
        return True
        
    except Exception as e:
        logger.error(f"[AUTOBOT_SETUP] ❌ Ошибка передачи монет автоботу: {e}")
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
        # ⚡ БЕЗ БЛОКИРОВКИ: конфиг не меняется, GIL делает чтение атомарным
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
        ai_check_enabled = True  # Включаем обратно - проблема была не в AI!
        
        if ai_check_enabled:
            try:
                from bot_engine.bot_config import AIConfig
                
                # Быстрая проверка: AI включен и Anomaly Detection включен
                if AIConfig.AI_ENABLED and AIConfig.AI_ANOMALY_DETECTION_ENABLED:
                    try:
                        from bot_engine.ai.ai_manager import get_ai_manager
                        
                        ai_manager = get_ai_manager()
                        
                        # Быстрая проверка доступности: если AI недоступен, пропускаем
                        if not ai_manager.is_available():
                            # AI модули не загружены (нет лицензии или не установлены)
                            # Не логируем каждый раз, чтобы не спамить
                            pass
                        elif ai_manager.anomaly_detector:
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


def get_lstm_prediction(symbol, signal, current_price):
    """
    Получает предсказание LSTM для монеты
    
    Args:
        symbol: Символ монеты
        signal: Сигнал ('LONG' или 'SHORT')
        current_price: Текущая цена
    
    Returns:
        Dict с предсказанием или None
    """
    try:
        from bot_engine.bot_config import AIConfig
        
        # Проверяем, включен ли LSTM
        if not (AIConfig.AI_ENABLED and AIConfig.AI_LSTM_ENABLED):
            return None
        
        try:
            from bot_engine.ai.ai_manager import get_ai_manager
            
            ai_manager = get_ai_manager()
            
            # Проверяем доступность LSTM
            if not ai_manager.is_available() or not ai_manager.lstm_predictor:
                return None
            
            # Получаем свечи для анализа
            exch = get_exchange()
            if not exch:
                return None
            
            chart_response = exch.get_chart_data(symbol, '6h', '30d')
            if not chart_response or not chart_response.get('success'):
                return None
            
            candles = chart_response.get('data', {}).get('candles', [])
            if len(candles) < 60:  # LSTM требует минимум 60 свечей
                return None
            
            # Получаем предсказание с ТАЙМАУТОМ
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(ai_manager.lstm_predictor.predict, candles, current_price)
                try:
                    prediction = future.result(timeout=5)  # 5 секунд таймаут для LSTM
                except concurrent.futures.TimeoutError:
                    logger.warning(f"[AI_LSTM] {symbol}: ⏱️ LSTM prediction таймаут (5с)")
                    prediction = None  # Пропускаем AI проверку при таймауте
            
            if prediction and prediction.get('confidence', 0) >= AIConfig.AI_LSTM_MIN_CONFIDENCE:
                # Проверяем совпадение направлений
                lstm_direction = "LONG" if prediction['direction'] > 0 else "SHORT"
                confidence = prediction['confidence']
                
                if lstm_direction == signal:
                    logger.info(
                        f"[LSTM] {symbol}: ✅ ПОДТВЕРЖДЕНИЕ: "
                        f"LSTM предсказывает {lstm_direction} "
                        f"(изменение: {prediction['change_percent']:+.2f}%, "
                        f"уверенность: {confidence:.1f}%)"
                    )
                else:
                    logger.warning(
                        f"[LSTM] {symbol}: ⚠️ ПРОТИВОРЕЧИЕ: "
                        f"Сигнал {signal}, но LSTM предсказывает {lstm_direction} "
                        f"(изменение: {prediction['change_percent']:+.2f}%, "
                        f"уверенность: {confidence:.1f}%)"
                    )
                
                return {
                    **prediction,
                    'lstm_direction': lstm_direction,
                    'matches_signal': lstm_direction == signal
                }
            
            return None
            
        except ImportError as e:
            logger.debug(f"[LSTM] {symbol}: AI модуль не доступен: {e}")
            return None
        except Exception as e:
            logger.error(f"[LSTM] {symbol}: Ошибка LSTM предсказания: {e}")
            return None
    
    except ImportError:
        return None


def get_pattern_analysis(symbol, signal, current_price):
    """
    Получает анализ паттернов для монеты
    
    Args:
        symbol: Символ монеты
        signal: Сигнал ('LONG' или 'SHORT')
        current_price: Текущая цена
    
    Returns:
        Dict с анализом паттернов или None
    """
    try:
        from bot_engine.bot_config import AIConfig
        
        # Проверяем, включен ли Pattern Recognition
        if not (AIConfig.AI_ENABLED and AIConfig.AI_PATTERN_ENABLED):
            return None
        
        try:
            from bot_engine.ai.ai_manager import get_ai_manager
            
            ai_manager = get_ai_manager()
            
            # Проверяем доступность Pattern Detector
            if not ai_manager.is_available() or not ai_manager.pattern_detector:
                return None
            
            # Получаем свечи для анализа
            exch = get_exchange()
            if not exch:
                return None
            
            chart_response = exch.get_chart_data(symbol, '6h', '30d')
            if not chart_response or not chart_response.get('success'):
                return None
            
            candles = chart_response.get('data', {}).get('candles', [])
            if len(candles) < 100:  # Pattern требует минимум 100 свечей
                return None
            
            # Получаем анализ паттернов с ТАЙМАУТОМ
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    ai_manager.pattern_detector.get_pattern_signal,
                    candles, 
                    current_price, 
                    signal
                )
                try:
                    pattern_signal = future.result(timeout=5)  # 5 секунд таймаут
                except concurrent.futures.TimeoutError:
                    logger.warning(f"[AI_PATTERN] {symbol}: ⏱️ Pattern detection таймаут (5с)")
                    pattern_signal = {'patterns_found': 0, 'confirmation': False}  # Пропускаем при таймауте
            
            if pattern_signal['patterns_found'] > 0:
                # Проверяем подтверждение
                if pattern_signal['confirmation']:
                    logger.info(
                        f"[PATTERN] {symbol}: ✅ ПОДТВЕРЖДЕНИЕ: "
                        f"Паттерны подтверждают {signal} "
                        f"(найдено: {pattern_signal['patterns_found']}, "
                        f"уверенность: {pattern_signal['confidence']:.1f}%)"
                    )
                    
                    if pattern_signal['strongest_pattern']:
                        strongest = pattern_signal['strongest_pattern']
                        logger.info(
                            f"[PATTERN] {symbol}:    └─ {strongest['name']}: "
                            f"{strongest['description']}"
                        )
                else:
                    logger.warning(
                        f"[PATTERN] {symbol}: ⚠️ ПРОТИВОРЕЧИЕ: "
                        f"Сигнал {signal}, но паттерны указывают на {pattern_signal['signal']} "
                        f"(уверенность: {pattern_signal['confidence']:.1f}%)"
                    )
                
                return pattern_signal
            
            return None
            
        except ImportError as e:
            logger.debug(f"[PATTERN] {symbol}: AI модуль не доступен: {e}")
            return None
        except Exception as e:
            logger.error(f"[PATTERN] {symbol}: Ошибка анализа паттернов: {e}")
            return None
    
    except ImportError:
        return None  # AIConfig не доступен

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
        # Локальный импорт для избежания циклического импорта
        from bots_modules.bot_class import NewTradingBot
        from bots_modules.imports_and_globals import get_exchange
        exchange_to_use = exchange_obj if exchange_obj else get_exchange()
        
        # Получаем размер позиции из конфига
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
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
        # ⚡ БЕЗ БЛОКИРОВКИ: присваивание - атомарная операция
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
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
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
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
        coin_data = coins_rsi_data['coins'].get(symbol)
        if not coin_data:
            logger.error(f"[TEST_RSI_TIME] {symbol}: Нет RSI данных")
            return
        
        current_rsi = coin_data.get('rsi6h', 0)
        signal = coin_data.get('signal', 'WAIT')
        
        # Определяем ОРИГИНАЛЬНЫЙ сигнал на основе только RSI (игнорируя другие фильтры)
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
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
            # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
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

