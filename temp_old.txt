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

from bots_modules.imports_and_globals import shutdown_flag, should_log_message

try:
    from bot_engine.filters import (
        check_rsi_time_filter as engine_check_rsi_time_filter,
        check_exit_scam_filter as engine_check_exit_scam_filter,
    )
except ImportError:
    engine_check_rsi_time_filter = None
    engine_check_exit_scam_filter = None

logger = logging.getLogger('BotsService')

# Кэш для предотвращения спама логов защиты от повторных входов
_loss_reentry_log_cache = {}
_loss_reentry_log_lock = threading.Lock()
_loss_reentry_log_interval = 60  # Логируем не чаще раза в минуту для каждой монеты

# ✅ КЭШИРОВАНИЕ AI MANAGER для избежания повторных инициализаций
_ai_manager_cache = None
_ai_available_cache = None
_ai_cache_lock = threading.Lock()
_delisted_cache = {'ts': 0.0, 'coins': {}}

def get_cached_ai_manager():
    """
    Получает закэшированный экземпляр AI Manager.
    Инициализируется только один раз для избежания повторных загрузок моделей.
    """
    global _ai_manager_cache, _ai_available_cache
    
    with _ai_cache_lock:
        # Если уже есть в кэше - возвращаем
        if _ai_manager_cache is not None:
            return _ai_manager_cache, _ai_available_cache
        
        # Инициализируем только один раз
        try:
            from bot_engine.bot_config import AIConfig
            if AIConfig.AI_ENABLED:
                from bot_engine.ai.ai_manager import get_ai_manager
                _ai_manager_cache = get_ai_manager()
                _ai_available_cache = _ai_manager_cache.is_available() if _ai_manager_cache else False
            else:
                _ai_manager_cache = None
                _ai_available_cache = False
        except Exception as e:
            logger.debug(f" AI Manager недоступен: {e}")
            _ai_manager_cache = None
            _ai_available_cache = False
        
        return _ai_manager_cache, _ai_available_cache


def _get_cached_delisted_coins():
    """Возвращает кэш делистинговых монет (обновляется раз в 60 секунд)."""
    global _delisted_cache
    now_ts = time.time()
    if now_ts - _delisted_cache['ts'] >= 60:
        try:
            delisted_data = load_delisted_coins()
            coins = delisted_data.get('delisted_coins', {}) or {}
            _delisted_cache = {'ts': now_ts, 'coins': coins}
        except Exception as exc:  # pragma: no cover
            logger.warning(f"⚠️ Не удалось обновить кэш делистинга: {exc}")
            # не обновляем ts, чтобы повторить попытку при следующем запросе
    return _delisted_cache['coins']

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

def calculate_ema_list(prices, period):
    """
    Рассчитывает список значений EMA для массива цен.
    Возвращает список значений EMA или None, если недостаточно данных.
    """
    if len(prices) < period:
        return None
    
    ema_values = []
    # Первое значение EMA = SMA
    sma = sum(prices[:period]) / period
    ema = sma
    multiplier = 2 / (period + 1)
    
    # Добавляем None для первых period-1 значений (где EMA еще не определен)
    ema_values.extend([None] * (period - 1))
    ema_values.append(ema)
    
    # Рассчитываем EMA для остальных значений
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
        ema_values.append(ema)
    
    return ema_values

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
    from bots_modules.sync_and_cache import load_delisted_coins, ensure_exchange_initialized
except ImportError as e:
    print(f"Warning: Could not import sync_and_cache helpers in filters: {e}")
    def load_delisted_coins():
        return {"delisted_coins": {}}
    def ensure_exchange_initialized():
        return False

# ❌ ОТКЛЮЧЕНО: optimal_ema перемещен в backup (используется заглушка из imports_and_globals)
# Импорт функции optimal_ema из модуля
# try:
#     from bots_modules.optimal_ema import get_optimal_ema_periods
# except ImportError as e:
#     print(f"Warning: Could not import optimal_ema functions in filters: {e}")
#     def get_optimal_ema_periods(symbol):
#         return {'ema_short': 50, 'ema_long': 200, 'accuracy': 0}

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
        BOT_STATUS, system_initialized, get_exchange,
        get_individual_coin_settings
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
    def get_individual_coin_settings(symbol):
        return None
    # Fallback для SystemConfig
    class SystemConfig:
        RSI_OVERSOLD = 29
        RSI_OVERBOUGHT = 71
        # ✅ Новые параметры с учетом тренда
        RSI_EXIT_LONG_WITH_TREND = 65
        RSI_EXIT_LONG_AGAINST_TREND = 60
        RSI_EXIT_SHORT_WITH_TREND = 35
        RSI_EXIT_SHORT_AGAINST_TREND = 40

def _legacy_check_rsi_time_filter(candles, rsi, signal, symbol=None, individual_settings=None):
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
        symbol: Символ монеты (опционально, для получения индивидуальных настроек)
        individual_settings: Индивидуальные настройки монеты (опционально)
    
    Returns:
        dict: {'allowed': bool, 'reason': str, 'last_extreme_candles_ago': int, 'calm_candles': int}
    """
    try:
        # ✅ Получаем настройки: сначала индивидуальные, затем глобальные
        # ⚡ БЕЗ БЛОКИРОВКИ: конфиг не меняется, GIL делает чтение атомарным
        if individual_settings is None and symbol:
            individual_settings = get_individual_coin_settings(symbol)
        
        auto_config = bots_data.get('auto_bot_config', {})
        
        # Используем индивидуальные настройки, если они есть, иначе глобальные
        rsi_time_filter_enabled = individual_settings.get('rsi_time_filter_enabled') if individual_settings else None
        if rsi_time_filter_enabled is None:
            rsi_time_filter_enabled = auto_config.get('rsi_time_filter_enabled', True)
        
        rsi_time_filter_candles = individual_settings.get('rsi_time_filter_candles') if individual_settings else None
        if rsi_time_filter_candles is None:
            rsi_time_filter_candles = auto_config.get('rsi_time_filter_candles', 8)
        rsi_time_filter_candles = max(2, rsi_time_filter_candles)  # Минимум 2 свечи (защита от некорректных значений)
        
        rsi_time_filter_upper = individual_settings.get('rsi_time_filter_upper') if individual_settings else None
        if rsi_time_filter_upper is None:
            rsi_time_filter_upper = auto_config.get('rsi_time_filter_upper', 65)  # Спокойная зона для SHORT
        
        rsi_time_filter_lower = individual_settings.get('rsi_time_filter_lower') if individual_settings else None
        if rsi_time_filter_lower is None:
            rsi_time_filter_lower = auto_config.get('rsi_time_filter_lower', 35)  # Спокойная зона для LONG
        
        rsi_long_threshold = individual_settings.get('rsi_long_threshold') if individual_settings else None
        if rsi_long_threshold is None:
            rsi_long_threshold = auto_config.get('rsi_long_threshold', 29)  # Экстремум для LONG
        
        rsi_short_threshold = individual_settings.get('rsi_short_threshold') if individual_settings else None
        if rsi_short_threshold is None:
            rsi_short_threshold = auto_config.get('rsi_short_threshold', 71)  # Экстремум для SHORT
        
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
        
        # Обновляем последний элемент истории переданным RSI, если он указан
        # Это важно для согласованности данных, так как переданный RSI может быть более актуальным
        if rsi is not None:
            rsi_history[current_index] = rsi
        
        if signal == 'ENTER_SHORT':
            # ЛОГИКА ДЛЯ SHORT (аналогично LONG, только наоборот):
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
        logger.error(f" Ошибка проверки временного фильтра: {e}")
        return {'allowed': False, 'reason': f'Ошибка анализа: {str(e)}', 'last_extreme_candles_ago': None, 'calm_candles': 0}

def get_coin_candles_only(symbol, exchange_obj=None):
    """⚡ БЫСТРАЯ загрузка ТОЛЬКО свечей БЕЗ расчетов"""
    try:
        if shutdown_flag.is_set():
            return None

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
            'timeframe': '6h',  # ✅ Добавляем timeframe для правильного накопления
            'last_update': datetime.now().isoformat()
        }
        
    except Exception as e:
        return None


def check_rsi_time_filter(candles, rsi, signal, symbol=None, individual_settings=None):
    """
    Обёртка над bot_engine.filters.check_rsi_time_filter с fallback на легаси-логику.
    
    Args:
        candles: Список свечей
        rsi: Текущее значение RSI
        signal: Торговый сигнал ('ENTER_LONG' или 'ENTER_SHORT')
        symbol: Символ монеты (опционально, для получения индивидуальных настроек)
        individual_settings: Индивидуальные настройки монеты (опционально)
    """
    try:
        if engine_check_rsi_time_filter is None:
            raise RuntimeError('engine filters unavailable')
        
        # ✅ Получаем конфиг с учетом индивидуальных настроек
        auto_config = bots_data.get('auto_bot_config', {}).copy()
        
        # Если переданы индивидуальные настройки - используем их
        if individual_settings is None and symbol:
            individual_settings = get_individual_coin_settings(symbol)
        
        if individual_settings:
            # Объединяем глобальные настройки с индивидуальными (индивидуальные имеют приоритет)
            for key in ['rsi_time_filter_enabled', 'rsi_time_filter_candles', 
                       'rsi_time_filter_lower', 'rsi_time_filter_upper',
                       'rsi_long_threshold', 'rsi_short_threshold']:
                if key in individual_settings:
                    auto_config[key] = individual_settings[key]
        
        result = engine_check_rsi_time_filter(
            candles,
            rsi,
            signal,
            auto_config,
            calculate_rsi_history_func=calculate_rsi_history,
        )
        return {
            'allowed': bool(result.get('allowed')),
            'reason': result.get('reason'),
            'last_extreme_candles_ago': result.get('last_extreme_candles_ago'),
            'calm_candles': result.get('calm_candles'),
        }
    except Exception as exc:
        logger.error(f" Ошибка проверки временного фильтра: {exc}")
        return _legacy_check_rsi_time_filter(candles, rsi, signal, symbol=symbol, individual_settings=individual_settings)


def _run_exit_scam_ai_detection(symbol, candles):
    """AI-анализ свечей на аномалии (reuse из легаси-логики)."""
    try:
        from bot_engine.bot_config import AIConfig
    except ImportError:
        return True

    if not (AIConfig.AI_ENABLED and AIConfig.AI_ANOMALY_DETECTION_ENABLED):
        return True

    try:
        ai_manager, ai_available = get_cached_ai_manager()
        if not ai_available or not ai_manager or not ai_manager.anomaly_detector:
            return True

        anomaly_result = ai_manager.anomaly_detector.detect(candles)
        if anomaly_result.get('is_anomaly'):
            severity = anomaly_result.get('severity', 0)
            anomaly_type = anomaly_result.get('anomaly_type', 'UNKNOWN')
            if severity > AIConfig.AI_ANOMALY_BLOCK_THRESHOLD:
                return False
            logger.warning(
                f"{symbol}: ⚠️ ПРЕДУПРЕЖДЕНИЕ (AI): "
                f"Аномалия {anomaly_type} "
                f"(severity: {severity:.2%} - ниже порога {AIConfig.AI_ANOMALY_BLOCK_THRESHOLD:.2%})"
            )
    except ImportError as exc:
        logger.debug(f"{symbol}: AI модуль не доступен: {exc}")
    except Exception as exc:
        logger.error(f"{symbol}: Ошибка AI проверки: {exc}")
    return True


def _check_loss_reentry_protection_static(symbol, candles, loss_reentry_count, loss_reentry_candles, individual_settings=None):
    """
    Статическая функция проверки защиты от повторных входов после убыточных закрытий
    
    Args:
        symbol: Символ монеты
        candles: Список свечей для подсчета прошедших свечей
        loss_reentry_count: Количество убыточных сделок для проверки (N)
        loss_reentry_candles: Количество свечей для ожидания (X)
        individual_settings: Индивидуальные настройки монеты (опционально)
    
    Returns:
        dict: {'allowed': bool, 'reason': str, 'candles_passed': int}
    """
    try:
        # ✅ УБРАНО: Проверка на открытую позицию должна быть только в should_open_long/short
        # Статическая функция всегда проверяет фильтр, проверка позиции делается на уровне бота
        
        # Получаем последние N закрытых сделок для этого символа
        from bot_engine.bots_database import get_bots_database
        bots_db = get_bots_database()
        
        # Получаем последние N закрытых сделок по символу, отсортированные по времени закрытия (новые первыми)
        closed_trades = bots_db.get_bot_trades_history(
            bot_id=None,
            symbol=symbol,
            status='CLOSED',
            decision_source=None,
            limit=loss_reentry_count,
            offset=0
        )
        
        # Если нет закрытых сделок - разрешаем вход
        if not closed_trades or len(closed_trades) < loss_reentry_count:
            return {
                'allowed': True,
                'reason': f'Недостаточно закрытых сделок ({len(closed_trades) if closed_trades else 0} < {loss_reentry_count})',
                'candles_passed': None
            }
        
        # ✅ ИСПРАВЛЕНО: Проверяем, все ли последние N сделок были в минус
        # Важно: проверяем именно ПОСЛЕДНИЕ N сделок по времени закрытия (они уже отсортированы DESC)
        all_losses = True
        for trade in closed_trades:
            pnl = trade.get('pnl', 0)
            # ✅ КРИТИЧНО: Проверяем что PnL определен и действительно отрицательный (строго < 0)
            try:
                pnl_float = float(pnl) if pnl is not None else 0.0
                # Если хотя бы одна сделка >= 0 (прибыльная или безубыточная) - не все в минус
                if pnl_float >= 0:
                    all_losses = False
                    break
            except (ValueError, TypeError):
                # Если не удалось преобразовать PnL - считаем что не убыточная
                all_losses = False
                break
        
        # ✅ КРИТИЧНО: Если НЕ ВСЕ последние N сделок в минус - РАЗРЕШАЕМ вход (фильтр НЕ работает)
        if not all_losses:
            return {
                'allowed': True,
                'reason': f'Не все последние {loss_reentry_count} сделок в минус - есть прибыльные сделки, фильтр не применяется',
                'candles_passed': None
            }
        
        # Все последние N сделок в минус - проверяем количество прошедших свечей
        last_trade = closed_trades[0]  # Самая последняя закрытая сделка
        
        # Получаем timestamp закрытия последней сделки
        exit_timestamp = last_trade.get('exit_timestamp')
        if not exit_timestamp:
            exit_time_str = last_trade.get('exit_time')
            if exit_time_str:
                try:
                    from datetime import datetime
                    if isinstance(exit_time_str, str):
                        exit_dt = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
                        exit_timestamp = int(exit_dt.timestamp())
                    else:
                        exit_timestamp = int(exit_time_str)
                except:
                    return {'allowed': True, 'reason': 'Не удалось получить время закрытия', 'candles_passed': None}
            else:
                return {'allowed': True, 'reason': 'Нет данных о времени закрытия', 'candles_passed': None}
        
        # Если exit_timestamp в миллисекундах, конвертируем в секунды
        if exit_timestamp > 1e12:
            exit_timestamp = exit_timestamp / 1000
        
        # Подсчитываем количество свечей, прошедших с момента закрытия
        CANDLE_INTERVAL_SECONDS = 6 * 3600  # 6 часов
        
        if not candles or len(candles) == 0:
            return {'allowed': True, 'reason': 'Нет свечей для проверки', 'candles_passed': None}
        
        # Получаем timestamp последней свечи
        last_candle = candles[-1]
        last_candle_timestamp = last_candle.get('timestamp', 0)
        if last_candle_timestamp > 1e12:
            last_candle_timestamp = last_candle_timestamp / 1000
        
        # ✅ ИСПРАВЛЕНО: Подсчитываем количество свечей с момента закрытия
        # Свечи уже отсортированы по времени (старые -> новые)
        candles_passed = 0
        
        # Ищем первую свечу, которая ПОЛНОСТЬЮ позже времени закрытия
        # Свеча считается прошедшей, если её начало >= времени закрытия
        for i, candle in enumerate(candles):
            candle_timestamp = candle.get('timestamp', 0)
            if candle_timestamp > 1e12:
                candle_timestamp = candle_timestamp / 1000
            
            # Если начало свечи >= времени закрытия, считаем эту и все последующие свечи
            if candle_timestamp >= exit_timestamp:
                candles_passed = len(candles) - i
                break
        
        # ✅ ИСПРАВЛЕНО: Если не нашли свечей через перебор, считаем по времени
        # Это более надежный метод для 6h свечей
        if candles_passed == 0:
            time_diff_seconds = last_candle_timestamp - exit_timestamp
            if time_diff_seconds > 0:
                # Считаем количество полных 6-часовых интервалов
                candles_passed = max(1, int(time_diff_seconds / CANDLE_INTERVAL_SECONDS))
        
        # ✅ ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: Если последняя свеча явно после закрытия
        if candles_passed == 0 and last_candle_timestamp > exit_timestamp:
            # Минимум 1 свеча прошла, если текущая свеча после закрытия
            candles_passed = 1
        
        # ✅ ИСПРАВЛЕНО: Конвертируем loss_reentry_candles в int для корректного сравнения
        try:
            loss_reentry_candles_int = int(loss_reentry_candles) if loss_reentry_candles is not None else 3
        except (ValueError, TypeError):
            loss_reentry_candles_int = 3
        
        # Проверяем, прошло ли достаточно свечей
        if candles_passed < loss_reentry_candles_int:
            return {
                'allowed': False,
                'reason': f'Последние {loss_reentry_count} сделок в минус, прошло только {candles_passed} свечей (требуется {loss_reentry_candles_int})',
                'candles_passed': candles_passed
            }
        
        return {
            'allowed': True,
            'reason': f'Прошло {candles_passed} свечей с последнего убытка (требуется {loss_reentry_candles})',
            'candles_passed': candles_passed
        }
        
    except Exception as e:
        # При ошибке разрешаем вход (безопаснее, как в bot_class.py)
        logger.error(f"{symbol}: ❌ Ошибка проверки защиты от повторных входов (static): {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {
            'allowed': True,
            'reason': f'Ошибка проверки: {str(e)}',
            'candles_passed': None
        }
        
    except Exception as e:
        logger.debug(f"{symbol}: Ошибка проверки защиты от повторных входов: {e}")
        return {'allowed': True, 'reason': f'Ошибка проверки: {str(e)}', 'candles_passed': None}


def check_exit_scam_filter(symbol, coin_data):
    """Унифицированный exit-scam фильтр с AI-анализом и fallback."""
    try:
        if engine_check_exit_scam_filter is None:
            raise RuntimeError('engine filters unavailable')
        
        # ✅ Получаем конфиг с учетом индивидуальных настроек
        auto_config = bots_data.get('auto_bot_config', {}).copy()
        individual_settings = get_individual_coin_settings(symbol)
        
        if individual_settings:
            # Объединяем глобальные настройки с индивидуальными (индивидуальные имеют приоритет)
            for key in ['exit_scam_enabled', 'exit_scam_candles', 
                       'exit_scam_single_candle_percent', 'exit_scam_multi_candle_count',
                       'exit_scam_multi_candle_percent']:
                if key in individual_settings:
                    auto_config[key] = individual_settings[key]
        
        exchange_obj = get_exchange()
        if not exchange_obj:
            return False

        base_allowed = engine_check_exit_scam_filter(
            symbol,
            coin_data,
            auto_config,
            exchange_obj,
            ensure_exchange_initialized,
        )

        if not base_allowed:
            return False

        chart_response = exchange_obj.get_chart_data(symbol, '6h', '30d')
        candles = chart_response.get('data', {}).get('candles', []) if chart_response and chart_response.get('success') else []
        if candles:
            return _run_exit_scam_ai_detection(symbol, candles)
        return True
    except Exception as exc:
        logger.error(f"{symbol}: Ошибка проверки exit-scam (core): {exc}")
        return _legacy_check_exit_scam_filter(symbol, coin_data, individual_settings=individual_settings)

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

    if shutdown_flag.is_set():
        logger.debug(f"{symbol}: Пропуск анализа RSI (shutdown requested)")
        return None
    
    # ⚡ СЕМАФОР: Ограничиваем одновременные API запросы к бирже (если нет в кэше)
    # Это предотвращает перегрузку API биржи
    global _exchange_api_semaphore
    try:
        _exchange_api_semaphore
    except NameError:
        _exchange_api_semaphore = threading.Semaphore(5)  # ⚡ Уменьшили до 5 для стабильности
    
    import time
    thread_start = time.time()
    data_source = 'cache'
    # print(f"[{time.strftime('%H:%M:%S')}] >>> НАЧАЛО get_coin_rsi_data({symbol})", flush=True)  # Отключено для скорости
    
    try:
        # ✅ ФИЛЬТР 0: ДЕЛИСТИНГОВЫЕ МОНЕТЫ - САМЫЙ ПЕРВЫЙ!
        # Исключаем делистинговые монеты ДО всех остальных проверок
        # Загружаем делистинговые монеты из файла
        delisted_coins = _get_cached_delisted_coins()
        
        if symbol in delisted_coins:
            delisting_info = delisted_coins.get(symbol, {})
            logger.info(f"{symbol}: Исключаем из всех проверок - {delisting_info.get('reason', 'Delisting detected')}")
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
                logger.debug(f"{symbol}: ❌ Режим WHITELIST - монета не в белом списке")
        
        elif scope == 'blacklist':
            # Режим ТОЛЬКО blacklist - работаем со ВСЕМИ монетами КРОМЕ черного списка
            if symbol in blacklist:
                is_blocked_by_scope = True
                logger.debug(f"{symbol}: ❌ Режим BLACKLIST - монета в черном списке")
        
        elif scope == 'all':
            # Режим ALL - работаем со ВСЕМИ монетами, но проверяем оба списка
            if symbol in blacklist:
                is_blocked_by_scope = True
                logger.debug(f"{symbol}: ❌ Монета в черном списке")
            # Если в whitelist - даем приоритет (логируем, но не блокируем)
            if whitelist and symbol in whitelist:
                logger.debug(f"{symbol}: ⭐ В белом списке (приоритет)")
        
        # БЕЗ задержки - семафор и ThreadPool уже контролируют rate limit
        
        # logger.debug(f"[DEBUG] Обработка {symbol}...")  # Отключено для ускорения
        
        # Используем переданную биржу или глобальную
        # print(f"[{time.strftime('%H:%M:%S')}] >>> {symbol}: Получение exchange...", flush=True)  # Отключено
        from bots_modules.imports_and_globals import get_exchange
        exchange_to_use = exchange_obj if exchange_obj is not None else get_exchange()
        
        # Проверяем, что биржа доступна
        if exchange_to_use is None:
            logger.error(f"Ошибка получения данных для {symbol}: 'NoneType' object has no attribute 'get_chart_data'")
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
            # Проверяем, есть ли кэш в памяти вообще (может быть еще не загружен при старте)
            cache_loaded = bool(coins_rsi_data.get('candles_cache', {}))
            if not cache_loaded:
                logger.debug(f"ℹ️ {symbol}: Кэш свечей еще не загружен, загружаем с биржи...")
            else:
                logger.info(f"ℹ️ {symbol}: Нет в кэше свечей, загружаем с биржи...")
            # ⚡ СЕМАФОР: Ограничиваем одновременные запросы к API биржи
            with _exchange_api_semaphore:
                import time as time_module
                api_start = time_module.time()
                logger.info(f"🌐 {symbol}: Начало запроса get_chart_data()...")
                
                chart_response = exchange_to_use.get_chart_data(symbol, '6h', '30d')
                
                api_duration = time_module.time() - api_start
                logger.info(f"🌐 {symbol}: get_chart_data() завершен за {api_duration:.1f}с")
                
                if not chart_response or not chart_response.get('success'):
                    logger.warning(f"❌ {symbol}: Ошибка: {chart_response.get('error', 'Неизвестная ошибка') if chart_response else 'Нет ответа'}")
                    return None
                
                candles = chart_response['data']['candles']
                logger.info(f"✅ {symbol}: Свечи загружены с биржи ({len(candles)} свечей)")
                data_source = 'api'
                
                # ✅ КРИТИЧНО: Сохраняем свечи в кэш после загрузки с биржи!
                # Это предотвращает повторные запросы к бирже для тех же монет
                try:
                    if candles and len(candles) >= 15:
                        # Сохраняем в том же формате, что и get_coin_candles_only
                        candles_cache[symbol] = {
                            'symbol': symbol,
                            'candles': candles,
                            'timeframe': '6h',
                            'last_update': datetime.now().isoformat()
                        }
                        # Обновляем глобальный кэш
                        coins_rsi_data['candles_cache'] = candles_cache
                        logger.debug(f"💾 {symbol}: Свечи сохранены в кэш ({len(candles)} свечей)")
                except Exception as cache_save_error:
                    logger.warning(f"⚠️ {symbol}: Ошибка сохранения свечей в кэш: {cache_save_error}")
        
        if not candles or len(candles) < 15:  # Базовая проверка для RSI(14)
            return None
        
        # Рассчитываем RSI для 6H
        # Bybit отправляет свечи в правильном порядке для RSI (от старой к новой)
        closes = [candle['close'] for candle in candles]
        
        rsi = calculate_rsi(closes, 14)
        
        if rsi is None:
            logger.warning(f"Не удалось рассчитать RSI для {symbol}")
            return None
        
        # ✅ РАСЧИТЫВАЕМ ТРЕНД СРАЗУ для всех монет - избегаем "гуляния" данных
        # НЕ УСТАНАВЛИВАЕМ ДЕФОЛТНЫХ ЗНАЧЕНИЙ! Только рассчитанные данные!
        trend = None  # Изначально None
        trend_analysis = None
        try:
            from bots_modules.calculations import analyze_trend_6h
            trend_analysis = analyze_trend_6h(symbol, exchange_obj=exchange_obj, candles_data=candles)
            if trend_analysis:
                trend = trend_analysis['trend']  # ТОЛЬКО рассчитанное значение!
            # НЕ устанавливаем дефолт если анализ не удался - оставляем None
        except Exception as e:
            logger.debug(f"{symbol}: Ошибка анализа тренда: {e}")
            # НЕ устанавливаем дефолт при ошибке - оставляем None
        
        # Рассчитываем изменение за 24h (примерно 4 свечи 6H)
        change_24h = 0
        if len(closes) >= 5:
            change_24h = round(((closes[-1] - closes[-5]) / closes[-5]) * 100, 2)
        
        # ✅ КРИТИЧНО: Получаем оптимальные EMA периоды ДО определения сигнала!
        # ❌ ОТКЛЮЧЕНО: EMA фильтр удален из системы
        # ema_periods = None
        # try:
        #     ema_periods = get_optimal_ema_periods(symbol)
        # except Exception as e:
        #     logger.debug(f"[EMA] Ошибка получения оптимальных EMA для {symbol}: {e}")
        #     ema_periods = {'ema_short': 50, 'ema_long': 200, 'accuracy': 0, 'analysis_method': 'default'}
        
        # ✅ КРИТИЧНО: Получаем индивидуальные настройки монеты ДО определения сигнала!
        # Это позволяет использовать индивидуальные пороги RSI для определения сигнала
        individual_settings = get_individual_coin_settings(symbol)
        
        # Определяем пороги RSI: сначала индивидуальные, затем глобальные
        rsi_long_threshold = individual_settings.get('rsi_long_threshold') if individual_settings else None
        if rsi_long_threshold is None:
            rsi_long_threshold = bots_data.get('auto_bot_config', {}).get('rsi_long_threshold', SystemConfig.RSI_OVERSOLD)
        
        rsi_short_threshold = individual_settings.get('rsi_short_threshold') if individual_settings else None
        if rsi_short_threshold is None:
            rsi_short_threshold = bots_data.get('auto_bot_config', {}).get('rsi_short_threshold', SystemConfig.RSI_OVERBOUGHT)
        
        # Определяем RSI зоны согласно техзаданию
        rsi_zone = 'NEUTRAL'
        signal = 'WAIT'
        
        # ✅ ФИЛЬТР 2: Базовый сигнал НА ОСНОВЕ OPTIMAL EMA ПЕРИОДОВ!
        # ✅ Получаем настройки фильтров по тренду: сначала индивидуальные, затем глобальные
        # ⚡ БЕЗ БЛОКИРОВКИ: конфиг не меняется во время выполнения, безопасно читать
        avoid_down_trend = individual_settings.get('avoid_down_trend') if individual_settings else None
        if avoid_down_trend is None:
            avoid_down_trend = bots_data.get('auto_bot_config', {}).get('avoid_down_trend', False)
        
        avoid_up_trend = individual_settings.get('avoid_up_trend') if individual_settings else None
        if avoid_up_trend is None:
            avoid_up_trend = bots_data.get('auto_bot_config', {}).get('avoid_up_trend', False)
        
        # ✅ КРИТИЧНО: Определяем сигнал на основе Optimal EMA периодов!
        # ✅ УПРОЩЕННАЯ ЛОГИКА: Убрали фильтр по EMA - используем только RSI
        # EMA слишком запаздывает и блокирует хорошие входы по RSI
        if True:  # Оставляем структуру для возможного возврата EMA в будущем
            try:
                # ✅ ИСПОЛЬЗУЕМ ИНДИВИДУАЛЬНЫЕ ПОРОГИ RSI для определения сигнала!
                # Определяем сигнал только на основе RSI с учетом индивидуальных настроек
                if rsi <= rsi_long_threshold:  # RSI ≤ порог LONG (индивидуальный или глобальный)
                    # ✅ ЧИСТЫЙ СИГНАЛ RSI: Входим сразу, без проверки тренда
                    # Защита от "падающего ножа" уже есть:
                    # - Временной фильтр RSI (блокирует если oversold слишком долго)
                    # - Pump-Dump фильтр (определяет искусственные движения)
                    # - ExitScam фильтр (защита от скама)
                    # - AI фильтр (дополнительный анализ)
                    # - Стоп-лосс 15% (ограничивает убытки)
                    rsi_zone = 'BUY_ZONE'
                    signal = 'ENTER_LONG'  # ✅ Входим в лонг по сигналу RSI
                
                elif rsi >= rsi_short_threshold:  # RSI ≥ порог SHORT (индивидуальный или глобальный)
                    # ✅ ЧИСТЫЙ СИГНАЛ RSI: Входим сразу, без проверки тренда
                    rsi_zone = 'SELL_ZONE'
                    signal = 'ENTER_SHORT'  # ✅ Входим в шорт по сигналу RSI
                else:
                    # RSI в нейтральной зоне
                    pass
            except Exception as e:
                logger.debug(f"{symbol}: Ошибка определения RSI сигнала: {e}")
                # Fallback к базовой логике при ошибке
                if rsi <= rsi_long_threshold:
                    rsi_zone = 'BUY_ZONE'
                    signal = 'ENTER_LONG'
                elif rsi >= rsi_short_threshold:
                    rsi_zone = 'SELL_ZONE'
                    signal = 'ENTER_SHORT'
        else:
            # Fallback к старой логике если EMA периоды недоступны
            if rsi <= rsi_long_threshold:
                rsi_zone = 'BUY_ZONE'
                if avoid_down_trend and trend == 'DOWN':
                    signal = 'WAIT'
                else:
                    signal = 'ENTER_LONG'
            elif rsi >= rsi_short_threshold:
                rsi_zone = 'SELL_ZONE'
                if avoid_up_trend and trend == 'UP':
                    signal = 'WAIT'
                else:
                    signal = 'ENTER_SHORT'
        # RSI между порогами - нейтральная зона
        
        # ✅ ФИЛЬТР 3: Существующие позиции (ОТКЛЮЧЕН для ускорения RSI расчета)
        # ⚡ ОПТИМИЗАЦИЯ: Проверка позиций слишком медленная (API запрос к бирже в каждом потоке!)
        # Эта проверка будет выполнена позже в process_auto_bot_signals() ПЕРЕД созданием бота
        has_existing_position = False
        # ПРОПУСКАЕМ ПРОВЕРКУ ПОЗИЦИЙ ЗДЕСЬ - экономим ~50 API запросов к бирже!
        
        # ✅ ФИЛЬТР 4: Enhanced RSI — считаем ТОЛЬКО когда есть потенциальный сигнал
        potential_signal = None
        enhanced_analysis = {
            'enabled': False,
            'warning_type': None,
            'warning_message': None,
            'extreme_duration': 0,
            'adaptive_levels': None,
            'confirmations': {},
            'enhanced_signal': None,
            'enhanced_reason': None,
        }

        if signal in ['ENTER_LONG', 'ENTER_SHORT'] or potential_signal in ['ENTER_LONG', 'ENTER_SHORT']:
            enhanced_analysis = perform_enhanced_rsi_analysis(candles, rsi, symbol) or enhanced_analysis

            # Если Enhanced RSI включен и дает другой сигнал - используем его
            if enhanced_analysis.get('enabled') and enhanced_analysis.get('enhanced_signal'):
                original_signal = signal
                enhanced_signal = enhanced_analysis.get('enhanced_signal')
                if enhanced_signal != original_signal:
                    logger.info(f"{symbol}: Сигнал изменен {original_signal} → {enhanced_signal}")
                    signal = enhanced_signal
                    # Если Enhanced RSI говорит WAIT - блокируем
                    if signal == 'WAIT':
                        rsi_zone = 'NEUTRAL'
        
        # ✅ ФИЛЬТР 5: Зрелость монеты (проверяем ПОСЛЕ Enhanced RSI)
        # 🔧 ИСПРАВЛЕНИЕ: Проверяем зрелость для ВСЕХ монет (для UI фильтра "Зрелые монеты")
        # ✅ Используем индивидуальные настройки, если они есть, иначе глобальные
        enable_maturity_check = individual_settings.get('enable_maturity_check') if individual_settings else None
        if enable_maturity_check is None:
            enable_maturity_check = bots_data.get('auto_bot_config', {}).get('enable_maturity_check', True)
        is_mature = True  # По умолчанию считаем зрелой (если проверка отключена)
        
        if enable_maturity_check:
            # ✅ ИСПОЛЬЗУЕМ хранилище зрелых монет для быстрой проверки
            is_mature = check_coin_maturity_stored_or_verify(symbol)
            
            # Если есть сигнал входа И монета незрелая - блокируем сигнал
            if signal in ['ENTER_LONG', 'ENTER_SHORT'] and not is_mature:
                # Ограничиваем частоту логирования - не более раза в 2 минуты для каждой монеты
                log_message = f"{symbol}: Монета незрелая - сигнал {signal} заблокирован"
                category = f'maturity_check_{symbol}'
                should_log, message = should_log_message(category, log_message, interval_seconds=120)
                if should_log:
                    logger.debug(message)
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
        loss_reentry_info = None  # ✅ Инициализируем ДО использования в result
        
        # ✅ Получаем пороги для фильтров с учетом индивидуальных настроек
        # Пороги RSI уже определены выше (с учетом индивидуальных настроек)
        # Получаем пороги временного фильтра: сначала индивидуальные, затем глобальные
        rsi_time_filter_lower = individual_settings.get('rsi_time_filter_lower') if individual_settings else None
        if rsi_time_filter_lower is None:
            rsi_time_filter_lower = bots_data.get('auto_bot_config', {}).get('rsi_time_filter_lower', 35)  # Нижняя граница для LONG
        
        rsi_time_filter_upper = individual_settings.get('rsi_time_filter_upper') if individual_settings else None
        if rsi_time_filter_upper is None:
            rsi_time_filter_upper = bots_data.get('auto_bot_config', {}).get('rsi_time_filter_upper', 65)  # Верхняя граница для SHORT
        
        # Определяем потенциальный сигнал для проверки фильтров
        # ВАЖНО: Проверяем фильтры если RSI в зоне фильтра:
        # - Для LONG: RSI <= 35 (нижняя граница)
        # - Для SHORT: RSI >= 65 (верхняя граница)
        
        # Определяем potential_signal для проверки фильтров
        if rsi is not None:
            # Проверяем, в какой зоне находится RSI
            if rsi <= rsi_time_filter_lower:
                # RSI в зоне фильтра для LONG - проверяем последние N свечей на наличие лоя
                potential_signal = 'ENTER_LONG'
            elif rsi >= rsi_time_filter_upper:
                # RSI в зоне фильтра для SHORT - проверяем последние N свечи на наличие пика
                potential_signal = 'ENTER_SHORT'
            else:
                # RSI вне зоны фильтра - показываем что фильтр не активен
                potential_signal = None  # Вне зоны входа
                time_filter_info = {
                    'blocked': False,
                    'reason': 'RSI временной фильтр вне зоны входа в сделку',
                    'filter_type': 'time_filter',
                    'last_extreme_candles_ago': None,
                    'calm_candles': None
                }
                # Для монет вне зоны входа ExitScam фильтр не проверяется (оптимизация)
                exit_scam_info = {
                    'blocked': False,
                    'reason': 'ExitScam фильтр: RSI вне зоны входа в сделку',
                    'filter_type': 'exit_scam'
                }
                # Для монет вне зоны входа защита от повторных входов не проверяется
                loss_reentry_info = {
                    'blocked': False,
                    'reason': 'Защита от повторных входов: RSI вне зоны входа в сделку',
                    'filter_type': 'loss_reentry_protection'
                }
        else:
            # RSI не определен - все фильтры не активны
            potential_signal = None
            time_filter_info = {
                'blocked': False,
                'reason': 'RSI временной фильтр: RSI не определен',
                'filter_type': 'time_filter',
                'last_extreme_candles_ago': None,
                'calm_candles': None
            }
            exit_scam_info = {
                'blocked': False,
                'reason': 'ExitScam фильтр: RSI не определен',
                'filter_type': 'exit_scam'
            }
            loss_reentry_info = {
                'blocked': False,
                'reason': 'Защита от повторных входов: RSI не определен',
                'filter_type': 'loss_reentry_protection'
            }
        
        # Проверяем фильтры если монета в зоне фильтра (LONG/SHORT)
        # ✅ ИСПРАВЛЕНИЕ: Проверяем фильтры для UI, чтобы показывать блокировки
        # ⚡ ОПТИМИЗАЦИЯ: Проверяем фильтры ТОЛЬКО для монет в зоне входа (RSI <= 35 для LONG или RSI >= 65 для SHORT)
        # Это не влияет на производительность бэкенда, так как проверки выполняются только для монет, которые уже должны пойти в лонг/шорт
        # Используем уже загруженные свечи из переменной candles, не делаем новых запросов к бирже!
        if potential_signal in ['ENTER_LONG', 'ENTER_SHORT']:
            # ✅ Проверяем RSI Time Filter для UI
            if len(candles) >= 50:
                try:
                    time_filter_result = check_rsi_time_filter(
                        candles, 
                        rsi, 
                        potential_signal, 
                        symbol=symbol, 
                        individual_settings=individual_settings
                    )
                    if time_filter_result:
                        time_filter_info = {
                            'blocked': not time_filter_result.get('allowed', True),
                            'reason': time_filter_result.get('reason', ''),
                            'filter_type': 'time_filter',
                            'last_extreme_candles_ago': time_filter_result.get('last_extreme_candles_ago'),
                            'calm_candles': time_filter_result.get('calm_candles')
                        }
                    else:
                        time_filter_info = {
                            'blocked': False,
                            'reason': 'RSI временной фильтр: проверка не выполнена',
                            'filter_type': 'time_filter',
                            'last_extreme_candles_ago': None,
                            'calm_candles': None
                        }
                except Exception as e:
                    logger.debug(f"{symbol}: Ошибка проверки RSI Time Filter для UI: {e}")
                    time_filter_info = {
                        'blocked': False,
                        'reason': f'Ошибка проверки: {str(e)}',
                        'filter_type': 'time_filter',
                        'last_extreme_candles_ago': None,
                        'calm_candles': None
                    }
            else:
                time_filter_info = {
                    'blocked': False,
                    'reason': 'Недостаточно свечей для проверки (требуется минимум 50)',
                    'filter_type': 'time_filter',
                    'last_extreme_candles_ago': None,
                    'calm_candles': None
                }
            
            # ✅ Проверяем ExitScam Filter для UI (используем уже загруженные свечи из candles)
            # ⚡ ОПТИМИЗАЦИЯ: Используем свечи, которые уже загружены выше, не делаем новый запрос к бирже!
            try:
                if len(candles) >= 10:  # Минимум свечей для проверки ExitScam
                    # Получаем конфиг с учетом индивидуальных настроек
                    auto_config = bots_data.get('auto_bot_config', {}).copy()
                    if individual_settings:
                        for key in ['exit_scam_enabled', 'exit_scam_candles', 
                                   'exit_scam_single_candle_percent', 'exit_scam_multi_candle_count',
                                   'exit_scam_multi_candle_percent']:
                            if key in individual_settings:
                                auto_config[key] = individual_settings[key]
                    
                    exit_scam_enabled = auto_config.get('exit_scam_enabled', True)
                    exit_scam_candles = auto_config.get('exit_scam_candles', 10)
                    single_candle_percent = auto_config.get('exit_scam_single_candle_percent', 15.0)
                    multi_candle_count = auto_config.get('exit_scam_multi_candle_count', 4)
                    multi_candle_percent = auto_config.get('exit_scam_multi_candle_percent', 50.0)
                    
                    exit_scam_allowed = True
                    exit_scam_reason = 'ExitScam фильтр пройден'
                    
                    if exit_scam_enabled and len(candles) >= exit_scam_candles:
                        # Проверяем последние N свечей (используем уже загруженные свечи!)
                        recent_candles = candles[-exit_scam_candles:]
                        
                        # 1. Проверка отдельных свечей
                        for candle in recent_candles:
                            open_price = candle['open']
                            close_price = candle['close']
                            price_change = abs((close_price - open_price) / open_price) * 100
                            
                            if price_change > single_candle_percent:
                                exit_scam_allowed = False
                                exit_scam_reason = f'ExitScam фильтр: одна свеча превысила лимит {single_candle_percent}% (было {price_change:.1f}%)'
                                break
                        
                        # 2. Проверка суммарного изменения (если первая проверка прошла)
                        if exit_scam_allowed and len(recent_candles) >= multi_candle_count:
                            multi_candles = recent_candles[-multi_candle_count:]
                            first_open = multi_candles[0]['open']
                            last_close = multi_candles[-1]['close']
                            total_change = abs((last_close - first_open) / first_open) * 100
                            
                            if total_change > multi_candle_percent:
                                exit_scam_allowed = False
                                exit_scam_reason = f'ExitScam фильтр: {multi_candle_count} свечей превысили суммарный лимит {multi_candle_percent}% (было {total_change:.1f}%)'
                        
                        # 3. AI детекция аномалий (если включена и базовые проверки прошли)
                        if exit_scam_allowed:
                            try:
                                from bot_engine.bot_config import AIConfig
                                if AIConfig.AI_ENABLED and AIConfig.AI_ANOMALY_DETECTION_ENABLED:
                                    exit_scam_allowed = _run_exit_scam_ai_detection(symbol, candles)
                                    if not exit_scam_allowed:
                                        exit_scam_reason = 'ExitScam фильтр: AI обнаружил аномалию'
                            except ImportError:
                                pass  # AI модуль не доступен
                    
                    exit_scam_info = {
                        'blocked': not exit_scam_allowed,
                        'reason': exit_scam_reason,
                        'filter_type': 'exit_scam'
                    }
                else:
                    # Недостаточно свечей для проверки
                    exit_scam_info = {
                        'blocked': False,
                        'reason': 'Недостаточно свечей для проверки (требуется минимум 10)',
                        'filter_type': 'exit_scam'
                    }
            except Exception as e:
                logger.debug(f"{symbol}: Ошибка проверки ExitScam Filter для UI: {e}")
                exit_scam_info = {
                    'blocked': False,
                    'reason': f'Ошибка проверки: {str(e)}',
                    'filter_type': 'exit_scam'
                }
            
            # ✅ Проверяем защиту от повторных входов после убыточных закрытий для UI
            try:
                # ✅ КРИТИЧНО: Проверяем наличие открытой позиции - если позиция уже открыта, фильтр НЕ применяется
                has_existing_position_check = False
                from bots_modules.imports_and_globals import bots_data_lock
                with bots_data_lock:
                    bots = bots_data.get('bots', {})
                    bot = bots.get(symbol)
                    if bot:
                        bot_status = bot.get('status', '')
                        position_side = bot.get('position_side')
                        has_existing_position_check = (bot_status == BOT_STATUS['IN_POSITION_LONG'] or 
                                                      bot_status == BOT_STATUS['IN_POSITION_SHORT'] or 
                                                      position_side is not None)
                
                if len(candles) >= 10:  # Минимум свечей для проверки
                    # Получаем конфиг с учетом индивидуальных настроек
                    auto_config = bots_data.get('auto_bot_config', {}).copy()
                    if individual_settings:
                        for key in ['loss_reentry_protection', 'loss_reentry_count', 'loss_reentry_candles']:
                            if key in individual_settings:
                                auto_config[key] = individual_settings[key]
                    
                    loss_reentry_protection_enabled = auto_config.get('loss_reentry_protection', True)
                    loss_reentry_count = auto_config.get('loss_reentry_count', 1)
                    loss_reentry_candles = auto_config.get('loss_reentry_candles', 3)
                    
                    # ✅ ИСПРАВЛЕНО: Всегда проверяем фильтр, проверка позиции делается на уровне бота
                    if loss_reentry_protection_enabled:
                        # Вызываем проверку защиты (проверка позиции убрана - она в should_open_long/short)
                        loss_reentry_result = _check_loss_reentry_protection_static(
                            symbol, candles, loss_reentry_count, loss_reentry_candles, individual_settings
                        )
                        if loss_reentry_result:
                            loss_reentry_info = {
                                'blocked': not loss_reentry_result.get('allowed', True),
                                'reason': loss_reentry_result.get('reason', ''),
                                'filter_type': 'loss_reentry_protection',
                                'candles_passed': loss_reentry_result.get('candles_passed'),
                                'required_candles': loss_reentry_candles,
                                'loss_count': loss_reentry_count
                            }
                        else:
                            loss_reentry_info = {
                                'blocked': False,
                                'reason': 'Защита от повторных входов: проверка не выполнена',
                                'filter_type': 'loss_reentry_protection'
                            }
                    else:
                        loss_reentry_info = {
                            'blocked': False,
                            'reason': 'Защита от повторных входов: выключена',
                            'filter_type': 'loss_reentry_protection'
                        }
                else:
                    loss_reentry_info = {
                        'blocked': False,
                        'reason': 'Недостаточно свечей для проверки',
                        'filter_type': 'loss_reentry_protection'
                    }
            except Exception as e:
                logger.debug(f"{symbol}: Ошибка проверки защиты от повторных входов для UI: {e}")
                loss_reentry_info = {
                    'blocked': False,
                    'reason': f'Ошибка проверки: {str(e)}',
                    'filter_type': 'loss_reentry_protection'
                }
        
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
            logger.info(f"{symbol}: Известная делистинговая монета")
        elif symbol in known_new_coins:
            trading_status = 'Delivering'
            is_delisting = True
            logger.info(f"{symbol}: Известная новая монета")
        
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
            # ⚡ ОПТИМИЗАЦИЯ: Enhanced RSI, фильтры и флаги ТОЛЬКО если проверялись
            'enhanced_rsi': enhanced_analysis if enhanced_analysis else {'enabled': False},
            'time_filter_info': time_filter_info,
            'exit_scam_info': exit_scam_info,  # None - проверка только при входе в позицию
            'blocked_by_scope': is_blocked_by_scope,
            'has_existing_position': has_existing_position,
            'is_mature': is_mature if enable_maturity_check else True,
            # ✅ КРИТИЧНО: Флаги блокировки для get_effective_signal и UI
            # Устанавливаем флаги на основе результатов проверки фильтров для UI
            'blocked_by_exit_scam': exit_scam_info.get('blocked', False) if exit_scam_info else False,
            'blocked_by_rsi_time': time_filter_info.get('blocked', False) if time_filter_info else False,
            'loss_reentry_info': loss_reentry_info,  # Информация о защите от повторных входов
            'blocked_by_loss_reentry': loss_reentry_info.get('blocked', False) if loss_reentry_info else False,
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
            logger.info(f"🎯 {symbol}: RSI={rsi:.1f} {trend_emoji}{trend_display} (${current_price:.4f}) → {signal}")
        elif signal == 'WAIT' and rsi <= SystemConfig.RSI_OVERSOLD and trend == 'DOWN' and avoid_down_trend:
            logger.debug(f"🚫 {symbol}: RSI={rsi:.1f} {trend_emoji}{trend_display} LONG заблокирован (фильтр DOWN тренда)")
        elif signal == 'WAIT' and rsi >= SystemConfig.RSI_OVERBOUGHT and trend == 'UP' and avoid_up_trend:
            logger.debug(f"🚫 {symbol}: RSI={rsi:.1f} {trend_emoji}{trend_display} SHORT заблокирован (фильтр UP тренда)")
        
        debug_payload = {
            'source': data_source,
            'duration': round(time.time() - thread_start, 3),
            'thread': threading.current_thread().name
        }
        result['debug_info'] = debug_payload
        return result
        
    except Exception as e:
        logger.error(f"Ошибка получения данных для {symbol}: {e}")
        return None

def load_all_coins_candles_fast():
    """⚡ БЫСТРАЯ загрузка ТОЛЬКО свечей для всех монет БЕЗ расчетов"""
    try:
        from bots_modules.imports_and_globals import get_exchange
        current_exchange = get_exchange()
        
        if not current_exchange:
            logger.error("❌ Биржа не инициализирована")
            return False
        
        if shutdown_flag.is_set():
            logger.warning("⏹️ Загрузка свечей отменена: система завершает работу")
            return False

        # Получаем список всех пар
        pairs = current_exchange.get_all_pairs()
        if not pairs:
            logger.error("❌ Не удалось получить список пар")
            return False
        
        # Загружаем ТОЛЬКО свечи пакетами (УСКОРЕННАЯ ВЕРСИЯ)
        batch_size = 100  # Увеличили с 50 до 100
        candles_cache = {}
        
        import concurrent.futures
        # ⚡ АДАПТИВНОЕ УПРАВЛЕНИЕ ВОРКЕРАМИ: начинаем с 20, временно уменьшаем при rate limit
        current_max_workers = 20  # Базовое количество воркеров
        rate_limit_detected = False  # Флаг обнаружения rate limit в предыдущем батче
        
        shutdown_requested = False

        for i in range(0, len(pairs), batch_size):
            if shutdown_flag.is_set():
                shutdown_requested = True
                break

            batch = pairs[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(pairs) + batch_size - 1)//batch_size
            
            # ⚡ ВРЕМЕННОЕ УМЕНЬШЕНИЕ ВОРКЕРОВ: если в предыдущем батче был rate limit
            if rate_limit_detected:
                current_max_workers = max(17, current_max_workers - 3)  # Уменьшаем на 3, но не меньше 17
                logger.warning(f"⚠️ Rate limit обнаружен в предыдущем батче. Временно уменьшаем воркеры до {current_max_workers}")
                rate_limit_detected = False  # Сбрасываем флаг для следующего батча
            elif current_max_workers < 20:
                # Возвращаем к базовому значению после успешного батча
                logger.info(f"✅ Возвращаем воркеры к базовому значению: {current_max_workers} → 20")
                current_max_workers = 20
            
            # ⚡ ОТСЛЕЖИВАНИЕ RATE LIMIT: проверяем задержку до и после батча
            delay_before_batch = current_exchange.current_request_delay if hasattr(current_exchange, 'current_request_delay') else None
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=current_max_workers) as executor:
                future_to_symbol = {
                    executor.submit(get_coin_candles_only, symbol, current_exchange): symbol
                    for symbol in batch
                }

                if shutdown_flag.is_set():
                    shutdown_requested = True
                    for future in future_to_symbol:
                        future.cancel()
                    break
                
                completed = 0
                done, not_done = concurrent.futures.wait(
                    future_to_symbol.keys(),
                    timeout=90,
                    return_when=concurrent.futures.ALL_COMPLETED
                )

                if shutdown_flag.is_set():
                    shutdown_requested = True
                    for future in future_to_symbol:
                        future.cancel()
                    break
                
                for future in done:
                    symbol = future_to_symbol.get(future)
                    try:
                        result = future.result()
                        if result:
                            candles_cache[result['symbol']] = result
                            completed += 1
                    except Exception:
                        pass
                
                if not_done:
                    unfinished_symbols = [future_to_symbol.get(future) for future in not_done if future in future_to_symbol]
                    logger.error(f"❌ Timeout: {len(unfinished_symbols)} (of {len(future_to_symbol)}) futures unfinished")
                    
                    # Отменяем незавершенные задачи и фиксируем возможный rate limit
                    for future in not_done:
                        try:
                            future.cancel()
                        except Exception:
                            pass
                    rate_limit_detected = True
                
                # Проверяем, увеличилась ли задержка после батча (признак rate limit)
                delay_after_batch = current_exchange.current_request_delay if hasattr(current_exchange, 'current_request_delay') else None
                if delay_before_batch is not None and delay_after_batch is not None:
                    if delay_after_batch > delay_before_batch:
                        # Задержка увеличилась - был rate limit
                        rate_limit_detected = True
                        logger.warning(f"⚠️ Rate limit обнаружен в батче {batch_num}/{total_batches}: задержка увеличилась {delay_before_batch:.3f}с → {delay_after_batch:.3f}с")
                
                # Уменьшили паузу между пакетами
                import time
                if shutdown_flag.wait(0.1):
                    shutdown_requested = True
                    break

            if shutdown_requested:
                break
        
        if shutdown_requested:
            logger.warning("⏹️ Загрузка свечей прервана из-за остановки системы")
            return False
        
        logger.info(f"✅ Загрузка завершена: {len(candles_cache)} монет")
        
        # ⚡ ИСПРАВЛЕНИЕ DEADLOCK: Сохраняем в глобальный кэш БЕЗ блокировки
        # rsi_data_lock может быть захвачен ContinuousDataLoader в другом потоке
        try:
            logger.info(f"💾 Сохраняем кэш в глобальное хранилище...")
            coins_rsi_data['candles_cache'] = candles_cache
            coins_rsi_data['last_candles_update'] = datetime.now().isoformat()
            logger.info(f"✅ Кэш сохранен: {len(candles_cache)} монет")
            logger.info(f"✅ Проверка: в глобальном кэше сейчас {len(coins_rsi_data.get('candles_cache', {}))} монет")
        except Exception as cache_error:
            logger.warning(f"⚠️ Ошибка сохранения кэша: {cache_error}")
        
        # ✅ Сохраняем свечи в БД БЕЗ накопления!
        # Запрашивается только 30 дней (~120 свечей), поэтому НЕ нужно накапливать старые данные
        # save_candles_cache() сам удалит старые свечи и вставит только новые
        # ⚠️ КРИТИЧНО: Проверяем, запущен ли процесс как ai.py - если да, НЕ сохраняем в bots_data.db!
        try:
            import sys
            import os
            # Более надежная проверка: смотрим имя скрипта, модуль __main__ и переменные окружения
            script_name = os.path.basename(sys.argv[0]) if sys.argv else ''
            main_file = None
            try:
                if hasattr(sys.modules.get('__main__', None), '__file__') and sys.modules['__main__'].__file__:
                    main_file = str(sys.modules['__main__'].__file__).lower()
            except:
                pass
            
            # ⚠️ КРИТИЧНО: Явно инициализируем переменные
            is_bots_process = False
            is_ai_process = False
            
            # Проверяем по имени скрипта, аргументам, файлу __main__ и переменной окружения
            # ⚠️ ВАЖНО: Сначала проверяем, что это НЕ bots.py, потом проверяем ai.py
            is_bots_process = (
                'bots.py' in script_name.lower() or 
                any('bots.py' in str(arg).lower() for arg in sys.argv) or
                (main_file and 'bots.py' in main_file)
            )
            
            # Если это точно bots.py - НЕ проверяем дальше и игнорируем переменную окружения
            if is_bots_process:
                is_ai_process = False
                logger.debug(f"🔍 Обнаружен процесс bots.py - сохраняем свечи в bots_data.db (script_name={script_name}, main_file={main_file})")
            else:
                # Проверяем, что это ai.py (переменная окружения учитывается ТОЛЬКО если это не bots.py)
                env_flag = os.environ.get('INFOBOT_AI_PROCESS', '').lower() == 'true'
                is_ai_process = (
                    'ai.py' in script_name.lower() or 
                    any('ai.py' in str(arg).lower() for arg in sys.argv) or
                    (main_file and 'ai.py' in main_file) or
                    env_flag
                )
                if is_ai_process:
                    logger.info(f"🔍 Обнаружен процесс ai.py - сохраняем свечи ТОЛЬКО в ai_data.db (script_name={script_name}, main_file={main_file}, env_flag={env_flag})")
            
            if is_ai_process:
                # Если это процесс ai.py - сохраняем ТОЛЬКО в ai_data.db, НЕ в bots_data.db!
                logger.info(f"🔍 Обнаружен процесс ai.py - сохраняем свечи ТОЛЬКО в ai_data.db (script_name={script_name}, main_file={main_file}, env={os.environ.get('INFOBOT_AI_PROCESS', '')})")
                try:
                    from bot_engine.ai.ai_database import get_ai_database
                    ai_db = get_ai_database()
                    if ai_db:
                        # Преобразуем формат для ai_database
                        saved_count = 0
                        for symbol, candle_data in candles_cache.items():
                            if isinstance(candle_data, dict):
                                candles = candle_data.get('candles', [])
                                if candles:
                                    ai_db.save_candles(symbol, candles, timeframe='6h')
                                    saved_count += 1
                        logger.info(f"✅ Свечи сохранены в ai_data.db: {saved_count} монет (процесс ai.py)")
                    else:
                        logger.error("❌ AI Database недоступна, свечи НЕ сохранены!")
                except Exception as ai_db_error:
                    logger.error(f"❌ Ошибка сохранения в ai_data.db: {ai_db_error}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                # Это процесс bots.py или неизвестный процесс - сохраняем в bots_data.db
                # ⚠️ ВАЖНО: Если это НЕ bots.py и НЕ ai.py - это может быть ошибка!
                if not is_bots_process:
                    logger.warning(f"⚠️ Неизвестный процесс вызывает load_all_coins_candles_fast()! script_name={script_name}, main_file={main_file}")
                    logger.warning(f"⚠️ Сохраняем в bots_data.db (по умолчанию)")
                
                from bot_engine.storage import save_candles_cache
                
                # Просто сохраняем текущие свечи - save_candles_cache() сам ограничит до 1000 и удалит старые
                if save_candles_cache(candles_cache):
                    logger.info(f"💾 Кэш свечей сохранен в bots_data.db: {len(candles_cache)} монет (процесс bots.py)")
                else:
                    logger.error(f"❌ Не удалось сохранить свечи в bots_data.db!")
            
        except Exception as db_error:
            logger.warning(f"⚠️ Ошибка сохранения в БД кэша: {db_error}")
        
        # 🔄 Сбрасываем задержку запросов после успешной загрузки раунда
        try:
            if current_exchange and hasattr(current_exchange, 'reset_request_delay'):
                current_exchange.reset_request_delay()
                logger.info(f"🔄 Задержка запросов сброшена к базовому значению")
        except Exception as reset_error:
            logger.warning(f"⚠️ Ошибка сброса задержки: {reset_error}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        return False

def load_all_coins_rsi():
    """Загружает RSI 6H для всех доступных монет"""
    global coins_rsi_data
    
    try:
        operation_start = time.time()
        logger.info("📊 RSI: запускаем полное обновление")
        # ⚡ БЕЗ БЛОКИРОВКИ: проверяем флаг без блокировки
        if coins_rsi_data['update_in_progress']:
            logger.info("Обновление RSI уже выполняется...")
            return False
        
        # ⚡ УСТАНАВЛИВАЕМ флаги БЕЗ блокировки
        coins_rsi_data['update_in_progress'] = True
        # ✅ UI блокировка уже установлена в continuous_data_loader
        
        if shutdown_flag.is_set():
            logger.warning("⏹️ Обновление RSI отменено: система завершает работу")
            coins_rsi_data['update_in_progress'] = False
            return False

        # ✅ КРИТИЧНО: Создаем ВРЕМЕННОЕ хранилище для всех монет
        # Обновляем coins_rsi_data ТОЛЬКО после завершения всех проверок!
        temp_coins_data = {}
        
        # Проверяем кэш свечей перед началом
        candles_cache_size = len(coins_rsi_data.get('candles_cache', {}))
        
        # Получаем актуальную ссылку на биржу
        try:
            from bots_modules.imports_and_globals import get_exchange
            current_exchange = get_exchange()
        except Exception as e:
            logger.error(f"❌ Ошибка получения биржи: {e}")
            current_exchange = None
        
        # Получаем список всех пар
        if not current_exchange:
            logger.error("❌ Биржа не инициализирована")
            coins_rsi_data['update_in_progress'] = False
            return False
            
        pairs = current_exchange.get_all_pairs()
        
        if not pairs or not isinstance(pairs, list):
            logger.error("❌ Не удалось получить список пар с биржи")
            return False
        
        logger.info(f"📊 RSI: получено {len(pairs)} пар, готовим батчи по 100 монет")
        
        # ⚡ БЕЗ БЛОКИРОВКИ: обновляем счетчики напрямую
        coins_rsi_data['total_coins'] = len(pairs)
        coins_rsi_data['successful_coins'] = 0
        coins_rsi_data['failed_coins'] = 0
        
        # ✅ ПАРАЛЛЕЛЬНАЯ загрузка с текстовым прогрессом (работает в лог-файле)
        batch_size = 100
        total_batches = (len(pairs) + batch_size - 1) // batch_size
        
        shutdown_requested = False

        for i in range(0, len(pairs), batch_size):
            if shutdown_flag.is_set():
                shutdown_requested = True
                break
            batch = pairs[i:i + batch_size]
            batch_num = i // batch_size + 1
            batch_start = time.time()
            request_delay = getattr(current_exchange, 'current_request_delay', 0) or 0
            logger.info(
                f"📦 RSI Batch {batch_num}/{total_batches}: size={len(batch)}, "
                f"workers=50, delay={request_delay:.2f}s"
            )
            batch_success = 0
            batch_fail = 0
            
            # Параллельная обработка пакета
            with ThreadPoolExecutor(max_workers=50) as executor:
                future_to_symbol = {
                    executor.submit(get_coin_rsi_data, symbol, current_exchange): symbol 
                    for symbol in batch
                }

                if shutdown_flag.is_set():
                    shutdown_requested = True
                    for future in future_to_symbol:
                        future.cancel()
                    break
                
                try:
                    for future in concurrent.futures.as_completed(future_to_symbol, timeout=60):
                        if shutdown_flag.is_set():
                            shutdown_requested = True
                            break

                        symbol = future_to_symbol[future]
                        try:
                            result = future.result(timeout=20)
                            if result:
                                temp_coins_data[result['symbol']] = result
                                coins_rsi_data['successful_coins'] += 1
                                batch_success += 1
                            else:
                                coins_rsi_data['failed_coins'] += 1
                                batch_fail += 1
                        except Exception as e:
                            logger.error(f"❌ {symbol}: {e}")
                            coins_rsi_data['failed_coins'] += 1
                            batch_fail += 1
                except concurrent.futures.TimeoutError:
                    pending = list(future_to_symbol.values())
                    logger.error(
                        f"⚠️ Timeout при загрузке RSI для пакета {batch_num} "
                        f"(ожидали {len(pending)} символов, примеры: {pending[:5]})"
                    )
                    coins_rsi_data['failed_coins'] += len(batch)
                    batch_fail += len(batch)

                if shutdown_flag.is_set():
                    shutdown_requested = True
                    for future in future_to_symbol:
                        future.cancel()
                    break
            
            logger.info(
                f"📦 RSI Batch {batch_num}/{total_batches} завершен: "
                f"{batch_success} успехов / {batch_fail} ошибок за "
                f"{time.time() - batch_start:.1f}s"
            )
            
            # ✅ Выводим прогресс в лог
            processed = coins_rsi_data['successful_coins'] + coins_rsi_data['failed_coins']
            if batch_num <= total_batches:
                logger.info(f"📊 Прогресс: {processed}/{len(pairs)} ({processed*100//len(pairs)}%)")

            if shutdown_requested:
                break

        if shutdown_requested:
            logger.warning("⏹️ Обновление RSI прервано из-за остановки системы")
            coins_rsi_data['update_in_progress'] = False
            return False
        
        # ✅ КРИТИЧНО: АТОМАРНОЕ обновление всех данных ОДНИМ МАХОМ!
        coins_rsi_data['coins'] = temp_coins_data
        coins_rsi_data['last_update'] = datetime.now().isoformat()
        coins_rsi_data['update_in_progress'] = False
        
        # Финальный отчет
        success_count = coins_rsi_data['successful_coins']
        failed_count = coins_rsi_data['failed_coins']
            
        # Подсчитываем сигналы
        enter_long_count = sum(1 for coin in coins_rsi_data['coins'].values() if coin.get('signal') == 'ENTER_LONG')
        enter_short_count = sum(1 for coin in coins_rsi_data['coins'].values() if coin.get('signal') == 'ENTER_SHORT')
        
        logger.info(f"✅ {success_count} монет | Сигналы: {enter_long_count} LONG + {enter_short_count} SHORT")
        
        if failed_count > 0:
            logger.warning(f"⚠️ Ошибок: {failed_count} монет")
        
        # Обновляем флаги is_mature
        try:
            update_is_mature_flags_in_rsi_data()
        except Exception as update_error:
            logger.warning(f"⚠️ Не удалось обновить is_mature: {update_error}")
        
        # 🔄 Сбрасываем задержку запросов после успешной загрузки раунда
        try:
            if current_exchange and hasattr(current_exchange, 'reset_request_delay'):
                current_exchange.reset_request_delay()
                logger.info(f"🔄 Задержка запросов сброшена к базовому значению")
        except Exception as reset_error:
            logger.warning(f"⚠️ Ошибка сброса задержки: {reset_error}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка загрузки RSI данных: {str(e)}")
        # ⚡ БЕЗ БЛОКИРОВКИ: атомарная операция
        coins_rsi_data['update_in_progress'] = False
        return False
    finally:
        elapsed = time.time() - operation_start
        logger.info(f"📊 RSI: полное обновление завершено за {elapsed:.1f}s")
        # Гарантированно сбрасываем флаг обновления
        # ⚡ БЕЗ БЛОКИРОВКИ: атомарная операция
        if coins_rsi_data['update_in_progress']:
            logger.warning(f"⚠️ Принудительный сброс флага update_in_progress")
            coins_rsi_data['update_in_progress'] = False

def _recalculate_signal_with_trend(rsi, trend, symbol):
    """Пересчитывает сигнал с учетом нового тренда"""
    try:
        # Получаем настройки автобота
        auto_config = bots_data.get('auto_bot_config', {})
        # ✅ ИСПРАВЛЕНО: Используем False по умолчанию (как в bot_config.py), а не True
        avoid_down_trend = auto_config.get('avoid_down_trend', False)
        avoid_up_trend = auto_config.get('avoid_up_trend', False)
        
        # Определяем базовый сигнал по RSI
        if rsi <= SystemConfig.RSI_OVERSOLD:  # RSI ≤ 29 
            # Проверяем нужно ли избегать DOWN тренда для LONG
            if avoid_down_trend and trend == 'DOWN':
                return 'WAIT'  # Ждем улучшения тренда
            else:
                return 'ENTER_LONG'  # Входим независимо от тренда или при хорошем тренде
        elif rsi >= SystemConfig.RSI_OVERBOUGHT:  # RSI ≥ 71
            # Проверяем нужно ли избегать UP тренда для SHORT
            if avoid_up_trend and trend == 'UP':
                return 'WAIT'  # Ждем ослабления тренда
            else:
                return 'ENTER_SHORT'  # Входим независимо от тренда или при хорошем тренде
        else:
            # RSI между 30-70 - нейтральная зона
            logger.debug(f"🔍 {symbol}: RSI {rsi:.1f} между 30-70 → WAIT")
            return 'WAIT'
            
    except Exception as e:
        logger.error(f"❌ Ошибка пересчета сигнала для {symbol}: {e}")
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
    # ✅ ИСПРАВЛЕНО: Используем False по умолчанию (как в bot_config.py), а не True
    avoid_down_trend = auto_config.get('avoid_down_trend', False)
    avoid_up_trend = auto_config.get('avoid_up_trend', False)
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
        logger.debug(f"{symbol}: ❌ {signal} заблокирован ExitScam фильтром")
        return 'WAIT'
    
    # Проверяем RSI Time фильтр
    if coin.get('blocked_by_rsi_time', False):
        logger.debug(f"{symbol}: ❌ {signal} заблокирован RSI Time фильтром")
        return 'WAIT'
    
    # ✅ Проверяем защиту от повторных входов после убыточных закрытий
    if coin.get('blocked_by_loss_reentry', False):
        loss_reentry_info = coin.get('loss_reentry_info', {})
        reason = loss_reentry_info.get('reason', 'Защита от повторных входов') if loss_reentry_info else 'Защита от повторных входов'
        
        # ✅ АНТИСПАМ: Логируем не чаще раза в минуту для каждой монеты
        current_time = time.time()
        with _loss_reentry_log_lock:
            last_log_time = _loss_reentry_log_cache.get(symbol, 0)
            if current_time - last_log_time >= _loss_reentry_log_interval:
                logger.debug(f"{symbol}: ❌ {signal} заблокирован защитой от повторных входов после убытка: {reason}")
                _loss_reentry_log_cache[symbol] = current_time
        
        return 'WAIT'
    
    # Проверяем зрелость монеты
    if not coin.get('is_mature', True):
        # Ограничиваем частоту логирования - не более раза в 2 минуты для каждой монеты
        log_message = f"{symbol}: ❌ {signal} заблокирован - монета незрелая"
        category = f'maturity_check_{symbol}'
        should_log, message = should_log_message(category, log_message, interval_seconds=120)
        if should_log:
            logger.debug(message)
        return 'WAIT'
    
    # УПРОЩЕННАЯ ПРОВЕРКА ТРЕНДОВ - только экстремальные случаи
    if signal == 'ENTER_SHORT' and avoid_up_trend and rsi >= rsi_short_threshold and trend == 'UP':
        logger.debug(f"{symbol}: ❌ SHORT заблокирован (RSI={rsi:.1f} >= {rsi_short_threshold} + UP тренд)")
        return 'WAIT'
    
    if signal == 'ENTER_LONG' and avoid_down_trend and rsi <= rsi_long_threshold and trend == 'DOWN':
        logger.debug(f"{symbol}: ❌ LONG заблокирован (RSI={rsi:.1f} <= {rsi_long_threshold} + DOWN тренд)")
        return 'WAIT'
    
    # Все проверки пройдены
    return signal

def process_auto_bot_signals(exchange_obj=None):
    """Новая логика автобота согласно требованиям"""
    try:
        # Проверяем, включен ли автобот
        # ⚡ БЕЗ БЛОКИРОВКИ: конфиг не меняется, чтение безопасно
        auto_bot_enabled = bots_data['auto_bot_config']['enabled']
        
        if not auto_bot_enabled:
            logger.info(" ⏹️ Автобот выключен")  # Изменено на INFO
            return
        
        logger.info(" ✅ Автобот включен, начинаем проверку сигналов")
        
        max_concurrent = bots_data['auto_bot_config']['max_concurrent']
        current_active = sum(1 for bot in bots_data['bots'].values() 
                           if bot['status'] not in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']])
        
        if current_active >= max_concurrent:
            logger.debug(f" 🚫 Достигнут лимит активных ботов ({current_active}/{max_concurrent})")
            return
        
        logger.info(" 🔍 Проверка сигналов для создания новых ботов...")
        
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
        
        logger.info(f" 🎯 Найдено {len(potential_coins)} потенциальных сигналов")
        
        # ✅ Логируем найденные сигналы для диагностики
        if potential_coins:
            logger.info(f" 📋 Потенциальные сигналы: {[(c['symbol'], c['signal'], f'RSI={c['rsi']:.1f}') for c in potential_coins[:10]]}")
        
        # Создаем ботов для найденных сигналов
        created_bots = 0
        for coin in potential_coins[:max_concurrent - current_active]:
            symbol = coin['symbol']
            
            # Проверяем, нет ли уже бота для этого символа
            # ⚡ БЕЗ БЛОКИРОВКИ: чтение безопасно
            if symbol in bots_data['bots']:
                logger.info(f" ⚠️ {symbol}: Бот уже существует (статус: {bots_data['bots'][symbol].get('status')})")
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
                        logger.warning(f" 🚫 {symbol}: Обнаружена РУЧНАЯ позиция на бирже - блокируем создание бота!")
                        continue
                        
            except Exception as pos_error:
                logger.warning(f" ⚠️ {symbol}: Ошибка проверки позиций: {pos_error}")
                # Продолжаем создание бота если проверка не удалась
            
            # ✅ КРИТИЧНО: Проверяем ВСЕ фильтры ПЕРЕД созданием бота!
            try:
                from bot_engine.ai.filter_utils import apply_entry_filters
                from bots_modules.imports_and_globals import get_config_snapshot
                
                # Получаем конфиг
                config_snapshot = get_config_snapshot(symbol)
                filter_config = config_snapshot.get('merged', {})
                
                # Получаем свечи для проверки фильтров
                candles = None
                # Сначала пробуем получить из кэша
                candles_cache = coins_rsi_data.get('candles_cache', {})
                if symbol in candles_cache:
                    cached_data = candles_cache[symbol]
                    candles = cached_data.get('candles')
                
                # Если нет в кэше, пробуем загрузить
                if not candles:
                    try:
                        candles_data = get_coin_candles_only(symbol, exchange_obj=exchange_obj)
                        if candles_data:
                            candles = candles_data.get('candles')
                    except Exception as candles_error:
                        logger.debug(f" {symbol}: Не удалось загрузить свечи для проверки фильтров: {candles_error}")
                
                # Если свечи все еще нет, пробуем из БД
                if not candles:
                    try:
                        from bot_engine.storage import get_candles_for_symbol
                        db_cached_data = get_candles_for_symbol(symbol)
                        if db_cached_data:
                            candles = db_cached_data.get('candles', [])
                    except Exception as db_error:
                        logger.debug(f" {symbol}: Не удалось загрузить свечи из БД: {db_error}")
                
                # Проверяем фильтры
                if candles and len(candles) >= 10:
                    current_rsi = coin.get('rsi') or coin_data.get('rsi6h')
                    current_trend = coin.get('trend') or coin_data.get('trend6h', 'NEUTRAL')
                    signal = coin['signal']
                    
                    filters_allowed, filters_reason = apply_entry_filters(
                        symbol,
                        candles,
                        current_rsi if current_rsi is not None else 50.0,
                        signal,
                        filter_config,
                        trend=current_trend
                    )
                    
                    if not filters_allowed:
                        logger.warning(f" 🚫 {symbol}: Фильтры заблокировали создание бота: {filters_reason}")
                        continue  # Пропускаем создание бота
                else:
                    logger.warning(f" ⚠️ {symbol}: Недостаточно свечей для проверки фильтров ({len(candles) if candles else 0}), пропускаем")
                    continue  # Пропускаем создание бота если нет свечей
                    
            except Exception as filter_check_error:
                logger.error(f" ❌ {symbol}: Ошибка проверки фильтров перед созданием бота: {filter_check_error}")
                import traceback
                logger.error(traceback.format_exc())
                # ⚠️ ВАЖНО: Если проверка фильтров не работает, БЛОКИРУЕМ создание бота для безопасности!
                logger.warning(f" 🚫 {symbol}: Блокируем создание бота из-за ошибки проверки фильтров!")
                continue
            
            # Создаем нового бота (фильтры уже проверены!)
            try:
                logger.info(f" 🚀 Создаем бота для {symbol} ({coin['signal']}, RSI: {coin['rsi']:.1f})")
                new_bot = create_new_bot(symbol, exchange_obj=exchange_obj)
                
                # ✅ КРИТИЧНО: Сразу входим в позицию!
                signal = coin['signal']
                direction = 'LONG' if signal == 'ENTER_LONG' else 'SHORT'
                logger.info(f" 📈 Входим в позицию {direction} для {symbol}")
                new_bot.enter_position(direction)
                
                created_bots += 1
                
            except Exception as e:
                # Блокировка фильтрами - это нормальная работа системы, логируем как WARNING
                error_str = str(e)
                if 'заблокирован фильтрами' in error_str or 'filters_blocked' in error_str:
                    logger.warning(f" ⚠️ Ошибка создания бота для {symbol}: {e}")
                else:
                    logger.error(f" ❌ Ошибка создания бота для {symbol}: {e}")
        
        if created_bots > 0:
            logger.info(f" ✅ Создано {created_bots} новых ботов")
        
    except Exception as e:
        logger.error(f" ❌ Ошибка обработки сигналов: {e}")

def process_trading_signals_for_all_bots(exchange_obj=None):
    """Обрабатывает торговые сигналы для всех активных ботов с новым классом"""
    try:
        logger.info("🔄 Начинаем обработку торговых сигналов для всех активных ботов...")
        
        # Проверяем, инициализирована ли система
        if not system_initialized:
            logger.warning("⏳ Система еще не инициализирована - пропускаем обработку")
            return
        
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
        # Фильтруем только активных ботов (исключаем IDLE и PAUSED)
        active_bots = {symbol: bot for symbol, bot in bots_data['bots'].items() 
                      if bot['status'] not in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']]}
        
        if not active_bots:
            logger.info("⏳ Нет активных ботов для обработки")
            return
        
        logger.debug(f"🔍 Обрабатываем {len(active_bots)} активных ботов: {list(active_bots.keys())}")
        
        for symbol, bot_data in active_bots.items():
            try:
                logger.info(f"🔍 Обрабатываем бота {symbol} (статус: {bot_data.get('status')}, позиция: {bot_data.get('position_side')})...")
                
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
                    logger.warning(f"❌ {symbol}: RSI данные не найдены - пропускаем проверку")
                    continue
                
                current_rsi = rsi_data.get('rsi6h')
                current_trend = rsi_data.get('trend6h')
                logger.info(f"✅ {symbol}: RSI={current_rsi}, Trend={current_trend}, Проверяем условия закрытия...")
                
                # Обрабатываем торговые сигналы через метод update
                external_signal = rsi_data.get('signal')
                external_trend = rsi_data.get('trend6h')
                
                signal_result = trading_bot.update(
                    force_analysis=True, 
                    external_signal=external_signal, 
                    external_trend=external_trend
                )
                
                logger.debug(f"🔄 {symbol}: Результат update: {signal_result}")
                
                # Обновляем данные бота в хранилище если есть изменения
                if signal_result and signal_result.get('success', False):
                    # ⚡ БЕЗ БЛОКИРОВКИ: присваивание - атомарная операция
                    bots_data['bots'][symbol] = trading_bot.to_dict()
                    
                    # Логируем торговые действия
                    action = signal_result.get('action')
                    if action in ['OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT']:
                        logger.info(f"🎯 {symbol}: {action} выполнено")
                else:
                    logger.debug(f"⏳ {symbol}: Нет торговых сигналов")
        
            except Exception as e:
                logger.error(f"❌ Ошибка обработки сигналов для {symbol}: {e}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки торговых сигналов: {str(e)}")

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
            return False
        
        # Дубль-проверка ExitScam
        if not check_exit_scam_filter(symbol, coin_data):
            logger.warning(f" {symbol}: ❌ БЛОКИРОВКА: Обнаружены резкие движения цены (ExitScam)")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f" {symbol}: Ошибка проверки фильтров: {e}")
        return False

def analyze_trends_for_signal_coins():
    """🎯 Определяет тренд для монет с сигналами (RSI ≤29 или ≥71)"""
    try:
        from bots_modules.imports_and_globals import rsi_data_lock, coins_rsi_data, get_exchange, get_auto_bot_config
        
        # Проверяем флаг trend_detection_enabled
        config = get_auto_bot_config()
        trend_detection_enabled = config.get('trend_detection_enabled', True)
        
        if not trend_detection_enabled:
            logger.info(" ⏸️ Анализ трендов отключен (trend_detection_enabled=False)")
            return False
        
        logger.info(" 🎯 Начинаем анализ трендов для сигнальных монет...")
        from bots_modules.calculations import analyze_trend_6h
        
        exchange = get_exchange()
        if not exchange:
            logger.error(" ❌ Биржа не инициализирована")
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
        
        logger.info(f" 📊 Найдено {len(signal_coins)} сигнальных монет для анализа тренда")
        
        if not signal_coins:
            logger.warning(" ⚠️ Нет сигнальных монет для анализа тренда")
            return False
        
        # Анализируем тренд для каждой сигнальной монеты
        analyzed_count = 0
        failed_count = 0
        
        for i, symbol in enumerate(signal_coins, 1):
            try:
                # Анализируем тренд
                trend_analysis = analyze_trend_6h(symbol, exchange_obj=exchange)
                
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
                            new_signal = 'WAIT'  # Оставляем WAIT
                        else:
                            new_signal = _recalculate_signal_with_trend(rsi, new_trend, symbol)
                        
                        # Сохраняем обновления во временном хранилище
                        temp_updates[symbol] = {
                            'trend6h': new_trend,
                            'trend_analysis': trend_analysis,
                            'signal': new_signal,
                            'old_signal': old_signal
                        }
                    
                    analyzed_count += 1
                else:
                    failed_count += 1
                
                # Выводим прогресс каждые 5 монет
                if i % 5 == 0 or i == len(signal_coins):
                    logger.info(f" 📊 Прогресс: {i}/{len(signal_coins)} ({i*100//len(signal_coins)}%)")
                
                # Небольшая пауза между запросами
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f" ❌ {symbol}: {e}")
                failed_count += 1
        
        # ✅ АТОМАРНО применяем ВСЕ обновления одним махом!
        for symbol, updates in temp_updates.items():
            coins_rsi_data['coins'][symbol]['trend6h'] = updates['trend6h']
            coins_rsi_data['coins'][symbol]['trend_analysis'] = updates['trend_analysis']
            coins_rsi_data['coins'][symbol]['signal'] = updates['signal']
        
        logger.info(f" ✅ {analyzed_count} проанализировано | {len(temp_updates)} обновлений")
        
        return True
        
    except Exception as e:
        logger.error(f" ❌ Ошибка анализа трендов: {e}")
        return False

def process_long_short_coins_with_filters():
    """🔍 Обрабатывает лонг/шорт монеты всеми фильтрами"""
    try:
        logger.info(" 🔍 Начинаем обработку лонг/шорт монет фильтрами...")
        
        from bots_modules.imports_and_globals import rsi_data_lock, coins_rsi_data
        
        # Находим монеты с сигналами лонг/шорт
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
        long_short_coins = []
        for symbol, coin_data in coins_rsi_data['coins'].items():
            signal = coin_data.get('signal', 'WAIT')
            if signal in ['ENTER_LONG', 'ENTER_SHORT']:
                long_short_coins.append(symbol)
        
        logger.info(f" 📊 Найдено {len(long_short_coins)} лонг/шорт монет для обработки")
        
        if not long_short_coins:
            logger.warning(" ⚠️ Нет лонг/шорт монет для обработки")
            return []
        
        # Обрабатываем каждую монету всеми фильтрами
        filtered_coins = []
        blocked_count = 0
        
        for i, symbol in enumerate(long_short_coins, 1):
            try:
                # Получаем данные монеты
                # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
                coin_data = coins_rsi_data['coins'].get(symbol, {})
                
                if not coin_data:
                    logger.warning(f" ⚠️ {symbol}: Нет данных")
                    blocked_count += 1
                    continue
                
                # Применяем все фильтры
                signal = coin_data.get('signal', 'WAIT')
                passes_filters = check_new_autobot_filters(symbol, signal, coin_data)
                
                if passes_filters:
                    filtered_coins.append(symbol)
                else:
                    blocked_count += 1
                
            except Exception as e:
                logger.error(f" ❌ {symbol}: Ошибка обработки фильтрами: {e}")
                blocked_count += 1
        
        logger.info(f" ✅ Обработка фильтрами завершена:")
        logger.info(f" 📊 Прошли фильтры: {len(filtered_coins)}")
        logger.info(f" 📊 Заблокированы: {blocked_count}")
        logger.info(f" 📊 Всего обработано: {len(filtered_coins) + blocked_count}")
        
        return filtered_coins
        
    except Exception as e:
        logger.error(f" ❌ Ошибка обработки фильтрами: {e}")
        return []

def set_filtered_coins_for_autobot(filtered_coins):
    """✅ Передает отфильтрованные монеты автоботу и СРАЗУ запускает проверку сигналов"""
    try:
        logger.info(f" ✅ Передаем {len(filtered_coins)} отфильтрованных монет автоботу...")
        
        from bots_modules.imports_and_globals import bots_data_lock, bots_data
        
        # Сохраняем отфильтрованные монеты в конфиг автобота
        # ⚡ БЕЗ БЛОКИРОВКИ: присваивание - атомарная операция
        if 'auto_bot_config' not in bots_data:
            bots_data['auto_bot_config'] = {}
        
        bots_data['auto_bot_config']['filtered_coins'] = filtered_coins
        bots_data['auto_bot_config']['last_filter_update'] = datetime.now().isoformat()
        
        logger.info(f" ✅ Отфильтрованные монеты сохранены в конфиг автобота")
        logger.info(f" 📊 Монеты для автобота: {', '.join(filtered_coins[:10])}{'...' if len(filtered_coins) > 10 else ''}")
        
        # ✅ КРИТИЧНО: СРАЗУ проверяем сигналы и создаем ботов без задержки!
        # Не ждем следующего цикла воркера (180 секунд) - обрабатываем немедленно!
        if filtered_coins and bots_data.get('auto_bot_config', {}).get('enabled', False):
            logger.info(f" 🚀 Немедленно проверяем сигналы для {len(filtered_coins)} монет...")
            try:
                from bots_modules.imports_and_globals import get_exchange
                exchange_obj = get_exchange()
                process_auto_bot_signals(exchange_obj=exchange_obj)
            except Exception as e:
                logger.error(f" ❌ Ошибка немедленной проверки сигналов: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f" ❌ Ошибка передачи монет автоботу: {e}")
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
            logger.warning(f"{symbol}: Биржа не инициализирована")
            return False
        
        chart_response = exch.get_chart_data(symbol, '6h', '30d')
        if not chart_response or not chart_response.get('success'):
            logger.warning(f"{symbol}: Не удалось получить свечи")
            return False
        
        candles = chart_response.get('data', {}).get('candles', [])
        if not candles:
            logger.warning(f"{symbol}: Нет свечей")
            return False
        
        maturity_result = check_coin_maturity_with_storage(symbol, candles)
        return maturity_result['is_mature']
        
    except Exception as e:
        logger.error(f"{symbol}: Ошибка проверки зрелости: {e}")
        return False

def update_is_mature_flags_in_rsi_data():
    """Обновляет флаги is_mature в кэшированных данных RSI на основе хранилища зрелых монет"""
    try:
        from bots_modules.imports_and_globals import is_coin_mature_stored
        
        updated_count = 0
        total_count = len(coins_rsi_data['coins'])
        
        # Обновляем флаги is_mature для всех монет в RSI данных
        for symbol, coin_data in coins_rsi_data['coins'].items():
            # Обновляем флаг is_mature на основе хранилища
            old_status = coin_data.get('is_mature', False)
            coin_data['is_mature'] = is_coin_mature_stored(symbol)
            
            # Подсчитываем обновленные
            if coin_data['is_mature']:
                updated_count += 1
        
        logger.info(f"✅ Обновлено флагов: {updated_count} зрелых из {total_count} монет")
        
    except Exception as e:
        logger.error(f"❌ Ошибка обновления флагов: {e}")

def _legacy_check_exit_scam_filter(symbol, coin_data, individual_settings=None):
    """
    EXIT SCAM ФИЛЬТР + AI ANOMALY DETECTION
    
    Защита от резких движений цены (памп/дамп/скам):
    1. Одна свеча превысила максимальный % изменения
    2. N свечей суммарно превысили максимальный % изменения
    3. ИИ обнаружил аномалию (если включен)
    
    Args:
        symbol: Символ монеты
        coin_data: Данные монеты
        individual_settings: Индивидуальные настройки монеты (опционально)
    """
    try:
        # ✅ Получаем настройки: сначала индивидуальные, затем глобальные
        # ⚡ БЕЗ БЛОКИРОВКИ: конфиг не меняется, GIL делает чтение атомарным
        if individual_settings is None:
            individual_settings = get_individual_coin_settings(symbol)
        
        auto_config = bots_data.get('auto_bot_config', {})
        
        exit_scam_enabled = individual_settings.get('exit_scam_enabled') if individual_settings else None
        if exit_scam_enabled is None:
            exit_scam_enabled = auto_config.get('exit_scam_enabled', True)
        
        exit_scam_candles = individual_settings.get('exit_scam_candles') if individual_settings else None
        if exit_scam_candles is None:
            exit_scam_candles = auto_config.get('exit_scam_candles', 10)
        
        single_candle_percent = individual_settings.get('exit_scam_single_candle_percent') if individual_settings else None
        if single_candle_percent is None:
            single_candle_percent = auto_config.get('exit_scam_single_candle_percent', 15.0)
        
        multi_candle_count = individual_settings.get('exit_scam_multi_candle_count') if individual_settings else None
        if multi_candle_count is None:
            multi_candle_count = auto_config.get('exit_scam_multi_candle_count', 4)
        
        multi_candle_percent = individual_settings.get('exit_scam_multi_candle_percent') if individual_settings else None
        if multi_candle_percent is None:
            multi_candle_percent = auto_config.get('exit_scam_multi_candle_percent', 50.0)
        
        # Если фильтр отключен - разрешаем
        if not exit_scam_enabled:
            logger.debug(f"{symbol}: Фильтр отключен")
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
        
        logger.debug(f"{symbol}: Анализ последних {exit_scam_candles} свечей")
        
        # 1. ПРОВЕРКА: Одна свеча превысила максимальный % изменения
        for i, candle in enumerate(recent_candles):
            open_price = candle['open']
            close_price = candle['close']
            
            # Процент изменения свечи (от открытия до закрытия)
            price_change = abs((close_price - open_price) / open_price) * 100
            
            if price_change > single_candle_percent:
                logger.debug(f"{symbol}: Свеча: O={open_price:.4f} C={close_price:.4f} H={candle['high']:.4f} L={candle['low']:.4f}")
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
                logger.warning(f"{symbol}: ❌ БЛОКИРОВКА: {multi_candle_count} свечей превысили суммарный лимит {multi_candle_percent}% (было {total_change:.1f}%)")
                logger.debug(f"{symbol}: Первая свеча: {first_open:.4f}, Последняя свеча: {last_close:.4f}")
                return False
        
        logger.debug(f"{symbol}: ✅ Базовые проверки пройдены")
        
        # 3. ПРОВЕРКА: AI Anomaly Detection (если включен)
        ai_check_enabled = True  # Включаем обратно - проблема была не в AI!
        
        if ai_check_enabled:
            try:
                from bot_engine.bot_config import AIConfig
                
                # Быстрая проверка: AI включен и Anomaly Detection включен
                if AIConfig.AI_ENABLED and AIConfig.AI_ANOMALY_DETECTION_ENABLED:
                    try:
                        # ✅ Используем закэшированный AI Manager
                        ai_manager, ai_available = get_cached_ai_manager()
                        
                        # Быстрая проверка доступности: если AI недоступен, пропускаем
                        if not ai_available or not ai_manager:
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
                                    return False
                                else:
                                    logger.warning(
                                        f"{symbol}: ⚠️ ПРЕДУПРЕЖДЕНИЕ (AI): "
                                        f"Аномалия {anomaly_type} "
                                        f"(severity: {severity:.2%} - ниже порога {AIConfig.AI_ANOMALY_BLOCK_THRESHOLD:.2%})"
                                    )
                            else:
                                logger.debug(f"{symbol}: ✅ AI: Аномалий не обнаружено")
                    
                    except ImportError as e:
                        logger.debug(f"{symbol}: AI модуль не доступен: {e}")
                    except Exception as e:
                        logger.error(f"{symbol}: Ошибка AI проверки: {e}")
        
            except ImportError:
                pass  # AIConfig не доступен - пропускаем AI проверку
        
                logger.debug(f"{symbol}: ✅ ПРОЙДЕН")
        return True
        
    except Exception as e:
        logger.error(f"{symbol}: Ошибка проверки: {e}")
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
            # ✅ Используем закэшированный AI Manager
            ai_manager, ai_available = get_cached_ai_manager()
            
            # Проверяем доступность LSTM
            if not ai_available or not ai_manager or not ai_manager.lstm_predictor:
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
                    logger.warning(f"{symbol}: ⏱️ LSTM prediction таймаут (5с)")
                    prediction = None  # Пропускаем AI проверку при таймауте
            
            if prediction and prediction.get('confidence', 0) >= AIConfig.AI_LSTM_MIN_CONFIDENCE:
                # Проверяем совпадение направлений
                lstm_direction = "LONG" if prediction['direction'] > 0 else "SHORT"
                confidence = prediction['confidence']
                
                if lstm_direction == signal:
                    logger.info(
                        f"{symbol}: ✅ ПОДТВЕРЖДЕНИЕ: "
                        f"LSTM предсказывает {lstm_direction} "
                        f"(изменение: {prediction['change_percent']:+.2f}%, "
                        f"уверенность: {confidence:.1f}%)"
                    )
                else:
                    logger.warning(
                        f"{symbol}: ⚠️ ПРОТИВОРЕЧИЕ: "
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
            logger.debug(f"{symbol}: AI модуль не доступен: {e}")
            return None
        except Exception as e:
            logger.error(f"{symbol}: Ошибка LSTM предсказания: {e}")
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
            # ✅ Используем закэшированный AI Manager
            ai_manager, ai_available = get_cached_ai_manager()
            
            # Проверяем доступность Pattern Detector
            if not ai_available or not ai_manager or not ai_manager.pattern_detector:
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
                    logger.warning(f"{symbol}: ⏱️ Pattern detection таймаут (5с)")
                    pattern_signal = {'patterns_found': 0, 'confirmation': False}  # Пропускаем при таймауте
            
            if pattern_signal['patterns_found'] > 0:
                # Проверяем подтверждение
                if pattern_signal['confirmation']:
                    logger.info(
                        f"{symbol}: ✅ ПОДТВЕРЖДЕНИЕ: "
                        f"Паттерны подтверждают {signal} "
                        f"(найдено: {pattern_signal['patterns_found']}, "
                        f"уверенность: {pattern_signal['confidence']:.1f}%)"
                    )
                    
                    if pattern_signal['strongest_pattern']:
                        strongest = pattern_signal['strongest_pattern']
                        logger.info(
                            f"{symbol}:    └─ {strongest['name']}: "
                            f"{strongest['description']}"
                        )
                else:
                    logger.warning(
                        f"{symbol}: ⚠️ ПРОТИВОРЕЧИЕ: "
                        f"Сигнал {signal}, но паттерны указывают на {pattern_signal['signal']} "
                        f"(уверенность: {pattern_signal['confidence']:.1f}%)"
                    )
                
                return pattern_signal
            
            return None
            
        except ImportError as e:
            logger.debug(f"{symbol}: AI модуль не доступен: {e}")
            return None
        except Exception as e:
            logger.error(f"{symbol}: Ошибка анализа паттернов: {e}")
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
                    logger.debug(f"{symbol}: Уже есть позиция {existing_side}")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"{symbol}: Ошибка проверки позиций: {e}")
        return False

def create_new_bot(symbol, config=None, exchange_obj=None):
    """Создает нового бота"""
    try:
        # Локальный импорт для избежания циклического импорта
        from bots_modules.bot_class import NewTradingBot
        from bots_modules.imports_and_globals import get_exchange
        exchange_to_use = exchange_obj if exchange_obj else get_exchange()
        
        # Получаем настройки размера позиции из конфига
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
        auto_bot_config = bots_data['auto_bot_config']
        default_volume = auto_bot_config.get('default_position_size')
        default_volume_mode = auto_bot_config.get('default_position_mode', 'usdt')
        
        # Создаем конфигурацию бота
        bot_config = {
            'symbol': symbol,
            'status': BOT_STATUS['RUNNING'],  # ✅ ИСПРАВЛЕНО: бот должен быть активным
            'created_at': datetime.now().isoformat(),
            'opened_by_autobot': True,
            'volume_mode': default_volume_mode,
            'volume_value': default_volume,  # ✅ ИСПРАВЛЕНО: используем значение из конфига
            'leverage': auto_bot_config.get('leverage', 1)  # ✅ Добавляем leverage из глобального конфига
        }

        individual_settings = get_individual_coin_settings(symbol)
        if individual_settings:
            bot_config.update(individual_settings)

        # Гарантируем обязательные поля
        bot_config['symbol'] = symbol
        bot_config['status'] = BOT_STATUS['RUNNING']
        bot_config.setdefault('volume_mode', default_volume_mode)
        if bot_config.get('volume_value') is None:
            bot_config['volume_value'] = default_volume
        if bot_config.get('leverage') is None:
            bot_config['leverage'] = auto_bot_config.get('leverage', 1)  # ✅ Fallback для leverage
        
        # Создаем бота
        new_bot = NewTradingBot(symbol, bot_config, exchange_to_use)
        
        # Сохраняем в bots_data
        # ⚡ БЕЗ БЛОКИРОВКИ: присваивание - атомарная операция
        bots_data['bots'][symbol] = new_bot.to_dict()
        
        logger.info(f"✅ Бот для {symbol} создан успешно")
        return new_bot
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания бота для {symbol}: {e}")
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
        
        logger.info(f"🔍 Тестируем ExitScam фильтр для {symbol}")
        logger.info(f"⚙️ Настройки:")
        logger.info(f"⚙️ - Включен: {exit_scam_enabled}")
        logger.info(f"⚙️ - Анализ свечей: {exit_scam_candles}")
        logger.info(f"⚙️ - Лимит одной свечи: {single_candle_percent}%")
        logger.info(f"⚙️ - Лимит {multi_candle_count} свечей: {multi_candle_percent}%")
        
        if not exit_scam_enabled:
            logger.info(f"{symbol}: ⚠️ Фильтр ОТКЛЮЧЕН в конфиге")
            return
        
        # Получаем свечи
        exch = get_exchange()
        if not exch:
            logger.error(f"{symbol}: Биржа не инициализирована")
            return
        
        chart_response = exch.get_chart_data(symbol, '6h', '30d')
        if not chart_response or not chart_response.get('success'):
            logger.error(f"{symbol}: Не удалось получить свечи")
            return
        
        candles = chart_response.get('data', {}).get('candles', [])
        if len(candles) < exit_scam_candles:
            logger.error(f"{symbol}: Недостаточно свечей ({len(candles)})")
            return
        
        # Анализируем последние N свечей (из конфига)
        recent_candles = candles[-exit_scam_candles:]
        
        # Показываем детали каждой свечи
        for i, candle in enumerate(recent_candles):
            open_price = candle['open']
            close_price = candle['close']
            high_price = candle['high']
            low_price = candle['low']
            
            price_change = ((close_price - open_price) / open_price) * 100
            candle_range = ((high_price - low_price) / open_price) * 100
            
            logger.info(f"{symbol}: Свеча {i+1}: O={open_price:.4f} C={close_price:.4f} H={high_price:.4f} L={low_price:.4f} | Изменение: {price_change:+.1f}% | Диапазон: {candle_range:.1f}%")
        
        # Тестируем фильтр с детальным логированием
        logger.info(f"{symbol}: 🔍 Запускаем проверку ExitScam фильтра...")
        result = check_exit_scam_filter(symbol, {})
        
        if result:
            logger.info(f"{symbol}: ✅ РЕЗУЛЬТАТ: ПРОЙДЕН")
        else:
            logger.warning(f"{symbol}: ❌ РЕЗУЛЬТАТ: ЗАБЛОКИРОВАН")
        
        # Дополнительный анализ
        logger.info(f"{symbol}: 📊 Дополнительный анализ:")
        
        # 1. Проверка отдельных свечей
        extreme_single_count = 0
        for i, candle in enumerate(recent_candles):
            open_price = candle['open']
            close_price = candle['close']
            
            price_change = abs((close_price - open_price) / open_price) * 100
            
            if price_change > single_candle_percent:
                extreme_single_count += 1
                logger.warning(f"{symbol}: ❌ Превышение лимита одной свечи #{i+1}: {price_change:.1f}% > {single_candle_percent}%")
        
        # 2. Проверка суммарного изменения за N свечей
        if len(recent_candles) >= multi_candle_count:
            multi_candles = recent_candles[-multi_candle_count:]
            first_open = multi_candles[0]['open']
            last_close = multi_candles[-1]['close']
            
            total_change = abs((last_close - first_open) / first_open) * 100
            
            logger.info(f"{symbol}: 📈 {multi_candle_count}-свечечный анализ: {total_change:.1f}% (порог: {multi_candle_percent}%)")
            
            if total_change > multi_candle_percent:
                logger.warning(f"{symbol}: ❌ Превышение суммарного лимита: {total_change:.1f}% > {multi_candle_percent}%")
        
    except Exception as e:
        logger.error(f"{symbol}: Ошибка тестирования: {e}")

# Алиас для обратной совместимости
test_anti_pump_filter = test_exit_scam_filter

def test_rsi_time_filter(symbol):
    """Тестирует RSI временной фильтр для конкретной монеты"""
    try:
        logger.info(f"🔍 Тестируем RSI временной фильтр для {symbol}")
        
        # Получаем свечи
        exch = get_exchange()
        if not exch:
            logger.error(f"{symbol}: Биржа не инициализирована")
            return
                
        chart_response = exch.get_chart_data(symbol, '6h', '30d')
        if not chart_response or not chart_response.get('success'):
            logger.error(f"{symbol}: Не удалось получить свечи")
            return
        
        candles = chart_response.get('data', {}).get('candles', [])
        if len(candles) < 50:
            logger.error(f"{symbol}: Недостаточно свечей ({len(candles)})")
            return
        
        # Получаем текущий RSI
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
        coin_data = coins_rsi_data['coins'].get(symbol)
        if not coin_data:
            logger.error(f"{symbol}: Нет RSI данных")
            return
        
        current_rsi = coin_data.get('rsi6h', 0)
        signal = coin_data.get('signal', 'WAIT')
        
        # ✅ Определяем ОРИГИНАЛЬНЫЙ сигнал на основе только RSI с учетом индивидуальных настроек
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
        individual_settings = get_individual_coin_settings(symbol)
        
        rsi_long_threshold = individual_settings.get('rsi_long_threshold') if individual_settings else None
        if rsi_long_threshold is None:
            rsi_long_threshold = bots_data.get('auto_bot_config', {}).get('rsi_long_threshold', 29)
        
        rsi_short_threshold = individual_settings.get('rsi_short_threshold') if individual_settings else None
        if rsi_short_threshold is None:
            rsi_short_threshold = bots_data.get('auto_bot_config', {}).get('rsi_short_threshold', 71)
        
        original_signal = 'WAIT'
        if current_rsi <= rsi_long_threshold:
            original_signal = 'ENTER_LONG'
        elif current_rsi >= rsi_short_threshold:
            original_signal = 'ENTER_SHORT'
        
        logger.info(f"{symbol}: Текущий RSI={current_rsi:.1f}, Оригинальный сигнал={original_signal}, Финальный сигнал={signal}")
        if individual_settings:
            logger.info(f"{symbol}: Используются индивидуальные настройки: rsi_long={rsi_long_threshold}, rsi_short={rsi_short_threshold}")
        
        # Тестируем временной фильтр с ОРИГИНАЛЬНЫМ сигналом и индивидуальными настройками
        time_filter_result = check_rsi_time_filter(candles, current_rsi, original_signal, symbol=symbol, individual_settings=individual_settings)
        
        logger.info(f"{symbol}: Результат временного фильтра:")
        logger.info(f"{symbol}: Разрешено: {time_filter_result['allowed']}")
        logger.info(f"{symbol}: Причина: {time_filter_result['reason']}")
        if 'calm_candles' in time_filter_result and time_filter_result['calm_candles'] is not None:
            logger.info(f"{symbol}: Спокойных свечей: {time_filter_result['calm_candles']}")
        if 'last_extreme_candles_ago' in time_filter_result and time_filter_result['last_extreme_candles_ago'] is not None:
            logger.info(f"{symbol}: Последний экстремум: {time_filter_result['last_extreme_candles_ago']} свечей назад")
        
        # Показываем историю RSI для анализа
        closes = [candle['close'] for candle in candles]
        rsi_history = calculate_rsi_history(closes, 14)
        
        if rsi_history:
            logger.info(f"{symbol}: Последние 20 значений RSI:")
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
                logger.info(f"{symbol}: Свеча -{index_from_end}: RSI={rsi_val:.1f} {marker_str}")
        
    except Exception as e:
        logger.error(f"{symbol}: Ошибка тестирования: {e}")

