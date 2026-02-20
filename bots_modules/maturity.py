"""Функции проверки зрелости монет

Включает:
- load_mature_coins_storage - загрузка хранилища зрелых монет
- save_mature_coins_storage - сохранение хранилища
- is_coin_mature_stored - проверка наличия в хранилище
- add_mature_coin_to_storage - добавление в хранилище
- remove_mature_coin_from_storage - удаление из хранилища
- update_mature_coin_verification - обновление времени проверки
- check_coin_maturity_with_storage - проверка зрелости с хранилищем
- check_coin_maturity - проверка зрелости
"""

import os
import json
import time
import threading
import logging
from datetime import datetime

logger = logging.getLogger('BotsService')

# Импорт глобальных переменных из imports_and_globals
try:
    from bots_modules.imports_and_globals import (
        bots_data_lock, bots_data,
        MIN_CANDLES_FOR_MATURITY, MIN_RSI_LOW, MAX_RSI_HIGH
    )
except ImportError:
    bots_data_lock = threading.Lock()
    bots_data = {}
    try:
        from bot_engine.config_loader import DEFAULT_AUTO_BOT_CONFIG
        MIN_CANDLES_FOR_MATURITY = DEFAULT_AUTO_BOT_CONFIG.get('min_candles_for_maturity')
        MIN_RSI_LOW = DEFAULT_AUTO_BOT_CONFIG.get('min_rsi_low')
        MAX_RSI_HIGH = DEFAULT_AUTO_BOT_CONFIG.get('max_rsi_high')
    except Exception:
        MIN_CANDLES_FOR_MATURITY = None
        MIN_RSI_LOW = None
        MAX_RSI_HIGH = None

# Импорт calculate_rsi_history из calculations
try:
    from bots_modules.calculations import calculate_rsi_history
except ImportError:
    def calculate_rsi_history(prices, period=14):
        return None

# Глобальные переменные (будут импортированы из главного файла)
mature_coins_storage = {}
MATURE_COINS_FILE = 'data/mature_coins.json'
MATURITY_CHECK_CACHE_FILE = 'data/maturity_check_cache.json'  # 🚀 Кэш последней проверки
mature_coins_lock = threading.Lock()

def get_maturity_timeframe():
    """Таймфрейм для проверки зрелости = системный (из конфига). 400 свечей на 5m = ~33ч, на 1m = ~7ч."""
    try:
        from bot_engine.config_loader import get_current_timeframe, TIMEFRAME
        return get_current_timeframe() or TIMEFRAME or '5m'
    except Exception:
        return '5m'

# 🚀 Кэш последней проверки зрелости (загружается из файла)
last_maturity_check = {'coins_count': 0, 'config_hash': None}
maturity_data_invalidated = False  # Флаг: True если данные были сброшены и не должны сохраняться

def load_maturity_check_cache():
    """🚀 Загружает кэш последней проверки зрелости из БД (с fallback на JSON)"""
    global last_maturity_check
    try:
        from bot_engine.storage import load_maturity_check_cache as storage_load_cache
        cached_data = storage_load_cache()
        if cached_data:
            last_maturity_check['coins_count'] = cached_data.get('coins_count', 0)
            last_maturity_check['config_hash'] = cached_data.get('config_hash', None)
            logger.info(f" 💾 Загружен кэш: {last_maturity_check['coins_count']} монет")
        else:
            logger.info(" 📝 Кэш не найден, создаем новый")
            last_maturity_check = {'coins_count': 0, 'config_hash': None}
    except Exception as e:
        logger.error(f" ❌ Ошибка загрузки кэша: {e}")
        last_maturity_check = {'coins_count': 0, 'config_hash': None}

def save_maturity_check_cache():
    """🚀 Сохраняет кэш последней проверки зрелости в БД (с fallback на JSON)"""
    global last_maturity_check
    try:
        from bot_engine.storage import save_maturity_check_cache as storage_save_cache
        storage_save_cache(
            last_maturity_check.get('coins_count', 0),
            last_maturity_check.get('config_hash')
        )
    except Exception as e:
        logger.error(f" ❌ Ошибка сохранения кэша: {e}")

def load_mature_coins_storage(expected_coins_count=None):
    """Загружает постоянное хранилище зрелых монет из БД"""
    global mature_coins_storage, maturity_data_invalidated
    try:
        # Загружаем из БД
        from bot_engine.storage import load_mature_coins as storage_load_mature
        loaded_data = storage_load_mature()
        
        # ✅ ПРОВЕРКА КОНФИГУРАЦИИ: Сравниваем настройки из БД с текущими
        need_recalculation = False
        if loaded_data:
            # 🎯 ПРОВЕРКА 1: Количество монет
            if expected_coins_count is not None and len(loaded_data) != expected_coins_count:
                logger.warning(f" 🔄 Количество монет изменилось: БД={len(loaded_data)}, биржа={expected_coins_count}")
                need_recalculation = True
            
            # Берем первую монету для проверки настроек
            first_coin = list(loaded_data.values())[0]
            if 'maturity_data' in first_coin and 'details' in first_coin['maturity_data']:
                db_min_required = first_coin['maturity_data']['details'].get('min_required')
                
                # Получаем текущие настройки
                from bots_modules.imports_and_globals import bots_data, bots_data_lock
                with bots_data_lock:
                    config = bots_data.get('auto_bot_config', {})
                
                current_min_candles = config.get('min_candles_for_maturity', MIN_CANDLES_FOR_MATURITY)
                current_min_rsi_low = config.get('min_rsi_low', MIN_RSI_LOW)
                current_max_rsi_high = config.get('max_rsi_high', MAX_RSI_HIGH)
                
                # Зрелость считается по текущему системному ТФ; при смене ТФ — пересчёт
                current_tf = get_maturity_timeframe()
                db_timeframe = first_coin['maturity_data']['details'].get('timeframe')
                if db_timeframe != current_tf:
                    logger.warning(f" ⚠️ В БД зрелость по ТФ {db_timeframe}, текущий ТФ {current_tf} — пересчитываем")
                    need_recalculation = True
                    from bot_engine.storage import save_mature_coins as storage_save_mature
                    storage_save_mature({})
                    loaded_data = {}
                    maturity_data_invalidated = True
                # Проверяем, изменились ли настройки
                elif (db_min_required != current_min_candles or
                    first_coin['maturity_data']['details'].get('config_min_rsi_low') != current_min_rsi_low or
                    first_coin['maturity_data']['details'].get('config_max_rsi_high') != current_max_rsi_high):
                    
                    logger.warning(f" ⚠️ Настройки зрелости изменились!")
                    logger.warning(f" БД: min_candles={db_min_required}, min_rsi={first_coin['maturity_data']['details'].get('config_min_rsi_low')}, max_rsi={first_coin['maturity_data']['details'].get('config_max_rsi_high')}")
                    logger.warning(f" Текущие: min_candles={current_min_candles}, min_rsi={current_min_rsi_low}, max_rsi={current_max_rsi_high}")
                    logger.warning(f" 🔄 Пересчитываем данные зрелости...")
                    
                    need_recalculation = True
                    
                    # Очищаем БД
                    if loaded_data:
                        from bot_engine.storage import save_mature_coins as storage_save_mature
                        storage_save_mature({})
                    
                    loaded_data = {}
                    
                    # ✅ УСТАНАВЛИВАЕМ ФЛАГ: данные недействительны и не должны сохраняться
                    maturity_data_invalidated = True
                    logger.warning(f" 🚫 Данные зрелости сброшены - сохранение ЗАПРЕЩЕНО до пересчета")
        
        # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Изменяем словарь in-place, а не переприсваиваем
        # Это важно, т.к. mature_coins_storage импортируется в другие модули
        with mature_coins_lock:
            mature_coins_storage.clear()
            mature_coins_storage.update(loaded_data if loaded_data else {})
        
        # ✅ ДОПОЛНИТЕЛЬНОЕ ИСПРАВЛЕНИЕ: Обновляем глобальную переменную в imports_and_globals
        try:
            import bots_modules.imports_and_globals as ig_module
            if hasattr(ig_module, 'mature_coins_storage'):
                with ig_module.mature_coins_lock:
                    ig_module.mature_coins_storage.clear()
                    ig_module.mature_coins_storage.update(loaded_data if loaded_data else {})
        except Exception as sync_error:
            logger.warning(f" ⚠️ Не удалось синхронизировать с imports_and_globals: {sync_error}")
        
        if need_recalculation:
            logger.info(f" 🔄 Данные будут пересчитаны при следующей проверке зрелости")
        elif loaded_data:
            logger.info(f" ✅ Загружено {len(mature_coins_storage)} зрелых монет из БД")
        else:
            logger.info(" 📝 БД хранилища пуста, создаем новое")
    except Exception as e:
        logger.error(f" ❌ Ошибка загрузки хранилища: {e}")
        with mature_coins_lock:
            mature_coins_storage.clear()

def save_mature_coins_storage():
    """Сохраняет постоянное хранилище зрелых монет в БД (с fallback на JSON)"""
    global maturity_data_invalidated
    
    # ✅ ПРОВЕРКА: Если данные были сброшены, не сохраняем их
    if maturity_data_invalidated:
        logger.warning(f" 🚫 Сохранение пропущено - данные недействительны (ждем пересчета)")
        return False
    
    try:
        with mature_coins_lock:
            # Создаем копию для безопасной сериализации
            storage_copy = mature_coins_storage.copy()
        
        # ПРИОРИТЕТ: Сохраняем в БД
        from bot_engine.storage import save_mature_coins as storage_save_mature
        if storage_save_mature(storage_copy):
            return True  # Успешно сохранили в БД
        
        # FALLBACK: Сохраняем в JSON (для обратной совместимости)
        os.makedirs(os.path.dirname(MATURE_COINS_FILE), exist_ok=True)
        from bot_engine.storage import save_json_file
        save_json_file(MATURE_COINS_FILE, storage_copy)
        return True
    except Exception as e:
        logger.error(f" Ошибка сохранения хранилища: {e}")
        return False

def is_coin_mature_stored(symbol):
    """Проверяет, есть ли монета в постоянном хранилище зрелых монет с актуальными настройками"""
    # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
    if symbol not in mature_coins_storage:
        return False
    
    # ✅ НОВАЯ ЛОГИКА: Сравниваем настройки зрелости
    stored_data = mature_coins_storage[symbol]
    maturity_data = stored_data.get('maturity_data', {})
    stored_details = maturity_data.get('details', {})
    
    # Получаем текущие настройки
    # ⚡ БЕЗ БЛОКИРОВКИ: конфиг не меняется, GIL делает чтение атомарным
    config = bots_data.get('auto_bot_config', {})
    
    current_min_candles = config.get('min_candles_for_maturity', MIN_CANDLES_FOR_MATURITY)
    current_min_rsi_low = config.get('min_rsi_low', MIN_RSI_LOW)
    current_max_rsi_high = config.get('max_rsi_high', MAX_RSI_HIGH)
    
    # Зрелость по текущему системному ТФ; при смене ТФ запись невалидна
    current_tf = get_maturity_timeframe()
    stored_timeframe = stored_details.get('timeframe')
    if stored_timeframe != current_tf:
        pass
        del mature_coins_storage[symbol]
        return False

    # ✅ СРАВНИВАЕМ С СОХРАНЕННЫМИ ПАРАМЕТРАМИ КОНФИГА
    stored_min_candles = stored_details.get('min_required', 0)
    stored_config_min_rsi_low = stored_details.get('config_min_rsi_low', 0)
    stored_config_max_rsi_high = stored_details.get('config_max_rsi_high', 0)
    
    # Если параметры конфига изменились - перепроверяем монету
    if stored_min_candles != current_min_candles:
        pass
        del mature_coins_storage[symbol]
        return False
    
    if stored_config_min_rsi_low != current_min_rsi_low:
        pass
        del mature_coins_storage[symbol]
        return False
    
    if stored_config_max_rsi_high != current_max_rsi_high:
        pass
        del mature_coins_storage[symbol]
        return False
    
    return True

def add_mature_coin_to_storage(symbol, maturity_data, auto_save=True):
    """Добавляет монету в постоянное хранилище зрелых монет (только если её там еще нет)"""
    global mature_coins_storage, maturity_data_invalidated
    
    with mature_coins_lock:
        # Проверяем, есть ли уже монета в хранилище
        if symbol in mature_coins_storage:
            # Монета уже есть - ничего не делаем
            pass
            return
        
        # ✅ СБРАСЫВАЕМ ФЛАГ: Если добавляем первую монету после сброса, данные снова валидны
        if maturity_data_invalidated:
            maturity_data_invalidated = False
            logger.info(f" ✅ Начат пересчет зрелости - сохранение разрешено")
        
        # Добавляем новую монету в хранилище
        mature_coins_storage[symbol] = {
            'timestamp': time.time(),
            'maturity_data': maturity_data
        }
    
    if auto_save:
        save_mature_coins_storage()
        logger.info(f" Монета {symbol} добавлена в постоянное хранилище зрелых монет")
    else:
        pass

def remove_mature_coin_from_storage(symbol):
    """Удаляет монету из постоянного хранилища зрелых монет"""
    global mature_coins_storage
    if symbol in mature_coins_storage:
        del mature_coins_storage[symbol]
        # Отключаем автоматическое сохранение - будет сохранено пакетно
        pass

# ❌ ОТКЛЮЧЕНО: Все функции optimal_ema удалены (EMA фильтр убран из системы)
# def load_optimal_ema_data():
#     """Загружает данные об оптимальных EMA из файла"""
#     pass

# ❌ ОТКЛЮЧЕНО
# def get_optimal_ema_periods(symbol):
#     """Получает оптимальные EMA периоды для монеты (поддержка отдельных LONG и SHORT EMA)"""
#     return {}

# def update_optimal_ema_data(new_data):
#     """Обновляет данные об оптимальных EMA из внешнего источника"""
#     return False

def check_coin_maturity_with_storage(symbol, candles):
    """Проверяет зрелость монеты с использованием постоянного хранилища"""
    # Сначала проверяем постоянное хранилище
    if is_coin_mature_stored(symbol):
        # Убрано избыточное логирование
        return {
            'is_mature': True,
            'details': {'stored': True, 'from_storage': True}
        }
    
    # Если не в хранилище, выполняем полную проверку
    maturity_result = check_coin_maturity(symbol, candles)
    
    # Если монета зрелая, добавляем в постоянное хранилище (с автосохранением)
    if maturity_result['is_mature']:
        add_mature_coin_to_storage(symbol, maturity_result, auto_save=True)
    
    return maturity_result

def check_coin_maturity(symbol, candles):
    """Проверяет зрелость монеты для торговли.
    
    Критерий (БЕЗ лишних проверок): за min_candles свечей (из конфига) RSI хотя бы раз достигал:
    - ≤ min_rsi_low (35) И ≥ max_rsi_high (65).
    Только 3 условия: достаточно свечей + rsi_min ≤ 35 + rsi_max ≥ 65.
    """
    try:
        # Получаем настройки из конфига (min_candles_for_maturity = 400 в конфиге)
        with bots_data_lock:
            config = bots_data.get('auto_bot_config', {})
        
        min_candles = config.get('min_candles_for_maturity') or MIN_CANDLES_FOR_MATURITY or 400
        min_rsi_low = config.get('min_rsi_low') or MIN_RSI_LOW or 35
        max_rsi_high = config.get('max_rsi_high') or MAX_RSI_HIGH or 65
        
        if not candles or len(candles) < min_candles:
            return {
                'is_mature': False,
                'reason': f'Недостаточно свечей: {len(candles) if candles else 0}/{min_candles}',
                'details': {
                    'candles_count': len(candles) if candles else 0,
                    'min_required': min_candles
                }
            }
        
        # Берём ПОСЛЕДНИЕ min_candles свечей (из конфига) — именно за этот период проверяем RSI
        recent_candles = candles[-min_candles:] if len(candles) >= min_candles else candles
        closes = [candle['close'] for candle in recent_candles]
        
        rsi_history = calculate_rsi_history(closes, 14)
        if not rsi_history:
            return {
                'is_mature': False,
                'reason': 'Не удалось рассчитать историю RSI',
                'details': {}
            }
        
        rsi_min = min(rsi_history)
        rsi_max = max(rsi_history)
        
        # Единственные 3 критерия: достаточно свечей, RSI достигал ≤35 и ≥65
        sufficient_candles = len(candles) >= min_candles
        rsi_reached_low = rsi_min <= min_rsi_low
        rsi_reached_high = rsi_max >= max_rsi_high
        is_mature = sufficient_candles and rsi_reached_low and rsi_reached_high
        
        # Детали с текущим таймфреймом (зрелость считается по нему)
        details = {
            'candles_count': len(candles),
            'min_required': min_candles,
            'config_min_rsi_low': min_rsi_low,
            'config_max_rsi_high': max_rsi_high,
            'rsi_min': round(rsi_min, 1),
            'rsi_max': round(rsi_max, 1),
            'timeframe': get_maturity_timeframe(),
        }
        
        # Причина незрелости (только для незрелых монет)
        if not is_mature:
            if not sufficient_candles:
                reason = f'Недостаточно свечей: {len(candles)}/{min_candles}'
            else:
                reason = (f'RSI не достигал ≤{min_rsi_low} и ≥{max_rsi_high} (min={rsi_min:.0f}, max={rsi_max:.0f})')
        else:
            reason = None  # Для зрелых монет reason не нужен
        
        result = {
            'is_mature': is_mature,
            'details': details
        }
        
        # Добавляем reason только для незрелых монет
        if reason:
            result['reason'] = reason
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка проверки зрелости {symbol}: {e}")
        return {
            'is_mature': False,
            'reason': f'Ошибка анализа: {str(e)}',
            'details': {}
        }

def _get_candles_from_cache(candles_cache, symbol, timeframe):
    """Достаёт свечи из candles_cache по символу и таймфрейму (без API)."""
    if not candles_cache or symbol not in candles_cache:
        return None
    symbol_cache = candles_cache[symbol]
    if not isinstance(symbol_cache, dict):
        return None
    # Новая структура: {timeframe: {candles: [...], ...}}
    if timeframe in symbol_cache:
        return symbol_cache[timeframe].get('candles')
    # Старая структура: {candles: [...], timeframe: '1m'}
    if symbol_cache.get('timeframe') == timeframe and 'candles' in symbol_cache:
        return symbol_cache.get('candles')
    return None


def calculate_all_coins_maturity():
    """🧮 Расчёт зрелости ТОЛЬКО по уже загруженным свечам (candles_cache после загрузки RSI).
    API не вызывается — все зрелые монеты заносятся в БД из данных загрузки RSI."""
    try:
        logger.info("🧮 Начинаем расчёт зрелости (только из кэша свечей, без API)...")
        
        from bots_modules.imports_and_globals import coins_rsi_data, bots_data
        
        # Получаем все монеты с RSI данными
        all_coins = []
        for symbol, coin_data in coins_rsi_data['coins'].items():
            from bot_engine.config_loader import get_rsi_from_coin_data
            if get_rsi_from_coin_data(coin_data) is not None:
                all_coins.append(symbol)
        
        logger.info(f"📊 Найдено {len(all_coins)} монет с RSI данными")
        
        global last_maturity_check
        config = bots_data.get('auto_bot_config', {})
        current_config_params = {
            'min_candles': config.get('min_candles_for_maturity', MIN_CANDLES_FOR_MATURITY),
            'min_rsi_low': config.get('min_rsi_low', MIN_RSI_LOW),
            'max_rsi_high': config.get('max_rsi_high', MAX_RSI_HIGH)
        }
        current_config_hash = str(current_config_params)
        current_coins_count = len(all_coins)
        
        if (last_maturity_check['coins_count'] == current_coins_count and
            last_maturity_check['config_hash'] == current_config_hash):
            logger.info(f"⚡ ПРОПУСК: Конфиг и количество монет ({current_coins_count}) не изменились!")
            return True
        
        if not all_coins:
            logger.warning("⚠️ Нет монет для проверки зрелости")
            return False
        
        maturity_tf = get_maturity_timeframe()
        candles_cache = coins_rsi_data.get('candles_cache', {})
        
        coins_to_check = []
        already_mature_count = 0
        for symbol in all_coins:
            if is_coin_mature_stored(symbol):
                already_mature_count += 1
            else:
                coins_to_check.append(symbol)
        
        logger.info(f"🎯 Уже зрелые (БД): {already_mature_count}, проверим по кэшу: {len(coins_to_check)}")
        
        if not coins_to_check:
            logger.info("✅ Все монеты уже зрелые - пересчет не нужен!")
            return True
        
        mature_count = 0
        immature_count = 0
        skipped_no_candles = 0
        
        for i, symbol in enumerate(coins_to_check, 1):
            try:
                if i == 1 or i % 10 == 0 or i == len(coins_to_check):
                    logger.info(f"📊 Прогресс: {i}/{len(coins_to_check)} монет ({round(i/len(coins_to_check)*100)}%)")
                
                candles = _get_candles_from_cache(candles_cache, symbol, maturity_tf)
                if not candles:
                    skipped_no_candles += 1
                    immature_count += 1
                    continue
                
                maturity_result = check_coin_maturity_with_storage(symbol, candles)
                if maturity_result['is_mature']:
                    mature_count += 1
                else:
                    immature_count += 1
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Ошибка проверки зрелости: {e}")
                immature_count += 1
        
        if skipped_no_candles:
            logger.info(f"📊 Без свечей в кэше по ТФ {maturity_tf} (остались незрелыми): {skipped_no_candles}")
        
        logger.info(f"✅ УМНЫЙ расчет зрелости завершен:")
        logger.info(f"📊 Уже были зрелыми: {already_mature_count}")
        logger.info(f"📊 Стали зрелыми: {mature_count}")
        logger.info(f"📊 Остались незрелыми: {immature_count}")
        logger.info(f"📊 Всего зрелых: {already_mature_count + mature_count}")
        logger.info(f"📊 Всего проверили: {len(coins_to_check)}")
        
        # 🚀 Обновляем кэш для следующего раза И СОХРАНЯЕМ В ФАЙЛ
        last_maturity_check['coins_count'] = current_coins_count
        last_maturity_check['config_hash'] = current_config_hash
        save_maturity_check_cache()  # 💾 Сохраняем в файл!
        logger.info(f"💾 Кэш обновлен и сохранен: {current_coins_count} монет")
        
        # 🔧 ОБНОВЛЯЕМ ФЛАГИ is_mature в кэшированных RSI данных
        try:
            from bots_modules.filters import update_is_mature_flags_in_rsi_data
            update_is_mature_flags_in_rsi_data()
            logger.info(f"✅ Флаги is_mature обновлены в UI данных")
        except Exception as update_error:
            logger.warning(f"⚠️ Не удалось обновить флаги is_mature: {update_error}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка умного расчета зрелости: {e}")
        return False

