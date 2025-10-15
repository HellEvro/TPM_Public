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
    MIN_CANDLES_FOR_MATURITY = 200
    MIN_RSI_LOW = 35
    MAX_RSI_HIGH = 65

# Импорт calculate_rsi_history из calculations
try:
    from bots_modules.calculations import calculate_rsi_history
except ImportError:
    def calculate_rsi_history(prices, period=14):
        return None

# Глобальные переменные (будут импортированы из главного файла)
mature_coins_storage = {}
MATURE_COINS_FILE = 'data/mature_coins.json'
mature_coins_lock = threading.Lock()

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

