"""Функции работы с оптимальными EMA периодами

Включает:
- load_optimal_ema_data - загрузка данных об оптимальных EMA
- get_optimal_ema_periods - получение оптимальных периодов для монеты
- update_optimal_ema_data - обновление данных
- save_optimal_ema_periods - сохранение данных
"""

import os
import json
import logging
import time

logger = logging.getLogger('BotsService')

# Глобальные переменные
optimal_ema_data = {}
OPTIMAL_EMA_FILE = 'data/optimal_ema.json'

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
    """Получает оптимальные EMA периоды для монеты (поддержка отдельных LONG и SHORT EMA)"""
    global optimal_ema_data
    if symbol in optimal_ema_data:
        data = optimal_ema_data[symbol]
        
        # ✅ НОВЫЙ ФОРМАТ: Отдельные EMA для LONG и SHORT
        if 'long' in data and 'short' in data:
            return {
                'long': {
                    'ema_short_period': data['long'].get('ema_short_period', 50),
                    'ema_long_period': data['long'].get('ema_long_period', 200),
                    'accuracy': data['long'].get('accuracy', 0),
                    'total_signals': data['long'].get('total_signals', 0),
                    'correct_signals': data['long'].get('correct_signals', 0)
                },
                'short': {
                    'ema_short_period': data['short'].get('ema_short_period', 50),
                    'ema_long_period': data['short'].get('ema_long_period', 200),
                    'accuracy': data['short'].get('accuracy', 0),
                    'total_signals': data['short'].get('total_signals', 0),
                    'correct_signals': data['short'].get('correct_signals', 0)
                },
                # Для обратной совместимости
                'ema_short': data.get('ema_short_period', data['long'].get('ema_short_period', 50)),
                'ema_long': data.get('ema_long_period', data['long'].get('ema_long_period', 200)),
                'accuracy': data.get('accuracy', 0),
                'long_signals': data.get('long_signals', 0),
                'short_signals': data.get('short_signals', 0),
                'analysis_method': data.get('analysis_method', 'separate_long_short')
            }
        # Поддержка формата (ema_short_period, ema_long_period)
        elif 'ema_short_period' in data and 'ema_long_period' in data:
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

def calculate_all_coins_optimal_ema(mode='auto', force_symbols=None):
    """📊 ПАКЕТНЫЙ расчет Optimal EMA через скрипт с параметрами
    
    Args:
        mode (str): Режим работы
            - 'auto': --all (только новые монеты)
            - 'force': --force (все монеты принудительно)
            - 'symbols': --force --coins LIST (конкретные монеты)
        force_symbols (list): Список монет для принудительного расчета (если mode='symbols')
    """
    try:
        logger.info(f"[OPTIMAL_EMA_BATCH] 📊 Начинаем расчет Optimal EMA (режим: {mode})...")
        
        from bots_modules.imports_and_globals import rsi_data_lock, coins_rsi_data
        import subprocess
        import os
        
        # Получаем все монеты с RSI данными
        coins_to_check = []
        with rsi_data_lock:
            for symbol, coin_data in coins_rsi_data['coins'].items():
                if coin_data.get('rsi6h') is not None:
                    coins_to_check.append(symbol)
        
        logger.info(f"[OPTIMAL_EMA_BATCH] 📊 Найдено {len(coins_to_check)} монет для расчета Optimal EMA")
        
        # 🧹 ОЧИСТКА: Удаляем неактуальные записи из Optimal EMA (только в режиме auto)
        if mode == 'auto':
            logger.info("[OPTIMAL_EMA_BATCH] 🧹 Очищаем Optimal EMA от неактуальных монет...")
            global optimal_ema_data
            original_count = len(optimal_ema_data)
            
            # Оставляем только монеты которые есть в RSI данных
            coins_to_keep = set(coins_to_check)
            optimal_ema_data = {symbol: data for symbol, data in optimal_ema_data.items() if symbol in coins_to_keep}
            
            removed_count = original_count - len(optimal_ema_data)
            if removed_count > 0:
                logger.info(f"[OPTIMAL_EMA_BATCH] 🗑️ Удалено {removed_count} неактуальных записей из Optimal EMA")
                logger.info(f"[OPTIMAL_EMA_BATCH] 📊 Осталось {len(optimal_ema_data)} актуальных записей")
        
        if not coins_to_check:
            logger.warning("[OPTIMAL_EMA_BATCH] ⚠️ Нет монет для расчета Optimal EMA")
            return False
        
        # 🚀 ЗАПУСКАЕМ СКРИПТ с нужными параметрами
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts', 'sync', 'optimal_ema.py')
        
        if not os.path.exists(script_path):
            logger.error(f"[OPTIMAL_EMA_BATCH] ❌ Скрипт не найден: {script_path}")
            return False
        
        # Формируем команду в зависимости от режима
        if mode == 'auto':
            cmd = ['python', script_path, '--all']
            logger.info("[OPTIMAL_EMA_BATCH] 🚀 Запускаем скрипт с параметром --all (только новые монеты)...")
        elif mode == 'force':
            cmd = ['python', script_path, '--force']
            logger.info("[OPTIMAL_EMA_BATCH] 🚀 Запускаем скрипт с параметром --force (все монеты принудительно)...")
        elif mode == 'symbols' and force_symbols:
            symbols_str = ','.join(force_symbols)
            cmd = ['python', script_path, '--force', '--coins', symbols_str]
            logger.info(f"[OPTIMAL_EMA_BATCH] 🚀 Запускаем скрипт с параметрами --force --coins {symbols_str}...")
        else:
            logger.error(f"[OPTIMAL_EMA_BATCH] ❌ Неверный режим или отсутствуют монеты: mode={mode}, symbols={force_symbols}")
            return False
        
        try:
            # Запускаем скрипт с нужными параметрами
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, 
                                  cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            
            if result.returncode == 0:
                logger.info("[OPTIMAL_EMA_BATCH] ✅ Скрипт optimal_ema.py завершен успешно")
                logger.info(f"[OPTIMAL_EMA_BATCH] 📊 Вывод: {result.stdout.strip()}")
                
                # Перезагружаем данные после расчета
                load_optimal_ema_data()
                
                logger.info(f"[OPTIMAL_EMA_BATCH] 📊 Актуальных записей в файле: {len(optimal_ema_data)}")
                return True
            else:
                logger.error(f"[OPTIMAL_EMA_BATCH] ❌ Скрипт завершился с ошибкой: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("[OPTIMAL_EMA_BATCH] ❌ Скрипт превысил время выполнения (10 минут)")
            return False
        except Exception as script_error:
            logger.error(f"[OPTIMAL_EMA_BATCH] ❌ Ошибка запуска скрипта: {script_error}")
            return False
        
    except Exception as e:
        logger.error(f"[OPTIMAL_EMA_BATCH] ❌ Ошибка пакетного расчета Optimal EMA: {e}")
        return False

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

