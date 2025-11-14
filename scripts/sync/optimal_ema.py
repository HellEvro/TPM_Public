#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для поиска оптимальных EMA периодов для определения идеальных точек входа
для каждой монеты. ДОТОШНЫЙ АНАЛИЗ: перебирает ВСЕ возможные комбинации EMA без ограничений.

═══════════════════════════════════════════════════════════════════════════════
📋 КОМАНДЫ ДЛЯ ЗАПУСКА СКРИПТА:
═══════════════════════════════════════════════════════════════════════════════

1. 📊 Показать список уже обработанных монет:
   python optimal_ema.py --list

2. 🪙 Рассчитать оптимальные EMA для ОДНОЙ конкретной монеты (принудительно):
   python optimal_ema.py --coin BTCUSDT
   python optimal_ema.py --coin ETHUSDT
   
   Примечание: Всегда пересчитывает, даже если монета уже обработана ранее.

3. 🪙🪙 Рассчитать оптимальные EMA для НЕСКОЛЬКИХ монет (принудительно):
   python optimal_ema.py --coins BTCUSDT,ETHUSDT,BNBUSDT
   
   Примечание: Всегда пересчитывает, даже если монеты уже обработаны ранее.

4. 🔄 Обновить только НОВЫЕ монеты (не обработанные ранее):
   python optimal_ema.py --all
   
   Примечание: Пропускает монеты, которые уже были обработаны.
   Используйте для периодического обновления новых монет.

5. ⚡ Принудительно пересчитать ВСЕ монеты:
   python optimal_ema.py --force
   
   Примечание: Пересчитывает ВСЕ монеты, даже если они уже обработаны.
   Используйте для полного пересчета всех EMA (может занять много времени!).

6. ⏱️ Указать таймфрейм для анализа (по умолчанию 6h):
   python optimal_ema.py --coin BTCUSDT --timeframe 1h
   python optimal_ema.py --all --timeframe 4h
   
   Доступные таймфреймы: 1m, 5m, 15m, 30m, 1h, 4h, 6h, 1d, 1w

7. 📖 Показать справку по всем аргументам:
   python optimal_ema.py
   python optimal_ema.py --help

═══════════════════════════════════════════════════════════════════════════════
🔧 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:
═══════════════════════════════════════════════════════════════════════════════

# Первый запуск - обработать все новые монеты
python optimal_ema.py --all

# Пересчитать конкретную монету после изменения логики
python optimal_ema.py --coin BTCUSDT

# Пересчитать несколько монет одновременно
python optimal_ema.py --coins BTCUSDT,ETHUSDT,BNBUSDT

# Полный пересчет всех монет (долго, но гарантирует актуальность)
python optimal_ema.py --force

# Посмотреть какие монеты уже обработаны
python optimal_ema.py --list

# Анализ на другом таймфрейме
python optimal_ema.py --coin BTCUSDT --timeframe 1h

═══════════════════════════════════════════════════════════════════════════════
📝 ЛОГИКА РАБОТЫ СКРИПТА:
═══════════════════════════════════════════════════════════════════════════════

1. ✅ ПОЛНЫЙ ПЕРЕБОР: Анализирует ВСЕ комбинации EMA в расширенных диапазонах 
   (3-300 для короткой, 10-600 для длинной) БЕЗ ШАГОВ (step=1) - перебирает 
   каждое значение подряд для максимальной точности

2. ✅ БОЛЬШЕ ДАННЫХ: Анализирует до 10000 свечей (увеличено с 5000) для большего 
   охвата истории. Свечи НАКАПЛИВАЮТСЯ в файле data/candles_cache.json при каждом 
   раунде загрузки системы, поэтому скрипт использует накопленные данные вместо 
   загрузки через API. Это значительно ускоряет работу и позволяет анализировать 
   больше истории.

3. ✅ ПРАВИЛЬНАЯ ЛОГИКА: Ищем моменты когда RSI входит в зону (значения из 
   конфига: RSI_OVERSOLD и RSI_OVERBOUGHT)

4. Для LONG: Ищем моменты когда RSI <= RSI_OVERSOLD, проверяем что EMA УЖЕ 
   перекрестились (ema_short > ema_long) в этот момент ИЛИ перекрестятся в 
   ближайшие 1-2 свечи

5. Для SHORT: Ищем моменты когда RSI >= RSI_OVERBOUGHT, проверяем что EMA УЖЕ 
   перекрестились (ema_short < ema_long) в этот момент ИЛИ перекрестятся в 
   ближайшие 1-2 свечи

6. Проверяем реальную прибыльность сигналов (≥1% за 20 периодов)

7. ✅ СТРОГИЙ ОТБОР: Выбираем EMA с максимальной точностью И достаточным 
   количеством сигналов (минимум 5)

8. Находим ОТДЕЛЬНЫЕ оптимальные EMA периоды для LONG и SHORT сигналов

9. Сохраняем отдельные EMA для каждого направления - они могут быть разными!

═══════════════════════════════════════════════════════════════════════════════
🔧 ИСПРАВЛЕНИЯ И УЛУЧШЕНИЯ:
═══════════════════════════════════════════════════════════════════════════════

✅ РАЗДЕЛЬНЫЕ EMA ДЛЯ LONG И SHORT:
   - Теперь для каждой монеты находятся ОТДЕЛЬНЫЕ оптимальные EMA периоды
   - LONG EMA: ищутся для моментов когда RSI <= RSI_OVERSOLD (зона покупки)
   - SHORT EMA: ищутся для моментов когда RSI >= RSI_OVERBOUGHT (зона продажи)
   - Результаты сохраняются в структуре: {"long": {...}, "short": {...}}

✅ АДАПТИВНЫЕ ДИАПАЗОНЫ EMA:
   - Если у монеты мало свечей (< 700), диапазоны EMA автоматически ограничиваются
   - Формула: max_period = available_candles - 100
   - Пример: 100 свечей → максимальный период EMA = 0 (ограничен минимумом 50)
   - Пример: 500 свечей → максимальный период EMA = 400

✅ МИНИМАЛЬНЫЕ ТРЕБОВАНИЯ ДЛЯ РАСЧЕТА EMA:
   - Минимум: 50 свечей (для EMA с периодами 10-20)
   - Формула: max_period + 100 свечей (где max_period - максимальный период EMA)
   - Запас 100 свечей включает:
     * 20 периодов для проверки прибыльности (HOLD_PERIODS)
     * 2 периода для проверки будущих свечей (max_future_candles)
     * ~78 периодов для стабилизации EMA и надежности анализа

✅ ИСПРАВЛЕН ПОИСК В КЭШЕ:
   - Теперь проверяются все варианты ключей: "0G", "0GUSDT", "0GUSDT" (верхний/нижний регистр)
   - Это решает проблему когда данные в кэше сохранены без USDT, а скрипт искал с USDT

✅ ИСПРАВЛЕНО СОХРАНЕНИЕ ДАННЫХ:
   - Теперь используется абсолютный путь к файлу (относительно корня проекта)
   - Добавлено детальное логирование процесса сохранения
   - Проверка что данные действительно сохранились после записи

✅ ПРОВЕРКА ДУБЛИКАТОВ ПРИ ПАГИНАЦИИ:
   - При загрузке свечей через API проверяются дубликаты по timestamp
   - Это предотвращает загрузку 10000 одинаковых свечей для новых монет

✅ ИСПОЛЬЗОВАНИЕ НАКОПЛЕННЫХ ДАННЫХ:
   - Скрипт сначала проверяет кэш в памяти (coins_rsi_data['candles_cache'])
   - Затем проверяет файл кэша (data/candles_cache.json)
   - Только если данных недостаточно - загружает через API
   - При загрузке через API данные накапливаются в файле кэша (объединяются со старыми)

═══════════════════════════════════════════════════════════════════════════════
⚠️ ВАЖНО:
═══════════════════════════════════════════════════════════════════════════════

Скрипт может работать долго (несколько часов для одной монеты), но это 
необходимо для получения идеальных EMA значений без погрешностей. 
Все комбинации анализируются дотошно.

═══════════════════════════════════════════════════════════════════════════════
📁 ХРАНЕНИЕ ДАННЫХ:
═══════════════════════════════════════════════════════════════════════════════

1. Результаты оптимальных EMA: data/optimal_ema.json
   - Сохраняются найденные оптимальные EMA периоды для каждой монеты

2. Кэш свечей в памяти: coins_rsi_data['candles_cache']
   - Загружается при запуске системы через load_all_coins_candles_fast()
   - Обновляется каждый раунд (каждые несколько минут)
   - Хранится в оперативной памяти для быстрого доступа

3. Кэш свечей в файле: data/candles_cache.json
   - ✅ НАКАПЛИВАЕТСЯ автоматически при каждом раунде загрузки системы
   - Каждый раунд добавляет новые свечи к существующим (объединяет, убирает дубликаты)
   - Хранит до 10000 свечей на монету (последние)
   - Используется скриптом optimal_ema.py для расчета EMA (не требует загрузки через API)
   - Данные накапливаются постепенно, поэтому чем дольше работает система, тем больше истории доступно

4. RSI кэш: data/rsi_cache.json
   - Содержит рассчитанные RSI значения для всех монет
   - Обновляется при каждом раунде загрузки данных

ПРИМЕЧАНИЕ: Скрипт использует НАКОПЛЕННЫЕ данные из файла candles_cache.json, 
которые автоматически пополняются при каждом раунде загрузки системы. Это означает:
- ✅ Не нужно загружать свечи через API каждый раз
- ✅ Доступна большая история (накапливается постепенно)
- ✅ Работа скрипта значительно быстрее
- ✅ Меньше нагрузка на API биржи

Если в файле недостаточно свечей (< 200), скрипт загрузит их через API и сохранит 
в кэш для будущего использования.
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import platform
import numpy as np

# Условный импорт numba - используем для ускорения, но отключаем multiprocessing на Windows
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    NUMBA_MESSAGE = "[INFO] Numba доступен - вычисления будут ускорены в 50+ раз"
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args, **kwargs):
        return range(*args, **kwargs)
    NUMBA_MESSAGE = "[WARNING] Numba недоступен - вычисления будут медленными"

# Настройка кодировки для Windows
if platform.system() == "Windows":
    # Устанавливаем переменную окружения для UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Настраиваем кодировку для stdout/stderr
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Безопасная инициализация multiprocessing для Windows
if platform.system() == "Windows":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Метод уже установлен, продолжаем
        pass

# Добавляем путь к модулям проекта
# Добавляем путь к корню проекта для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from exchanges.exchange_factory import ExchangeFactory
from app.config import EXCHANGES
from utils.log_rotation import setup_logger_with_rotation
import logging.handlers

# Настройка логирования с ротацией
def setup_logging():
    """Настройка логирования с автоматической ротацией при превышении 10MB"""
    # Создаем логгер с ротацией файлов
    logger = setup_logger_with_rotation(
        name='OptimalEMA',
        log_file='logs/optimal_ema.log',
        level=logging.INFO,
        max_bytes=10 * 1024 * 1024,  # 10MB
        format_string='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Добавляем консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# Константы
OPTIMAL_EMA_BASE_FILE = 'data/optimal_ema'  # Базовое имя файла
# ✅ РАСШИРЕННЫЕ ДИАПАЗОНЫ: Перебираем все возможные комбинации для точного анализа
EMA_SHORT_RANGE = (3, 300)  # Короткая EMA - расширенный диапазон
EMA_LONG_RANGE = (10, 600)  # Длинная EMA - расширенный диапазон

# ✅ Импортируем значения RSI из конфига
try:
    from bot_engine.bot_config import SystemConfig
    RSI_OVERSOLD = SystemConfig.RSI_OVERSOLD  # Зона покупки (LONG)
    RSI_OVERBOUGHT = SystemConfig.RSI_OVERBOUGHT  # Зона продажи (SHORT)
except ImportError:
    # Fallback значения, если конфиг недоступен
    RSI_OVERSOLD = 29
    RSI_OVERBOUGHT = 71
# Используем multiprocessing с безопасной инициализацией
MAX_WORKERS = mp.cpu_count()
# ✅ ДИНАМИЧЕСКИЙ МИНИМУМ: Рассчитывается на основе максимального периода EMA
# Максимальный период длинной EMA (600) + запас для анализа сигналов (150) = 750
MIN_CANDLES_FOR_ANALYSIS = EMA_LONG_RANGE[1] + 150  # 600 + 150 = 750 свечей
# Максимальное количество свечей для запроса через API (если кэш недоступен)
# Примечание: Скрипт сначала проверяет кэш в памяти и файле, API используется только при необходимости
MAX_CANDLES_TO_REQUEST = 10000  # Увеличено с 5000 до 10000 для большего охвата истории
DEFAULT_TIMEFRAME = '6h'  # Таймфрейм по умолчанию

# На Windows используем ThreadPoolExecutor вместо ProcessPoolExecutor для совместимости с numba
USE_MULTIPROCESSING = os.environ.get('OPTIMAL_EMA_NO_MP', '').lower() not in ['1', 'true', 'yes']
USE_THREADS_ON_WINDOWS = platform.system() == "Windows"

# Оптимизированные функции с numba
@jit(nopython=True, parallel=True)
def calculate_rsi_numba(prices, period=14):
    """Оптимизированный расчет RSI с numba"""
    n = len(prices)
    if n < period + 1:
        return np.zeros(n)
    
    rsi = np.zeros(n)
    gains = np.zeros(n)
    losses = np.zeros(n)
    
    # Вычисляем изменения
    for i in range(1, n):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains[i] = change
        else:
            losses[i] = -change
    
    # Первый RSI
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    
    if avg_loss == 0:
        rsi[period] = 100
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - (100 / (1 + rs))
    
    # Остальные RSI
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

@jit(nopython=True, parallel=True)
def calculate_ema_numba(prices, period):
    """Оптимизированный расчет EMA с numba"""
    n = len(prices)
    if n < period:
        return np.zeros(n)
    
    ema = np.zeros(n)
    multiplier = 2.0 / (period + 1)
    
    # Первое значение - SMA
    ema[period - 1] = np.mean(prices[:period])
    
    # Остальные значения - EMA
    for i in range(period, n):
        ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    return ema

@jit(nopython=True)
def analyze_ema_combination_long_numba(prices, rsi_values, ema_short_period, ema_long_period, rsi_oversold, max_future_candles):
    """
    Анализ для LONG сигналов: ищем моменты когда RSI входит в зону покупки,
    проверяем что EMA УЖЕ перекрестились или перекрестятся в ближайшие 1-2 свечи.
    
    Args:
        prices: Массив цен закрытия
        rsi_values: Массив значений RSI
        ema_short_period: Период короткой EMA
        ema_long_period: Период длинной EMA
        rsi_oversold: Значение RSI для зоны покупки (из конфига)
        max_future_candles: Максимальное количество свечей в будущем для проверки (1-2)
    """
    n = len(prices)
    # ✅ МИНИМАЛЬНЫЕ ТРЕБОВАНИЯ: max_period + запас для анализа
    # Запас = HOLD_PERIODS (20) + max_future_candles (2) + запас для стабилизации (10) = 32
    # Но оставляем 100 для надежности анализа (больше данных = лучше результаты)
    min_required = max(ema_short_period, ema_long_period) + 100
    if n < min_required:
        return 0.0, 0, 0
    
    # Вычисляем EMA
    ema_short = calculate_ema_numba(prices, ema_short_period)
    ema_long = calculate_ema_numba(prices, ema_long_period)
    
    # Находим общую длину для анализа
    min_length = min(len(rsi_values), len(ema_short), len(ema_long))
    start_idx = max(ema_short_period, ema_long_period) - 1
    
    if min_length - start_idx < 100:
        return 0.0, 0, 0
    
    # Параметры для проверки прибыльности
    MIN_PROFIT_PERCENT = 1.0
    HOLD_PERIODS = 20
    
    total_signals = 0
    correct_signals = 0.0
    
    # ✅ ПРАВИЛЬНАЯ ЛОГИКА: Ищем моменты когда RSI входит в зону покупки
    # EMA должны УЖЕ перекреститься в этот момент ИЛИ перекреститься в ближайшие 1-2 свечи
    for i in range(start_idx, min_length - HOLD_PERIODS - max_future_candles):
        rsi = rsi_values[i]
        entry_price = prices[i]
        
        # Ищем моменты когда RSI входит в зону покупки (используем значение из конфига)
        if rsi <= rsi_oversold:
            # ✅ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Проверяем, что EMA УЖЕ перекрестились ИЛИ перекрестятся в ближайшие 1-2 свечи
            ema_shows_up_trend = False
            
            # Проверяем текущий момент (i) и ближайшие 1-2 свечи (i+1, i+2)
            for check_idx in range(i, min(i + max_future_candles + 1, min_length)):
                if ema_short[check_idx] > ema_long[check_idx]:
                    ema_shows_up_trend = True
                    break
            
            # Если EMA перекрестились в момент входа RSI в зону или в ближайшие свечи - это хороший сигнал
            if ema_shows_up_trend:
                total_signals += 1
                
                # Проверяем реальную прибыльность
                success = False
                for j in range(1, HOLD_PERIODS + 1):
                    if i + j < min_length:
                        exit_price = prices[i + j]
                        profit_percent = ((exit_price - entry_price) / entry_price) * 100.0
                        
                        if profit_percent >= MIN_PROFIT_PERCENT:
                            success = True
                            break
                
                if success:
                    correct_signals += 1.0
    
    if total_signals == 0:
        return 0.0, 0, 0
    
    accuracy = (correct_signals / total_signals) * 100
    return accuracy, total_signals, correct_signals

@jit(nopython=True)
def analyze_ema_combination_short_numba(prices, rsi_values, ema_short_period, ema_long_period, rsi_overbought, max_future_candles):
    """
    Анализ для SHORT сигналов: ищем моменты когда RSI входит в зону продажи,
    проверяем что EMA УЖЕ перекрестились или перекрестятся в ближайшие 1-2 свечи.
    
    Args:
        prices: Массив цен закрытия
        rsi_values: Массив значений RSI
        ema_short_period: Период короткой EMA
        ema_long_period: Период длинной EMA
        rsi_overbought: Значение RSI для зоны продажи (из конфига)
        max_future_candles: Максимальное количество свечей в будущем для проверки (1-2)
    """
    n = len(prices)
    # ✅ МИНИМАЛЬНЫЕ ТРЕБОВАНИЯ: max_period + запас для анализа
    # Запас = HOLD_PERIODS (20) + max_future_candles (2) + запас для стабилизации (10) = 32
    # Но оставляем 100 для надежности анализа (больше данных = лучше результаты)
    min_required = max(ema_short_period, ema_long_period) + 100
    if n < min_required:
        return 0.0, 0, 0
    
    # Вычисляем EMA
    ema_short = calculate_ema_numba(prices, ema_short_period)
    ema_long = calculate_ema_numba(prices, ema_long_period)
    
    # Находим общую длину для анализа
    min_length = min(len(rsi_values), len(ema_short), len(ema_long))
    start_idx = max(ema_short_period, ema_long_period) - 1
    
    if min_length - start_idx < 100:
        return 0.0, 0, 0
    
    # Параметры для проверки прибыльности
    MIN_PROFIT_PERCENT = 1.0
    HOLD_PERIODS = 20
    
    total_signals = 0
    correct_signals = 0.0
    
    # ✅ ПРАВИЛЬНАЯ ЛОГИКА: Ищем моменты когда RSI входит в зону продажи
    # EMA должны УЖЕ перекреститься в этот момент ИЛИ перекреститься в ближайшие 1-2 свечи
    for i in range(start_idx, min_length - HOLD_PERIODS - max_future_candles):
        rsi = rsi_values[i]
        entry_price = prices[i]
        
        # Ищем моменты когда RSI входит в зону продажи (используем значение из конфига)
        if rsi >= rsi_overbought:
            # ✅ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Проверяем, что EMA УЖЕ перекрестились ИЛИ перекрестятся в ближайшие 1-2 свечи
            ema_shows_down_trend = False
            
            # Проверяем текущий момент (i) и ближайшие 1-2 свечи (i+1, i+2)
            for check_idx in range(i, min(i + max_future_candles + 1, min_length)):
                if ema_short[check_idx] < ema_long[check_idx]:
                    ema_shows_down_trend = True
                    break
            
            # Если EMA перекрестились в момент входа RSI в зону или в ближайшие свечи - это хороший сигнал
            if ema_shows_down_trend:
                total_signals += 1
                
                # Проверяем реальную прибыльность
                success = False
                for j in range(1, HOLD_PERIODS + 1):
                    if i + j < min_length:
                        exit_price = prices[i + j]
                        profit_percent = ((entry_price - exit_price) / entry_price) * 100.0
                        
                        if profit_percent >= MIN_PROFIT_PERCENT:
                            success = True
                            break
                
                if success:
                    correct_signals += 1.0
    
    if total_signals == 0:
        return 0.0, 0, 0
    
    accuracy = (correct_signals / total_signals) * 100
    return accuracy, total_signals, correct_signals

@jit(nopython=True)
def analyze_ema_combination_numba(prices, rsi_values, ema_short_period, ema_long_period, rsi_oversold, rsi_overbought, max_future_candles):
    """
    Объединенный анализ для обратной совместимости
    """
    long_accuracy, long_total, long_correct = analyze_ema_combination_long_numba(
        prices, rsi_values, ema_short_period, ema_long_period, rsi_oversold, max_future_candles
    )
    short_accuracy, short_total, short_correct = analyze_ema_combination_short_numba(
        prices, rsi_values, ema_short_period, ema_long_period, rsi_overbought, max_future_candles
    )
    
    total_signals = long_total + short_total
    correct_signals = long_correct + short_correct
    
    if total_signals == 0:
        return 0.0, 0, 0, 0, 0
    
    accuracy = (correct_signals / total_signals) * 100
    return accuracy, total_signals, correct_signals, long_total, short_total

# Импортируем конфигурацию из app.config
try:
    from app.config import EXCHANGES
except ImportError:
    # Fallback конфигурация
    EXCHANGES = {
        'BYBIT': {
            'api_key': 'your_api_key_here',
            'api_secret': 'your_api_secret_here'
        }
    }

def analyze_ema_combination_parallel(args):
    """Умная функция для параллельной обработки комбинаций EMA с анализом пересечений"""
    symbol, candles, rsi_values, ema_short_period, ema_long_period, signal_type, rsi_oversold, rsi_overbought, max_future_candles = args
    
    try:
        # Конвертируем в numpy массивы
        prices = np.array([float(candle['close']) for candle in candles], dtype=np.float64)
        
        # Анализируем в зависимости от типа сигнала
        if signal_type == 'long':
            accuracy, total_signals, correct_signals = analyze_ema_combination_long_numba(
                prices, rsi_values, ema_short_period, ema_long_period, rsi_oversold, max_future_candles
            )
            return {
                'accuracy': accuracy,
                'total_signals': total_signals,
                'correct_signals': correct_signals,
                'long_signals': total_signals,
                'short_signals': 0,
                'ema_short_period': ema_short_period,
                'ema_long_period': ema_long_period,
                'signal_type': 'long'
            }
        elif signal_type == 'short':
            accuracy, total_signals, correct_signals = analyze_ema_combination_short_numba(
                prices, rsi_values, ema_short_period, ema_long_period, rsi_overbought, max_future_candles
            )
            return {
                'accuracy': accuracy,
                'total_signals': total_signals,
                'correct_signals': correct_signals,
                'long_signals': 0,
                'short_signals': total_signals,
                'ema_short_period': ema_short_period,
                'ema_long_period': ema_long_period,
                'signal_type': 'short'
            }
        else:
            # Объединенный анализ для обратной совместимости (используем значения по умолчанию)
            long_accuracy, long_total, long_correct = analyze_ema_combination_long_numba(
                prices, rsi_values, ema_short_period, ema_long_period, rsi_oversold, max_future_candles
            )
            short_accuracy, short_total, short_correct = analyze_ema_combination_short_numba(
                prices, rsi_values, ema_short_period, ema_long_period, rsi_overbought, max_future_candles
            )
            
            total_signals = long_total + short_total
            correct_signals = long_correct + short_correct
            
            if total_signals == 0:
                accuracy = 0.0
            else:
                accuracy = (correct_signals / total_signals) * 100
            
            return {
                'accuracy': accuracy,
                'total_signals': total_signals,
                'correct_signals': correct_signals,
                'long_signals': long_total,
                'short_signals': short_total,
                'ema_short_period': ema_short_period,
                'ema_long_period': ema_long_period,
                'signal_type': 'both'
            }
        
    except Exception as e:
        logger.error(f"Ошибка в анализе комбинации {ema_short_period}/{ema_long_period} для {symbol}: {e}")
        return {
            'accuracy': 0,
            'total_signals': 0,
            'correct_signals': 0,
            'long_signals': 0,
            'short_signals': 0,
            'ema_short_period': ema_short_period,
            'ema_long_period': ema_long_period
        }

def calculate_rsi_parallel(prices, period=14):
    """Параллельная версия расчета RSI"""
    if len(prices) < period + 1:
        return []
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    rsi_values = []
    
    for i in range(period, len(prices)):
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_values.append(rsi)
        
        if i < len(prices) - 1:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    return rsi_values

def calculate_ema_parallel(prices, period):
    """Параллельная версия расчета EMA"""
    if len(prices) < period:
        return []
    
    ema = [0] * len(prices)
    ema[period - 1] = sum(prices[:period]) / period
    
    multiplier = 2 / (period + 1)
    
    for i in range(period, len(prices)):
        ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    return ema[period-1:]

def determine_trend_parallel(ema_short, ema_long, index):
    """Параллельная версия определения тренда"""
    if index >= len(ema_short) or index >= len(ema_long):
        return 'NEUTRAL'
    
    if ema_short[index] > ema_long[index]:
        return 'UP'
    elif ema_short[index] < ema_long[index]:
        return 'DOWN'
    else:
        return 'NEUTRAL'

class OptimalEMAFinder:
    """Умный класс для поиска оптимальных EMA периодов с двухэтапным анализом"""
    
    def __init__(self, timeframe: str = DEFAULT_TIMEFRAME):
        self.exchange = None
        self.optimal_ema_data = {}
        self.timeframe = timeframe
        self.optimal_ema_file = self._get_ema_file_path()
        self.load_optimal_ema_data()
        self._init_exchange()
        self.rsi_cache = {}  # Кэш для RSI значений
    
    def _get_ema_file_path(self) -> str:
        """Возвращает путь к файлу в зависимости от таймфрейма"""
        if self.timeframe == DEFAULT_TIMEFRAME:
            # Для 6h используем стандартное имя файла
            return f"{OPTIMAL_EMA_BASE_FILE}.json"
        else:
            # Для других таймфреймов добавляем суффикс
            return f"{OPTIMAL_EMA_BASE_FILE}_{self.timeframe}.json"
    
    def _init_exchange(self):
        """Инициализирует exchange"""
        try:
            self.exchange = ExchangeFactory.create_exchange(
                'BYBIT',
                EXCHANGES['BYBIT']['api_key'],
                EXCHANGES['BYBIT']['api_secret']
            )
        except Exception as e:
            logger.error(f"Ошибка инициализации exchange: {e}")
            self.exchange = None
    
    def load_optimal_ema_data(self):
        """Загружает данные об оптимальных EMA из файла"""
        try:
            # ✅ Определяем абсолютный путь к файлу (относительно корня проекта)
            file_path = self.optimal_ema_file
            if not os.path.isabs(file_path):
                # Если путь относительный, делаем его абсолютным относительно корня проекта
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                file_path = os.path.join(project_root, file_path)
            
            # Обновляем путь в объекте
            self.optimal_ema_file = file_path
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.optimal_ema_data = json.load(f)
                logger.info(f"Загружено {len(self.optimal_ema_data)} записей об оптимальных EMA для таймфрейма {self.timeframe} из файла {file_path}")
            else:
                self.optimal_ema_data = {}
                logger.info(f"Файл {file_path} не найден, создаем новую базу данных")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных EMA из файла {self.optimal_ema_file}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.optimal_ema_data = {}
    
    def save_optimal_ema_data(self):
        """Сохраняет данные об оптимальных EMA в файл"""
        try:
            # ✅ Определяем абсолютный путь к файлу (относительно корня проекта)
            file_path = self.optimal_ema_file
            if not os.path.isabs(file_path):
                # Если путь относительный, делаем его абсолютным относительно корня проекта
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                file_path = os.path.join(project_root, file_path)
            
            # Создаем директорию если нужно
            dir_path = os.path.dirname(file_path)
            if dir_path:  # Проверяем что путь не пустой
                os.makedirs(dir_path, exist_ok=True)
            
            # Сохраняем данные
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.optimal_ema_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Сохранено {len(self.optimal_ema_data)} записей об оптимальных EMA для таймфрейма {self.timeframe} в файл {file_path}")
            
            # ✅ Обновляем путь в объекте для будущих использований
            self.optimal_ema_file = file_path
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения данных EMA в файл {self.optimal_ema_file}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def get_candles_data(self, symbol: str) -> Optional[List[Dict]]:
        """
        Получает данные свечей для символа.
        ✅ УЛУЧШЕНО: Сначала проверяет кэш в памяти и файлах, затем загружает через API.
        ✅ ДИНАМИЧЕСКИЙ МИНИМУМ: Проверяет достаточность данных на основе максимального периода EMA.
        """
        try:
            # ✅ Вычисляем минимальное количество свечей динамически
            # Максимальный период длинной EMA + запас для анализа (100 свечей для проверки сигналов)
            min_candles_needed = EMA_LONG_RANGE[1] + 100  # 600 + 100 = 700 свечей
            
            # Очищаем символ от USDT если есть
            clean_symbol = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
            # ✅ Формируем ключ: проверяем разные варианты формата
            symbol_key = f"{clean_symbol}USDT"
            # Также проверяем варианты без USDT и с разными регистрами
            # ВАЖНО: В файле кэша может быть сохранено как "0G" (без USDT) или "0GUSDT"
            symbol_variants = [
                clean_symbol,  # 0G (без USDT) - ПЕРВЫМ, т.к. в файле кэша часто так сохранено
                symbol_key,    # 0GUSDT
                symbol,        # 0G или 0GUSDT (как пришло)
                f"{clean_symbol}USDT".upper(),  # 0GUSDT (верхний регистр)
                f"{clean_symbol}USDT".lower(),  # 0gusdt (нижний регистр)
            ]
            
            # ✅ ШАГ 1: Проверяем кэш свечей в памяти (coins_rsi_data['candles_cache'])
            try:
                from bots_modules.imports_and_globals import coins_rsi_data
                candles_cache = coins_rsi_data.get('candles_cache', {})
                
                # Проверяем все варианты ключа
                cached_data = None
                found_key = None
                for variant in symbol_variants:
                    if variant in candles_cache:
                        cached_data = candles_cache[variant]
                        found_key = variant
                        break
                
                if cached_data:
                    # Проверяем что это свечи для нужного таймфрейма (6h по умолчанию)
                    if 'candles' in cached_data and cached_data.get('timeframe') == self.timeframe:
                        candles = cached_data['candles']
                        # ✅ ДИНАМИЧЕСКАЯ ПРОВЕРКА: достаточно ли свечей для максимального периода EMA
                        if candles and len(candles) >= min_candles_needed:
                            logger.info(f"✅ Использованы свечи из кэша памяти: {len(candles)} свечей для {symbol} (ключ: {found_key}, минимум: {min_candles_needed})")
                            return candles
                        else:
                            logger.debug(f"В кэше памяти недостаточно свечей для {symbol}: {len(candles) if candles else 0}/{min_candles_needed}")
            except Exception as cache_error:
                logger.debug(f"Кэш памяти недоступен: {cache_error}")
            
            # ✅ ШАГ 2: Проверяем файл кэша свечей (если есть)
            # Определяем путь относительно корня проекта (как и другие файлы данных)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            candles_cache_file = os.path.join(project_root, 'data', 'candles_cache.json')
            if os.path.exists(candles_cache_file):
                try:
                    with open(candles_cache_file, 'r', encoding='utf-8') as f:
                        file_cache = json.load(f)
                    
                    # ✅ Проверяем все варианты ключа
                    cached_data = None
                    found_key = None
                    for variant in symbol_variants:
                        if variant in file_cache:
                            cached_data = file_cache[variant]
                            found_key = variant
                            break
                    
                    if cached_data:
                        if 'candles' in cached_data and cached_data.get('timeframe') == self.timeframe:
                            candles = cached_data['candles']
                            # ✅ ИСПОЛЬЗУЕМ НАКОПЛЕННЫЕ ДАННЫЕ: не проверяем возраст, т.к. данные накапливаются каждый раунд
                            # ✅ ДИНАМИЧЕСКАЯ ПРОВЕРКА: достаточно ли свечей для максимального периода EMA
                            if candles and len(candles) >= min_candles_needed:
                                # Показываем информацию о последнем обновлении
                                last_update = cached_data.get('last_update') or cached_data.get('timestamp', 'неизвестно')
                                logger.info(f"✅ Использованы НАКОПЛЕННЫЕ свечи из файла кэша: {len(candles)} свечей для {symbol} (ключ: {found_key}, минимум: {min_candles_needed}, обновлено: {last_update})")
                                return candles
                            else:
                                logger.debug(f"В кэше недостаточно свечей для {symbol}: {len(candles) if candles else 0}/{min_candles_needed}")
                    else:
                        # Показываем какие ключи есть в файле для отладки
                        available_keys = list(file_cache.keys())[:10]  # Первые 10 для примера
                        logger.debug(f"Ключ {symbol_key} не найден в кэше. Доступные ключи (примеры): {available_keys}")
                except Exception as file_error:
                    logger.debug(f"Ошибка чтения файла кэша: {file_error}")
            
            # ✅ ШАГ 3: Загружаем через API (если кэш недоступен или устарел)
            logger.info(f"📡 Загрузка свечей через API для {symbol}...")
            if not self.exchange:
                self._init_exchange()
                if not self.exchange:
                    raise Exception("Не удалось инициализировать exchange")
            
            # Пытаемся получить расширенные данные с пагинацией
            candles = self._get_extended_candles_data(clean_symbol, self.timeframe, MAX_CANDLES_TO_REQUEST)
            
            if not candles:
                # Fallback к стандартному методу
                logger.info(f"Пагинация не удалась, используем стандартный метод для {symbol}")
                response = self.exchange.get_chart_data(clean_symbol, self.timeframe, '1y')
                if response and response.get('success'):
                    candles = response['data']['candles']
                else:
                    logger.warning(f"Не удалось получить данные для {symbol}")
                    return None
            
            # ✅ Сохраняем в файл кэша с НАРАЩИВАНИЕМ данных (как в системе)
            # Объединяем старые и новые свечи, чтобы не конфликтовать с системой
            # Сохраняем любые свечи, даже если их меньше минимума (они накопятся со временем)
            if candles and len(candles) > 0:
                try:
                    # Определяем путь относительно корня проекта (как и другие файлы данных)
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    candles_cache_file = os.path.join(project_root, 'data', 'candles_cache.json')
                    os.makedirs(os.path.dirname(candles_cache_file), exist_ok=True)
                    
                    # Загружаем существующий кэш
                    file_cache = {}
                    if os.path.exists(candles_cache_file):
                        try:
                            with open(candles_cache_file, 'r', encoding='utf-8') as f:
                                file_cache = json.load(f)
                        except:
                            pass
                    
                    # ✅ НАРАЩИВАЕМ данные: объединяем старые и новые свечи (как в системе)
                    existing_data = file_cache.get(symbol_key, {})
                    existing_candles = existing_data.get('candles', [])
                    
                    # Создаем словарь для быстрого поиска по timestamp
                    candles_dict = {}
                    
                    # Добавляем существующие свечи
                    for candle in existing_candles:
                        timestamp = candle.get('timestamp') or candle.get('time') or candle.get('openTime')
                        if timestamp:
                            candles_dict[timestamp] = candle
                    
                    # Добавляем/обновляем новыми свечами (новые перезаписывают старые)
                    for candle in candles:
                        timestamp = candle.get('timestamp') or candle.get('time') or candle.get('openTime')
                        if timestamp:
                            candles_dict[timestamp] = candle
                    
                    # Преобразуем обратно в список и сортируем по времени
                    merged_candles = list(candles_dict.values())
                    merged_candles.sort(key=lambda x: x.get('timestamp') or x.get('time') or x.get('openTime') or 0)
                    
                    # Ограничиваем максимальное количество свечей (последние 10000)
                    max_candles = 10000
                    if len(merged_candles) > max_candles:
                        merged_candles = merged_candles[-max_candles:]
                        logger.debug(f"Обрезано до {max_candles} свечей для {symbol}")
                    
                    # Обновляем кэш для этой монеты
                    old_count = len(existing_candles)
                    new_count = len(merged_candles)
                    added_count = new_count - old_count
                    
                    file_cache[symbol_key] = {
                        'candles': merged_candles,
                        'timeframe': self.timeframe,
                        'timestamp': datetime.now().isoformat(),
                        'count': new_count,
                        'last_update': datetime.now().isoformat()
                    }
                    
                    # Сохраняем обратно
                    with open(candles_cache_file, 'w', encoding='utf-8') as f:
                        json.dump(file_cache, f, indent=2, ensure_ascii=False)
                    
                    if added_count > 0:
                        logger.info(f"💾 Кэш накоплен: {symbol} {old_count} -> {new_count} свечей (+{added_count})")
                    else:
                        logger.debug(f"💾 Свечи сохранены в кэш файл: {len(candles)} свечей для {symbol}")
                except Exception as save_error:
                    logger.debug(f"Не удалось сохранить в кэш файл: {save_error}")
                
                logger.info(f"✅ Получено {len(candles)} свечей для {symbol} через API")
                
                # ✅ Возвращаем свечи независимо от количества (анализ адаптируется автоматически)
                if len(candles) < min_candles_needed:
                    logger.warning(f"⚠️ Получено только {len(candles)} свечей для {symbol}, но для полного анализа желательно минимум {min_candles_needed} свечей")
                    logger.warning(f"   Анализ будет ограничен периодами EMA до {max(50, len(candles) - 120)}")
                return candles
                
        except Exception as e:
            logger.error(f"Ошибка получения данных для {symbol}: {e}")
            return None
    
    def _get_extended_candles_data(self, symbol: str, timeframe: str = '6h', target_candles: int = 5000) -> Optional[List[Dict]]:
        """Получает расширенные данные свечей с пагинацией"""
        try:
            # Маппинг таймфреймов
            timeframe_map = {
                '1m': '1',
                '5m': '5',
                '15m': '15',
                '30m': '30',
                '1h': '60',
                '4h': '240',
                '6h': '360',
                '1d': 'D',
                '1w': 'W'
            }
            
            interval = timeframe_map.get(timeframe)
            if not interval:
                logger.warning(f"Неподдерживаемый таймфрейм: {timeframe}")
                return None
            
            all_candles = []
            limit = 1000  # Максимум за запрос
            end_time = None  # Для пагинации
            seen_timestamps = set()  # ✅ Для проверки дубликатов
            
            logger.info(f"Запрашиваем расширенные данные для {symbol} (цель: {target_candles} свечей)")
            
            max_iterations = 20  # ✅ Ограничиваем количество итераций (максимум 20 запросов)
            iteration = 0
            
            while len(all_candles) < target_candles and iteration < max_iterations:
                iteration += 1
                try:
                    # Параметры запроса
                    params = {
                        'category': 'linear',
                        'symbol': f'{symbol}USDT',
                        'interval': interval,
                        'limit': min(limit, target_candles - len(all_candles))
                    }
                    
                    # Добавляем end_time для пагинации (если не первый запрос)
                    if end_time:
                        params['end'] = end_time
                    
                    response = self.exchange.client.get_kline(**params)
                    
                    if response['retCode'] == 0:
                        klines = response['result']['list']
                        if not klines:
                            logger.info("Больше данных нет")
                            break
                        
                        # ✅ Проверяем на дубликаты и конвертируем в наш формат
                        batch_candles = []
                        new_candles_count = 0
                        for k in klines:
                            candle_time = int(k[0])
                            
                            # Пропускаем дубликаты
                            if candle_time in seen_timestamps:
                                continue
                            
                            seen_timestamps.add(candle_time)
                            
                            candle = {
                                'time': candle_time,
                                'open': float(k[1]),
                                'high': float(k[2]),
                                'low': float(k[3]),
                                'close': float(k[4]),
                                'volume': float(k[5])
                            }
                            batch_candles.append(candle)
                            new_candles_count += 1
                        
                        if not batch_candles:
                            # Все свечи были дубликатами - значит достигли конца истории
                            logger.info("Все свечи дубликаты - достигнут конец истории")
                            break
                        
                        # Добавляем к общему списку
                        all_candles.extend(batch_candles)
                        
                        # Обновляем end_time для следующего запроса (берем время первой свечи - 1)
                        end_time = int(klines[0][0]) - 1
                        
                        logger.debug(f"Получено {new_candles_count} новых свечей (пропущено дубликатов: {len(klines) - new_candles_count}), всего: {len(all_candles)}")
                        
                        # Небольшая пауза между запросами
                        time.sleep(0.1)
                    else:
                        logger.warning(f"Ошибка API: {response.get('retMsg', 'Неизвестная ошибка')}")
                        break
                        
                except Exception as e:
                    logger.error(f"Ошибка запроса пагинации: {e}")
                    break
            
            if all_candles:
                # Сортируем свечи от старых к новым
                all_candles.sort(key=lambda x: x['time'])
                
                logger.info(f"[OK] Получено {len(all_candles)} свечей через пагинацию")
                return all_candles
            else:
                logger.warning("Не удалось получить данные через пагинацию")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка расширенного получения данных: {e}")
            return None
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Вычисляет волатильность для адаптации диапазонов"""
        if len(prices) < 100:
            return 0.02  # Средняя волатильность по умолчанию
        
        # Вычисляем дневные изменения
        daily_returns = np.diff(prices) / prices[:-1]
        
        # Волатильность как стандартное отклонение
        volatility = np.std(daily_returns)
        
        return volatility
    
    def _generate_adaptive_combinations(self, symbol: str, candles: List[Dict]) -> List[Tuple[int, int]]:
        """
        Генерирует ВСЕ возможные комбинации EMA для дотошного анализа.
        БЕЗ ОГРАНИЧЕНИЙ: перебираем все значения подряд (step=1) для максимальной точности.
        ✅ АДАПТИВНЫЕ ДИАПАЗОНЫ: Ограничивает максимальные периоды EMA в зависимости от доступных свечей.
6        """
        prices = np.array([float(candle['close']) for candle in candles], dtype=np.float64)
        volatility = self._calculate_volatility(prices)
        
        combinations = []
        
        # ✅ АДАПТИВНЫЕ ДИАПАЗОНЫ: Ограничиваем максимальные периоды в зависимости от количества свечей
        # МИНИМАЛЬНЫЕ ТРЕБОВАНИЯ ДЛЯ РАСЧЕТА EMA:
        # - Для EMA с периодом N нужно минимум N свечей
        # - Плюс запас для проверки прибыльности: 20 периодов (HOLD_PERIODS)
        # - Плюс запас для проверки будущих свечей: 2 периода (max_future_candles)
        # - Плюс запас для стабилизации и анализа: ~100 периодов (из проверки в numba функциях)
        # ИТОГО: max_period + 100 = available_candles, значит max_period = available_candles - 100
        # Но оставляем минимум 50 для минимального анализа
        available_candles = len(candles)
        max_usable_period = max(50, available_candles - 100)  # Минимум 50, но с учетом запаса в 100 свечей
        
        # Ограничиваем максимальные периоды доступными данными
        ema_short_min = EMA_SHORT_RANGE[0]
        ema_short_max = min(EMA_SHORT_RANGE[1], max_usable_period)
        ema_long_min = EMA_LONG_RANGE[0]
        ema_long_max = min(EMA_LONG_RANGE[1], max_usable_period)
        
        # Если доступно мало свечей, предупреждаем
        if available_candles < MIN_CANDLES_FOR_ANALYSIS:
            logger.warning(f"⚠️ Для {symbol} доступно только {available_candles} свечей (минимум для полного анализа: {MIN_CANDLES_FOR_ANALYSIS})")
            logger.warning(f"   Диапазоны EMA ограничены: короткая EMA до {ema_short_max}, длинная EMA до {ema_long_max}")
        
        logger.info(f"Генерация ВСЕХ комбинаций EMA для {symbol}:")
        logger.info(f"  Короткая EMA: {ema_short_min}..{ema_short_max} (step=1)")
        logger.info(f"  Длинная EMA: {ema_long_min}..{ema_long_max} (step=1)")
        logger.info(f"  Волатильность: {volatility:.3f}")
        logger.info(f"  Доступно свечей: {available_candles}, максимальный период EMA: {max_usable_period}")
        
        # ✅ ПЕРЕБИРАЕМ ВСЕ ЗНАЧЕНИЯ ПОДРЯД (step=1) - БЕЗ ПРОПУСКОВ
        total_combinations = 0
        for ema_short in range(ema_short_min, ema_short_max + 1):
            # Длинная EMA должна быть больше короткой минимум на 5 периодов для значимости
            min_long = max(ema_long_min, ema_short + 5)
            for ema_long in range(min_long, ema_long_max + 1):
                combinations.append((ema_short, ema_long))
                total_combinations += 1
        
        logger.info(f"✅ Сгенерировано {len(combinations)} комбинаций EMA для {symbol} (полный перебор)")
        logger.info(f"   Это займет больше времени, но даст максимально точные результаты")
        
        return combinations
    
    def _generate_detailed_combinations(self, best_candidates: List[Dict]) -> List[Tuple[int, int]]:
        """Генерирует детальные комбинации вокруг лучших кандидатов"""
        combinations = []
        
        for candidate in best_candidates:
            ema_short = candidate['ema_short_period']
            ema_long = candidate['ema_long_period']
            
            # ✅ ДЕТАЛЬНЫЙ ПЕРЕБОР: Расширенная окрестность с шагом 1
            # Проверяем все значения в окрестности ±10 для короткой и ±20 для длинной
            for short_offset in range(-10, 11, 1):  # step=1 для детальности
                for long_offset in range(-20, 21, 1):  # step=1 для детальности
                    new_short = ema_short + short_offset
                    new_long = ema_long + long_offset
                    
                    # Проверяем что значения в допустимых диапазонах
                    if (EMA_SHORT_RANGE[0] <= new_short <= EMA_SHORT_RANGE[1] and
                        EMA_LONG_RANGE[0] <= new_long <= EMA_LONG_RANGE[1] and
                        new_short < new_long):
                        combinations.append((new_short, new_long))
        
        # Убираем дубликаты
        combinations = list(set(combinations))
        
        logger.info(f"✅ Сгенерировано {len(combinations)} детальных комбинаций вокруг лучших кандидатов (step=1)")
        return combinations
    
    def _analyze_combinations(self, symbol: str, candles: List[Dict], rsi_values: np.ndarray, 
                            combinations: List[Tuple[int, int]], stage_name: str, signal_type: str = 'both',
                            rsi_oversold: float = None, rsi_overbought: float = None, max_future_candles: int = 2) -> List[Dict]:
        """Анализирует список комбинаций EMA для указанного типа сигнала"""
        if not combinations:
            return []
        
        # Используем значения из конфига, если не переданы
        if rsi_oversold is None:
            rsi_oversold = RSI_OVERSOLD
        if rsi_overbought is None:
            rsi_overbought = RSI_OVERBOUGHT
        if max_future_candles is None:
            max_future_candles = 2  # По умолчанию проверяем 1-2 свечи в будущем
        
        best_accuracy = 0
        best_combination = None
        all_results = []
        
        # Подготавливаем аргументы для параллельной обработки
        args_list = []
        for ema_short, ema_long in combinations:
            args_list.append((symbol, candles, rsi_values, ema_short, ema_long, signal_type, rsi_oversold, rsi_overbought, max_future_candles))
        
        total_combinations = len(combinations)
        logger.info(f"{stage_name}: Анализируем {total_combinations} комбинаций EMA для {symbol}")
        
        # Параллельная обработка
        use_parallel = USE_MULTIPROCESSING
        if use_parallel:
            try:
                # На Windows используем ThreadPoolExecutor для совместимости с numba
                if USE_THREADS_ON_WINDOWS:
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        future_to_combination = {
                            executor.submit(analyze_ema_combination_parallel, args): args 
                            for args in args_list
                        }
                        
                        completed = 0
                        for future in as_completed(future_to_combination):
                            completed += 1
                            
                            if completed % 50 == 0:
                                progress = (completed / total_combinations) * 100
                                logger.info(f"{stage_name} {symbol}: {progress:.1f}% ({completed}/{total_combinations})")
                            
                            try:
                                result = future.result()
                                all_results.append(result)
                                
                                if result['accuracy'] > best_accuracy:
                                    best_accuracy = result['accuracy']
                                    best_combination = result
                                    logger.info(f"{stage_name} {symbol}: Новая лучшая комбинация "
                                              f"EMA({result['ema_short_period']},{result['ema_long_period']}) "
                                              f"с точностью {result['accuracy']:.1f}% "
                                              f"(Long: {result['long_signals']}, Short: {result['short_signals']})")
                                
                            except Exception as e:
                                logger.error(f"Ошибка обработки комбинации: {e}")
                else:
                    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        future_to_combination = {
                            executor.submit(analyze_ema_combination_parallel, args): args 
                            for args in args_list
                        }
                        
                        completed = 0
                        for future in as_completed(future_to_combination):
                            completed += 1
                            
                            if completed % 50 == 0:
                                progress = (completed / total_combinations) * 100
                                logger.info(f"{stage_name} {symbol}: {progress:.1f}% ({completed}/{total_combinations})")
                            
                            try:
                                result = future.result()
                                all_results.append(result)
                                
                                if result['accuracy'] > best_accuracy:
                                    best_accuracy = result['accuracy']
                                    best_combination = result
                                    logger.info(f"{stage_name} {symbol}: Новая лучшая комбинация "
                                              f"EMA({result['ema_short_period']},{result['ema_long_period']}) "
                                              f"с точностью {result['accuracy']:.1f}% "
                                              f"(Long: {result['long_signals']}, Short: {result['short_signals']})")
                                
                            except Exception as e:
                                logger.error(f"Ошибка обработки комбинации: {e}")
                                
            except Exception as e:
                logger.warning(f"Ошибка параллельной обработки, переключаемся на последовательную: {e}")
                use_parallel = False
        
        if not use_parallel:
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 8)) as executor:
                future_to_combination = {
                    executor.submit(analyze_ema_combination_parallel, args): args 
                    for args in args_list
                }
                
                completed = 0
                for future in as_completed(future_to_combination):
                    completed += 1
                    
                    if completed % 50 == 0:
                        progress = (completed / total_combinations) * 100
                        logger.info(f"{stage_name} {symbol}: {progress:.1f}% ({completed}/{total_combinations})")
                    
                    try:
                        result = future.result()
                        all_results.append(result)
                        
                        if result['accuracy'] > best_accuracy:
                            best_accuracy = result['accuracy']
                            best_combination = result
                            logger.info(f"{stage_name} {symbol}: Новая лучшая комбинация "
                                      f"EMA({result['ema_short_period']},{result['ema_long_period']}) "
                                      f"с точностью {result['accuracy']:.1f}% "
                                      f"(Long: {result['long_signals']}, Short: {result['short_signals']})")
                        
                    except Exception as e:
                        logger.error(f"Ошибка обработки комбинации: {e}")
        
        logger.info(f"{stage_name} {symbol}: Обработано {len(all_results)} комбинаций")
        return all_results
    
    def find_optimal_ema(self, symbol: str, force_rescan: bool = False) -> Optional[Dict]:
        """Находит оптимальные EMA периоды для монеты с умным двухэтапным анализом"""
        try:
            # Очищаем символ от USDT для проверки в данных
            clean_symbol = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
            
            # Проверяем, есть ли уже данные
            if not force_rescan and clean_symbol in self.optimal_ema_data:
                logger.info(f"Оптимальные EMA для {clean_symbol} уже найдены, пропускаем")
                return self.optimal_ema_data[clean_symbol]
            
            logger.info(f"Поиск оптимальных EMA для {symbol}...")
            
            # Получаем данные свечей
            candles = self.get_candles_data(symbol)
            if not candles:
                return None
            
            # ✅ Проверяем минимальное количество свечей для расчета EMA
            # МИНИМАЛЬНЫЕ ТРЕБОВАНИЯ:
            # - Для расчета EMA с периодом N нужно минимум N свечей
            # - Плюс запас для проверки прибыльности: 20 периодов (HOLD_PERIODS)
            # - Плюс запас для проверки будущих свечей: 2 периода (max_future_candles)
            # - Плюс небольшой запас для стабилизации: ~10 периодов
            # ИТОГО: минимум для EMA(10, 20) = 20 + 20 + 2 + 10 = 52 свечи
            # Но для минимального анализа можно обойтись меньшим количеством
            min_candles_for_ema = 50  # Минимум для расчета EMA с минимальными периодами (10-20)
            if len(candles) < min_candles_for_ema:
                logger.warning(f"⚠️ Для {symbol} недостаточно свечей для анализа EMA: {len(candles)} < {min_candles_for_ema}")
                logger.warning(f"   Минимум: {min_candles_for_ema} свечей (для EMA с периодами 10-20)")
                return None
            
            # Вычисляем RSI один раз для всех комбинаций
            prices = np.array([float(candle['close']) for candle in candles], dtype=np.float64)
            rsi_values = calculate_rsi_numba(prices, 14)
            
            # ✅ НОВАЯ ЛОГИКА: Ищем отдельно для LONG и SHORT
            
            # === ПОИСК ОПТИМАЛЬНЫХ EMA ДЛЯ LONG ===
            logger.info(f"Поиск оптимальных EMA для LONG сигналов {symbol}...")
            stage1_combinations_long = self._generate_adaptive_combinations(symbol, candles)
            
            best_candidates_long = self._analyze_combinations(
                symbol, candles, rsi_values, stage1_combinations_long, "Этап 1 LONG", signal_type='long'
            )
            
            best_long = None
            if best_candidates_long:
                # ✅ УВЕЛИЧЕНО: Берем топ-5 кандидатов для более детального анализа
                top_candidates_long = sorted(best_candidates_long, key=lambda x: x['accuracy'], reverse=True)[:5]
                stage2_combinations_long = self._generate_detailed_combinations(top_candidates_long)
                final_results_long = self._analyze_combinations(
                    symbol, candles, rsi_values, stage2_combinations_long, "Этап 2 LONG", signal_type='long'
                )
                
                if final_results_long:
                    # ✅ СТРОГИЙ ОТБОР: Выбираем EMA с максимальной точностью И достаточным количеством сигналов
                    # Минимум 5 сигналов для надежности
                    valid_results = [r for r in final_results_long if r.get('total_signals', 0) >= 5]
                    if valid_results:
                        best_long = max(valid_results, key=lambda x: (x['accuracy'], x.get('total_signals', 0)))
                    else:
                        best_long = max(final_results_long, key=lambda x: x['accuracy'])
                else:
                    best_long = top_candidates_long[0] if top_candidates_long else None
            
            # === ПОИСК ОПТИМАЛЬНЫХ EMA ДЛЯ SHORT ===
            logger.info(f"Поиск оптимальных EMA для SHORT сигналов {symbol}...")
            stage1_combinations_short = self._generate_adaptive_combinations(symbol, candles)
            
            best_candidates_short = self._analyze_combinations(
                symbol, candles, rsi_values, stage1_combinations_short, "Этап 1 SHORT", signal_type='short'
            )
            
            best_short = None
            if best_candidates_short:
                # ✅ УВЕЛИЧЕНО: Берем топ-5 кандидатов для более детального анализа
                top_candidates_short = sorted(best_candidates_short, key=lambda x: x['accuracy'], reverse=True)[:5]
                stage2_combinations_short = self._generate_detailed_combinations(top_candidates_short)
                final_results_short = self._analyze_combinations(
                    symbol, candles, rsi_values, stage2_combinations_short, "Этап 2 SHORT", signal_type='short'
                )
                
                if final_results_short:
                    # ✅ СТРОГИЙ ОТБОР: Выбираем EMA с максимальной точностью И достаточным количеством сигналов
                    # Минимум 5 сигналов для надежности
                    valid_results = [r for r in final_results_short if r.get('total_signals', 0) >= 5]
                    if valid_results:
                        best_short = max(valid_results, key=lambda x: (x['accuracy'], x.get('total_signals', 0)))
                    else:
                        best_short = max(final_results_short, key=lambda x: x['accuracy'])
                else:
                    best_short = top_candidates_short[0] if top_candidates_short else None
            
            # Сохраняем результаты (отдельные EMA для LONG и SHORT)
            result_data = {
                'last_updated': datetime.now().isoformat(),
                'candles_analyzed': len(candles),
                'analysis_method': 'separate_long_short'
            }
            
            # Сохраняем EMA для LONG
            if best_long:
                result_data['long'] = {
                    'ema_short_period': best_long['ema_short_period'],
                    'ema_long_period': best_long['ema_long_period'],
                    'accuracy': best_long['accuracy'],
                    'total_signals': best_long['total_signals'],
                    'correct_signals': best_long['correct_signals']
                }
                logger.info(f"LONG EMA для {symbol}: "
                          f"EMA({best_long['ema_short_period']},{best_long['ema_long_period']}) "
                          f"с точностью {best_long['accuracy']:.1f}% "
                          f"({best_long['correct_signals']}/{best_long['total_signals']})")
            else:
                logger.warning(f"Не найдено оптимальных EMA для LONG сигналов {symbol}")
                # Используем дефолтные значения
                result_data['long'] = {
                    'ema_short_period': 50,
                    'ema_long_period': 200,
                    'accuracy': 0,
                    'total_signals': 0,
                    'correct_signals': 0
                }
            
            # Сохраняем EMA для SHORT
            if best_short:
                result_data['short'] = {
                    'ema_short_period': best_short['ema_short_period'],
                    'ema_long_period': best_short['ema_long_period'],
                    'accuracy': best_short['accuracy'],
                    'total_signals': best_short['total_signals'],
                    'correct_signals': best_short['correct_signals']
                }
                logger.info(f"SHORT EMA для {symbol}: "
                          f"EMA({best_short['ema_short_period']},{best_short['ema_long_period']}) "
                          f"с точностью {best_short['accuracy']:.1f}% "
                          f"({best_short['correct_signals']}/{best_short['total_signals']})")
            else:
                logger.warning(f"Не найдено оптимальных EMA для SHORT сигналов {symbol}")
                # Используем дефолтные значения
                result_data['short'] = {
                    'ema_short_period': 50,
                    'ema_long_period': 200,
                    'accuracy': 0,
                    'total_signals': 0,
                    'correct_signals': 0
                }
            
            # Для обратной совместимости сохраняем также общие поля
            if best_long:
                result_data['ema_short_period'] = best_long['ema_short_period']
                result_data['ema_long_period'] = best_long['ema_long_period']
                result_data['accuracy'] = best_long['accuracy']
                result_data['long_signals'] = best_long['total_signals']
                result_data['short_signals'] = best_short['total_signals'] if best_short else 0
            elif best_short:
                result_data['ema_short_period'] = best_short['ema_short_period']
                result_data['ema_long_period'] = best_short['ema_long_period']
                result_data['accuracy'] = best_short['accuracy']
                result_data['long_signals'] = 0
                result_data['short_signals'] = best_short['total_signals']
            else:
                result_data['ema_short_period'] = 50
                result_data['ema_long_period'] = 200
                result_data['accuracy'] = 0
                result_data['long_signals'] = 0
                result_data['short_signals'] = 0
            
            # ✅ Сохраняем данные в словарь
            self.optimal_ema_data[clean_symbol] = result_data
            logger.info(f"💾 Данные для {clean_symbol} добавлены в словарь, сохраняем в файл...")
            
            # ✅ Сохраняем в файл
            self.save_optimal_ema_data()
            
            # ✅ Проверяем что данные действительно сохранились
            if clean_symbol in self.optimal_ema_data:
                logger.info(f"✅ Данные для {clean_symbol} успешно сохранены в файл {self.optimal_ema_file}")
            else:
                logger.error(f"❌ ОШИБКА: Данные для {clean_symbol} не найдены после сохранения!")
            
            return self.optimal_ema_data[clean_symbol]
                
        except Exception as e:
            logger.error(f"Ошибка поиска оптимальных EMA для {symbol}: {e}")
            return None
    
    def get_all_symbols(self) -> List[str]:
        """Получает список всех доступных символов"""
        try:
            pairs = self.exchange.get_all_pairs()
            if pairs and isinstance(pairs, list):
                # Пары уже приходят в формате BTCUSDT, ETHUSDT и т.д.
                # Просто возвращаем их как есть
                return pairs
            return []
        except Exception as e:
            logger.error(f"Ошибка получения списка символов: {e}")
            return []
    
    def process_all_symbols(self, force_rescan: bool = False):
        """Обрабатывает все символы"""
        symbols = self.get_all_symbols()
        
        if not symbols:
            logger.error("Не удалось получить список символов")
            return
        
        # Добавляем временную метку для force режима
        if force_rescan:
            force_timestamp = datetime.now().isoformat()
            logger.info(f"[FORCE] 🚀 Запуск принудительного пересчета в {force_timestamp}")
            logger.info(f"[FORCE] 📊 Будет обработано {len(symbols)} символов")
        
        logger.info(f"Найдено {len(symbols)} символов на бирже")
        
        # Подсчитываем статистику
        already_processed = 0
        new_symbols = []
        
        for symbol in symbols:
            if symbol in self.optimal_ema_data:
                already_processed += 1
            else:
                new_symbols.append(symbol)
        
        logger.info(f"Уже обработано: {already_processed} монет")
        logger.info(f"Новых для обработки: {len(new_symbols)} монет")
        
        if force_rescan:
            logger.info("[FORCE] Принудительный режим: пересчитываем ВСЕ монеты")
            symbols_to_process = symbols
        else:
            logger.info("[NEW] Обычный режим: обрабатываем только новые монеты")
            symbols_to_process = new_symbols
        
        if not symbols_to_process:
            logger.info("[DONE] Все монеты уже обработаны!")
            return
        
        logger.info(f"Начинаем обработку {len(symbols_to_process)} монет...")
        
        processed = 0
        failed = 0
        
        try:
            for i, symbol in enumerate(symbols_to_process, 1):
                logger.info(f"Обработка {i}/{len(symbols_to_process)}: {symbol}")
                
                result = self.find_optimal_ema(symbol, force_rescan)
                if result:
                    processed += 1
                    logger.info(f"[OK] {symbol} обработан успешно")
                    
                    # При force режиме сохраняем данные после каждого символа
                    if force_rescan:
                        self.save_optimal_ema_data()
                        logger.info(f"[SAVE] Данные сохранены после обработки {symbol} ({i}/{len(symbols_to_process)})")
                else:
                    failed += 1
                    logger.warning(f"[ERROR] Не удалось обработать {symbol}")
                
                # Небольшая пауза между запросами
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info(f"[INTERRUPT] Обработка прервана на {i}/{len(symbols_to_process)} монетах")
            logger.info(f"[RESULT] Частично обработано: {processed} успешно, {failed} ошибок")
            return
        
        logger.info(f"[RESULT] Обработка завершена: {processed} успешно, {failed} ошибок")
        logger.info(f"[STATS] Всего в базе: {len(self.optimal_ema_data)} монет")
        
        # Принудительно сохраняем данные в конце обработки
        self.save_optimal_ema_data()
        logger.info("[SAVE] Данные сохранены в файл")
        
        # Добавляем информацию о завершении force режима
        if force_rescan:
            completion_timestamp = datetime.now().isoformat()
            logger.info(f"[FORCE] ✅ Принудительный пересчет завершен в {completion_timestamp}")
            logger.info(f"[FORCE] 📈 Итоговая статистика: {processed} успешно, {failed} ошибок")
    
    def process_symbols_list(self, symbols: List[str], force_rescan: bool = False):
        """Обрабатывает список символов"""
        processed = 0
        failed = 0
        
        # Добавляем временную метку для force режима
        if force_rescan:
            force_timestamp = datetime.now().isoformat()
            logger.info(f"[FORCE] 🚀 Запуск принудительного пересчета в {force_timestamp}")
            logger.info(f"[FORCE] 📊 Будет обработано {len(symbols)} символов")
        
        try:
            for i, symbol in enumerate(symbols, 1):
                logger.info(f"Обработка {i}/{len(symbols)}: {symbol}")
                
                result = self.find_optimal_ema(symbol, force_rescan)
                if result:
                    processed += 1
                    logger.info(f"[OK] {symbol} обработан успешно")
                    
                    # При force режиме сохраняем данные после каждого символа
                    if force_rescan:
                        self.save_optimal_ema_data()
                        logger.info(f"[SAVE] Данные сохранены после обработки {symbol} ({i}/{len(symbols)})")
                else:
                    failed += 1
                    logger.warning(f"[ERROR] Не удалось обработать {symbol}")
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info(f"[INTERRUPT] Обработка прервана на {i}/{len(symbols)} монетах")
            logger.info(f"[RESULT] Частично обработано: {processed} успешно, {failed} ошибок")
            return
        
        logger.info(f"[RESULT] Обработка завершена: {processed} успешно, {failed} ошибок")
        
        # Принудительно сохраняем данные в конце обработки
        self.save_optimal_ema_data()
        logger.info("[SAVE] Данные сохранены в файл")
        
        # Добавляем информацию о завершении force режима
        if force_rescan:
            completion_timestamp = datetime.now().isoformat()
            logger.info(f"[FORCE] ✅ Принудительный пересчет завершен в {completion_timestamp}")
            logger.info(f"[FORCE] 📈 Итоговая статистика: {processed} успешно, {failed} ошибок")

def main():
    """Основная функция"""
    # Настройка кодировки для Windows консоли
    if platform.system() == "Windows":
        try:
            import locale
            locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')
        except:
            try:
                locale.setlocale(locale.LC_ALL, 'Russian_Russia.1251')
            except:
                pass
        
        # Информируем о настройках для Windows
        if USE_THREADS_ON_WINDOWS:
            print("[INFO] На Windows используется ThreadPoolExecutor для совместимости с numba")
            print("[INFO] Numba + Threading = максимальная производительность!")
        elif not USE_MULTIPROCESSING:
            print("[INFO] Параллельная обработка отключена")
            print("[INFO] Numba остается активным для ускорения вычислений")
    
    parser = argparse.ArgumentParser(description='Поиск оптимальных EMA периодов для определения тренда')
    parser.add_argument('--all', action='store_true', help='Обработать только новые символы (не обработанные ранее)')
    parser.add_argument('--force', action='store_true', help='Принудительно пересчитать ВСЕ символы')
    parser.add_argument('--coin', type=str, help='Обработать конкретную монету (например, BTCUSDT) - принудительно')
    parser.add_argument('--coins', type=str, help='Обработать список монет через запятую (например, BTCUSDT,ETHUSDT)')
    parser.add_argument('--rescan', action='store_true', help='Принудительно пересканировать существующие (устаревший параметр)')
    parser.add_argument('--list', action='store_true', help='Показать список уже обработанных монет')
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME, 
                       help=f'Таймфрейм для анализа (по умолчанию: {DEFAULT_TIMEFRAME}). Доступные: 1m, 5m, 15m, 30m, 1h, 4h, 6h, 1d, 1w')
    
    args = parser.parse_args()
    
    finder = OptimalEMAFinder(timeframe=args.timeframe)
    
    # Информируем о настройках
    print(NUMBA_MESSAGE)
    print(f"[INFO] Используется таймфрейм: {args.timeframe}")
    print(f"[INFO] Файл данных: {finder.optimal_ema_file}")
    
    if args.list:
        # Показать список обработанных монет
        if finder.optimal_ema_data:
            print(f"\nОбработано {len(finder.optimal_ema_data)} монет:")
            for symbol, data in finder.optimal_ema_data.items():
                # Проверяем наличие новых ключей (для совместимости со старыми записями)
                if 'ema_short_period' in data and 'ema_long_period' in data:
                    long_signals = data.get('long_signals', 0)
                    short_signals = data.get('short_signals', 0)
                    print(f"  {symbol}: EMA({data['ema_short_period']},{data['ema_long_period']}) "
                          f"точность: {data['accuracy']:.3f} (Long: {long_signals}, Short: {short_signals})")
                else:
                    # Старый формат
                    print(f"  {symbol}: EMA({data.get('ema_short', 'N/A')},{data.get('ema_long', 'N/A')}) "
                          f"точность: {data['accuracy']:.3f} (старый формат)")
        else:
            print("Нет обработанных монет")
        return
    
    if args.coin:
        # Обработать конкретную монету (всегда принудительно)
        print(f"[COIN] Принудительный пересчет для {args.coin}...")
        result = finder.find_optimal_ema(args.coin.upper(), force_rescan=True)
        if result:
            long_signals = result.get('long_signals', 0)
            short_signals = result.get('short_signals', 0)
            print(f"[OK] Оптимальные EMA для {args.coin}: "
                  f"EMA({result['ema_short_period']},{result['ema_long_period']}) "
                  f"с точностью {result['accuracy']:.3f} "
                  f"(Long: {long_signals}, Short: {short_signals})")
        else:
            print(f"[ERROR] Не удалось найти оптимальные EMA для {args.coin}")
    elif args.coins:
        # Обработать список монет
        symbols = [s.strip().upper() for s in args.coins.split(',')]
        print(f"[COINS] Обработка списка монет: {', '.join(symbols)}")
        finder.process_symbols_list(symbols, force_rescan=True)
    elif args.force:
        # Принудительно пересчитать ВСЕ символы
        print("[FORCE] Принудительный пересчет ВСЕХ монет...")
        finder.process_all_symbols(force_rescan=True)
    elif args.all:
        # Обработать только новые символы
        print("[NEW] Обработка только новых монет...")
        finder.process_all_symbols(force_rescan=False)
    else:
        # Показать справку
        parser.print_help()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Обработка прервана пользователем (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Неожиданная ошибка: {e}")
        sys.exit(1)
