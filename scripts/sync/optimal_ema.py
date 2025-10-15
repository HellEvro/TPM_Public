#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для поиска оптимальных EMA периодов для определения тренда
для каждой монеты на основе корреляции с RSI 30/70.

Логика работы:
1. Ищет моменты входа RSI в зоны перепроданности (≤30) и перекупленности (≥70)
2. Проверяет, подтверждается ли соответствующий тренд EMA в течение следующих 5 периодов
3. Для LONG: RSI ≤30, затем EMA должны показать восходящий тренд
4. Для SHORT: RSI ≥70, затем EMA должны показать нисходящий тренд
5. Находит EMA периоды с максимальной точностью таких подтверждений
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

# Настройка логирования
def setup_logging():
    """Настройка логирования с правильной кодировкой"""
    # Создаем директорию для логов если её нет
    os.makedirs('logs', exist_ok=True)
    
    # Настраиваем форматтер
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Файловый хендлер с UTF-8
    file_handler = logging.FileHandler('logs/optimal_ema.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Консольный хендлер
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Настраиваем логгер
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# Константы
OPTIMAL_EMA_BASE_FILE = 'data/optimal_ema'  # Базовое имя файла
EMA_SHORT_RANGE = (5, 200)  # Короткая EMA
EMA_LONG_RANGE = (50, 500)  # Длинная EMA
RSI_OVERSOLD = 29  # Строгие условия для подтверждения тренда
RSI_OVERBOUGHT = 71
# Используем multiprocessing с безопасной инициализацией
MAX_WORKERS = mp.cpu_count()
MIN_CANDLES_FOR_ANALYSIS = 200
MAX_CANDLES_TO_REQUEST = 5000
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
def analyze_ema_combination_numba(prices, rsi_values, ema_short_period, ema_long_period):
    """Анализ EMA трендов с подтверждением RSI"""
    n = len(prices)
    if n < max(ema_short_period, ema_long_period) + 100:
        return 0.0, 0, 0, 0, 0
    
    # Вычисляем EMA
    ema_short = calculate_ema_numba(prices, ema_short_period)
    ema_long = calculate_ema_numba(prices, ema_long_period)
    
    # Находим общую длину для анализа
    min_length = min(len(rsi_values), len(ema_short), len(ema_long))
    start_idx = max(ema_short_period, ema_long_period) - 1
    
    if min_length - start_idx < 100:
        return 0.0, 0, 0, 0, 0
    
    # Счетчики для разных типов сигналов
    total_long_signals = 0
    correct_long_signals = 0
    total_short_signals = 0
    correct_short_signals = 0
    
    for i in range(start_idx, min_length - 10):
        rsi = rsi_values[i]
        
        # Проверяем тренды EMA
        if i > start_idx:
            # Восходящий тренд EMA (короткая выше длинной)
            if ema_short[i] > ema_long[i]:
                # Проверяем RSI на перепроданность для подтверждения
                if rsi <= 29:  # Строгие условия для перепроданности
                    total_long_signals += 1
                    
                    # Проверяем успешность в следующих 10 периодах
                    success = False
                    for j in range(1, 11):
                        if i + j < min_length:
                            if ema_short[i + j] > ema_long[i + j]:
                                success = True
                                break
                    
                    if success:
                        correct_long_signals += 1
            
            # Нисходящий тренд EMA (короткая ниже длинной)
            elif ema_short[i] < ema_long[i]:
                # Проверяем RSI на перекупленность для подтверждения
                if rsi >= 71:  # Строгие условия для перекупленности
                    total_short_signals += 1
                    
                    # Проверяем успешность в следующих 10 периодах
                    success = False
                    for j in range(1, 11):
                        if i + j < min_length:
                            if ema_short[i + j] < ema_long[i + j]:
                                success = True
                                break
                    
                    if success:
                        correct_short_signals += 1
    
    total_signals = total_long_signals + total_short_signals
    correct_signals = correct_long_signals + correct_short_signals
    
    if total_signals == 0:
        return 0.0, 0, 0, 0, 0
    
    accuracy = (correct_signals / total_signals) * 100
    return accuracy, total_signals, correct_signals, total_long_signals, total_short_signals

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
    symbol, candles, rsi_values, ema_short_period, ema_long_period = args
    
    try:
        # Конвертируем в numpy массивы
        prices = np.array([float(candle['close']) for candle in candles], dtype=np.float64)
        
        # Анализируем комбинацию с numba (теперь возвращает 5 значений)
        accuracy, total_signals, correct_signals, long_signals, short_signals = analyze_ema_combination_numba(
            prices, rsi_values, ema_short_period, ema_long_period
        )
        
        return {
            'accuracy': accuracy,
            'total_signals': total_signals,
            'correct_signals': correct_signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'ema_short_period': ema_short_period,
            'ema_long_period': ema_long_period
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
            if os.path.exists(self.optimal_ema_file):
                with open(self.optimal_ema_file, 'r', encoding='utf-8') as f:
                    self.optimal_ema_data = json.load(f)
                logger.info(f"Загружено {len(self.optimal_ema_data)} записей об оптимальных EMA для таймфрейма {self.timeframe}")
            else:
                self.optimal_ema_data = {}
                logger.info(f"Файл {self.optimal_ema_file} не найден, создаем новую базу данных")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных EMA: {e}")
            self.optimal_ema_data = {}
    
    def save_optimal_ema_data(self):
        """Сохраняет данные об оптимальных EMA в файл"""
        try:
            os.makedirs(os.path.dirname(self.optimal_ema_file), exist_ok=True)
            with open(self.optimal_ema_file, 'w', encoding='utf-8') as f:
                json.dump(self.optimal_ema_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Сохранено {len(self.optimal_ema_data)} записей об оптимальных EMA для таймфрейма {self.timeframe} в файл {self.optimal_ema_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения данных EMA: {e}")
    
    def get_candles_data(self, symbol: str) -> Optional[List[Dict]]:
        """Получает данные свечей для символа с поддержкой пагинации для получения большего количества данных"""
        try:
            if not self.exchange:
                self._init_exchange()
                if not self.exchange:
                    raise Exception("Не удалось инициализировать exchange")
            
            # Очищаем символ от USDT если есть
            clean_symbol = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
            
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
            
            if candles and len(candles) >= MIN_CANDLES_FOR_ANALYSIS:
                logger.info(f"Получено {len(candles)} свечей для {symbol}")
                return candles
            else:
                logger.warning(f"Недостаточно свечей для {symbol}: {len(candles) if candles else 0}/{MIN_CANDLES_FOR_ANALYSIS}")
                return None
                
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
            
            logger.info(f"Запрашиваем расширенные данные для {symbol} (цель: {target_candles} свечей)")
            
            while len(all_candles) < target_candles:
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
                        
                        # Конвертируем в наш формат
                        batch_candles = []
                        for k in klines:
                            candle = {
                                'time': int(k[0]),
                                'open': float(k[1]),
                                'high': float(k[2]),
                                'low': float(k[3]),
                                'close': float(k[4]),
                                'volume': float(k[5])
                            }
                            batch_candles.append(candle)
                        
                        # Добавляем к общему списку
                        all_candles.extend(batch_candles)
                        
                        # Обновляем end_time для следующего запроса (берем время первой свечи - 1)
                        end_time = int(klines[0][0]) - 1
                        
                        logger.debug(f"Получено {len(batch_candles)} свечей, всего: {len(all_candles)}")
                        
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
        """Генерирует адаптивные комбинации EMA на основе характеристик монеты"""
        prices = np.array([float(candle['close']) for candle in candles], dtype=np.float64)
        volatility = self._calculate_volatility(prices)
        
        combinations = []
        
        # Адаптивные диапазоны на основе волатильности
        if volatility > 0.05:  # Высокая волатильность (мемкоины, новые монеты)
            ema_short_range = (5, 50)
            ema_long_range = (20, 150)
            short_step = 3
            long_step = 10
        elif volatility > 0.03:  # Средняя волатильность (альткоины)
            ema_short_range = (5, 100)
            ema_long_range = (30, 200)
            short_step = 5
            long_step = 15
        else:  # Низкая волатильность (BTC, стабильные монеты)
            ema_short_range = (10, 150)
            ema_long_range = (50, 300)
            short_step = 10
            long_step = 25
        
        # Генерируем комбинации с адаптивными шагами
        for ema_short in range(ema_short_range[0], ema_short_range[1] + 1, short_step):
            for ema_long in range(ema_long_range[0], ema_long_range[1] + 1, long_step):
                if ema_short < ema_long:
                    combinations.append((ema_short, ema_long))
        
        # Добавляем универсальные комбинации для 6-часовых свечей
        universal_combinations = [
            (5, 15), (9, 21), (12, 26), (21, 55), (34, 89),
            (20, 50), (30, 70), (50, 200), (15, 45), (25, 75)
        ]
        
        for combo in universal_combinations:
            if combo not in combinations:
                combinations.append(combo)
        
        logger.info(f"Сгенерировано {len(combinations)} адаптивных комбинаций для {symbol} (волатильность: {volatility:.3f})")
        return combinations
    
    def _generate_detailed_combinations(self, best_candidates: List[Dict]) -> List[Tuple[int, int]]:
        """Генерирует детальные комбинации вокруг лучших кандидатов"""
        combinations = []
        
        for candidate in best_candidates:
            ema_short = candidate['ema_short_period']
            ema_long = candidate['ema_long_period']
            
            # Добавляем комбинации в окрестности лучших
            for short_offset in range(-5, 6, 2):
                for long_offset in range(-10, 11, 5):
                    new_short = ema_short + short_offset
                    new_long = ema_long + long_offset
                    
                    if 5 <= new_short < new_long <= 500:
                        combinations.append((new_short, new_long))
        
        # Убираем дубликаты
        combinations = list(set(combinations))
        
        logger.info(f"Сгенерировано {len(combinations)} детальных комбинаций")
        return combinations
    
    def _analyze_combinations(self, symbol: str, candles: List[Dict], rsi_values: np.ndarray, 
                            combinations: List[Tuple[int, int]], stage_name: str) -> List[Dict]:
        """Анализирует список комбинаций EMA"""
        if not combinations:
            return []
        
        best_accuracy = 0
        best_combination = None
        all_results = []
        
        # Подготавливаем аргументы для параллельной обработки
        args_list = []
        for ema_short, ema_long in combinations:
            args_list.append((symbol, candles, rsi_values, ema_short, ema_long))
        
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
            
            # Вычисляем RSI один раз для всех комбинаций
            prices = np.array([float(candle['close']) for candle in candles], dtype=np.float64)
            rsi_values = calculate_rsi_numba(prices, 14)
            
            # ЭТАП 1: Быстрый скрининг с адаптивными диапазонами
            logger.info(f"Этап 1: Быстрый скрининг {symbol}...")
            stage1_combinations = self._generate_adaptive_combinations(symbol, candles)
            
            best_candidates = self._analyze_combinations(
                symbol, candles, rsi_values, stage1_combinations, "Этап 1"
            )
            
            if not best_candidates:
                logger.warning(f"Не найдено подходящих EMA для {symbol} на этапе 1")
                return None
            
            # Берем топ-3 кандидата для детального анализа
            top_candidates = sorted(best_candidates, key=lambda x: x['accuracy'], reverse=True)[:3]
            
            # ЭТАП 2: Детальный анализ лучших кандидатов
            logger.info(f"Этап 2: Детальный анализ {symbol}...")
            stage2_combinations = self._generate_detailed_combinations(top_candidates)
            
            final_results = self._analyze_combinations(
                symbol, candles, rsi_values, stage2_combinations, "Этап 2"
            )
            
            if not final_results:
                # Если детальный анализ не дал результатов, используем лучший из этапа 1
                best_combination = top_candidates[0]
            else:
                best_combination = max(final_results, key=lambda x: x['accuracy'])
            
            # Сохраняем результат
            self.optimal_ema_data[clean_symbol] = {
                'ema_short_period': best_combination['ema_short_period'],
                'ema_long_period': best_combination['ema_long_period'],
                'accuracy': best_combination['accuracy'],
                'total_signals': best_combination['total_signals'],
                'correct_signals': best_combination['correct_signals'],
                'long_signals': best_combination['long_signals'],
                'short_signals': best_combination['short_signals'],
                'last_updated': datetime.now().isoformat(),
                'candles_analyzed': len(candles),
                'analysis_method': 'trend_confirmation'
            }
            
            self.save_optimal_ema_data()
            
            logger.info(f"Оптимальные EMA для {symbol}: "
                      f"EMA({best_combination['ema_short_period']},{best_combination['ema_long_period']}) "
                      f"с точностью {best_combination['accuracy']:.1f}% "
                      f"({best_combination['correct_signals']}/{best_combination['total_signals']}) "
                      f"Long: {best_combination['long_signals']}, Short: {best_combination['short_signals']}")
            
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
