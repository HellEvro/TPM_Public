#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль отслеживания использованных параметров обучения

Отслеживает какие комбинации параметров уже использовались для обучения,
чтобы избежать дубликатов и знать когда все комбинации исчерпаны.
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from threading import RLock

logger = logging.getLogger('AI.ParameterTracker')


class AIParameterTracker:
    """
    Отслеживает использованные комбинации параметров обучения
    """
    
    def __init__(self, data_dir: str = 'data/ai'):
        self.data_dir = data_dir
        self.lock = RLock()
        self.used_params_file = os.path.join(data_dir, 'used_training_parameters.json')
        
        # Создаем директорию если её нет
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Загружаем использованные параметры
        self.used_params = self._load_used_params()
        
        # Вычисляем общее количество возможных комбинаций
        self.total_combinations = self._calculate_total_combinations()
        
        logger.debug(f"📊 Загружено {len(self.used_params)} использованных комбинаций из {self.total_combinations} возможных")
    
    def _load_used_params(self) -> Dict:
        """Загрузить использованные параметры из файла"""
        try:
            if os.path.exists(self.used_params_file):
                with open(self.used_params_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('used_combinations', {})
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки использованных параметров: {e}")
        
        return {}
    
    def _save_used_params(self):
        """Сохранить использованные параметры в файл"""
        try:
            with self.lock:
                data = {
                    'last_update': datetime.now().isoformat(),
                    'total_combinations': self.total_combinations,
                    'used_count': len(self.used_params),
                    'used_combinations': self.used_params
                }
                
                with open(self.used_params_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения использованных параметров: {e}")
    
    def _calculate_total_combinations(self) -> int:
        """
        Вычислить общее количество возможных комбинаций параметров
        
        Параметры RSI с вариацией:
        - RSI_OVERSOLD: 20-35 (variation_range=3) → ~16 значений
        - RSI_OVERBOUGHT: 65-80 (variation_range=3) → ~16 значений
        - RSI_EXIT_LONG_WITH_TREND: 55-70 (±5) → ~16 значений
        - RSI_EXIT_LONG_AGAINST_TREND: 50-65 (±5) → ~16 значений
        - RSI_EXIT_SHORT_WITH_TREND: 25-40 (±5) → ~16 значений
        - RSI_EXIT_SHORT_AGAINST_TREND: 30-45 (±5) → ~16 значений
        
        Итого: 16^6 = 16,777,216 комбинаций (теоретически)
        Но на практике меньше из-за ограничений и вариации
        """
        # Реальные диапазоны с учетом вариации
        oversold_range = 16  # 20-35 с шагом ~1
        overbought_range = 16  # 65-80 с шагом ~1
        exit_long_with_range = 16  # 55-70 с шагом ~1
        exit_long_against_range = 16  # 50-65 с шагом ~1
        exit_short_with_range = 16  # 25-40 с шагом ~1
        exit_short_against_range = 16  # 30-45 с шагом ~1
        
        total = (oversold_range * overbought_range * 
                exit_long_with_range * exit_long_against_range *
                exit_short_with_range * exit_short_against_range)
        
        return total
    
    def _generate_param_hash(self, rsi_params: Dict) -> str:
        """
        Генерировать уникальный хеш для комбинации параметров
        
        Args:
            rsi_params: Словарь с параметрами RSI
        
        Returns:
            Хеш строку для идентификации комбинации
        """
        # Создаем строку из параметров для хеширования
        param_string = (
            f"{rsi_params.get('oversold', 0)}_"
            f"{rsi_params.get('overbought', 0)}_"
            f"{rsi_params.get('exit_long_with_trend', 0)}_"
            f"{rsi_params.get('exit_long_against_trend', 0)}_"
            f"{rsi_params.get('exit_short_with_trend', 0)}_"
            f"{rsi_params.get('exit_short_against_trend', 0)}"
        )
        
        # Генерируем MD5 хеш
        return hashlib.md5(param_string.encode()).hexdigest()
    
    def is_params_used(self, rsi_params: Dict) -> bool:
        """
        Проверить, использовались ли уже эти параметры
        
        Args:
            rsi_params: Словарь с параметрами RSI
        
        Returns:
            True если параметры уже использовались
        """
        param_hash = self._generate_param_hash(rsi_params)
        return param_hash in self.used_params
    
    def mark_params_used(self, rsi_params: Dict, training_seed: int, 
                         win_rate: float = 0.0, symbol: Optional[str] = None,
                         total_pnl: float = 0.0, signal_accuracy: float = 0.0,
                         trades_count: int = 0):
        """
        Отметить параметры как использованные и сохранить рейтинг
        
        Args:
            rsi_params: Словарь с параметрами RSI
            training_seed: Seed обучения
            win_rate: Win Rate достигнутый с этими параметрами
            symbol: Символ монеты (если обучение для конкретной монеты)
            total_pnl: Общий PnL достигнутый с этими параметрами
            signal_accuracy: Точность предсказания сигналов
            trades_count: Количество сделок
        """
        param_hash = self._generate_param_hash(rsi_params)
        
        # Вычисляем рейтинг параметров (комплексная метрика)
        rating = self.calculate_rating(win_rate, total_pnl, signal_accuracy, trades_count)
        
        with self.lock:
            # Если параметры уже использовались - обновляем если новый результат лучше
            if param_hash in self.used_params:
                existing = self.used_params[param_hash]
                existing_rating = existing.get('rating', 0)
                
                # Обновляем только если новый результат лучше
                if rating > existing_rating:
                    existing.update({
                        'rsi_params': rsi_params,
                        'training_seed': training_seed,
                        'used_at': datetime.now().isoformat(),
                        'win_rate': win_rate,
                        'total_pnl': total_pnl,
                        'signal_accuracy': signal_accuracy,
                        'trades_count': trades_count,
                        'rating': rating,
                        'symbol': symbol,
                        'update_count': existing.get('update_count', 0) + 1
                    })
                    logger.debug(f"✅ Обновлены параметры с лучшим рейтингом: {rating:.2f} (было: {existing_rating:.2f})")
            else:
                # Новые параметры
                self.used_params[param_hash] = {
                    'rsi_params': rsi_params,
                    'training_seed': training_seed,
                    'used_at': datetime.now().isoformat(),
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'signal_accuracy': signal_accuracy,
                    'trades_count': trades_count,
                    'rating': rating,
                    'symbol': symbol,
                    'update_count': 1
                }
            
            self._save_used_params()
            
            # Сохраняем лучшие параметры для монеты (если указана)
            if symbol:
                self._update_best_params_for_symbol(symbol, rsi_params, rating, win_rate, total_pnl)
    
    def calculate_rating(self, win_rate: float, total_pnl: float, 
                         signal_accuracy: float, trades_count: int) -> float:
        """
        Вычислить рейтинг параметров на основе метрик
        
        Рейтинг учитывает:
        - Win Rate (вес 40%) - основной показатель успешности
        - Signal Accuracy (вес 30%) - точность предсказаний
        - Total PnL (вес 20%) - прибыльность
        - Trades Count (вес 10%) - количество сделок (больше = надежнее)
        
        Returns:
            Рейтинг от 0 до 100
        """
        # Нормализуем метрики
        win_rate_score = min(win_rate, 100) / 100.0  # 0-1
        accuracy_score = min(signal_accuracy * 100, 100) / 100.0  # 0-1
        pnl_score = min(max(total_pnl / 1000.0, 0), 1)  # Нормализуем PnL (1000 USDT = 1.0)
        trades_score = min(trades_count / 100.0, 1)  # 100 сделок = 1.0
        
        # Взвешенная сумма
        rating = (
            win_rate_score * 0.4 +
            accuracy_score * 0.3 +
            pnl_score * 0.2 +
            trades_score * 0.1
        ) * 100
        
        return rating
    
    def _update_best_params_for_symbol(self, symbol: str, rsi_params: Dict, 
                                      rating: float, win_rate: float, total_pnl: float):
        """
        Обновить лучшие параметры для конкретной монеты
        
        Args:
            symbol: Символ монеты
            rsi_params: Параметры RSI
            rating: Рейтинг параметров
            win_rate: Win Rate
            total_pnl: Total PnL
        """
        best_params_file = os.path.join(self.data_dir, 'best_params_per_symbol.json')
        
        try:
            # Загружаем существующие лучшие параметры
            if os.path.exists(best_params_file):
                with open(best_params_file, 'r', encoding='utf-8') as f:
                    best_params = json.load(f)
            else:
                best_params = {}
            
            # Проверяем нужно ли обновить лучшие параметры для этой монеты
            if symbol not in best_params or rating > best_params[symbol].get('rating', 0):
                best_params[symbol] = {
                    'rsi_params': rsi_params,
                    'rating': rating,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'updated_at': datetime.now().isoformat()
                }
                
                # Сохраняем
                with open(best_params_file, 'w', encoding='utf-8') as f:
                    json.dump(best_params, f, indent=2, ensure_ascii=False)
                
                logger.debug(f"✅ Обновлены лучшие параметры для {symbol}: рейтинг {rating:.2f}, Win Rate {win_rate:.1f}%")
        except Exception as e:
            logger.debug(f"⚠️ Ошибка обновления лучших параметров для {symbol}: {e}")
    
    def get_usage_stats(self) -> Dict:
        """
        Получить статистику использования параметров
        
        Returns:
            Словарь со статистикой
        """
        used_count = len(self.used_params)
        total = self.total_combinations
        percentage = (used_count / total * 100) if total > 0 else 0
        
        return {
            'used_combinations': used_count,
            'total_combinations': total,
            'remaining_combinations': total - used_count,
            'usage_percentage': percentage,
            'is_exhausted': used_count >= total * 0.95  # Считаем исчерпанным если использовано 95%
        }
    
    def get_best_params(self, limit: int = 10, min_win_rate: float = 80.0) -> List[Dict]:
        """
        Получить лучшие использованные параметры (по рейтингу)
        
        Args:
            limit: Количество лучших комбинаций
            min_win_rate: Минимальный Win Rate для включения
        
        Returns:
            Список лучших комбинаций параметров
        """
        # Фильтруем по минимальному Win Rate и сортируем по рейтингу
        filtered_params = [
            p for p in self.used_params.values()
            if p.get('win_rate', 0) >= min_win_rate
        ]
        
        sorted_params = sorted(
            filtered_params,
            key=lambda x: x.get('rating', 0),
            reverse=True
        )
        
        return sorted_params[:limit]
    
    def get_best_params_for_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Получить лучшие параметры для конкретной монеты
        
        Args:
            symbol: Символ монеты
        
        Returns:
            Словарь с лучшими параметрами или None
        """
        best_params_file = os.path.join(self.data_dir, 'best_params_per_symbol.json')
        
        try:
            if os.path.exists(best_params_file):
                with open(best_params_file, 'r', encoding='utf-8') as f:
                    best_params = json.load(f)
                    return best_params.get(symbol)
        except Exception as e:
            logger.debug(f"⚠️ Ошибка загрузки лучших параметров для {symbol}: {e}")
        
        return None
    
    def get_all_best_params_per_symbol(self) -> Dict[str, Dict]:
        """
        Получить лучшие параметры для всех монет
        
        Returns:
            Словарь {symbol: best_params}
        """
        best_params_file = os.path.join(self.data_dir, 'best_params_per_symbol.json')
        
        try:
            if os.path.exists(best_params_file):
                with open(best_params_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"⚠️ Ошибка загрузки лучших параметров: {e}")
        
        return {}
    
    def get_unused_params_suggestion(self, base_params: Dict, 
                                     variation_range: int = 3) -> Optional[Dict]:
        """
        Предложить неиспользованные параметры на основе базовых
        
        Пробует найти комбинацию параметров которая еще не использовалась
        
        Args:
            base_params: Базовые параметры
            variation_range: Диапазон вариации
        
        Returns:
            Словарь с неиспользованными параметрами или None если не найдено
        """
        import random
        
        # Пробуем найти неиспользованную комбинацию (до 100 попыток)
        max_attempts = 100
        for attempt in range(max_attempts):
            # Генерируем случайные параметры
            rsi_params = {
                'oversold': max(20, min(35, 
                    base_params.get('oversold', 29) + random.randint(-variation_range, variation_range))),
                'overbought': max(65, min(80,
                    base_params.get('overbought', 71) + random.randint(-variation_range, variation_range))),
                'exit_long_with_trend': max(55, min(70,
                    base_params.get('exit_long_with_trend', 65) + random.randint(-5, 5))),
                'exit_long_against_trend': max(50, min(65,
                    base_params.get('exit_long_against_trend', 60) + random.randint(-5, 5))),
                'exit_short_with_trend': max(25, min(40,
                    base_params.get('exit_short_with_trend', 35) + random.randint(-5, 5))),
                'exit_short_against_trend': max(30, min(45,
                    base_params.get('exit_short_against_trend', 40) + random.randint(-5, 5)))
            }
            
            # Проверяем, использовались ли эти параметры
            if not self.is_params_used(rsi_params):
                return rsi_params
        
        # Если не нашли за 100 попыток - возвращаем None
        logger.warning("⚠️ Не удалось найти неиспользованные параметры за 100 попыток")
        return None
    
    def reset_used_params(self, confirm: bool = False):
        """
        Сбросить список использованных параметров
        
        Args:
            confirm: Подтверждение сброса (для безопасности)
        """
        if not confirm:
            logger.warning("⚠️ Для сброса параметров требуется подтверждение (confirm=True)")
            return
        
        with self.lock:
            self.used_params = {}
            self._save_used_params()
            logger.info("✅ Список использованных параметров сброшен")

