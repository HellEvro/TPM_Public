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
from typing import Dict, List, Optional, Tuple, Any
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
        
        # Подключаемся к БД
        try:
            from bot_engine.ai.ai_database import get_ai_database
            self.ai_db = get_ai_database()
            logger.debug("✅ AI Database подключена для AIParameterTracker")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось подключиться к AI Database: {e}")
            self.ai_db = None
        
        # Вычисляем общее количество возможных комбинаций
        self.total_combinations = self._calculate_total_combinations()
    
    def _get_used_params_dict(self) -> Dict:
        """Получить словарь использованных параметров из БД (для обратной совместимости)"""
        if not self.ai_db:
            return {}
        try:
            # Загружаем все использованные параметры
            count = self.ai_db.count_used_training_parameters()
            # Для обратной совместимости создаем словарь {hash: data}
            # Но это неэффективно для больших объемов, поэтому используем БД напрямую
            return {}  # Возвращаем пустой словарь, используем БД напрямую
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки использованных параметров: {e}")
            return {}
    
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
        if not self.ai_db:
            return False
        param_hash = self._generate_param_hash(rsi_params)
        used_param = self.ai_db.get_used_training_parameter(param_hash)
        return used_param is not None
    
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
        if not self.ai_db:
            return  # Не логируем предупреждение - это нормально если БД недоступна
        
        param_hash = self._generate_param_hash(rsi_params)
        
        # Вычисляем рейтинг параметров (комплексная метрика)
        rating = self.calculate_rating(win_rate, total_pnl, signal_accuracy, trades_count)
        
        # Убираем блокировку - SQLite WAL режим позволяет параллельные записи
        # Сохраняем в БД (метод сам проверит и обновит если нужно)
        param_id = self.ai_db.save_used_training_parameter(
            param_hash, rsi_params, training_seed,
            win_rate, total_pnl, signal_accuracy, trades_count,
            rating, symbol
        )
        
        # Сохраняем лучшие параметры для монеты (если указана) - делаем это в той же транзакции
        if param_id and symbol:
            # Проверяем нужно ли обновить лучшие параметры (быстрая проверка без блокировки)
            current_best = self.ai_db.get_best_params_for_symbol(symbol)
            if not current_best or rating > current_best.get('rating', 0):
                self.ai_db.save_best_params_for_symbol(symbol, rsi_params, rating, win_rate, total_pnl)
    
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
        if not self.ai_db:
            return
        
        try:
            # Проверяем текущие лучшие параметры
            current_best = self.ai_db.get_best_params_for_symbol(symbol)
            
            # Обновляем только если новый рейтинг лучше
            if not current_best or rating > current_best.get('rating', 0):
                self.ai_db.save_best_params_for_symbol(symbol, rsi_params, rating, win_rate, total_pnl)
                logger.debug(f"✅ Обновлены лучшие параметры для {symbol}: рейтинг {rating:.2f}, Win Rate {win_rate:.1f}%")
        except Exception as e:
            logger.debug(f"⚠️ Ошибка обновления лучших параметров для {symbol}: {e}")
    
    def get_usage_stats(self) -> Dict:
        """
        Получить статистику использования параметров
        
        Returns:
            Словарь со статистикой
        """
        if not self.ai_db:
            return {
                'used_combinations': 0,
                'total_combinations': self.total_combinations,
                'remaining_combinations': self.total_combinations,
                'usage_percentage': 0.0,
                'is_exhausted': False
            }
        
        used_count = self.ai_db.count_used_training_parameters()
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
        if not self.ai_db:
            return []
        return self.ai_db.get_best_used_parameters(limit, min_win_rate)
    
    def get_best_params_for_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Получить лучшие параметры для конкретной монеты
        
        Args:
            symbol: Символ монеты
        
        Returns:
            Словарь с лучшими параметрами или None
        """
        if not self.ai_db:
            return None
        return self.ai_db.get_best_params_for_symbol(symbol)
    
    def get_all_best_params_per_symbol(self) -> Dict[str, Dict]:
        """
        Получить лучшие параметры для всех монет
        
        Returns:
            Словарь {symbol: best_params}
        """
        if not self.ai_db:
            return {}
        return self.ai_db.get_all_best_params_per_symbol()
    
    def _load_blocked_params(self) -> List[Dict]:
        """Загрузить информацию о заблокированных параметрах"""
        if not self.ai_db:
            return []
        return self.ai_db.get_blocked_params(limit=None)
    
    def _analyze_blocking_patterns(self, blocked_params: List[Dict]) -> Dict[str, Any]:
        """
        Анализирует паттерны блокировок для понимания, какие параметры чаще блокируются
        
        Args:
            blocked_params: Список заблокированных параметров
        
        Returns:
            Словарь с анализом паттернов блокировок
        """
        if not blocked_params:
            return {}
        
        # Анализируем причины блокировок
        reason_counts = {}
        param_ranges = {
            'oversold': {'min': 100, 'max': 0, 'values': []},
            'overbought': {'min': 100, 'max': 0, 'values': []},
        }
        
        for blocked in blocked_params:
            # Подсчитываем причины
            reasons = blocked.get('block_reasons', {})
            for reason, count in reasons.items():
                reason_counts[reason] = reason_counts.get(reason, 0) + count
            
            # Анализируем диапазоны параметров
            rsi_params = blocked.get('rsi_params', {})
            for key in ['oversold', 'overbought']:
                if key in rsi_params:
                    value = rsi_params[key]
                    param_ranges[key]['values'].append(value)
                    param_ranges[key]['min'] = min(param_ranges[key]['min'], value)
                    param_ranges[key]['max'] = max(param_ranges[key]['max'], value)
        
        # Вычисляем средние значения
        for key in param_ranges:
            if param_ranges[key]['values']:
                param_ranges[key]['avg'] = sum(param_ranges[key]['values']) / len(param_ranges[key]['values'])
            else:
                param_ranges[key]['avg'] = 0
        
        return {
            'total_blocked': len(blocked_params),
            'reason_counts': reason_counts,
            'param_ranges': param_ranges,
            'top_blocking_reasons': sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _is_params_similar_to_blocked(self, rsi_params: Dict, blocked_params: List[Dict], 
                                      similarity_threshold: int = 2) -> bool:
        """
        Проверяет, похожи ли параметры на заблокированные
        
        Args:
            rsi_params: Параметры для проверки
            blocked_params: Список заблокированных параметров
            similarity_threshold: Порог схожести (разница в значениях)
        
        Returns:
            True если параметры похожи на заблокированные
        """
        for blocked in blocked_params:
            blocked_rsi = blocked.get('rsi_params', {})
            if not blocked_rsi:
                continue
            
            # Проверяем схожесть по каждому параметру
            differences = []
            for key in ['oversold', 'overbought', 'exit_long_with_trend', 
                       'exit_long_against_trend', 'exit_short_with_trend', 'exit_short_against_trend']:
                if key in rsi_params and key in blocked_rsi:
                    diff = abs(rsi_params[key] - blocked_rsi[key])
                    differences.append(diff)
            
            # Если все параметры очень похожи - считаем что это похожие параметры
            if differences and max(differences) <= similarity_threshold:
                return True
        
        return False
    
    def get_unused_params_suggestion(self, base_params: Dict, 
                                     variation_range: int = 3,
                                     avoid_blocked: bool = True) -> Optional[Dict]:
        """
        Предложить неиспользованные параметры на основе базовых
        
        Пробует найти комбинацию параметров которая еще не использовалась
        и (опционально) избегает параметров, похожих на заблокированные
        
        Args:
            base_params: Базовые параметры
            variation_range: Диапазон вариации
            avoid_blocked: Избегать параметров, похожих на заблокированные
        
        Returns:
            Словарь с неиспользованными параметрами или None если не найдено
        """
        import random
        
        # Получаем статистику использования для диагностики
        stats = self.get_usage_stats()
        usage_percentage = stats.get('usage_percentage', 0.0)
        
        # Загружаем заблокированные параметры если нужно их избегать
        blocked_params = []
        blocking_patterns = {}
        if avoid_blocked:
            blocked_params = self._load_blocked_params()
            if blocked_params:
                blocking_patterns = self._analyze_blocking_patterns(blocked_params)
                logger.debug(f"📊 Учитываем {len(blocked_params)} записей о блокировках при генерации параметров")
                if blocking_patterns.get('top_blocking_reasons'):
                    top_reasons = ", ".join([f"{r}: {c}" for r, c in blocking_patterns['top_blocking_reasons']])
                    logger.debug(f"   🔍 Топ причин блокировок: {top_reasons}")
        
        # Адаптивное количество попыток в зависимости от заполненности
        if usage_percentage < 50:
            max_attempts = 200
        elif usage_percentage < 80:
            max_attempts = 500  # Увеличиваем при высокой заполненности
        else:
            max_attempts = 1000  # Еще больше при очень высокой заполненности
        
        # Сначала пробуем систематический перебор вокруг базовых параметров
        # Это более эффективно чем чисто случайный поиск
        systematic_attempts = min(100, max_attempts // 2)
        for attempt in range(systematic_attempts):
            # Генерируем параметры с систематической вариацией
            # Используем attempt для создания более разнообразных комбинаций
            offset1 = (attempt % 7) - 3  # -3 до 3
            offset2 = ((attempt // 7) % 7) - 3
            offset3 = ((attempt // 49) % 5) - 2
            offset4 = ((attempt // 245) % 5) - 2
            
            rsi_params = {
                'oversold': max(20, min(35, 
                    base_params.get('oversold', 29) + offset1)),
                'overbought': max(65, min(80,
                    base_params.get('overbought', 71) + offset2)),
                'exit_long_with_trend': max(55, min(70,
                    base_params.get('exit_long_with_trend', 65) + offset3)),
                'exit_long_against_trend': max(50, min(65,
                    base_params.get('exit_long_against_trend', 60) + offset4)),
                'exit_short_with_trend': max(25, min(40,
                    base_params.get('exit_short_with_trend', 35) + random.randint(-5, 5))),
                'exit_short_against_trend': max(30, min(45,
                    base_params.get('exit_short_against_trend', 40) + random.randint(-5, 5)))
            }
            
            # Проверяем, использовались ли эти параметры
            if self.is_params_used(rsi_params):
                continue
            
            # Проверяем, похожи ли на заблокированные (если нужно избегать)
            if avoid_blocked and blocked_params:
                if self._is_params_similar_to_blocked(rsi_params, blocked_params):
                    continue  # Пропускаем похожие на заблокированные
            
            return rsi_params
        
        # Если систематический поиск не дал результата, пробуем случайный
        # Ослабляем проверку на заблокированные параметры если заполненность высокая
        strict_blocked_check = avoid_blocked and usage_percentage < 70
        
        for attempt in range(systematic_attempts, max_attempts):
            # Генерируем случайные параметры с большей вариацией
            rsi_params = {
                'oversold': max(20, min(35, 
                    base_params.get('oversold', 29) + random.randint(-variation_range * 2, variation_range * 2))),
                'overbought': max(65, min(80,
                    base_params.get('overbought', 71) + random.randint(-variation_range * 2, variation_range * 2))),
                'exit_long_with_trend': max(55, min(70,
                    base_params.get('exit_long_with_trend', 65) + random.randint(-8, 8))),
                'exit_long_against_trend': max(50, min(65,
                    base_params.get('exit_long_against_trend', 60) + random.randint(-8, 8))),
                'exit_short_with_trend': max(25, min(40,
                    base_params.get('exit_short_with_trend', 35) + random.randint(-8, 8))),
                'exit_short_against_trend': max(30, min(45,
                    base_params.get('exit_short_against_trend', 40) + random.randint(-8, 8)))
            }
            
            # Проверяем, использовались ли эти параметры
            if self.is_params_used(rsi_params):
                continue
            
            # Проверяем, похожи ли на заблокированные (только если строгая проверка)
            if strict_blocked_check and blocked_params:
                if self._is_params_similar_to_blocked(rsi_params, blocked_params):
                    continue  # Пропускаем похожие на заблокированные
            
            return rsi_params
        
        # Если не нашли за max_attempts попыток - пробуем найти любые параметры (даже использованные)
        # но с хорошим рейтингом, если параметры почти исчерпаны
        if usage_percentage > 80:
            logger.debug(f"💡 Параметры почти исчерпаны ({usage_percentage:.1f}%), ищем лучшие использованные параметры")
            best_params = self.get_best_params(limit=10, min_win_rate=0.0)
            if best_params:
                # Возвращаем лучшие параметры (даже если использовались)
                best = best_params[0]
                logger.debug(f"✅ Используем лучшие параметры (Win Rate: {best.get('win_rate', 0):.1f}%, Рейтинг: {best.get('rating', 0):.1f})")
                return best.get('rsi_params')
        
        # Если все попытки не удались - возвращаем None
        logger.warning(f"⚠️ Не удалось найти неиспользованные параметры за {max_attempts} попыток (использовано {stats.get('used_combinations', 0)}/{stats.get('total_combinations', 0)} комбинаций, {usage_percentage:.1f}%)")
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
        
        if not self.ai_db:
            logger.warning("⚠️ AI Database недоступна, сброс невозможен")
            return
        
        # ВАЖНО: Сброс БД - опасная операция, лучше не делать автоматически
        # Оставляем только предупреждение
        logger.warning("⚠️ Сброс параметров в БД не реализован для безопасности. Используйте SQL напрямую если необходимо.")

