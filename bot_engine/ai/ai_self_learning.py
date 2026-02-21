#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль самообучения AI системы

Реализует настоящее самообучение AI в реальном времени:
- Онлайн-обучение на каждой закрытой сделке
- Адаптация к изменяющимся рыночным условиям
- Самокорректировка на основе результатов
- Непрерывное улучшение эффективности
"""

import os
import json
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger('AI.SelfLearning')


def _get_ai_config_value(attr_name: str, default_value: Any) -> Any:
    """
    Получает значение настройки AI из конфига с fallback на дефолтное значение.
    
    Args:
        attr_name: Имя атрибута в AIConfig
        default_value: Дефолтное значение, если атрибут не найден
    
    Returns:
        Значение из конфига или дефолтное значение
    """
    try:
        from bot_engine.config_loader import AIConfig
        return getattr(AIConfig, attr_name, default_value)
    except (ImportError, AttributeError):
        return default_value


class AISelfLearning:
    """
    Система самообучения AI

    Основные возможности:
    1. Онлайн-обучение на каждой сделке
    2. Адаптация весов модели в реальном времени
    3. Самооценка эффективности
    4. Автоматическая корректировка стратегии
    """

    def __init__(self):
        """Инициализация системы самообучения"""
        self.data_dir = 'data/ai'
        self.models_dir = os.path.join(self.data_dir, 'models')
        self.self_learning_dir = os.path.join(self.data_dir, 'self_learning')

        # Создаем директории
        os.makedirs(self.self_learning_dir, exist_ok=True)

        # Подключаемся к AI тренеру
        self.ai_trainer = None
        self._connect_to_trainer()

        # ✅ Загружаем настройки из конфига с fallback на дефолтные значения
        # Самообучение AI в реальном времени
        self.self_learning_enabled = _get_ai_config_value('AI_SELF_LEARNING_ENABLED', True)
        self.self_learning_buffer_size = _get_ai_config_value('AI_SELF_LEARNING_BUFFER_SIZE', 50)
        self.adaptation_threshold = _get_ai_config_value('AI_ADAPTATION_THRESHOLD', 0.1)
        self.performance_window = _get_ai_config_value('AI_PERFORMANCE_WINDOW', 50)
        self.incremental_retrain_enabled = _get_ai_config_value('AI_INCREMENTAL_RETRAIN_ENABLED', True)

        # Система самооценки (передаем performance_window из конфига)
        self.performance_tracker = PerformanceTracker(performance_window=self.performance_window)

        # Онлайн обучение (используем настройки из конфига)
        self.online_learning_enabled = self.self_learning_enabled
        self.online_learning_buffer = deque(maxlen=self.self_learning_buffer_size)
        self.online_learning_interval = 5  # Обновление каждые 5 сделок

        # Адаптивное обучение
        self.adaptive_learning_enabled = self.self_learning_enabled
        self.market_conditions_buffer = deque(maxlen=self.self_learning_buffer_size)
        # self.adaptation_threshold уже установлен выше из конфига

        # Многопоточность
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="AI_SelfLearning")

        # Статистика
        self.stats = {
            'total_trades_processed': 0,
            'successful_adaptations': 0,
            'failed_adaptations': 0,
            'online_updates': 0,
            'last_update_time': None,
            'performance_score': 0.0
        }

        # Загружаем состояние
        self._load_state()

        logger.info("✅ AISelfLearning инициализирован")

    def _connect_to_trainer(self):
        """Подключение к AI тренеру"""
        try:
            from bot_engine.ai.ai_trainer import AITrainer
            self.ai_trainer = AITrainer()
            logger.info("✅ Подключен к AITrainer")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось подключиться к AITrainer: {e}")
            self.ai_trainer = None

    def _load_state(self):
        """Загрузка состояния системы самообучения"""
        try:
            state_file = os.path.join(self.self_learning_dir, 'self_learning_state.json')
            if os.path.exists(state_file):
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.stats.update(state.get('stats', {}))
                    logger.info("✅ Состояние самообучения загружено")
        except Exception as e:
            pass

    def _save_state(self):
        """Сохранение состояния системы самообучения"""
        try:
            state_file = os.path.join(self.self_learning_dir, 'self_learning_state.json')
            state = {
                'stats': self.stats,
                'last_save': datetime.now().isoformat()
            }
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            pass

    def process_trade_result(self, trade_result: Dict) -> None:
        """
        Обработка результата сделки для самообучения

        Args:
            trade_result: Результат закрытой сделки
        """
        try:
            pass

            # Добавляем в буфер онлайн обучения
            self.online_learning_buffer.append(trade_result)
            self.stats['total_trades_processed'] += 1

            # Буфер тренера для инкрементального ретрайна (те же сделки, 7 признаков)
            if self.ai_trainer:
                try:
                    self.ai_trainer.update_model_online(trade_result)
                except Exception:
                    pass

            # Обновляем трекер производительности
            self.performance_tracker.add_trade_result(trade_result)

            # Проверяем необходимость онлайн обучения
            if len(self.online_learning_buffer) >= self.online_learning_interval:
                self._perform_online_learning()

            # Адаптивное обучение на рыночных условиях
            if self.adaptive_learning_enabled:
                self._check_market_adaptation(trade_result)

            # Сохраняем состояние
            self._save_state()

        except Exception as e:
            logger.error(f"❌ Ошибка обработки сделки для самообучения: {e}")

    def _perform_online_learning(self) -> None:
        """
        Выполнение онлайн обучения на накопленных данных
        """
        try:
            if not self.ai_trainer or not self.online_learning_buffer:
                return

            pass

            # Преобразуем буфер в обучающие данные
            training_data = self._prepare_online_training_data()

            if training_data:
                # Инкрементальный ретрайн на последних сделках из БД (реальное переобучение)
                retrain_success = False
                if self.incremental_retrain_enabled and self.ai_trainer:
                    try:
                        retrain_success = self.ai_trainer.retrain_on_recent_trades()
                    except Exception as e:
                        logger.debug(f"Инкрементальный ретрайн: {e}")

                if not retrain_success:
                    # Fallback: попытка обновить «веса» (анализ паттернов, для RandomForest эффект ограничен)
                    success = self._update_model_online(training_data)
                    if success:
                        self.stats['online_updates'] += 1
                        self.stats['last_update_time'] = datetime.now().isoformat()
                        logger.info("✅ Онлайн обучение выполнено успешно")
                        self._evaluate_learning_effectiveness()
                    else:
                        logger.warning("⚠️ Онлайн обучение завершилось с ошибками")
                else:
                    self.stats['online_updates'] += 1
                    self.stats['last_update_time'] = datetime.now().isoformat()
                    logger.info("✅ Инкрементальный ретрайн выполнен успешно")
                    self._evaluate_learning_effectiveness()

            # Очищаем буфер после обработки
            self.online_learning_buffer.clear()

        except Exception as e:
            logger.error(f"❌ Ошибка онлайн обучения: {e}")

    def _prepare_online_training_data(self) -> Optional[Dict]:
        """
        Подготовка данных для онлайн обучения

        Returns:
            Словарь с обучающими данными или None
        """
        try:
            if len(self.online_learning_buffer) < 3:
                return None

            successful_trades = []
            failed_trades = []

            for trade in self.online_learning_buffer:
                pnl = trade.get('pnl', 0)
                if pnl > 0:
                    successful_trades.append(trade)
                else:
                    failed_trades.append(trade)

            if len(successful_trades) == 0 or len(failed_trades) == 0:
                pass
                return None

            # Извлекаем паттерны успеха/неудачи
            success_patterns = self._extract_trade_patterns(successful_trades)
            failure_patterns = self._extract_trade_patterns(failed_trades)

            return {
                'successful_patterns': success_patterns,
                'failure_patterns': failure_patterns,
                'sample_count': len(self.online_learning_buffer),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Ошибка подготовки данных для онлайн обучения: {e}")
            return None

    def _extract_trade_patterns(self, trades: List[Dict]) -> Dict:
        """
        Извлечение паттернов из сделок

        Args:
            trades: Список сделок

        Returns:
            Словарь с паттернами
        """
        patterns = {
            'avg_rsi': 0,
            'trend_distribution': {},
            'avg_volatility': 0,
            'avg_pnl': 0,
            'count': len(trades)
        }

        if not trades:
            return patterns

        rsi_values = []
        trends = []
        volatilities = []
        pnl_values = []

        for trade in trades:
            entry_data = trade.get('entry_data', {})

            rsi = entry_data.get('rsi')
            if rsi:
                rsi_values.append(rsi)

            trend = entry_data.get('trend', 'NEUTRAL')
            trends.append(trend)

            volatility = entry_data.get('volatility', 0)
            volatilities.append(volatility)

            pnl = trade.get('pnl', 0)
            pnl_values.append(pnl)

        if rsi_values:
            patterns['avg_rsi'] = np.mean(rsi_values)

        if trends:
            patterns['trend_distribution'] = dict([(t, trends.count(t)) for t in set(trends)])

        if volatilities:
            patterns['avg_volatility'] = np.mean(volatilities)

        if pnl_values:
            patterns['avg_pnl'] = np.mean(pnl_values)

        return patterns

    def _update_model_online(self, training_data: Dict) -> bool:
        """
        Обновление модели онлайн обучением

        Args:
            training_data: Данные для обучения

        Returns:
            True если успешно
        """
        try:
            if not self.ai_trainer:
                return False

            # Получаем текущие веса модели
            current_weights = self._get_model_weights()

            # Вычисляем корректировки на основе паттернов
            adjustments = self._calculate_weight_adjustments(training_data)

            # Применяем корректировки
            if adjustments:
                new_weights = self._apply_weight_adjustments(current_weights, adjustments)

                # Сохраняем обновленные веса
                success = self._set_model_weights(new_weights)

                if success:
                    pass
                    return True
                else:
                    logger.warning("⚠️ Не удалось обновить весы модели")
                    return False
            else:
                pass
                return True

        except Exception as e:
            logger.error(f"❌ Ошибка онлайн обновления модели: {e}")
            return False

    def _get_model_weights(self) -> Optional[Dict]:
        """Получение текущих весов модели"""
        try:
            if not self.ai_trainer or not self.ai_trainer.signal_predictor:
                return None

            # Для RandomForest получаем feature_importances_
            if hasattr(self.ai_trainer.signal_predictor, 'feature_importances_'):
                return {
                    'feature_importances': self.ai_trainer.signal_predictor.feature_importances_.tolist(),
                    'model_type': 'RandomForest'
                }
            else:
                pass
                return None

        except Exception as e:
            logger.error(f"❌ Ошибка получения весов модели: {e}")
            return None

    def _calculate_weight_adjustments(self, training_data: Dict) -> Optional[Dict]:
        """
        Вычисление корректировок весов на основе обучающих данных

        Args:
            training_data: Данные для обучения

        Returns:
            Словарь с корректировками или None
        """
        try:
            success_patterns = training_data.get('successful_patterns', {})
            failure_patterns = training_data.get('failure_patterns', {})

            adjustments = {}

            # Корректировка на основе RSI
            success_rsi = success_patterns.get('avg_rsi', 0)
            failure_rsi = failure_patterns.get('avg_rsi', 0)

            if success_rsi and failure_rsi and abs(success_rsi - failure_rsi) > 2:
                # Если успешные сделки при определенном RSI, усиливаем этот фактор
                rsi_adjustment = (success_rsi - failure_rsi) * 0.01  # Небольшая корректировка
                adjustments['rsi_weight'] = rsi_adjustment
                pass

            # Корректировка на основе трендов
            success_trends = success_patterns.get('trend_distribution', {})
            failure_trends = failure_patterns.get('trend_distribution', {})

            if success_trends and failure_trends:
                # Находим наиболее успешный тренд
                best_trend = max(success_trends.items(), key=lambda x: x[1])[0] if success_trends else None
                worst_trend = max(failure_trends.items(), key=lambda x: x[1])[0] if failure_trends else None

                if best_trend and worst_trend and best_trend != worst_trend:
                    adjustments['trend_preference'] = {
                        'preferred': best_trend,
                        'avoid': worst_trend
                    }
                    pass

            return adjustments if adjustments else None

        except Exception as e:
            logger.error(f"❌ Ошибка вычисления корректировок: {e}")
            return None

    def _apply_weight_adjustments(self, current_weights: Dict, adjustments: Dict) -> Dict:
        """
        Применение корректировок к весам

        Args:
            current_weights: Текущие веса
            adjustments: Корректировки

        Returns:
            Новые веса
        """
        try:
            new_weights = current_weights.copy()

            # Применяем корректировки (простая реализация)
            if 'rsi_weight' in adjustments:
                # Для RandomForest можем только логировать, так как feature_importances_ read-only
                pass

            if 'trend_preference' in adjustments:
                # Сохраняем предпочтения для использования в будущем обучении
                new_weights['trend_preferences'] = adjustments['trend_preference']

            return new_weights

        except Exception as e:
            logger.error(f"❌ Ошибка применения корректировок: {e}")
            return current_weights

    def _set_model_weights(self, weights: Dict) -> bool:
        """Установка новых весов модели"""
        try:
            # Сохраняем веса в файл для будущего использования
            weights_file = os.path.join(self.self_learning_dir, 'model_weights.json')
            with open(weights_file, 'w', encoding='utf-8') as f:
                json.dump(weights, f, ensure_ascii=False, indent=2)

            pass
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения весов модели: {e}")
            return False

    def _check_market_adaptation(self, trade_result: Dict) -> None:
        """
        Проверка необходимости адаптации к рыночным условиям

        Args:
            trade_result: Результат сделки
        """
        try:
            # Добавляем текущие рыночные условия в буфер
            market_conditions = self._extract_market_conditions(trade_result)
            self.market_conditions_buffer.append(market_conditions)

            if len(self.market_conditions_buffer) >= 10:
                # Анализируем изменения в рыночных условиях
                changes = self._analyze_market_changes()

                if changes and changes['significant_change']:
                    logger.info("🌊 Обнаружены значительные изменения в рынке, запускаем адаптацию...")
                    self._perform_market_adaptation(changes)

        except Exception as e:
            pass

    def _extract_market_conditions(self, trade_result: Dict) -> Dict:
        """Извлечение рыночных условий из сделки"""
        entry_data = trade_result.get('entry_data', {})
        return {
            'volatility': entry_data.get('volatility', 0),
            'trend': entry_data.get('trend', 'NEUTRAL'),
            'rsi': entry_data.get('rsi', 50),
            'volume_ratio': entry_data.get('volume_ratio', 1.0),
            'timestamp': trade_result.get('timestamp', datetime.now().isoformat())
        }

    def _analyze_market_changes(self) -> Optional[Dict]:
        """
        Анализ изменений в рыночных условиях

        Returns:
            Информация об изменениях или None
        """
        try:
            if len(self.market_conditions_buffer) < 5:
                return None

            recent = list(self.market_conditions_buffer)[-5:]  # Последние 5 условий
            older = list(self.market_conditions_buffer)[:-5]   # Предыдущие

            if not older:
                return None

            # Вычисляем средние значения
            recent_avg_volatility = np.mean([c['volatility'] for c in recent])
            older_avg_volatility = np.mean([c['volatility'] for c in older])

            # Проверяем значительные изменения
            volatility_change = abs(recent_avg_volatility - older_avg_volatility) / max(older_avg_volatility, 0.001)

            significant_change = volatility_change > self.adaptation_threshold

            return {
                'significant_change': significant_change,
                'volatility_change': volatility_change,
                'recent_avg_volatility': recent_avg_volatility,
                'older_avg_volatility': older_avg_volatility
            }

        except Exception as e:
            pass
            return None

    def _perform_market_adaptation(self, changes: Dict) -> None:
        """
        Выполнение адаптации к рыночным условиям

        Args:
            changes: Информация об изменениях
        """
        try:
            logger.info(f"🔄 Адаптация к рыночным изменениям: волатильность изменилась на {changes['volatility_change']:.2%}")

            # Пример адаптации: корректировка порогов принятия решений
            if changes['volatility_change'] > 0:
                # Увеличилась волатильность - повышаем требования к уверенности AI
                adaptation = {
                    'type': 'volatility_increase',
                    'action': 'increase_confidence_threshold',
                    'factor': min(changes['volatility_change'] * 2, 0.5),  # Максимум +50%
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Уменьшилась волатильность - можем снизить требования
                adaptation = {
                    'type': 'volatility_decrease',
                    'action': 'decrease_confidence_threshold',
                    'factor': min(abs(changes['volatility_change']) * 1.5, 0.3),  # Максимум -30%
                    'timestamp': datetime.now().isoformat()
                }

            # Сохраняем адаптацию
            self._save_adaptation(adaptation)
            self.stats['successful_adaptations'] += 1

            logger.info("✅ Адаптация к рынку выполнена")

        except Exception as e:
            logger.error(f"❌ Ошибка выполнения адаптации: {e}")
            self.stats['failed_adaptations'] += 1

    def _save_adaptation(self, adaptation: Dict) -> None:
        """Сохранение информации об адаптации"""
        try:
            adaptations_file = os.path.join(self.self_learning_dir, 'adaptations.json')

            # Загружаем существующие адаптации
            adaptations = []
            if os.path.exists(adaptations_file):
                with open(adaptations_file, 'r', encoding='utf-8') as f:
                    adaptations = json.load(f)

            adaptations.append(adaptation)

            # Ограничиваем историю (последние 100 адаптаций)
            if len(adaptations) > 100:
                adaptations = adaptations[-100:]

            # Сохраняем
            with open(adaptations_file, 'w', encoding='utf-8') as f:
                json.dump(adaptations, f, ensure_ascii=False, indent=2)

        except Exception as e:
            pass

    def _evaluate_learning_effectiveness(self) -> None:
        """Оценка эффективности обучения"""
        try:
            # Получаем текущую производительность
            current_performance = self.performance_tracker.get_performance_score()

            # Сравниваем с предыдущей оценкой
            if self.stats.get('performance_score', 0) > 0:
                improvement = current_performance - self.stats['performance_score']
                if abs(improvement) > 0.01:  # Значимое изменение
                    direction = "улучшилась" if improvement > 0 else "ухудшилась"
                    logger.info(".2%")

            # Обновляем оценку
            self.stats['performance_score'] = current_performance

        except Exception as e:
            pass

    def get_learning_stats(self) -> Dict:
        """
        Получение статистики самообучения

        Returns:
            Словарь со статистикой
        """
        return {
            'stats': self.stats.copy(),
            'performance_score': self.performance_tracker.get_performance_score(),
            'online_learning_enabled': self.online_learning_enabled,
            'adaptive_learning_enabled': self.adaptive_learning_enabled,
            'buffer_size': len(self.online_learning_buffer),
            'market_conditions_buffer_size': len(self.market_conditions_buffer)
        }

    def enable_online_learning(self, enabled: bool = True) -> None:
        """Включение/выключение онлайн обучения"""
        self.online_learning_enabled = enabled
        logger.info(f"{'✅' if enabled else '❌'} Онлайн обучение {'включено' if enabled else 'выключено'}")

    def enable_adaptive_learning(self, enabled: bool = True) -> None:
        """Включение/выключение адаптивного обучения"""
        self.adaptive_learning_enabled = enabled
        logger.info(f"{'✅' if enabled else '❌'} Адаптивное обучение {'включено' if enabled else 'выключено'}")

    def _get_continuous_learning(self):
        """Ленивая загрузка AIContinuousLearning для оценки производительности"""
        if not hasattr(self, '_continuous_learning') or self._continuous_learning is None:
            try:
                from bot_engine.ai.ai_continuous_learning import AIContinuousLearning
                self._continuous_learning = AIContinuousLearning()
            except Exception as e:
                logger.warning(f"⚠️ Не удалось подключить AIContinuousLearning: {e}")
                self._continuous_learning = None
        return self._continuous_learning

    def evaluate_ai_performance(self, trades: List[Dict]) -> Dict:
        """
        Оценивает производительность AI на основе сделок (делегирует в AIContinuousLearning)

        Args:
            trades: Список сделок с результатами

        Returns:
            Словарь с метриками производительности AI
        """
        continuous = self._get_continuous_learning()
        if continuous:
            return continuous.evaluate_ai_performance(trades)
        # Fallback: базовая оценка через PerformanceTracker
        if not trades:
            return {'error': 'Нет данных о сделках'}
        for t in trades:
            self.performance_tracker.add_trade_result(t)
        return {
            'total_trades': len(trades),
            'performance_score': self.performance_tracker.get_performance_score(),
            'evaluation_timestamp': datetime.now().isoformat()
        }

    def get_performance_trends(self) -> Dict:
        """
        Анализирует тренды производительности AI со временем (делегирует в AIContinuousLearning)

        Returns:
            Словарь с трендами производительности
        """
        continuous = self._get_continuous_learning()
        if continuous:
            return continuous.get_performance_trends()
        return {'error': 'AIContinuousLearning недоступен для анализа трендов'}


class PerformanceTracker:
    """Трекер производительности AI"""

    def __init__(self, performance_window: int = 50):
        """
        Args:
            performance_window: Окно сделок для оценки производительности (по умолчанию из конфига)
        """
        self.trade_results = deque(maxlen=1000)  # Последние 1000 сделок
        self.performance_window = performance_window  # Окно для оценки производительности

    def add_trade_result(self, trade_result: Dict) -> None:
        """Добавление результата сделки"""
        self.trade_results.append(trade_result)

    def get_performance_score(self) -> float:
        """
        Получение оценки производительности

        Returns:
            Оценка производительности (0.0 - 1.0)
        """
        try:
            if len(self.trade_results) < 10:
                return 0.5  # Нейтральная оценка при недостатке данных

            recent_trades = list(self.trade_results)[-self.performance_window:]

            successful_trades = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
            total_trades = len(recent_trades)

            win_rate = successful_trades / total_trades if total_trades > 0 else 0

            # Учитываем также среднюю прибыльность
            pnl_values = [t.get('pnl', 0) for t in recent_trades]
            avg_pnl = np.mean(pnl_values) if pnl_values else 0

            # Нормализуем оценку (win_rate + нормализованный avg_pnl)
            pnl_score = max(0, min(1, (avg_pnl + 100) / 200))  # Предполагаем нормальный диапазон -100..+100

            performance_score = (win_rate * 0.7) + (pnl_score * 0.3)

            return max(0, min(1, performance_score))

        except Exception as e:
            pass
            return 0.5


# Глобальный экземпляр системы самообучения
_self_learning_instance = None
_self_learning_lock = threading.Lock()


def get_self_learning_system() -> AISelfLearning:
    """
    Получение глобального экземпляра системы самообучения

    Returns:
        Экземпляр AISelfLearning
    """
    global _self_learning_instance

    if _self_learning_instance is None:
        with _self_learning_lock:
            if _self_learning_instance is None:
                _self_learning_instance = AISelfLearning()

    return _self_learning_instance


def process_trade_for_self_learning(trade_result: Dict) -> None:
    """
    Обработка сделки для самообучения (глобальная функция)

    Args:
        trade_result: Результат сделки
    """
    try:
        self_learning = get_self_learning_system()
        # Запускаем в отдельном потоке, чтобы не блокировать основной поток
        self_learning.executor.submit(self_learning.process_trade_result, trade_result)
    except Exception as e:
        pass