#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML модель для предсказания качества параметров торговли

Обучается на успешных/неуспешных параметрах и предсказывает:
- Вероятность успеха параметров
- Ожидаемый Win Rate
- Ожидаемый PnL

Используется для генерации оптимальных параметров вместо случайных
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
try:
    import utils.sklearn_parallel_config  # noqa: F401 — до импорта sklearn, подавляет UserWarning delayed/Parallel
except ImportError:
    pass
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib  # только dump/load; Parallel/delayed — оба из sklearn через utils.sklearn_parallel_config (патч joblib)  # только dump/load; Parallel/delayed — через sklearn (патч в utils.sklearn_parallel_config)

logger = logging.getLogger('AI.ParameterQualityPredictor')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    pass


class ParameterQualityPredictor:
    """
    ML модель для предсказания качества параметров торговли
    """
    
    def __init__(self, data_dir: str = 'data/ai'):
        # Определяем корень проекта для правильных путей
        try:
            from bot_engine.ai.ai_database import _get_project_root
            project_root = _get_project_root()
        except:
            # Fallback: используем текущую директорию
            import sys
            from pathlib import Path
            current_file = Path(__file__).resolve()
            project_root = None
            for parent in current_file.parents:
                if (parent / 'ai.py').exists() and (parent / 'bot_engine').exists():
                    project_root = parent
                    break
            if project_root is None:
                project_root = Path.cwd()
        
        # Модели сохраняются в data/ai/models относительно корня проекта
        self.data_dir = str(project_root / data_dir)
        self.models_dir = os.path.normpath(os.path.join(self.data_dir, 'models'))
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.model_file = os.path.normpath(os.path.join(self.models_dir, 'parameter_quality_predictor.pkl'))
        self.scaler_file = os.path.normpath(os.path.join(self.models_dir, 'parameter_quality_scaler.pkl'))
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.expected_features = None  # Количество признаков, которое ожидает загруженная модель
        
        # Метаданные для автоматического переобучения
        self._last_trained_samples_count = 0
        self._last_trained_time = None
        
        # Подключаемся к БД
        try:
            from bot_engine.ai.ai_database import get_ai_database
            self.ai_db = get_ai_database()
            pass
        except Exception as e:
            logger.warning(f"⚠️ Не удалось подключиться к AI Database: {e}")
            self.ai_db = None
        
        # Загружаем модель если есть
        self._load_model()
    
    def _load_model(self):
        """Загрузить обученную модель"""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
                self.model = joblib.load(self.model_file)
                self.scaler = joblib.load(self.scaler_file)
                # УЛУЧШЕНИЕ: Проверяем совместимость количества признаков
                # Генерируем тестовые признаки для проверки
                test_rsi_params = {
                    'oversold': 29,
                    'overbought': 71,
                    'exit_long_with_trend': 65,
                    'exit_long_against_trend': 60,
                    'exit_short_with_trend': 35,
                    'exit_short_against_trend': 40
                }
                test_risk_params = {
                    'stop_loss': 15.0,
                    'take_profit': 20.0,
                    'trailing_stop_activation': 30.0,
                    'trailing_stop_distance': 5.0
                }
                test_features = self._extract_features(test_rsi_params, test_risk_params)
                expected_features = test_features.shape[1]
                
                # Проверяем количество признаков в scaler
                scaler_features = None
                if hasattr(self.scaler, 'n_features_in_'):
                    scaler_features = self.scaler.n_features_in_
                elif hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                    scaler_features = len(self.scaler.mean_)
                else:
                    # Для старых версий sklearn проверяем через transform
                    try:
                        test_scaled = self.scaler.transform(test_features)
                        scaler_features = test_features.shape[1]  # Если успешно, значит совместимо
                    except ValueError as ve:
                        # Извлекаем количество из ошибки
                        error_msg = str(ve)
                        if 'expecting' in error_msg and 'features' in error_msg:
                            import re
                            match = re.search(r'expecting (\d+) features', error_msg)
                            if match:
                                scaler_features = int(match.group(1))
                
                if scaler_features is not None and scaler_features != expected_features:
                    # Проверяем, поддерживает ли legacy режим это количество признаков
                    if scaler_features in [7, 8, 10]:
                        logger.info(
                            f"ℹ️ Модель ожидает {scaler_features} признаков (старая версия), "
                            f"текущая версия генерирует {expected_features}. "
                            f"Будет использоваться legacy режим для обратной совместимости."
                        )
                        # Сохраняем количество признаков для использования в predict_quality
                        self.expected_features = scaler_features
                        # Модель может использоваться с legacy режимом
                        self.is_trained = True
                        logger.debug(f"✅ Загружена модель предсказания качества параметров (legacy режим: {scaler_features} признаков)")
                    else:
                        logger.warning(
                            f"⚠️ Несовместимость признаков: модель ожидает {scaler_features} признаков, "
                            f"а текущая версия генерирует {expected_features}. "
                            f"Legacy режим не поддерживает {scaler_features} признаков. "
                            f"Модель нужно переобучить!"
                        )
                        # Не помечаем модель как обученную, чтобы она не использовалась
                        self.is_trained = False
                        self.model = None
                        self.scaler = StandardScaler()  # Сбрасываем scaler
                        self.expected_features = None
                        return
                else:
                    # Количество признаков совпадает - используем новую версию
                    self.expected_features = expected_features
                    self.is_trained = True
                    logger.debug(f"✅ Загружена модель предсказания качества параметров ({expected_features} признаков)")
        except Exception as e:
            pass
            self.is_trained = False
    
    def _save_model(self):
        """Сохранить обученную модель"""
        try:
            if self.model:
                joblib.dump(self.model, self.model_file)
                joblib.dump(self.scaler, self.scaler_file)
                logger.info("✅ Сохранена модель предсказания качества параметров")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения модели: {e}")
    
    def _extract_features_legacy(self, rsi_params: Dict, risk_params: Optional[Dict] = None, 
                                 num_features: int = 7) -> np.ndarray:
        """
        Старая версия извлечения признаков (для обратной совместимости)
        
        Args:
            rsi_params: Параметры RSI
            risk_params: Параметры риск-менеджмента (опционально)
            num_features: Количество признаков (7 или 8)
        
        Returns:
            Массив признаков
        """
        features = [
            rsi_params.get('oversold', 29),
            rsi_params.get('overbought', 71),
            rsi_params.get('exit_long_with_trend', 65),
            rsi_params.get('exit_long_against_trend', 60),
            rsi_params.get('exit_short_with_trend', 35),
            rsi_params.get('exit_short_against_trend', 40),
        ]
        
        # Добавляем риск-параметры в зависимости от требуемого количества
        if num_features == 7:
            # Старая версия: 6 RSI + 1 риск-параметр
            if risk_params:
                features.append(risk_params.get('stop_loss', 15.0))
            else:
                features.append(0)
        elif num_features == 8:
            # Версия с 8 признаками: 6 RSI + 2 риск-параметра
            if risk_params:
                features.append(risk_params.get('stop_loss', 15.0))
                features.append(risk_params.get('take_profit', 20.0))
            else:
                features.extend([0, 0])
        elif num_features == 10:
            # Версия с 10 признаками: 6 RSI + 2 риск-параметра + 2 производных признака
            oversold = rsi_params.get('oversold', 29)
            overbought = rsi_params.get('overbought', 71)
            exit_long_with = rsi_params.get('exit_long_with_trend', 65)
            exit_short_with = rsi_params.get('exit_short_with_trend', 35)
            
            if risk_params:
                features.append(risk_params.get('stop_loss', 15.0))
                features.append(risk_params.get('take_profit', 20.0))
            else:
                features.extend([0, 0])
            
            # Добавляем 2 производных признака: ширина зон входа/выхода
            long_entry_zone_width = overbought - oversold
            long_exit_zone_width = exit_long_with - oversold
            features.extend([long_entry_zone_width, long_exit_zone_width])
        else:
            # Fallback: используем 7 признаков
            if risk_params:
                features.append(risk_params.get('stop_loss', 15.0))
            else:
                features.append(0)
        
        return np.array(features).reshape(1, -1)
    
    def _extract_features(self, rsi_params: Dict, risk_params: Optional[Dict] = None, 
                         use_extended: bool = True, expected_count: Optional[int] = None) -> np.ndarray:
        """
        Извлечь признаки из параметров для обучения
        
        УЛУЧШЕННАЯ ВЕРСИЯ: Добавлены производные признаки для лучшего предсказания
        Поддерживает обратную совместимость со старыми моделями
        
        Args:
            rsi_params: Параметры RSI
            risk_params: Параметры риск-менеджмента (опционально)
            use_extended: Использовать расширенные признаки (для новых моделей)
        
        Returns:
            Массив признаков
        """
        # Базовые параметры RSI (всегда используются)
        oversold = rsi_params.get('oversold', 29)
        overbought = rsi_params.get('overbought', 71)
        exit_long_with = rsi_params.get('exit_long_with_trend', 65)
        exit_long_against = rsi_params.get('exit_long_against_trend', 60)
        exit_short_with = rsi_params.get('exit_short_with_trend', 35)
        exit_short_against = rsi_params.get('exit_short_against_trend', 40)
        
        features = [
            # Базовые параметры (6 признаков)
            oversold,
            overbought,
            exit_long_with,
            exit_long_against,
            exit_short_with,
            exit_short_against,
        ]
        
        # Старая версия (для обратной совместимости): только базовые + риск-параметры
        if not use_extended:
            # Определяем, сколько признаков нужно на основе ожидаемого количества
            # Если не указано, используем 8 (6 базовых + 2 риск-параметра)
            # Но если модель ожидает 7, используем только 1 риск-параметр
            target_count = expected_count if expected_count is not None else 8
            
            if target_count == 7:
                # Старая модель с 7 признаками: 6 базовых + 1 риск-параметр
                if risk_params:
                    stop_loss = risk_params.get('stop_loss', 15.0)
                    features.append(stop_loss)
                else:
                    features.append(0)
            else:
                # Стандартная старая версия: 6 базовых + 2 риск-параметра = 8
                if risk_params:
                    stop_loss = risk_params.get('stop_loss', 15.0)
                    take_profit = risk_params.get('take_profit', 20.0)
                    features.extend([stop_loss, take_profit])
                else:
                    features.extend([0, 0])
            
            return np.array(features).reshape(1, -1)
        
        # НОВАЯ ВЕРСИЯ: Расширенные признаки
        # Ширина зон входа/выхода
        long_entry_zone_width = overbought - oversold
        long_exit_zone_width = exit_long_with - exit_long_against
        short_exit_zone_width = exit_short_against - exit_short_with
        
        features.extend([
            long_entry_zone_width,
            long_exit_zone_width,
            short_exit_zone_width,
        ])
        
        # Отношения параметров (нормализованные)
        oversold_ratio = oversold / 50.0
        overbought_ratio = overbought / 50.0
        exit_long_with_ratio = exit_long_with / 50.0
        exit_short_with_ratio = exit_short_with / 50.0
        
        features.extend([
            oversold_ratio,
            overbought_ratio,
            exit_long_with_ratio,
            exit_short_with_ratio,
        ])
        
        # Разница между входом и выходом
        long_entry_exit_diff = exit_long_with - oversold
        short_entry_exit_diff = overbought - exit_short_with
        
        features.extend([
            long_entry_exit_diff,
            short_entry_exit_diff,
        ])
        
        # Добавляем риск-параметры если есть
        if risk_params:
            stop_loss = risk_params.get('stop_loss', 15.0)
            take_profit = risk_params.get('take_profit', 20.0)
            trailing_activation = risk_params.get('trailing_stop_activation', 30.0)
            trailing_distance = risk_params.get('trailing_stop_distance', 5.0)
            
            features.extend([
                stop_loss,
                take_profit,
                trailing_activation,
                trailing_distance,
            ])
            
            # Производные признаки для риск-параметров
            risk_reward_ratio = take_profit / max(stop_loss, 0.1)
            trailing_coverage = trailing_distance / max(trailing_activation, 0.1)
            
            features.extend([
                risk_reward_ratio,
                trailing_coverage,
            ])
        else:
            # Заполняем нулями если нет
            features.extend([0, 0, 0, 0, 0, 0])
        
        return np.array(features).reshape(1, -1)
    
    def add_training_sample(self, rsi_params: Dict, win_rate: float, total_pnl: float,
                            trades_count: int, risk_params: Optional[Dict] = None,
                            symbol: Optional[str] = None, blocked: bool = False,
                            rsi_entered_zones: int = 0, filters_blocked: int = 0,
                            block_reasons: Optional[Dict[str, int]] = None):
        """
        Добавить образец для обучения
        
        Args:
            rsi_params: Параметры RSI
            win_rate: Win Rate (0-100)
            total_pnl: Total PnL
            trades_count: Количество сделок
            risk_params: Параметры риск-менеджмента
            symbol: Символ монеты
            blocked: Были ли входы заблокированы
            rsi_entered_zones: Сколько раз RSI входил в зоны входа (для градации качества)
        """
        if not self.ai_db:
            logger.warning("⚠️ AI Database недоступна, образец не сохранен")
            return
        
        try:
            # Вычисляем качество (target для обучения)
            # Качество = комбинация win_rate, pnl, trades_count
            # Если заблокировано - используем отрицательное качество для разнообразия
            if blocked or trades_count == 0:
                # ВАЖНО: Используем отрицательное качество вместо 0.0
                # Это позволяет модели различать заблокированные параметры
                # Градация качества для заблокированных:
                # -0.10: RSI не входил в зоны (параметры не подходят)
                # -0.05: RSI входил в зоны, но все заблокированы фильтрами
                # -0.02: Были попытки входа (win_rate > 0)
                
                if rsi_entered_zones > 0:
                    # RSI входил в зоны, но входы заблокированы фильтрами
                    # Это лучше чем параметры, которые вообще не дают сигналов
                    # Базовое качество зависит от количества попыток
                    base_quality = -0.05 + (0.01 * min(rsi_entered_zones / 20.0, 1.0))  # -0.05 до -0.04
                    
                    # Улучшаем качество если были заблокированные попытки (значит фильтры работают)
                    if filters_blocked > 0:
                        # Чем больше попыток было заблокировано, тем лучше параметры
                        # (значит они хотя бы генерируют сигналы)
                        blocked_ratio = min(filters_blocked / max(rsi_entered_zones, 1), 1.0)
                        base_quality += 0.01 * blocked_ratio  # До -0.03
                    
                    # Учитываем типы блокировок
                    if block_reasons:
                        # Если блокируется только одним типом фильтра - это лучше
                        # (значит можно оптимизировать параметры под этот фильтр)
                        unique_reasons = len(block_reasons)
                        if unique_reasons == 1:
                            base_quality += 0.005  # Немного лучше
                        elif unique_reasons >= 3:
                            base_quality -= 0.005  # Хуже если много разных причин
                    
                    quality = base_quality
                else:
                    # RSI не входил в зоны - параметры не подходят для этой монеты
                    quality = -0.10
                
                # Если есть win_rate > 0, значит были попытки, но заблокированы
                # Это лучше чем полное отсутствие сигналов
                if win_rate > 0:
                    quality = max(quality, -0.02)  # Не хуже -0.02 если были попытки
            else:
                # Нормализуем метрики
                win_rate_norm = win_rate / 100.0  # 0-1
                pnl_norm = min(max(total_pnl / 1000.0, -1), 1)  # -1 до 1 (1000 USDT = 1.0)
                trades_norm = min(trades_count / 50.0, 1)  # 0-1 (50 сделок = 1.0)
                
                # Взвешенная сумма (положительное качество)
                quality = (
                    win_rate_norm * 0.5 +
                    pnl_norm * 0.3 +
                    trades_norm * 0.2
                )
                
                # Обеспечиваем, что качество всегда положительное для успешных параметров
                quality = max(quality, 0.01)  # Минимум 0.01 для параметров с сделками
            
            # Сохраняем в БД
            sample = {
                'rsi_params': rsi_params,
                'risk_params': risk_params or {},
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'trades_count': trades_count,
                'quality': quality,
                'blocked': blocked,
                'rsi_entered_zones': rsi_entered_zones,
                'filters_blocked': filters_blocked,
                'block_reasons': block_reasons or {},
                'symbol': symbol
            }
            
            sample_id = self.ai_db.save_parameter_training_sample(sample)
            if sample_id:
                try:
                    pass
                except MemoryError:
                    # Не логируем при MemoryError
                    pass
            else:
                try:
                    logger.warning("⚠️ Не удалось сохранить образец в БД")
                except MemoryError:
                    # Не логируем при MemoryError
                    pass
                
        except MemoryError:
            # КРИТИЧНО: Не логируем при MemoryError (это вызывает рекурсию)
            # Просто пропускаем - graceful degradation
            pass
        except Exception as e:
            # Используем безопасное логирование
            try:
                logger.error(f"❌ Ошибка добавления образца: {e}")
            except MemoryError:
                # Не логируем при MemoryError
                pass
    
    def train(self, min_samples: int = 50) -> Optional[Dict[str, Any]]:
        """
        Обучить модель на накопленных данных
        
        Args:
            min_samples: Минимальное количество образцов для обучения
        
        Returns:
            Словарь с метриками обучения или None если обучение не удалось
        """
        if not self.ai_db:
            logger.warning("⚠️ AI Database недоступна, обучение невозможно")
            return {
                'success': False,
                'reason': 'database_unavailable'
            }
        
        try:
            # Загружаем данные из БД (последние 5000 образцов)
            training_data = self.ai_db.get_parameter_training_samples(limit=5000)
            
            samples_count = len(training_data)
            if samples_count < min_samples:
                logger.warning(f"⚠️ Недостаточно данных для обучения: {samples_count}/{min_samples}")
                return {
                    'success': False,
                    'samples_count': samples_count,
                    'min_samples_required': min_samples,
                    'reason': 'not_enough_samples'
                }
            
            # Подготавливаем данные
            X = []
            y = []
            
            for sample in training_data:
                features = self._extract_features(
                    sample['rsi_params'],
                    sample.get('risk_params')
                )
                X.append(features[0])
                y.append(sample['quality'])
            
            X = np.array(X)
            y = np.array(y)
            
            # Нормализуем признаки
            X_scaled = self.scaler.fit_transform(X)
            
            # УЛУЧШЕНИЕ: Пробуем несколько алгоритмов и выбираем лучший
            models_to_try = []
            
            # GradientBoostingRegressor (базовый)
            models_to_try.append((
                'GradientBoosting',
                GradientBoostingRegressor(
                    n_estimators=200,  # Увеличиваем количество деревьев
                    max_depth=6,  # Увеличиваем глубину
                    learning_rate=0.05,  # Уменьшаем learning rate для лучшей сходимости
                    random_state=42,
                    n_iter_no_change=15,
                    subsample=0.8  # Добавляем subsample для уменьшения переобучения
                )
            ))
            
            # RandomForestRegressor (альтернатива)
            models_to_try.append((
                'RandomForest',
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1  # без параллелизма — устраняет UserWarning про delayed/Parallel
                )
            ))
            
            # XGBoost (если доступен) - обычно лучший для табличных данных
            if XGBOOST_AVAILABLE:
                models_to_try.append((
                    'XGBoost',
                    XGBRegressor(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.05,
                        random_state=42,
                        n_jobs=1,  # без параллелизма — устраняет UserWarning про delayed/Parallel
                        subsample=0.8,
                        colsample_bytree=0.8
                    )
                ))
            
            # Обучаем все модели и выбираем лучшую
            best_model = None
            best_score = -float('inf')
            best_model_name = None
            
            logger.info(f"🎓 Обучение моделей предсказания качества параметров на {len(X)} образцах...")
            
            for model_name, model in models_to_try:
                try:
                    model.fit(X_scaled, y)
                    score = model.score(X_scaled, y)
                    
                    # Кросс-валидация для более надежной оценки
                    cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X) // 10), scoring='r2', n_jobs=1)
                    cv_mean = np.mean(cv_scores)
                    
                    logger.info(f"   📊 {model_name}: R² = {score:.3f}, CV R² = {cv_mean:.3f}")
                    
                    # Выбираем модель с лучшим CV score (более надежная метрика)
                    if cv_mean > best_score:
                        best_score = cv_mean
                        best_model = model
                        best_model_name = model_name
                except Exception as e:
                    pass
            
            if best_model is None:
                # Fallback на GradientBoosting если все не удались
                logger.warning("⚠️ Все модели не удалось обучить, используем GradientBoosting по умолчанию")
                best_model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                best_model.fit(X_scaled, y)
                best_model_name = "GradientBoosting (fallback)"
                best_score = best_model.score(X_scaled, y)
            
            self.model = best_model
            
            # Финальная оценка
            train_score = self.model.score(X_scaled, y)
            logger.info(f"✅ Модель обучена! Выбрана: {best_model_name}, R² score: {train_score:.3f}, CV R²: {best_score:.3f}")
            
            # Статистика по качеству образцов
            avg_quality = float(np.mean(y))
            max_quality = float(np.max(y))
            min_quality = float(np.min(y))
            blocked_count = sum(1 for s in training_data if s.get('blocked', False))
            
            self.is_trained = True
            # Устанавливаем expected_features на количество признаков новой версии
            self.expected_features = X.shape[1]
            
            # Сохраняем метаданные о последнем обучении для автоматического переобучения
            self._last_trained_samples_count = samples_count
            self._last_trained_time = datetime.now()
            
            self._save_model()
            
            return {
                'success': True,
                'samples_count': samples_count,
                'r2_score': float(train_score),
                'avg_quality': avg_quality,
                'max_quality': max_quality,
                'min_quality': min_quality,
                'blocked_samples': blocked_count,
                'successful_samples': samples_count - blocked_count
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения модели: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'reason': str(e)
            }
    
    def predict_quality(self, rsi_params: Dict, risk_params: Optional[Dict] = None) -> float:
        """
        Предсказать качество параметров
        
        Args:
            rsi_params: Параметры RSI
            risk_params: Параметры риск-менеджмента
        
        Returns:
            Предсказанное качество (может быть отрицательным для плохих параметров)
            Положительное = хорошие параметры, отрицательное = заблокированные/плохие
        """
        if not self.is_trained or not self.model:
            return 0.0  # Нейтральное значение если модель не обучена
        
        try:
            # УЛУЧШЕНИЕ: Используем сохраненное значение expected_features из _load_model
            # Если оно не определено, пытаемся определить через атрибуты scaler
            expected_features = self.expected_features
            if expected_features is None:
                if hasattr(self.scaler, 'n_features_in_'):
                    expected_features = self.scaler.n_features_in_
                elif hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                    expected_features = len(self.scaler.mean_)
            
            # Определяем, какую версию признаков использовать
            if expected_features is not None and expected_features < 21:
                # Старая модель - используем legacy версию сразу
                features = self._extract_features_legacy(rsi_params, risk_params, num_features=expected_features)
            else:
                # Новая модель или не можем определить - используем новую версию
                features = self._extract_features(rsi_params, risk_params)
                
                # Если не удалось определить через атрибуты, пробуем через попытку transform
                if expected_features is None:
                    try:
                        # Пробуем transform с текущими признаками - если не совпадает, получим ошибку
                        test_features = features.copy()
                        self.scaler.transform(test_features)
                        # Если успешно - количество совпадает
                        expected_features = features.shape[1]
                    except ValueError as ve:
                        # Извлекаем количество из ошибки
                        error_msg = str(ve)
                        import re
                        match = re.search(r'expecting (\d+) features', error_msg)
                        if match:
                            expected_features = int(match.group(1))
                            features = self._extract_features_legacy(rsi_params, risk_params, num_features=expected_features)
                else:
                    # Проверяем совместимость
                    actual_features = features.shape[1]
                    if actual_features != expected_features:
                        features = self._extract_features_legacy(rsi_params, risk_params, num_features=expected_features)
            
            features_scaled = self.scaler.transform(features)
            quality = self.model.predict(features_scaled)[0]
            # НЕ ограничиваем - модель может предсказывать отрицательные значения
            # Это важно для различения плохих и хороших параметров
            return float(quality)
        except ValueError as ve:
            # Специальная обработка ошибки несовместимости признаков
            error_msg = str(ve)
            if 'expecting' in error_msg and 'features' in error_msg:
                # Извлекаем количество ожидаемых признаков из ошибки
                import re
                match = re.search(r'expecting (\d+) features', error_msg)
                if match:
                    expected_features = int(match.group(1))
                    # Пробуем использовать legacy режим
                    if expected_features in [7, 8, 10]:
                        try:
                            features = self._extract_features_legacy(rsi_params, risk_params, num_features=expected_features)
                            features_scaled = self.scaler.transform(features)
                            quality = self.model.predict(features_scaled)[0]
                            return float(quality)
                        except Exception as e2:
                            logger.warning(
                                f"⚠️ Не удалось использовать legacy режим: {e2}. "
                                f"Модель нужно переобучить с новыми признаками!"
                            )
                    else:
                        logger.warning(
                            f"⚠️ Несовместимость признаков модели: {error_msg}. "
                            f"Модель ожидает {expected_features} признаков, но legacy режим не поддерживает это количество. "
                            f"Модель нужно переобучить с новыми признаками!"
                        )
                else:
                    logger.warning(
                        f"⚠️ Несовместимость признаков модели: {error_msg}. "
                        f"Модель нужно переобучить с новыми признаками!"
                    )
            else:
                pass
            return 0.0
        except Exception as e:
            pass
            return 0.0
    
    def suggest_optimal_params(self, base_params: Dict, risk_params: Optional[Dict] = None,
                               num_suggestions: int = 10) -> List[Tuple[Dict, float]]:
        """
        Предложить оптимальные параметры на основе модели
        
        Args:
            base_params: Базовые параметры
            risk_params: Параметры риск-менеджмента
            num_suggestions: Количество предложений
        
        Returns:
            Список кортежей (параметры, предсказанное_качество)
            Только параметры с положительным качеством (не заблокированные)
        """
        if not self.is_trained:
            return []
        
        import random
        
        suggestions = []
        
        # Генерируем больше вариантов, чтобы найти хорошие
        max_attempts = num_suggestions * 20  # Увеличиваем для лучшего поиска
        
        # УЛУЧШЕНИЕ: Убираем жесткие ограничения, позволяем ИИ генерировать параметры свободно
        # Базовые значения используются как отправная точка, но ИИ может выходить за их пределы
        base_oversold = base_params.get('oversold', 29)
        base_overbought = base_params.get('overbought', 71)
        base_exit_long_with = base_params.get('exit_long_with_trend', 65)
        base_exit_long_against = base_params.get('exit_long_against_trend', 60)
        base_exit_short_with = base_params.get('exit_short_with_trend', 35)
        base_exit_short_against = base_params.get('exit_short_against_trend', 40)
        
        for _ in range(max_attempts):
            # Генерируем параметры с широким диапазоном вариации
            # ИИ может генерировать параметры от 10 до 90 для RSI (разумные границы)
            # Вариация может быть до ±20 от базового значения
            variation_range = 20  # Широкий диапазон для адаптации
            
            rsi_params = {
                'oversold': max(10, min(60, 
                    base_oversold + random.randint(-variation_range, variation_range))),
                'overbought': max(40, min(90,
                    base_overbought + random.randint(-variation_range, variation_range))),
                'exit_long_with_trend': max(30, min(85,
                    base_exit_long_with + random.randint(-variation_range, variation_range))),
                'exit_long_against_trend': max(25, min(80,
                    base_exit_long_against + random.randint(-variation_range, variation_range))),
                'exit_short_with_trend': max(15, min(70,
                    base_exit_short_with + random.randint(-variation_range, variation_range))),
                'exit_short_against_trend': max(20, min(75,
                    base_exit_short_against + random.randint(-variation_range, variation_range)))
            }
            
            quality = self.predict_quality(rsi_params, risk_params)
            
            # ВАЖНО: Фильтруем только параметры с положительным качеством
            # Отрицательное качество = заблокированные/плохие параметры
            if quality > 0:
                suggestions.append((rsi_params, quality))
            
            # Если нашли достаточно хороших параметров - останавливаемся
            if len(suggestions) >= num_suggestions:
                break
        
        # Сортируем по качеству (лучшие первыми) и возвращаем топ
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:num_suggestions]

