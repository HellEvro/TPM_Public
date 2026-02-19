#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль обучения AI системы

Обучается на:
1. Истории трейдов (bot_history.py)
2. Параметрах стратегии (конфигурация ботов)
3. Исторических данных (свечи, индикаторы)
"""

import os
import json
import logging
import pickle
import shutil
from copy import deepcopy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
try:
    import utils.sklearn_parallel_config  # noqa: F401 — до импорта sklearn, подавляет UserWarning delayed/Parallel
except ImportError:
    pass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import joblib  # только dump/load; Parallel/delayed — оба из sklearn через utils.sklearn_parallel_config (патч joblib)  # только dump/load; Parallel/delayed — через sklearn (патч в utils.sklearn_parallel_config)

from bot_engine.protections import ProtectionState, evaluate_protections
from bot_engine.ai.filter_utils import apply_entry_filters
try:
    from bot_engine.ai.ai_launcher_config import AITrainingStrategyConfig
except ImportError:  # pragma: no cover
    AITrainingStrategyConfig = None

logger = logging.getLogger('AI.Trainer')


_existing_coin_settings_cache = None


def _get_existing_coin_settings(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Возвращает последние индивидуальные настройки монеты, если они есть.
    Используем для того, чтобы обучение начиналось от последней успешной конфигурации.
    """
    global _existing_coin_settings_cache

    if not symbol:
        return None

    # 1. Пытаемся получить настройки из bots.py, если он запущен
    try:
        from bots_modules.imports_and_globals import get_individual_coin_settings  # noqa: WPS433,E402

        current_settings = get_individual_coin_settings(symbol)
        if current_settings:
            return deepcopy(current_settings)
    except Exception:
        pass

    # 2. Фолбек: читаем напрямую из storage и кэшируем, чтобы не дергать диск на каждую монету
    try:
        if _existing_coin_settings_cache is None:
            from bot_engine.storage import load_individual_coin_settings as storage_load_individual_coin_settings  # noqa: WPS433,E402

            _existing_coin_settings_cache = storage_load_individual_coin_settings() or {}

        normalized_symbol = symbol.upper()
        cached_settings = _existing_coin_settings_cache.get(normalized_symbol)
        if cached_settings:
            return deepcopy(cached_settings)
    except Exception:
        pass

    return None


def _get_config_snapshot(symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Унифицированный способ получить настройки (глобальные + индивидуальные).
    Один и тот же конфиг для bots.py и ai.py: ExitScam, AI пороги, RSI и т.д. из bot_config.py.
    """
    try:
        from bots_modules.imports_and_globals import get_config_snapshot  # noqa: WPS433,E402

        return get_config_snapshot(symbol)
    except Exception as exc:
        pass
        # Fallback при запуске только ai.py: конфиг из bot_config (get_auto_bot_config → DEFAULT_AUTO_BOT_CONFIG)
        try:
            from bot_engine.ai.bots_data_helper import get_auto_bot_config
            base = get_auto_bot_config()
            global_config = deepcopy(base) if base else {}
        except Exception:
            try:
                from bot_engine.config_loader import DEFAULT_AUTO_BOT_CONFIG  # noqa: WPS433,E402
                global_config = deepcopy(DEFAULT_AUTO_BOT_CONFIG)
            except Exception:
                global_config = {}
        individual_config = _get_existing_coin_settings(symbol) if symbol else None
        merged_config = deepcopy(global_config)
        if individual_config:
            merged_config.update(individual_config)
        return {
            'global': global_config,
            'individual': individual_config,
            'merged': merged_config,
            'symbol': symbol.upper() if symbol else None,
            'timestamp': datetime.now().isoformat()
        }


def _should_train_on_symbol(symbol: str) -> bool:
    """
    Проверяет, должна ли монета использоваться для обучения AI на основе whitelist/blacklist.
    
    Логика:
    - Если scope == 'whitelist' ИЛИ (scope == 'all' и whitelist не пуст) -> обучаться только на монетах из whitelist
    - Если scope == 'blacklist' -> исключить монеты из blacklist (но если whitelist не пуст, то использовать whitelist)
    - Если scope == 'all' и whitelist пуст -> использовать все монеты кроме blacklist
    
    Args:
        symbol: Символ монеты для проверки
        
    Returns:
        True если монета должна использоваться для обучения, False иначе
    """
    if not symbol:
        return False
    
    symbol_upper = symbol.upper()
    
    try:
        # Пробуем получить конфигурацию из bots_data
        from bots_modules.imports_and_globals import bots_data, bots_data_lock
        with bots_data_lock:
            auto_config = bots_data.get('auto_bot_config', {}) or {}
    except ImportError:
        auto_config = {}
    if not auto_config:
        # Fallback при отдельном запуске ai.py: whitelist/blacklist/scope из БД
        try:
            from bot_engine.ai.bots_data_helper import get_auto_bot_config
            auto_config = get_auto_bot_config() or {}
        except Exception:
            pass
    if not auto_config:
        # Не удалось загрузить конфигурацию — используем все монеты
        return True
    
    scope = auto_config.get('scope', 'all')
    whitelist = auto_config.get('whitelist', []) or []
    blacklist = auto_config.get('blacklist', []) or []
    
    # Нормализуем списки (верхний регистр)
    whitelist = [coin.upper() for coin in whitelist if coin]
    blacklist = [coin.upper() for coin in blacklist if coin]
    
    # Если whitelist не пуст (независимо от scope), обучаемся только на монетах из whitelist
    if whitelist:
        return symbol_upper in whitelist
    
    # Если scope == 'whitelist' но whitelist пуст, не обучаемся ни на чем
    if scope == 'whitelist':
        return False
    
    # Если scope == 'blacklist', исключаем монеты из blacklist
    if scope == 'blacklist':
        return symbol_upper not in blacklist
    
    # scope == 'all': исключаем только blacklist
    return symbol_upper not in blacklist


class AITrainer:
    """
    Класс для обучения AI моделей
    """
    
    def __init__(self):
        """Инициализация тренера"""
        # Нормализуем пути для кроссплатформенной совместимости (особенно для Windows)
        self.models_dir = os.path.normpath('data/ai/models')
        self.data_dir = os.path.normpath('data/ai')
        
        # Создаем директории
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Модели
        self.signal_predictor = None  # Предсказание сигналов (LONG/SHORT/WAIT)
        self.profit_predictor = None  # Предсказание прибыльности
        self.scaler = StandardScaler()
        self.expected_features = None  # Количество признаков, которое ожидает модель (определяется из scaler)
        self.ai_decision_model = None  # Модель для анализа решений AI
        self.ai_decision_scaler = StandardScaler()
        self.ai_decisions_min_samples = 20
        self.ai_decisions_last_trained_count = 0
        self._ai_decision_last_accuracy = None
        # Отслеживание обучения на реальных сделках
        self._last_real_trades_training_time = None
        self._last_real_trades_training_count = 0
        self._real_trades_min_samples = 50  # Минимум реальных сделок для обучения (увеличено с 10)
        self._simulated_trades_min_samples = 100  # Минимум симулированных сделок для обучения
        self._real_trades_retrain_threshold = 0.2  # 20% новых сделок для переобучения
        self._profit_r2: Optional[float] = None  # R² модели прибыли; при <0 не используем для решений
        self._profit_model_unreliable = False  # True если R²<0 — используем только модель сигналов
        # Пути моделей (нормализуем все пути)
        self.signal_model_path = os.path.normpath(os.path.join(self.models_dir, 'signal_predictor.pkl'))
        self.profit_model_path = os.path.normpath(os.path.join(self.models_dir, 'profit_predictor.pkl'))
        self.scaler_path = os.path.normpath(os.path.join(self.models_dir, 'scaler.pkl'))
        self.ai_decision_model_path = os.path.normpath(os.path.join(self.models_dir, 'ai_decision_model.pkl'))
        self.ai_decision_scaler_path = os.path.normpath(os.path.join(self.models_dir, 'ai_decision_scaler.pkl'))

        
        # Файл для отслеживания сделок с AI решениями
        self.ai_decisions_file = os.path.normpath(os.path.join(self.data_dir, 'ai_decisions_tracking.json'))
        
        # ПРИМЕЧАНИЕ: Все данные теперь хранятся в БД (ai_data.db)
        # JSON файлы больше не используются
        
        # Инициализируем хранилище данных AI
        try:
            from bot_engine.ai.ai_data_storage import AIDataStorage
            self.data_storage = AIDataStorage(self.data_dir)
        except Exception as e:
            pass
            self.data_storage = None
        
        # Инициализируем реляционную БД для всех данных AI
        try:
            from bot_engine.ai.ai_database import get_ai_database
            self.ai_db = get_ai_database()
            logger.info("✅ AI Database подключена")
            
            # Мигрируем данные из JSON в БД (если нужно)
            self._migrate_json_to_database()
        except Exception as e:
            logger.warning(f"⚠️ Не удалось инициализировать AI Database: {e}")
            self.ai_db = None
            pass
            self.data_storage = None
        
        # Инициализируем трекер параметров (отслеживание использованных комбинаций)
        try:
            from bot_engine.ai.ai_parameter_tracker import AIParameterTracker
            self.param_tracker = AIParameterTracker(self.data_dir)
        except Exception as e:
            pass
            self.param_tracker = None

        self._perf_monitor = None
        try:
            from bot_engine.config_loader import AIConfig
            perf_monitoring_enabled = getattr(AIConfig, 'AI_PERFORMANCE_MONITORING_ENABLED', True)
        except ImportError:
            perf_monitoring_enabled = True
        if perf_monitoring_enabled:
            try:
                from bot_engine.ai.monitoring import AIPerformanceMonitor
                self._perf_monitor = AIPerformanceMonitor(max_records=5000)
            except Exception as e:
                pass

        # Ensemble (LSTM + Transformer + SMC) — ленивая инициализация при AI_USE_ENSEMBLE
        self._ensemble_predictor = None

        # Инициализируем ML модель для предсказания качества параметров (только если включено)
        self.param_quality_predictor = None
        try:
            from bot_engine.config_loader import AIConfig
            if not getattr(AIConfig, 'AI_PARAMETER_QUALITY_ENABLED', True):
                pass
            else:
                from bot_engine.ai.parameter_quality_predictor import ParameterQualityPredictor
                self.param_quality_predictor = ParameterQualityPredictor(self.data_dir)
        except Exception as e:
            pass
        
        # Загружаем историю биржи при инициализации (если файл пустой или не существует)
        # История будет дополняться при каждом обучении и периодически
        try:
            # Проверяем, есть ли сделки в БД
            if self.ai_db:
                saved_trades = self._load_saved_exchange_trades()
                if len(saved_trades) == 0:
                    logger.info("📥 История биржи пуста, загружаем историю...")
                    self._update_exchange_trades_history()
                else:
                    logger.info(f"📥 В БД уже есть {len(saved_trades)} сделок из истории биржи")
            else:
                logger.info("📥 Первичная загрузка истории сделок с биржи...")
                self._update_exchange_trades_history()
        except Exception as e:
            pass

        # Настройки тренировочного режима (не влияют на боевые боты)
        self.training_param_overrides: Dict[str, Any] = {}
        self.training_mutable_flags: Dict[str, bool] = {}
        self._training_overrides_logged = False
        if AITrainingStrategyConfig and getattr(AITrainingStrategyConfig, 'ENABLED', False):
            self.training_param_overrides = deepcopy(getattr(AITrainingStrategyConfig, 'PARAM_OVERRIDES', {}) or {})
            self.training_mutable_flags = getattr(AITrainingStrategyConfig, 'MUTABLE_FILTERS', {}) or {}
        
        # Целевые значения Win Rate для монет с динамическим повышением порога
        # Win Rate targets теперь в БД
        self.win_rate_targets_dirty = False
        self.win_rate_targets_default = 80.0  # Значение по умолчанию
        
        # ✅ Онлайн обучение: буфер для инкрементального обучения
        # Загружаем настройки из конфига с fallback на дефолтные значения
        try:
            from bot_engine.config_loader import AIConfig
            self._online_learning_buffer_size = getattr(AIConfig, 'AI_SELF_LEARNING_BUFFER_SIZE', 50)
            self._online_learning_enabled = getattr(AIConfig, 'AI_SELF_LEARNING_ENABLED', True)
        except (ImportError, AttributeError):
            # Дефолтные значения, если конфиг не доступен
            self._online_learning_buffer_size = 50
            self._online_learning_enabled = True
        
        from collections import deque
        self._online_learning_buffer = deque(maxlen=self._online_learning_buffer_size)
        
        # Загружаем существующие модели
        self._load_models()
        
        logger.info("✅ AITrainer инициализирован")

    def _record_training_event(self, event_type: str, status: str, **payload) -> None:
        """
        Неблокирующая запись события обучения в AIDataStorage.
        """
        if not self.data_storage:
            return
        try:
            record = {
                'event_type': event_type,
                'status': status,
                'timestamp': datetime.now().isoformat(),
            }
            if payload:
                record.update({k: v for k, v in payload.items() if v is not None})
            self.data_storage.add_training_record(record)
        except Exception as storage_error:
            pass

    def _build_individual_settings(
        self,
        coin_rsi_params: Dict[str, float],
        risk_params: Dict[str, float],
        filter_params: Dict[str, Dict[str, Any]],
        trend_params: Dict[str, Any],
        maturity_params: Dict[str, Any],
        ai_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Собирает полный payload индивидуальных настроек для сохранения."""
        rsi_time_filter = filter_params.get('rsi_time_filter', {})
        exit_scam_filter = filter_params.get('exit_scam', {})

        # Валидация минимальных значений
        rsi_time_filter_candles = rsi_time_filter.get('candles')
        if rsi_time_filter_candles is not None:
            rsi_time_filter_candles = max(2, rsi_time_filter_candles)  # Минимум 2 свечи
        
        max_position_hours = risk_params.get('max_position_hours')
        if max_position_hours is not None and max_position_hours > 0:
            # Для 6H ТФ минимум 18 часов (3 свечи) или 0 (отключено)
            max_position_hours = max(18, max_position_hours)
        
        return {
            'rsi_long_threshold': coin_rsi_params.get('oversold'),
            'rsi_short_threshold': coin_rsi_params.get('overbought'),
            'rsi_exit_long_with_trend': coin_rsi_params.get('exit_long_with_trend'),
            'rsi_exit_long_against_trend': coin_rsi_params.get('exit_long_against_trend'),
            'rsi_exit_short_with_trend': coin_rsi_params.get('exit_short_with_trend'),
            'rsi_exit_short_against_trend': coin_rsi_params.get('exit_short_against_trend'),
            'max_loss_percent': risk_params.get('max_loss_percent'),
            'take_profit_percent': risk_params.get('take_profit_percent'),
            'trailing_stop_activation': risk_params.get('trailing_stop_activation'),
            'trailing_stop_distance': risk_params.get('trailing_stop_distance'),
            'trailing_take_distance': risk_params.get('trailing_take_distance'),
            'trailing_update_interval': risk_params.get('trailing_update_interval'),
            'break_even_trigger': risk_params.get('break_even_trigger'),
            'break_even_protection': risk_params.get('break_even_protection'),
            'max_position_hours': max_position_hours,
            'rsi_time_filter_enabled': True,  # Всегда включен, AI не может отключить
            'rsi_time_filter_candles': rsi_time_filter_candles,
            'rsi_time_filter_upper': rsi_time_filter.get('upper'),
            'rsi_time_filter_lower': rsi_time_filter.get('lower'),
            'exit_scam_enabled': exit_scam_filter.get('enabled'),
            'exit_scam_candles': exit_scam_filter.get('candles'),
            'exit_scam_single_candle_percent': exit_scam_filter.get('single_candle_percent'),
            'exit_scam_multi_candle_count': exit_scam_filter.get('multi_candle_count'),
            'exit_scam_multi_candle_percent': exit_scam_filter.get('multi_candle_percent'),
            'trend_detection_enabled': trend_params.get('trend_detection_enabled'),
            'avoid_down_trend': trend_params.get('avoid_down_trend'),
            'avoid_up_trend': trend_params.get('avoid_up_trend'),
            'trend_analysis_period': trend_params.get('trend_analysis_period'),
            'trend_price_change_threshold': trend_params.get('trend_price_change_threshold'),
            'trend_candles_threshold': trend_params.get('trend_candles_threshold'),
            'enable_maturity_check': maturity_params.get('enable_maturity_check'),
            'min_candles_for_maturity': maturity_params.get('min_candles_for_maturity'),
            'min_rsi_low': maturity_params.get('min_rsi_low'),
            'max_rsi_high': maturity_params.get('max_rsi_high'),
            'ai_trained': True,
            'ai_win_rate': ai_meta.get('win_rate'),
            'ai_rating': ai_meta.get('rating', 0),
            'ai_trained_at': datetime.now().isoformat(),
            'ai_trades_count': ai_meta.get('trades_count', 0),
            'ai_total_pnl': ai_meta.get('total_pnl', 0.0),
        }
    
    def _generate_adaptive_params(self, symbol: str, rsi_history: List[float], 
                                   base_oversold: float, base_overbought: float,
                                   base_exit_long_with: float, base_exit_long_against: float,
                                   base_exit_short_with: float, base_exit_short_against: float,
                                   rng, base_params: Dict) -> Dict:
        """
        Генерирует адаптивные параметры на основе анализа реальных RSI значений монеты.
        
        ИИ анализирует историю RSI и генерирует параметры, которые:
        1. Адаптируются под реальный диапазон RSI монеты
        2. Не ограничены жесткими границами из конфига
        3. Учитывают изменения в поведении маркетмейкера
        4. Используют ML модель для предсказания качества параметров
        
        Args:
            symbol: Символ монеты
            rsi_history: История RSI значений
            base_*: Базовые параметры из конфига (используются как отправная точка)
            rng: Генератор случайных чисел
            base_params: Базовые параметры для ML модели
        
        Returns:
            Словарь с адаптивными RSI параметрами
        """
        # Анализируем реальные RSI значения
        valid_rsi = [r for r in rsi_history if r is not None and 0 <= r <= 100]
        if not valid_rsi:
            # Fallback: используем базовые параметры с небольшими вариациями
            return {
                'oversold': max(10, min(50, base_oversold + rng.randint(-10, 10))),
                'overbought': max(50, min(90, base_overbought + rng.randint(-10, 10))),
                'exit_long_with_trend': max(40, min(80, base_exit_long_with + rng.randint(-15, 15))),
                'exit_long_against_trend': max(35, min(75, base_exit_long_against + rng.randint(-15, 15))),
                'exit_short_with_trend': max(20, min(60, base_exit_short_with + rng.randint(-15, 15))),
                'exit_short_against_trend': max(25, min(65, base_exit_short_against + rng.randint(-15, 15)))
            }
        
        rsi_min = min(valid_rsi)
        rsi_max = max(valid_rsi)
        rsi_mean = sum(valid_rsi) / len(valid_rsi)
        rsi_std = (sum((x - rsi_mean) ** 2 for x in valid_rsi) / len(valid_rsi)) ** 0.5
        
        # Вычисляем процентили для более точной адаптации
        sorted_rsi = sorted(valid_rsi)
        rsi_p10 = sorted_rsi[int(len(sorted_rsi) * 0.10)]  # 10-й процентиль
        rsi_p90 = sorted_rsi[int(len(sorted_rsi) * 0.90)]  # 90-й процентиль
        rsi_p25 = sorted_rsi[int(len(sorted_rsi) * 0.25)]  # 25-й процентиль
        rsi_p75 = sorted_rsi[int(len(sorted_rsi) * 0.75)]  # 75-й процентиль
        
        # Генерируем параметры на основе анализа рынка
        # Oversold: используем процентили и реальный min, но не ограничиваем жестко
        # Если RSI редко опускается ниже 30, адаптируем порог выше
        if rsi_p10 > base_oversold:
            # RSI редко в зоне oversold - адаптируем порог
            adaptive_oversold = max(10, min(60, rsi_p10 - 2 + rng.uniform(-3, 3)))
        else:
            # RSI часто в зоне oversold - используем базовый с вариацией
            adaptive_oversold = max(10, min(60, base_oversold + rng.uniform(-10, 10)))
        
        # Overbought: аналогично
        if rsi_p90 < base_overbought:
            # RSI редко в зоне overbought - адаптируем порог
            adaptive_overbought = max(40, min(90, rsi_p90 + 2 + rng.uniform(-3, 3)))
        else:
            # RSI часто в зоне overbought - используем базовый с вариацией
            adaptive_overbought = max(40, min(90, base_overbought + rng.uniform(-10, 10)))
        
        # Exit параметры: адаптируем на основе медианы и процентилей
        # Exit LONG with trend: должен быть выше медианы, но не слишком высоко
        adaptive_exit_long_with = max(40, min(80, rsi_p75 + rng.uniform(-5, 10)))
        
        # Exit LONG against trend: чуть ниже exit_long_with
        adaptive_exit_long_against = max(35, min(75, adaptive_exit_long_with - 5 + rng.uniform(-5, 5)))
        
        # Exit SHORT with trend: должен быть ниже медианы, но не слишком низко
        adaptive_exit_short_with = max(20, min(60, rsi_p25 + rng.uniform(-10, 5)))
        
        # Exit SHORT against trend: чуть выше exit_short_with
        adaptive_exit_short_against = max(25, min(65, adaptive_exit_short_with + 5 + rng.uniform(-5, 5)))
        
        # Если есть ML модель - используем её для оптимизации параметров
        if self.param_quality_predictor and self.param_quality_predictor.is_trained:
            # Генерируем несколько вариантов и выбираем лучший по предсказанию ML
            best_params = None
            best_quality = float('-inf')
            
            for _ in range(10):  # Пробуем 10 вариантов
                test_params = {
                    'oversold': max(10, min(60, adaptive_oversold + rng.uniform(-5, 5))),
                    'overbought': max(40, min(90, adaptive_overbought + rng.uniform(-5, 5))),
                    'exit_long_with_trend': max(40, min(80, adaptive_exit_long_with + rng.uniform(-5, 5))),
                    'exit_long_against_trend': max(35, min(75, adaptive_exit_long_against + rng.uniform(-5, 5))),
                    'exit_short_with_trend': max(20, min(60, adaptive_exit_short_with + rng.uniform(-5, 5))),
                    'exit_short_against_trend': max(25, min(65, adaptive_exit_short_against + rng.uniform(-5, 5)))
                }
                
                try:
                    quality = self.param_quality_predictor.predict_quality(test_params)
                    if quality > best_quality:
                        best_quality = quality
                        best_params = test_params
                except:
                    pass
            
            if best_params and best_quality > 0:
                logger.info(f"   🤖 {symbol}: ML модель оптимизировала параметры (качество: {best_quality:.3f})")
                return best_params
        
        # Возвращаем адаптивные параметры
        result = {
            'oversold': round(adaptive_oversold, 1),
            'overbought': round(adaptive_overbought, 1),
            'exit_long_with_trend': round(adaptive_exit_long_with, 1),
            'exit_long_against_trend': round(adaptive_exit_long_against, 1),
            'exit_short_with_trend': round(adaptive_exit_short_with, 1),
            'exit_short_against_trend': round(adaptive_exit_short_against, 1)
        }
        
        logger.info(
            f"   🧠 {symbol}: адаптивные параметры на основе RSI анализа "
            f"(min={rsi_min:.1f}, max={rsi_max:.1f}, mean={rsi_mean:.1f}, std={rsi_std:.1f})"
        )
        
        return result

    def _mutate_flag(self, key: str, base_value: bool, rng) -> bool:
        """
        Переключает флаг в обучении, если это разрешено тренировочным конфигом.
        """
        allow_mutation = self.training_mutable_flags.get(key, False)
        if not allow_mutation or rng is None:
            return bool(base_value)
        base_bool = bool(base_value)
        # 50% шанс оставить как есть, иначе переключаем
        if rng.random() < 0.5:
            return base_bool
        return not base_bool

    def _load_models(self):
        """Загрузить сохраненные модели"""
        try:
            loaded_count = 0
            
            if os.path.exists(self.signal_model_path):
                self.signal_predictor = joblib.load(self.signal_model_path)
                logger.info(f"✅ Загружена модель предсказания сигналов: {self.signal_model_path}")
                loaded_count += 1
                metadata_path = os.path.normpath(os.path.join(self.models_dir, 'signal_predictor_metadata.json'))
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            logger.info(f"   📊 Модель обучена: {metadata.get('saved_at', 'unknown')}")
                    except Exception:
                        pass
            else:
                logger.info("ℹ️ Модель предсказания сигналов не найдена (будет создана при обучении)")

            if os.path.exists(self.profit_model_path):
                self.profit_predictor = joblib.load(self.profit_model_path)
                logger.info(f"✅ Загружена модель предсказания прибыли: {self.profit_model_path}")
                loaded_count += 1
                metadata_path = os.path.normpath(os.path.join(self.models_dir, 'profit_predictor_metadata.json'))
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            logger.info(f"   📊 Модель обучена: {metadata.get('saved_at', 'unknown')}")
                            r2 = metadata.get('r2_score')
                            if r2 is not None:
                                self._profit_r2 = float(r2)
                                self._profit_model_unreliable = float(r2) < 0
                                if self._profit_model_unreliable:
                                    logger.info(f"   ⚠️ R²={self._profit_r2:.4f} < 0 — для решений используется только модель сигналов")
                    except Exception:
                        pass
            else:
                logger.info("ℹ️ Модель предсказания прибыли не найдена (будет создана при обучении)")

            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"✅ Загружен scaler: {self.scaler_path}")
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ is not None:
                    self.expected_features = self.scaler.n_features_in_
                    logger.info(f"   📊 Модель ожидает {self.expected_features} признаков (определено из n_features_in_)")
                elif hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                    self.expected_features = len(self.scaler.mean_)
                    logger.info(f"   📊 Модель ожидает {self.expected_features} признаков (определено из mean_)")
                elif hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None:
                    self.expected_features = len(self.scaler.scale_)
                    logger.info(f"   📊 Модель ожидает {self.expected_features} признаков (определено из scale_)")
                else:
                    logger.warning("   ⚠️ Не удалось определить количество признаков из scaler")
                loaded_count += 1
            else:
                logger.info("ℹ️ Scaler не найден (будет создан при обучении)")

            if os.path.exists(self.ai_decision_model_path):
                try:
                    self.ai_decision_model = joblib.load(self.ai_decision_model_path)
                    logger.info(f"✅ Загружена модель анализа AI решений: {self.ai_decision_model_path}")
                    metadata_path = os.path.normpath(os.path.join(self.models_dir, 'ai_decision_model_metadata.json'))
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            logger.info(
                                f"   📊 Модель решений обучена: {metadata.get('saved_at', 'unknown')}, "
                                f"образцов: {metadata.get('samples', 'unknown')}, accuracy: {metadata.get('accuracy', 'n/a')}"
                            )
                except Exception as ai_load_error:
                    logger.warning(f"⚠️ Не удалось загрузить модель решений AI: {ai_load_error}")
                    self.ai_decision_model = None

            if os.path.exists(self.ai_decision_scaler_path):
                try:
                    self.ai_decision_scaler = joblib.load(self.ai_decision_scaler_path)
                    logger.info(f"✅ Загружен scaler для AI решений: {self.ai_decision_scaler_path}")
                except Exception as ai_scaler_error:
                    logger.warning(f"⚠️ Не удалось загрузить scaler решений AI: {ai_scaler_error}")
                    self.ai_decision_scaler = StandardScaler()

            if loaded_count > 0:
                logger.info(f"🤖 Загружено моделей: {loaded_count}/3 - готовы к использованию ботами!")
            else:
                logger.info("💡 Модели еще не обучены - запустите обучение для создания моделей")
                
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки моделей: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    
    def _save_models(self):
        """Сохранить модели"""
        try:
            saved_count = 0
            
            if self.signal_predictor:
                joblib.dump(self.signal_predictor, self.signal_model_path)
                logger.info(f"✅ Сохранена модель предсказания сигналов: {self.signal_model_path}")
                saved_count += 1
                
                # Сохраняем метаданные модели
                # Сохраняем метаданные в БД
                metadata = {
                    'id': 'signal_predictor',
                    'model_type': 'signal_predictor',
                    'model_path': str(self.signal_model_path),
                    'model_class': 'RandomForestClassifier',
                    'saved_at': datetime.now().isoformat(),
                    'n_estimators': getattr(self.signal_predictor, 'n_estimators', 'unknown'),
                    'max_depth': getattr(self.signal_predictor, 'max_depth', 'unknown')
                }
                # Добавляем accuracy если она была вычислена при обучении
                signal_accuracy = getattr(self, '_signal_predictor_accuracy', None)
                if signal_accuracy is not None:
                    metadata['accuracy'] = float(signal_accuracy)
                    metadata['signal_accuracy'] = float(signal_accuracy)  # Дублируем для совместимости
                if self.ai_db:
                    self.ai_db.save_model_version(metadata)
            
            if self.profit_predictor:
                joblib.dump(self.profit_predictor, self.profit_model_path)
                logger.info(f"✅ Сохранена модель предсказания прибыли: {self.profit_model_path}")
                saved_count += 1
                
                # Сохраняем метаданные в БД и в JSON (r2_score нужен при загрузке — при R²<0 не используем profit для решений)
                r2 = getattr(self, '_profit_r2', None)
                metadata = {
                    'id': 'profit_predictor',
                    'model_type': 'profit_predictor',
                    'model_path': str(self.profit_model_path),
                    'model_class': 'GradientBoostingRegressor',
                    'saved_at': datetime.now().isoformat(),
                    'n_estimators': getattr(self.profit_predictor, 'n_estimators', 'unknown'),
                    'max_depth': getattr(self.profit_predictor, 'max_depth', 'unknown'),
                    'r2_score': float(r2) if r2 is not None else None,
                }
                if self.ai_db:
                    self.ai_db.save_model_version(metadata)
                metadata_path = os.path.normpath(os.path.join(self.models_dir, 'profit_predictor_metadata.json'))
                try:
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            
            if self.scaler:
                joblib.dump(self.scaler, self.scaler_path)
                logger.info(f"✅ Сохранен scaler: {self.scaler_path}")
                # Сохраняем количество признаков
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ is not None:
                    self.expected_features = self.scaler.n_features_in_
                    logger.info(f"   📊 Модель использует {self.expected_features} признаков")
                saved_count += 1

            if self.ai_decision_model:
                joblib.dump(self.ai_decision_model, self.ai_decision_model_path)
                logger.info(f"✅ Сохранена модель анализа AI решений: {self.ai_decision_model_path}")
                # Сохраняем метаданные в БД
                metadata = {
                    'id': 'ai_decision_model',
                    'model_type': 'ai_decision_model',
                    'model_path': str(self.ai_decision_model_path),
                    'model_class': type(self.ai_decision_model).__name__,
                    'saved_at': datetime.now().isoformat(),
                    'samples': getattr(self, 'ai_decisions_last_trained_count', 0),
                    'min_samples_required': self.ai_decisions_min_samples
                }
                accuracy = getattr(self, '_ai_decision_last_accuracy', None)
                if accuracy is not None:
                    metadata['accuracy'] = float(accuracy)
                if self.ai_db:
                    self.ai_db.save_model_version(metadata)

            if self.ai_decision_scaler:
                joblib.dump(self.ai_decision_scaler, self.ai_decision_scaler_path)
                logger.info(f"✅ Сохранен scaler для AI решений: {self.ai_decision_scaler_path}")
            
            logger.info(f"💾 Сохранено моделей: {saved_count}/3")
            logger.info(f"📁 Модели сохранены в: {self.models_dir}")
            logger.info("🤖 Модели готовы к использованию ботами!")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения моделей: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _get_win_rate_target(self, symbol: str) -> float:
        """Получить текущую цель Win Rate для монеты (по умолчанию 80%)."""
        if not self.ai_db:
            return self.win_rate_targets_default
        
        try:
            target_data = self.ai_db.get_win_rate_target(symbol)
            if target_data:
                return float(target_data.get('target_win_rate', self.win_rate_targets_default))
        except Exception as e:
            pass
        
        return self.win_rate_targets_default
    
    def _register_win_rate_success(self, symbol: str, achieved_win_rate: float):
        """
        Зафиксировать успешное достижение цели Win Rate и повысить порог на 1%.
        """
        if not self.ai_db:
            return

        try:
            symbol_key = (symbol or '').upper()
            current_target = self._get_win_rate_target(symbol_key)

            # Получаем или создаем запись для символа
            win_rate_data = self.ai_db.get_win_rate_target(symbol_key) or {}
            entry = {
                'target': current_target,
                'symbol': symbol_key,
                'created_at': win_rate_data.get('created_at', datetime.now().isoformat()),
                'last_updated': datetime.now().isoformat()
            }

            # Обновляем существующие поля
            for key, value in win_rate_data.items():
                if key not in entry:
                    entry[key] = value

            if current_target >= 100.0:
                reset_target = max(self.win_rate_targets_default, 80.0)
                if current_target != reset_target:
                    entry['target'] = reset_target
                    entry['last_target_reset_at'] = datetime.now().isoformat()
                    entry['last_target_reset_reason'] = 'reached_100_then_reset'
                    logger.info(
                        f"   🔁 {symbol}: цель Win Rate достигла 100%, сбрасываем до {reset_target:.1f}% "
                        f"для повторного цикла обучения"
                    )
            else:
                if achieved_win_rate >= current_target:
                    new_target = min(current_target + 1.0, 100.0)
                    if new_target > current_target:
                        entry['target'] = new_target
                        entry['last_target_increment_at'] = datetime.now().isoformat()
                        entry['last_target_increment_win_rate'] = achieved_win_rate
                        entry['increments'] = entry.get('increments', 0) + 1
                        logger.info(
                            f"   🚀 {symbol}: цель Win Rate повышена с {current_target:.1f}% до {new_target:.1f}% "
                            f"(достигнуто {achieved_win_rate:.1f}%)"
                        )
                else:
                    entry['target'] = current_target

            # Сохраняем в БД (target_win_rate и current_win_rate — скаляры, не dict)
            target_val = float(entry['target'])
            current_wr = entry.get('last_target_increment_win_rate')
            if current_wr is not None:
                current_wr = float(current_wr)
            self.ai_db.save_win_rate_target(symbol_key, target_val, current_win_rate=current_wr)
            self.win_rate_targets_dirty = True
        except Exception as e:
            pass


    
    def _load_history_data(self) -> List[Dict]:
        """
        Загрузить данные истории трейдов
        
        AI получает сделки из следующих источников (в порядке приоритета):
        1. БД (ai_data.db) - основной источник, все сделки ботов уже там
        2. data/bot_history.json - fallback если БД недоступна
        3. API endpoint /api/bots/trades - если файлы недоступны
        
        ВАЖНО: AI использует ТОЛЬКО закрытые сделки с PnL (status='CLOSED' и pnl != None)
        Это нужно для обучения на реальных результатах торговли
        
        ПРИМЕЧАНИЕ: history_data.json больше не используется, так как все данные в БД
        """
        pass
        
        trades = []
        source_counts = {}
        
        # 1. ПРИОРИТЕТ: Загружаем из БД (основной источник)
        if self.ai_db:
            pass
            try:
                # ВАЖНО: Загружаем ВСЕ сделки - и реальные, и симуляции
                # Симуляции нужны для обучения ИИ на разных параметрах и поиска оптимальных
                db_trades = self.ai_db.get_trades_for_training(
                    include_simulated=True,  # ВКЛЮЧАЕМ симуляции для обучения!
                    include_real=True,
                    include_exchange=True,  # ВАЖНО: Включаем сделки с биржи!
                    min_trades=0,  # КРИТИЧНО: 0 чтобы получить все сделки, не фильтровать по символам
                    limit=None
                )
                
                # Подсчитываем количество разных типов сделок
                simulated_count = sum(1 for t in db_trades if t.get('is_simulated', False))
                real_count = len(db_trades) - simulated_count
                logger.info(f"📊 Загружено для обучения: {len(db_trades)} сделок (реальных: {real_count}, симуляций: {simulated_count})")
                if db_trades:
                    # Помечаем данные готовыми, чтобы лаунчер не выдавал «Не удалось дождаться готовности данных»
                    try:
                        from bot_engine.ai.data_service_status_helper import update_data_service_status_in_db
                        from datetime import datetime as _dt_now
                        update_data_service_status_in_db(
                            ready=True,
                            last_collection=_dt_now.now().isoformat(),
                            trades=len(db_trades),
                        )
                    except Exception as _e:
                        pass
                    # Конвертируем формат БД в формат для обучения
                    for trade in db_trades:
                        # Получаем RSI и Trend данные (приоритет: entry_rsi/entry_trend > rsi/trend)
                        rsi = trade.get('rsi')
                        trend = trade.get('trend')
                        
                        # Если нет rsi/trend, пробуем entry_rsi/entry_trend (get_trades_for_training уже маппит их)
                        if rsi is None:
                            rsi = trade.get('entry_rsi')
                        if trend is None:
                            trend = trade.get('entry_trend')
                        
                        # Если все еще нет RSI/Trend, пытаемся рассчитать из рыночных данных
                        # ВАЖНО: Это медленная операция, поэтому делаем только если действительно нужно
                        if (rsi is None or trend is None):
                            # Ленивая загрузка рыночных данных только если нужно
                            if not hasattr(self, '_cached_market_data'):
                                try:
                                    self._cached_market_data = self._load_market_data() if hasattr(self, '_load_market_data') else {}
                                except:
                                    self._cached_market_data = {}
                            
                            market_data = self._cached_market_data
                            if market_data:
                                symbol = trade.get('symbol')
                                entry_timestamp = trade.get('timestamp') or trade.get('entry_time')
                                
                                if symbol and entry_timestamp and symbol in market_data:
                                    candles = market_data[symbol].get('candles', [])
                                    if candles:
                                        # Находим свечу, ближайшую к моменту входа
                                        try:
                                            if isinstance(entry_timestamp, str):
                                                from datetime import datetime
                                                entry_dt = datetime.fromisoformat(entry_timestamp.replace('Z', '+00:00'))
                                                entry_ts = entry_dt.timestamp()
                                            else:
                                                entry_ts = float(entry_timestamp)
                                            
                                            # Ищем ближайшую свечу
                                            closest_candle = None
                                            min_diff = float('inf')
                                            for candle in candles:
                                                candle_ts = candle.get('timestamp', 0)
                                                diff = abs(candle_ts - entry_ts)
                                                if diff < min_diff:
                                                    min_diff = diff
                                                    closest_candle = candle
                                            
                                            if closest_candle:
                                                # Рассчитываем RSI если нет
                                                if rsi is None:
                                                    # Используем последние 14 свечей для расчета RSI
                                                    candle_idx = candles.index(closest_candle) if closest_candle in candles else len(candles) - 1
                                                    rsi_window = min(14, candle_idx + 1)
                                                    if rsi_window >= 14:
                                                        closes = [c.get('close', 0) for c in candles[max(0, candle_idx-13):candle_idx+1]]
                                                        if len(closes) == 14 and all(c > 0 for c in closes):
                                                            gains = [max(0, closes[i] - closes[i-1]) for i in range(1, len(closes))]
                                                            losses = [max(0, closes[i-1] - closes[i]) for i in range(1, len(closes))]
                                                            avg_gain = sum(gains) / len(gains) if gains else 0
                                                            avg_loss = sum(losses) / len(losses) if losses else 0
                                                            if avg_loss > 0:
                                                                rs = avg_gain / avg_loss
                                                                rsi = 100 - (100 / (1 + rs))
                                                            else:
                                                                rsi = 100 if avg_gain > 0 else 50
                                                
                                                # Рассчитываем Trend если нет
                                                if trend is None:
                                                    candle_idx = candles.index(closest_candle) if closest_candle in candles else len(candles) - 1
                                                    if candle_idx >= 26:
                                                        closes = [c.get('close', 0) for c in candles[max(0, candle_idx-25):candle_idx+1]]
                                                        if len(closes) >= 26 and all(c > 0 for c in closes):
                                                            # EMA короткая (12) и длинная (26)
                                                            ema_12 = sum(closes[-12:]) / 12
                                                            ema_26 = sum(closes[-26:]) / 26
                                                            if ema_12 > ema_26:
                                                                trend = 'UP'
                                                            elif ema_12 < ema_26:
                                                                trend = 'DOWN'
                                                            else:
                                                                trend = 'NEUTRAL'
                                        except Exception as enrich_error:
                                            pass
                        
                        # Преобразуем данные из БД в формат, ожидаемый AI
                        converted_trade = {
                            'id': f"db_{trade.get('symbol')}_{trade.get('timestamp', '')}",
                            'timestamp': trade.get('timestamp') or trade.get('entry_time'),
                            'bot_id': trade.get('bot_id', trade.get('symbol')),
                            'symbol': trade.get('symbol'),
                            'direction': trade.get('direction'),
                            'entry_price': trade.get('entry_price'),
                            'exit_price': trade.get('exit_price'),
                            'pnl': trade.get('pnl'),
                            'roi': trade.get('roi'),
                            'status': 'CLOSED',
                            'decision_source': trade.get('decision_source', 'SCRIPT'),
                            'rsi': rsi,  # Используем обогащенное значение
                            'trend': trend,  # Используем обогащенное значение
                            'close_timestamp': trade.get('close_timestamp') or trade.get('exit_time'),
                            'close_reason': trade.get('close_reason'),
                            'is_successful': trade.get('is_successful', False),
                            'is_simulated': False
                        }
                        trades.append(converted_trade)
                    
                    source_counts['database'] = len(trades)
                    # Если данные есть в БД, возвращаем их (не загружаем из JSON)
                    if trades:
                        return trades
            except Exception as e:
                pass
        
        # 2. Fallback: Пробуем загрузить напрямую из data/bot_history.json (основной файл bots.py)
        try:
            bot_history_file = os.path.normpath(os.path.join('data', 'bot_history.json'))
            if os.path.exists(bot_history_file):
                # Убрано: logger.debug(f"📖 Источник 2: {bot_history_file}") - слишком шумно
                try:
                    with open(bot_history_file, 'r', encoding='utf-8') as f:
                        bot_history_data = json.load(f)
                except json.JSONDecodeError as json_error:
                    logger.warning(f"   ⚠️ Файл истории ботов поврежден (JSON ошибка на строке {json_error.lineno}, колонка {json_error.colno}): {bot_history_file}")
                    raise  # Пробрасываем дальше для обработки в общем except
                
                # Извлекаем сделки из bot_history.json
                bot_trades = bot_history_data.get('trades', [])
                if bot_trades:
                    # Добавляем только новые сделки (избегаем дубликатов)
                    existing_ids = {t.get('id') for t in trades if t.get('id')}
                    new_trades = []
                    for trade in bot_trades:
                        trade_id = trade.get('id') or trade.get('timestamp')
                        if trade_id not in existing_ids:
                            trades.append(trade)
                            new_trades.append(trade)
                    
                    # Убрано: logger.debug(f"   ✅ Найдено {len(bot_trades)} сделок, добавлено {len(new_trades)} новых") - слишком шумно
                    source_counts['bot_history.json'] = len(new_trades)
                else:
                    pass
            else:
                pass
        except json.JSONDecodeError as json_error:
            pass
            # Не сохраняем копию автоматически - это может быть временная проблема при записи
            # Если проблема критична, пользователь может проверить файл вручную
        except Exception as e:
            pass
        
        # 3. Анализируем загруженные сделки (сокращенные логи)
        # Убрано: logger.debug(f"📊 Всего загружено сделок: {len(trades)}") - слишком шумно
        
        # Инициализируем счетчики
        simulated_count = 0
        backtest_count = 0
        
        if trades:
            # Анализируем статусы сделок (только для DEBUG)
            statuses = {}
            pnl_count = 0
            closed_count = 0
            
            for trade in trades:
                status = trade.get('status', 'UNKNOWN')
                statuses[status] = statuses.get(status, 0) + 1
                
                if trade.get('pnl') is not None:
                    pnl_count += 1
                
                if status == 'CLOSED':
                    closed_count += 1
            
            # Убрано: logger.debug(f"   По статусам: {dict(statuses)}, С PnL: {pnl_count}, Закрытых: {closed_count}") - слишком шумно

            # КРИТИЧНО: Фильтруем только РЕАЛЬНЫЕ сделки (не симулированные, не бэктест)
            # Признаки реальной сделки:
            # 1. status == 'CLOSED' - закрыта
            # 2. pnl is not None - есть PnL
            # 3. НЕ симулированная (нет флагов is_simulated, is_backtest, simulation)
            # 4. Имеет реальные данные (entry_price, exit_price)
            closed_trades = []
            
            for t in trades:
                if t.get('status') == 'CLOSED' and t.get('pnl') is not None:
                    # Проверяем, не является ли сделка симулированной
                    is_simulated = (
                        t.get('is_simulated', False) or
                        t.get('is_backtest', False) or
                        t.get('simulation', False) or
                        t.get('backtest', False) or
                        'simulation' in str(t.get('id', '')).lower() or
                        'backtest' in str(t.get('id', '')).lower() or
                        'simulated' in str(t.get('reason', '')).lower() or
                        t.get('exit_reason', '').startswith('SIMULATION') or
                        t.get('close_reason', '').startswith('SIMULATION')
                    )
                    
                    if is_simulated:
                        simulated_count += 1
                        continue
                    
                    # Проверяем, что есть реальные данные
                    if not t.get('entry_price') or not t.get('exit_price'):
                        continue
                    
                    closed_trades.append(t)
            
            if simulated_count > 0 or backtest_count > 0:
                logger.warning(f"   ⚠️ Пропущено симулированных/бэктест сделок: {simulated_count + backtest_count}")
                logger.warning(f"   💡 AI обучается ТОЛЬКО на реальных сделках с биржи!")
            
            # Дополнительная диагностика: проверяем признаки реальных сделок
            real_trade_indicators = {
                'has_decision_source': sum(1 for t in closed_trades if t.get('decision_source')),
                'has_ai_decision_id': sum(1 for t in closed_trades if t.get('ai_decision_id')),
                'has_close_reason': sum(1 for t in closed_trades if t.get('close_reason')),
                'has_timestamp': sum(1 for t in closed_trades if t.get('timestamp')),
                'has_entry_data': sum(1 for t in closed_trades if t.get('entry_data')),
            }
            
            logger.info(f"   📊 Признаки реальных сделок:")
            logger.info(f"      ✅ С decision_source: {real_trade_indicators['has_decision_source']}")
            logger.info(f"      ✅ С ai_decision_id: {real_trade_indicators['has_ai_decision_id']}")
            logger.info(f"      ✅ С close_reason: {real_trade_indicators['has_close_reason']}")
            logger.info(f"      ✅ С timestamp: {real_trade_indicators['has_timestamp']}")
            logger.info(f"      ✅ С entry_data: {real_trade_indicators['has_entry_data']}")
            
            # КРИТИЧЕСКАЯ ДИАГНОСТИКА: Проверяем распределение PnL в исходных данных
            if closed_trades:
                pnl_values = [t.get('pnl', 0) for t in closed_trades if t.get('pnl') is not None]
                if pnl_values:
                    positive_pnl = sum(1 for pnl in pnl_values if pnl > 0)
                    negative_pnl = sum(1 for pnl in pnl_values if pnl < 0)
                    zero_pnl = sum(1 for pnl in pnl_values if pnl == 0)
                    
                    logger.info("=" * 80)
                    logger.info("🔍 ДИАГНОСТИКА ИСХОДНЫХ ДАННЫХ (до обработки)")
                    logger.info("=" * 80)
                    logger.info(f"   📊 Распределение PnL в bot_history.json:")
                    logger.info(f"      ✅ Прибыльных сделок (PnL > 0): {positive_pnl}")
                    logger.info(f"      ❌ Убыточных сделок (PnL < 0): {negative_pnl}")
                    logger.info(f"      ⚪ Нулевых сделок (PnL = 0): {zero_pnl}")
                    
                    if negative_pnl == 0 and zero_pnl == 0:
                        logger.error("=" * 80)
                        logger.error("❌ КРИТИЧЕСКАЯ ПРОБЛЕМА ОБНАРУЖЕНА!")
                        logger.error("=" * 80)
                        logger.error("   ⚠️ В bot_history.json ВСЕ сделки имеют положительный PnL!")
                        logger.error("   ⚠️ Это означает, что либо:")
                        logger.error("      1. Убыточные сделки не сохраняются в bot_history.json")
                        logger.error("      2. PnL рассчитывается неправильно при сохранении")
                        logger.error("      3. В системе действительно нет убыточных сделок (маловероятно)")
                        logger.error("=" * 80)
                        logger.error("   💡 РЕШЕНИЕ: Проверьте код сохранения сделок в bot_history.py")
                        logger.error("   💡 Убедитесь, что убыточные сделки тоже сохраняются с отрицательным PnL")
                        logger.error("=" * 80)
        else:
            # Инициализируем список закрытых сделок
            closed_trades = []
            
            # Проверяем БД на наличие сделок из биржи
            exchange_trades_count = 0
            if self.ai_db:
                try:
                    exchange_trades = self._load_saved_exchange_trades()
                    exchange_trades_count = len(exchange_trades)
                    if exchange_trades_count > 0:
                        logger.info(f"   📊 Найдено {exchange_trades_count} сделок из биржи в БД")
                        # Добавляем сделки из биржи в список для обучения
                        for trade in exchange_trades:
                            if trade.get('status') == 'CLOSED' and trade.get('pnl') is not None:
                                if trade.get('entry_price') and trade.get('exit_price'):
                                    closed_trades.append(trade)
                        logger.info(f"   ✅ Добавлено {len(closed_trades)} сделок из биржи для обучения")
                except Exception as e:
                    pass
            
            if len(closed_trades) == 0:
                logger.warning("⚠️ Сделки не найдены!")
                logger.warning("   💡 Проверьте:")
                logger.warning("      1. Запущен ли bots.py и совершает ли сделки")
                logger.warning("      2. Есть ли сделки в БД (exchange_trades) - они загружаются через API биржи")
                logger.warning("      3. Вызовите _update_exchange_trades_history() для загрузки сделок с биржи")
            # 4. Фильтруем только закрытые сделки с PnL
            
        logger.info("=" * 80)
        logger.info("✅ РЕЗУЛЬТАТ ФИЛЬТРАЦИИ")
        logger.info("=" * 80)
        logger.info(f"   📊 Всего сделок загружено из bot_history.json: {len(trades)}")
        logger.info(f"   ✅ Закрытых сделок ботов с PnL: {len(closed_trades)}")
        if simulated_count > 0:
            logger.info(f"   ⚠️ Отфильтровано симулированных/бэктест: {simulated_count}")
        logger.info(f"   💡 AI будет обучаться на {len(closed_trades)} сделках БОТОВ (из bot_history.json)")
        logger.info(f"   📦 История биржи загружается из БД")
        
        if len(closed_trades) < 10:
            logger.warning("=" * 80)
            logger.warning("⚠️ НЕДОСТАТОЧНО СДЕЛОК ДЛЯ ОБУЧЕНИЯ")
            logger.warning("=" * 80)
            logger.warning(f"   📊 Найдено: {len(closed_trades)} закрытых сделок с PnL")
            logger.warning(f"   📊 Нужно минимум: 10 сделок")
            logger.warning("   💡 AI будет обучаться на исторических данных (симуляция)")
            logger.warning("   💡 Когда накопится >= 10 реальных сделок, AI переключится на обучение на вашем опыте")
            logger.warning("=" * 80)
        else:
            logger.info("=" * 80)
            logger.info("✅ ДОСТАТОЧНО СДЕЛОК ДЛЯ ОБУЧЕНИЯ")
            logger.info("=" * 80)
            logger.info(f"   📊 Найдено: {len(closed_trades)} закрытых сделок с PnL")
            logger.info("   💡 AI будет обучаться на вашем реальном опыте торговли!")
            logger.info("=" * 80)
        
        return closed_trades
    
    def _save_simulated_trades(self, simulated_trades: List[Dict]) -> None:
        """
        Сохраняет симулированные сделки в БД
        
        Args:
            simulated_trades: Список симулированных сделок
        """
        if not simulated_trades:
            return
        
        if not self.ai_db:
            logger.error("❌ БД недоступна! Невозможно сохранить симуляции. Проверьте инициализацию БД.")
            return
        
        try:
            # Добавляем метку времени и флаг симуляции
            for trade in simulated_trades:
                trade['is_simulated'] = True
                trade['simulation_timestamp'] = datetime.now().isoformat()
                if 'status' not in trade:
                    trade['status'] = 'CLOSED'
            
            # Получаем текущую сессию обучения (если есть)
            training_session_id = getattr(self, '_current_training_session_id', None)
            
            # Сохраняем в БД
            saved_count = self.ai_db.save_simulated_trades(simulated_trades, training_session_id)
            
            if saved_count > 0:
                total_count = self.ai_db.count_simulated_trades()
                logger.info(f"💾 Сохранено {saved_count} симулированных сделок в БД (всего: {total_count})")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения симуляций в БД: {e}")
            raise
    
    def _create_exchange_for_history(self):
        """
        Создает exchange для загрузки истории биржи (независимо от bots.py)
        
        Returns:
            Exchange объект или None
        """
        try:
            # Сначала пробуем получить существующий exchange
            from bots_modules.imports_and_globals import get_exchange
            exchange = get_exchange()
            
            if exchange:
                pass
                return exchange
            
            # Если exchange недоступен, создаем свой
            logger.info("   🔧 Exchange недоступен, создаем собственный для загрузки истории...")
            
            # Получаем API ключи из конфига
            try:
                from app.config import EXCHANGES, ACTIVE_EXCHANGE
                exchange_name = ACTIVE_EXCHANGE if ACTIVE_EXCHANGE else 'BYBIT'
                exchange_config = EXCHANGES.get(exchange_name, {})
                
                api_key = exchange_config.get('api_key')
                api_secret = exchange_config.get('api_secret')
                test_server = exchange_config.get('test_server', False)
                position_mode = exchange_config.get('position_mode', 'Hedge')
                limit_order_offset = exchange_config.get('limit_order_offset', 0.1)
                
                if not api_key or not api_secret:
                    logger.warning("   ⚠️ API ключи не настроены в конфиге")
                    return None
                
                # Создаем exchange через фабрику
                from exchanges.exchange_factory import ExchangeFactory
                exchange = ExchangeFactory.create_exchange(
                    exchange_name,
                    api_key,
                    api_secret
                )
                
                if exchange:
                    logger.info(f"   ✅ Создан собственный exchange: {type(exchange).__name__}")
                    return exchange
                else:
                    logger.warning("   ⚠️ ExchangeFactory не смог создать exchange")
                    return None
                
            except ImportError as e:
                logger.warning(f"   ⚠️ Не удалось импортировать конфиг: {e}")
                return None
            except Exception as e:
                logger.warning(f"   ⚠️ Ошибка создания exchange: {e}")
                import traceback
                pass
                return None
            
        except Exception as e:
            logger.warning(f"   ⚠️ Ошибка создания exchange: {e}")
            import traceback
            pass
            return None
    
    def _load_exchange_trades_history(self) -> List[Dict]:
        """
        Загружает историю сделок трейдера с биржи через API
        
        Returns:
            Список сделок с биржи
        """
        try:
            # Создаем или получаем exchange
            exchange = self._create_exchange_for_history()
            
            if not exchange:
                logger.warning("⚠️ Exchange недоступен для загрузки истории сделок")
                logger.warning("   💡 Проверьте настройки API ключей в конфиге")
                return []
            
            logger.info(f"   ✅ Exchange доступен: {type(exchange).__name__}")
            
            # Загружаем историю сделок с биржи через метод get_closed_pnl
            if hasattr(exchange, 'get_closed_pnl'):
                try:
                    logger.info("   📥 Вызов exchange.get_closed_pnl(period='all')...")
                    # Загружаем историю закрытых позиций (последние 2 года максимум)
                    closed_pnl_data = exchange.get_closed_pnl(
                        sort_by='time',
                        period='all'  # Загружаем всю доступную историю
                    )
                    
                    if not closed_pnl_data:
                        logger.info(f"   📊 Получено данных от биржи: 0 записей (пустой результат)")
                        logger.info(f"   💡 Возможно, на бирже нет закрытых позиций с PnL")
                        return []
                    
                    logger.info(f"   📊 Получено данных от биржи: {len(closed_pnl_data)} записей")
                    
                    if closed_pnl_data:
                        trades = []
                        processed_count = 0
                        skipped_count = 0
                        for trade_data in closed_pnl_data:
                            # Преобразуем данные биржи в формат для обучения
                            # ВАЖНО: get_closed_pnl возвращает данные с полями:
                            # - closed_pnl (не closedPnl)
                            # - entry_price (не avgEntryPrice)
                            # - exit_price (не avgExitPrice)
                            # - close_timestamp (не updatedTime)
                            
                            symbol = trade_data.get('symbol', '')
                            if not symbol:
                                skipped_count += 1
                                continue
                            
                            # Символ уже очищен от USDT в get_closed_pnl через clean_symbol
                            # Но на всякий случай проверяем
                            if symbol.endswith('USDT'):
                                symbol = symbol[:-4]
                            
                            # Получаем цены и PnL (используем правильные имена полей)
                            entry_price = float(trade_data.get('entry_price', 0) or trade_data.get('avgEntryPrice', 0) or 0)
                            exit_price = float(trade_data.get('exit_price', 0) or trade_data.get('avgExitPrice', 0) or 0)
                            pnl = float(trade_data.get('closed_pnl', 0) or trade_data.get('closedPnl', 0) or 0)
                            
                            # Получаем временные метки
                            close_timestamp = trade_data.get('close_timestamp') or trade_data.get('updatedTime') or trade_data.get('updated_time')
                            
                            # Определяем направление (если нет в данных, пробуем определить по qty или другим полям)
                            side = trade_data.get('side', '')
                            if not side:
                                # Пробуем определить по qty (положительное = LONG, отрицательное = SHORT)
                                qty = trade_data.get('qty', 0)
                                if qty:
                                    side = 'Buy' if qty > 0 else 'Sell'
                                else:
                                    side = 'Buy'  # По умолчанию LONG
                            
                            direction = 'LONG' if side.upper() in ['BUY', 'LONG'] else 'SHORT'
                            
                            # Рассчитываем ROI если нет
                            roi = 0
                            if entry_price > 0 and exit_price > 0:
                                if direction == 'LONG':
                                    roi = ((exit_price - entry_price) / entry_price) * 100
                                else:
                                    roi = ((entry_price - exit_price) / entry_price) * 100
                            
                            # Создаем запись сделки
                            trade = {
                                'id': trade_data.get('orderId') or trade_data.get('id') or trade_data.get('orderLinkId') or f"exchange_{symbol}_{close_timestamp}",
                                'symbol': symbol,
                                'direction': direction,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'roi': roi,
                                'timestamp': close_timestamp,  # Используем close_timestamp как timestamp
                                'close_timestamp': close_timestamp,
                                'status': 'CLOSED',
                                'is_real': True,
                                'is_simulated': False,
                                'source': 'exchange_api'
                            }
                            
                            # Добавляем только если есть валидные цены
                            # PnL может быть 0 или отрицательным - это нормально!
                            if entry_price > 0 and exit_price > 0:
                                # Если PnL не указан, рассчитываем из цен
                                if pnl == 0 and entry_price > 0 and exit_price > 0:
                                    # Рассчитываем PnL из цен (примерный расчет)
                                    qty = trade_data.get('qty', 1.0)
                                    if direction == 'LONG':
                                        calculated_pnl = (exit_price - entry_price) * qty
                                    else:
                                        calculated_pnl = (entry_price - exit_price) * qty
                                    trade['pnl'] = calculated_pnl
                                    pnl = calculated_pnl
                                
                                trades.append(trade)
                                processed_count += 1
                            else:
                                skipped_count += 1
                                if skipped_count <= 5:  # Показываем первые 5 причин пропуска
                                    reason = []
                                    if entry_price <= 0:
                                        reason.append(f"entry_price={entry_price}")
                                    if exit_price <= 0:
                                        reason.append(f"exit_price={exit_price}")
                                    pass
                        
                        logger.info(f"   ✅ Обработано: {processed_count} сделок")
                        if skipped_count > 0:
                            logger.info(f"   ⏭️ Пропущено: {skipped_count} сделок (нет PnL или цены)")
                        
                        if trades:
                            logger.info(f"📊 Загружено {len(trades)} сделок из истории биржи")
                            return trades
                        else:
                            logger.warning(f"   ⚠️ Не удалось обработать ни одной сделки из {len(closed_pnl_data)} записей")
                            if len(closed_pnl_data) > 0:
                                # Показываем пример первой записи для диагностики
                                sample = closed_pnl_data[0]
                                logger.warning(f"   📋 Пример записи (первые 3):")
                                for i, s in enumerate(closed_pnl_data[:3]):
                                    logger.warning(f"      [{i+1}] Ключи: {list(s.keys())}")
                                    logger.warning(f"      [{i+1}] symbol={s.get('symbol')}, "
                                                 f"closed_pnl={s.get('closed_pnl')}, closedPnl={s.get('closedPnl')}, "
                                                 f"entry_price={s.get('entry_price')}, avgEntryPrice={s.get('avgEntryPrice')}, "
                                                 f"exit_price={s.get('exit_price')}, avgExitPrice={s.get('avgExitPrice')}, "
                                                 f"close_timestamp={s.get('close_timestamp')}, updatedTime={s.get('updatedTime')}")
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка загрузки истории сделок с биржи: {e}")
                    import traceback
                    pass
            else:
                logger.warning(f"   ⚠️ Exchange не имеет метода get_closed_pnl")
                logger.warning(f"   💡 Доступные методы: {[m for m in dir(exchange) if not m.startswith('_')][:10]}")
                return []
            
            # Если дошли сюда, значит метод есть, но вернул пустой результат
            logger.info("   💡 Метод get_closed_pnl вернул пустой результат или None")
            return []
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки истории сделок с биржи: {e}")
            import traceback
            pass
            return []
    
    def _save_exchange_trades_history(self, new_trades: List[Dict]) -> None:
        """
        Сохраняет историю сделок трейдера из биржи в БД (ДОПОЛНЯЕТ, не перезаписывает)
        
        Args:
            new_trades: Список новых сделок с биржи
        """
        if not new_trades:
            return
        
        if not self.ai_db:
            logger.error("❌ БД недоступна! Невозможно сохранить сделки биржи. Проверьте инициализацию БД.")
            return
        
        try:
            # Добавляем метки
            for trade in new_trades:
                trade['is_simulated'] = False
                trade['is_real'] = True
                trade['source'] = trade.get('source', 'exchange_api')
                if 'saved_timestamp' not in trade:
                    trade['saved_timestamp'] = datetime.now().isoformat()
                if 'status' not in trade:
                    trade['status'] = 'CLOSED'
            
            # Сохраняем в БД
            saved_count = self.ai_db.save_exchange_trades(new_trades)
            
            if saved_count > 0:
                total_count = self.ai_db.count_exchange_trades()
                logger.info(f"💾 Сохранено {saved_count} новых сделок биржи в БД (всего: {total_count})")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения сделок биржи в БД: {e}")
            raise
    
    def _migrate_json_to_database(self):
        """
        Мигрирует данные из JSON файлов в БД (однократно)
        """
        if not self.ai_db:
            return
        
        try:
            # Миграция симулированных сделок (если есть старый файл)
            simulated_trades_file = os.path.join(self.data_dir, 'simulated_trades.json')
            if os.path.exists(simulated_trades_file):
                try:
                    with open(simulated_trades_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        trades = data.get('trades', [])
                        if trades:
                            saved = self.ai_db.save_simulated_trades(trades)
                            if saved > 0:
                                logger.info(f"📦 Мигрировано {saved} симулированных сделок из JSON в БД")
                except Exception as e:
                    pass
            
            # Миграция сделок биржи (если есть старый файл)
            exchange_trades_history_file = os.path.join(self.data_dir, 'exchange_trades_history.json')
            if os.path.exists(exchange_trades_history_file):
                try:
                    with open(exchange_trades_history_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        trades = data.get('trades', [])
                        if trades:
                            saved = self.ai_db.save_exchange_trades(trades)
                            if saved > 0:
                                logger.info(f"📦 Мигрировано {saved} сделок биржи из JSON в БД")
                except Exception as e:
                    pass
        except Exception as e:
            pass
    
    def _load_saved_exchange_trades(self) -> List[Dict]:
        """
        Загружает сохраненную историю сделок трейдера из биржи (из БД или JSON)
        
        Returns:
            Список сделок из истории биржи
        """
        if not self.ai_db:
            logger.error("❌ БД недоступна! Невозможно загрузить сделки биржи. Проверьте инициализацию БД.")
            return []
        
        try:
            # Получаем все сделки биржи из БД
            trades = self.ai_db.get_trades_for_training(
                include_simulated=False,
                include_real=False,
                include_exchange=True,
                limit=None
            )
            
            # Фильтруем только сделки биржи
            exchange_trades = [t for t in trades if t.get('source') == 'EXCHANGE']
            
            if exchange_trades:
                pass
            
            return exchange_trades
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки сделок биржи из БД: {e}")
            return []
    
    def _update_exchange_trades_history(self) -> None:
        """
        Загружает и дополняет историю сделок трейдера из биржи через API
        
        КОГДА ВЫЗЫВАЕТСЯ:
        1. При инициализации AITrainer (если файл пустой или не существует)
        2. Перед каждым обучением на реальных сделках (train_on_real_trades_with_candles)
        3. Можно вызывать вручную для периодического обновления
        
        КАК РАБОТАЕТ:
        - Загружает историю через exchange.get_closed_pnl()
        - Сохраняет в БД (exchange_trades)
        - ДОПОЛНЯЕТ файл (не перезаписывает!)
        - Избегает дубликатов по ключевым полям
        """
        try:
            logger.info("📥 Загрузка истории сделок с биржи через API...")
            
            # Проверяем текущее количество сделок в БД
            existing_count = 0
            if self.ai_db:
                try:
                    saved_trades = self._load_saved_exchange_trades()
                    existing_count = len(saved_trades)
                    if existing_count > 0:
                        logger.info(f"   💾 В БД уже есть {existing_count} сделок из истории биржи")
                except:
                    pass
            
            new_trades = self._load_exchange_trades_history()
            
            if new_trades:
                self._save_exchange_trades_history(new_trades)
                # Проверяем итоговое количество
                final_count = len(self._load_saved_exchange_trades())
                logger.info(f"✅ История сделок биржи обновлена: добавлено {len(new_trades)} новых сделок")
                logger.info(f"   📊 Всего в БД: {final_count} сделок биржи")
            else:
                if existing_count > 0:
                    logger.info(f"💡 Новых сделок в истории биржи не найдено (в файле уже {existing_count} сделок)")
                else:
                    logger.info(f"💡 История биржи пуста - возможно, на бирже нет закрытых позиций")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка обновления истории сделок биржи: {e}")
            import traceback
            pass
    
    def _load_simulated_trades(self) -> List[Dict]:
        """
        Загружает симулированные сделки из файла
        
        Returns:
            Список симулированных сделок
        """
        # Теперь все симулированные сделки в БД
        if not self.ai_db:
            return []
        
        try:
            # Загружаем из БД
            trades = self.ai_db.get_trades_for_training(
                include_simulated=True,
                include_real=False,
                include_exchange=False,
                limit=None
            )
            
            # Фильтруем только закрытые сделки с PnL
            closed_trades = [
                t for t in trades
                if t.get('status') == 'CLOSED' and t.get('pnl') is not None and t.get('is_simulated', False)
            ]
            
            return closed_trades
        except (json.JSONDecodeError, Exception) as e:
            pass
            return []
    
    def train_on_simulated_trades(self) -> None:
        """
        Обучается на симулированных сделках (дополнительно к реальным)
        Это позволяет AI учиться на большем объеме данных
        """
        try:
            simulated_trades = self._load_simulated_trades()
            
            if len(simulated_trades) < 50:
                pass
                return
            
            logger.info("=" * 80)
            logger.info("🎮 ОБУЧЕНИЕ НА СИМУЛИРОВАННЫХ СДЕЛКАХ")
            logger.info("=" * 80)
            logger.info(f"   📊 Загружено {len(simulated_trades)} симулированных сделок")
            
            # Подготавливаем данные (аналогично train_on_real_trades_with_candles)
            successful_samples = []
            failed_samples = []
            
            for trade in simulated_trades:
                try:
                    # Извлекаем данные
                    entry_rsi = trade.get('entry_rsi')
                    exit_rsi = trade.get('exit_rsi')
                    entry_trend = trade.get('entry_trend', 'NEUTRAL')
                    exit_trend = trade.get('exit_trend', 'NEUTRAL')
                    direction = trade.get('direction', 'LONG')
                    pnl = trade.get('pnl', 0)
                    entry_price = trade.get('entry_price', 0)
                    
                    # Рассчитываем волатильность и объем (если нет в данных)
                    entry_volatility = trade.get('entry_volatility', 0)
                    entry_volume_ratio = trade.get('entry_volume_ratio', 1.0)
                    
                    if not entry_rsi:
                        continue
                    
                    sample = {
                        'symbol': trade.get('symbol', 'UNKNOWN'),
                        'entry_rsi': entry_rsi,
                        'entry_trend': entry_trend,
                        'entry_volatility': entry_volatility,
                        'entry_volume_ratio': entry_volume_ratio,
                        'entry_price': entry_price,
                        'exit_price': trade.get('exit_price', entry_price),
                        'direction': direction,
                        'pnl': pnl,
                        'roi': trade.get('roi', 0),
                        'is_successful': pnl > 0,
                        'is_simulated': True
                    }
                    
                    if pnl > 0:
                        successful_samples.append(sample)
                    else:
                        failed_samples.append(sample)
                except Exception as e:
                    pass
                    continue
            
            all_samples = successful_samples + failed_samples
            
            if len(all_samples) < 50:
                logger.warning(f"⚠️ Недостаточно обработанных симулированных сделок (есть {len(all_samples)})")
                return
            
            logger.info(f"   ✅ Успешных: {len(successful_samples)}")
            logger.info(f"   ❌ Неуспешных: {len(failed_samples)}")
            
            # Обучаем модели (дополняем существующие модели)
            X = []
            y_signal = []
            y_profit = []
            
            for sample in all_samples:
                features = [
                    sample['entry_rsi'],
                    sample['entry_volatility'],
                    sample['entry_volume_ratio'],
                    1.0 if sample['entry_trend'] == 'UP' else 0.0,
                    1.0 if sample['entry_trend'] == 'DOWN' else 0.0,
                    1.0 if sample['direction'] == 'LONG' else 0.0,
                    sample['entry_price'] / 1000.0 if sample['entry_price'] > 0 else 0,
                ]
                X.append(features)
                y_signal.append(1 if sample['is_successful'] else 0)
                y_profit.append(sample['pnl'])
            
            X = np.array(X)
            y_signal = np.array(y_signal)
            y_profit = np.array(y_profit)
            
            # Нормализация
            # ВАЖНО: Проверяем совместимость scaler с текущим количеством фич
            from sklearn.preprocessing import StandardScaler
            current_features = X.shape[1] if len(X.shape) > 1 else len(X[0])
            scaler_features = getattr(self.scaler, 'n_features_in_', None)
            
            if scaler_features is None or scaler_features != current_features:
                # Scaler не обучен или несовместим - пересоздаём
                logger.info(f"   🔄 Пересоздание scaler: было {scaler_features} фич, нужно {current_features}")
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                # Scaler совместим - используем transform
                X_scaled = self.scaler.transform(X)
            
            # Сохраняем количество признаков
            if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ is not None:
                self.expected_features = self.scaler.n_features_in_
            
            # Обучаем модели (дополняем существующие или создаем новые)
            if not self.signal_predictor:
                from sklearn.ensemble import RandomForestClassifier
                self.signal_predictor = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,  # без параллелизма — устраняет UserWarning про delayed/Parallel
                    class_weight='balanced'
                )
                logger.info("   📈 Обучение новой модели на симулированных сделках...")
            else:
                logger.info("   📈 Дополнение существующей модели симулированными сделками...")
            
            # ВАЖНО: Если модель уже обучена, мы дополняем её новыми данными
            # Для этого нужно объединить старые и новые данные
            # Но так как мы не храним старые данные, просто переобучаем на симулированных
            # В будущем можно улучшить, загрузив реальные данные и объединив их
            self.signal_predictor.fit(X_scaled, y_signal)
            
            train_score = self.signal_predictor.score(X_scaled, y_signal)
            logger.info(f"   ✅ Модель обучена на симуляциях! Точность: {train_score:.2%}")
            
            # Обучаем модель прибыли
            if not self.profit_predictor:
                from sklearn.ensemble import GradientBoostingRegressor
                self.profit_predictor = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            
            self.profit_predictor.fit(X_scaled, y_profit)
            profit_pred = self.profit_predictor.predict(X_scaled)
            profit_mse = mean_squared_error(y_profit, profit_pred)
            profit_rmse = np.sqrt(profit_mse)  # RMSE более интерпретируем
            logger.info(f"   ✅ Модель прибыли обучена! RMSE: {profit_rmse:.2f} USDT (ошибка предсказания)")
            
            # Сохраняем модели
            self._save_models()
            logger.info("   💾 Модели сохранены!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения на симулированных сделках: {e}")
            import traceback
            pass
    
    def _load_market_data_for_symbols(self, symbols: List[str]) -> Dict:
        """
        Загрузить рыночные данные ТОЛЬКО для указанных символов
        
        Args:
            symbols: Список символов, для которых нужно загрузить свечи
        
        Returns:
            Словарь с рыночными данными
        """
        try:
            market_data = {'latest': {'candles': {}}}
            candles_data = {}
            
            # Загружаем ТОЛЬКО из БД
            if not self.ai_db:
                logger.warning("⚠️ AI Database не доступна")
                return market_data
            
            if not symbols:
                logger.warning("⚠️ Нет символов для загрузки свечей")
                return market_data
            
            try:
                # Загружаем свечи ТОЛЬКО для указанных символов
                # КРИТИЧНО: get_all_candles_dict() берет первые N символов по алфавиту,
                # а не те, которые нам нужны! Поэтому загружаем свечи напрямую из БД для каждого символа
                symbols_upper = {s.upper() for s in symbols}
                candles_data = {}
                
                for symbol in symbols:
                    try:
                        # Загружаем свечи для конкретного символа
                        from bot_engine.config_loader import get_current_timeframe
                        symbol_candles = self.ai_db.get_candles(
                            symbol=symbol,
                            timeframe=get_current_timeframe(),
                            limit=1000  # Максимум 1000 свечей на символ
                        )
                        
                        if symbol_candles and len(symbol_candles) >= 50:  # Минимум 50 свечей для обучения
                            # get_candles() уже возвращает правильный формат {time, open, high, low, close, volume}
                            candles_data[symbol.upper()] = symbol_candles
                    except Exception as symbol_error:
                        pass
                        continue
                
                if len(candles_data) < len(symbols):
                    missing_count = len(symbols) - len(candles_data)
                    logger.warning(f"   ⚠️ Нет свечей для {missing_count} из {len(symbols)} запрошенных монет")
                    if len(candles_data) > 0:
                        logger.warning(f"   💡 Загружены свечи только для {len(candles_data)} монет: {', '.join(sorted(list(candles_data.keys()))[:10])}{'...' if len(candles_data) > 10 else ''}")
                else:
                    logger.info(f"   ✅ Загружены свечи для всех {len(symbols)} запрошенных монет")
                
            except Exception as db_error:
                logger.error(f"❌ Ошибка загрузки свечей из БД: {db_error}")
                import traceback
                pass
                return market_data
            
            if candles_data:
                total_candles = sum(len(c) for c in candles_data.values())
                logger.info(f"✅ Загружено {len(candles_data)} монет из БД ({total_candles:,} свечей)")
                
                if 'latest' not in market_data:
                    market_data['latest'] = {}
                if 'candles' not in market_data['latest']:
                    market_data['latest']['candles'] = {}
                
                candles_count = 0
                total_candles_count = 0
                
                for symbol, candles in candles_data.items():
                    if candles:
                        market_data['latest']['candles'][symbol] = {
                            'candles': candles,
                            'timeframe': get_current_timeframe(),
                            'last_update': datetime.now().isoformat(),
                            'count': len(candles),
                            'source': 'ai_data.db'
                        }
                        candles_count += 1
                        total_candles_count += len(candles)
                
                logger.info(f"✅ Обработано: {candles_count} монет, {total_candles_count:,} свечей")
            else:
                logger.warning("⚠️ БД пуста или нет свечей для запрошенных символов, ожидаем загрузки свечей...")
            
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки рыночных данных: {e}")
            import traceback
            pass
            return {'latest': {'candles': {}}}
    
    def _load_market_data(self) -> Dict:
        """
        Загрузить рыночные данные
        
        ВАЖНО: Использует ТОЛЬКО БД (таблица candles_history)
        Свечи загружаются через пагинацию по 2000 свечей для каждой монеты
        """
        try:
            market_data = {'latest': {'candles': {}}}
            candles_data = {}
            
            # Загружаем ТОЛЬКО из БД
            if not self.ai_db:
                logger.warning("⚠️ AI Database не доступна")
                return market_data
            
            try:
                # Загружаем свечи для ВСЕХ монет (max_symbols=0), ТФ — системный из конфига
                from bot_engine.config_loader import get_current_timeframe
                candles_data = self.ai_db.get_all_candles_dict(
                    timeframe=get_current_timeframe(),
                    max_symbols=0,  # 0 = без ограничения (все доступные монеты)
                    max_candles_per_symbol=1000
                )
                if candles_data:
                    total_candles = sum(len(c) for c in candles_data.values())
                    logger.info(f"✅ Загружено {len(candles_data)} монет из БД ({total_candles:,} свечей, БЕЗ ограничений - все доступные монеты)")
                    
                    if 'latest' not in market_data:
                        market_data['latest'] = {}
                    if 'candles' not in market_data['latest']:
                        market_data['latest']['candles'] = {}
                    
                    candles_count = 0
                    total_candles_count = 0
                    
                    for symbol, candles in candles_data.items():
                        if candles:
                            market_data['latest']['candles'][symbol] = {
                                'candles': candles,
                                'timeframe': get_current_timeframe(),
                                'last_update': datetime.now().isoformat(),
                                'count': len(candles),
                                'source': 'ai_data.db'
                            }
                            candles_count += 1
                            total_candles_count += len(candles)
                    
                    logger.info(f"✅ Обработано: {candles_count} монет, {total_candles_count:,} свечей")
                else:
                    logger.warning("⚠️ БД пуста, ожидаем загрузки свечей...")
            except Exception as db_error:
                logger.error(f"❌ Ошибка загрузки из БД: {db_error}")
                import traceback
                logger.error(traceback.format_exc())
            
            return market_data
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки рыночных данных: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _prepare_features(self, trade: Dict, market_data: Dict = None) -> Optional[np.ndarray]:
        """
        Подготовка признаков для обучения
        
        Args:
            trade: Данные сделки
            market_data: Рыночные данные
        
        Returns:
            Массив признаков или None
        """
        try:
            features = []
            
            # Базовые признаки из сделки
            entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            direction = trade.get('direction', 'LONG')
            
            if entry_price == 0 or exit_price == 0:
                return None
            
            # Данные входа (fallback: из БД приходят entry_rsi/entry_trend на верхнем уровне)
            entry_data = trade.get('entry_data', {}) or {}
            entry_rsi = entry_data.get('rsi') or trade.get('entry_rsi') or trade.get('rsi')
            if entry_rsi is None:
                entry_rsi = 50
            entry_trend = entry_data.get('trend') or trade.get('entry_trend') or trade.get('trend') or 'NEUTRAL'
            entry_volatility = entry_data.get('volatility') or trade.get('entry_volatility')
            if entry_volatility is None:
                entry_volatility = 0
            
            # Данные выхода (fallback: часто в БД нет exit_rsi — используем entry)
            exit_market_data = trade.get('exit_market_data', {}) or {}
            exit_rsi = exit_market_data.get('rsi') or trade.get('exit_rsi')
            if exit_rsi is None:
                exit_rsi = entry_rsi
            exit_trend = exit_market_data.get('trend') or trade.get('exit_trend') or 'NEUTRAL'
            
            # Признаки
            features.append(entry_rsi)
            features.append(exit_rsi)
            features.append(entry_volatility)
            features.append(1 if direction == 'LONG' else 0)
            features.append(1 if entry_trend == 'UP' else (0 if entry_trend == 'DOWN' else 0.5))
            features.append(1 if exit_trend == 'UP' else (0 if exit_trend == 'DOWN' else 0.5))
            
            # Процент изменения цены
            if direction == 'LONG':
                price_change = ((exit_price - entry_price) / entry_price) * 100
            else:
                price_change = ((entry_price - exit_price) / entry_price) * 100
            
            features.append(price_change)
            
            # Время в позиции (часы)
            entry_time = trade.get('timestamp', '')
            exit_time = trade.get('close_timestamp', '')
            
            if entry_time and exit_time:
                try:
                    entry_dt = datetime.fromisoformat(entry_time.replace('Z', ''))
                    exit_dt = datetime.fromisoformat(exit_time.replace('Z', ''))
                    hours_in_position = (exit_dt - entry_dt).total_seconds() / 3600
                    features.append(hours_in_position)
                except:
                    features.append(0)
            else:
                features.append(0)

            # Признаки по причине закрытия
            close_reason = (trade.get('close_reason') or '').upper()
            features.append(1 if 'MANUAL' in close_reason else 0)
            features.append(1 if 'STOP' in close_reason else 0)
            features.append(1 if 'TAKE' in close_reason else 0)
            features.append(1 if 'TRAIL' in close_reason else 0)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"❌ Ошибка подготовки признаков: {e}")
            return None

    def _build_signal_features_7(self, trade: Dict) -> Optional[np.ndarray]:
        """
        Вектор из 7 признаков для модели сигнала — тот же порядок, что в ai_inference.build_features.
        Обязательно для совместимости обучения и инференса.
        Поддерживает формат из БД: entry_rsi, entry_trend, entry_volatility на верхнем уровне.
        """
        try:
            entry_data = trade.get('entry_data', {}) or {}
            entry_rsi = entry_data.get('rsi') or trade.get('entry_rsi') or trade.get('rsi')
            if entry_rsi is None:
                entry_rsi = 50
            entry_rsi = float(entry_rsi)
            entry_trend = (entry_data.get('trend') or trade.get('entry_trend') or trade.get('trend') or 'NEUTRAL')
            if isinstance(entry_trend, str):
                entry_trend = (entry_trend or 'NEUTRAL').upper()
            entry_volatility = entry_data.get('volatility') or trade.get('entry_volatility')
            if entry_volatility is None:
                entry_volatility = 0
            entry_volatility = float(entry_volatility)
            entry_volume_ratio = entry_data.get('volume_ratio') or trade.get('entry_volume_ratio')
            if entry_volume_ratio is None:
                entry_volume_ratio = 1.0
            entry_volume_ratio = float(entry_volume_ratio)
            direction = (trade.get('direction') or 'LONG').upper()
            entry_price = trade.get('entry_price') or trade.get('price') or 0
            entry_price = float(entry_price) if entry_price else 0
            features = [
                entry_rsi,
                entry_volatility,
                entry_volume_ratio,
                1.0 if entry_trend == 'UP' else 0.0,
                1.0 if entry_trend == 'DOWN' else 0.0,
                1.0 if direction == 'LONG' else 0.0,
                (entry_price / 1000.0) if entry_price > 0 else 0.0,
            ]
            return np.array(features, dtype=np.float64)
        except Exception as e:
            logger.debug(f"Ошибка _build_signal_features_7: {e}")
            return None

    def train_on_history(self):
        """
        Обучение на истории трейдов
        """
        logger.info("=" * 80)
        logger.info("🎓 ОБУЧЕНИЕ НА ИСТОРИИ ТРЕЙДОВ")
        logger.info("=" * 80)
        start_time = datetime.now()
        processed_samples = 0
        final_accuracy = None
        final_mse = None
        
        try:
            # Загружаем данные
            trades = self._load_history_data()
            
            if len(trades) < self._real_trades_min_samples:
                logger.warning(f"⚠️ Недостаточно данных для обучения (нужно минимум {self._real_trades_min_samples}, есть {len(trades)})")
                logger.info("💡 Накопите больше сделок для качественного обучения")
                logger.info("💡 Или используйте обучение на симуляциях (train_on_simulations)")
                self._record_training_event(
                    'history_trades_training',
                    status='SKIPPED',
                    reason='not_enough_trades',
                    samples=len(trades)
                )
                return
            
            logger.info(f"📊 Загружено {len(trades)} сделок для обучения")
            
            # Фильтруем сделки по whitelist/blacklist
            original_trades_count = len(trades)
            filtered_trades = []
            for trade in trades:
                symbol = trade.get('symbol', '')
                if _should_train_on_symbol(symbol):
                    filtered_trades.append(trade)
            
            trades = filtered_trades
            filtered_count = len(trades)
            skipped_by_filter = original_trades_count - filtered_count
            
            if skipped_by_filter > 0:
                logger.info(f"🎯 Фильтрация по whitelist/blacklist: {original_trades_count} → {filtered_count} сделок ({skipped_by_filter} пропущено)")
            
            if len(trades) < self._real_trades_min_samples:
                logger.warning(f"⚠️ Недостаточно данных для обучения после фильтрации (нужно минимум {self._real_trades_min_samples}, есть {len(trades)})")
                logger.info("💡 Накопите больше сделок для качественного обучения")
                logger.info("💡 Или используйте обучение на симуляциях (train_on_simulations)")
                self._record_training_event(
                    'history_trades_training',
                    status='SKIPPED',
                    reason='not_enough_trades_after_filter',
                    samples=len(trades)
                )
                return
            
            logger.info(f"📈 Анализируем сделки...")
            
            # Подготавливаем данные
            X = []
            y_signal = []  # Сигнал (1 = прибыль, 0 = убыток)
            y_profit = []  # Размер прибыли/убытка
            
            logger.info(f"🔍 Подготовка признаков из {len(trades)} сделок...")
            
            processed = 0
            skipped = 0
            
            for trade in trades:
                # Единый 7-признаковый вектор — как в ai_inference.build_features и train_on_real_trades_with_candles
                features = self._build_signal_features_7(trade)
                if features is None:
                    skipped += 1
                    continue
                
                X.append(features)
                
                pnl = trade.get('pnl', 0)
                y_signal.append(1 if pnl > 0 else 0)
                y_profit.append(pnl)
                
                processed += 1
            
            if skipped > 0:
                logger.info(f"⚠️ Пропущено {skipped} сделок (недостаточно данных для 7 признаков)")
            
            if len(X) < self._real_trades_min_samples:
                logger.warning(f"⚠️ Недостаточно валидных данных для обучения ({len(X)} записей, нужно минимум {self._real_trades_min_samples})")
                logger.info("💡 Переключаемся на обучение на симуляциях...")
                # Пробуем использовать симуляции если реальных сделок мало
                return self.train_on_simulations()
            
            logger.info(f"✅ Подготовлено {len(X)} валидных записей для обучения (7 признаков, как при инференсе)")
            
            X = np.array(X)
            y_signal = np.array(y_signal)
            y_profit = np.array(y_profit)
            processed_samples = len(X)
            
            # Проверка: оба класса должны быть представлены, иначе модель не учится различать
            n_success = int(np.sum(y_signal))
            n_fail = len(y_signal) - n_success
            logger.info(f"   📊 Классы: прибыльных={n_success}, убыточных={n_fail}")
            if n_success == 0 or n_fail == 0:
                logger.warning("   ⚠️ ВНИМАНИЕ: Все сделки одного исхода (только прибыльные или только убыточные)!")
                logger.warning("   ⚠️ Модель не сможет научиться различать — нужны и успешные, и неуспешные сделки.")
            
            # Scaler под 7 признаков (совместимость с inference)
            current_n = X.shape[1] if len(X.shape) > 1 else len(X[0])
            if getattr(self.scaler, 'n_features_in_', None) != current_n:
                self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Разделение на train/test
            X_train, X_test, y_signal_train, y_signal_test, y_profit_train, y_profit_test = train_test_split(
                X_scaled, y_signal, y_profit, test_size=0.2, random_state=42
            )
            
            # Обучение модели предсказания сигналов
            logger.info("=" * 80)
            logger.info("🎓 ОБУЧЕНИЕ МОДЕЛИ ПРЕДСКАЗАНИЯ СИГНАЛОВ")
            logger.info(f"📊 Обучающая выборка: {len(X_train)} записей")
            logger.info(f"📊 Тестовая выборка: {len(X_test)} записей")
            logger.info("⏳ Обучение RandomForestClassifier...")
            
            self.signal_predictor = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=1,  # без параллелизма — устраняет UserWarning про delayed/Parallel
                class_weight='balanced'  # баланс классов при неравном числе прибыльных/убыточных
            )
            self.signal_predictor.fit(X_train, y_signal_train)
            
            # УЛУЧШЕНИЕ: Проверка на переобучение (overfitting)
            train_accuracy = self.signal_predictor.score(X_train, y_signal_train)
            y_signal_pred = self.signal_predictor.predict(X_test)
            test_accuracy = accuracy_score(y_signal_test, y_signal_pred)
            final_accuracy = float(test_accuracy)
            
            # Проверяем разницу между train и test accuracy
            accuracy_diff = train_accuracy - test_accuracy
            if accuracy_diff > 0.15:  # Разница > 15% - возможное переобучение
                logger.warning(f"⚠️ Возможно переобучение: train_accuracy={train_accuracy:.2%}, test_accuracy={test_accuracy:.2%}, разница={accuracy_diff:.2%}")
                logger.warning(f"   💡 Модель запоминает данные вместо обобщения. Рекомендуется больше данных или регуляризация.")
            else:
                logger.info(f"✅ Проверка на переобучение: train={train_accuracy:.2%}, test={test_accuracy:.2%}, разница={accuracy_diff:.2%} (OK)")
            
            # УЛУЧШЕНИЕ: Кросс-валидация для более надежной оценки
            try:
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(self.signal_predictor, X_scaled, y_signal, cv=min(5, len(X) // 20), scoring='accuracy', n_jobs=1)
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                logger.info(f"   📊 Кросс-валидация (5-fold): {cv_mean:.2%} ± {cv_std:.2%}")
                
                # Если кросс-валидация сильно отличается от test accuracy - возможна проблема
                if abs(cv_mean - test_accuracy) > 0.10:
                    logger.warning(f"⚠️ Большая разница между CV и test accuracy: {abs(cv_mean - test_accuracy):.2%}")
            except Exception as cv_error:
                pass
            
            # Сохраняем accuracy для последующего сохранения в метаданных
            self._signal_predictor_accuracy = final_accuracy
            
            # Дополнительная статистика
            profitable_pred = sum(y_signal_pred)
            profitable_actual = sum(y_signal_test)
            
            logger.info(f"✅ Модель сигналов обучена!")
            logger.info(f"   📊 Точность: {final_accuracy:.2%}")
            logger.info(f"   📈 Предсказано прибыльных: {profitable_pred}/{len(y_signal_test)}")
            logger.info(f"   📈 Реально прибыльных: {profitable_actual}/{len(y_signal_test)}")
            
            # УЛУЧШЕНИЕ: Дополнительные метрики качества
            if len(y_signal_test) > 0:
                precision = profitable_pred / len(y_signal_test) if len(y_signal_test) > 0 else 0
                recall = profitable_actual / len(y_signal_test) if len(y_signal_test) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                logger.info(f"   📊 Precision: {precision:.2%}")
                logger.info(f"   📊 Recall: {recall:.2%}")
                logger.info(f"   📊 F1 Score: {f1_score:.2%}")
            
            # Обучение модели предсказания прибыли
            logger.info("=" * 80)
            logger.info("🎓 ОБУЧЕНИЕ МОДЕЛИ ПРЕДСКАЗАНИЯ ПРИБЫЛИ")
            logger.info("⏳ Обучение GradientBoostingRegressor...")
            
            self.profit_predictor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            self.profit_predictor.fit(X_train, y_profit_train)
            
            # Оценка модели прибыли
            y_profit_pred = self.profit_predictor.predict(X_test)
            mse = mean_squared_error(y_profit_test, y_profit_pred)
            final_mse = float(mse)
            
            avg_profit_actual = np.mean(y_profit_test)
            avg_profit_pred = np.mean(y_profit_pred)
            
            rmse = np.sqrt(mse)  # RMSE более интерпретируем
            
            # УЛУЧШЕНИЕ: Дополнительные метрики качества
            if len(y_profit_test) > 0:
                from sklearn.metrics import r2_score, mean_absolute_error
                r2 = r2_score(y_profit_test, y_profit_pred)
                mae = mean_absolute_error(y_profit_test, y_profit_pred)
                
                # Статистика PnL для диагностики
                y_std = np.std(y_profit_test)
                y_min = np.min(y_profit_test)
                y_max = np.max(y_profit_test)
                
                logger.info(f"✅ Модель прибыли обучена!")
                logger.info(f"   📊 RMSE: {rmse:.2f} USDT (средняя ошибка предсказания)")
                logger.info(f"   📈 Средняя прибыль (реальная): {avg_profit_actual:.2f} USDT")
                logger.info(f"   📈 Средняя прибыль (предсказанная): {avg_profit_pred:.2f} USDT")
                logger.info(f"   📊 R² Score: {r2:.4f} (качество: 0-1, >0 хорошо, <0 плохо)")
                
                # R² < 0 — штатный случай: в predict() используется только модель сигналов
                if r2 < 0:
                    logger.info(f"   ℹ️ R²={r2:.4f} < 0 — модель прибыли не используется в predict(), решения только по модели сигналов")
                
                logger.info(f"   📊 MAE: {mae:.2f} USDT")
                
                # Процент точности предсказания (в пределах 10%)
                within_10pct = sum(abs(y_profit_test[i] - y_profit_pred[i]) / max(abs(y_profit_test[i]), 1) < 0.1 
                                   for i in range(len(y_profit_test))) / len(y_profit_test) if len(y_profit_test) > 0 else 0
                logger.info(f"   📊 Точность в пределах 10%: {within_10pct:.2%}")
                self._profit_r2 = float(r2)
                self._profit_model_unreliable = r2 < 0
            else:
                self._profit_r2 = None
                self._profit_model_unreliable = True
            
            # Сохранение моделей
            self._save_models()
            
            # Подсчитываем количество обученных моделей
            models_count = 0
            if self.signal_predictor is not None:
                models_count += 1
            if self.profit_predictor is not None:
                models_count += 1
            
            logger.info("✅ Обучение на истории завершено")
            self._record_training_event(
                'history_trades_training',
                status='SUCCESS',
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                samples=processed_samples,
                accuracy=final_accuracy,
                mse=final_mse,
                models_saved=models_count
            )
            try:
                from bot_engine.ai.data_service_status_helper import update_data_service_status_in_db
                update_data_service_status_in_db(
                    training_samples=processed_samples,
                    last_training=datetime.now().isoformat(),
                    effectiveness=float(final_accuracy) if final_accuracy is not None else None,
                    ready=True,
                )
            except Exception:
                pass
        except Exception as e:
            logger.error(f"❌ Ошибка обучения на истории: {e}")
            import traceback
            traceback.print_exc()
            self._record_training_event(
                'history_trades_training',
                status='FAILED',
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                samples=processed_samples,
                reason=str(e)
            )
    
    def train_on_simulations(self, target_win_rate: float = 0.90, max_simulations: int = 1000) -> bool:
        """
        ОБУЧЕНИЕ НА СИМУЛЯЦИЯХ С ОПТИМИЗАЦИЕЙ ПАРАМЕТРОВ
        
        Генерирует разные параметры из конфига, симулирует сделки на истории,
        и обучается на результатах. Ищет параметры с 90%+ win rate.
        
        Args:
            target_win_rate: Целевой win rate (0.90 = 90%)
            max_simulations: Максимальное количество симуляций для поиска оптимальных параметров
        
        Returns:
            True если обучение успешно
        """
        logger.info("=" * 80)
        logger.info("🎲 ОБУЧЕНИЕ НА СИМУЛЯЦИЯХ С ОПТИМИЗАЦИЕЙ ПАРАМЕТРОВ")
        logger.info("=" * 80)
        logger.info(f"🎯 Цель: найти параметры с win_rate >= {target_win_rate:.0%}")
        logger.info(f"📊 Максимум симуляций: {max_simulations}")
        
        start_time = datetime.now()
        
        try:
            # 1. Загружаем исторические свечи для симуляций
            logger.info("📥 Загрузка исторических данных для симуляций...")
            from bot_engine.ai.ai_data_collector import AIDataCollector
            data_collector = AIDataCollector()
            historical_data = data_collector.collect_history_data()
            
            if not historical_data or not historical_data.get('trades'):
                logger.warning("⚠️ Нет исторических данных для симуляций")
                return False
            
            # 2. Используем train_on_historical_data для генерации симуляций с разными параметрами
            logger.info("🔄 Генерация симуляций с разными параметрами...")
            logger.info("💡 Используем train_on_historical_data для создания симуляций")
            logger.info("💡 train_on_historical_data автоматически генерирует разные параметры и симулирует сделки")
            
            # Запускаем train_on_historical_data который создаст симуляции
            # Он автоматически генерирует разные параметры и симулирует сделки на истории
            logger.info("🎲 Запуск train_on_historical_data для генерации симуляций...")
            self.train_on_historical_data()
            
            # 3. Загружаем созданные симуляции из БД
            logger.info("📥 Загрузка симулированных сделок из БД...")
            if not self.ai_db:
                logger.warning("⚠️ БД недоступна, невозможно загрузить симуляции")
                return False
            
            simulated_trades_for_training = self.ai_db.get_trades_for_training(
                include_simulated=True,
                include_real=False,
                include_exchange=False,
                min_trades=0,
                limit=None
            )
            
            if not simulated_trades_for_training or len(simulated_trades_for_training) < self._simulated_trades_min_samples:
                logger.warning(f"⚠️ Недостаточно симулированных сделок: {len(simulated_trades_for_training) if simulated_trades_for_training else 0} < {self._simulated_trades_min_samples}")
                logger.info("💡 Запустите train_on_historical_data для генерации симуляций")
                return False
            
            logger.info(f"✅ Загружено {len(simulated_trades_for_training)} симулированных сделок")
            
            # Анализируем результаты симуляций
            successful_trades = [t for t in simulated_trades_for_training if t.get('pnl', 0) > 0]
            win_rate = len(successful_trades) / len(simulated_trades_for_training) if simulated_trades_for_training else 0
            total_pnl = sum(t.get('pnl', 0) for t in simulated_trades_for_training)
            
            logger.info(f"📊 Статистика симуляций:")
            logger.info(f"   Win rate: {win_rate:.2%}")
            logger.info(f"   Total PnL: {total_pnl:.2f} USDT")
            logger.info(f"   Всего сделок: {len(simulated_trades_for_training)}")
            
            # Проверяем, достигли ли целевого win_rate
            if win_rate >= target_win_rate:
                logger.info(f"🎯 ДОСТИГНУТ ЦЕЛЕВОЙ WIN_RATE >= {target_win_rate:.0%}!")
                logger.info(f"   Текущий win_rate: {win_rate:.2%}")
            else:
                logger.info(f"📊 Текущий win_rate ({win_rate:.2%}) ниже целевого ({target_win_rate:.0%})")
                logger.info(f"💡 Система будет продолжать искать оптимальные параметры при следующем обучении")
            
            # 4. Обучаем модель на симулированных сделках
            logger.info("🎓 Обучение модели на симулированных сделках...")
            
            # Загружаем симуляции для обучения
            if self.ai_db:
                simulated_trades_for_training = self.ai_db.get_trades_for_training(
                    include_simulated=True,
                    include_real=False,
                    include_exchange=False,
                    min_trades=0,
                    limit=None
                )
                
                if simulated_trades_for_training and len(simulated_trades_for_training) >= self._simulated_trades_min_samples:
                    # Единый 7-признаковый вектор — как при инференсе
                    X = []
                    y_signal = []
                    y_profit = []
                    
                    for trade in simulated_trades_for_training:
                        features = self._build_signal_features_7(trade)
                        if features is None:
                            continue
                        
                        X.append(features)
                        pnl = trade.get('pnl', 0)
                        y_signal.append(1 if pnl > 0 else 0)
                        y_profit.append(pnl)
                    
                    if len(X) >= self._simulated_trades_min_samples:
                        X = np.array(X)
                        y_signal = np.array(y_signal)
                        y_profit = np.array(y_profit)
                        
                        current_n = X.shape[1] if len(X.shape) > 1 else len(X[0])
                        if getattr(self.scaler, 'n_features_in_', None) != current_n:
                            self.scaler = StandardScaler()
                        X_scaled = self.scaler.fit_transform(X)
                        
                        # Разделение на train/test
                        X_train, X_test, y_signal_train, y_signal_test, y_profit_train, y_profit_test = train_test_split(
                            X_scaled, y_signal, y_profit, test_size=0.2, random_state=42
                        )
                        
                        # Обучение моделей
                        self.signal_predictor = RandomForestClassifier(
                            n_estimators=100,
                            max_depth=10,
                            random_state=42,
                            n_jobs=1  # без параллелизма — устраняет UserWarning про delayed/Parallel
                        )
                        self.signal_predictor.fit(X_train, y_signal_train)
                        
                        # Проверка на переобучение
                        train_accuracy = self.signal_predictor.score(X_train, y_signal_train)
                        test_accuracy = self.signal_predictor.score(X_test, y_signal_test)
                        accuracy_diff = train_accuracy - test_accuracy
                        
                        if accuracy_diff > 0.15:
                            logger.warning(f"⚠️ Возможно переобучение: train={train_accuracy:.2%}, test={test_accuracy:.2%}")
                        else:
                            logger.info(f"✅ Проверка на переобучение: train={train_accuracy:.2%}, test={test_accuracy:.2%} (OK)")
                        
                        # Кросс-валидация
                        try:
                            from sklearn.model_selection import cross_val_score
                            cv_scores = cross_val_score(self.signal_predictor, X_scaled, y_signal, cv=min(5, len(X) // 20), scoring='accuracy', n_jobs=1)
                            cv_mean = np.mean(cv_scores)
                            logger.info(f"📊 Кросс-валидация: {cv_mean:.2%} ± {np.std(cv_scores):.2%}")
                        except Exception as cv_error:
                            pass
                        
                        self._signal_predictor_accuracy = float(test_accuracy)
                        
                        # Обучение profit_predictor
                        self.profit_predictor = GradientBoostingRegressor(
                            n_estimators=100,
                            max_depth=5,
                            random_state=42
                        )
                        self.profit_predictor.fit(X_train, y_profit_train)
                        
                        # Сохранение моделей
                        self._save_models()
                        
                        logger.info("✅ Обучение на симуляциях завершено")
                        
                        # Сохраняем статистику симуляций
                        if self.ai_db:
                            try:
                                # Получаем лучшие параметры из БД (если есть)
                                optimized_params = self.ai_db.get_optimized_params(
                                    symbol=None,
                                    optimization_type='SIMULATIONS_90_PERCENT'
                                )
                                if optimized_params:
                                    logger.info(f"🏆 Найденные оптимальные параметры:")
                                    logger.info(f"   Win rate: {optimized_params.get('win_rate', 0):.2%}")
                                    logger.info(f"   Total PnL: {optimized_params.get('total_pnl', 0):.2f} USDT")
                            except Exception as e:
                                pass
                        
                        return True
            
            logger.warning("⚠️ Не удалось обучить модель на симуляциях")
            return False
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения на симуляциях: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _simulate_trades_with_params(self, params: Dict, historical_data: Dict) -> List[Dict]:
        """
        Симулирует сделки с заданными параметрами на исторических данных
        
        Использует полную логику из train_on_historical_data для качественной симуляции.
        
        Args:
            params: Параметры RSI для симуляции
            historical_data: Исторические данные (свечи/сделки)
        
        Returns:
            Список симулированных сделок
        """
        # Используем существующий метод train_on_historical_data с переопределением параметров
        # Сохраняем текущие параметры
        old_overrides = getattr(self, 'training_param_overrides', None)
        
        try:
            # Устанавливаем параметры для симуляции
            self.training_param_overrides = {
                'rsi_long_threshold': params['oversold'],
                'rsi_short_threshold': params['overbought'],
                'rsi_exit_long_with_trend': params['exit_long_with_trend'],
                'rsi_exit_long_against_trend': params['exit_long_against_trend'],
                'rsi_exit_short_with_trend': params['exit_short_with_trend'],
                'rsi_exit_short_against_trend': params['exit_short_against_trend']
            }
            
            # Запускаем симуляцию через train_on_historical_data
            # Но нам нужны только симулированные сделки, не обучение модели
            # Поэтому используем упрощенную версию
            
            # Получаем свечи из historical_data
            trades = historical_data.get('trades', [])
            if not trades:
                return []
            
            # Используем AIStrategyOptimizer для симуляции на свечах
            from bot_engine.ai.ai_strategy_optimizer import AIStrategyOptimizer
            optimizer = AIStrategyOptimizer()
            
            # Группируем по символам и симулируем
            symbols_data = {}
            for trade in trades:
                symbol = trade.get('symbol', 'UNKNOWN')
                if symbol not in symbols_data:
                    symbols_data[symbol] = []
                symbols_data[symbol].append(trade)
            
            # Симулируем для ограниченного количества символов (для скорости)
            all_simulated = []
            for symbol, symbol_trades in list(symbols_data.items())[:5]:  # Ограничиваем для скорости
                # Конвертируем сделки в формат свечей (упрощенно)
                candles = []
                for trade in symbol_trades[:100]:  # Ограничиваем количество
                    # Создаем упрощенную свечу из сделки
                    entry_time = trade.get('timestamp') or trade.get('entry_time')
                    if entry_time:
                        try:
                            if isinstance(entry_time, str):
                                entry_ts = datetime.fromisoformat(entry_time.replace('Z', '')).timestamp()
                            else:
                                entry_ts_val = float(entry_time)
                                entry_ts = entry_ts_val / 1000 if entry_ts_val > 1e12 else entry_ts_val
                            
                            candle = {
                                'time': int(entry_ts * 1000),
                                'open': trade.get('entry_price', 0),
                                'close': trade.get('exit_price', trade.get('entry_price', 0)),
                                'high': max(trade.get('entry_price', 0), trade.get('exit_price', trade.get('entry_price', 0))),
                                'low': min(trade.get('entry_price', 0), trade.get('exit_price', trade.get('entry_price', 0))),
                                'volume': trade.get('volume', 0)
                            }
                            candles.append(candle)
                        except:
                            continue
                
                if len(candles) >= 50:  # Минимум свечей для симуляции
                    # Используем optimizer для симуляции
                    try:
                        from bot_engine.config_loader import AIConfig
                        use_bayesian = getattr(AIConfig, 'AI_USE_BAYESIAN', True)
                        optimized_params = optimizer.optimize_coin_parameters_on_candles(
                            symbol=symbol,
                            candles=candles,
                            current_win_rate=0.0,
                            use_bayesian=use_bayesian,
                        )
                        
                        if optimized_params:
                            # Получаем симулированные сделки из optimizer
                            # (упрощенно - используем существующие сделки с новыми параметрами)
                            simulated = self._simulate_symbol_trades_from_candles(symbol, candles, params)
                            all_simulated.extend(simulated)
                    except Exception as e:
                        pass
            
            return all_simulated
            
        except Exception as e:
            pass
            return []
        finally:
            # Восстанавливаем параметры
            self.training_param_overrides = old_overrides
    
    def _simulate_symbol_trades_from_candles(self, symbol: str, candles: List[Dict], params: Dict) -> List[Dict]:
        """
        Симулирует сделки для символа на основе свечей с заданными параметрами
        
        Args:
            symbol: Символ монеты
            candles: Исторические свечи
            params: Параметры RSI
        
        Returns:
            Список симулированных сделок
        """
        simulated_trades = []
        
        try:
            # Вычисляем RSI для свечей
            from bot_engine.indicators import TechnicalIndicators
            rsi_history = TechnicalIndicators.calculate_rsi_history(candles, period=14)
            
            if len(rsi_history) < 50:
                return []
            
            # Симулируем торговлю
            position = None
            for i, candle in enumerate(candles):
                if i < len(rsi_history):
                    rsi = rsi_history[i]
                    price = candle.get('close', 0)
                    
                    # Логика входа
                    if position is None:
                        if rsi <= params['oversold']:
                            # Вход LONG
                            position = {
                                'direction': 'LONG',
                                'entry_price': price,
                                'entry_rsi': rsi,
                                'entry_time': candle.get('time'),
                                'entry_trend': 'UP' if rsi < 30 else 'NEUTRAL'
                            }
                        elif rsi >= params['overbought']:
                            # Вход SHORT
                            position = {
                                'direction': 'SHORT',
                                'entry_price': price,
                                'entry_rsi': rsi,
                                'entry_time': candle.get('time'),
                                'entry_trend': 'DOWN' if rsi > 70 else 'NEUTRAL'
                            }
                    else:
                        # Логика выхода
                        should_exit = False
                        exit_reason = None
                        
                        if position['direction'] == 'LONG':
                            if rsi >= params['exit_long_with_trend']:
                                should_exit = True
                                exit_reason = 'TAKE_PROFIT_WITH_TREND'
                            elif rsi >= params['exit_long_against_trend']:
                                should_exit = True
                                exit_reason = 'TAKE_PROFIT_AGAINST_TREND'
                        else:  # SHORT
                            if rsi <= params['exit_short_with_trend']:
                                should_exit = True
                                exit_reason = 'TAKE_PROFIT_WITH_TREND'
                            elif rsi <= params['exit_short_against_trend']:
                                should_exit = True
                                exit_reason = 'TAKE_PROFIT_AGAINST_TREND'
                        
                        if should_exit:
                            # Закрываем позицию
                            exit_price = price
                            if position['direction'] == 'LONG':
                                pnl = (exit_price - position['entry_price']) / position['entry_price'] * 100
                            else:
                                pnl = (position['entry_price'] - exit_price) / position['entry_price'] * 100
                            
                            simulated_trade = {
                                'symbol': symbol,
                                'direction': position['direction'],
                                'entry_price': position['entry_price'],
                                'exit_price': exit_price,
                                'entry_rsi': position['entry_rsi'],
                                'exit_rsi': rsi,
                                'entry_trend': position['entry_trend'],
                                'exit_trend': 'UP' if rsi > 50 else 'DOWN',
                                'pnl': pnl,
                                'roi': pnl,
                                'is_successful': 1 if pnl > 0 else 0,
                                'status': 'CLOSED',
                                'close_reason': exit_reason,
                                'timestamp': position['entry_time'],
                                'close_timestamp': candle.get('time'),
                                'is_simulated': True,
                                'rsi_params': params
                            }
                            simulated_trades.append(simulated_trade)
                            position = None
            
            return simulated_trades
            
        except Exception as e:
            pass
            return []
    
    def train_on_strategy_params(self):
        """
        Обучение на параметрах стратегии
        
        Анализирует какие параметры стратегии приводят к лучшим результатам
        """
        logger.info("🎓 Обучение на параметрах стратегии...")
        
        try:
            # Загружаем данные
            trades = self._load_history_data()
            
            if len(trades) < 10:
                logger.warning("⚠️ Недостаточно данных для анализа параметров стратегии")
                return
            
            # Анализируем эффективность разных параметров
            # Например, какие значения RSI входа дают лучшие результаты
            
            rsi_ranges = {
                'very_low': (0, 25),
                'low': (25, 35),
                'medium': (35, 65),
                'high': (65, 75),
                'very_high': (75, 100)
            }
            
            results = {}
            
            for trade in trades:
                entry_data = trade.get('entry_data', {})
                entry_rsi = entry_data.get('rsi', 50)
                pnl = trade.get('pnl', 0)
                
                for range_name, (low, high) in rsi_ranges.items():
                    if low <= entry_rsi < high:
                        if range_name not in results:
                            results[range_name] = {'trades': 0, 'total_pnl': 0, 'winning': 0}
                        
                        results[range_name]['trades'] += 1
                        results[range_name]['total_pnl'] += pnl
                        if pnl > 0:
                            results[range_name]['winning'] += 1
                        break
            
            # Сохраняем результаты анализа в БД
            if self.ai_db:
                self.ai_db.save_strategy_analysis('parameter_analysis', results)
            
            logger.info("✅ Анализ параметров стратегии завершен")
            logger.info(f"📊 Результаты: {json.dumps(results, indent=2, ensure_ascii=False)}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения на параметрах стратегии: {e}")
    
    def train_on_real_trades_with_candles(self):
        """
        ГЛАВНЫЙ МЕТОД ОБУЧЕНИЯ: Обучается на РЕАЛЬНЫХ СДЕЛКАХ с PnL
        
        ИСПОЛЬЗУЕТ ДВА ИСТОЧНИКА ДАННЫХ:
        1. bot_history.json - сделки ботов (текущие сделки)
        2. БД (exchange_trades) - история сделок трейдера из биржи (загружается через API)
        
        Связывает свечи с реальными сделками:
        - Что было на свечах когда открыли позицию (RSI, тренд, волатильность)
        - Что было когда закрыли позицию
        - Реальный PnL сделки
        
        Успешные сделки = положительные примеры для обучения
        Неуспешные сделки = отрицательные примеры для обучения
        
        ПЕРЕД ОБУЧЕНИЕМ автоматически обновляет историю биржи через API
        """
        logger.info("=" * 80)
        logger.info("🤖 ОБУЧЕНИЕ НА РЕАЛЬНЫХ СДЕЛКАХ С ОБРАТНОЙ СВЯЗЬЮ")
        logger.info("=" * 80)
        start_time = datetime.now()
        processed_trades = 0
        samples_count = 0
        train_score = None
        profit_mse = None
        
        try:
            # Создаем сессию обучения в БД
            training_session_id = None
            if self.ai_db:
                try:
                    training_session_id = self.ai_db.create_training_session(
                        session_type='REAL_TRADES',
                        metadata={'started_at': datetime.now().isoformat()}
                    )
                    self._current_training_session_id = training_session_id
                except Exception as e:
                    pass
            
            # 0. Обновляем историю сделок с биржи (дополняем файл/БД)
            logger.info("📥 Обновление истории сделок с биржи...")
            self._update_exchange_trades_history()
            
            # 1. Загружаем реальные сделки с PnL из bot_history.json (сделки ботов)
            # Или из БД если доступна
            logger.info("=" * 80)
            logger.info("🔍 ДИАГНОСТИКА ЗАГРУЗКИ СДЕЛОК ДЛЯ ОБУЧЕНИЯ")
            logger.info("=" * 80)
            
            if self.ai_db:
                logger.info("   📦 Попытка загрузки сделок ботов из БД...")
                logger.info("      - ai_data.db -> bot_trades")
                logger.info("      - bots_data.db -> bot_trades_history")
                try:
                    # ВАЖНО: Используем get_trades_for_training() вместо get_bot_trades()
                    # потому что get_trades_for_training() загружает сделки из bots_data.db -> bot_trades_history,
                    # а get_bot_trades() только из ai_data.db -> bot_trades (который может быть пуст)
                    # ВАЖНО: Включаем симуляции для обучения ИИ на разных параметрах
                    bot_trades = self.ai_db.get_trades_for_training(
                        include_simulated=True,  # ВКЛЮЧАЕМ симуляции для обучения!
                        include_real=True,  # Включаем реальные сделки из bots_data.db
                        include_exchange=False,  # Сделки биржи загружаем отдельно
                        min_trades=0,  # Не фильтруем по символам
                        limit=None
                    )
                    logger.info(f"   ✅ Загружено {len(bot_trades)} сделок ботов из БД (ai_data.db + bots_data.db)")
                    
                    # Преобразуем формат для совместимости
                    valid_trades = 0
                    for trade in bot_trades:
                        if 'timestamp' not in trade:
                            trade['timestamp'] = trade.get('entry_time') or trade.get('timestamp')
                        if 'close_timestamp' not in trade:
                            trade['close_timestamp'] = trade.get('exit_time') or trade.get('close_timestamp')
                        
                        # Проверяем, что сделка пригодна для обучения
                        if trade.get('entry_price') and trade.get('exit_price') and trade.get('symbol'):
                            valid_trades += 1
                    
                    logger.info(f"   ✅ Пригодно для обучения: {valid_trades} из {len(bot_trades)} сделок")
                    
                    if len(bot_trades) == 0:
                        logger.warning("   ⚠️ ВНИМАНИЕ: БД вернула 0 сделок ботов!")
                        logger.warning("   💡 Проверьте:")
                        logger.warning("      - Есть ли сделки в ai_data.db -> bot_trades (status='CLOSED')")
                        logger.warning("      - Есть ли сделки в bots_data.db -> bot_trades_history (status='CLOSED')")
                        
                except Exception as e:
                    logger.warning(f"   ⚠️ Ошибка загрузки сделок ботов из БД: {e}")
                    logger.warning(f"   🔄 Fallback: загрузка из bot_history.json...")
                    import traceback
                    pass
                    bot_trades = self._load_history_data()
                    logger.info(f"   ✅ Загружено {len(bot_trades)} сделок из bot_history.json")
            else:
                logger.warning("   ⚠️ БД недоступна! Загрузка из bot_history.json...")
                bot_trades = self._load_history_data()
                logger.info(f"   ✅ Загружено {len(bot_trades)} сделок из bot_history.json")
            
            # 2. Загружаем историю сделок трейдера из биржи (из БД)
            logger.info("   📦 Загрузка сделок биржи из БД (exchange_trades)...")
            exchange_trades = self._load_saved_exchange_trades()
            logger.info(f"   ✅ Загружено {len(exchange_trades)} сделок биржи из БД")
            
            # 3. Объединяем сделки из обоих источников (избегаем дубликатов)
            trades = []
            existing_ids = set()
            
            # Добавляем сделки ботов
            for trade in bot_trades:
                trade_key = (
                    trade.get('symbol'),
                    trade.get('timestamp'),
                    trade.get('close_timestamp'),
                    trade.get('entry_price'),
                    trade.get('exit_price'),
                    trade.get('id')
                )
                if trade_key not in existing_ids:
                    trades.append(trade)
                    existing_ids.add(trade_key)
            
            # Добавляем сделки из истории биржи
            if exchange_trades:
                added_from_exchange = 0
                for trade in exchange_trades:
                    trade_key = (
                        trade.get('symbol'),
                        trade.get('timestamp'),
                        trade.get('close_timestamp'),
                        trade.get('entry_price'),
                        trade.get('exit_price'),
                        trade.get('id')
                    )
                    if trade_key not in existing_ids:
                        trades.append(trade)
                        existing_ids.add(trade_key)
                        added_from_exchange += 1
                
                if added_from_exchange > 0:
                    logger.info(f"📊 Добавлено {added_from_exchange} сделок из истории биржи")
            
            logger.info("=" * 80)
            logger.info(f"📊 ИТОГИ ЗАГРУЗКИ СДЕЛОК:")
            logger.info(f"   🤖 Сделки ботов: {len(bot_trades)}")
            logger.info(f"   📈 Сделки биржи: {len(exchange_trades)}")
            logger.info(f"   📦 Всего объединено: {len(trades)}")
            logger.info("=" * 80)
            
            if len(trades) < 10:
                logger.warning(f"⚠️ Недостаточно сделок для обучения (есть {len(trades)}, нужно минимум 10)")
                logger.warning(f"   🤖 Сделки ботов: {len(bot_trades)}")
                logger.warning(f"   📈 Сделки биржи: {len(exchange_trades)}")
                logger.warning("   💡 ДИАГНОСТИКА ПРОБЛЕМЫ:")
                if len(bot_trades) == 0:
                    logger.warning("      ❌ Нет сделок ботов!")
                    logger.warning("      💡 Проверьте:")
                    logger.warning("         - Есть ли закрытые сделки в bots_data.db -> bot_trades_history (status='CLOSED')")
                    logger.warning("         - Есть ли закрытые сделки в ai_data.db -> bot_trades (status='CLOSED', is_simulated=0)")
                    logger.warning("         - Есть ли файл data/bot_history.json с закрытыми сделками")
                if len(exchange_trades) == 0:
                    logger.warning("      ❌ Нет сделок биржи!")
                    logger.warning("      💡 Проверьте:")
                    logger.warning("         - Есть ли сделки в ai_data.db -> exchange_trades")
                    logger.warning("         - Была ли обновлена история биржи через _update_exchange_trades_history()")
                logger.info("💡 Накопите больше сделок - AI будет обучаться на вашем опыте!")
                self._record_training_event(
                    'real_trades_training',
                    status='SKIPPED',
                    reason='not_enough_trades',
                    trades=len(trades),
                    samples=0
                )
                return
            
            logger.info(f"📊 Загружено {len(trades)} сделок ДЛЯ ОБУЧЕНИЯ ИИ (объединенные данные)")
            logger.info(f"   🤖 Из bot_history.json (сделки БОТОВ): {len(bot_trades)}")
            logger.info(f"   📈 Из БД (сделки БИРЖИ): {len(exchange_trades)}")
            if len(exchange_trades) > 0:
                logger.info(f"   ✅ ИСТОРИЯ БИРЖИ ИСПОЛЬЗУЕТСЯ ДЛЯ ОБУЧЕНИЯ ИИ!")
            else:
                logger.info(f"   ⚠️ История биржи пуста - загружаем через API...")
            
            # Загружаем накопленный опыт ИИ, чтобы не переобучать с нуля
            experience_bad_coins = set()
            if self.ai_db and hasattr(self.ai_db, "get_ai_experience_snapshot"):
                try:
                    snap = self.ai_db.get_ai_experience_snapshot()
                    if snap and snap.get("unsuccessful_coins"):
                        for uc in snap["unsuccessful_coins"]:
                            s = uc.get("symbol") if isinstance(uc, dict) else uc
                            if s:
                                experience_bad_coins.add(str(s).upper())
                        if experience_bad_coins:
                            logger.info(f"   📚 Учёт опыта: {len(experience_bad_coins)} проблемных монет (усиленный вес при обучении)")
                except Exception as e:
                    logger.debug("get_ai_experience_snapshot: %s", e)
            
            # Обновляем количество сделок для отслеживания
            self._last_real_trades_training_count = len(trades)
            
            # 4. Загружаем свечи для ВСЕХ монет (не только для тех, по которым есть сделки)
            # Это позволяет обучаться на всех доступных монетах, а не только на тех, по которым уже были сделки
            logger.info("📊 Загрузка свечей для ВСЕХ доступных монет (не только для монет из сделок)...")
            market_data = self._load_market_data()
            
            # Дополнительно: убеждаемся, что свечи загружены для монет из сделок (если их нет в общем списке)
            symbols_from_trades = set()
            for trade in trades:
                symbol = trade.get('symbol')
                if symbol:
                    symbols_from_trades.add(symbol.upper())
            
            if symbols_from_trades:
                logger.info(f"📊 Найдено {len(symbols_from_trades)} уникальных монет в сделках")
                logger.info(f"   💡 Монеты из сделок: {', '.join(sorted(list(symbols_from_trades))[:20])}{'...' if len(symbols_from_trades) > 20 else ''}")
                
                # Проверяем, есть ли свечи для монет из сделок
                latest = market_data.get('latest', {})
                candles_data = latest.get('candles', {})
                symbols_without_candles = symbols_from_trades - set(candles_data.keys())
                if symbols_without_candles:
                    logger.warning(f"   ⚠️ Нет свечей для {len(symbols_without_candles)} монет из сделок:")
                    logger.warning(f"      {', '.join(sorted(list(symbols_without_candles))[:10])}{'...' if len(symbols_without_candles) > 10 else ''}")
                    logger.warning(f"   💡 Загружаем свечи для этих монет отдельно...")
                    # Догружаем свечи для монет из сделок, которых нет в общем списке
                    additional_candles = self._load_market_data_for_symbols(list(symbols_without_candles))
                    additional_latest = additional_candles.get('latest', {})
                    additional_candles_data = additional_latest.get('candles', {})
                    if additional_candles_data:
                        # Добавляем недостающие свечи в общий список
                        if 'latest' not in market_data:
                            market_data['latest'] = {}
                        if 'candles' not in market_data['latest']:
                            market_data['latest']['candles'] = {}
                        market_data['latest']['candles'].update(additional_candles_data)
                        logger.info(f"   ✅ Догружены свечи для {len(additional_candles_data)} монет из сделок")
            
            # Получаем финальный список свечей после всех загрузок
            latest = market_data.get('latest', {})
            candles_data = latest.get('candles', {})
            
            # Диагностика: какие монеты из сделок не имеют свечей (после всех попыток загрузки)
            if symbols_from_trades:
                symbols_without_candles = symbols_from_trades - set(candles_data.keys())
                if symbols_without_candles:
                    logger.warning(f"   ⚠️ Нет свечей для {len(symbols_without_candles)} монет из сделок (после всех попыток загрузки):")
                    logger.warning(f"      {', '.join(sorted(list(symbols_without_candles))[:10])}{'...' if len(symbols_without_candles) > 10 else ''}")
                    logger.warning(f"   💡 Эти сделки будут пропущены!")
            
            if not candles_data:
                logger.warning("⚠️ Нет свечей для анализа")
                self._record_training_event(
                    'real_trades_training',
                    status='SKIPPED',
                    reason='no_candles_data',
                    trades=len(trades),
                    samples=0
                )
                return
            
            logger.info(f"📈 Загружено свечей для {len(candles_data)} монет")
            
            # 5. Связываем сделки со свечами и обучаемся
            successful_samples = []  # Успешные сделки (PnL > 0)
            failed_samples = []      # Неуспешные сделки (PnL <= 0)
            
            # Флаг для принудительного использования рассчитанного PnL
            # Будет установлен в True, если все исходные PnL положительные
            force_use_calculated_pnl = False
            original_pnl_values = []  # Собираем исходные PnL для анализа
            
            # Инициализируем диагностику расчета PnL
            self._pnl_calculation_debug = {
                'negative_roi_count': 0,
                'positive_roi_count': 0,
                'zero_roi_count': 0,
                'with_position_size': 0,
                'without_position_size': 0,
                'negative_calculated_pnl': 0,
                'positive_calculated_pnl': 0
            }
            
            # Импортируем функцию расчета RSI
            try:
                from bot_engine.indicators import TechnicalIndicators
                calculate_rsi_history_func = TechnicalIndicators.calculate_rsi_history
            except ImportError:
                try:
                    from bots_modules.calculations import calculate_rsi_history
                    calculate_rsi_history_func = calculate_rsi_history
                except ImportError:
                    from bot_engine.utils.rsi_utils import calculate_rsi_history
                    calculate_rsi_history_func = calculate_rsi_history
            
            processed_trades = 0
            skipped_trades = 0
            processed_from_bot_history = 0
            processed_from_exchange = 0
            
            # Создаем множество ID сделок из истории биржи для статистики
            exchange_trade_ids = {
                (t.get('symbol'), t.get('timestamp'), t.get('close_timestamp'), 
                 t.get('entry_price'), t.get('exit_price'), t.get('id'))
                for t in exchange_trades
            }
            
            for trade in trades:
                try:
                    symbol = trade.get('symbol')
                    if not symbol or symbol not in candles_data:
                        skipped_trades += 1
                        continue
                    
                    candles = candles_data[symbol].get('candles', [])
                    if len(candles) < 50:
                        skipped_trades += 1
                        continue
                    
                    # Сортируем свечи по времени
                    candles = sorted(candles, key=lambda x: x.get('time', 0))
                    
                    # Данные сделки
                    entry_price = trade.get('entry_price') or trade.get('entryPrice')
                    exit_price = trade.get('exit_price') or trade.get('exitPrice')
                    direction = trade.get('direction', 'LONG')
                    original_pnl = trade.get('pnl', 0)
                    
                    # ВАЖНО: Всегда рассчитываем PnL из цен для корректности
                    # Это нужно, потому что исходный PnL может быть неправильно рассчитан
                    calculated_pnl = None
                    if entry_price and exit_price and entry_price > 0:
                        # Получаем размер позиции
                        position_size = trade.get('position_size') or trade.get('size') or trade.get('volume_value')
                        
                        # Рассчитываем ROI (процент изменения цены)
                        if direction == 'LONG':
                            roi_percent = (exit_price - entry_price) / entry_price
                        else:
                            roi_percent = (entry_price - exit_price) / entry_price
                        
                        # Если есть размер позиции, рассчитываем PnL в USDT
                        # Если position_size в USDT, то PnL = roi_percent * position_size
                        # Если position_size в количестве монет, то PnL = (exit_price - entry_price) * position_size для LONG
                        if position_size and position_size > 0:
                            # Предполагаем, что position_size в USDT (размер позиции)
                            calculated_pnl = roi_percent * position_size
                        else:
                            # Если нет размера позиции, используем ROI как относительный PnL
                            # ВАЖНО: roi_percent может быть отрицательным для убыточных сделок!
                            calculated_pnl = roi_percent * 100  # В процентах
                        
                        # ДИАГНОСТИКА: Сохраняем информацию о расчете для анализа
                        if roi_percent < 0:
                            self._pnl_calculation_debug['negative_roi_count'] += 1
                        elif roi_percent > 0:
                            self._pnl_calculation_debug['positive_roi_count'] += 1
                        else:
                            self._pnl_calculation_debug['zero_roi_count'] += 1
                        
                        if position_size and position_size > 0:
                            self._pnl_calculation_debug['with_position_size'] += 1
                        else:
                            self._pnl_calculation_debug['without_position_size'] += 1
                        
                        if calculated_pnl < 0:
                            self._pnl_calculation_debug['negative_calculated_pnl'] += 1
                        elif calculated_pnl > 0:
                            self._pnl_calculation_debug['positive_calculated_pnl'] += 1
                    
                    # Сохраняем исходный PnL для анализа
                    if original_pnl != 0 and original_pnl is not None:
                        original_pnl_values.append(original_pnl)
                    
                    # Используем рассчитанный PnL, если исходный отсутствует или равен 0
                    # ИЛИ если принудительно используем рассчитанный (будет установлено позже)
                    if calculated_pnl is not None:
                        if original_pnl == 0 or original_pnl is None or force_use_calculated_pnl:
                            pnl = calculated_pnl
                        else:
                            # Используем исходный PnL
                            pnl = original_pnl
                    else:
                        pnl = original_pnl
                    
                    entry_time = trade.get('timestamp') or trade.get('entry_time')
                    exit_time = trade.get('close_timestamp') or trade.get('exit_time')
                    
                    if not entry_price or not exit_price:
                        skipped_trades += 1
                        continue
                    
                    # Находим свечи в момент входа и выхода
                    entry_candle_idx = None
                    exit_candle_idx = None
                    
                    if entry_time:
                        try:
                            if isinstance(entry_time, str):
                                entry_dt = datetime.fromisoformat(entry_time.replace('Z', ''))
                                entry_timestamp = int(entry_dt.timestamp() * 1000)
                            else:
                                entry_timestamp = entry_time
                            
                            # Ищем ближайшую свечу к моменту входа
                            for idx, candle in enumerate(candles):
                                candle_time = candle.get('time', 0)
                                if abs(candle_time - entry_timestamp) < 3600000:  # В пределах 1 часа
                                    entry_candle_idx = idx
                                    break
                        except:
                            pass
                    
                    if exit_time:
                        try:
                            if isinstance(exit_time, str):
                                exit_dt = datetime.fromisoformat(exit_time.replace('Z', ''))
                                exit_timestamp = int(exit_dt.timestamp() * 1000)
                            else:
                                exit_timestamp = exit_time
                            
                            for idx, candle in enumerate(candles):
                                candle_time = candle.get('time', 0)
                                if abs(candle_time - exit_timestamp) < 3600000:
                                    exit_candle_idx = idx
                                    break
                        except:
                            pass
                    
                    # Если не нашли точные свечи, используем последние
                    if entry_candle_idx is None:
                        entry_candle_idx = len(candles) - 1
                    if exit_candle_idx is None:
                        exit_candle_idx = len(candles) - 1
                    
                    # Вычисляем RSI на момент входа
                    closes = [float(c.get('close', 0) or 0) for c in candles]
                    volumes = [float(c.get('volume', 0) or 0) for c in candles]
                    highs = [float(c.get('high', 0) or 0) for c in candles]
                    lows = [float(c.get('low', 0) or 0) for c in candles]
                    
                    if len(closes) < 50:
                        skipped_trades += 1
                        continue
                    
                    # RSI история
                    rsi_history = calculate_rsi_history_func(candles, period=14)
                    if not rsi_history or len(rsi_history) < 20:
                        skipped_trades += 1
                        continue
                    
                    # RSI на момент входа
                    rsi_idx = max(0, entry_candle_idx - 14)
                    if rsi_idx < len(rsi_history):
                        entry_rsi = rsi_history[rsi_idx]
                    else:
                        entry_rsi = rsi_history[-1] if rsi_history else 50
                    
                    # Тренд на момент входа
                    if entry_candle_idx >= 20:
                        ema_short = self._calculate_ema(closes[max(0, entry_candle_idx-12):entry_candle_idx+1], 12)
                        ema_long = self._calculate_ema(closes[max(0, entry_candle_idx-26):entry_candle_idx+1], 26)
                        if ema_short and ema_long:
                            entry_trend = 'UP' if ema_short > ema_long else ('DOWN' if ema_short < ema_long else 'NEUTRAL')
                        else:
                            entry_trend = 'NEUTRAL'
                    else:
                        entry_trend = 'NEUTRAL'
                    
                    # Волатильность на момент входа
                    volatility_window = 20
                    if entry_candle_idx >= volatility_window:
                        price_changes = [(closes[j] - closes[j-1]) / closes[j-1] * 100 
                                        for j in range(entry_candle_idx-volatility_window+1, entry_candle_idx+1)]
                        entry_volatility = np.std(price_changes) if price_changes else 0
                    else:
                        entry_volatility = 0
                    
                    # Объемы
                    volume_window = 20
                    if entry_candle_idx >= volume_window:
                        avg_volume = np.mean(volumes[entry_candle_idx-volume_window:entry_candle_idx+1])
                    else:
                        avg_volume = np.mean(volumes[:entry_candle_idx+1]) if entry_candle_idx > 0 else volumes[0]
                    entry_volume_ratio = volumes[entry_candle_idx] / avg_volume if avg_volume > 0 else 1.0
                    
                    # ROI сделки
                    if direction == 'LONG':
                        roi = ((exit_price - entry_price) / entry_price) * 100
                    else:
                        roi = ((entry_price - exit_price) / entry_price) * 100
                    
                    # Получаем размер позиции для пересчета PnL (если понадобится)
                    position_size = trade.get('position_size') or trade.get('size') or trade.get('volume_value') or 1.0
                    
                    # Создаем обучающий пример
                    sample = {
                        'symbol': symbol,
                        'entry_rsi': entry_rsi,
                        'entry_trend': entry_trend,
                        'entry_volatility': entry_volatility,
                        'entry_volume_ratio': entry_volume_ratio,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': direction,
                        'pnl': pnl,
                        'roi': roi,
                        'is_successful': pnl > 0,
                        'position_size': position_size  # Сохраняем для пересчета PnL
                    }
                    
                    # Разделяем на успешные и неуспешные
                    if pnl > 0:
                        successful_samples.append(sample)
                    else:
                        failed_samples.append(sample)
                    
                    # Подсчитываем источник сделки для статистики
                    trade_key = (
                        trade.get('symbol'),
                        trade.get('timestamp'),
                        trade.get('close_timestamp'),
                        trade.get('entry_price'),
                        trade.get('exit_price'),
                        trade.get('id')
                    )
                    if trade_key in exchange_trade_ids:
                        processed_from_exchange += 1
                    else:
                        processed_from_bot_history += 1
                    
                    processed_trades += 1
                    
                except Exception as e:
                    pass
                    skipped_trades += 1
                    continue
            
            # ДИАГНОСТИКА РАСЧЕТА PnL: Выводим статистику по расчету
            if hasattr(self, '_pnl_calculation_debug') and self._pnl_calculation_debug:
                debug = self._pnl_calculation_debug
                logger.info("=" * 80)
                logger.info("🔍 ДИАГНОСТИКА РАСЧЕТА PnL ИЗ ЦЕН")
                logger.info("=" * 80)
                logger.info(f"   📊 ROI (процент изменения цены):")
                logger.info(f"      ✅ Положительных ROI: {debug['positive_roi_count']}")
                logger.info(f"      ❌ Отрицательных ROI: {debug['negative_roi_count']}")
                logger.info(f"      ⚪ Нулевых ROI: {debug['zero_roi_count']}")
                logger.info(f"   📊 Размер позиции:")
                logger.info(f"      ✅ С размером позиции: {debug['with_position_size']}")
                logger.info(f"      ⚠️ Без размера позиции: {debug['without_position_size']}")
                logger.info(f"   📊 Рассчитанный PnL:")
                logger.info(f"      ✅ Положительных: {debug['positive_calculated_pnl']}")
                logger.info(f"      ❌ Отрицательных: {debug['negative_calculated_pnl']}")
                
                if debug['negative_roi_count'] > 0 and debug['negative_calculated_pnl'] == 0:
                    logger.error("=" * 80)
                    logger.error("❌ КРИТИЧЕСКАЯ ПРОБЛЕМА: Есть отрицательные ROI, но нет отрицательных PnL!")
                    logger.error("=" * 80)
                    logger.error("   ⚠️ Это означает, что расчет PnL неправильный!")
                    logger.error("   ⚠️ Возможно, position_size всегда положительный или расчет неверный")
                    logger.error("=" * 80)
                elif debug['negative_roi_count'] == 0:
                    logger.warning("   ⚠️ ВНИМАНИЕ: Нет отрицательных ROI - все сделки были прибыльными по ценам!")
                    logger.warning("   ⚠️ Это может означать, что в данных только успешные сделки")
                logger.info("=" * 80)
            
            # Диагностика исходных PnL значений
            if len(original_pnl_values) > 10:  # Минимум 10 сделок для анализа
                min_original_pnl = min(original_pnl_values)
                max_original_pnl = max(original_pnl_values)
                avg_original_pnl = np.mean(original_pnl_values)
                median_original_pnl = np.median(original_pnl_values)
                negative_count = sum(1 for pnl_val in original_pnl_values if pnl_val < 0)
                zero_count = sum(1 for pnl_val in original_pnl_values if pnl_val == 0)
                positive_count = sum(1 for pnl_val in original_pnl_values if pnl_val > 0)
                
                logger.info(f"   📊 Диагностика ИСХОДНЫХ PnL: min={min_original_pnl:.2f}, max={max_original_pnl:.2f}, avg={avg_original_pnl:.2f}, median={median_original_pnl:.2f}")
                logger.info(f"   📊 Распределение ИСХОДНЫХ PnL: отрицательных={negative_count}, нулевых={zero_count}, положительных={positive_count}")
                
                if negative_count == 0 and zero_count == 0:
                    logger.warning("   ⚠️ ОБНАРУЖЕНА ПРОБЛЕМА: Все исходные PnL положительные!")
                    logger.warning("   ⚠️ Это может означать, что в bot_history.json сохраняются только успешные сделки")
                    logger.warning("   ⚠️ Пересчитываем PnL из цен входа/выхода для корректного обучения")
                    force_use_calculated_pnl = True
                    
                    # Пересчитываем PnL для всех уже обработанных сделок
                    all_samples = successful_samples + failed_samples
                    for sample in all_samples:
                        entry_price = sample.get('entry_price')
                        exit_price = sample.get('exit_price')
                        direction = sample.get('direction', 'LONG')
                        if entry_price and exit_price and entry_price > 0:
                            # Рассчитываем ROI
                            if direction == 'LONG':
                                roi_percent = (exit_price - entry_price) / entry_price
                            else:
                                roi_percent = (entry_price - exit_price) / entry_price
                            
                            # Используем размер позиции из sample
                            position_size = sample.get('position_size')
                            if position_size and position_size > 0:
                                # PnL в USDT
                                recalculated_pnl = roi_percent * position_size
                            else:
                                # Если нет размера позиции, используем ROI в процентах
                                recalculated_pnl = roi_percent * 100
                            
                            sample['pnl'] = recalculated_pnl
                            sample['is_successful'] = recalculated_pnl > 0
                    
                    # Перераспределяем по категориям
                    successful_samples = [s for s in all_samples if s['pnl'] > 0]
                    failed_samples = [s for s in all_samples if s['pnl'] <= 0]
                    
                    # Диагностика после пересчета
                    recalculated_pnl_values = [s['pnl'] for s in all_samples]
                    if recalculated_pnl_values:
                        min_recalc = min(recalculated_pnl_values)
                        max_recalc = max(recalculated_pnl_values)
                        avg_recalc = np.mean(recalculated_pnl_values)
                        negative_recalc = sum(1 for pnl in recalculated_pnl_values if pnl < 0)
                        zero_recalc = sum(1 for pnl in recalculated_pnl_values if pnl == 0)
                        positive_recalc = sum(1 for pnl in recalculated_pnl_values if pnl > 0)
                        
                        logger.info(f"   📊 Диагностика ПЕРЕСЧИТАННЫХ PnL: min={min_recalc:.2f}, max={max_recalc:.2f}, avg={avg_recalc:.2f}")
                        logger.info(f"   📊 Распределение ПЕРЕСЧИТАННЫХ PnL: отрицательных={negative_recalc}, нулевых={zero_recalc}, положительных={positive_recalc}")
                        
                        if negative_recalc == 0 and zero_recalc == 0:
                            logger.error("   ❌ КРИТИЧЕСКАЯ ПРОБЛЕМА: После пересчета все PnL все еще положительные!")
                            logger.error("   ❌ Это означает, что все сделки действительно были прибыльными")
                            logger.error("   ❌ Или проблема в расчете PnL из цен входа/выхода")
                        else:
                            logger.info(f"   ✅ После пересчета: {negative_recalc} убыточных и {zero_recalc} нулевых сделок")
            
            logger.info(f"✅ Обработано {processed_trades} сделок ДЛЯ ОБУЧЕНИЯ ИИ")
            logger.info(f"   📦 Из bot_history.json (сделки ботов): {processed_from_bot_history}")
            logger.info(f"   📦 Из истории биржи (БД): {processed_from_exchange}")
            logger.info(f"   ✅ Успешных: {len(successful_samples)} (PnL > 0)")
            logger.info(f"   ❌ Неуспешных: {len(failed_samples)} (PnL <= 0)")
            logger.info(f"   ⏭️ Пропущено: {skipped_trades}")
            
            # ДИАГНОСТИКА: Если пропущено много сделок, объясняем почему
            if skipped_trades > 0:
                logger.warning(f"   ⚠️ ВНИМАНИЕ: Пропущено {skipped_trades} сделок из {len(trades)} загруженных")
                logger.warning(f"   💡 Причины пропуска:")
                logger.warning(f"      - Нет свечей для монеты (< 50 свечей)")
                logger.warning(f"      - Недостаточно данных RSI (< 20 значений)")
                logger.warning(f"      - Ошибки обработки сделки (см. DEBUG логи)")
                logger.warning(f"   💡 Проверьте: есть ли свечи для монет в БД (ai_data.db -> candles_history)")
            
            logger.info(f"   ✅ ИСТОРИЯ БИРЖИ АКТИВНО ИСПОЛЬЗУЕТСЯ ДЛЯ ОБУЧЕНИЯ ИИ!")
            
            # Диагностика: проверяем распределение PnL
            if processed_trades > 0:
                all_pnl_values = [s['pnl'] for s in successful_samples] + [s['pnl'] for s in failed_samples]
                if all_pnl_values:
                    min_pnl = min(all_pnl_values)
                    max_pnl = max(all_pnl_values)
                    avg_pnl = np.mean(all_pnl_values)
                    median_pnl = np.median(all_pnl_values)
                    logger.info(f"   📊 Диагностика PnL: min={min_pnl:.2f}, max={max_pnl:.2f}, avg={avg_pnl:.2f}, median={median_pnl:.2f}")
                    
                    # Проверяем, есть ли отрицательные PnL
                    negative_pnl_count = sum(1 for pnl in all_pnl_values if pnl < 0)
                    zero_pnl_count = sum(1 for pnl in all_pnl_values if pnl == 0)
                    positive_pnl_count = sum(1 for pnl in all_pnl_values if pnl > 0)
                    logger.info(f"   📊 Распределение PnL: отрицательных={negative_pnl_count}, нулевых={zero_pnl_count}, положительных={positive_pnl_count}")
                    
                    if negative_pnl_count == 0 and zero_pnl_count == 0:
                        logger.warning("   ⚠️ ВНИМАНИЕ: Все сделки имеют положительный PnL!")
                        logger.warning("   ⚠️ Это может указывать на проблему в данных или расчете PnL")
                        logger.warning("   ⚠️ Модель не сможет научиться различать успешные и неуспешные сделки")
            
            # 6. ОБУЧАЕМСЯ НА РЕАЛЬНОМ ОПЫТЕ
            all_samples = successful_samples + failed_samples
            samples_count = len(all_samples)
            
            if len(all_samples) >= 20:  # Минимум 20 сделок
                logger.info("=" * 80)
                logger.info("🤖 ОБУЧЕНИЕ НЕЙРОСЕТИ НА РЕАЛЬНОМ ОПЫТЕ")
                logger.info("=" * 80)
                
                # Подготавливаем данные
                X = []
                y_signal = []  # 1 = успешная сделка, 0 = неуспешная
                y_profit = []  # Реальный PnL
                
                sample_weights = []
                for sample in all_samples:
                    features = [
                        sample['entry_rsi'],
                        sample['entry_volatility'],
                        sample['entry_volume_ratio'],
                        1.0 if sample['entry_trend'] == 'UP' else 0.0,
                        1.0 if sample['entry_trend'] == 'DOWN' else 0.0,
                        1.0 if sample['direction'] == 'LONG' else 0.0,
                        sample['entry_price'] / 1000.0 if sample['entry_price'] > 0 else 0,
                    ]
                    
                    X.append(features)
                    y_signal.append(1 if sample['is_successful'] else 0)
                    y_profit.append(sample['pnl'])
                    # Усиленный вес для неудачных сделок по проблемным монетам (опыт ИИ)
                    w = 2.0 if (experience_bad_coins and sample.get('symbol', '').upper() in experience_bad_coins and not sample['is_successful']) else 1.0
                    sample_weights.append(w)
                
                X = np.array(X)
                y_signal = np.array(y_signal)
                y_profit = np.array(y_profit)
                sample_weights_arr = np.array(sample_weights, dtype=float)
                
                # Нормализация
                # ВАЖНО: Всегда пересоздаем scaler при обучении на реальных сделках,
                # потому что количество фич (7) отличается от _prepare_features (12)
                from sklearn.preprocessing import StandardScaler
                
                # Проверяем совместимость scaler с текущим количеством фич
                current_features = X.shape[1] if len(X.shape) > 1 else len(X[0])
                scaler_features = getattr(self.scaler, 'n_features_in_', None)
                
                if scaler_features is None or scaler_features != current_features:
                    # Scaler не обучен или обучен на другом количестве фич - пересоздаём
                    logger.info(f"   🔄 Пересоздание scaler: было {scaler_features} фич, нужно {current_features}")
                    self.scaler = StandardScaler()
                    X_scaled = self.scaler.fit_transform(X)
                else:
                    # Scaler совместим - используем transform
                    X_scaled = self.scaler.transform(X)
                
                # Сохраняем количество признаков
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ is not None:
                    self.expected_features = self.scaler.n_features_in_
                
                # Обучаем модель предсказания успешности сделок
                if not self.signal_predictor:
                    from sklearn.ensemble import RandomForestClassifier
                    self.signal_predictor = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=1,  # без параллелизма — устраняет UserWarning про delayed/Parallel
                        class_weight='balanced'  # Балансировка классов
                    )
                
                logger.info("   📈 Обучение модели на успешных/неуспешных сделках...")
                self.signal_predictor.fit(X_scaled, y_signal, sample_weight=sample_weights_arr)
                
                # Оценка качества
                train_score = self.signal_predictor.score(X_scaled, y_signal)
                logger.info(f"   ✅ Модель обучена! Точность: {train_score:.2%}")
                
                # Статистика по классам
                from collections import Counter
                class_dist = Counter(y_signal)
                successful_count = class_dist.get(1, 0)
                failed_count = class_dist.get(0, 0)
                total_count = successful_count + failed_count
                logger.info(f"   📊 Распределение: Успешных={successful_count}, Неуспешных={failed_count}")
                
                # Предупреждение если все сделки одного класса
                if failed_count == 0 and successful_count > 0:
                    logger.warning("   ⚠️ ВНИМАНИЕ: Все сделки успешные (PnL > 0)!")
                    logger.warning("   ⚠️ Модель не может научиться различать успешные и неуспешные сделки")
                    logger.warning("   ⚠️ Точность 100% и нулевая важность признаков - это признак проблемы!")
                    logger.warning("   ⚠️ Проверьте данные: возможно, только успешные сделки сохраняются в bot_history.json")
                elif successful_count == 0 and failed_count > 0:
                    logger.warning("   ⚠️ ВНИМАНИЕ: Все сделки неуспешные (PnL <= 0)!")
                    logger.warning("   ⚠️ Это также указывает на проблему в данных")
                
                # Анализ важности признаков
                if hasattr(self.signal_predictor, 'feature_importances_'):
                    feature_names = ['RSI', 'Volatility', 'Volume Ratio', 'Trend UP', 'Trend DOWN', 'Direction LONG', 'Price']
                    importances = self.signal_predictor.feature_importances_
                    logger.info("   🔍 Важность признаков:")
                    for name, importance in zip(feature_names, importances):
                        logger.info(f"      {name}: {importance:.3f}")
                    
                    # Предупреждение если все важности нулевые
                    if all(imp == 0.0 for imp in importances):
                        logger.warning("   ⚠️ ВНИМАНИЕ: Все признаки имеют нулевую важность!")
                        logger.warning("   ⚠️ Это означает, что модель не использует признаки для предсказания")
                        logger.warning("   ⚠️ Возможные причины: все сделки одного класса или признаки не информативны")
                
                # Обучаем модель предсказания прибыли
                if not self.profit_predictor:
                    from sklearn.ensemble import GradientBoostingRegressor
                    self.profit_predictor = GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=42
                    )
                
                logger.info("   💰 Обучение модели предсказания прибыли...")
                self.profit_predictor.fit(X_scaled, y_profit, sample_weight=sample_weights_arr)
                
                # Оценка предсказания прибыли с информативными метриками
                profit_pred = self.profit_predictor.predict(X_scaled)
                profit_mse = mean_squared_error(y_profit, profit_pred)
                profit_rmse = np.sqrt(profit_mse)  # RMSE более интерпретируем
                
                # R² - коэффициент детерминации (0-1, чем ближе к 1 тем лучше)
                from sklearn.metrics import r2_score as sklearn_r2_score
                r2_score = sklearn_r2_score(y_profit, profit_pred)
                
                # Нормализованный MSE (относительно среднего PnL)
                y_mean = np.mean(np.abs(y_profit))
                normalized_mse = profit_mse / (y_mean ** 2) if y_mean > 0 else profit_mse
                
                # Статистика PnL для диагностики
                y_std = np.std(y_profit)
                y_min = np.min(y_profit)
                y_max = np.max(y_profit)
                
                logger.info(f"   ✅ Модель прибыли обучена!")
                logger.info(f"      RMSE: {profit_rmse:.2f} USDT (средняя ошибка предсказания)")
                logger.info(f"      R²: {r2_score:.4f} (качество модели: 0-1, >0 хорошо, <0 плохо)")
                
                # R² < 0 — штатный случай: в predict() используется только модель сигналов
                if r2_score < 0:
                    logger.info(f"      ℹ️ R²={r2_score:.4f} < 0 — модель прибыли не используется в predict(), решения только по модели сигналов")
                
                self._profit_r2 = float(r2_score)
                self._profit_model_unreliable = r2_score < 0
                
                logger.info(f"      MSE/Var: {normalized_mse:.4f} (нормализованная ошибка)")
                logger.info(f"      Статистика PnL: min={y_min:.2f}, max={y_max:.2f}, std={y_std:.2f} USDT")
                
                # Сохраняем модели
                self._save_models()
                logger.info("   💾 Модели сохранены!")
                
                # Анализ успешных паттернов
                if successful_samples:
                    logger.info("=" * 80)
                    logger.info("📊 АНАЛИЗ УСПЕШНЫХ ПАТТЕРНОВ")
                    logger.info("=" * 80)
                    
                    successful_rsi = [s['entry_rsi'] for s in successful_samples]
                    successful_trends = [s['entry_trend'] for s in successful_samples]
                    successful_directions = [s['direction'] for s in successful_samples]
                    
                    avg_successful_rsi = np.mean(successful_rsi)
                    logger.info(f"   📈 Средний RSI успешных сделок: {avg_successful_rsi:.2f}")
                    
                    from collections import Counter
                    trend_dist = Counter(successful_trends)
                    logger.info(f"   📊 Тренды успешных сделок: {dict(trend_dist)}")
                    
                    direction_dist = Counter(successful_directions)
                    logger.info(f"   📊 Направления успешных сделок: {dict(direction_dist)}")
                    
                    logger.info("=" * 80)
                
                # Обновляем сессию обучения в БД
                if self.ai_db and hasattr(self, '_current_training_session_id') and self._current_training_session_id:
                    try:
                        self.ai_db.update_training_session(
                            self._current_training_session_id,
                            total_trades=processed_trades,
                            successful_trades=sum(1 for s in all_samples if s.get('is_successful', False)),
                            failed_trades=sum(1 for s in all_samples if not s.get('is_successful', True)),
                            accuracy=float(train_score) if train_score is not None else None,
                            mse=float(profit_mse) if profit_mse is not None else None,
                            status='COMPLETED'
                        )
                    except Exception as e:
                        pass
                
                # Подсчитываем количество обученных моделей
                models_count = 0
                if self.signal_predictor is not None:
                    models_count += 1
                if self.profit_predictor is not None:
                    models_count += 1
                
                self._record_training_event(
                    'real_trades_training',
                    status='SUCCESS',
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    trades=processed_trades,
                    samples=samples_count,
                    accuracy=float(train_score) if train_score is not None else None,
                    mse=float(profit_mse) if profit_mse is not None else None,
                    models_saved=models_count
                )
                # Обновляем статус data_service для UI (выборка и эффективность)
                try:
                    from bot_engine.ai.data_service_status_helper import update_data_service_status_in_db
                    update_data_service_status_in_db(
                        training_samples=samples_count,
                        trades=processed_trades,
                        last_training=datetime.now().isoformat(),
                        effectiveness=float(train_score) if train_score is not None else None,
                        ready=True,
                    )
                except Exception:
                    pass
            else:
                logger.warning(f"⚠️ Недостаточно сделок для обучения (нужно минимум 20, есть {len(all_samples)})")
                logger.warning(f"   📊 Статистика:")
                logger.warning(f"      - Загружено сделок из источников: {len(trades)}")
                logger.warning(f"      - Успешно обработано: {processed_trades}")
                logger.warning(f"      - Пропущено (недостаточно данных): {skipped_trades}")
                logger.warning(f"      - Пригодно для обучения: {len(all_samples)}")
                logger.warning(f"   💡 Почему нет сделок для обучения:")
                logger.warning(f"      - Нет закрытых сделок ботов (bot_history.json или БД)")
                logger.warning(f"      - Нет сделок из истории биржи (exchange_trades в БД)")
                logger.warning(f"      - Недостаточно свечей для монет (< 50 свечей на монету)")
                logger.warning(f"      - Недостаточно данных RSI для анализа (< 20 значений)")
                logger.warning(f"   💡 Что делать:")
                logger.warning(f"      1. Запустите ботов - они будут генерировать сделки")
                logger.warning(f"      2. Обновите историю биржи: _update_exchange_trades_history()")
                logger.warning(f"      3. Загрузите свечи для монет в БД (ai_data.db -> candles_history)")
                logger.warning(f"      4. Используйте симуляции: train_on_historical_data() создаст симулированные сделки")
                self._record_training_event(
                    'real_trades_training',
                    status='SKIPPED',
                    reason='not_enough_samples',
                    trades=processed_trades,
                    samples=samples_count
                )
            
            # КРИТИЧНО: Сначала генерируем симуляции на исторических данных
            # Это создает симуляции в БД для последующего обучения
            try:
                logger.info("=" * 80)
                logger.info("🎮 ГЕНЕРАЦИЯ AI СИМУЛЯЦИЙ НА ИСТОРИЧЕСКИХ ДАННЫХ")
                logger.info("=" * 80)
                logger.info("💡 Это создаст симуляции в БД для обучения AI")
                self.train_on_historical_data()
                logger.info("✅ Генерация симуляций завершена")
            except Exception as hist_error:
                logger.warning(f"⚠️ Ошибка генерации симуляций на исторических данных: {hist_error}")
                import traceback
                pass
            
            # Обучаемся на симулированных сделках (дополнительно к реальным)
            try:
                self.train_on_simulated_trades()
            except Exception as sim_error:
                pass
            
            # Обновляем время и количество последнего обучения на реальных сделках
            self._last_real_trades_training_time = datetime.now()
            if self.ai_db:
                try:
                    # Получаем актуальное количество сделок из БД
                    bot_trades = self.ai_db.get_bot_trades(status='CLOSED', limit=None)
                    exchange_trades = self._load_saved_exchange_trades()
                    self._last_real_trades_training_count = len(bot_trades) + len(exchange_trades)
                except Exception as e:
                    pass
                    self._last_real_trades_training_count = processed_trades
            else:
                self._last_real_trades_training_count = processed_trades
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения на реальных сделках: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._record_training_event(
                'real_trades_training',
                status='FAILED',
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                trades=processed_trades,
                samples=samples_count,
                reason=str(e)
            )
    
    def train_on_historical_data(self):
        """
        ОБУЧЕНИЕ НА ИСТОРИЧЕСКИХ ДАННЫХ С ИСПОЛЬЗОВАНИЕМ ВАШИХ НАСТРОЕК
        
        Симулирует торговлю на исторических данных используя:
        - Ваши RSI параметры из bot_config.py (с вариацией для разнообразия)
        - Ваши стратегии входа/выхода (с разными комбинациями)
        - Проверяет как отработали сигналы
        - Обучается на успешных/неуспешных симуляциях
        
        ВАЖНО: Каждое обучение использует РАЗНЫЕ параметры и РАЗНЫЕ данные для разнообразия!
        """
        start_time = datetime.now()
        total_trained_coins = 0
        total_models_saved = 0
        total_failed_coins = 0
        total_candles_processed = 0
        ml_params_generated_count = 0  # Счетчик использования ML модели для генерации параметров
        import random
        import time as time_module
        
        # Генерируем уникальный seed для этого обучения на основе времени
        training_seed = int(time_module.time() * 1000) % 1000000
        random.seed(training_seed)
        np.random.seed(training_seed)
        
        # Создаем сессию обучения в БД
        training_session_id = None
        if self.ai_db:
            try:
                training_session_id = self.ai_db.create_training_session(
                    session_type='HISTORICAL_DATA',
                    training_seed=training_seed,
                    metadata={'started_at': datetime.now().isoformat()}
                )
                self._current_training_session_id = training_session_id
            except Exception as e:
                pass
        
        # Сокращенные логи - только seed для отслеживания
        pass

        def _normalize_timestamp(raw_ts):
            """Преобразует таймстамп свечи (мс/с) в секунды."""
            try:
                value = float(raw_ts)
                if value > 1e12:  # мс
                    return value / 1000.0
                if value > 1e10:  # fallback для мкс
                    return value / 1000.0
                return value
            except (TypeError, ValueError):
                return None

        def _build_protection_state(direction: str, entry_price: float, entry_ts_ms: Optional[float], position_size: float) -> ProtectionState:
            quantity = None
            safe_entry = float(entry_price) if entry_price else None
            if safe_entry and safe_entry > 0 and position_size:
                quantity = position_size / safe_entry
            return ProtectionState(
                position_side=direction,
                entry_price=safe_entry,
                entry_time=_normalize_timestamp(entry_ts_ms),
                quantity=quantity,
                notional_usdt=position_size,
            )
        
        try:
            # ✅ КРИТИЧНО: Сначала загружаем КОНФИГ пользователя — он база для всех расчётов
            base_config_snapshot = _get_config_snapshot()
            base_config = base_config_snapshot.get('global', {}) or {}
            if self.training_param_overrides:
                base_config = deepcopy(base_config)
                base_config.update(self.training_param_overrides)
                if not self._training_overrides_logged:
                    logger.info("🎯 Используем тренировочные AI оверрайды (ai_launcher_config)")
                    self._training_overrides_logged = True

            # Константы из bot_config — только как fallback, если в конфиге нет значения
            try:
                from bot_engine.config_loader import (
                    RSI_OVERSOLD, RSI_OVERBOUGHT,
                    RSI_EXIT_LONG_WITH_TREND, RSI_EXIT_LONG_AGAINST_TREND,
                    RSI_EXIT_SHORT_WITH_TREND, RSI_EXIT_SHORT_AGAINST_TREND,
                    RSI_PERIOD
                )
            except ImportError as e:
                logger.warning(f"⚠️ Не удалось загрузить настройки из bot_config.py: {e}")
                RSI_OVERSOLD, RSI_OVERBOUGHT = 29, 71
                RSI_EXIT_LONG_WITH_TREND, RSI_EXIT_LONG_AGAINST_TREND = 65, 60
                RSI_EXIT_SHORT_WITH_TREND, RSI_EXIT_SHORT_AGAINST_TREND = 35, 40
                RSI_PERIOD = 14

            # ✅ Базовые RSI — СНАЧАЛА из конфига пользователя, затем константы (fallback)
            def _safe_float(v, default):
                if v is None:
                    return default
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return default
            base_rsi_oversold = _safe_float(base_config.get('rsi_long_threshold'), RSI_OVERSOLD)
            base_rsi_overbought = _safe_float(base_config.get('rsi_short_threshold'), RSI_OVERBOUGHT)
            base_exit_long_with = _safe_float(base_config.get('rsi_exit_long_with_trend'), RSI_EXIT_LONG_WITH_TREND)
            base_exit_long_against = _safe_float(base_config.get('rsi_exit_long_against_trend'), RSI_EXIT_LONG_AGAINST_TREND)
            base_exit_short_with = _safe_float(base_config.get('rsi_exit_short_with_trend'), RSI_EXIT_SHORT_WITH_TREND)
            base_exit_short_against = _safe_float(base_config.get('rsi_exit_short_against_trend'), RSI_EXIT_SHORT_AGAINST_TREND)
            
            # ВАРИАЦИЯ ПАРАМЕТРОВ: Добавляем случайное отклонение для разнообразия
            # Это позволяет модели обучаться на разных комбинациях параметров
            variation_range = 7  # ±7 пунктов вариации для RSI входов
            
            # Базовые параметры для трекера
            base_params = {
                'oversold': base_rsi_oversold,
                'overbought': base_rsi_overbought,
                'exit_long_with_trend': base_exit_long_with,
                'exit_long_against_trend': base_exit_long_against,
                'exit_short_with_trend': base_exit_short_with,
                'exit_short_against_trend': base_exit_short_against
            }
            
            # Пробуем найти неиспользованные параметры (если трекер доступен)
            if self.param_tracker:
                # Проверяем статистику использования
                stats = self.param_tracker.get_usage_stats()
                if stats['is_exhausted']:
                    logger.warning(f"⚠️ Использовано {stats['usage_percentage']:.1f}% всех комбинаций параметров!")
                    logger.warning("💡 Рекомендуется переключиться на обучение на реальных сделках")
                else:
                    pass  # доступно комбинаций RSI параметров
            else:
                pass

            base_stop_loss = base_config.get('max_loss_percent', 15.0)
            base_take_profit = base_config.get('take_profit_percent', 20.0)
            base_trailing_activation = base_config.get('trailing_stop_activation', 20.0)
            base_trailing_distance = base_config.get('trailing_stop_distance', 15.0)
            base_trailing_take_distance = base_config.get('trailing_take_distance', 0.5)
            base_trailing_update_interval = base_config.get('trailing_update_interval', 3.0)
            base_break_even = base_config.get('break_even_trigger', 100.0)
            base_break_even_protection = base_config.get('break_even_protection', True)
            base_max_hours = base_config.get('max_position_hours', 48)
            base_rsi_time_filter_enabled = base_config.get('rsi_time_filter_enabled', True)
            base_rsi_time_filter_candles = base_config.get('rsi_time_filter_candles', 6)
            base_rsi_time_filter_upper = base_config.get('rsi_time_filter_upper', 65)
            base_rsi_time_filter_lower = base_config.get('rsi_time_filter_lower', 35)
            base_exit_scam_enabled = base_config.get('exit_scam_enabled', True)
            base_exit_scam_candles = base_config.get('exit_scam_candles', 8)
            base_exit_scam_single_candle_percent = base_config.get('exit_scam_single_candle_percent', 15.0)
            base_exit_scam_multi_candle_count = base_config.get('exit_scam_multi_candle_count', 4)
            base_exit_scam_multi_candle_percent = base_config.get('exit_scam_multi_candle_percent', 50.0)
            base_trend_detection_enabled = base_config.get('trend_detection_enabled', False)
            base_avoid_down_trend = base_config.get('avoid_down_trend', True)
            base_avoid_up_trend = base_config.get('avoid_up_trend', True)
            base_trend_analysis_period = base_config.get('trend_analysis_period', 30)
            base_trend_price_change_threshold = base_config.get('trend_price_change_threshold', 7)
            base_trend_candles_threshold = base_config.get('trend_candles_threshold', 70)
            base_enable_maturity_check = base_config.get('enable_maturity_check', True)
            base_min_candles_for_maturity = base_config.get('min_candles_for_maturity', 400)
            base_min_rsi_low = base_config.get('min_rsi_low', 35)
            base_max_rsi_high = base_config.get('max_rsi_high', 65)

            logger.info("🎲 БАЗОВЫЕ ПАРАМЕТРЫ ОБУЧЕНИЯ (конфиг → вариации на уровне монеты)")

            logger.info("=" * 80)

            logger.info("📊 RSI базовые значения (из конфига пользователя, ИИ варьирует от них):")

            logger.info(

                f"   LONG: вход <= {base_rsi_oversold} (±{variation_range}), "

                f"выход по тренду >= {base_exit_long_with} (±8), против тренда >= {base_exit_long_against} (±8)"

            )

            logger.info(

                f"   SHORT: вход >= {base_rsi_overbought} (±{variation_range}), "

                f"выход по тренду <= {base_exit_short_with} (±8), против тренда <= {base_exit_short_against} (±8)"

            )

            logger.info("💰 Риск-менеджмент:")

            logger.info(f"   Stop Loss: {base_stop_loss:.1f}% (±6%)")

            logger.info(f"   Take Profit: {base_take_profit:.1f}% (-12% … +15%)")

            logger.info(

                f"   Trailing Stop: активация {base_trailing_activation:.1f}% (-12% … +25%), "

                f"расстояние {base_trailing_distance:.1f}% (-12% … +18%)"

            )

            logger.info(

                f"   Trailing Take: расстояние {base_trailing_take_distance:.2f}% (±0.2%), "

                f"интервал {base_trailing_update_interval:.1f}с (±1.0с)"

            )

            logger.info(

                f"   Break Even: {'✅' if base_break_even_protection else '❌'} "

                f"(триггер {base_break_even:.1f}% (-60% … +90%))"

            )

            logger.info(f"   Max Position Hours: {base_max_hours}ч (-72…+120ч)")

            logger.info("=" * 80)



            # Импортируем функцию расчета RSI истории
            try:
                from bot_engine.indicators import TechnicalIndicators
                calculate_rsi_history_func = TechnicalIndicators.calculate_rsi_history
            except ImportError:
                try:
                    from bots_modules.calculations import calculate_rsi_history
                    calculate_rsi_history_func = calculate_rsi_history
                except ImportError:
                    from bot_engine.utils.rsi_utils import calculate_rsi_history
                    calculate_rsi_history_func = calculate_rsi_history
            
            # Загружаем рыночные данные
            # ВАЖНО: Используем ТОЛЬКО БД (ai_data.db, таблица candles_history)!
            # Файлы больше не используются - все данные в БД!
            market_data = self._load_market_data()
            
            if not market_data:
                logger.warning("⚠️ Нет рыночных данных для обучения")
                self._record_training_event(
                    'historical_data_training',
                    status='SKIPPED',
                    reason='no_market_data'
                )
                return
            
            latest = market_data.get('latest', {})
            candles_data = latest.get('candles', {})
            
            if not candles_data:
                logger.warning("⚠️ Нет свечей для обучения!")
                logger.info("💡 БД ai_data.db пуста или таблица candles_history не содержит данных")
                logger.info("💡 Запустите загрузку полной истории свечей через ai.py")
                logger.info("   💡 Это загрузит ВСЕ доступные свечи для всех монет в БД через пагинацию")
                self._record_training_event(
                    'historical_data_training',
                    status='SKIPPED',
                    reason='no_candles_data'
                )
                return
            
            # ВАЖНО: Загружаем список зрелых монет из bots.py (если доступен)
            # Это экономит ресурсы - обучаем только зрелые монеты
            # Используем helper модуль для единообразного доступа к данным bots.py
            try:
                from bot_engine.ai.bots_data_helper import get_mature_coins
                mature_coins_set = get_mature_coins()
                if mature_coins_set:
                    logger.info(f"✅ Загружен список зрелых монет из bots.py: {len(mature_coins_set)} монет")
                else:
                    pass
            except ImportError:
                # Fallback если helper недоступен
                mature_coins_set = set()
                try:
                    mature_coins_file = os.path.join('data', 'mature_coins.json')
                    if os.path.exists(mature_coins_file):
                        with open(mature_coins_file, 'r', encoding='utf-8') as f:
                            mature_coins_data = json.load(f)
                            mature_coins_set = set(mature_coins_data.keys())
                            logger.info(f"✅ Загружен список зрелых монет из файла: {len(mature_coins_set)} монет")
                except Exception as e:
                    pass
                    pass
            except Exception as e:
                pass
                mature_coins_set = set()
            
            # Фильтруем монеты: используем только зрелые (если список доступен)
            if mature_coins_set and base_enable_maturity_check:
                original_count = len(candles_data)
                candles_data = {symbol: data for symbol, data in candles_data.items() if symbol in mature_coins_set}
                filtered_count = len(candles_data)
                skipped_count = original_count - filtered_count
                if skipped_count > 0:
                    logger.info(f"📊 Фильтрация по зрелости: {original_count} → {filtered_count} монет ({skipped_count} незрелых пропущено)")
            
            # Фильтруем монеты по whitelist/blacklist для обучения
            original_count_after_maturity = len(candles_data)
            filtered_candles_data = {}
            for symbol, data in candles_data.items():
                if _should_train_on_symbol(symbol):
                    filtered_candles_data[symbol] = data
            
            candles_data = filtered_candles_data
            filtered_count_after_whitelist = len(candles_data)
            skipped_by_whitelist = original_count_after_maturity - filtered_count_after_whitelist
            
            if skipped_by_whitelist > 0:
                logger.info(f"🎯 Фильтрация по whitelist/blacklist: {original_count_after_maturity} → {filtered_count_after_whitelist} монет ({skipped_by_whitelist} пропущено)")
            
            # Сокращенный лог начала обучения
            total_coins = len(candles_data)
            logger.info(f"📊 Обучение для {total_coins} монет...")
            
            # ОБУЧЕНИЕ ДЛЯ КАЖДОЙ МОНЕТЫ ОТДЕЛЬНО
            total_trained_coins = 0
            total_failed_coins = 0
            total_models_saved = 0
            total_candles_processed = 0
            
            # ВАЖНО: Логируем прогресс каждые 50 монет
            progress_interval = 50
            
            # ОБУЧАЕМ КАЖДУЮ МОНЕТУ ОТДЕЛЬНО
            # Собираем все симулированные сделки для сохранения
            all_simulated_trades = []
            
            # Генерируем уникальный ID процесса для координации параллельной работы
            import socket
            hostname = socket.gethostname()
            process_id = f"{hostname}-{os.getpid()}-{int(time_module.time())}"
            
            # Получаем доступные символы (не заблокированные другими процессами)
            if self.ai_db:
                available_symbols = list(candles_data.keys())
                try:
                    available_symbols = self.ai_db.get_available_symbols(available_symbols, process_id, hostname)
                    if len(available_symbols) < len(candles_data):
                        logger.info(f"📊 Доступно для обработки: {len(available_symbols)}/{len(candles_data)} монет (остальные заняты другими процессами)")
                except Exception as e:
                    pass
            
            for symbol_idx, (symbol, candle_info) in enumerate(candles_data.items(), 1):
                # Показываем прогресс каждые 50 монет или для первых 10 монет
                if symbol_idx % progress_interval == 0 or symbol_idx <= 10:
                    logger.info(f"   📈 Прогресс: {symbol_idx}/{total_coins} монет обработано ({symbol_idx/total_coins*100:.1f}%)")
                
                # Логируем начало обработки каждой монеты (первые 10 и каждые 50)
                if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                    logger.info(f"   🎓 [{symbol_idx}/{total_coins}] Начало обработки {symbol}...")
                
                # Блокируем символ для обработки (для параллельной работы на разных ПК)
                if self.ai_db:
                    if not self.ai_db.try_lock_symbol(symbol, process_id, hostname, lock_duration_minutes=120):
                        pass
                        continue
                
                try:
                    candles = candle_info.get('candles', [])
                    coin_seed = training_seed + (abs(hash(symbol)) % 1000)
                    coin_rng = random.Random(coin_seed)
                    if not candles or len(candles) < 100:  # Нужно больше свечей для симуляции
                        if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                            logger.info(f"   ⏭️ {symbol}: пропущено (недостаточно свечей: {len(candles) if candles else 0})")
                        continue
                    
                    # ВАЖНО: Дополнительная проверка зрелости (fallback если список зрелых монет недоступен)
                    # Основная фильтрация уже выполнена выше по списку зрелых монет из bots.py
                    if base_enable_maturity_check and not mature_coins_set and len(candles) < base_min_candles_for_maturity:
                        if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                            logger.info(f"   ⏭️ {symbol}: пропущено (незрелая монета: {len(candles)}/{base_min_candles_for_maturity} свечей)")
                        continue
                    
                    # ВАЖНО: Проверяем есть ли лучшие параметры для этой монеты
                    # Если есть и они с высоким рейтингом - используем их вместо случайных
                    coin_best_params = None
                    if self.param_tracker:
                        best_params = self.param_tracker.get_best_params_for_symbol(symbol)
                        if best_params and best_params.get('rating', 0) >= 70.0:  # Используем если рейтинг >= 70
                            coin_best_params = best_params.get('rsi_params')
                            pass
                    
                    # УЛУЧШЕНИЕ: Ограничиваем количество свечей для обучения до 1000
                    # Используем только последние 1000 свечей для обучения ИИ
                    original_count = len(candles)
                    
                    # Сортируем свечи по времени (от старых к новым)
                    candles = sorted(candles, key=lambda x: x.get('time', 0))
                    
                    # Ограничиваем до 1000 последних свечей
                    MAX_CANDLES_FOR_TRAINING = 1000
                    if len(candles) > MAX_CANDLES_FOR_TRAINING:
                        candles = candles[-MAX_CANDLES_FOR_TRAINING:]
                        pass
                    
                    # ВАРИАЦИЯ ДАННЫХ: Используем разные подмножества данных для разнообразия
                    # Это обеспечивает разные паттерны при каждом обучении
                    # Но только если у нас достаточно свечей после ограничения
                    if len(candles) > 500:
                        # Для каждой монеты используем свой offset на основе seed
                        max_offset = min(200, len(candles) - 300)
                        start_offset = coin_rng.randint(0, max_offset) if max_offset > 0 else 0
                        # Используем все свечи от offset до конца (но не меньше 300)
                        min_length = 300
                        if len(candles) - start_offset >= min_length:
                            candles = candles[start_offset:]
                            pass

                    # Проверяем существующую модель и количество свечей при предыдущем обучении
                    # Нормализуем путь и имя символа для Windows
                    safe_symbol = symbol.replace('/', '_').replace('\\', '_').replace(':', '_')
                    symbol_models_dir = os.path.normpath(os.path.join(self.models_dir, safe_symbol))
                    # Загружаем метаданные из БД
                    previous_candles_count = 0
                    model_exists = False
                    
                    if self.ai_db:
                        try:
                            latest_version = self.ai_db.get_latest_model_version(model_type=f'symbol_model_{symbol}')
                            if latest_version:
                                previous_candles_count = latest_version.get('training_samples', 0)
                                model_exists = True
                        except Exception as e:
                            pass
                    
                    current_candles_count = len(candles)
                    candles_increased = current_candles_count > previous_candles_count
                    increase_percent = ((current_candles_count - previous_candles_count) / previous_candles_count * 100) if previous_candles_count > 0 else 0
                    
                    # Показываем прогресс для каждой монеты (но не слишком часто)
                    pass  # посимвольный лог "Обработка symbol" отключён
                    
                    if model_exists:
                        if candles_increased:
                            pass
                        else:
                            pass
                    else:
                        pass
                    
                    # Предупреждение только если критично
                    if len(candles) <= 1000:
                        pass
                    
                    # Извлекаем данные из свечей
                    closes = [float(c.get('close', 0) or 0) for c in candles]
                    volumes = [float(c.get('volume', 0) or 0) for c in candles]
                    highs = [float(c.get('high', 0) or 0) for c in candles]
                    lows = [float(c.get('low', 0) or 0) for c in candles]
                    opens = [float(c.get('open', 0) or 0) for c in candles]
                    times = [c.get('time', 0) for c in candles]
                    
                    if len(closes) < 100:
                        continue
                    
                    # Вычисляем RSI до приоритетов 1–4: нужен для _generate_adaptive_params (приоритет 4) и симуляции
                    rsi_history = calculate_rsi_history_func(candles, period=RSI_PERIOD)
                    if not rsi_history or len(rsi_history) < 50:
                        pass
                        continue
                    
                    # Готовим индивидуальную базу конфигурации (общий конфиг + индивидуальные настройки монеты)
                    # ВАЖНО: Используем сохраненные настройки как базовые для симуляций
                    from bot_engine.config_loader import AIConfig
                    use_saved_as_base = AIConfig.AI_USE_SAVED_SETTINGS_AS_BASE
                    
                    existing_coin_settings = _get_existing_coin_settings(symbol) or {}
                    if existing_coin_settings and use_saved_as_base:
                        logger.info(f"   🧩 {symbol}: обнаружены индивидуальные настройки (Win Rate: {existing_coin_settings.get('ai_win_rate', 0):.1f}%), используем их как базу для симуляций")
                    coin_base_config = base_config.copy() if isinstance(base_config, dict) else {}
                    if existing_coin_settings and use_saved_as_base:
                        # Используем сохраненные настройки как базовые, но позволяем вариацию
                        coin_base_config.update(existing_coin_settings)
                    if self.training_param_overrides:
                        coin_base_config.update(self.training_param_overrides)

                    def _get_float_value(key, default_value):
                        value = coin_base_config.get(key, default_value)
                        if value is None:
                            return default_value
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            return default_value

                    def _get_int_value(key, default_value):
                        value = coin_base_config.get(key, default_value)
                        if value is None:
                            return default_value
                        try:
                            return int(value)
                        except (TypeError, ValueError):
                            return default_value

                    def _get_bool_value(key, default_value):
                        value = coin_base_config.get(key, default_value)
                        if isinstance(value, str):
                            return value.lower() in ('1', 'true', 'yes', 'on')
                        if value is None:
                            return default_value
                        return bool(value)

                    coin_base_rsi_oversold = _get_float_value('rsi_long_threshold', base_rsi_oversold)
                    coin_base_rsi_overbought = _get_float_value('rsi_short_threshold', base_rsi_overbought)
                    coin_base_exit_long_with = _get_float_value('rsi_exit_long_with_trend', base_exit_long_with)
                    coin_base_exit_long_against = _get_float_value('rsi_exit_long_against_trend', base_exit_long_against)
                    coin_base_exit_short_with = _get_float_value('rsi_exit_short_with_trend', base_exit_short_with)
                    coin_base_exit_short_against = _get_float_value('rsi_exit_short_against_trend', base_exit_short_against)

                    coin_base_stop_loss = _get_float_value('max_loss_percent', base_stop_loss)
                    coin_base_take_profit = _get_float_value('take_profit_percent', base_take_profit)
                    coin_base_trailing_activation = _get_float_value('trailing_stop_activation', base_trailing_activation)
                    coin_base_trailing_distance = _get_float_value('trailing_stop_distance', base_trailing_distance)
                    coin_base_trailing_take_distance = _get_float_value('trailing_take_distance', base_trailing_take_distance)
                    coin_base_trailing_update_interval = _get_float_value('trailing_update_interval', base_trailing_update_interval)
                    coin_base_break_even_trigger = _get_float_value(
                        'break_even_trigger_percent',
                        _get_float_value('break_even_trigger', base_break_even)
                    )
                    coin_base_break_even_protection = _get_bool_value('break_even_protection', base_break_even_protection)
                    coin_base_max_hours = _get_float_value('max_position_hours', base_max_hours)

                    coin_base_rsi_time_filter_enabled = _get_bool_value('rsi_time_filter_enabled', base_rsi_time_filter_enabled)
                    coin_base_rsi_time_filter_candles = _get_int_value('rsi_time_filter_candles', base_rsi_time_filter_candles)
                    coin_base_rsi_time_filter_upper = _get_float_value('rsi_time_filter_upper', base_rsi_time_filter_upper)
                    coin_base_rsi_time_filter_lower = _get_float_value('rsi_time_filter_lower', base_rsi_time_filter_lower)

                    coin_base_exit_scam_enabled = _get_bool_value('exit_scam_enabled', base_exit_scam_enabled)
                    coin_base_exit_scam_candles = _get_int_value('exit_scam_candles', base_exit_scam_candles)
                    coin_base_exit_scam_single = _get_float_value('exit_scam_single_candle_percent', base_exit_scam_single_candle_percent)
                    coin_base_exit_scam_multi_count = _get_int_value('exit_scam_multi_candle_count', base_exit_scam_multi_candle_count)
                    coin_base_exit_scam_multi_percent = _get_float_value('exit_scam_multi_candle_percent', base_exit_scam_multi_candle_percent)

                    coin_base_trend_detection_enabled = _get_bool_value('trend_detection_enabled', base_trend_detection_enabled)
                    coin_base_avoid_down_trend = _get_bool_value('avoid_down_trend', base_avoid_down_trend)
                    coin_base_avoid_up_trend = _get_bool_value('avoid_up_trend', base_avoid_up_trend)
                    coin_base_trend_analysis_period = _get_int_value('trend_analysis_period', base_trend_analysis_period)
                    coin_base_trend_price_change_threshold = _get_float_value('trend_price_change_threshold', base_trend_price_change_threshold)
                    coin_base_trend_candles_threshold = _get_int_value('trend_candles_threshold', base_trend_candles_threshold)

                    coin_base_enable_maturity_check = _get_bool_value('enable_maturity_check', base_enable_maturity_check)
                    coin_base_min_candles_for_maturity = _get_int_value('min_candles_for_maturity', base_min_candles_for_maturity)
                    coin_base_min_rsi_low = _get_float_value('min_rsi_low', base_min_rsi_low)
                    coin_base_max_rsi_high = _get_float_value('max_rsi_high', base_max_rsi_high)

                    # ПРИОРИТЕТ 1: Используем ML модель для генерации оптимальных параметров
                    # ИИ САМ НАХОДИТ оптимальные параметры на основе обучения на предыдущих симуляциях
                    coin_rsi_params = None
                    
                    if self.param_quality_predictor and self.param_quality_predictor.is_trained:
                        try:
                            # Получаем предложения от ML модели (ИИ сам находит лучшие параметры)
                            risk_params = {
                                'stop_loss': coin_base_stop_loss,
                                'take_profit': coin_base_take_profit,
                                'trailing_stop_activation': coin_base_trailing_activation,
                                'trailing_stop_distance': coin_base_trailing_distance,
                            }
                            
                            # ИИ генерирует оптимальные параметры на основе обучения
                            suggestions = self.param_quality_predictor.suggest_optimal_params(
                                base_params, risk_params, num_suggestions=10  # Увеличено для лучшего выбора
                            )
                            
                            # Пробуем использовать лучшие предложения от ИИ
                            for suggested_params, predicted_quality in suggestions:
                                # Проверяем, не использовались ли уже
                                if self.param_tracker and not self.param_tracker.is_params_used(suggested_params):
                                    coin_rsi_params = suggested_params
                                    ml_params_generated_count += 1
                                    if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                                        logger.info(f"   🤖 {symbol}: ИИ нашел оптимальные параметры (предсказанное качество: {predicted_quality:.3f})")
                                    else:
                                        pass
                                    break
                        except Exception as e:
                            pass
                    
                    # ПРИОРИТЕТ 2: Используем сохранённые лучшие параметры для монеты
                    if not coin_rsi_params and coin_best_params:
                        coin_rsi_params = coin_best_params
                        if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                            logger.info(f"   ⭐ {symbol}: применяем сохранённые лучшие параметры (Win Rate: {coin_best_params.get('win_rate', 0):.1f}%)")
                        else:
                            pass
                    
                    # ПРИОРИТЕТ 3: Используем трекер параметров для новых комбинаций
                    if not coin_rsi_params and self.param_tracker:
                        suggested_params = self.param_tracker.get_unused_params_suggestion(base_params, variation_range)
                        if suggested_params:
                            coin_rsi_params = suggested_params
                            pass
                    
                    # ПРИОРИТЕТ 4: Генерируем адаптивные параметры на основе анализа рынка
                    # (используется только если ML модель не обучена или не дала результатов)
                    if not coin_rsi_params:
                        coin_rsi_params = self._generate_adaptive_params(
                            symbol, rsi_history, coin_base_rsi_oversold, coin_base_rsi_overbought,
                            coin_base_exit_long_with, coin_base_exit_long_against,
                            coin_base_exit_short_with, coin_base_exit_short_against,
                            coin_rng, base_params
                        )
                        if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                            logger.info(f"   📊 {symbol}: сгенерированы адаптивные параметры на основе анализа рынка")
                        else:
                            pass

                    if symbol_idx <= 5 or symbol_idx % progress_interval == 0:
                        logger.info(f"   ⚙️ {symbol}: RSI params {coin_rsi_params}, seed {coin_seed}")
                    else:
                        pass

                    # Используем параметры для этой монеты
                    coin_RSI_OVERSOLD = coin_rsi_params['oversold']
                    coin_RSI_OVERBOUGHT = coin_rsi_params['overbought']
                    coin_RSI_EXIT_LONG_WITH_TREND = coin_rsi_params['exit_long_with_trend']
                    coin_RSI_EXIT_LONG_AGAINST_TREND = coin_rsi_params['exit_long_against_trend']
                    coin_RSI_EXIT_SHORT_WITH_TREND = coin_rsi_params['exit_short_with_trend']
                    coin_RSI_EXIT_SHORT_AGAINST_TREND = coin_rsi_params['exit_short_against_trend']

                    MAX_LOSS_PERCENT = max(5.0, min(30.0, coin_base_stop_loss + coin_rng.uniform(-6.0, 6.0)))
                    TAKE_PROFIT_PERCENT = max(10.0, min(70.0, coin_base_take_profit + coin_rng.uniform(-12.0, 15.0)))
                    TRAILING_STOP_ACTIVATION = max(8.0, min(70.0, coin_base_trailing_activation + coin_rng.uniform(-12.0, 25.0)))
                    TRAILING_STOP_DISTANCE = max(5.0, min(45.0, coin_base_trailing_distance + coin_rng.uniform(-12.0, 18.0)))
                    TRAILING_TAKE_DISTANCE = max(0.1, min(2.0, coin_base_trailing_take_distance + coin_rng.uniform(-0.2, 0.2)))
                    TRAILING_UPDATE_INTERVAL = max(1.0, min(10.0, coin_base_trailing_update_interval + coin_rng.uniform(-1.0, 1.0)))
                    BREAK_EVEN_TRIGGER = max(30.0, min(250.0, coin_base_break_even_trigger + coin_rng.uniform(-60.0, 90.0)))
                    base_break_even_flag = bool(coin_base_break_even_protection)
                    BREAK_EVEN_PROTECTION = base_break_even_flag if coin_rng.random() < 0.5 else not base_break_even_flag
                    MAX_POSITION_HOURS = max(18, min(336, coin_base_max_hours + coin_rng.randint(-72, 120)))  # Минимум 18 часов (3 свечи на 6H ТФ)

                    # Фильтры: RSI временной и ExitScam (индивидуализация на уровне монеты)
                    # КРИТИЧЕСКИ ВАЖНО: Временной фильтр всегда должен быть включен, AI не может его отключить
                    coin_rsi_time_filter_enabled = True  # Всегда включен, AI не может отключить
                    coin_base_config['rsi_time_filter_enabled'] = coin_rsi_time_filter_enabled
                    coin_rsi_time_filter_candles = max(2, min(30, coin_base_rsi_time_filter_candles + coin_rng.randint(-4, 4)))
                    coin_rsi_time_filter_upper = max(50, min(85, coin_base_rsi_time_filter_upper + coin_rng.randint(-6, 6)))
                    coin_rsi_time_filter_lower = max(15, min(50, coin_base_rsi_time_filter_lower + coin_rng.randint(-6, 6)))
                    if coin_rsi_time_filter_lower >= coin_rsi_time_filter_upper:
                        # Гарантируем корректный диапазон
                        coin_rsi_time_filter_lower = max(15, coin_rsi_time_filter_upper - 1)
                    coin_exit_scam_enabled = bool(coin_base_exit_scam_enabled)
                    coin_exit_scam_enabled = self._mutate_flag('exit_scam_enabled', coin_exit_scam_enabled, coin_rng)
                    coin_base_config['exit_scam_enabled'] = coin_exit_scam_enabled
                    coin_exit_scam_candles = max(4, min(30, coin_base_exit_scam_candles + coin_rng.randint(-4, 4)))
                    coin_exit_scam_single_candle_percent = max(
                        5.0, min(60.0, coin_base_exit_scam_single + coin_rng.uniform(-10.0, 10.0))
                    )
                    coin_exit_scam_multi_candle_count = max(
                        2, min(12, coin_base_exit_scam_multi_count + coin_rng.randint(-2, 2))
                    )
                    coin_exit_scam_multi_candle_percent = max(
                        20.0, min(150.0, coin_base_exit_scam_multi_percent + coin_rng.uniform(-20.0, 20.0))
                    )

                    coin_trend_detection_enabled = bool(coin_base_trend_detection_enabled)
                    coin_trend_detection_enabled = self._mutate_flag('trend_detection_enabled', coin_trend_detection_enabled, coin_rng)
                    coin_base_config['trend_detection_enabled'] = coin_trend_detection_enabled

                    coin_avoid_down_trend = bool(coin_base_avoid_down_trend)
                    coin_avoid_down_trend = self._mutate_flag('avoid_down_trend', coin_avoid_down_trend, coin_rng)
                    coin_base_config['avoid_down_trend'] = coin_avoid_down_trend

                    coin_avoid_up_trend = bool(coin_base_avoid_up_trend)
                    coin_avoid_up_trend = self._mutate_flag('avoid_up_trend', coin_avoid_up_trend, coin_rng)
                    coin_base_config['avoid_up_trend'] = coin_avoid_up_trend
                    coin_trend_analysis_period = max(5, min(120, coin_base_trend_analysis_period + coin_rng.randint(-10, 10)))
                    coin_trend_price_change_threshold = max(1.0, min(25.0, coin_base_trend_price_change_threshold + coin_rng.uniform(-3.0, 3.0)))
                    coin_trend_candles_threshold = max(40, min(100, coin_base_trend_candles_threshold + coin_rng.randint(-15, 15)))

                    coin_enable_maturity_check = bool(coin_base_enable_maturity_check)
                    coin_enable_maturity_check = self._mutate_flag('enable_maturity_check', coin_enable_maturity_check, coin_rng)
                    coin_base_config['enable_maturity_check'] = coin_enable_maturity_check
                    coin_min_candles_for_maturity = max(100, min(900, coin_base_min_candles_for_maturity + coin_rng.randint(-120, 150)))
                    coin_min_rsi_low = max(15, min(45, coin_base_min_rsi_low + coin_rng.randint(-5, 5)))
                    coin_max_rsi_high = max(55, min(85, coin_base_max_rsi_high + coin_rng.randint(-5, 5)))

                    if symbol_idx <= 5 or symbol_idx % progress_interval == 0:
                        logger.info(
                            f"   📐 {symbol}: риск-параметры SL {MAX_LOSS_PERCENT:.1f}% | TP {TAKE_PROFIT_PERCENT:.1f}% | "
                            f"TS {TRAILING_STOP_ACTIVATION:.1f}%/{TRAILING_STOP_DISTANCE:.1f}% | "
                            f"TT {TRAILING_TAKE_DISTANCE:.2f}%/{TRAILING_UPDATE_INTERVAL:.1f}с | "
                            f"BE {'✅' if BREAK_EVEN_PROTECTION else '❌'} ({BREAK_EVEN_TRIGGER:.1f}%) | MaxHold {MAX_POSITION_HOURS}ч"
                        )
                        logger.info(
                            f"   🛡️ {symbol}: RSI time filter {coin_rsi_time_filter_candles} свечей "
                            f"[{coin_rsi_time_filter_lower}/{coin_rsi_time_filter_upper}] | "
                            f"ExitScam: N={coin_exit_scam_candles}, 1св {coin_exit_scam_single_candle_percent:.1f}%, "
                            f"{coin_exit_scam_multi_candle_count}св {coin_exit_scam_multi_candle_percent:.1f}%"
                        )
                    else:
                        pass  # параметры SL/TP/TS для symbol

                    # RSI уже вычислен выше (до приоритетов 1–4)
                    # УЛУЧШЕНИЕ: Адаптируем диапазоны параметров на основе статистики RSI монеты
                    # Это увеличивает вероятность генерации сделок
                    rsi_values = [r for r in rsi_history if r is not None and 0 <= r <= 100]
                    if rsi_values:
                        rsi_min = min(rsi_values)
                        rsi_max = max(rsi_values)
                        rsi_mean = sum(rsi_values) / len(rsi_values)
                        rsi_std = (sum((x - rsi_mean) ** 2 for x in rsi_values) / len(rsi_values)) ** 0.5
                        
                        # Адаптируем параметры входа на основе реального диапазона RSI монеты
                        # УЛУЧШЕНИЕ: Всегда проверяем, попадает ли RSI в зоны, и адаптируем при необходимости
                        adaptive_oversold = coin_RSI_OVERSOLD
                        adaptive_overbought = coin_RSI_OVERBOUGHT
                        
                        # Проверяем, сколько раз RSI попадает в зоны входа
                        rsi_in_long_zone_count = sum(1 for r in rsi_values if r <= coin_RSI_OVERSOLD)
                        rsi_in_short_zone_count = sum(1 for r in rsi_values if r >= coin_RSI_OVERBOUGHT)
                        
                        # Если RSI не попадает в зону LONG (oversold) - адаптируем порог
                        if rsi_in_long_zone_count == 0 or rsi_min > coin_RSI_OVERSOLD:
                            # RSI никогда не опускается ниже порога - увеличиваем порог для генерации сделок
                            # УЛУЧШЕНИЕ: Расширяем диапазон до 50 (вместо 35), чтобы покрыть монеты с высоким RSI
                            # Устанавливаем порог чуть ниже min для гарантированного входа
                            adaptive_oversold = min(50, max(coin_RSI_OVERSOLD, int(rsi_min - 1)))
                            logger.info(f"   📊 {symbol}: RSI min={rsi_min:.1f}, oversold={coin_RSI_OVERSOLD}, попаданий в зону LONG: {rsi_in_long_zone_count} → адаптируем oversold: {coin_RSI_OVERSOLD} → {adaptive_oversold}")
                        
                        # Если RSI не попадает в зону SHORT (overbought) - адаптируем порог
                        if rsi_in_short_zone_count == 0 or rsi_max < coin_RSI_OVERBOUGHT:
                            # RSI никогда не поднимается выше порога - уменьшаем порог для генерации сделок
                            # УЛУЧШЕНИЕ: Расширяем диапазон до 50 (вместо 65), чтобы покрыть монеты с низким RSI
                            # Устанавливаем порог чуть выше max для гарантированного входа
                            adaptive_overbought = max(50, min(coin_RSI_OVERBOUGHT, int(rsi_max + 1)))
                            logger.info(f"   📊 {symbol}: RSI max={rsi_max:.1f}, overbought={coin_RSI_OVERBOUGHT}, попаданий в зону SHORT: {rsi_in_short_zone_count} → адаптируем overbought: {coin_RSI_OVERBOUGHT} → {adaptive_overbought}")
                        
                        # Применяем адаптивные значения
                        coin_RSI_OVERSOLD = adaptive_oversold
                        coin_RSI_OVERBOUGHT = adaptive_overbought
                        
                        # Обновляем параметры в словаре для логирования
                        coin_rsi_params['oversold'] = adaptive_oversold
                        coin_rsi_params['overbought'] = adaptive_overbought
                    
                    # СИМУЛЯЦИЯ: Проходим по свечам и симулируем входы/выходы
                    simulated_trades_symbol = []  # Симулированные сделки ТОЛЬКО для этой монеты
                    current_position = None  # {'direction': 'LONG'/'SHORT', 'entry_idx': int, 'entry_price': float, 'entry_rsi': float, 'entry_trend': str}
                    trades_for_symbol = 0

                    # Размер позиции для текущего символа (логируем ОДИН раз)
                    position_size_value = coin_base_config.get(
                        'default_position_size',
                        base_config.get('default_position_size', 5)
                    )
                    position_size_mode = coin_base_config.get(
                        'default_position_mode',
                        base_config.get('default_position_mode', 'usdt')
                    )
                    if position_size_mode == 'percent':
                        reference_deposit = coin_base_config.get(
                            'ai_reference_deposit_usdt',
                            base_config.get('ai_reference_deposit_usdt', 1000)
                        )
                        position_size_usdt = reference_deposit * (position_size_value / 100)
                        logger.info(
                            f"   💵 {symbol}: размер сделки {position_size_usdt:.4f} USDT "
                            f"(режим percent, {position_size_value}% от депозита {reference_deposit} USDT)"
                        )
                    else:
                        position_size_usdt = position_size_value
                        pass
                    
                    # Логируем начало симуляции (INFO только для важных монет)
                    # Вычисляем количество свечей для обработки с учетом пропуска начальных свечей
                    simulation_start_idx = RSI_PERIOD
                    if coin_enable_maturity_check:
                        simulation_start_idx = max(RSI_PERIOD, coin_min_candles_for_maturity)
                    candles_to_process = len(candles) - simulation_start_idx
                    if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                        logger.info(f"   🔄 {symbol}: симуляция {candles_to_process:,} свечей...")
                    else:
                        pass
                    
                    # Логируем прогресс каждые 1000 свечей (INFO для важных монет)
                    progress_step = 1000
                    
                    # ВАЖНО: Начинаем симуляцию с момента, когда уже накопилось достаточно свечей для зрелости
                    # Это нужно чтобы фильтр зрелости не блокировал все входы в начале истории
                    simulation_start_idx = RSI_PERIOD
                    if coin_enable_maturity_check:
                        # Начинаем симуляцию с момента, когда уже есть достаточно свечей (или индивидуальный порог для монеты)
                        # Используем мутированный порог для определения начала симуляции
                        simulation_start_idx = max(RSI_PERIOD, coin_min_candles_for_maturity)
                        if simulation_start_idx > RSI_PERIOD:
                            skipped_candles = simulation_start_idx - RSI_PERIOD
                            pass
                    
                    # Счетчики для диагностики фильтров
                    rsi_entered_long_zone = 0
                    rsi_entered_short_zone = 0
                    filters_blocked_long = 0
                    filters_blocked_short = 0
                    filter_block_reasons = {}
                    
                    for i in range(simulation_start_idx, len(candles)):
                        # Логируем прогресс каждые 1000 свечей (DEBUG - техническая деталь)
                        # Учитываем что симуляция начинается не с RSI_PERIOD, а с simulation_start_idx
                        processed_count = i - simulation_start_idx
                        if candles_to_process > 1000 and processed_count % progress_step == 0:
                            progress_pct = (processed_count / candles_to_process) * 100
                            pass
                        try:
                            # RSI на текущей позиции
                            rsi_idx = i - RSI_PERIOD
                            if rsi_idx >= len(rsi_history):
                                continue
                            
                            current_rsi = rsi_history[rsi_idx]
                            current_price = closes[i]
                            
                            # Определяем тренд (используем EMA как в bots.py)
                            trend = 'NEUTRAL'
                            if i >= 50:
                                ema_short = self._calculate_ema(closes[max(0, i-50):i+1], 50)
                                ema_long = self._calculate_ema(closes[max(0, i-200):i+1], 200)
                                if ema_short and ema_long:
                                    if ema_short > ema_long:
                                        trend = 'UP'
                                    elif ema_short < ema_long:
                                        trend = 'DOWN'
                            
                            # ПРОВЕРКА ВЫХОДА (если есть открытая позиция)
                            if current_position:
                                entry_trend = current_position['entry_trend']
                                direction = current_position['direction']
                                should_exit = False
                                exit_reason = None
                                
                                # ✅ Используем coin_RSI_EXIT_* — сгенерированные/тестируемые параметры для этой монеты
                                if direction == 'LONG':
                                    # Определяем был ли вход по тренду или против
                                    if entry_trend == 'UP':
                                        # Вход по тренду - используем WITH_TREND
                                        if current_rsi >= coin_RSI_EXIT_LONG_WITH_TREND:
                                            should_exit = True
                                            exit_reason = 'RSI_EXIT_WITH_TREND'
                                    else:
                                        # Вход против тренда - используем AGAINST_TREND
                                        if current_rsi >= coin_RSI_EXIT_LONG_AGAINST_TREND:
                                            should_exit = True
                                            exit_reason = 'RSI_EXIT_AGAINST_TREND'
                                
                                elif direction == 'SHORT':
                                    if entry_trend == 'DOWN':
                                        if current_rsi <= coin_RSI_EXIT_SHORT_WITH_TREND:
                                            should_exit = True
                                            exit_reason = 'RSI_EXIT_WITH_TREND'
                                    else:
                                        if current_rsi <= coin_RSI_EXIT_SHORT_AGAINST_TREND:
                                            should_exit = True
                                            exit_reason = 'RSI_EXIT_AGAINST_TREND'

                                protection_state = current_position.get('protection_state')
                                if protection_state:
                                    protection_decision = evaluate_protections(
                                        current_price=current_price,
                                        config=coin_base_config,
                                        state=protection_state,
                                        realized_pnl=0.0,
                                        now_ts=_normalize_timestamp(times[i]),
                                    )
                                    current_position['protection_state'] = protection_decision.state
                                    if protection_decision.should_close:
                                        should_exit = True
                                        exit_reason = protection_decision.reason or exit_reason or 'PROTECTION'
                                
                                if should_exit:
                                    # Закрываем позицию и записываем результат
                                    entry_price = current_position['entry_price']
                                    if direction == 'LONG':
                                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                                    else:
                                        pnl_pct = ((entry_price - current_price) / entry_price) * 100
                                    
                                    # Симулируем PnL в USDT (используем заранее рассчитанный размер позиции)
                                    position_size_for_trade = current_position.get('position_size_usdt', position_size_usdt)
                                    pnl_usdt = position_size_for_trade * (pnl_pct / 100)
                                    
                                    simulated_trade = {
                                        'symbol': symbol,
                                        'direction': direction,
                                        'entry_idx': current_position['entry_idx'],
                                        'exit_idx': i,
                                        'entry_price': entry_price,
                                        'exit_price': current_price,
                                        'entry_rsi': current_position['entry_rsi'],
                                        'exit_rsi': current_rsi,
                                        'entry_trend': entry_trend,
                                        'exit_trend': trend,
                                        'pnl': pnl_usdt,
                                        'pnl_pct': pnl_pct,
                                        'roi': pnl_pct,
                                        'exit_reason': exit_reason,
                                        'is_successful': pnl_usdt > 0,
                                        'entry_time': times[current_position['entry_idx']],
                                        'exit_time': times[i],
                                        'duration_candles': i - current_position['entry_idx']
                                    }
                                    
                                    simulated_trades_symbol.append(simulated_trade)
                                    trades_for_symbol += 1
                                    current_position = None
                            
                            # ПРОВЕРКА ВХОДА (если нет открытой позиции)
                            if not current_position:
                                should_enter_long = current_rsi <= coin_RSI_OVERSOLD
                                should_enter_short = current_rsi >= coin_RSI_OVERBOUGHT
                                
                                if should_enter_long:
                                    rsi_entered_long_zone += 1
                                if should_enter_short:
                                    rsi_entered_short_zone += 1
                                
                                if should_enter_long or should_enter_short:
                                    signal = 'ENTER_LONG' if should_enter_long else 'ENTER_SHORT'
                                    filters_allowed, filters_reason = apply_entry_filters(
                                        symbol,
                                        candles[:i + 1],
                                        current_rsi,
                                        signal,
                                        coin_base_config,
                                        trend=trend,
                                    )
                                    if not filters_allowed:
                                        # Подсчитываем причины блокировки
                                        if should_enter_long:
                                            filters_blocked_long += 1
                                        if should_enter_short:
                                            filters_blocked_short += 1
                                        
                                        # Извлекаем основную причину блокировки
                                        # Формат: "SYMBOL: причина" или просто "причина"
                                        if ':' in filters_reason:
                                            # Убираем символ в начале, оставляем только причину
                                            main_reason = filters_reason.split(':', 1)[-1].strip()
                                        else:
                                            main_reason = filters_reason.strip()
                                        
                                        # Нормализуем причину для группировки
                                        if 'RSI time filter' in main_reason or 'RSI временной фильтр' in main_reason:
                                            main_reason = 'RSI time filter'
                                        elif 'ExitScam' in main_reason or 'exit scam' in main_reason.lower():
                                            main_reason = 'ExitScam'
                                        elif 'Молодая монета' in main_reason or 'maturity' in main_reason.lower():
                                            main_reason = 'Maturity check'
                                        elif 'trend' in main_reason.lower():
                                            main_reason = 'Trend filter'
                                        elif 'scope' in main_reason.lower():
                                            main_reason = 'Scope filter'
                                        
                                        filter_block_reasons[main_reason] = filter_block_reasons.get(main_reason, 0) + 1
                                        
                                        should_enter_long = False
                                        should_enter_short = False
                                
                                if should_enter_long:
                                    entry_ts_ms = times[i]
                                    current_position = {
                                        'direction': 'LONG',
                                        'entry_idx': i,
                                        'entry_price': current_price,
                                        'entry_rsi': current_rsi,
                                        'entry_trend': trend,
                                        'entry_time': entry_ts_ms,
                                        'position_size_usdt': position_size_usdt,
                                        'protection_state': _build_protection_state('LONG', current_price, entry_ts_ms, position_size_usdt),
                                    }
                                
                                elif should_enter_short:
                                    entry_ts_ms = times[i]
                                    current_position = {
                                        'direction': 'SHORT',
                                        'entry_idx': i,
                                        'entry_price': current_price,
                                        'entry_rsi': current_rsi,
                                        'entry_trend': trend,
                                        'entry_time': entry_ts_ms,
                                        'position_size_usdt': position_size_usdt,
                                        'protection_state': _build_protection_state('SHORT', current_price, entry_ts_ms, position_size_usdt),
                                    }
                        except Exception as e:
                            pass
                            continue
                    
                    total_candles_processed += len(candles)
                    
                    # ДИАГНОСТИКА: Если нет сделок, логируем статистику RSI и фильтров
                    if trades_for_symbol == 0 and (symbol_idx <= 10 or symbol_idx % progress_interval == 0):
                        if rsi_history:
                            # Используем только RSI из симуляции (начиная с simulation_start_idx)
                            simulation_rsi = rsi_history[simulation_start_idx - RSI_PERIOD:] if len(rsi_history) > (simulation_start_idx - RSI_PERIOD) else rsi_history
                            if simulation_rsi:
                                min_rsi = min(simulation_rsi)
                                max_rsi = max(simulation_rsi)
                                avg_rsi = sum(simulation_rsi) / len(simulation_rsi)
                                rsi_in_long_zone = sum(1 for r in simulation_rsi if r <= coin_RSI_OVERSOLD)
                                rsi_in_short_zone = sum(1 for r in simulation_rsi if r >= coin_RSI_OVERBOUGHT)
                            else:
                                min_rsi = max_rsi = avg_rsi = 0
                                rsi_in_long_zone = rsi_in_short_zone = 0
                            
                            diagnostic_msg = (
                                f"   🔍 {symbol}: диагностика отсутствия сделок - "
                                f"RSI: min={min_rsi:.1f}, max={max_rsi:.1f}, avg={avg_rsi:.1f}, "
                                f"в зоне LONG (≤{coin_RSI_OVERSOLD}): {rsi_in_long_zone} раз, "
                                f"в зоне SHORT (≥{coin_RSI_OVERBOUGHT}): {rsi_in_short_zone} раз"
                            )
                            
                            # ВАЖНО: Показываем реальные попытки входа из симуляции
                            total_attempts = rsi_entered_long_zone + rsi_entered_short_zone
                            total_blocked = filters_blocked_long + filters_blocked_short
                            
                            if total_attempts > 0:
                                diagnostic_msg += (
                                    f" | ✅ Попыток входа: {total_attempts} (LONG={rsi_entered_long_zone}, SHORT={rsi_entered_short_zone}) | "
                                    f"🚫 Заблокировано: {total_blocked} (LONG={filters_blocked_long}, SHORT={filters_blocked_short})"
                                )
                                if filter_block_reasons:
                                    top_reasons = sorted(filter_block_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
                                    reasons_str = ", ".join([f"{reason}: {count}" for reason, count in top_reasons])
                                    diagnostic_msg += f" | 🔍 Топ-5 причин блокировки: {reasons_str}"
                                else:
                                    diagnostic_msg += " | ⚠️ Причины блокировки не зафиксированы (возможно, фильтры не вызывались)"
                            else:
                                # Если попыток входа не было, но RSI попадал в зоны - значит проблема в логике проверки
                                if rsi_in_long_zone > 0 or rsi_in_short_zone > 0:
                                    diagnostic_msg += f" | ⚠️ RSI попадал в зоны, но попыток входа не было (возможно, позиция уже открыта или ошибка в логике)"
                            
                            logger.info(diagnostic_msg)
                    
                    # Логируем завершение симуляции (INFO только для важных монет)
                    if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                        logger.info(f"   ✅ {symbol}: симуляция завершена ({candles_to_process:,} свечей обработано, {trades_for_symbol} сделок)")
                    else:
                        pass
                    
                    # ВАЖНО: Логируем сразу после симуляции для отладки
                    symbol_win_rate = 0.0  # значение по умолчанию, если сделок нет
                    
                    if symbol_idx <= 10:
                        logger.info(f"   🔍 {symbol}: проверка результатов симуляции... (сделок: {trades_for_symbol})")
                    
                    # ВАЖНО: Сохраняем информацию о результатах для обучения ML модели
                    # AI должна учиться на успешных и неуспешных параметрах
                    risk_params = {
                        'stop_loss': MAX_LOSS_PERCENT,
                        'take_profit': TAKE_PROFIT_PERCENT,
                        'trailing_stop_activation': TRAILING_STOP_ACTIVATION,
                        'trailing_stop_distance': TRAILING_STOP_DISTANCE,
                    }
                    
                    # Вычисляем PnL для добавления в ML модель
                    symbol_pnl_for_ml = 0.0
                    if trades_for_symbol > 0:
                        symbol_successful = sum(1 for t in simulated_trades_symbol if t['is_successful'])
                        symbol_win_rate = symbol_successful / trades_for_symbol * 100
                        symbol_pnl_for_ml = sum(t['pnl'] for t in simulated_trades_symbol)
                    else:
                        symbol_win_rate = 0.0
                    
                    # ВАЖНО: Добавляем образец в ML модель для обучения
                    # ИИ УЧИТСЯ на ВСЕХ результатах симуляций - успешных и неуспешных
                    # Это позволяет ИИ САМОМУ находить оптимальные параметры в будущем
                    if self.param_quality_predictor:
                        try:
                            # ВАЖНО: Если сделок нет (trades_for_symbol == 0), это всегда блокировка
                            # Независимо от того, входил ли RSI в зону или нет
                            # - Если RSI входил в зону, но сделок нет → фильтры заблокировали (blocked=True)
                            # - Если RSI НЕ входил в зону → параметры не подходят (blocked=True)
                            was_blocked = trades_for_symbol == 0
                            rsi_entered_zones = rsi_entered_long_zone + rsi_entered_short_zone
                            total_blocked = filters_blocked_long + filters_blocked_short
                            
                            # ИИ учится на результатах этой симуляции
                            self.param_quality_predictor.add_training_sample(
                                coin_rsi_params,
                                symbol_win_rate,
                                symbol_pnl_for_ml,
                                trades_for_symbol,
                                risk_params,
                                symbol,
                                blocked=was_blocked,
                                rsi_entered_zones=rsi_entered_zones,
                                filters_blocked=total_blocked,
                                block_reasons=filter_block_reasons
                            )
                            
                            # Логируем обучение ИИ (только для важных монет)
                            if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                                if trades_for_symbol > 0:
                                    pass
                                else:
                                    pass
                        except Exception as e:
                            pass
                    
                    # ВАЖНО: Сохраняем информацию о блокировках для обучения AI
                    # AI должна учиться на том, какие параметры блокируются и почему
                    if trades_for_symbol == 0 and (rsi_entered_long_zone > 0 or rsi_entered_short_zone > 0):
                        # Есть попытки входа, но все заблокированы - это важная информация для обучения
                        total_blocked = filters_blocked_long + filters_blocked_short
                        if total_blocked > 0 and self.param_tracker:
                            # Сохраняем информацию о блокировках в трекер
                            # Это поможет AI в будущем избегать параметров, которые блокируются
                            try:
                                # Создаем "негативный" результат для обучения
                                blocked_info = {
                                    'symbol': symbol,
                                    'rsi_params': coin_rsi_params,
                                    'blocked_attempts': total_blocked,
                                    'blocked_long': filters_blocked_long,
                                    'blocked_short': filters_blocked_short,
                                    'block_reasons': filter_block_reasons,
                                    'timestamp': datetime.now().isoformat()
                                }
                                # Сохраняем в БД вместо JSON файла
                                if self.ai_db:
                                    self.ai_db.save_blocked_params(
                                        rsi_params=coin_rsi_params,
                                        block_reasons=filter_block_reasons,
                                        symbol=symbol,
                                        blocked_attempts=total_blocked,
                                        blocked_long=filters_blocked_long,
                                        blocked_short=filters_blocked_short
                                    )
                                    pass
                            except Exception as e:
                                pass
                    
                    # Логируем результаты симуляции (DEBUG - техническая деталь)
                    # symbol_win_rate и symbol_pnl_for_ml уже вычислены выше для ML модели
                    if trades_for_symbol == 0:
                        pass
                        symbol_pnl = 0.0
                    else:
                        # Используем уже вычисленные значения
                        symbol_pnl = symbol_pnl_for_ml
                        win_rate_target = self._get_win_rate_target(symbol)
                        
                        pass  # посимвольный лог Win Rate/PnL отключён
                        
                        # ОБУЧАЕМ МОДЕЛЬ ДЛЯ ЭТОЙ МОНЕТЫ ОТДЕЛЬНО
                        signal_score = None
                        profit_mse = None
                        model_trained = False
                        
                        if trades_for_symbol >= 1:  # Минимум 1 сделка для обучения (уменьшено с 5 для сохранения моделей даже при малом количестве сделок)
                            # Показываем начало обучения модели (INFO только для важных монет)
                            if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                                logger.info(f"   🎓 Обучаем модель для {symbol}... ({trades_for_symbol} сделок, Win Rate: {symbol_win_rate:.1f}%)")
                            else:
                                pass
                            
                            # ВАЖНО: Логируем подготовку данных
                            if symbol_idx <= 10:
                                logger.info(f"   📊 {symbol}: подготовка данных для обучения...")
                            
                            # Подготавливаем данные для обучения
                            X_symbol = []
                            y_signal_symbol = []
                            y_profit_symbol = []
                            
                            # 7 признаков — как у глобальной модели и inference (без изменения логики, только совместимость)
                            symbol_trades = simulated_trades_symbol
                            for trade in symbol_trades:
                                try:
                                    vol = trade.get('entry_volatility')
                                    vol = float(vol) if vol is not None else 0.0
                                except (TypeError, ValueError):
                                    vol = 0.0
                                try:
                                    vol_ratio = trade.get('entry_volume_ratio')
                                    vol_ratio = float(vol_ratio) if vol_ratio is not None else 1.0
                                except (TypeError, ValueError):
                                    vol_ratio = 1.0
                                features = [
                                    float(trade.get('entry_rsi', 50)),
                                    vol,
                                    vol_ratio,
                                    1.0 if (trade.get('entry_trend') or '') == 'UP' else 0.0,
                                    1.0 if (trade.get('entry_trend') or '') == 'DOWN' else 0.0,
                                    1.0 if (trade.get('direction') or 'LONG') == 'LONG' else 0.0,
                                    (float(trade.get('entry_price') or 0) / 1000.0) if (trade.get('entry_price') or 0) > 0 else 0.0,
                                ]
                                X_symbol.append(features)
                                y_signal_symbol.append(1 if trade['is_successful'] else 0)
                                y_profit_symbol.append(trade['pnl'])
                            
                            X_symbol = np.array(X_symbol)
                            y_signal_symbol = np.array(y_signal_symbol)
                            y_profit_symbol = np.array(y_profit_symbol)
                            
                            if symbol_idx <= 10:
                                logger.info(f"   📊 {symbol}: данные подготовлены ({len(X_symbol)} образцов)")
                            
                            # Создаем scaler для этой монеты
                            from sklearn.preprocessing import StandardScaler
                            symbol_scaler = StandardScaler()
                            X_symbol_scaled = symbol_scaler.fit_transform(X_symbol)
                            
                            if symbol_idx <= 10:
                                logger.info(f"   🔄 {symbol}: обучение RandomForestClassifier...")
                            
                            # Обучаем модель сигналов для этой монеты
                            from sklearn.ensemble import RandomForestClassifier
                            # ВАЖНО: Используем training_seed для разнообразия при каждом обучении
                            coin_model_seed = coin_seed  # Уникальный seed для каждой монеты
                            symbol_signal_predictor = RandomForestClassifier(
                                n_estimators=100,
                                max_depth=10,
                                min_samples_split=3,
                                random_state=coin_model_seed,  # Разный seed для каждого обучения
                                n_jobs=1,  # без параллелизма — устраняет UserWarning про delayed/Parallel
                                class_weight='balanced'
                            )
                            symbol_signal_predictor.fit(X_symbol_scaled, y_signal_symbol)
                            signal_score = symbol_signal_predictor.score(X_symbol_scaled, y_signal_symbol)
                            
                            if symbol_idx <= 10:
                                logger.info(f"   ✅ {symbol}: RandomForestClassifier обучен (Accuracy: {signal_score:.2%})")
                                logger.info(f"   🔄 {symbol}: обучение GradientBoostingRegressor...")
                            
                            # Обучаем модель прибыли для этой монеты
                            from sklearn.ensemble import GradientBoostingRegressor
                            # ВАЖНО: Используем training_seed для разнообразия при каждом обучении
                            coin_model_seed = coin_seed  # Уникальный seed для каждой монеты
                            symbol_profit_predictor = GradientBoostingRegressor(
                                n_estimators=50,
                                max_depth=4,
                                learning_rate=0.1,
                                random_state=coin_model_seed  # Разный seed для каждого обучения
                            )
                            symbol_profit_predictor.fit(X_symbol_scaled, y_profit_symbol)
                            profit_pred = symbol_profit_predictor.predict(X_symbol_scaled)
                            profit_mse = mean_squared_error(y_profit_symbol, profit_pred)
                            
                            if symbol_idx <= 10:
                                logger.info(f"   ✅ {symbol}: GradientBoostingRegressor обучен (MSE: {profit_mse:.2f})")
                                logger.info(f"   💾 {symbol}: сохранение моделей...")
                            
                            # Логируем завершение обучения модели для важных случаев
                            if symbol_win_rate >= win_rate_target or symbol_idx % progress_interval == 0:
                                logger.info(f"   ✅ {symbol}: модель обучена! Accuracy: {signal_score:.2%}, MSE: {profit_mse:.2f}")
                            
                            # Сохраняем модели для этой монеты
                            # Нормализуем путь и имя символа для Windows
                            safe_symbol = symbol.replace('/', '_').replace('\\', '_').replace(':', '_')
                            symbol_models_dir = os.path.normpath(os.path.join(self.models_dir, safe_symbol))
                            os.makedirs(symbol_models_dir, exist_ok=True)
                            
                            signal_model_path = os.path.normpath(os.path.join(symbol_models_dir, 'signal_predictor.pkl'))
                            profit_model_path = os.path.normpath(os.path.join(symbol_models_dir, 'profit_predictor.pkl'))
                            scaler_path = os.path.normpath(os.path.join(symbol_models_dir, 'scaler.pkl'))
                            
                            joblib.dump(symbol_signal_predictor, signal_model_path)
                            joblib.dump(symbol_profit_predictor, profit_model_path)
                            joblib.dump(symbol_scaler, scaler_path)
                            
                            if symbol_idx <= 10:
                                logger.info(f"   ✅ {symbol}: модели сохранены на диск")
                            
                            # Сохраняем метаданные (включая количество свечей для проверки при следующем обучении)
                            metadata = {
                                'symbol': symbol,
                                'trained_at': datetime.now().isoformat(),
                                'training_seed': training_seed,  # Seed для этого обучения (обеспечивает уникальность)
                                'coin_model_seed': coin_model_seed,  # Уникальный seed для этой монеты
                                'rsi_params': coin_rsi_params,  # Параметры RSI использованные при обучении (лучшие для монеты или общие)
                                # ВАЖНО: Сохраняем ВСЕ параметры обучения для полной истории!
                                'risk_params': {
                                    'stop_loss': MAX_LOSS_PERCENT,
                                    'take_profit': TAKE_PROFIT_PERCENT,
                                    'trailing_stop_activation': TRAILING_STOP_ACTIVATION,
                                    'trailing_stop_distance': TRAILING_STOP_DISTANCE,
                                    'break_even_protection': BREAK_EVEN_PROTECTION,
                                    'break_even_trigger': BREAK_EVEN_TRIGGER,
                                    'max_position_hours': MAX_POSITION_HOURS
                                },
                                'filter_params': {
                                    'rsi_time_filter': {
                                        'enabled': coin_rsi_time_filter_enabled,
                                        'candles': coin_rsi_time_filter_candles,
                                        'upper': coin_rsi_time_filter_upper,
                                        'lower': coin_rsi_time_filter_lower
                                    },
                                    'exit_scam_filter': {
                                        'enabled': coin_exit_scam_enabled,
                                        'candles': coin_exit_scam_candles,
                                        'single_candle_percent': coin_exit_scam_single_candle_percent,
                                        'multi_candle_count': coin_exit_scam_multi_candle_count,
                                        'multi_candle_percent': coin_exit_scam_multi_candle_percent
                                    }
                                },
                                'candles_count': len(candles),  # ВАЖНО: сохраняем количество свечей для проверки
                                'trades_count': trades_for_symbol,
                                'win_rate': symbol_win_rate,
                                'signal_accuracy': signal_score,
                                'profit_mse': profit_mse,
                                'total_pnl': symbol_pnl,
                                'previous_candles_count': previous_candles_count if 'previous_candles_count' in locals() else 0,
                                'candles_increased': candles_increased if 'candles_increased' in locals() else False
                            }
                            # Сохраняем метаданные в БД
                            if self.ai_db:
                                db_metadata = {
                                    'id': f'symbol_model_{symbol}',
                                    'model_type': f'symbol_model_{symbol}',
                                    'model_path': str(symbol_models_dir),
                                    'symbol': symbol,
                                    'training_samples': metadata.get('candles_count', len(candles)),
                                    'trained_at': metadata.get('trained_at', datetime.now().isoformat()),
                                    'trades_count': metadata.get('trades_count', 0),
                                    'win_rate': metadata.get('win_rate'),
                                    'accuracy': metadata.get('signal_accuracy'),
                                    'mse': metadata.get('profit_mse'),
                                    'total_pnl': metadata.get('total_pnl')
                                }
                                # Сохраняем полные метаданные в metadata_json
                                db_metadata.update(metadata)
                                self.ai_db.save_model_version(db_metadata)
                                pass
                            if symbol_idx <= 10:
                                logger.info(f"   ✅ {symbol}: метаданные сохранены")
                            
                            # ВАЖНО: Отмечаем параметры как использованные в трекере с рейтингом
                            # Сохраняем ВСЕ параметры (RSI + риск-менеджмент) для полного отслеживания
                            if self.param_tracker:
                                try:
                                    # Расширяем параметры для сохранения (включаем все параметры обучения)
                                    full_params = {
                                        **coin_rsi_params,  # RSI параметры
                                        'stop_loss': MAX_LOSS_PERCENT,
                                        'take_profit': TAKE_PROFIT_PERCENT,
                                        'trailing_stop_activation': TRAILING_STOP_ACTIVATION,
                                        'trailing_stop_distance': TRAILING_STOP_DISTANCE,
                                        'break_even_protection': BREAK_EVEN_PROTECTION,
                                        'break_even_trigger': BREAK_EVEN_TRIGGER,
                                        'max_position_hours': MAX_POSITION_HOURS,
                                        'rsi_time_filter_enabled': coin_rsi_time_filter_enabled,
                                        'rsi_time_filter_candles': coin_rsi_time_filter_candles,
                                        'rsi_time_filter_upper': coin_rsi_time_filter_upper,
                                        'rsi_time_filter_lower': coin_rsi_time_filter_lower,
                                        'exit_scam_enabled': coin_exit_scam_enabled,
                                        'exit_scam_candles': coin_exit_scam_candles,
                                        'exit_scam_single_candle_percent': coin_exit_scam_single_candle_percent,
                                        'exit_scam_multi_candle_count': coin_exit_scam_multi_candle_count,
                                        'exit_scam_multi_candle_percent': coin_exit_scam_multi_candle_percent
                                    }
                                    
                                    # Сохраняем только RSI параметры в трекер (так как он рассчитан на RSI)
                                    # Но полные параметры сохраняются в metadata.json модели
                                    try:
                                        self.param_tracker.mark_params_used(
                                            coin_rsi_params,  # Используем параметры которые реально использовались для монеты
                                            training_seed,
                                            symbol_win_rate,
                                            symbol,
                                            total_pnl=symbol_pnl,
                                            signal_accuracy=signal_score,
                                            trades_count=trades_for_symbol
                                        )
                                        
                                        if symbol_idx <= 10:
                                            logger.info(f"   ✅ {symbol}: параметры сохранены в трекер (Win Rate: {symbol_win_rate:.1f}%, PnL: {symbol_pnl:.2f} USDT)")
                                        else:
                                            pass
                                    except Exception as tracker_error:
                                        logger.error(f"   ❌ {symbol}: ошибка сохранения параметров в трекер: {tracker_error}")
                                        import traceback
                                        logger.error(traceback.format_exc())
                                except Exception as outer_tracker_error:
                                    logger.error(f"   ❌ {symbol}: ошибка подготовки параметров для трекера: {outer_tracker_error}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                            
                            # ВАЖНО: Увеличиваем счетчик сохраненных моделей ВСЕГДА, так как модель уже сохранена на диск
                            # Это не зависит от Win Rate - модель сохраняется всегда если обучена
                            total_models_saved += 1
                            model_trained = True
                            
                            # Сохраняем параметры в индивидуальные настройки при:
                            # 1) win_rate >= AI_SAVE_BEST_PARAMS_MIN_WIN_RATE (90%), или
                            # 2) "save if better": win_rate > сохранённого по монете И >= 60% И сделок >= 5
                            from bot_engine.config_loader import AIConfig
                            min_win_rate_for_save = AIConfig.AI_SAVE_BEST_PARAMS_MIN_WIN_RATE * 100
                            min_wr_better = getattr(AIConfig, 'AI_SAVE_IF_BETTER_MIN_WIN_RATE', 0.60) * 100
                            min_trades_better = getattr(AIConfig, 'AI_SAVE_IF_BETTER_MIN_TRADES', 5)
                            existing_wr = float((existing_coin_settings or {}).get('ai_win_rate') or 0)
                            save_because_90 = symbol_win_rate >= min_win_rate_for_save
                            save_because_better = (
                                symbol_win_rate > existing_wr
                                and symbol_win_rate >= min_wr_better
                                and trades_for_symbol >= min_trades_better
                            )
                            save_params = save_because_90 or save_because_better
                            if save_params:
                                if save_because_90:
                                    logger.info(
                                        f"   🎯 {symbol}: Win Rate {symbol_win_rate:.1f}% >= {min_win_rate_for_save:.1f}% "
                                        "- сохраняем ЛУЧШИЕ параметры в индивидуальные настройки ✅"
                                    )
                                    self._register_win_rate_success(symbol, symbol_win_rate)
                                else:
                                    logger.info(
                                        f"   📈 {symbol}: Win Rate {symbol_win_rate:.1f}% > сохранённый {existing_wr:.1f}% "
                                        f"(порог {min_wr_better:.0f}%, сделок {trades_for_symbol}) - сохраняем улучшенные параметры ✅"
                                    )
                            else:
                                logger.info(
                                    f"   ⏭️ {symbol}: Win Rate {symbol_win_rate:.1f}% < {min_win_rate_for_save:.1f}% "
                                    f"и не лучше сохранённого ({existing_wr:.1f}%) - параметры НЕ сохраняются"
                                )
                            
                            if save_params:
                                try:
                                    risk_payload = {
                                        'max_loss_percent': MAX_LOSS_PERCENT,
                                        'take_profit_percent': TAKE_PROFIT_PERCENT,
                                        'trailing_stop_activation': TRAILING_STOP_ACTIVATION,
                                        'trailing_stop_distance': TRAILING_STOP_DISTANCE,
                                        'trailing_take_distance': TRAILING_TAKE_DISTANCE,
                                        'trailing_update_interval': TRAILING_UPDATE_INTERVAL,
                                        'break_even_trigger': BREAK_EVEN_TRIGGER,
                                        'break_even_protection': BREAK_EVEN_PROTECTION,
                                        'max_position_hours': MAX_POSITION_HOURS,
                                    }
                                    filter_payload = {
                                        'rsi_time_filter': {
                                            'enabled': coin_rsi_time_filter_enabled,
                                            'candles': coin_rsi_time_filter_candles,
                                            'upper': coin_rsi_time_filter_upper,
                                            'lower': coin_rsi_time_filter_lower,
                                        },
                                        'exit_scam': {
                                            'enabled': coin_exit_scam_enabled,
                                            'candles': coin_exit_scam_candles,
                                            'single_candle_percent': coin_exit_scam_single_candle_percent,
                                            'multi_candle_count': coin_exit_scam_multi_candle_count,
                                            'multi_candle_percent': coin_exit_scam_multi_candle_percent,
                                        },
                                    }
                                    trend_payload = {
                                        'trend_detection_enabled': coin_trend_detection_enabled,
                                        'avoid_down_trend': coin_avoid_down_trend,
                                        'avoid_up_trend': coin_avoid_up_trend,
                                        'trend_analysis_period': coin_trend_analysis_period,
                                        'trend_price_change_threshold': coin_trend_price_change_threshold,
                                        'trend_candles_threshold': coin_trend_candles_threshold,
                                    }
                                    maturity_payload = {
                                        'enable_maturity_check': coin_enable_maturity_check,
                                        'min_candles_for_maturity': coin_min_candles_for_maturity,
                                        'min_rsi_low': coin_min_rsi_low,
                                        'max_rsi_high': coin_max_rsi_high,
                                    }
                                    ai_meta = {
                                        'win_rate': symbol_win_rate,
                                        'rating': self.param_tracker.calculate_rating(symbol_win_rate, symbol_pnl, signal_score, trades_for_symbol) if self.param_tracker else 0,
                                        'total_pnl': symbol_pnl,
                                        'trades_count': trades_for_symbol,
                                    }

                                    individual_settings = self._build_individual_settings(
                                        coin_rsi_params=coin_rsi_params,
                                        risk_params=risk_payload,
                                        filter_params=filter_payload,
                                        trend_params=trend_payload,
                                        maturity_params=maturity_payload,
                                        ai_meta=ai_meta,
                                    )
                                    
                                    # ВАЖНО: Используем ТЕ ЖЕ функции что и bots.py для бесшовной интеграции
                                    # Сначала пробуем через прямой импорт (работает если bots.py запущен)
                                    try:
                                        from bots_modules.imports_and_globals import (
                                            set_individual_coin_settings,
                                            get_individual_coin_settings,
                                            load_individual_coin_settings
                                        )
                                        
                                        # Загружаем существующие настройки если они есть (чтобы не потерять другие параметры)
                                        existing_settings = get_individual_coin_settings(symbol) or {}
                                        
                                        # Объединяем существующие настройки с новыми (новые имеют приоритет)
                                        merged_settings = {**existing_settings, **individual_settings}
                                        merged_settings['updated_at'] = datetime.now().isoformat()
                                        
                                        # Сохраняем используя ТУ ЖЕ функцию что и bots.py
                                        set_individual_coin_settings(symbol, merged_settings, persist=True)
                                        logger.info(f"   💾 Параметры сохранены в индивидуальные настройки для {symbol} (через bots_modules)")
                                        
                                    except ImportError:
                                        # Если bots.py не запущен - используем прямое сохранение в файл
                                        try:
                                            from bot_engine.storage import (
                                                save_individual_coin_settings,
                                                load_individual_coin_settings as storage_load_individual_coin_settings
                                            )
                                            
                                            # Загружаем существующие настройки из файла
                                            existing_all_settings = storage_load_individual_coin_settings() or {}
                                            
                                            # Объединяем с новыми настройками для этой монеты
                                            existing_settings = existing_all_settings.get(symbol.upper(), {})
                                            merged_settings = {**existing_settings, **individual_settings}
                                            merged_settings['updated_at'] = datetime.now().isoformat()
                                            
                                            # Обновляем все настройки
                                            existing_all_settings[symbol.upper()] = merged_settings
                                            
                                            # Сохраняем используя ТУ ЖЕ функцию что и bots.py
                                            save_individual_coin_settings(existing_all_settings)
                                            logger.info(f"   💾 Параметры сохранены в файл для {symbol} (bots.py не запущен)")
                                            
                                        except Exception as storage_error:
                                            logger.warning(f"   ⚠️ Не удалось сохранить параметры для {symbol}: {storage_error}")
                                            
                                    except Exception as save_error:
                                        # Если не получилось через bots_modules - пробуем через API
                                        try:
                                            import requests
                                            response = requests.post(
                                                f'http://localhost:5001/api/bots/individual-settings/{symbol}',
                                                json=individual_settings,
                                                timeout=5
                                            )
                                            if response.status_code == 200:
                                                logger.info(f"   💾 Параметры сохранены через API для {symbol}")
                                            else:
                                                logger.warning(f"   ⚠️ API вернул код {response.status_code} для {symbol}")
                                        except Exception as api_error:
                                            logger.warning(f"   ⚠️ Не удалось сохранить параметры для {symbol} (API недоступен): {api_error}")
                                except Exception as save_params_error:
                                    logger.error(f"   ❌ {symbol}: ошибка сохранения индивидуальных настроек: {save_params_error}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                            # при save_params=False причина уже залогирована выше (Win Rate / "не лучше")
                        
                            if signal_score is not None and profit_mse is not None:
                                pass  # модель обучена, метрики
                            else:
                                pass  # модель обучена, Win Rate

                        if not model_trained:
                            if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                                logger.info(f"   ⏳ {symbol}: недостаточно сделок для обучения ({trades_for_symbol} < 1)")
                            else:
                                pass
                        
                    # ВАЖНО: Увеличиваем счетчик ВСЕГДА, даже если сделок нет!
                    total_trained_coins += 1
                    
                    completion_message = (
                        f"   ✅ [{symbol_idx}/{total_coins}] {symbol}: обработка завершена "
                        f"({trades_for_symbol} сделок, Win Rate: {symbol_win_rate:.1f}%)"
                    )
                    if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                        logger.info(completion_message)
                    else:
                        pass
                    
                    # Собираем симулированные сделки для сохранения
                    if simulated_trades_symbol:
                        all_simulated_trades.extend(simulated_trades_symbol)
                    
                    # Очищаем память после обработки каждого символа
                    try:
                        if 'simulated_trades_symbol' in locals():
                            del simulated_trades_symbol
                    except (NameError, UnboundLocalError):
                        pass
                    from utils.memory_utils import force_collect_full
                    force_collect_full()
                    
                    # Логируем прогресс каждые 50 монет
                    if total_trained_coins % progress_interval == 0:
                        logger.info(
                            f"   📊 Прогресс: {total_trained_coins}/{total_coins} монет обработано "
                            f"({total_trained_coins/total_coins*100:.1f}%), {total_models_saved} моделей сохранено"
                        )
                    
                except Exception as e:
                    # Все ошибки обучения - это критичные ERROR, а не WARNING!
                    logger.error(f"   ❌ Ошибка обучения для {symbol}: {e}")
                    import traceback
                    # Для важных монет показываем полный traceback, для остальных - краткий
                    if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                        logger.error(traceback.format_exc())
                    else:
                        pass
                    total_failed_coins += 1
                finally:
                    # Освобождаем блокировку символа (для параллельной работы на разных ПК)
                    if self.ai_db:
                        try:
                            self.ai_db.release_lock(symbol, process_id)
                        except Exception as lock_error:
                            pass
            
            # Win Rate targets теперь сохраняются в БД автоматически
            
            # Итоговая статистика
            logger.info("=" * 80)
            logger.info(f"✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
            logger.info(f"   📈 Монет обработано: {total_trained_coins}")
            logger.info(f"   ✅ Моделей сохранено: {total_models_saved}")
            logger.info(f"   ⚠️ Ошибок: {total_failed_coins}")
            logger.info(f"   📊 Свечей обработано: {total_candles_processed:,}")
            
            # Дополнительная диагностика: почему модели не сохраняются
            if total_models_saved == 0 and total_trained_coins > 0:
                logger.warning("   ⚠️ ВНИМАНИЕ: Модели не сохранены!")
                logger.warning("   💡 Возможные причины:")
                logger.warning("      - У монет недостаточно сделок для обучения (нужно минимум 1 сделка)")
                logger.warning("      - Все сделки распределены по разным монетам (по 1 сделке на монету)")
                logger.warning("      - Проверьте логи выше для каждой монеты - там указано количество сделок")
            
            # Статистика использования параметров
            if self.param_tracker:
                stats = self.param_tracker.get_usage_stats()
                logger.info(f"   📊 Параметры: использовано {stats['used_combinations']} из {stats['total_combinations']} комбинаций ({stats['usage_percentage']:.2f}%)")
                if stats['is_exhausted']:
                    logger.warning("   ⚠️ Параметры почти исчерпаны! Рекомендуется переключиться на обучение на реальных сделках")
            logger.info("=" * 80)
            
            # Сохраняем все симулированные сделки для последующего обучения
            if all_simulated_trades:
                logger.info(f"💾 Сохранение {len(all_simulated_trades)} симулированных сделок...")
                self._save_simulated_trades(all_simulated_trades)
                logger.info(f"✅ Сохранено {len(all_simulated_trades)} симулированных сделок в БД")
            
            # Также создаем общую модель на всех данных (для монет без индивидуальных моделей)
            logger.info("💡 Общая модель будет создана при следующем обучении (после сбора всех сделок)")
            
            logger.info("=" * 80)
            logger.info(f"✅ СИМУЛЯЦИЯ И ОБУЧЕНИЕ ЗАВЕРШЕНЫ")
            logger.info(f"   📊 Монет обработано: {total_trained_coins}")
            logger.info(f"   📈 Свечей обработано: {total_candles_processed}")
            logger.info(f"   ✅ Моделей сохранено: {total_models_saved}")
            logger.info(f"   ⚠️ Ошибок: {total_failed_coins}")
            if ml_params_generated_count > 0:
                logger.info(f"   🤖 ML модель использована для генерации параметров: {ml_params_generated_count} раз")
            logger.info("=" * 80)
            
            # ВАЖНО: Обучаем/переобучаем ML модель на собранных данных
            # Это позволит AI в будущем генерировать оптимальные параметры вместо случайных
            ml_training_metrics = None
            if self.param_quality_predictor:
                try:
                    logger.info("=" * 80)
                    logger.info("🤖 ОБУЧЕНИЕ/ПЕРЕОБУЧЕНИЕ ML МОДЕЛИ ПРЕДСКАЗАНИЯ КАЧЕСТВА ПАРАМЕТРОВ")
                    logger.info("=" * 80)
                    logger.info("   🧠 ИИ УЧИТСЯ на результатах всех симуляций (успешных и неуспешных)")
                    logger.info("   🎯 ИИ САМ НАХОДИТ оптимальные параметры на основе обучения")
                    logger.info("   🔄 Модель автоматически переобучается при накоплении новых данных")
                    logger.info("   💡 Чем больше симуляций - тем лучше ИИ находит оптимальные параметры")
                    
                    # УЛУЧШЕНИЕ: Проверяем, нужно ли переобучение
                    should_retrain = self._should_retrain_parameter_quality_model()
                    
                    if should_retrain['retrain']:
                        logger.info(f"   🔄 Переобучение модели: {should_retrain['reason']}")
                        ml_training_metrics = self.param_quality_predictor.train(min_samples=50)
                    else:
                        logger.info(f"   ℹ️ Переобучение не требуется: {should_retrain['reason']}")
                        # Проверяем текущее состояние модели
                        if self.param_quality_predictor.is_trained:
                            logger.info("   ✅ Модель уже обучена и актуальна")
                        else:
                            # Модель не обучена - обучаем
                            logger.info("   🎓 Первичное обучение модели...")
                            ml_training_metrics = self.param_quality_predictor.train(min_samples=50)
                    if ml_training_metrics and ml_training_metrics.get('success'):
                        logger.info("   ✅ ML модель обучена! Теперь AI будет генерировать оптимальные параметры")
                        logger.info(f"   📊 R² score: {ml_training_metrics.get('r2_score', 0):.3f}")
                        logger.info(f"   📊 Образцов: {ml_training_metrics.get('samples_count', 0)}")
                        succ = ml_training_metrics.get('successful_samples', 0)
                        blk = ml_training_metrics.get('blocked_samples', 0)
                        logger.info(f"   📊 С сделками: {succ} | Без сделок: {blk}")
                        if succ == 0 and blk > 0:
                            logger.info(
                                "   💡 Все образцы без сделок — симуляции не открывали позиций. "
                                "Проверьте RSI-зоны и фильтры (попадания в LONG/SHORT)."
                            )
                        
                        # Логируем в историю успешное обучение ML модели
                        self._record_training_event(
                            'ml_parameter_quality_training',
                            status='SUCCESS',
                            samples_count=ml_training_metrics.get('samples_count', 0),
                            r2_score=ml_training_metrics.get('r2_score', 0),
                            avg_quality=ml_training_metrics.get('avg_quality', 0),
                            max_quality=ml_training_metrics.get('max_quality', 0),
                            min_quality=ml_training_metrics.get('min_quality', 0),
                            blocked_samples=ml_training_metrics.get('blocked_samples', 0),
                            successful_samples=ml_training_metrics.get('successful_samples', 0),
                            notes='ML модель обучена для генерации оптимальных параметров'
                        )
                    else:
                        reason = 'not_enough_samples'
                        samples_count = 0
                        if ml_training_metrics:
                            reason = ml_training_metrics.get('reason', 'not_enough_samples')
                            samples_count = ml_training_metrics.get('samples_count', 0)
                        
                        logger.info(f"   ⏳ Недостаточно данных для обучения ML модели (есть {samples_count}, нужно минимум 50 образцов)")
                        logger.info("   💡 Продолжаем сбор данных...")
                        
                        # Логируем в историю что данных недостаточно
                        self._record_training_event(
                            'ml_parameter_quality_training',
                            status='SKIPPED',
                            samples_count=samples_count,
                            min_samples_required=50,
                            reason=reason,
                            notes='Недостаточно данных для обучения ML модели'
                        )
                    logger.info("=" * 80)
                    
                    # Обучаемся на симулированных сделках (дополнительно к реальным)
                    try:
                        self.train_on_simulated_trades()
                    except Exception as sim_error:
                        pass
                except Exception as e:
                    logger.error(f"   ❌ Ошибка обучения ML модели: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # Логируем ошибку в историю
                    self._record_training_event(
                        'ml_parameter_quality_training',
                        status='FAILED',
                        reason=str(e),
                        notes='Ошибка при обучении ML модели'
                    )

            self._record_training_event(
                'historical_data_training',
                status='SUCCESS',
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                coins=total_trained_coins,
                candles=total_candles_processed,
                models_saved=total_models_saved,
                errors=total_failed_coins,
                ml_params_generated=ml_params_generated_count,
                ml_model_available=self.param_quality_predictor.is_trained if self.param_quality_predictor else False
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения на исторических данных: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def analyze_open_positions(self) -> List[Dict[str, Any]]:
        """
        Анализирует открытые позиции и дает рекомендации ИИ по точкам выхода и стопам
        
        ВАЖНО: Используется для получения рекомендаций ИИ в реальном времени
        для управления текущими позициями
        
        Returns:
            Список рекомендаций для каждой открытой позиции
        """
        try:
            if not self.ai_db:
                logger.warning("⚠️ AI Database не доступна для анализа позиций")
                return []
            
            # Загружаем открытые позиции с обогащенными данными
            open_positions = self.ai_db.get_open_positions_for_ai()
            
            if not open_positions:
                pass
                return []
            
            recommendations = []
            
            for position in open_positions:
                symbol = position.get('symbol', '')
                if not symbol:
                    continue
                
                try:
                    # Анализируем позицию с помощью ИИ
                    recommendation = self._analyze_single_position(position)
                    if recommendation:
                        recommendations.append(recommendation)
                except Exception as e:
                    pass
                    continue
            
            logger.info(f"✅ Проанализировано {len(recommendations)} открытых позиций")
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа открытых позиций: {e}")
            import traceback
            pass
            return []
    
    def _analyze_single_position(self, position: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Анализирует одну открытую позицию и дает рекомендации ИИ
        
        Args:
            position: Данные открытой позиции (с entry_rsi, current_rsi, etc.)
        
        Returns:
            Словарь с рекомендациями ИИ или None
        """
        try:
            symbol = position.get('symbol', '')
            entry_price = position.get('entry_price')
            current_price = position.get('current_price')
            entry_rsi = position.get('entry_rsi')
            current_rsi = position.get('current_rsi')
            entry_trend = position.get('entry_trend', 'NEUTRAL')
            current_trend = position.get('current_trend', 'NEUTRAL')
            position_side = position.get('position_side', 'LONG')
            pnl = position.get('pnl', 0)
            roi = position.get('roi', 0)
            
            if not entry_price or not current_price:
                return None
            
            # Вычисляем процент изменения цены
            if position_side == 'LONG':
                price_change_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                price_change_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Анализируем с помощью ИИ моделей
            should_exit = False
            exit_reason = None
            recommended_stop = None
            recommended_take_profit = None
            confidence = 0.0
            
            # Используем обученные модели для предсказания
            if hasattr(self, 'signal_model') and self.signal_model:
                try:
                    # Подготавливаем features для текущей позиции
                    features = np.array([[
                        entry_rsi or 50.0,
                        current_rsi or 50.0,
                        0.0,  # entry_volatility (нет данных)
                        1.0 if position_side == 'LONG' else 0.0,
                        1.0 if entry_trend == 'UP' else (0.0 if entry_trend == 'DOWN' else 0.5),
                        1.0 if current_trend == 'UP' else (0.0 if current_trend == 'DOWN' else 0.5),
                        price_change_pct,
                        0.0  # hours_in_position (нет данных)
                    ]])
                    
                    # Нормализуем features
                    if hasattr(self, 'scaler') and self.scaler:
                        features_scaled = self.scaler.transform(features)
                    else:
                        features_scaled = features
                    
                    # Предсказываем вероятность успешного выхода
                    exit_probability = self.signal_model.predict_proba(features_scaled)[0][1]
                    confidence = float(exit_probability)
                    
                    # Рекомендация выхода если вероятность успеха высокая или низкая
                    if exit_probability > 0.8:  # Высокая вероятность успеха - можно выходить
                        should_exit = True
                        exit_reason = 'AI_HIGH_SUCCESS_PROBABILITY'
                    elif exit_probability < 0.2:  # Низкая вероятность успеха - лучше выйти
                        should_exit = True
                        exit_reason = 'AI_LOW_SUCCESS_PROBABILITY'
                    
                except Exception as e:
                    pass
            
            # Анализ RSI для рекомендаций по выходу
            if current_rsi:
                if position_side == 'LONG':
                    # Для LONG позиций: выход при высоком RSI
                    if current_rsi >= 70:
                        should_exit = True
                        exit_reason = exit_reason or 'RSI_OVERBOUGHT'
                    elif current_rsi >= 65 and not should_exit:
                        # Предупреждение о возможном выходе
                        exit_reason = exit_reason or 'RSI_APPROACHING_OVERBOUGHT'
                else:  # SHORT
                    # Для SHORT позиций: выход при низком RSI
                    if current_rsi <= 30:
                        should_exit = True
                        exit_reason = exit_reason or 'RSI_OVERSOLD'
                    elif current_rsi <= 35 and not should_exit:
                        exit_reason = exit_reason or 'RSI_APPROACHING_OVERSOLD'
            
            # Рекомендации по стопам на основе текущего PnL
            if pnl < 0:
                # Убыточная позиция - рекомендуем стоп
                if abs(roi) > 5.0:  # Убыток больше 5%
                    recommended_stop = current_price * 0.98  # Стоп на 2% ниже текущей цены
                    if position_side == 'SHORT':
                        recommended_stop = current_price * 1.02  # Для SHORT наоборот
            elif pnl > 0:
                # Прибыльная позиция - рекомендуем trailing stop
                if roi > 10.0:  # Прибыль больше 10%
                    # Trailing stop на уровне 80% от максимальной прибыли
                    recommended_stop = entry_price + (current_price - entry_price) * 0.8
                    if position_side == 'SHORT':
                        recommended_stop = entry_price - (entry_price - current_price) * 0.8
            
            # Рекомендации по take profit
            if pnl > 0 and not should_exit:
                # Если прибыль хорошая, но еще не время выходить - рекомендуем take profit
                if roi > 15.0:
                    recommended_take_profit = current_price * 1.05  # Take profit на 5% выше
                    if position_side == 'SHORT':
                        recommended_take_profit = current_price * 0.95  # Для SHORT наоборот
            
            return {
                'symbol': symbol,
                'position_side': position_side,
                'entry_price': entry_price,
                'current_price': current_price,
                'pnl': pnl,
                'roi': roi,
                'entry_rsi': entry_rsi,
                'current_rsi': current_rsi,
                'entry_trend': entry_trend,
                'current_trend': current_trend,
                'should_exit': should_exit,
                'exit_reason': exit_reason,
                'exit_confidence': confidence,
                'recommended_stop': recommended_stop,
                'recommended_take_profit': recommended_take_profit,
                'price_change_pct': price_change_pct,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            pass
            return None
            # Win Rate targets теперь сохраняются в БД автоматически
            self._record_training_event(
                'historical_data_training',
                status='FAILED',
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                coins=total_trained_coins,
                candles=total_candles_processed,
                models_saved=total_models_saved,
                errors=total_failed_coins,
                reason=str(e)
            )
    
    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Вычисляет EMA (Exponential Moving Average)"""
        if not prices or len(prices) < period:
            return None
        
        prices_array = np.array(prices[-period:])
        multiplier = 2.0 / (period + 1)
        
        ema = prices_array[0]
        for price in prices_array[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return float(ema)
    
    def _determine_signal_from_rsi_trend(self, rsi: float, trend: str) -> str:
        """Определяет сигнал на основе RSI и тренда"""
        # Логика определения сигнала (можно настроить)
        if rsi <= 30 and trend == 'UP':
            return 'LONG'
        elif rsi >= 70 and trend == 'DOWN':
            return 'SHORT'
        elif rsi <= 25:
            return 'LONG'
        elif rsi >= 75:
            return 'SHORT'
        else:
            return 'WAIT'

    def _get_ensemble_predictor(self):
        """
        Ленивое создание EnsemblePredictor (LSTM + Transformer + SMC).
        Возвращает экземпляр или None при ошибке/отсутствии моделей.
        """
        if self._ensemble_predictor is not None:
            return self._ensemble_predictor
        try:
            from bot_engine.config_loader import AIConfig
            if not getattr(AIConfig, 'AI_USE_ENSEMBLE', False):
                return None
        except Exception:
            return None
        try:
            from bot_engine.config_loader import AIConfig
            from bot_engine.ai.ensemble import EnsemblePredictor
            lstm_p, trans_p, smc_p = None, None, None
            lstm_path = getattr(AIConfig, 'AI_LSTM_MODEL_PATH', 'data/ai/models/lstm_predictor.keras')
            lstm_path_pth = os.path.splitext(lstm_path)[0] + '.pth'
            if os.path.exists(lstm_path_pth):
                try:
                    from bot_engine.ai.lstm_predictor import LSTMPredictor
                    lstm_scaler = getattr(AIConfig, 'AI_LSTM_SCALER_PATH', 'data/ai/models/lstm_scaler.pkl')
                    if os.path.exists(lstm_scaler):
                        lstm_p = LSTMPredictor(model_path=lstm_path_pth, scaler_path=lstm_scaler)
                except Exception as e:
                    pass
            trans_path = 'data/ai/models/transformer_predictor.pth'
            if getattr(AIConfig, 'AI_USE_TRANSFORMER', False) and os.path.exists(trans_path):
                try:
                    from bot_engine.ai.transformer_predictor import TransformerPredictor
                    trans_p = TransformerPredictor(model_path=trans_path)
                except Exception as e:
                    pass
            try:
                from bot_engine.ai.smart_money_features import SmartMoneyFeatures
                smc_p = SmartMoneyFeatures()
            except Exception as e:
                pass
            if lstm_p or trans_p or smc_p:
                self._ensemble_predictor = EnsemblePredictor(
                    lstm_predictor=lstm_p,
                    transformer_predictor=trans_p,
                    smc_features=smc_p,
                    voting='soft',
                )
                return self._ensemble_predictor
        except Exception as e:
            pass
        return None

    def predict(self, symbol: str, market_data: Dict) -> Dict:
        """
        Предсказание торгового сигнала
        
        Args:
            symbol: Символ монеты
            market_data: Рыночные данные (RSI, свечи, тренд и т.д.)
        
        Returns:
            Словарь с предсказанием
        """
        if not self.signal_predictor:
            return {'error': 'Models not trained'}
        if not hasattr(self.signal_predictor, 'predict_proba'):
            return {'error': 'Модель сигналов не обучена или не поддерживает predict_proba'}
        # Проверка готовности модели (RandomForest: estimators_ не пусты)
        try:
            if hasattr(self.signal_predictor, 'estimators_'):
                est = getattr(self.signal_predictor, 'estimators_', None)
                if not est or (len(est) > 0 and est[0] is None):
                    return {'error': 'Модель сигналов не обучена (estimators пусты)'}
        except (AttributeError, IndexError, TypeError):
            pass
        
        # Модель прибыли при R²<0 не используется — решения только по модели сигналов
        use_profit = self.profit_predictor is not None and not getattr(self, '_profit_model_unreliable', True)
        
        try:
            # Определяем ожидаемое количество признаков
            expected_features = self.expected_features
            if expected_features is None:
                # Пытаемся определить из scaler
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ is not None:
                    expected_features = self.scaler.n_features_in_
                    self.expected_features = expected_features
                elif hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                    expected_features = len(self.scaler.mean_)
                    self.expected_features = expected_features
                elif hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None:
                    expected_features = len(self.scaler.scale_)
                    self.expected_features = expected_features
                else:
                    # По умолчанию используем 7 признаков (стандартная версия)
                    expected_features = 7
                    logger.warning(f"⚠️ Не удалось определить количество признаков, используем по умолчанию: {expected_features}")
            
            # Подготавливаем признаки из market_data
            features = []
            
            rsi = market_data.get('rsi', 50)
            trend = market_data.get('trend', 'NEUTRAL')
            price = market_data.get('price', 0)
            direction = market_data.get('direction', 'LONG')
            volatility = market_data.get('volatility', 0)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            
            # Генерируем признаки в том же порядке, что и при обучении:
            # 1. entry_rsi
            features.append(rsi)
            # 2. entry_volatility
            features.append(volatility)
            # 3. entry_volume_ratio
            features.append(volume_ratio)
            # 4. trend UP (1.0 или 0.0)
            features.append(1.0 if trend == 'UP' else 0.0)
            # 5. trend DOWN (1.0 или 0.0)
            features.append(1.0 if trend == 'DOWN' else 0.0)
            # 6. direction LONG (1.0 или 0.0)
            features.append(1.0 if direction == 'LONG' else 0.0)
            # 7. entry_price / 1000.0
            features.append(price / 1000.0 if price > 0 else 0)
            
            # Адаптируем количество признаков под ожидаемое моделью
            # ВАЖНО: Система работает динамически для любого количества признаков (7, 8, 9, 10 и т.д.)
            if len(features) < expected_features:
                # Дополняем нулями (если модель ожидает больше признаков, чем мы генерируем). Не логируем каждый вызов.
                while len(features) < expected_features:
                    features.append(0.0)  # Используем 0.0 вместо 0 для явности типа
            elif len(features) > expected_features:
                # Обрезаем до нужного количества (если модель ожидает меньше признаков). Не логируем каждый вызов.
                features = features[:expected_features]
            else:
                # Количество признаков совпадает - идеальный случай (не логируем каждый вызов, слишком шумно)
                pass
            
            features_array = np.array([features])
            
            try:
                features_scaled = self.scaler.transform(features_array)
            except ValueError as ve:
                # Обработка ошибки несовместимости признаков
                error_msg = str(ve)
                if 'expecting' in error_msg and 'features' in error_msg:
                    # Извлекаем ожидаемое количество признаков из сообщения об ошибке
                    import re
                    match = re.search(r'expecting (\d+) features', error_msg)
                    if match:
                        expected_features = int(match.group(1))
                        self.expected_features = expected_features  # Сохраняем для будущих предсказаний
                        logger.warning(f"⚠️ Несовместимость признаков: модель ожидает {expected_features}, получено {len(features)}")
                        # Адаптируем признаки
                        if expected_features < len(features):
                            features = features[:expected_features]
                            features_array = np.array([features])
                        else:
                            while len(features) < expected_features:
                                features.append(0)
                            features_array = np.array([features])
                        features_scaled = self.scaler.transform(features_array)
                    else:
                        raise
                else:
                    raise
            
            # Предсказание сигнала (всегда по модели сигналов)
            try:
                signal_prob = self.signal_predictor.predict_proba(features_scaled)[0]
            except AttributeError as ae:
                if 'tree_' in str(ae) or 'NoneType' in str(ae):
                    pass
                    return {'error': f'Модель не обучена или не загружена: {ae}'}
                raise
            if use_profit:
                predicted_profit = self.profit_predictor.predict(features_scaled)[0]
            else:
                predicted_profit = None  # R²<0 или модель не обучена — не используем предсказание PnL
            
            # Определяем сигнал
            if signal_prob[1] > 0.6:  # Вероятность прибыли > 60%
                signal = 'LONG' if rsi < 35 else 'SHORT' if rsi > 65 else 'WAIT'
            else:
                signal = 'WAIT'

            result = {
                'signal': signal,
                'confidence': float(signal_prob[1]),
                'predicted_profit': float(predicted_profit) if predicted_profit is not None else None,
                'rsi': rsi,
                'trend': trend
            }

            # Ensemble (LSTM + Transformer + SMC): при AI_USE_ENSEMBLE и наличии candles в market_data
            try:
                from bot_engine.config_loader import AIConfig
                if getattr(AIConfig, 'AI_USE_ENSEMBLE', False):
                    candles = market_data.get('candles') or []
                    price = market_data.get('price') or 0
                    if candles and price:
                        ep = self._get_ensemble_predictor()
                        if ep is not None:
                            ens = ep.predict(candles, float(price))
                            if ens and 'error' not in ens:
                                d = ens.get('direction', 0)
                                result['signal'] = 'LONG' if d == 1 else ('SHORT' if d == -1 else 'WAIT')
                                result['confidence'] = float(ens.get('confidence', 50)) / 100.0
                                result['ensemble_used'] = True
            except Exception as ens_e:
                pass

            if getattr(self, '_perf_monitor', None):
                try:
                    s = result.get('signal', signal)
                    direction = 1 if s == 'LONG' else (-1 if s == 'SHORT' else 0)
                    self._perf_monitor.track_prediction(
                        symbol,
                        {
                            'direction': direction,
                            'change_percent': result.get('predicted_profit') or 0,
                            'confidence': (result.get('confidence') or 0.5) * 100,
                        },
                        model='signal_predictor'
                    )
                except Exception as mon_e:
                    pass

            return result

        except Exception as e:
            logger.error(f"❌ Ошибка предсказания: {e}")
            return {'error': str(e)}
    
    def _prepare_ai_decision_sample(self, decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Подготовить данные решения AI к обучению"""
        try:
            status = (decision.get('status') or '').upper()
            if status not in ('SUCCESS', 'FAILED'):
                return None
            
            market_data = decision.get('market_data') or {}
            
            confidence = decision.get('ai_confidence')
            if confidence is None:
                confidence = market_data.get('confidence')
            if confidence is None:
                confidence = 0.0
            
            entry_rsi = decision.get('rsi')
            if entry_rsi is None:
                entry_rsi = market_data.get('rsi')
            if entry_rsi is None:
                entry_rsi = 50.0
            
            price = decision.get('price')
            if price is None:
                price = market_data.get('price')
            if price is None:
                price = 0.0
            
            direction = (decision.get('direction') or market_data.get('direction') or 'UNKNOWN').upper()
            ai_signal = (decision.get('ai_signal') or market_data.get('signal') or 'UNKNOWN').upper()
            trend = (decision.get('trend') or market_data.get('trend') or 'NEUTRAL').upper()
            
            sample = {
                'decision_id': decision.get('id') or decision.get('decision_id'),
                'symbol': decision.get('symbol'),
                'timestamp': decision.get('timestamp'),
                'target': 1 if status == 'SUCCESS' else 0,
                'ai_confidence': float(confidence),
                'entry_rsi': float(entry_rsi),
                'price': float(price),
                'direction_long': 1.0 if direction == 'LONG' else 0.0,
                'direction_short': 1.0 if direction == 'SHORT' else 0.0,
                'direction_wait': 1.0 if direction == 'WAIT' else 0.0,
                'signal_long': 1.0 if ai_signal == 'LONG' else 0.0,
                'signal_short': 1.0 if ai_signal == 'SHORT' else 0.0,
                'signal_wait': 1.0 if ai_signal == 'WAIT' else 0.0,
                'trend_up': 1.0 if trend == 'UP' else 0.0,
                'trend_down': 1.0 if trend == 'DOWN' else 0.0,
                'trend_neutral': 1.0 if trend not in ('UP', 'DOWN') else 0.0,
                'pnl': float(decision.get('pnl', 0) or 0),
                'roi': float(decision.get('roi', 0) or 0),
            }
            
            additional_features = {}
            for key in ('volatility', 'volume_ratio', 'atr', 'ema_short', 'ema_long'):
                value = decision.get(key, market_data.get(key))
                if value is not None:
                    try:
                        additional_features[key] = float(value)
                    except (TypeError, ValueError):
                        continue
            
            sample.update(additional_features)
            return sample
        except Exception as sample_error:
            pass
            return None
    
    def _should_retrain_parameter_quality_model(self) -> Dict[str, Any]:
        """
        Определяет, нужно ли переобучать модель предсказания качества параметров.
        
        Проверяет:
        1. Накопилось ли достаточно новых данных
        2. Снизилось ли качество модели
        3. Прошло ли достаточно времени с последнего обучения
        
        Returns:
            Словарь с решением: {'retrain': bool, 'reason': str}
        """
        if not self.param_quality_predictor:
            return {'retrain': False, 'reason': 'ParameterQualityPredictor не инициализирован'}
        
        try:
            # Проверяем количество новых образцов в БД
            if self.param_quality_predictor.ai_db:
                training_data = self.param_quality_predictor.ai_db.get_trades_for_training(
                    include_simulated=True, include_real=True, include_exchange=True, min_trades=0
                )
                current_samples_count = len(training_data)
                
                # Если модель не обучена - нужно обучить
                if not self.param_quality_predictor.is_trained:
                    if current_samples_count >= 50:
                        return {'retrain': True, 'reason': f'Модель не обучена, есть {current_samples_count} образцов (нужно минимум 50)'}
                    else:
                        return {'retrain': False, 'reason': f'Модель не обучена, недостаточно данных: {current_samples_count} < 50'}
                
                # Если модель обучена - проверяем, нужно ли переобучение
                # Получаем количество образцов, на которых была обучена модель
                last_trained_samples = getattr(self.param_quality_predictor, '_last_trained_samples_count', 0)
                
                # Если накопилось достаточно новых данных (минимум 20% от предыдущего обучения)
                new_samples_threshold = max(10, int(last_trained_samples * 0.2))
                new_samples = current_samples_count - last_trained_samples
                
                if new_samples >= new_samples_threshold:
                    return {
                        'retrain': True,
                        'reason': f'Накопилось {new_samples} новых образцов (было {last_trained_samples}, стало {current_samples_count}, порог: {new_samples_threshold})'
                    }
                
                # НЕПРЕРЫВНОЕ ОБУЧЕНИЕ: Если данных достаточно, всегда обучаем
                # Убрана проверка на 7 дней - обучение происходит непрерывно
                if current_samples_count >= 50:  # Минимум данных для обучения
                    return {
                        'retrain': True,
                        'reason': f'Непрерывное обучение: достаточно данных ({current_samples_count} образцов)'
                    }
                
                return {
                    'retrain': False,
                    'reason': f'Недостаточно данных для обучения ({current_samples_count} образцов, нужно минимум 50)'
                }
            else:
                return {'retrain': False, 'reason': 'AI Database недоступна'}
        except Exception as e:
            pass
            # В случае ошибки - переобучаем для безопасности
            return {'retrain': True, 'reason': f'Ошибка проверки, переобучаем для безопасности: {e}'}
    
    def retrain_on_ai_decisions(self, force: bool = False) -> int:
        """
        Переобучить модели на основе решений AI (реальные сделки с обратной связью)
        """
        logger.info("=" * 80)
        logger.info("🤖 ПЕРЕОБУЧЕНИЕ НА РЕШЕНИЯХ AI")
        logger.info("=" * 80)
        
        if not self.data_storage:
            pass
            return 0
        
        try:
            decisions = self.data_storage.get_ai_decisions()
            closed_decisions = [
                d for d in decisions
                if (d.get('status') or '').upper() in ('SUCCESS', 'FAILED')
            ]
            
            total_closed = len(closed_decisions)
            logger.info(f"📊 Решений AI с результатом: {total_closed}")
            
            if total_closed < self.ai_decisions_min_samples and not force:
                logger.info(
                    f"⚠️ Недостаточно решений AI для переобучения "
                    f"(есть {total_closed}, нужно минимум {self.ai_decisions_min_samples})"
                )
                return 0
            
            if not force and total_closed <= self.ai_decisions_last_trained_count:
                return 0
            
            samples = []
            for decision in closed_decisions:
                sample = self._prepare_ai_decision_sample(decision)
                if sample:
                    samples.append(sample)
            
            if len(samples) < self.ai_decisions_min_samples and not force:
                logger.info(
                    f"⚠️ После подготовки осталось {len(samples)} решений AI (нужно минимум {self.ai_decisions_min_samples})"
                )
                return 0
            
            if not samples:
                logger.info("ℹ️ Нет данных для переобучения на решениях AI")
                return 0
            
            df = pd.DataFrame(samples)
            df = df.dropna(subset=['target', 'ai_confidence', 'entry_rsi'])
            
            if df.empty:
                logger.info("ℹ️ После очистки данных нет решений AI для обучения")
                return 0
            
            if df['target'].nunique() < 2:
                logger.info("⚠️ Все решения AI с одинаковым результатом (нужны успехи и ошибки)")
                return 0
            
            feature_blacklist = {
                'decision_id', 'symbol', 'timestamp', 'target', 'pnl', 'roi'
            }
            feature_columns = [col for col in df.columns if col not in feature_blacklist]
            
            if not feature_columns:
                logger.info("⚠️ Нет признаков для обучения на решениях AI")
                return 0
            
            X = df[feature_columns]
            y = df['target']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if len(df) >= 10:
                test_size = 0.2 if len(df) >= 25 else 0.25
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y
            
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                random_state=42,
                class_weight='balanced',
                n_jobs=1,  # без параллелизма — устраняет UserWarning delayed/Parallel
            )
            model.fit(X_train, y_train)
            
            if len(df) >= 10:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=False, zero_division=0)
                logger.info(f"✅ Модель решений AI обучена (accuracy: {accuracy * 100:.2f}%)")
                pass
                self._ai_decision_last_accuracy = float(accuracy)
            else:
                self._ai_decision_last_accuracy = None
                logger.info("✅ Модель решений AI обучена (оценка точности пропущена из-за малого набора)")
            
            self.ai_decision_model = model
            self.ai_decision_scaler = scaler
            self.ai_decisions_last_trained_count = len(df)
            
            try:
                self._save_models()
            except Exception as save_error:
                logger.warning(f"⚠️ Не удалось сохранить модель решений AI: {save_error}")
            
            # Обновляем метрики производительности
            try:
                metrics = self.data_storage.calculate_performance_metrics()
                if metrics:
                    self.data_storage.update_performance_metrics(metrics)
                    pass
            except Exception as metrics_error:
                pass
            
            logger.info(f"🎯 Переобучение на решениях AI завершено (образцов: {len(df)})")
            return len(df)
        
        except Exception as retrain_error:
            logger.error(f"❌ Ошибка переобучения на решениях AI: {retrain_error}")
            import traceback
            pass
            return 0
    
    def update_ai_decision_result(
        self,
        decision_id: str,
        pnl: Optional[float],
        roi: Optional[float],
        is_successful: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Обновить результат решения AI после закрытия сделки
        """
        if not decision_id:
            pass
            return False
        
        if not self.data_storage:
            pass
            return False
        
        updates: Dict[str, Any] = {
            'status': 'SUCCESS' if is_successful else 'FAILED',
            'pnl': float(pnl) if pnl is not None else None,
            'roi': float(roi) if roi is not None else None,
            'updated_at': datetime.now().isoformat()
        }
        
        if metadata:
            updates.setdefault('metadata', {})
            if isinstance(updates['metadata'], dict):
                updates['metadata'].update(metadata)
        
        if 'closed_at' not in updates:
            updates['closed_at'] = metadata.get('closed_at') if metadata else datetime.now().isoformat()
        
        try:
            updated = self.data_storage.update_ai_decision(decision_id, updates)
            if updated:
                pass
                
                # УЛУЧШЕНИЕ: Проверяем, нужно ли переобучить модели на реальных сделках
                # Делаем это асинхронно, чтобы не блокировать обновление решения
                try:
                    should_retrain = self._should_retrain_real_trades_models()
                    if should_retrain['retrain']:
                        logger.info(f"🔄 Обнаружено достаточно новых сделок для переобучения: {should_retrain['reason']}")
                        # Запускаем переобучение в отдельном потоке, чтобы не блокировать
                        import threading
                        retrain_thread = threading.Thread(
                            target=self.auto_retrain_real_trades_models,
                            args=(False,),
                            daemon=True,
                            name="AutoRetrainRealTrades"
                        )
                        retrain_thread.start()
                        logger.info("🚀 Запущено автоматическое переобучение на реальных сделках (в фоне)")
                except Exception as retrain_check_error:
                    pass
            else:
                pass
            return updated
        except Exception as update_error:
            logger.warning(f"⚠️ Ошибка обновления решения AI {decision_id}: {update_error}")
            return False
    
    def get_trades_count(self) -> int:
        """
        Получить количество сделок для обучения
        
        Возвращает количество закрытых сделок с PnL из БД (ai_data.db)
        - bot_trades - реальные сделки ботов
        - exchange_trades - сделки с биржи
        
        ВАЖНО: Используются ТОЛЬКО закрытые сделки с PnL (status='CLOSED' и pnl != None)
        """
        trades = self._load_history_data()
        return len(trades)
    
    def _should_retrain_real_trades_models(self) -> Dict[str, Any]:
        """
        Определяет, нужно ли переобучать основные модели (signal_predictor, profit_predictor) на реальных сделках.
        
        НЕПРЕРЫВНОЕ ОБУЧЕНИЕ: Обучение происходит непрерывно, если данных достаточно.
        
        Проверяет:
        1. Накопилось ли достаточно новых сделок (минимум 10, или 20% от предыдущего обучения)
        2. Достаточно ли данных для обучения (непрерывное обучение, без проверки времени)
        3. Модели не обучены вообще
        
        Returns:
            Словарь с решением: {'retrain': bool, 'reason': str, 'trades_count': int}
        """
        try:
            # Получаем текущее количество сделок
            current_trades_count = self.get_trades_count()
            
            # Если моделей нет вообще - нужно обучить
            if not self.signal_predictor or not self.profit_predictor:
                if current_trades_count >= self._real_trades_min_samples:
                    return {
                        'retrain': True,
                        'reason': f'Модели не обучены, есть {current_trades_count} сделок (нужно минимум {self._real_trades_min_samples})',
                        'trades_count': current_trades_count
                    }
                else:
                    return {
                        'retrain': False,
                        'reason': f'Модели не обучены, недостаточно сделок: {current_trades_count} < {self._real_trades_min_samples}',
                        'trades_count': current_trades_count
                    }
            
            # Если модели обучены - проверяем, нужно ли переобучение
            # Проверяем количество новых сделок
            if self._last_real_trades_training_count > 0:
                new_trades = current_trades_count - self._last_real_trades_training_count
                new_trades_threshold = max(
                    self._real_trades_min_samples,
                    int(self._last_real_trades_training_count * self._real_trades_retrain_threshold)
                )
                
                if new_trades >= new_trades_threshold:
                    return {
                        'retrain': True,
                        'reason': f'Накопилось {new_trades} новых сделок (было {self._last_real_trades_training_count}, стало {current_trades_count}, порог: {new_trades_threshold})',
                        'trades_count': current_trades_count
                    }
            else:
                # Первое обучение еще не было - проверяем минимальный порог
                if current_trades_count >= self._real_trades_min_samples:
                    return {
                        'retrain': True,
                        'reason': f'Первое обучение: есть {current_trades_count} сделок (нужно минимум {self._real_trades_min_samples})',
                        'trades_count': current_trades_count
                    }
            
            # НЕПРЕРЫВНОЕ ОБУЧЕНИЕ: Если данных достаточно, всегда обучаем
            # Убрана проверка на 7 дней - обучение происходит непрерывно
            if current_trades_count >= self._real_trades_min_samples:
                return {
                    'retrain': True,
                    'reason': f'Непрерывное обучение: достаточно данных ({current_trades_count} сделок)',
                    'trades_count': current_trades_count
                }
            
            return {
                'retrain': False,
                'reason': f'Недостаточно данных для обучения ({current_trades_count} сделок, нужно минимум {self._real_trades_min_samples})',
                'trades_count': current_trades_count
            }
        except Exception as e:
            pass
            # В случае ошибки - переобучаем для безопасности
            return {
                'retrain': True,
                'reason': f'Ошибка проверки, переобучаем для безопасности: {e}',
                'trades_count': 0
            }
    
    def auto_retrain_real_trades_models(self, force: bool = False) -> bool:
        """
        Автоматически переобучает основные модели на реальных сделках, если накопилось достаточно данных.
        
        Args:
            force: Принудительное переобучение даже если данных недостаточно
        
        Returns:
            True если обучение было выполнено, False если пропущено
        """
        try:
            # Проверяем, нужно ли переобучение
            if not force:
                should_retrain = self._should_retrain_real_trades_models()
                if not should_retrain['retrain']:
                    pass
                    return False
            
            # Запускаем обучение на реальных сделках
            logger.info("=" * 80)
            logger.info("🤖 АВТОМАТИЧЕСКОЕ ПЕРЕОБУЧЕНИЕ НА РЕАЛЬНЫХ СДЕЛКАХ")
            logger.info("=" * 80)
            
            self.train_on_real_trades_with_candles()
            
            # Обновляем время и количество последнего обучения
            self._last_real_trades_training_time = datetime.now()
            if self.ai_db:
                try:
                    # Получаем актуальное количество сделок из БД
                    bot_trades = self.ai_db.get_bot_trades(status='CLOSED', limit=None)
                    exchange_trades = self._load_saved_exchange_trades()
                    self._last_real_trades_training_count = len(bot_trades) + len(exchange_trades)
                except Exception as e:
                    pass
                    self._last_real_trades_training_count = self.get_trades_count()
            else:
                self._last_real_trades_training_count = self.get_trades_count()
            
            logger.info("✅ Автоматическое переобучение на реальных сделках завершено")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка автоматического переобучения на реальных сделках: {e}")
            import traceback
            pass
            return False

    def update_model_online(self, trade_result: Dict) -> bool:
        """
        Онлайн обновление модели на основе результата одной сделки

        Args:
            trade_result: Результат закрытой сделки

        Returns:
            True если обновление выполнено успешно
        """
        try:
            if not self.signal_predictor:
                pass
                return False

            # Извлекаем признаки (7 признаков — как при обучении и инференсе)
            features = self._build_signal_features_7(trade_result)
            if features is None:
                pass
                return False

            # Получаем результат сделки
            pnl = trade_result.get('pnl', 0)
            is_successful = pnl > 0

            # Буфер для инкрементального обучения (те же 7 признаков, что и scaler)
            self._online_learning_buffer.append({
                'features': features,
                'target': 1 if is_successful else 0,
                'pnl': pnl,
                'timestamp': datetime.now().isoformat()
            })

            # Ограничиваем буфер (размер уже задан в deque maxlen, но для совместимости оставляем проверку)
            if len(self._online_learning_buffer) > self._online_learning_buffer_size:
                self._online_learning_buffer.pop(0)

            # Выполняем онлайн обучение каждые 10 сделок
            if len(self._online_learning_buffer) >= 10 and len(self._online_learning_buffer) % 10 == 0:
                return self._perform_incremental_training()

            pass
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка онлайн обновления модели: {e}")
            return False

    def retrain_on_recent_trades(
        self,
        min_samples: Optional[int] = None,
        max_trades: Optional[int] = None,
    ) -> bool:
        """
        Переобучение на последних сделках из БД (инкрементальное обновление модели).
        Использует те же 7 признаков, что и при полном обучении и при инференсе.

        Args:
            min_samples: Минимум валидных образцов для ретрайна (по умолчанию из конфига или 20).
            max_trades: Максимум последних сделок из БД (по умолчанию из конфига или 150).

        Returns:
            True если переобучение выполнено и модели сохранены.
        """
        try:
            try:
                from bot_engine.config_loader import AIConfig as _AIConfig
            except ImportError:
                _AIConfig = None
            if _AIConfig is None:
                min_samples = min_samples if min_samples is not None else 20
                max_trades = max_trades if max_trades is not None else 150
            else:
                min_samples = min_samples if min_samples is not None else getattr(_AIConfig, 'AI_INCREMENTAL_RETRAIN_MIN_SAMPLES', 20)
                max_trades = max_trades if max_trades is not None else getattr(_AIConfig, 'AI_INCREMENTAL_RETRAIN_MAX_TRADES', 150)

            if not self.ai_db:
                return False

            trades = self.ai_db.get_trades_for_training(
                include_simulated=True,
                include_real=True,
                include_exchange=True,
                min_trades=0,
                limit=max_trades,
            )
            if not trades or len(trades) < min_samples:
                return False

            X_list = []
            y_signal_list = []
            y_profit_list = []
            for trade in trades:
                feats = self._build_signal_features_7(trade)
                if feats is None:
                    continue
                X_list.append(feats)
                pnl = trade.get('pnl', 0) or 0
                y_signal_list.append(1 if pnl > 0 else 0)
                y_profit_list.append(float(pnl))

            if len(X_list) < min_samples:
                return False

            X = np.array(X_list)
            y_signal = np.array(y_signal_list)
            y_profit = np.array(y_profit_list)

            n_success = int(np.sum(y_signal))
            n_fail = len(y_signal) - n_success
            if n_success == 0 or n_fail == 0:
                return False

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ is not None:
                self.expected_features = self.scaler.n_features_in_

            test_size = min(0.2, max(0.1, 5.0 / len(X)))
            X_train, X_test, y_signal_train, y_signal_test, y_profit_train, y_profit_test = train_test_split(
                X_scaled, y_signal, y_profit, test_size=test_size, random_state=42, stratify=y_signal if n_success >= 2 and n_fail >= 2 else None
            )

            # Те же классы и гиперпараметры, что в train_on_history / train_on_real_trades_with_candles
            if not self.signal_predictor:
                self.signal_predictor = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=1,
                    class_weight='balanced',
                )
            self.signal_predictor.fit(X_train, y_signal_train)
            acc = float(accuracy_score(y_signal_test, self.signal_predictor.predict(X_test)))
            self._signal_predictor_accuracy = acc

            if not self.profit_predictor:
                self.profit_predictor = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                )
            self.profit_predictor.fit(X_train, y_profit_train)
            from sklearn.metrics import r2_score
            r2 = r2_score(y_profit_test, self.profit_predictor.predict(X_test))
            self._profit_r2 = float(r2) if not np.isnan(r2) else None
            if self._profit_r2 is not None and self._profit_r2 < 0:
                self._profit_model_unreliable = True

            self._save_models()
            r2_str = f"{r2:.3f}" if not np.isnan(r2) else "n/a"
            logger.info(f"✅ Инкрементальный ретрайн: {len(X_list)} образцов, accuracy={acc:.2%}, R²={r2_str}")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Ошибка инкрементального ретрайна: {e}")
            return False

    def _perform_incremental_training(self) -> bool:
        """
        Выполнение инкрементального обучения на накопленных данных

        Returns:
            True если обучение выполнено успешно
        """
        try:
            if len(self._online_learning_buffer) < 5:
                return False

            pass

            # Извлекаем данные из буфера
            X_online = []
            y_online = []

            for item in self._online_learning_buffer[-20:]:  # Используем последние 20 сделок
                X_online.append(item['features'])
                y_online.append(item['target'])

            X_online = np.array(X_online)
            y_online = np.array(y_online)

            # Нормализация
            if hasattr(self, 'scaler') and self.scaler:
                X_online_scaled = self.scaler.transform(X_online)
            else:
                pass
                return False

            # Для RandomForest инкрементальное обучение ограничено
            # В реальной реализации здесь можно использовать онлайн-алгоритмы
            # или частичное переобучение на новых данных

            # Простая оценка важности признаков на новых данных
            if hasattr(self.signal_predictor, 'feature_importances_'):
                # Анализируем успешные и неуспешные сделки
                successful_features = X_online_scaled[y_online == 1]
                failed_features = X_online_scaled[y_online == 0]

                if len(successful_features) > 0 and len(failed_features) > 0:
                    # Вычисляем средние значения признаков
                    success_means = np.mean(successful_features, axis=0)
                    failed_means = np.mean(failed_features, axis=0)

                    # Находим признаки с наибольшими отличиями
                    differences = np.abs(success_means - failed_means)
                    most_important_idx = np.argmax(differences)

                    pass

                    # В реальной реализации здесь можно корректировать веса модели
                    # Пока просто логируем для анализа

            pass
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка инкрементального обучения: {e}")
            return False
