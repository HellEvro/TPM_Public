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
from copy import deepcopy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import joblib

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
    Унифицированный способ получить настройки (глобальные + индивидуальные) даже без запущенного bots.py.
    """
    try:
        from bots_modules.imports_and_globals import get_config_snapshot  # noqa: WPS433,E402

        return get_config_snapshot(symbol)
    except Exception as exc:
        logger.debug(f"⚠️ Не удалось получить конфиг через bots_modules ({exc}), используем запасной путь")
        try:
            from bot_engine.bot_config import DEFAULT_AUTO_BOT_CONFIG  # noqa: WPS433,E402

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
        self.ai_decision_model = None  # Модель для анализа решений AI
        self.ai_decision_scaler = StandardScaler()
        self.ai_decisions_min_samples = 20
        self.ai_decisions_last_trained_count = 0
        self._ai_decision_last_accuracy = None
        # Пути моделей (нормализуем все пути)
        self.signal_model_path = os.path.normpath(os.path.join(self.models_dir, 'signal_predictor.pkl'))
        self.profit_model_path = os.path.normpath(os.path.join(self.models_dir, 'profit_predictor.pkl'))
        self.scaler_path = os.path.normpath(os.path.join(self.models_dir, 'scaler.pkl'))
        self.ai_decision_model_path = os.path.normpath(os.path.join(self.models_dir, 'ai_decision_model.pkl'))
        self.ai_decision_scaler_path = os.path.normpath(os.path.join(self.models_dir, 'ai_decision_scaler.pkl'))

        
        # Файл для отслеживания сделок с AI решениями
        self.ai_decisions_file = os.path.normpath(os.path.join(self.data_dir, 'ai_decisions_tracking.json'))
        
        # Инициализируем хранилище данных AI
        try:
            from bot_engine.ai.ai_data_storage import AIDataStorage
            self.data_storage = AIDataStorage(self.data_dir)
        except Exception as e:
            logger.debug(f"⚠️ Не удалось инициализировать AIDataStorage: {e}")
            self.data_storage = None
        
        # Инициализируем трекер параметров (отслеживание использованных комбинаций)
        try:
            from bot_engine.ai.ai_parameter_tracker import AIParameterTracker
            self.param_tracker = AIParameterTracker(self.data_dir)
        except Exception as e:
            logger.debug(f"⚠️ Не удалось инициализировать AIParameterTracker: {e}")
            self.param_tracker = None
        
        # Целевые значения Win Rate для монет с динамическим повышением порога
        self.win_rate_targets_path = os.path.normpath(os.path.join(self.data_dir, 'win_rate_targets.json'))
        self.win_rate_targets = self._load_win_rate_targets()
        self.win_rate_targets_dirty = False
        
        # Загружаем существующие модели
        self._load_models()
        
        logger.info("✅ AITrainer инициализирован")
    
    def _load_models(self):
        """Загрузить сохраненные модели"""
        try:
            loaded_count = 0
            
            if os.path.exists(self.signal_model_path):
                self.signal_predictor = joblib.load(self.signal_model_path)
                logger.info(f"✅ Загружена модель предсказания сигналов: {self.signal_model_path}")
                loaded_count += 1
                
                # Загружаем метаданные если есть
                metadata_path = os.path.normpath(os.path.join(self.models_dir, 'signal_predictor_metadata.json'))
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            logger.info(f"   📊 Модель обучена: {metadata.get('saved_at', 'unknown')}")
                    except:
                        pass
            else:
                logger.info("ℹ️ Модель предсказания сигналов не найдена (будет создана при обучении)")
            
            if os.path.exists(self.profit_model_path):
                self.profit_predictor = joblib.load(self.profit_model_path)
                logger.info(f"✅ Загружена модель предсказания прибыли: {self.profit_model_path}")
                loaded_count += 1
                
                # Загружаем метаданные если есть
                metadata_path = os.path.normpath(os.path.join(self.models_dir, 'profit_predictor_metadata.json'))
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            logger.info(f"   📊 Модель обучена: {metadata.get('saved_at', 'unknown')}")
                    except:
                        pass
            else:
                logger.info("ℹ️ Модель предсказания прибыли не найдена (будет создана при обучении)")
            
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"✅ Загружен scaler: {self.scaler_path}")
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
                metadata_path = os.path.normpath(os.path.join(self.models_dir, 'signal_predictor_metadata.json'))
                metadata = {
                    'model_type': 'RandomForestClassifier',
                    'saved_at': datetime.now().isoformat(),
                    'n_estimators': getattr(self.signal_predictor, 'n_estimators', 'unknown'),
                    'max_depth': getattr(self.signal_predictor, 'max_depth', 'unknown')
                }
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            if self.profit_predictor:
                joblib.dump(self.profit_predictor, self.profit_model_path)
                logger.info(f"✅ Сохранена модель предсказания прибыли: {self.profit_model_path}")
                saved_count += 1
                
                # Сохраняем метаданные модели
                metadata_path = os.path.normpath(os.path.join(self.models_dir, 'profit_predictor_metadata.json'))
                metadata = {
                    'model_type': 'GradientBoostingRegressor',
                    'saved_at': datetime.now().isoformat(),
                    'n_estimators': getattr(self.profit_predictor, 'n_estimators', 'unknown'),
                    'max_depth': getattr(self.profit_predictor, 'max_depth', 'unknown')
                }
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            if self.scaler:
                joblib.dump(self.scaler, self.scaler_path)
                logger.info(f"✅ Сохранен scaler: {self.scaler_path}")
                saved_count += 1

            if self.ai_decision_model:
                joblib.dump(self.ai_decision_model, self.ai_decision_model_path)
                logger.info(f"✅ Сохранена модель анализа AI решений: {self.ai_decision_model_path}")
                metadata_path = os.path.normpath(os.path.join(self.models_dir, 'ai_decision_model_metadata.json'))
                metadata = {
                    'model_type': type(self.ai_decision_model).__name__,
                    'saved_at': datetime.now().isoformat(),
                    'samples': getattr(self, 'ai_decisions_last_trained_count', 0),
                    'min_samples_required': self.ai_decisions_min_samples
                }
                accuracy = getattr(self, '_ai_decision_last_accuracy', None)
                if accuracy is not None:
                    metadata['accuracy'] = float(accuracy)
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

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
    
    def _load_win_rate_targets(self) -> Dict[str, Any]:
        """
        Загрузить целевые значения Win Rate для монет.
        
        Возвращает словарь формата:
        {
            "default_target": 80.0,
            "symbols": {
                "BTCUSDT": {"target": 84.0, ...},
                ...
            }
        }
        """
        default_data = {'default_target': 80.0, 'symbols': {}}
        try:
            if os.path.exists(self.win_rate_targets_path):
                with open(self.win_rate_targets_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if 'symbols' not in data or not isinstance(data.get('symbols'), dict):
                        data['symbols'] = {}
                    if 'default_target' not in data:
                        data['default_target'] = 80.0
                    return data
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить цели Win Rate: {e}")
        return default_data
    
    def _save_win_rate_targets(self):
        """Сохранить целевые значения Win Rate, если были изменения."""
        try:
            payload = {
                'default_target': float(self.win_rate_targets.get('default_target', 80.0)),
                'symbols': self.win_rate_targets.get('symbols', {}),
                'updated_at': datetime.now().isoformat()
            }
            with open(self.win_rate_targets_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            self.win_rate_targets_dirty = False
        except Exception as e:
            logger.warning(f"⚠️ Не удалось сохранить цели Win Rate: {e}")
    
    def _get_win_rate_target(self, symbol: str) -> float:
        """Получить текущую цель Win Rate для монеты (по умолчанию 80%)."""
        default_target = float(self.win_rate_targets.get('default_target', 80.0))
        symbols = self.win_rate_targets.get('symbols', {})
        entry = symbols.get((symbol or '').upper())
        if isinstance(entry, dict):
            return float(entry.get('target', default_target))
        if isinstance(entry, (int, float)):
            return float(entry)
        return default_target
    
    def _register_win_rate_success(self, symbol: str, achieved_win_rate: float):
        """
        Зафиксировать успешное достижение цели Win Rate и повысить порог на 1%.
        """
        try:
            symbol_key = (symbol or '').upper()
            default_target = float(self.win_rate_targets.get('default_target', 80.0))
            symbols = self.win_rate_targets.setdefault('symbols', {})
            entry = symbols.get(symbol_key)
            if not isinstance(entry, dict):
                entry = {'target': self._get_win_rate_target(symbol_key)}
            
            current_target = float(entry.get('target', default_target))
            entry['last_success_at'] = datetime.now().isoformat()
            entry['last_success_win_rate'] = achieved_win_rate
            entry['achievements'] = entry.get('achievements', 0) + 1
            
            if current_target >= 100.0:
                reset_target = max(default_target, 80.0)
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
            
            symbols[symbol_key] = entry
            self.win_rate_targets_dirty = True
        except Exception as e:
            logger.debug(f"⚠️ Не удалось обновить цель Win Rate для {symbol}: {e}")
    
    def _load_history_data(self) -> List[Dict]:
        """
        Загрузить данные истории трейдов
        
        AI получает сделки из следующих источников:
        1. data/ai/history_data.json - данные собранные через API из bots.py
        2. data/bot_history.json - основной файл где bots.py сохраняет все сделки
        3. API endpoint /api/bots/trades - если файлы недоступны
        
        ВАЖНО: AI использует ТОЛЬКО закрытые сделки с PnL (status='CLOSED' и pnl != None)
        Это нужно для обучения на реальных результатах торговли
        """
        # Сокращенные логи (детали только для DEBUG)
        logger.debug("📊 Загрузка реальных сделок для обучения AI")
        logger.debug("   Источники: history_data.json, bot_history.json, API /api/bots/trades")
        logger.debug("   Используются только закрытые сделки с PnL")
        
        trades = []
        source_counts = {}
        
        # 1. Пробуем загрузить из data/ai/history_data.json (данные собранные через API)
        try:
            history_file = os.path.normpath(os.path.join(self.data_dir, 'history_data.json'))
            if os.path.exists(history_file):
                logger.debug(f"📖 Источник 1: {history_file}")
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Извлекаем все сделки из истории
                latest = data.get('latest', {})
                history = data.get('history', [])
                
                # Добавляем сделки из latest
                latest_trades = latest.get('trades', []) if latest else []
                if latest_trades:
                    trades.extend(latest_trades)
                    logger.debug(f"   ✅ Загружено {len(latest_trades)} сделок из 'latest'")
                
                # Добавляем сделки из истории
                history_trades_count = 0
                for entry in history:
                    entry_trades = entry.get('trades', [])
                    if entry_trades:
                        trades.extend(entry_trades)
                        history_trades_count += len(entry_trades)
                
                if history_trades_count > 0:
                    logger.debug(f"   ✅ Загружено {history_trades_count} сделок из 'history'")
                
                source_counts['history_data.json'] = len(latest_trades) + history_trades_count
            else:
                logger.debug(f"   ⏳ Файл {history_file} не найден")
        except Exception as e:
            logger.warning(f"   ⚠️ Ошибка загрузки history_data.json: {e}")
        
        # 2. Пробуем загрузить напрямую из data/bot_history.json (основной файл bots.py)
        try:
            bot_history_file = os.path.normpath(os.path.join('data', 'bot_history.json'))
            if os.path.exists(bot_history_file):
                logger.debug(f"📖 Источник 2: {bot_history_file}")
                with open(bot_history_file, 'r', encoding='utf-8') as f:
                    bot_history_data = json.load(f)
                
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
                    
                    logger.debug(f"   ✅ Найдено {len(bot_trades)} сделок, добавлено {len(new_trades)} новых")
                    source_counts['bot_history.json'] = len(new_trades)
                else:
                    logger.debug(f"   ⏳ В файле нет сделок")
            else:
                logger.debug(f"   ⏳ Файл {bot_history_file} не найден")
        except json.JSONDecodeError as json_error:
            logger.warning(f"   ⚠️ Файл bot_history.json поврежден (JSON ошибка на позиции {json_error.pos})")
            logger.info("   🗑️ Удаляем поврежденный файл, bots.py пересоздаст его автоматически")
            try:
                os.remove(bot_history_file)
                logger.info("   ✅ Поврежденный файл удален")
            except Exception as del_error:
                logger.debug(f"   ⚠️ Не удалось удалить файл: {del_error}")
        except Exception as e:
            logger.warning(f"   ⚠️ Ошибка загрузки bot_history.json: {e}")
        
        # 3. Анализируем загруженные сделки (сокращенные логи)
        logger.debug(f"📊 Всего загружено сделок: {len(trades)}")
        
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
            
            logger.debug(f"   По статусам: {dict(statuses)}, С PnL: {pnl_count}, Закрытых: {closed_count}")

            closed_trades = [
                t for t in trades
                if t.get('status') == 'CLOSED' and t.get('pnl') is not None
            ]
        else:
            logger.warning("⚠️ Сделки не найдены! Убедитесь что bots.py запущен и совершает сделки.")
            # 4. Фильтруем только закрытые сделки с PnL (пустой список)
            closed_trades = []
            
        logger.info("=" * 80)
        logger.info("✅ РЕЗУЛЬТАТ ФИЛЬТРАЦИИ")
        logger.info("=" * 80)
        logger.info(f"   📊 Всего сделок загружено: {len(trades)}")
        logger.info(f"   ✅ Закрытых сделок с PnL: {len(closed_trades)}")
        logger.info(f"   💡 AI будет обучаться на {len(closed_trades)} реальных сделках")
        
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
    
    def _load_market_data(self) -> Dict:
        """
        Загрузить рыночные данные
        
        ВАЖНО: Использует ТОЛЬКО полную историю свечей из data/ai/candles_full_history.json
        (загруженную через пагинацию по 2000 свечей для каждой монеты)
        
        НЕ использует candles_cache.json - только полная история для качественного обучения!
        """
        try:
            # ВАЖНО: Загружаем ТОЛЬКО из полной истории свечей (data/ai/candles_full_history.json)
            # НЕ используем market_data.json - свечи всегда из candles_full_history.json!
            # НЕ используем candles_cache.json - только полная история!
            # Если файла нет - возвращаем пустые данные (не fallback на кэш!)
            full_history_file = os.path.normpath(os.path.join('data', 'ai', 'candles_full_history.json'))
            market_data = {'latest': {'candles': {}}}
            
            # Загружаем свечи напрямую из полной истории (ВСЕГДА)
            if not os.path.exists(full_history_file):
                logger.error("=" * 80)
                logger.error("❌ ФАЙЛ ПОЛНОЙ ИСТОРИИ СВЕЧЕЙ НЕ НАЙДЕН!")
                logger.error("=" * 80)
                logger.error(f"   📁 Файл: {full_history_file}")
                logger.error("   💡 Файл должен быть загружен через load_full_candles_history()")
                logger.error("   💡 Загрузка запускается автоматически при старте ai.py")
                logger.error("   ⏳ ДОЖДИТЕСЬ пока файл не будет создан и загружен")
                logger.error("   ❌ НЕ используем candles_cache.json - только полная история!")
                logger.error("   ⏸️ Обучение будет пропущено до загрузки файла")
                logger.error("=" * 80)
                return market_data
            
            # Читаем ТОЛЬКО из полной истории свечей
            try:
                logger.info(f"📖 Загрузка полной истории свечей из {full_history_file}...")
                logger.info("   💡 Это файл загружен через пагинацию по 2000 свечей для каждой монеты")
                logger.info("   💡 Содержит ВСЕ доступные свечи для качественного обучения AI")
                logger.info("   ✅ Используем ТОЛЬКО полную историю (не используем candles_cache.json)")
                
                with open(full_history_file, 'r', encoding='utf-8') as f:
                    full_data = json.load(f)
                
                # Извлекаем свечи из структуры с метаданными
                candles_data = {}
                if 'candles' in full_data:
                    candles_data = full_data['candles']
                elif isinstance(full_data, dict) and not full_data.get('metadata'):
                    candles_data = full_data
                else:
                    logger.warning("⚠️ Неожиданная структура файла candles_full_history.json")
                    candles_data = {}
                
                if candles_data:
                    logger.info(f"✅ Загружено полной истории для {len(candles_data)} монет")
                    
                    if 'latest' not in market_data:
                        market_data['latest'] = {}
                    if 'candles' not in market_data['latest']:
                        market_data['latest']['candles'] = {}
                    
                    candles_count = 0
                    total_candles = 0
                    
                    for symbol, candle_info in candles_data.items():
                        candles = candle_info.get('candles', []) if isinstance(candle_info, dict) else []
                        if candles:
                            market_data['latest']['candles'][symbol] = {
                                'candles': candles,
                                'timeframe': candle_info.get('timeframe', '6h') if isinstance(candle_info, dict) else '6h',
                                'last_update': candle_info.get('last_update') or candle_info.get('loaded_at') if isinstance(candle_info, dict) else None,
                                'count': len(candles),
                                'source': 'candles_full_history.json'
                            }
                            candles_count += 1
                            total_candles += len(candles)
                    
                    logger.info(f"✅ Обработано: {candles_count} монет, {total_candles} свечей")
                else:
                    logger.error("=" * 80)
                    logger.error("❌ ФАЙЛ ПОЛНОЙ ИСТОРИИ СВЕЧЕЙ ПУСТ ИЛИ ПОВРЕЖДЕН!")
                    logger.error("=" * 80)
                    logger.error(f"   📁 Файл: {full_history_file}")
                    logger.error("   ⏳ Дождитесь перезагрузки файла через load_full_candles_history()")
                    logger.error("   ⏸️ Обучение будет пропущено до загрузки файла")
                    logger.error("=" * 80)
                    
            except json.JSONDecodeError as json_error:
                logger.error("=" * 80)
                logger.error("❌ ФАЙЛ ПОЛНОЙ ИСТОРИИ СВЕЧЕЙ ПОВРЕЖДЕН!")
                logger.error("=" * 80)
                logger.error(f"   📁 Файл: {full_history_file}")
                logger.error(f"   ⚠️ JSON ошибка на позиции {json_error.pos}")
                logger.error("   🗑️ Удаляем поврежденный файл, он будет пересоздан при следующей загрузке")
                try:
                    os.remove(full_history_file)
                    logger.info("   ✅ Поврежденный файл удален")
                except Exception as del_error:
                    logger.debug(f"   ⚠️ Не удалось удалить файл: {del_error}")
                logger.error("   ⏳ Дождитесь перезагрузки файла через load_full_candles_history()")
                logger.error("   ⏸️ Обучение будет пропущено до загрузки файла")
                logger.error("=" * 80)
            except Exception as e:
                logger.error(f"❌ Ошибка чтения candles_full_history.json: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.error("   ⏸️ Обучение будет пропущено до загрузки файла")
            
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
            
            # Данные входа
            entry_data = trade.get('entry_data', {})
            entry_rsi = entry_data.get('rsi', 50)
            entry_trend = entry_data.get('trend', 'NEUTRAL')
            entry_volatility = entry_data.get('volatility', 0)
            
            # Данные выхода
            exit_market_data = trade.get('exit_market_data', {})
            exit_rsi = exit_market_data.get('rsi', 50)
            exit_trend = exit_market_data.get('trend', 'NEUTRAL')
            
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
    
    def train_on_history(self):
        """
        Обучение на истории трейдов
        """
        logger.info("=" * 80)
        logger.info("🎓 ОБУЧЕНИЕ НА ИСТОРИИ ТРЕЙДОВ")
        logger.info("=" * 80)
        
        try:
            # Загружаем данные
            trades = self._load_history_data()
            
            if len(trades) < 10:
                logger.warning(f"⚠️ Недостаточно данных для обучения (нужно минимум 10, есть {len(trades)})")
                logger.info("💡 Накопите больше сделок для качественного обучения")
                return
            
            logger.info(f"📊 Загружено {len(trades)} сделок для обучения")
            logger.info(f"📈 Анализируем сделки...")
            
            # Подготавливаем данные
            X = []
            y_signal = []  # Сигнал (1 = прибыль, 0 = убыток)
            y_profit = []  # Размер прибыли/убытка
            
            logger.info(f"🔍 Подготовка признаков из {len(trades)} сделок...")
            
            processed = 0
            skipped = 0
            
            for trade in trades:
                features = self._prepare_features(trade)
                if features is None:
                    skipped += 1
                    continue
                
                X.append(features)
                
                pnl = trade.get('pnl', 0)
                y_signal.append(1 if pnl > 0 else 0)
                y_profit.append(pnl)
                
                processed += 1
                
                # Логируем прогресс каждые 20 сделок
                if processed % 20 == 0:
                    logger.info(f"📊 Обработано {processed}/{len(trades)} сделок...")
            
            if skipped > 0:
                logger.info(f"⚠️ Пропущено {skipped} сделок (недостаточно данных)")
            
            if len(X) < 10:
                logger.warning(f"⚠️ Недостаточно валидных данных для обучения ({len(X)} записей)")
                return
            
            logger.info(f"✅ Подготовлено {len(X)} валидных записей для обучения")
            
            X = np.array(X)
            y_signal = np.array(y_signal)
            y_profit = np.array(y_profit)
            
            # Нормализация признаков
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
                n_jobs=-1
            )
            self.signal_predictor.fit(X_train, y_signal_train)
            
            # Оценка модели сигналов
            y_signal_pred = self.signal_predictor.predict(X_test)
            accuracy = accuracy_score(y_signal_test, y_signal_pred)
            
            # Дополнительная статистика
            profitable_pred = sum(y_signal_pred)
            profitable_actual = sum(y_signal_test)
            
            logger.info(f"✅ Модель сигналов обучена!")
            logger.info(f"   📊 Точность: {accuracy:.2%}")
            logger.info(f"   📈 Предсказано прибыльных: {profitable_pred}/{len(y_signal_test)}")
            logger.info(f"   📈 Реально прибыльных: {profitable_actual}/{len(y_signal_test)}")
            
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
            
            avg_profit_actual = np.mean(y_profit_test)
            avg_profit_pred = np.mean(y_profit_pred)
            
            logger.info(f"✅ Модель прибыли обучена!")
            logger.info(f"   📊 MSE: {mse:.2f}")
            logger.info(f"   📈 Средняя прибыль (реальная): {avg_profit_actual:.2f} USDT")
            logger.info(f"   📈 Средняя прибыль (предсказанная): {avg_profit_pred:.2f} USDT")
            
            # Сохранение моделей
            self._save_models()
            
            logger.info("✅ Обучение на истории завершено")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения на истории: {e}")
            import traceback
            traceback.print_exc()
    
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
            
            # Сохраняем результаты анализа
            analysis_file = os.path.normpath(os.path.join(self.models_dir, 'strategy_analysis.json'))
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info("✅ Анализ параметров стратегии завершен")
            logger.info(f"📊 Результаты: {json.dumps(results, indent=2, ensure_ascii=False)}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения на параметрах стратегии: {e}")
    
    def train_on_real_trades_with_candles(self):
        """
        ГЛАВНЫЙ МЕТОД ОБУЧЕНИЯ: Обучается на РЕАЛЬНЫХ СДЕЛКАХ с PnL
        
        Связывает свечи с реальными сделками:
        - Что было на свечах когда открыли позицию (RSI, тренд, волатильность)
        - Что было когда закрыли позицию
        - Реальный PnL сделки
        
        Успешные сделки = положительные примеры для обучения
        Неуспешные сделки = отрицательные примеры для обучения
        """
        logger.info("=" * 80)
        logger.info("🤖 ОБУЧЕНИЕ НА РЕАЛЬНЫХ СДЕЛКАХ С ОБРАТНОЙ СВЯЗЬЮ")
        logger.info("=" * 80)
        
        try:
            # 1. Загружаем реальные сделки с PnL
            trades = self._load_history_data()
            
            if len(trades) < 10:
                logger.warning(f"⚠️ Недостаточно реальных сделок для обучения (есть {len(trades)})")
                logger.info("💡 Накопите больше сделок - AI будет обучаться на вашем опыте!")
                return
            
            logger.info(f"📊 Загружено {len(trades)} реальных сделок с PnL")
            
            # 2. Загружаем свечи для анализа
            market_data = self._load_market_data()
            latest = market_data.get('latest', {})
            candles_data = latest.get('candles', {})
            
            if not candles_data:
                logger.warning("⚠️ Нет свечей для анализа")
                return
            
            logger.info(f"📈 Загружено свечей для {len(candles_data)} монет")
            
            # 3. Связываем сделки со свечами и обучаемся
            successful_samples = []  # Успешные сделки (PnL > 0)
            failed_samples = []      # Неуспешные сделки (PnL <= 0)
            
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
                    pnl = trade.get('pnl', 0)
                    direction = trade.get('direction', 'LONG')
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
                                from datetime import datetime
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
                                from datetime import datetime
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
                        'is_successful': pnl > 0
                    }
                    
                    # Разделяем на успешные и неуспешные
                    if pnl > 0:
                        successful_samples.append(sample)
                    else:
                        failed_samples.append(sample)
                    
                    processed_trades += 1
                    
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка обработки сделки {trade.get('symbol', 'unknown')}: {e}")
                    skipped_trades += 1
                    continue
            
            logger.info(f"✅ Обработано {processed_trades} сделок")
            logger.info(f"   ✅ Успешных: {len(successful_samples)} (PnL > 0)")
            logger.info(f"   ❌ Неуспешных: {len(failed_samples)} (PnL <= 0)")
            logger.info(f"   ⏭️ Пропущено: {skipped_trades}")
            
            # 4. ОБУЧАЕМСЯ НА РЕАЛЬНОМ ОПЫТЕ
            all_samples = successful_samples + failed_samples
            
            if len(all_samples) >= 20:  # Минимум 20 сделок
                logger.info("=" * 80)
                logger.info("🤖 ОБУЧЕНИЕ НЕЙРОСЕТИ НА РЕАЛЬНОМ ОПЫТЕ")
                logger.info("=" * 80)
                
                # Подготавливаем данные
                X = []
                y_signal = []  # 1 = успешная сделка, 0 = неуспешная
                y_profit = []  # Реальный PnL
                
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
                if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
                    from sklearn.preprocessing import StandardScaler
                    self.scaler = StandardScaler()
                    X_scaled = self.scaler.fit_transform(X)
                else:
                    # Переобучение на новых данных (incremental learning)
                    X_scaled = self.scaler.transform(X)
                
                # Обучаем модель предсказания успешности сделок
                if not self.signal_predictor:
                    from sklearn.ensemble import RandomForestClassifier
                    self.signal_predictor = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                        class_weight='balanced'  # Балансировка классов
                    )
                
                logger.info("   📈 Обучение модели на успешных/неуспешных сделках...")
                self.signal_predictor.fit(X_scaled, y_signal)
                
                # Оценка качества
                train_score = self.signal_predictor.score(X_scaled, y_signal)
                logger.info(f"   ✅ Модель обучена! Точность: {train_score:.2%}")
                
                # Статистика по классам
                from collections import Counter
                class_dist = Counter(y_signal)
                logger.info(f"   📊 Распределение: Успешных={class_dist.get(1, 0)}, Неуспешных={class_dist.get(0, 0)}")
                
                # Анализ важности признаков
                if hasattr(self.signal_predictor, 'feature_importances_'):
                    feature_names = ['RSI', 'Volatility', 'Volume Ratio', 'Trend UP', 'Trend DOWN', 'Direction LONG', 'Price']
                    importances = self.signal_predictor.feature_importances_
                    logger.info("   🔍 Важность признаков:")
                    for name, importance in zip(feature_names, importances):
                        logger.info(f"      {name}: {importance:.3f}")
                
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
                self.profit_predictor.fit(X_scaled, y_profit)
                
                # Оценка предсказания прибыли
                profit_pred = self.profit_predictor.predict(X_scaled)
                profit_mse = mean_squared_error(y_profit, profit_pred)
                logger.info(f"   ✅ Модель прибыли обучена! MSE: {profit_mse:.2f}")
                
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
            else:
                logger.warning(f"⚠️ Недостаточно сделок для обучения (нужно минимум 20, есть {len(all_samples)})")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения на реальных сделках: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
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
        import random
        import time as time_module
        
        # Генерируем уникальный seed для этого обучения на основе времени
        training_seed = int(time_module.time() * 1000) % 1000000
        random.seed(training_seed)
        np.random.seed(training_seed)
        
        # Сокращенные логи - только seed для отслеживания
        logger.debug(f"🎲 Seed обучения: {training_seed}")
        
        try:
            # Импортируем ВАШИ настройки из bots.py
            try:
                from bot_engine.bot_config import (
                    RSI_OVERSOLD, RSI_OVERBOUGHT,
                    RSI_EXIT_LONG_WITH_TREND, RSI_EXIT_LONG_AGAINST_TREND,
                    RSI_EXIT_SHORT_WITH_TREND, RSI_EXIT_SHORT_AGAINST_TREND,
                    RSI_PERIOD
                )
                base_rsi_oversold = RSI_OVERSOLD
                base_rsi_overbought = RSI_OVERBOUGHT
                base_exit_long_with = RSI_EXIT_LONG_WITH_TREND
                base_exit_long_against = RSI_EXIT_LONG_AGAINST_TREND
                base_exit_short_with = RSI_EXIT_SHORT_WITH_TREND
                base_exit_short_against = RSI_EXIT_SHORT_AGAINST_TREND
            except ImportError as e:
                logger.warning(f"⚠️ Не удалось загрузить настройки из bot_config.py: {e}")
                # Используем значения по умолчанию
                base_rsi_oversold = 29
                base_rsi_overbought = 71
                base_exit_long_with = 65
                base_exit_long_against = 60
                base_exit_short_with = 35
                base_exit_short_against = 40
                RSI_PERIOD = 14
            
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
                    logger.debug(
                        f"   📚 Доступно ещё {stats['remaining_combinations']:,} комбинаций RSI параметров"
                    )
            else:
                logger.debug("   ⚙️ Трекер параметров недоступен — используем случайные комбинации на монету")

            base_config_snapshot = _get_config_snapshot()
            base_config = base_config_snapshot.get('global', {})

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

            logger.info("🎲 БАЗОВЫЕ ПАРАМЕТРЫ ОБУЧЕНИЯ (индивидуализация на уровне монеты)")

            logger.info("=" * 80)

            logger.info("📊 RSI базовые значения:")

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
            # ВАЖНО: Используем ТОЛЬКО полную историю свечей из candles_full_history.json
            market_data = self._load_market_data()
            
            if not market_data:
                logger.warning("⚠️ Нет рыночных данных для обучения")
                return
            
            latest = market_data.get('latest', {})
            candles_data = latest.get('candles', {})
            
            if not candles_data:
                logger.warning("⚠️ Нет свечей для обучения!")
                logger.info("💡 Файл data/ai/candles_full_history.json не найден или пуст")
                logger.info("💡 Запустите загрузку полной истории свечей через ai.py")
                logger.info("   💡 Это загрузит ВСЕ доступные свечи для всех монет через пагинацию")
                return
            
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
            for symbol_idx, (symbol, candle_info) in enumerate(candles_data.items(), 1):
                # Показываем прогресс каждые 50 монет или для первых 10 монет
                if symbol_idx % progress_interval == 0 or symbol_idx <= 10:
                    logger.info(f"   📈 Прогресс: {symbol_idx}/{total_coins} монет обработано ({symbol_idx/total_coins*100:.1f}%)")
                
                # Логируем начало обработки каждой монеты (первые 10 и каждые 50)
                if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                    logger.info(f"   🎓 [{symbol_idx}/{total_coins}] Начало обработки {symbol}...")
                
                try:
                    candles = candle_info.get('candles', [])
                    coin_seed = training_seed + (abs(hash(symbol)) % 1000)
                    coin_rng = random.Random(coin_seed)
                    if not candles or len(candles) < 100:  # Нужно больше свечей для симуляции
                        if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                            logger.info(f"   ⏭️ {symbol}: пропущено (недостаточно свечей: {len(candles) if candles else 0})")
                        continue
                    
                    # ВАЖНО: Проверяем есть ли лучшие параметры для этой монеты
                    # Если есть и они с высоким рейтингом - используем их вместо случайных
                    coin_best_params = None
                    if self.param_tracker:
                        best_params = self.param_tracker.get_best_params_for_symbol(symbol)
                        if best_params and best_params.get('rating', 0) >= 70.0:  # Используем если рейтинг >= 70
                            coin_best_params = best_params.get('rsi_params')
                            logger.debug(f"   ⭐ {symbol}: используем лучшие параметры (рейтинг {best_params.get('rating', 0):.1f}, Win Rate {best_params.get('win_rate', 0):.1f}%)")
                    
                    # ВАЖНО: Используем ВСЕ свечи, без ограничений!
                    # Проверяем что не обрезаны свечи
                    original_count = len(candles)
                    
                    # Сортируем свечи по времени (от старых к новым)
                    candles = sorted(candles, key=lambda x: x.get('time', 0))
                    
                    # ВАРИАЦИЯ ДАННЫХ: Используем разные подмножества данных для разнообразия
                    # Это обеспечивает разные паттерны при каждом обучении
                    if len(candles) > 500:
                        # Для каждой монеты используем свой offset на основе seed
                        max_offset = min(200, len(candles) - 300)
                        start_offset = coin_rng.randint(0, max_offset) if max_offset > 0 else 0
                        # Используем все свечи от offset до конца (но не меньше 300)
                        min_length = 300
                        if len(candles) - start_offset >= min_length:
                            candles = candles[start_offset:]
                            logger.debug(f"   🎲 {symbol}: используем подмножество свечей с offset {start_offset} (всего {len(candles)} свечей)")

                    # Проверяем существующую модель и количество свечей при предыдущем обучении
                    # Нормализуем путь и имя символа для Windows
                    safe_symbol = symbol.replace('/', '_').replace('\\', '_').replace(':', '_')
                    symbol_models_dir = os.path.normpath(os.path.join(self.models_dir, safe_symbol))
                    metadata_path = os.path.normpath(os.path.join(symbol_models_dir, 'metadata.json'))
                    previous_candles_count = 0
                    model_exists = False
                    
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                existing_metadata = json.load(f)
                            previous_candles_count = existing_metadata.get('candles_count', 0)
                            model_exists = True
                        except Exception as e:
                            logger.debug(f"   ⚠️ Ошибка чтения метаданных модели для {symbol}: {e}")
                    
                    current_candles_count = len(candles)
                    candles_increased = current_candles_count > previous_candles_count
                    increase_percent = ((current_candles_count - previous_candles_count) / previous_candles_count * 100) if previous_candles_count > 0 else 0
                    
                    # Показываем прогресс для каждой монеты (но не слишком часто)
                    if symbol_idx % progress_interval == 0 or symbol_idx == 1 or symbol_idx == total_coins:
                        logger.info(f"   🎓 [{symbol_idx}/{total_coins}] Обработка {symbol}... ({len(candles)} свечей)")
                    else:
                        logger.debug(f"🎓 [{symbol_idx}/{total_coins}] ОБУЧЕНИЕ ДЛЯ {symbol}")
                        logger.debug(f"   📊 Свечей: {len(candles)}")
                    
                    if model_exists:
                        if candles_increased:
                            logger.debug(f"   🔄 Переобучение: {previous_candles_count} → {current_candles_count} (+{increase_percent:.1f}%)")
                        else:
                            logger.debug(f"   ✅ Модель существует, переобучаем на {current_candles_count} свечах")
                    else:
                        logger.debug(f"   🆕 Новая модель на {current_candles_count} свечах")
                    
                    # Предупреждение только если критично
                    if len(candles) <= 1000:
                        logger.debug(f"   ⚠️ {symbol}: только {len(candles)} свечей (возможно кэш)")
                    
                    # Извлекаем данные из свечей
                    closes = [float(c.get('close', 0) or 0) for c in candles]
                    volumes = [float(c.get('volume', 0) or 0) for c in candles]
                    highs = [float(c.get('high', 0) or 0) for c in candles]
                    lows = [float(c.get('low', 0) or 0) for c in candles]
                    opens = [float(c.get('open', 0) or 0) for c in candles]
                    times = [c.get('time', 0) for c in candles]
                    
                    if len(closes) < 100:
                        continue
                    
                    # Готовим индивидуальную базу конфигурации (общий конфиг + индивидуальные настройки монеты)
                    existing_coin_settings = _get_existing_coin_settings(symbol) or {}
                    if existing_coin_settings:
                        logger.debug(f"   🧩 {symbol}: обнаружены индивидуальные настройки, используем их как базу")
                    coin_base_config = base_config.copy() if isinstance(base_config, dict) else {}
                    coin_base_config.update(existing_coin_settings)

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

                    # ВАЖНО: Используем лучшие параметры для монеты если они есть
                    # Иначе используем общие параметры из начала функции
                    if coin_best_params:
                        coin_rsi_params = coin_best_params
                        logger.debug(f"   ⭐ {symbol}: применяем сохранённые лучшие параметры")
                    else:
                        coin_rsi_params = None
                        if self.param_tracker:
                            suggested_params = self.param_tracker.get_unused_params_suggestion(base_params, variation_range)
                            if suggested_params:
                                coin_rsi_params = suggested_params
                                logger.debug(f"   🎯 {symbol}: получили новую комбинацию параметров из трекера")
                        if not coin_rsi_params:
                            exit_variation = 8
                            coin_rsi_params = {
                                'oversold': max(20, min(35, coin_base_rsi_oversold + coin_rng.randint(-variation_range, variation_range))),
                                'overbought': max(65, min(80, coin_base_rsi_overbought + coin_rng.randint(-variation_range, variation_range))),
                                'exit_long_with_trend': max(55, min(70, coin_base_exit_long_with + coin_rng.randint(-exit_variation, exit_variation))),
                                'exit_long_against_trend': max(50, min(65, coin_base_exit_long_against + coin_rng.randint(-exit_variation, exit_variation))),
                                'exit_short_with_trend': max(25, min(40, coin_base_exit_short_with + coin_rng.randint(-exit_variation, exit_variation))),
                                'exit_short_against_trend': max(30, min(45, coin_base_exit_short_against + coin_rng.randint(-exit_variation, exit_variation)))
                            }
                            logger.debug(f"   🎲 {symbol}: сгенерировали уникальные RSI параметры")

                    if symbol_idx <= 5 or symbol_idx % progress_interval == 0:
                        logger.info(f"   ⚙️ {symbol}: RSI params {coin_rsi_params}, seed {coin_seed}")
                    else:
                        logger.debug(f"   ⚙️ {symbol}: RSI params {coin_rsi_params}")

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
                    MAX_POSITION_HOURS = max(12, min(336, coin_base_max_hours + coin_rng.randint(-72, 120)))

                    # Фильтры: RSI временной и ExitScam (индивидуализация на уровне монеты)
                    coin_rsi_time_filter_enabled = bool(coin_base_rsi_time_filter_enabled)
                    coin_rsi_time_filter_candles = max(3, min(30, coin_base_rsi_time_filter_candles + coin_rng.randint(-4, 4)))
                    coin_rsi_time_filter_upper = max(50, min(85, coin_base_rsi_time_filter_upper + coin_rng.randint(-6, 6)))
                    coin_rsi_time_filter_lower = max(15, min(50, coin_base_rsi_time_filter_lower + coin_rng.randint(-6, 6)))
                    if coin_rsi_time_filter_lower >= coin_rsi_time_filter_upper:
                        # Гарантируем корректный диапазон
                        coin_rsi_time_filter_lower = max(15, coin_rsi_time_filter_upper - 1)
                    coin_exit_scam_enabled = bool(coin_base_exit_scam_enabled)
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
                    if coin_rng.random() > 0.7:
                        coin_trend_detection_enabled = not coin_trend_detection_enabled
                    coin_avoid_down_trend = bool(coin_base_avoid_down_trend)
                    if coin_rng.random() > 0.8:
                        coin_avoid_down_trend = not coin_avoid_down_trend
                    coin_avoid_up_trend = bool(coin_base_avoid_up_trend)
                    if coin_rng.random() > 0.8:
                        coin_avoid_up_trend = not coin_avoid_up_trend
                    coin_trend_analysis_period = max(5, min(120, coin_base_trend_analysis_period + coin_rng.randint(-10, 10)))
                    coin_trend_price_change_threshold = max(1.0, min(25.0, coin_base_trend_price_change_threshold + coin_rng.uniform(-3.0, 3.0)))
                    coin_trend_candles_threshold = max(40, min(100, coin_base_trend_candles_threshold + coin_rng.randint(-15, 15)))

                    coin_enable_maturity_check = bool(coin_base_enable_maturity_check)
                    if coin_rng.random() > 0.85:
                        coin_enable_maturity_check = not coin_enable_maturity_check
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
                        logger.debug(
                            f"   📐 {symbol}: SL {MAX_LOSS_PERCENT:.1f}%, TP {TAKE_PROFIT_PERCENT:.1f}%, "
                            f"TS {TRAILING_STOP_ACTIVATION:.1f}%/{TRAILING_STOP_DISTANCE:.1f}%, "
                            f"TT {TRAILING_TAKE_DISTANCE:.2f}%/{TRAILING_UPDATE_INTERVAL:.1f}с, "
                            f"BE {'✅' if BREAK_EVEN_PROTECTION else '❌'} ({BREAK_EVEN_TRIGGER:.1f}%), MaxHold {MAX_POSITION_HOURS}ч"
                        )
                        logger.debug(
                            f"   🛡️ {symbol}: RSI TF {coin_rsi_time_filter_candles} [{coin_rsi_time_filter_lower}/{coin_rsi_time_filter_upper}] | "
                            f"ExitScam: N={coin_exit_scam_candles}, 1св {coin_exit_scam_single_candle_percent:.1f}%, "
                            f"{coin_exit_scam_multi_candle_count}св {coin_exit_scam_multi_candle_percent:.1f}%"
                        )

                    
                    # Вычисляем RSI для КАЖДОЙ свечи
                    rsi_history = calculate_rsi_history_func(candles, period=RSI_PERIOD)
                    
                    if not rsi_history or len(rsi_history) < 50:
                        logger.debug(f"   ⚠️ Недостаточно данных для расчета RSI ({len(rsi_history) if rsi_history else 0})")
                        continue
                    
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
                        logger.info(f"   💵 {symbol}: размер сделки {position_size_usdt:.4f} USDT (режим fixed_usdt)")
                    
                    # Логируем начало симуляции для ВСЕХ монет (INFO для первых 10 и каждых 50)
                    candles_to_process = len(candles) - RSI_PERIOD
                    if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                        logger.info(f"   🔄 {symbol}: симуляция {candles_to_process:,} свечей...")
                    else:
                        logger.debug(f"   🔄 {symbol}: симуляция {candles_to_process:,} свечей...")
                    
                    # Логируем прогресс каждые 1000 свечей (INFO для важных монет)
                    progress_step = 1000
                    
                    for i in range(RSI_PERIOD, len(candles)):
                        # Логируем прогресс каждые 1000 свечей (INFO для важных монет)
                        if candles_to_process > 1000 and (i - RSI_PERIOD) % progress_step == 0:
                            progress_pct = ((i - RSI_PERIOD) / candles_to_process) * 100
                            if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                                logger.info(f"   📊 {symbol}: обработано {i - RSI_PERIOD:,}/{candles_to_process:,} свечей ({progress_pct:.1f}%)")
                            else:
                                logger.debug(f"   📊 {symbol}: обработано {i - RSI_PERIOD:,}/{candles_to_process:,} свечей ({progress_pct:.1f}%)")
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
                                
                                # Используем ВАШИ правила выхода из bot_config.py
                                if direction == 'LONG':
                                    # Определяем был ли вход по тренду или против
                                    if entry_trend == 'UP':
                                        # Вход по тренду - используем WITH_TREND
                                        if current_rsi >= RSI_EXIT_LONG_WITH_TREND:
                                            should_exit = True
                                            exit_reason = 'RSI_EXIT_WITH_TREND'
                                    else:
                                        # Вход против тренда - используем AGAINST_TREND
                                        if current_rsi >= RSI_EXIT_LONG_AGAINST_TREND:
                                            should_exit = True
                                            exit_reason = 'RSI_EXIT_AGAINST_TREND'
                                    
                                    # Стоп-лосс (используем СГЕНЕРИРОВАННЫЕ параметры для этого обучения!)
                                    if current_price <= current_position['entry_price'] * (1 - MAX_LOSS_PERCENT / 100):
                                        should_exit = True
                                        exit_reason = 'STOP_LOSS'
                                    
                                    # Take Profit (используем СГЕНЕРИРОВАННЫЕ параметры!)
                                    if current_price >= current_position['entry_price'] * (1 + TAKE_PROFIT_PERCENT / 100):
                                        should_exit = True
                                        exit_reason = 'TAKE_PROFIT'
                                    
                                    # Trailing Stop (если активирован)
                                    if current_position.get('max_profit', 0) > 0:
                                        max_profit_pct = ((current_position['max_profit'] - current_position['entry_price']) / current_position['entry_price']) * 100
                                        if max_profit_pct >= TRAILING_STOP_ACTIVATION:
                                            # Trailing stop активирован
                                            trailing_stop_price = current_position['entry_price'] * (1 + (max_profit_pct - TRAILING_STOP_DISTANCE) / 100)
                                            if current_price <= trailing_stop_price:
                                                should_exit = True
                                                exit_reason = 'TRAILING_STOP'
                                    
                                    # Break Even Protection
                                    if BREAK_EVEN_PROTECTION and current_position.get('max_profit', 0) > 0:
                                        max_profit_pct = ((current_position['max_profit'] - current_position['entry_price']) / current_position['entry_price']) * 100
                                        if max_profit_pct >= BREAK_EVEN_TRIGGER:
                                            # Break even активирован - защищаем безубыточность
                                            if current_price <= current_position['entry_price']:
                                                should_exit = True
                                                exit_reason = 'BREAK_EVEN'
                                    
                                    # Max Position Hours
                                    if current_position.get('entry_time'):
                                        from datetime import datetime
                                        entry_time = datetime.fromtimestamp(current_position['entry_time'] / 1000)
                                        current_time = datetime.fromtimestamp(times[i] / 1000)
                                        hours_held = (current_time - entry_time).total_seconds() / 3600
                                        if hours_held >= MAX_POSITION_HOURS:
                                            should_exit = True
                                            exit_reason = 'MAX_POSITION_HOURS'
                                
                                elif direction == 'SHORT':
                                    if entry_trend == 'DOWN':
                                        if current_rsi <= RSI_EXIT_SHORT_WITH_TREND:
                                            should_exit = True
                                            exit_reason = 'RSI_EXIT_WITH_TREND'
                                    else:
                                        if current_rsi <= RSI_EXIT_SHORT_AGAINST_TREND:
                                            should_exit = True
                                            exit_reason = 'RSI_EXIT_AGAINST_TREND'
                                    
                                    # Стоп-лосс (используем СГЕНЕРИРОВАННЫЕ параметры для этого обучения!)
                                    if current_price >= current_position['entry_price'] * (1 + MAX_LOSS_PERCENT / 100):
                                        should_exit = True
                                        exit_reason = 'STOP_LOSS'
                                    
                                    # Take Profit (используем СГЕНЕРИРОВАННЫЕ параметры!)
                                    if current_price <= current_position['entry_price'] * (1 - TAKE_PROFIT_PERCENT / 100):
                                        should_exit = True
                                        exit_reason = 'TAKE_PROFIT'
                                    
                                    # Trailing Stop (если активирован)
                                    if current_position.get('max_profit', 0) > 0:
                                        max_profit_pct = ((current_position['entry_price'] - current_position['max_profit']) / current_position['entry_price']) * 100
                                        if max_profit_pct >= TRAILING_STOP_ACTIVATION:
                                            # Trailing stop активирован
                                            trailing_stop_price = current_position['entry_price'] * (1 - (max_profit_pct - TRAILING_STOP_DISTANCE) / 100)
                                            if current_price >= trailing_stop_price:
                                                should_exit = True
                                                exit_reason = 'TRAILING_STOP'
                                    
                                    # Break Even Protection
                                    if BREAK_EVEN_PROTECTION and current_position.get('max_profit', 0) > 0:
                                        max_profit_pct = ((current_position['entry_price'] - current_position['max_profit']) / current_position['entry_price']) * 100
                                        if max_profit_pct >= BREAK_EVEN_TRIGGER:
                                            # Break even активирован - защищаем безубыточность
                                            if current_price >= current_position['entry_price']:
                                                should_exit = True
                                                exit_reason = 'BREAK_EVEN'
                                    
                                    # Max Position Hours
                                    if current_position.get('entry_time'):
                                        from datetime import datetime
                                        entry_time = datetime.fromtimestamp(current_position['entry_time'] / 1000)
                                        current_time = datetime.fromtimestamp(times[i] / 1000)
                                        hours_held = (current_time - entry_time).total_seconds() / 3600
                                        if hours_held >= MAX_POSITION_HOURS:
                                            should_exit = True
                                            exit_reason = 'MAX_POSITION_HOURS'
                                
                                if should_exit:
                                    # Закрываем позицию и записываем результат
                                    entry_price = current_position['entry_price']
                                    if direction == 'LONG':
                                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                                    else:
                                        pnl_pct = ((entry_price - current_price) / entry_price) * 100
                                    
                                    # Симулируем PnL в USDT (используем заранее рассчитанный размер позиции)
                                    pnl_usdt = position_size_usdt * (pnl_pct / 100)
                                    
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
                            
                            # ОБНОВЛЯЕМ max_profit для открытых позиций (для trailing stop и break even)
                            if current_position:
                                if current_position['direction'] == 'LONG':
                                    if current_price > current_position.get('max_profit', current_position['entry_price']):
                                        current_position['max_profit'] = current_price
                                else:  # SHORT
                                    if current_price < current_position.get('max_profit', current_position['entry_price']):
                                        current_position['max_profit'] = current_price
                            
                            # ПРОВЕРКА ВХОДА (если нет открытой позиции)
                            if not current_position:
                                # Используем ВАШИ правила входа из bot_config.py
                                should_enter_long = False
                                should_enter_short = False
                                
                                # LONG: RSI <= RSI_OVERSOLD (используем параметры для монеты)
                                if current_rsi <= coin_RSI_OVERSOLD:
                                    should_enter_long = True
                                    current_position = {
                                        'direction': 'LONG',
                                        'entry_idx': i,
                                        'entry_price': current_price,
                                        'entry_rsi': current_rsi,
                                        'entry_trend': trend,
                                        'entry_time': times[i],
                                        'max_profit': current_price  # Отслеживаем максимальную прибыль для trailing stop
                                    }
                                
                                # SHORT: RSI >= RSI_OVERBOUGHT (используем параметры для монеты)
                                if current_rsi >= coin_RSI_OVERBOUGHT:
                                    should_enter_short = True
                                    current_position = {
                                        'direction': 'SHORT',
                                        'entry_idx': i,
                                        'entry_price': current_price,
                                        'entry_rsi': current_rsi,
                                        'entry_trend': trend,
                                        'entry_time': times[i],
                                        'max_profit': current_price  # Для SHORT это минимум цены (максимальная прибыль)
                                    }
                            
                        except Exception as e:
                            logger.debug(f"   ⚠️ Ошибка симуляции свечи {i} для {symbol}: {e}")
                            continue
                    
                    total_candles_processed += len(candles)
                    
                    # Логируем завершение симуляции (INFO для важных монет)
                    if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                        logger.info(f"   ✅ {symbol}: симуляция завершена ({candles_to_process:,} свечей обработано, {trades_for_symbol} сделок)")
                    elif candles_to_process > 1000:
                        logger.debug(f"   ✅ {symbol}: симуляция завершена ({candles_to_process:,} свечей обработано, {trades_for_symbol} сделок)")
                    
                    # ВАЖНО: Логируем сразу после симуляции для отладки
                    symbol_win_rate = 0.0  # значение по умолчанию, если сделок нет
                    
                    if symbol_idx <= 10:
                        logger.info(f"   🔍 {symbol}: проверка результатов симуляции... (сделок: {trades_for_symbol})")
                    
                    # Логируем результаты симуляции (даже если сделок нет)
                    if trades_for_symbol == 0:
                        if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                            logger.info(f"   ⏭️ {symbol}: сделок не найдено (симуляция завершена)")
                        else:
                            logger.debug(f"   ⏭️ {symbol}: сделок не найдено")
                    else:
                        symbol_successful = sum(1 for t in simulated_trades_symbol if t['is_successful'])
                        symbol_win_rate = symbol_successful / trades_for_symbol * 100
                        symbol_pnl = sum(t['pnl'] for t in simulated_trades_symbol)
                        win_rate_target = self._get_win_rate_target(symbol)
                        
                        if symbol_idx <= 10:
                            logger.info(f"   🎯 {symbol}: текущая цель Win Rate: {win_rate_target:.1f}%")
                        
                        # Показываем результаты для монет с хорошими результатами или при каждом 50-м прогрессе
                        if symbol_win_rate >= win_rate_target or symbol_idx % progress_interval == 0:
                            logger.info(
                                f"   ✅ {symbol}: {trades_for_symbol} сделок, Win Rate: {symbol_win_rate:.1f}% "
                                f"(цель: {win_rate_target:.1f}%), PnL: {symbol_pnl:.2f} USDT"
                            )
                        else:
                            logger.debug(
                                f"   ✅ {symbol}: {trades_for_symbol} сделок, Win Rate: {symbol_win_rate:.1f}% "
                                f"(цель: {win_rate_target:.1f}%), PnL: {symbol_pnl:.2f} USDT"
                            )
                        
                        # ОБУЧАЕМ МОДЕЛЬ ДЛЯ ЭТОЙ МОНЕТЫ ОТДЕЛЬНО
                        signal_score = None
                        profit_mse = None
                        model_trained = False
                        
                        if trades_for_symbol >= 5:  # Минимум 5 сделок для обучения
                            # Показываем начало обучения модели для важных случаев
                            if symbol_win_rate >= win_rate_target or symbol_idx % progress_interval == 0 or symbol_idx <= 10:
                                logger.info(f"   🎓 Обучаем модель для {symbol}... ({trades_for_symbol} сделок, Win Rate: {symbol_win_rate:.1f}%)")
                            else:
                                logger.debug(f"   🎓 Обучаем модель для {symbol}... ({trades_for_symbol} сделок)")
                            
                            # ВАЖНО: Логируем подготовку данных
                            if symbol_idx <= 10:
                                logger.info(f"   📊 {symbol}: подготовка данных для обучения...")
                            
                            # Подготавливаем данные для обучения
                            X_symbol = []
                            y_signal_symbol = []
                            y_profit_symbol = []
                            
                            symbol_trades = simulated_trades_symbol
                            for trade in symbol_trades:
                                features = [
                                    trade['entry_rsi'],
                                    trade['entry_trend'] == 'UP',
                                    trade['entry_trend'] == 'DOWN',
                                    trade['direction'] == 'LONG',
                                    trade['entry_price'] / 1000.0 if trade['entry_price'] > 0 else 0,
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
                                n_jobs=-1,
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
                            metadata_path = os.path.normpath(os.path.join(symbol_models_dir, 'metadata.json'))
                            with open(metadata_path, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2, ensure_ascii=False)
                            logger.debug(f"   🗄️ {symbol}: metadata.json обновлён")
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
                                        logger.info(f"   ✅ {symbol}: параметры сохранены в трекер")
                                        logger.debug(f"   🧾 {symbol}: параметры отмечены в трекере")
                                except Exception as tracker_error:
                                    logger.error(f"   ❌ {symbol}: ошибка сохранения параметров в трекер: {tracker_error}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                            
                            # ВАЖНО: Сохраняем параметры в индивидуальные настройки ТОЛЬКО если Win Rate достиг цели
                            if symbol_win_rate >= win_rate_target:
                                try:
                                    logger.info(
                                        f"   🎯 {symbol}: Win Rate {symbol_win_rate:.1f}% >= цель {win_rate_target:.1f}% "
                                        "- сохраняем параметры в индивидуальные настройки"
                                    )
                                    self._register_win_rate_success(symbol, symbol_win_rate)
                                    
                                    # Формируем настройки в формате для bots.py (используем формат из bot_config.py)
                                    individual_settings = {
                                        'rsi_long_threshold': coin_rsi_params['oversold'],  # Вход в LONG при RSI <=
                                        'rsi_short_threshold': coin_rsi_params['overbought'],  # Вход в SHORT при RSI >=
                                        'rsi_exit_long_with_trend': coin_rsi_params['exit_long_with_trend'],
                                        'rsi_exit_long_against_trend': coin_rsi_params['exit_long_against_trend'],
                                        'rsi_exit_short_with_trend': coin_rsi_params['exit_short_with_trend'],
                                        'rsi_exit_short_against_trend': coin_rsi_params['exit_short_against_trend'],
                                        'max_loss_percent': MAX_LOSS_PERCENT,
                                        'take_profit_percent': TAKE_PROFIT_PERCENT,
                                        'trailing_stop_activation': TRAILING_STOP_ACTIVATION,
                                        'trailing_stop_distance': TRAILING_STOP_DISTANCE,
                                        'trailing_take_distance': TRAILING_TAKE_DISTANCE,
                                        'trailing_update_interval': TRAILING_UPDATE_INTERVAL,
                                        'break_even_trigger': BREAK_EVEN_TRIGGER,
                                        'break_even_protection': BREAK_EVEN_PROTECTION,
                                        'max_position_hours': MAX_POSITION_HOURS,
                                        'rsi_time_filter_enabled': coin_rsi_time_filter_enabled,
                                        'rsi_time_filter_candles': coin_rsi_time_filter_candles,
                                        'rsi_time_filter_upper': coin_rsi_time_filter_upper,
                                        'rsi_time_filter_lower': coin_rsi_time_filter_lower,
                                        'exit_scam_enabled': coin_exit_scam_enabled,
                                        'exit_scam_candles': coin_exit_scam_candles,
                                        'exit_scam_single_candle_percent': coin_exit_scam_single_candle_percent,
                                        'exit_scam_multi_candle_count': coin_exit_scam_multi_candle_count,
                                        'exit_scam_multi_candle_percent': coin_exit_scam_multi_candle_percent,
                                        'trend_detection_enabled': coin_trend_detection_enabled,
                                        'avoid_down_trend': coin_avoid_down_trend,
                                        'avoid_up_trend': coin_avoid_up_trend,
                                        'trend_analysis_period': coin_trend_analysis_period,
                                        'trend_price_change_threshold': coin_trend_price_change_threshold,
                                        'trend_candles_threshold': coin_trend_candles_threshold,
                                        'enable_maturity_check': coin_enable_maturity_check,
                                        'min_candles_for_maturity': coin_min_candles_for_maturity,
                                        'min_rsi_low': coin_min_rsi_low,
                                        'max_rsi_high': coin_max_rsi_high,
                                        'ai_trained': True,
                                        'ai_win_rate': symbol_win_rate,
                                        'ai_rating': self.param_tracker.calculate_rating(symbol_win_rate, symbol_pnl, signal_score, trades_for_symbol) if self.param_tracker else 0,
                                        'ai_trained_at': datetime.now().isoformat(),
                                        'ai_trades_count': trades_for_symbol,
                                        'ai_total_pnl': symbol_pnl
                                    }
                                    
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
                        else:
                            logger.debug(
                                f"   ⏳ {symbol}: Win Rate {symbol_win_rate:.1f}% < цель {win_rate_target:.1f}% "
                                "- параметры НЕ сохраняются в индивидуальные настройки"
                            )
                        
                            # Детальные метрики только для DEBUG
                            if signal_score is not None and profit_mse is not None:
                                logger.debug(
                                    f"   ✅ {symbol}: модель обучена! Accuracy: {signal_score:.2%}, "
                                    f"MSE: {profit_mse:.2f}, Win Rate: {symbol_win_rate:.1f}%"
                                )
                            else:
                                logger.debug(
                                    f"   ✅ {symbol}: модель обучена! Win Rate: {symbol_win_rate:.1f}% "
                                    "(метрики не вычислены)"
                                )
                            total_models_saved += 1
                            model_trained = True

                        if not model_trained:
                            if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                                logger.info(f"   ⏳ {symbol}: недостаточно сделок для обучения ({trades_for_symbol} < 5)")
                            else:
                                logger.debug(f"   ⏳ {symbol}: недостаточно сделок ({trades_for_symbol} < 5)")
                        
                    # ВАЖНО: Увеличиваем счетчик ВСЕГДА, даже если сделок нет!
                    total_trained_coins += 1
                    
                    completion_message = (
                        f"   ✅ [{symbol_idx}/{total_coins}] {symbol}: обработка завершена "
                        f"({trades_for_symbol} сделок, Win Rate: {symbol_win_rate:.1f}%)"
                    )
                    if symbol_idx <= 10 or symbol_idx % progress_interval == 0:
                        logger.info(completion_message)
                    else:
                        logger.debug(completion_message)
                    
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
                        logger.debug(traceback.format_exc())
                    total_failed_coins += 1
                    continue
            
            if self.win_rate_targets_dirty:
                self._save_win_rate_targets()
            
            # Итоговая статистика
            logger.info("=" * 80)
            logger.info(f"✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
            logger.info(f"   📈 Монет обработано: {total_trained_coins}")
            logger.info(f"   ✅ Моделей сохранено: {total_models_saved}")
            logger.info(f"   ⚠️ Ошибок: {total_failed_coins}")
            logger.info(f"   📊 Свечей обработано: {total_candles_processed:,}")
            
            # Статистика использования параметров
            if self.param_tracker:
                stats = self.param_tracker.get_usage_stats()
                logger.info(f"   📊 Параметры: использовано {stats['used_combinations']} из {stats['total_combinations']} комбинаций ({stats['usage_percentage']:.2f}%)")
                if stats['is_exhausted']:
                    logger.warning("   ⚠️ Параметры почти исчерпаны! Рекомендуется переключиться на обучение на реальных сделках")
            logger.info("=" * 80)
            
            # Также создаем общую модель на всех данных (для монет без индивидуальных моделей)
            logger.info("💡 Общая модель будет создана при следующем обучении (после сбора всех сделок)")
            
            logger.info("=" * 80)
            logger.info(f"✅ СИМУЛЯЦИЯ И ОБУЧЕНИЕ ЗАВЕРШЕНЫ")
            logger.info(f"   📊 Монет обработано: {total_trained_coins}")
            logger.info(f"   📈 Свечей обработано: {total_candles_processed}")
            logger.info(f"   ✅ Моделей сохранено: {total_models_saved}")
            logger.info(f"   ⚠️ Ошибок: {total_failed_coins}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения на исторических данных: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if self.win_rate_targets_dirty:
                try:
                    self._save_win_rate_targets()
                except Exception as save_error:
                    logger.debug(f"⚠️ Не удалось сохранить цели Win Rate после ошибки: {save_error}")
    
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
    
    def predict(self, symbol: str, market_data: Dict) -> Dict:
        """
        Предсказание торгового сигнала
        
        Args:
            symbol: Символ монеты
            market_data: Рыночные данные (RSI, свечи, тренд и т.д.)
        
        Returns:
            Словарь с предсказанием
        """
        if not self.signal_predictor or not self.profit_predictor:
            return {'error': 'Models not trained'}
        
        try:
            # Подготавливаем признаки из market_data
            features = []
            
            rsi = market_data.get('rsi', 50)
            trend = market_data.get('trend', 'NEUTRAL')
            price = market_data.get('price', 0)
            
            # Упрощенная подготовка признаков
            features.append(rsi)
            features.append(1 if trend == 'UP' else (0 if trend == 'DOWN' else 0.5))
            features.append(price)
            
            # Добавляем нули для остальных признаков (упрощение)
            while len(features) < 8:
                features.append(0)
            
            features_array = np.array([features])
            features_scaled = self.scaler.transform(features_array)
            
            # Предсказание сигнала
            signal_prob = self.signal_predictor.predict_proba(features_scaled)[0]
            predicted_profit = self.profit_predictor.predict(features_scaled)[0]
            
            # Определяем сигнал
            if signal_prob[1] > 0.6:  # Вероятность прибыли > 60%
                signal = 'LONG' if rsi < 35 else 'SHORT' if rsi > 65 else 'WAIT'
            else:
                signal = 'WAIT'
            
            return {
                'signal': signal,
                'confidence': float(signal_prob[1]),
                'predicted_profit': float(predicted_profit),
                'rsi': rsi,
                'trend': trend
            }
            
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
            logger.debug(f"⚠️ Не удалось подготовить решение AI {decision.get('id')}: {sample_error}")
            return None
    
    def retrain_on_ai_decisions(self, force: bool = False) -> int:
        """
        Переобучить модели на основе решений AI (реальные сделки с обратной связью)
        """
        logger.info("=" * 80)
        logger.info("🤖 ПЕРЕОБУЧЕНИЕ НА РЕШЕНИЯХ AI")
        logger.info("=" * 80)
        
        if not self.data_storage:
            logger.debug("⚠️ AIDataStorage недоступен - пропускаем переобучение на решениях AI")
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
                logger.debug(
                    f"ℹ️ Новых решений AI нет (последнее переобучение на {self.ai_decisions_last_trained_count} решениях)"
                )
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
                class_weight='balanced'
            )
            model.fit(X_train, y_train)
            
            if len(df) >= 10:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=False, zero_division=0)
                logger.info(f"✅ Модель решений AI обучена (accuracy: {accuracy * 100:.2f}%)")
                logger.debug(f"📄 Classification report:\n{report}")
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
                    logger.debug("📊 Метрики производительности AI обновлены")
            except Exception as metrics_error:
                logger.debug(f"⚠️ Не удалось обновить метрики AI решений: {metrics_error}")
            
            logger.info(f"🎯 Переобучение на решениях AI завершено (образцов: {len(df)})")
            return len(df)
        
        except Exception as retrain_error:
            logger.error(f"❌ Ошибка переобучения на решениях AI: {retrain_error}")
            import traceback
            logger.debug(traceback.format_exc())
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
            logger.debug("⚠️ Пустой decision_id для обновления решения AI")
            return False
        
        if not self.data_storage:
            logger.debug("⚠️ AIDataStorage недоступен - не можем обновить решение AI")
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
                logger.debug(f"✅ Решение AI {decision_id} обновлено (pnl={updates.get('pnl')}, roi={updates.get('roi')})")
            else:
                logger.debug(f"⚠️ Решение AI {decision_id} не найдено в хранилище")
            return updated
        except Exception as update_error:
            logger.warning(f"⚠️ Ошибка обновления решения AI {decision_id}: {update_error}")
            return False
    
    def get_trades_count(self) -> int:
        """
        Получить количество сделок для обучения
        
        Возвращает количество закрытых сделок с PnL из:
        - data/bot_history.json (основной файл bots.py)
        - data/ai/history_data.json (данные через API)
        
        ВАЖНО: Используются ТОЛЬКО закрытые сделки с PnL (status='CLOSED' и pnl != None)
        """
        trades = self._load_history_data()
        return len(trades)

