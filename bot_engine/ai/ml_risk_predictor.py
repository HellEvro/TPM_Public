"""
ML Risk Predictor - Машинное обучение для предсказания оптимальных SL/TP

Создает и обучает модель которая:
- Принимает на вход: RSI, волатильность, тренд, историю стопов
- Предсказывает: оптимальный SL%, оптимальный TP%
- Обучается на реальных результатах торговли
- Улучшается с каждым стопом

ТРЕБУЕТ ПРЕМИУМ ЛИЦЕНЗИИ!
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
import pickle

logger = logging.getLogger('AI.MLPredictor')

try:
    import utils.sklearn_parallel_config  # noqa: F401 — до импорта sklearn, подавляет UserWarning delayed/Parallel
except ImportError:
    pass

# Проверяем ML библиотеки
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("[MLPredictor] scikit-learn не установлен. pip install scikit-learn")


class MLRiskPredictor:
    """ML модель для предсказания оптимальных SL/TP"""
    
    def __init__(self):
        """Инициализация ML модели"""
        
        # Проверяем лицензию
        try:
            from bot_engine.ai import check_premium_license
            PREMIUM_AVAILABLE = check_premium_license()
            if not PREMIUM_AVAILABLE:
                raise ImportError("ML модель требует премиум лицензию. Для активации: python scripts/activate_premium.py")
        except ImportError:
            raise ImportError("Не удалось проверить лицензию")
        
        if not ML_AVAILABLE:
            raise ImportError("ML библиотеки недоступны. Установите: pip install scikit-learn")
        
        self.logger = logger
        self.model_sl = None  # Модель для предсказания SL
        self.model_tp = None  # Модель для предсказания TP
        self.model_entry_timing = None  # Модель для предсказания оптимального входа
        self.scaler = StandardScaler()
        self.entry_timing_scaler = StandardScaler()
        
        self.model_path = Path('data/ai/models/risk_predictor.pkl')
        self.scaler_path = Path('data/ai/models/risk_scaler.pkl')
        self.training_data_path = Path('data/ai/training/ml_training_data.json')
        
        # Создаем директории
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.training_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Загружаем модель если есть
        self.load_model()
        
        logger.info("[MLPredictor] ✅ ML модель готова")
    
    def predict(self, features: Dict) -> Dict:
        """
        Предсказывает оптимальные SL/TP
        
        Args:
            features: {
                'rsi': 16.7,
                'volatility': 1.2,
                'trend_strength': 0.68,
                'volume': 1000000,
                'price': 13.9387,
                'coin_stops_count': 5,
                'avg_stop_duration_hours': 12.5
            }
        
        Returns:
            {
                'optimal_sl': 12.5,
                'optimal_tp': 105.3,
                'confidence': 0.75
            }
        """
        try:
            if self.model_sl is None or self.model_tp is None:
                return self._default_prediction()
            
            # Подготавливаем features
            X = self._prepare_features(features)
            
            # Предсказываем
            predicted_sl = self.model_sl.predict([X])[0]
            predicted_tp = self.model_tp.predict([X])[0]
            
            # Ограничиваем диапазоны
            predicted_sl = max(5.0, min(25.0, predicted_sl))
            predicted_tp = max(50.0, min(200.0, predicted_tp))
            
            return {
                'optimal_sl': float(predicted_sl),
                'optimal_tp': float(predicted_tp),
                'confidence': 0.7  # TODO: рассчитывать на основе качества модели
            }
            
        except Exception as e:
            logger.error(f"[MLPredictor] Ошибка предсказания: {e}")
            return self._default_prediction()
    
    def train(self, training_data: List[Dict]):
        """
        Обучает модель на исторических данных
        
        Args:
            training_data: [
                {
                    'features': {'rsi': 16.7, 'volatility': 1.2, ...},
                    'actual_sl': 12.0,  # Реально использованный SL
                    'actual_tp': 100.0,  # Реально использованный TP
                    'actual_roi': -14.99,  # Реальный результат
                    'score': 0.65  # Насколько хорошо сработало
                },
                ...
            ]
        """
        try:
            if not training_data or len(training_data) < 10:
                logger.warning("[MLPredictor] Недостаточно данных для обучения (минимум 10)")
                return False
            
            # Подготавливаем данные
            X = []
            y_sl = []
            y_tp = []
            
            for record in training_data:
                features = record.get('features', {})
                actual_sl = record.get('actual_sl', 15.0)
                actual_tp = record.get('actual_tp', 100.0)
                score = record.get('score', 0.5)
                
                # Используем только хорошие примеры для обучения
                if score > 0.5:
                    X.append(self._prepare_features(features))
                    y_sl.append(actual_sl)
                    y_tp.append(actual_tp)
            
            if len(X) < 10:
                logger.warning(f"[MLPredictor] Недостаточно хороших примеров: {len(X)}")
                return False
            
            X = np.array(X)
            
            # Нормализуем features
            X_scaled = self.scaler.fit_transform(X)
            
            # Обучаем модель для SL
            self.model_sl = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
            self.model_sl.fit(X_scaled, y_sl)
            
            # Обучаем модель для TP
            self.model_tp = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
            self.model_tp.fit(X_scaled, y_tp)
            
            # Сохраняем модель
            self.save_model()
            
            # Рассчитываем метрики качества
            from sklearn.metrics import mean_absolute_error
            
            # Test set для оценки (20% данных)
            if len(X) > 20:
                train_X, test_X, train_y_sl, test_y_sl = train_test_split(X_scaled, y_sl, test_size=0.2, random_state=42)
                train_X, test_X, train_y_tp, test_y_tp = train_test_split(X_scaled, y_tp, test_size=0.2, random_state=42)
                
                pred_sl = self.model_sl.predict(test_X)
                pred_tp = self.model_tp.predict(test_X)
                
                mae_sl = mean_absolute_error(test_y_sl, pred_sl)
                mae_tp = mean_absolute_error(test_y_tp, pred_tp)
                
                # Сохраняем метаданные
                self._save_metadata(len(X), mae_sl, mae_tp)
            
            logger.info(f"[MLPredictor] ✅ Модель обучена на {len(X)} примерах")
            return True
            
        except Exception as e:
            logger.error(f"[MLPredictor] Ошибка обучения: {e}")
            return False
    
    def _prepare_features(self, features: Dict) -> np.ndarray:
        """Подготавливает features для модели"""
        return np.array([
            features.get('rsi', 50.0),
            features.get('volatility', 1.0),
            features.get('trend_strength', 0.5),
            features.get('volume', 0.0) / 1000000,  # Нормализуем
            features.get('price', 0.0) / 100,  # Нормализуем
            features.get('coin_stops_count', 0),
            features.get('avg_stop_duration_hours', 0.0) / 48  # Нормализуем к макс 48ч
        ])
    
    def _default_prediction(self) -> Dict:
        """Возвращает дефолтное предсказание"""
        return {
            'optimal_sl': 15.0,
            'optimal_tp': 100.0,
            'confidence': 0.3
        }
    
    def save_model(self):
        """Сохраняет модель"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({'model_sl': self.model_sl, 'model_tp': self.model_tp}, f)
            
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            logger.info("[MLPredictor] ✅ Модель сохранена")
        except Exception as e:
            logger.error(f"[MLPredictor] Ошибка сохранения: {e}")
    
    def _save_metadata(self, training_samples: int, mae_sl: float, mae_tp: float):
        """Сохраняет метаданные модели"""
        try:
            metadata = {
                "version": "1.0",
                "trained_date": datetime.now().isoformat(),
                "training_samples": training_samples,
                "model_type": "GradientBoostingRegressor",
                "features": ["rsi", "volatility", "trend_strength", "volume", "price", "coin_stops_count", "avg_stop_duration_hours"],
                "mae_sl": float(mae_sl),
                "mae_tp": float(mae_tp),
                "accuracy": max(0.0, min(1.0, 1.0 - mae_sl / 15.0))  # Нормализуем к 0-1
            }
            
            metadata_path = self.model_path.parent / 'model_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"[MLPredictor] 📊 Метаданные: samples={training_samples}, MAE_SL={mae_sl:.2f}%, MAE_TP={mae_tp:.2f}%")
        except Exception as e:
            logger.error(f"[MLPredictor] Ошибка сохранения метаданных: {e}")
    
    def load_model(self):
        """Загружает модель"""
        try:
            if self.model_path.exists() and self.scaler_path.exists():
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model_sl = data['model_sl']
                    self.model_tp = data['model_tp']
                
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                logger.info("[MLPredictor] ✅ Модель загружена")
                return True
            else:
                logger.info("[MLPredictor] ℹ️ Модель не найдена, будет создана при обучении")
                return False
        except Exception as e:
            logger.warning(f"[MLPredictor] Не удалось загрузить модель: {e}")
            return False
    
    def predict_entry_timing(self, features: Dict, side: str) -> Dict:
        """
        🤖 Предсказывает оптимальное время входа
        
        Args:
            features: {
                'rsi': 16.7,
                'volatility': 1.2,
                'trend_strength': 0.68,
                'volume': 1000000,
                'price': 13.9387
            }
            side: 'LONG' или 'SHORT'
        
        Returns:
            {
                'optimal_entry_price': 13.5,
                'wait_minutes': 0,
                'confidence': 0.75
            }
        """
        try:
            if self.model_entry_timing is None:
                # Простая эвристика без ML
                return self._simple_entry_prediction(features, side)
            
            # Подготавливаем features
            X = self._prepare_features(features)
            X_scaled = self.entry_timing_scaler.transform([X])
            
            # Предсказываем оптимальную цену
            optimal_price_diff = self.model_entry_timing.predict(X_scaled)[0]
            
            # Рассчитываем оптимальную цену
            current_price = features.get('price', 1.0)
            
            if side == 'LONG':
                # Для LONG ожидаем падение цены перед входом
                optimal_price = current_price * (1 - abs(optimal_price_diff) / 100)
            else:  # SHORT
                # Для SHORT ожидаем рост цены перед входом
                optimal_price = current_price * (1 + abs(optimal_price_diff) / 100)
            
            # Оцениваем время ожидания
            wait_minutes = min(int(abs(optimal_price_diff) * 5), 30)
            
            # Confidence на основе качества модели
            confidence = 0.6 if abs(optimal_price_diff) < 3.0 else 0.5
            
            return {
                'optimal_entry_price': float(optimal_price),
                'wait_minutes': wait_minutes,
                'confidence': confidence,
                'price_deviation_percent': float(optimal_price_diff)
            }
            
        except Exception as e:
            logger.error(f"[MLPredictor] Ошибка предсказания входа: {e}")
            return self._simple_entry_prediction(features, side)
    
    def _simple_entry_prediction(self, features: Dict, side: str) -> Dict:
        """Простое предсказание без ML (fallback)"""
        current_price = features.get('price', 1.0)
        
        # Простая эвристика: ждем 1-3% отклонения
        if side == 'LONG':
            optimal_price = current_price * 0.98  # На 2% ниже
        else:  # SHORT
            optimal_price = current_price * 1.02  # На 2% выше
        
        return {
            'optimal_entry_price': float(optimal_price),
            'wait_minutes': 10,
            'confidence': 0.5,
            'price_deviation_percent': -2.0 if side == 'LONG' else 2.0
        }

