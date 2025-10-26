"""
LSTM Predictor для предсказания движения цены криптовалют

Этот модуль использует LSTM нейронную сеть для предсказания:
- Направления движения цены (вверх/вниз)
- Ожидаемого изменения цены в %
- Вероятности движения

Используется для улучшения точности входов в сделки.
"""

import os
import json
import pickle
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger('LSTM')

# Отключаем предупреждения TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

# Проверяем доступность TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("[LSTM] TensorFlow не установлен. LSTM Predictor недоступен.")


class LSTMPredictor:
    """
    LSTM модель для предсказания движения цены криптовалют
    """
    
    def __init__(
        self,
        model_path: str = "data/ai/models/lstm_predictor.keras",  # ✅ Keras 3 формат
        scaler_path: str = "data/ai/models/lstm_scaler.pkl",
        config_path: str = "data/ai/models/lstm_config.json"
    ):
        """
        Инициализация LSTM предиктора
        
        Args:
            model_path: Путь к сохраненной модели
            scaler_path: Путь к сохраненному scaler'у
            config_path: Путь к конфигурации модели
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config_path = config_path
        
        self.model = None
        self.scaler = None
        self.config = {
            'sequence_length': 60,  # 60 свечей для предсказания
            'features': ['close', 'volume', 'high', 'low', 'rsi', 'ema_fast', 'ema_slow'],
            'prediction_horizon': 6,  # Предсказание на 6 часов вперед (1 свеча)
            'model_version': '1.0',
            'trained_at': None,
            'training_samples': 0
        }
        
        if not TENSORFLOW_AVAILABLE:
            logger.error("[LSTM] TensorFlow недоступен. Установите: pip install tensorflow")
            return
        
        # Загружаем модель, если существует
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.load_model()
        else:
            logger.info("[LSTM] Модель не найдена, создаем новую")
            self._create_new_model()
    
    def _create_new_model(self):
        """Создает новую LSTM модель"""
        if not TENSORFLOW_AVAILABLE:
            return
        
        sequence_length = self.config['sequence_length']
        n_features = len(self.config['features'])
        
        # Архитектура LSTM модели (современный подход)
        self.model = Sequential([
            # Входной слой
            Input(shape=(sequence_length, n_features)),
            
            # Первый LSTM слой с возвратом последовательностей
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            # Второй LSTM слой
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            # Третий LSTM слой
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            # Полносвязные слои
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            
            # Выходной слой: 3 значения [направление, изменение_%, вероятность]
            Dense(3, activation='linear')
        ])
        
        # Компиляция модели
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Создаем scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        logger.info("[LSTM] ✅ Создана новая модель")
        logger.info(f"[LSTM] Архитектура: {sequence_length} свечей → {n_features} признаков")
    
    def prepare_features(self, candles: List[Dict]) -> np.ndarray:
        """
        Подготавливает признаки из свечей для модели
        
        Args:
            candles: Список свечей с OHLCV данными
        
        Returns:
            Массив признаков для модели
        """
        if len(candles) < self.config['sequence_length']:
            return None
        
        df = pd.DataFrame(candles)
        
        # Извлекаем необходимые признаки
        features = []
        for feature in self.config['features']:
            if feature in df.columns:
                features.append(df[feature].values)
            else:
                # Если признака нет, заполняем нулями
                logger.warning(f"[LSTM] Признак {feature} не найден в данных")
                features.append(np.zeros(len(df)))
        
        # Транспонируем, чтобы получить (samples, features)
        features = np.array(features).T
        
        # Берем последние sequence_length свечей
        features = features[-self.config['sequence_length']:]
        
        return features
    
    def predict(
        self,
        candles: List[Dict],
        current_price: float
    ) -> Optional[Dict]:
        """
        Предсказывает движение цены
        
        Args:
            candles: История свечей для анализа
            current_price: Текущая цена
        
        Returns:
            {
                'direction': 1 (вверх) или -1 (вниз),
                'change_percent': ожидаемое изменение в %,
                'confidence': уверенность модели (0-100),
                'predicted_price': предсказанная цена,
                'horizon_hours': горизонт предсказания в часах
            }
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return None
        
        try:
            # Подготавливаем признаки
            features = self.prepare_features(candles)
            if features is None:
                return None
            
            # Нормализуем данные
            features_scaled = self.scaler.transform(features)
            
            # Добавляем batch dimension
            features_scaled = features_scaled.reshape(1, self.config['sequence_length'], -1)
            
            # Предсказание
            prediction = self.model.predict(features_scaled, verbose=0)[0]
            
            # Распаковываем результат
            direction_raw = prediction[0]  # -1 до 1
            change_percent = prediction[1]  # % изменения
            confidence = prediction[2]  # 0-1
            
            # Определяем направление
            direction = 1 if direction_raw > 0 else -1
            
            # Нормализуем уверенность
            confidence = min(max(abs(confidence) * 100, 0), 100)
            
            # Вычисляем предсказанную цену
            predicted_price = current_price * (1 + change_percent / 100)
            
            result = {
                'direction': direction,
                'change_percent': float(change_percent),
                'confidence': float(confidence),
                'predicted_price': float(predicted_price),
                'horizon_hours': self.config['prediction_horizon'],
                'current_price': current_price
            }
            
            return result
            
        except Exception as e:
            logger.error(f"[LSTM] Ошибка предсказания: {e}")
            return None
    
    def train(
        self,
        training_data: List[Tuple[np.ndarray, np.ndarray]],
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict:
        """
        Обучает LSTM модель
        
        Args:
            training_data: Список (X, y) где X - признаки, y - целевые значения
            validation_split: Доля данных для валидации
            epochs: Количество эпох обучения
            batch_size: Размер батча
        
        Returns:
            История обучения
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return {'error': 'TensorFlow unavailable'}
        
        try:
            # Объединяем данные
            X_list, y_list = zip(*training_data)
            X = np.array(X_list)
            y = np.array(y_list)
            
            logger.info(f"[LSTM] Начало обучения: {len(X)} образцов")
            logger.info(f"[LSTM] Форма X: {X.shape}, форма y: {y.shape}")
            
            # Настраиваем callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001
                )
            ]
            
            # Обучаем модель
            history = self.model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Обновляем конфигурацию
            self.config['trained_at'] = datetime.now().isoformat()
            self.config['training_samples'] = len(X)
            
            # Сохраняем модель
            self.save_model()
            
            logger.info("[LSTM] ✅ Обучение завершено успешно")
            
            return {
                'success': True,
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'epochs_trained': len(history.history['loss']),
                'training_samples': len(X)
            }
            
        except Exception as e:
            logger.error(f"[LSTM] ❌ Ошибка обучения: {e}")
            return {'error': str(e)}
    
    def save_model(self):
        """Сохраняет модель, scaler и конфигурацию"""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return
        
        try:
            # Создаем директорию, если не существует
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Сохраняем модель
            self.model.save(self.model_path)
            
            # Сохраняем scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Сохраняем конфигурацию
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"[LSTM] ✅ Модель сохранена: {self.model_path}")
            
        except Exception as e:
            logger.error(f"[LSTM] ❌ Ошибка сохранения модели: {e}")
    
    def load_model(self):
        """Загружает модель, scaler и конфигурацию"""
        if not TENSORFLOW_AVAILABLE:
            return
        
        try:
            # Загружаем модель
            self.model = load_model(self.model_path)
            
            # Загружаем scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Загружаем конфигурацию
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config.update(json.load(f))
            
            logger.info(f"[LSTM] ✅ Модель загружена: {self.model_path}")
            logger.info(f"[LSTM] Обучена: {self.config.get('trained_at', 'неизвестно')}")
            logger.info(f"[LSTM] Образцов: {self.config.get('training_samples', 0)}")
            
        except Exception as e:
            logger.error(f"[LSTM] ❌ Ошибка загрузки модели: {e}")
            self._create_new_model()
    
    def get_status(self) -> Dict:
        """Возвращает статус модели"""
        if not TENSORFLOW_AVAILABLE:
            return {
                'available': False,
                'error': 'TensorFlow not installed'
            }
        
        is_trained = (
            self.model is not None and
            os.path.exists(self.model_path) and
            self.config.get('training_samples', 0) > 0
        )
        
        return {
            'available': True,
            'trained': is_trained,
            'model_path': self.model_path,
            'sequence_length': self.config['sequence_length'],
            'prediction_horizon': self.config['prediction_horizon'],
            'trained_at': self.config.get('trained_at'),
            'training_samples': self.config.get('training_samples', 0),
            'features': self.config['features']
        }

