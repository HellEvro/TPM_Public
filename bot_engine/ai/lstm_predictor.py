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

try:
    from sklearn.exceptions import NotFittedError
except ImportError:  # pragma: no cover - fallback если scikit-learn не установлен
    class NotFittedError(Exception):
        """Локальный NotFittedError, если scikit-learn недоступен"""
        pass

logger = logging.getLogger('LSTM')

# Отключаем предупреждения TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

# Отключаем DEBUG логи TensorFlow через Python logging
tensorflow_logger = logging.getLogger('tensorflow')
tensorflow_logger.setLevel(logging.WARNING)
tensorflow_python_logger = logging.getLogger('tensorflow.python')
tensorflow_python_logger.setLevel(logging.WARNING)
tensorflow_core_logger = logging.getLogger('tensorflow.core')
tensorflow_core_logger.setLevel(logging.WARNING)
# Удаляем все обработчики из TensorFlow логгеров
for handler in tensorflow_logger.handlers[:]:
    tensorflow_logger.removeHandler(handler)
for handler in tensorflow_python_logger.handlers[:]:
    tensorflow_python_logger.removeHandler(handler)
for handler in tensorflow_core_logger.handlers[:]:
    tensorflow_core_logger.removeHandler(handler)

# Проверяем доступность TensorFlow
try:
    import tensorflow as tf
    # Дополнительно отключаем логирование после импорта
    tf.get_logger().setLevel(logging.WARNING)
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
    
    # Настройка GPU для TensorFlow
    def configure_gpu():
        """Настраивает TensorFlow для использования GPU NVIDIA (если доступен)"""
        try:
            # Проверяем доступность GPU
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                try:
                    # В TensorFlow с CUDA обычно все найденные GPU - это NVIDIA
                    # Используем первый доступный GPU
                    primary_gpu = gpus[0]
                    
                    # Включаем рост памяти GPU по мере необходимости
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    logger.info(f"✅ Найдено GPU устройств: {len(gpus)}")
                    for i, gpu in enumerate(gpus):
                        logger.info(f"   GPU {i}: {gpu.name}")
                    
                    # Проверяем, что GPU действительно доступен для вычислений
                    # TensorFlow автоматически будет использовать GPU для всех операций
                    logger.info(f"✅ GPU NVIDIA доступен и будет использоваться для обучения и предсказаний")
                    logger.info(f"   Основное устройство: {primary_gpu.name}")
                    
                    return True, primary_gpu
                except RuntimeError as e:
                    logger.warning(f"⚠️ Ошибка настройки GPU: {e}")
                    logger.info("Продолжаем с CPU...")
                    return False, None
            else:
                logger.info("ℹ️ GPU устройства не найдены, используется CPU")
                return False, None
        except Exception as e:
            logger.warning(f"⚠️ Ошибка проверки GPU: {e}")
            logger.info("Продолжаем с CPU...")
            return False, None
    
    # Настраиваем GPU при импорте модуля
    GPU_AVAILABLE, GPU_DEVICE = configure_gpu()
    
except ImportError:
    TENSORFLOW_AVAILABLE = False
    GPU_AVAILABLE = False
    GPU_DEVICE = None
    logger.warning("TensorFlow не установлен. LSTM Predictor недоступен.")


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
            logger.error("TensorFlow недоступен. Установите: pip install tensorflow")
            return
        
        # Выводим информацию о GPU при инициализации
        if TENSORFLOW_AVAILABLE:
            if GPU_AVAILABLE and GPU_DEVICE:
                logger.info(f"🚀 LSTM Predictor инициализирован с поддержкой GPU NVIDIA: {GPU_DEVICE.name}")
            else:
                logger.info("💻 LSTM Predictor инициализирован (CPU режим)")
        
        # Загружаем модель, если существует
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.load_model()
        else:
            logger.info("Модель не найдена, создаем новую")
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
        
        logger.info("✅ Создана новая модель")
        logger.info(f"Архитектура: {sequence_length} свечей → {n_features} признаков")
    
    def prepare_features(self, candles: List[Dict]) -> np.ndarray:
        """
        Подготавливает признаки из свечей для модели
        
        Args:
            candles: Список свечей с OHLCV данными
        
        Returns:
            Массив признаков для модели
        """
        if len(candles) < self.config['sequence_length']:
            logger.debug(f"Недостаточно свечей: {len(candles)} < {self.config['sequence_length']}")
            return None
        
        df = pd.DataFrame(candles)
        
        # Извлекаем необходимые признаки
        features = []
        for feature in self.config['features']:
            if feature in df.columns:
                features.append(df[feature].values)
            else:
                # Если признака нет, заполняем нулями
                logger.warning(f"Признак {feature} не найден в данных")
                features.append(np.zeros(len(df)))
        
        # Транспонируем, чтобы получить (samples, features)
        features = np.array(features).T
        
        # Берем последние sequence_length свечей
        features = features[-self.config['sequence_length']:]
        
        return features.astype(np.float32)
    
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
            try:
                features_scaled = self.scaler.transform(features)
            except NotFittedError:
                logger.error("Scaler не обучен. Выполните обучение модели")
                return None
            except Exception as transform_error:
                logger.error(f"Ошибка нормализации: {transform_error}")
                return None
            
            # Добавляем batch dimension
            features_scaled = features_scaled.reshape(1, self.config['sequence_length'], -1).astype(np.float32)
            
            # Предсказание с явным указанием устройства (GPU если доступен, иначе CPU)
            if TENSORFLOW_AVAILABLE and GPU_AVAILABLE and GPU_DEVICE:
                # Используем GPU для предсказания
                with tf.device(GPU_DEVICE.name):
                    prediction = self.model.predict(features_scaled, verbose=0)[0]
            else:
                # Используем CPU для предсказания
                with tf.device('/CPU:0'):
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
            logger.error(f"Ошибка предсказания: {e}")
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

        if not training_data:
            logger.error("Пустой набор данных для обучения")
            return {'error': 'No training data provided'}
        
        try:
            # Объединяем данные
            X_list, y_list = zip(*training_data)
            X = np.array(X_list)
            y = np.array(y_list)

            if X.ndim != 3:
                raise ValueError(f"Training data X должен иметь размерность (samples, seq_len, features), получено: {X.shape}")

            if np.isnan(X).any() or np.isinf(X).any():
                logger.warning("Обнаружены NaN/Inf в обучающих данных. Выполняем замену на нули")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Проверяем наличие признаков
            if X.shape[-1] != len(self.config['features']):
                logger.warning(
                    "Количество признаков (%s) не совпадает с конфигурацией (%s). Обновляем конфиг.",
                    X.shape[-1], len(self.config['features'])
                )
                self.config['features'] = [f'feature_{i}' for i in range(X.shape[-1])]

            # Обучаем scaler на всем массиве
            if self.scaler is None:
                self.scaler = MinMaxScaler(feature_range=(0, 1))

            flat_X = X.reshape(-1, X.shape[-1])
            self.scaler.fit(flat_X)
            X_scaled = self.scaler.transform(flat_X).reshape(X.shape).astype(np.float32)
            
            # Проверяем и логируем информацию о GPU перед обучением
            device_info = "CPU"
            if TENSORFLOW_AVAILABLE and GPU_AVAILABLE and GPU_DEVICE:
                device_info = f"GPU NVIDIA ({GPU_DEVICE.name})"
                logger.info(f"🚀 Обучение на {device_info}")
            else:
                logger.info(f"💻 Обучение на {device_info}")
            
            logger.info(f"Начало обучения: {len(X)} образцов")
            logger.info(f"Форма X: {X.shape}, форма y: {y.shape}")
            
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
            
            # Обучаем модель с явным указанием устройства (GPU если доступен, иначе CPU)
            if TENSORFLOW_AVAILABLE and GPU_AVAILABLE and GPU_DEVICE:
                # Используем GPU для обучения
                with tf.device(GPU_DEVICE.name):
                    history = self.model.fit(
                        X_scaled, y,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=1
                    )
            else:
                # Используем CPU для обучения
                with tf.device('/CPU:0'):
                    history = self.model.fit(
                        X_scaled, y,
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
            
            logger.info("✅ Обучение завершено успешно")
            
            return {
                'success': True,
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'epochs_trained': len(history.history['loss']),
                'training_samples': len(X)
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения: {e}")
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
            
            logger.info(f"✅ Модель сохранена: {self.model_path}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения модели: {e}")
    
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
            
            logger.info(f"✅ Модель загружена: {self.model_path}")
            logger.info(f"Обучена: {self.config.get('trained_at', 'неизвестно')}")
            logger.info(f"Образцов: {self.config.get('training_samples', 0)}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
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

