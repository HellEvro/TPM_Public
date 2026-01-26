"""
LSTM Predictor для предсказания движения цены криптовалют

Этот модуль использует LSTM нейронную сеть для предсказания:
- Направления движения цены (вверх/вниз)
- Ожидаемого изменения цены в %
- Вероятности движения

Используется для улучшения точности входов в сделки.

Теперь использует PyTorch вместо TensorFlow для лучшей поддержки Python 3.14+ и GPU.
"""

import os
import json
import pickle
import logging
import warnings
import time
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

# Отключаем предупреждения PyTorch
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# Проверяем доступность PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler
    PYTORCH_AVAILABLE = True
    
    # Настройка GPU для PyTorch
    def configure_gpu():
        """Настраивает PyTorch для использования GPU NVIDIA (если доступен)"""
        try:
            # Выводим информацию о версии PyTorch
            torch_version = torch.__version__
            logger.debug(f"PyTorch версия: {torch_version}")
            
            # Проверяем доступность CUDA
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                primary_gpu = torch.device('cuda:0')
                gpu_name = torch.cuda.get_device_name(0)
                
                logger.info(f"✅ Найдено GPU устройств: {gpu_count}")
                for i in range(gpu_count):
                    logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                
                logger.info(f"✅ GPU NVIDIA доступен и будет использоваться для обучения и предсказаний")
                logger.info(f"   Основное устройство: {gpu_name}")
                
                return True, primary_gpu
            else:
                logger.info("ℹ️ GPU устройства не найдены, используется CPU")
                return False, torch.device('cpu')
        except Exception as e:
            logger.warning(f"⚠️ Ошибка проверки GPU: {e}")
            logger.info("Продолжаем с CPU...")
            return False, torch.device('cpu')
    
    # Настраиваем GPU при импорте модуля
    GPU_AVAILABLE, DEVICE = configure_gpu()
    
except ImportError:
    PYTORCH_AVAILABLE = False
    GPU_AVAILABLE = False
    DEVICE = None
    logger.warning("PyTorch не установлен. LSTM Predictor недоступен.")


# Определяем класс LSTMModel только если PyTorch доступен
if PYTORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """
        PyTorch LSTM модель для предсказания движения цены
        """
        
        def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32], dropout: float = 0.2):
            super(LSTMModel, self).__init__()
            
            self.hidden_sizes = hidden_sizes
            self.num_layers = len(hidden_sizes)
            
            # LSTM слои
            self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True, num_layers=1)
            self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
            self.dropout1 = nn.Dropout(dropout)
            
            self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True, num_layers=1)
            self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
            self.dropout2 = nn.Dropout(dropout)
            
            self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True, num_layers=1)
            self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
            self.dropout3 = nn.Dropout(dropout)
            
            # Полносвязные слои
            self.fc1 = nn.Linear(hidden_sizes[2], 32)
            self.dropout4 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 3)  # Выход: [направление, изменение_%, вероятность]
            
        def forward(self, x):
            # Первый LSTM слой (возвращает последовательность)
            lstm_out1, _ = self.lstm1(x)  # (batch, seq_len, hidden1)
            # Применяем BatchNorm к каждому временному шагу
            batch_size, seq_len, hidden = lstm_out1.shape
            lstm_out1 = lstm_out1.reshape(-1, hidden)
            lstm_out1 = self.bn1(lstm_out1)
            lstm_out1 = lstm_out1.reshape(batch_size, seq_len, hidden)
            lstm_out1 = self.dropout1(lstm_out1)
            
            # Второй LSTM слой (возвращает последовательность)
            lstm_out2, _ = self.lstm2(lstm_out1)  # (batch, seq_len, hidden2)
            batch_size, seq_len, hidden = lstm_out2.shape
            lstm_out2 = lstm_out2.reshape(-1, hidden)
            lstm_out2 = self.bn2(lstm_out2)
            lstm_out2 = lstm_out2.reshape(batch_size, seq_len, hidden)
            lstm_out2 = self.dropout2(lstm_out2)
            
            # Третий LSTM слой (не возвращает последовательность)
            lstm_out3, _ = self.lstm3(lstm_out2)  # (batch, hidden3)
            lstm_out3 = self.bn3(lstm_out3)
            lstm_out3 = self.dropout3(lstm_out3)
            
            # Полносвязные слои
            out = torch.relu(self.fc1(lstm_out3))
            out = self.dropout4(out)
            out = torch.relu(self.fc2(out))
            out = self.fc3(out)  # Линейный выход
            
            return out
else:
    # Заглушка для случая, когда PyTorch недоступен
    class LSTMModel:
        """Заглушка для LSTMModel когда PyTorch недоступен"""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch не установлен. Установите: pip install torch")


class LSTMPredictor:
    """
    LSTM модель для предсказания движения цены криптовалют (PyTorch версия)
    """
    
    def __init__(
        self,
        model_path: str = "data/ai/models/lstm_predictor.pth",  # PyTorch формат
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
            'model_version': '2.0',  # Версия 2.0 для PyTorch
            'trained_at': None,
            'training_samples': 0
        }
        
        if not PYTORCH_AVAILABLE:
            logger.error("PyTorch недоступен. Установите: pip install torch")
            return
        
        # Выводим информацию о GPU при инициализации
        if PYTORCH_AVAILABLE:
            if GPU_AVAILABLE and DEVICE:
                logger.info(f"🚀 LSTM Predictor инициализирован с поддержкой GPU NVIDIA: {DEVICE}")
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
        if not PYTORCH_AVAILABLE:
            return
        
        try:
            sequence_length = self.config['sequence_length']
            n_features = len(self.config['features'])
            
            # Создаем PyTorch модель
            self.model = LSTMModel(input_size=n_features)
            self.model.to(DEVICE)
            self.model.eval()  # Режим оценки по умолчанию
        except NameError as e:
            logger.error(f"Ошибка создания модели: {e}. PyTorch недоступен.")
            return
        
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
        if not PYTORCH_AVAILABLE or self.model is None:
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
            
            # Добавляем batch dimension и конвертируем в tensor
            features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(DEVICE)
            
            # Логируем использование GPU для предсказания
            if GPU_AVAILABLE and DEVICE and features_tensor.device.type == 'cuda':
                logger.debug(f"🚀 Предсказание выполняется на GPU: {DEVICE}")
            
            # Предсказание
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(features_tensor)
                # Синхронизируем GPU перед переносом на CPU
                if GPU_AVAILABLE and DEVICE and features_tensor.device.type == 'cuda':
                    torch.cuda.synchronize()
                prediction = prediction.cpu().numpy()[0]
            
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
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        Обучает LSTM модель
        
        Args:
            training_data: Список (X, y) где X - признаки, y - целевые значения
            validation_split: Доля данных для валидации
            epochs: Количество эпох обучения
            batch_size: Размер батча
            learning_rate: Скорость обучения
        
        Returns:
            История обучения
        """
        if not PYTORCH_AVAILABLE or self.model is None:
            return {'error': 'PyTorch unavailable'}

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
                # Пересоздаем модель с новым количеством признаков
                self._create_new_model()

            # Обучаем scaler на всем массиве
            if self.scaler is None:
                self.scaler = MinMaxScaler(feature_range=(0, 1))

            flat_X = X.reshape(-1, X.shape[-1])
            self.scaler.fit(flat_X)
            X_scaled = self.scaler.transform(flat_X).reshape(X.shape).astype(np.float32)
            
            # Разделяем на train и validation
            split_idx = int(len(X_scaled) * (1 - validation_split))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Конвертируем в PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
            y_train_tensor = torch.FloatTensor(y_train).to(DEVICE)
            X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
            y_val_tensor = torch.FloatTensor(y_val).to(DEVICE)
            
            # Создаем DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Настраиваем оптимизатор и функцию потерь
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # Проверяем и логируем информацию о GPU перед обучением
            device_info = "CPU"
            if PYTORCH_AVAILABLE and GPU_AVAILABLE and DEVICE:
                device_info = f"GPU NVIDIA ({DEVICE})"
                logger.info(f"🚀 Обучение на {device_info}")
            else:
                logger.info(f"💻 Обучение на {device_info}")
            
            logger.info(f"Начало обучения: {len(X_train)} образцов (train), {len(X_val)} образцов (val)")
            logger.info(f"Форма X: {X.shape}, форма y: {y.shape}")
            
            # Обучение
            self.model.train()
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            history = {'loss': [], 'val_loss': []}
            
            # Логируем использование GPU при обучении
            if GPU_AVAILABLE and DEVICE:
                logger.info(f"🚀 Обучение на GPU: {DEVICE}")
                memory_before = torch.cuda.memory_allocated(0) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
                logger.info(f"📊 Память GPU до обучения: {memory_before:.2f} MB (выделено) / {memory_reserved:.2f} MB (зарезервировано)")
            
            for epoch in range(epochs):
                # Обучение
                epoch_loss = 0.0
                epoch_start_time = time.time()
                
                for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                    # Проверяем, что данные на правильном устройстве
                    if GPU_AVAILABLE and DEVICE:
                        if batch_X.device.type != 'cuda':
                            logger.warning(f"⚠️ Batch {batch_idx}: данные не на GPU! Перемещаю...")
                            batch_X = batch_X.to(DEVICE)
                            batch_y = batch_y.to(DEVICE)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                    # Логируем использование GPU каждые 10 батчей (только для первых 3 эпох)
                    if GPU_AVAILABLE and DEVICE and epoch < 3 and batch_idx % 10 == 0:
                        memory_used = torch.cuda.memory_allocated(0) / 1024**2
                        logger.debug(f"📊 Эпоха {epoch+1}/{epochs}, Батч {batch_idx}: GPU память = {memory_used:.2f} MB, Loss = {loss.item():.6f}")
                
                # Синхронизируем GPU после каждой эпохи
                if GPU_AVAILABLE and DEVICE:
                    torch.cuda.synchronize()
                    memory_after = torch.cuda.memory_allocated(0) / 1024**2
                    if epoch % 5 == 0 or epoch == 0:  # Логируем каждые 5 эпох
                        logger.info(f"📊 Эпоха {epoch+1}/{epochs}: Loss={avg_train_loss:.6f}, GPU память={memory_after:.2f} MB")
                
                epoch_time = time.time() - epoch_start_time
                avg_train_loss = epoch_loss / len(train_loader)
                
                avg_train_loss = epoch_loss / len(train_loader)
                
                # Валидация
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                self.model.train()
                
                history['loss'].append(avg_train_loss)
                history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Сохраняем лучшую модель
                    torch.save(self.model.state_dict(), self.model_path + '.best')
                else:
                    patience_counter += 1
                
                # Learning rate scheduling
                if patience_counter >= 5:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                        if param_group['lr'] < 0.00001:
                            param_group['lr'] = 0.00001
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping на эпохе {epoch+1}")
                    # Загружаем лучшую модель
                    self.model.load_state_dict(torch.load(self.model_path + '.best'))
                    break
            
            # Обновляем конфигурацию
            self.config['trained_at'] = datetime.now().isoformat()
            self.config['training_samples'] = len(X)
            
            # Сохраняем модель
            self.save_model()
            
            logger.info("✅ Обучение завершено успешно")
            
            return {
                'success': True,
                'final_loss': float(history['loss'][-1]),
                'final_val_loss': float(history['val_loss'][-1]),
                'epochs_trained': len(history['loss']),
                'training_samples': len(X)
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения: {e}", exc_info=True)
            return {'error': str(e)}
    
    def save_model(self):
        """Сохраняет модель, scaler и конфигурацию"""
        if not PYTORCH_AVAILABLE or self.model is None:
            return
        
        try:
            # Создаем директорию, если не существует
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Сохраняем модель
            torch.save(self.model.state_dict(), self.model_path)
            
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
        if not PYTORCH_AVAILABLE:
            return
        
        try:
            # Загружаем конфигурацию
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            
            # Создаем модель с правильными параметрами
            n_features = len(self.config['features'])
            self.model = LSTMModel(input_size=n_features)
            self.model.to(DEVICE)
            
            # Загружаем веса модели
            self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
            self.model.eval()
        except NameError as e:
            logger.error(f"Ошибка загрузки модели: {e}. PyTorch недоступен.")
            self._create_new_model()
            
            # Загружаем scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            logger.info(f"✅ Модель загружена: {self.model_path}")
            logger.info(f"Обучена: {self.config.get('trained_at', 'неизвестно')}")
            logger.info(f"Образцов: {self.config.get('training_samples', 0)}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            self._create_new_model()
    
    def get_status(self) -> Dict:
        """Возвращает статус модели"""
        if not PYTORCH_AVAILABLE:
            return {
                'available': False,
                'error': 'PyTorch not installed'
            }
        
        is_trained = (
            self.model is not None and
            os.path.exists(self.model_path) and
            self.config.get('training_samples', 0) > 0
        )
        
        status = {
            'available': True,
            'trained': is_trained,
            'model_path': self.model_path,
            'sequence_length': self.config['sequence_length'],
            'prediction_horizon': self.config['prediction_horizon'],
            'trained_at': self.config.get('trained_at'),
            'training_samples': self.config.get('training_samples', 0),
            'features': self.config['features'],
            'framework': 'PyTorch'
        }
        
        # Добавляем информацию о GPU
        if PYTORCH_AVAILABLE:
            status['gpu_available'] = GPU_AVAILABLE
            status['device'] = str(DEVICE) if DEVICE else 'cpu'
            if GPU_AVAILABLE and DEVICE:
                try:
                    import torch
                    status['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                    status['cuda_version'] = torch.version.cuda if torch.cuda.is_available() else None
                except:
                    pass
        
        return status
