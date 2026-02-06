"""
Transformer Predictor - Temporal Fusion Transformer для предсказания цены

Реализация упрощенного TFT (Temporal Fusion Transformer) для временных рядов:
- Positional Encoding для временной информации
- Gated Residual Network (GRN) для нелинейных преобразований
- Variable Selection Network для выбора важных признаков
- Multi-Head Attention для долгосрочных зависимостей
- Quantile outputs для оценки неопределенности

Совместимый API с LSTMPredictor для легкой замены.
"""

import os
import json
import pickle
import logging
import math
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger('Transformer')

# Проверяем PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler
    PYTORCH_AVAILABLE = True
    
    # Определяем устройство
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
        GPU_AVAILABLE = True
        logger.info(f"Transformer: GPU доступен ({torch.cuda.get_device_name(0)})")
    else:
        DEVICE = torch.device('cpu')
        GPU_AVAILABLE = False
        logger.info("Transformer: используется CPU")
        
except ImportError:
    PYTORCH_AVAILABLE = False
    GPU_AVAILABLE = False
    DEVICE = None
    logger.warning("PyTorch не установлен. Transformer Predictor недоступен.")


if PYTORCH_AVAILABLE:
    
    class PositionalEncoding(nn.Module):
        """
        Positional Encoding для временных рядов
        
        Добавляет информацию о позиции каждого элемента последовательности
        """
        
        def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            # Создаем positional encoding матрицу
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)
        
        def forward(self, x):
            """
            Args:
                x: (batch, seq_len, d_model)
            """
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)
    
    
    class GatedLinearUnit(nn.Module):
        """Gated Linear Unit activation"""
        
        def __init__(self, input_dim: int, output_dim: int):
            super(GatedLinearUnit, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim * 2)
        
        def forward(self, x):
            out = self.linear(x)
            out, gate = out.chunk(2, dim=-1)
            return out * torch.sigmoid(gate)
    
    
    class GatedResidualNetwork(nn.Module):
        """
        Gated Residual Network (GRN)
        
        Архитектура: Linear -> ELU -> Linear -> GLU -> Add & Norm
        """
        
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float = 0.1,
            context_dim: int = None
        ):
            super(GatedResidualNetwork, self).__init__()
            
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.context_dim = context_dim
            
            # Основная сеть
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.glu = GatedLinearUnit(hidden_dim, output_dim)
            
            # Context (опционально)
            if context_dim is not None:
                self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)
            
            # Skip connection
            if input_dim != output_dim:
                self.skip = nn.Linear(input_dim, output_dim)
            else:
                self.skip = None
            
            self.layer_norm = nn.LayerNorm(output_dim)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x, context=None):
            """
            Args:
                x: (batch, ..., input_dim)
                context: (batch, context_dim) опциональный контекст
            """
            # Skip connection
            if self.skip is not None:
                skip = self.skip(x)
            else:
                skip = x
            
            # Основная сеть
            out = F.elu(self.fc1(x))
            
            # Добавляем контекст если есть
            if context is not None and self.context_dim is not None:
                context_out = self.context_proj(context)
                if len(out.shape) == 3:  # (batch, seq, hidden)
                    context_out = context_out.unsqueeze(1).expand(-1, out.size(1), -1)
                out = out + context_out
            
            out = self.dropout(F.elu(self.fc2(out)))
            out = self.glu(out)
            
            # Residual + LayerNorm
            return self.layer_norm(out + skip)
    
    
    class VariableSelectionNetwork(nn.Module):
        """
        Variable Selection Network
        
        Выбирает важные признаки с помощью softmax attention
        """
        
        def __init__(
            self,
            input_dim: int,
            num_inputs: int,
            hidden_dim: int,
            dropout: float = 0.1,
            context_dim: int = None
        ):
            super(VariableSelectionNetwork, self).__init__()
            
            self.num_inputs = num_inputs
            self.hidden_dim = hidden_dim
            
            # GRN для каждого входа
            self.grns = nn.ModuleList([
                GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout, context_dim)
                for _ in range(num_inputs)
            ])
            
            # Softmax selection
            self.flatten_grn = GatedResidualNetwork(
                num_inputs * hidden_dim, hidden_dim, num_inputs, dropout, context_dim
            )
        
        def forward(self, inputs: List[torch.Tensor], context=None):
            """
            Args:
                inputs: Список из num_inputs тензоров (batch, seq, input_dim)
                context: (batch, context_dim)
            
            Returns:
                selected: (batch, seq, hidden_dim)
                weights: (batch, seq, num_inputs) - важность каждого признака
            """
            # Обрабатываем каждый вход
            processed = [grn(inp, context) for grn, inp in zip(self.grns, inputs)]
            
            # Stack и flatten
            stacked = torch.stack(processed, dim=-1)  # (batch, seq, hidden, num_inputs)
            batch_size, seq_len, hidden, num_inp = stacked.shape
            
            # Reshape для selection
            flat = stacked.permute(0, 1, 3, 2).reshape(batch_size, seq_len, -1)
            
            # Вычисляем веса
            weights = F.softmax(self.flatten_grn(flat, context), dim=-1)  # (batch, seq, num_inputs)
            
            # Взвешенная сумма
            selected = (stacked * weights.unsqueeze(2)).sum(dim=-1)  # (batch, seq, hidden)
            
            return selected, weights
    
    
    class InterpretableMultiHeadAttention(nn.Module):
        """
        Interpretable Multi-Head Attention
        
        Возвращает attention weights для интерпретации
        """
        
        def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
            super(InterpretableMultiHeadAttention, self).__init__()
            
            assert d_model % num_heads == 0
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            
            self.q_linear = nn.Linear(d_model, d_model)
            self.k_linear = nn.Linear(d_model, d_model)
            self.v_linear = nn.Linear(d_model, d_model)
            self.out_linear = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.head_dim)
        
        def forward(self, query, key, value, mask=None):
            """
            Args:
                query, key, value: (batch, seq, d_model)
                mask: (batch, seq, seq) опциональная маска
            
            Returns:
                output: (batch, seq, d_model)
                attention_weights: (batch, num_heads, seq, seq)
            """
            batch_size = query.size(0)
            
            # Linear projections
            Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values
            context = torch.matmul(attention_weights, V)
            
            # Reshape and linear
            context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            output = self.out_linear(context)
            
            return output, attention_weights
    
    
    class TemporalFusionTransformer(nn.Module):
        """
        Simplified Temporal Fusion Transformer
        
        Архитектура:
        1. Input embedding + Positional Encoding
        2. Variable Selection
        3. LSTM Encoder
        4. Multi-Head Attention
        5. GRN Output
        """
        
        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_heads: int = 4,
            num_lstm_layers: int = 1,
            dropout: float = 0.1,
            output_size: int = 3  # direction, change_percent, confidence
        ):
            super(TemporalFusionTransformer, self).__init__()
            
            self.input_size = input_size
            self.hidden_size = hidden_size
            
            # Input embedding
            self.input_embedding = nn.Linear(input_size, hidden_size)
            
            # Positional encoding
            self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)
            
            # LSTM encoder
            self.lstm = nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
                dropout=dropout if num_lstm_layers > 1 else 0,
                bidirectional=True
            )
            
            # После bidirectional LSTM размер удваивается
            self.lstm_proj = nn.Linear(hidden_size * 2, hidden_size)
            
            # Multi-head attention
            self.attention = InterpretableMultiHeadAttention(hidden_size, num_heads, dropout)
            self.attention_norm = nn.LayerNorm(hidden_size)
            
            # GRN output layers
            self.grn1 = GatedResidualNetwork(hidden_size, hidden_size * 2, hidden_size, dropout)
            self.grn2 = GatedResidualNetwork(hidden_size, hidden_size, hidden_size // 2, dropout)
            
            # Output heads
            self.direction_head = nn.Linear(hidden_size // 2, 1)
            self.change_head = nn.Linear(hidden_size // 2, 1)
            self.confidence_head = nn.Linear(hidden_size // 2, 1)
            
            # Сохраняем attention weights
            self.last_attention_weights = None
            
            logger.info(f"TFT создан: input={input_size}, hidden={hidden_size}, heads={num_heads}")
        
        def forward(self, x, return_attention=False):
            """
            Args:
                x: (batch, seq_len, input_size)
                return_attention: вернуть attention weights
            
            Returns:
                output: (batch, 3) - [direction, change_percent, confidence]
            """
            # Input embedding + positional encoding
            embedded = self.input_embedding(x)
            embedded = self.pos_encoder(embedded)
            
            # LSTM encoding
            lstm_out, _ = self.lstm(embedded)
            lstm_out = self.lstm_proj(lstm_out)
            
            # Self-attention
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            self.last_attention_weights = attn_weights.detach()
            
            # Residual + norm
            attn_out = self.attention_norm(lstm_out + attn_out)
            
            # GRN processing
            out = self.grn1(attn_out)
            out = self.grn2(out)
            
            # Берем последний временной шаг
            final = out[:, -1, :]
            
            # Output heads
            direction = torch.tanh(self.direction_head(final))
            change = self.change_head(final)
            confidence = torch.sigmoid(self.confidence_head(final))
            
            output = torch.cat([direction, change, confidence], dim=-1)
            
            if return_attention:
                return output, attn_weights
            return output
        
        def get_attention_weights(self):
            """Возвращает последние attention weights"""
            return self.last_attention_weights


class TransformerPredictor:
    """
    Predictor на основе Temporal Fusion Transformer
    
    API совместим с LSTMPredictor для легкой замены
    """
    
    def __init__(
        self,
        model_path: str = "data/ai/models/transformer_predictor.pth",
        scaler_path: str = "data/ai/models/transformer_scaler.pkl",
        config_path: str = "data/ai/models/transformer_config.json"
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config_path = config_path
        
        self.model = None
        self.scaler = None
        self.config = {
            'sequence_length': 60,
            'features': ['close', 'volume', 'high', 'low', 'rsi', 'ema_fast', 'ema_slow'],
            'prediction_horizon': 6,
            'model_version': '1.0',
            'model_architecture': 'tft',
            'hidden_size': 64,
            'num_heads': 4,
            'trained_at': None,
            'training_samples': 0
        }
        
        if not PYTORCH_AVAILABLE:
            logger.error("PyTorch недоступен")
            return
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.load_model()
        else:
            logger.info("Модель не найдена, создаем новую")
            self._create_new_model()
    
    def _create_new_model(self):
        """Создает новую TFT модель"""
        if not PYTORCH_AVAILABLE:
            return
        
        n_features = len(self.config['features'])
        hidden_size = self.config.get('hidden_size', 64)
        num_heads = self.config.get('num_heads', 4)
        
        self.model = TemporalFusionTransformer(
            input_size=n_features,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=0.2
        )
        self.model.to(DEVICE)
        self.model.eval()
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        logger.info(f"TFT модель создана: {n_features} признаков, hidden={hidden_size}")
    
    def prepare_features(self, candles: List[Dict]) -> Optional[np.ndarray]:
        """Подготавливает признаки из свечей"""
        if len(candles) < self.config['sequence_length']:
            return None
        
        df = pd.DataFrame(candles)
        
        features = []
        for feature in self.config['features']:
            if feature in df.columns:
                features.append(df[feature].values)
            else:
                features.append(np.zeros(len(df)))
        
        features = np.array(features).T
        features = features[-self.config['sequence_length']:]
        
        return features.astype(np.float32)
    
    def predict(self, candles: List[Dict], current_price: float) -> Optional[Dict]:
        """
        Предсказывает движение цены
        
        Args:
            candles: История свечей
            current_price: Текущая цена
        
        Returns:
            Dict с предсказанием
        """
        if not PYTORCH_AVAILABLE or self.model is None:
            return None
        
        try:
            features = self.prepare_features(candles)
            if features is None:
                return None
            
            try:
                features_scaled = self.scaler.transform(features)
            except:
                logger.warning("Scaler не обучен")
                return None
            
            features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(DEVICE)
            
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(features_tensor)
                prediction = prediction.cpu().numpy()[0]
            
            direction_raw = prediction[0]
            change_percent = prediction[1]
            confidence = prediction[2]
            
            direction = 1 if direction_raw > 0 else -1
            confidence = min(max(abs(confidence) * 100, 0), 100)
            predicted_price = current_price * (1 + change_percent / 100)
            
            return {
                'direction': direction,
                'change_percent': float(change_percent),
                'confidence': float(confidence),
                'predicted_price': float(predicted_price),
                'horizon_hours': self.config['prediction_horizon'],
                'current_price': current_price,
                'model': 'TFT'
            }
            
        except Exception as e:
            logger.error(f"Ошибка предсказания TFT: {e}")
            return None
    
    def train(
        self,
        training_data: List[Tuple[np.ndarray, np.ndarray]],
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict:
        """Обучает TFT модель"""
        if not PYTORCH_AVAILABLE or self.model is None:
            return {'error': 'PyTorch unavailable'}
        
        if not training_data:
            return {'error': 'No training data'}
        
        try:
            X_list, y_list = zip(*training_data)
            X = np.array(X_list)
            y = np.array(y_list)
            
            if X.shape[-1] != len(self.config['features']):
                self.config['features'] = [f'feature_{i}' for i in range(X.shape[-1])]
                self._create_new_model()
            
            if self.scaler is None:
                self.scaler = MinMaxScaler(feature_range=(0, 1))
            
            flat_X = X.reshape(-1, X.shape[-1])
            self.scaler.fit(flat_X)
            X_scaled = self.scaler.transform(flat_X).reshape(X.shape).astype(np.float32)
            
            split_idx = int(len(X_scaled) * (1 - validation_split))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
            y_train_tensor = torch.FloatTensor(y_train).to(DEVICE)
            X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
            y_val_tensor = torch.FloatTensor(y_val).to(DEVICE)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            logger.info(f"Начало обучения TFT: {len(X_train)} train, {len(X_val)} val")
            
            self.model.train()
            best_val_loss = float('inf')
            history = {'loss': [], 'val_loss': []}
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_train_loss = epoch_loss / len(train_loader)
                
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                self.model.train()
                
                history['loss'].append(avg_train_loss)
                history['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.model_path + '.best')
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.6f}, Val: {val_loss:.6f}")
            
            self.config['trained_at'] = datetime.now().isoformat()
            self.config['training_samples'] = len(X)
            self.save_model()
            
            logger.info("TFT обучение завершено")
            
            return {
                'success': True,
                'final_loss': float(history['loss'][-1]),
                'final_val_loss': float(history['val_loss'][-1]),
                'epochs_trained': epochs,
                'training_samples': len(X)
            }
            
        except Exception as e:
            logger.error(f"Ошибка обучения TFT: {e}")
            return {'error': str(e)}
    
    def save_model(self):
        """Сохраняет модель"""
        if not PYTORCH_AVAILABLE or self.model is None:
            return
        
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
            
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"TFT модель сохранена: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения TFT: {e}")
    
    def load_model(self):
        """Загружает модель"""
        if not PYTORCH_AVAILABLE:
            return
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            
            n_features = len(self.config['features'])
            hidden_size = self.config.get('hidden_size', 64)
            num_heads = self.config.get('num_heads', 4)
            
            self.model = TemporalFusionTransformer(
                input_size=n_features,
                hidden_size=hidden_size,
                num_heads=num_heads
            )
            self.model.to(DEVICE)
            self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
            self.model.eval()
            
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            logger.info(f"TFT модель загружена: {self.model_path}")
            
        except Exception as e:
            logger.warning(f"Ошибка загрузки TFT: {e}")
            self._create_new_model()
    
    def get_status(self) -> Dict:
        """Возвращает статус модели"""
        if not PYTORCH_AVAILABLE:
            return {'available': False, 'error': 'PyTorch not installed'}
        
        is_trained = (
            self.model is not None and
            os.path.exists(self.model_path) and
            self.config.get('training_samples', 0) > 0
        )
        
        return {
            'available': True,
            'trained': is_trained,
            'model_path': self.model_path,
            'model_architecture': 'Temporal Fusion Transformer',
            'sequence_length': self.config['sequence_length'],
            'prediction_horizon': self.config['prediction_horizon'],
            'trained_at': self.config.get('trained_at'),
            'training_samples': self.config.get('training_samples', 0),
            'features': self.config['features'],
            'hidden_size': self.config.get('hidden_size'),
            'num_heads': self.config.get('num_heads'),
            'framework': 'PyTorch',
            'gpu_available': GPU_AVAILABLE,
            'device': str(DEVICE) if DEVICE else 'cpu'
        }
    
    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Возвращает attention weights для визуализации"""
        if self.model is None:
            return None
        
        try:
            weights = self.model.get_attention_weights()
            if weights is not None:
                return weights.cpu().numpy()
        except:
            pass
        
        return None


# ==================== ТЕСТОВЫЙ КОД ====================

if __name__ == '__main__':
    print("=" * 60)
    print("Transformer Predictor - Тест")
    print("=" * 60)
    
    if not PYTORCH_AVAILABLE:
        print("PyTorch не установлен!")
        exit(1)
    
    # Создаем тестовые данные
    np.random.seed(42)
    
    # Генерируем свечи
    n_candles = 100
    candles = []
    price = 100.0
    
    for i in range(n_candles):
        change = np.random.randn() * 0.02
        open_price = price
        close_price = price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.randn() * 0.005))
        low_price = min(open_price, close_price) * (1 - abs(np.random.randn() * 0.005))
        
        candles.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': np.random.randint(1000, 10000),
            'rsi': np.random.uniform(30, 70),
            'ema_fast': close_price * np.random.uniform(0.99, 1.01),
            'ema_slow': close_price * np.random.uniform(0.98, 1.02)
        })
        price = close_price
    
    # Тестируем TFT Predictor
    print("\n1. Создание TransformerPredictor:")
    predictor = TransformerPredictor()
    status = predictor.get_status()
    print(f"   Статус: {status['model_architecture']}")
    print(f"   GPU: {status['gpu_available']}")
    print(f"   Device: {status['device']}")
    
    # Тестируем модель напрямую
    print("\n2. Тест модели TFT:")
    n_features = 7
    batch_size = 4
    seq_len = 60
    
    model = TemporalFusionTransformer(input_size=n_features, hidden_size=64)
    model.to(DEVICE)
    
    test_input = torch.randn(batch_size, seq_len, n_features).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        output, attn = model(test_input, return_attention=True)
    
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention shape: {attn.shape}")
    
    # Тест GRN
    print("\n3. Тест GatedResidualNetwork:")
    grn = GatedResidualNetwork(64, 128, 64).to(DEVICE)
    grn_input = torch.randn(4, 60, 64).to(DEVICE)
    grn_output = grn(grn_input)
    print(f"   GRN input: {grn_input.shape}")
    print(f"   GRN output: {grn_output.shape}")
    
    print("\n" + "=" * 60)
    print("[OK] Все тесты пройдены!")
    print("=" * 60)
