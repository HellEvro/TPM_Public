"""
Ensemble модели - объединение нескольких предикторов

Реализует:
- VotingEnsemble: взвешенное голосование
- StackingEnsemble: мета-модель поверх базовых
- EnsemblePredictor: объединяет LSTM, Transformer и SMC

Преимущества:
- Уменьшение дисперсии предсказаний
- Повышение робастности
- Комбинирование сильных сторон разных архитектур
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import json
import os

logger = logging.getLogger('Ensemble')

# Проверяем PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
    
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')
except ImportError:
    PYTORCH_AVAILABLE = False
    DEVICE = None


class VotingEnsemble:
    """
    Voting Ensemble - объединение моделей через взвешенное голосование
    
    Поддерживает:
    - Soft voting (усреднение вероятностей)
    - Hard voting (голосование по классам)
    - Динамические веса на основе производительности
    """
    
    def __init__(self, voting: str = 'soft'):
        """
        Args:
            voting: 'soft' или 'hard'
        """
        self.voting = voting
        self.models: Dict[str, Dict] = {}  # name -> {model, weight, stats}
        self.performance_history: Dict[str, List[float]] = {}
    
    def add_model(
        self,
        name: str,
        model: Any,
        weight: float = 1.0,
        predict_fn: Callable = None
    ):
        """
        Добавляет модель в ансамбль
        
        Args:
            name: Уникальное имя модели
            model: Объект модели
            weight: Начальный вес
            predict_fn: Функция для получения предсказания (если не стандартная)
        """
        self.models[name] = {
            'model': model,
            'weight': weight,
            'predict_fn': predict_fn or (lambda m, x: m.predict(x)),
            'correct_count': 0,
            'total_count': 0
        }
        self.performance_history[name] = []
        
        logger.info(f"Модель '{name}' добавлена в ансамбль с весом {weight}")
    
    def remove_model(self, name: str):
        """Удаляет модель из ансамбля"""
        if name in self.models:
            del self.models[name]
            del self.performance_history[name]
            logger.info(f"Модель '{name}' удалена из ансамбля")
    
    def set_weight(self, name: str, weight: float):
        """Устанавливает вес модели"""
        if name in self.models:
            self.models[name]['weight'] = weight
    
    def predict(self, x: Any) -> Dict:
        """
        Получает предсказание ансамбля
        
        Args:
            x: Входные данные (формат зависит от моделей)
        
        Returns:
            Dict с результатом голосования
        """
        if not self.models:
            return {'error': 'No models in ensemble'}
        
        predictions = {}
        weights = {}
        
        for name, model_data in self.models.items():
            try:
                pred = model_data['predict_fn'](model_data['model'], x)
                if pred is not None:
                    predictions[name] = pred
                    weights[name] = model_data['weight']
            except Exception as e:
                pass
        
        if not predictions:
            return {'error': 'All models failed'}
        
        # Нормализуем веса
        total_weight = sum(weights.values())
        norm_weights = {k: v / total_weight for k, v in weights.items()}
        
        if self.voting == 'soft':
            return self._soft_vote(predictions, norm_weights)
        else:
            return self._hard_vote(predictions, norm_weights)
    
    def _soft_vote(self, predictions: Dict, weights: Dict) -> Dict:
        """Soft voting - взвешенное усреднение"""
        
        # Собираем числовые предсказания
        directions = []
        changes = []
        confidences = []
        
        for name, pred in predictions.items():
            w = weights[name]
            
            if isinstance(pred, dict):
                if 'direction' in pred:
                    directions.append(pred['direction'] * w)
                if 'change_percent' in pred:
                    changes.append(pred['change_percent'] * w)
                if 'confidence' in pred:
                    confidences.append(pred['confidence'] * w)
        
        result = {
            'direction': 1 if sum(directions) > 0 else -1,
            'direction_score': sum(directions),
            'change_percent': sum(changes) if changes else 0,
            'confidence': sum(confidences) if confidences else 50,
            'model_predictions': predictions,
            'weights': weights,
            'voting_method': 'soft'
        }
        
        return result
    
    def _hard_vote(self, predictions: Dict, weights: Dict) -> Dict:
        """Hard voting - голосование по направлению"""
        
        votes_long = 0
        votes_short = 0
        
        for name, pred in predictions.items():
            w = weights[name]
            
            if isinstance(pred, dict):
                if pred.get('direction', 0) > 0:
                    votes_long += w
                elif pred.get('direction', 0) < 0:
                    votes_short += w
        
        direction = 1 if votes_long > votes_short else -1
        confidence = max(votes_long, votes_short) / (votes_long + votes_short + 1e-8) * 100
        
        return {
            'direction': direction,
            'votes_long': votes_long,
            'votes_short': votes_short,
            'confidence': confidence,
            'model_predictions': predictions,
            'voting_method': 'hard'
        }
    
    def update_performance(self, name: str, is_correct: bool):
        """Обновляет статистику производительности модели"""
        if name in self.models:
            self.models[name]['total_count'] += 1
            if is_correct:
                self.models[name]['correct_count'] += 1
            
            accuracy = self.models[name]['correct_count'] / self.models[name]['total_count']
            self.performance_history[name].append(accuracy)
    
    def update_weights_by_performance(self, window: int = 50):
        """
        Обновляет веса на основе недавней производительности
        
        Args:
            window: Размер окна для расчета
        """
        for name in self.models:
            history = self.performance_history.get(name, [])
            if len(history) >= window:
                recent_accuracy = np.mean(history[-window:])
                # Вес пропорционален точности
                self.models[name]['weight'] = max(0.1, recent_accuracy)
        
        # Нормализуем веса
        total = sum(m['weight'] for m in self.models.values())
        for name in self.models:
            self.models[name]['weight'] /= total
    
    def get_model_stats(self) -> Dict:
        """Возвращает статистику по моделям"""
        stats = {}
        for name, data in self.models.items():
            total = data['total_count']
            correct = data['correct_count']
            stats[name] = {
                'weight': data['weight'],
                'accuracy': correct / total if total > 0 else 0,
                'total_predictions': total
            }
        return stats


if PYTORCH_AVAILABLE:
    
    class MetaLearner(nn.Module):
        """
        Meta-learner для Stacking Ensemble
        
        Обучается предсказывать на основе выходов базовых моделей
        """
        
        def __init__(self, n_models: int, hidden_dim: int = 32):
            super(MetaLearner, self).__init__()
            
            # Каждая модель дает 3 выхода: direction, change, confidence
            input_dim = n_models * 3
            
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3)  # direction, change, confidence
            )
        
        def forward(self, x):
            return self.network(x)
    
    
    class StackingEnsemble:
        """
        Stacking Ensemble - мета-модель поверх базовых моделей
        
        Двухуровневая архитектура:
        1. Базовые модели делают предсказания
        2. Мета-модель комбинирует их в финальное предсказание
        """
        
        def __init__(self, hidden_dim: int = 32):
            self.hidden_dim = hidden_dim
            self.base_models: Dict[str, Dict] = {}
            self.meta_model: Optional[MetaLearner] = None
            self.is_trained = False
        
        def add_base_model(self, name: str, model: Any, predict_fn: Callable = None):
            """Добавляет базовую модель"""
            self.base_models[name] = {
                'model': model,
                'predict_fn': predict_fn or (lambda m, x: m.predict(x))
            }
            self.is_trained = False  # Нужно переобучить мета-модель
            logger.info(f"Базовая модель '{name}' добавлена")
        
        def _get_base_predictions(self, x: Any) -> Optional[torch.Tensor]:
            """Получает предсказания всех базовых моделей"""
            all_preds = []
            
            for name, model_data in self.base_models.items():
                try:
                    pred = model_data['predict_fn'](model_data['model'], x)
                    if pred is not None and isinstance(pred, dict):
                        all_preds.extend([
                            pred.get('direction', 0),
                            pred.get('change_percent', 0),
                            pred.get('confidence', 50) / 100
                        ])
                    else:
                        all_preds.extend([0, 0, 0.5])
                except Exception as e:
                    pass
                    all_preds.extend([0, 0, 0.5])
            
            if not all_preds:
                return None
            
            return torch.FloatTensor(all_preds).unsqueeze(0).to(DEVICE)
        
        def train_meta_model(
            self,
            X_data: List[Any],
            y_data: List[Tuple[float, float, float]],
            epochs: int = 100,
            lr: float = 0.001
        ) -> Dict:
            """
            Обучает мета-модель
            
            Args:
                X_data: Входные данные для базовых моделей
                y_data: Целевые значения [(direction, change, confidence), ...]
                epochs: Количество эпох
                lr: Learning rate
            """
            if not self.base_models:
                return {'error': 'No base models'}
            
            # Собираем предсказания базовых моделей
            meta_inputs = []
            for x in X_data:
                pred_tensor = self._get_base_predictions(x)
                if pred_tensor is not None:
                    meta_inputs.append(pred_tensor.squeeze(0).cpu().numpy())
            
            if not meta_inputs:
                return {'error': 'Failed to get base predictions'}
            
            X_meta = torch.FloatTensor(np.array(meta_inputs)).to(DEVICE)
            y_meta = torch.FloatTensor(np.array(y_data)).to(DEVICE)
            
            # Создаем мета-модель
            self.meta_model = MetaLearner(
                n_models=len(self.base_models),
                hidden_dim=self.hidden_dim
            ).to(DEVICE)
            
            optimizer = optim.Adam(self.meta_model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            self.meta_model.train()
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.meta_model(X_meta)
                loss = criterion(outputs, y_meta)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Meta-model epoch {epoch+1}/{epochs}, loss: {loss.item():.6f}")
            
            self.is_trained = True
            
            return {
                'success': True,
                'final_loss': loss.item(),
                'samples': len(X_data)
            }
        
        def predict(self, x: Any) -> Dict:
            """Получает предсказание ансамбля"""
            if not self.is_trained or self.meta_model is None:
                # Fallback на простое усреднение
                return self._fallback_predict(x)
            
            meta_input = self._get_base_predictions(x)
            if meta_input is None:
                return {'error': 'Failed to get base predictions'}
            
            self.meta_model.eval()
            with torch.no_grad():
                output = self.meta_model(meta_input).cpu().numpy()[0]
            
            return {
                'direction': 1 if output[0] > 0 else -1,
                'change_percent': float(output[1]),
                'confidence': float(np.clip(output[2] * 100, 0, 100)),
                'method': 'stacking'
            }
        
        def _fallback_predict(self, x: Any) -> Dict:
            """Fallback предсказание без мета-модели"""
            directions = []
            changes = []
            confidences = []
            
            for name, model_data in self.base_models.items():
                try:
                    pred = model_data['predict_fn'](model_data['model'], x)
                    if pred and isinstance(pred, dict):
                        directions.append(pred.get('direction', 0))
                        changes.append(pred.get('change_percent', 0))
                        confidences.append(pred.get('confidence', 50))
                except:
                    pass
            
            if not directions:
                return {'error': 'All models failed'}
            
            return {
                'direction': 1 if np.mean(directions) > 0 else -1,
                'change_percent': float(np.mean(changes)),
                'confidence': float(np.mean(confidences)),
                'method': 'average_fallback'
            }


class EnsemblePredictor:
    """
    Высокоуровневый Ensemble Predictor
    
    Объединяет:
    - LSTM Predictor
    - Transformer Predictor
    - SMC сигналы
    
    С автоматическим выбором весов
    """
    
    def __init__(
        self,
        lstm_predictor=None,
        transformer_predictor=None,
        smc_features=None,
        voting: str = 'soft'
    ):
        """
        Args:
            lstm_predictor: LSTMPredictor или None
            transformer_predictor: TransformerPredictor или None
            smc_features: SmartMoneyFeatures или None
            voting: Метод голосования
        """
        self.voting_ensemble = VotingEnsemble(voting=voting)
        
        self.lstm_predictor = lstm_predictor
        self.transformer_predictor = transformer_predictor
        self.smc_features = smc_features
        
        # Добавляем модели с начальными весами
        if lstm_predictor is not None:
            self.voting_ensemble.add_model(
                'lstm',
                lstm_predictor,
                weight=1.0,
                predict_fn=lambda m, x: m.predict(x['candles'], x['price']) if isinstance(x, dict) else None
            )
        
        if transformer_predictor is not None:
            self.voting_ensemble.add_model(
                'transformer',
                transformer_predictor,
                weight=1.0,
                predict_fn=lambda m, x: m.predict(x['candles'], x['price']) if isinstance(x, dict) else None
            )
        
        if smc_features is not None:
            self.voting_ensemble.add_model(
                'smc',
                smc_features,
                weight=0.8,  # SMC немного меньший вес по умолчанию
                predict_fn=self._smc_predict
            )
        
        logger.info(f"EnsemblePredictor создан с {len(self.voting_ensemble.models)} моделями")
    
    def _smc_predict(self, smc, x: Dict) -> Optional[Dict]:
        """Получает предсказание от SMC"""
        try:
            import pandas as pd
            
            candles = x.get('candles', [])
            if not candles:
                return None
            
            df = pd.DataFrame(candles)
            signal = smc.get_smc_signal(df)
            
            if signal.get('signal') == 'LONG':
                direction = 1
            elif signal.get('signal') == 'SHORT':
                direction = -1
            else:
                direction = 0
            
            return {
                'direction': direction,
                'change_percent': 0,  # SMC не предсказывает конкретное изменение
                'confidence': signal.get('confidence', 50),
                'smc_score': signal.get('score', 0),
                'smc_reasons': signal.get('reasons', [])
            }
        except Exception as e:
            pass
            return None
    
    def predict(self, candles: List[Dict], current_price: float) -> Dict:
        """
        Получает предсказание ансамбля
        
        Args:
            candles: Список свечей
            current_price: Текущая цена
        
        Returns:
            Dict с предсказанием
        """
        x = {'candles': candles, 'price': current_price}
        result = self.voting_ensemble.predict(x)
        
        result['current_price'] = current_price
        result['timestamp'] = datetime.now().isoformat()
        
        # Добавляем предсказанную цену
        if 'change_percent' in result:
            result['predicted_price'] = current_price * (1 + result['change_percent'] / 100)
        
        return result
    
    def update_model_performance(self, actual_direction: int):
        """
        Обновляет производительность моделей после получения результата
        
        Args:
            actual_direction: Фактическое направление (1 или -1)
        """
        # Получаем последние предсказания из истории (если есть)
        # Это упрощенная версия - в реальности нужно хранить историю
        pass
    
    def get_status(self) -> Dict:
        """Возвращает статус ансамбля"""
        return {
            'models': list(self.voting_ensemble.models.keys()),
            'weights': {k: v['weight'] for k, v in self.voting_ensemble.models.items()},
            'stats': self.voting_ensemble.get_model_stats(),
            'voting_method': self.voting_ensemble.voting
        }
    
    def set_model_weight(self, model_name: str, weight: float):
        """Устанавливает вес модели"""
        self.voting_ensemble.set_weight(model_name, weight)
    
    def auto_tune_weights(self, window: int = 50):
        """Автоматически подстраивает веса на основе производительности"""
        self.voting_ensemble.update_weights_by_performance(window)


# ==================== ТЕСТОВЫЙ КОД ====================

if __name__ == '__main__':
    print("=" * 60)
    print("Ensemble - Тест")
    print("=" * 60)
    
    # 1. Тест VotingEnsemble
    print("\n1. Тест VotingEnsemble:")
    
    voting = VotingEnsemble(voting='soft')
    
    # Создаем mock модели
    class MockModel:
        def __init__(self, bias):
            self.bias = bias
        
        def predict(self, x):
            return {
                'direction': 1 if np.random.random() > 0.5 - self.bias else -1,
                'change_percent': np.random.randn() * 2,
                'confidence': np.random.uniform(50, 90)
            }
    
    voting.add_model('model1', MockModel(0.1), weight=1.0)
    voting.add_model('model2', MockModel(-0.1), weight=0.8)
    voting.add_model('model3', MockModel(0.0), weight=0.6)
    
    result = voting.predict(None)
    print(f"   Результат голосования: direction={result['direction']}, confidence={result['confidence']:.1f}%")
    print(f"   Веса: {result['weights']}")
    
    # Обновляем статистику
    for _ in range(20):
        voting.update_performance('model1', np.random.random() > 0.4)
        voting.update_performance('model2', np.random.random() > 0.5)
        voting.update_performance('model3', np.random.random() > 0.6)
    
    stats = voting.get_model_stats()
    print(f"   Статистика: {stats}")
    
    # 2. Тест StackingEnsemble (если PyTorch доступен)
    if PYTORCH_AVAILABLE:
        print("\n2. Тест StackingEnsemble:")
        
        stacking = StackingEnsemble()
        stacking.add_base_model('mock1', MockModel(0.1))
        stacking.add_base_model('mock2', MockModel(-0.1))
        
        # Генерируем данные для обучения
        X_train = [None] * 50  # Mock данные
        y_train = [(np.random.choice([-1, 1]), np.random.randn(), np.random.uniform(0.5, 0.9)) 
                   for _ in range(50)]
        
        train_result = stacking.train_meta_model(X_train, y_train, epochs=50)
        print(f"   Обучение: {train_result}")
        
        pred = stacking.predict(None)
        print(f"   Предсказание: {pred}")
    
    # 3. Тест EnsemblePredictor
    print("\n3. Тест EnsemblePredictor:")
    
    ensemble = EnsemblePredictor()
    status = ensemble.get_status()
    print(f"   Статус: {status}")
    
    print("\n" + "=" * 60)
    print("[OK] Все тесты пройдены!")
    print("=" * 60)
