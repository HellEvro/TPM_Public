"""
Anomaly Detector - обнаружение аномалий (pump/dump)

Использует Isolation Forest для обнаружения аномальных движений цены.
Помогает улучшить ExitScam фильтр и избежать входов в pump/dump схемы.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional
try:
    import utils.sklearn_parallel_config  # noqa: F401 — до импорта sklearn, подавляет UserWarning delayed/Parallel
except ImportError:
    pass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib  # только dump/load; Parallel/delayed — оба из sklearn через utils.sklearn_parallel_config (патч joblib)
import os

logger = logging.getLogger('AI.AnomalyDetector')


class AnomalyDetector:
    """Детектор аномалий для обнаружения pump/dump"""
    
    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """
        Инициализация детектора
        
        Args:
            model_path: Путь к сохраненной модели (если None, создается новая)
            scaler_path: Путь к scaler (если None, определяется автоматически)
        """
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        # Параметры модели
        self.contamination = 0.1  # 10% данных считаем аномалиями
        self.random_state = 42
        
        # Пытаемся загрузить существующую модель
        if model_path and os.path.exists(model_path):
            self.load_model(model_path, scaler_path)
        else:
            # Создаем новую модель
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=100,
                max_samples='auto',
                max_features=1.0,
                bootstrap=False,
                n_jobs=1,  # без параллелизма — устраняет UserWarning про delayed/Parallel
                verbose=0
            )
            self.scaler = StandardScaler()
            logger.info(" Создана новая модель (не обучена)")
    
    def extract_features(self, candles: List[dict]) -> Optional[np.ndarray]:
        """
        Извлекает признаки из свечей для анализа аномалий
        
        Args:
            candles: Список свечей с полями close, high, low, volume
        
        Returns:
            Массив признаков или None если недостаточно данных
        """
        if len(candles) < 20:
            return None
        
        # Берем последние 20 свечей
        recent = candles[-20:]
        
        features = []
        
        # 1. Резкие изменения цены (последние 19 свечей) — реальные %: (close - prev_close)/prev_close*100
        price_changes = []
        for i in range(1, len(recent)):
            change = (recent[i]['close'] - recent[i-1]['close']) / recent[i-1]['close'] * 100
            price_changes.append(abs(change))
        
        # Добавляем статистики изменений цены
        features.append(np.max(price_changes))      # Максимальное изменение
        features.append(np.mean(price_changes))     # Среднее изменение
        features.append(np.std(price_changes))      # Стандартное отклонение
        
        # 2. Объем относительно среднего
        volumes = [c['volume'] for c in recent]
        avg_volume = np.mean(volumes[:-1])
        current_volume = volumes[-1]
        volume_spike = current_volume / avg_volume if avg_volume > 0 else 1.0
        features.append(volume_spike)
        
        # 3. Волатильность (коэффициент вариации)
        closes = [c['close'] for c in recent]
        volatility = np.std(closes) / np.mean(closes) if np.mean(closes) > 0 else 0
        features.append(volatility)
        
        # 4. Размах свечей (последние 5 свечей)
        for candle in recent[-5:]:
            candle_range = (candle['high'] - candle['low']) / candle['close']
            features.append(candle_range)
        
        # 5. Momentum (изменение за последние N свечей)
        momentum_3 = (closes[-1] - closes[-4]) / closes[-4] * 100 if len(closes) >= 4 else 0
        momentum_5 = (closes[-1] - closes[-6]) / closes[-6] * 100 if len(closes) >= 6 else 0
        momentum_10 = (closes[-1] - closes[-11]) / closes[-11] * 100 if len(closes) >= 11 else 0
        features.append(abs(momentum_3))
        features.append(abs(momentum_5))
        features.append(abs(momentum_10))
        
        # 6. Ускорение цены (вторая производная)
        if len(price_changes) >= 2:
            acceleration = price_changes[-1] - price_changes[-2]
            features.append(abs(acceleration))
        else:
            features.append(0)
        
        # 7. Объемный импульс
        volume_momentum = (volumes[-1] - np.mean(volumes[-5:])) / np.mean(volumes[-5:]) if np.mean(volumes[-5:]) > 0 else 0
        features.append(abs(volume_momentum))
        
        return np.array(features).reshape(1, -1)
    
    def detect(self, candles: List[dict]) -> Dict[str, Any]:
        """
        Обнаруживает аномалии в данных свечей
        
        Args:
            candles: Список свечей для анализа
        
        Returns:
            Словарь с результатами анализа:
            - is_anomaly: bool - обнаружена ли аномалия
            - severity: float - степень аномальности (0.0-1.0)
            - anomaly_type: str - тип аномалии (PUMP/DUMP/MANIPULATION)
            - anomaly_score: float - сырой скор от модели
        """
        # Извлекаем признаки
        features = self.extract_features(candles)
        
        if features is None:
            return {
                'is_anomaly': False,
                'severity': 0.0,
                'anomaly_type': None,
                'anomaly_score': 0.0,
                'reason': 'Insufficient data (need at least 20 candles)'
            }
        
        # Если модель не обучена, используем эвристический подход
        if not self.is_trained:
            return self._heuristic_detection(candles, features)
        
        # Нормализация признаков
        try:
            features_scaled = self.scaler.transform(features)
        except Exception as e:
            logger.error(f" Ошибка нормализации: {e}")
            features_scaled = features
        
        # Предсказание (-1 = аномалия, 1 = нормально)
        try:
            prediction = self.model.predict(features_scaled)[0]
            is_anomaly = (prediction == -1)
            
            # Вычисляем severity (насколько сильная аномалия)
            # Isolation Forest score_samples() даёт примерно [-0.5, 0.5]: чем отрицательнее — тем аномальнее.
            # Формула: severity = 1.0 - (anomaly_score + 0.5) → шкала 0–1 (98% = очень сильная аномалия).
            anomaly_score = self.model.score_samples(features_scaled)[0]
            severity = max(0.0, min(1.0, 1.0 - (anomaly_score + 0.5)))
            
        except Exception as e:
            logger.error(f" Ошибка предсказания: {e}")
            return self._heuristic_detection(candles, features)
        
        # Определяем тип аномалии
        anomaly_type = None
        if is_anomaly:
            anomaly_type = self._classify_anomaly_type(candles)
        
        return {
            'is_anomaly': is_anomaly,
            'severity': float(severity),
            'anomaly_type': anomaly_type,
            'anomaly_score': float(anomaly_score),
            'features_count': features.shape[1]
        }
    
    def _heuristic_detection(self, candles: List[dict], features: np.ndarray) -> Dict[str, Any]:
        """
        Эвристическое обнаружение аномалий (когда модель не обучена)
        
        Использует простые правила для обнаружения pump/dump
        """
        recent = candles[-20:]
        
        # Проверяем резкие изменения цены
        price_changes = []
        for i in range(1, len(recent)):
            change = (recent[i]['close'] - recent[i-1]['close']) / recent[i-1]['close'] * 100
            price_changes.append(change)
        
        max_change = max(abs(c) for c in price_changes)
        
        # Проверяем объемные всплески
        volumes = [c['volume'] for c in recent]
        avg_volume = np.mean(volumes[:-1])
        current_volume = volumes[-1]
        volume_spike = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Правила для обнаружения аномалий (свечи могут расти/падать на 100%, 200%+ — шкала по %)
        is_anomaly = False
        severity = 0.0
        reason = []
        
        # Правило 1: Резкое изменение цены (>10% за одну свечу). Severity по %: 100% хода = 100% severity, 200% = cap 100%
        if max_change > 10:
            is_anomaly = True
            severity = max(severity, min(1.0, max_change / 100.0))
            reason.append(f"Резкое изменение цены: {max_change:.1f}%")
        
        # Правило 2: Объемный всплеск (>3x среднего)
        if volume_spike > 3:
            is_anomaly = True
            severity = max(severity, min(1.0, volume_spike / 5))
            reason.append(f"Объемный всплеск: {volume_spike:.1f}x")
        
        # Правило 3: Серия резких изменений
        recent_changes = price_changes[-5:]
        if len([c for c in recent_changes if abs(c) > 5]) >= 3:
            is_anomaly = True
            severity = max(severity, 0.7)
            reason.append("Серия резких изменений")
        
        # Определяем тип аномалии
        anomaly_type = None
        if is_anomaly:
            anomaly_type = self._classify_anomaly_type(candles)
        
        return {
            'is_anomaly': is_anomaly,
            'severity': float(severity),
            'anomaly_type': anomaly_type,
            'anomaly_score': -severity,  # Имитируем score
            'method': 'heuristic',
            'reason': ' | '.join(reason) if reason else None
        }
    
    def _classify_anomaly_type(self, candles: List[dict]) -> str:
        """
        Классифицирует тип аномалии
        
        Returns:
            'PUMP' | 'DUMP' | 'MANIPULATION'
        """
        recent = candles[-6:]
        
        # Смотрим на изменения последних 5 свечей
        changes = [
            (candle['close'] - candles[i-1]['close']) / candles[i-1]['close'] * 100
            for i, candle in enumerate(recent[1:], start=len(candles)-5)
        ]
        
        # PUMP: несколько свечей подряд растут >3%
        if all(c > 3 for c in changes):
            return 'PUMP'
        
        # DUMP: несколько свечей подряд падают >3%
        if all(c < -3 for c in changes):
            return 'DUMP'
        
        # Иначе - манипуляция (резкие изменения в разные стороны)
        return 'MANIPULATION'
    
    def train(self, training_data: List[List[dict]]) -> bool:
        """
        Обучает модель на исторических данных
        
        Args:
            training_data: Список списков свечей для обучения
        
        Returns:
            True если обучение успешно
        """
        if not training_data:
            logger.error(" Нет данных для обучения")
            return False
        
        logger.info(f" Начинаем обучение на {len(training_data)} примерах...")
        
        # Извлекаем признаки из всех примеров
        all_features = []
        for candles in training_data:
            features = self.extract_features(candles)
            if features is not None:
                all_features.append(features[0])
        
        if len(all_features) < 10:
            logger.error(f" Недостаточно данных: {len(all_features)} примеров")
            return False
        
        X = np.array(all_features)
        
        logger.info(f" Подготовлено {X.shape[0]} примеров с {X.shape[1]} признаками")
        
        # Нормализация
        try:
            X_scaled = self.scaler.fit_transform(X)
            logger.info(" ✅ Нормализация выполнена")
        except Exception as e:
            logger.error(f" ❌ Ошибка нормализации: {e}")
            return False
        
        # Обучение Isolation Forest
        try:
            self.model.fit(X_scaled)
            self.is_trained = True
            logger.info(" ✅ Модель успешно обучена")
            return True
        except Exception as e:
            logger.error(f" ❌ Ошибка обучения: {e}")
            return False
    
    def save_model(self, model_path: str, scaler_path: str):
        """
        Сохраняет обученную модель
        
        Args:
            model_path: Путь для сохранения модели
            scaler_path: Путь для сохранения scaler
        """
        if not self.is_trained:
            logger.warning(" Модель не обучена, нечего сохранять")
            return
        
        try:
            # Создаем директорию если не существует
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Сохраняем модель и scaler
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f" ✅ Модель сохранена: {model_path}")
            logger.info(f" ✅ Scaler сохранен: {scaler_path}")
        except Exception as e:
            logger.error(f" ❌ Ошибка сохранения: {e}")
    
    def load_model(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Загружает обученную модель
        
        Args:
            model_path: Путь к файлу модели
            scaler_path: Путь к файлу scaler (если None, пытается найти рядом)
        """
        if scaler_path is None:
            scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        
        try:
            # Загружаем модель
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f" ✅ Модель загружена: {model_path}")
                self.is_trained = True
            else:
                logger.warning(f" ⚠️ Файл модели не найден: {model_path}")
                return False
            
            # Загружаем scaler
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f" ✅ Scaler загружен: {scaler_path}")
            else:
                logger.warning(f" ⚠️ Файл scaler не найден: {scaler_path}")
                self.scaler = StandardScaler()
            
            return True
            
        except Exception as e:
            logger.error(f" ❌ Ошибка загрузки: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает статус детектора
        
        Returns:
            Словарь со статусом
        """
        return {
            'is_trained': self.is_trained,
            'contamination': self.contamination,
            'model_type': 'IsolationForest'
        }

