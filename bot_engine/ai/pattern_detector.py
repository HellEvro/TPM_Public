"""
Pattern Recognition Module для технического анализа криптовалют

Этот модуль распознает классические паттерны технического анализа:
- Head & Shoulders (Голова и плечи)
- Double Top/Bottom (Двойная вершина/дно)
- Triangle (Треугольник)
- Flag/Pennant (Флаг/Вымпел)
- Support/Resistance Breakout (Пробой уровней)
- Candlestick patterns (Свечные паттерны)

Использует комбинацию алгоритмов:
1. Геометрический анализ (поиск локальных экстремумов)
2. Статистический анализ (вероятность паттерна)
3. Machine Learning (классификация паттернов)
"""

import os
import logging
import pickle
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import linregress

logger = logging.getLogger('AI')

# Проверяем доступность scikit-learn
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("[PATTERN] scikit-learn не установлен. Pattern Recognition недоступен.")


class PatternDetector:
    """
    Детектор классических паттернов технического анализа
    """
    
    def __init__(
        self,
        model_path: str = "data/ai/models/pattern_detector.pkl",
        scaler_path: str = "data/ai/models/pattern_scaler.pkl"
    ):
        """
        Инициализация детектора паттернов
        
        Args:
            model_path: Путь к сохраненной ML модели
            scaler_path: Путь к scaler'у
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        self.model = None  # RandomForest для классификации
        self.scaler = None
        
        # Конфигурация паттернов
        self.config = {
            'lookback_period': 100,  # Сколько свечей анализировать
            'min_pattern_length': 10,  # Минимальная длина паттерна
            'confidence_threshold': 0.6,  # Минимальная уверенность
            'model_version': '1.0'
        }
        
        if not SKLEARN_AVAILABLE:
            logger.error("[PATTERN] scikit-learn недоступен")
            return
        
        # Загружаем модель, если существует
        if os.path.exists(model_path):
            self.load_model()
        else:
            self._create_new_model()
    
    def _create_new_model(self):
        """Создает новую модель для классификации паттернов"""
        if not SKLEARN_AVAILABLE:
            return
        
        # RandomForest для классификации паттернов
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        
        # Scaler для нормализации признаков
        self.scaler = StandardScaler()
        
        logger.info("[PATTERN] Создана новая модель для распознавания паттернов")
    
    def detect_patterns(
        self,
        candles: List[Dict],
        current_price: float
    ) -> Dict:
        """
        Распознает паттерны в свечных данных
        
        Args:
            candles: История свечей
            current_price: Текущая цена
        
        Returns:
            {
                'patterns': [список найденных паттернов],
                'signal': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
                'confidence': 0-100,
                'strongest_pattern': паттерн с наибольшей вероятностью
            }
        """
        if len(candles) < self.config['lookback_period']:
            return {
                'patterns': [],
                'signal': 'NEUTRAL',
                'confidence': 0,
                'strongest_pattern': None
            }
        
        # Берем последние N свечей
        recent_candles = candles[-self.config['lookback_period']:]
        
        # Извлекаем цены
        closes = np.array([c['close'] for c in recent_candles])
        highs = np.array([c['high'] for c in recent_candles])
        lows = np.array([c['low'] for c in recent_candles])
        opens = np.array([c['open'] for c in recent_candles])
        volumes = np.array([c.get('volume', 0) for c in recent_candles])
        
        # Ищем различные паттерны
        detected_patterns = []
        
        # 1. Head & Shoulders
        hs_pattern = self._detect_head_shoulders(closes, highs, lows)
        if hs_pattern:
            detected_patterns.append(hs_pattern)
        
        # 2. Double Top/Bottom
        double_pattern = self._detect_double_top_bottom(closes, highs, lows)
        if double_pattern:
            detected_patterns.append(double_pattern)
        
        # 3. Triangle
        triangle_pattern = self._detect_triangle(closes, highs, lows)
        if triangle_pattern:
            detected_patterns.append(triangle_pattern)
        
        # 4. Support/Resistance Breakout
        breakout_pattern = self._detect_breakout(closes, highs, lows, current_price)
        if breakout_pattern:
            detected_patterns.append(breakout_pattern)
        
        # 5. Candlestick Patterns
        candle_patterns = self._detect_candlestick_patterns(opens, closes, highs, lows)
        detected_patterns.extend(candle_patterns)
        
        # Определяем общий сигнал
        if not detected_patterns:
            return {
                'patterns': [],
                'signal': 'NEUTRAL',
                'confidence': 0,
                'strongest_pattern': None
            }
        
        # Вычисляем средний сигнал и уверенность
        bullish_count = sum(1 for p in detected_patterns if p['signal'] == 'BULLISH')
        bearish_count = sum(1 for p in detected_patterns if p['signal'] == 'BEARISH')
        
        avg_confidence = np.mean([p['confidence'] for p in detected_patterns])
        
        if bullish_count > bearish_count:
            signal = 'BULLISH'
        elif bearish_count > bullish_count:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'
        
        # Находим самый сильный паттерн
        strongest = max(detected_patterns, key=lambda x: x['confidence'])
        
        return {
            'patterns': detected_patterns,
            'signal': signal,
            'confidence': avg_confidence,
            'strongest_pattern': strongest
        }
    
    def _find_local_extrema(
        self,
        data: np.ndarray,
        order: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Находит локальные максимумы и минимумы
        
        Args:
            data: Массив цен
            order: Размер окна для поиска экстремумов
        
        Returns:
            (индексы максимумов, индексы минимумов)
        """
        maxima = argrelextrema(data, np.greater, order=order)[0]
        minima = argrelextrema(data, np.less, order=order)[0]
        
        return maxima, minima
    
    def _detect_head_shoulders(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> Optional[Dict]:
        """
        Распознает паттерн Head & Shoulders
        
        Структура: Левое плечо - Голова - Правое плечо
        Разворотный паттерн на вершине тренда
        """
        maxima, minima = self._find_local_extrema(closes, order=5)
        
        if len(maxima) < 3:
            return None
        
        # Берем последние 3 максимума
        last_3_maxima = maxima[-3:]
        
        # Проверяем паттерн H&S:
        # - Средний пик (голова) выше двух других (плечей)
        # - Плечи примерно на одном уровне
        left_shoulder = closes[last_3_maxima[0]]
        head = closes[last_3_maxima[1]]
        right_shoulder = closes[last_3_maxima[2]]
        
        # Голова должна быть выше плеч
        if head <= left_shoulder or head <= right_shoulder:
            return None
        
        # Плечи должны быть примерно одинаковы (±5%)
        shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
        if shoulder_diff > 0.05:
            return None
        
        # Вычисляем уверенность
        head_prominence = (head - max(left_shoulder, right_shoulder)) / head
        confidence = min(head_prominence * 100, 100)
        
        if confidence < 50:
            return None
        
        return {
            'name': 'Head & Shoulders',
            'type': 'reversal',
            'signal': 'BEARISH',  # H&S - медвежий паттерн
            'confidence': confidence,
            'description': f'Разворотный паттерн: голова {head:.2f}, плечи {left_shoulder:.2f}/{right_shoulder:.2f}'
        }
    
    def _detect_double_top_bottom(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> Optional[Dict]:
        """
        Распознает паттерн Double Top (двойная вершина) или Double Bottom (двойное дно)
        
        Разворотный паттерн
        """
        maxima, minima = self._find_local_extrema(closes, order=5)
        
        # Проверяем Double Top
        if len(maxima) >= 2:
            last_2_maxima = maxima[-2:]
            peak1 = closes[last_2_maxima[0]]
            peak2 = closes[last_2_maxima[1]]
            
            # Вершины должны быть примерно на одном уровне (±2%)
            peak_diff = abs(peak1 - peak2) / peak1
            if peak_diff < 0.02:
                confidence = (1 - peak_diff) * 100
                
                return {
                    'name': 'Double Top',
                    'type': 'reversal',
                    'signal': 'BEARISH',
                    'confidence': confidence,
                    'description': f'Двойная вершина: {peak1:.2f} и {peak2:.2f}'
                }
        
        # Проверяем Double Bottom
        if len(minima) >= 2:
            last_2_minima = minima[-2:]
            bottom1 = closes[last_2_minima[0]]
            bottom2 = closes[last_2_minima[1]]
            
            # Дно должно быть примерно на одном уровне (±2%)
            bottom_diff = abs(bottom1 - bottom2) / bottom1
            if bottom_diff < 0.02:
                confidence = (1 - bottom_diff) * 100
                
                return {
                    'name': 'Double Bottom',
                    'type': 'reversal',
                    'signal': 'BULLISH',
                    'confidence': confidence,
                    'description': f'Двойное дно: {bottom1:.2f} и {bottom2:.2f}'
                }
        
        return None
    
    def _detect_triangle(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> Optional[Dict]:
        """
        Распознает треугольник (сужающийся диапазон)
        
        Паттерн продолжения тренда или консолидации
        """
        if len(closes) < 30:
            return None
        
        # Берем последние 30 свечей
        recent_highs = highs[-30:]
        recent_lows = lows[-30:]
        
        # Вычисляем линейную регрессию для максимумов и минимумов
        x = np.arange(len(recent_highs))
        
        highs_slope, _, highs_r, _, _ = linregress(x, recent_highs)
        lows_slope, _, lows_r, _, _ = linregress(x, recent_lows)
        
        # Проверяем сходимость линий (треугольник)
        # Верхняя линия должна идти вниз, нижняя - вверх
        if highs_slope < -0.01 and lows_slope > 0.01:
            # Симметричный треугольник
            avg_r = (abs(highs_r) + abs(lows_r)) / 2
            confidence = avg_r * 100
            
            if confidence > 60:
                return {
                    'name': 'Symmetrical Triangle',
                    'type': 'continuation',
                    'signal': 'NEUTRAL',  # Ждем пробоя
                    'confidence': confidence,
                    'description': f'Симметричный треугольник (R²={avg_r:.2f})'
                }
        
        # Восходящий треугольник (бычий)
        elif abs(highs_slope) < 0.005 and lows_slope > 0.01:
            confidence = abs(lows_r) * 100
            
            if confidence > 60:
                return {
                    'name': 'Ascending Triangle',
                    'type': 'continuation',
                    'signal': 'BULLISH',
                    'confidence': confidence,
                    'description': f'Восходящий треугольник (поддержка растет)'
                }
        
        # Нисходящий треугольник (медвежий)
        elif highs_slope < -0.01 and abs(lows_slope) < 0.005:
            confidence = abs(highs_r) * 100
            
            if confidence > 60:
                return {
                    'name': 'Descending Triangle',
                    'type': 'continuation',
                    'signal': 'BEARISH',
                    'confidence': confidence,
                    'description': f'Нисходящий треугольник (сопротивление падает)'
                }
        
        return None
    
    def _detect_breakout(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        current_price: float
    ) -> Optional[Dict]:
        """
        Распознает пробой уровня поддержки/сопротивления
        """
        if len(closes) < 50:
            return None
        
        # Ищем уровни поддержки и сопротивления
        maxima, minima = self._find_local_extrema(closes, order=5)
        
        if len(maxima) < 2 or len(minima) < 2:
            return None
        
        # Уровень сопротивления (среднее по максимумам)
        resistance_level = np.mean(closes[maxima[-3:]])
        
        # Уровень поддержки (среднее по минимумам)
        support_level = np.mean(closes[minima[-3:]])
        
        # Проверяем пробой вверх
        if current_price > resistance_level * 1.005:  # +0.5% выше сопротивления
            breakout_strength = (current_price - resistance_level) / resistance_level * 100
            confidence = min(breakout_strength * 20, 100)
            
            if confidence > 50:
                return {
                    'name': 'Resistance Breakout',
                    'type': 'breakout',
                    'signal': 'BULLISH',
                    'confidence': confidence,
                    'description': f'Пробой сопротивления {resistance_level:.2f} → {current_price:.2f} (+{breakout_strength:.1f}%)'
                }
        
        # Проверяем пробой вниз
        elif current_price < support_level * 0.995:  # -0.5% ниже поддержки
            breakout_strength = (support_level - current_price) / support_level * 100
            confidence = min(breakout_strength * 20, 100)
            
            if confidence > 50:
                return {
                    'name': 'Support Breakdown',
                    'type': 'breakout',
                    'signal': 'BEARISH',
                    'confidence': confidence,
                    'description': f'Пробой поддержки {support_level:.2f} → {current_price:.2f} (-{breakout_strength:.1f}%)'
                }
        
        return None
    
    def _detect_candlestick_patterns(
        self,
        opens: np.ndarray,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> List[Dict]:
        """
        Распознает свечные паттерны
        
        Returns:
            Список найденных свечных паттернов
        """
        patterns = []
        
        if len(closes) < 3:
            return patterns
        
        # Берем последние 3 свечи
        o = opens[-3:]
        c = closes[-3:]
        h = highs[-3:]
        l = lows[-3:]
        
        # Последняя свеча
        last_open = o[-1]
        last_close = c[-1]
        last_high = h[-1]
        last_low = l[-1]
        
        body = abs(last_close - last_open)
        total_range = last_high - last_low
        
        if total_range == 0:
            return patterns
        
        # 1. Hammer (Молот) - бычий разворотный паттерн
        lower_shadow = min(last_open, last_close) - last_low
        upper_shadow = last_high - max(last_open, last_close)
        
        if body > 0 and lower_shadow > body * 2 and upper_shadow < body * 0.3:
            confidence = min((lower_shadow / body) * 30, 100)
            
            patterns.append({
                'name': 'Hammer',
                'type': 'candlestick',
                'signal': 'BULLISH',
                'confidence': confidence,
                'description': 'Молот: длинная нижняя тень, разворот вверх'
            })
        
        # 2. Shooting Star (Падающая звезда) - медвежий разворотный паттерн
        if body > 0 and upper_shadow > body * 2 and lower_shadow < body * 0.3:
            confidence = min((upper_shadow / body) * 30, 100)
            
            patterns.append({
                'name': 'Shooting Star',
                'type': 'candlestick',
                'signal': 'BEARISH',
                'confidence': confidence,
                'description': 'Падающая звезда: длинная верхняя тень, разворот вниз'
            })
        
        # 3. Bullish Engulfing (Бычье поглощение)
        if len(c) >= 2:
            prev_open = o[-2]
            prev_close = c[-2]
            
            # Предыдущая свеча медвежья, текущая бычья и поглощает предыдущую
            if prev_close < prev_open and last_close > last_open:
                if last_close > prev_open and last_open < prev_close:
                    engulfing_ratio = body / abs(prev_close - prev_open)
                    confidence = min(engulfing_ratio * 50, 100)
                    
                    if confidence > 60:
                        patterns.append({
                            'name': 'Bullish Engulfing',
                            'type': 'candlestick',
                            'signal': 'BULLISH',
                            'confidence': confidence,
                            'description': 'Бычье поглощение: сильный разворот вверх'
                        })
        
        # 4. Bearish Engulfing (Медвежье поглощение)
        if len(c) >= 2:
            prev_open = o[-2]
            prev_close = c[-2]
            
            # Предыдущая свеча бычья, текущая медвежья и поглощает предыдущую
            if prev_close > prev_open and last_close < last_open:
                if last_close < prev_open and last_open > prev_close:
                    engulfing_ratio = body / abs(prev_close - prev_open)
                    confidence = min(engulfing_ratio * 50, 100)
                    
                    if confidence > 60:
                        patterns.append({
                            'name': 'Bearish Engulfing',
                            'type': 'candlestick',
                            'signal': 'BEARISH',
                            'confidence': confidence,
                            'description': 'Медвежье поглощение: сильный разворот вниз'
                        })
        
        # 5. Doji (Доджи) - неопределенность
        if body / total_range < 0.1:  # Тело меньше 10% от диапазона
            confidence = (1 - (body / total_range)) * 80
            
            patterns.append({
                'name': 'Doji',
                'type': 'candlestick',
                'signal': 'NEUTRAL',
                'confidence': confidence,
                'description': 'Доджи: неопределенность, возможный разворот'
            })
        
        return patterns
    
    def get_pattern_signal(
        self,
        candles: List[Dict],
        current_price: float,
        signal_type: str  # 'LONG' или 'SHORT'
    ) -> Dict:
        """
        Получает сигнал от паттернов для конкретного типа сделки
        
        Args:
            candles: История свечей
            current_price: Текущая цена
            signal_type: Тип сигнала ('LONG' или 'SHORT')
        
        Returns:
            {
                'confirmation': True/False - подтверждает ли паттерн сигнал,
                'confidence': уверенность,
                'patterns_found': количество найденных паттернов,
                'details': детали паттернов
            }
        """
        result = self.detect_patterns(candles, current_price)
        
        # Определяем совместимость паттернов с сигналом
        if signal_type == 'LONG':
            confirmation = result['signal'] == 'BULLISH'
        elif signal_type == 'SHORT':
            confirmation = result['signal'] == 'BEARISH'
        else:
            confirmation = False
        
        return {
            'confirmation': confirmation,
            'confidence': result['confidence'],
            'patterns_found': len(result['patterns']),
            'signal': result['signal'],
            'strongest_pattern': result['strongest_pattern'],
            'all_patterns': result['patterns']
        }
    
    def train(
        self,
        training_data: List[Tuple[np.ndarray, str]],
        validation_split: float = 0.2
    ) -> Dict:
        """
        Обучает ML модель для классификации паттернов
        
        Args:
            training_data: Список (признаки, метка_паттерна)
            validation_split: Доля данных для валидации
        
        Returns:
            Результаты обучения
        """
        if not SKLEARN_AVAILABLE or self.model is None:
            return {'success': False, 'error': 'sklearn unavailable'}
        
        try:
            X_list, y_list = zip(*training_data)
            X = np.array(X_list)
            y = np.array(y_list)
            
            # Нормализуем признаки
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Разделяем на train/val
            split_idx = int(len(X_scaled) * (1 - validation_split))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Обучаем модель
            logger.info(f"[PATTERN] Начало обучения: {len(X_train)} образцов")
            self.model.fit(X_train, y_train)
            
            # Проверяем точность
            train_score = self.model.score(X_train, y_train)
            val_score = self.model.score(X_val, y_val)
            
            logger.info(f"[PATTERN] Train accuracy: {train_score:.3f}")
            logger.info(f"[PATTERN] Val accuracy: {val_score:.3f}")
            
            # Сохраняем модель
            self.save_model()
            
            return {
                'success': True,
                'train_accuracy': train_score,
                'val_accuracy': val_score,
                'training_samples': len(X_train)
            }
            
        except Exception as e:
            logger.error(f"[PATTERN] Ошибка обучения: {e}")
            return {'success': False, 'error': str(e)}
    
    def load_model(self):
        """Загружает обученную модель из файла"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                logger.info(f"[PATTERN] Модель загружена: {self.model_path}")
            
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                logger.info(f"[PATTERN] Scaler загружен: {self.scaler_path}")
            
            if self.model is None:
                logger.warning("[PATTERN] Модель не найдена, создаем новую")
                self._create_new_model()
        
        except Exception as e:
            logger.error(f"[PATTERN] Ошибка загрузки модели: {e}")
            self._create_new_model()
    
    def save_model(self):
        """Сохраняет модель в файл"""
        try:
            # Создаем директорию, если не существует
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Сохраняем модель
            if self.model is not None:
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                
                logger.info(f"[PATTERN] Модель сохранена: {self.model_path}")
            
            # Сохраняем scaler
            if self.scaler is not None:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                
                logger.info(f"[PATTERN] Scaler сохранен: {self.scaler_path}")
        
        except Exception as e:
            logger.error(f"[PATTERN] Ошибка сохранения модели: {e}")
    
    def get_status(self) -> Dict:
        """Возвращает статус модели"""
        return {
            'is_trained': self.model is not None and hasattr(self.model, 'estimators_'),
            'model_type': 'RandomForestClassifier',
            'lookback_period': self.config['lookback_period'],
            'confidence_threshold': self.config['confidence_threshold']
        }

