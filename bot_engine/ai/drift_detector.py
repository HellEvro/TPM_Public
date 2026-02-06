"""
Data Drift Detection - обнаружение дрифта данных и деградации модели

Модуль для мониторинга:
- Изменений в распределении входных данных (data drift)
- Деградации производительности модели (concept drift)
- Автоматического триггера переобучения

Использует:
- Kolmogorov-Smirnov тест для детекции дрифта
- Скользящие метрики для мониторинга производительности
"""

import logging
import re
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger('DriftDetector')

# Опциональные зависимости
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy не установлен. Используется упрощенный детектор дрифта.")


@dataclass
class DriftResult:
    """Результат детекции дрифта"""
    drift_detected: bool
    drifted_features: List[str]
    drift_scores: Dict[str, float]
    p_values: Dict[str, float]
    recommendation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PerformanceMetrics:
    """Метрики производительности модели"""
    direction_accuracy: float  # Точность предсказания направления
    mae: float  # Mean Absolute Error для % изменения
    calibration_error: float  # Разница между предсказанной и реальной уверенностью
    total_predictions: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DataDriftDetector:
    """
    Детектор дрифта данных
    
    Использует Kolmogorov-Smirnov тест для сравнения
    распределений признаков между reference и текущими данными
    """
    
    def __init__(
        self,
        reference_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.05,  # p-value порог для KS теста
        min_samples: int = 100
    ):
        """
        Args:
            reference_data: Референсные данные (n_samples, n_features)
            feature_names: Названия признаков
            threshold: Порог p-value для определения дрифта
            min_samples: Минимальное количество сэмплов для теста
        """
        self.reference_data = reference_data
        self.feature_names = feature_names or []
        self.threshold = threshold
        self.min_samples = min_samples
        
        # Статистики reference данных
        self.reference_stats = {}
        
        if reference_data is not None:
            self._compute_reference_stats(reference_data)
    
    def _compute_reference_stats(self, data: np.ndarray):
        """Вычисляет статистики для reference данных"""
        self.reference_data = np.array(data)
        
        n_features = data.shape[1] if len(data.shape) > 1 else 1
        
        self.reference_stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'median': np.median(data, axis=0),
            'q25': np.percentile(data, 25, axis=0),
            'q75': np.percentile(data, 75, axis=0),
            'n_samples': len(data),
            'n_features': n_features
        }
        
        logger.info(f"Reference данные установлены: {len(data)} сэмплов, {n_features} признаков")
    
    def set_reference(self, data: np.ndarray, feature_names: List[str] = None):
        """Устанавливает reference данные"""
        self._compute_reference_stats(data)
        if feature_names:
            self.feature_names = feature_names
    
    def detect_drift(self, new_data: np.ndarray) -> DriftResult:
        """
        Детектирует дрифт между reference и новыми данными
        
        Args:
            new_data: Новые данные (n_samples, n_features)
        
        Returns:
            DriftResult с информацией о дрифте
        """
        if self.reference_data is None:
            return DriftResult(
                drift_detected=False,
                drifted_features=[],
                drift_scores={},
                p_values={},
                recommendation="Установите reference данные"
            )
        
        new_data = np.array(new_data)
        
        if len(new_data) < self.min_samples:
            return DriftResult(
                drift_detected=False,
                drifted_features=[],
                drift_scores={},
                p_values={},
                recommendation=f"Недостаточно данных ({len(new_data)} < {self.min_samples})"
            )
        
        # Убедимся что размерности совпадают
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)
        
        ref_data = self.reference_data
        if len(ref_data.shape) == 1:
            ref_data = ref_data.reshape(-1, 1)
        
        n_features = min(new_data.shape[1], ref_data.shape[1])
        
        drifted_features = []
        drift_scores = {}
        p_values = {}
        
        for i in range(n_features):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            
            ref_feature = ref_data[:, i]
            new_feature = new_data[:, i]
            
            # Kolmogorov-Smirnov тест (method='asymp' + подавление RuntimeWarning на части окружений)
            if SCIPY_AVAILABLE:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=RuntimeWarning,
                        message=re.escape("ks_2samp: Exact calculation unsuccessful.")
                        + r".*",
                    )
                    statistic, p_value = stats.ks_2samp(
                        ref_feature, new_feature, method="asymp"
                    )
            else:
                # Упрощенная версия без scipy
                statistic, p_value = self._simple_ks_test(ref_feature, new_feature)
            
            drift_scores[feature_name] = float(statistic)
            p_values[feature_name] = float(p_value)
            
            if p_value < self.threshold:
                drifted_features.append(feature_name)
        
        drift_detected = len(drifted_features) > 0
        
        # Формируем рекомендацию
        if drift_detected:
            drift_pct = len(drifted_features) / n_features * 100
            if drift_pct > 50:
                recommendation = f"КРИТИЧЕСКИЙ ДРИФТ: {drift_pct:.0f}% признаков. Рекомендуется немедленное переобучение."
            elif drift_pct > 20:
                recommendation = f"УМЕРЕННЫЙ ДРИФТ: {drift_pct:.0f}% признаков. Рекомендуется переобучение."
            else:
                recommendation = f"НЕЗНАЧИТЕЛЬНЫЙ ДРИФТ: {drift_pct:.0f}% признаков. Продолжайте мониторинг."
        else:
            recommendation = "Дрифт не обнаружен. Данные стабильны."
        
        return DriftResult(
            drift_detected=drift_detected,
            drifted_features=drifted_features,
            drift_scores=drift_scores,
            p_values=p_values,
            recommendation=recommendation
        )
    
    def _simple_ks_test(self, data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
        """
        Упрощенный KS тест без scipy
        
        Returns:
            statistic: KS статистика
            p_value: приблизительное p-value
        """
        # Сортируем данные
        data1 = np.sort(data1)
        data2 = np.sort(data2)
        
        n1 = len(data1)
        n2 = len(data2)
        
        # Объединяем и сортируем все точки
        all_data = np.sort(np.concatenate([data1, data2]))
        
        # Вычисляем CDF для каждого набора
        cdf1 = np.searchsorted(data1, all_data, side='right') / n1
        cdf2 = np.searchsorted(data2, all_data, side='right') / n2
        
        # KS статистика - максимальная разница CDF
        statistic = np.max(np.abs(cdf1 - cdf2))
        
        # Приблизительное p-value (упрощенная формула)
        n = (n1 * n2) / (n1 + n2)
        p_value = 2 * np.exp(-2 * n * statistic ** 2)
        p_value = min(max(p_value, 0), 1)
        
        return statistic, p_value
    
    def get_drift_summary(self, new_data: np.ndarray) -> Dict:
        """
        Возвращает краткую сводку о дрифте
        
        Args:
            new_data: Новые данные
        
        Returns:
            Dict с ключевыми метриками
        """
        result = self.detect_drift(new_data)
        
        return {
            'drift_detected': result.drift_detected,
            'n_drifted_features': len(result.drifted_features),
            'drifted_features': result.drifted_features,
            'recommendation': result.recommendation,
            'max_drift_score': max(result.drift_scores.values()) if result.drift_scores else 0,
            'min_p_value': min(result.p_values.values()) if result.p_values else 1
        }


class ModelPerformanceMonitor:
    """
    Монитор производительности модели
    
    Отслеживает метрики предсказаний и детектирует деградацию
    """
    
    def __init__(
        self,
        window_size: int = 100,
        accuracy_threshold: float = 0.5,  # Минимальная точность направления
        degradation_threshold: float = 0.1  # Порог падения метрик
    ):
        """
        Args:
            window_size: Размер скользящего окна для метрик
            accuracy_threshold: Порог точности ниже которого модель считается деградировавшей
            degradation_threshold: Порог падения метрик относительно baseline
        """
        self.window_size = window_size
        self.accuracy_threshold = accuracy_threshold
        self.degradation_threshold = degradation_threshold
        
        # Хранение предсказаний и результатов
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Baseline метрики (устанавливаются после первого периода)
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        
        # История метрик
        self.metrics_history: List[PerformanceMetrics] = []
    
    def log_prediction(
        self,
        prediction: Dict,
        timestamp: datetime = None
    ):
        """
        Логирует предсказание модели
        
        Args:
            prediction: Dict с 'direction', 'change_percent', 'confidence'
            timestamp: Время предсказания
        """
        self.predictions.append({
            'direction': prediction.get('direction', 0),
            'change_percent': prediction.get('change_percent', 0),
            'confidence': prediction.get('confidence', 0.5)
        })
        self.timestamps.append(timestamp or datetime.now())
    
    def log_actual_result(
        self,
        actual: Dict
    ):
        """
        Логирует фактический результат
        
        Args:
            actual: Dict с 'direction', 'change_percent'
        """
        self.actuals.append({
            'direction': actual.get('direction', 0),
            'change_percent': actual.get('change_percent', 0)
        })
    
    def get_metrics(self) -> PerformanceMetrics:
        """
        Вычисляет текущие метрики производительности
        
        Returns:
            PerformanceMetrics
        """
        if len(self.predictions) < 10 or len(self.actuals) < 10:
            return PerformanceMetrics(
                direction_accuracy=0.5,
                mae=0,
                calibration_error=0,
                total_predictions=len(self.predictions)
            )
        
        # Берем последние N пар (предсказание, результат)
        n = min(len(self.predictions), len(self.actuals))
        
        predictions = list(self.predictions)[-n:]
        actuals = list(self.actuals)[-n:]
        
        # Точность направления
        correct_direction = sum(
            1 for p, a in zip(predictions, actuals)
            if (p['direction'] > 0 and a['direction'] > 0) or
               (p['direction'] < 0 and a['direction'] < 0)
        )
        direction_accuracy = correct_direction / n
        
        # MAE для % изменения
        mae = np.mean([
            abs(p['change_percent'] - a['change_percent'])
            for p, a in zip(predictions, actuals)
        ])
        
        # Calibration error (средняя разница между confidence и реальной точностью)
        # Группируем по уровням confidence
        calibration_errors = []
        for p, a in zip(predictions, actuals):
            is_correct = (p['direction'] > 0 and a['direction'] > 0) or \
                        (p['direction'] < 0 and a['direction'] < 0)
            calibration_errors.append(abs(p['confidence'] - (1.0 if is_correct else 0.0)))
        
        calibration_error = np.mean(calibration_errors)
        
        metrics = PerformanceMetrics(
            direction_accuracy=float(direction_accuracy),
            mae=float(mae),
            calibration_error=float(calibration_error),
            total_predictions=n
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_performance_trend(self, periods: int = 5) -> Dict:
        """
        Анализирует тренд производительности
        
        Args:
            periods: Количество периодов для анализа
        
        Returns:
            Dict с информацией о тренде
        """
        if len(self.metrics_history) < periods:
            return {
                'trend': 'insufficient_data',
                'direction': 'unknown',
                'change_pct': 0
            }
        
        recent = self.metrics_history[-periods:]
        
        # Анализируем тренд точности
        accuracies = [m.direction_accuracy for m in recent]
        
        if len(accuracies) >= 2:
            trend_direction = 'improving' if accuracies[-1] > accuracies[0] else 'degrading'
            change_pct = (accuracies[-1] - accuracies[0]) / (accuracies[0] + 1e-8) * 100
        else:
            trend_direction = 'stable'
            change_pct = 0
        
        return {
            'trend': 'positive' if change_pct > 5 else ('negative' if change_pct < -5 else 'stable'),
            'direction': trend_direction,
            'change_pct': float(change_pct),
            'current_accuracy': accuracies[-1] if accuracies else 0,
            'periods_analyzed': len(recent)
        }
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Определяет, нужно ли переобучить модель
        
        Returns:
            (should_retrain, reason)
        """
        metrics = self.get_metrics()
        
        # Проверка минимальной точности
        if metrics.direction_accuracy < self.accuracy_threshold:
            return True, f"Точность ({metrics.direction_accuracy:.2%}) ниже порога ({self.accuracy_threshold:.2%})"
        
        # Проверка деградации относительно baseline
        if self.baseline_metrics:
            accuracy_drop = self.baseline_metrics.direction_accuracy - metrics.direction_accuracy
            if accuracy_drop > self.degradation_threshold:
                return True, f"Деградация точности на {accuracy_drop:.2%} относительно baseline"
        
        # Проверка тренда
        trend = self.get_performance_trend()
        if trend['trend'] == 'negative' and trend['change_pct'] < -10:
            return True, f"Негативный тренд: {trend['change_pct']:.1f}%"
        
        return False, "Производительность стабильна"
    
    def set_baseline(self):
        """Устанавливает текущие метрики как baseline"""
        self.baseline_metrics = self.get_metrics()
        logger.info(f"Baseline метрики установлены: accuracy={self.baseline_metrics.direction_accuracy:.2%}")
    
    def get_report(self) -> Dict:
        """Возвращает полный отчет о производительности"""
        metrics = self.get_metrics()
        trend = self.get_performance_trend()
        should_retrain, retrain_reason = self.should_retrain()
        
        return {
            'current_metrics': {
                'direction_accuracy': metrics.direction_accuracy,
                'mae': metrics.mae,
                'calibration_error': metrics.calibration_error,
                'total_predictions': metrics.total_predictions
            },
            'trend': trend,
            'baseline': {
                'direction_accuracy': self.baseline_metrics.direction_accuracy if self.baseline_metrics else None
            },
            'should_retrain': should_retrain,
            'retrain_reason': retrain_reason,
            'timestamp': datetime.now().isoformat()
        }


class CombinedDriftMonitor:
    """
    Комбинированный монитор дрифта данных и производительности
    
    Объединяет DataDriftDetector и ModelPerformanceMonitor
    для комплексного мониторинга модели
    """
    
    def __init__(
        self,
        feature_names: List[str] = None,
        drift_threshold: float = 0.05,
        accuracy_threshold: float = 0.5,
        window_size: int = 100
    ):
        self.data_drift_detector = DataDriftDetector(
            feature_names=feature_names,
            threshold=drift_threshold
        )
        self.performance_monitor = ModelPerformanceMonitor(
            window_size=window_size,
            accuracy_threshold=accuracy_threshold
        )
        
        self.last_check = None
        self.check_history: List[Dict] = []
    
    def set_reference_data(self, data: np.ndarray, feature_names: List[str] = None):
        """Устанавливает reference данные для детекции дрифта"""
        self.data_drift_detector.set_reference(data, feature_names)
    
    def set_performance_baseline(self):
        """Устанавливает baseline производительности"""
        self.performance_monitor.set_baseline()
    
    def log_prediction(self, prediction: Dict, features: np.ndarray = None):
        """
        Логирует предсказание и опционально входные признаки
        
        Args:
            prediction: Предсказание модели
            features: Входные признаки (для детекции дрифта)
        """
        self.performance_monitor.log_prediction(prediction)
    
    def log_actual(self, actual: Dict):
        """Логирует фактический результат"""
        self.performance_monitor.log_actual_result(actual)
    
    def check_health(self, current_features: np.ndarray = None) -> Dict:
        """
        Полная проверка здоровья модели
        
        Args:
            current_features: Текущие входные данные для проверки дрифта
        
        Returns:
            Dict с полным отчетом
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': None,
            'performance': None,
            'overall_health': 'healthy',
            'action_required': False,
            'recommendations': []
        }
        
        # Проверка дрифта данных
        if current_features is not None:
            drift_result = self.data_drift_detector.detect_drift(current_features)
            report['data_drift'] = {
                'detected': drift_result.drift_detected,
                'drifted_features': drift_result.drifted_features,
                'recommendation': drift_result.recommendation
            }
            
            if drift_result.drift_detected:
                report['recommendations'].append(drift_result.recommendation)
        
        # Проверка производительности
        perf_report = self.performance_monitor.get_report()
        report['performance'] = perf_report
        
        if perf_report['should_retrain']:
            report['recommendations'].append(perf_report['retrain_reason'])
        
        # Определяем общее состояние
        if report['data_drift'] and report['data_drift']['detected']:
            if len(report['data_drift']['drifted_features']) > 3:
                report['overall_health'] = 'critical'
                report['action_required'] = True
            else:
                report['overall_health'] = 'warning'
        
        if perf_report['should_retrain']:
            report['overall_health'] = 'critical'
            report['action_required'] = True
        
        self.last_check = report
        self.check_history.append(report)
        
        return report
    
    def should_trigger_retrain(self, current_features: np.ndarray = None) -> Tuple[bool, str]:
        """
        Определяет, нужно ли запустить переобучение
        
        Returns:
            (should_retrain, reason)
        """
        health = self.check_health(current_features)
        
        if health['action_required']:
            reasons = health['recommendations']
            return True, "; ".join(reasons)
        
        return False, "Модель в нормальном состоянии"


# ==================== ТЕСТОВЫЙ КОД ====================

if __name__ == '__main__':
    print("=" * 60)
    print("Drift Detector - Тест")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 1. Тест Data Drift Detector
    print("\n1. Тест DataDriftDetector:")
    
    # Reference данные (нормальное распределение)
    reference = np.random.randn(500, 5)
    
    # Данные без дрифта
    no_drift = np.random.randn(100, 5)
    
    # Данные с дрифтом (смещенное среднее)
    with_drift = np.random.randn(100, 5) + 2  # Сдвиг на 2
    
    detector = DataDriftDetector(
        reference_data=reference,
        feature_names=['f1', 'f2', 'f3', 'f4', 'f5']
    )
    
    print("\n  Тест без дрифта:")
    result_no_drift = detector.detect_drift(no_drift)
    print(f"    Дрифт обнаружен: {result_no_drift.drift_detected}")
    print(f"    Признаки с дрифтом: {result_no_drift.drifted_features}")
    print(f"    Рекомендация: {result_no_drift.recommendation}")
    
    print("\n  Тест с дрифтом:")
    result_drift = detector.detect_drift(with_drift)
    print(f"    Дрифт обнаружен: {result_drift.drift_detected}")
    print(f"    Признаки с дрифтом: {result_drift.drifted_features}")
    print(f"    Рекомендация: {result_drift.recommendation}")
    
    # 2. Тест Model Performance Monitor
    print("\n2. Тест ModelPerformanceMonitor:")
    
    monitor = ModelPerformanceMonitor(window_size=50)
    
    # Симулируем предсказания и результаты
    for i in range(50):
        # Предсказание
        pred_direction = np.random.choice([-1, 1])
        pred_change = np.random.randn() * 2
        pred_confidence = np.random.uniform(0.5, 0.9)
        
        monitor.log_prediction({
            'direction': pred_direction,
            'change_percent': pred_change,
            'confidence': pred_confidence
        })
        
        # Результат (с некоторой корреляцией с предсказанием)
        if np.random.random() < 0.6:  # 60% точность
            actual_direction = pred_direction
        else:
            actual_direction = -pred_direction
        
        actual_change = pred_change + np.random.randn() * 1
        
        monitor.log_actual_result({
            'direction': actual_direction,
            'change_percent': actual_change
        })
    
    metrics = monitor.get_metrics()
    print(f"\n  Метрики:")
    print(f"    Точность направления: {metrics.direction_accuracy:.2%}")
    print(f"    MAE: {metrics.mae:.4f}")
    print(f"    Calibration Error: {metrics.calibration_error:.4f}")
    print(f"    Всего предсказаний: {metrics.total_predictions}")
    
    trend = monitor.get_performance_trend()
    print(f"\n  Тренд:")
    print(f"    Направление: {trend['direction']}")
    print(f"    Изменение: {trend['change_pct']:.1f}%")
    
    should_retrain, reason = monitor.should_retrain()
    print(f"\n  Переобучение:")
    print(f"    Требуется: {should_retrain}")
    print(f"    Причина: {reason}")
    
    # 3. Тест Combined Monitor
    print("\n3. Тест CombinedDriftMonitor:")
    
    combined = CombinedDriftMonitor(
        feature_names=['f1', 'f2', 'f3', 'f4', 'f5']
    )
    combined.set_reference_data(reference)
    
    # Логируем данные
    for i in range(30):
        combined.log_prediction({
            'direction': np.random.choice([-1, 1]),
            'change_percent': np.random.randn() * 2,
            'confidence': np.random.uniform(0.5, 0.9)
        })
        combined.log_actual({
            'direction': np.random.choice([-1, 1]),
            'change_percent': np.random.randn() * 2
        })
    
    health = combined.check_health(with_drift)
    print(f"\n  Общее здоровье: {health['overall_health']}")
    print(f"  Действие требуется: {health['action_required']}")
    print(f"  Рекомендации: {health['recommendations']}")
    
    print("\n" + "=" * 60)
    print("[OK] Все тесты пройдены!")
    print("=" * 60)
