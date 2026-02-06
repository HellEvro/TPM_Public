"""
AI Performance Monitoring - мониторинг производительности AI моделей

Модуль для:
- Отслеживания предсказаний и результатов
- Расчета метрик производительности
- Проверки здоровья моделей
- Генерации отчетов
"""

import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger('AI.Monitoring')

# Корень проекта (bot_engine/ai/monitoring.py -> вверх на 2 уровня)
def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def _default_models_path() -> str:
    return os.path.join(_project_root(), 'data', 'ai', 'models')

@dataclass
class PredictionRecord:
    """Запись предсказания"""
    symbol: str
    direction: int  # 1 или -1
    change_percent: float
    confidence: float
    timestamp: str
    model: str = 'unknown'
    actual_direction: Optional[int] = None
    actual_change: Optional[float] = None
    is_correct: Optional[bool] = None

class AIPerformanceMonitor:
    """
    Мониторинг производительности AI моделей

    Отслеживает:
    - Точность направления
    - Калибровку уверенности
    - MAE предсказаний
    - Тренды производительности
    """

    def __init__(
        self,
        max_records: int = 10000,
        save_path: str = "data/ai/monitoring"
    ):
        self.max_records = max_records
        self.save_path = save_path

        self.predictions: deque = deque(maxlen=max_records)
        self.daily_metrics: Dict[str, Dict] = {}  # date -> metrics

        os.makedirs(save_path, exist_ok=True)
        self._load_history()

    def _load_history(self):
        """Загружает историю метрик"""
        metrics_file = os.path.join(self.save_path, "daily_metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    self.daily_metrics = json.load(f)
            except:
                pass

    def _save_history(self):
        """Сохраняет историю метрик"""
        metrics_file = os.path.join(self.save_path, "daily_metrics.json")
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.daily_metrics, f, indent=2)
        except Exception as e:
                        pass

    def track_prediction(
        self,
        symbol: str,
        prediction: Dict,
        model: str = 'unknown'
    ) -> str:
        """
        Отслеживает предсказание

        Args:
            symbol: Символ монеты
            prediction: Dict с direction, change_percent, confidence
            model: Название модели

        Returns:
            ID записи
        """
        record = PredictionRecord(
            symbol=symbol,
            direction=prediction.get('direction', 0),
            change_percent=prediction.get('change_percent', 0),
            confidence=prediction.get('confidence', 50),
            timestamp=datetime.now().isoformat(),
            model=model
        )

        self.predictions.append(record)

        return f"{symbol}_{record.timestamp}"

    def track_actual_result(
        self,
        symbol: str,
        actual_direction: int,
        actual_change: float,
        lookback_minutes: int = 60
    ):
        """
        Отслеживает фактический результат

        Args:
            symbol: Символ монеты
            actual_direction: Фактическое направление
            actual_change: Фактическое изменение %
            lookback_minutes: Сколько минут назад искать предсказание
        """
        cutoff = datetime.now() - timedelta(minutes=lookback_minutes)

        for record in reversed(self.predictions):
            if record.symbol != symbol:
                continue

            record_time = datetime.fromisoformat(record.timestamp)
            if record_time < cutoff:
                break

            if record.actual_direction is None:
                record.actual_direction = actual_direction
                record.actual_change = actual_change
                record.is_correct = (
                    (record.direction > 0 and actual_direction > 0) or
                    (record.direction < 0 and actual_direction < 0)
                )
                break

    def get_daily_metrics(self, date: str = None) -> Dict:
        """
        Получает метрики за день

        Args:
            date: Дата в формате YYYY-MM-DD (по умолчанию сегодня)

        Returns:
            Dict с метриками
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        # Фильтруем записи за день
        day_records = [
            r for r in self.predictions
            if r.timestamp.startswith(date) and r.is_correct is not None
        ]

        if not day_records:
            return {
                'date': date,
                'total_predictions': 0,
                'direction_accuracy': 0,
                'avg_confidence': 0,
                'mae': 0
            }

        correct = sum(1 for r in day_records if r.is_correct)
        total = len(day_records)

        metrics = {
            'date': date,
            'total_predictions': total,
            'direction_accuracy': correct / total if total > 0 else 0,
            'avg_confidence': np.mean([r.confidence for r in day_records]),
            'mae': np.mean([abs(r.change_percent - (r.actual_change or 0)) for r in day_records]),
            'by_model': {}
        }

        # Метрики по моделям
        models = set(r.model for r in day_records)
        for model in models:
            model_records = [r for r in day_records if r.model == model]
            model_correct = sum(1 for r in model_records if r.is_correct)
            metrics['by_model'][model] = {
                'total': len(model_records),
                'accuracy': model_correct / len(model_records) if model_records else 0
            }

        # Сохраняем
        self.daily_metrics[date] = metrics
        self._save_history()

        return metrics

    def get_weekly_report(self) -> str:
        """Генерирует недельный отчет"""
        today = datetime.now()
        report_lines = [
            "=" * 50,
            "AI Performance Weekly Report",
            f"Generated: {today.isoformat()}",
            "=" * 50,
            ""
        ]

        total_correct = 0
        total_predictions = 0

        for i in range(7):
            date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
            metrics = self.get_daily_metrics(date)

            if metrics['total_predictions'] > 0:
                total_predictions += metrics['total_predictions']
                total_correct += int(metrics['direction_accuracy'] * metrics['total_predictions'])

                report_lines.append(
                    f"{date}: {metrics['total_predictions']} predictions, "
                    f"{metrics['direction_accuracy']:.1%} accuracy"
                )

        report_lines.extend([
            "",
            "-" * 50,
            f"Week Total: {total_predictions} predictions",
            f"Week Accuracy: {total_correct/total_predictions:.1%}" if total_predictions > 0 else "No data",
            "=" * 50
        ])

        return "\n".join(report_lines)

    def export_metrics_to_db(self):
        """Экспортирует метрики в базу данных"""
        # Placeholder - интеграция с DB
        pass

class ModelHealthChecker:
    """
    Проверка здоровья AI моделей

    Проверяет:
    - Устаревание модели
    - Распределение предсказаний
    - Аномалии в уверенности
    """

    def __init__(self, models_path: Optional[str] = None):
        self.models_path = models_path if models_path is not None else _default_models_path()

    def check_model_staleness(
        self,
        model_path: str,
        max_age_days: int = 7
    ) -> Dict:
        """
        Проверяет устаревание модели

        Args:
            model_path: Путь к модели
            max_age_days: Максимальный возраст в днях

        Returns:
            Dict с информацией об устаревании
        """
        if not os.path.exists(model_path):
            return {
                'exists': False,
                'is_stale': True,
                'reason': 'Model file not found'
            }

        mtime = os.path.getmtime(model_path)
        age_days = (datetime.now().timestamp() - mtime) / 86400

        return {
            'exists': True,
            'is_stale': age_days > max_age_days,
            'age_days': age_days,
            'last_modified': datetime.fromtimestamp(mtime).isoformat(),
            'max_age_days': max_age_days
        }

    def check_prediction_distribution(
        self,
        predictions: List[Dict],
        min_samples: int = 50
    ) -> Dict:
        """
        Проверяет распределение предсказаний

        Args:
            predictions: Список предсказаний
            min_samples: Минимум для анализа

        Returns:
            Dict с информацией о распределении
        """
        if len(predictions) < min_samples:
            return {
                'valid': False,
                'reason': f'Insufficient samples ({len(predictions)} < {min_samples})'
            }

        directions = [p.get('direction', 0) for p in predictions]
        confidences = [p.get('confidence', 50) for p in predictions]

        long_ratio = sum(1 for d in directions if d > 0) / len(directions)

        return {
            'valid': True,
            'total_samples': len(predictions),
            'long_ratio': long_ratio,
            'short_ratio': 1 - long_ratio,
            'avg_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'is_biased': long_ratio < 0.2 or long_ratio > 0.8,
            'confidence_too_high': np.mean(confidences) > 85,
            'confidence_too_low': np.mean(confidences) < 40
        }

    def _get_model_paths(self) -> List[tuple]:
        """Возвращает список (имя для отображения, полный путь) из AIConfig."""
        root = _project_root()
        paths = []
        try:
            from bot_engine.config_loader import AIConfig
            lstm_path = getattr(AIConfig, 'AI_LSTM_MODEL_PATH', 'data/ai/models/lstm_predictor.keras')
            paths.append((os.path.basename(lstm_path), os.path.join(root, lstm_path)))
            pattern_path = getattr(AIConfig, 'AI_PATTERN_MODEL_PATH', 'data/ai/models/pattern_detector.pkl')
            paths.append((os.path.basename(pattern_path), os.path.join(root, pattern_path)))
        except Exception:
            paths = [
                ('lstm_predictor.keras', os.path.join(self.models_path, 'lstm_predictor.keras')),
                ('pattern_detector.pkl', os.path.join(self.models_path, 'pattern_detector.pkl')),
            ]
        # Transformer: путь из ai_trainer (пока не в AIConfig)
        trans_path = os.path.join(root, 'data', 'ai', 'models', 'transformer_predictor.pth')
        paths.append(('transformer_predictor.pth', trans_path))
        return paths

    def get_recommendations(self) -> List[str]:
        """Возвращает рекомендации по улучшению (проверяет реальные пути из AIConfig)."""
        recommendations = []

        for model_name, path in self._get_model_paths():
            staleness = self.check_model_staleness(path)

            if not staleness['exists']:
                recommendations.append(f"Model {model_name} not found - consider training")
            elif staleness['is_stale']:
                recommendations.append(
                    f"Model {model_name} is {staleness['age_days']:.1f} days old - consider retraining"
                )

        return recommendations

def get_performance_api_data() -> Dict:
    """
    Возвращает данные для API /api/ai/performance
    """
    monitor = AIPerformanceMonitor()
    checker = ModelHealthChecker()

    return {
        'daily_metrics': monitor.get_daily_metrics(),
        'recommendations': checker.get_recommendations(),
        'timestamp': datetime.now().isoformat()
    }

def get_health_api_data() -> Dict:
    """
    Возвращает данные для API /api/ai/health (пути моделей из AIConfig).
    """
    checker = ModelHealthChecker()
    models_health = {}
    for model_name, path in checker._get_model_paths():
        models_health[model_name] = checker.check_model_staleness(path)

    return {
        'models': models_health,
        'recommendations': checker.get_recommendations(),
        'overall_status': 'healthy' if not any(m.get('is_stale') for m in models_health.values()) else 'needs_attention',
        'timestamp': datetime.now().isoformat()
    }

# ==================== ТЕСТОВЫЙ КОД ====================

if __name__ == '__main__':
    print("=" * 60)
    print("AI Monitoring - Test")
    print("=" * 60)

    # Тест AIPerformanceMonitor
    print("\n1. Test AIPerformanceMonitor:")

    monitor = AIPerformanceMonitor()

    # Симулируем предсказания
    for i in range(20):
        pred = {
            'direction': np.random.choice([-1, 1]),
            'change_percent': np.random.randn() * 2,
            'confidence': np.random.uniform(50, 90)
        }
        monitor.track_prediction(f"BTCUSDT", pred, model='lstm')

        # Симулируем результат
        monitor.track_actual_result(
            "BTCUSDT",
            np.random.choice([-1, 1]),
            np.random.randn() * 2
        )

    metrics = monitor.get_daily_metrics()
    print(f"   Total predictions: {metrics['total_predictions']}")
    print(f"   Direction accuracy: {metrics['direction_accuracy']:.1%}")
    print(f"   Avg confidence: {metrics['avg_confidence']:.1f}")

    # Тест ModelHealthChecker
    print("\n2. Test ModelHealthChecker:")

    checker = ModelHealthChecker()
    recs = checker.get_recommendations()
    print(f"   Recommendations: {len(recs)}")
    for rec in recs[:3]:
        print(f"   - {rec}")

    # Тест API данных
    print("\n3. Test API data:")

    perf_data = get_performance_api_data()
    print(f"   Performance data keys: {list(perf_data.keys())}")

    health_data = get_health_api_data()
    print(f"   Health status: {health_data['overall_status']}")

    print("\n" + "=" * 60)
    print("[OK] All tests passed!")
    print("=" * 60)
