"""
Оптимизация threshold для Anomaly Detector

Этот скрипт помогает подобрать оптимальный AI_ANOMALY_BLOCK_THRESHOLD
на основе исторических данных или симуляции.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь (устойчиво к запуску из любого cwd)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
os.environ['AI_DEV_MODE'] = '1'

# ВАЖНО: до любых импортов sklearn — подавляет UserWarning delayed/Parallel и фиксирует joblib
import utils.sklearn_parallel_config  # noqa: F401

from bot_engine.ai.anomaly_detector import AnomalyDetector
import numpy as np


def generate_test_data(num_normal=100, num_pump=20, num_dump=20):
    """Генерирует тестовые данные"""
    
    def generate_normal_candles(count=50):
        """Нормальные свечи"""
        candles = []
        base_price = 1000.0
        base_volume = 100000.0
        
        for i in range(count):
            change = np.random.normal(0, 0.5)
            base_price *= (1 + change / 100)
            
            candles.append({
                'close': base_price,
                'high': base_price * 1.01,
                'low': base_price * 0.99,
                'volume': base_volume * (1 + np.random.uniform(-0.2, 0.2))
            })
        
        return candles
    
    def generate_pump_candles(count=50):
        """PUMP свечи"""
        candles = generate_normal_candles(40)
        base_price = candles[-1]['close']
        
        for i in range(10):
            change = np.random.uniform(5, 8)
            base_price *= (1 + change / 100)
            
            candles.append({
                'close': base_price,
                'high': base_price * 1.02,
                'low': base_price * 0.98,
                'volume': candles[-1]['volume'] * np.random.uniform(2, 4)
            })
        
        return candles
    
    def generate_dump_candles(count=50):
        """DUMP свечи"""
        candles = generate_normal_candles(40)
        base_price = candles[-1]['close']
        
        for i in range(10):
            change = np.random.uniform(-8, -5)
            base_price *= (1 + change / 100)
            
            candles.append({
                'close': base_price,
                'high': base_price * 1.02,
                'low': base_price * 0.98,
                'volume': candles[-1]['volume'] * np.random.uniform(2, 4)
            })
        
        return candles
    
    # Генерируем данные
    test_data = []
    
    print(f"Generating test data...")
    print(f"  - Normal: {num_normal} examples")
    print(f"  - PUMP: {num_pump} examples")
    print(f"  - DUMP: {num_dump} examples")
    
    for _ in range(num_normal):
        test_data.append(('NORMAL', generate_normal_candles(50)))
    
    for _ in range(num_pump):
        test_data.append(('PUMP', generate_pump_candles(50)))
    
    for _ in range(num_dump):
        test_data.append(('DUMP', generate_dump_candles(50)))
    
    return test_data


def test_threshold(detector, test_data, threshold):
    """Тестирует detector с заданным threshold"""
    
    results = {
        'NORMAL': {'correct': 0, 'total': 0, 'blocked': 0},
        'PUMP': {'correct': 0, 'total': 0, 'blocked': 0},
        'DUMP': {'correct': 0, 'total': 0, 'blocked': 0}
    }
    
    for label, candles in test_data:
        result = detector.detect(candles)
        
        is_blocked = False
        if result.get('is_anomaly'):
            severity = result.get('severity', 0)
            if severity > threshold:
                is_blocked = True
        
        results[label]['total'] += 1
        
        if label == 'NORMAL':
            # Для нормальных свечей правильно = НЕ блокировать
            if not is_blocked:
                results[label]['correct'] += 1
            else:
                results[label]['blocked'] += 1
        else:
            # Для PUMP/DUMP правильно = БЛОКИРОВАТЬ
            if is_blocked:
                results[label]['correct'] += 1
            results[label]['blocked'] += 1
    
    # Вычисляем метрики
    total_correct = sum(r['correct'] for r in results.values())
    total = sum(r['total'] for r in results.values())
    accuracy = total_correct / total if total > 0 else 0
    
    # False Positive Rate (нормальные, но заблокированные)
    fpr = results['NORMAL']['blocked'] / results['NORMAL']['total'] if results['NORMAL']['total'] > 0 else 0
    
    # True Positive Rate (аномалии заблокированы)
    anomalies_total = results['PUMP']['total'] + results['DUMP']['total']
    anomalies_blocked = results['PUMP']['blocked'] + results['DUMP']['blocked']
    tpr = anomalies_blocked / anomalies_total if anomalies_total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'fpr': fpr,  # False Positive Rate (ложные срабатывания)
        'tpr': tpr,  # True Positive Rate (обнаруженные аномалии)
        'results': results
    }


def optimize_threshold():
    """Подбирает оптимальный threshold"""
    
    print("=" * 60)
    print("THRESHOLD OPTIMIZATION FOR ANOMALY DETECTOR")
    print("=" * 60)
    print()
    
    # Создаем детектор
    print("Creating detector...")
    detector = AnomalyDetector()
    print(f"Status: {detector.get_status()}")
    print()
    
    # Генерируем тестовые данные
    print("Generating test data...")
    test_data = generate_test_data(
        num_normal=100,
        num_pump=30,
        num_dump=30
    )
    print(f"Total examples: {len(test_data)}")
    print()
    
    # Тестируем разные threshold
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("Testing different thresholds...")
    print("-" * 60)
    print(f"{'Threshold':<12} {'Accuracy':<12} {'FPR':<12} {'TPR':<12} {'Score'}")
    print("-" * 60)
    
    best_threshold = None
    best_score = 0
    all_results = []
    
    for threshold in thresholds:
        metrics = test_threshold(detector, test_data, threshold)
        
        # Вычисляем общий score
        # Score = Accuracy - FPR * 0.5 (штраф за ложные срабатывания)
        score = metrics['accuracy'] - metrics['fpr'] * 0.5
        
        all_results.append({
            'threshold': threshold,
            'metrics': metrics,
            'score': score
        })
        
        print(f"{threshold:<12.2f} {metrics['accuracy']:<12.1%} {metrics['fpr']:<12.1%} {metrics['tpr']:<12.1%} {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    print("-" * 60)
    print()
    
    # Детальная статистика для лучшего threshold
    print(f"RECOMMENDED THRESHOLD: {best_threshold:.2f}")
    print(f"Score: {best_score:.3f}")
    print()
    
    best_result = next(r for r in all_results if r['threshold'] == best_threshold)
    best_metrics = best_result['metrics']
    
    print("Detailed statistics:")
    print(f"  Accuracy: {best_metrics['accuracy']:.1%}")
    print(f"  False Positive Rate: {best_metrics['fpr']:.1%} (normal candles blocked)")
    print(f"  True Positive Rate: {best_metrics['tpr']:.1%} (anomalies detected)")
    print()
    
    print("Breakdown by category:")
    for label, stats in best_metrics['results'].items():
        accuracy_label = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {label:8s}: {stats['correct']:3d}/{stats['total']:3d} correct ({accuracy_label:5.1f}%)")
    print()
    
    # Рекомендации
    print("RECOMMENDATIONS:")
    print()
    
    if best_metrics['fpr'] < 0.05:
        print("[OK] Very low false positive rate - recommended for production")
    elif best_metrics['fpr'] < 0.10:
        print("[OK] Low false positive rate - acceptable for production")
    elif best_metrics['fpr'] < 0.20:
        print("[WARNING] Moderate false positive rate - consider increasing threshold")
    else:
        print("[WARNING] High false positive rate - increase threshold")
    print()
    
    if best_metrics['tpr'] > 0.90:
        print("[OK] Excellent anomaly detection rate")
    elif best_metrics['tpr'] > 0.80:
        print("[OK] Good anomaly detection rate")
    elif best_metrics['tpr'] > 0.70:
        print("[WARNING] Moderate anomaly detection - consider decreasing threshold")
    else:
        print("[WARNING] Low anomaly detection - decrease threshold")
    print()
    
    # Инструкции по применению
    print("TO APPLY:")
    print()
    print("Edit configs/bot_config.py:")
    print()
    print("class AIConfig:")
    print(f"    AI_ANOMALY_BLOCK_THRESHOLD = {best_threshold}  # Recommended")
    print()
    
    # Визуализация (текстовая)
    print("THRESHOLD COMPARISON:")
    print()
    print("Threshold | Accuracy | FPR | TPR | Score")
    print("-" * 60)
    
    for result in all_results:
        t = result['threshold']
        m = result['metrics']
        s = result['score']
        
        marker = " <-- BEST" if t == best_threshold else ""
        
        acc_bar = "#" * int(m['accuracy'] * 20)
        fpr_bar = "#" * int(m['fpr'] * 20)
        tpr_bar = "#" * int(m['tpr'] * 20)
        
        print(f"{t:.2f}      | {acc_bar:<20s} | {fpr_bar:<20s} | {tpr_bar:<20s} | {s:.3f}{marker}")
    
    print()
    print("=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    optimize_threshold()

