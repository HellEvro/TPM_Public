"""
Тест детектора аномалий

Тестирует AnomalyDetector на реальных и синтетических данных
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь (устойчиво к запуску из любого cwd)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
# Включаем режим разработки
os.environ['AI_DEV_MODE'] = '1'

# ВАЖНО: до любых импортов sklearn — подавляет UserWarning delayed/Parallel и фиксирует joblib
import utils.sklearn_parallel_config  # noqa: F401

from bot_engine.ai.anomaly_detector import AnomalyDetector
import numpy as np


def generate_normal_candles(count=50):
    """Генерирует нормальные свечи (без аномалий)"""
    candles = []
    base_price = 1000.0
    base_volume = 100000.0
    
    for i in range(count):
        # Небольшие случайные изменения
        change = np.random.normal(0, 0.5)  # ±0.5%
        base_price *= (1 + change / 100)
        
        candles.append({
            'close': base_price,
            'high': base_price * 1.01,
            'low': base_price * 0.99,
            'volume': base_volume * (1 + np.random.uniform(-0.2, 0.2))
        })
    
    return candles


def generate_pump_candles(count=50):
    """Генерирует свечи с PUMP"""
    candles = generate_normal_candles(40)
    
    # Добавляем PUMP в последние 10 свечей
    base_price = candles[-1]['close']
    for i in range(10):
        # Резкий рост 5-8% за свечу
        change = np.random.uniform(5, 8)
        base_price *= (1 + change / 100)
        
        candles.append({
            'close': base_price,
            'high': base_price * 1.02,
            'low': base_price * 0.98,
            'volume': candles[-1]['volume'] * np.random.uniform(2, 4)  # Объемный всплеск
        })
    
    return candles


def generate_dump_candles(count=50):
    """Генерирует свечи с DUMP"""
    candles = generate_normal_candles(40)
    
    # Добавляем DUMP в последние 10 свечей
    base_price = candles[-1]['close']
    for i in range(10):
        # Резкое падение 5-8% за свечу
        change = np.random.uniform(-8, -5)
        base_price *= (1 + change / 100)
        
        candles.append({
            'close': base_price,
            'high': base_price * 1.02,
            'low': base_price * 0.98,
            'volume': candles[-1]['volume'] * np.random.uniform(2, 4)  # Объемный всплеск
        })
    
    return candles


def test_anomaly_detector():
    """Тестирует детектор аномалий"""
    print("=" * 60)
    print("TEST ANOMALY DETECTOR")
    print("=" * 60)
    print()
    
    # Создаем детектор
    detector = AnomalyDetector()
    print(f"[OK] Detector created")
    print(f"   Status: {detector.get_status()}")
    print()
    
    # Тест 1: Нормальные свечи
    print("Test 1: Normal candles (no anomalies)")
    print("-" * 60)
    normal_candles = generate_normal_candles(50)
    result = detector.detect(normal_candles)
    
    print(f"   Anomaly: {result['is_anomaly']}")
    print(f"   Severity: {result['severity']:.2%}")
    print(f"   Type: {result['anomaly_type']}")
    print(f"   Method: {result.get('method', 'model')}")
    
    if result['is_anomaly']:
        print(f"   [FAIL] Anomaly detected in normal data!")
    else:
        print(f"   [PASS] No anomalies detected")
    print()
    
    # Тест 2: PUMP
    print("Test 2: PUMP (sharp price increase)")
    print("-" * 60)
    pump_candles = generate_pump_candles(50)
    result = detector.detect(pump_candles)
    
    print(f"   Anomaly: {result['is_anomaly']}")
    print(f"   Severity: {result['severity']:.2%}")
    print(f"   Type: {result['anomaly_type']}")
    print(f"   Method: {result.get('method', 'model')}")
    
    if result['is_anomaly'] and result['anomaly_type'] == 'PUMP':
        print(f"   [PASS] PUMP successfully detected")
    else:
        print(f"   [WARNING] PUMP not detected or wrong type")
    print()
    
    # Тест 3: DUMP
    print("Test 3: DUMP (sharp price decrease)")
    print("-" * 60)
    dump_candles = generate_dump_candles(50)
    result = detector.detect(dump_candles)
    
    print(f"   Anomaly: {result['is_anomaly']}")
    print(f"   Severity: {result['severity']:.2%}")
    print(f"   Type: {result['anomaly_type']}")
    print(f"   Method: {result.get('method', 'model')}")
    
    if result['is_anomaly'] and result['anomaly_type'] == 'DUMP':
        print(f"   [PASS] DUMP successfully detected")
    else:
        print(f"   [WARNING] DUMP not detected or wrong type")
    print()
    
    # Тест 4: Обучение модели
    print("Test 4: Model training")
    print("-" * 60)
    
    # Генерируем тренировочные данные
    training_data = []
    for _ in range(100):
        training_data.append(generate_normal_candles(50))
    for _ in range(10):
        training_data.append(generate_pump_candles(50))
    for _ in range(10):
        training_data.append(generate_dump_candles(50))
    
    print(f"   Генерировано {len(training_data)} примеров для обучения")
    
    success = detector.train(training_data)
    
    if success:
        print(f"   [PASS] Model successfully trained")
        print(f"   Status: {detector.get_status()}")
    else:
        print(f"   [FAIL] Training error")
    print()
    
    # Тест 5: Проверка обученной модели
    if success:
        print("Test 5: Testing trained model")
        print("-" * 60)
        
        # Тестируем на новых данных
        test_normal = detector.detect(generate_normal_candles(50))
        test_pump = detector.detect(generate_pump_candles(50))
        test_dump = detector.detect(generate_dump_candles(50))
        
        print(f"   Normal: anomaly={test_normal['is_anomaly']}, severity={test_normal['severity']:.2%}")
        print(f"   PUMP:   anomaly={test_pump['is_anomaly']}, severity={test_pump['severity']:.2%}, type={test_pump['anomaly_type']}")
        print(f"   DUMP:   anomaly={test_dump['is_anomaly']}, severity={test_dump['severity']:.2%}, type={test_dump['anomaly_type']}")
        
        # Подсчет точности
        correct = 0
        if not test_normal['is_anomaly']:
            correct += 1
        if test_pump['is_anomaly']:
            correct += 1
        if test_dump['is_anomaly']:
            correct += 1
        
        accuracy = correct / 3 * 100
        print(f"   Accuracy: {accuracy:.0f}%")
        
        if accuracy >= 66:
            print(f"   [PASS] Acceptable accuracy")
        else:
            print(f"   [WARNING] Low accuracy")
        print()
        
        # Тест 6: Сохранение/Загрузка модели
        print("Test 6: Save and load model")
        print("-" * 60)
        
        model_path = 'data/ai/models/anomaly_detector_test.pkl'
        scaler_path = 'data/ai/models/anomaly_detector_test_scaler.pkl'
        
        detector.save_model(model_path, scaler_path)
        print(f"   [OK] Model saved")
        
        # Загружаем в новый экземпляр
        detector2 = AnomalyDetector()
        detector2.load_model(model_path, scaler_path)
        print(f"   [OK] Model loaded into new instance")
        
        # Проверяем что работает также
        test_result = detector2.detect(generate_pump_candles(50))
        print(f"   Test PUMP: anomaly={test_result['is_anomaly']}, type={test_result['anomaly_type']}")
        
        if test_result['is_anomaly']:
            print(f"   [PASS] Loaded model works")
        else:
            print(f"   [WARNING] Loaded model did not detect anomaly")
        print()
    
    # Итоговая статистика
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"[OK] All tests completed")
    print(f"Mode: {'trained model' if detector.is_trained else 'heuristic'}")
    print()


if __name__ == '__main__':
    test_anomaly_detector()

