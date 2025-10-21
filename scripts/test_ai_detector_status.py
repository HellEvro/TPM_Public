#!/usr/bin/env python3
"""
Тест: Проверка статуса обученной модели Anomaly Detector
"""

import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_detector_status():
    """Проверяет статус и детали обученной модели"""
    
    print("\n" + "="*80)
    print("ТЕСТ: Статус Anomaly Detector")
    print("="*80 + "\n")
    
    # Инициализируем AI Manager
    from bot_engine.ai.ai_manager import get_ai_manager
    
    ai_manager = get_ai_manager()
    
    if not ai_manager.anomaly_detector:
        print("[X] Anomaly Detector не загружен!")
        return
    
    detector = ai_manager.anomaly_detector
    
    print("[1] Статус детектора:")
    print(f"   Модель обучена: {detector.is_trained}")
    print(f"   Contamination: {detector.contamination}")
    print(f"   Random State: {detector.random_state}")
    print(f"   Модель тип: {type(detector.model).__name__}")
    print(f"   Scaler тип: {type(detector.scaler).__name__}")
    
    # Проверяем параметры модели
    if detector.model:
        print(f"\n[2] Параметры модели:")
        print(f"   N estimators: {detector.model.n_estimators}")
        print(f"   Max samples: {detector.model.max_samples}")
        print(f"   Обучена: {hasattr(detector.model, 'estimators_')}")
        if hasattr(detector.model, 'estimators_'):
            print(f"   Количество деревьев: {len(detector.model.estimators_)}")
    
    # Проверяем scaler
    if detector.scaler:
        print(f"\n[3] Параметры Scaler:")
        print(f"   Тип: {type(detector.scaler).__name__}")
        if hasattr(detector.scaler, 'mean_'):
            print(f"   Размерность: {len(detector.scaler.mean_)}")
            print(f"   Mean (первые 5): {detector.scaler.mean_[:5]}")
            print(f"   Scale (первые 5): {detector.scaler.scale_[:5]}")
        else:
            print(f"   [!] Scaler не обучен (нет mean_)")
    
    print("\n" + "="*80)
    
    if detector.is_trained:
        print("РЕЗУЛЬТАТ: [OK] Модель полностью загружена и обучена!")
        print("           Детектор готов к обнаружению аномалий")
    else:
        print("РЕЗУЛЬТАТ: [X] Модель создана, но не обучена")
        print("           Требуется обучение: python scripts/ai/train_anomaly_on_real_data.py")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        test_detector_status()
    except Exception as e:
        print(f"\n[X] Ошибка теста: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

