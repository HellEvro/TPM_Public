#!/usr/bin/env python3
"""
Тест: Проверка правильности инициализации AI модулей при запуске
"""

import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ai_initialization():
    """Тестирует логику инициализации AI модулей"""
    
    print("\n" + "="*80)
    print("ТЕСТ: Инициализация AI модулей при запуске")
    print("="*80 + "\n")
    
    # Шаг 1: Проверяем AI_ENABLED
    print("[1] Проверяем настройки AI...")
    from bot_engine.bot_config import AIConfig
    
    print(f"   AI_ENABLED: {AIConfig.AI_ENABLED}")
    print(f"   AI_ANOMALY_DETECTION_ENABLED: {AIConfig.AI_ANOMALY_DETECTION_ENABLED}")
    print(f"   AI_AUTO_TRAIN_ENABLED: {AIConfig.AI_AUTO_TRAIN_ENABLED}")
    
    if not AIConfig.AI_ENABLED:
        print("   [X] AI отключен в конфигурации")
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТ: AI не будет инициализирован (отключен в конфигурации)")
        print("="*80 + "\n")
        return
    
    print("   [OK] AI включен\n")
    
    # Шаг 2: Инициализируем AI Manager
    print("[2] Инициализируем AI Manager...")
    from bot_engine.ai.ai_manager import get_ai_manager
    
    ai_manager = get_ai_manager()
    print(f"   AI Manager создан: {ai_manager is not None}\n")
    
    # Шаг 3: Проверяем доступность
    print("[3] Проверяем доступность AI...")
    is_available = ai_manager.is_available()
    
    print(f"   Premium доступен: {ai_manager.premium_loader.premium_available}")
    print(f"   Лицензия валидна: {ai_manager.premium_loader.license_valid}")
    print(f"   AI доступен: {is_available}\n")
    
    # Шаг 4: Проверяем загруженные модули
    print("[4] Проверяем загруженные модули...")
    print(f"   Anomaly Detector: {'[OK] Загружен' if ai_manager.anomaly_detector else '[X] Не загружен'}")
    print(f"   LSTM Predictor: {'[OK] Загружен' if ai_manager.lstm_predictor else '[X] Не загружен'}")
    print(f"   Pattern Detector: {'[OK] Загружен' if ai_manager.pattern_detector else '[X] Не загружен'}")
    print(f"   Risk Manager: {'[OK] Загружен' if ai_manager.risk_manager else '[X] Не загружен'}\n")
    
    # Шаг 5: Получаем статус
    print("[5] Получаем статус AI системы...")
    status = ai_manager.get_status()
    
    print(f"   Enabled: {status['enabled']}")
    print(f"   Available: {status['available']}")
    
    if status['license']['valid']:
        print(f"   Лицензия: {status['license']['type']}")
        print(f"   Действительна до: {status['license']['expires_at']}")
    else:
        print(f"   Лицензия: [X] Недействительна")
    
    print("\n" + "="*80)
    
    # Финальный результат
    if is_available:
        loaded_modules = sum([
            ai_manager.anomaly_detector is not None,
            ai_manager.lstm_predictor is not None,
            ai_manager.pattern_detector is not None,
            ai_manager.risk_manager is not None
        ])
        print(f"РЕЗУЛЬТАТ: [OK] AI инициализирован ({loaded_modules}/4 модулей загружено)")
        print("           AI будет использоваться в фильтрах")
        print("           Auto Trainer " + ("запустится" if AIConfig.AI_AUTO_TRAIN_ENABLED else "не запустится (отключен)"))
    else:
        print("РЕЗУЛЬТАТ: [X] AI недоступен (нет лицензии или не установлен)")
        print("           AI не будет использоваться в фильтрах")
        print("           Auto Trainer не запустится")
        print("")
        print("           [!] Для активации:")
        print("           1. Создайте license.lic с текстом: DEVELOPER_LICENSE_DO_NOT_COMMIT")
        print("           2. Перезапустите бот")
    
    print("="*80 + "\n")
    
    # Шаг 6: Тестируем производительность проверки is_available()
    print("[6] Тест производительности is_available() (кэширование)...")
    import time
    
    # Первый вызов
    start = time.perf_counter()
    result1 = ai_manager.is_available()
    time1 = time.perf_counter() - start
    
    # Второй вызов (кэшированный)
    start = time.perf_counter()
    result2 = ai_manager.is_available()
    time2 = time.perf_counter() - start
    
    print(f"   Первый вызов: {time1*1000:.3f} мс")
    print(f"   Второй вызов (кэш): {time2*1000:.3f} мс")
    print(f"   Ускорение: {time1/time2:.1f}x\n")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        test_ai_initialization()
    except Exception as e:
        print(f"\n[X] Ошибка теста: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

