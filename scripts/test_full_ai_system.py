#!/usr/bin/env python3
"""
Комплексный тест всей AI системы
Проверяет работу всех модулей вместе
"""

import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_full_ai_system():
    """Тестирует полную AI систему"""
    
    print("\n" + "="*80)
    print("КОМПЛЕКСНЫЙ ТЕСТ: Полная AI система")
    print("="*80 + "\n")
    
    # 1. Проверка конфигурации
    print("[1] Проверка конфигурации AI...")
    from bot_engine.bot_config import AIConfig
    
    print(f"   AI_ENABLED: {AIConfig.AI_ENABLED}")
    print(f"   AI_ANOMALY_DETECTION_ENABLED: {AIConfig.AI_ANOMALY_DETECTION_ENABLED}")
    print(f"   AI_RISK_MANAGEMENT_ENABLED: {AIConfig.AI_RISK_MANAGEMENT_ENABLED}")
    print(f"   AI_AUTO_TRAIN_ENABLED: {AIConfig.AI_AUTO_TRAIN_ENABLED}\n")
    
    # 2. Инициализация AI Manager
    print("[2] Инициализация AI Manager...")
    from bot_engine.ai.ai_manager import get_ai_manager
    
    ai_manager = get_ai_manager()
    
    print(f"   AI Manager создан: {ai_manager is not None}")
    print(f"   AI доступен: {ai_manager.is_available()}")
    print(f"   Anomaly Detector: {'[OK]' if ai_manager.anomaly_detector else '[X]'}")
    print(f"   Risk Manager: {'[OK]' if ai_manager.risk_manager else '[X]'}\n")
    
    if not ai_manager.is_available():
        print("[X] AI недоступен, проверьте лицензию!")
        return False
    
    # 3. Проверка Anomaly Detector
    print("[3] Тест Anomaly Detector...")
    
    if ai_manager.anomaly_detector:
        status = ai_manager.anomaly_detector.get_status()
        print(f"   Модель обучена: {status['is_trained']}")
        print(f"   Contamination: {status['contamination']}")
        print(f"   Model type: {status['model_type']}")
        
        if status['is_trained']:
            print("   [OK] Модель готова к обнаружению аномалий\n")
        else:
            print("   [!] ВНИМАНИЕ: Модель не обучена!\n")
    else:
        print("   [X] Anomaly Detector не загружен\n")
    
    # 4. Проверка Risk Manager
    print("[4] Тест Risk Manager...")
    
    if ai_manager.risk_manager:
        status = ai_manager.risk_manager.get_status()
        print(f"   Active: {status['active']}")
        print(f"   SL диапазон: {status['sl_range'][0]}-{status['sl_range'][1]}%")
        print(f"   TP диапазон: {status['tp_range'][0]}-{status['tp_range'][1]}%")
        print("   [OK] Risk Manager готов к управлению рисками\n")
    else:
        print("   [X] Risk Manager не загружен\n")
    
    # 5. Тест комплексного анализа монеты
    print("[5] Тест комплексного анализа монеты (симуляция)...")
    
    # Генерируем тестовые свечи
    import random
    test_candles = []
    base_price = 50000
    
    for i in range(50):
        price = base_price * (1 + random.uniform(-0.02, 0.02))
        candle = {
            'timestamp': 1729000000000 + i * 21600000,
            'time': 1729000000000 + i * 21600000,
            'open': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'close': price * (1 + random.uniform(-0.01, 0.01)),
            'volume': random.uniform(1000000, 3000000)
        }
        test_candles.append(candle)
        base_price = candle['close']
    
    # Тестовые данные монеты
    coin_data = {
        'symbol': 'TESTCOIN',
        'in_position': False,
        'trend': 'UP',
        'rsi': 28
    }
    
    # Анализ монеты
    analysis = ai_manager.analyze_coin('TESTCOIN', coin_data, test_candles)
    
    print(f"   AI доступен: {analysis['available']}")
    
    if analysis['available']:
        anomaly = analysis.get('anomaly_score')
        if anomaly:
            print(f"   Аномалия обнаружена: {anomaly.get('is_anomaly')}")
            if anomaly.get('is_anomaly'):
                print(f"   Тип аномалии: {anomaly.get('anomaly_type')}")
                print(f"   Severity: {anomaly.get('severity', 0):.0%}")
        else:
            print(f"   Anomaly score: None (модель не обучена)")
    
    print()
    
    # 6. Тест адаптивных параметров для сделки
    print("[6] Тест адаптивных параметров для сделки...")
    
    if ai_manager.risk_manager:
        # Динамический SL
        dynamic_sl = ai_manager.risk_manager.calculate_dynamic_sl(
            'TESTCOIN', test_candles, 'LONG'
        )
        print(f"   Адаптивный SL: {dynamic_sl['sl_percent']}%")
        print(f"   Волатильность: {dynamic_sl['volatility']:.3f}")
        print(f"   Причина: {dynamic_sl['reason']}")
        
        # Динамический TP
        dynamic_tp = ai_manager.risk_manager.calculate_dynamic_tp(
            'TESTCOIN', test_candles, 'LONG'
        )
        print(f"   Адаптивный TP: {dynamic_tp['tp_percent']}%")
        print(f"   Сила тренда: {dynamic_tp['trend_strength']:.3f}")
        print(f"   Причина: {dynamic_tp['reason']}")
        
        # Размер позиции
        position_size = ai_manager.risk_manager.calculate_position_size(
            'TESTCOIN', test_candles, balance_usdt=1000, signal_confidence=0.8
        )
        print(f"   Размер позиции: {position_size['size_usdt']} USDT")
        print(f"   Множитель: {position_size['size_multiplier']:.2f}x")
        print(f"   Причина: {position_size['reason']}")
    
    print()
    
    # 7. Итоговая статистика AI
    print("[7] Итоговая статистика AI системы...")
    
    status = ai_manager.get_status()
    
    print(f"   Enabled: {status['enabled']}")
    print(f"   Available: {status['available']}")
    print(f"   License type: {status['license']['type']}")
    print(f"   License valid: {status['license']['valid']}")
    
    modules = status['modules']
    loaded_count = sum([
        modules['anomaly_detector'],
        modules['lstm_predictor'],
        modules['pattern_detector'],
        modules['risk_manager']
    ])
    
    print(f"\n   Загружено модулей: {loaded_count}/4")
    print(f"     - Anomaly Detector: {'[OK]' if modules['anomaly_detector'] else '[X]'}")
    print(f"     - LSTM Predictor: {'[OK]' if modules['lstm_predictor'] else '[X]'}")
    print(f"     - Pattern Detector: {'[OK]' if modules['pattern_detector'] else '[X]'}")
    print(f"     - Risk Manager: {'[OK]' if modules['risk_manager'] else '[X]'}")
    
    print("\n" + "="*80)
    
    if loaded_count >= 2:
        print("РЕЗУЛЬТАТ: [OK] AI система работает! (загружено 2+ модулей)")
        print("\nДоступные функции:")
        if modules['anomaly_detector']:
            print("  - Обнаружение аномалий (PUMP/DUMP) в Exit Scam фильтре")
        if modules['risk_manager']:
            print("  - Адаптивные SL/TP при открытии позиций")
            print("  - Умное управление размером позиции")
            print("  - Предсказание разворотов тренда")
    else:
        print("РЕЗУЛЬТАТ: [!] AI система частично доступна")
        print(f"           Загружено {loaded_count}/4 модулей")
    
    print("="*80 + "\n")
    
    return True


if __name__ == '__main__':
    try:
        success = test_full_ai_system()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[X] Ошибка теста: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

