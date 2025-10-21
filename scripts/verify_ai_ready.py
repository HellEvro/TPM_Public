#!/usr/bin/env python3
"""
Финальная проверка готовности AI системы к продакшн
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_ai_ready():
    """Проверяет готовность AI системы"""
    
    print("\n" + "="*80)
    print("ФИНАЛЬНАЯ ПРОВЕРКА: AI система готова к продакшн?")
    print("="*80 + "\n")
    
    checks_passed = 0
    total_checks = 10
    
    # 1. Конфигурация
    print("[1/10] Проверка конфигурации...")
    from bot_engine.bot_config import AIConfig
    
    if AIConfig.AI_ENABLED and AIConfig.AI_ANOMALY_DETECTION_ENABLED and AIConfig.AI_RISK_MANAGEMENT_ENABLED:
        print("   [OK] AI модули включены в конфигурации")
        checks_passed += 1
    else:
        print("   [X] AI модули отключены!")
        return False
    
    # 2. Лицензия
    print("[2/10] Проверка лицензии...")
    license_file = Path('license.lic')
    
    if license_file.exists():
        print("   [OK] Файл лицензии найден")
        checks_passed += 1
    else:
        print("   [X] Файл лицензии не найден!")
    
    # 3. AI Manager
    print("[3/10] Инициализация AI Manager...")
    from bot_engine.ai.ai_manager import get_ai_manager
    
    ai_manager = get_ai_manager()
    
    if ai_manager and ai_manager.is_available():
        print("   [OK] AI Manager доступен")
        checks_passed += 1
    else:
        print("   [X] AI Manager недоступен!")
        return False
    
    # 4. Anomaly Detector
    print("[4/10] Проверка Anomaly Detector...")
    
    if ai_manager.anomaly_detector and ai_manager.anomaly_detector.is_trained:
        print("   [OK] Anomaly Detector загружен и обучен")
        checks_passed += 1
    else:
        print("   [!] Anomaly Detector не обучен (но загружен)")
        if ai_manager.anomaly_detector:
            checks_passed += 0.5
    
    # 5. Risk Manager
    print("[5/10] Проверка Risk Manager...")
    
    if ai_manager.risk_manager:
        status = ai_manager.risk_manager.get_status()
        if status['active']:
            print("   [OK] Risk Manager активен")
            checks_passed += 1
        else:
            print("   [X] Risk Manager неактивен!")
    else:
        print("   [X] Risk Manager не загружен!")
    
    # 6. Модели
    print("[6/10] Проверка файлов моделей...")
    
    model_path = Path(AIConfig.AI_ANOMALY_MODEL_PATH)
    scaler_path = Path(AIConfig.AI_ANOMALY_SCALER_PATH)
    
    if model_path.exists() and scaler_path.exists():
        model_size = model_path.stat().st_size / 1024  # KB
        scaler_size = scaler_path.stat().st_size / 1024
        print(f"   [OK] Модели найдены ({model_size:.1f} KB + {scaler_size:.1f} KB)")
        checks_passed += 1
    else:
        print("   [X] Файлы моделей не найдены!")
    
    # 7. Исторические данные
    print("[7/10] Проверка исторических данных...")
    
    hist_dir = Path('data/ai/historical')
    
    if hist_dir.exists():
        csv_files = list(hist_dir.glob('*_6h_historical.csv'))
        if len(csv_files) >= 500:
            print(f"   [OK] Найдено {len(csv_files)} файлов с историей")
            checks_passed += 1
        else:
            print(f"   [!] Мало файлов: {len(csv_files)} (ожидается 583)")
            checks_passed += 0.5
    else:
        print("   [X] Директория с данными не найдена!")
    
    # 8. Auto Trainer
    print("[8/10] Проверка Auto Trainer...")
    
    if AIConfig.AI_AUTO_TRAIN_ENABLED:
        print("   [OK] Auto Trainer включён")
        print(f"       Интервал обновления: {AIConfig.AI_DATA_UPDATE_INTERVAL/3600:.0f} часов")
        print(f"       Интервал переобучения: {AIConfig.AI_RETRAIN_INTERVAL/86400:.0f} дней")
        checks_passed += 1
    else:
        print("   [!] Auto Trainer отключён")
        checks_passed += 0.5
    
    # 9. Интеграция в фильтры
    print("[9/10] Проверка интеграции в фильтры...")
    
    try:
        from bots_modules import filters
        
        # Проверяем что есть импорт AI
        import inspect
        source = inspect.getsource(filters.check_exit_scam_filter)
        
        if 'AI' in source and 'anomaly_detector' in source:
            print("   [OK] AI интегрирован в Exit Scam фильтр")
            checks_passed += 1
        else:
            print("   [X] AI не найден в фильтрах!")
    except Exception as e:
        print(f"   [X] Ошибка проверки: {e}")
    
    # 10. Интеграция в trading_bot
    print("[10/10] Проверка интеграции в trading_bot...")
    
    try:
        from bot_engine import trading_bot
        
        source = inspect.getsource(trading_bot.TradingBot)
        
        if 'risk_manager' in source and 'calculate_dynamic_sl' in source:
            print("   [OK] Risk Manager интегрирован в trading_bot")
            checks_passed += 1
        else:
            print("   [!] Risk Manager не найден в trading_bot")
            checks_passed += 0.5
    except Exception as e:
        print(f"   [X] Ошибка проверки: {e}")
    
    # Итоговая оценка
    print("\n" + "="*80)
    
    score = (checks_passed / total_checks) * 100
    
    print(f"РЕЗУЛЬТАТ: {checks_passed}/{total_checks} проверок пройдено ({score:.0f}%)")
    print("="*80 + "\n")
    
    if score >= 90:
        print("✅ ОТЛИЧНО! AI система полностью готова к продакшн!")
        print("\nМожете запускать бот:")
        print("  python bots.py")
        print("\nAI будет автоматически:")
        print("  - Блокировать pump/dump (Anomaly Detection)")
        print("  - Адаптировать SL (8-25% по волатильности)")
        print("  - Адаптировать размер позиции (5-20 USDT)")
        print("  - Обновлять данные (ежедневно)")
        print("  - Переобучаться (еженедельно)")
        return True
    elif score >= 70:
        print("⚠️ ХОРОШО! AI система почти готова")
        print("\nНекоторые компоненты требуют внимания")
        print("Но основные функции работают")
        return True
    else:
        print("❌ ПРОБЛЕМЫ! AI система не готова")
        print("\nТребуется дополнительная настройка")
        return False


if __name__ == '__main__':
    try:
        ready = verify_ai_ready()
        sys.exit(0 if ready else 1)
    except Exception as e:
        print(f"\n[X] Ошибка проверки: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

