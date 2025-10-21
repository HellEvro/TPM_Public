#!/usr/bin/env python3
"""
Быстрая проверка импортов всех AI модулей
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Проверка импортов AI модулей...")

try:
    # 1. Основные модули
    print("[1] bot_engine.bot_config...", end=" ")
    from bot_engine.bot_config import AIConfig, SystemConfig, RiskConfig
    print("OK")
    
    # 2. AI Manager
    print("[2] bot_engine.ai.ai_manager...", end=" ")
    from bot_engine.ai.ai_manager import get_ai_manager, AIManager
    print("OK")
    
    # 3. Anomaly Detector
    print("[3] bot_engine.ai.anomaly_detector...", end=" ")
    from bot_engine.ai.anomaly_detector import AnomalyDetector
    print("OK")
    
    # 4. Risk Manager
    print("[4] bot_engine.ai.risk_manager...", end=" ")
    from bot_engine.ai.risk_manager import DynamicRiskManager
    print("OK")
    
    # 5. Auto Trainer
    print("[5] bot_engine.ai.auto_trainer...", end=" ")
    from bot_engine.ai.auto_trainer import AutoTrainer, start_auto_trainer, stop_auto_trainer
    print("OK")
    
    # 6. Trading Bot
    print("[6] bot_engine.trading_bot...", end=" ")
    from bot_engine.trading_bot import TradingBot
    print("OK")
    
    # 7. Filters
    print("[7] bots_modules.filters...", end=" ")
    from bots_modules.filters import check_exit_scam_filter
    print("OK")
    
    print("\n✅ Все импорты успешны! Бот готов к запуску.")
    
except Exception as e:
    print(f"\n❌ Ошибка импорта: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

