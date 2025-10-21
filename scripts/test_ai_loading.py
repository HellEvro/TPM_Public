"""
Тест загрузки AI модулей
"""

import sys
sys.path.append('.')

print("=" * 60)
print("AI MODULE LOADING TEST")
print("=" * 60)
print()

# Шаг 1: Проверка лицензии
print("Step 1: Checking license...")
print("-" * 60)

from bot_engine.ai._premium_loader import get_premium_loader

loader = get_premium_loader()

print(f"Premium available: {loader.premium_available}")
print(f"License valid: {loader.license_valid}")

if loader.license_info:
    print(f"License type: {loader.license_info['type']}")
    print(f"Expires: {loader.license_info['expires_at']}")
    print(f"Features: {loader.license_info['features']}")

print()

# Шаг 2: Загрузка AI Manager
print("Step 2: Loading AI Manager...")
print("-" * 60)

from bot_engine.ai.ai_manager import get_ai_manager

ai_manager = get_ai_manager()

print(f"AI Manager created: {ai_manager is not None}")
print(f"AI available: {ai_manager.is_available()}")
print()

# Шаг 3: Проверка модулей
print("Step 3: Checking modules...")
print("-" * 60)

print(f"Anomaly Detector: {ai_manager.anomaly_detector is not None}")
print(f"LSTM Predictor: {ai_manager.lstm_predictor is not None}")
print(f"Pattern Detector: {ai_manager.pattern_detector is not None}")
print(f"Risk Manager: {ai_manager.risk_manager is not None}")

print()

# Шаг 4: Статус
print("Step 4: Status...")
print("-" * 60)

status = ai_manager.get_status()

for key, value in status.items():
    print(f"{key}: {value}")

print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)

