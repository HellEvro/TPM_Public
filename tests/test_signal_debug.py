#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Детальный тест для отладки логики установки сигналов
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot_engine.config_loader import SystemConfig
from bots_modules.imports_and_globals import load_auto_bot_config, bots_data, bots_data_lock
import json

def print_header(text):
    print(f"\n{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}")

def print_section(text):
    print(f"\n{'-'*80}")
    print(f"  {text}")
    print(f"{'-'*80}")

def debug_signal_logic():
    """Отладка логики установки сигналов"""
    
    print_header("🔍 ОТЛАДКА ЛОГИКИ УСТАНОВКИ СИГНАЛОВ")
    
    # Загружаем конфигурацию
    print_section("📋 Загрузка конфигурации")
    load_auto_bot_config()
    
    with bots_data_lock:
        config = bots_data.get('auto_bot_config', {})
    
    print("🔍 Настройки тренда:")
    print(f"  • avoid_down_trend: {config.get('avoid_down_trend', 'НЕ НАЙДЕНО')}")
    print(f"  • avoid_up_trend: {config.get('avoid_up_trend', 'НЕ НАЙДЕНО')}")
    print(f"  • RSI_OVERSOLD: {SystemConfig.RSI_OVERSOLD}")
    print(f"  • RSI_OVERBOUGHT: {SystemConfig.RSI_OVERBOUGHT}")
    
    # Тестовые случаи
    test_cases = [
        {"rsi": 25.0, "trend": "DOWN", "expected": "ENTER_LONG"},
        {"rsi": 28.0, "trend": "DOWN", "expected": "ENTER_LONG"},
        {"rsi": 75.0, "trend": "UP", "expected": "ENTER_SHORT"},
        {"rsi": 80.0, "trend": "UP", "expected": "ENTER_SHORT"},
        {"rsi": 45.0, "trend": "DOWN", "expected": "WAIT"},
        {"rsi": 50.0, "trend": "UP", "expected": "WAIT"},
    ]
    
    print_section("🧪 Симуляция логики установки сигналов")
    
    for i, case in enumerate(test_cases, 1):
        rsi = case["rsi"]
        trend = case["trend"]
        expected = case["expected"]
        
        print(f"\n{i}. RSI={rsi}, Trend={trend}")
        
        # Симулируем логику из get_coin_rsi_data()
        signal = 'WAIT'
        rsi_zone = 'NEUTRAL'
        
        # Получаем настройки
        avoid_down_trend = config.get('avoid_down_trend', True)
        avoid_up_trend = config.get('avoid_up_trend', True)
        
        print(f"   • avoid_down_trend: {avoid_down_trend}")
        print(f"   • avoid_up_trend: {avoid_up_trend}")
        
        if rsi <= SystemConfig.RSI_OVERSOLD:  # RSI ≤ 29 
            rsi_zone = 'BUY_ZONE'
            print(f"   • RSI <= {SystemConfig.RSI_OVERSOLD} → BUY_ZONE")
            
            # Проверяем нужно ли избегать DOWN тренда для LONG
            if avoid_down_trend and trend == 'DOWN':
                signal = 'WAIT'  # Ждем улучшения тренда
                print(f"   • avoid_down_trend=True AND trend=DOWN → WAIT")
            else:
                signal = 'ENTER_LONG'  # Входим независимо от тренда или при хорошем тренде
                print(f"   • avoid_down_trend=False OR trend!=DOWN → ENTER_LONG")
                
        elif rsi >= SystemConfig.RSI_OVERBOUGHT:  # RSI ≥ 71
            rsi_zone = 'SELL_ZONE'
            print(f"   • RSI >= {SystemConfig.RSI_OVERBOUGHT} → SELL_ZONE")
            
            # Проверяем нужно ли избегать UP тренда для SHORT
            if avoid_up_trend and trend == 'UP':
                signal = 'WAIT'  # Ждем ослабления тренда
                print(f"   • avoid_up_trend=True AND trend=UP → WAIT")
            else:
                signal = 'ENTER_SHORT'  # Входим независимо от тренда или при хорошем тренде
                print(f"   • avoid_up_trend=False OR trend!=UP → ENTER_SHORT")
        else:
            print(f"   • RSI между {SystemConfig.RSI_OVERSOLD} и {SystemConfig.RSI_OVERBOUGHT} → NEUTRAL")
        
        print(f"   • Результат: signal = {signal}")
        print(f"   • Ожидается: {expected}")
        
        if signal == expected:
            print(f"   ✅ ПРАВИЛЬНО")
        else:
            print(f"   ❌ ОШИБКА!")
    
    print_section("💡 Анализ проблемы")
    
    # Проверяем реальные настройки
    avoid_down_trend = config.get('avoid_down_trend', True)
    avoid_up_trend = config.get('avoid_up_trend', True)
    
    print(f"🔍 Текущие настройки:")
    print(f"  • avoid_down_trend: {avoid_down_trend}")
    print(f"  • avoid_up_trend: {avoid_up_trend}")
    
    if avoid_down_trend:
        print(f"⚠️  avoid_down_trend=True блокирует LONG при DOWN тренде")
    else:
        print(f"✅ avoid_down_trend=False позволяет LONG при любом тренде")
        
    if avoid_up_trend:
        print(f"⚠️  avoid_up_trend=True блокирует SHORT при UP тренде")
    else:
        print(f"✅ avoid_up_trend=False позволяет SHORT при любом тренде")
    
    print(f"\n🎯 РЕКОМЕНДАЦИИ:")
    if avoid_down_trend or avoid_up_trend:
        print(f"  • Проблема в настройках тренда!")
        print(f"  • Нужно отключить avoid_down_trend и avoid_up_trend")
    else:
        print(f"  • Настройки тренда правильные")
        print(f"  • Проблема в другом месте логики")

if __name__ == "__main__":
    debug_signal_logic()
