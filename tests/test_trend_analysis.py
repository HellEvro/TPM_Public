#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Тест анализа трендов и сигналов
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots_modules.filters import get_coin_rsi_data
from bots_modules.imports_and_globals import load_auto_bot_config, bots_data, bots_data_lock
from bot_engine.config_loader import SystemConfig
import json

def print_header(text):
    print(f"\n{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}")

def print_section(text):
    print(f"\n{'-'*80}")
    print(f"  {text}")
    print(f"{'-'*80}")

def main():
    print_header("🔍 АНАЛИЗ ПРОБЛЕМЫ С ТРЕНДАМИ И СИГНАЛАМИ")
    
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
    
    # Тестовые монеты с разными RSI и трендами
    test_coins = [
        {"symbol": "BTC", "rsi6h": 25.0, "trend": "DOWN", "is_mature": True},
        {"symbol": "ETH", "rsi6h": 28.0, "trend": "DOWN", "is_mature": True},
        {"symbol": "BNB", "rsi6h": 75.0, "trend": "UP", "is_mature": True},
        {"symbol": "ADA", "rsi6h": 45.0, "trend": "DOWN", "is_mature": True},
        {"symbol": "SOL", "rsi6h": 20.0, "trend": "DOWN", "is_mature": True},
        {"symbol": "DOT", "rsi6h": 80.0, "trend": "UP", "is_mature": True},
    ]
    
    print_section("🧪 Тестирование сигналов для разных монет")
    
    results = []
    for coin in test_coins:
        try:
            # Получаем данные через фильтр
            coin_data = get_coin_rsi_data(coin['symbol'])
            
            result = {
                'symbol': coin['symbol'],
                'input_rsi': coin['rsi6h'],
                'input_trend': coin['trend'],
                'input_mature': coin['is_mature'],
                'output_signal': coin_data.get('signal', 'НЕ НАЙДЕНО'),
                'output_rsi': coin_data.get('rsi6h', 'НЕ НАЙДЕНО'),
                'output_trend': coin_data.get('trend', 'НЕ НАЙДЕНО'),
                'output_mature': coin_data.get('is_mature', 'НЕ НАЙДЕНО'),
                'blocked_by': coin_data.get('blocked_by', 'НЕТ')
            }
            results.append(result)
            
        except Exception as e:
            print(f"❌ Ошибка для {coin['symbol']}: {e}")
    
    # Выводим результаты
    print_section("📊 РЕЗУЛЬТАТЫ АНАЛИЗА")
    
    print("🎯 Анализ сигналов:")
    for result in results:
        symbol = result['symbol']
        input_rsi = result['input_rsi']
        input_trend = result['input_trend']
        output_signal = result['output_signal']
        blocked_by = result['blocked_by']
        
        print(f"\n💎 {symbol}:")
        print(f"  • Вход: RSI={input_rsi}, Trend={input_trend}")
        print(f"  • Выход: Signal={output_signal}")
        print(f"  • Заблокирован: {blocked_by}")
        
        # Анализ ожидаемого поведения
        if input_rsi <= SystemConfig.RSI_OVERSOLD:  # RSI ≤ 29
            expected = "ENTER_LONG"
            if input_trend == "DOWN" and config.get('avoid_down_trend', True):
                expected = "WAIT (из-за DOWN тренда)"
        elif input_rsi >= SystemConfig.RSI_OVERBOUGHT:  # RSI ≥ 71
            expected = "ENTER_SHORT"
            if input_trend == "UP" and config.get('avoid_up_trend', True):
                expected = "WAIT (из-за UP тренда)"
        else:
            expected = "WAIT (нейтральная зона RSI)"
        
        print(f"  • Ожидается: {expected}")
        
        if output_signal == expected or (expected.startswith("WAIT") and output_signal == "WAIT"):
            print(f"  ✅ ПРАВИЛЬНО")
        else:
            print(f"  ❌ ОШИБКА! Ожидалось: {expected}, получено: {output_signal}")
    
    print_section("💡 ВЫВОДЫ")
    
    # Подсчитываем статистику
    total_coins = len(results)
    enter_long_count = len([r for r in results if r['output_signal'] == 'ENTER_LONG'])
    enter_short_count = len([r for r in results if r['output_signal'] == 'ENTER_SHORT'])
    wait_count = len([r for r in results if r['output_signal'] == 'WAIT'])
    
    print(f"📊 Статистика сигналов:")
    print(f"  • Всего монет: {total_coins}")
    print(f"  • ENTER_LONG: {enter_long_count}")
    print(f"  • ENTER_SHORT: {enter_short_count}")
    print(f"  • WAIT: {wait_count}")
    
    if wait_count == total_coins:
        print(f"\n⚠️  ВСЕ МОНЕТЫ В СОСТОЯНИИ WAIT!")
        print(f"🔍 Возможные причины:")
        print(f"  • avoid_down_trend=True блокирует LONG при DOWN тренде")
        print(f"  • avoid_up_trend=True блокирует SHORT при UP тренде")
        print(f"  • Все монеты имеют неподходящий тренд")
    
    print(f"\n🎯 РЕКОМЕНДАЦИИ:")
    if config.get('avoid_down_trend', True):
        print(f"  • Отключить avoid_down_trend для входа в LONG при любом тренде")
    if config.get('avoid_up_trend', True):
        print(f"  • Отключить avoid_up_trend для входа в SHORT при любом тренде")

if __name__ == "__main__":
    main()
