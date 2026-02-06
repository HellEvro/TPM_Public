#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Тест реальной функции get_coin_rsi_data с отладкой
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
    print_header("🔍 ОТЛАДКА РЕАЛЬНОЙ ФУНКЦИИ get_coin_rsi_data")
    
    # Загружаем конфигурацию
    print_section("📋 Загрузка конфигурации")
    load_auto_bot_config()
    
    with bots_data_lock:
        config = bots_data.get('auto_bot_config', {})
    
    print("🔍 Настройки:")
    print(f"  • avoid_down_trend: {config.get('avoid_down_trend', 'НЕ НАЙДЕНО')}")
    print(f"  • avoid_up_trend: {config.get('avoid_up_trend', 'НЕ НАЙДЕНО')}")
    print(f"  • RSI_OVERSOLD: {SystemConfig.RSI_OVERSOLD}")
    print(f"  • RSI_OVERBOUGHT: {SystemConfig.RSI_OVERBOUGHT}")
    
    # Тестируем несколько реальных монет
    test_symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
    
    print_section("🧪 Тестирование реальных монет")
    
    for symbol in test_symbols:
        print(f"\n🔍 Тестируем {symbol}:")
        
        try:
            # Получаем данные через реальную функцию
            coin_data = get_coin_rsi_data(symbol)
            
            if coin_data:
                rsi = coin_data.get('rsi6h', 'НЕТ')
                trend = coin_data.get('trend', 'НЕТ')
                signal = coin_data.get('signal', 'НЕТ')
                is_mature = coin_data.get('is_mature', 'НЕТ')
                blocked_by = coin_data.get('blocked_by', 'НЕТ')
                
                print(f"  • RSI: {rsi}")
                print(f"  • Trend: {trend}")
                print(f"  • Signal: {signal}")
                print(f"  • Mature: {is_mature}")
                print(f"  • Blocked by: {blocked_by}")
                
                # Анализируем логику
                if isinstance(rsi, (int, float)):
                    if rsi <= SystemConfig.RSI_OVERSOLD:
                        expected_signal = "ENTER_LONG"
                        if config.get('avoid_down_trend', False) and trend == 'DOWN':
                            expected_signal = "WAIT (DOWN тренд)"
                    elif rsi >= SystemConfig.RSI_OVERBOUGHT:
                        expected_signal = "ENTER_SHORT"
                        if config.get('avoid_up_trend', False) and trend == 'UP':
                            expected_signal = "WAIT (UP тренд)"
                    else:
                        expected_signal = "WAIT (нейтральная зона)"
                    
                    print(f"  • Ожидается: {expected_signal}")
                    
                    if signal == expected_signal or (expected_signal.startswith("WAIT") and signal == "WAIT"):
                        print(f"  ✅ ПРАВИЛЬНО")
                    else:
                        print(f"  ❌ ОШИБКА! Ожидалось: {expected_signal}, получено: {signal}")
                else:
                    print(f"  ⚠️  RSI не число: {rsi}")
            else:
                print(f"  ❌ Нет данных для {symbol}")
                
        except Exception as e:
            print(f"  ❌ Ошибка для {symbol}: {e}")
    
    print_section("💡 Дополнительная отладка")
    
    # Проверим что происходит с конкретной монетой из теста
    print(f"\n🔍 Детальная отладка для BTC:")
    try:
        coin_data = get_coin_rsi_data('BTC')
        print(f"Полные данные BTC: {json.dumps(coin_data, indent=2, default=str)}")
    except Exception as e:
        print(f"Ошибка получения данных BTC: {e}")

if __name__ == "__main__":
    main()
