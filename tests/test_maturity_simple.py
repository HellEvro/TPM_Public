#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Простой тест проверки зрелости нескольких монет
"""
import sys
import io
import requests
import json

# Устанавливаем UTF-8 для консоли Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_maturity_simple():
    """Тестирует зрелость нескольких монет"""
    print("🔍 ТЕСТ ПРОВЕРКИ ЗРЕЛОСТИ")
    print("=" * 40)
    
    # Тестируем только 3 монеты
    test_symbols = ['BTC', 'ETH', 'BIO']
    
    for symbol in test_symbols:
        print(f"\n📊 Тестируем {symbol}:")
        try:
            # Принудительно обновляем RSI данные
            response = requests.get(f'http://localhost:5001/api/bots/coins-with-rsi?refresh_symbol={symbol}', timeout=10)
            
            if response.status_code == 200:
                print(f"  ✅ RSI данные обновлены")
                
                # Получаем общие данные
                data_response = requests.get('http://localhost:5001/api/bots/coins-with-rsi', timeout=10)
                if data_response.status_code == 200:
                    data = data_response.json()
                    coins = data.get('coins', {})
                    
                    if symbol in coins:
                        coin_data = coins[symbol]
                        print(f"  📈 RSI: {coin_data.get('rsi6h', 'N/A')}")
                        print(f"  🎯 Сигнал: {coin_data.get('signal', 'N/A')}")
                        print(f"  🕒 Тренд: {coin_data.get('trend6h', 'N/A')}")
                    else:
                        print(f"  ❌ Монета {symbol} не найдена в данных")
                else:
                    print(f"  ❌ Ошибка получения данных: {data_response.status_code}")
            else:
                print(f"  ❌ Ошибка обновления: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
    
    # Проверяем хранилище зрелых монет
    print(f"\n🏆 Хранилище зрелых монет:")
    try:
        with open('data/mature_coins.json', 'r', encoding='utf-8') as f:
            mature_data = json.load(f)
        print(f"  📊 Зрелых монет: {len(mature_data)}")
        if mature_data:
            print(f"  📝 Примеры: {list(mature_data.keys())[:5]}")
        else:
            print(f"  ⚠️ Хранилище пустое")
    except Exception as e:
        print(f"  ❌ Ошибка чтения хранилища: {e}")

if __name__ == '__main__':
    test_maturity_simple()
