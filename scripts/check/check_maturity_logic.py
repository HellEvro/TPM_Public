#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Проверка логики зрелости монет
"""
import sys
import io
import requests
import json

# Устанавливаем UTF-8 для консоли Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_maturity_logic():
    """Проверяет логику зрелости монет"""
    print("🔍 ПРОВЕРКА ЛОГИКИ ЗРЕЛОСТИ МОНЕТ")
    print("=" * 50)
    
    try:
        # Получаем данные всех монет
        response = requests.get('http://localhost:5001/api/bots/coins-with-rsi', timeout=30)
        if response.status_code != 200:
            print(f"❌ Ошибка: {response.status_code}")
            return
        
        data = response.json()
        coins = data.get('coins', {})
        
        print(f"📊 Всего монет: {len(coins)}")
        
        # Проверяем несколько монет вручную
        test_symbols = ['BTC', 'ETH', 'BIO', 'DOGE', 'ADA']
        
        for symbol in test_symbols:
            if symbol in coins:
                print(f"\n📈 {symbol}:")
                coin = coins[symbol]
                print(f"  RSI: {coin.get('rsi6h', 'N/A')}")
                print(f"  Сигнал: {coin.get('signal', 'N/A')}")
                print(f"  Тренд: {coin.get('trend6h', 'N/A')}")
                
                # Принудительно обновляем для проверки зрелости
                print(f"  🔄 Обновляем для проверки зрелости...")
                update_response = requests.get(f'http://localhost:5001/api/bots/coins-with-rsi?refresh_symbol={symbol}', timeout=10)
                
                if update_response.status_code == 200:
                    print(f"  ✅ Обновлено")
                else:
                    print(f"  ❌ Ошибка обновления: {update_response.status_code}")
        
        # Проверяем хранилище зрелых монет
        print(f"\n🏆 Хранилище зрелых монет:")
        try:
            with open('data/mature_coins.json', 'r', encoding='utf-8') as f:
                mature_data = json.load(f)
            print(f"  📊 Зрелых монет: {len(mature_data)}")
            
            if len(mature_data) > 50:
                print(f"  ⚠️ СЛИШКОМ МНОГО зрелых монет! Возможно ошибка в логике")
            elif len(mature_data) > 0:
                print(f"  📝 Примеры зрелых монет: {list(mature_data.keys())[:10]}")
            else:
                print(f"  ✅ Хранилище пустое (нормально для новых критериев)")
                
        except Exception as e:
            print(f"  ❌ Ошибка чтения хранилища: {e}")
            
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")

if __name__ == '__main__':
    check_maturity_logic()
