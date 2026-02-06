#!/usr/bin/env python3
"""
Тестовый скрипт для проверки данных Stochastic RSI
"""

import requests
import json

def test_stochastic_data():
    """Тестирует получение данных Stochastic RSI через API"""
    try:
        # Получаем данные монет
        response = requests.get('http://localhost:5001/api/bots/coins-with-rsi')
        if response.status_code == 200:
            data = response.json()
            coins = data.get('coins', {})
            
            print(f"Получено {len(coins)} монет")
            
            # Проверяем первые 5 монет на наличие Stochastic RSI
            count = 0
            for symbol, coin in coins.items():
                if count >= 5:
                    break
                    
                print(f"\n=== {symbol} ===")
                print(f"RSI: {coin.get('rsi6h')}")
                
                # Проверяем прямые данные стохастика
                stoch_k = coin.get('stoch_rsi_k')
                stoch_d = coin.get('stoch_rsi_d')
                print(f"Stoch RSI K: {stoch_k}")
                print(f"Stoch RSI D: {stoch_d}")
                
                # Проверяем enhanced_rsi данные
                enhanced_rsi = coin.get('enhanced_rsi')
                if enhanced_rsi:
                    confirmations = enhanced_rsi.get('confirmations', {})
                    stoch_k_enhanced = confirmations.get('stoch_rsi_k')
                    stoch_d_enhanced = confirmations.get('stoch_rsi_d')
                    print(f"Enhanced Stoch RSI K: {stoch_k_enhanced}")
                    print(f"Enhanced Stoch RSI D: {stoch_d_enhanced}")
                else:
                    print("Enhanced RSI: НЕТ")
                
                count += 1
        else:
            print(f"Ошибка API: {response.status_code}")
            
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    test_stochastic_data()
