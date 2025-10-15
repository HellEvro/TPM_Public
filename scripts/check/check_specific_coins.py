#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Проверка конкретных монет"""

import requests
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_coins():
    """Проверяет конкретные монеты"""
    try:
        print("🔍 ПРОВЕРКА КОНКРЕТНЫХ МОНЕТ")
        print("=" * 60)
        
        r = requests.get('http://localhost:5001/api/bots/coins-with-rsi')
        coins = r.json().get('coins', {})
        
        test_coins = ['BTC', 'ETH', 'BIO', 'SOL', 'BNB', 'DOGE', 'XRP', 'ADA']
        
        for symbol in test_coins:
            if symbol in coins:
                coin = coins[symbol]
                print(f"\n{symbol}:")
                print(f"  RSI: {coin.get('rsi')}")
                print(f"  Signal: {coin.get('signal')}")
                print(f"  Trend: {coin.get('trend')}")
                print(f"  Price: {coin.get('price')}")
                
                # Проверяем дополнительные данные
                if 'time_filter_info' in coin:
                    tf = coin['time_filter_info']
                    print(f"  Time Filter: {tf}")
                
                if 'exit_scam_info' in coin:
                    es = coin['exit_scam_info']
                    print(f"  ExitScam: {es}")
            else:
                print(f"\n{symbol}: НЕ НАЙДЕНА")
        
        # Проверяем BIO отдельно через test endpoint
        print("\n" + "=" * 60)
        print("🧪 ТЕСТ BIO через test-coin-filters:")
        r = requests.get('http://localhost:5001/api/bots/test-coin-filters/BIO')
        if r.status_code == 200:
            data = r.json()
            print(f"  RSI: {data.get('rsi')}")
            print(f"  Signal: {data.get('signal')}")
            print(f"  Trend: {data.get('trend')}")
        else:
            print(f"  Ошибка: {r.status_code}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_coins()

