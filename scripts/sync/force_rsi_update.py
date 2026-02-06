#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Принудительное обновление RSI данных"""

import requests
import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def force_update():
    """Принудительно обновляет RSI данные"""
    try:
        print("🔄 ПРИНУДИТЕЛЬНОЕ ОБНОВЛЕНИЕ RSI")
        print("=" * 60)
        
        # Получаем список всех монет
        r = requests.get('http://localhost:5001/api/bots/coins-with-rsi')
        coins = r.json().get('coins', {})
        symbols = list(coins.keys())
        
        print(f"📊 Найдено {len(symbols)} монет")
        print(f"🔄 Обновляем RSI данные...")
        
        # Обновляем несколько ключевых монет для теста
        test_symbols = ['BIO', 'BTC', 'ETH', 'SOL', 'BNB']
        
        for symbol in test_symbols:
            if symbol in symbols:
                print(f"  🔄 {symbol}...", end='', flush=True)
                r = requests.get(f'http://localhost:5001/api/bots/coins-with-rsi?refresh_symbol={symbol}')
                if r.status_code == 200:
                    print(" ✅")
                else:
                    print(f" ❌ ({r.status_code})")
                time.sleep(0.5)
        
        print(f"\n✅ Обновление завершено")
        
        # Проверяем результат
        print(f"\n📊 Проверка результатов:")
        r = requests.get('http://localhost:5001/api/bots/coins-with-rsi')
        coins = r.json().get('coins', {})
        
        for symbol in test_symbols:
            if symbol in coins:
                coin = coins[symbol]
                print(f"\n{symbol}:")
                print(f"  RSI: {coin.get('rsi6h')}")
                print(f"  Signal: {coin.get('signal')}")
                print(f"  RSI Zone: {coin.get('rsi_zone')}")
                
                tf = coin.get('time_filter_info', {})
                if tf:
                    print(f"  Time Filter: {'❌ Blocked' if tf.get('blocked') else '✅ Passed'}")
                    print(f"    Reason: {tf.get('reason')}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    force_update()

