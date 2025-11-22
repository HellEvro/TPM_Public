#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Поиск монет с экстремальными RSI"""

import requests
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def find_extreme():
    """Находит монеты с экстремальными RSI"""
    try:
        print("🔍 ПОИСК МОНЕТ С ЭКСТРЕМАЛЬНЫМИ RSI")
        print("=" * 60)
        
        r = requests.get('http://localhost:5001/api/bots/coins-with-rsi')
        coins = r.json().get('coins', {})
        
        print(f"📊 Всего монет: {len(coins)}")
        
        # Ищем монеты с низким RSI
        low_rsi = []
        for symbol, coin in coins.items():
            rsi = coin.get('rsi6h')
            if rsi and rsi < 35:
                low_rsi.append((symbol, rsi, coin.get('signal'), coin.get('rsi_zone')))
        
        low_rsi.sort(key=lambda x: x[1])
        
        print(f"\n🟢 Монеты с RSI < 35: {len(low_rsi)}")
        for symbol, rsi, signal, zone in low_rsi[:20]:
            print(f"  {symbol}: RSI={rsi:.1f}, Signal={signal}, Zone={zone}")
        
        # Ищем монеты с высоким RSI
        high_rsi = []
        for symbol, coin in coins.items():
            rsi = coin.get('rsi6h')
            if rsi and rsi > 65:
                high_rsi.append((symbol, rsi, coin.get('signal'), coin.get('rsi_zone')))
        
        high_rsi.sort(key=lambda x: -x[1])
        
        print(f"\n🔴 Монеты с RSI > 65: {len(high_rsi)}")
        for symbol, rsi, signal, zone in high_rsi[:20]:
            print(f"  {symbol}: RSI={rsi:.1f}, Signal={signal}, Zone={zone}")
        
        # Проверяем конфигурацию
        r = requests.get('http://localhost:5001/api/bots/auto-bot')
        config = r.json().get('config', {})
        
        print(f"\n⚙️ Конфигурация:")
        print(f"  RSI LONG порог: {config.get('rsi_long_threshold')}")
        print(f"  RSI SHORT порог: {config.get('rsi_short_threshold')}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    find_extreme()

