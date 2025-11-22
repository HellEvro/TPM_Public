#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Поиск подходящих кандидатов для торговли"""

import requests
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def find_candidates():
    """Находит подходящих кандидатов"""
    try:
        print("🔍 ПОИСК ПОДХОДЯЩИХ КАНДИДАТОВ")
        print("=" * 60)
        
        r = requests.get('http://localhost:5001/api/bots/coins-with-rsi')
        coins = r.json().get('coins', {})
        
        print(f"📊 Всего монет: {len(coins)}")
        
        # Ищем LONG кандидатов (RSI <= 29, trend != DOWN)
        long_candidates = []
        for symbol, coin in coins.items():
            rsi = coin.get('rsi6h')
            trend = coin.get('trend6h')
            zone = coin.get('rsi_zone')
            signal = coin.get('signal')
            
            if rsi and rsi <= 29 and zone == 'BUY_ZONE':
                long_candidates.append((symbol, rsi, trend, signal))
        
        long_candidates.sort(key=lambda x: x[1])
        
        print(f"\n🟢 LONG кандидаты (RSI <= 29, BUY_ZONE): {len(long_candidates)}")
        for symbol, rsi, trend, signal in long_candidates:
            trend_emoji = '📈' if trend == 'UP' else '📉' if trend == 'DOWN' else '➡️'
            signal_emoji = '✅' if signal == 'ENTER_LONG' else '❌'
            print(f"  {signal_emoji} {symbol}: RSI={rsi:.1f}, Trend={trend_emoji}{trend}, Signal={signal}")
        
        # Ищем SHORT кандидатов (RSI >= 71, trend != UP)
        short_candidates = []
        for symbol, coin in coins.items():
            rsi = coin.get('rsi6h')
            trend = coin.get('trend6h')
            zone = coin.get('rsi_zone')
            signal = coin.get('signal')
            
            if rsi and rsi >= 71 and zone == 'SELL_ZONE':
                short_candidates.append((symbol, rsi, trend, signal))
        
        short_candidates.sort(key=lambda x: -x[1])
        
        print(f"\n🔴 SHORT кандидаты (RSI >= 71, SELL_ZONE): {len(short_candidates)}")
        for symbol, rsi, trend, signal in short_candidates:
            trend_emoji = '📈' if trend == 'UP' else '📉' if trend == 'DOWN' else '➡️'
            signal_emoji = '✅' if signal == 'ENTER_SHORT' else '❌'
            print(f"  {signal_emoji} {symbol}: RSI={rsi:.1f}, Trend={trend_emoji}{trend}, Signal={signal}")
        
        # Подсчитываем сколько монет прошли фильтр тренда
        long_with_good_trend = [c for c in long_candidates if c[3] == 'ENTER_LONG']
        short_with_good_trend = [c for c in short_candidates if c[3] == 'ENTER_SHORT']
        
        print(f"\n📊 Итого:")
        print(f"  LONG с подходящим трендом: {len(long_with_good_trend)}/{len(long_candidates)}")
        print(f"  SHORT с подходящим трендом: {len(short_with_good_trend)}/{len(short_candidates)}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    find_candidates()

