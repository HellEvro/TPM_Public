#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Проверка сигналов монет"""

import requests
import sys
import io

# Настройка кодировки для Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_signals():
    """Проверяет распределение сигналов по монетам"""
    try:
        print("🔍 АНАЛИЗ СИГНАЛОВ МОНЕТ")
        print("=" * 60)
        
        # Получаем данные RSI
        r = requests.get('http://localhost:5001/api/bots/coins-with-rsi')
        data = r.json()
        coins = data.get('coins', {})
        
        print(f"\n📊 Всего монет загружено: {len(coins)}")
        
        # Подсчитываем сигналы
        signals = {}
        for symbol, coin in coins.items():
            sig = coin.get('signal', 'UNKNOWN')
            signals[sig] = signals.get(sig, 0) + 1
        
        print(f"\n📈 Распределение сигналов:")
        for sig, count in sorted(signals.items()):
            print(f"  {sig}: {count}")
        
        # Ищем монеты с ENTER_LONG/SHORT
        long_coins = [(s, c) for s, c in coins.items() if c.get('signal') == 'ENTER_LONG']
        short_coins = [(s, c) for s, c in coins.items() if c.get('signal') == 'ENTER_SHORT']
        
        print(f"\n🟢 LONG кандидаты: {len(long_coins)}")
        if long_coins:
            for symbol, coin in long_coins[:5]:
                print(f"  {symbol}: RSI={coin.get('rsi')}, Trend={coin.get('trend')}")
        
        print(f"\n🔴 SHORT кандидаты: {len(short_coins)}")
        if short_coins:
            for symbol, coin in short_coins[:5]:
                print(f"  {symbol}: RSI={coin.get('rsi')}, Trend={coin.get('trend')}")
        
        # Ищем монеты в экстремальных зонах RSI
        print(f"\n🔥 Монеты в экстремальных зонах RSI:")
        extreme_low = [(s, c) for s, c in coins.items() if c.get('rsi') and c.get('rsi') < 30]
        extreme_high = [(s, c) for s, c in coins.items() if c.get('rsi') and c.get('rsi') > 70]
        
        print(f"  RSI < 30: {len(extreme_low)}")
        if extreme_low:
            for symbol, coin in extreme_low[:10]:
                print(f"    {symbol}: RSI={coin.get('rsi'):.1f}, Signal={coin.get('signal')}, Trend={coin.get('trend')}")
        
        print(f"\n  RSI > 70: {len(extreme_high)}")
        if extreme_high:
            for symbol, coin in extreme_high[:10]:
                print(f"    {symbol}: RSI={coin.get('rsi'):.1f}, Signal={coin.get('signal')}, Trend={coin.get('trend')}")
        
        # Проверяем конфигурацию
        print(f"\n⚙️ Проверка конфигурации:")
        config_r = requests.get('http://localhost:5001/api/bots/auto-bot')
        config_data = config_r.json()
        config = config_data.get('config', {})
        
        print(f"  RSI LONG порог: {config.get('rsi_long_threshold')}")
        print(f"  RSI SHORT порог: {config.get('rsi_short_threshold')}")
        print(f"  RSI Time Filter: {config.get('rsi_time_filter_enabled')}")
        print(f"  RSI Time Filter Candles: {config.get('rsi_time_filter_candles')}")
        print(f"  ExitScam Filter: {config.get('exit_scam_enabled')}")
        print(f"  Maturity Check: {config.get('enable_maturity_check')}")
        print(f"  Min Candles for Maturity: {config.get('min_candles_for_maturity')}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_signals()

