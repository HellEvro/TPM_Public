#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Тест логики проверки зрелости монет"""

import requests
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_maturity():
    """Тестирует логику зрелости"""
    try:
        print("🔍 ТЕСТ ЛОГИКИ ЗРЕЛОСТИ МОНЕТ")
        print("=" * 60)
        
        # Получаем конфигурацию
        r = requests.get('http://localhost:5001/api/bots/auto-bot')
        config = r.json().get('config', {})
        
        min_candles = config.get('min_candles_for_maturity', 200)
        min_rsi_low = config.get('min_rsi_low', 35)
        max_rsi_high = config.get('max_rsi_high', 65)
        
        print(f"\n⚙️ Критерии зрелости:")
        print(f"  Минимум свечей: {min_candles}")
        print(f"  RSI должен достигать: ≤{min_rsi_low} и ≥{max_rsi_high}")
        
        # Получаем зрелые монеты
        r = requests.get('http://localhost:5001/api/bots/mature-coins-count')
        mature_count = r.json().get('count', 0)
        
        print(f"\n📊 Зрелых монет в хранилище: {mature_count}")
        
        # Получаем примеры
        r = requests.get('http://localhost:5001/api/bots/mature-coins')
        mature_data = r.json()
        mature_coins = mature_data.get('mature_coins', {})
        
        print(f"\n📝 Примеры зрелых монет:")
        for i, (symbol, data) in enumerate(list(mature_coins.items())[:10]):
            print(f"  {i+1}. {symbol}")
            if isinstance(data, dict):
                details = data.get('maturity_check', {}).get('details', {})
                print(f"     Свечей: {details.get('candles_count', 'N/A')}")
                print(f"     RSI мин: {details.get('rsi_min', 'N/A')}")
                print(f"     RSI макс: {details.get('rsi_max', 'N/A')}")
        
        # Проверяем несколько конкретных монет
        print(f"\n🧪 Проверка конкретных монет:")
        test_coins = ['BTC', 'ETH', 'BIO', 'REX', 'ZORA']
        
        r = requests.get('http://localhost:5001/api/bots/coins-with-rsi')
        all_coins = r.json().get('coins', {})
        
        for symbol in test_coins:
            if symbol in all_coins:
                # Проверяем зрелость через API
                is_mature = symbol in mature_coins
                coin = all_coins[symbol]
                
                print(f"\n  {symbol}:")
                print(f"    Зрелая: {'✅ Да' if is_mature else '❌ Нет'}")
                print(f"    RSI: {coin.get('rsi6h')}")
                print(f"    Signal: {coin.get('signal')}")
        
        # Проверяем логику: монеты без 200 свечей НЕ должны быть зрелыми
        print(f"\n🔍 Проверка: все зрелые монеты имеют {min_candles}+ свечей?")
        print(f"   (Это можно проверить только через прямой запрос к бирже)")
        print(f"   Логика в коде:")
        print(f"   1. Если свечей < {min_candles} → is_mature = False")
        print(f"   2. Если свечей ≥ {min_candles} → анализируем последние {min_candles} свечей")
        print(f"   3. Проверяем что RSI достигал ≤{min_rsi_low} и ≥{max_rsi_high}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_maturity()

