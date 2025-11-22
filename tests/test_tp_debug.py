#!/usr/bin/env python3
import requests
import json

# Получаем монеты с сигналами
r = requests.get('http://localhost:5001/api/bots/coins-with-rsi')
data = r.json()

print("Монеты с сигналами:")
signals_found = False
for coin in data.get('coins', []):
    if coin.get('signal') in ['ENTER_LONG', 'ENTER_SHORT']:
        print(f"{coin['symbol']}: {coin['signal']} (RSI: {coin.get('rsi6h', 'N/A')})")
        signals_found = True

if not signals_found:
    print("Нет монет с сигналами ENTER_LONG/ENTER_SHORT")
    print("Первые 5 монет:")
    for coin in data.get('coins', [])[:5]:
        print(f"{coin['symbol']}: {coin.get('signal', 'N/A')} (RSI: {coin.get('rsi6h', 'N/A')})")
