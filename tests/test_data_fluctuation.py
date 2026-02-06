#!/usr/bin/env python3
"""
🔍 ТЕСТ КОЛЕБАНИЙ ДАННЫХ В UI
Проверяем, что данные не "гуляют" между этапами обработки
"""

import requests
import time
import json
from datetime import datetime

API = "http://127.0.0.1:5001"

def get_coin_data(symbol):
    """Получает данные монеты"""
    try:
        response = requests.get(f"{API}/api/bots/coins-with-rsi", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get('coins', {}).get(symbol)
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении данных {symbol}: {e}")
        return None

def main():
    print("🔍 ТЕСТ КОЛЕБАНИЙ ДАННЫХ В UI")
    print("=" * 50)
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Даем серверу время на запуск
    print("\n⏳ Ожидание запуска сервера (30 секунд)...")
    time.sleep(30)
    
    # Монеты для мониторинга
    test_coins = ['GORK', 'IOTX', 'AUCTION', 'XAUT']
    
    print(f"\n🔍 Мониторим изменения данных для {len(test_coins)} монет...")
    print("=" * 50)
    
    # Сохраняем предыдущие состояния
    previous_states = {}
    
    for i in range(1, 31):  # 30 проверок с интервалом 2 секунды
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f"\n⏰ Проверка #{i} ({current_time}):")
        
        changes_detected = False
        
        for symbol in test_coins:
            coin_data = get_coin_data(symbol)
            
            if coin_data:
                current_state = {
                    'signal': coin_data.get('signal', 'UNKNOWN'),
                    'trend': coin_data.get('trend6h', 'UNKNOWN'),
                    'rsi': coin_data.get('rsi6h', 0),
                    'price': coin_data.get('price', 0)
                }
                
                # Проверяем изменения
                if symbol in previous_states:
                    prev_state = previous_states[symbol]
                    
                    if (current_state['signal'] != prev_state['signal'] or 
                        current_state['trend'] != prev_state['trend']):
                        
                        changes_detected = True
                        print(f"  🔄 {symbol}: ИЗМЕНЕНИЕ!")
                        print(f"    Сигнал: {prev_state['signal']} → {current_state['signal']}")
                        print(f"    Тренд: {prev_state['trend']} → {current_state['trend']}")
                        print(f"    RSI: {prev_state['rsi']:.1f} → {current_state['rsi']:.1f}")
                
                # Показываем текущее состояние
                if changes_detected or symbol not in previous_states:
                    print(f"  📊 {symbol}: {current_state['signal']} | {current_state['trend']} | RSI:{current_state['rsi']:.1f}")
                
                previous_states[symbol] = current_state
            else:
                print(f"  ❌ {symbol}: Данные недоступны")
        
        if not changes_detected and i > 1:
            print("  ✅ Изменений не обнаружено")
        
        time.sleep(2)  # Проверяем каждые 2 секунды
    
    print("\n" + "=" * 50)
    print("🎯 ЗАКЛЮЧЕНИЕ:")
    print("=" * 50)
    print("📊 Проверьте логи выше на предмет:")
    print("   - Колебаний сигналов (WAIT ↔ ENTER_LONG/ENTER_SHORT)")
    print("   - Колебаний трендов (NEUTRAL ↔ DOWN/UP)")
    print("   - Неожиданных изменений RSI")
    print("✅ Если данные стабильны - проблема решена!")

if __name__ == "__main__":
    main()
