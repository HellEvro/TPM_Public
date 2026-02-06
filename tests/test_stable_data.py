#!/usr/bin/env python3
"""
🔍 ТЕСТ СТАБИЛЬНОСТИ ДАННЫХ
Проверяем, что данные не "гуляют" после исправлений
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

def get_service_status():
    """Получает статус сервиса"""
    try:
        response = requests.get(f"{API}/api/bots/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении статуса сервиса: {e}")
        return None

def main():
    print("🔍 ТЕСТ СТАБИЛЬНОСТИ ДАННЫХ ПОСЛЕ ИСПРАВЛЕНИЙ")
    print("=" * 60)
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Даем серверу время на запуск
    print("\n⏳ Ожидание запуска сервера (30 секунд)...")
    time.sleep(30)
    
    # Монеты для мониторинга
    test_coins = ['GORK', 'IOTX', 'AUCTION', 'XAUT']
    
    print(f"\n🔍 Мониторим стабильность данных для {len(test_coins)} монет...")
    print("=" * 60)
    
    # Сохраняем предыдущие состояния
    previous_states = {}
    changes_count = 0
    stable_checks = 0
    
    for i in range(1, 21):  # 20 проверок с интервалом 3 секунды
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
                        changes_count += 1
                        print(f"  🔄 {symbol}: ИЗМЕНЕНИЕ #{changes_count}!")
                        print(f"    Сигнал: {prev_state['signal']} → {current_state['signal']}")
                        print(f"    Тренд: {prev_state['trend']} → {current_state['trend']}")
                        print(f"    RSI: {prev_state['rsi']:.1f} → {current_state['rsi']:.1f}")
                    else:
                        stable_checks += 1
                
                # Показываем текущее состояние
                if changes_detected or symbol not in previous_states:
                    print(f"  📊 {symbol}: {current_state['signal']} | {current_state['trend']} | RSI:{current_state['rsi']:.1f}")
                
                previous_states[symbol] = current_state
            else:
                print(f"  ❌ {symbol}: Данные недоступны")
        
        if not changes_detected and i > 1:
            print("  ✅ Изменений не обнаружено")
        
        # Проверяем статус сервиса
        status = get_service_status()
        if status and status.get('success'):
            print(f"  📊 Сервис: {status.get('status')}, версия данных: {status.get('data_version', 'N/A')}")
        
        time.sleep(3)  # Проверяем каждые 3 секунды
    
    print("\n" + "=" * 60)
    print("🎯 РЕЗУЛЬТАТЫ ТЕСТА:")
    print("=" * 60)
    print(f"📊 Всего проверок: {i}")
    print(f"📊 Стабильных проверок: {stable_checks}")
    print(f"📊 Изменений данных: {changes_count}")
    
    if changes_count == 0:
        print("✅ ОТЛИЧНО! Данные полностью стабильны!")
        print("✅ Проблема 'гуляющих' данных решена!")
    elif changes_count <= 2:
        print("⚠️ ХОРОШО! Минимальные изменения данных")
        print("⚠️ Возможно, это нормальные обновления")
    else:
        print("❌ ПЛОХО! Данные все еще нестабильны")
        print("❌ Требуется дополнительная отладка")
    
    print(f"\n📈 Стабильность: {(stable_checks / (i * len(test_coins))) * 100:.1f}%")

if __name__ == "__main__":
    main()
