#!/usr/bin/env python3
"""
🔍 ТЕСТ БЛОКИРОВКИ ФИЛЬТРОВ
Проверяем, что сигналы правильно блокируются фильтрами
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
    print("🔍 ТЕСТ БЛОКИРОВКИ ФИЛЬТРОВ")
    print("=" * 50)
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Даем серверу время на запуск
    print("\n⏳ Ожидание запуска сервера (30 секунд)...")
    time.sleep(30)
    
    # Монеты для проверки (особенно JST, которая должна быть заблокирована)
    test_coins = ['JST', 'GORK', 'IOTX', 'AUCTION', 'XAUT']
    
    print(f"\n🔍 Проверяем блокировку фильтров для {len(test_coins)} монет...")
    print("=" * 50)
    
    for symbol in test_coins:
        print(f"\n📊 Анализ монеты {symbol}:")
        coin_data = get_coin_data(symbol)
        
        if coin_data:
            signal = coin_data.get('signal', 'UNKNOWN')
            rsi = coin_data.get('rsi6h', 0)
            trend = coin_data.get('trend6h', 'UNKNOWN')
            
            # Проверяем флаги блокировки
            blocked_by_exit_scam = coin_data.get('blocked_by_exit_scam', False)
            blocked_by_rsi_time = coin_data.get('blocked_by_rsi_time', False)
            blocked_by_scope = coin_data.get('blocked_by_scope', False)
            is_mature = coin_data.get('is_mature', True)
            
            # Получаем детальную информацию о фильтрах
            exit_scam_info = coin_data.get('exit_scam_info', {})
            time_filter_info = coin_data.get('time_filter_info', {})
            
            print(f"  📈 Основные данные:")
            print(f"    Сигнал: {signal}")
            print(f"    RSI: {rsi:.1f}")
            print(f"    Тренд: {trend}")
            
            print(f"  🚫 Флаги блокировки:")
            print(f"    ExitScam: {'❌ ЗАБЛОКИРОВАН' if blocked_by_exit_scam else '✅ ПРОЙДЕН'}")
            print(f"    RSI Time: {'❌ ЗАБЛОКИРОВАН' if blocked_by_rsi_time else '✅ ПРОЙДЕН'}")
            print(f"    Scope: {'❌ ЗАБЛОКИРОВАН' if blocked_by_scope else '✅ ПРОЙДЕН'}")
            print(f"    Зрелость: {'✅ ЗРЕЛАЯ' if is_mature else '❌ НЕЗРЕЛАЯ'}")
            
            if exit_scam_info:
                print(f"  🔍 ExitScam детали: {exit_scam_info.get('reason', 'Нет информации')}")
            
            if time_filter_info:
                print(f"  🔍 RSI Time детали: {time_filter_info.get('reason', 'Нет информации')}")
                if 'last_extreme_candles_ago' in time_filter_info:
                    print(f"    Последний экстремум: {time_filter_info['last_extreme_candles_ago']} свечей назад")
            
            # Проверяем логику блокировки
            if signal in ['ENTER_LONG', 'ENTER_SHORT']:
                if blocked_by_exit_scam or blocked_by_rsi_time or blocked_by_scope or not is_mature:
                    print(f"  ⚠️ ОШИБКА: Показывается сигнал {signal}, но есть блокировки!")
                    print(f"    Это недопустимо! Сигнал должен быть WAIT")
                else:
                    print(f"  ✅ ОК: Сигнал {signal} разрешен - все фильтры пройдены")
            else:
                print(f"  ✅ ОК: Сигнал {signal} - нет торговых сигналов")
                
        else:
            print(f"  ❌ Данные недоступны")
    
    print("\n" + "=" * 50)
    print("🎯 ЗАКЛЮЧЕНИЕ:")
    print("=" * 50)
    print("📊 Проверьте результаты выше:")
    print("   - Нет ли сигналов ENTER_LONG/ENTER_SHORT при заблокированных фильтрах")
    print("   - Правильно ли работают ExitScam и RSI Time фильтры")
    print("   - Корректно ли отображается информация о блокировках")
    print("✅ Если все сигналы корректны - проблема решена!")

if __name__ == "__main__":
    main()
