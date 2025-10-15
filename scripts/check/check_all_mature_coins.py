#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для проверки зрелости всех монет
"""
import sys
import io
import requests
import time

# Устанавливаем UTF-8 для консоли Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_all_mature_coins():
    """Проверяет зрелость всех монет через API"""
    print("=" * 60)
    print("🔍 ПРОВЕРКА ЗРЕЛОСТИ ВСЕХ МОНЕТ")
    print("=" * 60)
    
    try:
        # Получаем список всех монет
        print("📊 Получаем список всех монет...")
        response = requests.get('http://localhost:5001/api/bots/coins-with-rsi', timeout=30)
        if response.status_code != 200:
            print(f"❌ Ошибка получения списка монет: {response.status_code}")
            return
        
        data = response.json()
        coins = data.get('coins', {})
        
        if not coins:
            print("❌ Нет данных о монетах")
            return
        
        print(f"📈 Найдено {len(coins)} монет для проверки")
        print("🔄 Начинаем проверку зрелости...")
        
        mature_count = 0
        immature_count = 0
        error_count = 0
        
        for i, symbol in enumerate(coins.keys(), 1):
            try:
                print(f"[{i}/{len(coins)}] Проверяем {symbol}...", end=" ")
                
                # Принудительно обновляем RSI данные для монеты
                update_response = requests.get(f'http://localhost:5001/api/bots/coins-with-rsi?refresh_symbol={symbol}', timeout=15)
                
                if update_response.status_code == 200:
                    print("✅")
                    mature_count += 1
                else:
                    print(f"❌ {update_response.status_code}")
                    error_count += 1
                
                # Небольшая задержка чтобы не перегрузить API
                time.sleep(0.1)
                
                # Показываем прогресс каждые 50 монет
                if i % 50 == 0:
                    print(f"📊 Прогресс: {i}/{len(coins)} монет проверено")
                
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                error_count += 1
        
        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТЫ ПРОВЕРКИ:")
        print(f"✅ Успешно проверено: {mature_count}")
        print(f"❌ Ошибок: {error_count}")
        print(f"📈 Всего монет: {len(coins)}")
        
        # Проверяем сколько зрелых монет теперь в хранилище
        print("\n🔍 Проверяем хранилище зрелых монет...")
        try:
            import json
            with open('data/mature_coins.json', 'r', encoding='utf-8') as f:
                mature_data = json.load(f)
            print(f"🏆 Зрелых монет в хранилище: {len(mature_data)}")
            if mature_data:
                print(f"📝 Примеры зрелых монет: {list(mature_data.keys())[:10]}")
        except Exception as e:
            print(f"❌ Ошибка чтения хранилища: {e}")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")

if __name__ == '__main__':
    check_all_mature_coins()
