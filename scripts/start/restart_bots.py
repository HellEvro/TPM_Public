#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрый перезапуск bots.py через API
"""

import sys
import io
import requests
import json
import time

# Исправляем кодировку для Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def restart_bots_service():
    """Перезапускает сервис ботов через API"""
    try:
        print("🔄 Перезапускаем сервис ботов...")
        
        # Отправляем запрос на перезапуск
        response = requests.post(
            'http://localhost:5001/api/bots/restart-service',
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result.get('message', 'Сервис перезапущен')}")
            return True
        else:
            print(f"❌ Ошибка перезапуска: {response.status_code}")
            print(f"Ответ: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Сервис ботов недоступен на порту 5001")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def reload_modules():
    """Перезагружает модули"""
    try:
        print("🔄 Перезагружаем модули...")
        
        response = requests.post(
            'http://localhost:5001/api/bots/reload-modules',
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result.get('message', 'Модули перезагружены')}")
            return True
        else:
            print(f"❌ Ошибка перезагрузки модулей: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def refresh_rsi_for_coin(symbol):
    """Обновляет RSI данные для конкретной монеты"""
    try:
        print(f"🔄 Обновляем RSI данные для {symbol}...")
        
        response = requests.post(
            f'http://localhost:5001/api/bots/refresh-rsi/{symbol}',
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result.get('message', f'RSI данные для {symbol} обновлены')}")
            return True
        else:
            print(f"❌ Ошибка обновления RSI: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def main():
    """Основная функция"""
    import sys
    
    print("🚀 Умный перезапуск bots.py")
    print("=" * 60)
    print("📋 Что будет перезагружено:")
    print("  ✅ Модули Python (bot_engine, bots)")
    print("  ✅ Конфигурация Auto Bot")
    print("  ✅ Состояние ботов")
    print("  ⏭️  RSI данные НЕ трогаем (используется кэш)")
    print("=" * 60)
    
    # Проверяем, передан ли символ монеты для обновления
    symbol_to_refresh = None
    if len(sys.argv) > 1:
        symbol_to_refresh = sys.argv[1].upper()
        print(f"\n🎯 Дополнительно: обновить RSI данные для {symbol_to_refresh}")
    
    # Шаг 1: Перезагрузка модулей
    print("\n📦 Шаг 1/2: Перезагрузка Python модулей...")
    if reload_modules():
        print("✅ Модули перезагружены")
    else:
        print("❌ Ошибка перезагрузки модулей")
        return
    
    # Шаг 2: Перезагрузка сервиса (конфиг + состояние ботов)
    print("\n⚙️ Шаг 2/2: Перезагрузка сервиса (конфиг + боты)...")
    try:
        response = requests.post(
            'http://localhost:5001/api/bots/restart-service',
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result.get('message', 'Сервис перезагружен')}")
        else:
            print(f"⚠️ Частичная перезагрузка (код: {response.status_code})")
    except Exception as e:
        print(f"⚠️ Не удалось перезагрузить сервис: {e}")
    
    # Ждём применения изменений
    print("\n⏳ Ждем 2 секунды для применения изменений...")
    time.sleep(2)
    
    # Опционально: обновить RSI для конкретной монеты
    if symbol_to_refresh:
        print(f"\n🔄 Обновление RSI данных для {symbol_to_refresh}...")
        refresh_rsi_for_coin(symbol_to_refresh)
    
    # Проверка здоровья сервиса
    print("\n🏥 Проверка здоровья сервиса...")
    try:
        response = requests.get('http://localhost:5001/health', timeout=5)
        if response.status_code == 200:
            print("✅ Сервис работает корректно")
            print("\n" + "=" * 60)
            print("🎉 Перезапуск завершен успешно!")
            print("=" * 60)
        else:
            print(f"⚠️ Сервис отвечает с кодом: {response.status_code}")
    except Exception as e:
        print(f"❌ Сервис недоступен: {e}")
        print("\n💡 Возможно, нужен полный перезапуск: python bots.py")

if __name__ == '__main__':
    main()
