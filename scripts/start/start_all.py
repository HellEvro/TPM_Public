#!/usr/bin/env python3
"""
Скрипт для запуска всех компонентов InfoBot
- Основное приложение (app.py) на порту 5000
- Сервис ботов (bots.py) на порту 5001
"""

import os
import sys
import time
import signal
import subprocess
import threading
from datetime import datetime

def print_banner():
    """Красивый баннер"""
    print("=" * 60)
    print("🚀 InfoBot - Complete Trading System")
    print("=" * 60)
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    print("📊 Main App:    http://localhost:5000")
    print("🤖 Bots API:    http://localhost:5001")
    print("")
    print("Команды управления:")
    print("  Ctrl+C - Остановить все сервисы")
    print("=" * 60)

def start_service(name, script, port):
    """Запуск отдельного сервиса"""
    try:
        print(f"🔄 Запуск {name}...")
        
        # Запуск в отдельном процессе
        process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print(f"✅ {name} запущен (PID: {process.pid})")
        return process
        
    except Exception as e:
        print(f"❌ Ошибка запуска {name}: {str(e)}")
        return None

def monitor_process(process, name):
    """Мониторинг процесса и вывод логов"""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                # Префикс для логов каждого сервиса
                prefix = f"[{name}]"
                print(f"{prefix} {line.rstrip()}")
        
        process.stdout.close()
        return_code = process.wait()
        print(f"⚠️ {name} завершился с кодом {return_code}")
        
    except Exception as e:
        print(f"❌ Ошибка мониторинга {name}: {str(e)}")

def main():
    """Основная функция"""
    print_banner()
    
    processes = []
    monitor_threads = []
    
    # Создаем директорию для логов
    if not os.path.exists('logs'):
        os.makedirs('logs')
        print("📁 Создана директория logs/")
    
    try:
        # Запускаем сервис ботов первым
        print("\n🤖 Запуск сервиса ботов...")
        bots_process = start_service("BOTS", "bots.py", 5001)
        if bots_process:
            processes.append(("BOTS", bots_process))
            
            # Запускаем мониторинг в отдельном потоке
            monitor_thread = threading.Thread(
                target=monitor_process, 
                args=(bots_process, "BOTS"),
                daemon=True
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)
        
        # Ждем немного, чтобы сервис ботов запустился
        time.sleep(3)
        
        # Запускаем основное приложение
        print("\n📊 Запуск основного приложения...")
        app_process = start_service("MAIN", "app.py", 5000)
        if app_process:
            processes.append(("MAIN", app_process))
            
            # Запускаем мониторинг в отдельном потоке
            monitor_thread = threading.Thread(
                target=monitor_process, 
                args=(app_process, "MAIN"),
                daemon=True
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)
        
        if not processes:
            print("❌ Не удалось запустить ни один сервис!")
            return
        
        print(f"\n✅ Все сервисы запущены! ({len(processes)} из 2)")
        print("\n🌐 Откройте http://localhost:5000 в браузере")
        print("⏹️  Нажмите Ctrl+C для остановки всех сервисов\n")
        
        # Ожидаем сигнал прерывания
        try:
            while True:
                # Проверяем, что все процессы еще живы
                for name, process in processes:
                    if process.poll() is not None:
                        print(f"⚠️  Процесс {name} неожиданно завершился!")
                        return
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Получен сигнал прерывания...")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {str(e)}")
    
    finally:
        # Корректное завершение всех процессов
        print("\n🛑 Завершение работы всех сервисов...")
        
        for name, process in processes:
            try:
                print(f"⏹️  Остановка {name}...")
                process.terminate()
                
                # Ждем корректного завершения
                try:
                    process.wait(timeout=5)
                    print(f"✅ {name} корректно завершен")
                except subprocess.TimeoutExpired:
                    print(f"⚠️  {name} не отвечает, принудительное завершение...")
                    process.kill()
                    process.wait()
                    print(f"🔴 {name} принудительно завершен")
                    
            except Exception as e:
                print(f"❌ Ошибка при завершении {name}: {str(e)}")
        
        print("\n🔚 Все сервисы остановлены.")
        print(f"⏰ Время работы: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
