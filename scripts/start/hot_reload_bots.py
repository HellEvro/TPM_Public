#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hot Reload для bots.py - автоматический перезапуск при изменении файла
"""

import os
import sys
import time
import subprocess
import signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class BotsReloadHandler(FileSystemEventHandler):
    """Обработчик изменений файла bots.py"""
    
    def __init__(self):
        self.bots_process = None
        self.last_reload = 0
        self.reload_cooldown = 5  # Минимальная пауза между перезапусками (секунды)
    
    def start_bots(self):
        """Запускает bots.py"""
        try:
            if self.bots_process:
                print("🔄 Останавливаем старый процесс...")
                self.bots_process.terminate()
                self.bots_process.wait(timeout=10)
            
            print("🚀 Запускаем bots.py...")
            self.bots_process = subprocess.Popen([
                sys.executable, 'bots.py'
            ], cwd=os.getcwd())
            
            print(f"✅ bots.py запущен (PID: {self.bots_process.pid})")
            
        except Exception as e:
            print(f"❌ Ошибка запуска bots.py: {e}")
    
    def stop_bots(self):
        """Останавливает bots.py"""
        try:
            if self.bots_process:
                print("🛑 Останавливаем bots.py...")
                self.bots_process.terminate()
                self.bots_process.wait(timeout=10)
                self.bots_process = None
                print("✅ bots.py остановлен")
        except Exception as e:
            print(f"❌ Ошибка остановки bots.py: {e}")
    
    def on_modified(self, event):
        """Вызывается при изменении файла"""
        if event.is_directory:
            return
            
        # Проверяем, что изменился именно bots.py
        if os.path.basename(event.src_path) == 'bots.py':
            current_time = time.time()
            
            # Защита от частых перезапусков
            if current_time - self.last_reload < self.reload_cooldown:
                return
            
            self.last_reload = current_time
            
            print(f"\n📝 Обнаружено изменение в bots.py: {time.strftime('%H:%M:%S')}")
            print("🔄 Перезапускаем bots.py...")
            
            self.start_bots()
    
    def on_created(self, event):
        """Вызывается при создании файла"""
        if os.path.basename(event.src_path) == 'bots.py':
            self.on_modified(event)

def signal_handler(signum, frame):
    """Обработчик сигналов для корректного завершения"""
    print("\n🛑 Получен сигнал завершения...")
    handler.stop_bots()
    observer.stop()
    sys.exit(0)

def main():
    """Основная функция"""
    global handler, observer
    
    print("🔥 Hot Reload для bots.py запущен!")
    print("📁 Мониторим изменения в файле bots.py")
    print("⏹️  Нажмите Ctrl+C для остановки")
    print("-" * 50)
    
    # Создаем обработчик
    handler = BotsReloadHandler()
    
    # Запускаем bots.py в первый раз
    handler.start_bots()
    
    # Настраиваем мониторинг файлов
    observer = Observer()
    observer.schedule(handler, path='.', recursive=False)
    observer.start()
    
    # Регистрируем обработчики сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Ждем изменения файлов
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Завершение работы...")
    finally:
        handler.stop_bots()
        observer.stop()

if __name__ == '__main__':
    main()
