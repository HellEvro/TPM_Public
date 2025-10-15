#!/usr/bin/env python3
"""
Главный файл bots.py - импортирует все модули
"""

# Базовые импорты
import os
import sys

# Настройка кодировки для Windows консоли
if os.name == 'nt':
    try:
        # Пытаемся установить UTF-8 для консоли Windows
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        # Если не получилось, пробуем через os
        try:
            import subprocess
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
        except:
            pass

# Проверка наличия конфигурации ПЕРЕД всеми импортами
if not os.path.exists('app/config.py'):
    print("\n" + "="*80)
    print("❌ ОШИБКА: Файл конфигурации не найден!")
    print("="*80)
    print()
    print("📝 Для первого запуска выполните:")
    print()
    print("   1. Скопируйте файл конфигурации:")
    if os.name == 'nt':  # Windows
        print("      copy app\\config.example.py app\\config.py")
    else:  # Linux/Mac
        print("      cp app/config.example.py app/config.py")
    print()
    print("   2. Отредактируйте app/config.py:")
    print("      - Добавьте свои API ключи бирж")
    print("      - Настройте Telegram (опционально)")
    print()
    print("   3. Запустите снова:")
    print("      python bots.py")
    print()
    print("   📖 Подробная инструкция: docs/INSTALL.md")
    print()
    print("="*80)
    print()
    sys.exit(1)

import signal
import threading
import time
import logging
import json
from datetime import datetime
from flask import Flask
from flask_cors import CORS

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Функция проверки порта (должна быть до всех импортов)
from bots_modules.imports_and_globals import check_and_stop_existing_bots_processes

# КРИТИЧЕСКИ ВАЖНО: Проверяем порт 5001 ПЕРЕД загрузкой остальных модулей
if __name__ == '__main__':
    # Эта проверка должна быть ПЕРВОЙ при запуске
    can_continue = check_and_stop_existing_bots_processes()
    if not can_continue:
        print("Не удалось освободить порт 5001, завершаем работу")
        sys.exit(1)

# Импорт цветного логирования
from utils.color_logger import setup_color_logging

# Импортируем все модули
print("Загрузка модулей...")
from bots_modules.imports_and_globals import *
from bots_modules.calculations import *
from bots_modules.maturity import *
from bots_modules.optimal_ema import *
from bots_modules.filters import *
from bots_modules.bot_class import *
from bots_modules.sync_and_cache import *
from bots_modules.workers import *
from bots_modules.init_functions import *
from bots_modules.api_endpoints import *

print("Все модули загружены!")

# Импорт системы истории ботов (после импорта модулей, чтобы логирование было настроено)
try:
    from bot_engine.bot_history import (
        bot_history_manager, log_bot_start, log_bot_stop, log_bot_signal,
        log_position_opened, log_position_closed
    )
    BOT_HISTORY_AVAILABLE = True
    logger = logging.getLogger('BotsService')
    logger.info("[BOT_HISTORY] ✅ Модуль bot_history загружен успешно")
except ImportError as e:
    print(f"[WARNING] Модуль bot_history недоступен: {e}")
    # Создаем заглушки
    class DummyHistoryManager:
        def get_bot_history(self, *args, **kwargs): return []
        def get_bot_trades(self, *args, **kwargs): return []
        def get_bot_statistics(self, *args, **kwargs): return {}
        def clear_history(self, *args, **kwargs): pass
    
    bot_history_manager = DummyHistoryManager()
    def log_bot_start(*args, **kwargs): pass
    def log_bot_stop(*args, **kwargs): pass
    def log_bot_signal(*args, **kwargs): pass
    def log_position_opened(*args, **kwargs): pass
    def log_position_closed(*args, **kwargs): pass
    BOT_HISTORY_AVAILABLE = False

# Настройка логирования
setup_color_logging()

# Добавляем файловый логгер
file_handler = logging.FileHandler('logs/bots.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('[BOTS] %(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)

# Настройка кодировки для stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger('BotsService')

# Глобальные переменные для shutdown
graceful_shutdown = False

# Signal handlers
def signal_handler(signum, frame):
    """Обработчик сигналов для graceful shutdown"""
    global graceful_shutdown
    
    logger.warning(f"Получен сигнал {signum}, начинаем graceful shutdown...")
    graceful_shutdown = True
    shutdown_flag.set()
    
    # Принудительно останавливаем Flask
    print("\n🛑 Остановка сервиса...")
    cleanup_bot_service()
    print("✅ Сервис остановлен")
    
    # Убиваем все потоки принудительно
    os._exit(0)

_cleanup_done = False

def cleanup_bot_service():
    """Очистка ресурсов перед остановкой"""
    global _cleanup_done
    
    if _cleanup_done:
        return
    
    _cleanup_done = True
    
    logger.info("=" * 80)
    logger.info("ОСТАНОВКА СИСТЕМЫ INFOBOT")
    logger.info("=" * 80)
    
    try:
        if async_processor:
            logger.info("Остановка асинхронного процессора...")
            stop_async_processor()
        
        logger.info("Сохранение состояния ботов...")
        save_bots_state()
        
        logger.info("Сохранение хранилища зрелых монет...")
        save_mature_coins_storage()
        
        logger.info("Система остановлена")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Ошибка при очистке: {e}")

def run_bots_service():
    """Запуск Flask сервера для API ботов"""
    try:
        logger.info("=" * 80)
        logger.info("ЗАПУСК BOTS SERVICE API (Порт 5001)")
        logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print("✅ BOTS SERVICE ЗАПУЩЕН И РАБОТАЕТ!")
        print("=" * 80)
        print("🌐 API доступен на: http://localhost:5001")
        print("📊 Статус: http://localhost:5001/api/status")
        print("🤖 Боты: http://localhost:5001/api/bots")
        print("=" * 80)
        print("💡 Нажмите Ctrl+C для остановки")
        print("=" * 80 + "\n")
        
        bots_app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f"Ошибка запуска Flask сервера: {e}")
        raise

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    import atexit
    atexit.register(cleanup_bot_service)
    
    try:
        from bots_modules.workers import auto_save_worker, auto_bot_worker
        
        load_auto_bot_config()
        
        try:
            init_bot_service()
        except Exception as init_error:
            logger.error(f"Ошибка инициализации (продолжаем запуск): {init_error}")
            import traceback
            traceback.print_exc()
        
        # Запускаем воркер Optimal EMA
        try:
            from bot_engine.optimal_ema_worker import start_optimal_ema_worker
            optimal_ema_worker = start_optimal_ema_worker(update_interval=21600)  # 6 часов
            if optimal_ema_worker:
                logger.info("📊 Optimal EMA Worker запущен (обновление каждые 6 часов)")
            else:
                logger.warning("⚠️ Optimal EMA Worker не запущен (скрипт не найден)")
        except Exception as ema_error:
            logger.warning(f"⚠️ Не удалось запустить Optimal EMA Worker: {ema_error}")
        
        auto_save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        auto_save_thread.start()
        logger.info("Auto Save Worker запущен")
        
        auto_bot_thread = threading.Thread(target=auto_bot_worker, daemon=True)
        auto_bot_thread.start()
        logger.info("Auto Bot Worker запущен")
        
        run_bots_service()
        
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания...")
        print("\n🛑 Остановка сервиса...")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            cleanup_bot_service()
            print("✅ Сервис остановлен\n")
        except:
            pass
        os._exit(0)
