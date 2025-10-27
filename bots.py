#!/usr/bin/env python3
"""
Главный файл bots.py - импортирует все модули
"""

# Базовые импорты
import os
import sys

# 🔍 ТРЕЙСИНГ из конфига (после импорта sys, но до остальных импортов)
try:
    # Читаем настройку трейсинга из конфига
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bot_engine.bot_config import SystemConfig
    ENABLE_TRACE = SystemConfig.ENABLE_CODE_TRACING
    
    if ENABLE_TRACE:
        from trace_debug import enable_trace
        enable_trace()
        print("=" * 80)
        print("TRACE: ENABLED - all code execution will be logged with timing")
        print("WARNING: This will slow down the system significantly!")
        print("=" * 80, flush=True)
    else:
        print("[INFO] Code tracing DISABLED (set SystemConfig.ENABLE_CODE_TRACING = True to enable)")
except Exception as e:
    print(f"[WARNING] Could not initialize tracing: {e}")
    ENABLE_TRACE = False

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

# Проверка API ключей
def check_api_keys():
    """Проверяет наличие настроенных API ключей"""
    try:
        # Проверяем наличие файла с ключами
        if not os.path.exists('app/keys.py'):
            return False
            
        from app.config import EXCHANGES, ACTIVE_EXCHANGE
        active_exchange = EXCHANGES.get(ACTIVE_EXCHANGE, {})
        api_key = active_exchange.get('api_key', '')
        api_secret = active_exchange.get('api_secret', '')
        
        # Проверяем что ключи не пустые и не содержат "YOUR_" (из примера)
        if not api_key or not api_secret:
            return False
        if 'YOUR_' in api_key or 'YOUR_' in api_secret:
            return False
        if api_key == 'YOUR_API_KEY_HERE' or api_secret == 'YOUR_SECRET_KEY_HERE':
            return False
            
        return True
    except:
        return False

# КРИТИЧЕСКИ ВАЖНО: Проверяем порт 5001 ПЕРЕД загрузкой остальных модулей
if __name__ == '__main__':
    # Эта проверка должна быть ПЕРВОЙ при запуске
    can_continue = check_and_stop_existing_bots_processes()
    if not can_continue:
        print("Не удалось освободить порт 5001, завершаем работу")
        sys.exit(1)
    
    # Проверяем API ключи
    if not check_api_keys():
        print("\n" + "="*80)
        print("⚠️  ВНИМАНИЕ: API ключи не настроены!")
        print("="*80)
        print()
        print("📌 Текущий статус:")
        try:
            from app.config import ACTIVE_EXCHANGE
            print(f"   Биржа: {ACTIVE_EXCHANGE}")
        except:
            print("   Биржа: НЕ ОПРЕДЕЛЕНА")
        
        if not os.path.exists('app/keys.py'):
            print("   Файл с ключами: app/keys.py НЕ НАЙДЕН")
        else:
            print("   API ключи: НЕ НАСТРОЕНЫ или СОДЕРЖАТ ПРИМЕРЫ")
        print()
        print("💡 Что нужно сделать:")
        print("   1. Создайте app/keys.py с реальными ключами от биржи")
        print("   2. Или добавьте ключи в app/config.py (EXCHANGES)")
        print("   3. Перезапустите bots.py")
        print()
        print("⚠️  Сервис запущен, но торговля НЕВОЗМОЖНА без ключей!")
        print("   Будут ошибки: 'Http status code is not 200. (ErrCode: 401)'")
        print()
        print("="*80)
        print()

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

# Импорт системы истории ботов (ПЕРЕД импортом API endpoints!)
try:
    print("[BOT_HISTORY] 🔄 Попытка импорта bot_history...")
    from bot_engine.bot_history import (
        bot_history_manager, log_bot_start, log_bot_stop, log_bot_signal,
        log_position_opened, log_position_closed
    )
    print(f"[BOT_HISTORY] ✅ Импорт успешен, bot_history_manager: {bot_history_manager}")
    BOT_HISTORY_AVAILABLE = True
    logger = logging.getLogger('BotsService')
    logger.info("[BOT_HISTORY] ✅ Модуль bot_history загружен успешно")
    
    # Устанавливаем bot_history_manager в глобальный модуль
    import bots_modules.imports_and_globals as globals_module
    globals_module.bot_history_manager = bot_history_manager
    globals_module.BOT_HISTORY_AVAILABLE = True
    print(f"[BOT_HISTORY] ✅ Установлен в глобальный модуль: {globals_module.bot_history_manager}")
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
    
    # Устанавливаем заглушку в глобальный модуль
    import bots_modules.imports_and_globals as globals_module
    globals_module.bot_history_manager = bot_history_manager
    globals_module.BOT_HISTORY_AVAILABLE = False
    print(f"[BOT_HISTORY] ⚠️ Установлена заглушка в глобальный модуль: {globals_module.bot_history_manager}")
except Exception as e:
    print(f"[ERROR] Неожиданная ошибка при импорте bot_history: {e}")
    import traceback
    traceback.print_exc()

# Теперь импортируем API endpoints (после установки bot_history_manager)
from bots_modules.api_endpoints import *

print("Все модули загружены!")

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

def open_firewall_port_5001():
    """Открывает порт 5001 в брандмауэре при запуске (Windows/macOS/Linux)"""
    try:
        import subprocess
        import platform
        
        print("[BOTS] 🔥 Проверка открытия порта 5001 в брандмауэре...")
        
        system = platform.system()
        port = 5001
        
        if system == 'Windows':
            # Проверяем правило для порта 5001
            result = subprocess.run(
                ['netsh', 'advfirewall', 'firewall', 'show', 'rule', 'name=InfoBot Bot Service'],
                capture_output=True,
                text=True
            )
            
            if 'InfoBot Bot Service' not in result.stdout:
                print("[BOTS] 🔥 Открываем порт 5001...")
                subprocess.run([
                    'netsh', 'advfirewall', 'firewall', 'add', 'rule',
                    'name=InfoBot Bot Service',
                    'dir=in',
                    'action=allow',
                    'protocol=TCP',
                    f'localport={port}'
                ], check=True)
                print("[BOTS] ✅ Порт 5001 открыт")
            else:
                print("[BOTS] ✅ Порт 5001 уже открыт")
        
        elif system == 'Darwin':  # macOS
            print("[BOTS] 💡 На macOS откройте порт 5001 вручную")
        
        elif system == 'Linux':
            try:
                # Проверяем наличие ufw
                subprocess.run(['which', 'ufw'], check=True)
                result = subprocess.run(['ufw', 'status'], capture_output=True, text=True)
                if f'{port}/tcp' not in result.stdout:
                    subprocess.run(['sudo', 'ufw', 'allow', str(port), '/tcp'], check=True)
                    print(f"[BOTS] ✅ Порт {port} открыт")
                else:
                    print(f"[BOTS] ✅ Порт {port} уже открыт")
            except:
                print(f"[BOTS] ⚠️ Настройте порт {port} вручную")
        
        else:
            print(f"[BOTS] ⚠️ Неизвестная система: {system}")
            print("[BOTS] 💡 Настройте порт вручную см. docs/INSTALL.md")
            
    except Exception as e:
        print(f"[BOTS] ⚠️ Не удалось открыть порт 5001 автоматически: {e}")
        print("[BOTS] 💡 Откройте порт вручную см. docs/INSTALL.md")

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
        # 🔄 Останавливаем непрерывный загрузчик данных
        logger.info("🔄 Останавливаем непрерывный загрузчик данных...")
        from bots_modules.continuous_data_loader import stop_continuous_loader
        stop_continuous_loader()
        
        if async_processor:
            logger.info("Остановка асинхронного процессора...")
            stop_async_processor()
        
        logger.info("Сохранение состояния ботов...")
        save_bots_state()
        
        # ✅ ВОССТАНОВЛЕНО: Сохранение зрелых монет при завершении (ТОЛЬКО если данные валидны)
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
    except SystemExit as e:
        if e.code == 42:
            # Специальный код для горячей перезагрузки
            logger.info("🔄 Горячая перезагрузка: перезапуск сервера...")
            print("🔄 Горячая перезагрузка: перезапуск сервера...")
            # Запускаем новый процесс
            import subprocess
            subprocess.Popen([sys.executable] + sys.argv)
            sys.exit(0)
        else:
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
        from bots_modules.workers import auto_save_worker, auto_bot_worker, positions_monitor_worker
        
        load_auto_bot_config()
        
        # Инициализируем ботов в отдельном потоке, чтобы не блокировать запуск сервера
        def init_bots_async():
            try:
                init_bot_service()
            except Exception as init_error:
                logger.error(f"Ошибка инициализации (продолжаем запуск): {init_error}")
                import traceback
                traceback.print_exc()
        
        init_thread = threading.Thread(target=init_bots_async, daemon=True)
        init_thread.start()
        logger.info("🔧 Инициализация ботов начата в фоне...")
        
        # ✅ Optimal EMA Worker - расчет оптимальных EMA в фоне
        from bot_engine.optimal_ema_worker import start_optimal_ema_worker
        optimal_ema_worker = start_optimal_ema_worker(update_interval=21600) # 6 часов
        if optimal_ema_worker:
            logger.info("✅ Optimal EMA Worker запущен (обновление каждые 6 часов)")
        else:
            logger.warning("⚠️ Не удалось запустить Optimal EMA Worker")
        
        auto_save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        auto_save_thread.start()
        logger.info("Auto Save Worker запущен")
        
        auto_bot_thread = threading.Thread(target=auto_bot_worker, daemon=True)
        auto_bot_thread.start()
        logger.info("Auto Bot Worker запущен")
        
        # ✅ Positions Monitor Worker - мониторинг позиций каждые 5 секунд
        positions_monitor_thread = threading.Thread(target=positions_monitor_worker, daemon=True)
        positions_monitor_thread.start()
        logger.info("📊 Positions Monitor Worker запущен (обновление каждые 5с)")
        
        # Инициализируем AI Manager (проверка лицензии и загрузка модулей)
        ai_manager = None
        try:
            from bot_engine.bot_config import AIConfig
            
            if AIConfig.AI_ENABLED:
                logger.info("🤖 Инициализация AI модулей...")
                from bot_engine.ai.ai_manager import get_ai_manager
                ai_manager = get_ai_manager()
                
                # Если лицензия валидна и модули загружены, запускаем Auto Trainer
                if ai_manager.is_available() and AIConfig.AI_AUTO_TRAIN_ENABLED:
                    from bot_engine.ai.auto_trainer import start_auto_trainer
                    start_auto_trainer()
                    logger.info("🤖 AI Auto Trainer запущен (автообновление данных и переобучение)")
                elif ai_manager.is_available():
                    logger.info("🤖 AI модули активны (автообучение отключено)")
                else:
                    logger.warning("⚠️ AI модули не загружены (проверьте лицензию)")
            else:
                logger.info("ℹ️ AI модули отключены в конфигурации")
        except ImportError as ai_import_error:
            logger.debug(f"AI модули не доступны: {ai_import_error}")
        except Exception as ai_error:
            logger.warning(f"⚠️ Ошибка инициализации AI: {ai_error}")
        
        # Открываем порт 5001 в брандмауэре
        open_firewall_port_5001()
        
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
            # Останавливаем AI Auto Trainer
            try:
                from bot_engine.ai.auto_trainer import stop_auto_trainer
                stop_auto_trainer()
            except:
                pass
            
            cleanup_bot_service()
            print("✅ Сервис остановлен\n")
        except:
            pass
        os._exit(0)
