#!/usr/bin/env python3
"""
Главный файл bots.py - импортирует все модули
"""

# Базовые импорты
import errno
import os
import sys
import warnings

# Подавление FutureWarning LeafSpec (PyTorch 2.x / Python 3.14) — до любых импортов torch
warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*TreeSpec.*is_leaf.*")
# Корень проекта в path до импорта utils — иначе sklearn_parallel_config не найдётся при запуске из другой директории
_root = os.path.dirname(os.path.abspath(__file__))
if _root and _root not in sys.path:
    sys.path.insert(0, _root)
# Подавление UserWarning sklearn delayed/Parallel (дочерние процессы наследуют PYTHONWARNINGS)
_pw = os.environ.get("PYTHONWARNINGS", "").strip()
_add = "ignore::UserWarning:sklearn.utils.parallel"
os.environ["PYTHONWARNINGS"] = f"{_pw},{_add}" if _pw else _add
# Вариант 1: joblib → sklearn.utils.parallel до любых импортов sklearn
import utils.sklearn_parallel_config  # noqa: F401

# Ограничение ОЗУ процесса (AI_MEMORY_LIMIT_MB / AI_MEMORY_PCT из bot_config) — как для ai.py
try:
    from utils.process_limits import (
        compute_memory_limit_mb,
        apply_memory_limit_setrlimit,
        apply_windows_job_limits,
    )
    _limit_mb, _kind, _total_mb, _pct = compute_memory_limit_mb()
    if _limit_mb is not None:
        if _kind == 'pct' and _total_mb is not None and _pct is not None:
            sys.stderr.write(f"[Bots] Лимит ОЗУ: {_limit_mb} MB ({_pct:.0f}% от {_total_mb} MB)\n")
        else:
            sys.stderr.write(f"[Bots] Лимит ОЗУ: {_limit_mb} MB (AI_MEMORY_LIMIT_MB)\n")
        if sys.platform == 'win32':
            apply_windows_job_limits(memory_mb=_limit_mb, cpu_pct=0, process_name='Bots')
        else:
            apply_memory_limit_setrlimit(_limit_mb)
except Exception:
    pass

# 🔍 Проверка и создание configs/bot_config.py из configs/bot_config.example.py (если отсутствует)
# Также настраиваем git skip-worktree для игнорирования локальных изменений
_bot_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'bot_config.py')
_example_bot_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'bot_config.example.py')

if not os.path.exists(_bot_config_path):
    if os.path.exists(_example_bot_config_path):
        try:
            import shutil
            os.makedirs(os.path.dirname(_bot_config_path), exist_ok=True)
            shutil.copy2(_example_bot_config_path, _bot_config_path)
            import sys
            sys.stderr.write(f"[INFO] ✅ Создан configs/bot_config.py из configs/bot_config.example.py\n")
        except Exception as e:
            import sys
            sys.stderr.write(f"[WARNING] Не удалось создать configs/bot_config.py: {e}\n")
            sys.stderr.write(f"[WARNING] Продолжаем с configs/bot_config.example.py...\n")
    else:
        import sys
        sys.stderr.write(f"[WARNING] Файл configs/bot_config.example.py не найден, configs/bot_config.py не будет создан автоматически\n")

# RSI-фикс: автопатч configs/bot_config.py (устаревший fallback rsi6h → делегирование в config_loader)
try:
    from bot_engine import ensure_rsi_fix_applied
    ensure_rsi_fix_applied()
except Exception:
    pass

# Патчи обратной совместимости (например FullAI в DefaultAutoBotConfig) — при любом запуске
try:
    from pathlib import Path
    from patches.runner import run_patches
    _applied = run_patches(Path(_root))
    if _applied:
        sys.stderr.write(f"[Bots] Применены патчи: {', '.join(_applied)}\n")
except Exception:
    pass

# Настройка git skip-worktree для игнорирования локальных изменений в bot_config.py
# Это позволяет файлу оставаться в git, но локальные изменения не будут коммититься
# И защищает от перезаписи при git pull - локальная версия всегда имеет приоритет
if os.path.exists(_bot_config_path):
    try:
        import subprocess
        git_dir = os.path.dirname(os.path.abspath(__file__))

        # Проверяем, установлен ли уже skip-worktree
        result = subprocess.run(
            ['git', 'ls-files', '-v', _bot_config_path],
            capture_output=True,
            text=True,
            cwd=git_dir,
            timeout=5
        )
        # Если файл отслеживается и не имеет флага skip-worktree (не начинается с 'S')
        if result.returncode == 0 and result.stdout.strip() and not result.stdout.strip().startswith('S'):
            subprocess.run(
                ['git', 'update-index', '--skip-worktree', _bot_config_path],
                cwd=git_dir,
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Логгер еще не настроен, используем stderr
            import sys
            sys.stderr.write(f"[INFO] ✅ Защита configs/bot_config.py от перезаписи при git pull активирована\n")

        # Дополнительная защита: если configs/bot_config.py был изменён в удалённом репозитории,
        # но у нас есть локальная версия — восстанавливаем её из бэкапа (если есть)
        backup_path = _bot_config_path + '.local_backup'
        if os.path.exists(backup_path):
            try:
                import shutil
                # Восстанавливаем локальную версию из бэкапа
                shutil.copy2(backup_path, _bot_config_path)
                # Удаляем бэкап после восстановления
                try:
                    os.remove(backup_path)
                except Exception:
                    pass
                # Логгер еще не настроен, используем stderr
                import sys
                sys.stderr.write(f"[INFO] ✅ Восстановлена локальная версия configs/bot_config.py после git pull\n")
            except Exception:
                pass

        # Автоматическая установка git hooks для защиты bot_config.py
        try:
            hooks_install_script = os.path.join(git_dir, 'scripts', 'install_git_hooks.py')
            if os.path.exists(hooks_install_script):
                # Проверяем, установлены ли уже хуки
                hooks_target_dir = os.path.join(git_dir, '.git', 'hooks')
                post_merge_hook = os.path.join(hooks_target_dir, 'post-merge')
                if os.path.exists(hooks_target_dir) and not os.path.exists(post_merge_hook):
                    # Устанавливаем хуки автоматически
                    install_result = subprocess.run(
                        [sys.executable, hooks_install_script],
                        cwd=git_dir,
                        timeout=10,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    if install_result.returncode == 0:
                        import sys
                        sys.stderr.write(f"[INFO] ✅ Git hooks для защиты bot_config.py установлены автоматически\n")
        except Exception:
            # Игнорируем ошибки установки хуков
            pass

    except Exception:
        # Игнорируем ошибки git (если это не git репозиторий или git не установлен)
        pass

# 🔍 ТРЕЙСИНГ из конфига (после импорта sys, но до остальных импортов)
try:
    # Читаем настройку трейсинга из конфига
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bot_engine.config_loader import SystemConfig
    ENABLE_TRACE = SystemConfig.ENABLE_CODE_TRACING

    if ENABLE_TRACE:
        from trace_debug import enable_trace
        enable_trace()
        # Логгер еще не настроен, используем stderr
        import sys
        sys.stderr.write("=" * 80 + "\n")
        sys.stderr.write("TRACE: ENABLED - all code execution will be logged with timing\n")
        sys.stderr.write("WARNING: This will slow down the system significantly!\n")
        sys.stderr.write("=" * 80 + "\n")
    else:
        # Логгер еще не настроен, используем stderr
        import sys
        sys.stderr.write("[INFO] Code tracing DISABLED (set SystemConfig.ENABLE_CODE_TRACING = True to enable)\n")
except Exception as e:
    # Логгер еще не настроен, используем stderr
    import sys
    sys.stderr.write(f"[WARNING] Could not initialize tracing: {e}\n")
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

# Проверка наличия конфигурации (все конфиги в configs/)
if not os.path.exists('configs/app_config.py') and not os.path.exists('app/config.py'):
    import sys
    sys.stderr.write("\n" + "="*80 + "\n")
    sys.stderr.write("❌ ОШИБКА: Конфигурация не найдена!\n")
    sys.stderr.write("="*80 + "\n\n")
    sys.stderr.write("📝 Все конфиги в папке configs/. Выполните:\n\n")
    if os.name == 'nt':
        sys.stderr.write("      copy configs\\app_config.example.py configs\\app_config.py\n")
        sys.stderr.write("      copy configs\\keys.example.py configs\\keys.py\n")
    else:
        sys.stderr.write("      cp configs/app_config.example.py configs/app_config.py\n")
        sys.stderr.write("      cp configs/keys.example.py configs/keys.py\n")
    sys.stderr.write("\n   Отредактируйте configs/keys.py (API ключи) и configs/app_config.py\n")
    sys.stderr.write("   📖 Подробнее: docs/INSTALL.md\n\n" + "="*80 + "\n\n")
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
        if not os.path.exists('configs/keys.py') and not os.path.exists('app/keys.py'):
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
        # Логгер еще не настроен, используем stderr
        import sys
        sys.stderr.write("Не удалось освободить порт 5001, завершаем работу\n")
        sys.exit(1)

    # Проверяем API ключи
    if not check_api_keys():
        # Логгер еще не настроен, используем stderr для предупреждений
        import sys
        sys.stderr.write("\n" + "="*80 + "\n")
        sys.stderr.write("⚠️  ВНИМАНИЕ: API ключи не настроены!\n")
        sys.stderr.write("="*80 + "\n")
        sys.stderr.write("\n")
        sys.stderr.write("📌 Текущий статус:\n")
        try:
            from app.config import ACTIVE_EXCHANGE
            sys.stderr.write(f"   Биржа: {ACTIVE_EXCHANGE}\n")
        except:
            sys.stderr.write("   Биржа: НЕ ОПРЕДЕЛЕНА\n")

        if not os.path.exists('configs/keys.py'):
            sys.stderr.write("   Файл с ключами: configs/keys.py НЕ НАЙДЕН\n")
        else:
            sys.stderr.write("   API ключи: НЕ НАСТРОЕНЫ или СОДЕРЖАТ ПРИМЕРЫ\n")
        sys.stderr.write("\n")
        sys.stderr.write("💡 Что нужно сделать:\n")
        sys.stderr.write("   1. Отредактируйте configs/keys.py — добавьте реальные ключи биржи\n")
        sys.stderr.write("   2. Или добавьте EXCHANGES в configs/app_config.py\n")
        sys.stderr.write("   3. Перезапустите bots.py\n")
        sys.stderr.write("\n")
        sys.stderr.write("⚠️  Сервис запущен, но торговля НЕВОЗМОЖНА без ключей!\n")
        sys.stderr.write("   Будут ошибки: 'Http status code is not 200. (ErrCode: 401)'\n")
        sys.stderr.write("\n")
        sys.stderr.write("="*80 + "\n")
        sys.stderr.write("\n")

# Импорт цветного логирования
from utils.color_logger import setup_color_logging

# Импортируем все модули
from bots_modules.imports_and_globals import *
from bots_modules.calculations import *
from bots_modules.maturity import *
# ❌ ОТКЛЮЧЕНО: optimal_ema перемещен в backup (используются заглушки из imports_and_globals)
# from bots_modules.optimal_ema import *
from bots_modules.filters import *
from bots_modules.bot_class import *
from bots_modules.sync_and_cache import *
from bots_modules.workers import *
from bots_modules.init_functions import *

# Импорт системы истории ботов (ПЕРЕД импортом API endpoints!)
# Настройка логирования (раньше, чтобы использовать logger)
# Применяем фильтр уровней логирования из конфига
try:
    from bot_engine.config_loader import SystemConfig
    console_levels = getattr(SystemConfig, 'CONSOLE_LOG_LEVELS', [])
    setup_color_logging(console_log_levels=console_levels if console_levels else None, log_file='logs/bots.log')
except Exception as e:
    # Если не удалось загрузить конфиг, используем стандартную настройку
    setup_color_logging(log_file='logs/bots.log')

# Отключаем DEBUG логи от внешних библиотек ДО их импорта
# matplotlib - логирует неформатированные сообщения при импорте
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)
for handler in matplotlib_logger.handlers[:]:
    matplotlib_logger.removeHandler(handler)

matplotlib_font_manager_logger = logging.getLogger('matplotlib.font_manager')
matplotlib_font_manager_logger.setLevel(logging.WARNING)
for handler in matplotlib_font_manager_logger.handlers[:]:
    matplotlib_font_manager_logger.removeHandler(handler)

matplotlib_backends_logger = logging.getLogger('matplotlib.backends')
matplotlib_backends_logger.setLevel(logging.WARNING)
for handler in matplotlib_backends_logger.handlers[:]:
    matplotlib_backends_logger.removeHandler(handler)

# TensorFlow - логирует "Falling back to TensorFlow client..."
tensorflow_logger = logging.getLogger('tensorflow')
tensorflow_logger.setLevel(logging.WARNING)
tensorflow_python_logger = logging.getLogger('tensorflow.python')
tensorflow_python_logger.setLevel(logging.WARNING)
tensorflow_core_logger = logging.getLogger('tensorflow.core')
tensorflow_core_logger.setLevel(logging.WARNING)
for handler in tensorflow_logger.handlers[:]:
    tensorflow_logger.removeHandler(handler)
for handler in tensorflow_python_logger.handlers[:]:
    tensorflow_python_logger.removeHandler(handler)
for handler in tensorflow_core_logger.handlers[:]:
    tensorflow_core_logger.removeHandler(handler)

logger = logging.getLogger('BotsService')

try:
    from bot_engine.bot_history import (
        bot_history_manager, log_bot_start, log_bot_stop, log_bot_signal,
        log_position_opened, log_position_closed
    )
    BOT_HISTORY_AVAILABLE = True
    logger.info("✅ Модуль bot_history загружен успешно")

    # Устанавливаем bot_history_manager в глобальный модуль
    import bots_modules.imports_and_globals as globals_module
    globals_module.bot_history_manager = bot_history_manager
    globals_module.BOT_HISTORY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Модуль bot_history недоступен: {e}")
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
    logger.warning("⚠️ Установлена заглушка для bot_history")
except Exception as e:
    logger.error(f"❌ Неожиданная ошибка при импорте bot_history: {e}")
    import traceback
    traceback.print_exc()

# Теперь импортируем API endpoints (после установки bot_history_manager)
from bots_modules.api_endpoints import *

# Файловый логгер уже настроен в setup_color_logging() выше, не нужно дублировать

# Настройка кодировки для stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger('BotsService')

# Глобальные переменные для shutdown
graceful_shutdown = False
_flask_server = None  # Глобальная ссылка на Flask сервер для корректной остановки

# Signal handlers
def signal_handler(signum, frame):
    """Обработчик сигналов для graceful shutdown"""
    global graceful_shutdown, _flask_server

    if graceful_shutdown:
        # Уже идет завершение, принудительно выходим
        logger.warning("⚠️ Принудительное завершение процесса...")
        os._exit(0)

    logger.warning(f"\n🛑 Получен сигнал {signum}, начинаем graceful shutdown...")
    graceful_shutdown = True
    shutdown_flag.set()

    # Останавливаем Flask сервер
    if _flask_server:
        try:
            logger.info("🛑 Остановка Flask сервера...")
            _flask_server.shutdown()
            logger.info("✅ Flask сервер остановлен")
        except Exception as e:
                        pass

    # Очистка ресурсов
    try:
        cleanup_bot_service()
    except Exception as e:
                pass

    logger.info("✅ Сервис остановлен")

    # Принудительное завершение через небольшой таймаут
    def force_exit():
        time.sleep(1.5)  # Даём 1.5 секунды на graceful shutdown
        logger.warning("⏱️ Таймаут graceful shutdown, принудительное завершение...")
        os._exit(0)

    exit_thread = threading.Thread(target=force_exit, daemon=True)
    exit_thread.start()

_cleanup_done = False

def open_firewall_port_5001():
    """Открывает порт 5001 в брандмауэре при запуске (Windows/macOS/Linux)"""
    try:
        import subprocess
        import platform

        logger.info("🔥 Проверка открытия порта 5001 в брандмауэре...")

        system = platform.system()
        port = 5001

        if system == 'Windows':
            # Проверяем правило для порта 5001
            try:
                check_result = subprocess.run(
                    ['netsh', 'advfirewall', 'firewall', 'show', 'rule', 'name=InfoBot Bot Service'],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )

                # Если правило не найдено (код возврата != 0 или имя не в выводе)
                rule_exists = (
                    check_result.returncode == 0 and 
                    check_result.stdout and 
                    'InfoBot Bot Service' in check_result.stdout
                )

                if not rule_exists:
                    logger.info("🔥 Открываем порт 5001...")
                    add_result = subprocess.run([
                        'netsh', 'advfirewall', 'firewall', 'add', 'rule',
                        'name=InfoBot Bot Service',
                        'dir=in',
                        'action=allow',
                        'protocol=TCP',
                        f'localport={port}'
                    ], capture_output=True, text=True, encoding='utf-8', errors='replace')

                    if add_result.returncode == 0:
                        logger.info("✅ Порт 5001 открыт")
                    else:
                        # Возможно правило уже существует или нужны права администратора
                        if 'уже существует' in add_result.stderr or 'already exists' in add_result.stderr.lower():
                            logger.info("✅ Порт 5001 уже открыт")
                        else:
                            logger.warning(f"⚠️ Не удалось открыть порт 5001: {add_result.stderr or add_result.stdout}")
                else:
                    logger.info("✅ Порт 5001 уже открыт")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось проверить/открыть порт 5001: {e}")

        elif system == 'Darwin':  # macOS
            logger.info("💡 На macOS откройте порт 5001 вручную")

        elif system == 'Linux':
            try:
                # Проверяем наличие ufw
                subprocess.run(['which', 'ufw'], check=True)
                result = subprocess.run(['ufw', 'status'], capture_output=True, text=True)
                if f'{port}/tcp' not in result.stdout:
                    subprocess.run(['sudo', 'ufw', 'allow', str(port), '/tcp'], check=True)
                    logger.info(f"✅ Порт {port} открыт")
                else:
                    logger.info(f"✅ Порт {port} уже открыт")
            except:
                logger.warning(f"⚠️ Настройте порт {port} вручную")

        else:
            logger.warning(f"⚠️ Неизвестная система: {system}")
            logger.info("💡 Настройте порт вручную см. docs/INSTALL.md")

    except Exception as e:
        logger.warning(f"⚠️ Не удалось открыть порт 5001 автоматически: {e}")
        logger.info("💡 Откройте порт вручную см. docs/INSTALL.md")

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

        from utils.memory_utils import force_collect_full
        force_collect_full()
        logger.info("Система остановлена")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Ошибка при очистке: {e}")

def run_bots_service():
    """Запуск Flask сервера для API ботов"""
    global graceful_shutdown, _flask_server

    try:
        logger.info("=" * 80)
        logger.info("ЗАПУСК BOTS SERVICE API (Порт 5001)")
        logger.info("=" * 80)

        logger.info("\n" + "=" * 80)
        logger.info("✅ BOTS SERVICE ЗАПУЩЕН И РАБОТАЕТ!")
        logger.info("=" * 80)
        logger.info("🌐 API доступен на: http://localhost:5001")
        logger.info("📊 Статус: http://localhost:5001/api/status")
        logger.info("🤖 Боты: http://localhost:5001/api/bots")
        logger.info("=" * 80)
        logger.info("💡 Нажмите Ctrl+C для остановки")
        logger.info("=" * 80 + "\n")

        # Используем Werkzeug сервер для возможности корректной остановки
        from werkzeug.serving import make_server

        _flask_server = None
        server_thread = None

        try:
            _flask_server = make_server('0.0.0.0', 5001, bots_app, threaded=True)

            # Запускаем сервер в отдельном потоке
            def run_server():
                try:
                    _flask_server.serve_forever(poll_interval=0.5)
                except (KeyboardInterrupt, SystemExit):
                    pass
                except Exception as e:
                    if not graceful_shutdown:
                        logger.error(f"Ошибка сервера: {e}")

            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()

            # Ждем завершения или сигнала (периодическая сборка мусора раз в ~60 с)
            try:
                _gc_ticks = 0
                while server_thread.is_alive() and not graceful_shutdown:
                    time.sleep(0.1)
                    _gc_ticks += 1
                    if _gc_ticks >= 600:
                        from utils.memory_utils import force_collect_full
                        force_collect_full()
                        _gc_ticks = 0
            except KeyboardInterrupt:
                logger.info("\n🛑 Получен KeyboardInterrupt, останавливаем сервер...")
                graceful_shutdown = True
                shutdown_flag.set()

        finally:
            # Останавливаем сервер
            if _flask_server:
                try:
                    if not graceful_shutdown:
                        logger.info("🛑 Остановка Flask сервера...")
                    _flask_server.shutdown()
                    _flask_server = None
                    if not graceful_shutdown:
                        logger.info("✅ Flask сервер остановлен")
                except Exception as e:
                                        pass

                # Ждем завершения потока сервера
                if server_thread and server_thread.is_alive():
                    server_thread.join(timeout=2.0)

    except KeyboardInterrupt:
        logger.info("\n🛑 KeyboardInterrupt в run_bots_service")
        graceful_shutdown = True
        shutdown_flag.set()
    except SystemExit as e:
        if e.code == 42:
            # Специальный код для горячей перезагрузки
            logger.info("🔄 Горячая перезагрузка: перезапуск сервера...")
            # Запускаем новый процесс
            import subprocess
            subprocess.Popen([sys.executable] + sys.argv)
            sys.exit(0)
        else:
            raise
    except OSError as e:
        if getattr(e, "errno", None) in (errno.EADDRINUSE, 10048):
            logger.error(
                "Порт 5001 занят. Закройте другой процесс на этом порту или измените BOTS_SERVICE_PORT в bot_config."
            )
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

        logger.info("📋 Загрузка конфигурации Auto Bot...")
        load_auto_bot_config()
        logger.info("✅ Конфигурация Auto Bot загружена")

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

        # ❌ ОТКЛЮЧЕНО: Optimal EMA Worker - больше не используется
        # EMA фильтр убран, расчет оптимальных EMA не нужен
        # from bot_engine.optimal_ema_worker import start_optimal_ema_worker
        # optimal_ema_worker = start_optimal_ema_worker(update_interval=21600) # 6 часов
        logger.info("ℹ️ Optimal EMA Worker отключен (не используется)")

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

        # ✅ FullAI Monitor - ежесекундная проверка позиций при full_ai_control (БД, свечи, индикаторы)
        try:
            from bot_engine.fullai_monitor import start_fullai_monitor
            start_fullai_monitor()
            logger.info("🧠 FullAI Monitor запущен (ежесекундно при full_ai_control)")
        except Exception as fa_err:
            logger.debug("FullAI Monitor не запущен: %s", fa_err)

        # Планировщик бэкапов только bots_data.db (своя БД bots.py)
        try:
            from configs.app_config import DATABASE_BACKUP
            from bot_engine.backup_service import run_backup_scheduler_loop
            _backup_cfg = DATABASE_BACKUP if isinstance(DATABASE_BACKUP, dict) else {}
            _backup_cfg = {**_backup_cfg, 'APP_ENABLED': False, 'AI_ENABLED': False, 'BOTS_ENABLED': True}
            if _backup_cfg.get('ENABLED', True) and _backup_cfg.get('BOTS_ENABLED', False):
                _backup_thread = threading.Thread(
                    target=run_backup_scheduler_loop,
                    args=(_backup_cfg,),
                    name='DatabaseBackupScheduler',
                    daemon=True
                )
                _backup_thread.start()
                logger.info("💾 Планировщик бэкапов Bots БД (bots_data.db) запущен")
        except Exception as backup_err:
            logger.debug("Планировщик бэкапов не запущен: %s", backup_err)

        # Инициализируем AI Manager (проверка лицензии и загрузка модулей)
        ai_manager = None
        try:
            from bot_engine.config_loader import AIConfig

            if AIConfig.AI_ENABLED:
                logger.info("🤖 Инициализация AI модулей...")
                from bot_engine.ai import get_ai_manager
                ai_manager = get_ai_manager()

                # ✅ Обучение перенесено в ai.py - здесь только проверка доступности модулей
                if ai_manager.is_available():
                    logger.info("")
                    logger.info("=" * 80)
                    logger.info("🟢 AI МОДУЛИ АКТИВНЫ - ЛИЦЕНЗИЯ ВАЛИДНА 🟢")
                    logger.info("=" * 80)
                    logger.info("🤖 AI модули активны (обучение выполняется в ai.py)")
                    logger.info("=" * 80)
                    logger.info("")
                else:
                    logger.warning("")
                    logger.warning("=" * 80)
                    logger.warning("🔴 AI МОДУЛИ НЕ ЗАГРУЖЕНЫ - ЛИЦЕНЗИЯ НЕ ВАЛИДНА 🔴")
                    logger.warning("=" * 80)
                    logger.warning("⚠️ AI модули не загружены (проверьте лицензию)")
                    logger.warning("💡 Получите HWID: python scripts/activate_premium.py")
                    logger.warning("=" * 80)
                    logger.warning("")
            else:
                logger.info("ℹ️ AI модули отключены в конфигурации")
        except ImportError as ai_import_error:
                        pass
        except Exception as ai_error:
            logger.warning(f"⚠️ Ошибка инициализации AI: {ai_error}")

        # Открываем порт 5001 в брандмауэре
        open_firewall_port_5001()

        run_bots_service()

    except KeyboardInterrupt:
        logger.info("\n🛑 Получен KeyboardInterrupt, останавливаем сервис...")
        graceful_shutdown = True
        shutdown_flag.set()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            # ✅ Auto Trainer останавливается в ai.py, здесь не требуется
            cleanup_bot_service()
            logger.info("✅ Сервис остановлен\n")
        except Exception as cleanup_error:
                        pass

        # Принудительное завершение процесса
        logger.info("🚪 Завершение процесса...")
        os._exit(0)
