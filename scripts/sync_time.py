#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для синхронизации времени в Windows 11
Использует Windows Time Service (w32tm) для синхронизации с серверами времени

Режимы работы:
  - Однократная синхронизация: python sync_time.py
  - Постоянная работа: python sync_time.py --daemon [--interval MINUTES]
"""

import subprocess
import sys
import os
import time
import argparse
import datetime
import signal
from typing import Optional, Tuple
from pathlib import Path


def run_command(command: list, check: bool = True) -> Tuple[bool, str]:
    """
    Выполняет команду и возвращает результат
    
    Args:
        command: Список аргументов команды
        check: Если True, вызывает исключение при ошибке
        
    Returns:
        Tuple[bool, str]: (успех, вывод команды)
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=check,
            encoding='utf-8',
            errors='ignore'
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr if e.stderr else str(e)
    except Exception as e:
        return False, str(e)


def check_admin_rights() -> bool:
    """Проверяет, запущен ли скрипт с правами администратора"""
    try:
        # Пытаемся выполнить команду, требующую прав администратора
        result = subprocess.run(
            ['net', 'session'],
            capture_output=True,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def get_time_status() -> Optional[str]:
    """Получает текущий статус службы времени"""
    success, output = run_command(['w32tm', '/query', '/status'], check=False)
    if success:
        return output
    return None


def configure_time_service(server: str = "time.windows.com", silent: bool = False) -> Tuple[bool, str]:
    """
    Настраивает службу времени для синхронизации с указанным сервером
    
    Args:
        server: Адрес сервера времени (по умолчанию time.windows.com)
        silent: Если True, не выводит сообщения в консоль
        
    Returns:
        Tuple[bool, str]: (успех, сообщение)
    """
    if not check_admin_rights():
        return False, "Требуются права администратора для настройки службы времени"
    
    # Настройка службы времени
    command = [
        'w32tm', '/config',
        f'/manualpeerlist:"{server}"',
        '/syncfromflags:manual',
        '/reliable:yes',
        '/update'
    ]
    
    success, output = run_command(command, check=False)
    if not success:
        return False, f"Ошибка настройки службы времени: {output}"
    
    # Перезапуск службы времени
    if not silent:
        print("Перезапуск службы времени...")
    run_command(['net', 'stop', 'w32time'], check=False)
    run_command(['net', 'start', 'w32time'], check=False)
    
    return True, "Служба времени успешно настроена"


def sync_time(silent: bool = False) -> Tuple[bool, str]:
    """
    Выполняет принудительную синхронизацию времени
    
    Args:
        silent: Если True, не выводит сообщения в консоль
        
    Returns:
        Tuple[bool, str]: (успех, сообщение)
    """
    if not check_admin_rights():
        return False, "Требуются права администратора для синхронизации времени"
    
    if not silent:
        print("Выполняется синхронизация времени...")
    
    success, output = run_command(['w32tm', '/resync'], check=False)
    
    if success:
        return True, "Время успешно синхронизировано"
    else:
        # Если синхронизация не удалась, попробуем настроить службу
        if not silent:
            print("Попытка настроить службу времени...")
        config_success, config_msg = configure_time_service()
        if config_success:
            # Повторная попытка синхронизации
            success, output = run_command(['w32tm', '/resync'], check=False)
            if success:
                return True, "Время успешно синхронизировано после настройки службы"
        
        return False, f"Ошибка синхронизации времени: {output}"


def show_current_time() -> None:
    """Показывает текущее системное время"""
    success, output = run_command(['w32tm', '/query', '/status'], check=False)
    if success:
        print("\nТекущий статус службы времени:")
        print(output)
    else:
        # Альтернативный способ показать время
        import datetime
        print(f"\nТекущее системное время: {datetime.datetime.now()}")


def log_message(message: str, log_file: Optional[Path] = None, console: bool = True) -> None:
    """
    Логирует сообщение в файл и/или консоль
    
    Args:
        message: Сообщение для логирования
        log_file: Путь к файлу лога (опционально)
        console: Выводить ли в консоль
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    
    if console:
        print(log_line)
    
    if log_file:
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_line + '\n')
        except Exception as e:
            if console:
                print(f"Ошибка записи в лог: {e}")


def run_daemon_mode(interval_minutes: int = 60, log_file: Optional[Path] = None) -> None:
    """
    Запускает скрипт в режиме постоянной работы (daemon)
    
    Args:
        interval_minutes: Интервал синхронизации в минутах
        log_file: Путь к файлу лога (опционально)
    """
    # Обработчик сигнала для корректного завершения
    def signal_handler(sig, frame):
        log_message("Получен сигнал завершения. Остановка скрипта...", log_file)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    log_message("=" * 60, log_file)
    log_message("Запуск режима постоянной синхронизации времени", log_file)
    log_message(f"Интервал синхронизации: {interval_minutes} минут", log_file)
    log_message("Для остановки нажмите Ctrl+C", log_file)
    log_message("=" * 60, log_file)
    
    # Первоначальная настройка службы времени
    log_message("Первоначальная настройка службы времени...", log_file)
    configure_time_service(silent=True)
    
    sync_count = 0
    error_count = 0
    
    try:
        while True:
            sync_count += 1
            log_message(f"\n--- Синхронизация #{sync_count} ---", log_file)
            
            success, message = sync_time(silent=True)
            
            if success:
                log_message(f"✓ {message}", log_file)
                error_count = 0  # Сбрасываем счетчик ошибок при успехе
            else:
                error_count += 1
                log_message(f"✗ {message}", log_file)
                
                # Если много ошибок подряд, попробуем переконфигурировать службу
                if error_count >= 3:
                    log_message("Много ошибок подряд. Попытка переконфигурации службы...", log_file)
                    configure_time_service(silent=True)
                    error_count = 0
            
            # Показываем время до следующей синхронизации
            next_sync = datetime.datetime.now() + datetime.timedelta(minutes=interval_minutes)
            log_message(f"Следующая синхронизация: {next_sync.strftime('%Y-%m-%d %H:%M:%S')}", log_file)
            log_message(f"Ожидание {interval_minutes} минут...", log_file)
            
            # Ожидание с периодической проверкой (чтобы можно было прервать)
            wait_seconds = interval_minutes * 60
            check_interval = 60  # Проверяем каждую минуту
            
            for _ in range(0, wait_seconds, check_interval):
                time.sleep(min(check_interval, wait_seconds - _))
                
    except KeyboardInterrupt:
        log_message("\nПолучен сигнал прерывания. Остановка скрипта...", log_file)
    except Exception as e:
        log_message(f"\nКритическая ошибка: {e}", log_file)
        raise
    finally:
        log_message("=" * 60, log_file)
        log_message(f"Скрипт остановлен. Всего синхронизаций: {sync_count}", log_file)
        log_message("=" * 60, log_file)


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description='Скрипт синхронизации времени Windows 11',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python sync_time.py                    # Однократная синхронизация
  python sync_time.py --daemon          # Постоянная работа (интервал 60 мин)
  python sync_time.py --daemon --interval 30  # Постоянная работа (интервал 30 мин)
  python sync_time.py --daemon --interval 15 --log sync_time.log  # С логом
        """
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Запустить в режиме постоянной работы'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Интервал синхронизации в минутах (по умолчанию: 60)'
    )
    parser.add_argument(
        '--log',
        type=str,
        help='Путь к файлу лога (опционально)'
    )
    
    args = parser.parse_args()
    
    # Проверка операционной системы
    if sys.platform != 'win32':
        print("Ошибка: Этот скрипт предназначен только для Windows")
        sys.exit(1)
    
    # Проверка прав администратора
    if not check_admin_rights():
        print("\n⚠ ВНИМАНИЕ: Скрипт запущен без прав администратора!")
        print("Для синхронизации времени требуются права администратора.")
        print("\nЗапустите скрипт от имени администратора:")
        print("  1. Правый клик на файл -> 'Запуск от имени администратора'")
        print("  2. Или через PowerShell: Start-Process python -ArgumentList 'sync_time.py' -Verb RunAs")
        sys.exit(1)
    
    # Определяем файл лога
    log_file = None
    if args.log:
        log_file = Path(args.log)
    elif args.daemon:
        # По умолчанию создаем лог в той же директории
        script_dir = Path(__file__).parent
        log_file = script_dir / 'sync_time.log'
    
    # Режим постоянной работы
    if args.daemon:
        if args.interval < 1:
            print("Ошибка: Интервал должен быть не менее 1 минуты")
            sys.exit(1)
        run_daemon_mode(interval_minutes=args.interval, log_file=log_file)
        return
    
    # Режим однократной синхронизации
    print("=" * 60)
    print("Скрипт синхронизации времени Windows 11")
    print("=" * 60)
    
    # Показываем текущее время
    show_current_time()
    
    # Выполняем синхронизацию
    print("\n" + "=" * 60)
    success, message = sync_time()
    
    if success:
        print(f"\n✓ {message}")
    else:
        print(f"\n✗ {message}")
        sys.exit(1)
    
    # Показываем время после синхронизации
    print("\n" + "=" * 60)
    show_current_time()
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)


if __name__ == "__main__":
    main()
