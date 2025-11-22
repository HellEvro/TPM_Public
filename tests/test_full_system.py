#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полное тестирование системы: запуск всех сервисов, проверка лицензий и конфигов
"""

import os
import sys
import time
import subprocess
import requests
import signal
import threading
from pathlib import Path
from queue import Queue

# Настройка кодировки
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("ПОЛНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ")
print("=" * 80)
print()

# Процессы и логи
processes = {}
process_logs = {}
log_queues = {}
errors = []
warnings = []

def log_reader(proc, name, queue):
    """Читает логи процесса"""
    try:
        for line in iter(proc.stdout.readline, ''):
            if line:
                queue.put((name, line.strip()))
            if proc.poll() is not None:
                break
    except:
        pass

def cleanup():
    """Остановка всех процессов"""
    print("\n" + "=" * 80)
    print("ОСТАНОВКА ВСЕХ ПРОЦЕССОВ")
    print("=" * 80)
    for name, proc in processes.items():
        if proc and proc.poll() is None:
            print(f"Останавливаем {name}...")
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                try:
                    proc.kill()
                except:
                    pass
            print(f"  [OK] {name} остановлен")

import atexit
atexit.register(cleanup)

# 1. Тест лицензии
print("1. ТЕСТ ЛИЦЕНЗИИ")
print("-" * 80)
try:
    from bot_engine.ai.license_checker import get_license_checker
    checker = get_license_checker(project_root=PROJECT_ROOT)
    valid, info = checker.check_license()
    if valid:
        print(f"  [OK] Лицензия валидна: {info.get('type', 'N/A')}")
        print(f"  [INFO] Действительна до: {info.get('expires_at', 'N/A')}")
    else:
        print("  [ERROR] Лицензия не валидна!")
        errors.append("Лицензия не валидна")
        sys.exit(1)
except Exception as e:
    print(f"  [ERROR] Ошибка проверки лицензии: {e}")
    errors.append(f"Ошибка проверки лицензии: {e}")
    sys.exit(1)

print()

# 2. Запуск bots.py
print("2. ЗАПУСК BOTS.PY")
print("-" * 80)
try:
    bots_proc = subprocess.Popen(
        [sys.executable, "bots.py"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace'
    )
    processes['bots.py'] = bots_proc
    log_queue = Queue()
    log_queues['bots.py'] = log_queue
    thread = threading.Thread(target=log_reader, args=(bots_proc, 'bots.py', log_queue), daemon=True)
    thread.start()
    print("  [OK] bots.py запущен (PID: {})".format(bots_proc.pid))
    
    # Ждем запуска и проверяем логи на ошибки
    print("  [INFO] Ожидание запуска сервиса (до 30 секунд)...")
    for i in range(30):
        time.sleep(1)
        # Проверяем логи на ошибки
        try:
            while not log_queue.empty():
                name, line = log_queue.get_nowait()
                if 'ERROR' in line.upper() or 'CRITICAL' in line.upper():
                    if 'license' in line.lower() or 'лиценз' in line.lower():
                        errors.append(f"bots.py: {line}")
                        print(f"  [ERROR] {line}")
                    elif 'config' in line.lower() or 'конфиг' in line.lower():
                        errors.append(f"bots.py: {line}")
                        print(f"  [ERROR] {line}")
        except:
            pass
        
        # Проверяем health
        try:
            response = requests.get("http://127.0.0.1:5001/api/bots/health", timeout=2)
            if response.status_code == 200:
                print("  [OK] bots.py отвечает на health check")
                break
        except:
            pass
        
        # Проверяем что процесс не упал
        if bots_proc.poll() is not None:
            print(f"  [ERROR] bots.py завершился с кодом {bots_proc.returncode}")
            # Читаем последние логи
            print("  [INFO] Последние логи bots.py:")
            try:
                while not log_queue.empty():
                    name, line = log_queue.get_nowait()
                    print(f"    {line}")
            except:
                pass
            errors.append(f"bots.py завершился с кодом {bots_proc.returncode}")
            break
    else:
        print("  [WARN] bots.py не ответил за 30 секунд")
except Exception as e:
    print(f"  [ERROR] Ошибка запуска bots.py: {e}")
    errors.append(f"Ошибка запуска bots.py: {e}")

print()

# 3. Запуск app.py
print("3. ЗАПУСК APP.PY")
print("-" * 80)
try:
    app_proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace'
    )
    processes['app.py'] = app_proc
    log_queue = Queue()
    log_queues['app.py'] = log_queue
    thread = threading.Thread(target=log_reader, args=(app_proc, 'app.py', log_queue), daemon=True)
    thread.start()
    print("  [OK] app.py запущен (PID: {})".format(app_proc.pid))
    
    # Ждем запуска
    print("  [INFO] Ожидание запуска сервиса (до 30 секунд)...")
    for i in range(30):
        time.sleep(1)
        # Проверяем логи на ошибки
        try:
            while not log_queue.empty():
                name, line = log_queue.get_nowait()
                if 'ERROR' in line.upper() or 'CRITICAL' in line.upper():
                    if 'license' in line.lower() or 'лиценз' in line.lower():
                        errors.append(f"app.py: {line}")
                        print(f"  [ERROR] {line}")
                    elif 'config' in line.lower() or 'конфиг' in line.lower():
                        errors.append(f"app.py: {line}")
                        print(f"  [ERROR] {line}")
        except:
            pass
        
        # Проверяем доступность
        try:
            response = requests.get("http://127.0.0.1:5000/", timeout=2)
            if response.status_code == 200:
                print("  [OK] app.py отвечает")
                break
        except:
            pass
        
        if app_proc.poll() is not None:
            print(f"  [ERROR] app.py завершился с кодом {app_proc.returncode}")
            errors.append(f"app.py завершился с кодом {app_proc.returncode}")
            break
    else:
        print("  [WARN] app.py не ответил за 30 секунд")
except Exception as e:
    print(f"  [ERROR] Ошибка запуска app.py: {e}")
    errors.append(f"Ошибка запуска app.py: {e}")

print()

# 4. Запуск ai.py
print("4. ЗАПУСК AI.PY")
print("-" * 80)
try:
    ai_proc = subprocess.Popen(
        [sys.executable, "ai.py"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace'
    )
    processes['ai.py'] = ai_proc
    log_queue = Queue()
    log_queues['ai.py'] = log_queue
    thread = threading.Thread(target=log_reader, args=(ai_proc, 'ai.py', log_queue), daemon=True)
    thread.start()
    print("  [OK] ai.py запущен (PID: {})".format(ai_proc.pid))
    
    # Ждем запуска и проверяем логи
    print("  [INFO] Ожидание запуска сервиса (до 30 секунд)...")
    for i in range(30):
        time.sleep(1)
        # Проверяем логи на ошибки
        try:
            while not log_queue.empty():
                name, line = log_queue.get_nowait()
                if 'ERROR' in line.upper() or 'CRITICAL' in line.upper():
                    if 'license' in line.lower() or 'лиценз' in line.lower():
                        errors.append(f"ai.py: {line}")
                        print(f"  [ERROR] {line}")
                    elif 'config' in line.lower() or 'конфиг' in line.lower():
                        errors.append(f"ai.py: {line}")
                        print(f"  [ERROR] {line}")
        except:
            pass
        
        # Проверяем что процесс жив
        if ai_proc.poll() is None:
            if i >= 5:  # Даем минимум 5 секунд
                print("  [OK] ai.py работает")
                break
        else:
            print(f"  [ERROR] ai.py завершился с кодом {ai_proc.returncode}")
            errors.append(f"ai.py завершился с кодом {ai_proc.returncode}")
            break
except Exception as e:
    print(f"  [ERROR] Ошибка запуска ai.py: {e}")
    errors.append(f"Ошибка запуска ai.py: {e}")

print()

# 5. Тест конфигов через API
print("5. ТЕСТ КОНФИГОВ ЧЕРЕЗ API")
print("-" * 80)
if 'bots.py' in processes and processes['bots.py'] and processes['bots.py'].poll() is None:
    try:
        # Загружаем текущую конфигурацию
        response = requests.get("http://127.0.0.1:5001/api/bots/auto-bot", timeout=5)
        if response.status_code != 200:
            print(f"  [ERROR] Не удалось получить конфигурацию: {response.status_code}")
            errors.append(f"Не удалось получить конфигурацию: {response.status_code}")
        else:
            config = response.json().get('config', {})
            original_value = config.get('break_even_trigger_percent', 20.0)
            print(f"  [OK] Конфигурация загружена")
            print(f"  [INFO] break_even_trigger_percent = {original_value}")
            
            # Изменяем значение
            new_value = 25.0 if abs(original_value - 25.0) > 0.01 else 30.0
            print(f"  [INFO] Изменяем break_even_trigger_percent → {new_value}")
            
            response = requests.post(
                "http://127.0.0.1:5001/api/bots/auto-bot",
                json={"break_even_trigger_percent": new_value},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code != 200:
                print(f"  [ERROR] Не удалось обновить конфигурацию: {response.status_code}")
                print(f"  [INFO] Ответ: {response.text}")
                errors.append(f"Не удалось обновить конфигурацию: {response.status_code}")
            else:
                print(f"  [OK] Конфигурация обновлена")
                
                # Проверяем что изменилось
                time.sleep(2)
                response = requests.get("http://127.0.0.1:5001/api/bots/auto-bot", timeout=5)
                if response.status_code == 200:
                    config = response.json().get('config', {})
                    updated_value = config.get('break_even_trigger_percent')
                    if abs(updated_value - new_value) < 0.01:
                        print(f"  [OK] Изменение подтверждено: {updated_value}")
                    else:
                        print(f"  [ERROR] Значение не изменилось: {updated_value} (ожидалось {new_value})")
                        errors.append(f"Значение не изменилось: {updated_value} (ожидалось {new_value})")
                    
                    # Восстанавливаем
                    response = requests.post(
                        "http://127.0.0.1:5001/api/bots/auto-bot",
                        json={"break_even_trigger_percent": original_value},
                        headers={"Content-Type": "application/json"},
                        timeout=5
                    )
                    if response.status_code == 200:
                        print(f"  [OK] Исходное значение восстановлено")
                    else:
                        print(f"  [WARN] Не удалось восстановить исходное значение")
    except Exception as e:
        print(f"  [ERROR] Ошибка теста конфигов: {e}")
        import traceback
        traceback.print_exc()
        errors.append(f"Ошибка теста конфигов: {e}")
else:
    print("  [SKIP] bots.py не запущен, пропускаем тест конфигов")

print()

# 6. Мониторинг логов на ошибки
print("6. МОНИТОРИНГ ЛОГОВ (60 секунд)")
print("-" * 80)
print("  [INFO] Отслеживание ошибок в логах...")
start_time = time.time()
monitored_errors = []

for i in range(60):
    time.sleep(1)
    
    # Проверяем логи всех процессов
    for name, queue in log_queues.items():
        try:
            while not queue.empty():
                proc_name, line = queue.get_nowait()
                if 'ERROR' in line.upper() or 'CRITICAL' in line.upper():
                    if 'license' in line.lower() or 'лиценз' in line.lower():
                        if line not in monitored_errors:
                            monitored_errors.append(line)
                            print(f"  [ERROR] {proc_name}: {line}")
                    elif 'config' in line.lower() or 'конфиг' in line.lower():
                        if line not in monitored_errors:
                            monitored_errors.append(line)
                            print(f"  [ERROR] {proc_name}: {line}")
        except:
            pass
    
    # Проверяем что процессы не упали
    for name, proc in processes.items():
        if proc and proc.poll() is not None:
            if f"{name} завершился" not in [e for e in errors if name in e]:
                print(f"  [ERROR] {name} завершился неожиданно (код: {proc.returncode})")
                monitored_errors.append(f"{name} завершился с кодом {proc.returncode}")

print()

# Итоги
print("=" * 80)
print("ИТОГИ ТЕСТИРОВАНИЯ")
print("=" * 80)

if errors:
    print(f"\n[ERROR] Найдено ошибок: {len(errors)}")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
else:
    print("\n[OK] Критических ошибок не найдено")

if monitored_errors:
    print(f"\n[WARN] Проблемы во время мониторинга: {len(monitored_errors)}")
    for i, error in enumerate(monitored_errors, 1):
        print(f"  {i}. {error}")

# Проверяем что все процессы еще работают
print("\nСтатус процессов:")
for name, proc in processes.items():
    if proc:
        if proc.poll() is None:
            print(f"  [OK] {name} работает (PID: {proc.pid})")
        else:
            print(f"  [ERROR] {name} завершился (код: {proc.returncode})")
            if f"{name} завершился" not in [e for e in errors if name in e]:
                errors.append(f"{name} завершился с кодом {proc.returncode}")

if not errors and not monitored_errors:
    print("\n[OK] ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    sys.exit(0)
else:
    print("\n[FAIL] НАЙДЕНЫ ОШИБКИ!")
    sys.exit(1)
