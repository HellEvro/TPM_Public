#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПОЛНЫЙ ТРЕЙСЕР - логирует КАЖДУЮ строку выполняемого кода
"""
import sys
import logging

logger = logging.getLogger('TRACE')

# Файлы которые НЕ трейсим (библиотеки)
SKIP_FILES = [
    'logging', 'threading', 'queue', 'socket', 'ssl', 
    'http', 'urllib', 'json', 'datetime', 'traceback',
    'site-packages', 'AppData', 'Python313', 'pybit'
]

def should_trace_file(filename):
    """Проверяет нужно ли трейсить этот файл"""
    if not filename:
        return False
    
    # Пропускаем библиотеки
    for skip in SKIP_FILES:
        if skip in filename:
            return False
    
    # Трейсим только наш код
    return 'InfoBot' in filename or 'bots_modules' in filename or 'exchanges' in filename or 'bot_engine' in filename

last_logged = {}
last_exec_time = {}
global_last_time = None

def trace_calls(frame, event, arg):
    """Функция трейсинга - вызывается для КАЖДОЙ строки кода"""
    global global_last_time
    
    if event != 'line':
        return trace_calls
    
    filename = frame.f_code.co_filename
    
    if not should_trace_file(filename):
        return trace_calls
    
    lineno = frame.f_lineno
    func_name = frame.f_code.co_name
    
    # Получаем имя файла без пути
    short_file = filename.split('\\')[-1] if '\\' in filename else filename.split('/')[-1]
    
    # Ключ для дедупликации (не спамим циклы)
    key = f"{short_file}:{lineno}"
    
    import time
    from datetime import datetime
    current_time = time.time()
    
    # Рассчитываем время с ПОСЛЕДНЕЙ строки (любой)
    if global_last_time is not None:
        delta = current_time - global_last_time
    else:
        delta = 0
    
    global_last_time = current_time
    
    # Логируем только если прошло >0.3 сек с последнего лога ЭТОЙ строки (убираем спам циклов)
    # ИЛИ если delta > 1 сек (показываем медленные операции)
    if delta > 1.0 or key not in last_logged or (current_time - last_logged[key]) > 0.3:
        # Получаем КОД строки
        try:
            import linecache
            line_code = linecache.getline(filename, lineno).strip()
            if len(line_code) > 80:
                line_code = line_code[:80] + "..."
        except:
            line_code = "???"
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # ⚡ Показываем время выполнения если >0.01 сек
        if delta > 0.01:
            delta_str = f"+{delta:.3f}s"
        else:
            delta_str = ""
        
        print(f"[{timestamp}] {delta_str:>10} | {short_file}:{lineno:4} | {func_name:30} | {line_code}", flush=True)
        last_logged[key] = current_time
    
    return trace_calls

def enable_trace():
    """Включает трейсинг ВСЕХ строк кода"""
    print("=" * 80)
    print("TRACE: FULL CODE TRACING ENABLED")
    print("=" * 80)
    sys.settrace(trace_calls)

def disable_trace():
    """Выключает трейсинг"""
    sys.settrace(None)
    print("TRACE: DISABLED")

