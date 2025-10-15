#!/usr/bin/env python3
"""
Быстрая проверка синтаксиса Python файлов
"""

import py_compile
import sys

def check_syntax(filename):
    """Проверяет синтаксис файла"""
    try:
        py_compile.compile(filename, doraise=True)
        print(f"[OK] {filename}: Syntax OK")
        return True
    except py_compile.PyCompileError as e:
        print(f"[ERROR] {filename}: SYNTAX ERROR")
        print(f"   Line {e.exc_value.lineno}: {e.exc_value.msg}")
        return False

if __name__ == '__main__':
    files = ['bots.py', 'app.py']
    
    print("=" * 60)
    print("ПРОВЕРКА СИНТАКСИСА")
    print("=" * 60)
    
    all_ok = True
    for filename in files:
        if not check_syntax(filename):
            all_ok = False
    
    print("=" * 60)
    if all_ok:
        print("[SUCCESS] All files OK!")
        sys.exit(0)
    else:
        print("[ERROR] Errors found!")
        sys.exit(1)

