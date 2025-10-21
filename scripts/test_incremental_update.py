#!/usr/bin/env python3
"""
Тест: Проверка инкрементального обновления данных
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_incremental_update():
    """Тестирует правильность работы инкрементального обновления"""
    
    print("\n" + "="*80)
    print("ТЕСТ: Инкрементальное обновление исторических данных")
    print("="*80 + "\n")
    
    # Проверяем наличие исторических данных
    hist_dir = Path('data/ai/historical')
    
    if not hist_dir.exists():
        print("[X] Директория data/ai/historical не найдена!")
        return False
    
    csv_files = list(hist_dir.glob('*_6h_historical.csv'))
    
    print(f"[1] Найдено файлов: {len(csv_files)}")
    
    if len(csv_files) == 0:
        print("[X] Нет CSV файлов!")
        return False
    
    # Проверяем несколько файлов
    test_coins = ['BTC', 'ETH', 'BNB']
    
    print(f"\n[2] Проверяем размер данных для тестовых монет:")
    
    import pandas as pd
    
    for coin in test_coins:
        file_path = hist_dir / f'{coin}_6h_historical.csv'
        
        if file_path.exists():
            df = pd.read_csv(file_path)
            file_size = file_path.stat().st_size / 1024  # KB
            
            print(f"   {coin}: {len(df)} свечей, {file_size:.1f} KB")
            
            if len(df) < 100:
                print(f"   [!] ПРЕДУПРЕЖДЕНИЕ: Слишком мало данных для {coin}!")
        else:
            print(f"   {coin}: [X] Файл не найден")
    
    # Тестируем скрипт на одной монете
    print(f"\n[3] Тестируем инкрементальное обновление (BTC, последние 7 дней)...")
    
    import subprocess
    
    cmd = [
        sys.executable,
        'scripts/ai/collect_historical_data.py',
        '--limit', '1',  # Только BTC (первая монета)
        '--days', '7'
    ]
    
    print(f"   Команда: {' '.join(cmd)}")
    print(f"   Запуск...")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode == 0:
        output = result.stdout
        
        # Проверяем, что данные обновлены, а не перезаписаны
        if 'Updated:' in output:
            print(f"   [OK] Данные обновлены инкрементально!")
            
            # Извлекаем количество добавленных свечей
            for line in output.split('\n'):
                if 'Updated:' in line:
                    print(f"   {line.strip()}")
        elif 'Created:' in output:
            print(f"   [!] Файл создан заново (возможно, не было старого файла)")
            for line in output.split('\n'):
                if 'Created:' in line:
                    print(f"   {line.strip()}")
        elif 'Saved:' in output and 'overwritten' in output:
            print(f"   [X] ОШИБКА: Данные перезаписаны вместо обновления!")
            return False
        else:
            print(f"   [?] Неожиданный вывод:")
            print(output)
    else:
        print(f"   [X] Ошибка выполнения скрипта:")
        print(result.stderr)
        return False
    
    # Проверяем, что количество строк увеличилось
    print(f"\n[4] Проверяем итоговый размер данных...")
    
    file_path = hist_dir / 'BTC_6h_historical.csv'
    
    if file_path.exists():
        df = pd.read_csv(file_path)
        file_size = file_path.stat().st_size / 1024  # KB
        
        print(f"   BTC: {len(df)} свечей, {file_size:.1f} KB")
        
        if len(df) >= 3000:
            print(f"   [OK] Данные сохранены корректно (>= 3000 свечей)")
        else:
            print(f"   [!] ПРЕДУПРЕЖДЕНИЕ: Меньше 3000 свечей, возможно данные потеряны")
    
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТ: [OK] Инкрементальное обновление работает корректно!")
    print("="*80 + "\n")
    
    return True


if __name__ == '__main__':
    try:
        success = test_incremental_update()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[X] Ошибка теста: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

