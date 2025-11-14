#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для разбиения больших JSON файлов на части по 100MB
"""

import os
import json
import sys

# Настройка кодировки для Windows консоли
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

def split_json_file(filepath: str, max_size_mb: int = 100):
    """
    Разбивает большой JSON файл на части
    
    Args:
        filepath: Путь к файлу
        max_size_mb: Максимальный размер части в MB
    """
    if not os.path.exists(filepath):
        print(f"❌ Файл {filepath} не найден")
        return False
    
    file_size = os.path.getsize(filepath)
    file_size_mb = file_size / 1024 / 1024
    max_size = max_size_mb * 1024 * 1024
    
    print(f"📊 Размер файла: {file_size_mb:.2f} MB")
    
    if file_size <= max_size:
        print(f"✅ Файл меньше {max_size_mb}MB, разбиение не требуется")
        return True
    
    print(f"📦 Разбиваем файл на части по {max_size_mb}MB...")
    
    # Загружаем данные
    with open(filepath, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # Разбиваем данные на части
    part_num = 1
    current_part = {}
    current_size = 0
    
    # Сохраняем метаданные в первую часть
    if 'metadata' in full_data:
        current_part['metadata'] = full_data['metadata']
    
    # Разбиваем историю на части
    if 'history' in full_data and isinstance(full_data['history'], list):
        history = full_data['history']
        current_history = []
        
        for item in history:
            # Сериализуем элемент чтобы узнать его размер
            item_json = json.dumps(item, ensure_ascii=False)
            item_size = len(item_json.encode('utf-8'))
            
            if current_size + item_size > max_size and current_history:
                # Сохраняем текущую часть
                current_part['history'] = current_history
                part_file = f"{filepath}.part{part_num}"
                with open(part_file, 'w', encoding='utf-8') as f:
                    json.dump(current_part, f, ensure_ascii=False, indent=2)
                part_size = os.path.getsize(part_file) / 1024 / 1024
                print(f"   💾 Часть {part_num}: {part_size:.2f} MB")
                
                # Начинаем новую часть
                part_num += 1
                current_part = {'metadata': full_data.get('metadata', {})}
                current_history = []
                current_size = 0
            
            current_history.append(item)
            current_size += item_size
        
        # Сохраняем последнюю часть истории
        if current_history:
            current_part['history'] = current_history
    
    # Сохраняем latest в последнюю часть
    if 'latest' in full_data:
        current_part['latest'] = full_data['latest']
    
    # Сохраняем последнюю часть
    if current_part:
        part_file = f"{filepath}.part{part_num}"
        with open(part_file, 'w', encoding='utf-8') as f:
            json.dump(current_part, f, ensure_ascii=False, indent=2)
        part_size = os.path.getsize(part_file) / 1024 / 1024
        print(f"   💾 Часть {part_num}: {part_size:.2f} MB")
    
    print(f"✅ Файл разбит на {part_num} частей")
    
    # Удаляем основной файл (он будет собираться из частей при загрузке)
    print(f"🗑️ Удаляем основной файл {filepath} (будет собираться из частей)")
    os.remove(filepath)
    
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Использование: python scripts/split_large_json.py <путь_к_файлу> [max_size_mb]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    max_size_mb = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    success = split_json_file(filepath, max_size_mb)
    sys.exit(0 if success else 1)

