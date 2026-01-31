#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт автоматического патчинга bot_config.py

Сравнивает bot_config.example.py с bot_config.py и добавляет недостающие параметры.
Не перезаписывает существующие значения пользователя.

Использование:
    python scripts/patch_bot_config.py          # Патчинг с выводом изменений
    python scripts/patch_bot_config.py --dry    # Только показать что будет изменено
    python scripts/patch_bot_config.py --force  # Принудительный патчинг даже если файл не существует
"""

import os
import sys
import re
import ast
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

# Настройка кодировки для Windows консоли
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


# Определяем корневую директорию проекта
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
BOT_ENGINE_DIR = PROJECT_ROOT / "bot_engine"
EXAMPLE_CONFIG = BOT_ENGINE_DIR / "bot_config.example.py"
USER_CONFIG = BOT_ENGINE_DIR / "bot_config.py"


def parse_config_file(filepath: Path) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Парсит конфигурационный файл и извлекает все классы и их атрибуты.
    
    Returns:
        Tuple[classes_dict, full_content]:
            - classes_dict: {class_name: {attr_name: (value, line_number, raw_line)}}
            - full_content: полный текст файла
    """
    if not filepath.exists():
        return {}, ""
    
    content = filepath.read_text(encoding='utf-8')
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"❌ Ошибка синтаксиса в {filepath}: {e}")
        return {}, content
    
    classes = {}
    lines = content.split('\n')
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            attrs = {}
            
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            attr_name = target.id
                            line_num = item.lineno
                            raw_line = lines[line_num - 1] if line_num <= len(lines) else ""
                            
                            # Получаем значение как строку из исходного кода
                            try:
                                value = ast.literal_eval(ast.unparse(item.value))
                            except:
                                value = ast.unparse(item.value)
                            
                            attrs[attr_name] = (value, line_num, raw_line)
            
            classes[class_name] = attrs
    
    return classes, content


def find_class_end_line(content: str, class_name: str) -> int:
    """
    Находит номер последней строки класса.
    """
    lines = content.split('\n')
    in_class = False
    class_indent = 0
    last_line = 0
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Нашли начало класса
        if re.match(rf'^class\s+{class_name}\s*[\(:]', stripped):
            in_class = True
            class_indent = len(line) - len(line.lstrip())
            last_line = i
            continue
        
        if in_class:
            # Пустая строка или комментарий - продолжаем
            if not stripped or stripped.startswith('#'):
                continue
            
            # Проверяем отступ
            current_indent = len(line) - len(line.lstrip())
            
            # Если отступ больше чем у класса - это содержимое класса
            if current_indent > class_indent:
                last_line = i
            # Если отступ такой же или меньше - класс закончился
            elif current_indent <= class_indent and stripped and not stripped.startswith('#'):
                break
    
    return last_line


def generate_patch(example_classes: Dict, user_classes: Dict, example_content: str) -> List[Tuple[str, str, List[str]]]:
    """
    Генерирует патч - список изменений для применения.
    
    Returns:
        List of (class_name, action, lines_to_add)
        action: 'add_attrs' - добавить атрибуты в существующий класс
                'add_class' - добавить новый класс целиком
    """
    patches = []
    
    for class_name, example_attrs in example_classes.items():
        if class_name not in user_classes:
            # Класс отсутствует - нужно добавить целиком
            # Извлекаем весь класс из example
            class_lines = extract_class_from_content(example_content, class_name)
            if class_lines:
                patches.append((class_name, 'add_class', class_lines))
        else:
            # Класс есть - проверяем недостающие атрибуты
            user_attrs = user_classes[class_name]
            missing_attrs = []
            
            for attr_name, (value, line_num, raw_line) in example_attrs.items():
                if attr_name not in user_attrs:
                    missing_attrs.append(raw_line)
            
            if missing_attrs:
                patches.append((class_name, 'add_attrs', missing_attrs))
    
    return patches


def extract_class_from_content(content: str, class_name: str) -> List[str]:
    """
    Извлекает полный текст класса из файла.
    """
    lines = content.split('\n')
    result = []
    in_class = False
    class_indent = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Нашли начало класса
        if re.match(rf'^class\s+{class_name}\s*[\(:]', stripped):
            in_class = True
            class_indent = len(line) - len(line.lstrip())
            result.append(line)
            continue
        
        if in_class:
            # Пустая строка внутри класса
            if not stripped:
                result.append(line)
                continue
            
            # Комментарий
            if stripped.startswith('#'):
                result.append(line)
                continue
            
            current_indent = len(line) - len(line.lstrip())
            
            # Если отступ больше - содержимое класса
            if current_indent > class_indent:
                result.append(line)
            # Новый класс или что-то на том же уровне - конец
            elif current_indent <= class_indent:
                break
    
    return result


def apply_patch(user_config_path: Path, user_content: str, patches: List[Tuple[str, str, List[str]]]) -> str:
    """
    Применяет патч к файлу конфигурации.
    """
    if not user_content:
        # Файл не существует - копируем example
        return EXAMPLE_CONFIG.read_text(encoding='utf-8')
    
    lines = user_content.split('\n')
    
    for class_name, action, patch_lines in patches:
        if action == 'add_attrs':
            # Находим конец класса и добавляем атрибуты перед ним
            end_line = find_class_end_line(user_content, class_name)
            if end_line > 0:
                # Добавляем пустую строку и новые атрибуты
                insert_lines = [''] + patch_lines
                lines = lines[:end_line] + insert_lines + lines[end_line:]
                # Обновляем content для следующих итераций
                user_content = '\n'.join(lines)
        
        elif action == 'add_class':
            # Добавляем класс в конец файла
            lines.extend(['', ''])
            lines.extend(patch_lines)
            user_content = '\n'.join(lines)
    
    return user_content


def main():
    parser = argparse.ArgumentParser(description='Патчинг bot_config.py')
    parser.add_argument('--dry', '-d', action='store_true', help='Только показать изменения')
    parser.add_argument('--force', '-f', action='store_true', help='Создать файл если не существует')
    parser.add_argument('--quiet', '-q', action='store_true', help='Минимальный вывод')
    args = parser.parse_args()
    
    if not EXAMPLE_CONFIG.exists():
        print(f"❌ Файл примера не найден: {EXAMPLE_CONFIG}")
        return 1
    
    if not USER_CONFIG.exists():
        if args.force:
            print(f"📝 Создание {USER_CONFIG} из примера...")
            if not args.dry:
                USER_CONFIG.write_text(EXAMPLE_CONFIG.read_text(encoding='utf-8'), encoding='utf-8')
            print("✅ Файл создан")
            return 0
        else:
            print(f"⚠️ Файл {USER_CONFIG} не существует")
            print("   Используйте --force для создания из примера")
            return 1
    
    # Парсим оба файла
    example_classes, example_content = parse_config_file(EXAMPLE_CONFIG)
    user_classes, user_content = parse_config_file(USER_CONFIG)
    
    if not example_classes:
        print("❌ Не удалось распарсить example файл")
        return 1
    
    # Генерируем патч
    patches = generate_patch(example_classes, user_classes, example_content)
    
    if not patches:
        if not args.quiet:
            print("✅ Конфигурация актуальна, патчинг не требуется")
        return 0
    
    # Выводим информацию о патче
    print("=" * 60)
    print("📋 ПАТЧИНГ BOT_CONFIG.PY")
    print("=" * 60)
    
    total_attrs = 0
    total_classes = 0
    
    for class_name, action, patch_lines in patches:
        if action == 'add_attrs':
            print(f"\n🔧 Класс {class_name}: добавление {len(patch_lines)} атрибутов")
            for line in patch_lines:
                attr_match = re.match(r'\s*(\w+)\s*=', line)
                if attr_match:
                    print(f"   + {attr_match.group(1)}")
                total_attrs += 1
        elif action == 'add_class':
            print(f"\n📦 Новый класс: {class_name}")
            total_classes += 1
    
    print(f"\n📊 Итого: {total_attrs} атрибутов, {total_classes} классов")
    
    if args.dry:
        print("\n⚠️ Режим --dry: изменения НЕ применены")
        return 0
    
    # Применяем патч
    new_content = apply_patch(USER_CONFIG, user_content, patches)
    
    # Проверяем синтаксис перед записью
    try:
        ast.parse(new_content)
    except SyntaxError as e:
        print(f"\n❌ Ошибка синтаксиса после патчинга: {e}")
        print("   Патч НЕ применён")
        return 1
    
    # Записываем
    USER_CONFIG.write_text(new_content, encoding='utf-8')
    print(f"\n✅ Патч успешно применён к {USER_CONFIG}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
