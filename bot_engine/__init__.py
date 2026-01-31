# Торговый движок ботов

# ===========================================
# АВТОПАТЧИНГ КОНФИГУРАЦИИ
# ===========================================
# При импорте bot_engine автоматически проверяется и патчится bot_config.py
# если в нём отсутствуют новые параметры из bot_config.example.py

import os
from pathlib import Path

def _auto_patch_config():
    """
    Автоматически патчит bot_config.py при импорте модуля.
    Добавляет недостающие параметры из bot_config.example.py.
    """
    bot_engine_dir = Path(__file__).parent
    example_config = bot_engine_dir / "bot_config.example.py"
    user_config = bot_engine_dir / "bot_config.py"
    
    # Если example не существует - пропускаем
    if not example_config.exists():
        return
    
    # Если user config не существует - копируем из example
    if not user_config.exists():
        try:
            user_config.write_text(example_config.read_text(encoding='utf-8'), encoding='utf-8')
            print(f"[CONFIG] Создан bot_config.py из примера")
        except Exception as e:
            print(f"[CONFIG] Ошибка создания bot_config.py: {e}")
        return
    
    # Проверяем нужен ли патчинг (быстрая проверка по размеру файла)
    example_size = example_config.stat().st_size
    user_size = user_config.stat().st_size
    
    # Если user config значительно меньше example - вероятно нужен патчинг
    if user_size < example_size * 0.9:
        try:
            # Импортируем и запускаем патчер
            import sys
            scripts_dir = bot_engine_dir.parent / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            
            from patch_bot_config import parse_config_file, generate_patch, apply_patch, EXAMPLE_CONFIG, USER_CONFIG
            import ast
            
            example_classes, example_content = parse_config_file(example_config)
            user_classes, user_content = parse_config_file(user_config)
            
            if example_classes and user_classes:
                patches = generate_patch(example_classes, user_classes, example_content)
                
                if patches:
                    new_content = apply_patch(user_config, user_content, patches)
                    
                    # Проверяем синтаксис
                    try:
                        ast.parse(new_content)
                        user_config.write_text(new_content, encoding='utf-8')
                        
                        # Подсчитываем изменения
                        total_attrs = sum(len(p[2]) for p in patches if p[1] == 'add_attrs')
                        total_classes = sum(1 for p in patches if p[1] == 'add_class')
                        print(f"[CONFIG] Автопатчинг: добавлено {total_attrs} атрибутов, {total_classes} классов")
                    except SyntaxError:
                        pass  # Не применяем битый патч
        except Exception:
            pass  # Молча пропускаем ошибки патчинга

# Запускаем автопатчинг при импорте
_auto_patch_config()
