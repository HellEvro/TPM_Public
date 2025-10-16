"""
Модуль для безопасной записи конфигурации в bot_config.py
"""
import re
import os
import logging
from typing import Dict, Any

logger = logging.getLogger('ConfigWriter')

def save_auto_bot_config_to_py(config: Dict[str, Any]) -> bool:
    """
    Безопасно обновляет DEFAULT_AUTO_BOT_CONFIG в bot_config.py
    
    Алгоритм:
    1. Читает файл bot_config.py
    2. Находит блок DEFAULT_AUTO_BOT_CONFIG = {...}
    3. Обновляет только значения, сохраняя комментарии
    4. Записывает обратно в файл
    
    Args:
        config: Словарь с новыми значениями конфигурации
        
    Returns:
        True если успешно, False если ошибка
    """
    try:
        config_file = 'bot_engine/bot_config.py'
        
        if not os.path.exists(config_file):
            logger.error(f"[CONFIG_WRITER] ❌ Файл {config_file} не найден")
            return False
        
        # Читаем файл
        with open(config_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Находим начало и конец блока DEFAULT_AUTO_BOT_CONFIG
        start_idx = None
        end_idx = None
        in_config_block = False
        brace_count = 0
        
        for i, line in enumerate(lines):
            if 'DEFAULT_AUTO_BOT_CONFIG' in line and '=' in line and '{' in line:
                start_idx = i
                in_config_block = True
                brace_count = line.count('{') - line.count('}')
                continue
            
            if in_config_block:
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    end_idx = i
                    break
        
        if start_idx is None or end_idx is None:
            logger.error(f"[CONFIG_WRITER] ❌ Не найден блок DEFAULT_AUTO_BOT_CONFIG")
            return False
        
        logger.info(f"[CONFIG_WRITER] 📝 Найден блок конфигурации: строки {start_idx+1}-{end_idx+1}")
        
        # Обновляем значения в блоке конфигурации
        updated_lines = lines[:start_idx + 1]  # Все строки до начала блока + строка с DEFAULT_AUTO_BOT_CONFIG
        
        for i in range(start_idx + 1, end_idx + 1):
            line = lines[i]
            updated_line = line
            
            # Ищем строки с ключами конфигурации
            # Формат: '    'key': value,  # комментарий' или '    'key': value,'
            match = re.match(r"^(\s*)'([^']+)':\s*([^,#]+)(,?)(.*)$", line)
            
            if match:
                indent = match.group(1)
                key = match.group(2)
                old_value = match.group(3).strip()
                comma = match.group(4)
                comment = match.group(5)
                
                # Если этот ключ есть в новой конфигурации, обновляем значение
                if key in config:
                    new_value = config[key]
                    
                    # Форматируем новое значение в Python-синтаксис
                    if isinstance(new_value, bool):
                        new_value_str = str(new_value)
                    elif isinstance(new_value, str):
                        new_value_str = f"'{new_value}'"
                    elif isinstance(new_value, (int, float)):
                        new_value_str = str(new_value)
                    elif isinstance(new_value, list):
                        new_value_str = str(new_value)
                    else:
                        new_value_str = str(new_value)
                    
                    # Собираем обновленную строку
                    updated_line = f"{indent}'{key}': {new_value_str}{comma}{comment}\n"
                    logger.debug(f"[CONFIG_WRITER] ✏️ {key}: {old_value} → {new_value_str}")
            
            updated_lines.append(updated_line)
        
        # Добавляем все строки после блока конфигурации
        updated_lines.extend(lines[end_idx + 1:])
        
        # Записываем обратно в файл
        with open(config_file, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        
        logger.info(f"[CONFIG_WRITER] ✅ Конфигурация успешно сохранена в {config_file}")
        return True
        
    except Exception as e:
        logger.error(f"[CONFIG_WRITER] ❌ Ошибка сохранения конфигурации: {e}")
        import traceback
        traceback.print_exc()
        return False

