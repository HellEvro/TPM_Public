"""
Модуль для безопасной записи конфигурации в bot_config.py
"""
import re
import os
import logging
from typing import Dict, Any

logger = logging.getLogger('ConfigWriter')


def _format_python_value(value: Any) -> str:
    """Возвращает строковое представление значения в синтаксисе Python."""
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if isinstance(value, str):
        return repr(value)
    if value is None:
        return 'None'
    if isinstance(value, (list, tuple)):
        # Правильно форматируем списки и кортежи
        items = ', '.join(_format_python_value(item) for item in value)
        return f'[{items}]' if isinstance(value, list) else f'({items})'
    if isinstance(value, dict):
        # Форматируем словари
        items = ', '.join(f"{repr(k)}: {_format_python_value(v)}" for k, v in value.items())
        return f'{{{items}}}'
    return str(value)


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
        
        # ✅ Логируем ключевые значения, которые будут сохранены
        logger.info(f"[CONFIG_WRITER] 🔍 Сохраняемые значения:")
        logger.info(f"  trailing_stop_activation: {config.get('trailing_stop_activation')}")
        logger.info(f"  trailing_stop_distance: {config.get('trailing_stop_distance')}")
        logger.info(f"  break_even_trigger: {config.get('break_even_trigger')}")
        logger.info(f"  avoid_down_trend: {config.get('avoid_down_trend')}")
        logger.info(f"  avoid_up_trend: {config.get('avoid_up_trend')}")
        
        # Обновляем значения в блоке конфигурации
        updated_lines = lines[:start_idx + 1]  # Все строки до начала блока + строка с DEFAULT_AUTO_BOT_CONFIG
        
        for i in range(start_idx + 1, end_idx + 1):
            line = lines[i]
            updated_line = line
            
            # Ищем строки с ключами конфигурации
            # Формат: '    'key': value,  # комментарий' или '    'key': value,'
            # Улучшенный парсинг для обработки массивов и сложных значений
            
            # Сначала извлекаем комментарий
            comment_match = re.search(r'\s*#.*$', line)
            comment = comment_match.group(0) if comment_match else ''
            
            # Убираем комментарий из строки для парсинга
            line_without_comment = re.sub(r'\s*#.*$', '', line).rstrip()
            
            # Парсим ключ
            key_match = re.match(r"^(\s*)'([^']+)':\s*", line_without_comment)
            if not key_match:
                updated_lines.append(updated_line)
                continue
                
            indent = key_match.group(1)
            key = key_match.group(2)
            
            # Извлекаем значение (все что после ': ' до запятой или конца строки)
            # Но нужно учесть, что значение может быть массивом со скобками
            value_part = line_without_comment[len(key_match.group(0)):].rstrip()
            
            # Убираем запятую в конце, если есть
            has_comma = value_part.endswith(',')
            if has_comma:
                value_part = value_part[:-1].rstrip()
            
            old_value = value_part
            
            # Если этот ключ есть в новой конфигурации, обновляем значение
            if key in config:
                new_value = config[key]
                
                # Форматируем новое значение в Python-синтаксис
                new_value_str = _format_python_value(new_value)
                
                # Для массивов и сложных значений сравниваем нормализованные версии
                old_normalized = old_value.rstrip(',').strip()
                new_normalized = new_value_str.strip()
                
                if old_normalized == new_normalized:
                    # Значение не изменилось — оставляем строку как есть
                    logger.debug(f"[CONFIG_WRITER] ↩️ {key}: без изменений")
                else:
                    # Собираем обновленную строку
                    # Сохраняем комментарий, если он был
                    if comment:
                        comment_str = f' {comment.strip()}' if comment.strip().startswith('#') else f'  {comment.strip()}'
                    else:
                        comment_str = ''
                    
                    # Всегда добавляем запятую перед комментарием
                    updated_line = f"{indent}'{key}': {new_value_str},{comment_str}\n"
                    # ✅ Логируем ключевые изменения
                    if key in ('trailing_stop_activation', 'trailing_stop_distance', 'break_even_trigger', 'avoid_down_trend', 'avoid_up_trend', 'limit_orders_entry_enabled', 'limit_orders_percent_steps', 'limit_orders_margin_amounts'):
                        logger.info(f"[CONFIG_WRITER] ✏️ {key}: {old_normalized[:50]}... → {new_normalized[:50]}...")
                    else:
                        logger.debug(f"[CONFIG_WRITER] ✏️ {key}: {old_normalized[:50]}... → {new_normalized[:50]}...")
            
            updated_lines.append(updated_line)
        
        # Добавляем все строки после блока конфигурации
        updated_lines.extend(lines[end_idx + 1:])
        
        # Записываем обратно в файл
        with open(config_file, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        
        # ✅ ПРОВЕРЯЕМ, что файл действительно обновлен - читаем обратно ключевые значения
        try:
            import importlib
            import sys
            # Принудительно перезагружаем модуль
            if 'bot_engine.bot_config' in sys.modules:
                import bot_engine.bot_config
                importlib.reload(bot_engine.bot_config)
                from bot_engine.bot_config import DEFAULT_AUTO_BOT_CONFIG
                logger.info(f"[CONFIG_WRITER] ✅ Проверка сохраненных значений:")
                logger.info(f"  trailing_stop_activation: {DEFAULT_AUTO_BOT_CONFIG.get('trailing_stop_activation')}")
                logger.info(f"  trailing_stop_distance: {DEFAULT_AUTO_BOT_CONFIG.get('trailing_stop_distance')}")
                logger.info(f"  break_even_trigger: {DEFAULT_AUTO_BOT_CONFIG.get('break_even_trigger')}")
                logger.info(f"  avoid_down_trend: {DEFAULT_AUTO_BOT_CONFIG.get('avoid_down_trend')}")
                logger.info(f"  avoid_up_trend: {DEFAULT_AUTO_BOT_CONFIG.get('avoid_up_trend')}")
        except Exception as check_error:
            logger.warning(f"[CONFIG_WRITER] ⚠️ Не удалось проверить сохраненные значения: {check_error}")
        
        logger.info(f"[CONFIG_WRITER] ✅ Конфигурация успешно сохранена в {config_file}")
        return True
        
    except Exception as e:
        logger.error(f"[CONFIG_WRITER] ❌ Ошибка сохранения конфигурации: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_system_config_to_py(config: Dict[str, Any]) -> bool:
    """
    Безопасно обновляет класс SystemConfig в bot_config.py.
    config — словарь { 'ATTRIBUTE_NAME': value }.
    """
    try:
        # ✅ Определяем путь к файлу относительно корня проекта
        # Получаем директорию текущего модуля (bots_modules)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Поднимаемся на уровень выше (в корень проекта)
        project_root = os.path.dirname(current_dir)
        # Формируем путь к bot_config.py
        config_file = os.path.join(project_root, 'bot_engine', 'bot_config.py')
        
        if not os.path.exists(config_file):
            # ✅ Попробуем альтернативный путь (относительный)
            alt_config_file = 'bot_engine/bot_config.py'
            if os.path.exists(alt_config_file):
                config_file = alt_config_file
            else:
                logger.error(f"[CONFIG_WRITER] ❌ Файл {config_file} не найден (проверяли также {alt_config_file})")
                return False

        logger.debug(f"[CONFIG_WRITER] 📝 Открываем файл: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            # ✅ Ищем класс SystemConfig (может быть с комментарием или наследованием)
            if 'class SystemConfig' in line or line.strip().startswith('class SystemConfig'):
                start_idx = i
                logger.debug(f"[CONFIG_WRITER] ✅ Найден класс SystemConfig на строке {i+1}: {line.strip()}")
                break

        if start_idx is None:
            logger.error(f"[CONFIG_WRITER] ❌ Не найден класс SystemConfig в файле {config_file}")
            logger.debug(f"[CONFIG_WRITER] 🔍 Первые 20 строк файла:")
            for i, line in enumerate(lines[:20]):
                logger.debug(f"  {i+1}: {line.rstrip()}")
            return False

        for j in range(start_idx + 1, len(lines)):
            line = lines[j]
            if line.startswith('class ') and not line.startswith('class SystemConfig'):
                end_idx = j
                break
        if end_idx is None:
            end_idx = len(lines)

        updated_lines = lines[:start_idx + 1]

        for i in range(start_idx + 1, end_idx):
            line = lines[i]
            match = re.match(r"^(\s+)([A-Z0-9_]+)\s*=\s*([^#\n]+)(.*)$", line)
            if match:
                indent, attr_name, old_value, comment = match.groups()
                attr_name = attr_name.strip()
                if attr_name in config:
                    new_value = _format_python_value(config[attr_name])
                    if old_value.strip() != new_value:
                        comment_fragment = comment or ''
                        if comment_fragment and not comment_fragment.startswith(' '):
                            comment_fragment = f' {comment_fragment}'
                        line = f"{indent}{attr_name} = {new_value}{comment_fragment}\n"
                        logger.debug(f"[CONFIG_WRITER] ✏️ {attr_name}: {old_value.strip()} → {new_value}")
            updated_lines.append(line)

        updated_lines.extend(lines[end_idx:])

        with open(config_file, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)

        logger.info("[CONFIG_WRITER] ✅ SystemConfig обновлен в bot_config.py")
        return True

    except Exception as e:
        logger.error(f"[CONFIG_WRITER] ❌ Ошибка сохранения SystemConfig: {e}")
        import traceback
        traceback.print_exc()
        return False

