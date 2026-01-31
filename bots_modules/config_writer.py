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
        # Абсолютный путь к bot_config.py (не зависит от cwd процесса)
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_current_dir)
        config_file = os.path.join(_project_root, 'bot_engine', 'bot_config.py')
        
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
        logger.info(f"[CONFIG_WRITER] 🔍 Сохраняемые значения (всего {len(config)} параметров):")
        logger.info(f"  enabled: {config.get('enabled')}")
        logger.info(f"  max_concurrent: {config.get('max_concurrent')}")
        logger.info(f"  risk_cap_percent: {config.get('risk_cap_percent')}")
        logger.info(f"  scope: {config.get('scope')} (тип: {type(config.get('scope')).__name__})")
        logger.info(f"  whitelist: {len(config.get('whitelist', []))} элементов")
        logger.info(f"  blacklist: {len(config.get('blacklist', []))} элементов")
        logger.info(f"  ai_enabled: {config.get('ai_enabled')}")
        logger.info(f"  ai_min_confidence: {config.get('ai_min_confidence')}")
        logger.info(f"  ai_override_original: {config.get('ai_override_original')}")
        logger.info(f"  leverage: {config.get('leverage')}")
        logger.info(f"  trailing_stop_activation: {config.get('trailing_stop_activation')}")
        logger.info(f"  trailing_stop_distance: {config.get('trailing_stop_distance')}")
        logger.info(f"  break_even_trigger: {config.get('break_even_trigger')}")
        logger.info(f"  avoid_down_trend: {config.get('avoid_down_trend')}")
        logger.info(f"  avoid_up_trend: {config.get('avoid_up_trend')}")
        logger.info(f"  exit_scam_enabled: {config.get('exit_scam_enabled')}")
        logger.info(f"  exit_scam_candles: {config.get('exit_scam_candles')}")
        logger.info(f"  exit_scam_single_candle_percent: {config.get('exit_scam_single_candle_percent')}")
        logger.info(f"  exit_scam_multi_candle_count: {config.get('exit_scam_multi_candle_count')}")
        logger.info(f"  exit_scam_multi_candle_percent: {config.get('exit_scam_multi_candle_percent')}")
        logger.info(f"  rsi_long_threshold: {config.get('rsi_long_threshold')}, rsi_short_threshold: {config.get('rsi_short_threshold')}")
        logger.info(f"  rsi_exit: LONG with={config.get('rsi_exit_long_with_trend')}, against={config.get('rsi_exit_long_against_trend')}, SHORT with={config.get('rsi_exit_short_with_trend')}, against={config.get('rsi_exit_short_against_trend')}")
        
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
                    pass
                else:
                    # Собираем обновленную строку
                    # Сохраняем комментарий, если он был
                    if comment:
                        comment_str = f' {comment.strip()}' if comment.strip().startswith('#') else f'  {comment.strip()}'
                    else:
                        comment_str = ''
                    
                    # Всегда добавляем запятую перед комментарием
                    updated_line = f"{indent}'{key}': {new_value_str},{comment_str}\n"
                    # ✅ Логируем ключевые изменения (включая RSI вход/выход и ExitScam)
                    log_keys = (
                        'enabled', 'max_concurrent', 'risk_cap_percent', 'scope', 'whitelist', 'blacklist',
                        'rsi_long_threshold', 'rsi_short_threshold',
                        'rsi_exit_long_with_trend', 'rsi_exit_long_against_trend',
                        'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend',
                        'ai_enabled', 'ai_min_confidence', 'ai_override_original', 'leverage',
                        'trailing_stop_activation', 'trailing_stop_distance', 'break_even_trigger',
                        'avoid_down_trend', 'avoid_up_trend', 'limit_orders_entry_enabled',
                        'limit_orders_percent_steps', 'limit_orders_margin_amounts',
                        'exit_scam_enabled', 'exit_scam_candles', 'exit_scam_single_candle_percent',
                        'exit_scam_multi_candle_count', 'exit_scam_multi_candle_percent',
                    )
                    if key in log_keys:
                        logger.info(f"[CONFIG_WRITER] ✏️ {key}: {old_normalized[:50] if len(old_normalized) <= 50 else old_normalized[:50] + '...'} → {new_normalized[:50] if len(new_normalized) <= 50 else new_normalized[:50] + '...'}")
            
            updated_lines.append(updated_line)
        
        # ✅ КРИТИЧЕСКИ ВАЖНО: Добавляем ключи из config, которых нет в файле
        # Собираем все ключи, которые уже есть в файле
        existing_keys = set()
        for i in range(start_idx + 1, end_idx + 1):
            line = lines[i]
            line_without_comment = re.sub(r'\s*#.*$', '', line).rstrip()
            key_match = re.match(r"^(\s*)'([^']+)':\s*", line_without_comment)
            if key_match:
                existing_keys.add(key_match.group(2))
        
        # ✅ ГАРАНТИРУЕМ, что leverage всегда есть в конфиге
        # Примечание: если leverage не передан в config, это нормально - значит он не изменялся
        # Проверяем только если он действительно отсутствует в файле
        if 'leverage' not in config:
            # Проверяем, есть ли leverage в существующих ключах файла
            if 'leverage' not in existing_keys:
                logger.info(f"[CONFIG_WRITER] ℹ️ leverage отсутствует в файле, добавляем значение по умолчанию: 1")
                config['leverage'] = 1
            else:
                # leverage есть в файле, но не передан в config - это нормально (не изменялся)
                pass
        
        # Добавляем недостающие ключи перед закрывающей скобкой
        # Находим последнюю строку перед закрывающей скобкой
        last_config_line_idx = len(updated_lines) - 1
        for i in range(len(updated_lines) - 1, -1, -1):
            if updated_lines[i].strip() == '}':
                last_config_line_idx = i - 1
                break
        
        # Получаем отступ из последней строки конфига
        last_line = updated_lines[last_config_line_idx] if last_config_line_idx >= 0 else '    '
        indent_match = re.match(r'^(\s*)', last_line)
        indent = indent_match.group(1) if indent_match else '    '
        
        # Добавляем недостающие ключи
        missing_keys = []
        for key in config.keys():
            if key not in existing_keys:
                missing_keys.append(key)
        
        if missing_keys:
            logger.info(f"[CONFIG_WRITER] ➕ Добавляем недостающие ключи: {missing_keys}")
            if 'leverage' in missing_keys:
                logger.warning(f"[CONFIG_WRITER] ⚠️ leverage отсутствовал в файле! Добавляем обратно.")
            # Добавляем запятую к последней строке, если её нет
            if last_config_line_idx >= 0 and not updated_lines[last_config_line_idx].rstrip().endswith(','):
                updated_lines[last_config_line_idx] = updated_lines[last_config_line_idx].rstrip() + ',\n'
            
            # Добавляем новые ключи (leverage первым, если он отсутствует)
            sorted_keys = sorted(missing_keys)
            if 'leverage' in sorted_keys:
                sorted_keys.remove('leverage')
                sorted_keys.insert(0, 'leverage')  # leverage всегда первым
            
            for key in sorted_keys:
                value = config[key]
                value_str = _format_python_value(value)
                # Определяем комментарий на основе ключа
                comment = ''
                if key == 'leverage':
                    comment = '  # ✅ Кредитное плечо (1-125x)'
                elif key == 'default_position_size':
                    comment = '  # Базовый размер позиции (в единицах согласно default_position_mode)'
                elif key == 'default_position_mode':
                    comment = '  # Режим расчета: usdt | percent'
                
                new_line = f"{indent}'{key}': {value_str},{comment}\n"
                updated_lines.insert(last_config_line_idx + 1, new_line)
                last_config_line_idx += 1
        
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
                logger.info(f"  enabled: {DEFAULT_AUTO_BOT_CONFIG.get('enabled')}")
                logger.info(f"  max_concurrent: {DEFAULT_AUTO_BOT_CONFIG.get('max_concurrent')}")
                logger.info(f"  risk_cap_percent: {DEFAULT_AUTO_BOT_CONFIG.get('risk_cap_percent')}")
                logger.info(f"  scope: {DEFAULT_AUTO_BOT_CONFIG.get('scope')}")
                logger.info(f"  ai_enabled: {DEFAULT_AUTO_BOT_CONFIG.get('ai_enabled')}")
                logger.info(f"  ai_min_confidence: {DEFAULT_AUTO_BOT_CONFIG.get('ai_min_confidence')}")
                logger.info(f"  ai_override_original: {DEFAULT_AUTO_BOT_CONFIG.get('ai_override_original')}")
                logger.info(f"  leverage: {DEFAULT_AUTO_BOT_CONFIG.get('leverage')}")
                logger.info(f"  trailing_stop_activation: {DEFAULT_AUTO_BOT_CONFIG.get('trailing_stop_activation')}")
                logger.info(f"  trailing_stop_distance: {DEFAULT_AUTO_BOT_CONFIG.get('trailing_stop_distance')}")
                logger.info(f"  break_even_trigger: {DEFAULT_AUTO_BOT_CONFIG.get('break_even_trigger')}")
                logger.info(f"  avoid_down_trend: {DEFAULT_AUTO_BOT_CONFIG.get('avoid_down_trend')}")
                logger.info(f"  avoid_up_trend: {DEFAULT_AUTO_BOT_CONFIG.get('avoid_up_trend')}")
                
                # ✅ Проверяем, что основные настройки действительно сохранились
                if 'enabled' in config:
                    saved_enabled = DEFAULT_AUTO_BOT_CONFIG.get('enabled')
                    if saved_enabled != config.get('enabled'):
                        logger.error(f"[CONFIG_WRITER] ❌ ОШИБКА: enabled не сохранился! Ожидалось: {config.get('enabled')}, сохранено: {saved_enabled}")
                    else:
                        logger.info(f"[CONFIG_WRITER] ✅ enabled успешно сохранен: {saved_enabled}")
                
                if 'max_concurrent' in config:
                    saved_max_concurrent = DEFAULT_AUTO_BOT_CONFIG.get('max_concurrent')
                    if saved_max_concurrent != config.get('max_concurrent'):
                        logger.error(f"[CONFIG_WRITER] ❌ ОШИБКА: max_concurrent не сохранился! Ожидалось: {config.get('max_concurrent')}, сохранено: {saved_max_concurrent}")
                    else:
                        logger.info(f"[CONFIG_WRITER] ✅ max_concurrent успешно сохранен: {saved_max_concurrent}")
                
                # ✅ КРИТИЧЕСКИ ВАЖНО: Проверяем scope
                if 'scope' in config:
                    saved_scope = DEFAULT_AUTO_BOT_CONFIG.get('scope')
                    expected_scope = config.get('scope')
                    if saved_scope != expected_scope:
                        logger.error(f"[CONFIG_WRITER] ❌ ОШИБКА: scope не сохранился! Ожидалось: {expected_scope} (тип: {type(expected_scope).__name__}), сохранено: {saved_scope} (тип: {type(saved_scope).__name__})")
                    else:
                        logger.info(f"[CONFIG_WRITER] ✅ scope успешно сохранен: {saved_scope}")
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

        pass
        with open(config_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            # ✅ Ищем класс SystemConfig (может быть с комментарием или наследованием)
            if 'class SystemConfig' in line or line.strip().startswith('class SystemConfig'):
                start_idx = i
                pass
                break

        if start_idx is None:
            logger.error(f"[CONFIG_WRITER] ❌ Не найден класс SystemConfig в файле {config_file}")
            pass
            for i, line in enumerate(lines[:20]):
                pass
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
                        pass
            updated_lines.append(line)

        updated_lines.extend(lines[end_idx:])

        # ✅ При сохранении SYSTEM_TIMEFRAME также обновляем модульную константу TIMEFRAME (fallback после перезапуска)
        if 'SYSTEM_TIMEFRAME' in config:
            new_tf = _format_python_value(config['SYSTEM_TIMEFRAME']).strip("'\"")
            for i, line in enumerate(updated_lines):
                if re.match(r"^TIMEFRAME\s*=\s*['\"]", line.strip()) and not line.strip().startswith('#'):
                    updated_lines[i] = f"TIMEFRAME = {repr(new_tf)}\n"
                    break

        with open(config_file, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)

        logger.info("[CONFIG_WRITER] ✅ SystemConfig обновлен в bot_config.py")
        return True

    except Exception as e:
        logger.error(f"[CONFIG_WRITER] ❌ Ошибка сохранения SystemConfig: {e}")
        import traceback
        traceback.print_exc()
        return False

