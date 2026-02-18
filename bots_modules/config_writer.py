"""
Модуль для безопасной записи конфигурации в bot_config.py.
Сохранение: обновление каждого параметра в файле по отдельности (не перезапись всего блока).
Загрузка: чтение файла и извлечение каждого параметра по строке с применением в конфиг.
Миграция: создание configs/bot_config.py из примера (при отсутствии). Конфиги только в configs/.
"""
import ast
import re
import os
import shutil
import logging
import threading
import importlib.util
import json as _json
from typing import Dict, Any, Optional

logger = logging.getLogger('ConfigWriter')
_config_write_lock = threading.Lock()


def _ensure_bot_config_exists(config_file: str) -> bool:
    """
    Если bot_config.py отсутствует — создаёт его из bot_config.example.py.
    Возвращает True если файл есть (или успешно создан), False если создать не удалось.
    """
    if os.path.exists(config_file):
        return True
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_current_dir)
    example_file = os.path.join(_project_root, 'configs', 'bot_config.example.py')
    if not os.path.exists(example_file):
        logger.error(f"[CONFIG_WRITER] ❌ Файл-пример {example_file} не найден, невозможно создать bot_config.py")
        return False
    try:
        shutil.copy2(example_file, config_file)
        logger.info(f"[CONFIG_WRITER] ✅ Создан {config_file} из configs/bot_config.example.py")
        return True
    except Exception as e:
        logger.error(f"[CONFIG_WRITER] ❌ Не удалось создать bot_config.py из примера: {e}")
        return False


def migrate_old_bot_config_to_configs(project_root: Optional[str] = None) -> bool:
    """
    Создаёт configs/bot_config.py из примера при отсутствии; накладывает data/auto_bot_config.json при миграции.
    Конфиги только в configs/. Возвращает True если миграция/создание выполнены, False иначе.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configs_bot = os.path.join(project_root, 'configs', 'bot_config.py')
    example_bot = os.path.join(project_root, 'configs', 'bot_config.example.py')
    data_json = os.path.join(project_root, 'data', 'auto_bot_config.json')

    if os.path.exists(configs_bot):
        return False

    merged: Dict[str, Any] = {}

    # 1) Наложить данные из data/auto_bot_config.json (если были)
    if os.path.isfile(data_json):
        try:
            with open(data_json, 'r', encoding='utf-8') as f:
                data_cfg = _json.load(f)
            if isinstance(data_cfg, dict):
                merged.update(data_cfg)
                logger.info("[CONFIG_WRITER] 📥 Наложены настройки из data/auto_bot_config.json")
        except Exception as e:
            logger.warning(f"[CONFIG_WRITER] ⚠️ Не удалось прочитать data/auto_bot_config.json: {e}")

    # 2) Создать configs/bot_config.py из configs/bot_config.example.py
    if not os.path.isfile(example_bot):
        logger.error("[CONFIG_WRITER] ❌ configs/bot_config.example.py не найден, миграция отменена")
        return False
    try:
        os.makedirs(os.path.dirname(configs_bot), exist_ok=True)
        shutil.copy2(example_bot, configs_bot)
        logger.info("[CONFIG_WRITER] ✅ Создан configs/bot_config.py из примера")
    except Exception as e:
        logger.error(f"[CONFIG_WRITER] ❌ Не удалось создать configs/bot_config.py: {e}")
        return False

    # 4) Записать перенесённые настройки в класс AutoBotConfig
    if merged:
        if save_auto_bot_config_current_to_py(merged):
            logger.info("[CONFIG_WRITER] ✅ Миграция завершена: настройки перенесены в configs/bot_config.py")
        else:
            logger.warning("[CONFIG_WRITER] ⚠️ Миграция создала файл, но не удалось обновить AutoBotConfig")
    return True


def migrate_old_keys_to_configs(project_root: Optional[str] = None) -> bool:
    """
    Переносит ключи из app/keys.py в configs/keys.py.
    - Если configs/keys.py нет: создаём из app/keys.py (если не заглушка) или из configs/keys.example.py.
    - Если configs/keys.py есть, но это шаблон (плейсхолдеры), а в app/keys.py реальные ключи — перезаписываем.
    Возвращает True если ключи перенесены из app/, False иначе.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configs_keys = os.path.join(project_root, 'configs', 'keys.py')
    app_keys = os.path.join(project_root, 'app', 'keys.py')
    example_keys = os.path.join(project_root, 'configs', 'keys.example.py')

    os.makedirs(os.path.dirname(configs_keys), exist_ok=True)

    def app_has_real_keys():
        if not os.path.isfile(app_keys):
            return False
        try:
            with open(app_keys, 'r', encoding='utf-8') as f:
                c = f.read()
            return 'from configs.keys import' not in c
        except Exception:
            return False

    def configs_is_template():
        if not os.path.isfile(configs_keys):
            return True
        try:
            with open(configs_keys, 'r', encoding='utf-8') as f:
                return 'YOUR_BYBIT_API_KEY_HERE' in f.read()
        except Exception:
            return True

    # configs/keys.py есть и не шаблон — не трогаем
    if os.path.exists(configs_keys) and not configs_is_template():
        return False

    # Есть app/keys.py с реальными ключами — переносим в configs/
    if app_has_real_keys():
        try:
            shutil.copy2(app_keys, configs_keys)
            logger.info("[CONFIG_WRITER] ✅ Ключи перенесены из app/keys.py в configs/keys.py")
            return True
        except Exception as e:
            logger.warning(f"[CONFIG_WRITER] ⚠️ Не удалось скопировать app/keys.py в configs/: {e}")

    # configs/keys.py нет — создаём из примера
    if not os.path.exists(configs_keys) and os.path.isfile(example_keys):
        try:
            shutil.copy2(example_keys, configs_keys)
            logger.info("[CONFIG_WRITER] ✅ Создан configs/keys.py из примера (заполните ключи)")
        except Exception as e:
            logger.error(f"[CONFIG_WRITER] ❌ Не удалось создать configs/keys.py: {e}")
    return False


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


def _find_class_block(lines: list, class_name: str):
    """
    Находит блок класса в файле: начало (строка с "class ClassName") и конец (первая строка вне тела класса).
    Возвращает (start_idx, end_idx) или (None, None) если не найдено.
    КРИТИЧНО: не считать концом тела строки только с ] или ) — это закрытие многострочного
    WHITELIST/BLACKLIST; иначе новые ключи вставлялись бы перед ] и конфиг ломался.
    """
    start_idx = None
    for i, line in enumerate(lines):
        if re.match(rf'^class\s+{re.escape(class_name)}\s*[:(]', line.strip()):
            start_idx = i
            break
    if start_idx is None:
        return None, None
    # Конец тела: первая строка с нулевым отступом, которая реально начинает новый блок
    # (class / секция # ===), а НЕ закрывающая скобка ] или ) от списка/кортежа
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            continue
        if line[0] != ' ' and line[0] != '\t':
            # Строка с нулевым отступом — конец тела только если это новый класс или секция
            if re.match(r'^class\s+\w+\s*[:(]', stripped) or stripped.startswith('# ==='):
                end_idx = i
                break
            # Иначе это ] или ) или что-то ещё — не конец класса, ищем дальше
    return start_idx, end_idx


def _generate_class_body(config: Dict[str, Any]) -> list:
    """Генерирует строки тела класса (атрибуты В ВЕРХНЕМ РЕГИСТРЕ + комментарий) из словаря конфига."""
    out = []
    for key, value in config.items():
        attr_name = key.upper() if isinstance(key, str) else key
        out.append(f'    {attr_name} = {_format_python_value(value)}  # настройка\n')
    return out


def _parse_attr_line(line: str) -> Optional[tuple]:
    """
    Парсит строку атрибута класса: "    ENABLED = True  # comment" -> ('enabled', True).
    Возвращает (key_lower, value) или None если строка не атрибут.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith('#') or stripped.startswith('"""'):
        return None
    match = re.match(r'^([A-Z_0-9]+)\s*=\s*(.+)$', stripped)
    if not match:
        return None
    attr_upper, value_part = match.groups()
    value_part = value_part.strip()
    if '#' in value_part:
        value_part = value_part.split('#', 1)[0].strip().rstrip(',')
    try:
        value = ast.literal_eval(value_part)
    except (ValueError, SyntaxError):
        return None
    return (attr_upper.lower(), value)


# Обратная совместимость: при загрузке ключ из файла подменяется на текущий (API/код).
CONFIG_KEY_ALIASES: Dict[str, str] = {
    'fullai_adaptive_virtual_success': 'fullai_adaptive_virtual_success_count',
    'fullai_adaptive_real_loss': 'fullai_adaptive_real_loss_to_retry',
    'fullai_adaptive_round_size': 'fullai_adaptive_virtual_round_size',
    'fullai_adaptive_max_failures': 'fullai_adaptive_virtual_max_failures',
}

# При сохранении: ключ в config (API) -> имя атрибута в bot_config.py (чтобы не дублировать строки).
CONFIG_KEY_TO_FILE_ATTR: Dict[str, str] = {
    'fullai_adaptive_virtual_success_count': 'FULLAI_ADAPTIVE_VIRTUAL_SUCCESS',
    'fullai_adaptive_real_loss_to_retry': 'FULLAI_ADAPTIVE_REAL_LOSS',
    'fullai_adaptive_virtual_round_size': 'FULLAI_ADAPTIVE_ROUND_SIZE',
    'fullai_adaptive_virtual_max_failures': 'FULLAI_ADAPTIVE_MAX_FAILURES',
}


def load_auto_bot_config_from_file(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Загружает конфиг из файла, парсируя КАЖДУЮ строку класса AutoBotConfig отдельно.
    Возвращает словарь {ключ_в_нижнем_регистре: значение}. Не перезагружает модуль — только чтение файла.
    Обратная совместимость: неизвестные строки пропускаются, старые ключи маппятся через CONFIG_KEY_ALIASES.
    """
    if config_file is None:
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_current_dir)
        config_file = os.path.join(_project_root, 'configs', 'bot_config.py')
    if not os.path.exists(config_file):
        return {}
    with open(config_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    start_idx, end_idx = _find_class_block(lines, 'AutoBotConfig')
    if start_idx is None or end_idx is None:
        return {}
    result = {}
    for i in range(start_idx + 1, end_idx):
        parsed = _parse_attr_line(lines[i])
        if parsed:
            key_lower, value = parsed
            canonical_key = CONFIG_KEY_ALIASES.get(key_lower, key_lower)
            result[canonical_key] = value
    return result


def save_auto_bot_config_to_py(config: Dict[str, Any]) -> bool:
    """
    Безопасно обновляет класс DefaultAutoBotConfig в configs/bot_config.py.
    UI сохраняет текущие настройки в класс AutoBotConfig (save_auto_bot_config_current_to_py).
    DefaultAutoBotConfig — только шаблон для кнопки «Сбросить к стандарту».
    """
    try:
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_current_dir)
        config_file = os.path.join(_project_root, 'configs', 'bot_config.py')
        if not _ensure_bot_config_exists(config_file):
            logger.error(f"[CONFIG_WRITER] ❌ Файл {config_file} не найден и не удалось создать из примера")
            return False
        with open(config_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        start_idx, end_idx = _find_class_block(lines, 'DefaultAutoBotConfig')
        if start_idx is None or end_idx is None:
            logger.error(f"[CONFIG_WRITER] ❌ Не найден класс DefaultAutoBotConfig")
            return False
        logger.info(f"[CONFIG_WRITER] 📝 Найден класс DefaultAutoBotConfig: строки {start_idx+1}-{end_idx}")
        config = dict(config)
        if 'leverage' not in config:
            config['leverage'] = 10
        # Сохраняем строку класса и докстринг (все строки тела до первой атрибутивной "key = value")
        doc_end = end_idx
        for i in range(start_idx + 1, end_idx):
            line = lines[i]
            if re.match(r'^\s+\w+\s*=', line):
                doc_end = i
                break
        new_body = _generate_class_body(config)
        updated_lines = lines[: doc_end] + new_body + lines[end_idx:]
        with _config_write_lock:
            with open(config_file, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
                f.flush()
                os.fsync(f.fileno())
        logger.info(f"[CONFIG_WRITER] ✅ DefaultAutoBotConfig обновлён в {config_file}")
        return True
    except Exception as e:
        logger.error(f"[CONFIG_WRITER] ❌ Ошибка сохранения конфигурации: {e}")
        import traceback
        traceback.print_exc()
        return False


def _update_attr_value_in_line(line: str, attr_upper: str, new_value: Any) -> Optional[str]:
    """
    Если строка — атрибут класса вида "    ATTR = value  # comment", подменяет только value.
    Возвращает обновлённую строку или None, если строка не этот атрибут.
    КРИТИЧНО: всегда пишем отступ 4 пробела, чтобы не тиражировать ошибочный отступ (8 пробелов).
    """
    match = re.match(r'^(\s+)([A-Z_0-9]+)\s*=\s*([^#\n]*?)(\s*(?:#.*)?)$', line)
    if not match:
        return None
    name, _old_val, rest = match.group(2), match.group(3), match.group(4)
    if name != attr_upper:
        return None
    new_val_str = _format_python_value(new_value)
    return f'    {name} = {new_val_str}{rest.rstrip()}\n'


def save_auto_bot_config_current_to_py(config: Dict[str, Any]) -> bool:
    """
    Обновляет класс AutoBotConfig в configs/bot_config.py по каждой настройке отдельно:
    для каждого параметра из config находится строка в файле и заменяется только значение (комментарии сохраняются).
    Если в файле нет строки для ключа (устаревший файл или новый параметр) — строка добавляется в конец тела класса.
    Остальные строки файла не трогаются. После записи перезагружает конфиг.
    """
    try:
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_current_dir)
        config_file = os.path.join(_project_root, 'configs', 'bot_config.py')
        if not os.path.exists(config_file):
            logger.error(f"[CONFIG_WRITER] ❌ Файл {config_file} не найден")
            return False
        with open(config_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        start_idx, end_idx = _find_class_block(lines, 'AutoBotConfig')
        if start_idx is None or end_idx is None:
            logger.error(f"[CONFIG_WRITER] ❌ Не найден класс AutoBotConfig в {config_file}")
            return False
        # Защита от перезаписи повреждённого конфига: если после AutoBotConfig нет других классов — не трогать
        content_after = ''.join(lines[end_idx:]) if end_idx < len(lines) else ''
        if end_idx >= len(lines) or not re.search(r'\bclass\s+(SystemConfig|DefaultBotConfig|RiskConfig)\b', content_after):
            logger.error(
                "[CONFIG_WRITER] ❌ Файл конфига повреждён (нет класса SystemConfig/DefaultBotConfig после AutoBotConfig). "
                "Восстановите configs/bot_config.py из configs/bot_config.example.py вручную."
            )
            return False
        updated_lines = list(lines)
        keys_not_found: list = []  # (attr_upper, value) для вставки
        for key, value in config.items():
            key_str = key.lower() if isinstance(key, str) else key
            attr_upper = CONFIG_KEY_TO_FILE_ATTR.get(key_str, key.upper() if isinstance(key, str) else key)
            found = False
            for i in range(start_idx + 1, end_idx):
                new_line = _update_attr_value_in_line(updated_lines[i], attr_upper, value)
                if new_line is not None:
                    updated_lines[i] = new_line
                    found = True
                    break
            if not found:
                keys_not_found.append((attr_upper, value))
        # Добавляем отсутствующие ключи в конец тела класса (обратная совместимость: файл мог быть старым)
        for attr_upper, value in keys_not_found:
            new_line = f'    {attr_upper} = {_format_python_value(value)}  # настройка\n'
            updated_lines.insert(end_idx, new_line)
            end_idx += 1
        with _config_write_lock:
            with open(config_file, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
                f.flush()
                os.fsync(f.fileno())
        try:
            from bot_engine.config_loader import reload_config
            reload_config()
        except Exception as reload_err:
            logger.warning(f"[CONFIG_WRITER] ⚠️ Не удалось перезагрузить конфиг после сохранения: {reload_err}")
        logger.info(f"[CONFIG_WRITER] ✅ AutoBotConfig сохранён в {config_file} (по одной настройке в блок)")
        return True
    except Exception as e:
        logger.error(f"[CONFIG_WRITER] ❌ Ошибка сохранения AutoBotConfig: {e}")
        return False


def patch_system_config_add_bybit_margin_mode(config_file: Optional[str] = None) -> bool:
    """
    Одноразовый патч для существующих пользователей: добавляет BYBIT_MARGIN_MODE в SystemConfig
    в configs/bot_config.py, если этой строки ещё нет. После обновления кода у пользователей
    настройка появится в конфиге и будет сохраняться из UI.
    """
    if config_file is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_file = os.path.join(project_root, 'configs', 'bot_config.py')
    if not os.path.exists(config_file):
        return False
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        if 'BYBIT_MARGIN_MODE' in content:
            return True
        lines = content.splitlines(keepends=True)
        insert_idx = None
        in_system_config = False
        for i, line in enumerate(lines):
            if 'class SystemConfig' in line or line.strip().startswith('class SystemConfig'):
                in_system_config = True
                continue
            if in_system_config and line.strip().startswith('class ') and 'SystemConfig' not in line:
                break
            if in_system_config and 'TREND_REQUIRE_CANDLES' in line and '=' in line:
                insert_idx = i + 1
        if insert_idx is None:
            logger.debug("[CONFIG_WRITER] Патч BYBIT_MARGIN_MODE: не найдена строка TREND_REQUIRE_CANDLES в SystemConfig")
            return False
        new_line = "    BYBIT_MARGIN_MODE = 'auto'  # Bybit: auto | cross | isolated\n"
        lines.insert(insert_idx, new_line)
        with _config_write_lock:
            with open(config_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                f.flush()
                os.fsync(f.fileno())
        logger.info("[CONFIG_WRITER] ✅ В конфиг добавлена настройка BYBIT_MARGIN_MODE (режим маржи Bybit)")
        return True
    except Exception as e:
        logger.warning(f"[CONFIG_WRITER] Патч BYBIT_MARGIN_MODE не применён: {e}")
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
        config_file = os.path.join(project_root, 'configs', 'bot_config.py')
        if not os.path.exists(config_file):
            if _ensure_bot_config_exists(config_file):
                pass
            else:
                alt = os.path.join(os.getcwd(), 'configs', 'bot_config.py')
                if os.path.exists(alt):
                    config_file = alt
                else:
                    logger.error(f"[CONFIG_WRITER] ❌ Файл configs/bot_config.py не найден и не удалось создать из примера")
                    return False
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

        with _config_write_lock:
            with open(config_file, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
                f.flush()
                os.fsync(f.fileno())

        logger.info("[CONFIG_WRITER] ✅ SystemConfig обновлен в bot_config.py")
        return True

    except Exception as e:
        logger.error(f"[CONFIG_WRITER] ❌ Ошибка сохранения SystemConfig: {e}")
        import traceback
        traceback.print_exc()
        return False

