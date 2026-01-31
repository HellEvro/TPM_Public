"""
Автопатч bot_config.py из bot_config.example.py.

- Если bot_config.py отсутствует — копируется bot_config.example.py.
- Если bot_config.py есть — в конец дописываются только недостающие классы из example
  (функции верхнего уровня не добавляются).

Вызывается при импорте bot_engine (см. bot_engine/__init__.py) и из scripts/patch_bot_config.py.
"""

import ast
import os
import shutil


def _project_root() -> str:
    """Корень проекта (родитель каталога bot_engine)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _module_level_class_names(tree: ast.Module) -> set:
    return {n.name for n in tree.body if isinstance(n, ast.ClassDef)}


def _classes_with_lines(tree: ast.Module) -> list:
    """Список (name, lineno, end_lineno) для каждого класса на верхнем уровне."""
    result = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            end = getattr(node, 'end_lineno', node.lineno)
            result.append((node.name, node.lineno, end))
    return result


def run_patch(root_dir: str = None) -> None:
    """
    Обеспечивает наличие bot_config.py и дополняет его недостающими классами из example.

    - Если bot_config.py нет — копирует bot_config.example.py.
    - Если оба есть — парсит оба файла и дописывает в bot_config.py только те классы,
      которые есть в example, но отсутствуют в bot_config (без добавления функций).
    """
    root = root_dir or _project_root()
    config_path = os.path.join(root, 'bot_engine', 'bot_config.py')
    example_path = os.path.join(root, 'bot_engine', 'bot_config.example.py')

    if not os.path.exists(example_path):
        return

    if not os.path.exists(config_path):
        try:
            shutil.copy2(example_path, config_path)
        except Exception:
            pass
        return

    try:
        with open(example_path, 'r', encoding='utf-8') as f:
            example_src = f.read()
        with open(config_path, 'r', encoding='utf-8') as f:
            config_src = f.read()
    except Exception:
        return

    try:
        example_tree = ast.parse(example_src)
        config_tree = ast.parse(config_src)
    except SyntaxError:
        return

    if not isinstance(example_tree, ast.Module) or not isinstance(config_tree, ast.Module):
        return

    config_classes = _module_level_class_names(config_tree)
    example_classes = _classes_with_lines(example_tree)
    example_lines = example_src.splitlines()

    to_append = []
    for name, lineno, end_lineno in example_classes:
        if name in config_classes:
            continue
        # извлекаем строки класса (lineno 1-based)
        block = '\n'.join(example_lines[lineno - 1:end_lineno])
        to_append.append(block)

    if not to_append:
        return

    suffix = '\n\n# --- Добавлено автопатчем из bot_config.example.py ---\n\n' + '\n\n'.join(to_append) + '\n'
    try:
        with open(config_path, 'a', encoding='utf-8') as f:
            f.write(suffix)
    except Exception:
        pass
