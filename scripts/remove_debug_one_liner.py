"""
Удаляет только однострочные вызовы logger.debug(...) и self.logger.debug(...).
Строка заменяется на отступ + pass.
"""
import re
import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCLUDE = {'.git', '__pycache__', 'venv', '.venv', 'env', 'node_modules', '.cursor'}
EXCLUDE_FILES = {'remove_debug_logs.py', 'remove_debug_one_liner.py'}


def process_file(path: Path) -> tuple[int, bool]:
    """Заменяет однострочные logger.debug/self.logger.debug на pass. Возвращает (количество, изменился ли файл)."""
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return 0, False
    lines = text.split('\n')
    count = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        # Любой логгер: logger.debug, app_logger.debug, cache_logger.debug и т.д. (однострочный вызов)
        if '.debug(' not in stripped:
            continue
        if not re.match(r'^[\w.]*\.debug\s*\(', stripped):
            continue
        if ')' not in stripped:
            continue
        # Проверяем, что закрывающая скобка вызова на этой же строке (однострочный вызов)
        depth = 0
        in_str = None
        j = 0
        while j < len(stripped):
            c = stripped[j]
            if in_str:
                if c == '\\' and j + 1 < len(stripped):
                    j += 2
                    continue
                if c == in_str:
                    in_str = None
                j += 1
                continue
            if c in '"\'':
                in_str = c
                j += 1
                continue
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            continue
        # Вся строка — один вызов (до конца или до комментария)
        if '#' in stripped:
            rest = stripped.split('#', 1)[0].rstrip()
            if not rest.endswith(')'):
                continue
        indent = line[:len(line) - len(stripped)]
        lines[i] = indent + 'pass'
        count += 1
    if count == 0:
        return 0, False
    new_text = '\n'.join(lines)
    try:
        ast.parse(new_text)
    except SyntaxError:
        return count, False
    path.write_text(new_text, encoding='utf-8')
    return count, True


def main():
    total = 0
    for py in PROJECT_ROOT.rglob('*.py'):
        if py.name in EXCLUDE_FILES:
            continue
        if any(d in py.parts for d in EXCLUDE):
            continue
        n, ok = process_file(py)
        if ok:
            total += n
            print(f"  {py.relative_to(PROJECT_ROOT)}: {n}")
    print(f"\nВсего заменено однострочных logger.debug: {total}")


if __name__ == '__main__':
    main()
