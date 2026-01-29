"""
Скрипт удаления всех вызовов logger.debug и self.logger.debug из .py файлов проекта.
Обрабатывает однострочные и многострочные вызовы (с учётом вложенных скобок).
"""
import re
import ast
import sys
from pathlib import Path

# Корень проекта (родитель папки scripts)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {
    '.git', '__pycache__', 'venv', '.venv', 'env', 'node_modules',
    '.cursor', 'migrations', '.pytest_cache', 'htmlcov', '.tox'
}
EXCLUDE_FILES = {'remove_debug_logs.py'}


def find_matching_paren(text: str, start: int) -> int:
    """Найти позицию закрывающей скобки для открывающей в text[start]."""
    depth = 1
    i = start + 1
    in_string = None
    escape = False
    while i < len(text):
        c = text[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == '\\' and in_string:
            escape = True
            i += 1
            continue
        if in_string:
            if c == in_string:
                in_string = None
            i += 1
            continue
        if c in '"\'':
            in_string = c
            i += 1
            continue
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def get_line_start(content: str, pos: int) -> int:
    """Вернуть индекс начала строки для позиции pos."""
    i = content.rfind('\n', 0, pos)
    return i + 1 if i >= 0 else 0


def remove_debug_calls(content: str) -> str:
    """Заменить все вызовы logger.debug(...) и self.logger.debug(...) на pass."""
    pattern = re.compile(
        r'(\blogger\.debug\s*\(|self\.logger\.debug\s*\()',
        re.MULTILINE
    )
    result = []
    last_end = 0
    for m in pattern.finditer(content):
        result.append(content[last_end:m.start()])
        line_start = get_line_start(content, m.start())
        indent = content[line_start:m.start()]  # отступ до вызова
        open_pos = m.end() - 1
        close_pos = find_matching_paren(content, open_pos)
        if close_pos == -1:
            result.append(content[m.start():m.end()])
            last_end = m.end()
            continue
        call_end = close_pos + 1
        rest = content[call_end:call_end + 20]
        if re.match(r'\s*,', rest):
            call_end += len(re.match(r'\s*,', rest).group(0))
        # Заменяем вызов на pass (чтобы не оставлять пустые блоки try/except/if)
        result.append(indent + 'pass')
        last_end = call_end
    result.append(content[last_end:])
    out = ''.join(result)
    out = re.sub(r'^\s+$', '', out, flags=re.MULTILINE)
    out = re.sub(r'\n{4,}', '\n\n\n', out)
    return out


def process_file(path: Path) -> tuple[bool, int]:
    """Обработать один файл. Возвращает (changed, count_removed)."""
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"  [SKIP] {path.relative_to(PROJECT_ROOT)}: не удалось прочитать - {e}")
        return False, 0
    original_count = len(re.findall(r'\blogger\.debug\s*\(|self\.logger\.debug\s*\(', text))
    if original_count == 0:
        return False, 0
    new_text = remove_debug_calls(text)
    if new_text == text:
        return False, 0
    try:
        ast.parse(new_text)
    except SyntaxError as err:
        print(f"  [SKIP] {path.relative_to(PROJECT_ROOT)}: после удаления синтаксическая ошибка - {err}")
        return False, 0
    path.write_text(new_text, encoding='utf-8')
    return True, original_count


def main():
    total_files = 0
    total_calls = 0
    for py_path in PROJECT_ROOT.rglob('*.py'):
        if py_path.name in EXCLUDE_FILES:
            continue
        parts = py_path.relative_to(PROJECT_ROOT).parts
        if any(d in EXCLUDE_DIRS for d in parts):
            continue
        changed, count = process_file(py_path)
        if changed:
            total_files += 1
            total_calls += count
            print(f"  [OK] {py_path.relative_to(PROJECT_ROOT)} — удалено вызовов: {count}")
    print(f"\nИтого: обработано файлов {total_files}, удалено вызовов logger.debug: {total_calls}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
