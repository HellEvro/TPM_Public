"""
Патч 015: исправление сломанного многострочного BLACKLIST/WHITELIST в configs/bot_config.py.

Если список закрыт на одной строке (ATTR = [...]) и следом идут «осиротевшие» строки
с элементами списка — конфиг даёт IndentationError при загрузке. Патч склеивает в один
корректный многострочный список.
"""
import re
from pathlib import Path


def apply(project_root: Path) -> bool:
    path = project_root / "configs" / "bot_config.py"
    if not path.exists():
        return True
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    if not lines:
        return True

    # Ищем паттерн: строка "    BLACKLIST = [..., 'last']" (закрыта ]) и следующая непустая — с отступом и кавычкой (осиротевшие элементы)
    attr_pattern = re.compile(r"^(\s*)(BLACKLIST|WHITELIST)\s*=\s*\[(.*)\]\s*((?:#.*)?)$")
    continuation_pattern = re.compile(r"^\s{4,}['\"].*$")  # строка с отступом и строковым литералом
    closing_only = re.compile(r"^\s*\]\s*((?:#.*)?)$")

    new_lines = []
    i = 0
    fixed = False
    while i < len(lines):
        line = lines[i]
        m = attr_pattern.match(line.rstrip())
        if m:
            indent, attr_name, inner, comment = m.groups()
            # Проверяем, есть ли следом «осиротевшие» продолжения
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith("#")):
                j += 1
            if j < len(lines) and continuation_pattern.match(lines[j]):
                # Сломанный случай: после закрытого списка идут строки-продолжения
                # Собираем все продолжения до закрывающей ]
                continuation_lines = []
                while j < len(lines):
                    stripped = lines[j].strip()
                    if not stripped:
                        j += 1
                        continue
                    if closing_only.match(lines[j].rstrip()):
                        continuation_lines.append(lines[j])
                        j += 1
                        break
                    if stripped.startswith("#"):
                        j += 1
                        continue
                    if continuation_pattern.match(lines[j]) or (stripped.startswith("'") or stripped.startswith('"')):
                        continuation_lines.append(lines[j])
                        j += 1
                    else:
                        break
                if continuation_lines:
                    # Формируем корректный многострочный список: первая строка без ], с запятой
                    first_line_content = inner.strip()
                    if first_line_content and not first_line_content.endswith(","):
                        first_line_content += ","
                    new_first = f"{indent}{attr_name} = [\n"
                    new_second = f"{indent}    {first_line_content}\n"
                    new_lines.append(new_first)
                    new_lines.append(new_second)
                    new_lines.extend(continuation_lines)
                    i = j
                    fixed = True
                    continue
        new_lines.append(line)
        i += 1

    if fixed:
        path.write_text("".join(new_lines), encoding="utf-8")
    return True
