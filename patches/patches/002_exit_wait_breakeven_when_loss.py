"""
Патч 002: добавление настройки EXIT_WAIT_BREAKEVEN_WHEN_LOSS в configs/bot_config.py.
Ждать безубыток при выходе в минусе (когда RSI/тейки уже в зоне закрытия).
"""
from pathlib import Path

NEW_LINE = "    EXIT_WAIT_BREAKEVEN_WHEN_LOSS = False   # Ждать безубыток при выходе в минусе (RSI/тейки в зоне закрытия)\n"
ANCHOR = "BREAK_EVEN_TRIGGER_PERCENT"
SKIP_IF = "EXIT_WAIT_BREAKEVEN_WHEN_LOSS"


def apply(project_root: Path) -> bool:
    config_path = project_root / "configs" / "bot_config.py"
    if not config_path.exists():
        return False

    content = config_path.read_text(encoding="utf-8")
    if SKIP_IF in content:
        return True

    if ANCHOR not in content:
        return False

    lines = content.splitlines(keepends=True)
    new_lines = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        if ANCHOR in line and "=" in line:
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if SKIP_IF not in next_line:
                new_lines.append(NEW_LINE)
                changed = True
        i += 1

    if not changed:
        return True

    config_path.write_text("".join(new_lines), encoding="utf-8")
    return True
