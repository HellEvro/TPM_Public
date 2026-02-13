"""
Патч 003: включить exit_wait_breakeven_when_loss = True по умолчанию.
Исправляет закрытие в минус при RSI/тейках — теперь ждём безубыток.
"""
from pathlib import Path

SEARCH = "EXIT_WAIT_BREAKEVEN_WHEN_LOSS = False"
REPLACE = "EXIT_WAIT_BREAKEVEN_WHEN_LOSS = True"


def apply(project_root: Path) -> bool:
    config_path = project_root / "configs" / "bot_config.py"
    if not config_path.exists():
        return False

    content = config_path.read_text(encoding="utf-8")
    if SEARCH not in content:
        return True

    content = content.replace(SEARCH, REPLACE)
    config_path.write_text(content, encoding="utf-8")
    return True
