"""
Патч 008: добавить configs/fullai_config.json в .gitignore (рабочий конфиг только локально).
"""
from pathlib import Path

ANCHOR = "configs/default_auto_bot_config.json\n"
INSERT = "configs/default_auto_bot_config.json\n# Full AI (ПРИИ) — рабочий конфиг, только локально; пример: configs/fullai_config.example.json\nconfigs/fullai_config.json\n"


def apply(project_root: Path) -> bool:
    gitignore = project_root / ".gitignore"
    if not gitignore.exists():
        return False
    text = gitignore.read_text(encoding="utf-8")
    if "configs/fullai_config.json" in text:
        return True
    if ANCHOR not in text:
        return True
    text = text.replace(
        ANCHOR,
        INSERT,
        1,
    )
    gitignore.write_text(text, encoding="utf-8")
    return True

