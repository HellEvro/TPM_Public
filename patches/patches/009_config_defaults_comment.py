"""
Патч 009: комментарий в config.js над DEFAULTS — «единственный источник правды»
для порога PnL и других дефолтов (чтобы не дублировать значения в HTML/JS).
"""
from pathlib import Path

OLD = "// Значения по умолчанию\nconst DEFAULTS = {"
NEW = "// Значения по умолчанию (единственный источник правды для порога PnL и др.)\nconst DEFAULTS = {"


def apply(project_root: Path) -> bool:
    path = project_root / "static" / "js" / "config.js"
    if not path.exists():
        return True
    text = path.read_text(encoding="utf-8")
    if "единственный источник правды" in text:
        return True
    if OLD not in text:
        return True
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    return True
