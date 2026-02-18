"""
Патч 004: добавление блока FullAI (FULL_AI_CONTROL, FULLAI_ADAPTIVE_*) в DefaultAutoBotConfig
в configs/bot_config.py. Иначе при «Сбросить к стандарту» эти настройки теряются.
"""
from pathlib import Path

FULLAI_BLOCK = '''
    # --- FullAI (полный режим ИИ и виртуальная обкатка) ---
    FULL_AI_CONTROL = False                 # Полный режим ИИ: ИИ сам управляет входами/выходами
    FULLAI_ADAPTIVE_ENABLED = True          # Виртуальная обкатка (N успешных виртуальных → 1 реальная)
    FULLAI_ADAPTIVE_DEAD_CANDLES = 100      # Свечей без сделок → смена параметров
    FULLAI_ADAPTIVE_VIRTUAL_SUCCESS = 3     # Удачных виртуальных подряд → 1 реальная (0 = только реальные)
    FULLAI_ADAPTIVE_REAL_LOSS = 1           # Убыточных реальных подряд → снова виртуальные
    FULLAI_ADAPTIVE_ROUND_SIZE = 3          # Размер серии виртуальных
    FULLAI_ADAPTIVE_MAX_FAILURES = 0        # Макс. неудачных виртуальных в серии (0 = все удачны)
'''


def apply(project_root: Path) -> bool:
    config_path = project_root / "configs" / "bot_config.py"
    if not config_path.exists():
        return False
    text = config_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Найти начало и конец класса DefaultAutoBotConfig
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("class DefaultAutoBotConfig"):
            start_idx = i
            continue
        if start_idx is not None and end_idx is None:
            # Конец класса: следующая строка class или блок с # ====
            if line.strip().startswith("class ") and "DefaultAutoBotConfig" not in line:
                end_idx = i
                break
            if "# ====================" in line and "БЛОК 2" in line and i > start_idx + 5:
                end_idx = i
                break

    if start_idx is None or end_idx is None:
        return False

    class_body = "".join(lines[start_idx:end_idx])
    if "FULLAI_ADAPTIVE_ENABLED" in class_body or "FULLAI_ADAPTIVE_DEAD_CANDLES" in class_body:
        return True  # Уже есть

    # Вставить блок перед концом класса (перед end_idx)
    insert_line = end_idx
    # Подставить перед строкой с "# ====" или "class AutoBotConfig"
    new_lines = lines[:insert_line] + [FULLAI_BLOCK + "\n"] + lines[insert_line:]
    config_path.write_text("".join(new_lines), encoding="utf-8")
    return True
