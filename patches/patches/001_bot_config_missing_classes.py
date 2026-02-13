"""
Патч 001: добавление недостающих классов в configs/bot_config.py для старых пользователей.
Исправляет ImportError: cannot import name 'DefaultBotConfig' (и др.) при запуске bots.py / app.py.
"""
from pathlib import Path


def apply(project_root: Path) -> bool:
    config_path = project_root / "configs" / "bot_config.py"
    example_path = project_root / "configs" / "bot_config.example.py"
    if not config_path.exists() or not example_path.exists():
        return False
    content = config_path.read_text(encoding="utf-8")
    if "class DefaultBotConfig" in content:
        # Уже есть, помечаем как успех чтобы не повторять
        return True
    # Читаем из примера блок с DefaultBotConfig до конца файла
    example_lines = example_path.read_text(encoding="utf-8").splitlines()
    start = None
    for i, line in enumerate(example_lines):
        if "class DefaultBotConfig" in line:
            # Включаем комментарий выше, если есть
            if i > 0 and "# ====================" in example_lines[i - 1]:
                start = i - 1
            else:
                start = i
            break
    if start is None:
        return False
    block = "\n".join(example_lines[start:]).strip()
    if not block:
        return False
    new_content = content.rstrip() + "\n\n\n# ----- добавлено патчем 001 (недостающие классы для config_loader) -----\n\n" + block + "\n"
    config_path.write_text(new_content, encoding="utf-8")
    return True
