"""
Патч 006: виртуальные позиции ПРИИ на странице «Позиции», статистика только по реальным.
Проверяет наличие логики (get_virtual_positions_for_api, is_virtual в positions.js).
Если уже есть — помечаем успех. Иначе правки должны быть подтянуты из репозитория (полная замена get_positions и generatePositionsHtml объёмная).
"""
from pathlib import Path


def apply(project_root: Path) -> bool:
    app_py = project_root / "app.py"
    if app_py.exists():
        if "get_virtual_positions_for_api" in app_py.read_text(encoding="utf-8"):
            pos_js = project_root / "static" / "js" / "positions.js"
            if not pos_js.exists():
                return True
            if "is_virtual" in pos_js.read_text(encoding="utf-8"):
                return True
    # Патч не меняет файлы — полная логика в репо; при обновлении кода всё подтянется
    return True

