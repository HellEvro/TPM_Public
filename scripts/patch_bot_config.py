"""
Ручной запуск автопатча bot_config.py из bot_config.example.py.

- Если bot_config.py отсутствует — копируется bot_config.example.py.
- Если bot_config.py есть — в конец дописываются только недостающие классы из example
  (функции верхнего уровня не добавляются).

Запуск из корня проекта: python scripts/patch_bot_config.py
"""

import os
import sys

# корень проекта (родитель каталога scripts)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bot_engine._patch_config import run_patch


def main() -> None:
    run_patch(root_dir=_ROOT)
    print("Автопатч bot_config выполнен.")


if __name__ == "__main__":
    main()
