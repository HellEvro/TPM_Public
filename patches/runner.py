"""
Механизм патчей: при запуске лаунчера выполняются патчи из папки patches/patches/.
Каждый патч выполняется один раз (учёт в launcher/.patches_applied.json).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Tuple

# Корень проекта (папка patches на уровень ниже корня)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCHES_DIR = Path(__file__).resolve().parent / "patches"
STATE_FILE = PROJECT_ROOT / "launcher" / ".patches_applied.json"


def _load_applied_ids() -> List[str]:
    """Загружает список уже применённых ID патчей."""
    if not STATE_FILE.exists():
        return []
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return list(data.get("applied", []))
    except Exception:
        return []


def _save_applied_ids(ids: List[str]) -> None:
    """Сохраняет список применённых ID."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps({"applied": ids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _discover_patches() -> List[Tuple[str, Path]]:
    """Возвращает список (patch_id, path) отсортированный по имени файла."""
    if not PATCHES_DIR.is_dir():
        return []
    result = []
    for path in sorted(PATCHES_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        # ID патча = имя файла без расширения (например 001_bot_config_missing_classes)
        patch_id = path.stem
        result.append((patch_id, path))
    return result


def run_patches(project_root: Path | None = None) -> List[str]:
    """
    Запускает все ещё не применённые патчи.
    project_root: корень проекта (по умолчанию родитель папки patches).
    Возвращает список ID применённых патчей в этом запуске.
    """
    root = project_root or PROJECT_ROOT
    applied = _load_applied_ids()
    discovered = _discover_patches()
    newly_applied: List[str] = []

    for patch_id, path in discovered:
        if patch_id in applied:
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"patches.patches.{patch_id}", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            if not hasattr(mod, "apply"):
                continue
            ok = mod.apply(root)
            if ok:
                applied.append(patch_id)
                newly_applied.append(patch_id)
                _save_applied_ids(applied)
        except Exception:
            # Не падаем на одном патче, продолжаем
            pass

    return newly_applied
