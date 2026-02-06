#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка версий AI/ML зависимостей (sklearn, joblib) и тест загрузки моделей.
Используется после pip install -r requirements.txt. При несовпадении версий
или ошибках загрузки — переустановите: pip install 'scikit-learn>=1.7.0,<1.8'
или запустите «Обновить venv» в лаунчере.

Выход: 0 — OK, 1 — ошибка (неверные версии или не удалось загрузить модели).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Фикс кодировки stderr/stdout на Windows (лаунчер читает UTF-8)
if os.name == "nt":
    try:
        if getattr(sys.stderr, "encoding", None) != "utf-8":
            import io
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        if getattr(sys.stdout, "encoding", None) != "utf-8":
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _check_versions() -> tuple[bool, str]:
    try:
        import sklearn
        import joblib
    except ImportError as e:
        return False, f"sklearn/joblib не установлены: {e}"

    sk = getattr(sklearn, "__version__", "?")
    jb = getattr(joblib, "__version__", "?")

    # Ожидаем sklearn >= 1.7, < 1.8 (совместимо с pickle моделями)
    major, minor = 0, 0
    try:
        parts = sk.split(".")
        if len(parts) >= 2:
            major, minor = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass

    if major < 1 or (major == 1 and minor < 7):
        return False, f"scikit-learn {sk} — требуется >=1.7.0,<1.8 (сейчас {sk})"
    if major > 1 or (major == 1 and minor >= 8):
        return False, f"scikit-learn {sk} — требуется >=1.7.0,<1.8 (сейчас {sk})"

    return True, f"sklearn={sk} joblib={jb}"


def _test_load_models() -> tuple[bool, str]:
    root = _project_root()
    models_dir = root / "data" / "ai" / "models"
    if not models_dir.exists():
        return True, "модели не найдены (проверка пропущена)"

    try:
        import joblib
    except ImportError:
        return False, "joblib не установлен"

    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except ImportError:
        InconsistentVersionWarning = None

    candidates = [
        "signal_predictor.pkl",
        "profit_predictor.pkl",
        "scaler.pkl",
    ]
    loaded = 0
    errors: list[str] = []

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for name in candidates:
            path = models_dir / name
            if not path.is_file():
                continue
            try:
                joblib.load(path)
                loaded += 1
            except Exception as e:
                errors.append(f"{name}: {e}")
        inconsistent = []
        for x in w:
            if InconsistentVersionWarning and x.category and issubclass(x.category, InconsistentVersionWarning):
                inconsistent.append(x)
            elif "unpickle" in str(getattr(x, "message", "")).lower() and "version" in str(getattr(x, "message", "")).lower():
                inconsistent.append(x)

    if errors:
        return False, "; ".join(errors)
    if loaded:
        if inconsistent:
            return True, f"загружено {loaded} моделей OK (версия sklearn при сохранении отличалась — используем как есть)"
        return True, f"загружено {loaded} моделей OK"
    return True, "моделей нет"


def main() -> int:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    ok, msg = _check_versions()
    if not ok:
        print(f"[verify_ai_deps] Ошибка версий: {msg}", file=sys.stderr)
        print("[verify_ai_deps] Установите: pip install 'scikit-learn>=1.7.0,<1.8'", file=sys.stderr)
        return 1

    print(f"[verify_ai_deps] Версии: {msg}")

    ok, msg = _test_load_models()
    if not ok:
        print(f"[verify_ai_deps] Ошибка загрузки моделей: {msg}", file=sys.stderr)
        print("[verify_ai_deps] Сделайте: «Обновить venv» в лаунчере. Если не поможет — запустите AI Engine и дождитесь обучения; модели обновятся.", file=sys.stderr)
        return 1

    print(f"[verify_ai_deps] Модели: {msg}")
    print("[verify_ai_deps] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
