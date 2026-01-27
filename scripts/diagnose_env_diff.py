#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диагностика различий окружений (Win10 vs Win11, разные ПК).

Запусти на «хорошем» и «плохом» ПК, сохрани вывод в файлы, затем сравни.
Помогает выявить отличия в версиях, CPU, путях, переменных окружения.

Использование:
  python scripts/diagnose_env_diff.py
  python scripts/diagnose_env_diff.py > env_good_pc.txt
  python scripts/diagnose_env_diff.py > env_bad_pc.txt
"""

from __future__ import annotations

import os
import platform
import struct
import sys
from pathlib import Path


def _v(m: object, attr: str, default: str = "?") -> str:
    return str(getattr(m, attr, default))


def _try(fn, default: str = "?") -> str:
    try:
        return str(fn())
    except Exception as e:
        return f"error: {e}"


def main() -> None:
    lines: list[str] = []
    sep = "=" * 60

    lines.append("InfoBot env diagnostic")
    lines.append(sep)
    lines.append(f"Python:     {sys.version}")
    lines.append(f"executable: {sys.executable}")
    lines.append(f"platform:   {platform.platform()}")
    lines.append(f"system:     {platform.system()} {platform.release()} ({platform.version()})")
    lines.append(f"machine:    {platform.machine()}")
    lines.append(f"proc:       {platform.processor()}")
    lines.append(f"cpu count:  {os.cpu_count()}")
    lines.append(f"64-bit:     {struct.calcsize('P') * 8}-bit")
    lines.append("")

    # CWD and project
    try:
        root = Path(__file__).resolve().parents[1]
        lines.append(f"project root: {root}")
        lines.append(f"cwd:          {os.getcwd()}")
    except Exception as e:
        lines.append(f"paths error: {e}")
    lines.append("")

    # Env
    lines.append("Relevant env:")
    for k in (
        "PYTHONWARNINGS",
        "LOKY_MAX_CPU_COUNT",
        "JOBLIB_START_METHOD",
        "INFOBOT_SKLEARN_SINGLE_THREAD",
        "INFOBOT_AI_PROCESS",
        "PATH",
    ):
        v = os.environ.get(k, "")
        if k == "PATH" and v:
            v = v[:200] + "..." if len(v) > 200 else v
        lines.append(f"  {k}={v!r}")
    lines.append("")

    # Packages
    lines.append("Packages:")
    for pkg, mod, attr in (
        ("numpy", "numpy", "__version__"),
        ("scipy", "scipy", "__version__"),
        ("sklearn", "sklearn", "__version__"),
        ("joblib", "joblib", "__version__"),
        ("torch", "torch", "__version__"),
        ("pandas", "pandas", "__version__"),
    ):
        try:
            m = __import__(mod)
            lines.append(f"  {pkg}: {_v(m, attr)}")
        except Exception as e:
            lines.append(f"  {pkg}: not installed ({e})")

    lines.append("")
    lines.append("sklearn.utils.parallel:")
    try:
        from sklearn.utils import parallel as skp

        pmod = getattr(skp.Parallel, "__module__", "?")
        dmod = getattr(skp.delayed, "__module__", "?")
        lines.append(f"  Parallel: {pmod}")
        lines.append(f"  delayed:  {dmod}")
    except Exception as e:
        lines.append(f"  error: {e}")

    lines.append("")
    lines.append("joblib (after app: patched to sklearn if config loaded):")
    try:
        import joblib

        pmod = getattr(joblib.Parallel, "__module__", "?")
        dmod = getattr(joblib.delayed, "__module__", "?")
        lines.append(f"  Parallel: {pmod}")
        lines.append(f"  delayed:  {dmod}")
    except Exception as e:
        lines.append(f"  error: {e}")

    lines.append("")
    lines.append("If sklearn UserWarning/lock/FakeTensor spam on this PC but not the other:")
    lines.append("  1. Compare this output with the other PC (env_*.txt).")
    lines.append("  2. Run with: set INFOBOT_SKLEARN_SINGLE_THREAD=1 && python ai.py")
    lines.append("     (PowerShell: $env:INFOBOT_SKLEARN_SINGLE_THREAD='1'; python ai.py)")
    lines.append("")
    lines.append(sep)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
